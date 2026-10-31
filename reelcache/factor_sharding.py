"""Non-replicated distributed factor ownership (SGLang Step 16; plan section 4.8).

A lossy matrix is ``A_hat = P @ Q`` with coefficients ``P`` ``[tokens, rank]`` (FP32, HEVC coded) and an
FP16 basis ``Q`` ``[rank, channels]``. This module partitions the *rank directions* across a process
group so that ``A_hat = sum_s P_s @ Q_s``: owner ``s`` stores and decodes only its coefficient columns
``P_s`` (an independently encoded field subset) and the matching basis rows ``Q_s``. No owner ever holds
another owner's factor rows, and nothing is permanently all-gathered.

Defaults (coordinator decision 2026-10-15, user confirmation pending; a measured deviation from the
literal reconstruction layout of plan section 4.8, see runtime_gate/step16/README.md): whole-matrix owner
blocks (``block_rows=None`` -> one block per matrix, owners alternating by layer/target) and
``mode="factors"`` reconstruction. On 2x RTX PRO 6000 (PCIe) with real K98/V95 nodes of all five models
this restored 1.8-2.2x faster than a single-GPU restore plus dense BF16 owner scatter (prewarmed decoder
pools; 1.55-1.61x on an NVLink A100-40GB pair for the four Wan models, where ``partials`` reached
1.41-1.56x), its busiest sender moving 4-9x fewer bytes than the dense scatter's sender, with identical
accuracy. The plan-literal layout -- 32-row blocks
(``PLAN_BLOCK_ROWS``) and ``mode="partials"`` (FP32 partial-sum reduce-scatter, cheap over NVLink) --
stays available as an explicit, documented ablation; neither choice is ever switched silently.

Two layouts stay distinct:

* :class:`FactorShardSpec` -- factor ownership. The global rank axis of a basis version is cut into
  immutable ``block_rows``-row blocks; block ``b`` covers rows ``[b * block_rows, (b + 1) * block_rows)``
  and has one recorded, stable owner. ``block_rows >= channels`` makes the whole matrix one block (the
  default); ``PLAN_BLOCK_ROWS`` = 32 (the rank method's bucket, ``types.RANK_BUCKET``) is the plan's grid.
  Augmentation appends rows; rows that land in an existing (partial) block keep that block's owner, new
  blocks get ``(b + rotation) % world``. Old rows never move, so a basis version shared by forks has one
  owner per row on every branch. Ranges are explicit and validated: an owner may hold zero rows of a
  matrix; nothing is padded to an unreported full factor.
* :class:`NativeOwners` -- the engine's native KV ownership (head/channel shards as under Ulysses/TP,
  reusing :class:`reelcache.distributed.KVLayout`, or token-row shards such as LongCat-Video's 2-D spatial
  context-parallel split). Restores deliver every native owner its rows/columns of this layout.

Storage. Basis rows are stored as *segments* = intersections of the block grid with the basis versions'
added-row blocks: immutable, content addressed and stored once on their owner, so forks and consecutive
nodes share them. Each owner keeps its fragment pack(s) per node (:func:`encode_fragment_pack`): an
independently encoded all-intra HEVC stream holding only that owner's coefficient columns of the node's
matrices (temporal canvas, per-tile byte ranges; the channel-tile mosaic of :mod:`reelcache._hevc`), never a
byte slice of another stream. ``svd`` keeps exact FP32 coefficients.

Reconstruction (:meth:`FactorShardDomain.reconstruct`) works in bounded token tiles. ``mode="factors"``
(default): owners send factor tiles -- per matrix ONE batched point-to-point round with the native owner's
basis column slice (heads) or all columns (rows) of every owner's rows and the coefficient rows that native
owner needs (bounded by ``FACTOR_EXCHANGE_BYTES`` per round; rounds pipelined across matrices) -- and every
native owner runs one FP32 GEMM per token tile over the full rank (global row order); the transient
working set is reported and released after the matrix. ``mode="partials"``
(ablation): each owner computes the FP32 partial ``P_s[tile] @ Q_s`` into per-destination buffers and one
reduce-scatter (SUM) per tile delivers the sum; an owner without rows contributes the reduction identity
(counted, never hidden). The native owner applies RoPE (pre-RoPE K only) in FP32 and casts to the native
dtype. FP32 partial sums and GEMM shapes differ from a single GEMM only by summation order: that numerical
error is reported separately from RAW/native transport identity.

Archival uses the bounded *owner-fit reference*: one canonical matrix is gathered to a declared fit owner
(and, for augmentation, the parent basis rows), fitted there, and the factor blocks are scattered to their
owners. Its transient bytes are reported. Gram summation over channel (head) shards is rejected: it drops
the cross-channel terms; it is valid only for row shards (:func:`row_shard_gram`).

Ordering and failure. Every phase starts with a control checkpoint of the coordinator
(:class:`reelcache.distributed.DistributedOwner`, a CPU/Gloo group); payloads use a separate data group
(NCCL on CUDA, Gloo on CPU). All fallible local work (decode, allocation, validation) precedes the payload
collectives; failures and cancellations are agreed before any payload moves. Publication is two-phase
(prepare, agree on one node-map digest, commit; any failed commit rolls back every owner's reference),
so a distributed node/basis map becomes visible only after all owners agree. A transport failure marks
the domain broken (process death is bounded by the process-group timeout).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
import sys
import time
import traceback
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import torch

from . import _hevc
from .distributed import CollectiveCancelled, DistributedOwner, KVLayout, PeerFailure
from .errors import ConfigurationError, IntegrityError, StateError, UnsupportedError
from .types import (
    RANK_BUCKET, CodecSpec, EncodedTensor, MatrixKey, TensorContext, canonical_digest, entry_key,
    entry_order, entry_parts, json_copy, object_ref,
)

PLAN_BLOCK_ROWS = RANK_BUCKET  # plan section 4.8's 32-row grid (ablation); the default is whole-matrix blocks
BLOCK_ROWS = PLAN_BLOCK_ROWS  # backward-compatible name of the plan grid
DEFAULT_MODE = "factors"
SPEC_SCHEMA = "reelcache.factor_shard_spec.v1"
FRAGMENT_SCHEMA = "reelcache.factor_fragment_pack.v1"
NODE_MAP_SCHEMA = "reelcache.factor_shard_node_map.v1"
OWNER_SET_TYPE = "factor_owner_set"
NODE_MAP_TYPE = "factor_shard_node_map"
REF_PREFIX = "factor_shard/"
FRAGMENT_KINDS = ("svd", "svd_hevc")
RECONSTRUCT_MODES = ("partials", "factors")
# ``factors`` mode: coefficient bytes one exchange round may carry per peer and matrix (one round per matrix
# for every real node; larger matrices are exchanged in several rounds of at least ``tile_tokens`` tokens).
FACTOR_EXCHANGE_BYTES = 64 * 1024**2
_BASIS_TAG, _COEFFICIENT_TAG = 1, 2


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _int(value: Any, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ConfigurationError(f"{name} must be an integer >= {minimum}")
    return value


def entry_to_dict(entry: Any) -> dict[str, Any]:
    key, stream = entry_parts(entry)
    return {"key": key.to_dict()} if stream is None else {"key": key.to_dict(), "stream": stream}


def entry_from_dict(data: Mapping[str, Any]) -> Any:
    try:
        key = MatrixKey.from_dict(data["key"])
        stream = data.get("stream")
    except (KeyError, TypeError) as exc:
        raise ConfigurationError("invalid factor-shard entry") from exc
    return key if stream is None else entry_key((key, stream))


def entry_label(entry: Any) -> str:
    key, stream = entry_parts(entry)
    return f"{key.layer}:{key.target}" + ("" if stream is None else f":{stream}")


def default_rotation(entry: Any, world_size: int) -> int:
    """Deterministic first owner of a matrix: spreads one-block matrices over owners (layer, K/V)."""
    key, _ = entry_parts(entry)
    return (key.layer + (key.target == "V")) % world_size


def _merge(ranges: Iterable[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    merged: list[list[int]] = []
    for start, stop in sorted(ranges):
        if start >= stop:
            continue
        if merged and merged[-1][1] == start:
            merged[-1][1] = stop
        else:
            merged.append([start, stop])
    return tuple((start, stop) for start, stop in merged)


def ranges_rows(ranges: Sequence[tuple[int, int]]) -> int:
    return sum(stop - start for start, stop in ranges)


def ranges_index(ranges: Sequence[tuple[int, int]], *, device=None) -> torch.Tensor:
    if not ranges:
        return torch.empty(0, dtype=torch.long, device=device)
    return torch.cat([torch.arange(start, stop, dtype=torch.long, device=device) for start, stop in ranges])


# ---------------------------------------------------------------------------------------------- spec

@dataclass(frozen=True)
class FactorShardSpec:
    """Stable factor rank-row ownership of one matrix's basis version (and its rank view).

    A validated record: construct it with :meth:`fresh` / :meth:`extend` / :func:`spec_for_basis_version`,
    whose default is whole-matrix ownership (``block_rows = channels``); the plain dataclass default
    ``block_rows`` = ``PLAN_BLOCK_ROWS`` only applies to direct construction of a plan-grid record.

    ``version_rows`` are the cumulative row boundaries of the basis-version chain (root rows first, then
    each augmentation), so ``version_rows[-1]`` is the version's total rank; segments never straddle a
    version boundary. ``block_owners[b]`` owns grid rows ``[b * block_rows, (b + 1) * block_rows)``.
    ``rank_view`` (<= total rank) is the prefix the matrix's coefficients use (rank trimming keeps the
    owners). ``basis_group_id`` / ``basis_version_ref`` bind the spec to a :class:`~reelcache.types.BasisVersion`
    when the runtime store is used; standalone factors leave them unset.
    """

    key: MatrixKey
    channels: int
    world_size: int
    version_rows: tuple[int, ...]
    block_owners: tuple[int, ...]
    rank_view: int
    rotation: int = 0
    block_rows: int = BLOCK_ROWS
    stream: str | None = None
    basis_group_id: str | None = None
    basis_version_ref: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.key, MatrixKey):
            raise ConfigurationError("factor-shard spec requires a MatrixKey")
        _int(self.channels, "channels", minimum=1)
        _int(self.world_size, "world_size", minimum=1)
        _int(self.block_rows, "block_rows", minimum=1)
        _int(self.rotation, "rotation")
        if self.rotation >= self.world_size:
            raise ConfigurationError("rotation must be a rank of the group")
        rows = tuple(self.version_rows)
        if not rows or any(type(value) is not int for value in rows) or rows[0] < 1 or any(
            later <= earlier for earlier, later in zip(rows, rows[1:])
        ):
            raise ConfigurationError("version_rows must be strictly increasing positive row boundaries")
        if rows[-1] > self.channels:
            raise ConfigurationError("a basis cannot have more rows than channels")
        object.__setattr__(self, "version_rows", rows)
        owners = tuple(self.block_owners)
        if len(owners) != math.ceil(rows[-1] / self.block_rows) or any(
            type(owner) is not int or not 0 <= owner < self.world_size for owner in owners
        ):
            raise ConfigurationError("block_owners must name one valid owner per basis row block")
        object.__setattr__(self, "block_owners", owners)
        _int(self.rank_view, "rank_view", minimum=1)
        if self.rank_view > rows[-1]:
            raise ConfigurationError("rank_view cannot exceed the basis version's total rank")
        if self.stream is not None:
            entry_key((self.key, self.stream), "factor-shard stream")
        if self.basis_group_id is not None and (
            not isinstance(self.basis_group_id, str) or len(self.basis_group_id) != 64
            or any(c not in "0123456789abcdef" for c in self.basis_group_id)
        ):
            raise ConfigurationError("basis_group_id must be a SHA256 hex digest")
        if self.basis_version_ref is not None:
            object_ref(self.basis_version_ref, "basis version reference")

    # -- construction -------------------------------------------------------------------------------
    @classmethod
    def fresh(cls, entry: Any, *, channels: int, rank: int, world_size: int, rotation: int | None = None,
              block_rows: int | None = None, basis_group_id: str | None = None,
              basis_version_ref: str | None = None) -> FactorShardSpec:
        """Ownership of a new basis group (a fresh fit or a refit).

        ``block_rows=None`` (default) = whole-matrix owner (one block of ``channels`` rows, so the matrix
        keeps one owner forever); ``PLAN_BLOCK_ROWS`` = the plan's 32-row grid (ablation).
        """
        key, stream = entry_parts(entry)
        _int(rank, "rank", minimum=1)
        _int(world_size, "world_size", minimum=1)
        _int(channels, "channels", minimum=1)
        rotation = default_rotation(entry, world_size) if rotation is None else rotation
        block_rows = channels if block_rows is None else block_rows
        _int(block_rows, "block_rows", minimum=1)
        owners = tuple((block + rotation) % world_size for block in range(math.ceil(rank / block_rows)))
        return cls(key=key, channels=channels, world_size=world_size, version_rows=(rank,),
                   block_owners=owners, rank_view=rank, rotation=rotation, block_rows=block_rows,
                   stream=stream, basis_group_id=basis_group_id, basis_version_ref=basis_version_ref)

    def extend(self, added_rows: int, *, rank_view: int | None = None,
               basis_version_ref: str | None = None) -> FactorShardSpec:
        """Append an immutable row block (augmentation); every existing row keeps its owner."""
        _int(added_rows, "added_rows", minimum=1)
        total = self.total_rank + added_rows
        if total > self.channels:
            raise ConfigurationError("an extension cannot exceed the channel count")
        blocks = math.ceil(total / self.block_rows)
        owners = self.block_owners + tuple(
            (block + self.rotation) % self.world_size for block in range(len(self.block_owners), blocks)
        )
        return FactorShardSpec(
            key=self.key, channels=self.channels, world_size=self.world_size,
            version_rows=self.version_rows + (total,), block_owners=owners,
            rank_view=total if rank_view is None else rank_view, rotation=self.rotation,
            block_rows=self.block_rows, stream=self.stream, basis_group_id=self.basis_group_id,
            basis_version_ref=basis_version_ref,
        )

    def with_view(self, rank_view: int) -> FactorShardSpec:
        """Same basis version and owners, another legal prefix view (trim or regrow inside the version)."""
        return FactorShardSpec(
            key=self.key, channels=self.channels, world_size=self.world_size, version_rows=self.version_rows,
            block_owners=self.block_owners, rank_view=rank_view, rotation=self.rotation,
            block_rows=self.block_rows, stream=self.stream, basis_group_id=self.basis_group_id,
            basis_version_ref=self.basis_version_ref,
        )

    # -- geometry -----------------------------------------------------------------------------------
    @property
    def entry(self) -> Any:
        return self.key if self.stream is None else (self.key, self.stream)

    @property
    def total_rank(self) -> int:
        return self.version_rows[-1]

    @property
    def whole_matrix(self) -> bool:
        """One block can hold every row the matrix can ever have: a single stable owner."""
        return self.block_rows >= self.channels

    @property
    def blocks(self) -> int:
        return len(self.block_owners)

    def block_range(self, block: int) -> tuple[int, int]:
        _int(block, "block")
        if block >= self.blocks:
            raise ConfigurationError("block is outside the basis version")
        start = block * self.block_rows
        return start, min(start + self.block_rows, self.total_rank)

    def owner_of(self, row: int) -> int:
        _int(row, "row")
        if row >= self.total_rank:
            raise ConfigurationError("row is outside the basis version")
        return self.block_owners[row // self.block_rows]

    def _check_owner(self, owner: int) -> None:
        if type(owner) is not int or not 0 <= owner < self.world_size:
            raise ConfigurationError("owner is outside the factor group")

    def basis_ranges(self, owner: int) -> tuple[tuple[int, int], ...]:
        """Basis rows ``[start, stop)`` stored by ``owner`` (whole version)."""
        self._check_owner(owner)
        return _merge(self.block_range(block) for block, item in enumerate(self.block_owners) if item == owner)

    def view_ranges(self, owner: int) -> tuple[tuple[int, int], ...]:
        """Rows of the rank view owned by ``owner``: its coefficient columns, in global row order."""
        return tuple((start, min(stop, self.rank_view)) for start, stop in self.basis_ranges(owner)
                     if start < self.rank_view)

    def view_rows(self, owner: int) -> int:
        return ranges_rows(self.view_ranges(owner))

    @property
    def view_loads(self) -> tuple[int, ...]:
        return tuple(self.view_rows(owner) for owner in range(self.world_size))

    @property
    def basis_loads(self) -> tuple[int, ...]:
        return tuple(ranges_rows(self.basis_ranges(owner)) for owner in range(self.world_size))

    def segments(self) -> tuple[tuple[int, int, int], ...]:
        """Immutable basis segments ``(start, stop, owner)``: grid blocks cut at version boundaries."""
        cuts = sorted({0, *self.version_rows, *range(0, self.total_rank, self.block_rows)})
        cuts = [cut for cut in cuts if cut <= self.total_rank]
        return tuple((start, stop, self.owner_of(start)) for start, stop in zip(cuts, cuts[1:]))

    def owner_segments(self, owner: int) -> tuple[tuple[int, int], ...]:
        self._check_owner(owner)
        return tuple((start, stop) for start, stop, item in self.segments() if item == owner)

    def check_partition(self) -> None:
        """Owner view ranges partition ``[0, rank_view)`` exactly (no overlap, no gap, no padding)."""
        rows = sorted(item for owner in range(self.world_size) for item in self.view_ranges(owner))
        cursor = 0
        for start, stop in rows:
            if start != cursor or stop <= start:
                raise IntegrityError("factor-shard owner ranges do not partition the rank view")
            cursor = stop
        if cursor != self.rank_view:
            raise IntegrityError("factor-shard owner ranges do not cover the rank view")

    def check_stable(self, parent: FactorShardSpec) -> None:
        """Every row shared with ``parent`` keeps its owner (augmentation/fork invariant)."""
        if (parent.key, parent.stream, parent.channels, parent.world_size, parent.block_rows) != (
            self.key, self.stream, self.channels, self.world_size, self.block_rows
        ):
            raise IntegrityError("specs describe different matrices or groups")
        shared = min(parent.total_rank, self.total_rank)
        if self.version_rows[:len(parent.version_rows)] != parent.version_rows and shared == parent.total_rank:
            raise IntegrityError("an extension rewrote its parent's version boundaries")
        for block in range(math.ceil(shared / self.block_rows)):
            if parent.block_owners[block] != self.block_owners[block]:
                raise IntegrityError(f"basis row block {block} moved from owner "
                                     f"{parent.block_owners[block]} to {self.block_owners[block]}")

    # -- serialization ------------------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SPEC_SCHEMA, "entry": entry_to_dict(self.entry), "channels": self.channels,
            "world_size": self.world_size, "version_rows": list(self.version_rows),
            "block_owners": list(self.block_owners), "rank_view": self.rank_view, "rotation": self.rotation,
            "block_rows": self.block_rows, "basis_group_id": self.basis_group_id,
            "basis_version_ref": self.basis_version_ref,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> FactorShardSpec:
        try:
            if data["schema"] != SPEC_SCHEMA:
                raise ConfigurationError("unsupported factor-shard spec schema")
            key, stream = entry_parts(entry_from_dict(data["entry"]))
            spec = cls(key=key, channels=data["channels"], world_size=data["world_size"],
                       version_rows=tuple(data["version_rows"]), block_owners=tuple(data["block_owners"]),
                       rank_view=data["rank_view"], rotation=data["rotation"], block_rows=data["block_rows"],
                       stream=stream, basis_group_id=data["basis_group_id"],
                       basis_version_ref=data["basis_version_ref"])
        except (KeyError, TypeError) as exc:
            raise ConfigurationError("invalid factor-shard spec fields") from exc
        spec.check_partition()
        return spec

    @property
    def digest(self) -> str:
        return canonical_digest(self.to_dict())


def split_factors(spec: FactorShardSpec, coefficients: torch.Tensor, basis: torch.Tensor
                  ) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    """Reference split of full factors into per-owner ``(P_s, Q_s)`` (view columns, view basis rows)."""
    if coefficients.ndim != 2 or coefficients.shape[1] != spec.rank_view:
        raise ConfigurationError("coefficients must be [tokens, rank_view]")
    if basis.ndim != 2 or basis.shape[0] < spec.rank_view or basis.shape[1] != spec.channels:
        raise ConfigurationError("basis must hold at least rank_view rows of the spec's channels")
    result = {}
    for owner in range(spec.world_size):
        index = ranges_index(spec.view_ranges(owner), device=coefficients.device)
        result[owner] = (coefficients.index_select(1, index).contiguous(),
                         basis.index_select(0, index.to(basis.device)).contiguous())
    return result


def spec_for_basis_version(entry: Any, versions, version_ref: str, *, world_size: int, rank_view: int | None = None,
                           rotation: int | None = None, block_rows: int | None = None) -> FactorShardSpec:
    """Ownership of a runtime :class:`~reelcache.types.BasisVersion` (``basis_versions.BasisVersionStore``).

    ``version_rows`` are the cumulative ranks of the version's chain (root first), so a fork or a later node
    that extends a shared ancestor derives exactly the same owners for every shared row (owners depend only
    on the block index and the matrix's rotation, never on the total rank). ``block_rows=None`` (default)
    = whole-matrix owner; ``PLAN_BLOCK_ROWS`` = the plan's 32-row grid.
    """
    from .basis_versions import BasisVersionStore

    if not isinstance(versions, BasisVersionStore):
        raise ConfigurationError("versions must be the runtime's BasisVersionStore")
    key, stream = entry_parts(entry_key(entry))
    chain = versions.chain(version_ref)
    head = chain[-1][1]
    if head.key != key:
        raise ConfigurationError("basis version belongs to another matrix")
    rows = tuple(version.total_rank for _, version in chain)
    rotation = default_rotation(entry, world_size) if rotation is None else rotation
    block_rows = head.channels if block_rows is None else block_rows
    _int(block_rows, "block_rows", minimum=1)
    owners = tuple((block + rotation) % world_size for block in range(math.ceil(rows[-1] / block_rows)))
    return FactorShardSpec(key=key, channels=head.channels, world_size=world_size, version_rows=rows,
                           block_owners=owners, rank_view=rows[-1] if rank_view is None else rank_view,
                           rotation=rotation, block_rows=block_rows, stream=stream, basis_group_id=head.group_id,
                           basis_version_ref=version_ref)


def row_shard_gram(local_rows: torch.Tensor, *, layout: str) -> torch.Tensor:
    """FP64 Gram of a token-row shard; summing these over row owners equals ``X^T X``.

    Head/channel shards are rejected: their Grams lack the cross-channel blocks ``X_a^T X_b``.
    """
    if layout != "rows":
        raise ConfigurationError("summing per-shard Grams is valid only for token-row shards; channel/head "
                                 "shards drop the cross-channel terms (gather or use the owner-fit reference)")
    value = local_rows.detach().double()
    return value.T @ value


# ------------------------------------------------------------------------------------- native owners

class NativeOwners:
    """Native KV ownership a restore targets: head (channel) shards or token-row shards.

    ``kind="heads"``: rank ``r`` owns KV channels ``layout.columns(r)`` of every token (Ulysses/TP).
    ``kind="rows"``: rank ``r`` owns token rows ``row_index[r]`` of every channel (contiguous
    ``layout.rows(r)`` by default, or an explicit partition such as a 2-D spatial split).
    """

    KINDS = ("heads", "rows")

    def __init__(self, layout: KVLayout, kind: str, row_index: Sequence[Sequence[int]] | None = None):
        if not isinstance(layout, KVLayout):
            raise ConfigurationError("native owners require a KVLayout")
        if kind not in self.KINDS:
            raise ConfigurationError(f"native owner kind must be one of {self.KINDS}")
        if kind == "heads" and row_index is not None:
            raise ConfigurationError("head-sharded owners hold every token row")
        self.layout, self.kind = layout, kind
        if kind == "rows":
            if row_index is None:
                row_index = [range(layout.rows(rank).start, layout.rows(rank).stop)
                             for rank in range(layout.world_size)]
            rows = [torch.as_tensor(list(item), dtype=torch.long) for item in row_index]
            if len(rows) != layout.world_size:
                raise ConfigurationError("row_index needs one row set per rank")
            flat = torch.cat(rows) if rows else torch.empty(0, dtype=torch.long)
            if flat.numel() != layout.tokens or not torch.equal(torch.sort(flat).values,
                                                                  torch.arange(layout.tokens)):
                raise ConfigurationError("row_index must partition the native token rows exactly")
            if any(item.numel() == 0 for item in rows):
                raise ConfigurationError("every row owner needs at least one token")
            self.row_index = tuple(rows)
        else:
            self.row_index = None

    @classmethod
    def spatial_rows(cls, *, frames: int, height: int, width: int, split_h: int, split_w: int,
                     kv_heads: int, head_dim: int) -> NativeOwners:
        """LongCat-Video context-parallel rows: rank ``hi * split_w + wi`` owns spatial block ``(hi, wi)``."""
        if height % split_h or width % split_w:
            raise ConfigurationError("spatial split must divide the latent height and width")
        world = split_h * split_w
        index = torch.arange(frames * height * width).reshape(frames, height, width)
        hs, ws = height // split_h, width // split_w
        rows = []
        for rank in range(world):
            hi, wi = divmod(rank, split_w)
            rows.append(index[:, hi * hs:(hi + 1) * hs, wi * ws:(wi + 1) * ws].reshape(-1).tolist())
        layout = KVLayout(tokens=frames * height * width, kv_heads=kv_heads, head_dim=head_dim, world_size=world)
        return cls(layout, "rows", rows)

    @property
    def world_size(self) -> int:
        return self.layout.world_size

    @property
    def tokens(self) -> int:
        return self.layout.tokens

    @property
    def channels(self) -> int:
        return self.layout.channels

    def columns(self, rank: int) -> slice:
        return self.layout.columns(rank) if self.kind == "heads" else slice(0, self.channels)

    def rows(self, rank: int) -> torch.Tensor:
        if self.kind == "rows":
            return self.row_index[rank]
        return torch.arange(self.tokens, dtype=torch.long)

    def shape(self, rank: int) -> tuple[int, int]:
        if self.kind == "rows":
            return int(self.row_index[rank].numel()), self.channels
        return self.tokens, self.layout.local_channels

    def to_dict(self) -> dict[str, Any]:
        result = {"kind": self.kind, "tokens": self.tokens, "kv_heads": self.layout.kv_heads,
                  "head_dim": self.layout.head_dim, "world_size": self.world_size}
        if self.kind == "rows":
            result["rows_sha256"] = [_sha(item.numpy().astype("<i8").tobytes()) for item in self.row_index]
        return result

    def tiles(self, tile_tokens: int) -> int:
        """Reduce-scatter steps: token tiles of the longest owner row set (heads: of all tokens)."""
        longest = max(self.shape(rank)[0] for rank in range(self.world_size)) if self.kind == "rows" else self.tokens
        return math.ceil(longest / tile_tokens)


# ------------------------------------------------------------------------------------ fragment packs

def _descriptor(raw: bytes, dtype: str, shape: Sequence[int]) -> dict[str, Any]:
    return {"bytes": len(raw), "sha256": _sha(raw), "dtype": dtype, "shape": list(shape)}


def _le_bytes(tensor: torch.Tensor) -> bytes:
    if sys.byteorder != "little":
        raise ConfigurationError("factor fragments require a little-endian host")
    value = tensor.detach().contiguous().cpu()
    return value.view(torch.uint8).numpy().tobytes()


def basis_bytes(rows: torch.Tensor) -> bytes:
    if not isinstance(rows, torch.Tensor) or rows.dtype != torch.float16 or rows.ndim != 2:
        raise ConfigurationError("basis rows are FP16 [rows, channels] tensors")
    return _le_bytes(rows)


def _hevc_codec(codec: CodecSpec) -> dict[str, Any]:
    if not isinstance(codec, CodecSpec) or codec.kind != "svd_hevc":
        raise ConfigurationError("HEVC fragment packs need an svd_hevc CodecSpec")
    if codec.inter_prediction is not False:
        raise ConfigurationError("fragment packs are all-intra; inter prediction is rejected (as for node packs)")
    _hevc.validate_parameters(codec.pix_fmt, codec.qp, codec.preset)
    return {"pix_fmt": codec.pix_fmt, "qp": codec.qp, "preset": codec.preset, "inter_prediction": False,
            "normalization": codec.normalization, "protected_fraction": codec.protected_fraction}


@dataclass(frozen=True)
class FragmentInput:
    """One matrix's owner fragment: the owner's view coefficient columns and view basis rows."""

    entry: Any
    spec: FactorShardSpec
    context: TensorContext
    coefficients: torch.Tensor
    basis: torch.Tensor

    def __post_init__(self) -> None:
        entry_key(self.entry, "fragment entry")
        if not isinstance(self.spec, FactorShardSpec) or self.spec.entry != self.entry:
            raise ConfigurationError("fragment spec must describe the fragment's entry")
        if not isinstance(self.context, TensorContext) or self.context.key != self.spec.key:
            raise ConfigurationError("fragment context must describe the spec's matrix")
        if self.context.channels != self.spec.channels:
            raise ConfigurationError("fragment context and spec disagree on channels")
        if self.context.key.target == "K" and self.context.key_space != "pre_rope":
            raise ConfigurationError("factor fragments of K are normalized pre-RoPE K only")
        rows = self.coefficients.shape[1] if self.coefficients.ndim == 2 else -1
        if (self.coefficients.dtype != torch.float32 or self.coefficients.ndim != 2
                or self.coefficients.shape[0] != self.context.token_count or rows < 1):
            raise ConfigurationError("fragment coefficients must be FP32 [tokens, owned rows >= 1]")
        if tuple(self.basis.shape) != (rows, self.spec.channels) or self.basis.dtype != torch.float16:
            raise ConfigurationError("fragment basis rows must be FP16 [owned rows, channels]")


def _placements(videos: Sequence[Mapping[str, Any]]) -> tuple[int, int, int, list]:
    """Temporal canvas (CTU aligned, >= NVDEC minimum) and per-tile (pictures, rect)."""
    width = max(_hevc.align(max(item["image_width"] for item in videos)), _hevc.MIN_CANVAS)
    height = max(_hevc.align(max(item["image_height"] for item in videos)), _hevc.MIN_CANVAS)
    if width > _hevc.MAX_DIMENSION or height > _hevc.MAX_DIMENSION:
        raise UnsupportedError("fragment canvas exceeds the HEVC provider's maximum picture dimension")
    placements, cursor = [], 0
    for item in videos:
        count = item["frames"] * item["segments"]
        placements.append(([cursor, count], [0, 0, item["image_width"], item["image_height"]]))
        cursor += count
    return width, height, cursor, placements


def encode_fragment_pack(node_id: str, owner: int, world_size: int, fragments: Sequence[FragmentInput], *,
                         kind: str = "svd_hevc", codec: CodecSpec | None = None, ffmpeg_path: str | None = None,
                         threads: int = 1, validate: bool = False, empty_entries: Sequence[Any] = ()
                         ) -> EncodedTensor:
    """Encode ONE owner's coefficient columns of several matrices of ONE node as an independent pack.

    ``svd_hevc``: per-channel quantization of exactly the owner's columns, channel-tile mosaics placed
    temporally on one CTU-aligned canvas, ONE all-intra x265 stream with per-tile byte ranges (an
    independently encoded field subset, decodable without any other owner's stream). ``svd``: exact FP32
    coefficient payloads. Basis rows are not embedded: each tile binds the SHA256 of its view rows, which
    the owner stores as content-addressed segments (:class:`OwnerFactorStore`).
    """
    if not isinstance(node_id, str) or not node_id.strip():
        raise ConfigurationError("node_id must be a nonempty string")
    _int(world_size, "world_size", minimum=1)
    if type(owner) is not int or not 0 <= owner < world_size:
        raise ConfigurationError("owner is outside the group")
    if kind not in FRAGMENT_KINDS:
        raise ConfigurationError(f"fragment kind must be one of {FRAGMENT_KINDS}")
    if type(validate) is not bool:
        raise ConfigurationError("validate must be boolean")
    items = sorted(fragments, key=lambda item: entry_order(item.entry))
    if not items:
        raise ConfigurationError("a fragment pack needs at least one fragment with owned rows")
    if len({entry_order(item.entry) for item in items}) != len(items):
        raise ConfigurationError("duplicate fragment entry")
    started = time.perf_counter()
    first = items[0].context
    for item in items:
        if not isinstance(item, FragmentInput):
            raise ConfigurationError("fragments must be FragmentInput values")
        if item.spec.world_size != world_size:
            raise ConfigurationError("fragment spec belongs to another group size")
        if (item.context.model != first.model or item.context.token_start != first.token_start or
                (item.context.frames, item.context.height, item.context.width)
                != (first.frames, first.height, first.width)):
            raise ConfigurationError("a fragment pack cannot span two nodes")
        if item.coefficients.shape[1] != item.spec.view_rows(owner):
            raise ConfigurationError("fragment columns differ from the owner's view rows")
    tiles, payloads, descriptors = [], {}, {}
    hevc_codec = _hevc_codec(codec) if kind == "svd_hevc" else None
    quantized = []
    for index, item in enumerate(items):
        ranges = item.spec.view_ranges(owner)
        raw_basis = basis_bytes(item.basis)
        tile = {
            "entry": entry_to_dict(item.entry), "spec_sha256": item.spec.digest,
            "context": item.context.to_dict(), "ranges": [list(r) for r in ranges], "rows": ranges_rows(ranges),
            "coefficient_shape": list(item.coefficients.shape),
            "basis": {"sha256": _sha(raw_basis), "bytes": len(raw_basis), "shape": list(item.basis.shape),
                      "dtype": "float16"},
        }
        values = item.coefficients.detach().cpu().contiguous()
        if not bool(torch.isfinite(values).all()):
            raise ConfigurationError("fragment coefficients must be finite")
        if kind == "svd":
            raw = _le_bytes(values)
            name = f"coefficients.{index:03d}.f32"
            payloads[name] = raw
            descriptors[name] = _descriptor(raw, "float32", values.shape)
            tile["coefficients"] = name
        else:
            pixels, affine, normalization = _hevc.quantize_coefficients(
                values.numpy(), codec.normalization, codec.protected_fraction)
            video = _hevc.video_layout(item.context.frames, item.context.height, item.context.width,
                                       tile["rows"], codec.pix_fmt)
            raw = affine.astype("<f4", copy=False).tobytes()
            name = f"affine.{index:03d}.f32"
            payloads[name] = raw
            descriptors[name] = _descriptor(raw, "float32", normalization["affine_shape"])
            tile.update(affine=name, normalization=normalization, video=video, pictures=None, rect=None,
                        byte_range=None, decoded_coefficients_sha256=None)
            quantized.append((pixels, affine, normalization, video))
        tiles.append(tile)
    canvas, encode_seconds, version, decode_seconds = None, 0.0, None, 0.0
    if kind == "svd_hevc":
        width, height, pictures, placements = _placements([item[3] for item in quantized])
        planes = 3 if codec.pix_fmt == "yuv444p" else 1
        images = np.zeros((pictures, planes, height, width), dtype=np.uint8)
        for tile, (pixels, _, _, video), (span, rect) in zip(tiles, quantized, placements):
            tile["pictures"], tile["rect"] = span, rect
            values = pixels.reshape(video["frames"], video["height"], video["width"], video["rank"])
            for segment in range(video["segments"]):
                at = span[0] + segment * video["frames"]
                images[at:at + video["frames"], :, :rect[3], :rect[2]] = _hevc._mosaic(values, video, segment)
        stream, encode_seconds, version = _hevc.encode_pictures(images, codec.pix_fmt, ffmpeg_path=ffmpeg_path,
                                                                threads=threads)
        units = _hevc.all_intra_access_units(stream, pictures)
        random_access = all(unit["self_contained"] for unit in units)
        for tile in tiles:
            if random_access:
                start, count = tile["pictures"]
                head, tail = units[start], units[start + count - 1]
                tile["byte_range"] = [head["offset"], tail["offset"] + tail["length"] - head["offset"]]
        if validate:
            decoded, decode_seconds = _hevc.decode_pictures(stream, pictures=pictures, height=height, width=width,
                                                            pix_fmt=codec.pix_fmt, ffmpeg_path=ffmpeg_path)
            for tile, (_, affine, normalization, video) in zip(tiles, quantized):
                restored = _unpack_tile(decoded, tile, tile["pictures"][0], video)
                coefficients = _hevc.dequantize_coefficients(restored, affine, normalization)
                tile["decoded_coefficients_sha256"] = _sha(coefficients.astype("<f4", copy=False).tobytes())
            del decoded
        del images
        payloads["stream.hevc"] = stream
        descriptors["stream.hevc"] = _descriptor(stream, "uint8", (len(stream),))
        canvas = {"width": width, "height": height, "pictures": pictures, "planes": planes,
                  "pix_fmt": codec.pix_fmt, "alignment": _hevc.CANVAS_ALIGNMENT, "fill": 0,
                  "random_access": "tile_byte_ranges" if random_access else "whole_stream"}
    metadata = {
        "schema": FRAGMENT_SCHEMA, "byte_order": "little", "node_id": node_id, "owner": owner,
        "world_size": world_size, "kind": kind, "codec": hevc_codec,
        "hevc": _hevc.encoder_record(False) if kind == "svd_hevc" else None, "canvas": canvas, "tiles": tiles,
        "empty_entries": [entry_to_dict(entry) for entry in sorted(empty_entries, key=entry_order)],
        "payloads": descriptors,
    }
    metrics = {
        "tiles": len(tiles), "coefficient_values": int(sum(math.prod(t["coefficient_shape"]) for t in tiles)),
        "payload_bytes": int(sum(map(len, payloads.values()))), "hevc_encode_seconds": encode_seconds,
        "hevc_encode_threads": threads if kind == "svd_hevc" else 0, "ffmpeg_version": version,
        "validation_decode_seconds": decode_seconds, "encode_seconds": time.perf_counter() - started,
    }
    return EncodedTensor(metadata, payloads, None, metrics)


def _unpack_tile(images: np.ndarray, tile: Mapping[str, Any], first: int, video: Mapping[str, Any]) -> np.ndarray:
    x, y, width, height = tile["rect"]
    values = np.empty((video["frames"], video["height"], video["width"], video["rank"]), dtype=np.uint8)
    for segment in range(video["segments"]):
        at = first + segment * video["frames"]
        _hevc._unmosaic(images[at:at + video["frames"], :, y:y + height, x:x + width], video, segment, values)
    return values.reshape(-1, video["rank"])


def parse_fragment_pack(encoded: EncodedTensor, *, verify: bool = True) -> dict[str, Any]:
    """Structural validation (always); with ``verify`` also every payload's SHA-256 and the stream's all-intra
    access units. The restore path reads packs whose container address was just checked and parses them with
    ``verify=False``; writers and audits keep ``verify=True``."""
    if not isinstance(encoded, EncodedTensor):
        raise ConfigurationError("fragment pack must be an EncodedTensor")
    meta = encoded.metadata
    try:
        if meta["schema"] != FRAGMENT_SCHEMA or meta["byte_order"] != "little" or meta["kind"] not in FRAGMENT_KINDS:
            raise IntegrityError("unsupported factor fragment pack schema")
        owner, world = meta["owner"], meta["world_size"]
        if type(world) is not int or world < 1 or type(owner) is not int or not 0 <= owner < world:
            raise IntegrityError("fragment pack owner is outside its group")
        descriptors = meta["payloads"]
        if not isinstance(descriptors, Mapping) or set(descriptors) != set(encoded.payloads):
            raise IntegrityError("fragment pack has missing or uncharged payloads")
        for name, row in descriptors.items():
            raw = encoded.payloads[name]
            if row["bytes"] != len(raw):
                raise IntegrityError("fragment payload length differs from its descriptor")
            if verify and row["sha256"] != _sha(raw):
                raise IntegrityError(f"fragment payload SHA256 mismatch: {name}")
        tiles, entries = [], set()
        for tile in meta["tiles"]:
            entry = entry_from_dict(tile["entry"])
            if entry_order(entry) in entries:
                raise IntegrityError("duplicate fragment tile entry")
            entries.add(entry_order(entry))
            context = TensorContext.from_dict(tile["context"])
            ranges = tuple((int(a), int(b)) for a, b in tile["ranges"])
            rows = ranges_rows(ranges)
            if rows != tile["rows"] or rows < 1 or tile["coefficient_shape"] != [context.token_count, rows]:
                raise IntegrityError("fragment tile rows/ranges/coefficient shape disagree")
            if tile["basis"]["shape"] != [rows, context.channels] or tile["basis"]["bytes"] != rows * context.channels * 2:
                raise IntegrityError("fragment tile basis binding disagrees with its rows")
            if meta["kind"] == "svd":
                row = descriptors[tile["coefficients"]]
                if row["shape"] != [context.token_count, rows] or row["bytes"] != context.token_count * rows * 4:
                    raise IntegrityError("FP32 fragment payload disagrees with its tile")
            else:
                video = tile["video"]
                expected = _hevc.video_layout(context.frames, context.height, context.width, rows,
                                              meta["codec"]["pix_fmt"])
                if json.dumps(expected, sort_keys=True) != json.dumps(video, sort_keys=True):
                    raise IntegrityError("fragment tile video layout differs from its rows")
                canvas = meta["canvas"]
                first, count = tile["pictures"]
                x, y, width, height = tile["rect"]
                if (count != video["frames"] * video["segments"] or first < 0 or first + count > canvas["pictures"]
                        or x + width > canvas["width"] or y + height > canvas["height"]
                        or (width, height) != (video["image_width"], video["image_height"])):
                    raise IntegrityError("fragment tile lies outside its canvas")
                affine = descriptors[tile["affine"]]
                if affine["shape"] != tile["normalization"]["affine_shape"]:
                    raise IntegrityError("fragment affine payload disagrees with its normalization")
            tiles.append((entry, context, ranges, tile))
        if meta["kind"] == "svd_hevc":
            stream = encoded.payloads["stream.hevc"]
            if verify:
                _hevc.all_intra_access_units(stream, meta["canvas"]["pictures"])
    except IntegrityError:
        raise
    except (KeyError, TypeError, ValueError, ConfigurationError) as exc:
        raise IntegrityError(f"malformed factor fragment pack: {exc}") from exc
    return {"meta": meta, "tiles": tiles, "index": {entry_order(item[0]): item for item in tiles}}


def _select(parsed: Mapping[str, Any], entries: Iterable[Any] | None) -> list:
    if entries is None:
        return list(parsed["tiles"])
    chosen = []
    for entry in entries:
        item = parsed["index"].get(entry_order(entry_key(entry)))
        if item is None:
            raise ConfigurationError(f"fragment pack holds no tile for {entry_label(entry)}")
        chosen.append(item)
    return sorted(chosen, key=lambda item: item[3]["pictures"][0] if item[3].get("pictures") else 0)


def fragment_units(encoded: EncodedTensor, parsed: Mapping[str, Any], selected: Sequence) -> tuple[list, int, dict]:
    """NVDEC access units of the selected tiles (byte ranges) and each tile's first canvas picture."""
    stream = encoded.payloads["stream.hevc"]
    if not all(item[3]["byte_range"] is not None for item in selected):
        units = [(stream, unit["offset"], unit["length"], 1)
                 for unit in _hevc.all_intra_access_units(stream, parsed["meta"]["canvas"]["pictures"])]
        return units, parsed["meta"]["canvas"]["pictures"], {
            entry_order(item[0]): item[3]["pictures"][0] for item in selected}
    units, offsets, pictures = [], {}, 0
    for entry, _, _, tile in selected:
        offset, length = tile["byte_range"]
        count = tile["pictures"][1]
        units.extend((stream, start, size, 1) for start, size in _hevc.picture_units(stream, offset, length, count))
        offsets[entry_order(entry)] = pictures
        pictures += count
    return units, pictures, offsets


def decode_fragment_pack(encoded: EncodedTensor, *, entries: Iterable[Any] | None = None, device="cpu",
                         pool=None, verify: bool = True, ffmpeg_path: str | None = None,
                         parsed: Mapping[str, Any] | None = None) -> dict[Any, torch.Tensor]:
    """Owner-local decode of a fragment pack to FP32 coefficient columns ``{entry: [tokens, rows]}``.

    CPU: software FFmpeg decode of the selected tiles' self-contained pictures. CUDA: an explicit
    :class:`reelcache.decoder_pool.DecoderPool` on the output device decodes the same access units into a
    device canvas (no software fallback). Coefficients are dequantized exactly as ReelCache packets.
    """
    device = torch.device(device)
    if pool is not None and device.type != "cuda":
        raise ConfigurationError("a DecoderPool decodes to CUDA devices only")
    if parsed is None:
        parsed = parse_fragment_pack(encoded, verify=verify)
    meta = parsed["meta"]
    selected = _select(parsed, entries)
    result = {}
    if meta["kind"] == "svd":
        for entry, context, _, tile in selected:
            raw = encoded.payloads[tile["coefficients"]]
            value = torch.frombuffer(bytearray(raw), dtype=torch.float32).reshape(context.token_count, tile["rows"])
            result[entry] = value.to(device)
        return result
    canvas = meta["canvas"]
    if device.type == "cuda":
        return submit_fragment_decode(encoded, pool=pool, device=device, entries=entries, verify=verify,
                                      parsed=parsed).result()
    stream = encoded.payloads["stream.hevc"]
    partial = all(item[3]["byte_range"] is not None for item in selected) and len(selected) < len(parsed["tiles"])
    if partial:
        pieces, offsets, pictures = [], {}, 0
        for entry, _, _, tile in selected:
            offset, length = tile["byte_range"]
            pieces.append(stream[offset:offset + length])
            offsets[entry_order(entry)] = pictures
            pictures += tile["pictures"][1]
        source = b"".join(pieces)
    else:
        source, pictures = stream, canvas["pictures"]
        offsets = {entry_order(entry): tile["pictures"][0] for entry, _, _, tile in selected}
    images, _ = _hevc.decode_pictures(source, pictures=pictures, height=canvas["height"], width=canvas["width"],
                                      pix_fmt=canvas["pix_fmt"], ffmpeg_path=ffmpeg_path)
    for entry, _, _, tile in selected:
        values = _unpack_tile(images, tile, offsets[entry_order(entry)], tile["video"])
        affine = np.frombuffer(encoded.payloads[tile["affine"]], dtype="<f4").reshape(
            tile["normalization"]["affine_shape"])
        decoded = _hevc.dequantize_coefficients(values, affine, tile["normalization"])
        digest = tile["decoded_coefficients_sha256"]
        if digest is not None and _sha(decoded.astype("<f4", copy=False).tobytes()) != digest:
            raise IntegrityError("fragment coefficients differ from their validated checksum")
        result[entry] = torch.from_numpy(decoded).to(device)
    return result


class PendingFragmentDecode:
    """An NVDEC decode of one fragment pack in flight; :meth:`result` dequantizes on the caller's stream."""

    def __init__(self, encoded, selected, offsets, images, ticket, verify, arena=None):
        self._encoded, self._selected, self._offsets = encoded, selected, offsets
        self._images, self._ticket, self._verify, self._arena = images, ticket, verify, arena
        self._result = None

    def result(self, stream=None) -> dict[Any, torch.Tensor]:
        from .codecs import dequantize_device_coefficients
        from .nvdec import unpack_device_tile

        if self._result is not None:
            return self._result
        stream = torch.cuda.current_stream(self._images.device) if stream is None else stream
        self._ticket.wait(stream)
        result = {}
        with torch.cuda.stream(stream):
            for entry, _, _, tile in self._selected:
                pixels = unpack_device_tile(self._images, self._offsets[entry_order(entry)], tile["rect"], tile["video"])
                affine = np.frombuffer(self._encoded.payloads[tile["affine"]], dtype="<f4").reshape(
                    tile["normalization"]["affine_shape"])
                result[entry] = dequantize_device_coefficients(pixels, affine, tile["normalization"],
                                                               validate=self._verify)
        if self._arena is not None:
            done = torch.cuda.Event()
            done.record(stream)
            self._arena.release(self._images, after=[done])
        else:
            self._images.record_stream(stream)
        self._result, self._images = result, None
        return result

    def discard(self, timeout: float | None = 60.0) -> None:
        """Abandon an unconsumed decode (e.g. a failed load): an arena canvas returns to the arena once the
        decoder and any read a failed :meth:`result` queued on the current stream are done with it.

        A job not issued within ``timeout`` may still write the canvas later, so that canvas is never handed out
        again (it stays leased); a failed job no longer writes it. A canvas without an arena is protected by the
        decoder's ``record_stream`` and is simply dropped."""
        images, self._images = self._images, None
        if images is None or self._arena is None:
            return
        try:
            event = self._ticket.event(timeout)
        except TimeoutError:
            return
        except Exception:  # noqa: BLE001 - the job failed: nothing writes the canvas any more
            event = None
        reads = torch.cuda.Event()
        reads.record(torch.cuda.current_stream(images.device))
        self._arena.release(images, after=[event, reads])


def submit_fragment_decode(encoded: EncodedTensor, *, pool, device, entries: Iterable[Any] | None = None,
                           verify: bool = True, parsed: Mapping[str, Any] | None = None,
                           arena=None) -> PendingFragmentDecode:
    """Queue one HEVC fragment pack on an explicit :class:`reelcache.decoder_pool.DecoderPool` (never blocks on
    decoding), so an owner can keep several packs in flight on several NVDEC sessions/engines.

    Canvas ordering: with a :class:`reelcache.decoder_pool.CanvasArena` the canvas has no pending users and the
    decode starts at once (it is recycled after the dequantization event); without one, a fresh allocation
    may alias memory that work still queued on the current stream uses, so the decode waits for that stream.
    """
    device = torch.device(device)
    if device.type != "cuda" or pool is None:
        raise ConfigurationError("NVDEC fragment decode needs a CUDA device and an explicit DecoderPool")
    if parsed is None:
        parsed = parse_fragment_pack(encoded, verify=verify)
    if parsed["meta"]["kind"] != "svd_hevc":
        raise ConfigurationError("only svd_hevc fragment packs carry an HEVC stream")
    canvas = parsed["meta"]["canvas"]
    selected = _select(parsed, entries)
    units, pictures, offsets = fragment_units(encoded, parsed, selected)
    shape = (pictures, canvas["planes"], canvas["height"], canvas["width"])
    if arena is not None:
        images = arena.acquire(shape)
        ticket = pool.submit(units, images, pix_fmt=canvas["pix_fmt"], after=None)
    else:
        images = torch.empty(shape, dtype=torch.uint8, device=device)
        ticket = pool.submit(units, images, pix_fmt=canvas["pix_fmt"], after="current")
    return PendingFragmentDecode(encoded, selected, offsets, images, ticket, verify, arena)


# ------------------------------------------------------------------------------------ owner storage

class OwnerFactorStore:
    """One owner's rank-local storage: basis segments (``basis:``), fragment packs (``blob:``) and
    node manifests in its own :class:`reelcache.store.ObjectStore` root. Content addressing stores a
    segment shared by consecutive nodes or forks once; nothing of another owner is ever stored here."""

    def __init__(self, store, *, owner: int, world_size: int):
        from .store import ObjectStore

        if not isinstance(store, ObjectStore):
            raise ConfigurationError("owner storage needs an ObjectStore (one root per owner)")
        _int(world_size, "world_size", minimum=1)
        if type(owner) is not int or not 0 <= owner < world_size:
            raise ConfigurationError("owner is outside the group")
        self.store, self.owner, self.world_size = store, owner, world_size

    def put_segment(self, rows: torch.Tensor) -> str:
        return self.store.put(basis_bytes(rows), kind="basis")

    def _store_verified_read(self) -> bool:
        """``ObjectStore.get`` already hash-checked the bytes it returned against their address (``verify_reads``):
        every stored byte is then checked once per read, by the store; otherwise this class checks it."""
        return getattr(self.store, "verify_reads", False) is True

    def get_segment(self, ref: str, rows: int, channels: int) -> torch.Tensor:
        object_ref(ref, "basis segment", kinds=("basis",))
        data = self.store.get(ref)
        if len(data) != rows * channels * 2 or (
                not self._store_verified_read() and _sha(data) != ref.split(":", 1)[1]):
            raise IntegrityError("basis segment bytes differ from their address or geometry")
        return torch.frombuffer(bytearray(data), dtype=torch.float16).reshape(rows, channels).clone()

    def put_pack(self, encoded: EncodedTensor) -> str:
        from .sp_kv import pack_encoded

        return self.store.put(pack_encoded(encoded), kind="blob")

    def get_pack(self, ref: str) -> EncodedTensor:
        from .sp_kv import unpack_encoded

        object_ref(ref, "fragment pack", kinds=("blob",))
        data = self.store.get(ref)
        if not self._store_verified_read() and _sha(data) != ref.split(":", 1)[1]:
            raise IntegrityError("fragment pack bytes differ from their content address")
        return unpack_encoded(data)

    def pack_size(self, ref: str) -> int:
        return self.store.object_size(ref)

    def commit(self, node_map: ShardedNodeMap) -> str:
        """Write the agreed node map and this owner's set (referencing its local objects); publish it."""
        mine = node_map.owner_record(self.owner)
        refs = [item["ref"] for item in mine["packs"]] + sorted({item["ref"] for item in mine["segments"]})
        with self.store.transaction():
            map_ref = self.store.put_manifest({"type": NODE_MAP_TYPE, "map": node_map.to_dict()})
            owner_set = self.store.put_manifest(
                {"type": OWNER_SET_TYPE, "node_id": node_map.node_id, "owner": self.owner,
                 "node_map_sha256": node_map.digest},
                references=[map_ref, *refs])
            name = REF_PREFIX + node_map.node_id
            current = self.store.get_ref(name)
            if current is not None and current != owner_set:
                raise StateError(f"node {node_map.node_id} is already published with another map")
            self.store.set_ref(name, owner_set, expected=current)
        return owner_set

    def retract(self, node_id: str, owner_set: str) -> None:
        name = REF_PREFIX + node_id
        if self.store.get_ref(name) == owner_set:
            self.store.delete_ref(name, expected=owner_set)

    def published(self, node_id: str) -> str | None:
        return self.store.get_ref(REF_PREFIX + node_id)

    def node_map(self, node_id: str) -> ShardedNodeMap:
        owner_set = self.published(node_id)
        if owner_set is None:
            raise StateError(f"node {node_id} is not published on owner {self.owner}")
        data, references = self.store.get_manifest_with_references(owner_set)
        if data.get("type") != OWNER_SET_TYPE or data.get("owner") != self.owner:
            raise IntegrityError("owner-set manifest does not belong to this owner")
        maps = [ref for ref in references
                if ref.startswith("manifest:") and self.store.get_manifest(ref).get("type") == NODE_MAP_TYPE]
        if len(maps) != 1:
            raise IntegrityError("owner set must reference exactly one node map")
        node_map = ShardedNodeMap.from_dict(self.store.get_manifest(maps[0])["map"])
        if node_map.digest != data["node_map_sha256"]:
            raise IntegrityError("node map differs from the digest its owners agreed on")
        return node_map

    def stats(self) -> dict[str, int]:
        stats = self.store.stats()
        return {name: int(stats.get(name, 0)) for name in (
            "object_count", "object_bytes", "blob_bytes", "basis_bytes", "manifest_bytes", "references")}


# ----------------------------------------------------------------------------------------- node map

@dataclass(frozen=True)
class ShardedNodeMap:
    """The distributed node/basis map every owner agreed on (identical bytes on every owner).

    ``specs``: one :class:`FactorShardSpec` per entry. ``owners[s]``: owner ``s``'s fragment packs (store
    references, bytes, tiles with their ranges and spec digests) and its basis segments per entry.
    """

    node_id: str
    world_size: int
    kind: str
    specs: tuple
    contexts: tuple
    owners: tuple
    parent_node_id: str | None = None
    codec: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.node_id, str) or not self.node_id.strip():
            raise ConfigurationError("node_id must be a nonempty string")
        _int(self.world_size, "world_size", minimum=1)
        if self.kind not in FRAGMENT_KINDS:
            raise ConfigurationError("unsupported fragment kind")
        specs = tuple(sorted(self.specs, key=lambda spec: entry_order(spec.entry)))
        if not specs or any(not isinstance(spec, FactorShardSpec) or spec.world_size != self.world_size
                            for spec in specs):
            raise ConfigurationError("node map specs must be FactorShardSpec values of this group")
        if len({entry_order(spec.entry) for spec in specs}) != len(specs):
            raise ConfigurationError("duplicate node map entry")
        object.__setattr__(self, "specs", specs)
        contexts = {}
        for item in self.contexts:
            if not isinstance(item, tuple) or len(item) != 2 or not isinstance(item[1], TensorContext):
                raise ConfigurationError("node map contexts are (entry, TensorContext) pairs")
            contexts[entry_order(entry_key(item[0]))] = item[1]
        if len(contexts) != len(self.contexts) or sorted(contexts) != [entry_order(spec.entry) for spec in specs]:
            raise ConfigurationError("node map needs exactly one TensorContext per matrix entry")
        for spec in specs:
            context = contexts[entry_order(spec.entry)]
            if context.key != spec.key or context.channels != spec.channels:
                raise ConfigurationError("node map context and spec disagree")
        object.__setattr__(self, "contexts", tuple((spec.entry, contexts[entry_order(spec.entry)]) for spec in specs))
        owners = tuple(json_copy(item, name="owner record") for item in self.owners)
        if [item.get("owner") for item in owners] != list(range(self.world_size)):
            raise ConfigurationError("node map needs exactly one record per owner, in rank order")
        object.__setattr__(self, "owners", owners)
        if self.codec is not None:
            object.__setattr__(self, "codec", json_copy(self.codec, name="node map codec"))
        self.validate()

    def spec(self, entry: Any) -> FactorShardSpec:
        order = entry_order(entry_key(entry))
        for spec in self.specs:
            if entry_order(spec.entry) == order:
                return spec
        raise ConfigurationError(f"node map has no entry {entry_label(entry)}")

    @property
    def entries(self) -> tuple:
        return tuple(spec.entry for spec in self.specs)

    def context(self, entry: Any) -> TensorContext:
        order = entry_order(entry_key(entry))
        for item, context in self.contexts:
            if entry_order(item) == order:
                return context
        raise ConfigurationError(f"node map has no entry {entry_label(entry)}")

    def owner_record(self, owner: int) -> dict[str, Any]:
        return self.owners[owner]

    def validate(self) -> None:
        """Every owner holds exactly its spec ranges and segments; nothing is missing or extra."""
        for spec in self.specs:
            spec.check_partition()
        for owner, record in enumerate(self.owners):
            tiles = {}
            for pack_index, pack in enumerate(record["packs"]):
                object_ref(pack["ref"], "fragment pack", kinds=("blob",))
                for tile in pack["tiles"]:
                    entry = entry_from_dict(tile["entry"])
                    if entry_order(entry) in tiles:
                        raise IntegrityError("an owner holds two fragments of one matrix")
                    tiles[entry_order(entry)] = (tile, pack_index)
            segments: dict[tuple, list] = {}
            for item in record["segments"]:
                object_ref(item["ref"], "basis segment", kinds=("basis",))
                segments.setdefault(entry_order(entry_from_dict(item["entry"])), []).append(
                    (item["start"], item["stop"]))
            for spec in self.specs:
                order = entry_order(spec.entry)
                ranges = spec.view_ranges(owner)
                tile = tiles.pop(order, None)
                if ranges:
                    if tile is None:
                        raise IntegrityError(f"owner {owner} lacks its fragment of {entry_label(spec.entry)}")
                    tile = tile[0]
                    if tile["spec_sha256"] != spec.digest or [tuple(r) for r in tile["ranges"]] != list(ranges):
                        raise IntegrityError(f"owner {owner} fragment of {entry_label(spec.entry)} disagrees "
                                             "with the agreed spec")
                elif tile is not None:
                    raise IntegrityError(f"owner {owner} holds rows it does not own ({entry_label(spec.entry)})")
                if sorted(segments.pop(order, [])) != list(spec.owner_segments(owner)):
                    raise IntegrityError(f"owner {owner} basis segments of {entry_label(spec.entry)} disagree "
                                         "with the agreed ownership")
            if tiles or segments:
                raise IntegrityError(f"owner {owner} reports fragments or segments outside the node map")

    def segment_refs(self, owner: int, entry: Any) -> list[tuple[int, int, str]]:
        order = entry_order(entry_key(entry))
        return sorted((item["start"], item["stop"], item["ref"]) for item in self.owners[owner]["segments"]
                      if entry_order(entry_from_dict(item["entry"])) == order)

    def to_dict(self) -> dict[str, Any]:
        return {"schema": NODE_MAP_SCHEMA, "node_id": self.node_id, "world_size": self.world_size,
                "kind": self.kind, "parent_node_id": self.parent_node_id, "codec": self.codec,
                "specs": [spec.to_dict() for spec in self.specs],
                "contexts": [{"entry": entry_to_dict(entry), "context": context.to_dict()}
                             for entry, context in self.contexts],
                "owners": [dict(item) for item in self.owners]}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ShardedNodeMap:
        try:
            if data["schema"] != NODE_MAP_SCHEMA:
                raise IntegrityError("unsupported factor-shard node map schema")
            return cls(node_id=data["node_id"], world_size=data["world_size"], kind=data["kind"],
                       specs=tuple(FactorShardSpec.from_dict(item) for item in data["specs"]),
                       contexts=tuple((entry_from_dict(item["entry"]), TensorContext.from_dict(item["context"]))
                                      for item in data["contexts"]),
                       owners=tuple(data["owners"]), parent_node_id=data["parent_node_id"], codec=data["codec"])
        except (KeyError, TypeError, ConfigurationError) as exc:
            raise IntegrityError(f"invalid factor-shard node map: {exc}") from exc

    @property
    def digest(self) -> str:
        return canonical_digest(self.to_dict())

    def storage(self) -> dict[str, Any]:
        """Unique bytes per owner (packs + segments counted once) vs a replicated-basis layout."""
        per_owner, segments_seen = [], {}
        for record in self.owners:
            packs = sum(item["bytes"] for item in record["packs"])
            own = {item["ref"]: item["bytes"] for item in record["segments"]}
            segments_seen.update(own)
            per_owner.append({"owner": record["owner"], "pack_bytes": packs, "segment_bytes": sum(own.values())})
        basis_rows_bytes = sum(spec.total_rank * spec.channels * 2 for spec in self.specs)
        return {"per_owner": per_owner, "pack_bytes": sum(item["pack_bytes"] for item in per_owner),
                "segment_bytes": sum(item["segment_bytes"] for item in per_owner),
                "basis_version_bytes": basis_rows_bytes,
                "replicated_basis_bytes": basis_rows_bytes * self.world_size}


# ---------------------------------------------------------------------------------- fitting helpers

@dataclass(frozen=True)
class FitResult:
    """Owner-fit outcome: coefficients over basis rows ``[0, rank_view)`` plus the rows it adds.

    Exactly one of: ``fresh_basis`` (a new basis group: fresh fit or refit), ``added_rows`` (an immutable
    block appended to the parent version: augmentation), or neither (reuse of the parent version).
    """

    coefficients: torch.Tensor
    rank_view: int
    fresh_basis: torch.Tensor | None = None
    added_rows: torch.Tensor | None = None
    metrics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.fresh_basis is not None and self.added_rows is not None:
            raise ConfigurationError("a fit is either fresh or an augmentation, not both")
        _int(self.rank_view, "rank_view", minimum=1)
        if self.coefficients.dtype != torch.float32 or self.coefficients.ndim != 2 or (
                self.coefficients.shape[1] != self.rank_view):
            raise ConfigurationError("fit coefficients must be FP32 [tokens, rank_view]")
        for rows in (self.fresh_basis, self.added_rows):
            if rows is not None and (rows.dtype != torch.float16 or rows.ndim != 2 or rows.shape[0] < 1):
                raise ConfigurationError("fitted basis rows must be nonempty FP16 [rows, channels]")


def _fp32_context(context: TensorContext) -> TensorContext:
    import dataclasses

    return dataclasses.replace(context, source_dtype="float32")


def fresh_fit(rank: int, **options) -> Callable:
    """Owner-fit callable: a new basis group of ``rank`` rows (``factorization.fit_basis``)."""
    _int(rank, "rank", minimum=1)

    def fit(matrix, context, parent_basis):
        from .factorization import fit_basis

        factors = fit_basis(matrix, context, rank, **options)
        return FitResult(coefficients=factors.coefficients.float().contiguous(), rank_view=rank,
                         fresh_basis=factors.basis.contiguous(),
                         metrics={"method": factors.metrics.get("factorization_method"),
                                  "projection_relative_residual": factors.metrics.get("projection_relative_residual")})
    return fit


def augment_fit(added: int) -> Callable:
    """Owner-fit callable: append ``added`` residual directions to the parent version, then project."""
    _int(added, "added", minimum=1)

    def fit(matrix, context, parent_basis):
        from .factorization import fit_basis, project_onto_basis

        if parent_basis is None:
            raise ConfigurationError("augmentation needs the parent basis version")
        parent = project_onto_basis(matrix, parent_basis)
        residual = matrix.float() - parent.coefficients.to(matrix.device) @ parent_basis.float()
        new_rows = fit_basis(residual.contiguous(), _fp32_context(context), added).basis.contiguous()
        basis = torch.cat((parent_basis, new_rows.to(parent_basis.device)))
        projection = project_onto_basis(matrix, basis)
        return FitResult(coefficients=projection.coefficients.float().contiguous(), rank_view=basis.shape[0],
                         added_rows=new_rows, metrics={
                             "parent_relative_residual": parent.relative_residual,
                             "projection_relative_residual": projection.relative_residual})
    return fit


def reuse_fit(rank_view: int | None = None) -> Callable:
    """Owner-fit callable: project onto a prefix of the parent version (no new rows)."""
    def fit(matrix, context, parent_basis):
        from .factorization import project_onto_basis

        if parent_basis is None:
            raise ConfigurationError("reuse needs the parent basis version")
        view = parent_basis.shape[0] if rank_view is None else rank_view
        projection = project_onto_basis(matrix, parent_basis[:view].contiguous())
        return FitResult(coefficients=projection.coefficients.float().contiguous(), rank_view=view,
                         metrics={"projection_relative_residual": projection.relative_residual})
    return fit


@dataclass
class OwnerShard:
    """This rank's factor data of one matrix: its view coefficient columns and ALL basis rows it owns
    (whole version, in global row order; the view rows are a prefix). Empty owners hold ``None``."""

    entry: Any
    spec: FactorShardSpec
    context: TensorContext
    owner: int
    coefficients: torch.Tensor | None
    basis: torch.Tensor | None

    def __post_init__(self) -> None:
        entry_key(self.entry, "owner shard entry")
        if self.spec.entry != self.entry or self.context.key != self.spec.key:
            raise ConfigurationError("owner shard spec/context describe another matrix")
        view, owned = self.spec.view_rows(self.owner), ranges_rows(self.spec.basis_ranges(self.owner))
        if view == 0:
            if self.coefficients is not None:
                raise ConfigurationError("an owner without view rows holds no coefficients")
        elif (self.coefficients is None or self.coefficients.dtype != torch.float32
              or tuple(self.coefficients.shape) != (self.context.token_count, view)):
            raise ConfigurationError("owner coefficients must be FP32 [tokens, owned view rows]")
        if owned == 0:
            if self.basis is not None:
                raise ConfigurationError("an owner without basis rows holds no basis")
        elif (self.basis is None or self.basis.dtype != torch.float16
              or tuple(self.basis.shape) != (owned, self.spec.channels)):
            raise ConfigurationError("owner basis must be FP16 [owned version rows, channels]")

    def view_basis(self) -> torch.Tensor | None:
        rows = self.spec.view_rows(self.owner)
        return None if rows == 0 else self.basis[:rows]

    def segment_rows(self, start: int, stop: int) -> torch.Tensor:
        index = ranges_index(self.spec.basis_ranges(self.owner))
        lookup = {int(row): position for position, row in enumerate(index.tolist())}
        if start not in lookup or stop - 1 not in lookup or lookup[stop - 1] - lookup[start] != stop - 1 - start:
            raise ConfigurationError("segment rows are not owned contiguously by this owner")
        return self.basis[lookup[start]:lookup[start] + stop - start]

    def nbytes(self) -> int:
        return sum(0 if value is None else value.numel() * value.element_size()
                   for value in (self.coefficients, self.basis))


# --------------------------------------------------------------------------------- collective domain

class FactorShardDomain:
    """One rank's view of a non-replicated factor group; construct identically on every rank.

    ``coordinator``: CPU/Gloo :class:`DistributedOwner` for ordered control checkpoints. ``data_group``:
    a separate process group for payloads (NCCL with CUDA tensors, Gloo with CPU tensors). ``store``:
    this owner's :class:`OwnerFactorStore` (required for staging/publication/restore from storage).
    ``tile_tokens`` bounds the ``partials`` workspace to ``inflight`` x (world + 1) tiles of ``tile_tokens``
    x native-owner columns in FP32; ``factors`` holds at most ``inflight`` + 1 exchange rounds (each one
    matrix's FP32 basis columns plus at most ``FACTOR_EXCHANGE_BYTES`` of coefficient rows per peer) and
    one ``tile_tokens`` FP32 GEMM tile. ``checkpoint_every`` > 0 adds an agreed cancellation
    point every that many matrices inside a reconstruction (0: only around the payload phase).
    ``mode`` (default ``"factors"``) is the reconstruction default (per-call override allowed) and
    ``block_rows`` (default None = whole-matrix owners; ``PLAN_BLOCK_ROWS`` = the plan's 32-row grid) the
    block size of the specs this domain creates for fresh fits (extensions keep their parent's blocks).
    The plan-literal ablation is ``mode="partials", block_rows=PLAN_BLOCK_ROWS``.
    """

    def __init__(self, coordinator: DistributedOwner, *, data_group=None, device="cpu",
                 store: OwnerFactorStore | None = None, tile_tokens: int = 2048, inflight: int = 2,
                 checkpoint_every: int = 0, control_chunk: int = 192 * 1024, mode: str = DEFAULT_MODE,
                 block_rows: int | None = None):
        import torch.distributed as dist

        if not isinstance(coordinator, DistributedOwner):
            raise ConfigurationError("coordinator must be a DistributedOwner")
        self.coordinator, self.dist = coordinator, dist
        self.rank, self.world_size = coordinator.rank, coordinator.world_size
        self.device = torch.device(device)
        if coordinator.initialized and self.world_size > 1:
            if data_group is None or data_group is coordinator.group:
                raise ConfigurationError("factor payloads need their own process group, distinct from control")
            backend = str(dist.get_backend(data_group))
            if backend == "nccl" and self.device.type != "cuda":
                raise ConfigurationError("an NCCL data group requires CUDA tensors")
            if backend == "gloo" and self.device.type != "cpu":
                raise ConfigurationError("a Gloo data group requires CPU tensors")
            if dist.get_world_size(data_group) != self.world_size:
                raise ConfigurationError("data group size differs from the control group")
        self.data_group = data_group
        if store is not None and (not isinstance(store, OwnerFactorStore) or store.owner != self.rank
                                  or store.world_size != self.world_size):
            raise ConfigurationError("store must be this rank's OwnerFactorStore")
        self.store = store
        _int(tile_tokens, "tile_tokens", minimum=1)
        if type(inflight) is not int or not 1 <= inflight <= 4:
            raise ConfigurationError("inflight must be an integer in [1, 4]")
        _int(checkpoint_every, "checkpoint_every")
        _int(control_chunk, "control_chunk", minimum=1024)
        if control_chunk > 256 * 1024:
            raise ConfigurationError("control_chunk must stay below the coordinator's checkpoint bound")
        self.tile_tokens, self.inflight, self.checkpoint_every = tile_tokens, inflight, checkpoint_every
        self.control_chunk = control_chunk
        if mode not in RECONSTRUCT_MODES:
            raise ConfigurationError(f"reconstruction mode must be one of {RECONSTRUCT_MODES}")
        if block_rows is not None:
            _int(block_rows, "block_rows", minimum=1)
        self.mode, self.block_rows = mode, block_rows
        self._broken = False
        self.counters = {
            "p2p_bytes_sent": 0, "p2p_bytes_received": 0, "p2p_messages": 0,
            "reduce_scatter_calls": 0, "reduce_scatter_bytes_sent": 0, "reduce_scatter_bytes_received": 0,
            "reduce_scatter_padding_bytes": 0, "zero_partial_calls": 0, "tiles": 0, "matrices_restored": 0,
            "factor_exchange_bytes_sent": 0, "factor_exchange_bytes_received": 0, "factor_exchange_rounds": 0,
            "factor_transient_peak_bytes": 0,
            "owner_fit": [], "retired_transfers": 0, "cancelled_operations": 0,
        }
        rows = coordinator.checkpoint("factor_shard_config", details={
            "tile_tokens": tile_tokens, "inflight": inflight, "checkpoint_every": checkpoint_every,
            "device_type": self.device.type, "store": store is not None, "control_chunk": control_chunk,
            "mode": mode, "block_rows": block_rows,
        })
        if any(row != rows[0] for row in rows):
            raise ConfigurationError("factor-shard ranks disagree on the domain configuration")

    # -- ordered control and transport --------------------------------------------------------------
    def _agree(self, operation, *, error=None, cancelled=False, details=None, same=False):
        if self._broken:
            raise PeerFailure("the factor-shard data group failed; create a new domain")
        try:
            rows = self.coordinator.checkpoint(operation, error=error, cancelled=cancelled, details=details)
        except CollectiveCancelled:
            self.counters["cancelled_operations"] += 1
            raise
        if same and any(row != rows[0] for row in rows):
            raise PeerFailure(f"{operation}: ranks disagree: {rows}")
        return rows

    def _all_records(self, operation, record, *, error=None, cancelled=False):
        """Exchange one JSON record per rank in bounded control chunks (node maps outgrow one checkpoint).

        The first round carries every rank's error/cancellation and chunk count, so failures are agreed
        before any record bytes move; all ranks then run the same number of chunk rounds.
        """
        raw = "" if record is None else json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False)
        pieces = [raw[start:start + self.control_chunk] for start in range(0, len(raw), self.control_chunk)] or [""]
        counts = self._agree(f"{operation}:chunks", error=error, cancelled=cancelled, details=len(pieces))
        parts = [[] for _ in range(self.world_size)]
        for index in range(max(counts)):
            rows = self._agree(f"{operation}:chunk{index}", details=pieces[index] if index < len(pieces) else None)
            for rank, row in enumerate(rows):
                if index < counts[rank]:
                    if not isinstance(row, str):
                        raise PeerFailure(f"{operation}: rank {rank} sent a malformed record chunk")
                    parts[rank].append(row)
        try:
            return [json.loads("".join(items)) if items and "".join(items) else None for items in parts]
        except ValueError as exc:
            raise PeerFailure(f"{operation}: a peer record is not valid JSON") from exc

    def _transport(self, function, *args, **kwargs):
        try:
            return function(*args, **kwargs)
        except Exception as exc:
            self._broken = True
            raise PeerFailure(f"factor-shard transport failed: {type(exc).__name__}: {exc}") from exc

    def _global(self, rank: int) -> int:
        return self.dist.get_global_rank(self.data_group, rank) if self.data_group is not None else rank

    def _send(self, tensor: torch.Tensor, destination: int) -> None:
        tensor = tensor.contiguous()
        self._transport(self.dist.send, tensor, dst=self._global(destination), group=self.data_group)
        self.counters["p2p_bytes_sent"] += tensor.numel() * tensor.element_size()
        self.counters["p2p_messages"] += 1

    def _recv(self, tensor: torch.Tensor, source: int) -> None:
        self._transport(self.dist.recv, tensor, src=self._global(source), group=self.data_group)
        self.counters["p2p_bytes_received"] += tensor.numel() * tensor.element_size()

    # -- archival: bounded owner-fit reference ------------------------------------------------------------
    def gather_canonical(self, local: torch.Tensor, native: NativeOwners, *, fit_owner: int,
                         operation: str = "factor_gather") -> torch.Tensor | None:
        """Gather one canonical ``[tokens, channels]`` matrix onto ``fit_owner`` (transient, reported)."""
        error, details = None, None
        try:
            if not isinstance(native, NativeOwners) or native.world_size != self.world_size:
                raise ConfigurationError("native owners must describe this group")
            if type(fit_owner) is not int or not 0 <= fit_owner < self.world_size:
                raise ConfigurationError("fit owner is outside the group")
            if tuple(local.shape) != native.shape(self.rank) or local.device.type != self.device.type:
                raise ConfigurationError("local native shard does not match its declared layout/device")
            if not local.is_floating_point():
                raise ConfigurationError("native KV shards are floating point")
            details = {"dtype": str(local.dtype), "native": native.to_dict(), "fit_owner": fit_owner}
        except Exception as exc:  # noqa: BLE001 - reported to every rank
            error = exc
        self._agree(operation, error=error, details=details, same=error is None)
        canonical, error = None, None
        try:
            if self.rank == fit_owner:
                canonical = torch.empty((native.tokens, native.channels), dtype=local.dtype, device=self.device)
        except Exception as exc:  # noqa: BLE001
            error = exc
        self._agree(f"{operation}:allocated", error=error)

        def place(value, source):
            if native.kind == "heads":
                canonical[:, native.columns(source)].copy_(value)
            else:
                canonical.index_copy_(0, native.rows(source).to(self.device), value)

        for source in range(self.world_size):
            if source == fit_owner:
                if self.rank == fit_owner:
                    place(local, source)
            elif self.rank == source:
                self._send(local, fit_owner)
            elif self.rank == fit_owner:
                buffer = torch.empty(native.shape(source), dtype=local.dtype, device=self.device)
                self._recv(buffer, source)
                place(buffer, source)
                del buffer
        return canonical

    def _gather_parent_basis(self, parent: OwnerShard, fit_owner: int) -> torch.Tensor | None:
        spec = parent.spec
        full = None
        if self.rank == fit_owner:
            full = torch.empty((spec.total_rank, spec.channels), dtype=torch.float16, device=self.device)
        for source in range(self.world_size):
            ranges = spec.basis_ranges(source)
            if not ranges:
                continue
            index = ranges_index(ranges, device=self.device)
            if source == fit_owner:
                if self.rank == fit_owner:
                    full.index_copy_(0, index, parent.basis.to(self.device))
            elif self.rank == source:
                self._send(parent.basis.to(self.device), fit_owner)
            elif self.rank == fit_owner:
                buffer = torch.empty((index.numel(), spec.channels), dtype=torch.float16, device=self.device)
                self._recv(buffer, source)
                full.index_copy_(0, index, buffer)
        return full

    def fit_and_scatter(self, entry: Any, local: torch.Tensor, native: NativeOwners, context: TensorContext, *,
                        fit: Callable, fit_owner: int | None = None, parent: OwnerShard | None = None,
                        operation: str | None = None) -> OwnerShard:
        """Owner-fit reference: gather -> fit on ``fit_owner`` -> scatter factor blocks to their owners.

        With ``parent`` (this rank's shard of the parent node's matrix) the parent's basis rows are
        gathered transiently to the fit owner so it can reuse/augment the version; existing rows keep
        their owners and only coefficient columns and NEW rows move. Returns this rank's shard.
        """
        entry = entry_key(entry)
        op = operation or f"factor_fit:{entry_label(entry)}"
        fit_owner = default_rotation(entry, self.world_size) if fit_owner is None else fit_owner
        error = None
        try:
            if not isinstance(context, TensorContext) or context.key != entry_parts(entry)[0]:
                raise ConfigurationError("context must describe the fitted entry")
            if (context.token_count, context.channels) != (native.tokens, native.channels):
                raise ConfigurationError("context geometry differs from the native layout")
            if parent is not None:
                if not isinstance(parent, OwnerShard) or parent.entry != entry or parent.owner != self.rank:
                    raise ConfigurationError("parent must be this rank's shard of the same matrix")
                if parent.spec.world_size != self.world_size:
                    raise ConfigurationError("parent spec belongs to another group")
        except Exception as exc:  # noqa: BLE001
            error = exc
        self._agree(f"{op}:begin", error=error, details={
            "entry": entry_label(entry), "parent": None if parent is None else parent.spec.digest,
            "fit_owner": fit_owner}, same=error is None)
        started = time.perf_counter()
        canonical = self.gather_canonical(local, native, fit_owner=fit_owner, operation=f"{op}:gather")
        parent_basis = None if parent is None else self._gather_parent_basis(parent, fit_owner)
        spec, result, error, report = None, None, None, None
        if self.rank == fit_owner:
            try:
                fit_started = time.perf_counter()
                result = fit(canonical, context, parent_basis)
                if not isinstance(result, FitResult) or result.coefficients.shape[0] != context.token_count:
                    raise ConfigurationError("fit must return a FitResult over the matrix tokens")
                if result.fresh_basis is not None:
                    spec = FactorShardSpec.fresh(entry, channels=context.channels, rank=result.fresh_basis.shape[0],
                                                 world_size=self.world_size, block_rows=self.block_rows
                                                 ).with_view(result.rank_view)
                elif parent is None:
                    raise ConfigurationError("a fit without a parent must return a fresh basis")
                elif result.added_rows is not None:
                    spec = parent.spec.extend(result.added_rows.shape[0], rank_view=result.rank_view)
                else:
                    spec = parent.spec.with_view(result.rank_view)
                spec.check_partition()
                report = {
                    "entry": entry_label(entry), "fit_owner": fit_owner,
                    "transient_canonical_bytes": canonical.numel() * canonical.element_size(),
                    "transient_parent_basis_bytes": 0 if parent_basis is None else parent_basis.numel() * 2,
                    "factor_bytes": result.coefficients.numel() * 4 + sum(
                        0 if rows is None else rows.numel() * 2 for rows in (result.fresh_basis, result.added_rows)),
                    "fit_seconds": time.perf_counter() - fit_started, "fresh": result.fresh_basis is not None,
                    "added_rows": 0 if result.added_rows is None else int(result.added_rows.shape[0]),
                    "rank_view": result.rank_view, "metrics": json_copy(dict(result.metrics)),
                }
            except Exception as exc:  # noqa: BLE001 - reported before any factor block moves
                error = exc
        rows = self._agree(f"{op}:fitted", error=error, details={
            "spec": spec.to_dict(), "fresh": result.fresh_basis is not None}
            if self.rank == fit_owner and error is None else None)
        spec, fresh = FactorShardSpec.from_dict(rows[fit_owner]["spec"]), rows[fit_owner]["fresh"]
        if type(fresh) is not bool or (not fresh and parent is None):
            raise PeerFailure("fit owner published an inconsistent fit decision")
        if not fresh:
            spec.check_stable(parent.spec)
        new_start = 0 if fresh else parent.spec.total_rank
        new_rows = None
        if self.rank == fit_owner:
            new_rows = result.fresh_basis if fresh else result.added_rows
        coefficients = basis_new = None
        for owner in range(self.world_size):
            view_index = ranges_index(spec.view_ranges(owner), device=self.device)
            owned_new = tuple((max(start, new_start), stop) for start, stop in spec.basis_ranges(owner)
                              if stop > new_start)
            new_index = ranges_index(owned_new, device=self.device) - new_start
            if self.rank == fit_owner:
                block_p = result.coefficients.to(self.device).index_select(1, view_index) if view_index.numel() else None
                block_q = new_rows.to(self.device).index_select(0, new_index) if new_index.numel() else None
                if owner == fit_owner:
                    coefficients, basis_new = block_p, block_q
                else:
                    if block_p is not None:
                        self._send(block_p, owner)
                    if block_q is not None:
                        self._send(block_q, owner)
            elif self.rank == owner:
                if view_index.numel():
                    coefficients = torch.empty((context.token_count, view_index.numel()), dtype=torch.float32,
                                               device=self.device)
                    self._recv(coefficients, fit_owner)
                if new_index.numel():
                    basis_new = torch.empty((new_index.numel(), spec.channels), dtype=torch.float16,
                                            device=self.device)
                    self._recv(basis_new, fit_owner)
        parts = []
        if not fresh and parent.basis is not None:
            parts.append(parent.basis.to(self.device))
        if basis_new is not None:
            parts.append(basis_new)
        basis = torch.cat(parts).contiguous() if parts else None
        shard = OwnerShard(entry, spec, context, self.rank,
                           None if coefficients is None else coefficients.contiguous(), basis)
        details = None
        if report is not None:
            report["wall_seconds"] = time.perf_counter() - started
            details = report
        rows = self._agree(f"{op}:scattered", details=details)
        self.counters["owner_fit"].append(rows[fit_owner])
        del canonical, parent_basis
        return shard

    # -- staging and two-phase publication ---------------------------------------------------------------
    def stage_node(self, node_id: str, shards: Mapping[Any, OwnerShard], *, kind: str = "svd_hevc",
                   codec: CodecSpec | None = None, ffmpeg_path: str | None = None, threads: int = 1,
                   pack_groups: int = 1, validate: bool = False) -> dict[str, Any]:
        """Write this owner's basis segments and fragment pack(s) of one node (unreferenced until published).

        Local; never raises for content errors: they are recorded and reported at publication so every
        owner fails the same publication. ``pack_groups`` > 1 splits the owner's tiles into that many
        independent streams (e.g. to decode on several NVDEC engines concurrently).
        """
        record = {"node_id": node_id, "owner": self.rank, "kind": kind, "packs": [], "segments": [],
                  "specs": {}, "contexts": {}, "codec": None, "error": None, "encode_seconds": 0.0,
                  "staged_refs": []}
        try:
            if self.store is None:
                raise StateError("staging needs this owner's OwnerFactorStore")
            if kind not in FRAGMENT_KINDS:
                raise ConfigurationError(f"fragment kind must be one of {FRAGMENT_KINDS}")
            _int(pack_groups, "pack_groups", minimum=1)
            fragments, empty = [], []
            for entry, shard in sorted(shards.items(), key=lambda item: entry_order(item[0])):
                if (not isinstance(shard, OwnerShard) or shard.owner != self.rank or shard.entry != entry_key(entry)
                        or shard.spec.world_size != self.world_size):
                    raise ConfigurationError("stage_node needs this rank's OwnerShard of every entry")
                label = entry_label(entry)
                record["specs"][label] = shard.spec.to_dict()
                record["contexts"][label] = {"entry": entry_to_dict(entry), "context": shard.context.to_dict()}
                for start, stop in shard.spec.owner_segments(self.rank):
                    ref = self.store.put_segment(shard.segment_rows(start, stop).cpu())
                    record["segments"].append({"entry": entry_to_dict(entry), "start": start, "stop": stop,
                                               "ref": ref, "bytes": (stop - start) * shard.spec.channels * 2})
                    record["staged_refs"].append(ref)
                if shard.spec.view_rows(self.rank):
                    fragments.append(FragmentInput(entry, shard.spec, shard.context, shard.coefficients.cpu(),
                                                   shard.view_basis().cpu()))
                else:
                    empty.append(entry)
            if kind == "svd_hevc":
                record["codec"] = _hevc_codec(codec)
            groups = min(pack_groups, len(fragments))
            for index in range(groups):
                lo, hi = index * len(fragments) // groups, (index + 1) * len(fragments) // groups
                pack = encode_fragment_pack(node_id, self.rank, self.world_size, fragments[lo:hi], kind=kind,
                                            codec=codec, ffmpeg_path=ffmpeg_path, threads=threads,
                                            validate=validate, empty_entries=empty if index == 0 else ())
                ref = self.store.put_pack(pack)
                record["staged_refs"].append(ref)
                record["packs"].append({
                    "ref": ref, "bytes": self.store.pack_size(ref), "payload_bytes": pack.metrics["payload_bytes"],
                    "tiles": [{"entry": tile["entry"], "spec_sha256": tile["spec_sha256"], "ranges": tile["ranges"]}
                              for tile in pack.metadata["tiles"]]})
                record["encode_seconds"] += pack.metrics["encode_seconds"]
        except Exception as exc:  # noqa: BLE001 - reported at publication on every owner
            record["error"] = f"{type(exc).__name__}: {exc}"
        return record

    def discard_staged(self, staged: Mapping[str, Any]) -> dict[str, int]:
        """Collect a failed/cancelled staging's unreferenced objects (shared live segments survive)."""
        if self.store is None or not staged.get("staged_refs"):
            return {"objects_removed": 0, "bytes_reclaimed": 0}
        return self.store.store.collect_garbage(candidates=staged["staged_refs"])

    def publish_node(self, staged: Mapping[str, Any], *, parent_node_id: str | None = None,
                     cancelled: bool = False) -> ShardedNodeMap:
        """Two-phase publication: prepare (all staged), agree (one node-map digest), commit (all owners).

        Nothing is visible before every owner agreed; a failed commit rolls every owner back. A
        cancellation or failure leaves only unreferenced staged objects (see :meth:`discard_staged`).
        """
        node_id = staged["node_id"]
        op = f"factor_publish:{node_id}"
        details = {name: staged[name] for name in ("node_id", "owner", "kind", "packs", "segments", "specs",
                                                    "contexts", "codec")}
        rows = self._all_records(f"{op}:prepare", details, error=staged.get("error"), cancelled=cancelled)
        node_map, error = None, None
        try:
            if [row["owner"] for row in rows] != list(range(self.world_size)) or any(
                    row["node_id"] != node_id or row["kind"] != rows[0]["kind"] or row["codec"] != rows[0]["codec"]
                    for row in rows):
                raise IntegrityError("owners staged different nodes, kinds or codecs")
            if any(row["specs"] != rows[0]["specs"] or row["contexts"] != rows[0]["contexts"] for row in rows):
                raise IntegrityError("owners disagree on the factor ownership of the node")
            node_map = ShardedNodeMap(
                node_id=node_id, world_size=self.world_size, kind=rows[0]["kind"],
                specs=tuple(FactorShardSpec.from_dict(item) for item in rows[0]["specs"].values()),
                contexts=tuple((entry_from_dict(item["entry"]), TensorContext.from_dict(item["context"]))
                               for item in rows[0]["contexts"].values()),
                owners=tuple({"owner": row["owner"], "packs": row["packs"], "segments": row["segments"]}
                             for row in rows),
                parent_node_id=parent_node_id, codec=rows[0]["codec"])
        except Exception as exc:  # noqa: BLE001 - deterministic on every rank, still agreed explicitly
            error = exc
        agreed = self._agree(f"{op}:agree", error=error, details=None if node_map is None else node_map.digest)
        if any(row != agreed[0] for row in agreed):
            raise PeerFailure(f"{op}: owners computed different node maps")
        owner_set, error = None, None
        try:
            owner_set = self.store.commit(node_map)
        except Exception as exc:  # noqa: BLE001
            error = exc
        try:
            self._agree(f"{op}:commit", error=error)
        except PeerFailure as failure:
            rollback = None
            if owner_set is not None:
                try:
                    self.store.retract(node_id, owner_set)
                except Exception as exc:  # noqa: BLE001
                    rollback = exc
            if not self._broken and not self.coordinator._broken:
                self._agree(f"{op}:rolled_back", error=rollback)
            raise failure
        return node_map

    # -- restore from storage ---------------------------------------------------------------------------
    def load_node(self, node_map: ShardedNodeMap, *, entries: Iterable[Any] | None = None, device=None,
                  pool=None, verify: bool = True, ffmpeg_path: str | None = None, arena=None
                  ) -> dict[Any, OwnerShard]:
        """Local: decode this owner's fragments and assemble its basis rows from its own segments.

        Every pack and basis segment read is checked once against its content address (by the store when it
        verifies reads, otherwise by :class:`OwnerFactorStore`); ``verify`` validates every decoded tile's affine."""
        if self.store is None:
            raise StateError("loading needs this owner's OwnerFactorStore")
        device = self.device if device is None else torch.device(device)
        if pool is not None and device.type != "cuda":  # checked on every owner, even one without packs
            raise ConfigurationError("a DecoderPool decodes to CUDA devices only")
        if self.store.node_map(node_map.node_id).digest != node_map.digest:
            raise IntegrityError("this owner published a different map for the node")
        wanted = list(node_map.entries) if entries is None else [entry_key(entry) for entry in entries]
        orders = {entry_order(entry) for entry in wanted}
        coefficients, pending = {}, []
        try:
            return self._load_owned(node_map, wanted, orders, coefficients, pending, device=device, pool=pool,
                                    verify=verify, ffmpeg_path=ffmpeg_path, arena=arena)
        except BaseException as failure:
            for item in pending:  # decodes already queued: their canvases return to the arena once written
                try:
                    item.discard()
                except Exception:  # noqa: BLE001 - the load failure is raised
                    pass
            coefficients.clear()
            pending.clear()
            traceback.clear_frames(failure.__traceback__)  # partially loaded factors in the finished callee frames
            raise

    def _load_owned(self, node_map, wanted, orders, coefficients, pending, *, device, pool, verify, ffmpeg_path,
                    arena) -> dict[Any, OwnerShard]:
        for pack in node_map.owner_record(self.rank)["packs"]:
            chosen = [entry_from_dict(tile["entry"]) for tile in pack["tiles"]
                      if entry_order(entry_from_dict(tile["entry"])) in orders]
            if not chosen:
                continue
            encoded = self.store.get_pack(pack["ref"])
            # The container's bytes were just checked against their address (get_pack or the store), so its payload
            # digests and the stream's access-unit structure are the ones validated when the pack was written:
            # structural checks only here; the selected tiles' byte ranges are still scanned (fragment_units) and
            # every decoded tile's affine is still validated (verify).
            parsed = parse_fragment_pack(encoded, verify=False)
            if parsed["meta"]["node_id"] != node_map.node_id or parsed["meta"]["owner"] != self.rank:
                raise IntegrityError("fragment pack belongs to another node or owner")
            for entry in chosen:
                tile = parsed["index"][entry_order(entry)][3]
                spec = node_map.spec(entry)
                if tile["spec_sha256"] != spec.digest or [tuple(r) for r in tile["ranges"]] != list(
                        spec.view_ranges(self.rank)):
                    raise IntegrityError(f"fragment of {entry_label(entry)} disagrees with the agreed map")
            if device.type == "cuda" and parsed["meta"]["kind"] == "svd_hevc":
                pending.append(submit_fragment_decode(encoded, pool=pool, device=device, entries=chosen,
                                                      verify=verify, parsed=parsed, arena=arena))
            else:
                coefficients.update(decode_fragment_pack(encoded, entries=chosen, device=device, pool=pool,
                                                         verify=verify, ffmpeg_path=ffmpeg_path, parsed=parsed))
        for item in pending:  # every pack was queued before the first wait: owner packs decode concurrently
            coefficients.update(item.result())
        shards = {}
        for entry in wanted:
            spec, context = node_map.spec(entry), node_map.context(entry)
            parts = [self.store.get_segment(ref, stop - start, spec.channels)
                     for start, stop, ref in node_map.segment_refs(self.rank, entry)]
            basis = torch.cat(parts).to(device) if parts else None
            value = coefficients.get(entry)
            if (value is None) != (spec.view_rows(self.rank) == 0):
                raise IntegrityError(f"owner {self.rank} fragment presence disagrees with {entry_label(entry)}")
            shards[entry] = OwnerShard(entry, spec, context, self.rank, value, basis)
        return shards

    def restore_node(self, node_map: ShardedNodeMap, native: NativeOwners, *, entries: Iterable[Any] | None = None,
                     dtype=torch.bfloat16, rope: Callable | None = None, pool=None, verify: bool = True,
                     out: Mapping[Any, torch.Tensor] | None = None, cancel: Callable | None = None,
                     ffmpeg_path: str | None = None, arena=None, mode: str | None = None) -> dict[Any, torch.Tensor]:
        """Collective: every owner decodes its own fragments; tiled partials reduce-scatter to native owners."""
        op = f"factor_restore:{node_map.node_id}"
        wanted = list(node_map.entries) if entries is None else [entry_key(entry) for entry in entries]
        self._agree(f"{op}:begin", details={"map": node_map.digest, "entries": [entry_label(e) for e in wanted]},
                    same=True)
        shards, error = None, None
        try:
            shards = self.load_node(node_map, entries=wanted, pool=pool, verify=verify, ffmpeg_path=ffmpeg_path,
                                    arena=arena)
        except Exception as exc:  # noqa: BLE001 - agreed before any payload collective
            error = exc
        try:
            self._agree(f"{op}:loaded", error=error, cancelled=bool(cancel and cancel()))
            error = None
            return self.reconstruct(shards, native, dtype=dtype, rope=rope, out=out, operation=op, cancel=cancel,
                                    mode=mode)
        except BaseException:
            shards = error = None  # a failed restore keeps no decoded factors alive in its traceback
            raise

    # -- tiled reconstruction + reduce-scatter -----------------------------------------------------------------
    def reconstruct(self, shards: Mapping[Any, OwnerShard], native: NativeOwners, *, dtype=torch.bfloat16,
                    rope: Callable | None = None, out: Mapping[Any, torch.Tensor] | None = None,
                    operation: str = "factor_reconstruct", cancel: Callable | None = None,
                    mode: str | None = None) -> dict[Any, torch.Tensor]:
        """Collective: native-layout shards ``{entry: [native rows, native columns]}`` on every rank.

        ``mode=None`` uses the domain default (``"factors"`` unless configured otherwise).
        ``mode="factors"`` (default; coordinator decision, user confirmation pending): owners send factor
        tiles instead of products. Per matrix ONE batched point-to-point round carries, for every native
        owner, the basis columns it owns (heads; all columns for rows) of every factor owner's rows together
        with the coefficient rows it needs (all of them unless they exceed ``FACTOR_EXCHANGE_BYTES``, then
        further rounds of at least ``tile_tokens`` tokens); rounds are pipelined ``inflight`` deep across
        matrices. Each native owner then runs one FP32 GEMM per ``tile_tokens`` tile over the full rank in
        global row order (about ``rank / channels`` of the partials bytes; nothing is retained after the
        matrix; the transient working set is reported). ``mode="partials"`` (plan section 4.8 literal layout;
        explicit ablation): each owner
        computes FP32 partials ``P_s[tile] @ Q_s`` straight into per-destination buffers and one
        reduce-scatter per tile delivers each native owner the sum over factor owners; it moves
        ``(w-1)/w * tokens * channels * 4`` bytes per rank and matrix. Owners store and decode only their
        own rows in both modes. ``rope(entry,
        values, tokens, columns)`` (pre-RoPE K entries only) runs in FP32 on the native owner before the
        cast to ``dtype``. If ``out`` is given and the operation fails or is cancelled, its contents are
        undefined (partially written); retired transfers are drained before the error propagates.
        """
        items = sorted(shards.items(), key=lambda item: entry_order(item[0]))
        error, details = None, None
        try:
            from .types import canonical_dtype

            canonical_dtype(dtype)
            if not isinstance(native, NativeOwners) or native.world_size != self.world_size:
                raise ConfigurationError("native owners must describe this group")
            for entry, shard in items:
                if not isinstance(shard, OwnerShard) or shard.owner != self.rank or shard.entry != entry_key(entry):
                    raise ConfigurationError("reconstruct needs this rank's OwnerShard of every entry")
                if shard.spec.world_size != self.world_size or shard.spec.channels != native.channels:
                    raise ConfigurationError("shard spec does not match the group or native channels")
                if shard.context.token_count != native.tokens:
                    raise ConfigurationError("shard tokens differ from the native layout")
                for value in (shard.coefficients, shard.basis):
                    if value is not None and value.device.type != self.device.type:
                        raise ConfigurationError("owner factors must live on the domain device")
            mode = self.mode if mode is None else mode
            if mode not in RECONSTRUCT_MODES:
                raise ConfigurationError(f"reconstruction mode must be one of {RECONSTRUCT_MODES}")
            details = {"entries": [[entry_label(entry), shard.spec.digest] for entry, shard in items],
                       "native": native.to_dict(), "dtype": str(dtype), "tile_tokens": self.tile_tokens,
                       "mode": mode}
        except Exception as exc:  # noqa: BLE001
            error = exc
        outputs = slots = target = shard = value = pending = state = row_sets = None
        try:
            self._agree(f"{operation}:plan", error=error, details=details, same=error is None)
            outputs, slots, error = {}, None, None
            try:
                columns = native.layout.local_channels if native.kind == "heads" else native.channels
                longest = native.tokens if native.kind == "heads" else max(
                    native.shape(rank)[0] for rank in range(self.world_size))
                tile = min(self.tile_tokens, longest)
                slots = [] if mode == "factors" else [
                    (torch.empty(self.world_size * tile * columns, dtype=torch.float32, device=self.device),
                     torch.empty(tile * columns, dtype=torch.float32, device=self.device))
                    for _ in range(self.inflight)]
                for entry in [item[0] for item in items]:
                    target = None if out is None else out.get(entry)
                    if target is None:
                        target = torch.empty(native.shape(self.rank), dtype=dtype, device=self.device)
                    elif (tuple(target.shape) != native.shape(self.rank) or target.dtype != dtype
                          or target.device.type != self.device.type or not target.is_contiguous()):
                        raise ConfigurationError("output buffer differs from this rank's native shard")
                    outputs[entry] = target
                target = None
                row_sets = None if native.kind == "heads" else [native.rows(rank).to(self.device)
                                                               for rank in range(self.world_size)]
            except Exception as exc:  # noqa: BLE001
                error = exc
            self._agree(f"{operation}:allocated", error=error, cancelled=bool(cancel and cancel()))
            pending, state = [], {"slot": 0}
            try:
                for index, (entry, shard) in enumerate(items):
                    if self.checkpoint_every and index and index % self.checkpoint_every == 0:
                        while pending:
                            self._finish(pending.pop(0))
                        self._agree(f"{operation}:matrix{index}", cancelled=bool(cancel and cancel()))
                    if mode == "factors":
                        self._reconstruct_factors_one(entry, shard, native, outputs[entry], rope, tile, row_sets,
                                                      pending)
                    else:
                        self._reconstruct_one(entry, shard, native, outputs[entry], rope, slots, tile, columns,
                                              row_sets, pending, state)
                while pending:
                    self._finish(pending.pop(0))
            except BaseException:
                self._retire(pending)
                raise
            self._agree(f"{operation}:complete")
        except BaseException as failure:
            # A failed or cancelled reconstruction keeps nothing alive in its traceback: the (partially restored)
            # outputs this call allocated, its workspaces, its references to the owners' decoded factors and the
            # finished callee frames' per-matrix buffers are dropped before the failure propagates
            # (caller-provided ``out`` buffers remain the caller's).
            outputs = slots = target = shard = value = items = shards = error = pending = state = row_sets = None
            traceback.clear_frames(failure.__traceback__)
            raise
        self.counters["matrices_restored"] += len(items)
        return outputs

    def _finish(self, item) -> None:
        work, finalize = item
        for each in (work if isinstance(work, list) else [work]):
            if each is not None:
                self._transport(each.wait)
        finalize()

    def _retire(self, pending) -> None:
        for work, _ in pending:
            for each in (work if isinstance(work, list) else [work]):
                if each is not None:
                    try:
                        each.wait()
                    except Exception:  # noqa: BLE001 - the failure that triggered retirement is raised
                        pass
                self.counters["retired_transfers"] += 1
        pending.clear()

    def _exchange(self, sends, receives) -> list:
        """One batched point-to-point round on the data group; returns its works (empty: nothing to move).

        ``sends`` / ``receives`` hold ``(tensor, peer, tag)``; both sides list the messages of a peer pair in
        the same order (NCCL matches them in order, Gloo by tag)."""
        if self.world_size == 1 or (not sends and not receives):
            return []
        ops = [self.dist.P2POp(self.dist.isend, tensor, self._global(peer), self.data_group, tag)
               for tensor, peer, tag in sends]
        ops += [self.dist.P2POp(self.dist.irecv, tensor, self._global(peer), self.data_group, tag)
                for tensor, peer, tag in receives]
        works = self._transport(self.dist.batch_isend_irecv, ops)
        self.counters["factor_exchange_bytes_sent"] += sum(t.numel() * t.element_size() for t, _, _ in sends)
        self.counters["factor_exchange_bytes_received"] += sum(t.numel() * t.element_size() for t, _, _ in receives)
        self.counters["factor_exchange_rounds"] += 1
        return list(works)

    def _reconstruct_factors_one(self, entry, shard, native, target, rope, tile, row_sets, pending) -> None:
        world, rank, device = self.world_size, self.rank, self.device
        spec = shard.spec
        heads = native.kind == "heads"
        width = native.layout.local_channels if heads else native.channels
        own_columns = native.columns(rank)
        rows = [spec.view_rows(owner) for owner in range(world)]
        owners = [owner for owner in range(world) if rows[owner]]
        # global row positions, only needed to interleave several owners' rows (none for whole-matrix owners)
        index = {owner: ranges_index(spec.view_ranges(owner), device=device) for owner in owners} \
            if len(owners) > 1 else {}
        P, Qv = shard.coefficients, shard.view_basis()
        rotate = rope if (rope is not None and entry_parts(entry)[0].target == "K") else None
        counts = [native.tokens] * world if heads else [int(item.numel()) for item in row_sets]
        # Tokens per exchange round: the whole matrix unless its coefficient rows exceed FACTOR_EXCHANGE_BYTES.
        chunk = max(tile, FACTOR_EXCHANGE_BYTES // (4 * max(1, spec.rank_view)))
        starts = list(range(0, max(counts), chunk)) or [0]
        # This native owner's FP32 basis columns of every owner's rows (global row order), built by the first
        # round's finalize and released by the last one; rounds are pipelined across matrices.
        state = {}
        transient = 0
        for step, start in enumerate(starts):
            sizes = [max(0, min(chunk, count - start)) for count in counts]
            mine = sizes[rank]
            staged, parts, sends, receives = [None] * world, [None] * world, [], []
            if step == 0:
                if rows[rank]:
                    staged[rank] = Qv[:, own_columns] if heads else Qv
                    for other in range(world):
                        if other != rank:
                            sends.append((Qv[:, native.columns(other)].contiguous() if heads else Qv.contiguous(),
                                          other, _BASIS_TAG))
                for other in range(world):
                    if other != rank and rows[other]:
                        staged[other] = torch.empty((rows[other], width), dtype=torch.float16, device=device)
                        receives.append((staged[other], other, _BASIS_TAG))
            if rows[rank]:
                for other in range(world):
                    if sizes[other] == 0:
                        continue
                    block = (P[start:start + sizes[other]] if heads
                             else P.index_select(0, row_sets[other][start:start + sizes[other]]))
                    if other == rank:
                        parts[rank] = block
                    else:
                        sends.append((block.contiguous(), other, _COEFFICIENT_TAG))
            for other in range(world):
                if other != rank and rows[other] and mine:
                    parts[other] = torch.empty((mine, rows[other]), dtype=torch.float32, device=device)
                    receives.append((parts[other], other, _COEFFICIENT_TAG))
            works = self._exchange(sends, receives)
            transient = max(transient, spec.rank_view * width * 6 + 2 * mine * spec.rank_view * 4)

            def finalize(sends=sends, staged=staged, parts=parts, start=start, mine=mine, step=step,
                         last=step == len(starts) - 1):
                del sends  # kept alive until the round completed
                if step == 0 and counts[rank]:
                    if len(owners) == 1:
                        state["basis"] = staged[owners[0]].float()
                    else:
                        basis = torch.zeros((spec.rank_view, width), dtype=torch.float32, device=device)
                        for owner in owners:
                            basis.index_copy_(0, index[owner], staged[owner].float())
                        state["basis"] = basis
                basis = state.pop("basis", None) if last else state.get("basis")
                if mine == 0:
                    return
                if len(owners) == 1:
                    coefficients = parts[owners[0]]
                else:
                    coefficients = torch.zeros((mine, spec.rank_view), dtype=torch.float32, device=device)
                    for owner in owners:
                        coefficients.index_copy_(1, index[owner], parts[owner])
                for t0 in range(0, mine, tile):
                    t1 = min(mine, t0 + tile)
                    values = coefficients[t0:t1] @ basis
                    if rotate is not None:
                        positions = (torch.arange(start + t0, start + t1, device=device) if heads
                                     else row_sets[rank][start + t0:start + t1])
                        values = rotate(entry, values, positions, own_columns)
                    target[start + t0:start + t1].copy_(values)

            self.counters["tiles"] += 1
            pending.append((works, finalize))
            while len(pending) > self.inflight:
                self._finish(pending.pop(0))
        self.counters["factor_transient_peak_bytes"] = max(self.counters["factor_transient_peak_bytes"], transient)

    def _issue(self, send, recv, finalize, pending, columns) -> None:
        self.counters["tiles"] += 1
        if self.world_size == 1:
            recv.copy_(send[0])
            finalize()
            return
        chunk = recv.numel() * 4
        self.counters["reduce_scatter_calls"] += 1
        self.counters["reduce_scatter_bytes_sent"] += (self.world_size - 1) * chunk
        self.counters["reduce_scatter_bytes_received"] += (self.world_size - 1) * chunk
        work = self._transport(self.dist.reduce_scatter_tensor, recv, send.reshape(-1, columns),
                               group=self.data_group, async_op=True)
        pending.append((work, finalize))

    def _slot(self, slots, pending, state):
        if len(pending) >= self.inflight:
            self._finish(pending.pop(0))
        slot = slots[state["slot"] % self.inflight]
        state["slot"] += 1
        return slot

    def _reconstruct_one(self, entry, shard, native, target, rope, slots, tile, columns, row_sets,
                         pending, state) -> None:
        world, rank = self.world_size, self.rank
        P = shard.coefficients
        Q = None if P is None else shard.view_basis().to(self.device, torch.float32)
        rotate = rope if (rope is not None and entry_parts(entry)[0].target == "K") else None
        if P is None:
            self.counters["zero_partial_calls"] += 1
        if native.kind == "heads":
            blocks = None if Q is None else [Q[:, native.columns(owner)].contiguous() for owner in range(world)]
            own = native.columns(rank)
            for t0 in range(0, native.tokens, tile):
                t1 = min(native.tokens, t0 + tile)
                n = t1 - t0
                flat_send, flat_recv = self._slot(slots, pending, state)
                send = flat_send[:world * n * columns].view(world, n, columns)
                recv = flat_recv[:n * columns].view(n, columns)
                if P is None:
                    send.zero_()
                else:
                    rows = P[t0:t1]
                    for owner in range(world):
                        torch.matmul(rows, blocks[owner], out=send[owner])

                def finalize(recv=recv, t0=t0, t1=t1):
                    values = recv
                    if rotate is not None:
                        values = rotate(entry, values, torch.arange(t0, t1, device=self.device), own)
                    target[t0:t1].copy_(values)

                self._issue(send, recv, finalize, pending, columns)
            return
        sizes = [int(rows.numel()) for rows in row_sets]
        for start in range(0, max(sizes), tile):
            n = max(min(tile, size - start) for size in sizes)
            flat_send, flat_recv = self._slot(slots, pending, state)
            send = flat_send[:world * n * columns].view(world, n, columns)
            recv = flat_recv[:n * columns].view(n, columns)
            for owner in range(world):
                index = row_sets[owner][start:start + n]
                m = int(index.numel())
                if P is None or m == 0:
                    send[owner].zero_()
                else:
                    torch.matmul(P.index_select(0, index), Q, out=send[owner, :m])
                    if m < n:
                        send[owner, m:].zero_()
                if owner != rank and m < n:
                    self.counters["reduce_scatter_padding_bytes"] += (n - m) * columns * 4
            mine = row_sets[rank][start:start + n]

            def finalize(recv=recv, start=start, mine=mine):
                m = int(mine.numel())
                if m == 0:
                    return
                values = recv[:m]
                if rotate is not None:
                    values = rotate(entry, values, mine, slice(0, native.channels))
                target[start:start + m].copy_(values)

            self._issue(send, recv, finalize, pending, columns)

    # -- accounting ----------------------------------------------------------------------------------------
    def aggregate_storage(self) -> dict[str, Any]:
        """Collective: every owner's store statistics and their totals (unique bytes, each object once)."""
        stats = None if self.store is None else self.store.stats()
        rows = self._agree("factor_storage", details=stats)
        totals = {}
        for row in rows:
            for name, value in (row or {}).items():
                totals[name] = totals.get(name, 0) + value
        return {"per_owner": rows, "totals": totals}

    def report(self) -> dict[str, Any]:
        return {**{name: (list(value) if isinstance(value, list) else value) for name, value in self.counters.items()},
                "world_size": self.world_size, "rank": self.rank, "tile_tokens": self.tile_tokens,
                "inflight": self.inflight, "checkpoint_every": self.checkpoint_every, "broken": self._broken,
                "accounting": "logical application bytes; NCCL/Gloo algorithm and protocol overhead excluded"}


def reference_reconstruction(coefficients: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """Single-process FP32 reference ``P @ Q`` (one GEMM) of a full rank view."""
    return coefficients.float() @ basis[:coefficients.shape[1]].float()
