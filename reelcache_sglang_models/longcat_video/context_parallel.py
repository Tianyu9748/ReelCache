"""LongCat-Video context parallelism with the native ``context_parallel_util`` semantics (pure torch).

Native LongCat-Video (``longcat_video/context_parallel/{context_parallel_util,ulysses_wrapper}.py``) runs one
sample on ``cp`` ranks:

* after the patch embedding every rank keeps one spatial block of every latent frame: a 2-D split
  ``(split_h, split_w) = get_optimal_split(cp)`` of the token grid (``(1, 2)`` for 2 ranks). Rank
  ``hi * split_w + wi`` owns token rows ``hi*H/split_h ..`` and columns ``wi*W/split_w ..`` of every frame, in
  frame-major, then row, then column order (= ``reelcache.factor_sharding.NativeOwners.spatial_rows``);
* self-attention is Ulysses: one all-to-all per q/k/v scatters the heads and gathers the sequence (the rank-major
  concatenation of the local blocks) before the kernel, the inverse all-to-all follows it; cross-attention and the
  per-token layers run on the local rows; 3D RoPE uses the rank's rows of the angle table (split first, then cos/sin);
* the final-layer output is all-gathered and reassembled (``GatherFunction2D``); every rank then holds the full
  prediction and steps the scheduler redundantly;
* the prompt embeddings and the initial latents are broadcast from CP rank 0 (``cp_broadcast``).

The conditioning cache each rank builds holds only its own token rows (pre-RoPE K and V; ``[B, T_local, H, D]`` in
this port, ``[B, H, T_local, D]`` natively): see :class:`RowShard` and the per-rank seam in :mod:`.conditioning_cache`.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist


class ContextParallelError(RuntimeError):
    """A context-parallel configuration or tensor violates the native LongCat CP contract."""


def optimal_split(size: int) -> tuple[int, int]:
    """Native ``get_optimal_split``: the factor pair ``(h, w)`` of ``size`` closest to square, ``h <= w``."""
    size = int(size)
    if size < 1:
        raise ContextParallelError("context-parallel size must be >= 1")
    factors = [(i, size // i) for i in range(1, int(size ** 0.5) + 1) if size % i == 0]
    return min(factors, key=lambda pair: abs(pair[0] - pair[1]))


def _grid(token_grid) -> tuple[int, int, int]:
    grid = tuple(int(x) for x in token_grid)
    if len(grid) != 3 or min(grid) <= 0:
        raise ContextParallelError(f"invalid token grid {token_grid}")
    return grid


@dataclass(frozen=True)
class RowShard:
    """One rank's spatial block of a token grid under the native 2-D split (plain, hashable data)."""

    world_size: int
    rank: int
    split_hw: tuple[int, int]

    def __post_init__(self) -> None:
        split = tuple(int(x) for x in self.split_hw)
        if len(split) != 2 or min(split) < 1:
            raise ContextParallelError(f"invalid split {self.split_hw}")
        object.__setattr__(self, "split_hw", split)
        if split[0] * split[1] != int(self.world_size):
            raise ContextParallelError(f"split {split} does not cover {self.world_size} ranks")
        if not 0 <= int(self.rank) < int(self.world_size):
            raise ContextParallelError(f"rank {self.rank} outside [0, {self.world_size})")

    @property
    def block(self) -> tuple[int, int]:
        """``(hi, wi)``: the rank's position in the split (native ``split_tensor_in_cp_2d`` order)."""
        return divmod(int(self.rank), self.split_hw[1])

    def check_grid(self, token_grid) -> tuple[int, int, int]:
        t, h, w = _grid(token_grid)
        if h % self.split_hw[0] or w % self.split_hw[1]:
            raise ContextParallelError(f"token grid {(t, h, w)} is not divisible by the split {self.split_hw}")
        return t, h, w

    def local_grid(self, token_grid) -> tuple[int, int, int]:
        t, h, w = self.check_grid(token_grid)
        return t, h // self.split_hw[0], w // self.split_hw[1]

    def local_tokens(self, token_grid) -> int:
        t, h, w = self.local_grid(token_grid)
        return t * h * w

    def rows(self, token_grid) -> torch.Tensor:
        """Native token rows (int64, CPU) owned by this rank, in the order the rank stores them."""
        t, h, w = self.check_grid(token_grid)
        hs, ws = h // self.split_hw[0], w // self.split_hw[1]
        hi, wi = self.block
        index = torch.arange(t * h * w, dtype=torch.long).reshape(t, h, w)
        return index[:, hi * hs:(hi + 1) * hs, wi * ws:(wi + 1) * ws].reshape(-1).contiguous()

    def native_owners_kwargs(self, token_grid, *, kv_heads: int, head_dim: int) -> dict:
        """Arguments of ``reelcache.factor_sharding.NativeOwners.spatial_rows`` describing this split."""
        t, h, w = self.check_grid(token_grid)
        return {"frames": t, "height": h, "width": w, "split_h": self.split_hw[0], "split_w": self.split_hw[1],
                "kv_heads": int(kv_heads), "head_dim": int(head_dim)}

    def as_dict(self) -> dict:
        return {"world_size": int(self.world_size), "rank": int(self.rank), "split_hw": list(self.split_hw),
                "owner_rule": "native 2-D spatial split of every latent frame (NativeOwners.spatial_rows)"}

    @classmethod
    def from_dict(cls, data: dict) -> "RowShard":
        return cls(int(data["world_size"]), int(data["rank"]), tuple(data["split_hw"]))


def split_tokens(x: torch.Tensor, token_grid, shard: RowShard) -> torch.Tensor:
    """``[B, T*H*W, C]`` -> the shard's ``[B, T*h*w, C]`` rows (native ``split_cp_2d`` after the patch embed)."""
    t, h, w = shard.check_grid(token_grid)
    if x.ndim != 3 or x.shape[1] != t * h * w:
        raise ContextParallelError(f"tokens {tuple(x.shape)} do not match the grid {(t, h, w)}")
    hs, ws = h // shard.split_hw[0], w // shard.split_hw[1]
    hi, wi = shard.block
    view = x.view(x.shape[0], t, h, w, x.shape[2])
    return view[:, :, hi * hs:(hi + 1) * hs, wi * ws:(wi + 1) * ws].reshape(x.shape[0], -1, x.shape[2])


def assemble_tokens(blocks, token_grid, split_hw) -> torch.Tensor:
    """Reassemble every rank's ``[B, T*h*w, C]`` block into ``[B, T*H*W, C]`` (native ``GatherFunction2D``)."""
    split_h, split_w = (int(x) for x in split_hw)
    t, h, w = _grid(token_grid)
    if len(blocks) != split_h * split_w:
        raise ContextParallelError(f"{len(blocks)} blocks for split {split_hw}")
    hs, ws = h // split_h, w // split_w
    shaped = [b.reshape(b.shape[0], t, hs, ws, b.shape[-1]) for b in blocks]
    rows = [torch.cat(shaped[i * split_w:(i + 1) * split_w], dim=3) for i in range(split_h)]
    out = torch.cat(rows, dim=2)
    return out.reshape(out.shape[0], t * h * w, out.shape[-1])


def all_to_all(tensor: torch.Tensor, scatter_idx: int, gather_idx: int, group, world_size: int) -> torch.Tensor:
    """Native ``ulysses_wrapper.all_to_all`` (``gather=True``): split ``scatter_idx`` into ``world_size`` chunks
    (chunk ``r`` goes to rank ``r``) and concatenate the received chunks along ``gather_idx`` in rank order."""
    if world_size == 1:
        return tensor
    if scatter_idx == gather_idx:
        raise ContextParallelError("scatter_idx and gather_idx must be different")
    shape = list(tensor.shape)
    if shape[scatter_idx] % world_size:
        raise ContextParallelError(f"dimension {scatter_idx} ({shape[scatter_idx]}) is not divisible by {world_size}")
    chunk = shape[scatter_idx] // world_size
    split_shape = shape[:scatter_idx] + [world_size, chunk] + shape[scatter_idx + 1:]
    order = [scatter_idx] + [i for i in range(len(split_shape)) if i != scatter_idx]
    send = tensor.reshape(*split_shape).permute(*order).contiguous()
    recv = torch.empty_like(send)
    dist.all_to_all_single(recv, send, group=group)
    # recv: [world (source rank), *shape with scatter_idx chunked]; move the source axis in front of gather_idx
    dims = list(recv.shape)
    permute = []
    for i in range(1, len(dims)):
        if i == gather_idx + 1:
            permute.extend([0, i])
        else:
            permute.append(i)
    out = recv.permute(*permute).contiguous()
    merged = [dims[i] if i != gather_idx + 1 else world_size * dims[i] for i in range(1, len(dims))]
    return out.reshape(*merged)


class LongCatContextParallel:
    """A context-parallel group (``torch.distributed`` process group) with the native split and collectives.

    Tensors use this port's ``[B, S, H, D]`` attention layout: ``ulysses_in`` scatters heads (dim 2) and gathers the
    sequence (dim 1), ``ulysses_out`` is the inverse -- the native ``[B, H, S, D]`` exchange with permuted axes.
    """

    def __init__(self, group=None, *, split_hw: tuple[int, int] | None = None, label: str = "torch.distributed") -> None:
        if not dist.is_available() or not dist.is_initialized():
            raise ContextParallelError("context parallelism needs an initialized torch.distributed process group")
        self.group = group
        self.size = dist.get_world_size(group)
        self.rank = dist.get_rank(group)
        if self.size < 2:
            raise ContextParallelError("a context-parallel group needs at least 2 ranks")
        split = optimal_split(self.size) if split_hw is None else tuple(int(x) for x in split_hw)
        self.shard = RowShard(self.size, self.rank, split)
        self.split_hw = self.shard.split_hw
        self.src_global_rank = dist.get_global_rank(group, 0) if group is not None else 0
        self.backend = str(dist.get_backend(group))
        self.label = label
        self._rows: dict = {}

    # -- geometry ------------------------------------------------------------------------------------
    @property
    def rope_shard(self) -> tuple[int, int, int]:
        return (self.split_hw[0], self.split_hw[1], self.rank)

    def row_shard(self) -> RowShard:
        return self.shard

    def local_rows(self, token_grid) -> torch.Tensor:
        key = _grid(token_grid)
        if key not in self._rows:
            self._rows[key] = self.shard.rows(key)
        return self._rows[key]

    def split_tokens(self, x: torch.Tensor, token_grid) -> torch.Tensor:
        return split_tokens(x, token_grid, self.shard)

    # -- collectives (native semantics) -------------------------------------------------------------------
    def gather_tokens(self, x_local: torch.Tensor, token_grid) -> torch.Tensor:
        """All-gather every rank's ``[B, T*h*w, C]`` block and reassemble ``[B, T*H*W, C]`` on every rank."""
        expect = self.shard.local_tokens(token_grid)
        if x_local.ndim != 3 or x_local.shape[1] != expect:
            raise ContextParallelError(f"local tokens {tuple(x_local.shape)} != {expect} rows of {token_grid}")
        x_local = x_local.contiguous()
        blocks = [torch.empty_like(x_local) for _ in range(self.size)]
        dist.all_gather(blocks, x_local, group=self.group)
        return assemble_tokens(blocks, token_grid, self.split_hw)

    def ulysses_in(self, x: torch.Tensor) -> torch.Tensor:
        """``[B, S/N, H, D]`` -> ``[B, S, H/N, D]`` (heads scattered, sequence gathered in rank order)."""
        if x.ndim != 4:
            raise ContextParallelError("Ulysses exchange expects [B, S, H, D]")
        return all_to_all(x, 2, 1, self.group, self.size)

    def ulysses_out(self, x: torch.Tensor) -> torch.Tensor:
        """``[B, S, H/N, D]`` -> ``[B, S/N, H, D]``."""
        if x.ndim != 4:
            raise ContextParallelError("Ulysses exchange expects [B, S, H, D]")
        return all_to_all(x, 1, 2, self.group, self.size)

    def broadcast(self, tensor: torch.Tensor) -> torch.Tensor:
        """In-place broadcast from CP rank 0 (native ``cp_broadcast``)."""
        if not tensor.is_contiguous():
            raise ContextParallelError("broadcast needs a contiguous tensor")
        dist.broadcast(tensor, src=self.src_global_rank, group=self.group)
        return tensor

    def barrier(self) -> None:
        dist.barrier(group=self.group)

    def describe(self) -> dict:
        return {"cp_size": self.size, "cp_rank": self.rank, "split_hw": list(self.split_hw), "backend": self.backend,
                "group": self.label, "src_global_rank": self.src_global_rank,
                "exchange": "Ulysses all-to-all (heads scattered, sequence gathered rank-major); final-layer "
                            "all-gather; prompt/latents broadcast from CP rank 0"}
