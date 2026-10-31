"""LongCat-Video 3D rotary position embedding (pure torch).

Bitwise replica of ``longcat_video/modules/rope_3d.py`` (``RotaryPositionalEmbedding`` with
``cp_split_hw=[1, 1]``):

* head_dim 128 is split into temporal / height / width bands of 44 / 42 / 42 channels;
* every band uses ``1 / base ** (arange(0, d, 2)[: d // 2] / d)`` frequencies, repeated pairwise
  (``(n r)`` with ``r = 2``), i.e. interleaved (GPT-J style) rotation pairs;
* the angle table is built on the CPU in FP32 (as native does); cos/sin are taken on the compute
  device in FP32 and the rotation is computed in FP32 and cast back to the input dtype.

Positions are segment-local: a continuation segment's conditioning latents occupy temporal
positions ``0 .. n_cond-1`` and its new latents ``n_cond .. n_total-1``. Rows of a ``(T, H, W)``
table equal the first rows of any longer ``(T', H, W)`` table, which is why the conditioning cache
can hold pre-RoPE keys and have RoPE applied at read time.

Tensor layout in this port is ``[B, S, H, D]`` (SGLang/FlashAttention layout); the native code uses
``[B, H, S, D]``. Both layouts apply the same per-row math.

Under context parallelism (``cp_split_hw`` > 1 natively) a rank rotates only its spatial block of every frame:
the native code splits the FP32 angle table first and takes cos/sin of the split table; :meth:`RopeTables.cos_sin`
with ``shard=(split_h, split_w, rank)`` does the same (rows of :class:`~.context_parallel.RowShard`).
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import torch

from .context_parallel import RowShard


def rope_axis_dims(head_dim: int) -> tuple[int, int, int]:
    if head_dim % 8:
        raise ValueError("Dim must be a multiple of 8 for 3D RoPE")
    return head_dim - 4 * (head_dim // 6), 2 * (head_dim // 6), 2 * (head_dim // 6)


def _axis_angles(n: int, dim: int, base: float) -> torch.Tensor:
    inv = 1.0 / (base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    grid = torch.from_numpy(np.linspace(0, n, n, endpoint=False, dtype=np.float32)).float()
    angles = torch.einsum("..., f -> ... f", grid, inv)
    return angles.repeat_interleave(2, dim=-1)


def rope_angles_3d(grid_size: tuple[int, int, int], head_dim: int = 128, base: float = 10000.0) -> torch.Tensor:
    """FP32 CPU angle table ``[T*H*W, head_dim]`` in (t, h, w) row-major token order."""
    num_frames, height, width = (int(x) for x in grid_size)
    if min(num_frames, height, width) <= 0:
        raise ValueError(f"invalid RoPE grid {grid_size}")
    dim_t, dim_h, dim_w = rope_axis_dims(head_dim)
    ft = _axis_angles(num_frames, dim_t, base)
    fh = _axis_angles(height, dim_h, base)
    fw = _axis_angles(width, dim_w, base)
    freqs = torch.cat([
        ft[:, None, None, :].expand(num_frames, height, width, dim_t),
        fh[None, :, None, :].expand(num_frames, height, width, dim_h),
        fw[None, None, :, :].expand(num_frames, height, width, dim_w),
    ], dim=-1)
    return freqs.reshape(num_frames * height * width, head_dim)


def rotate_half_interleaved(x: torch.Tensor) -> torch.Tensor:
    """``rearrange(x, '... (d r) -> ... d r', r=2)``, then ``(-x2, x1)`` and back."""
    pairs = x.unflatten(-1, (-1, 2))
    x1, x2 = pairs.unbind(dim=-1)
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, *, seq_dim: int = 1) -> torch.Tensor:
    """Rotate ``x`` (sequence on ``seq_dim``, head_dim last) with ``[S, D]`` FP32 cos/sin tables."""
    if cos.shape != sin.shape or cos.ndim != 2:
        raise ValueError("cos/sin must be [S, D]")
    seq_dim = seq_dim % x.ndim
    if x.shape[seq_dim] != cos.shape[0] or x.shape[-1] != cos.shape[1]:
        raise ValueError(f"RoPE table {tuple(cos.shape)} does not match x {tuple(x.shape)} on dim {seq_dim}")
    view = [1] * x.ndim
    view[seq_dim] = cos.shape[0]
    view[-1] = cos.shape[1]
    c = cos.view(view)
    s = sin.view(view)
    x_ = x.float()
    out = (x_ * c) + (rotate_half_interleaved(x_) * s)
    return out.type_as(x)


class RopeTables:
    """Per-device cache of FP32 cos/sin tables keyed by token grid (the native ``freqs_dict``)."""

    def __init__(self, head_dim: int = 128, base: float = 10000.0) -> None:
        self.head_dim = int(head_dim)
        self.base = float(base)
        self._angles: dict[tuple[int, int, int], torch.Tensor] = {}
        self._tables: dict[tuple[tuple[int, int, int], str], tuple[torch.Tensor, torch.Tensor]] = {}

    def angles(self, grid_size: tuple[int, int, int]) -> torch.Tensor:
        key = tuple(int(x) for x in grid_size)
        if key not in self._angles:
            self._angles[key] = rope_angles_3d(key, self.head_dim, self.base)
        return self._angles[key]

    def cos_sin(self, grid_size: tuple[int, int, int], device: torch.device | str,
                shard: tuple[int, int, int] | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """FP32 cos/sin ``[S, D]`` of ``grid_size``; with ``shard=(split_h, split_w, rank)`` only that rank's rows
        (the angle table is split before cos/sin, as native ``precompute_freqs_cis_3d`` does under CP)."""
        grid = tuple(int(x) for x in grid_size)
        key = (grid, str(torch.device(device)), None if shard is None else tuple(int(x) for x in shard))
        if key not in self._tables:
            angles = self.angles(grid)
            if shard is not None:
                split_h, split_w, rank = key[2]
                angles = angles.index_select(0, RowShard(split_h * split_w, rank, (split_h, split_w)).rows(grid))
            freqs = angles.to(device).float()
            self._tables[key] = (freqs.cos(), freqs.sin())
        return self._tables[key]

    def rows(self, grid_size: tuple[int, int, int], device, start_frame: int, num_frames: int):
        """cos/sin rows of temporal frames ``[start_frame, start_frame + num_frames)`` of ``grid_size``."""
        t, h, w = (int(x) for x in grid_size)
        if start_frame < 0 or start_frame + num_frames > t:
            raise ValueError(f"frames [{start_frame}, {start_frame + num_frames}) outside grid T={t}")
        cos, sin = self.cos_sin(grid_size, device)
        lo, hi = start_frame * h * w, (start_frame + num_frames) * h * w
        return cos[lo:hi], sin[lo:hi]

    def clear(self) -> None:
        self._angles.clear()
        self._tables.clear()


def segment_positions(num_cond_latents: int, num_new_latents: int) -> dict:
    """Segment-local temporal positions of one continuation segment (no cross-segment offset)."""
    if num_cond_latents < 0 or num_new_latents < 0:
        raise ValueError("latent counts must be non-negative")
    return {"cache_frames": list(range(num_cond_latents)),
            "new_frames": list(range(num_cond_latents, num_cond_latents + num_new_latents)),
            "rope_grid_frames": num_cond_latents + num_new_latents,
            "convention": "segment_local: conditioning latents at t=0..n_cond-1, new latents follow"}


def native_rope_reference_layout(tensors: Iterable[torch.Tensor]) -> list[torch.Tensor]:
    """Permute ``[B, S, H, D]`` tensors to the native ``[B, H, S, D]`` layout (for parity checks)."""
    return [t.permute(0, 2, 1, 3) for t in tensors]
