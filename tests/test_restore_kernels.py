"""CPU tests for the batched restore reconstruction (reelcache.restore_kernels).

These run in the standard CPU suite without skips. GPU parity, native-code
RoPE parity and performance live in
artifacts/sglang_integration/benchmarks/restore_kernels/ and run in GPU jobs.

The native RoPE functions below are line-by-line replicas (the model repos and
SGLang are not imported by the CPU suite). Their sources, with the SHA256 of
each native function's source segment when replicated, are listed in
``NATIVE_SOURCES``; the GPU test script imports the real native code, checks
those hashes and compares replicas, native code and the kernels.
"""

import ast
import contextlib
import hashlib
import itertools
import json
import math
import os
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from torch.overrides import TorchFunctionMode

from reelcache import _hevc, restore_kernels as rk
from reelcache.codecs import dequantize_device_coefficients
from reelcache.errors import ConfigurationError, IntegrityError, UnsupportedError
from reelcache.nvdec import unpack_device_frame

ROOT = Path(__file__).resolve().parents[1]

NATIVE_SOURCES = {
    "wan_rope_params": ("video-models/RAVEN/third_party/causvid/modeling/model.py::rope_params "
                        "(identical in RAVEN project and LongLive wan_5b)",
                        "8fd95284724635e63b2b52c5425e9a28dac14666350cf7562114c071d95366db"),
    "causvid_causal_rope_apply": ("video-models/RAVEN/third_party/causvid/modeling/causal_model.py::causal_rope_apply "
                                  "(LongLive 1.0 third_party/longlive differs only in line wrapping)",
                                  "e505532f737d5bce34d20f0a89071a279e301ab9f004ca0b49ea5ffd61fa22cc"),
    "raven_apply_latent_pos_embed": ("video-models/RAVEN/project/models/wan2_1/causal_model.py::apply_latent_pos_embed",
                                     "1a6b91c86124b0966558dd2bbae47e045a906615747ec3bca00313836f6ee9de"),
    "longlive2_causal_rope_apply": ("video-models/LongLive/wan_5b/modules/causal_model.py::causal_rope_apply "
                                    "(float64 path, LLV2_TRITON_ROPE=0)",
                                    "14529b8bf1d5f2d61ed2fffed1aa0e94b5180d9d88ebdad643a459792b91664b"),
    "longlive2_compute_temporal_freqs": ("video-models/LongLive/utils/position_embedding_utils.py::compute_temporal_freqs",
                                         "5c2339ec01edc3a09869618ce6dd5dd75c2321ac44b1a409f7dbbf39e2bfb9f4"),
    "longcat_rope_3d": ("video-models/LongCat-Video/longcat_video/modules/rope_3d.py::RotaryPositionalEmbedding "
                        "(einops rearrange/repeat written as torch reshapes)",
                        "c77aeedfcd15d18a8b7f0c127795b2495643e2598886bdf34ef61442ea017f20"),
    "sglang_rotary_native": ("sglang/kernels/ops/diffusion/common/fallback_torch.py::apply_rotary_embedding_native",
                             "3aaef21716443651d6e49fc7aa6f3e46c9354ffe1c249e43dd97e184605ff724"),
    "sglang_nd_rotary_grid": ("sglang/multimodal_gen/runtime/layers/rotary_embedding/mrope.py::"
                              "NDRotaryEmbedding.forward_from_grid (sp_world_size=1)",
                              "768d9d42676f126272663b51b7425d2f568784faef62449d99c3f81f653c1680"),
}


# ---------------------------------------------------------------------------
# Native RoPE replicas
# ---------------------------------------------------------------------------


def wan_rope_params(max_seq_len, dim, theta=10000):
    assert dim % 2 == 0
    freqs = torch.outer(
        torch.arange(max_seq_len),
        1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float64).div(dim)))
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs


def wan_freqs(head_dim, rows=64):
    d = head_dim
    return torch.cat([wan_rope_params(rows, d - 4 * (d // 6)), wan_rope_params(rows, 2 * (d // 6)),
                      wan_rope_params(rows, 2 * (d // 6))], dim=1)


def causvid_causal_rope_apply(x, grid_sizes, freqs, start_frame=0):
    n, c = x.size(2), x.size(3) // 2
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(seq_len, n, -1, 2))
        freqs_i = torch.cat([
            freqs[0][start_frame:start_frame + f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ], dim=-1).reshape(seq_len, 1, -1)
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])
        output.append(x_i)
    return torch.stack(output).type_as(x)


def raven_apply_latent_pos_embed(xs, grid_sizes, freqs, frame_shifts=None, packed=False):
    c = freqs.size(1)
    if frame_shifts is None:
        frame_shifts = [0] * len(grid_sizes)
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
    output = []
    curr = 0
    for i, ((f, h, w), fs) in enumerate(zip(grid_sizes, frame_shifts)):
        seq_len = f * h * w
        if packed:
            x = xs[curr:curr + seq_len]
            curr += seq_len
        else:
            x = xs[i]
        x_i = torch.view_as_complex(x.contiguous().to(torch.float64).reshape(seq_len, -1, c, 2))
        freqs_i = torch.cat([
            freqs[0][fs:fs + f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ], dim=-1).reshape(seq_len, 1, -1)
        if packed:
            x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        else:
            x_i = torch.view_as_real(x_i * freqs_i).flatten(1)
        output.append(x_i)
    if packed:
        assert curr == xs.size(0)
        return torch.cat(output).float()
    return [u.float() for u in output]


def longlive2_compute_temporal_freqs(freqs_t, f, start_frame, t_scale, device, method="linear",
                                     original_seq_len=None, temporal_offset=0.0):
    if temporal_offset is None:
        temporal_offset = 0.0
    if t_scale == 1.0 and not torch.is_tensor(temporal_offset) and float(temporal_offset) == 0.0:
        return freqs_t[start_frame:start_frame + f]
    base_angles = torch.angle(freqs_t[1]).to(torch.float64)
    positions = torch.arange(f, device=device, dtype=torch.float64) + start_frame
    if torch.is_tensor(temporal_offset):
        offset = temporal_offset.to(device=device, dtype=torch.float64)
        if offset.ndim == 0:
            positions = positions + offset
        elif offset.numel() == f:
            positions = positions + offset
        else:
            positions = positions + offset[start_frame:start_frame + f]
    else:
        positions = positions + float(temporal_offset)
    positions = positions * t_scale
    angles = positions.unsqueeze(-1) * base_angles.unsqueeze(0)
    return torch.polar(torch.ones_like(angles), angles)


def longlive2_causal_rope_apply(x, grid_sizes, freqs, start_frame=0, t_scale=1.0, temporal_offset=0.0):
    """LongLive 2.0 float64 path (Triton RoPE disabled), one sample, scalar/[F] offsets."""
    n, c = x.size(2), x.size(3) // 2
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
    output = []
    for i, (f, h, w) in enumerate(grid_sizes):
        seq_len = f * h * w
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(seq_len, n, -1, 2))
        temporal_freqs = longlive2_compute_temporal_freqs(freqs[0], f, start_frame, t_scale, x.device,
                                                          temporal_offset=temporal_offset)
        freqs_i = torch.cat([
            temporal_freqs.view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
        ], dim=-1).reshape(seq_len, 1, -1)
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])
        output.append(x_i)
    return torch.stack(output).type_as(x)


def longcat_rotate_half(x):
    x = x.reshape(*x.shape[:-1], -1, 2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return x.reshape(*x.shape[:-2], -1)


def longcat_precompute_freqs_cis_3d(head_dim, grid_size, base=10000):
    num_frames, height, width = grid_size
    dim_t = head_dim - 4 * (head_dim // 6)
    dim_h = 2 * (head_dim // 6)
    dim_w = 2 * (head_dim // 6)
    freqs_t = 1.0 / (base ** (torch.arange(0, dim_t, 2)[: (dim_t // 2)].float() / dim_t))
    freqs_h = 1.0 / (base ** (torch.arange(0, dim_h, 2)[: (dim_h // 2)].float() / dim_h))
    freqs_w = 1.0 / (base ** (torch.arange(0, dim_w, 2)[: (dim_w // 2)].float() / dim_w))
    grid_t = torch.from_numpy(np.linspace(0, num_frames, num_frames, endpoint=False, dtype=np.float32)).float()
    grid_h = torch.from_numpy(np.linspace(0, height, height, endpoint=False, dtype=np.float32)).float()
    grid_w = torch.from_numpy(np.linspace(0, width, width, endpoint=False, dtype=np.float32)).float()
    freqs_t = torch.einsum("..., f -> ... f", grid_t, freqs_t).repeat_interleave(2, dim=-1)
    freqs_h = torch.einsum("..., f -> ... f", grid_h, freqs_h).repeat_interleave(2, dim=-1)
    freqs_w = torch.einsum("..., f -> ... f", grid_w, freqs_w).repeat_interleave(2, dim=-1)
    shape = (num_frames, height, width)
    freqs = torch.cat([freqs_t[:, None, None, :].expand(*shape, -1), freqs_h[None, :, None, :].expand(*shape, -1),
                       freqs_w[None, None, :, :].expand(*shape, -1)], dim=-1)
    return freqs.reshape(num_frames * height * width, -1)


def longcat_rope_3d_key(k, grid_size):
    """LongCat ``RotaryPositionalEmbedding.forward`` for K ``[B, H, S, D]``."""
    freqs_cis = longcat_precompute_freqs_cis_3d(k.shape[-1], grid_size).to(k.device)
    k_ = k.float()
    freqs_cis = freqs_cis.float().to(k.device)
    cos, sin = freqs_cis.cos(), freqs_cis.sin()
    cos, sin = cos[None, None], sin[None, None]
    k_ = (k_ * cos) + (longcat_rotate_half(k_) * sin)
    return k_.type_as(k)


def sglang_apply_rotary_embedding_native(x, cos, sin, interleaved=False):
    if interleaved and cos.shape[-1] == x.shape[-1]:
        cos = cos[..., ::2]
        sin = sin[..., ::2]
    cos = cos.unsqueeze(-2).to(x.dtype)
    sin = sin.unsqueeze(-2).to(x.dtype)
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    return torch.stack((o1, o2), dim=-1).flatten(-2)


def sglang_rotary_cos_sin(grid, start_frame, head_dim, theta=10000.0, dtype=torch.float64):
    """``NDRotaryEmbedding.forward_from_grid`` with OneD generators, one SP rank."""
    d = head_dim
    rope_dim_list = [d - 4 * (d // 6), 2 * (d // 6), 2 * (d // 6)]
    sizes = grid
    num_tokens = math.prod(sizes)
    cos = torch.empty((num_tokens, sum(rope_dim_list) // 2), dtype=dtype)
    sin = torch.empty_like(cos)
    col = 0
    for i, dim in enumerate(rope_dim_list):
        offset = start_frame if i == 0 else 0
        pos = torch.arange(offset, offset + sizes[i], dtype=dtype)
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=dtype)[: (dim // 2)] / dim))
        angles = torch.outer(pos, freqs)
        cos_1d, sin_1d = angles.cos().float(), angles.sin().float()
        repeats = math.prod(sizes[i + 1:])
        tiles = math.prod(sizes[:i])
        cos_e = cos_1d.repeat_interleave(repeats, dim=0).repeat(tiles, 1)
        sin_e = sin_1d.repeat_interleave(repeats, dim=0).repeat(tiles, 1)
        cos[:, col:col + dim // 2] = cos_e
        sin[:, col:col + dim // 2] = sin_e
        col += dim // 2
    return cos.float(), sin.float()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def custom_layout(frames, height, width, rank, columns, rows, planes):
    slots = columns * rows * planes
    return {
        "schema": "reelcache.channel_tiles.v1", "frames": frames, "height": height, "width": width, "rank": rank,
        "pix_fmt": "yuv444p" if planes == 3 else "yuv420p", "grid": [columns, rows], "planes": planes,
        "channels_per_segment": slots, "segments": math.ceil(rank / slots),
    }


def pack_with_cells(pixels, layout, origin=(0, 0), extra=(3, 5), filler=None):
    """Canvases [segments*frames, 3, H, W] built with the codec's own cell loop."""
    frames, height, width, rank = layout["frames"], layout["height"], layout["width"], layout["rank"]
    columns, rows = layout["grid"]
    slots = layout["channels_per_segment"]
    values = pixels.reshape(frames, height, width, rank)
    y0, x0 = origin
    shape = (layout["segments"] * frames, 3, rows * height + y0 + extra[0], columns * width + x0 + extra[1])
    canvas = torch.full(shape, 7, dtype=torch.uint8) if filler is None else filler(shape)
    indices = []
    for segment in range(layout["segments"]):
        first = segment * slots
        count = min(slots, rank - first)
        for frame in range(frames):
            for index, plane, y, x in _hevc._cells(layout, count):
                canvas[segment * frames + frame, plane, y0 + y:y0 + y + height, x0 + x:x0 + x + width] = \
                    values[frame, :, :, first + index]
        indices.append([segment * frames + frame for frame in range(frames)])
    return canvas, indices


def random_affine(rank, rng):
    lo = rng.uniform(-4, 0, size=rank).astype(np.float32)
    hi = (lo.astype(np.float64) + rng.uniform(0.5, 6, size=rank)).astype(np.float32)
    affine = _hevc._covering_affine(lo, hi)
    return affine[0].copy(), affine[1].copy()


def make_picture_job(rng, destination, grid, basis, rank, *, layout=None, rope=None, origin=(2, 3)):
    frames, height, width = grid
    tokens = frames * height * width
    layout = _hevc.video_layout(frames, height, width, rank, "yuv444p") if layout is None else layout
    pixels = torch.from_numpy(rng.integers(0, 256, size=(tokens, rank), dtype=np.uint8))
    scale, offset = random_affine(rank, rng)
    canvas, indices = pack_with_cells(pixels, layout, origin)
    source = rk.PictureSource.from_canvases(canvas, layout, indices, origin, scale, offset)
    return rk.ReconstructionJob(source, destination, grid, basis, rank, rope), pixels


def explicit_values(job):
    """Independent FP64 reconstruction: dequantize, project, rotate with complex math."""
    coefficients = rk.reference_coefficients(job).double()
    values = coefficients @ job.basis[:job.rank].double()
    if job.rope is not None:
        heads, head_dim = job.destination.heads, job.destination.head_dim
        x = torch.view_as_complex(values.reshape(job.tokens, heads, head_dim // 2, 2).contiguous())
        cos, sin = rk.rope_cos_sin(job.rope, job.grid)
        values = torch.view_as_real(x * torch.complex(cos, sin)[:, None, :]).reshape(job.tokens, -1)
    return values


class ReferenceLayoutTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.rng = np.random.default_rng(0)
        self.grid = (2, 3, 4)
        self.tokens = 24
        self.heads, self.head_dim = 2, 12
        self.channels = 24
        self.basis = torch.randn(10, self.channels).half()
        self.table = rk.RopeTable.from_complex(wan_freqs(self.head_dim))

    def test_layouts_offsets_and_untouched_tokens(self):
        sentinel = -123.0
        results = {}
        for layout in rk.LAYOUTS:
            if layout == "BTHD":
                tensor = torch.full((1, 3 * self.tokens, self.heads, self.head_dim), sentinel)
            elif layout == "THD":
                tensor = torch.full((3 * self.tokens, self.heads, self.head_dim), sentinel)
            else:
                tensor = torch.full((1, self.heads, 3 * self.tokens, self.head_dim), sentinel)
            rng = np.random.default_rng(1)
            job, _ = make_picture_job(rng, rk.NativeView(tensor, layout, token_offset=self.tokens), self.grid,
                                      self.basis, 7, rope=rk.RopePositions(self.table, (2, 0, 0)))
            rk.reconstruct([job], backend="reference")
            slot = job.destination.slot(self.tokens).reshape(self.tokens, -1)
            torch.testing.assert_close(slot.double(), explicit_values(job).float().double(), rtol=0, atol=0)
            canonical = tensor if layout != "BHTD" else tensor.transpose(1, 2)
            canonical = canonical.reshape(3 * self.tokens, -1)
            self.assertTrue(torch.all(canonical[:self.tokens] == sentinel))
            self.assertTrue(torch.all(canonical[2 * self.tokens:] == sentinel))
            results[layout] = canonical[self.tokens:2 * self.tokens].clone()
        torch.testing.assert_close(results["BTHD"], results["THD"], rtol=0, atol=0)
        torch.testing.assert_close(results["BTHD"], results["BHTD"], rtol=0, atol=0)

    def test_non_contiguous_destinations(self):
        padded = torch.zeros(1, self.tokens + 5, self.heads, self.head_dim + 4)
        view = padded[:, 2:, :, 1:1 + self.head_dim]
        permuted_storage = torch.zeros(self.head_dim, self.heads, self.tokens)
        permuted = permuted_storage.permute(2, 1, 0)
        rng = np.random.default_rng(2)
        jobs = []
        for tensor, layout in ((view, "BTHD"), (permuted, "THD")):
            job, _ = make_picture_job(rng, rk.NativeView(tensor, layout), self.grid, self.basis, 9,
                                      rope=rk.RopePositions(self.table, (5, 1, 0)))
            jobs.append(job)
        rk.reconstruct(jobs, backend="reference")
        for job in jobs:
            slot = job.destination.slot(self.tokens).reshape(self.tokens, -1)
            torch.testing.assert_close(slot.double(), explicit_values(job).float().double(), rtol=0, atol=0)
        self.assertTrue(torch.all(padded[:, :2] == 0))
        self.assertTrue(torch.all(padded[..., 0] == 0) and torch.all(padded[..., -3:] == 0))

    def test_prefix_rank_views_and_mixed_ranks(self):
        destination = torch.zeros(1, 3 * self.tokens, self.heads, self.head_dim)
        coefficients = torch.randn(self.tokens, 10)
        jobs = [
            rk.ReconstructionJob(rk.CoefficientSource(coefficients), rk.NativeView(destination, token_offset=index * self.tokens),
                                 self.grid, self.basis, rank)
            for index, rank in enumerate((3, 10, 6))
        ]
        rk.reconstruct(jobs, backend="reference", compute_dtype=torch.float32)
        for job, rank in zip(jobs, (3, 10, 6)):
            expected = coefficients[:, :rank] @ self.basis[:rank].float()
            torch.testing.assert_close(job.destination.slot(self.tokens).reshape(self.tokens, -1), expected,
                                       rtol=0, atol=0)

    def test_empty_single_and_dense(self):
        handle = rk.reconstruct([], backend="auto")
        self.assertTrue(handle.query())
        self.assertIsNone(handle.event)
        handle.release()
        self.assertTrue(handle.released)
        destination = torch.zeros(self.tokens, self.heads, self.head_dim, dtype=torch.bfloat16)
        dense = torch.randn(self.tokens, self.channels)
        job = rk.ReconstructionJob(rk.DenseSource(dense), rk.NativeView(destination, "THD"), self.grid)
        handle = rk.reconstruct(job)
        self.assertEqual(handle.stats["backend"], "reference")
        torch.testing.assert_close(destination.reshape(self.tokens, -1), dense.bfloat16(), rtol=0, atol=0)
        self.assertTrue(handle.query())

    def test_quantized_coefficient_matrix_source(self):
        destination = torch.zeros(1, self.tokens, self.heads, self.head_dim, dtype=torch.float32)
        pixels = torch.from_numpy(self.rng.integers(0, 256, size=(self.tokens, 8), dtype=np.uint8))
        scale, offset = random_affine(8, self.rng)
        job = rk.ReconstructionJob(rk.CoefficientSource(pixels, scale, offset), rk.NativeView(destination),
                                   self.grid, self.basis, 5)
        rk.reconstruct([job])
        expected = (pixels[:, :5].float() * torch.from_numpy(scale[:5]) + torch.from_numpy(offset[:5])).double() \
            @ self.basis[:5].double()
        torch.testing.assert_close(destination.reshape(self.tokens, -1), expected.float(), rtol=0, atol=0)

    def test_error_bound_holds_for_reference_casts(self):
        for dtype in rk.OUTPUT_DTYPES:
            destination = torch.zeros(1, self.tokens, self.heads, self.head_dim, dtype=dtype)
            job, _ = make_picture_job(self.rng, rk.NativeView(destination), self.grid, self.basis, 10,
                                      rope=rk.RopePositions(self.table, (1, 0, 0)))
            rk.reconstruct([job])
            exact, bound = rk.error_bound(job, "fp32")
            written = destination.reshape(self.tokens, -1).double()
            self.assertTrue(bool(((written - exact).abs() <= bound).all()), dtype)


class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.grid = (1, 2, 3)
        self.destination = torch.zeros(1, 12, 2, 4)
        self.basis = torch.randn(5, 8).half()
        self.values = torch.randn(6, 5)

    def job(self, **changes):
        arguments = dict(source=rk.CoefficientSource(self.values), destination=rk.NativeView(self.destination),
                         grid=self.grid, basis=self.basis, rank=5)
        arguments.update(changes)
        return rk.ReconstructionJob(**arguments)

    def test_rejects_inconsistent_jobs(self):
        with self.assertRaises(ConfigurationError):
            self.job(rank=6)
        with self.assertRaises(ConfigurationError):
            self.job(grid=(2, 2, 3))
        with self.assertRaises(ConfigurationError):
            self.job(basis=torch.randn(5, 6).half())
        with self.assertRaises(ConfigurationError):
            self.job(destination=rk.NativeView(self.destination, token_offset=8))
        with self.assertRaises(ConfigurationError):
            rk.NativeView(self.destination, "THD")
        with self.assertRaises(ConfigurationError):
            rk.NativeView(torch.zeros(1, 4, 2, 4, dtype=torch.int32))
        with self.assertRaises(ConfigurationError):
            rk.NativeView(torch.zeros(1, 1, 1, 4).expand(1, 4, 2, 4))
        with self.assertRaises(ConfigurationError):
            rk.CoefficientSource(torch.zeros(6, 5, dtype=torch.uint8))
        with self.assertRaises(ConfigurationError):
            rk.CoefficientSource(torch.zeros(6, 5, dtype=torch.uint8), np.ones(4, np.float32), np.zeros(4, np.float32))
        with self.assertRaises(ConfigurationError):
            rk.CoefficientSource(torch.zeros(6, 5), bound=-1.0)
        with self.assertRaises(ConfigurationError):
            rk.DenseSource(torch.zeros(6, 8, dtype=torch.int64))
        table = rk.RopeTable.from_complex(wan_freqs(4, rows=2))
        with self.assertRaises(ConfigurationError):
            self.job(rope=rk.RopePositions(table, (2, 0, 0)))
        with self.assertRaises(ConfigurationError):
            rk.RopeTable(torch.zeros(3, 2), torch.zeros(3, 2), (1, 1, 1))

    def test_rejects_overlapping_destinations_and_bad_options(self):
        overlapping = [self.job(), self.job(destination=rk.NativeView(self.destination, token_offset=5))]
        with self.assertRaises(ConfigurationError):
            rk.reconstruct(overlapping)
        disjoint = [self.job(), self.job(destination=rk.NativeView(self.destination, token_offset=6))]
        rk.reconstruct(disjoint)
        with self.assertRaises(ConfigurationError):
            rk.reconstruct(disjoint, backend="nope")
        with self.assertRaises(ConfigurationError):
            rk.reconstruct(disjoint, precision="int8")
        with self.assertRaises(UnsupportedError):
            rk.reconstruct(disjoint, backend="fused")
        with self.assertRaises(UnsupportedError):
            rk.reconstruct(disjoint, backend="staged")
        with self.assertRaises(ConfigurationError):
            rk.reconstruct([object()])

    def test_native_rope_must_return_matching_size(self):
        table = rk.RopeTable.from_complex(wan_freqs(4, rows=4))
        job = self.job(rope=rk.RopePositions(table, (0, 0, 0)))
        with self.assertRaises(IntegrityError):
            rk.reconstruct([job], native_rope=lambda values, job: values[:1])


class BackendSelectionTests(unittest.TestCase):
    """No silent fallback: on CUDA, ``auto`` raises like ``fused`` when the compiled kernels are unavailable."""

    def resolve_all(self):
        return {(backend, cuda): rk._resolve_backend(backend, cuda) for backend in ("auto", "reference")
                for cuda in (True, False)}

    def test_resolution_with_compiled_kernels(self):
        with mock.patch.object(rk, "TRITON_AVAILABLE", True), mock.patch.object(rk, "_interpreted", lambda: False):
            self.assertEqual(self.resolve_all(), {
                ("auto", True): ("fused", False), ("auto", False): ("reference", False),
                ("reference", True): ("reference", False), ("reference", False): ("reference", False)})
            for backend in ("fused", "two_pass", "staged"):
                self.assertEqual(rk._resolve_backend(backend, True), (backend, False))
                with self.assertRaisesRegex(UnsupportedError, "needs CUDA tensors"):
                    rk._resolve_backend(backend, False)
            for cuda in (True, False):
                with self.assertRaises(UnsupportedError):
                    rk._resolve_backend("interpret", cuda)

    def test_auto_on_cuda_raises_when_triton_failed_to_import(self):
        error = "ImportError: simulated libtriton failure"
        with mock.patch.object(rk, "TRITON_AVAILABLE", False), mock.patch.object(rk, "TRITON_IMPORT_ERROR", error):
            for backend in ("auto", "fused", "two_pass", "staged"):
                with self.assertRaises(UnsupportedError) as caught:
                    rk._resolve_backend(backend, True)
                self.assertIn(error, str(caught.exception))
            with self.assertRaises(UnsupportedError) as caught:
                rk._resolve_backend("auto", True)
            self.assertIn("backend='reference'", str(caught.exception))
            with self.assertRaisesRegex(UnsupportedError, "simulated libtriton failure"):
                rk._resolve_backend("interpret", False)
            # Explicit reference (any device) and CPU tensors keep the plain-torch path.
            self.assertEqual(rk._resolve_backend("reference", True), ("reference", False))
            self.assertEqual(rk._resolve_backend("auto", False), ("reference", False))

    def test_auto_on_cuda_raises_in_interpreter_mode(self):
        with mock.patch.object(rk, "TRITON_AVAILABLE", True), mock.patch.object(rk, "_interpreted", lambda: True):
            for backend in ("auto", "fused", "two_pass", "staged"):
                with self.assertRaisesRegex(UnsupportedError, "TRITON_INTERPRET=1"):
                    rk._resolve_backend(backend, True)
            self.assertEqual(rk._resolve_backend("auto", False), ("reference", False))
            self.assertEqual(rk._resolve_backend("fused", False), ("fused", True))
            self.assertEqual(rk._resolve_backend("two_pass", False), ("two_pass", True))
            self.assertEqual(rk._resolve_backend("interpret", False), ("fused", True))

    def test_failed_triton_import_is_reported_not_substituted(self):
        """A real failed import (``sys.modules['triton'] = None``) in a fresh interpreter."""
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES="")
        environment.pop("TRITON_INTERPRET", None)
        completed = subprocess.run(
            [sys.executable, "-c", _NO_TRITON_SCRIPT, str(ROOT)], capture_output=True, text=True,
            env=environment, timeout=600, cwd=str(ROOT), check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr[-4000:])
        report = json.loads(completed.stdout.strip().splitlines()[-1])
        self.assertFalse(report["available"])
        self.assertIn("triton", report["import_error"])
        for backend in ("auto", "fused", "two_pass", "staged"):
            self.assertEqual(report["cuda"][backend]["error"], "UnsupportedError", backend)
            self.assertIn(report["import_error"], report["cuda"][backend]["message"], backend)
        self.assertEqual(report["cuda"]["reference"], {"resolved": ["reference", False]})
        self.assertEqual(report["cpu_auto"], {"backend": "reference", "exact": True})
        self.assertEqual(report["cpu_interpret"]["error"], "UnsupportedError")
        self.assertIn(report["import_error"], report["cpu_interpret"]["message"])


class StagedGemmTests(unittest.TestCase):
    """The staged cuBLAS GEMM writes FP32 or raises; it never retries with an FP16/BF16 output."""

    def setUp(self):
        self.coefficients = torch.randn(6, 4).half()
        self.basis = torch.randn(4, 8).half()

    def run_with(self, behaviour, precision="fp16"):
        calls = []

        def fake_mm(*args, **kwargs):
            calls.append(kwargs)
            return behaviour(*args, **kwargs)

        with mock.patch.object(torch, "mm", fake_mm):
            try:
                return rk._staged_gemm(self.coefficients, self.basis, precision), calls
            except Exception as error:  # noqa: BLE001 - inspected by the caller
                return error, calls

    def test_requests_fp32_output_once(self):
        for precision in ("fp16", "bf16"):
            result, calls = self.run_with(
                lambda a, b, out_dtype=None: (a.double() @ b.double()).to(out_dtype), precision)
            self.assertEqual(calls, [{"out_dtype": torch.float32}])
            self.assertEqual(result.dtype, torch.float32)

    def test_no_silent_retry(self):
        def raiser(error):
            def behaviour(*args, **kwargs):
                raise error
            return behaviour

        for error in (TypeError("mm() got an unexpected keyword argument 'out_dtype'"),
                      NotImplementedError("Could not run 'aten::mm.dtype'")):
            result, calls = self.run_with(raiser(error))
            self.assertIsInstance(result, UnsupportedError)
            self.assertIs(result.__cause__, error)
            self.assertIn("out_dtype=torch.float32", str(result))
            self.assertEqual(len(calls), 1)
        out_of_memory = RuntimeError("CUDA out of memory")
        result, calls = self.run_with(raiser(out_of_memory))
        self.assertIs(result, out_of_memory)
        self.assertEqual(len(calls), 1)

    def test_unsupported_out_dtype_is_not_replaced_by_a_half_precision_product(self):
        # Torch 2.8 / 2.13 CPU builds have no FP32-output FP16 GEMM (aten::mm.dtype): previously retried silently
        # as an FP16-output product.
        try:
            result = rk._staged_gemm(self.coefficients, self.basis, "fp16")
        except UnsupportedError as error:
            self.assertIsInstance(error.__cause__, NotImplementedError)
        else:
            self.assertEqual(result.dtype, torch.float32)


class RopeTableCacheTests(unittest.TestCase):
    def test_host_copies_cached_and_refreshed_after_in_place_update(self):
        table = rk.RopeTable.from_complex(wan_freqs(8, rows=6))
        self.assertEqual(table.cos.dtype, torch.float64)
        cos, sin = table.kernel_tables("cpu")
        self.assertTrue(cos.dtype == sin.dtype == torch.float32 and cos.is_contiguous() and sin.is_contiguous())
        self.assertTrue(torch.equal(cos, table.cos.float()) and torch.equal(sin, table.sin.float()))
        again = table.kernel_tables("cpu")
        self.assertTrue(again[0] is cos and again[1] is sin)
        table.cos.mul_(0.5)
        refreshed, _ = table.kernel_tables("cpu")
        self.assertTrue(torch.equal(refreshed, table.cos.float()))
        self.assertFalse(torch.equal(refreshed, cos))

    def test_consumers_wait_on_the_copy_event_until_it_completes(self):
        class Event:
            done = False

            def query(self):
                return self.done

        class Stream:
            def __init__(self):
                self.waited = []

            def wait_event(self, event):
                self.waited.append(event)

        event, staging = Event(), object()
        entry = rk._KernelTables(torch.zeros(2, 2), torch.zeros(2, 2), (0, 0), event, staging)
        first, second = Stream(), Stream()
        entry.order_before(torch.device("cuda", 0), first)
        entry.order_before(torch.device("cuda", 0), second)
        self.assertEqual((first.waited, second.waited), ([event], [event]))
        self.assertIs(entry.staging, staging)
        event.done = True
        third = Stream()
        entry.order_before(torch.device("cuda", 0), third)
        self.assertEqual(third.waited, [])
        self.assertIsNone(entry.event)
        self.assertIsNone(entry.staging)


def _mode(inside):
    """``torch.inference_mode()`` (SGLang runs requests under it) or no change."""
    return torch.inference_mode() if inside else contextlib.nullcontext()


def _read(probe):
    try:
        value = probe()
    except Exception as error:  # noqa: BLE001 - e.g. Torch >= 2.9 refuses legacy getters after new-API use
        return f"raises {type(error).__name__}"
    return value if isinstance(value, (bool, int, float, str, type(None))) else str(value)


def global_flags():
    """Read-only snapshot of process-wide torch settings that restore_kernels must never change."""
    backends, matmul, cudnn = torch.backends, torch.backends.cuda.matmul, torch.backends.cudnn
    mkldnn = getattr(backends, "mkldnn", None)
    probes = {
        "cuda.matmul.allow_tf32": lambda: matmul.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision,
        "fp32_precision": lambda: getattr(backends, "fp32_precision", None),
        "cuda.matmul.fp32_precision": lambda: getattr(matmul, "fp32_precision", None),
        "mkldnn.fp32_precision": lambda: getattr(mkldnn, "fp32_precision", None),
        "mkldnn.matmul.fp32_precision": lambda: getattr(getattr(mkldnn, "matmul", None), "fp32_precision", None),
        "cudnn.fp32_precision": lambda: getattr(cudnn, "fp32_precision", None),
        "cudnn.conv.fp32_precision": lambda: getattr(getattr(cudnn, "conv", None), "fp32_precision", None),
        "cuda.matmul.allow_fp16_reduced_precision_reduction": lambda: matmul.allow_fp16_reduced_precision_reduction,
        "cuda.matmul.allow_bf16_reduced_precision_reduction": lambda: matmul.allow_bf16_reduced_precision_reduction,
        "cuda.matmul.allow_fp16_accumulation": lambda: getattr(matmul, "allow_fp16_accumulation", None),
        "cudnn.allow_tf32": lambda: cudnn.allow_tf32, "cudnn.benchmark": lambda: cudnn.benchmark,
        "cudnn.deterministic": lambda: cudnn.deterministic, "cudnn.enabled": lambda: cudnn.enabled,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled,
        "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled,
        "default_dtype": torch.get_default_dtype, "grad_enabled": torch.is_grad_enabled,
        "inference_mode": torch.is_inference_mode_enabled, "num_threads": torch.get_num_threads,
        "num_interop_threads": torch.get_num_interop_threads,
        "environment": lambda: hashlib.sha256(json.dumps(sorted(os.environ.items())).encode()).hexdigest(),
    }
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        probes.update({
            "cuda.current_device": torch.cuda.current_device,
            "cuda.current_stream": lambda: torch.cuda.current_stream().cuda_stream,
            "cuda.preferred_blas_library": torch.backends.cuda.preferred_blas_library,
            "cuda.preferred_linalg_library": torch.backends.cuda.preferred_linalg_library,
        })
    return {name: _read(probe) for name, probe in probes.items()}


@contextlib.contextmanager
def process_fp32_precision(mode):
    """The caller (a test) sets the process's FP32 matmul precision ("ieee" or "tf32") and restores it exactly.

    A no-op when the process already runs FP32 matmuls at ``mode`` (Torch >= 2.9 then keeps accepting the legacy
    getters, which raise once the new fp32_precision API has set a value).
    """
    matmul = torch.backends.cuda.matmul
    holders = [holder for holder in (matmul, getattr(getattr(torch.backends, "mkldnn", None), "matmul", None))
               if isinstance(getattr(holder, "fp32_precision", None), str)]
    if holders:  # Torch >= 2.9 API; "none" = the generic setting, whose "none" default is IEEE
        saved = [holder.fp32_precision for holder in holders]
        generic = getattr(torch.backends, "fp32_precision", "none")
        if all((generic if value == "none" else value).replace("none", "ieee") == mode for value in saved):
            yield
            return
        try:
            for holder in holders:
                holder.fp32_precision = mode
            yield
        finally:
            for holder, value in zip(holders, saved):
                holder.fp32_precision = value
    else:  # Torch <= 2.8: one legacy setting for CUDA (allow_tf32) and oneDNN
        saved = torch.get_float32_matmul_precision()
        wanted = "high" if mode == "tf32" else "highest"
        if saved == wanted:
            yield
            return
        try:
            torch.set_float32_matmul_precision(wanted)
            yield
        finally:
            torch.set_float32_matmul_precision(saved)


_SCOPED_FLAGS = frozenset({"grad_enabled", "inference_mode", "cuda.current_device", "cuda.current_stream"})


def process_wide_flags():
    """global_flags() without the thread-local states that ``with`` blocks enter and leave (grad/inference mode,
    current CUDA device and stream); the module may scope those, never the process-wide settings."""
    return {name: value for name, value in global_flags().items() if name not in _SCOPED_FLAGS}


def flag_diff(before, after):
    return {name: [before.get(name), after.get(name)] for name in sorted(set(before) | set(after))
            if before.get(name) != after.get(name)}


class FlagWatch(TorchFunctionMode):
    """Checks the process-wide settings against their values at entry at every torch call made inside the block.

    A before/after comparison cannot see a toggle that is undone before the call returns (the pre-fix staged backend
    set allow_tf32 around its GEMM and restored it in a ``finally``); meanwhile model work in other threads runs with
    the toggled setting. Torch calls made while such a toggle is active are recorded in ``deviations``.
    """

    def __init__(self):
        super().__init__()
        self.baseline, self.calls, self.deviations = None, 0, []

    def __enter__(self):
        self.baseline = process_wide_flags()
        return super().__enter__()

    def __torch_function__(self, func, types, args=(), kwargs=None):
        self.calls += 1
        current = process_wide_flags()
        if current != self.baseline and len(self.deviations) < 8:
            self.deviations.append({"op": getattr(func, "__qualname__", repr(func)),
                                    "changed": flag_diff(self.baseline, current)})
        return func(*args, **(kwargs or {}))


_MODES = tuple(itertools.product((False, True), repeat=2))  # (built inside inference_mode, used inside it)


class InferenceModeTests(unittest.TestCase):
    """Tables, jobs and calls built or run inside ``torch.inference_mode`` (inference tensors), outside, or mixed."""

    def test_kernel_tables_cache_across_modes(self):
        for built_inside, used_inside in _MODES:
            with self.subTest(built_inside=built_inside, used_inside=used_inside):
                with _mode(built_inside):
                    table = rk.RopeTable.from_complex(wan_freqs(12))
                self.assertEqual(table.cos.is_inference(), built_inside)
                with _mode(used_inside):
                    first = table.kernel_tables("cpu")
                    second = table.kernel_tables("cpu")
                with _mode(not used_inside):
                    third = table.kernel_tables("cpu")
                self.assertTrue(first[0] is second[0] is third[0] and first[1] is second[1] is third[1])
                self.assertFalse(first[0].is_inference() or first[1].is_inference())  # persistent copies
                self.assertTrue(torch.equal(first[0], table.cos.float()) and torch.equal(first[1], table.sin.float()))

    def test_inference_tables_are_immutable_until_invalidated(self):
        with torch.inference_mode():
            table = rk.RopeTable.from_complex(wan_freqs(12))
        cos, _ = table.kernel_tables("cpu")
        with torch.inference_mode():
            table.cos.mul_(0.5)  # untracked: inference tensors have no version counter
        self.assertIs(table.kernel_tables("cpu")[0], cos)
        table.invalidate()
        refreshed, _ = table.kernel_tables("cpu")
        self.assertTrue(torch.equal(refreshed, table.cos.float()) and not torch.equal(refreshed, cos))
        normal = rk.RopeTable.from_complex(wan_freqs(12))
        cos, _ = normal.kernel_tables("cpu")
        with torch.inference_mode():
            normal.cos.mul_(0.5)  # normal tensors keep their version counter inside inference mode
        refreshed, _ = normal.kernel_tables("cpu")
        self.assertTrue(torch.equal(refreshed, normal.cos.float()) and not torch.equal(refreshed, cos))

    def test_reference_reconstruction_inside_outside_and_mixed(self):
        grid, heads, head_dim = (2, 3, 4), 2, 12
        tokens = 24

        def run(built_inside, used_inside, destination_inside):
            with _mode(built_inside):
                table = rk.RopeTable.from_complex(wan_freqs(head_dim))
                basis = torch.from_numpy(np.random.default_rng(4).standard_normal((10, heads * head_dim))).half()
            with _mode(destination_inside):
                destination = torch.zeros(1, tokens, heads, head_dim)
            with _mode(built_inside):
                job, _ = make_picture_job(np.random.default_rng(5), rk.NativeView(destination), grid, basis, 7,
                                          rope=rk.RopePositions(table, (2, 0, 0)))
            with _mode(used_inside):
                handle = rk.reconstruct([job])
            self.assertEqual(handle.stats["backend"], "reference")
            return destination

        expected = run(False, False, False)
        for built_inside, used_inside in _MODES:
            for destination_inside in (False, True):
                with self.subTest(built_inside=built_inside, used_inside=used_inside,
                                  destination_inside=destination_inside):
                    if destination_inside and not used_inside:
                        # The reference backend writes with PyTorch in-place copies: an inference-mode cache
                        # must be written inside torch.inference_mode (the Triton backends write through pointers).
                        with self.assertRaisesRegex(RuntimeError, "Inplace update to inference tensor"):
                            run(built_inside, used_inside, destination_inside)
                        continue
                    self.assertTrue(torch.equal(run(built_inside, used_inside, destination_inside), expected))


class GlobalFlagsTests(unittest.TestCase):
    """restore_kernels never changes process-wide torch settings (TF32 / matmul precision etc.)."""

    def setUp(self):
        self.grid, self.heads, self.head_dim = (2, 3, 4), 2, 12
        self.tokens = 24
        self.basis = torch.from_numpy(np.random.default_rng(8).standard_normal((10, 24))).half()
        self.table = rk.RopeTable.from_complex(wan_freqs(self.head_dim))

    def jobs(self):
        rng = np.random.default_rng(6)
        destination = torch.zeros(1, 4 * self.tokens, self.heads, self.head_dim)
        picture, pixels = make_picture_job(rng, rk.NativeView(destination), self.grid, self.basis, 7,
                                           rope=rk.RopePositions(self.table, (1, 0, 0)))
        quantized = rk.ReconstructionJob(
            rk.CoefficientSource(pixels, picture.source.scale, picture.source.offset),
            rk.NativeView(destination, token_offset=self.tokens), self.grid, self.basis, 7)
        floats = rk.ReconstructionJob(rk.CoefficientSource(rk.reference_coefficients(picture)),
                                      rk.NativeView(destination, token_offset=2 * self.tokens), self.grid, self.basis,
                                      7, rk.RopePositions(self.table, (3, 0, 0)))
        dense = rk.ReconstructionJob(rk.DenseSource(torch.randn(self.tokens, 24)),
                                     rk.NativeView(destination, token_offset=3 * self.tokens), self.grid,
                                     rope=rk.RopePositions(self.table, (0, 0, 0)))
        return [picture, quantized, floats, dense]

    def attempts(self):
        """Every CPU backend and torch-GEMM path; each returns normally or raises UnsupportedError."""
        coefficients, basis = torch.randn(6, 4), torch.randn(4, 8).half()
        calls = {f"reconstruct/{backend}/{dtype}": (lambda backend=backend, dtype=dtype: rk.reconstruct(
                     self.jobs(), backend=backend, compute_dtype=dtype))
                 for backend in ("auto", "reference", "fused", "two_pass", "staged", "interpret")
                 for dtype in (torch.float64, torch.float32)}
        calls["reconstruct_reference/fp32"] = lambda: rk.reconstruct_reference(self.jobs(), compute_dtype=torch.float32)
        calls["kernel_tables"] = lambda: rk.RopeTable.from_complex(wan_freqs(self.head_dim)).kernel_tables("cpu")
        operands = {"fp32": coefficients, "tf32": coefficients, "fp16": coefficients.half(), "bf16": coefficients.bfloat16()}
        for precision, operand in operands.items():
            calls[f"staged_gemm/{precision}"] = (lambda precision=precision, operand=operand: rk._staged_gemm(
                operand, basis, precision))
        return calls

    def run_all(self):
        outcomes = {}
        for name, call in self.attempts().items():
            before = global_flags()
            with FlagWatch() as watch:
                try:
                    call()
                    outcomes[name] = "ok"
                except UnsupportedError as error:
                    outcomes[name] = f"UnsupportedError: {error}"
            self.assertEqual(global_flags(), before, name)
            self.assertEqual(watch.deviations, [], name)  # not even temporarily (during the call)
            if outcomes[name] == "ok":
                self.assertGreater(watch.calls, 0, name)
        return outcomes

    def test_backends_never_change_global_flags(self):
        start = global_flags()
        outcomes = self.run_all()
        self.assertEqual(global_flags(), start)
        self.assertEqual(outcomes["reconstruct/auto/torch.float64"], "ok")
        self.assertEqual(outcomes["reconstruct/reference/torch.float32"], "ok")  # IEEE FP32 (default setting)
        self.assertEqual(outcomes["staged_gemm/fp32"], "ok")
        self.assertIn("staged tf32 GEMM", outcomes["staged_gemm/tf32"])
        self.assertIn("never changes it", outcomes["staged_gemm/tf32"])

    def test_torch_gemm_paths_follow_but_never_set_the_process_precision(self):
        start = global_flags()
        with process_fp32_precision("tf32"):
            self.assertEqual(rk.fp32_matmul_precision("cuda"), "tf32" if hasattr(
                torch.backends.cuda.matmul, "fp32_precision") else "high")
            self.assertNotEqual(rk.fp32_matmul_precision("cpu"), "ieee")
            outcomes = self.run_all()  # also asserts unchanged flags around every call
        self.assertEqual(global_flags(), start)
        self.assertEqual(rk.fp32_matmul_precision("cuda"), "ieee")
        self.assertEqual(outcomes["staged_gemm/tf32"], "ok")
        self.assertIn("staged fp32 GEMM", outcomes["staged_gemm/fp32"])
        for name in ("reconstruct/reference/torch.float32", "reconstruct/auto/torch.float32",
                     "reconstruct_reference/fp32"):
            self.assertIn("FP32 reference", outcomes[name])
        self.assertEqual(outcomes["reconstruct/reference/torch.float64"], "ok")

    def test_source_never_writes_process_wide_settings(self):
        """Static: no assignment to torch / os settings, no torch setter calls (also on paths no test runs)."""
        self.assertEqual(process_wide_writes(Path(rk.__file__).read_text()), [])
        toggles = ["torch.backends.cuda.matmul.allow_tf32 = True", "m = torch.backends.cuda.matmul\nm.allow_tf32 = x",
                   "torch.set_float32_matmul_precision('high')", "torch.backends.cuda.matmul.fp32_precision = 'tf32'",
                   "with torch.backends.cudnn.flags(enabled=False):\n    pass", "os.environ['A'] = '1'",
                   "setattr(torch.backends.cudnn, 'benchmark', True)", "setattr(m, 'allow_tf32', True)",
                   "torch.use_deterministic_algorithms(True)", "a, torch.backends.cudnn.benchmark = 1, True"]
        for snippet in toggles:
            self.assertEqual(len(process_wide_writes(snippet)), 1, snippet)


_FLAG_ATTRIBUTES = frozenset({
    "allow_tf32", "fp32_precision", "allow_fp16_reduced_precision_reduction",
    "allow_bf16_reduced_precision_reduction", "allow_fp16_accumulation", "benchmark", "benchmark_limit",
    "deterministic", "enabled", "verbose", "matmul", "cudnn", "mkldnn", "environ"})


def _dotted(node):
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def process_wide_writes(source):
    """Statements of Python ``source`` that write process-wide torch / environment settings (``line: source``)."""
    found = []

    def writes(target):
        if isinstance(target, (ast.Tuple, ast.List)):
            return any(writes(element) for element in target.elts)
        if isinstance(target, ast.Subscript):
            return (_dotted(target.value) or "") == "os.environ"
        name = _dotted(target) if isinstance(target, ast.Attribute) else None
        return name is not None and (name.split(".")[0] in ("torch", "os") or target.attr in _FLAG_ATTRIBUTES)

    for node in ast.walk(ast.parse(source)):
        hit = False
        if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign, ast.Delete)):
            targets = node.targets if isinstance(node, (ast.Assign, ast.Delete)) else [node.target]
            hit = any(writes(target) for target in targets)
        elif isinstance(node, ast.Call):
            name = _dotted(node.func) or ""
            first = node.args[0] if node.args else None
            hit = (name.startswith(("torch.set_", "torch.backends.", "torch._C._set", "torch.cuda.set_",
                                    "torch.cuda.memory.set_", "os.environ.", "os.putenv", "os.unsetenv"))
                   or name in ("torch.use_deterministic_algorithms", "torch.manual_seed", "torch.seed")
                   or (name == "setattr" and first is not None and (
                       (_dotted(first) or "").split(".")[0] in ("torch", "os")
                       or (len(node.args) > 1 and isinstance(node.args[1], ast.Constant)
                           and node.args[1].value in _FLAG_ATTRIBUTES))))
        if hit:
            found.append(f"{node.lineno}: {ast.get_source_segment(source, node)}")
    return found


class WarmupTests(unittest.TestCase):
    """warmup(): shape validation, the call-dependent specializations it enumerates, CPU behaviour."""

    def test_shapes_are_validated_and_hashable(self):
        shape = rk.WarmupShape(12, 128, ranks=[1, 269])
        self.assertEqual(shape.ranks, (1, 269))
        self.assertEqual(hash(shape), hash(rk.WarmupShape(12, 128, ranks=(1, 269))))
        self.assertEqual(shape.describe()["dtype"], "torch.bfloat16")
        bad = [{"heads": 0, "head_dim": 8}, {"heads": 2, "head_dim": 7}, {"heads": 2, "head_dim": 8, "dtype": torch.int8},
               {"heads": 2, "head_dim": 8, "layout": "TBHD"}, {"heads": 2, "head_dim": 8, "source": "rgb"},
               {"heads": 2, "head_dim": 8, "rope": "post"}, {"heads": 2, "head_dim": 8, "ranks": (0, 4)},
               {"heads": 2, "head_dim": 8, "ranks": (5, 4)}, {"heads": 2, "head_dim": 8, "ranks": (1, 2, 3)},
               {"heads": 2, "head_dim": 8, "ranks": (1, 70000)}, {"heads": 2, "head_dim": 8, "source_dtype": torch.float32},
               {"heads": 2, "head_dim": 8, "source": "float", "source_dtype": torch.uint8},
               {"heads": 2, "head_dim": 8, "native_rope_dtype": torch.float32},
               {"heads": 2, "head_dim": 8, "rope": "native", "native_rope_dtype": torch.float64}]
        for arguments in bad:
            with self.assertRaises(ConfigurationError, msg=arguments):
                rk.WarmupShape(**arguments)
        rk.WarmupShape(2, 7, rope="none")  # odd head_dim without RoPE

    def test_rank_recipes_cover_every_reachable_launch_group(self):
        """Brute force: the (k_tiles classes, row alignment) of every group of ranks in the range is warmed."""
        cases = (((1, 140), (64,), None), ((1, 300), (32,), 4), ((60, 70), (32, 64), 1), ((950, 1030), (64,), None),
                 ((5, 40), (), 2), ((470, 520), (32,), 2), ((7, 7), (64,), 1))
        for ranks, block_ks, itemsize in cases:
            recipes = rk._rank_recipes(ranks, list(block_ks), itemsize)

            def key(group, block_ks=block_ks, itemsize=itemsize):
                classes = tuple(rk._k_class(-(-max(group) // block)) for block in block_ks)
                return classes, all(itemsize is None or rank * itemsize % 16 == 0 for rank in group)

            low, high = ranks
            reachable = {key((rank, other)) for rank in range(low, high + 1) for other in range(low, rank + 1)}
            self.assertEqual({key(recipe) for recipe in recipes}, reachable, ranks)
            self.assertEqual(len(recipes), len(reachable))  # one synthetic call per specialization
            self.assertTrue(all(low <= rank <= high for recipe in recipes for rank in recipe))
        self.assertIn((961,), rk._rank_recipes((1, 1024), [64], None))  # k_tiles = 16: Triton's "divisible by 16"
        self.assertEqual(rk._rank_recipes((65, 269), [64], None), [(65,)])  # no group can have k_tiles == 1

    def test_plans_follow_the_launch_path(self):
        shape = rk.WarmupShape(12, 128, ranks=(1, 269))
        legacy = {**rk._DEFAULT_CONFIG, "BLOCK_M": 64, "N_GROUP": 4}
        current = {**rk._DEFAULT_CONFIG, "BLOCK_M": 32, "N_GROUP": 1}
        with mock.patch.object(rk, "_LEGACY_TRITON", True):
            # Triton <= 3.4: pictures on the fused kernel, whose N_GROUP (1..4) depends on the call's tiles.
            self.assertEqual(rk._warmup_plan(shape, "auto", "fp16x2", legacy, True), ("fused", [(1,)], [1, 2, 3, 4]))
            self.assertEqual(rk._warmup_plan(shape, "auto", "fp16x2", legacy, False), ("fused", [(1,)], [None]))
        with mock.patch.object(rk, "_LEGACY_TRITON", False):
            # Triton >= 3.5: two-pass pictures; the unpack's k_tiles = ceil(max rank / 64) is 1 or not.
            self.assertEqual(rk._warmup_plan(shape, "auto", "fp16x2", current, True), ("two_pass", [(1,), (65,)], [1]))
        dense = rk.WarmupShape(12, 128, source="dense", rope="native")
        self.assertEqual(rk._warmup_plan(dense, "auto", "fp16x2", current, True), ("rope_store", [()], [None]))
        small = rk.WarmupShape(1, 64, rope="none", ranks=(1, 40))  # one channel tile: N_GROUP is always 1
        self.assertEqual(rk._warmup_plan(small, "fused", "fp16x2", legacy, True)[2], [1])
        floats = rk.WarmupShape(12, 128, source="float", source_dtype=torch.bfloat16, ranks=(1, 64))
        path, recipes, _ = rk._warmup_plan(floats, "staged", "fp16", current, True)
        self.assertEqual((path, recipes), ("staged", [(1,), (8,), (33,), (40,)]))  # unpack/bound k_tiles x alignment
        self.assertEqual(rk._warmup_plan(floats, "staged", "fp32", current, True)[1], [(1,), (8,), (33,), (40,)])
        self.assertEqual(rk._warmup_plan(floats, "fused", "fp32", current, True)[1], [(1,), (8,)])  # no bound kernel

    def test_cpu_auto_is_the_reference_backend_and_compiles_nothing(self):
        before = global_flags()
        with FlagWatch() as watch:
            report = rk.warmup("cpu", {"heads": 2, "head_dim": 8})
        self.assertEqual(global_flags(), before)
        self.assertEqual(watch.deviations, [])
        self.assertEqual((report["resolved_backend"], report["calls"], report["launches"]), ("reference", 0, 0))
        self.assertIn("compiles no kernels", report["note"])
        with self.assertRaisesRegex(UnsupportedError, "needs CUDA tensors"):
            rk.warmup("cpu", rk.WarmupShape(2, 8), backend="fused")
        with self.assertRaises(ConfigurationError):
            rk.warmup("cpu", [], backend="reference")
        with self.assertRaises(ConfigurationError):
            rk.warmup("cpu", rk.WarmupShape(2, 8), backend="fastest")
        with self.assertRaises(ConfigurationError):
            rk._warmup_precisions(["fp16x2", "int8"], torch.device("cpu"))
        self.assertEqual(rk._warmup_precisions(["auto", "fp16x2", "fp32"], torch.device("cpu")), ["fp16x2", "fp32"])

    def test_cuda_without_compiled_kernels_raises_like_reconstruct(self):
        simulated = "ImportError: simulated (warmup test)"
        with (mock.patch.object(rk, "TRITON_AVAILABLE", False), mock.patch.object(rk, "TRITON_IMPORT_ERROR", simulated),
              self.assertRaisesRegex(UnsupportedError, "simulated")):
            rk.warmup("cuda", rk.WarmupShape(2, 8))
        with mock.patch.object(rk, "_interpreted", lambda: True), self.assertRaisesRegex(UnsupportedError,
                                                                                       "TRITON_INTERPRET"):
            rk.warmup("cuda", rk.WarmupShape(2, 8))


class UnpackDequantTests(unittest.TestCase):
    def test_unpack_matches_codec_cells_and_nvdec_unpack(self):
        rng = np.random.default_rng(3)
        layouts = [
            _hevc.video_layout(3, 5, 7, 11, "yuv444p"), _hevc.video_layout(2, 4, 6, 9, "yuv420p"),
            custom_layout(2, 3, 4, 10, 2, 1, 3), custom_layout(1, 2, 5, 7, 1, 1, 1),
        ]
        for layout in layouts:
            tokens = layout["frames"] * layout["height"] * layout["width"]
            pixels = torch.from_numpy(rng.integers(0, 256, size=(tokens, layout["rank"]), dtype=np.uint8))
            canvas, indices = pack_with_cells(pixels, layout, origin=(1, 2))
            scale, offset = random_affine(layout["rank"], rng)
            source = rk.PictureSource.from_canvases(canvas, layout, indices, (1, 2), scale, offset)
            self.assertTrue(torch.equal(rk.unpack_pictures(source), pixels))
            if "image_height" in layout:
                slots = layout["channels_per_segment"]
                frames = layout["frames"]
                for segment in range(layout["segments"]):
                    count = min(slots, layout["rank"] - segment * slots)
                    for frame in range(frames):
                        image = canvas[segment * frames + frame, :, 1:, 2:]
                        unpacked = unpack_device_frame(image, layout, count)
                        expected = pixels.reshape(frames, layout["height"], layout["width"], -1)[
                            frame, :, :, segment * slots:segment * slots + count]
                        self.assertTrue(torch.equal(unpacked, expected))

    def test_per_picture_origins(self):
        rng = np.random.default_rng(4)
        layout = custom_layout(2, 3, 4, 7, 1, 1, 3)
        tokens = 24
        pixels = torch.from_numpy(rng.integers(0, 256, size=(tokens, 7), dtype=np.uint8))
        canvas, indices = pack_with_cells(pixels, layout, origin=(0, 0), extra=(4, 9))
        shifted = torch.zeros_like(canvas)
        origins = []
        for segment, row in enumerate(indices):
            origin_row = []
            for frame, picture in enumerate(row):
                y, x = (segment + frame) % 3, (2 * segment + frame) % 5
                shifted[picture, :, y:, x:] = canvas[picture, :, :canvas.shape[2] - y, :canvas.shape[3] - x]
                origin_row.append((y, x))
            origins.append(origin_row)
        scale, offset = random_affine(7, rng)
        source = rk.PictureSource.from_canvases(shifted, layout, indices, origins, scale, offset)
        self.assertTrue(torch.equal(rk.unpack_pictures(source), pixels))

    def test_dequantize_matches_reelcache_codecs(self):
        rng = np.random.default_rng(5)
        for mode, fraction in (("per_channel", 1.0), ("global", 0.0), ("leading_columns", 0.25),
                               ("leading_columns", 1.0)):
            coefficients = (rng.standard_normal((40, 12)) * np.linspace(5, 0.1, 12)).astype(np.float32)
            pixels, affine, layout = _hevc.quantize_coefficients(coefficients, mode, fraction)
            scale, offset = rk.expand_affine(affine, layout, 12)
            expected_scale, expected_offset = _hevc._expanded_affine(affine, layout, 12)
            np.testing.assert_array_equal(scale, expected_scale)
            np.testing.assert_array_equal(offset, expected_offset)
            torch_pixels = torch.from_numpy(pixels)
            ours = rk.dequantize(torch_pixels, scale, offset)
            reelcache = dequantize_device_coefficients(torch_pixels, affine, layout)
            numpy_reference = _hevc.dequantize_coefficients(pixels, affine, layout)
            self.assertTrue(torch.equal(ours, reelcache))
            np.testing.assert_array_equal(ours.numpy(), numpy_reference)


class RopeParityTests(unittest.TestCase):
    """Rotation of the same reconstructed pre-RoPE K by the module vs replicas of native code."""

    def setUp(self):
        torch.manual_seed(7)
        self.heads, self.head_dim = 3, 32
        self.grid = (3, 4, 5)
        self.tokens = 60

    def module_rotate(self, values, positions, out_dtype=torch.bfloat16, layout="BTHD"):
        if layout == "BTHD":
            destination = torch.zeros(1, self.tokens, self.heads, self.head_dim, dtype=out_dtype)
        else:
            destination = torch.zeros(self.tokens, self.heads, self.head_dim, dtype=out_dtype)
        job = rk.ReconstructionJob(rk.DenseSource(values.reshape(self.tokens, -1)), rk.NativeView(destination, layout),
                                   self.grid, rope=positions)
        rk.reconstruct([job], backend="reference")
        return job.destination.slot(self.tokens)

    def assert_rounding_parity(self, ours, native, *, max_ulps=1, min_exact=0.999):
        ours, native = ours.double(), native.double()
        exact = (ours == native).double().mean().item()
        spacing = torch.pow(2.0, torch.floor(torch.log2(native.abs().clamp_min(1e-30))) - 7)
        ulps = ((ours - native).abs() / spacing).max().item()
        self.assertLessEqual(ulps, max_ulps)
        self.assertGreaterEqual(exact, min_exact)

    def test_causvid_and_longlive1_causal_rope_apply(self):
        freqs = wan_freqs(self.head_dim)
        table = rk.RopeTable.from_complex(freqs)
        values = torch.randn(1, self.tokens, self.heads, self.head_dim) * 3
        for start in (0, 9, 30):
            native = causvid_causal_rope_apply(values, torch.tensor([list(self.grid)]), freqs, start_frame=start)
            ours = self.module_rotate(values, rk.RopePositions(table, (start, 0, 0)), out_dtype=torch.float32)
            torch.testing.assert_close(ours, native[0], rtol=0, atol=0)
            native_bf16 = native.bfloat16()[0]
            ours_bf16 = self.module_rotate(values, rk.RopePositions(table, (start, 0, 0)))
            self.assertTrue(torch.equal(ours_bf16, native_bf16))

    def test_raven_packed_float32_keys(self):
        freqs = wan_freqs(self.head_dim)
        table = rk.RopeTable.from_complex(freqs)
        values = torch.randn(self.tokens, self.heads, self.head_dim).bfloat16()
        for shift in (0, 4, 17):
            native = raven_apply_latent_pos_embed(values, [self.grid], freqs, [shift], packed=True)
            self.assertEqual(native.dtype, torch.float32)
            ours = self.module_rotate(values.float(), rk.RopePositions(table, (shift, 0, 0)), torch.float32, "THD")
            torch.testing.assert_close(ours, native, rtol=0, atol=0)

    def test_longlive2_multishot_offsets(self):
        freqs = wan_freqs(self.head_dim)
        values = torch.randn(1, self.tokens, self.heads, self.head_dim) * 2
        sections = rk.wan_sections(self.head_dim // 2)
        t_freqs, h_freqs, w_freqs = freqs.split(list(sections), dim=1)
        for start, scale, offset in ((6, 1.0, 0.0), (6, 1.0, 2.5), (3, 0.5, torch.tensor([0.0, 1.0, 1.0]))):
            native = longlive2_causal_rope_apply(values, [self.grid], freqs, start_frame=start, t_scale=scale,
                                                 temporal_offset=offset)
            temporal = longlive2_compute_temporal_freqs(t_freqs, self.grid[0], start, scale, values.device,
                                                        temporal_offset=offset)
            table = rk.RopeTable.from_axes(temporal, h_freqs[:self.grid[1]], w_freqs[:self.grid[2]])
            ours = self.module_rotate(values, rk.RopePositions(table, (0, 0, 0)), torch.float32)
            torch.testing.assert_close(ours, native[0], rtol=0, atol=0)

    def test_longcat_rope_3d(self):
        grid = self.grid
        table = rk.RopeTable.longcat(self.head_dim, *grid)
        key = torch.randn(1, self.heads, self.tokens, self.head_dim).bfloat16()
        native = longcat_rope_3d_key(key, grid)
        ours = self.module_rotate(key[0].transpose(0, 1).float(), rk.RopePositions(table, (0, 0, 0)))
        self.assert_rounding_parity(ours, native[0].transpose(0, 1))
        cos = torch.from_numpy(np.asarray(longcat_precompute_freqs_cis_3d(self.head_dim, grid)))[:, 0::2]
        cos_ours, _ = rk.rope_cos_sin(rk.RopePositions(table, (0, 0, 0)), grid, dtype=torch.float32)
        torch.testing.assert_close(cos_ours, cos.cos(), rtol=0, atol=0)

    def test_sglang_causal_wan_tables_and_rotation(self):
        values = torch.randn(self.tokens, self.heads, self.head_dim) * 2
        freqs = wan_freqs(self.head_dim)
        wan_table = rk.RopeTable.from_complex(freqs)
        for start in (0, 6):
            cos, sin = sglang_rotary_cos_sin(self.grid, start, self.head_dim)
            dense = rk.RopeTable.from_cos_sin(cos, sin)
            native = sglang_apply_rotary_embedding_native(values, cos, sin)
            ours = self.module_rotate(values, rk.RopePositions(dense, 0), torch.float32, "THD")
            torch.testing.assert_close(ours, native, rtol=2e-6, atol=2e-6)
            separable = self.module_rotate(values, rk.RopePositions(wan_table, (start, 0, 0)), torch.bfloat16, "THD")
            self.assert_rounding_parity(separable, native.bfloat16())


class ExactNativePathTests(unittest.TestCase):
    def test_raw_identity_through_native_callable(self):
        torch.manual_seed(11)
        heads, head_dim, grid = 2, 16, (2, 3, 4)
        tokens = 24
        freqs = wan_freqs(head_dim)
        table = rk.RopeTable.from_complex(freqs)
        raw = (torch.randn(tokens, heads * head_dim) * 4).bfloat16()
        destination = torch.zeros(1, 2 * tokens, heads, head_dim, dtype=torch.bfloat16)
        calls = []

        def native(values, job):
            calls.append((values.dtype, tuple(values.shape), job.label))
            start = job.rope.rows[0]
            return causvid_causal_rope_apply(values[None], torch.tensor([list(job.grid)]), freqs, start)[0]

        jobs = [rk.ReconstructionJob(rk.DenseSource(raw), rk.NativeView(destination, token_offset=tokens), grid,
                                     rope=rk.RopePositions(table, (5, 0, 0)), label="raw")]
        rk.reconstruct(jobs, native_rope=native)
        expected = causvid_causal_rope_apply(raw.reshape(1, tokens, heads, head_dim), torch.tensor([list(grid)]),
                                             freqs, 5)
        self.assertTrue(torch.equal(destination[:, tokens:], expected))
        self.assertTrue(torch.all(destination[:, :tokens] == 0))
        self.assertEqual(calls, [(torch.bfloat16, (tokens, heads, head_dim), "raw")])

    def test_factorized_source_through_native_callable(self):
        torch.manual_seed(12)
        heads, head_dim, grid = 2, 16, (1, 4, 6)
        tokens = 24
        freqs = wan_freqs(head_dim)
        basis = torch.randn(9, heads * head_dim).half()
        coefficients = torch.randn(tokens, 9)
        destination = torch.zeros(tokens, heads, head_dim, dtype=torch.bfloat16)
        job = rk.ReconstructionJob(rk.CoefficientSource(coefficients), rk.NativeView(destination, "THD"), grid, basis,
                                   9, rk.RopePositions(rk.RopeTable.from_complex(freqs), (2, 0, 0)))
        rk.reconstruct([job], native_rope=lambda values, job: causvid_causal_rope_apply(
            values[None], torch.tensor([list(job.grid)]), freqs, 2)[0], native_rope_dtype=torch.float32)
        pre = (coefficients.double() @ basis.double()).float().reshape(1, tokens, heads, head_dim)
        expected = causvid_causal_rope_apply(pre, torch.tensor([list(grid)]), freqs, 2)[0].bfloat16()
        self.assertTrue(torch.equal(destination, expected))


_NO_TRITON_SCRIPT = r"""
import json, sys
sys.modules["triton"] = None  # every `import triton` fails, like a missing or broken install
sys.path.insert(0, sys.argv[1])
import torch
from reelcache import restore_kernels as rk
from reelcache.errors import UnsupportedError
report = {"available": rk.TRITON_AVAILABLE, "import_error": rk.TRITON_IMPORT_ERROR, "cuda": {}}
for backend in ("auto", "fused", "two_pass", "staged", "reference"):
    try:
        report["cuda"][backend] = {"resolved": list(rk._resolve_backend(backend, True))}
    except UnsupportedError as error:
        report["cuda"][backend] = {"error": type(error).__name__, "message": str(error)}
torch.manual_seed(0)
basis = torch.randn(5, 8).half()
values = torch.randn(6, 5)
dest = torch.zeros(1, 6, 2, 4)
job = rk.ReconstructionJob(rk.CoefficientSource(values), rk.NativeView(dest), (1, 2, 3), basis, 5)
handle = rk.reconstruct([job])
expected = (values.double() @ basis.double()).float().reshape(1, 6, 2, 4)
report["cpu_auto"] = {"backend": handle.stats["backend"], "exact": bool(torch.equal(dest, expected))}
try:
    rk.reconstruct([job], backend="interpret")
    report["cpu_interpret"] = {"error": None}
except UnsupportedError as error:
    report["cpu_interpret"] = {"error": type(error).__name__, "message": str(error)}
print(json.dumps(report))
"""


_INTERPRETER_SCRIPT = r"""
import json, os, sys
os.environ["TRITON_INTERPRET"] = "1"
sys.path.insert(0, sys.argv[1])
import numpy as np, torch
from tests.test_restore_kernels import (make_picture_job, custom_layout, wan_freqs, random_affine,
                                         pack_with_cells, global_flags, process_fp32_precision, FlagWatch)
from reelcache import restore_kernels as rk
report = {"triton": rk.TRITON_AVAILABLE, "import_error": rk.TRITON_IMPORT_ERROR, "cases": []}
if not rk.TRITON_AVAILABLE:
    print(json.dumps(report)); raise SystemExit(0)
import triton
report["triton_version"] = triton.__version__
report["interpreted"] = rk._interpreted()
flags_start = global_flags()
flag_changes = []
def checked_reconstruct(*args, **kwargs):
    # Every backend call must leave the process-wide torch settings (TF32 etc.) as it found them, also while it runs.
    before = global_flags()
    with FlagWatch() as watch:
        rk.reconstruct(*args, **kwargs)
    after = global_flags()
    if after != before or watch.deviations or not watch.calls:
        flag_changes.append({"backend": kwargs.get("backend"), "precision": kwargs.get("precision"),
                             "torch_calls": watch.calls, "during": watch.deviations,
                             "after": {k: [before.get(k), after.get(k)] for k in set(before) | set(after)
                                       if before.get(k) != after.get(k)}})
torch.manual_seed(0)
heads, head_dim = 4, 16
grid = (2, 3, 4); tokens = 24; channels = 64
table = rk.RopeTable.from_complex(wan_freqs(head_dim))
dense_table = rk.RopeTable.from_cos_sin(torch.randn(80, 8), torch.randn(80, 8))
basis = torch.randn(24, channels).half()
basis_b = torch.randn(30, channels).half()

def destination(layout, dtype, total):
    if layout == "BTHD":
        return torch.zeros(1, total, heads, head_dim, dtype=dtype)
    if layout == "THD":
        return torch.zeros(total + 3, heads, head_dim + 2, dtype=dtype)[3:, :, 1:1 + head_dim]
    return torch.zeros(1, heads, total, head_dim, dtype=dtype)

def shifted_origin_job(rng, view, rank, rope):
    layout = custom_layout(*grid, rank, 1, 1, 3)
    pixels = torch.from_numpy(rng.integers(0, 256, size=(tokens, rank), dtype=np.uint8))
    canvas, indices = pack_with_cells(pixels, layout, origin=(0, 0), extra=(3, 6))
    shifted = torch.zeros_like(canvas)
    origins = []
    for segment, row in enumerate(indices):
        origin_row = []
        for frame, picture in enumerate(row):
            y, x = (segment + frame) % 3, (2 * segment + frame) % 5
            shifted[picture, :, y:, x:] = canvas[picture, :, :canvas.shape[2] - y, :canvas.shape[3] - x]
            origin_row.append((y, x))
        origins.append(origin_row)
    scale, offset = random_affine(rank, rng)
    source = rk.PictureSource.from_canvases(shifted, layout, indices, origins, scale, offset)
    return rk.ReconstructionJob(source, view, grid, basis_b, rank, rope)

def build(seed, layout, dtype, kind, affine):
    rng = np.random.default_rng(seed)
    dst = destination(layout, dtype, 5 * tokens)
    jobs = []
    specs = [(7, None, rk.RopePositions(table, (1, 0, 0)), basis), (20, custom_layout(*grid, 20, 1, 1, 3), None, basis),
             (13, custom_layout(*grid, 13, 2, 1, 1), rk.RopePositions(table, (4, 1, 2)), basis_b),
             (9, None, rk.RopePositions(dense_table, 11), basis_b)]
    for index, (rank, layout_spec, rope, base) in enumerate(specs):
        view = rk.NativeView(dst, layout, token_offset=index * tokens)
        if index == 3 and not affine:
            job = shifted_origin_job(rng, view, rank, rope)
            pixels = rk.unpack_pictures(job.source)
        else:
            job, pixels = make_picture_job(rng, view, grid, base, rank, layout=layout_spec, rope=rope)
        if kind == "uint8":
            job = rk.ReconstructionJob(rk.CoefficientSource(pixels, job.source.scale, job.source.offset), view, grid,
                                       job.basis, rank, rope)
        elif kind == "float":
            job = rk.ReconstructionJob(rk.CoefficientSource(rk.reference_coefficients(job)), view, grid, job.basis,
                                       rank, rope)
        jobs.append(job)
    dense = torch.randn(tokens, channels)
    jobs.append(rk.ReconstructionJob(rk.DenseSource(dense), rk.NativeView(dst, layout, token_offset=4 * tokens), grid,
                                     rope=rk.RopePositions(table, (2, 0, 0))))
    return jobs

configs = {"g1_picT": {"BLOCK_M": 16, "BLOCK_N": 32, "BLOCK_K": 16, "N_GROUP": 1, "PIC_T": True},
           "g2_rowmajor": {"BLOCK_M": 16, "BLOCK_N": 32, "BLOCK_K": 16, "N_GROUP": 2, "PIC_T": False,
                           "UNPACK_BLOCK_M": 16, "UNPACK_BLOCK_K": 16}}
cases = []
for layout, dtype in (("BTHD", torch.float32), ("BTHD", torch.float16), ("THD", torch.float32), ("BHTD", torch.float32)):
    for backend in ("fused", "two_pass"):
        for config in configs:
            cases.append(("pictures", layout, dtype, backend, config, layout != "THD"))
for kind in ("uint8", "float"):
    cases.append((kind, "BTHD", torch.float32, "fused", "g2_rowmajor", True))
for kind, layout, dtype, backend, config_name, affine in cases:
    for precision in ("fp32", "tf32x3", "fp16", "fp16x2"):
        jobs = build(1, layout, dtype, kind, affine)
        checked_reconstruct(jobs, backend=backend, precision=precision, config=configs[config_name])
        worst, violations = 0.0, 0
        for job in jobs:
            exact, bound = rk.error_bound(job, precision)
            written = job.destination.slot(job.tokens).reshape(job.tokens, -1).double()
            ratio = ((written - exact).abs() / bound)
            worst = max(worst, ratio.max().item())
            violations += int((ratio > 1).sum())
        report["cases"].append({"kind": kind, "layout": layout, "dtype": str(dtype), "backend": backend,
                                "config": config_name, "affine": affine, "precision": precision,
                                "worst_ratio": worst, "violations": violations})
rng = np.random.default_rng(3)
dst = torch.zeros(1, tokens, heads, head_dim)
job, _ = make_picture_job(rng, rk.NativeView(dst), grid, basis, 11, rope=rk.RopePositions(table, (0, 0, 0)))
calls = []
def native(values, job):
    calls.append(values.dtype)
    return values * 2
checked_reconstruct([job], backend="interpret", precision="fp32", native_rope=native, native_rope_dtype=torch.float32,
               config=configs["g1_picT"])
pre = (rk.reference_coefficients(job).double() @ basis[:11].double()).float()
report["native_path_max_error"] = (dst.reshape(tokens, -1) - 2 * pre).abs().max().item()
report["native_calls"] = [str(c) for c in calls]
# torch.inference_mode (SGLang): table/sources, destination and the call each inside or outside it, compared with the
# all-outside result. The Triton backends write through pointers, so even an inference-mode destination written
# outside inference mode works.
import contextlib, itertools
def mode(inside):
    return torch.inference_mode() if inside else contextlib.nullcontext()
def inference_run(built, run, dest, backend="fused"):
    with mode(built):
        table_i = rk.RopeTable.from_complex(wan_freqs(head_dim))
    with mode(dest):
        out = torch.zeros(1, tokens, heads, head_dim)
    with mode(built):
        job_i, _ = make_picture_job(np.random.default_rng(9), rk.NativeView(out), grid, basis, 13,
                                    rope=rk.RopePositions(table_i, (1, 0, 0)))
    with mode(run):
        checked_reconstruct([job_i], backend=backend, precision="fp16x2", config=configs["g1_picT"])
    return out
reference_out = inference_run(False, False, False)
modes = {f"built={b},run={r},dest={d}": bool(torch.equal(inference_run(b, r, d), reference_out))
         for b, r, d in itertools.product((False, True), repeat=3)}
modes["two_pass,all_inside"] = bool(torch.equal(inference_run(True, True, True, "two_pass"),
                                                inference_run(False, False, False, "two_pass")))
report["inference_mode"] = modes
# The Triton kernels pass an explicit input_precision to every FP32 dot, so a caller that enabled TF32 for its own
# model does not change their results (the module never sets that flag itself).
def tf32_run(backend, precision):
    torch.manual_seed(5)  # build() draws its dense job from the global generator
    jobs = build(1, "BTHD", torch.float32, "pictures", True)
    checked_reconstruct(jobs, backend=backend, precision=precision, config=configs["g1_picT"])
    return jobs[0].destination.tensor.clone()
tf32_process = {}
for backend, precision in (("fused", "fp32"), ("two_pass", "fp32"), ("fused", "tf32x3")):
    expected = tf32_run(backend, precision)
    with process_fp32_precision("tf32"):
        tf32_process[f"{backend}/{precision}"] = bool(torch.equal(tf32_run(backend, precision), expected))
report["tf32_process_bitwise"] = tf32_process
# warmup(): every synthetic call of each shape runs the real kernels (here: the interpreter) without changing
# process-wide settings; the repeat call does nothing.
warm_shapes = [rk.WarmupShape(heads, head_dim, ranks=(1, 40)),
               rk.WarmupShape(heads, head_dim, source="uint8", rope="none", layout="THD", ranks=(30, 40)),
               rk.WarmupShape(heads, head_dim, source="float", dtype=torch.float32, layout="BHTD", ranks=(1, 40)),
               rk.WarmupShape(heads, head_dim, source="dense", rope="native", source_dtype=torch.float32),
               rk.WarmupShape(heads, head_dim, rope="native", native_rope_dtype=torch.float32, ranks=(5, 9))]
warm_precisions = ["fp16x2", "fp32"]
expected_calls = sum(len(plan[1]) * len(plan[2]) for shape in warm_shapes for name in warm_precisions
                     for plan in [rk._warmup_plan(shape, "interpret", name, dict(rk._DEFAULT_CONFIG), False)])
warm_before = global_flags()
with FlagWatch() as watch:
    first = rk.warmup("cpu", warm_shapes, backend="interpret", precision=warm_precisions)
    second = rk.warmup("cpu", warm_shapes, backend="interpret", precision=warm_precisions)
warm_keys = ("calls", "launches", "warmed", "skipped", "repeat")
report["warmup"] = {"first": {k: first[k] for k in warm_keys}, "second": {k: second[k] for k in warm_keys},
                    "expected_calls": expected_calls, "during": watch.deviations, "torch_calls": watch.calls,
                    "flags_equal": global_flags() == warm_before}
report["flag_changes"] = flag_changes
report["flags_end_equal"] = global_flags() == flags_start
print(json.dumps(report))
"""


class InterpreterKernelTests(unittest.TestCase):
    """The real Triton kernels under the Triton CPU interpreter (separate process).

    Triton is a required dependency of this test: without it the kernels cannot
    be checked, so the test fails instead of passing vacuously.
    """

    def test_kernels_match_reference_within_documented_bound(self):
        environment = dict(os.environ, TRITON_INTERPRET="1", CUDA_VISIBLE_DEVICES="")
        completed = subprocess.run(
            [sys.executable, "-c", _INTERPRETER_SCRIPT, str(ROOT)], capture_output=True, text=True,
            env=environment, timeout=1800, cwd=str(ROOT),
        )
        self.assertEqual(completed.returncode, 0, completed.stderr[-4000:])
        report = json.loads(completed.stdout.strip().splitlines()[-1])
        self.assertTrue(report["triton"], f"Triton is not importable by {sys.executable} ({report['import_error']}); "
                                          "the fused/two-pass kernels cannot be verified")
        self.assertTrue(report["interpreted"], "the kernels did not run under the Triton interpreter")
        self.assertGreaterEqual(len(report["cases"]), 72)
        for case in report["cases"]:
            self.assertEqual(case["violations"], 0, case)
        self.assertLess(report["native_path_max_error"], 1e-4)
        self.assertEqual(report["native_calls"], ["torch.float32"])
        self.assertEqual(len(report["inference_mode"]), 9)
        self.assertTrue(all(report["inference_mode"].values()), report["inference_mode"])
        self.assertEqual(report["flag_changes"], [])
        self.assertTrue(report["flags_end_equal"])
        self.assertEqual(len(report["tf32_process_bitwise"]), 3)
        self.assertTrue(all(report["tf32_process_bitwise"].values()), report["tf32_process_bitwise"])
        warm = report["warmup"]
        self.assertEqual(warm["first"]["calls"], warm["expected_calls"])
        self.assertGreaterEqual(warm["first"]["launches"], warm["first"]["calls"])
        self.assertEqual((warm["first"]["warmed"], warm["first"]["repeat"]), (10, False))
        self.assertEqual(warm["second"], {"calls": 0, "launches": 0, "warmed": 0, "skipped": 10, "repeat": True})
        self.assertEqual(warm["during"], [])
        self.assertGreater(warm["torch_calls"], 0)
        self.assertTrue(warm["flags_equal"])


if __name__ == "__main__":
    unittest.main()
