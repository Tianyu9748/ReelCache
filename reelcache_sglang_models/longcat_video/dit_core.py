"""LongCat-Video 13.6B DiT math (pure torch), shared by the SGLang model and the CPU reference core.

The math is an op-for-op port of ``longcat_video/modules/{longcat_video_dit,attention,blocks}.py`` at
commit ``6b3f4b8`` with the same module/parameter names (checkpoint keys load without remapping):

* native ``amp.autocast('cuda', float32)`` regions (timestep MLP, adaLN modulation, final linear) are
  written as explicit FP32 linears on FP32-cast weights; LayerNorm/RMSNorm run in FP32 and cast back;
  gated residuals are formed in FP32 and cast to BF16 -- exactly the native dtype flow;
* attention runs through an injected op on ``[B, S, H, D]`` tensors (SGLang ``LocalAttention`` in the
  serving model, torch SDPA in the CPU core); softmax scale ``head_dim ** -0.5`` for self- and
  cross-attention;
* the conditioning-cache build is the native timestep-zero, cross-attention-skipped forward; it
  returns every layer's key after ``k_norm`` and before RoPE, and its value, in ``[B, T, H, D]``;
* a forward with a conditioning cache concatenates cache + current keys/values and applies the
  segment-local 3D RoPE to the concatenation at read time (native ``forward_with_kv_cache``);
* context parallelism (optional, :mod:`.context_parallel`) follows native ``context_parallel_util``: each rank
  keeps its 2-D spatial block of every frame after the patch embedding, self-attention runs between Ulysses
  all-to-alls, RoPE uses the rank's rows, the final-layer output is all-gathered; a conditioning cache built or
  read under CP holds the rank's token rows only.

The native final layer of the cache-build forward is skipped (its output is discarded natively).
"""

from __future__ import annotations

import math
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from .context_parallel import ContextParallelError, LongCatContextParallel
from .lora import LoRALinear
from .rope import RopeTables, apply_rope
from .spec import LONGCAT_VIDEO_ARCH, LongCatVideoArch

AttentionFactory = Callable[[str, int, int, float], nn.Module]


def _no_autocast(device_type: str):
    return torch.autocast(device_type=device_type, enabled=False)


def linear_fp32(module: nn.Linear, x: torch.Tensor) -> torch.Tensor:
    """``module(x)`` inside native ``autocast(float32)``: input and parameters cast to FP32."""
    bias = None if module.bias is None else module.bias.float()
    return F.linear(x.float(), module.weight.float(), bias)


class SDPAAttention(nn.Module):
    """Reference attention op on ``[B, S, H, D]`` tensors (torch SDPA)."""

    def __init__(self, kind: str, num_heads: int, head_dim: int, softmax_scale: float) -> None:
        super().__init__()
        self.kind = kind
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.softmax_scale = softmax_scale

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        out = F.scaled_dot_product_attention(q.transpose(1, 2).contiguous(), k.transpose(1, 2).contiguous(),
                                             v.transpose(1, 2).contiguous(), scale=self.softmax_scale)
        return out.transpose(1, 2)


def sdpa_attention_factory(kind: str, num_heads: int, head_dim: int, softmax_scale: float) -> nn.Module:
    return SDPAAttention(kind, num_heads, head_dim, softmax_scale)


class RMSNormFP32(nn.Module):
    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        out = (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)).type_as(x)
        return out * self.weight


class LayerNormFP32(nn.LayerNorm):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        origin_dtype = inputs.dtype
        return F.layer_norm(inputs.float(), self.normalized_shape,
                            None if self.weight is None else self.weight.float(),
                            None if self.bias is None else self.bias.float(), self.eps).to(origin_dtype)


def modulate_fp32(norm: nn.Module, x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    if shift.dtype != torch.float32 or scale.dtype != torch.float32:
        raise TypeError("LongCat modulation parameters must be FP32")
    dtype = x.dtype
    y = norm(x.to(torch.float32))
    y = y * (scale + 1) + shift
    return y.to(dtype)


class PatchEmbed3D(nn.Module):
    def __init__(self, patch_size, in_chans: int, embed_dim: int) -> None:
        super().__init__()
        self.patch_size = tuple(patch_size)
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=self.patch_size, stride=self.patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, d, h, w = x.size()
        pt, ph, pw = self.patch_size
        if w % pw:
            x = F.pad(x, (0, pw - w % pw))
        if h % ph:
            x = F.pad(x, (0, 0, 0, ph - h % ph))
        if d % pt:
            x = F.pad(x, (0, 0, 0, 0, 0, pt - d % pt))
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)


def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half)
    freqs = freqs.to(device=t.device)  # native builds the frequencies on the CPU, then moves them
    args = t[:, None].float() * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class TimestepEmbedder(nn.Module):
    def __init__(self, t_embed_dim: int, frequency_embedding_size: int = 256) -> None:
        super().__init__()
        self.t_embed_dim = t_embed_dim
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(nn.Linear(frequency_embedding_size, t_embed_dim, bias=True), nn.SiLU(),
                                 nn.Linear(t_embed_dim, t_embed_dim, bias=True))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq = timestep_embedding(t, self.frequency_embedding_size)
        h = linear_fp32(self.mlp[0], t_freq)
        h = F.silu(h)
        return linear_fp32(self.mlp[2], h)


class CaptionEmbedder(nn.Module):
    def __init__(self, in_channels: int, hidden_size: int) -> None:
        super().__init__()
        self.y_proj = nn.Sequential(nn.Linear(in_channels, hidden_size, bias=True), nn.GELU(approximate="tanh"),
                                    nn.Linear(hidden_size, hidden_size, bias=True))

    def forward(self, caption: torch.Tensor) -> torch.Tensor:
        return self.y_proj(caption)


class FeedForwardSwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.w1 = LoRALinear(dim, hidden_dim, bias=False)
        self.w2 = LoRALinear(hidden_dim, dim, bias=False)
        self.w3 = LoRALinear(dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class LongCatSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float, attention_factory: AttentionFactory) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = LoRALinear(dim, dim * 3, bias=True)
        self.q_norm = RMSNormFP32(self.head_dim, eps)
        self.k_norm = RMSNormFP32(self.head_dim, eps)
        self.proj = LoRALinear(dim, dim, bias=True)
        self.attn_op = attention_factory("self", num_heads, self.head_dim, self.scale)

    def _qkv(self, x: torch.Tensor):
        b, n, _ = x.shape
        q, k, v = self.qkv(x).view(b, n, 3, self.num_heads, self.head_dim).unbind(2)
        return self.q_norm(q), self.k_norm(k), v

    def _attend(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, cp: LongCatContextParallel | None):
        if cp is None:
            return self.attn_op(q, k, v)
        # native ulysses_wrapper: a2a in per q/k/v, the kernel on H/N heads of the full sequence, a2a out
        return cp.ulysses_out(self.attn_op(cp.ulysses_in(q), cp.ulysses_in(k), cp.ulysses_in(v)))

    def forward(self, x: torch.Tensor, latent_shape: tuple[int, int, int], rope: RopeTables, *,
                num_cond_latents: int = 0, return_kv: bool = False, kv_cache=None,
                cp: LongCatContextParallel | None = None):
        b, n, c = x.shape
        t_frames, th, tw = latent_shape
        shard = None if cp is None else cp.rope_shard
        q, k, v = self._qkv(x)
        kv_out = None
        if kv_cache is not None:
            if not num_cond_latents:
                raise ValueError("a conditioning-cache forward needs num_cond_latents > 0")
            k_cache, v_cache = kv_cache
            if k_cache.shape[0] != v_cache.shape[0] or k_cache.shape[0] not in (1, b):
                raise ValueError("conditioning cache batch must be 1 or the input batch")
            if k_cache.shape[0] == 1 and b > 1:
                k_cache = k_cache.repeat(b, 1, 1, 1)
                v_cache = v_cache.repeat(b, 1, 1, 1)
            n_cache = k_cache.shape[1]
            if n_cache != num_cond_latents * (n // t_frames):
                raise ValueError(f"cache holds {n_cache} tokens, expected {num_cond_latents} latent frames")
            k_full = torch.cat([k_cache, k], dim=1).contiguous()
            v_full = torch.cat([v_cache, v], dim=1).contiguous()
            cos, sin = rope.cos_sin((t_frames + num_cond_latents, th, tw), x.device, shard)
            k_full = apply_rope(k_full, cos, sin)
            q = apply_rope(q, cos[n_cache:], sin[n_cache:])
            out = self._attend(q, k_full, v_full, cp)
        else:
            if return_kv:
                kv_out = (k.contiguous(), v.contiguous())
            cos, sin = rope.cos_sin((t_frames, th, tw), x.device, shard)
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)
            if num_cond_latents:
                nc = num_cond_latents * (n // t_frames)
                x_cond = self._attend(q[:, :nc].contiguous(), k[:, :nc].contiguous(), v[:, :nc].contiguous(), cp)
                x_noise = self._attend(q[:, nc:].contiguous(), k, v, cp)
                out = torch.cat([x_cond, x_noise], dim=1).contiguous()
            else:
                out = self._attend(q, k, v, cp)
        out = self.proj(out.reshape(b, n, c))
        return (out, kv_out) if return_kv else out


class LongCatCrossAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float, attention_factory: AttentionFactory) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q_linear = LoRALinear(dim, dim)
        self.kv_linear = LoRALinear(dim, dim * 2)
        self.proj = LoRALinear(dim, dim)
        self.q_norm = RMSNormFP32(self.head_dim, eps)
        self.k_norm = RMSNormFP32(self.head_dim, eps)
        self.attn_op = attention_factory("cross", num_heads, self.head_dim, self.head_dim ** -0.5)

    def _attend(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        b, n, c = x.shape
        m = cond.shape[1]
        q = self.q_linear(x).view(b, n, self.num_heads, self.head_dim)
        k, v = self.kv_linear(cond).view(b, m, 2, self.num_heads, self.head_dim).unbind(2)
        q, k = self.q_norm(q), self.k_norm(k)
        out = self.attn_op(q, k, v.contiguous())
        return self.proj(out.reshape(b, n, c))

    def forward(self, x: torch.Tensor, cond: torch.Tensor, latent_shape, num_cond_latents: int | None = None):
        if not num_cond_latents:
            return self._attend(x, cond)
        b, n, c = x.shape
        nc = num_cond_latents * (n // latent_shape[0])
        out_noise = self._attend(x[:, nc:], cond)
        return torch.cat([torch.zeros((b, nc, c), dtype=out_noise.dtype, device=out_noise.device), out_noise],
                         dim=1).contiguous()


class LongCatVideoBlock(nn.Module):
    def __init__(self, arch: LongCatVideoArch, attention_factory: AttentionFactory) -> None:
        super().__init__()
        c = arch.hidden_size
        self.hidden_size = c
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(arch.adaln_tembed_dim, 6 * c, bias=True))
        self.mod_norm_attn = LayerNormFP32(c, eps=arch.norm_eps, elementwise_affine=False)
        self.mod_norm_ffn = LayerNormFP32(c, eps=arch.norm_eps, elementwise_affine=False)
        self.pre_crs_attn_norm = LayerNormFP32(c, eps=arch.norm_eps, elementwise_affine=True)
        self.attn = LongCatSelfAttention(c, arch.num_heads, arch.norm_eps, attention_factory)
        self.cross_attn = LongCatCrossAttention(c, arch.num_heads, arch.norm_eps, attention_factory)
        self.ffn = FeedForwardSwiGLU(c, arch.ffn_hidden_dim)

    def forward(self, x, y, t, latent_shape, rope: RopeTables, *, num_cond_latents: int = 0,
                return_kv: bool = False, kv_cache=None, skip_crs_attn: bool = False,
                cp: LongCatContextParallel | None = None):
        x_dtype = x.dtype
        b, n, c = x.shape
        t_frames = latent_shape[0]
        with _no_autocast(x.device.type):
            mod = linear_fp32(self.adaLN_modulation[1], F.silu(t.float()))
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod.unsqueeze(2).chunk(6, dim=-1)
            x_m = modulate_fp32(self.mod_norm_attn, x.view(b, t_frames, -1, c), shift_msa, scale_msa).view(b, n, c)
            attn_out = self.attn(x_m, latent_shape, rope, num_cond_latents=num_cond_latents, return_kv=return_kv,
                                 kv_cache=kv_cache, cp=cp)
            if return_kv:
                x_s, kv = attn_out
            else:
                x_s, kv = attn_out, None
            x = (x + (gate_msa * x_s.view(b, -1, n // t_frames, c)).view(b, -1, c)).to(x_dtype)
            if not skip_crs_attn:
                cross_cond = None if kv_cache is not None else num_cond_latents
                x = x + self.cross_attn(self.pre_crs_attn_norm(x), y, latent_shape, num_cond_latents=cross_cond)
            x_m = modulate_fp32(self.mod_norm_ffn, x.view(b, -1, n // t_frames, c), shift_mlp, scale_mlp).view(b, -1, c)
            x_s = self.ffn(x_m)
            x = (x + (gate_mlp * x_s.view(b, -1, n // t_frames, c)).view(b, -1, c)).to(x_dtype)
        return (x, kv) if return_kv else x


class FinalLayerFP32(nn.Module):
    def __init__(self, hidden_size: int, num_patch: int, out_channels: int, adaln_tembed_dim: int,
                 eps: float = 1e-6) -> None:
        super().__init__()
        self.norm_final = LayerNormFP32(hidden_size, elementwise_affine=False, eps=eps)
        self.linear = nn.Linear(hidden_size, num_patch * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(adaln_tembed_dim, 2 * hidden_size, bias=True))

    def forward(self, x: torch.Tensor, t: torch.Tensor, latent_shape) -> torch.Tensor:
        b, n, c = x.shape
        t_frames = latent_shape[0]
        with _no_autocast(x.device.type):
            shift, scale = linear_fp32(self.adaLN_modulation[1], F.silu(t.float())).unsqueeze(2).chunk(2, dim=-1)
            x = modulate_fp32(self.norm_final, x.view(b, t_frames, -1, c), shift, scale).view(b, n, c)
            return linear_fp32(self.linear, x)


class LongCatVideoDiTMixin:
    """Module construction and forward math; mixed into an ``nn.Module`` subclass."""

    longcat_arch: LongCatVideoArch

    def _build_longcat_modules(self, arch: LongCatVideoArch, attention_factory: AttentionFactory) -> None:
        self.longcat_arch = arch
        self.patch_size = tuple(arch.patch_size)
        self.in_channels = arch.in_channels
        self.out_channels = arch.out_channels
        self.x_embedder = PatchEmbed3D(arch.patch_size, arch.in_channels, arch.hidden_size)
        self.t_embedder = TimestepEmbedder(arch.adaln_tembed_dim, arch.frequency_embedding_size)
        self.y_embedder = CaptionEmbedder(arch.caption_channels, arch.hidden_size)
        self.blocks = nn.ModuleList(LongCatVideoBlock(arch, attention_factory) for _ in range(arch.depth))
        self.final_layer = FinalLayerFP32(arch.hidden_size, arch.patch_numel, arch.out_channels,
                                          arch.adaln_tembed_dim, eps=arch.norm_eps)
        self.rope_tables = RopeTables(arch.head_dim, arch.rope_base)
        self.longcat_cp: LongCatContextParallel | None = None

    # -- context parallelism -----------------------------------------------------------------------
    def set_context_parallel(self, cp: LongCatContextParallel | None) -> None:
        """Run forwards context-parallel over ``cp`` (``None``: single device). Weights stay replicated."""
        if cp is not None:
            if not isinstance(cp, LongCatContextParallel):
                raise ContextParallelError("set_context_parallel needs a LongCatContextParallel")
            if self.longcat_arch.num_heads % cp.size:
                raise ContextParallelError(f"{self.longcat_arch.num_heads} heads are not divisible by cp={cp.size}")
        self.longcat_cp = cp

    def context_parallel(self) -> LongCatContextParallel | None:
        return self.longcat_cp

    # -- helpers ---------------------------------------------------------------------------------
    @property
    def compute_dtype(self) -> torch.dtype:
        return self.x_embedder.proj.weight.dtype

    def _token_grid(self, hidden_states: torch.Tensor) -> tuple[int, int, int]:
        _, _, t, h, w = hidden_states.shape
        pt, ph, pw = self.patch_size
        if pt != 1:
            raise ValueError("temporal patching is not supported")
        return t // pt, h // ph, w // pw

    def _embed_timestep(self, timestep: torch.Tensor, batch: int, n_t: int, dtype: torch.dtype) -> torch.Tensor:
        if timestep.ndim == 1:
            timestep = timestep.unsqueeze(1).expand(-1, n_t)
        timestep = timestep.to(dtype)
        with _no_autocast(timestep.device.type):
            return self.t_embedder(timestep.float().flatten()).reshape(batch, n_t, -1)

    def _embed_text(self, encoder_hidden_states: torch.Tensor, encoder_attention_mask, dtype) -> torch.Tensor:
        y = self.y_embedder(encoder_hidden_states.to(dtype))  # native: [B, 1, N_text, C]
        if y.ndim == 3:
            y = y.unsqueeze(1)
        b = y.shape[0]
        if encoder_attention_mask is not None:
            mask = encoder_attention_mask
            if mask.ndim == 4:
                mask = mask.squeeze(1).squeeze(1)
            if mask.ndim != 2 or mask.shape != (b, y.shape[2]):
                raise ValueError(f"encoder_attention_mask {tuple(encoder_attention_mask.shape)} does not match text")
            if not self.longcat_arch.text_tokens_zero_pad:
                raise NotImplementedError("variable-length text (text_tokens_zero_pad=False) is not ported")
            y = y * mask[:, None, :, None]
        return y.squeeze(1)

    def unpatchify(self, x: torch.Tensor, n_t: int, n_h: int, n_w: int) -> torch.Tensor:
        pt, ph, pw = self.patch_size
        c_out = self.out_channels
        b = x.shape[0]
        x = x.view(b, n_t, n_h, n_w, pt, ph, pw, c_out)
        x = x.permute(0, 7, 1, 4, 2, 5, 3, 6)
        return x.reshape(b, c_out, n_t * pt, n_h * ph, n_w * pw)

    # -- forwards ----------------------------------------------------------------------------------
    def longcat_forward(self, hidden_states: torch.Tensor, timestep: torch.Tensor,
                        encoder_hidden_states: torch.Tensor | None, encoder_attention_mask=None, *,
                        num_cond_latents: int = 0, kv_cache_layers=None, return_kv: bool = False,
                        skip_crs_attn: bool = False, kv_cache_ready=None):
        """Native ``LongCatVideoTransformer3DModel.forward``; with ``return_kv`` returns the per-layer
        pre-RoPE ``(keys, values)`` in ``[B, T, H, D]`` instead of the prediction (under context parallelism:
        this rank's token rows, ``[B, T_local, H, D]``; ``kv_cache_layers`` must then hold the same rows).
        ``kv_cache_ready`` (a :class:`~.conditioning_cache.LayerReadiness`): the cache is still being restored; block
        ``i`` waits for layer ``i`` (host: until issued; device: the layer's event) -- layer-ahead overlap."""
        if return_kv and kv_cache_layers is not None:
            raise ValueError("return_kv and a conditioning cache are mutually exclusive")
        cp = self.context_parallel()
        b = hidden_states.shape[0]
        n_t, n_h, n_w = self._token_grid(hidden_states)
        dtype = self.compute_dtype
        hidden_states = hidden_states.to(dtype)
        x = self.x_embedder(hidden_states)
        if cp is not None:
            x = cp.split_tokens(x, (n_t, n_h, n_w))  # native split_cp_2d: this rank's spatial block of every frame
        t = self._embed_timestep(timestep, b, n_t, dtype)
        y = None
        if not skip_crs_attn:
            if encoder_hidden_states is None:
                raise ValueError("encoder_hidden_states are required unless cross-attention is skipped")
            y = self._embed_text(encoder_hidden_states, encoder_attention_mask, dtype)
        if kv_cache_layers is not None and len(kv_cache_layers) != len(self.blocks):
            raise ValueError(f"conditioning cache has {len(kv_cache_layers)} layers, model has {len(self.blocks)}")
        keys, values = [], []
        shape = (n_t, n_h, n_w)
        if kv_cache_ready is not None and kv_cache_layers is None:
            raise ValueError("kv_cache_ready needs kv_cache_layers")
        for i, block in enumerate(self.blocks):
            kv = None if kv_cache_layers is None else kv_cache_layers[i]
            if kv_cache_ready is not None:
                kv_cache_ready.wait(i)
            out = block(x, y, t, shape, self.rope_tables, num_cond_latents=num_cond_latents, return_kv=return_kv,
                        kv_cache=kv, skip_crs_attn=skip_crs_attn, cp=cp)
            if return_kv:
                x, (k_i, v_i) = out
                keys.append(k_i)
                values.append(v_i)
            else:
                x = out
        if return_kv:
            return keys, values
        x = self.final_layer(x, t, shape)
        if cp is not None:
            x = cp.gather_tokens(x, shape)  # native gather_cp_2d: every rank gets the full prediction
        x = self.unpatchify(x, n_t, n_h, n_w)
        return x.to(torch.float32)

    def build_conditioning_kv(self, cond_latents: torch.Tensor):
        """Native ``_cache_clean_latents``: timestep 0, cross-attention skipped, keys before RoPE."""
        b, _, t = cond_latents.shape[:3]
        timestep = torch.zeros(b, t).to(device=cond_latents.device, dtype=self.compute_dtype)
        return self.longcat_forward(cond_latents, timestep, None, return_kv=True, skip_crs_attn=True)

    def forward_with_conditioning(self, hidden_states: torch.Tensor, timestep: torch.Tensor,
                                  encoder_hidden_states: torch.Tensor, encoder_attention_mask, kv_cache_layers,
                                  num_cond_latents: int, kv_cache_ready=None) -> torch.Tensor:
        """Native denoising call of ``generate_vc(use_kv_cache=True)``."""
        return self.longcat_forward(hidden_states, timestep, encoder_hidden_states, encoder_attention_mask,
                                    num_cond_latents=num_cond_latents, kv_cache_layers=kv_cache_layers,
                                    kv_cache_ready=kv_cache_ready)


class LongCatVideoDiTCore(LongCatVideoDiTMixin, nn.Module):
    """SGLang-free model with the native parameter names (CPU reference and unit tests)."""

    def __init__(self, arch: LongCatVideoArch = LONGCAT_VIDEO_ARCH,
                 attention_factory: AttentionFactory = sdpa_attention_factory) -> None:
        nn.Module.__init__(self)
        self._build_longcat_modules(arch, attention_factory)

    def forward(self, hidden_states, timestep, encoder_hidden_states=None, encoder_attention_mask=None, *,
                num_cond_latents: int = 0, kv_cache_layers=None, return_kv: bool = False,
                skip_crs_attn: bool = False, kv_cache_ready=None):
        return self.longcat_forward(hidden_states, timestep, encoder_hidden_states, encoder_attention_mask,
                                    num_cond_latents=num_cond_latents, kv_cache_layers=kv_cache_layers,
                                    return_kv=return_kv, skip_crs_attn=skip_crs_attn, kv_cache_ready=kv_cache_ready)
