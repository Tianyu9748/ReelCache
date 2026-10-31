"""Causal-Wan DiT used by the profiles (imports SGLang).

At e00930c ``CausalWanTransformerBlock.forward`` calls ``self.attn2(..., crossattn_cache=...)`` but
``WanT2VCrossAttention.forward(x, context, context_lens)`` has no such argument, so the stock
``CausalWanTransformer3DModel`` cannot run KV-cache inference (LongLive 2.0 avoids it with its own block's
``_cross_attn_with_cache``). ``ReelCacheCausalWanTransformer3DModel`` keeps every parameter and all other math of
``CausalWanTransformer3DModel`` and only gives each block's cross-attention the cache semantics of the original
repositories (compute the prompt K/V on the first forward of a request, reuse them until the cache is reset).
It is registered under its own architecture name, so SGLang's ``CausalWanTransformer3DModel`` is untouched. On the
SGLang branch ``reelcache-integration`` the stock block already builds a cache-aware ``CausalWanT2VCrossAttention``
(same math); it is re-classed the same way, so the profile DiT is identical on both trees. The branch's base class
also provides the causal RoPE hooks the ReelCache bridge uses to re-rotate restored pre-RoPE keys
(``causal_rotary_cos_sin`` / ``causal_rotate_key``); the profile DiT inherits them unchanged (all three profiles
apply SGLang's causal RoPE inside ``CausalWanSelfAttention``).

DiT precision policies (``apply_precision_policy``, chosen by the profile's ``dtypes.sglang_dit_compute``):

* ``bf16``: SGLang's causal-Wan block math unchanged (bf16 weights; fp32 modulation and norms with fused kernels,
  bf16 residual stream).
* ``self_forcing_bf16``: the Self-Forcing lineage (CausVid, LongLive 1.0) as it runs natively: ``pipe.to(bfloat16)``
  with the upstream ``amp.autocast(dtype=torch.float32)`` regions commented out, so the modulation sum
  ``modulation + e``, ``norm(x) * (1 + scale) + shift``, the gated residual adds, LayerNorms (bf16 in/out) and the head
  modulation all run in bf16; WanRMSNorm normalises in fp32 and rounds before the bf16 weight; the sinusoid is
  float64 rounded to bf16.
* ``wan_fp32_regions``: the original Wan2.1 mixed precision that RAVEN runs: fp32 weights and fp32 residual stream;
  sinusoid in float64, time embedding/projection, modulation, residual adds, LayerNorms and the head in fp32 (the
  ``amp.autocast(dtype=torch.float32)`` regions); every q/k/v/o, FFN, patch and text-embedding matmul stays under the
  stage's bf16 autocast; WanRMSNorm rounds the normalised value to the matmul dtype before the fp32 weight; attention
  inputs are bf16.

Both native policies also compute the patch embedding as the native ``Conv3d`` (SGLang's ``PatchEmbed`` uses an
equivalent GEMM). Modules are re-classed in place (same parameters, no copy); the q/k/v/o, FFN and text-embedding
matmuls, RoPE and attention stay SGLang's.
"""

from __future__ import annotations

import contextlib

import torch
import torch.nn.functional as F
from torch import nn

from sglang.multimodal_gen.runtime.layers.kvcache.causal_attention_cache import CrossAttentionKVCache
from sglang.multimodal_gen.runtime.layers.layernorm import LayerNormScaleShift, tensor_parallel_rms_norm
from sglang.multimodal_gen.runtime.layers.visual_embedding import ModulateProjection, PatchEmbed, TimestepEmbedder
from sglang.multimodal_gen.runtime.models.dits.causal_wanvideo import (
    CausalWanTransformer3DModel,
    CausalWanTransformerBlock,
)
from sglang.multimodal_gen.runtime.models.dits.wanvideo import WanT2VCrossAttention

try:  # SGLang branch reelcache-integration (Step 11 CR a): the stock causal block already has a cache-aware class
    from sglang.multimodal_gen.runtime.models.dits.causal_wanvideo import (
        CausalWanT2VCrossAttention as _BranchCausalCrossAttention,
    )
except ImportError:  # e00930c
    _BranchCausalCrossAttention = None

DIT_ARCH_NAME = "ReelCacheCausalWanTransformer3DModel"
DIT_MODULE_PATH = "reelcache_sglang_models.causal_wan.sglang_dit:ReelCacheCausalWanTransformer3DModel"
DIT_PRECISION_POLICIES = ("bf16", "self_forcing_bf16", "wan_fp32_regions")
# Cross-attention classes the stock causal block builds (re-classed to ``CachedWanT2VCrossAttention``, same math).
STOCK_CROSS_ATTENTION = tuple(c for c in (WanT2VCrossAttention, _BranchCausalCrossAttention) if c is not None)


class CachedWanT2VCrossAttention(WanT2VCrossAttention):
    """``WanT2VCrossAttention`` + the ``crossattn_cache`` argument (same math as LongLive2's ``_cross_attn_with_cache``)."""

    def forward(self, x, context, context_lens=None, crossattn_cache: CrossAttentionKVCache | None = None):
        if crossattn_cache is None:
            return super().forward(x, context, context_lens)
        q, _ = self.to_q(x)
        q = tensor_parallel_rms_norm(q, self.norm_q) if self.tp_rmsnorm else self.norm_q(q)
        q = q.unflatten(2, (self.local_num_heads, self.head_dim))
        if crossattn_cache.is_init:
            k, v = crossattn_cache.k, crossattn_cache.v
        else:
            k, _ = self.to_k(context)
            k = tensor_parallel_rms_norm(k, self.norm_k) if self.tp_rmsnorm else self.norm_k(k)
            k = k.unflatten(2, (self.local_num_heads, self.head_dim))
            v, _ = self.to_v(context)
            v = v.unflatten(2, (self.local_num_heads, self.head_dim))
            crossattn_cache.store(k, v)
        out = self.attn(q, k, v).flatten(2)
        out, _ = self.to_out(out)
        return out


# ------------------------------------------------------------------------------------ wan_fp32_regions policy
@contextlib.contextmanager
def fp32_region(device_type: str, tf32: bool = False):
    """A Wan2.1 ``amp.autocast(dtype=torch.float32)`` region: autocast off, fp32 math; TF32 matmuls iff ``tf32``.

    Uses the legacy ``allow_tf32`` flag only (as the native harness does) and restores it on exit, also on errors;
    it writes the flag only when it differs (``tf32=False`` regions write nothing while TF32 is off, the default).
    The flag is process-wide, but parity with the native TF32 run needs it: PyTorch has no thread-local or per-op
    TF32 switch (``NoTF32Guard`` is C++-only and can only disable), and no flag-free computation reproduces the
    cuBLAS TF32 GEMM bit for bit (phase-2 ``tf32_probe``: TF32-rounded operands with IEEE FP32 or FP64 products
    differ by 2e-7 to 4e-6, plain IEEE FP32 by 3e-4). RAVEN opens it around three GEMM sites per forward (time
    embedding MLP, time projection, head). Thread-safety assumption: one thread runs the forwards (SGLang's GPU
    worker) and no other thread of the process depends on the FP32 cuBLAS precision meanwhile; see
    ``profiles.archive_matmul_precision`` for the ReelCache side."""
    prev = torch.backends.cuda.matmul.allow_tf32
    if prev != tf32:
        torch.backends.cuda.matmul.allow_tf32 = tf32
    try:
        with torch.autocast(device_type=device_type, enabled=False):
            yield
    finally:
        if prev != tf32:
            torch.backends.cuda.matmul.allow_tf32 = prev


def wan_rms_norm(norm, x: torch.Tensor, weight_dtype: torch.dtype | None = None) -> torch.Tensor:
    """``WanRMSNorm.forward``: normalise in fp32, round to ``x.dtype``, then scale by the weight.

    The weight is used in its own dtype (fp32 for RAVEN: fp32 result) unless ``weight_dtype`` asks for the
    native module's parameter dtype (bf16 for the Self-Forcing lineage: bf16 result)."""
    xf = x.float()
    w = norm.weight if weight_dtype is None else norm.weight.to(weight_dtype)
    return (xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + norm.variance_epsilon)).type_as(x) * w


def wan_sinusoidal_embedding_1d(dim: int, position: torch.Tensor) -> torch.Tensor:
    """Wan2.1 ``sinusoidal_embedding_1d`` (float64, [cos | sin])."""
    half = dim // 2
    position = position.to(torch.float64)
    sinusoid = torch.outer(position, torch.pow(10000, -torch.arange(half, device=position.device).to(position).div(half)))
    return torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)


class WanFP32TimestepEmbedder(TimestepEmbedder):
    """Wan2.1 ``time_embedding(sinusoidal_embedding_1d(t).float())`` computed in fp32."""

    reelcache_tf32 = False

    def forward(self, t: torch.Tensor, timestep_seq_len: int | None = None) -> torch.Tensor:
        t_freq = wan_sinusoidal_embedding_1d(self.frequency_embedding_size, t.flatten()).float()
        if timestep_seq_len is not None:
            t_freq = t_freq.unflatten(0, (t_freq.shape[0] // timestep_seq_len, timestep_seq_len))
        with fp32_region(t_freq.device.type, self.reelcache_tf32):
            return self.mlp(t_freq)


class WanFP32ModulateProjection(ModulateProjection):
    """Wan2.1 ``time_projection`` (SiLU + Linear) computed in fp32."""

    reelcache_tf32 = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with fp32_region(x.device.type, self.reelcache_tf32):
            x, _ = self.linear(self.act(x.float()))
        return x


class WanFP32HeadNorm(LayerNormScaleShift):
    """Head ``norm(x) * (1 + scale) + shift`` in fp32 (Wan2.1 ``Head`` inside its fp32 region)."""

    def forward(self, x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        with fp32_region(x.device.type):
            n = F.layer_norm(x.float(), (x.shape[-1],), None, None, self.eps)
            if scale.dim() == 4:  # [B, F, 1, C] per latent frame
                f = scale.shape[1]
                return (n.unflatten(1, (f, n.shape[1] // f)) * (1 + scale.float()) + shift.float()).flatten(1, 2)
            return n * (1 + scale.float()) + shift.float()


class WanFP32Linear(nn.Linear):
    """Head projection in fp32 (Wan2.1 ``Head.head`` inside its fp32 region)."""

    reelcache_tf32 = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with fp32_region(x.device.type, self.reelcache_tf32):
            return F.linear(x.float(), self.weight.float(), None if self.bias is None else self.bias.float())


class WanNativeNormCrossAttention(CachedWanT2VCrossAttention):
    """Cross-attention with native WanRMSNorm q/k semantics; attention inputs in the matmul dtype (bf16)."""

    reelcache_norm_weight_dtype: torch.dtype | None = None  # bf16 for the Self-Forcing lineage (native param dtype)

    def forward(self, x, context, context_lens=None, crossattn_cache: CrossAttentionKVCache | None = None):
        q, _ = self.to_q(x)
        dt = q.dtype
        wd = self.reelcache_norm_weight_dtype
        q = wan_rms_norm(self.norm_q, q, wd).to(dt).unflatten(2, (self.local_num_heads, self.head_dim))
        if crossattn_cache is not None and crossattn_cache.is_init:
            k, v = crossattn_cache.k, crossattn_cache.v
        else:
            k, _ = self.to_k(context)
            k = wan_rms_norm(self.norm_k, k, wd).to(dt).unflatten(2, (self.local_num_heads, self.head_dim))
            v, _ = self.to_v(context)
            v = v.unflatten(2, (self.local_num_heads, self.head_dim))
            if crossattn_cache is not None:
                crossattn_cache.store(k, v)
        out = self.attn(q, k, v).flatten(2)
        out, _ = self.to_out(out)
        return out


class WanFP32RegionsCausalBlock(CausalWanTransformerBlock):
    """``CausalWanTransformerBlock`` with RAVEN/Wan2.1 precision: fp32 residual stream, modulation and LayerNorms."""

    def forward(self, hidden_states, encoder_hidden_states, temb, freqs_cis, block_mask, kv_cache=None,
                crossattn_cache=None, current_start: int = 0, cache_start: int | None = None):
        if hidden_states.dim() == 4:
            hidden_states = hidden_states.squeeze(1)
        num_frames = temb.shape[1]
        frame_seqlen = hidden_states.shape[1] // num_frames
        dev = hidden_states.device.type
        dim = hidden_states.shape[-1]

        def modulate(t: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
            return (t.unflatten(1, (num_frames, frame_seqlen)) * (1 + scale) + shift).flatten(1, 2)

        def gated(t: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:  # [B, L, C] x [B, F, 1, C]
            return (t.unflatten(1, (num_frames, frame_seqlen)) * gate).flatten(1, 2)

        with fp32_region(dev):
            e = self.scale_shift_table.float() + temb.float()
            shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = e.chunk(6, dim=2)
            x = hidden_states.float()
            m1 = modulate(F.layer_norm(x, (dim,), None, None, self.norm1.eps), scale_msa, shift_msa)
        query, _ = self.to_q(m1)
        key, _ = self.to_k(m1)
        value, _ = self.to_v(m1)
        query = wan_rms_norm(self.norm_q, query).unflatten(2, (self.local_num_heads, self.dim_head))
        key = wan_rms_norm(self.norm_k, key).unflatten(2, (self.local_num_heads, self.dim_head))
        value = value.unflatten(2, (self.local_num_heads, self.dim_head))
        attn_output = self.attn1(query, key, value, freqs_cis, block_mask, kv_cache, current_start, cache_start)
        attn_output, _ = self.to_out(attn_output.flatten(2))
        n3_mod = self.self_attn_residual_norm.norm
        with fp32_region(dev):
            x = x + gated(attn_output.float(), gate_msa)
            n3 = F.layer_norm(x, (dim,), None if n3_mod.weight is None else n3_mod.weight.float(),
                              None if n3_mod.bias is None else n3_mod.bias.float(), n3_mod.eps)
        y = self.attn2(n3, context=encoder_hidden_states, context_lens=None, crossattn_cache=crossattn_cache)
        with fp32_region(dev):
            x = x + y.float()
            m2 = modulate(F.layer_norm(x, (dim,), None, None, self.cross_attn_residual_norm.norm.eps),
                          c_scale_msa, c_shift_msa)
        ff = self.ffn(m2)
        with fp32_region(dev):
            x = x + gated(ff.float(), c_gate_msa)
        return x


class SelfForcingBF16CausalBlock(CausalWanTransformerBlock):
    """``CausalWanTransformerBlock`` with the native Self-Forcing/CausVid/LongLive 1.0 bf16 elementwise math."""

    def forward(self, hidden_states, encoder_hidden_states, temb, freqs_cis, block_mask, kv_cache=None,
                crossattn_cache=None, current_start: int = 0, cache_start: int | None = None):
        if hidden_states.dim() == 4:
            hidden_states = hidden_states.squeeze(1)
        num_frames = temb.shape[1]
        frame_seqlen = hidden_states.shape[1] // num_frames
        dev = hidden_states.device.type
        dim = hidden_states.shape[-1]
        x = hidden_states

        def modulate(t: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
            return (t.unflatten(1, (num_frames, frame_seqlen)) * (1 + scale) + shift).flatten(1, 2)

        def gated(t: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
            return (t.unflatten(1, (num_frames, frame_seqlen)) * gate).flatten(1, 2)

        dt = x.dtype  # bf16: the native modules' parameter dtype (SGLang keeps some norm params in fp32)
        with torch.autocast(device_type=dev, enabled=False):  # LayerNorm in/out bf16, as without autocast
            e = self.scale_shift_table.to(dt) + temb.to(dt)
            shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = e.chunk(6, dim=2)
            m1 = modulate(F.layer_norm(x, (dim,), None, None, self.norm1.eps), scale_msa, shift_msa)
        query, _ = self.to_q(m1)
        key, _ = self.to_k(m1)
        value, _ = self.to_v(m1)
        query = wan_rms_norm(self.norm_q, query, dt).unflatten(2, (self.local_num_heads, self.dim_head))
        key = wan_rms_norm(self.norm_k, key, dt).unflatten(2, (self.local_num_heads, self.dim_head))
        value = value.unflatten(2, (self.local_num_heads, self.dim_head))
        attn_output = self.attn1(query, key, value, freqs_cis, block_mask, kv_cache, current_start, cache_start)
        attn_output, _ = self.to_out(attn_output.flatten(2))
        n3_mod = self.self_attn_residual_norm.norm
        with torch.autocast(device_type=dev, enabled=False):
            x = x + gated(attn_output.to(dt), gate_msa)
            n3 = F.layer_norm(x, (dim,), None if n3_mod.weight is None else n3_mod.weight.to(dt),
                              None if n3_mod.bias is None else n3_mod.bias.to(dt), n3_mod.eps)
        y = self.attn2(n3, context=encoder_hidden_states, context_lens=None, crossattn_cache=crossattn_cache)
        with torch.autocast(device_type=dev, enabled=False):
            x = x + y.to(dt)
            m2 = modulate(F.layer_norm(x, (dim,), None, None, self.cross_attn_residual_norm.norm.eps),
                          c_scale_msa, c_shift_msa)
        ff = self.ffn(m2)
        with torch.autocast(device_type=dev, enabled=False):
            x = x + gated(ff.to(dt), c_gate_msa)
        return x


class SelfForcingBF16HeadNorm(LayerNormScaleShift):
    """Head ``norm(x) * (1 + scale) + shift`` in bf16 (the native CausalHead without its fp32 autocast region)."""

    def forward(self, x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=x.device.type, enabled=False):
            n = F.layer_norm(x, (x.shape[-1],), None, None, self.eps)
            scale, shift = scale.to(x.dtype), shift.to(x.dtype)
            if scale.dim() == 4:
                f = scale.shape[1]
                return (n.unflatten(1, (f, n.shape[1] // f)) * (1 + scale) + shift).flatten(1, 2)
            return n * (1 + scale) + shift


class WanSinusoidTimestepEmbedder(TimestepEmbedder):
    """``time_embedding(sinusoidal_embedding_1d(t).type_as(x))``: float64 sinusoid rounded to the weight dtype."""

    def forward(self, t: torch.Tensor, timestep_seq_len: int | None = None) -> torch.Tensor:
        t_freq = wan_sinusoidal_embedding_1d(self.frequency_embedding_size, t.flatten()).to(self.mlp.fc_in.weight.dtype)
        if timestep_seq_len is not None:
            t_freq = t_freq.unflatten(0, (t_freq.shape[0] // timestep_seq_len, timestep_seq_len))
        return self.mlp(t_freq)


class NativeConvPatchEmbed(PatchEmbed):
    """Patch embedding as the native ``nn.Conv3d`` call (same weights as SGLang's GEMM formulation)."""

    def forward(self, x):
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)
        return self.norm(x)


def _fp32_patch_output(module, inputs, output):
    return output.float()


class ReelCacheCausalWanTransformer3DModel(CausalWanTransformer3DModel):
    def __init__(self, config, hf_config, quant_config=None) -> None:
        super().__init__(config=config, hf_config=hf_config, quant_config=quant_config)
        for i, block in enumerate(self.blocks):
            if type(block.attn2) not in STOCK_CROSS_ATTENTION:
                raise TypeError(f"block {i}: unexpected cross-attention {type(block.attn2).__name__}")
            block.attn2.__class__ = CachedWanT2VCrossAttention
        self.reelcache_precision_policy = "bf16"
        self.reelcache_fp32_regions_tf32 = False
        self.reelcache_params_rounded_at_use: list[str] = []

    def apply_precision_policy(self, policy: str, *, tf32: bool = False) -> dict:
        """Switch the loaded DiT to a profile precision policy (idempotent; parameters are not copied)."""
        if policy not in DIT_PRECISION_POLICIES:
            raise ValueError(f"unknown DiT precision policy {policy!r}")
        if policy == self.reelcache_precision_policy and tf32 == self.reelcache_fp32_regions_tf32:
            return self.precision_report()
        if self.reelcache_precision_policy != "bf16":
            raise RuntimeError(f"precision policy already set to {self.reelcache_precision_policy!r}")
        if tf32 and policy != "wan_fp32_regions":
            raise ValueError("tf32 applies to the fp32 regions only")
        if policy == "bf16":
            return self.precision_report()
        want = torch.float32 if policy == "wan_fp32_regions" else torch.bfloat16
        off = [n for n, p in self.named_parameters() if p.dtype != want]
        # SGLang keeps LayerNorm affine params in fp32 even for bf16 DiTs; the bf16 policy rounds them at use as the
        # native bf16 modules hold them. Every other parameter (all matmul weights) must already be ``want``.
        bad = [n for n in off if not (want == torch.bfloat16 and n.endswith(("_residual_norm.norm.weight",
                                                                            "_residual_norm.norm.bias")))]
        if bad:
            raise ValueError(f"{policy} needs {want} DiT weights; {len(bad)} parameters differ, e.g. {bad[:4]}")
        self.reelcache_params_rounded_at_use = off
        block_cls = WanFP32RegionsCausalBlock if policy == "wan_fp32_regions" else SelfForcingBF16CausalBlock
        for i, block in enumerate(self.blocks):
            if type(block) is not CausalWanTransformerBlock or type(block.attn2) is not CachedWanT2VCrossAttention:
                raise TypeError(f"block {i}: unexpected {type(block).__name__}/{type(block.attn2).__name__}")
            if block.tp_rmsnorm or block.attn2.tp_rmsnorm:
                raise ValueError(f"{policy} supports single-rank (non-TP) RMSNorm only")
            block.__class__ = block_cls
            block.attn2.__class__ = WanNativeNormCrossAttention
            block.attn2.reelcache_norm_weight_dtype = None if policy == "wan_fp32_regions" else torch.bfloat16
        emb = self.condition_embedder
        if type(emb.time_embedder) is not TimestepEmbedder or type(emb.time_modulation) is not ModulateProjection:
            raise TypeError("unexpected time embedding modules")
        if type(self.norm_out) is not LayerNormScaleShift or type(self.proj_out) is not nn.Linear:
            raise TypeError("unexpected head modules")
        if type(self.patch_embedding) is not PatchEmbed:
            raise TypeError("unexpected patch embedding module")
        self.patch_embedding.__class__ = NativeConvPatchEmbed
        if policy == "wan_fp32_regions":
            emb.time_embedder.__class__ = WanFP32TimestepEmbedder
            emb.time_modulation.__class__ = WanFP32ModulateProjection
            self.norm_out.__class__ = WanFP32HeadNorm
            self.proj_out.__class__ = WanFP32Linear
            for m in (emb.time_embedder, emb.time_modulation, self.proj_out):
                m.reelcache_tf32 = bool(tf32)
            self._reelcache_patch_hook = self.patch_embedding.register_forward_hook(_fp32_patch_output)
        else:
            emb.time_embedder.__class__ = WanSinusoidTimestepEmbedder
            self.norm_out.__class__ = SelfForcingBF16HeadNorm
        self.reelcache_precision_policy = policy
        self.reelcache_fp32_regions_tf32 = bool(tf32)
        return self.precision_report()

    def precision_report(self) -> dict:
        return {"policy": self.reelcache_precision_policy, "fp32_regions_tf32": self.reelcache_fp32_regions_tf32,
                "param_dtypes": sorted({str(p.dtype) for p in self.parameters()}),
                "block_class": type(self.blocks[0]).__name__, "cross_attention_class": type(self.blocks[0].attn2).__name__,
                "time_embedder_class": type(self.condition_embedder.time_embedder).__name__,
                "patch_embedding_class": type(self.patch_embedding).__name__,
                "params_rounded_at_use": len(self.reelcache_params_rounded_at_use),
                "params_rounded_at_use_examples": self.reelcache_params_rounded_at_use[:4],
                "head_classes": [type(self.norm_out).__name__, type(self.proj_out).__name__]}
