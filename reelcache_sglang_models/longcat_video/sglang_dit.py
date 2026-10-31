"""SGLang DiT module for LongCat-Video (imports SGLang).

``LongCatVideoTransformer3DModel`` is registered with SGLang's ``ModelRegistry`` under the native
``_class_name`` of ``dit/config.json``. It reuses the pure-torch math of :mod:`.dit_core` and runs
self-/cross-attention through SGLang's ``LocalAttention`` (backend chosen by SGLang: FlashAttention,
FA2 or torch SDPA). Module names equal the native checkpoint keys (identity weight mapping).

Context parallelism: with ``sp_degree = ulysses_degree = N > 1`` (ring 1, tp 1) the module runs native LongCat CP
(:mod:`.context_parallel`) over SGLang's Ulysses process group -- 2-D spatial split ``get_optimal_split(N)``,
Ulysses all-to-all around the attention kernel, final-layer all-gather. Weights stay replicated on every rank.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from sglang.multimodal_gen.runtime.distributed import (
    get_sp_group,
    get_sp_world_size,
    get_tp_world_size,
    model_parallel_is_initialized,
)
from sglang.multimodal_gen.runtime.layers.attention import LocalAttention
from sglang.multimodal_gen.runtime.models.dits.base import CachableDiT
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum

from .context_parallel import LongCatContextParallel
from .dit_core import LongCatVideoBlock, LongCatVideoDiTMixin

SUPPORTED_ATTENTION_BACKENDS = {
    AttentionBackendEnum.FA,
    AttentionBackendEnum.FA2,
    AttentionBackendEnum.TORCH_SDPA,
}


class SGLangAttentionOp(nn.Module):
    """``[B, S, H, D]`` attention through SGLang's ``LocalAttention`` (no mask, not causal)."""

    def __init__(self, kind: str, num_heads: int, head_dim: int, softmax_scale: float) -> None:
        super().__init__()
        self.kind = kind
        self.attn = LocalAttention(num_heads=num_heads, head_size=head_dim, softmax_scale=softmax_scale,
                                   causal=False, supported_attention_backends=SUPPORTED_ATTENTION_BACKENDS,
                                   is_cross_attention=(kind == "cross"))

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return self.attn(q.contiguous(), k.contiguous(), v.contiguous())

    @property
    def backend(self):
        return self.attn.backend


def sglang_attention_factory(kind: str, num_heads: int, head_dim: int, softmax_scale: float) -> nn.Module:
    return SGLangAttentionOp(kind, num_heads, head_dim, softmax_scale)


def sglang_context_parallel() -> LongCatContextParallel | None:
    """Native LongCat CP over SGLang's sequence-parallel (Ulysses) group, or ``None`` for one GPU.

    Only the native exchange is accepted: Ulysses over the whole SP group (``ulysses_degree == sp_degree``,
    ring 1) and no tensor parallelism; anything else raises (no silent single-GPU fallback).
    """
    if not model_parallel_is_initialized():
        return None
    sp = get_sp_world_size()
    tp = get_tp_world_size()
    if tp != 1:
        raise NotImplementedError(f"LongCat-Video port supports tp_size=1 only (got {tp})")
    if sp == 1:
        return None
    group = get_sp_group()
    if group.ring_world_size != 1 or group.ulysses_world_size != sp:
        raise NotImplementedError(
            f"LongCat-Video context parallelism is native Ulysses: need ulysses_degree == sp_degree ({sp}) and "
            f"ring_degree 1, got ulysses {group.ulysses_world_size} / ring {group.ring_world_size}")
    return LongCatContextParallel(group.ulysses_group, label=f"sglang sp group (ulysses {sp})")


def _is_block(name: str, module: nn.Module) -> bool:
    return isinstance(module, LongCatVideoBlock)


class LongCatVideoTransformer3DModel(LongCatVideoDiTMixin, CachableDiT):
    _fsdp_shard_conditions = [_is_block]
    _compile_conditions = [_is_block]
    param_names_mapping = {}
    reverse_param_names_mapping = {}
    lora_param_names_mapping = {}
    _supported_attention_backends = SUPPORTED_ATTENTION_BACKENDS

    def __init__(self, config, hf_config: dict | None = None, quant_config=None, **kwargs) -> None:
        if quant_config is not None:
            raise NotImplementedError("quantized LongCat-Video checkpoints are not supported by this port")
        if kwargs:
            raise TypeError(f"unexpected LongCat-Video model arguments: {sorted(kwargs)}")
        if model_parallel_is_initialized() and get_tp_world_size() != 1:
            raise NotImplementedError("LongCat-Video port supports tp_size=1 only")
        CachableDiT.__init__(self, config=config, hf_config=hf_config or {})
        arch = config.arch_config.to_longcat_arch()
        self._build_longcat_modules(arch, sglang_attention_factory)
        self._longcat_cp_pending = True  # resolved from SGLang's groups at first use (after distributed init)
        self.hidden_size = arch.hidden_size
        self.num_attention_heads = arch.num_heads
        self.num_channels_latents = arch.in_channels
        self.__post_init__()

    def context_parallel(self) -> LongCatContextParallel | None:
        if self._longcat_cp_pending:
            cp = sglang_context_parallel()
            self._longcat_cp_pending = False
            if cp is not None:
                LongCatVideoDiTMixin.set_context_parallel(self, cp)
        return self.longcat_cp

    def set_context_parallel(self, cp: LongCatContextParallel | None) -> None:
        """Explicit override (e.g. a single-GPU reference forward on one rank of a CP group)."""
        self._longcat_cp_pending = False
        LongCatVideoDiTMixin.set_context_parallel(self, cp)

    def attention_backends(self) -> dict:
        backends = {}
        for module in self.modules():
            if isinstance(module, SGLangAttentionOp):
                backends.setdefault(module.kind, set()).add(str(module.backend))
        return {kind: sorted(values) for kind, values in backends.items()}

    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor | None = None,
                timestep: torch.Tensor | None = None, encoder_hidden_states_image=None, guidance=None, *,
                encoder_attention_mask: torch.Tensor | None = None, num_cond_latents: int = 0,
                kv_cache_layers=None, return_kv: bool = False, skip_crs_attn: bool = False, kv_cache_ready=None,
                **kwargs):
        if encoder_hidden_states_image is not None or guidance is not None or kwargs:
            raise TypeError("LongCat-Video takes no image embeddings, guidance embedding or extra kwargs: "
                            f"{sorted(kwargs)}")
        if timestep is None:
            raise ValueError("timestep is required")
        return self.longcat_forward(hidden_states, timestep, encoder_hidden_states, encoder_attention_mask,
                                    num_cond_latents=num_cond_latents, kv_cache_layers=kv_cache_layers,
                                    return_kv=return_kv, skip_crs_attn=skip_crs_attn, kv_cache_ready=kv_cache_ready)


EntryClass = LongCatVideoTransformer3DModel
