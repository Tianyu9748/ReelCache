"""LongCat-Video LoRA branches (``cfg_step_lora``) applied exactly like the native runtime.

Native ``LongCatVideoTransformer3DModel.enable_loras`` replaces each targeted ``nn.Linear.forward`` with

    org = linear(x); lx = lora_up(lora_down(x.to(down.dtype)))
    out = org + (0 + lx.to(x.dtype) * multiplier * alpha_scale)

where ``lora_up`` may be split into ``n`` parallel blocks (qkv: 3, kv_linear: 2) applied to equal chunks
of the down projection. The branches are not merged into the base weights (merging changes BF16
rounding). LoRA tensors are cast to the model dtype/device, as ``enable_loras`` does (``alpha_scale``,
0.5 in the shipped file, included).
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

LORA_PREFIX = "lora___lorahyphen___"
LORA_HYPHEN = "___lorahyphen___"


class LoRALinear(nn.Linear):
    """``nn.Linear`` with optional additive LoRA branches (state-dict compatible with ``nn.Linear``)."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None) -> None:
        super().__init__(in_features, out_features, bias=bias, device=device, dtype=dtype)
        self.lora_branches = nn.ModuleList()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        out = F.linear(x, self.weight, self.bias)
        if len(self.lora_branches) == 0:
            return out
        total = 0
        for branch in self.lora_branches:
            if branch.enabled:
                total += branch(x, x.dtype)
        return out + total


class LoRABranch(nn.Module):
    def __init__(self, in_features: int, rank_total: int, up_out: list[int], alpha_scale: float | torch.Tensor,
                 multiplier: float = 1.0, name: str = "") -> None:
        super().__init__()
        if not up_out or rank_total % len(up_out):
            raise ValueError(f"LoRA {name}: rank {rank_total} not divisible into {len(up_out)} up blocks")
        rank = rank_total // len(up_out)
        self.name = name
        self.lora_down = nn.Linear(in_features, rank_total, bias=False)
        self.lora_ups = nn.ModuleList(nn.Linear(rank, out, bias=False) for out in up_out)
        self.register_buffer("alpha_scale", torch.as_tensor(alpha_scale, dtype=torch.float32).reshape(()))
        self.multiplier = float(multiplier)
        self.enabled = True

    @property
    def out_features(self) -> int:
        return sum(up.out_features for up in self.lora_ups)

    def forward(self, x: torch.Tensor, weight_dtype: torch.dtype) -> torch.Tensor:
        lx = self.lora_down(x.to(self.lora_down.weight.dtype))
        if len(self.lora_ups) == 1:
            lx = self.lora_ups[0](lx)
        else:
            chunks = torch.chunk(lx, len(self.lora_ups), dim=-1)
            lx = torch.cat([self.lora_ups[i](chunks[i]) for i in range(len(self.lora_ups))], dim=-1)
        return lx.to(weight_dtype) * self.multiplier * self.alpha_scale


@dataclass(frozen=True)
class LoRAEntry:
    module_name: str
    down: torch.Tensor
    ups: tuple[torch.Tensor, ...]
    alpha_scale: torch.Tensor


def lora_module_name(lora_name: str) -> str:
    return lora_name.replace(LORA_PREFIX, "").replace(LORA_HYPHEN, ".")


def parse_lora_state_dict(state_dict: dict[str, torch.Tensor]) -> "OrderedDict[str, LoRAEntry]":
    """Group a native LongCat LoRA state dict by target module (strict: every key must be consumed)."""
    names = sorted({k[: -len(".lora_down.weight")] for k in state_dict if k.endswith(".lora_down.weight")})
    consumed: set[str] = set()
    entries: "OrderedDict[str, LoRAEntry]" = OrderedDict()
    for lora_name in names:
        down_key = f"{lora_name}.lora_down.weight"
        alpha_key = f"{lora_name}.alpha_scale"
        block_prefix = f"{lora_name}.lora_up.blocks."
        block_keys = sorted((k for k in state_dict if k.startswith(block_prefix)),
                            key=lambda k: int(k[len(block_prefix):].split(".")[0]))
        if block_keys:
            expect = [f"{block_prefix}{i}.weight" for i in range(len(block_keys))]
            if block_keys != expect:
                raise ValueError(f"LoRA {lora_name}: unexpected up-block keys {block_keys}")
            ups = tuple(state_dict[k] for k in block_keys)
        else:
            up_key = f"{lora_name}.lora_up.weight"
            if up_key not in state_dict:
                raise ValueError(f"LoRA {lora_name}: missing {up_key}")
            block_keys = [up_key]
            ups = (state_dict[up_key],)
        if alpha_key not in state_dict:
            raise ValueError(f"LoRA {lora_name}: missing {alpha_key}")
        down = state_dict[down_key]
        if down.ndim != 2 or any(u.ndim != 2 or u.shape[1] * len(ups) != down.shape[0] for u in ups):
            raise ValueError(f"LoRA {lora_name}: inconsistent shapes down {tuple(down.shape)} "
                             f"ups {[tuple(u.shape) for u in ups]}")
        consumed.update((down_key, alpha_key, *block_keys))
        module = lora_module_name(lora_name)
        if module in entries:
            raise ValueError(f"duplicate LoRA target {module}")
        entries[module] = LoRAEntry(module, down, ups, state_dict[alpha_key])
    leftover = sorted(set(state_dict) - consumed)
    if leftover:
        raise ValueError(f"unconsumed LoRA keys: {leftover[:8]}{'...' if len(leftover) > 8 else ''}")
    return entries


def attach_lora(model: nn.Module, state_dict: dict[str, torch.Tensor], *, multiplier: float = 1.0,
                dtype: torch.dtype | None = None, device: torch.device | str | None = None) -> dict:
    """Attach native-equivalent LoRA branches to the ``LoRALinear`` modules of ``model`` (strict)."""
    entries = parse_lora_state_dict(state_dict)
    modules = dict(model.named_modules())
    attached = []
    for name, entry in entries.items():
        target = modules.get(name)
        if not isinstance(target, LoRALinear):
            raise ValueError(f"LoRA target {name!r} is not a LoRALinear in {type(model).__name__}")
        if entry.down.shape[1] != target.in_features or sum(u.shape[0] for u in entry.ups) != target.out_features:
            raise ValueError(f"LoRA {name}: shapes do not match {target.in_features}->{target.out_features}")
        if len(target.lora_branches):
            raise ValueError(f"LoRA target {name!r} already has a branch attached")
        w_dtype = dtype or target.weight.dtype
        w_device = device or target.weight.device
        branch = LoRABranch(entry.down.shape[1], entry.down.shape[0], [u.shape[0] for u in entry.ups],
                            entry.alpha_scale, multiplier=multiplier, name=name)
        with torch.no_grad():
            branch.lora_down.weight.copy_(entry.down)
            for up, w in zip(branch.lora_ups, entry.ups):
                up.weight.copy_(w)
        branch.to(device=w_device, dtype=w_dtype)
        target.lora_branches.append(branch)
        attached.append(name)
    return {"modules": len(attached), "targets": attached, "multiplier": float(multiplier),
            "alpha_scales": sorted({float(e.alpha_scale) for e in entries.values()}),
            "dtype": str(dtype) if dtype is not None else "model"}


def set_lora_enabled(model: nn.Module, enabled: bool) -> int:
    count = 0
    for module in model.modules():
        if isinstance(module, LoRABranch):
            module.enabled = bool(enabled)
            count += 1
    return count
