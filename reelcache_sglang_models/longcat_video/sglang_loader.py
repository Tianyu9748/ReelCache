"""Pipeline-local SGLang component loader for the LongCat-Video DiT (imports SGLang).

Loads the base DiT through SGLang's ``TransformerLoader`` (``dit/`` safetensors, BF16, identity key
mapping), then checks that every checkpoint tensor was consumed, and attaches the hash-bound
``cfg_step_lora`` branches with native semantics (unmerged, multiplier 1.0). It is registered only
as this pipeline's ``component_loaders["transformer"]`` (``component_names`` is empty so SGLang's
global loader table is not touched).
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

from sglang.multimodal_gen.runtime.loader.component_loaders.transformer_loader import TransformerLoader

from .lora import attach_lora
from .spec import DIT_CLASS_NAME
from .weights import LORA_RELPATH, MODEL_FINGERPRINT, verify_checkpoint

_VERIFIED: dict[str, dict] = {}
_LOCK = threading.Lock()


def record_verified_checkpoint(root: str | Path, report: dict) -> None:
    with _LOCK:
        _VERIFIED[str(Path(root).resolve())] = dict(report)


def verified_checkpoint(root: str | Path) -> dict | None:
    with _LOCK:
        report = _VERIFIED.get(str(Path(root).resolve()))
    return None if report is None else dict(report)


class LongCatVideoTransformerLoader(TransformerLoader):
    component_names: list[str] = []  # pipeline-local only; never replace the global "transformer" loader

    def load_customized(self, component_model_path: str, server_args, component_name: str,
                        cpu_offload_flag: bool = False):
        dit_dir = Path(component_model_path)
        if dit_dir.name != "dit":
            raise ValueError(f"LongCat-Video DiT must load from <checkpoint>/dit, got {dit_dir}")
        root = dit_dir.parent
        report = verified_checkpoint(root)
        if report is None or not {"dit", "distill_lora_cfg_step"} <= set(report.get("roles", ())):
            report = verify_checkpoint(root, roles=("dit", "distill_lora_cfg_step"))
            record_verified_checkpoint(root, report)
        t0 = time.perf_counter()
        model = super().load_customized(component_model_path, server_args, component_name,
                                        cpu_offload_flag=cpu_offload_flag)
        if type(model).__name__ != DIT_CLASS_NAME:
            raise TypeError(f"expected {DIT_CLASS_NAME}, the registry resolved {type(model).__name__}")
        index = json.loads((dit_dir / "diffusion_pytorch_model.safetensors.index.json").read_text())["weight_map"]
        params = {name for name, _ in model.named_parameters()}
        missing, unexpected = sorted(params - set(index)), sorted(set(index) - params)
        if missing or unexpected:
            raise RuntimeError(f"LongCat-Video DiT keys differ from the checkpoint: missing={missing[:5]} "
                               f"unexpected={unexpected[:5]} ({len(missing)}/{len(unexpected)})")
        from safetensors.torch import load_file

        lora_path = root / LORA_RELPATH
        first = next(model.parameters())
        lora_report = attach_lora(model, load_file(str(lora_path), device="cpu"), multiplier=1.0,
                                  dtype=first.dtype, device=first.device)
        model.longcat_load_report = {
            "checkpoint_root": str(root.resolve()), "binding": report, "model_fingerprint": MODEL_FINGERPRINT,
            "checkpoint_tensors": len(index), "lora": {k: v for k, v in lora_report.items() if k != "targets"},
            "lora_path": str(lora_path), "param_dtype": str(first.dtype), "load_seconds": time.perf_counter() - t0,
        }
        return model
