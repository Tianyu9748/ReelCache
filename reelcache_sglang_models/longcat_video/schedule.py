"""Distilled LongCat-Video continuation schedule (pure torch/numpy).

Replicates ``LongCatVideoPipeline.get_timesteps_sigmas(steps, use_distill=True)`` followed by the
flow-matching Euler scheduler's ``set_timesteps(steps, sigmas=...)`` with the checkpoint's
``scheduler_config.json`` (``shift = 12``, ``use_dynamic_shifting = False``), and its ``step``.
The SGLang stage uses SGLang's own ``FlowMatchEulerDiscreteScheduler`` and checks its schedule
against :func:`continuation_schedule` bitwise before denoising.
"""

from __future__ import annotations

import numpy as np
import torch


def distill_sigmas(num_inference_steps: int = 16, num_train_timesteps: int = 1000,
                   num_distill_sample_steps: int = 50) -> torch.Tensor:
    """``get_timesteps_sigmas(sampling_steps, use_distill=True)`` (FP32, before the flow shift)."""
    if not 0 < num_inference_steps <= num_distill_sample_steps:
        raise ValueError("num_inference_steps must be in [1, num_distill_sample_steps]")
    distill_indices = torch.arange(1, num_distill_sample_steps + 1, dtype=torch.float32)
    distill_indices = (distill_indices * (num_train_timesteps // num_distill_sample_steps)).round().long()
    inference_indices = np.linspace(0, num_distill_sample_steps, num=num_inference_steps, endpoint=False)
    inference_indices = np.floor(inference_indices).astype(np.int64)
    sigmas = torch.flip(distill_indices, [0])[inference_indices].float() / num_train_timesteps
    return sigmas.to(torch.float32)


def shift_sigmas(sigmas, shift: float = 12.0) -> np.ndarray:
    """``FlowMatchEulerDiscreteScheduler.set_timesteps(sigmas=...)`` shift (static, FP32 numpy)."""
    s = np.array(sigmas).astype(np.float32)
    return shift * s / (1 + (shift - 1) * s)


def continuation_schedule(num_inference_steps: int = 16, shift: float = 12.0, num_train_timesteps: int = 1000,
                          num_distill_sample_steps: int = 50, device=None) -> tuple[torch.Tensor, torch.Tensor]:
    """``(timesteps [steps], sigmas [steps + 1])`` FP32, terminal sigma 0 appended (native scheduler state)."""
    shifted = shift_sigmas(distill_sigmas(num_inference_steps, num_train_timesteps, num_distill_sample_steps), shift)
    sigmas = torch.from_numpy(shifted).to(dtype=torch.float32, device=device)
    timesteps = sigmas * num_train_timesteps
    sigmas = torch.cat([sigmas, torch.zeros(1, device=sigmas.device)])
    return timesteps, sigmas


def euler_step(model_output: torch.Tensor, sample: torch.Tensor, sigma: torch.Tensor,
               sigma_next: torch.Tensor) -> torch.Tensor:
    """Deterministic flow-matching Euler step (scheduler ``step`` without per-token timesteps)."""
    upcast = sample.to(torch.float32)
    dt = sigma_next - sigma
    return (upcast + dt * model_output).to(model_output.dtype)


def dit_timestep(t: torch.Tensor, batch: int, frames: int, dtype: torch.dtype) -> torch.Tensor:
    """Native ``generate_vc`` timestep tensor: ``t.expand(B).to(dit_dtype)`` repeated per latent frame."""
    return t.expand(batch).to(dtype).unsqueeze(-1).repeat(1, frames)
