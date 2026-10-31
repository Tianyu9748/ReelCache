"""Native sampler semantics of the causal-Wan profiles (torch only; no SGLang import).

* ``FlowMatchScheduler`` reproduces the DiffSynth scheduler that CausVid, Self-Forcing and LongLive 1.0 use
  (``shift``, ``sigma_min=0``, ``extra_one_step=True``, ``set_timesteps(1000, training=True)``): same float32
  ``sigmas``/``timesteps`` tensors, the same nearest-timestep lookup and ``add_noise``. SGLang's causal DMD
  stage reads ``sigmas``/``timesteps`` (``pred_noise_to_pred_video``) and calls ``add_noise``, so installing
  this object as the pipeline scheduler gives the original repositories' arithmetic.
* RAVEN helpers: integer trailing timesteps, the v-lerp consistency step in float32 and ``yield_seed`` (the
  LCG RAVEN uses to fork the step-noise generator).
"""

from __future__ import annotations

import numpy as np
import torch

from .profiles import ModelProfile, SamplerSpec


class FlowMatchScheduler:
    """DiffSynth ``FlowMatchScheduler`` as configured by CausVid/LongLive (inference subset)."""

    order = 1

    def __init__(self, shift: float, num_train_timesteps: int = 1000, sigma_max: float = 1.0,
                 sigma_min: float = 0.0, extra_one_step: bool = True):
        self.shift = float(shift)
        self.num_train_timesteps = int(num_train_timesteps)
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.extra_one_step = extra_one_step
        self.config = {"shift": self.shift, "num_train_timesteps": self.num_train_timesteps,
                       "sigma_min": sigma_min, "sigma_max": sigma_max, "extra_one_step": extra_one_step,
                       "_class_name": "ReelCacheNativeFlowMatchScheduler"}
        self.set_timesteps(1000)

    def set_timesteps(self, num_inference_steps: int = 1000, denoising_strength: float = 1.0, **_):
        sigma_start = self.sigma_min + (self.sigma_max - self.sigma_min) * denoising_strength
        if self.extra_one_step:
            sigmas = torch.linspace(sigma_start, self.sigma_min, num_inference_steps + 1)[:-1]
        else:
            sigmas = torch.linspace(sigma_start, self.sigma_min, num_inference_steps)
        self.sigmas = self.shift * sigmas / (1 + (self.shift - 1) * sigmas)
        self.timesteps = self.sigmas * self.num_train_timesteps

    def _sigma(self, timestep: torch.Tensor, device) -> torch.Tensor:
        self.sigmas = self.sigmas.to(device)
        self.timesteps = self.timesteps.to(device)
        timestep_id = torch.argmin((self.timesteps.unsqueeze(0) - timestep.to(device).unsqueeze(1)).abs(), dim=1)
        return self.sigmas[timestep_id].reshape(-1, 1, 1, 1)

    def add_noise(self, original_samples: torch.Tensor, noise: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        sigma = self._sigma(timestep, noise.device)
        sample = (1 - sigma) * original_samples + sigma * noise
        return sample.type_as(noise)

    def state_dict(self) -> dict:
        return dict(self.config)


def resolve_dmd_timesteps(spec: SamplerSpec, scheduler: FlowMatchScheduler) -> torch.Tensor:
    """The per-step timesteps exactly as the native pipelines build them (and SGLang's causal DMD stage)."""
    steps = torch.tensor(spec.denoising_steps, dtype=torch.long)
    if not spec.warp_denoising_step:
        return steps
    table = torch.cat((scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32)))
    return table[spec.num_train_timesteps - steps]


def scheduler_for(profile: ModelProfile) -> FlowMatchScheduler:
    s = profile.sampler
    shift = s.timestep_shift if s.kind == "dmd_flow_x0" else 1.0
    return FlowMatchScheduler(shift=shift, num_train_timesteps=s.num_train_timesteps)


# ------------------------------------------------------------------------------------------------ RAVEN
def yield_seed(seed: int, a: int = 1103515245, c: int = 12345, m: int = 2 ** 31) -> int:
    """RAVEN ``project.utils.random.yield_seed``; ``RandomState(seed).fork(0)`` uses ``yield_seed(seed)``."""
    return (a * int(seed) + c) % m


def raven_trailing_timesteps(num_steps: int, shift: float, T: int = 1000) -> list[int]:
    """RAVEN ``TrailingSamplingTimesteps`` with an integer ``T``: round(T * shift*t/(1+(shift-1)*t))."""
    t = np.arange(1.0, 0, -1.0 / num_steps)
    t = shift * t / (1 + (shift - 1) * t)
    return [int(x) for x in (t * T).round().astype(np.int64)]


def raven_x0(pred: torch.Tensor, x_t: torch.Tensor, t: int, T: int = 1000) -> torch.Tensor:
    """``LinearInterpolationSchedule.convert_from_pred`` for v_lerp: x0 = (x_t - B_t v) / (A_t + B_t)."""
    tt = torch.tensor([t], device=x_t.device).reshape([1] * x_t.ndim)
    a = 1 - (tt / T)
    b = tt / T
    return (x_t - b * pred) / (a + b)


def raven_forward(x0: torch.Tensor, noise: torch.Tensor, s: int, T: int = 1000) -> torch.Tensor:
    """``LinearInterpolationSchedule.forward``: x_s = A_s x0 + B_s noise."""
    ss = torch.tensor([s], device=x0.device).reshape([1] * x0.ndim)
    return (1 - (ss / T)) * x0 + (ss / T) * noise
