"""Conditioning-frame preprocessing, latent (de)normalization and native output conversion.

Bitwise replicas of the native pixel/latent conventions of ``LongCatVideoPipeline``:

* ``video_processor.preprocess_video(frames, height, width)`` (diffusers ``VideoProcessor`` with PIL
  input): Lanczos resize only when the size differs, ``uint8 / 255`` in NumPy FP32, then
  ``2 * x - 1`` in torch FP32, layout ``[B, C, F, H, W]``;
* ``normalize_latents`` / ``denormalize_latents`` with the Wan VAE mean/std, computed in the latents'
  dtype (BF16 in the pipeline; ``1 / std`` is rounded to that dtype first);
* decode tail: ``(x * 0.5 + 0.5).clamp(0, 1)`` in the VAE output dtype (BF16), FP32 numpy, and
  ``(x * 255).astype(uint8)`` truncation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .spec import VAE_LATENTS_MEAN, VAE_LATENTS_STD


def load_condition_frames(source) -> np.ndarray:
    """Load ``[F, H, W, 3]`` uint8 frames from an array/tensor, ``.npz`` (key ``frames``), ``.npy`` or a
    directory of ``.png`` frames. Lossy video containers are rejected (decode them explicitly first)."""
    if isinstance(source, torch.Tensor):
        frames = source.detach().cpu().numpy()
    elif isinstance(source, np.ndarray):
        frames = source
    else:
        path = Path(source)
        if path.is_dir():
            from PIL import Image

            files = sorted(path.glob("*.png"))
            if not files:
                raise ValueError(f"no .png frames in {path}")
            frames = np.stack([np.array(Image.open(f).convert("RGB")) for f in files])
        elif path.suffix == ".npz":
            with np.load(path, allow_pickle=False) as z:
                if "frames" not in z:
                    raise ValueError(f"{path} has no 'frames' array")
                frames = z["frames"].copy()
        elif path.suffix == ".npy":
            frames = np.load(path, allow_pickle=False)
        else:
            raise ValueError(f"unsupported condition source {path}; use .npz/.npy/png-dir/uint8 array")
    frames = np.ascontiguousarray(frames)
    if frames.dtype != np.uint8 or frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"condition frames must be uint8 [F, H, W, 3], got {frames.dtype} {frames.shape}")
    return frames


def frames_to_vae_input(frames: np.ndarray, height: int, width: int) -> torch.Tensor:
    """``VideoProcessor.preprocess_video(list_of_PIL, height, width)`` -> FP32 ``[1, 3, F, H, W]`` in [-1, 1]."""
    frames = load_condition_frames(frames)
    if frames.shape[1:3] != (height, width):
        from PIL import Image

        resample = getattr(Image, "Resampling", Image).LANCZOS
        frames = np.stack([np.array(Image.fromarray(f).resize((width, height), resample=resample)) for f in frames])
    images = np.stack([f.astype(np.float32) / 255.0 for f in frames], axis=0)
    pt = torch.from_numpy(images.transpose(0, 3, 1, 2))
    pt = 2.0 * pt - 1.0
    return pt.unsqueeze(0).permute(0, 2, 1, 3, 4)


def _latent_stats(latents: torch.Tensor):
    z = len(VAE_LATENTS_MEAN)
    mean = torch.tensor(VAE_LATENTS_MEAN).view(1, z, 1, 1, 1).to(latents.device, latents.dtype)
    inv_std = 1.0 / torch.tensor(VAE_LATENTS_STD).view(1, z, 1, 1, 1).to(latents.device, latents.dtype)
    return mean, inv_std


def normalize_latents(latents: torch.Tensor) -> torch.Tensor:
    mean, inv_std = _latent_stats(latents)
    return (latents - mean) * inv_std


def denormalize_latents(latents: torch.Tensor) -> torch.Tensor:
    mean, inv_std = _latent_stats(latents)
    return latents / inv_std + mean


def postprocess_decoded(video: torch.Tensor) -> torch.Tensor:
    """VAE output ``[B, 3, F, H, W]`` (VAE dtype) -> FP32 ``[B, F, H, W, 3]`` in [0, 1] (native ``postprocess``)."""
    out = (video * 0.5 + 0.5).clamp(0, 1)
    return out.permute(0, 2, 3, 4, 1).float()


def to_uint8_native(frames01: torch.Tensor | np.ndarray) -> np.ndarray:
    """Native ``(video * 255).astype(np.uint8)`` (truncation) of FP32 [0, 1] frames."""
    arr = frames01.detach().cpu().numpy() if isinstance(frames01, torch.Tensor) else np.asarray(frames01)
    if arr.dtype != np.float32:
        raise TypeError("native uint8 conversion expects FP32 frames")
    return (arr * 255).astype(np.uint8)
