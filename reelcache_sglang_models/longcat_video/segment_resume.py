"""Segment-boundary resume state of the LongCat-Video continuation (pure torch, no SGLang).

A continuation segment is reproduced exactly from its boundary: the previous segment's last 13 decoded frames, the
segment index and the request generator's state at the segment start (CUDA Philox: seed + offset), from which the
segment's noise and VAE posterior sample are drawn in the native order. The stage records the state in every segment
record (``generator_state``) and a request may resume from it (``start_segment_index`` + ``generator_state``).
"""

from __future__ import annotations

import hashlib

import torch


def generator_state_hex(generator: torch.Generator) -> str:
    """The generator's full state as a hex string (read-only: the generator is not advanced)."""
    return bytes(generator.get_state().numpy().tobytes()).hex()


def generator_state_bytes(state: str) -> bytes:
    try:
        return bytes.fromhex(state)
    except (TypeError, ValueError) as exc:
        raise ValueError("generator_state must be the hex string recorded in a segment record") from exc


def set_generator_state_hex(generator: torch.Generator, state: str) -> None:
    """Restore a state recorded by :func:`generator_state_hex` (same generator type; size-checked)."""
    raw = generator_state_bytes(state)
    expected = generator.get_state().numel()
    if len(raw) != expected:
        raise ValueError(f"generator_state has {len(raw)} bytes; this generator's state has {expected}")
    generator.set_state(torch.tensor(list(raw), dtype=torch.uint8))


def generator_state_sha256(state: str) -> str:
    return hashlib.sha256(generator_state_bytes(state)).hexdigest()
