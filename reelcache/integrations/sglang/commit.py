"""Commit inputs of the bridge: lineage-keyed matrices and contexts of one captured range.

Every committed range carries all layers' normalized pre-RoPE K and direct V of
every CFG stream, keyed ``(MatrixKey, stream)`` in canonical ``[N, C]`` views
of the captured tensors (no copies): the Step 6 archive pipeline reads them on
its fitter stream and drops them after its last read; the RAW reference copies
them to the host synchronously.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch

from ...errors import IntegrityError
from ...types import MatrixKey, ModelIdentity, TensorContext


def matrix_contexts(model: ModelIdentity, *, token_start: int, frames: int, grid: tuple[int, int],
                    streams: Sequence[str]) -> dict[tuple[MatrixKey, str], TensorContext]:
    contexts = {}
    for layer in range(model.layers):
        for target in ("K", "V"):
            key = MatrixKey(layer, target)
            context = TensorContext(model, key, token_start, frames, grid[0], grid[1],
                                    "pre_rope" if target == "K" else "none")
            for stream in streams:
                contexts[(key, stream)] = context
    return contexts


def stream_matrices(builder: Any, streams: Sequence[str]) -> dict[tuple[MatrixKey, str], torch.Tensor]:
    """``(MatrixKey, stream) -> [N, C]`` views of a completed capture (checks in-place mutation)."""
    matrices = {}
    for stream in streams:
        for key, value in builder.matrices(stream).items():
            matrices[(key, stream)] = value
    return matrices


def slice_matrices(matrices: Mapping[tuple[MatrixKey, str], torch.Tensor], offset: int,
                   num_tokens: int) -> dict[tuple[MatrixKey, str], torch.Tensor]:
    """Rows ``[offset, offset + num_tokens)`` of every matrix (a written range of a recache window)."""
    result = {}
    for entry, value in matrices.items():
        if offset < 0 or offset + num_tokens > value.shape[0]:
            raise IntegrityError(f"range [{offset}, {offset + num_tokens}) is outside the captured window")
        result[entry] = value[offset:offset + num_tokens]
    return result


__all__ = ["matrix_contexts", "slice_matrices", "stream_matrices"]
