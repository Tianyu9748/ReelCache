"""Clean-chunk capture transaction: normalized pre-RoPE K and V of every layer and CFG stream.

The SGLang seam hands the attention module's own ``key``/``value`` tensors to
the bridge during a clean forward only (``[1, T, H, D]``, native dtype,
*before* ``_apply_rotary_emb``): the clean context forward of a chunk
(``clean_commit``) or a clean recache forward over a window of earlier frames
(``kv_recache``). The builder keeps references (no copies), checks that
nothing mutated them in place before the commit reads them, and verifies that
the RoPE tables the attention used are exactly the engine's tables at the
recorded position, so a restore can re-rotate at the same position. After the
commit submitted them, the archive pipeline holds the references until its
last read.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping

import torch

from ...errors import ConfigurationError, IntegrityError, StateError
from ...types import MatrixKey


class CleanChunkBuilder:
    """Mutable capture/staging transaction of one chunk (process-local, not serializable)."""

    def __init__(self, chunk: Any, *, streams: tuple[str, ...], num_layers: int, num_heads: int, head_dim: int,
                 dtype: torch.dtype | Mapping[str, torch.dtype],
                 rope: Callable[[float, int], tuple[torch.Tensor, torch.Tensor]] | None, kind: str = "clean_commit"):
        if kind not in ("clean_commit", "kv_recache"):
            raise ConfigurationError(f"unknown capture kind {kind!r}")
        self.chunk = chunk
        self.kind = kind
        self.streams = tuple(streams)
        self.num_layers, self.num_heads, self.head_dim = num_layers, num_heads, head_dim
        # Captured dtypes per target: the cache dtype, or e.g. FP32 pre-RoPE keys with BF16 values (RAVEN).
        self.dtypes = {"K": dtype, "V": dtype} if isinstance(dtype, torch.dtype) else dict(dtype)
        if set(self.dtypes) != {"K", "V"}:
            raise ConfigurationError("capture dtypes map exactly K and V")
        self._rope = rope
        self._rope_checked: set[str] = set()
        self._parts: dict[str, dict[int, tuple[torch.Tensor, torch.Tensor, int, int]]] = {s: {} for s in self.streams}
        self._ended: set[str] = set()
        self.rope_check = {"verified_streams": [], "bitwise_equal": None}

    def add(self, stream: str, layer: int, key: torch.Tensor, value: torch.Tensor,
            freqs_cis: tuple[torch.Tensor, torch.Tensor]) -> None:
        if stream not in self._parts or stream in self._ended:
            raise StateError(f"capture for stream {stream!r} is not open")
        if type(layer) is not int or not 0 <= layer < self.num_layers:
            raise ConfigurationError(f"captured layer index {layer!r} is outside the model")
        if layer in self._parts[stream]:
            raise IntegrityError(f"layer {layer} of stream {stream!r} was captured twice in one clean forward")
        expected = (1, self.chunk.num_tokens, self.num_heads, self.head_dim)
        for name, tensor, dtype in (("key", key, self.dtypes["K"]), ("value", value, self.dtypes["V"])):
            if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != expected or tensor.dtype != dtype:
                raise IntegrityError(
                    f"captured {name} of layer {layer} is {getattr(tensor, 'shape', None)} "
                    f"{getattr(tensor, 'dtype', None)}, expected {expected} {dtype}"
                )
        if stream not in self._rope_checked and self._rope is not None:
            self._check_rope(stream, freqs_cis)
        self._parts[stream][layer] = (key, value, self._version(key), self._version(value))

    @staticmethod
    def _version(tensor: torch.Tensor) -> int | None:
        """The in-place mutation counter, or None for inference tensors (SGLang denoises under
        ``torch.inference_mode``, whose tensors track no version: mutation cannot be detected there; the GPU
        capture-correctness test checks the archived K/V against the native cache bitwise instead)."""
        return None if tensor.is_inference() else tensor._version

    @property
    def mutation_check(self) -> str:
        versions = [version for parts in self._parts.values() for _, _, *pair in parts.values() for version in pair]
        if not versions:
            return "none"
        return "version_counter" if all(version is not None for version in versions) else "unavailable_inference_tensors"

    def _check_rope(self, stream: str, freqs_cis: tuple[torch.Tensor, torch.Tensor]) -> None:
        cos, sin = self._rope(self.chunk.rope_start_frame, self.chunk.num_frames)
        used_cos, used_sin = freqs_cis
        if (tuple(cos.shape) != tuple(used_cos.shape) or tuple(sin.shape) != tuple(used_sin.shape)
                or not torch.equal(cos.to(used_cos.device), used_cos) or not torch.equal(sin.to(used_sin.device), used_sin)):
            raise IntegrityError(
                f"the RoPE tables of chunk {self.chunk.index} differ from the engine's tables at the recorded "
                f"position {self.chunk.rope_start_frame}; restored keys could not be re-rotated exactly"
            )
        self._rope_checked.add(stream)
        self.rope_check["verified_streams"].append(stream)
        self.rope_check["bitwise_equal"] = True

    def end_stream(self, stream: str) -> None:
        parts = self._parts.get(stream)
        if parts is None:
            raise StateError(f"unknown capture stream {stream!r}")
        if sorted(parts) != list(range(self.num_layers)):
            missing = sorted(set(range(self.num_layers)) - set(parts))
            raise IntegrityError(f"clean forward of stream {stream!r} did not capture layers {missing}")
        self._ended.add(stream)

    @property
    def complete(self) -> bool:
        return self._ended == set(self.streams)

    def matrices(self, stream: str) -> dict[MatrixKey, torch.Tensor]:
        """Canonical ``[N, C]`` views (``C = H * D``) of a completed stream; checks for in-place mutation."""
        if stream not in self._ended:
            raise StateError(f"stream {stream!r} has not completed its clean forward")
        result = {}
        for layer, (key, value, key_version, value_version) in sorted(self._parts[stream].items()):
            if (key_version is not None and key._version != key_version) or (
                    value_version is not None and value._version != value_version):
                raise IntegrityError(f"captured K/V of layer {layer} were modified in place after capture")
            channels = self.num_heads * self.head_dim
            result[MatrixKey(layer, "K")] = key.reshape(self.chunk.num_tokens, channels)
            result[MatrixKey(layer, "V")] = value.reshape(self.chunk.num_tokens, channels)
        return result

    def source_bytes(self) -> int:
        return sum(t.numel() * t.element_size() for parts in self._parts.values() for k, v, _, _ in parts.values()
                   for t in (k, v))

    def release(self) -> None:
        self._parts = {stream: {} for stream in self.streams}
        self._ended.clear()


__all__ = ["CleanChunkBuilder"]
