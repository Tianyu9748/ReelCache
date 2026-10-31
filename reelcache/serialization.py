"""Portable, bit-preserving tensor state; never executes pickle payloads."""

from __future__ import annotations

import hashlib
import json
import math
import struct
import sys
from typing import Any, Mapping, Sequence

import torch

from .errors import CapacityError, ConfigurationError, IntegrityError, UnsupportedError
from .types import json_copy


_MAGIC = b"REELCACHE_TENSORS\x00\x01"
_MAX_HEADER_BYTES = 16 * 1024 * 1024
_DTYPES = {
    name: getattr(torch, name)
    for name in (
        "bool", "uint8", "int8", "int16", "int32", "int64",
        "float16", "bfloat16", "float32", "float64", "complex64", "complex128",
        "uint16", "uint32", "uint64",
    )
    if hasattr(torch, name)
}


def canonical_json(value: Any) -> bytes:
    value = json_copy(value)
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def _unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = dict(pairs)
    if len(result) != len(pairs):
        seen: set[str] = set()
        for key, _ in pairs:
            if key in seen:
                raise IntegrityError(f"duplicate JSON key: {key}")
            seen.add(key)
    return result


def _finite_float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value):
        raise ValueError("non-finite JSON number")
    return value


def _reject_constant(name: str) -> Any:
    raise ValueError(f"non-finite JSON constant: {name}")


# Documents nested deeper than this are validated by the former full ``json_copy`` pass (see ``parse_json``).
_SHALLOW_JSON_DEPTH = 64


class _DeepDocument(Exception):
    pass


def _shallow(value: dict | list, budget: int) -> None:
    """Raise _DeepDocument when containers nest deeper than ``budget`` levels below ``value``."""
    if not budget:
        raise _DeepDocument
    for child in (value.values() if type(value) is dict else value):
        if type(child) is dict or type(child) is list:
            _shallow(child, budget - 1)


def parse_json(data: bytes) -> Any:
    """Parse stored JSON: UTF-8, unique object keys, finite numbers only (no NaN/Infinity constants, no number
    literal outside the float range); any violation raises IntegrityError.

    One ``json.loads`` whose hooks reject duplicate keys, non-finite numbers and constants while parsing (SGLang
    Step 7: ~6 ms -> ~1 ms for a CausVid node pack manifest). It accepts and rejects exactly what the former parse
    followed by a full ``json_copy`` validation did and returns equal values: ``json_copy`` added only the finiteness
    check (now in the hooks) and a recursive walk whose depth limit is the interpreter's recursion limit (it depends
    on the caller's stack and on ABC caches), so documents nested deeper than ``_SHALLOW_JSON_DEPTH`` still go
    through that same ``json_copy`` call; shallower ones cannot reach the limit.
    """
    try:
        value = json.loads(data.decode("utf-8"), object_pairs_hook=_unique_pairs, parse_float=_finite_float,
                           parse_constant=_reject_constant)
        if type(value) is dict or type(value) is list:
            try:
                _shallow(value, _SHALLOW_JSON_DEPTH)
            except _DeepDocument:
                value = json_copy(value)
        return value
    except (ValueError, UnicodeError, TypeError, RecursionError, ConfigurationError) as exc:
        raise IntegrityError("invalid JSON metadata") from exc


def resolve_dtype(dtype: str | torch.dtype) -> torch.dtype:
    name = str(dtype).removeprefix("torch.")
    if name not in _DTYPES:
        raise UnsupportedError(f"unsupported tensor state dtype: {dtype!r}")
    return _DTYPES[name]


def dtype_name(dtype: str | torch.dtype) -> str:
    return str(resolve_dtype(dtype)).removeprefix("torch.")


def tensor_nbytes(shape: Sequence[int], dtype: str | torch.dtype) -> int:
    if not isinstance(shape, (list, tuple)) or len(shape) > 64:
        raise ConfigurationError("tensor shape must have at most 64 dimensions")
    if any(type(size) is not int or not 0 <= size <= sys.maxsize for size in shape):
        raise ConfigurationError("tensor dimensions must be representable nonnegative integers")
    count = math.prod(shape)
    size = count * torch.empty((), dtype=resolve_dtype(dtype)).element_size()
    if size > sys.maxsize:
        raise ConfigurationError("tensor storage exceeds the host address space")
    return size


def tensor_bytes(tensor: torch.Tensor) -> bytes:
    if not isinstance(tensor, torch.Tensor):
        raise ConfigurationError("tensor state values must be torch.Tensor instances")
    resolve_dtype(tensor.dtype)
    if tensor.layout != torch.strided or tensor.is_quantized or tensor.device.type == "meta":
        raise UnsupportedError("only materialized dense strided tensors can be serialized")
    if sys.byteorder != "little":
        raise UnsupportedError("tensor state currently requires a little-endian host")
    value = tensor.detach().resolve_conj().resolve_neg().to("cpu").contiguous()
    return value.reshape(-1).view(torch.uint8).numpy().tobytes()


def tensor_from_bytes(
    data: bytes, *, dtype: str | torch.dtype, shape: Sequence[int], device: Any = "cpu"
) -> torch.Tensor:
    if sys.byteorder != "little":
        raise UnsupportedError("tensor state currently requires a little-endian host")
    if not isinstance(data, bytes):
        raise IntegrityError("tensor payload must be immutable bytes")
    try:
        resolved = resolve_dtype(dtype)
        expected = tensor_nbytes(shape, resolved)
    except (ConfigurationError, UnsupportedError) as exc:
        raise IntegrityError("invalid serialized tensor geometry or dtype") from exc
    if len(data) != expected:
        raise IntegrityError(f"tensor byte length mismatch: expected {expected}, got {len(data)}")
    try:
        if expected == 0:
            value = torch.empty(tuple(shape), dtype=resolved)
        else:
            value = torch.frombuffer(bytearray(data), dtype=resolved).reshape(tuple(shape))
    except (RuntimeError, TypeError, ValueError, OverflowError) as exc:
        raise IntegrityError("serialized tensor shape cannot be materialized") from exc
    try:
        return value.to(device=device)
    except (RuntimeError, TypeError, ValueError) as exc:
        raise ConfigurationError(f"cannot restore tensor on device {device!r}") from exc


def serialize_tensors(tensors: Mapping[str, torch.Tensor]) -> bytes:
    if not isinstance(tensors, Mapping):
        raise ConfigurationError("named tensor state must be a mapping")
    if any(not isinstance(name, str) or not name for name in tensors):
        raise ConfigurationError("tensor state names must be nonempty strings")
    entries: dict[str, Any] = {}
    payloads: list[bytes] = []
    offset = 0
    for name in sorted(tensors):
        tensor = tensors[name]
        payload = tensor_bytes(tensor)
        entries[name] = {
            "dtype": dtype_name(tensor.dtype), "shape": list(tensor.shape),
            "offset": offset, "nbytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
        payloads.append(payload)
        offset += len(payload)
    header = canonical_json({"version": 1, "byteorder": "little", "tensors": entries})
    if len(header) > _MAX_HEADER_BYTES:
        raise CapacityError("tensor state header exceeds the format limit")
    return _MAGIC + struct.pack("<Q", len(header)) + header + b"".join(payloads)


def deserialize_tensors(
    data: bytes, *, device: Any = "cpu", max_bytes: int | None = None
) -> dict[str, torch.Tensor]:
    if not isinstance(data, bytes) or not data.startswith(_MAGIC):
        raise IntegrityError("not a ReelCache non-pickle tensor state")
    prefix = len(_MAGIC) + 8
    if len(data) < prefix:
        raise IntegrityError("truncated tensor state header")
    header_size = struct.unpack("<Q", data[len(_MAGIC):prefix])[0]
    if header_size > _MAX_HEADER_BYTES or header_size > len(data) - prefix:
        raise IntegrityError("invalid tensor state header length")
    header = parse_json(data[prefix:prefix + header_size])
    if (not isinstance(header, dict)
            or set(header) != {"version", "byteorder", "tensors"}
            or type(header["version"]) is not int or header["version"] != 1
            or header["byteorder"] != "little"
            or not isinstance(header["tensors"], dict)):
        raise IntegrityError("unsupported tensor state format")
    body = memoryview(data)[prefix + header_size:]
    if max_bytes is not None:
        if type(max_bytes) is not int or max_bytes < 0:
            raise ConfigurationError("max_bytes must be nonnegative")
        if len(body) > max_bytes:
            raise CapacityError("serialized tensor state exceeds the restoration budget")
    entries = header["tensors"]
    offset = 0
    validated: list[tuple[str, dict[str, Any], bytes]] = []
    for name in sorted(entries):
        desc = entries[name]
        if (not name or not isinstance(desc, dict)
                or set(desc) != {"dtype", "shape", "offset", "nbytes", "sha256"}
                or type(desc["offset"]) is not int or desc["offset"] != offset
                or type(desc["nbytes"]) is not int or desc["nbytes"] < 0):
            raise IntegrityError("invalid tensor state entry or noncontiguous offsets")
        try:
            expected = tensor_nbytes(desc["shape"], desc["dtype"])
        except (ConfigurationError, UnsupportedError) as exc:
            raise IntegrityError("invalid tensor state dtype or shape") from exc
        if expected != desc["nbytes"] or offset + expected > len(body):
            raise IntegrityError("tensor state shape or length mismatch")
        payload = bytes(body[offset:offset + expected])
        if hashlib.sha256(payload).hexdigest() != desc["sha256"]:
            raise IntegrityError(f"corrupt tensor state: {name}")
        validated.append((name, desc, payload))
        offset += expected
    if offset != len(body):
        raise IntegrityError("tensor state has unclaimed trailing bytes")
    return {
        name: tensor_from_bytes(payload, dtype=desc["dtype"], shape=desc["shape"], device=device)
        for name, desc, payload in validated
    }
