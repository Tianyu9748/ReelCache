"""Portable RAW, SVD, and software SVD+HEVC tensor representations.

Packets contain only JSON metadata and immutable byte payloads. The entire
canonical FP16 basis is separate from coefficient payloads, even for a smaller
rank view. RAW is bit preserving; SVD and HEVC are explicitly lossy. HEVC uses
QP8/ultrafast and either genuine yuv444p or genuine yuv420p, never a fallback.

Packet schemas: RAW and SVD-only packets keep ``reelcache.tensor.v1`` unchanged.
New svd_hevc packets are version 2: ``reelcache.tensor.v2`` (one matrix in its
own-size channel-tile layout) or ``reelcache.tensor_pack.v2`` (one or many
matrices of one node on a common all-intra canvas with a validated tile map).
Both record ``CodecSpec.inter_prediction`` and the exact x265 contract.
Version-1 svd_hevc packets keep their historical (x265 preset-default GOP)
identity and decode unchanged; they are never re-labelled as all-intra.

``with using_ffmpeg("/path/to/ffmpeg"): ...`` supplies an executable for both
encoding and independent reload. Otherwise an existing PATH/imageio-ffmpeg
binary is discovered. No FFmpeg path is required in persisted metadata.
For worker-based runtimes, inject ``HEVCProvider(path).encode_tensor`` and
``.decode_tensor``; each bound call establishes its own context-local selection.
"""

from __future__ import annotations

import contextlib
import functools
import hashlib
import json
import math
import re
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
import torch

from . import _hevc
from . import accel_c as _accel_c
from .errors import ConfigurationError, IntegrityError, UnsupportedError
from .factorization import (
    Factors, basis_condition_bound, rank_projection_workspace_bytes, tensor_sha256,
    validate_matrix, validate_rank, validate_stored_basis,
)
from .reuse import factorize_tensor, validate_threshold
from .types import CodecSpec, EncodedTensor, MatrixKey, TensorContext, canonical_dtype, json_copy
from .telemetry import cuda_span, span
from .nvdec import NVDECDecoder
from .gpu_fitting import GPUFitter


using_ffmpeg = _hevc.using_ffmpeg
hevc_capabilities = _hevc.capabilities
SCHEMA = "reelcache.tensor.v1"
HEVC_SCHEMA = "reelcache.tensor.v2"
PACK_SCHEMA = "reelcache.tensor_pack.v2"
#: wl-base (workload-v2 W1c): node packs of FP16 factors without HEVC (``reelcache.factor_packs``, the SVD-only baseline).
FACTOR_PACK_SCHEMA = "reelcache.factor_pack.v1"
TENSOR_SCHEMAS = (SCHEMA, HEVC_SCHEMA)
PACKING_MODES = ("per_matrix", "node", "layer_group")
PACKED_LAYOUTS = ("temporal", "spatial")
_DTYPES = {
    "float16": torch.float16, "bfloat16": torch.bfloat16,
    "float32": torch.float32, "float64": torch.float64,
}
_ITEM_SIZES = {"float16": 2, "bfloat16": 2, "float32": 4, "float64": 8, "uint8": 1}
_METADATA_FIELDS = {
    "schema", "kind", "context", "shape", "dtype", "byte_order", "codec",
    "payloads", "basis", "coefficient_shape", "normalization", "video",
    "decoded_coefficients_sha256", "factorization_domain",
}
_METADATA_FIELDS_V2 = _METADATA_FIELDS | {"hevc"}
_PACK_FIELDS = {
    "schema", "byte_order", "node", "codec", "hevc", "packing", "canvas", "tiles",
    "payloads", "padding",
}
_TILE_FIELDS = {
    "key", "stream", "context", "codec", "shape", "dtype", "factorization_domain",
    "coefficient_shape", "basis", "affine", "normalization", "video", "pictures", "rect",
    "byte_range", "decoded_coefficients_sha256",
}
_PREPARED_BINDING_FIELDS = {
    "schema", "policy_identity", "source_sha256", "context_sha256", "spec_sha256",
    "basis_sha256", "coefficients_sha256", "metrics_sha256",
}


def tensor_metadata_fields(schema):
    """Exact single-matrix header fields of a supported tensor packet schema."""
    if schema == SCHEMA:
        return set(_METADATA_FIELDS)
    if schema == HEVC_SCHEMA:
        return set(_METADATA_FIELDS_V2)
    raise IntegrityError("unsupported tensor packet schema")


def _json_bytes(value):
    return json.dumps(
        json_copy(value), sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")


_HEX64 = re.compile("[0-9a-f]{64}")


def _plain_json(value):
    """True when ``value`` holds only str-keyed dicts, lists/tuples, str, int, bool, None and finite floats."""
    kind = type(value)
    if kind is dict:
        for key, item in value.items():
            if type(key) is not str or not _plain_json(item):
                return False
        return True
    if kind is list or kind is tuple:
        for item in value:
            if not _plain_json(item):
                return False
        return True
    if kind is float:
        return math.isfinite(value)
    return value is None or kind is str or kind is int or kind is bool


def _canon(value):
    """Exactly ``_json_bytes(value)``; one C-encoder pass when ``value`` is plain JSON (e.g. a parsed manifest)."""
    if _plain_json(value):
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return _json_bytes(value)


def _frozen_layout(layout):
    return MappingProxyType({key: tuple(value) if isinstance(value, list) else value for key, value in layout.items()})


# Pure derived layouts of packed-tile validation, shared by every pack of a model geometry (restore reads many packs
# with the same ranks; typed so that e.g. True and 1 stay distinct arguments). Failures are not cached.
@functools.lru_cache(maxsize=4096, typed=True)
def _normalization_canon(rank, normalization, protected_fraction):
    layout = _hevc.normalization_layout(rank, normalization, protected_fraction)
    return _frozen_layout(layout), _canon(layout)


@functools.lru_cache(maxsize=4096, typed=True)
def _video_canon(frames, height, width, rank, pix_fmt):
    video = _hevc.video_layout(frames, height, width, rank, pix_fmt)
    return _frozen_layout(video), _canon(video)


_PACKED_SPECS = {}


def _packed_tile_spec(codec):
    """``CodecSpec`` of a packed tile's codec record, validated and canonical (memoized by its canonical JSON)."""
    text = _canon(codec)
    spec = _PACKED_SPECS.get(text)
    if spec is None:
        spec = CodecSpec.from_dict(codec)
        _validate_spec(spec)
        if _canon(_portable_spec(spec)) != text:
            raise IntegrityError("packed tile codec differs from its all-intra stream")
        if len(_PACKED_SPECS) < 4096:
            _PACKED_SPECS[text] = spec
    return spec


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _tensor_bytes(tensor):
    value = tensor.detach().contiguous().cpu()
    raw = value.view(torch.uint8).numpy().reshape(-1, value.element_size())
    if sys.byteorder != "little":
        raw = raw[:, ::-1]
    return raw.tobytes()


def _from_bytes(raw, dtype, shape, *, columns=None):
    item_size = _ITEM_SIZES[dtype]
    values = np.frombuffer(raw, dtype=np.uint8).reshape(*shape, item_size)
    if columns is not None:
        values = values[:, columns, :]
    if sys.byteorder != "little":
        values = values[..., ::-1]
    owned = np.ascontiguousarray(values).copy()
    return torch.from_numpy(owned).view(_DTYPES[dtype]).reshape(owned.shape[:-1])


def _descriptor(raw, dtype, shape, **extra):
    return {
        "bytes": len(raw), "sha256": _sha(raw), "dtype": dtype,
        "shape": list(shape), **extra,
    }


def _portable_spec(spec):
    config = spec.to_dict()
    config.pop("ffmpeg_path", None)
    return config


def factor_binding(tensor, context, spec, factors, policy_identity=None):
    """Bind prepared factors to their exact input, request, and diagnostics.

    This is a portable integrity record, not permission to skip arithmetic
    validation. Version 2 binds the guard's row tiling as part of its numerical
    execution recipe, outside the unchanged v1 packet header.
    """
    _validate_spec(spec)
    value = validate_matrix(tensor, context, lossy=True)
    _prepared_geometry(value, spec, factors)
    tile_rows = _prepared_guard_tile_rows(factors.metrics)
    if policy_identity is not None and (
        not isinstance(policy_identity, str) or len(policy_identity) != 64
        or any(character not in "0123456789abcdef" for character in policy_identity)
    ):
        raise ConfigurationError("prepared policy identity must be a SHA256 or None")
    return {
        "schema": "reelcache.prepared_factors.v2",
        "policy_identity": policy_identity,
        "guard_tile_rows": tile_rows,
        "source_sha256": tensor_sha256(value, tile_rows=tile_rows),
        "context_sha256": _sha(_json_bytes(context.to_dict())),
        "spec_sha256": _sha(_json_bytes(spec.to_dict())),
        "basis_sha256": tensor_sha256(factors.basis, tile_rows=tile_rows),
        "coefficients_sha256": tensor_sha256(factors.coefficients, tile_rows=tile_rows),
        "metrics_sha256": _sha(_json_bytes(factors.metrics)),
    }


def _prepared_guard_tile_rows(metrics):
    tile_rows = metrics.get("tile_rows", 1024)
    if type(tile_rows) is not int or tile_rows < 1:
        raise ConfigurationError("prepared guard tile_rows must be a positive integer")
    return tile_rows


def _prepared_binding_tile_rows(binding, metrics):
    if not isinstance(binding, Mapping):
        raise IntegrityError("prepared factor binding must be an object")
    schema = binding.get("schema")
    if not isinstance(schema, str) or schema not in {"reelcache.prepared_factors.v1", "reelcache.prepared_factors.v2"}:
        raise IntegrityError("unsupported prepared factor binding schema")
    fields = _PREPARED_BINDING_FIELDS
    if schema == "reelcache.prepared_factors.v2":
        fields = fields | {"guard_tile_rows"}
    if set(binding) != fields:
        raise IntegrityError("prepared factor binding fields are incomplete or unsupported")
    # Legacy records already bind this recipe indirectly through metrics_sha256.
    tile_rows = (
        binding["guard_tile_rows"] if schema == "reelcache.prepared_factors.v2"
        else _prepared_guard_tile_rows(metrics)
    )
    if type(tile_rows) is not int or tile_rows < 1:
        raise IntegrityError("prepared binding guard tiling must be a positive integer")
    return tile_rows


def _prepared_geometry(value, spec, factors):
    if spec.kind == "raw":
        raise ConfigurationError("RAW encoding cannot consume prepared lossy factors")
    if not isinstance(factors, Factors):
        raise ConfigurationError("prepared_factors must be Factors")
    validate_rank(spec.rank, tuple(value.shape))
    if type(factors.rank) is not int or factors.rank != spec.rank:
        raise ConfigurationError("prepared factor rank differs from the selected specification")
    stored = validate_stored_basis(factors.basis, value.shape[1])
    coefficients = factors.coefficients
    if stored.shape[0] < spec.rank:
        raise ConfigurationError("prepared factor view exceeds the canonical basis")
    if (
        not isinstance(coefficients, torch.Tensor) or coefficients.layout != torch.strided
        or coefficients.device.type == "meta" or coefficients.dtype != torch.float32
        or tuple(coefficients.shape) != (value.shape[0], spec.rank)
        or not bool(torch.isfinite(coefficients).all())
    ):
        raise ConfigurationError("prepared coefficients must be finite canonical FP32 P")
    if not isinstance(factors.metrics, Mapping):
        raise ConfigurationError("prepared factor diagnostics must be a mapping")
    json_copy(factors.metrics, name="prepared factor metrics")


def prepared_validation_workspace_bytes(context, spec, *, tile_rows=1024):
    """Verifier workspace for the bound guard tiling, excluding source/factors."""
    if not isinstance(context, TensorContext) or not isinstance(spec, CodecSpec) or spec.kind == "raw":
        raise ConfigurationError("prepared validation requires a lossy canonical matrix")
    if type(tile_rows) is not int or tile_rows < 1:
        raise ConfigurationError("prepared guard tile_rows must be a positive integer")
    validate_rank(spec.rank, context.shape)
    rows, channels = context.shape
    rank = spec.rank
    # Also reserve a coefficient transfer and the stored-dtype residual buffers.
    return (
        rank_projection_workspace_bytes(context.shape, rank, tile_rows=tile_rows)
        + rows * rank * 4 + min(tile_rows, rows) * channels * 24
    )


@torch.no_grad()
def validate_prepared_factors(tensor, context, spec, factors, binding):
    """Verify FP16 B, FP32 P and source binding without another decomposition."""
    started = time.perf_counter()
    if not isinstance(binding, Mapping):
        raise ConfigurationError("prepared factors require an explicit source/request binding")
    checked = json_copy(binding, name="prepared factor binding")
    actual = factor_binding(
        tensor, context, spec, factors, checked.get("policy_identity"),
    )
    tile_rows = _prepared_binding_tile_rows(checked, factors.metrics)
    if tile_rows != actual["guard_tile_rows"]:
        raise IntegrityError("prepared binding changed the accepted guard tiling")
    if checked["schema"] == "reelcache.prepared_factors.v1":
        actual.pop("guard_tile_rows")
        actual["schema"] = "reelcache.prepared_factors.v1"
    if checked != actual:
        raise IntegrityError("prepared factors differ from their source/context/spec/metrics binding")
    value = tensor.detach()
    view = factors.basis[:spec.rank].to(device=value.device, dtype=torch.float32)
    condition = basis_condition_bound(view)
    if not math.isfinite(condition) or condition > 1e6:
        raise ConfigurationError("prepared FP16 basis is singular or ill-conditioned")
    coefficients = factors.coefficients.to(device=value.device)
    expected = value.float() @ view.T
    if not torch.equal(coefficients, expected):
        raise IntegrityError("prepared coefficients are not the declared source's FP32 projection")
    del expected
    source_squared = torch.zeros((), dtype=torch.float64, device=value.device)
    error_squared = torch.zeros_like(source_squared)
    stored_error_squared = torch.zeros_like(source_squared)
    scale = float(torch.maximum(value.amax().abs(), value.amin().abs()))
    for start in range(0, value.shape[0], tile_rows):
        stop = min(value.shape[0], start + tile_rows)
        source = value[start:stop].double()
        reconstructed = coefficients[start:stop] @ view
        error = source - reconstructed.double()
        stored_error = source - reconstructed.to(value.dtype).double()
        if scale:
            source = source / scale
            error = error / scale
            stored_error = stored_error / scale
        source_squared += source.square().sum()
        error_squared += error.square().sum()
        stored_error_squared += stored_error.square().sum()
    source_squared = float(source_squared)
    residual = 0.0 if not scale else math.sqrt(float(error_squared) / source_squared)
    stored_residual = 0.0 if not scale else math.sqrt(float(stored_error_squared) / source_squared)
    if not math.isfinite(residual) or not math.isfinite(stored_residual):
        raise IntegrityError("prepared factor residual is not finite")
    metrics = factors.metrics
    declared = metrics.get("projection_relative_residual")
    declared_condition = metrics.get("basis_condition_bound")
    if (
        isinstance(declared, bool) or not isinstance(declared, (int, float))
        or not math.isfinite(declared)
        or not math.isclose(declared, residual, rel_tol=1e-6, abs_tol=1e-12)
        or type(metrics.get("zero_input")) is not bool
        or metrics["zero_input"] != (scale == 0.0)
        or isinstance(declared_condition, bool) or not isinstance(declared_condition, (int, float))
        or not math.isfinite(declared_condition)
        or not math.isclose(declared_condition, condition, rel_tol=1e-6, abs_tol=1e-12)
    ):
        raise IntegrityError("prepared factor diagnostics disagree with the actual projection")
    return {
        "seconds": time.perf_counter() - started,
        "projection_count": 1, "decomposition_count": 0,
        "projection_relative_residual": residual, "basis_condition_bound": condition,
        "stored_svd_relative_residual": stored_residual,
        "guard_tile_rows": tile_rows,
        "workspace_estimate_bytes": prepared_validation_workspace_bytes(context, spec, tile_rows=tile_rows),
        "workspace_is_measured_peak": False,
    }


def _codec_comparison_header(config):
    """Normalize only ignored fractions in a copy of a validated v1 header."""
    if not isinstance(config, dict):
        raise IntegrityError("portable codec metadata must be an object")
    comparable = dict(config)
    if "protected_fraction" in comparable:
        if comparable.get("normalization") == "global":
            comparable["protected_fraction"] = 0.0
        elif comparable.get("normalization") == "per_channel":
            comparable["protected_fraction"] = 1.0
    return comparable


def _validate_spec(spec):
    if not isinstance(spec, CodecSpec):
        raise ConfigurationError("spec must be a CodecSpec")
    if isinstance(spec.preset, str) and spec.preset.startswith("nvenc-"):
        from .nvenc import validate_spec_parameters  # explicit NVENC contract (Step 15)
        validate_spec_parameters(spec)
    else:
        _hevc.validate_parameters(spec.pix_fmt, spec.qp, spec.preset)
    if spec.kind == "svd_hevc":
        _hevc.normalization_layout(spec.rank, spec.normalization, spec.protected_fraction)


def _require_encodable(spec, encoder=None):
    if spec.kind == "svd_hevc" and spec.inter_prediction is None:
        raise ConfigurationError(
            "legacy version-1 svd_hevc specifications (implicit x265 preset GOP) are "
            "decode-only; set inter_prediction=False (all-intra default) or True explicitly",
        )
    # An explicit picture encoder (reelcache.nvenc) and its specifications never substitute for libx265.
    if spec.kind == "svd_hevc" and encoder is not None:
        encoder.validate_spec(spec)
    elif spec.kind == "svd_hevc" and spec.preset.startswith("nvenc-"):
        raise UnsupportedError(
            "NVENC specifications are encoded only by an explicit reelcache.nvenc.NVENCEncoder "
            "(packed node streams); libx265 never substitutes for NVENC",
        )


def encoded_nbytes(encoded: EncodedTensor, *, include_metrics=False) -> int:
    """Exact logical bytes: canonical metadata + payloads + whole basis.

    Optional diagnostics can be counted explicitly. Persistent stores also
    charge their own manifests/envelopes and count shared objects uniquely;
    this function does not pretend that those store-specific bytes are zero.
    """
    if not isinstance(encoded, EncodedTensor):
        raise ConfigurationError("encoded must be an EncodedTensor")
    result = len(_json_bytes(encoded.metadata))
    result += sum(len(raw) for raw in encoded.payloads.values())
    result += len(encoded.basis) if encoded.basis is not None else 0
    if include_metrics and encoded.metrics is not None:
        result += len(_json_bytes(encoded.metrics))
    return result


@dataclass(frozen=True)
class HEVCProvider:
    """Injectable provider with an explicit executable, including on workers.

    RAW/SVD remain dependency-free when ``ffmpeg_path`` is omitted. Encoding
    and decoding signatures match KVRuntime's provider injection points; the
    executable is operational configuration, not part of a portable packet.
    ``packing`` (node (default) | layer_group | per_matrix) only affects
    :meth:`encode_node`; the per-matrix ``encode_tensor`` path is unchanged.
    The GOP choice is per specification (``CodecSpec.inter_prediction``);
    inter-coded specifications require ``packing="per_matrix"``.
    """

    ffmpeg_path: str | None = None
    nvdec: NVDECDecoder | None = None
    fitter: GPUFitter | None = None
    verify_decoded_coefficients: bool = False
    verify_restored_packets: bool = False
    validate_hevc_encode: bool = False
    x265_threads: int = 4
    packing: str = "node"
    layer_group_size: int | None = None
    packed_layout: str = "temporal"

    def __post_init__(self):
        _hevc.x265_thread_params(self.x265_threads, 1)
        if (type(self.verify_decoded_coefficients) is not bool or type(self.verify_restored_packets) is not bool
                or type(self.validate_hevc_encode) is not bool):
            raise ConfigurationError("verification options must be boolean")
        if self.fitter is not None and not isinstance(self.fitter, GPUFitter):
            raise ConfigurationError("fitter must be an explicitly configured GPUFitter")
        if self.nvdec is not None and not isinstance(self.nvdec, NVDECDecoder):
            raise ConfigurationError("nvdec must be an explicitly configured NVDECDecoder")
        _validate_packing(self.packing, self.layer_group_size, self.packed_layout)
        if self.ffmpeg_path is not None:
            object.__setattr__(self, "ffmpeg_path", _hevc.discover_ffmpeg(self.ffmpeg_path))

    def _scope(self):
        return (
            contextlib.nullcontext() if self.ffmpeg_path is None
            else using_ffmpeg(self.ffmpeg_path)
        )

    def encode_tensor(
        self, tensor, context, spec, *, reuse_basis=None, reuse_threshold=None,
        prepared_factors=None, prepared_binding=None,
    ):
        with self._scope():
            return encode_tensor(
                tensor, context, spec, reuse_basis=reuse_basis, reuse_threshold=reuse_threshold,
                prepared_factors=prepared_factors, prepared_binding=prepared_binding,
                fitter=self.fitter, validate_hevc=self.validate_hevc_encode,
                hevc_threads=self.x265_threads,
            )

    @contextlib.contextmanager
    def prefit(self, matrices, contexts, specs):
        """Batch GPU basis fitting for one chunk before per-matrix ``encode_tensor`` calls.

        A no-op unless the fitter uses ``gram_batched``. Cached bases are
        dropped on exit, so an interrupted seal cannot leak them into a later one.
        """
        if self.fitter is None or self.fitter.method != "gram_batched":
            yield
            return
        items = {
            key: (matrices[key], contexts[key], specs[key].rank)
            for key in matrices if specs[key].kind in ("svd", "svd_hevc")
        }
        try:
            self.fitter.prefit(items)
            yield
        finally:
            self.fitter.clear_prefit()

    def decode_tensor(self, encoded, *, device="cpu", dtype=None, columns=None):
        with self._scope():
            return decode_tensor(encoded, device=device, dtype=dtype, columns=columns, nvdec=self.nvdec,
                                 verify_decoded_coefficients=self.verify_decoded_coefficients,
                                 verify_packet=self.verify_restored_packets)

    def decode_column_views(self, encoded, *, column_groups, device="cpu", dtype=None):
        with self._scope():
            return decode_column_views(encoded, column_groups=column_groups, device=device, dtype=dtype,
                                       nvdec=self.nvdec, verify_decoded_coefficients=self.verify_decoded_coefficients,
                                       verify_packet=self.verify_restored_packets)

    def encode_node(self, matrices, contexts, specs, *, node_id=None, reuse_bases=None,
                    reuse_threshold=None, prepared=None):
        """Encode one node with this provider's explicit packing configuration."""
        with self._scope():
            return encode_node(
                matrices, contexts, specs, packing=self.packing,
                layer_group_size=self.layer_group_size, layout=self.packed_layout, node_id=node_id,
                reuse_bases=reuse_bases, reuse_threshold=reuse_threshold, prepared=prepared,
                fitter=self.fitter, validate_hevc=self.validate_hevc_encode,
                hevc_threads=self.x265_threads,
            )

    def decode_packed(self, encoded, *, keys=None, device="cpu", dtype=None):
        with self._scope():
            return decode_packed(encoded, keys=keys, device=device, dtype=dtype, nvdec=self.nvdec,
                                 verify_decoded_coefficients=self.verify_decoded_coefficients,
                                 verify_packet=self.verify_restored_packets)


def _check_encode_request(
    tensor, context, spec, *, reuse_basis, reuse_threshold, prepared_factors, prepared_binding,
    fitter, validate_hevc, hevc_threads, encoder=None,
):
    """Shared encode admission; returns the validated matrix."""
    if type(validate_hevc) is not bool:
        raise ConfigurationError("validate_hevc must be boolean")
    _hevc.x265_thread_params(hevc_threads, 1)
    if fitter is not None and not isinstance(fitter, GPUFitter):
        raise ConfigurationError("fitter must be an explicitly configured GPUFitter")
    if fitter is not None and prepared_factors is not None:
        raise ConfigurationError("GPU fitting cannot replace an adaptive prepared-factor decision")
    _validate_spec(spec)
    _require_encodable(spec, encoder)
    # A GPU fitter repeats the finite-value scan on its device copy; skip the host pass.
    value = validate_matrix(
        tensor, context, lossy=spec.kind != "raw",
        check_values=fitter is None or prepared_factors is not None or spec.kind == "raw",
    )
    if reuse_threshold is not None:
        validate_threshold(reuse_threshold)
    if (prepared_factors is None) != (prepared_binding is None):
        raise ConfigurationError("prepared factors and their binding must be supplied together")
    if prepared_factors is not None and spec.kind == "raw":
        raise ConfigurationError("RAW encoding cannot consume prepared lossy factors")
    if prepared_factors is not None and (
        reuse_basis is not None or reuse_threshold is not None
    ):
        raise ConfigurationError("prepared factors cannot request another reuse decision or fit")
    return value


def _initial_metrics(spec):
    return {
        "reuse_attempts": 0, "reuse_accepted": False, "reuse_rejections": 0,
        "reuse_reason": "raw" if spec.kind == "raw" else "no_candidate",
        "reuse_seconds": 0.0, "fit_seconds": 0.0, "decomposition_seconds": 0.0,
        "decomposition_count": 0, "svd_count": 0, "eigh_count": 0,
        "qr_count": 0, "projection_count": 0, "projection_seconds": 0.0,
    }


def _fit_factors(
    value, context, spec, metrics, *, reuse_basis, reuse_threshold, prepared_factors,
    prepared_binding, fitter,
):
    validate_rank(spec.rank, tuple(value.shape))
    if prepared_factors is None:
        factorizer = factorize_tensor if fitter is None else fitter.fit
        factors = factorizer(
            value, context, spec.rank,
            reuse_basis=reuse_basis, reuse_threshold=reuse_threshold,
        )
    else:
        validation = validate_prepared_factors(
            value, context, spec, prepared_factors, prepared_binding,
        )
        factors = prepared_factors
        metrics["prepared_binding"] = json_copy(prepared_binding)
        metrics["prepared_validation"] = validation
    metrics.update(factors.metrics)
    return factors


def _factorization_domain(context):
    return "normalized_pre_rope_k" if context.key.target == "K" else "direct_v"


def _quantized_tiles(factors, context, spec, metrics):
    normalization_started = time.perf_counter()
    pixels, affine, normalization = _hevc.quantize_coefficients(
        factors.coefficients.cpu().numpy(), spec.normalization, spec.protected_fraction,
    )
    metrics["normalization_seconds"] = time.perf_counter() - normalization_started
    layout = _hevc.video_layout(
        context.frames, context.height, context.width, spec.rank, spec.pix_fmt,
    )
    return pixels, affine, normalization, layout


@torch.no_grad()
def encode_tensor(
    tensor, context, spec, *, reuse_basis=None, reuse_threshold=None,
    prepared_factors=None, prepared_binding=None,
    fitter=None, validate_hevc=False, hevc_threads=1,
) -> EncodedTensor:
    """Encode one canonical ``[frames*height*width, native_KV_channels]`` matrix.

    ``prepared_factors`` bypasses fitting, not validation. Its explicit binding
    is obtained from :func:`factor_binding`; only the accepted FP16 basis and
    FP32 projection are packed. RAW/SVD packets keep the v1 bytes; svd_hevc
    packets use ``reelcache.tensor.v2`` with the spec's explicit GOP contract.
    """
    started = time.perf_counter()
    value = _check_encode_request(
        tensor, context, spec, reuse_basis=reuse_basis, reuse_threshold=reuse_threshold,
        prepared_factors=prepared_factors, prepared_binding=prepared_binding,
        fitter=fitter, validate_hevc=validate_hevc, hevc_threads=hevc_threads,
    )
    # Prepared (adaptive) packets are independently re-validated against this checksum.
    validate_hevc = validate_hevc or prepared_factors is not None
    metadata = {
        "schema": HEVC_SCHEMA if spec.kind == "svd_hevc" else SCHEMA,
        "kind": spec.kind, "context": context.to_dict(),
        "shape": list(value.shape), "dtype": canonical_dtype(value.dtype),
        "byte_order": "little", "codec": _portable_spec(spec),
        "payloads": {}, "basis": None, "coefficient_shape": None,
        "normalization": None, "video": None,
        "decoded_coefficients_sha256": None, "factorization_domain": "raw",
    }
    if spec.kind == "svd_hevc":
        metadata["hevc"] = _hevc.encoder_record(spec.inter_prediction)
    payloads = {}
    basis_bytes = None
    metrics = _initial_metrics(spec)
    if spec.kind == "raw":
        if reuse_basis is not None:
            raise ConfigurationError("RAW encoding does not consume a reuse basis")
        raw = _tensor_bytes(value)
        payloads["tensor.bin"] = raw
        metadata["payloads"]["tensor.bin"] = _descriptor(raw, metadata["dtype"], value.shape)
    else:
        factors = _fit_factors(
            value, context, spec, metrics, reuse_basis=reuse_basis, reuse_threshold=reuse_threshold,
            prepared_factors=prepared_factors, prepared_binding=prepared_binding, fitter=fitter,
        )
        basis_bytes = _tensor_bytes(factors.basis)
        metadata["basis"] = _descriptor(
            basis_bytes, "float16", factors.basis.shape, view_rank=spec.rank,
        )
        metadata["coefficient_shape"] = list(factors.coefficients.shape)
        metadata["factorization_domain"] = _factorization_domain(context)
        if spec.kind == "svd":
            raw = _tensor_bytes(factors.coefficients)
            payloads["coefficients.f32"] = raw
            metadata["payloads"]["coefficients.f32"] = _descriptor(
                raw, "float32", factors.coefficients.shape,
            )
            metadata["decoded_coefficients_sha256"] = _sha(raw)
        else:
            pixels, affine, normalization, layout = _quantized_tiles(factors, context, spec, metrics)
            ffmpeg_path = getattr(spec, "ffmpeg_path", None)
            streams, encode_seconds, version = _hevc.encode_pixels(
                pixels, layout, ffmpeg_path=ffmpeg_path, threads=hevc_threads,
                inter_prediction=spec.inter_prediction,
            )
            decoded_coefficients, decode_seconds = None, 0.0
            if validate_hevc:
                decoded_pixels, decode_seconds = _hevc.decode_pixels(
                    streams, layout, ffmpeg_path=ffmpeg_path,
                )
                decoded_coefficients = _hevc.dequantize_coefficients(decoded_pixels, affine, normalization)
            raw = affine.astype("<f4", copy=False).tobytes()
            payloads["affine.f32"] = raw
            metadata["payloads"]["affine.f32"] = _descriptor(raw, "float32", affine.shape)
            for index, stream in enumerate(streams):
                name = f"coefficients.{index:03d}.hevc"
                payloads[name] = stream
                metadata["payloads"][name] = _descriptor(stream, "uint8", (len(stream),))
            metadata["normalization"] = normalization
            metadata["video"] = layout
            metadata["decoded_coefficients_sha256"] = None if decoded_coefficients is None else _sha(
                decoded_coefficients.astype("<f4", copy=False).tobytes()
            )
            metrics.update(
                hevc_encode_seconds=encode_seconds,
                hevc_encode_threads=hevc_threads,
                hevc_validation_decode=validate_hevc,
                hevc_validation_decode_seconds=decode_seconds,
                hevc_inter_prediction=spec.inter_prediction,
                hevc_picture_structure=metadata["hevc"]["picture_structure"],
                ffmpeg_version=version,
                hevc_software_only=True,
            )
    metadata_bytes = len(_json_bytes(metadata))
    payload_bytes = sum(map(len, payloads.values()))
    basis_size = len(basis_bytes) if basis_bytes is not None else 0
    metrics.update(
        source_bytes=value.numel() * value.element_size(),
        payload_bytes=payload_bytes, basis_bytes=basis_size,
        metadata_bytes=metadata_bytes,
        representation_bytes=metadata_bytes + payload_bytes + basis_size,
        encode_seconds=time.perf_counter() - started,
    )
    return EncodedTensor(metadata, payloads, basis_bytes, metrics)


def validate_prepared_packet(
    encoded, context, spec, factors, binding, *, source=None, ffmpeg_path=None,
):
    """Reject a provider that substitutes another prepared representation."""
    started = time.perf_counter()
    metadata, declared_context, declared_spec = _validate_packet(encoded)
    if declared_context != context or _portable_spec(declared_spec) != _portable_spec(spec):
        raise IntegrityError("prepared packet does not represent the authoritative request")
    tile_rows = _prepared_binding_tile_rows(binding, factors.metrics)
    if tile_rows != _prepared_guard_tile_rows(factors.metrics):
        raise IntegrityError("prepared packet changed the accepted guard tiling")
    if (
        not isinstance(encoded.metrics, Mapping)
        or encoded.metrics.get("prepared_binding") != json_copy(binding)
        or _sha(encoded.basis) != binding["basis_sha256"]
        or tensor_sha256(factors.coefficients, tile_rows=tile_rows) != binding["coefficients_sha256"]
        or _sha(_json_bytes(factors.metrics)) != binding["metrics_sha256"]
    ):
        raise IntegrityError("provider did not preserve the accepted factor binding")
    for name, value in json_copy(factors.metrics).items():
        if encoded.metrics.get(name) != value:
            raise IntegrityError("provider changed the accepted factor diagnostics")
    metadata_bytes = len(_json_bytes(metadata))
    payload_bytes = sum(map(len, encoded.payloads.values()))
    expected_bytes = {
        "source_bytes": math.prod(context.shape) * _ITEM_SIZES[context.dtype],
        "basis_bytes": len(encoded.basis), "payload_bytes": payload_bytes,
        "metadata_bytes": metadata_bytes,
        "representation_bytes": metadata_bytes + payload_bytes + len(encoded.basis),
    }
    if any(
        type(encoded.metrics.get(name)) is not int or encoded.metrics[name] != count
        for name, count in expected_bytes.items()
    ):
        raise IntegrityError("provider byte accounting differs from the actual prepared packet")
    diagnostics = {"hevc_validation_decode_count": 0, "decomposition_count": 0}
    if spec.kind == "svd":
        if _sha(encoded.payloads["coefficients.f32"]) != binding["coefficients_sha256"]:
            raise IntegrityError("provider substituted the accepted FP32 projection")
    else:
        _, affine, normalization = _hevc.quantize_coefficients(
            factors.coefficients.detach().cpu().numpy(),
            spec.normalization, spec.protected_fraction,
        )
        if (
            normalization != metadata["normalization"]
            or affine.astype("<f4", copy=False).tobytes() != encoded.payloads["affine.f32"]
        ):
            raise IntegrityError("provider changed the accepted coefficient normalization")
        streams = [
            encoded.payloads[f"coefficients.{index:03d}.hevc"]
            for index in range(metadata["video"]["segments"])
        ]
        pixels, seconds = _hevc.decode_pixels(
            streams, metadata["video"],
            ffmpeg_path=spec.ffmpeg_path or ffmpeg_path,
        )
        decoded = _hevc.dequantize_coefficients(pixels, affine, normalization)
        if _sha(decoded.astype("<f4", copy=False).tobytes()) != metadata["decoded_coefficients_sha256"]:
            raise IntegrityError("prepared HEVC packet failed independent coefficient validation")
        diagnostics.update(
            hevc_validation_decode_count=1, hevc_validation_decode_seconds=seconds,
        )
        if source is not None:
            value = validate_matrix(source, context)
            device = value.device
            coefficients = torch.from_numpy(decoded).to(device=device)
            basis = factors.basis[:spec.rank].to(device=device, dtype=torch.float32)
            source_energy = torch.zeros((), dtype=torch.float64, device=device)
            error_energy = torch.zeros_like(source_energy)
            scale = float(torch.maximum(value.amax().abs(), value.amin().abs()))
            for start in range(0, context.token_count, 1024):
                stop = min(start + 1024, context.token_count)
                original = value[start:stop].double()
                reconstructed = (coefficients[start:stop] @ basis).to(dtype=value.dtype).double()
                if scale:
                    original, reconstructed = original / scale, reconstructed / scale
                source_energy += original.square().sum()
                error_energy += (original - reconstructed).square().sum()
            error = 0.0 if not scale else math.sqrt(float(error_energy / source_energy))
            if not math.isfinite(error):
                raise IntegrityError("prepared HEVC reconstruction is not finite")
            diagnostics["post_hevc_relative_residual"] = error
    diagnostics["seconds"] = time.perf_counter() - started
    return diagnostics


def _positive_shape(value, dimensions, label):
    if (
        not isinstance(value, list) or len(value) != dimensions
        or any(type(d) is not int or d < 1 for d in value)
    ):
        raise IntegrityError(f"invalid {label} shape")
    return tuple(value)


def _validate_descriptor(row, raw, *, dtype, shape, basis=False, verify=True, external=False):
    """Check a payload descriptor against its bytes; ``external`` (raw None): structure only, the bytes are
    supplied and checked against the descriptor by the caller (externally staged pack bases)."""
    keys = {"bytes", "sha256", "dtype", "shape"} | ({"view_rank"} if basis else set())
    if not isinstance(row, dict) or set(row) != keys or not (raw is None if external else isinstance(raw, bytes)):
        raise IntegrityError("invalid payload descriptor fields")
    actual_shape = _positive_shape(row["shape"], len(shape), "payload")
    if actual_shape != tuple(shape) or row["dtype"] != dtype:
        raise IntegrityError("payload dtype/shape disagrees with representation")
    expected_bytes = math.prod(shape) * _ITEM_SIZES[dtype]
    if type(row["bytes"]) is not int or row["bytes"] != expected_bytes or (not external and len(raw) != expected_bytes):
        raise IntegrityError("payload byte count disagrees with dtype/shape")
    if external and (not isinstance(row["sha256"], str) or _HEX64.fullmatch(row["sha256"]) is None):
        raise IntegrityError("invalid external payload SHA256")
    if verify and not external and (
        not isinstance(row["sha256"], str) or len(row["sha256"]) != 64
        or any(c not in "0123456789abcdef" for c in row["sha256"])
        or _sha(raw) != row["sha256"]
    ):
        raise IntegrityError("payload SHA256 mismatch")


def _parse_packet(encoded):
    """Unverified fast path: read only what decoding needs; no cross-checks.

    Malformed metadata fails with ordinary lookup/reshape errors rather than a
    precise integrity diagnosis. Use ``verify_packet=True`` for full checks.
    """
    if not isinstance(encoded, EncodedTensor):
        raise ConfigurationError("encoded must be an EncodedTensor")
    metadata = encoded.metadata
    try:
        context = TensorContext.from_dict(metadata["context"])
        spec = CodecSpec.from_dict(metadata["codec"])
    except (ConfigurationError, UnsupportedError, KeyError, TypeError, ValueError) as exc:
        raise IntegrityError(f"cannot parse packet context or codec: {exc}") from exc
    return metadata, context, spec


def _validate_packet(encoded, *, verify=True):
    """Parse packet structure; ``verify`` additionally hashes every payload.

    Structural bounds (schema, geometry, payload names and exact byte counts)
    are always checked because decoding depends on them.
    """
    if not verify:
        return _parse_packet(encoded)
    if not isinstance(encoded, EncodedTensor):
        raise ConfigurationError("encoded must be an EncodedTensor")
    try:
        metadata = json_copy(encoded.metadata)
    except ConfigurationError as exc:
        raise IntegrityError("packet metadata is not finite JSON") from exc
    schema = metadata.get("schema") if isinstance(metadata, dict) else None
    if (
        schema not in TENSOR_SCHEMAS or set(metadata) != tensor_metadata_fields(schema)
        or metadata["byte_order"] != "little"
    ):
        raise IntegrityError("unsupported or malformed tensor packet schema")
    if schema == SCHEMA and isinstance(metadata["codec"], dict) and "inter_prediction" in metadata["codec"]:
        raise IntegrityError("version-1 tensor packets never record inter_prediction")
    try:
        context = TensorContext.from_dict(metadata["context"])
        spec = CodecSpec.from_dict(metadata["codec"])
        _validate_spec(spec)
    except (ConfigurationError, UnsupportedError, TypeError, ValueError) as exc:
        raise IntegrityError(f"invalid context or codec metadata: {exc}") from exc
    if schema == HEVC_SCHEMA and (
        spec.kind != "svd_hevc" or type(spec.inter_prediction) is not bool
        or _json_bytes(metadata["hevc"]) != _json_bytes(_hevc.encoder_record(spec.inter_prediction))
    ):
        raise IntegrityError("version-2 packet lacks its explicit svd_hevc GOP contract")
    canonical_codec = _codec_comparison_header(_portable_spec(spec))
    stored_codec = _codec_comparison_header(metadata["codec"])
    if (
        _json_bytes(context.to_dict()) != _json_bytes(metadata["context"])
        or _json_bytes(canonical_codec) != _json_bytes(stored_codec)
        or metadata["kind"] != spec.kind or metadata["dtype"] != context.dtype
        or _positive_shape(metadata["shape"], 2, "tensor") != context.shape
    ):
        raise IntegrityError("tensor identity, geometry, dtype, and codec disagree")
    if context.key.target == "K" and (
        context.key_space not in {"pre_rope", "post_rope"}
        or (spec.kind != "raw" and context.key_space != "pre_rope")
    ):
        raise IntegrityError("packet violates the pre-/post-RoPE representation contract")
    descriptors = metadata["payloads"]
    if not isinstance(descriptors, dict) or set(descriptors) != set(encoded.payloads):
        raise IntegrityError("packet has missing or uncharged extra payloads")
    if spec.kind == "raw":
        if (
            encoded.basis is not None or metadata["basis"] is not None
            or any(metadata[name] is not None for name in (
                "coefficient_shape", "normalization", "video", "decoded_coefficients_sha256",
            ))
            or metadata["factorization_domain"] != "raw"
            or set(descriptors) != {"tensor.bin"}
        ):
            raise IntegrityError("RAW packet contains inconsistent factorization fields")
        _validate_descriptor(
            descriptors["tensor.bin"], encoded.payloads["tensor.bin"],
            dtype=metadata["dtype"], shape=context.shape, verify=verify,
        )
        return metadata, context, spec
    try:
        validate_rank(spec.rank, context.shape)
    except ConfigurationError as exc:
        raise IntegrityError("factor rank exceeds matrix geometry") from exc
    if _positive_shape(metadata["coefficient_shape"], 2, "coefficient") != (context.token_count, spec.rank):
        raise IntegrityError("coefficient shape differs from rank or token count")
    domain = "normalized_pre_rope_k" if context.key.target == "K" else "direct_v"
    if metadata["factorization_domain"] != domain:
        raise IntegrityError("invalid factorization domain")
    basis_row = metadata["basis"]
    if not isinstance(basis_row, dict) or "shape" not in basis_row:
        raise IntegrityError("factorization packet lacks its canonical basis descriptor")
    canonical_rank, channels = _positive_shape(basis_row["shape"], 2, "basis")
    if (
        channels != context.channels or not spec.rank <= canonical_rank <= channels
        or type(basis_row.get("view_rank")) is not int or basis_row["view_rank"] != spec.rank
    ):
        raise IntegrityError("canonical basis shape/view rank is incompatible")
    _validate_descriptor(
        basis_row, encoded.basis, dtype="float16", shape=(canonical_rank, channels), basis=True,
        verify=verify,
    )
    decoded_hash = metadata["decoded_coefficients_sha256"]
    # svd_hevc packets encoded without the (default-off) validation decode carry no checksum.
    if not (decoded_hash is None and spec.kind == "svd_hevc") and (
        not isinstance(decoded_hash, str) or len(decoded_hash) != 64
        or any(c not in "0123456789abcdef" for c in decoded_hash)
    ):
        raise IntegrityError("invalid decoded coefficient checksum")
    if spec.kind == "svd":
        if set(descriptors) != {"coefficients.f32"} or any(
            metadata[name] is not None for name in ("normalization", "video")
        ):
            raise IntegrityError("SVD-only packet contains codec fields or extra payloads")
        _validate_descriptor(
            descriptors["coefficients.f32"], encoded.payloads["coefficients.f32"],
            dtype="float32", shape=(context.token_count, spec.rank), verify=verify,
        )
        if verify and _sha(encoded.payloads["coefficients.f32"]) != decoded_hash:
            raise IntegrityError("SVD coefficient reconstruction checksum mismatch")
    else:
        expected_normalization = _hevc.normalization_layout(
            spec.rank, spec.normalization, spec.protected_fraction,
        )
        try:
            expected_video = _hevc.video_layout(
                context.frames, context.height, context.width, spec.rank, spec.pix_fmt,
            )
        except (ConfigurationError, UnsupportedError) as exc:
            raise IntegrityError("invalid HEVC geometry") from exc
        if (
            _json_bytes(expected_normalization) != _json_bytes(metadata["normalization"])
            or _json_bytes(expected_video) != _json_bytes(metadata["video"])
        ):
            raise IntegrityError("coefficient normalization/video layout differs from the codec contract")
        stream_names = {
            f"coefficients.{index:03d}.hevc" for index in range(expected_video["segments"])
        }
        if set(descriptors) != stream_names | {"affine.f32"}:
            raise IntegrityError("HEVC payload names/segment count differ from the layout")
        _validate_descriptor(
            descriptors["affine.f32"], encoded.payloads["affine.f32"],
            dtype="float32", shape=expected_normalization["affine_shape"], verify=verify,
        )
        for name in stream_names:
            raw = encoded.payloads[name]
            _validate_descriptor(descriptors[name], raw, dtype="uint8", shape=(len(raw),), verify=verify)
            if verify and schema == HEVC_SCHEMA:
                if spec.inter_prediction:
                    if len(_hevc.access_units(raw)) != context.frames:
                        raise IntegrityError("HEVC stream picture count differs from its layout")
                else:
                    _hevc.all_intra_access_units(raw, context.frames)
    return metadata, context, spec


def _column_selection(columns, channels):
    if columns is None:
        return slice(None)
    if isinstance(columns, slice):
        start = 0 if columns.start is None else columns.start
        stop = channels if columns.stop is None else columns.stop
        step = 1 if columns.step is None else columns.step
        if (
            any(type(v) is not int for v in (start, stop, step))
            or not 0 <= start < stop <= channels or step < 1
        ):
            raise ConfigurationError("columns slice must be nonempty and within native KV channels")
        return slice(start, stop, step)
    if isinstance(columns, torch.Tensor):
        if columns.ndim != 1 or columns.dtype not in (torch.int32, torch.int64):
            raise ConfigurationError("columns tensor must be one-dimensional integer indices")
        columns = columns.detach().cpu().tolist()
    elif isinstance(columns, np.ndarray):
        if columns.ndim != 1 or columns.dtype.kind not in "iu":
            raise ConfigurationError("columns array must be one-dimensional integer indices")
        columns = columns.tolist()
    if not isinstance(columns, Sequence) or isinstance(columns, (str, bytes)):
        raise ConfigurationError("columns must be a slice or an index sequence")
    if (
        not columns or any(type(v) is not int or not 0 <= v < channels for v in columns)
        or len(set(columns)) != len(columns)
    ):
        raise ConfigurationError("columns must be unique native KV-channel indices; no head expansion")
    return list(columns)


def _decode_columns(encoded, *, device, dtype, column_groups, nvdec=None, verify_decoded_coefficients=False,
                    verify_packet=False):
    if type(verify_decoded_coefficients) is not bool or type(verify_packet) is not bool:
        raise ConfigurationError("verification options must be boolean")
    if nvdec is not None and not isinstance(nvdec, NVDECDecoder):
        raise ConfigurationError("nvdec must be an explicitly configured NVDECDecoder")
    if _is_pack(encoded):
        with span("decode.validate"):
            pack = _validate_pack(encoded, verify=verify_packet)
        if len(pack["tiles"]) != 1:
            raise ConfigurationError("packed packet carries several matrices; use decode_packed")
        _, tile, context, spec = pack["tiles"][0]
        selections, output_dtype, output_device = _decode_targets(
            context, column_groups, dtype, device, tile["dtype"],
        )
        basis, coefficients = _pack_factors(
            encoded, pack, [0], verify_decoded_coefficients=verify_decoded_coefficients,
            verify_packet=verify_packet, nvdec=nvdec,
        )[0]
        return _reconstruct(
            basis, coefficients, spec, selections, output_device, output_dtype, verify_packet,
        )
    with span("decode.validate"):
        metadata, context, spec = _validate_packet(encoded, verify=verify_packet)
    selections, output_dtype, output_device = _decode_targets(
        context, column_groups, dtype, device, metadata["dtype"],
    )
    if spec.kind == "raw":
        results = []
        for selected in selections:
            with span("decode.reconstruct"):
                result = _from_bytes(
                    encoded.payloads["tensor.bin"], metadata["dtype"], context.shape, columns=selected,
                )
                results.append(result.to(device=output_device, dtype=output_dtype))
        return results
    with span("decode.basis", nbytes=len(encoded.basis)):
        basis = _from_bytes(encoded.basis, "float16", metadata["basis"]["shape"])
        if verify_packet:
            if not bool(torch.isfinite(basis).all()):
                raise IntegrityError("stored FP16 basis is nonfinite")
            condition = basis_condition_bound(basis[:spec.rank])
            if not math.isfinite(condition) or condition > 1e6:
                raise IntegrityError("stored FP16 basis is singular or ill-conditioned")
    if spec.kind == "svd":
        with span("decode.coefficients", nbytes=len(encoded.payloads["coefficients.f32"])):
            coefficients = _from_bytes(
                encoded.payloads["coefficients.f32"], "float32",
                (context.token_count, spec.rank),
            )
    else:
        affine = np.frombuffer(encoded.payloads["affine.f32"], dtype="<f4").reshape(
            metadata["normalization"]["affine_shape"],
        )
        if verify_packet:
            _hevc.validate_affine(affine, metadata["normalization"], spec.rank)
        streams = [
            encoded.payloads[f"coefficients.{index:03d}.hevc"]
            for index in range(metadata["video"]["segments"])
        ]
        with span("decode.hevc", nbytes=sum(map(len, streams))):
            pixels, _ = (_hevc.decode_pixels(streams, metadata["video"]) if nvdec is None
                         else nvdec.decode_pixels(streams, metadata["video"]))
        with span("decode.dequantize", nbytes=pixels.nbytes):
            if isinstance(pixels, torch.Tensor):
                with cuda_span("decode.dequantize", pixels.device):
                    coefficients = dequantize_device_coefficients(
                        pixels, affine, metadata["normalization"], validate=verify_packet,
                    )
                if verify_decoded_coefficients:
                    if metadata["decoded_coefficients_sha256"] is None:
                        raise IntegrityError("packet was encoded without validation decode; no checksum to verify")
                    with span("decode.checksum_readback"):
                        if _sha(_tensor_bytes(coefficients)) != metadata["decoded_coefficients_sha256"]:
                            raise IntegrityError("HEVC decoded coefficients differ from their authoritative checksum")
            else:
                decoded = _hevc.dequantize_coefficients(pixels, affine, metadata["normalization"])
                digest = metadata["decoded_coefficients_sha256"]
                if digest is not None and _sha(decoded.astype("<f4", copy=False).tobytes()) != digest:
                    raise IntegrityError("HEVC decoded coefficients differ from their authoritative checksum")
                coefficients = torch.from_numpy(decoded)
    return _reconstruct(basis, coefficients, spec, selections, output_device, output_dtype, verify_packet)


def _decode_targets(context, column_groups, dtype, device, stored_dtype):
    selections = [_column_selection(columns, context.channels) for columns in column_groups]
    if len(selections) > 1:
        indices = [
            list(range(*selected.indices(context.channels))) if isinstance(selected, slice) else selected
            for selected in selections
        ]
        flattened = [column for group in indices for column in group]
        if len({len(group) for group in indices}) != 1 or len(set(flattened)) != len(flattened):
            raise ConfigurationError("column groups must have equal widths and no repeated channels")
    output_dtype = _DTYPES[canonical_dtype(dtype)] if dtype is not None else _DTYPES[stored_dtype]
    try:
        output_device = torch.device(device)
    except (TypeError, RuntimeError) as exc:
        raise ConfigurationError("invalid decode device") from exc
    if output_device.type == "meta":
        raise ConfigurationError("decoding to a meta device would discard the payload")
    return selections, output_dtype, output_device


def _reconstruct(basis, coefficients, spec, selections, output_device, output_dtype, verify_packet):
    if verify_packet and not bool(torch.isfinite(coefficients).all()):
        raise IntegrityError("decoded coefficients are nonfinite")
    results = []
    for selected in selections:
        with span("decode.transfer"), cuda_span("decode.transfer", output_device):
            local_basis = basis[:spec.rank, selected].to(device=output_device, dtype=torch.float32)
            coefficients = coefficients.to(device=output_device)
        with span("decode.reconstruct"), cuda_span("decode.reconstruct", output_device):
            result = coefficients @ local_basis
            result = result.to(dtype=output_dtype)
            if verify_packet and not bool(torch.isfinite(result).all()):
                raise IntegrityError("reconstruction is nonfinite in the requested dtype")
            results.append(result)
    return results


@torch.no_grad()
def decode_tensor(encoded, *, device="cpu", dtype=None, columns=None, nvdec=None,
                  verify_decoded_coefficients=False, verify_packet=False) -> torch.Tensor:
    """Decode a packet and reconstruct only the requested columns.

    ``verify_packet=True`` re-hashes payloads, checks basis finiteness and
    numerical conditioning, and checks finite outputs. Structural bounds needed
    for safe decoding always run. Stores already hash-verify objects on read.
    HEVC decodes full coefficient streams; no random-access decoding is implied.
    """
    return _decode_columns(encoded, device=device, dtype=dtype, column_groups=(columns,), nvdec=nvdec,
                           verify_decoded_coefficients=verify_decoded_coefficients,
                           verify_packet=verify_packet)[0]


@torch.no_grad()
def decode_column_views(encoded, *, column_groups, device="cpu", dtype=None, nvdec=None,
                        verify_decoded_coefficients=False, verify_packet=False) -> torch.Tensor:
    """Decode once, preserving separate equal-width, disjoint column products.

    Returns contiguous [groups, tokens, columns] storage. All group outputs and
    packing workspace must be admitted by the caller; this is not a persistent
    factor cache or a smaller-wire representation.
    """
    if not isinstance(column_groups, (tuple, list)) or not column_groups:
        raise ConfigurationError("column_groups must be a nonempty sequence")
    views = _decode_columns(encoded, device=device, dtype=dtype, column_groups=column_groups, nvdec=nvdec,
                            verify_decoded_coefficients=verify_decoded_coefficients,
                            verify_packet=verify_packet)
    return torch.stack(views)


def dequantize_device_coefficients(pixels, affine, layout, *, validate=True):
    if validate:
        if pixels.dtype != torch.uint8 or pixels.ndim != 2 or min(pixels.shape) <= 0:
            raise IntegrityError("device coefficient pixels must be a nonempty uint8 matrix")
        _hevc.validate_affine(affine, layout, pixels.shape[1])
    scale, offset = _hevc._expanded_affine(affine, layout, pixels.shape[1])
    scales = torch.from_numpy(scale.copy()).to(pixels.device)
    offsets = torch.from_numpy(offset.copy()).to(pixels.device)
    # Separate eager operations preserve the CPU multiplication-then-add rounding.
    return pixels.float().mul_(scales).add_(offsets)


# Packet schema v2 for several matrices: one all-intra stream per node or layer group.

@dataclass(frozen=True)
class NodeEncoding:
    """Packets of one node and each matrix's ``(packet index, tile index | None)``.

    Keys are ``MatrixKey`` or ``(MatrixKey, stream label)`` for CFG streams.
    A tile index of None marks an ordinary single-matrix packet.
    """

    packets: tuple
    locations: Mapping
    packing: str
    layout: str | None


def _validate_packing(packing, layer_group_size, layout):
    if not isinstance(packing, str) or packing not in PACKING_MODES:
        raise ConfigurationError(f"packing must be one of {PACKING_MODES}")
    if packing == "layer_group":
        if type(layer_group_size) is not int or layer_group_size < 1:
            raise ConfigurationError("layer_group packing requires a positive integer layer_group_size")
    elif layer_group_size is not None:
        raise ConfigurationError("layer_group_size applies only to layer_group packing")
    if not isinstance(layout, str) or layout not in PACKED_LAYOUTS:
        raise ConfigurationError(f"packed layout must be one of {PACKED_LAYOUTS}")


def _node_key(key):
    if isinstance(key, MatrixKey):
        return key, None
    if (
        isinstance(key, tuple) and len(key) == 2 and isinstance(key[0], MatrixKey)
        and isinstance(key[1], str) and key[1] and key[1] == key[1].strip()
    ):
        return key
    raise ConfigurationError("node matrix keys must be MatrixKey or (MatrixKey, stream label)")


def _public_key(key, stream):
    return key if stream is None else (key, stream)


def _key_order(item):
    key, stream = item
    return key.layer, key.target, stream is not None, stream or ""


def _node_items(matrices, contexts, specs):
    if not isinstance(matrices, Mapping) or not isinstance(contexts, Mapping) or not matrices:
        raise ConfigurationError("node encoding requires nonempty matrix and context mappings")
    if set(matrices) != set(contexts) or (
        not isinstance(specs, CodecSpec) and (not isinstance(specs, Mapping) or set(specs) != set(matrices))
    ):
        raise ConfigurationError("node matrices, contexts and specs must cover the same keys")
    items = {}
    for public in matrices:
        canonical = _node_key(public)
        if canonical in items:
            raise ConfigurationError("duplicate node matrix key")
        context = contexts[public]
        spec = specs if isinstance(specs, CodecSpec) else specs[public]
        if not isinstance(context, TensorContext) or not isinstance(spec, CodecSpec):
            raise ConfigurationError("node entries require TensorContext and CodecSpec values")
        if context.key != canonical[0]:
            raise ConfigurationError("node matrix key differs from its tensor context")
        items[canonical] = (public, context, spec)
    return [(key, *items[key]) for key in sorted(items, key=_key_order)]


def _node_descriptor(contexts, node_id):
    first = contexts[0]
    for context in contexts[1:]:
        if (
            context.model != first.model or context.token_start != first.token_start
            or (context.frames, context.height, context.width) != (first.frames, first.height, first.width)
        ):
            raise ConfigurationError(
                "a packet cannot span two nodes: matrices differ in model, token range or geometry",
            )
    if node_id is not None and (not isinstance(node_id, str) or not node_id.strip()):
        raise ConfigurationError("node_id must be a nonempty string or None")
    return {
        "model": first.model.to_dict(), "token_start": first.token_start,
        "frames": first.frames, "height": first.height, "width": first.width, "node_id": node_id,
    }


def _pack_codec(specs, encoder=None):
    first = specs[0]
    for spec in specs:
        _validate_spec(spec)
        _require_encodable(spec, encoder)
        if spec.kind != "svd_hevc":
            raise ConfigurationError("packed streams carry only svd_hevc coefficient matrices")
        if spec.inter_prediction:
            raise ConfigurationError(
                "inter prediction is rejected for packed streams; use per_matrix packets for inter coding",
            )
        if (spec.pix_fmt, spec.qp, spec.preset, spec.ffmpeg_path) != (
            first.pix_fmt, first.qp, first.preset, first.ffmpeg_path,
        ):
            raise ConfigurationError("one packed stream requires one pixel format, QP, preset and FFmpeg")
    return {
        "kind": "svd_hevc", "pix_fmt": first.pix_fmt, "qp": first.qp, "preset": first.preset,
        "inter_prediction": False,
    }


def _pack_padding(tiles, canvas):
    planes = canvas["planes"]
    canvas_samples = canvas["pictures"] * canvas["width"] * canvas["height"] * planes
    tile_samples = sum(tile["pictures"][1] * tile["rect"][2] * tile["rect"][3] * planes for tile in tiles)
    coefficient_samples = sum(math.prod(tile["coefficient_shape"]) for tile in tiles)
    return {
        "unit": "uncompressed 8-bit samples in coefficient planes",
        "canvas_samples": canvas_samples, "tile_samples": tile_samples,
        "coefficient_samples": coefficient_samples,
        "canvas_padding_bytes": canvas_samples - tile_samples,
        "mosaic_padding_bytes": tile_samples - coefficient_samples,
        "canvas_padding_fraction": (canvas_samples - tile_samples) / canvas_samples,
    }


def _pack_placements(layouts, layout, canvas):
    """Canvas size and per-tile ``(pictures, rect)`` for temporal or spatial packing."""
    alignment, floor = _hevc.CANVAS_ALIGNMENT, _hevc.MIN_CANVAS
    cell_width = _hevc.align(max(item["image_width"] for item in layouts))
    cell_height = _hevc.align(max(item["image_height"] for item in layouts))
    if layout == "temporal":
        width, height = max(cell_width, floor), max(cell_height, floor)
        if canvas is not None:
            if (
                not isinstance(canvas, (tuple, list)) or len(canvas) != 2
                or any(type(value) is not int or value < 1 or value % alignment for value in canvas)
                or canvas[0] < width or canvas[1] < height
            ):
                raise ConfigurationError(
                    f"canvas must be (width, height) multiples of {alignment}, at least {floor}x{floor} "
                    "(NVDEC minimum picture), covering every tile",
                )
            width, height = canvas
        placements, cursor = [], 0
        for item in layouts:
            count = item["frames"] * item["segments"]
            placements.append(([cursor, count], [0, 0, item["image_width"], item["image_height"]]))
            cursor += count
        pictures = cursor
    else:
        if canvas is not None:
            raise ConfigurationError("spatial packing derives its own canvas")
        if any(item["segments"] != 1 for item in layouts):
            raise UnsupportedError("spatial packing requires every matrix to fit one channel-tile segment")
        columns = max(1, min(len(layouts), _hevc.MAX_DIMENSION // cell_width))
        rows = math.ceil(len(layouts) / columns)
        width, height = max(columns * cell_width, floor), max(rows * cell_height, floor)
        placements = []
        for index, item in enumerate(layouts):
            row, column = divmod(index, columns)
            placements.append((
                [0, item["frames"]],
                [column * cell_width, row * cell_height, item["image_width"], item["image_height"]],
            ))
        pictures = layouts[0]["frames"]
    if width > _hevc.MAX_DIMENSION or height > _hevc.MAX_DIMENSION:
        raise UnsupportedError("packed canvas exceeds the HEVC provider's maximum picture dimension")
    return width, height, pictures, placements


@torch.no_grad()
def encode_packed(
    matrices, contexts, specs, *, layout="temporal", packing="node", node_id=None,
    reuse_bases=None, reuse_threshold=None, prepared=None, fitter=None,
    validate_hevc=False, hevc_threads=1, canvas=None, encoder=None,
) -> EncodedTensor:
    """Encode one or many svd_hevc matrices of ONE node into one all-intra stream.

    ``layout="temporal"`` (default) keeps each matrix's own pictures at the
    top-left of a common CTU-aligned canvas (the maximum over the matrices, at
    least ``_hevc.MIN_CANVAS`` square so NVDEC can decode it, or an explicit
    larger ``canvas``): its decoded tiles are pixel-identical to a
    single-matrix packet encoded on the same canvas. ``layout="spatial"`` puts
    the matrices of each frame side by side in one picture; intra prediction
    and CABAC state then cross tile edges, so it is NOT pixel-identical.
    Inter prediction is rejected. ``packing`` is a provenance label: "node",
    "layer_group" or "explicit" (a caller-chosen subset such as a reference).
    """
    started = time.perf_counter()
    if packing not in ("node", "layer_group", "explicit"):
        raise ConfigurationError("packed packing label must be node, layer_group or explicit")
    if not isinstance(layout, str) or layout not in PACKED_LAYOUTS:
        raise ConfigurationError(f"packed layout must be one of {PACKED_LAYOUTS}")
    if type(validate_hevc) is not bool:
        raise ConfigurationError("validate_hevc must be boolean")
    items = _node_items(matrices, contexts, specs)
    node = _node_descriptor([item[2] for item in items], node_id)
    codec = _pack_codec([item[3] for item in items], encoder)
    for name, values in (("reuse_bases", reuse_bases), ("prepared", prepared)):
        if values is not None and (not isinstance(values, Mapping) or not set(values) <= set(matrices)):
            raise ConfigurationError(f"{name} must map a subset of the node's matrix keys")
    reuse_bases, prepared = reuse_bases or {}, prepared or {}
    validate_hevc = validate_hevc or bool(prepared)
    tiles, sources, factor_metrics, quantized = [], [], [], []
    source_bytes = 0
    for index, ((key, stream_label), public, context, spec) in enumerate(items):
        prepared_factors, prepared_binding = prepared.get(public, (None, None))
        value = _check_encode_request(
            matrices[public], context, spec, reuse_basis=reuse_bases.get(public),
            reuse_threshold=reuse_threshold, prepared_factors=prepared_factors,
            prepared_binding=prepared_binding, fitter=fitter, validate_hevc=validate_hevc,
            hevc_threads=hevc_threads, encoder=encoder,
        )
        metrics = _initial_metrics(spec)
        factors = _fit_factors(
            value, context, spec, metrics, reuse_basis=reuse_bases.get(public),
            reuse_threshold=reuse_threshold, prepared_factors=prepared_factors,
            prepared_binding=prepared_binding, fitter=fitter,
        )
        pixels, affine, normalization, video = _quantized_tiles(factors, context, spec, metrics)
        source_bytes += value.numel() * value.element_size()
        sources.append((_tensor_bytes(factors.basis), affine.astype("<f4", copy=False).tobytes()))
        quantized.append((pixels, affine, normalization, video))
        factor_metrics.append(metrics)
        tiles.append({
            "key": key.to_dict(), "stream": stream_label, "context": context.to_dict(),
            "codec": _portable_spec(spec), "shape": list(value.shape),
            "dtype": canonical_dtype(value.dtype), "factorization_domain": _factorization_domain(context),
            "coefficient_shape": list(factors.coefficients.shape),
            "basis": {"payload": f"basis.{index:03d}.f16", "view_rank": spec.rank},
            "affine": f"affine.{index:03d}.f32", "normalization": normalization, "video": video,
            "pictures": None, "rect": None, "byte_range": None,
            "decoded_coefficients_sha256": None,
            "_basis_shape": list(factors.basis.shape),
        })
    width, height, pictures, placements = _pack_placements(
        [item[3] for item in quantized], layout, canvas,
    )
    planes = 3 if codec["pix_fmt"] == "yuv444p" else 1
    images = np.zeros((pictures, planes, height, width), dtype=np.uint8)
    for tile, (pixels, _, _, video), (span_pictures, rect) in zip(tiles, quantized, placements):
        tile["pictures"], tile["rect"] = span_pictures, rect
        x, y, tile_width, tile_height = rect
        values = pixels.reshape(video["frames"], video["height"], video["width"], video["rank"])
        for segment in range(video["segments"]):
            first = span_pictures[0] + segment * video["frames"]
            images[first:first + video["frames"], :, y:y + tile_height, x:x + tile_width] = _hevc._mosaic(
                values, video, segment,
            )
    ffmpeg_path = items[0][3].ffmpeg_path
    with span("pack.encode", nbytes=images.nbytes):
        if encoder is None:
            stream, encode_seconds, version = _hevc.encode_pictures(
                images, codec["pix_fmt"], ffmpeg_path=ffmpeg_path, threads=hevc_threads,
            )
        else:
            stream, encode_seconds, version = encoder.encode_pictures(images, codec["pix_fmt"])
    del images
    units = _hevc.all_intra_access_units(stream, pictures)
    random_access = layout == "temporal" and all(unit["self_contained"] for unit in units)
    if random_access:
        for tile in tiles:
            first, count = tile["pictures"]
            start, last = units[first], units[first + count - 1]
            tile["byte_range"] = [start["offset"], last["offset"] + last["length"] - start["offset"]]
    decode_seconds = 0.0
    if validate_hevc:
        decoded, decode_seconds = _hevc.decode_pictures(
            stream, pictures=pictures, height=height, width=width, pix_fmt=codec["pix_fmt"],
            ffmpeg_path=ffmpeg_path,
        )
        for tile, (_, affine, normalization, video) in zip(tiles, quantized):
            restored = _unpack_tile(decoded, tile, tile["pictures"][0], video)
            coefficients = _hevc.dequantize_coefficients(restored, affine, normalization)
            tile["decoded_coefficients_sha256"] = _sha(coefficients.astype("<f4", copy=False).tobytes())
        del decoded
    payloads = {"stream.hevc": stream}
    descriptors = {"stream.hevc": _descriptor(stream, "uint8", (len(stream),))}
    basis_bytes = 0
    for tile, (basis, affine) in zip(tiles, sources):
        basis_shape = tile.pop("_basis_shape")
        payloads[tile["basis"]["payload"]] = basis
        descriptors[tile["basis"]["payload"]] = _descriptor(basis, "float16", basis_shape)
        payloads[tile["affine"]] = affine
        descriptors[tile["affine"]] = _descriptor(affine, "float32", tile["normalization"]["affine_shape"])
        basis_bytes += len(basis)
    canvas_record = {
        "width": width, "height": height, "alignment": _hevc.CANVAS_ALIGNMENT,
        "pictures": pictures, "planes": planes, "pix_fmt": codec["pix_fmt"], "fill": 0,
    }
    metadata = {
        "schema": PACK_SCHEMA, "byte_order": "little", "node": node, "codec": codec,
        "hevc": _hevc.encoder_record(False) if encoder is None else encoder.encoder_record(codec["pix_fmt"]),
        "packing": {
            "mode": packing, "layout": layout,
            "pixel_identical_to_same_canvas_matrix": layout == "temporal",
            "tile_order": "layer,target,stream",
            "random_access": "tile_byte_ranges" if random_access else "whole_stream",
        },
        "canvas": canvas_record, "tiles": tiles, "payloads": descriptors,
        "padding": _pack_padding(tiles, canvas_record),
    }
    metadata_bytes = len(_json_bytes(metadata))
    payload_bytes = sum(map(len, payloads.values()))
    metrics = {
        "tiles": factor_metrics, "tile_count": len(tiles), "pictures": pictures,
        "hevc_encode_seconds": encode_seconds, "hevc_encode_threads": hevc_threads,
        "hevc_validation_decode": validate_hevc, "hevc_validation_decode_seconds": decode_seconds,
        "hevc_inter_prediction": False, "hevc_picture_structure": "all_intra_idr",
        "ffmpeg_version": version, "hevc_software_only": True,
        "stream_bytes": len(stream), "basis_bytes": basis_bytes,
        "source_bytes": source_bytes, "payload_bytes": payload_bytes,
        "metadata_bytes": metadata_bytes, "representation_bytes": metadata_bytes + payload_bytes,
        "encode_seconds": time.perf_counter() - started,
    }
    if encoder is not None:
        metrics.update(ffmpeg_version=None, hevc_software_only=False, hevc_encoder=encoder.name,
                       hevc_encoder_version=version)
    return EncodedTensor(metadata, payloads, None, metrics)


@torch.no_grad()
def encode_node(
    matrices, contexts, specs, *, packing="node", layer_group_size=None, layout="temporal",
    node_id=None, reuse_bases=None, reuse_threshold=None, prepared=None, fitter=None,
    validate_hevc=False, hevc_threads=1, encoder=None,
) -> NodeEncoding:
    """Encode every matrix of one node under an explicit packing mode.

    ``node`` (default, chosen from the Step 5a NVDEC measurements) packs all
    svd_hevc matrices into one all-intra temporal stream with per-tile byte
    ranges; ``layer_group`` packs each run of ``layer_group_size`` layers;
    ``per_matrix`` emits one independent packet per matrix, exactly as
    :func:`encode_tensor` (required for inter-coded matrices, which the
    packed modes reject; low-rank own-size packets can fall below NVDEC's
    144x144 minimum picture). RAW/SVD matrices have no stream and stay
    single-matrix packets. ``encoder`` is an explicit packed-stream encoder
    (``reelcache.nvenc.NVENCEncoder``) for its own specifications; None is libx265.
    """
    _validate_packing(packing, layer_group_size, layout)
    if encoder is not None and packing == "per_matrix":
        raise ConfigurationError("an explicit packed-stream encoder requires node or layer_group packing")
    items = _node_items(matrices, contexts, specs)
    _node_descriptor([item[2] for item in items], node_id)
    for name, values in (("reuse_bases", reuse_bases), ("prepared", prepared)):
        if values is not None and (not isinstance(values, Mapping) or not set(values) <= set(matrices)):
            raise ConfigurationError(f"{name} must map a subset of the node's matrix keys")
    reuse_bases, prepared = reuse_bases or {}, prepared or {}
    packets, locations = [], {}

    def single(public, context, spec):
        prepared_factors, prepared_binding = prepared.get(public, (None, None))
        locations[public] = (len(packets), None)
        packets.append(encode_tensor(
            matrices[public], context, spec, reuse_basis=reuse_bases.get(public),
            reuse_threshold=reuse_threshold, prepared_factors=prepared_factors,
            prepared_binding=prepared_binding, fitter=fitter, validate_hevc=validate_hevc,
            hevc_threads=hevc_threads,
        ))

    if packing == "per_matrix":
        for _, public, context, spec in items:
            single(public, context, spec)
        return NodeEncoding(tuple(packets), MappingProxyType(locations), packing, None)
    hevc = [item for item in items if item[3].kind == "svd_hevc"]
    if any(item[3].inter_prediction for item in hevc):
        raise ConfigurationError(
            "inter prediction is rejected for packed streams; use packing='per_matrix' for inter coding",
        )
    groups = {}
    for item in hevc:
        group = 0 if packing == "node" else item[0][0].layer // layer_group_size
        groups.setdefault(group, []).append(item)
    for group in sorted(groups):
        members = groups[group]
        publics = [item[1] for item in members]
        packet = encode_packed(
            {public: matrices[public] for public in publics},
            {item[1]: item[2] for item in members}, {item[1]: item[3] for item in members},
            layout=layout, packing=packing, node_id=node_id,
            reuse_bases={public: reuse_bases[public] for public in publics if public in reuse_bases},
            reuse_threshold=reuse_threshold,
            prepared={public: prepared[public] for public in publics if public in prepared},
            fitter=fitter, validate_hevc=validate_hevc, hevc_threads=hevc_threads, encoder=encoder,
        )
        for tile_index, public in enumerate(publics):
            locations[public] = (len(packets), tile_index)
        packets.append(packet)
    for _, public, context, spec in items:
        if spec.kind != "svd_hevc":
            single(public, context, spec)
    return NodeEncoding(tuple(packets), MappingProxyType(locations), packing, layout)


def _is_pack(encoded):
    return (
        isinstance(encoded, EncodedTensor) and isinstance(encoded.metadata, Mapping)
        and encoded.metadata.get("schema") == PACK_SCHEMA
    )


def _int_list(value, length, label, *, minimum=0):
    if (
        not isinstance(value, list) or len(value) != length
        or any(type(item) is not int or item < minimum for item in value)
    ):
        raise IntegrityError(f"invalid packed {label}")
    return value


def _validate_pack(encoded, *, verify=True, external_bases=False):
    """Validate a packed packet; structural checks always run, hashing on ``verify``.

    ``external_bases``: the tile basis payloads are not in ``encoded.payloads`` (a restore stages them from the
    basis store, e.g. ``reelcache.restore_staging``); their descriptors are checked structurally here and the caller
    checks the staged bytes against each descriptor's length and SHA256.
    """
    if not isinstance(encoded, EncodedTensor):
        raise ConfigurationError("encoded must be an EncodedTensor")
    if type(external_bases) is not bool:
        raise ConfigurationError("external_bases must be boolean")
    try:
        metadata = json_copy(encoded.metadata) if verify else encoded.metadata
        if (
            not isinstance(metadata, Mapping) or set(metadata) != _PACK_FIELDS
            or metadata["schema"] != PACK_SCHEMA or metadata["byte_order"] != "little"
            or encoded.basis is not None
        ):
            raise IntegrityError("unsupported or malformed packed tensor schema")
        codec = metadata["codec"]
        if not isinstance(codec, Mapping) or set(codec) != {"kind", "pix_fmt", "qp", "preset", "inter_prediction"}:
            raise IntegrityError("invalid packed stream codec")
        if codec["inter_prediction"] is not False:
            raise IntegrityError("inter prediction is rejected for packed streams")
        if codec["kind"] != "svd_hevc":
            raise IntegrityError("packed streams carry only svd_hevc coefficient matrices")
        record = metadata["hevc"]
        if isinstance(record, Mapping) and record.get("encoder") == "nvenc":
            from .nvenc import validate_pack_contract  # explicit NVENC streams (Step 15)
            validate_pack_contract(codec, record)
        else:
            try:
                _hevc.validate_parameters(codec["pix_fmt"], codec["qp"], codec["preset"])
            except UnsupportedError as exc:
                raise IntegrityError(f"invalid packed stream codec: {exc}") from exc
            if _canon(record) != _canon(_hevc.encoder_record(False)):
                raise IntegrityError("packed stream does not declare the all-intra encoder contract")
        node = metadata["node"]
        if (
            not isinstance(node, Mapping)
            or set(node) != {"model", "token_start", "frames", "height", "width", "node_id"}
            or (node["node_id"] is not None and (not isinstance(node["node_id"], str) or not node["node_id"].strip()))
        ):
            raise IntegrityError("invalid packed node descriptor")
        canvas = metadata["canvas"]
        alignment = _hevc.CANVAS_ALIGNMENT
        planes = 3 if codec["pix_fmt"] == "yuv444p" else 1
        if (
            not isinstance(canvas, Mapping)
            or set(canvas) != {"width", "height", "alignment", "pictures", "planes", "pix_fmt", "fill"}
            or canvas["alignment"] != alignment or canvas["planes"] != planes
            or canvas["pix_fmt"] != codec["pix_fmt"] or canvas["fill"] != 0
            or any(
                type(canvas[name]) is not int or canvas[name] < 1
                for name in ("width", "height", "pictures")
            )
            or canvas["width"] % alignment or canvas["height"] % alignment
            or canvas["width"] > _hevc.MAX_DIMENSION or canvas["height"] > _hevc.MAX_DIMENSION
        ):
            raise IntegrityError("packed canvas violates its CTU alignment or geometry contract")
        packing = metadata["packing"]
        if (
            not isinstance(packing, Mapping)
            or set(packing) != {"mode", "layout", "pixel_identical_to_same_canvas_matrix", "tile_order", "random_access"}
            or packing["mode"] not in ("node", "layer_group", "explicit")
            or packing["layout"] not in PACKED_LAYOUTS
            or packing["pixel_identical_to_same_canvas_matrix"] is not (packing["layout"] == "temporal")
            or packing["tile_order"] != "layer,target,stream"
            or packing["random_access"] not in ("tile_byte_ranges", "whole_stream")
            or (packing["random_access"] == "tile_byte_ranges" and packing["layout"] != "temporal")
        ):
            raise IntegrityError("invalid packed layout descriptor")
        temporal = packing["layout"] == "temporal"
        ranged = packing["random_access"] == "tile_byte_ranges"
        tiles = metadata["tiles"]
        if not isinstance(tiles, list) or not tiles:
            raise IntegrityError("packed packets require a nonempty tile map")
        descriptors = metadata["payloads"]
        names, basis_names = {"stream.hevc"}, set()
        stream = encoded.payloads.get("stream.hevc")
        if not isinstance(stream, bytes) or not isinstance(descriptors, Mapping):
            raise IntegrityError("packed packet lacks its HEVC stream")
        node_model = _canon(node["model"])
        parsed, seen, cursor, byte_cursor = [], set(), 0, 0
        for index, tile in enumerate(tiles):
            if not isinstance(tile, Mapping) or set(tile) != _TILE_FIELDS:
                raise IntegrityError("invalid packed tile fields")
            key = MatrixKey.from_dict(tile["key"])
            label = tile["stream"]
            if label is not None and (not isinstance(label, str) or not label or label != label.strip()):
                raise IntegrityError("invalid packed tile stream label")
            if (key, label) in seen:
                raise IntegrityError("duplicate matrix in packed tile map")
            if parsed and _key_order((key, label)) <= _key_order(parsed[-1][4]):
                raise IntegrityError("packed tiles are not in layer,target,stream order")
            seen.add((key, label))
            context = TensorContext.from_dict(tile["context"])
            if _canon(context.to_dict()) != _canon(tile["context"]) or context.key != key:
                raise IntegrityError("packed tile context differs from its key")
            # Equal canonical contexts embed equal canonical models: this is the context model's check.
            if (
                _canon(tile["context"]["model"]) != node_model
                or [context.token_start, context.frames, context.height, context.width]
                != [node["token_start"], node["frames"], node["height"], node["width"]]
            ):
                raise IntegrityError("a packed stream cannot span two nodes")
            spec = _packed_tile_spec(tile["codec"])
            if (
                spec.kind != "svd_hevc" or spec.inter_prediction is not False
                or [spec.pix_fmt, spec.qp, spec.preset] != [codec["pix_fmt"], codec["qp"], codec["preset"]]
            ):
                raise IntegrityError("packed tile codec differs from its all-intra stream")
            if context.key.target == "K" and context.key_space != "pre_rope":
                raise IntegrityError("packed K tiles must use normalized pre-RoPE coordinates")
            validate_rank(spec.rank, context.shape)
            normalization, normalization_text = _normalization_canon(
                spec.rank, spec.normalization, spec.protected_fraction)
            video, video_text = _video_canon(context.frames, context.height, context.width, spec.rank, spec.pix_fmt)
            if (
                tile["shape"] != list(context.shape) or tile["dtype"] != context.dtype
                or tile["factorization_domain"] != _factorization_domain(context)
                or tile["coefficient_shape"] != [context.token_count, spec.rank]
                or _canon(tile["normalization"]) != normalization_text
                or _canon(tile["video"]) != video_text
                or tile["basis"] != {"payload": f"basis.{index:03d}.f16", "view_rank": spec.rank}
                or tile["affine"] != f"affine.{index:03d}.f32"
            ):
                raise IntegrityError("packed tile geometry, normalization or payload names are inconsistent")
            digest = tile["decoded_coefficients_sha256"]
            if digest is not None and (not isinstance(digest, str) or _HEX64.fullmatch(digest) is None):
                raise IntegrityError("invalid packed decoded coefficient checksum")
            first, count = _int_list(tile["pictures"], 2, "picture range")
            x, y, width, height = _int_list(tile["rect"], 4, "rectangle")
            if count != video["frames"] * video["segments"] or first + count > canvas["pictures"]:
                raise IntegrityError("packed tile picture range differs from its frames x segments or the canvas")
            if width != video["image_width"] or height != video["image_height"]:
                raise IntegrityError("packed tile rectangle size differs from its channel-tile mosaic")
            if x % alignment or y % alignment:
                raise IntegrityError("packed tile rectangle origin is not CTU-aligned")
            if x + width > canvas["width"] or y + height > canvas["height"]:
                raise IntegrityError("packed tile rectangle extends past the canvas")
            if temporal:
                if (x, y) != (0, 0) or first != cursor:
                    raise IntegrityError("temporal tiles must own consecutive pictures at the canvas origin")
                cursor += count
            elif first != 0 or video["segments"] != 1 or count != canvas["pictures"]:
                raise IntegrityError("spatial tiles must share every picture of a single segment")
            if ranged:
                offset, length = _int_list(tile["byte_range"], 2, "byte range", minimum=0)
                if offset != byte_cursor or length < 1:
                    raise IntegrityError("packed tile byte ranges are not contiguous")
                byte_cursor += length
            elif tile["byte_range"] is not None:
                raise IntegrityError("whole-stream packets cannot declare tile byte ranges")
            basis_name, affine_name = tile["basis"]["payload"], tile["affine"]
            names |= {basis_name, affine_name}
            basis_names.add(basis_name)
            basis_raw, affine_raw = encoded.payloads.get(basis_name), encoded.payloads.get(affine_name)
            basis_row = descriptors.get(basis_name)
            if not isinstance(basis_row, Mapping) or "shape" not in basis_row:
                raise IntegrityError("packed tile lacks its basis descriptor")
            canonical_rank, channels = _positive_shape(basis_row["shape"], 2, "basis")
            if channels != context.channels or not spec.rank <= canonical_rank <= channels:
                raise IntegrityError("packed tile basis shape/view rank is incompatible")
            _validate_descriptor(
                dict(basis_row), basis_raw, dtype="float16", shape=(canonical_rank, channels), verify=verify,
                external=external_bases,
            )
            _validate_descriptor(
                dict(descriptors.get(affine_name) or {}), affine_raw, dtype="float32",
                shape=normalization["affine_shape"], verify=verify,
            )
            parsed.append((tile, context, spec, (canonical_rank, channels), (key, label)))
        if temporal and cursor != canvas["pictures"]:
            raise IntegrityError("temporal tiles do not cover every canvas picture")
        if ranged and byte_cursor != len(stream):
            raise IntegrityError("packed tile byte ranges do not cover the stream")
        if not temporal:
            rectangles = [item[0]["rect"] for item in parsed]
            for left in range(len(rectangles)):
                ax, ay, aw, ah = rectangles[left]
                for right in range(left + 1, len(rectangles)):
                    bx, by, bw, bh = rectangles[right]
                    if ax < bx + bw and bx < ax + aw and ay < by + bh and by < ay + ah:
                        raise IntegrityError("packed tiles overlap on a shared picture")
        if set(descriptors) != names or set(encoded.payloads) != (names - basis_names if external_bases else names):
            raise IntegrityError("packed packet has missing or uncharged extra payloads")
        _validate_descriptor(
            dict(descriptors["stream.hevc"]), stream, dtype="uint8", shape=(len(stream),), verify=verify,
        )
        if _canon(metadata["padding"]) != _canon(_pack_padding([item[0] for item in parsed], canvas)):
            raise IntegrityError("packed padding accounting differs from the tile map")
        if verify:
            units = _hevc.all_intra_access_units(stream, canvas["pictures"])
            if ranged:
                for tile, *_ in parsed:
                    first, count = tile["pictures"]
                    start, last = units[first], units[first + count - 1]
                    if (
                        not all(unit["self_contained"] for unit in units[first:first + count])
                        or tile["byte_range"] != [start["offset"], last["offset"] + last["length"] - start["offset"]]
                    ):
                        raise IntegrityError("packed tile byte range differs from its access units")
    except (ConfigurationError, UnsupportedError, KeyError, TypeError, ValueError, AttributeError) as exc:
        raise IntegrityError(f"invalid packed tensor packet: {exc}") from exc
    return {
        "metadata": metadata, "canvas": canvas, "stream": stream,
        "tiles": [(index, item[0], item[1], item[2]) for index, item in enumerate(parsed)],
        "basis_shapes": [item[3] for item in parsed], "keys": [item[4] for item in parsed],
        "payload_lengths": {name: len(raw) for name, raw in encoded.payloads.items()},
        "external_bases": external_bases,
    }


def _unpack_tile(images, tile, first_picture, video):
    if _accel_c.current().native_ext:  # accel C (native_ext): the Rust copy, GIL released; identical pixels
        values = _accel_c.native_unpack_tile(images, tile, first_picture, video)
        if values is not None:
            return values
    x, y, width, height = tile["rect"]
    values = np.empty((video["frames"], video["height"], video["width"], video["rank"]), dtype=np.uint8)
    for segment in range(video["segments"]):
        first = first_picture + segment * video["frames"]
        _hevc._unmosaic(
            images[first:first + video["frames"], :, y:y + height, x:x + width], video, segment, values,
        )
    return values.reshape(-1, video["rank"])


def _pack_units(pack, indices):
    """NVDEC access units ``(bytes, offset, length, 1)`` and each selected tile's first job picture.

    Zero-latency NVDEC returns one picture per ``Decode`` call, so every unit is
    one access unit. Temporal packets with tile byte ranges decode only the
    selected tiles' (self-contained) pictures; other packets decode every picture.
    """
    tiles, canvas, stream = pack["tiles"], pack["canvas"], pack["stream"]
    if not all(tiles[index][1]["byte_range"] is not None for index in indices):
        offsets = {index: tiles[index][1]["pictures"][0] for index in indices}
        units = [(stream, unit["offset"], unit["length"], 1)
                 for unit in _hevc.all_intra_access_units(stream, canvas["pictures"])]
        return units, canvas["pictures"], offsets
    units, offsets, pictures = [], {}, 0
    for index in indices:
        offset, length = tiles[index][1]["byte_range"]
        count = tiles[index][1]["pictures"][1]
        units.extend((stream, start, size, 1) for start, size in _hevc.picture_units(stream, offset, length, count))
        offsets[index] = pictures
        pictures += count
    return units, pictures, offsets


def _pack_pixels(encoded, pack, indices, nvdec=None):
    """Decode only what the selected tiles need; per-tile byte ranges allow subsets.

    Software decoding returns NumPy ``[tokens, rank]`` tiles; an explicit
    ``NVDECDecoder`` decodes the same access units on its zero-latency canvas
    session (device output keeps tiles on the GPU). There is no fallback.
    """
    tiles, canvas, stream = pack["tiles"], pack["canvas"], pack["stream"]
    if nvdec is not None:
        from .nvdec import unpack_device_tile

        units, pictures, offsets = _pack_units(pack, indices)
        nbytes = sum(unit[2] for unit in units)
        started = time.perf_counter()
        with span("decode.nvdec", nbytes=nbytes):
            images = nvdec.decode_canvas(
                units, width=canvas["width"], height=canvas["height"], pix_fmt=canvas["pix_fmt"],
            )
        seconds = time.perf_counter() - started
        if isinstance(images, np.ndarray):
            pixels = {
                index: _unpack_tile(images, tiles[index][1], offsets[index], tiles[index][1]["video"])
                for index in indices
            }
        else:
            pixels = {
                index: unpack_device_tile(images, offsets[index], tiles[index][1]["rect"], tiles[index][1]["video"])
                for index in indices
            }
        return pixels, {"decoded_pictures": pictures, "decoded_stream_bytes": nbytes, "decode_seconds": seconds}
    partial = len(indices) < len(tiles) and all(tiles[index][1]["byte_range"] is not None for index in indices)
    if partial:
        pieces, offsets, pictures = [], {}, 0
        for index in indices:
            offset, length = tiles[index][1]["byte_range"]
            pieces.append(stream[offset:offset + length])
            offsets[index] = pictures
            pictures += tiles[index][1]["pictures"][1]
        source = b"".join(pieces)
    else:
        source, pictures = stream, canvas["pictures"]
        offsets = {index: tiles[index][1]["pictures"][0] for index in indices}
    with span("decode.hevc", nbytes=len(source)):
        images, seconds = _hevc.decode_pictures(
            source, pictures=pictures, height=canvas["height"], width=canvas["width"],
            pix_fmt=canvas["pix_fmt"],
        )
    return {
        index: _unpack_tile(images, tiles[index][1], offsets[index], tiles[index][1]["video"])
        for index in indices
    }, {"decoded_pictures": pictures, "decoded_stream_bytes": len(source), "decode_seconds": seconds}


def _pack_factors(encoded, pack, indices, *, verify_decoded_coefficients, verify_packet, nvdec=None):
    pixels, _ = _pack_pixels(encoded, pack, indices, nvdec)
    result = []
    for index in indices:
        _, tile, context, spec = pack["tiles"][index]
        with span("decode.basis"):
            basis = _from_bytes(
                encoded.payloads[tile["basis"]["payload"]], "float16", pack["basis_shapes"][index],
            )
            if verify_packet:
                if not bool(torch.isfinite(basis).all()):
                    raise IntegrityError("stored FP16 basis is nonfinite")
                condition = basis_condition_bound(basis[:spec.rank])
                if not math.isfinite(condition) or condition > 1e6:
                    raise IntegrityError("stored FP16 basis is singular or ill-conditioned")
        affine = np.frombuffer(encoded.payloads[tile["affine"]], dtype="<f4").reshape(
            tile["normalization"]["affine_shape"],
        )
        if verify_packet:
            _hevc.validate_affine(affine, tile["normalization"], spec.rank)
        digest = tile["decoded_coefficients_sha256"]
        if digest is None and verify_decoded_coefficients:
            raise IntegrityError("packed tile was encoded without validation decode; no checksum to verify")
        values = pixels.pop(index)
        if isinstance(values, torch.Tensor):
            with span("decode.dequantize"), cuda_span("decode.dequantize", values.device):
                coefficients = dequantize_device_coefficients(
                    values, affine, tile["normalization"], validate=verify_packet,
                )
            if verify_decoded_coefficients:
                with span("decode.checksum_readback"):
                    if _sha(_tensor_bytes(coefficients)) != digest:
                        raise IntegrityError("HEVC decoded coefficients differ from their authoritative checksum")
            result.append((basis, coefficients))
            continue
        with span("decode.dequantize"):
            decoded = _hevc.dequantize_coefficients(values, affine, tile["normalization"])
        if digest is not None and _sha(decoded.astype("<f4", copy=False).tobytes()) != digest:
            raise IntegrityError("HEVC decoded coefficients differ from their authoritative checksum")
        result.append((basis, torch.from_numpy(decoded)))
    return result


def _pack_indices(pack, keys):
    if keys is None:
        return list(range(len(pack["tiles"])))
    if isinstance(keys, (str, bytes)) or not isinstance(keys, (list, tuple, set, frozenset)):
        raise ConfigurationError("keys must be a collection of node matrix keys")
    lookup = {key: index for index, key in enumerate(pack["keys"])}
    indices = []
    for public in keys:
        canonical = _node_key(public)
        if canonical not in lookup:
            raise ConfigurationError(f"packed packet does not contain {public!r}")
        indices.append(lookup[canonical])
    if not indices or len(set(indices)) != len(indices):
        raise ConfigurationError("keys must name distinct matrices of the packed packet")
    return sorted(indices)


def packed_keys(encoded):
    """Tile-map matrix keys in stream order (``MatrixKey`` or ``(MatrixKey, stream)``)."""
    return [_public_key(*key) for key in _validate_pack(encoded, verify=False)["keys"]]


def decode_packed_pixels(encoded, *, keys=None, verify_packet=False):
    """Decoded uint8 coefficient pixels ``[tokens, rank]`` per tile (identity checks)."""
    if type(verify_packet) is not bool:
        raise ConfigurationError("verification options must be boolean")
    pack = _validate_pack(encoded, verify=verify_packet)
    indices = _pack_indices(pack, keys)
    pixels, _ = _pack_pixels(encoded, pack, indices)
    return {_public_key(*pack["keys"][index]): pixels[index] for index in indices}


@torch.no_grad()
def decode_packed(encoded, *, keys=None, device="cpu", dtype=None, nvdec=None,
                  verify_decoded_coefficients=False, verify_packet=False):
    """Decode selected (default: all) matrices of a packed packet.

    Temporal all-intra packets decode only the selected tiles' access units.
    Returns ``{key: tensor}`` in tile order. An explicit ``nvdec`` decodes the
    stream on NVDEC (zero-latency canvas session); unsupported formats raise
    instead of falling back to software. Asynchronous multi-session device
    restore without dense reconstruction lives in :mod:`reelcache.packed_decode`.
    """
    if type(verify_decoded_coefficients) is not bool or type(verify_packet) is not bool:
        raise ConfigurationError("verification options must be boolean")
    if nvdec is not None and not isinstance(nvdec, NVDECDecoder):
        raise ConfigurationError("nvdec must be an explicitly configured NVDECDecoder")
    with span("decode.validate"):
        pack = _validate_pack(encoded, verify=verify_packet)
    indices = _pack_indices(pack, keys)
    targets = {}
    for index in indices:
        _, tile, context, _ = pack["tiles"][index]
        targets[index] = _decode_targets(context, (None,), dtype, device, tile["dtype"])
    factors = _pack_factors(
        encoded, pack, indices, verify_decoded_coefficients=verify_decoded_coefficients,
        verify_packet=verify_packet, nvdec=nvdec,
    )
    result = {}
    for index, (basis, coefficients) in zip(indices, factors):
        spec = pack["tiles"][index][3]
        selections, output_dtype, output_device = targets[index]
        result[_public_key(*pack["keys"][index])] = _reconstruct(
            basis, coefficients, spec, selections, output_device, output_dtype, verify_packet,
        )[0]
    return result
