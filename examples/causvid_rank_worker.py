"""Gated CausVid480 study workers; importing this module never imports a model.

``native_causvid.py worker --help`` describes the launch interface. Native
launches require the controller's immutable ``reelcache.rank_requests.v1`` batch
and its source-bound ``reelcache.rank_request.v1`` rows. They are translated only
after ``validate_request_batch`` succeeds. Rows must explicitly bind case_id,
trajectory_id and arm_id before request hashing; a case is never inferred from
an arm-specific request_id. Fixed vector controls use the parent's
``representation.layer_ranks={K:[L integers],V:[L integers]}``, with no inline
policy or uniform-rank fields. The complete immutable per-key CodecSpec map
remains static for the request and snapshots. Result/source metadata explicitly
labels ``rank_selection`` as raw, uniform, static-layerwise or adaptive, hashes
the complete requested rank configuration, and includes static K/V vectors.
Native spectra include the shared numeric identity
before capture and persist a separately validated observation beside each file.
Adaptive commits receive the same prebound source metadata; exact-reference
exports reuse the committed oracle result without a second decomposition.
RAW calibration census also requires hash-covered ``capture.detector`` with
``spec`` (the complete shared DetectorSpec), ``workspace_bytes`` and
``solver_workspace_bytes``. Its fixed QR/q=2 acquisition is separate offline
calibration work, runs while the original is borrowed, and emits both .95/.99
crossings (including unresolved nulls) without consulting oracle ranks.
Detector artifacts use ``reelcache_rank_detector_observations_v1`` with the full
``detector_spec``, ``detector_sha256`` and both ``observations`` from the shared
acquisition exporter. Detector CPU threads are explicit, recipe-hashed, and
must match the admitted ``--threads`` count across the whole serial epoch.
The internal CPU-testable request
uses ``reelcache.causvid.rank_request.v1`` and explicit ``CausVidMemoryBudgets``;
it also binds ``disk_budget`` (disk_bytes/disk_reserve_bytes) from the shared
grant. Store/staging/archive maxima, saved state, spectra/metadata and decoded
RGB outputs are admitted separately; each artifact write preserves the reserve.
It is not an alternative launch authorization format.

The manifested asset-map JSON has schema ``reelcache.causvid.assets.v1``, cell_id
``causvid480``, relative source_root/base_model/checkpoint/config paths and fps.
Generation never loads a VAE. A separately admitted ``--operation decode``
epoch consumes requests with a hashed ``latent_source`` relative to output-root.
Development-only ``--operation native-reference`` calls the original native
pipeline, never the adapter, and exports latents at its VAE boundary. It requires
a hash-covered ``reference_source={path,sha256}`` pointing to the matched,
completed RAW development generation result; this supplies noise/RNG checks,
not a clean prefix. Its resident budget includes the original full 32760-token
KV allocation plus cross-attention caches. Decode remains a separate real VAE
epoch. No operation imports quality metrics or grants a passed preflight.

``capture.native_history_audit=true`` is opt-in, development/native_preflight
generation only (at most three chunks). Its new-only ``native-history.jsonl``
streams actual attention-ready post-RoPE K/direct V hashes, exact heads and
coordinates, and measured clean-publication comparisons including the newest
quality-hot mutable slot. Device readbacks, scalar metadata and extra reload
decodes are explicitly diagnostic overhead, not frontier/calibration timing.
These observations alone never grant native parity or advance a phase gate.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict, dataclass, fields, is_dataclass
import gc
import hashlib
import importlib
import inspect
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
import threading
import time
import types
from typing import Any, Mapping

from reelcache.errors import CapacityError, ConfigurationError, IntegrityError, StateError
from reelcache.study import (
    NATIVE_SOURCE_FPS, canonical_json, file_sha256, identity, read_json, study_cell, validate_protocol,
    validate_request_batch, related_prompts, verify_assets, verify_launch_receipt, write_json_new,
)
from reelcache.types import CodecSpec, MatrixKey, ModelIdentity, TensorContext, json_copy


ROOT = Path(__file__).resolve().parents[1]
REQUEST_SCHEMA = "reelcache.causvid.rank_request.v1"
ASSET_MAP_SCHEMA = "reelcache.causvid.assets.v1"
NATIVE_FPS = NATIVE_SOURCE_FPS["causvid480"]
NATIVE_DENSE_TOKENS = 32760
NATIVE_DENSE_KV_BYTES = 30 * 2 * NATIVE_DENSE_TOKENS * 1536 * 2
NATIVE_CROSSATTN_BYTES = 30 * 2 * 512 * 1536 * 2


def _name(value, field):
    if not isinstance(value, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", value) is None:
        raise ConfigurationError(f"{field} must be a simple nonempty identifier")
    return value


def _relative(value, field):
    if not isinstance(value, str) or not value or "\\" in value:
        raise ConfigurationError(f"{field} must be a relative POSIX path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or str(path) != value or value == ".":
        raise ConfigurationError(f"{field} must be normalized and remain inside its root")
    return value


def _positive(value, field, *, zero=False):
    if type(value) is not int or value < (0 if zero else 1):
        raise ConfigurationError(f"{field} must be a {'nonnegative' if zero else 'positive'} integer")
    return value


def _sha(value, field):
    if not isinstance(value, str) or re.fullmatch("[0-9a-f]{64}", value) is None:
        raise ConfigurationError(f"{field} must be a lowercase SHA256")
    return value


def conditioning_identity(text, asset_id):
    """Identity known before encoding; the actual embedding hash is also saved."""
    return identity({
        "schema": "reelcache.causvid.conditioning.v1", "cell_id": "causvid480",
        "text": text, "asset_id": asset_id,
        "encoder": "native_WanTextEncoder_exact_text_BF16_512",
    })


def model_identity(asset_id):
    return ModelIdentity(
        "causvid", identity({
            "asset_id": asset_id, "cell": study_cell("causvid480").to_dict(),
            "attention": "native_self_attention; instance_local_full_SDPA_cross_attention",
            "native_normalized_K_dtype": "bfloat16", "native_V_dtype": "bfloat16",
        }),
        30, 1536, "tokens_channels", "bfloat16",
    )


def build_rank_policy(document, model, config):
    """Use the shared typed policy unchanged; never rebind a foreign policy."""
    from reelcache.online_rank import RankPolicy

    policy = RankPolicy.from_dict(document)
    for key in policy.keys:
        policy.validate_context(TensorContext(
            model, key, 0, config.frames_per_chunk, config.grid_height, config.grid_width,
            "pre_rope" if key.target == "K" else "none",
        ))
    return policy


def _admit_policy(policy, document, ffmpeg):
    if "study_request" in document:
        _validate_policy_codec(document["study_request"]["representation"], policy.to_dict())
    if not policy.history_min <= 1 <= document["chunks"] <= policy.history_max:
        raise ConfigurationError("The entire request must fit the calibrated policy history support")
    budget = document["memory_budgets"]
    policy_host = policy.workspace_limit_bytes
    if budget["clean_disk_bytes"] and policy.anchor_mode == "staged":
        policy_host += policy.staging_limit_bytes
    latent_bytes = document["chunks"] * 3 * 16 * 60 * 104 * 2
    other_host = (
        budget["model_host_bytes"] + budget["clean_host_bytes"] + budget["spectrum_workspace_bytes"]
        + 6 * latent_bytes + (16 << 20)
    )
    if policy_host > budget["host_bytes"] - other_host:
        raise CapacityError("Rank-policy workspace/staged inputs exceed their host admission")
    if policy.codec.kind == "svd_hevc" and (
        ffmpeg is None or policy.codec.ffmpeg_path not in (None, str(ffmpeg))
    ):
        raise ConfigurationError("Adaptive HEVC requires the same explicit verified FFmpeg asset")
    if policy.codec.kind == "svd_hevc" and policy.codec.inter_prediction is None:
        raise ConfigurationError(
            "Adaptive HEVC policies must declare codec.inter_prediction (version-2 specification); "
            "a version-1 policy keeps its legacy GOP identity and cannot encode new chunks",
        )


def rgb_windows(chunks, fps):
    _positive(chunks, "chunks")
    if isinstance(fps, bool) or not isinstance(fps, (int, float)) or not math.isfinite(fps) or fps != NATIVE_FPS:
        raise ConfigurationError("Native CausVid source FPS must be 16")
    counts = [4 * 3 * (index + 1) - 3 for index in range(chunks)]
    return {
        "fps": fps, "latent_frames_per_chunk": 3, "rgb_rule": "4F-3",
        "prefix_frame_counts": counts,
        "windows": [
            {
                "chunk_id": index, "start_frame": 0 if index == 0 else counts[index - 1],
                "end_frame": count, "kind": "first_chunk" if index == 0 else "genuine_continuation",
                "start_seconds": (0 if index == 0 else counts[index - 1]) / fps,
                "end_seconds": count / fps,
            }
            for index, count in enumerate(counts)
        ],
    }


@dataclass(frozen=True)
class CausVidMemoryBudgets:
    """Separate admitted maxima, not a claim of measured total allocator peaks."""

    device_bytes: int
    model_device_bytes: int
    model_workspace_bytes: int
    host_bytes: int
    model_host_bytes: int
    resident_bytes: int
    attention_bytes: int
    current_bytes: int
    clean_host_bytes: int
    clean_disk_bytes: int
    quality_history_bytes: int
    store_bytes: int
    spectrum_workspace_bytes: int
    source_archive_bytes: int

    def __post_init__(self):
        optional = {"clean_disk_bytes", "quality_history_bytes", "spectrum_workspace_bytes", "source_archive_bytes"}
        for item in fields(self):
            _positive(getattr(self, item.name), item.name, zero=item.name in optional)

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, Mapping) or set(value) != {item.name for item in fields(cls)}:
            raise ConfigurationError("memory_budgets must explicitly cover every CausVid budget field")
        return cls(**dict(value))

    def admit(self, *, chunks, history_mode, capture):
        cell = study_cell("causvid480")
        if self.current_bytes < cell.native_kv_bytes:
            raise CapacityError("Native current KV was not admitted")
        matrix = cell.tokens * cell.channels * 2
        if self.clean_disk_bytes:
            if (
                self.clean_host_bytes < 4 * matrix + 65536
                or self.clean_disk_bytes < cell.source_bytes + 2 * cell.layers * 65536
            ):
                raise CapacityError("Whole-clean-chunk disk staging and its host workspace were not admitted")
        elif self.clean_host_bytes < cell.source_bytes + 3 * matrix + 65536:
            raise CapacityError("Whole-clean-chunk host staging was not admitted")
        quality = chunks * cell.native_kv_bytes if history_mode == "quality_hot" else 0
        if self.quality_history_bytes < quality or (history_mode == "reload" and self.quality_history_bytes):
            raise CapacityError("Decoded native quality history must be fully and separately admitted")
        required = sum(getattr(self, name) for name in (
            "model_device_bytes", "model_workspace_bytes", "resident_bytes", "attention_bytes",
            "current_bytes", "quality_history_bytes",
        ))
        if self.device_bytes < required:
            raise CapacityError("The aggregate device grant does not cover all declared live allocations")
        latent_bytes = chunks * 3 * 16 * 60 * 104 * 2
        detector = capture.get("detector")
        detector_workspace = 0 if detector is None else detector["workspace_bytes"]
        if detector is not None:
            from reelcache.factorization import DetectorSpec, rank_fit_workspace_bytes

            spec = DetectorSpec.from_dict(detector["spec"])
            needed = matrix + rank_fit_workspace_bytes(
                (cell.tokens, cell.channels), min(cell.tokens, cell.channels), tile_rows=spec.tile_rows,
                solver_workspace_bytes=detector["solver_workspace_bytes"],
            )
            if detector_workspace < needed:
                raise CapacityError("Fixed detector must admit its original CPU copy and complete width ladder")
        host = self.model_host_bytes + self.clean_host_bytes + max(self.spectrum_workspace_bytes, detector_workspace)
        if self.host_bytes < host + 6 * latent_bytes + (16 << 20):
            raise CapacityError("The host grant does not cover models, capture, spectra and saved request state")
        if capture["spectra"] and not self.spectrum_workspace_bytes:
            raise CapacityError("Exact spectra require an explicit separate offline workspace budget")
        if len(capture["retain"]) * (matrix + 65536) > self.source_archive_bytes:
            raise CapacityError("The selected pristine-source archive exceeds its explicit disk budget")


def _capture_config(value, chunks):
    if not isinstance(value, Mapping) or set(value) - {
        "spectra", "retain", "tile_rows", "detector", "native_history_audit",
    }:
        raise ConfigurationError("capture supports spectra, retain, tile_rows, detector and native_history_audit")
    result = {"spectra": value.get("spectra", False), "retain": value.get("retain", []),
              "tile_rows": value.get("tile_rows", 1024), "detector": value.get("detector")}
    if "native_history_audit" in value:
        if type(value["native_history_audit"]) is not bool:
            raise ConfigurationError("native_history_audit must be an explicit boolean")
        result["native_history_audit"] = value["native_history_audit"]
    if type(result["spectra"]) is not bool or not isinstance(result["retain"], list):
        raise ConfigurationError("Invalid clean-source capture declaration")
    _positive(result["tile_rows"], "tile_rows")
    seen = set()
    for entry in result["retain"]:
        if not isinstance(entry, dict) or set(entry) != {"chunk_id", "layer", "target"}:
            raise ConfigurationError("retained sources must specify chunk_id/layer/target")
        chunk = _positive(entry["chunk_id"], "retained chunk", zero=True)
        key = MatrixKey(entry["layer"], entry["target"])
        marker = (chunk, key.layer, key.target)
        if chunk >= chunks or key.layer >= 30 or marker in seen:
            raise ConfigurationError("Duplicate or out-of-range pristine-source selection")
        seen.add(marker)
    if result["detector"] is not None:
        from reelcache.factorization import DetectorSpec

        detector = result["detector"]
        if not isinstance(detector, Mapping) or set(detector) != {"spec", "workspace_bytes", "solver_workspace_bytes"}:
            raise ConfigurationError("Detector capture must bind its complete recipe and both workspace limits")
        spec = DetectorSpec.from_dict(detector["spec"])
        if spec.to_dict() != DetectorSpec(cpu_threads=spec.cpu_threads).to_dict():
            raise ConfigurationError("Detector recipe must match the current frozen CPU implementation/backend")
        _positive(detector["workspace_bytes"], "detector workspace_bytes")
        _positive(detector["solver_workspace_bytes"], "detector solver_workspace_bytes", zero=True)
        result["detector"] = json_copy(detector)
    return result


def _validate_detector_capture(document):
    phase = document.get("study_phase", document.get("study_request", {}).get("phase"))
    detector = document["capture"].get("detector")
    if "latent_source" in document:
        if detector is not None:
            raise ConfigurationError("Fixed detector acquisition belongs to original generation, not VAE decode")
        return
    if phase == "calibration_census":
        if (
            document["split"] != "calibration" or document["rank_policy"] is not None
            or any(spec.kind != "raw" for spec in codec_specs(document["spec_config"], 30).values())
            or document["capture"]["spectra"] is not True or detector is None
        ):
            raise ConfigurationError("RAW calibration census requires both exact spectra and its fixed detector recipe")
    elif detector is not None:
        raise ConfigurationError("Standalone fixed detector capture is only admitted for RAW calibration census")


def codec_specs(spec_config, layers, *, ffmpeg=None):
    if isinstance(spec_config, Mapping) and set(spec_config) == {"K", "V"}:
        rows = [
            {"key": {"layer": layer, "target": target}, "spec": spec_config[target]}
            for layer in range(layers) for target in ("K", "V")
        ]
    elif isinstance(spec_config, list):
        rows = spec_config
    else:
        raise ConfigurationError("spec_config must contain K/V specs or a complete layer/key list")
    result = {}
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != {"key", "spec"} or not isinstance(row["spec"], Mapping):
            raise ConfigurationError("Invalid per-matrix codec declaration")
        key = MatrixKey.from_dict(row["key"])
        if key.layer >= layers or key in result:
            raise ConfigurationError("Duplicate or out-of-range codec key")
        spec = dict(row["spec"])
        if spec.get("ffmpeg_path") is not None:
            raise ConfigurationError("Worker codecs use only the verified asset-map FFmpeg, not request paths")
        if spec.get("kind") == "svd_hevc":
            # Request codec rows are configuration: an omitted GOP choice means the all-intra default.
            spec.setdefault("inter_prediction", False)
            if ffmpeg is not None:
                spec["ffmpeg_path"] = str(ffmpeg)
        result[key] = CodecSpec.from_dict(spec)
    expected = {MatrixKey(layer, target) for layer in range(layers) for target in ("K", "V")}
    if set(result) != expected:
        raise ConfigurationError("Every layer's K and V need an explicit codec specification")
    if len({spec.kind == "raw" for key, spec in result.items() if key.target == "K"}) != 1:
        raise ConfigurationError("All CausVid K sources must use one fixed coordinate domain")
    return result


def rank_control_metadata(document):
    """Describe the declared control, never infer adaptivity from resolved ranks."""
    policy, configuration = document.get("rank_policy"), document.get("spec_config")
    if policy is None and configuration is None:
        return {}
    rows = configuration if isinstance(configuration, list) else (
        [] if configuration is None else [{"spec": value} for value in configuration.values()]
    )
    if policy is not None:
        mode = "adaptive"
    elif all(row["spec"].get("kind", "raw") == "raw" for row in rows):
        mode = "raw"
    else:
        mode = "static-layerwise" if isinstance(configuration, list) else "uniform"
    result = {
        "rank_selection": mode,
        "rank_configuration_sha256": identity({"spec_config": configuration, "rank_policy": policy}),
    }
    if mode == "static-layerwise":
        result["layer_ranks"] = {
            target: [row["spec"]["rank"] for row in sorted(
                configuration, key=lambda row: row["key"]["layer"],
            ) if row["key"]["target"] == target]
            for target in ("K", "V")
        }
    return result


def _validate_policy_codec(representation, policy_document):
    if not isinstance(representation, Mapping) or not isinstance(policy_document, Mapping):
        raise ConfigurationError("The admitted representation and policy must be explicit codec objects")
    codec = CodecSpec.from_dict(policy_document.get("codec", {})).to_dict()
    kind = representation.get("kind")
    defaults = CodecSpec(kind=kind, rank=None if kind == "raw" else 1).to_dict()
    for name in defaults:
        if name == "rank" or (name == "ffmpeg_path" and name not in representation):
            continue
        declared = representation.get(name, defaults[name])
        if canonical_json(declared) != canonical_json(codec.get(name)):
            raise ConfigurationError(f"Outer representation {name} differs from the adaptive policy codec")


def _disk_grant(document):
    declared = document.get("disk_budget")
    shared = document.get("study_request", {}).get("budgets")
    if declared is None and isinstance(shared, Mapping):
        declared = {name: shared[name] for name in ("disk_bytes", "disk_reserve_bytes")}
    if not isinstance(declared, Mapping) or set(declared) != {"disk_bytes", "disk_reserve_bytes"}:
        raise ConfigurationError("Execution requires an explicit total output disk grant and reserve")
    grant = {
        "disk_bytes": _positive(declared["disk_bytes"], "disk_bytes"),
        "disk_reserve_bytes": _positive(declared["disk_reserve_bytes"], "disk_reserve_bytes", zero=True),
    }
    if isinstance(shared, Mapping) and (
        grant["disk_bytes"] > shared["disk_bytes"]
        or grant["disk_reserve_bytes"] < shared["disk_reserve_bytes"]
    ):
        raise CapacityError("The execution output disk budget cannot widen the shared grant or consume its reserve")
    return grant


def _output_reservation(document, operation):
    if operation not in ("generate", "decode", "native-reference"):
        raise ConfigurationError("Unknown CausVid output operation")
    grant = _disk_grant(document)
    budget = document["memory_budgets"]
    chunks = _positive(document["chunks"], "chunks")
    request_bytes = len(canonical_json(document)) + 1
    failure_bytes = (64 << 10) + 2 * request_bytes
    metadata_bytes = (1 << 20) + 8 * request_bytes
    latent_bytes = rgb_bytes = runtime_bytes = staging_bytes = archive_bytes = 0
    if operation in ("generate", "native-reference"):
        runtime_bytes, staging_bytes, archive_bytes = (
            budget["store_bytes"], budget["clean_disk_bytes"], budget["source_archive_bytes"],
        )
        if operation == "native-reference":
            runtime_bytes = staging_bytes = 0
        # Native saved state: latents/noise, up to 512x4096 BF16 text, RNGs and headers.
        latent_bytes = 2 * chunks * 3 * 16 * 60 * 104 * 2 + 512 * 4096 * 2 + (256 << 10)
        matrices = 2 * study_cell("causvid480").layers * chunks
        metadata_bytes += matrices * ((64 << 10) + 4 * request_bytes)
        if document.get("capture", {}).get("spectra") or (
            document.get("rank_policy") is not None
            and document["rank_policy"].get("solver") == "exact_oracle"
        ):
            metadata_bytes += matrices * (1536 * 64 + (128 << 10) + 4 * request_bytes)
        if document.get("capture", {}).get("detector") is not None:
            metadata_bytes += matrices * ((256 << 10) + 8 * request_bytes)
        if document.get("capture", {}).get("native_history_audit", False):
            # At most four noisy passes, one clean pass and one publication per matrix.
            metadata_bytes += matrices * 6 * (64 << 10)
    else:
        rgb_bytes = rgb_windows(chunks, NATIVE_FPS)["prefix_frame_counts"][-1] * 480 * 832 * 3 + (64 << 10)
    artifacts = latent_bytes + rgb_bytes + metadata_bytes + failure_bytes
    reserved = runtime_bytes + staging_bytes + archive_bytes
    if reserved + artifacts + grant["disk_reserve_bytes"] > grant["disk_bytes"]:
        raise CapacityError("Complete CausVid outputs and disk reserve exceed the admitted disk grant")
    return {
        **grant, "operation": operation, "runtime_bytes": runtime_bytes, "staging_bytes": staging_bytes,
        "archive_bytes": archive_bytes, "latent_file_bytes": latent_bytes, "rgb_file_bytes": rgb_bytes,
        "metadata_bytes": metadata_bytes, "failure_bytes": failure_bytes,
        "artifact_floor_bytes": artifacts, "reserved_component_bytes": reserved,
    }


class _RequestOutputBudget:
    """Account caller artifacts separately from the bounded store/staging/archive."""

    def __init__(self, document, root, operation, *, input_bytes=0):
        self.plan = _output_reservation(document, operation)
        self.root = Path(root)
        self.used = {"artifacts": 0, "archive": 0}
        self.artifact_limit = (
            self.plan["disk_bytes"] - self.plan["disk_reserve_bytes"]
            - self.plan["reserved_component_bytes"] - input_bytes
        )
        if self.artifact_limit < self.plan["artifact_floor_bytes"]:
            raise CapacityError("Retained inputs plus complete outputs would consume the admitted disk reserve")
        self._check_free(self.plan["reserved_component_bytes"] + self.plan["artifact_floor_bytes"])

    def _check_free(self, additional):
        path = self.root
        while not path.exists():
            if path.parent == path:
                raise CapacityError("No existing filesystem can admit the output namespace")
            path = path.parent
        if shutil.disk_usage(path).free < additional + self.plan["disk_reserve_bytes"]:
            raise CapacityError("Output filesystem has insufficient free bytes to preserve the disk reserve")

    def charge(self, count, *, category="artifacts", failure=False):
        _positive(count, "artifact bytes", zero=True)
        limit = self.plan["archive_bytes"] if category == "archive" else (
            self.artifact_limit - (0 if failure else self.plan["failure_bytes"])
        )
        if category not in self.used or self.used[category] + count > limit:
            raise CapacityError("CausVid artifact write exceeds its admitted output disk budget")
        self._check_free(count)
        self.used[category] += count


def _validate_reference_request(document):
    phase = document.get("study_phase", document.get("study_request", {}).get("phase"))
    if (
        phase != "native_preflight" or document["split"] != "development"
        or "reference_source" not in document or "latent_source" in document
        or document["rank_policy"] is not None
        or any(spec.kind != "raw" for spec in codec_specs(document["spec_config"], 30).values())
    ):
        raise ConfigurationError("Native reference requires a paired RAW development/native_preflight request")
    if document["capture"]["spectra"] or document["capture"].get("detector") is not None:
        raise ConfigurationError("Native reference admits scalar audits and selected originals, not a second census")
    if document["chunks"] * 4680 > NATIVE_DENSE_TOKENS:
        raise CapacityError("Requested horizon exceeds the unchanged original native dense KV allocation")
    if document["memory_budgets"]["resident_bytes"] < NATIVE_DENSE_KV_BYTES + NATIVE_CROSSATTN_BYTES:
        raise CapacityError("Native reference must admit full 32760-token dense KV and cross-attention caches")


@dataclass(frozen=True)
class CausVidRankRequest:
    """Canonical JSON storage prevents caller aliases from changing a request."""

    canonical: bytes

    @property
    def document(self):
        import json

        return json.loads(self.canonical)

    @classmethod
    def from_dict(cls, value, *, protocol=None, asset_id=None):
        required = {
            "schema", "request_id", "cell_id", "prompt_group_id", "text", "seed", "chunks",
            "conditioning_identity", "history_mode", "spec_config", "rank_policy",
            "memory_budgets", "output_namespace", "case_id", "trajectory_id", "arm_id", "split",
        }
        optional = {"capture", "latent_source", "reference_source", "study_request", "study_phase", "disk_budget"}
        if not isinstance(value, Mapping) or required - set(value) or set(value) - required - optional:
            raise ConfigurationError("CausVid request fields differ from the frozen worker schema")
        value = json_copy(value)
        if value["schema"] != REQUEST_SCHEMA or value["cell_id"] != "causvid480":
            raise ConfigurationError("This worker supports only the native causvid480 request schema")
        _name(value["request_id"], "request_id")
        _name(value["prompt_group_id"], "prompt_group_id")
        for name in ("case_id", "trajectory_id", "arm_id"):
            if not isinstance(value[name], str) or not value[name].strip():
                raise ConfigurationError(f"{name} must be explicit; never infer a case from an arm request ID")
        if value["split"] not in ("development", "calibration", "heldout"):
            raise ConfigurationError("The request must bind its explicit protocol split")
        if not isinstance(value["text"], str) or not value["text"].strip():
            raise ConfigurationError("The exact official prompt text is required")
        if _positive(value["seed"], "seed", zero=True) >= 2**63:
            raise ConfigurationError("The explicit generation seed must fit a nonnegative int64")
        _positive(value["chunks"], "chunks")
        _sha(value["conditioning_identity"], "conditioning_identity")
        _relative(value["output_namespace"], "output_namespace")
        if PurePosixPath(value["output_namespace"]).parts[0] == "build_cache":
            raise ConfigurationError("build_cache is reserved for the worker's local compiler caches")
        if value["history_mode"] not in ("reload", "quality_hot"):
            raise ConfigurationError("history_mode must be reload or quality_hot")
        if value["rank_policy"] is not None and not isinstance(value["rank_policy"], dict):
            raise ConfigurationError("rank_policy must be null or an explicit versioned policy object")
        if value["rank_policy"] is None:
            codec_specs(value["spec_config"], 30)
        elif value["spec_config"] is not None:
            raise ConfigurationError("rank_policy owns resolved specs; adaptive spec_config must be null")
        if value["rank_policy"] is not None and "study_request" in value:
            _validate_policy_codec(value["study_request"]["representation"], value["rank_policy"])
        value["capture"] = _capture_config(value.get("capture", {}), value["chunks"])
        if value["capture"].get("native_history_audit", False):
            phases = [
                phase for phase in (value.get("study_phase"), value.get("study_request", {}).get("phase"))
                if phase is not None
            ]
            if (
                not phases or any(phase != "native_preflight" for phase in phases)
                or value["split"] != "development" or value["chunks"] > 3
                or "latent_source" in value or "reference_source" in value
            ):
                raise ConfigurationError("Native-history audits are bounded development/native_preflight generation only")
        _validate_detector_capture(value)
        if (
            value["rank_policy"] is not None and value["rank_policy"].get("solver") == "exact_oracle"
            and value["capture"]["spectra"]
        ):
            raise ConfigurationError(
                "Exact-reference factorization owns its single spectrum solve; disable separate observer spectra",
            )
        budgets = CausVidMemoryBudgets.from_dict(value["memory_budgets"])
        budgets.admit(chunks=value["chunks"], history_mode=value["history_mode"], capture=value["capture"])
        if asset_id is not None and value["conditioning_identity"] != conditioning_identity(value["text"], asset_id):
            raise ConfigurationError("Conditioning identity does not bind this exact prompt and staged asset set")
        if protocol is not None:
            validate_protocol(protocol)
            if value.get("split") == "development":
                if any(related_prompts(value["text"], row["text"]) for row in protocol["prompts"]):
                    raise ConfigurationError("Development requests must not inspect fresh study prompt groups")
                catalog = read_json(protocol["catalog"]["path"])
                if not any(row.get("prompt_en") == value["text"] for row in catalog):
                    raise ConfigurationError("Development generation still requires an unchanged official prompt")
            else:
                matches = [row for row in protocol["prompts"] if row["group_id"] == value["prompt_group_id"]]
                if len(matches) != 1 or value["text"] != matches[0]["text"]:
                    raise ConfigurationError("Request prompt text/group/seed is not an exact frozen protocol member")
                if value.get("split", matches[0]["split"]) != matches[0]["split"]:
                    raise ConfigurationError("Request split differs from its frozen prompt group")
                value["split"] = matches[0]["split"]
            if value["seed"] not in protocol["seeds"]:
                raise ConfigurationError("Request seed is outside the frozen protocol")
        for name in ("latent_source", "reference_source"):
            if name not in value:
                continue
            source = value[name]
            if not isinstance(source, dict) or set(source) != {"path", "sha256"}:
                raise ConfigurationError(f"{name} must name one hashed local result.json")
            _relative(source["path"], name + ".path")
            _sha(source["sha256"], name + ".sha256")
        if "reference_source" in value:
            _validate_reference_request(value)
        value["disk_budget"] = _disk_grant(value)
        operation = "decode" if "latent_source" in value else (
            "native-reference" if "reference_source" in value else "generate"
        )
        _output_reservation(value, operation)
        return cls(canonical_json(value))


@dataclass(frozen=True)
class CausVidAssets:
    root: Path
    asset_id: str
    paths: Mapping[str, Path]
    fps: float
    manifest: Mapping[str, Any]

    @classmethod
    def verified(cls, manifest_path, mapping_name, *, launch_receipt):
        manifest_path = Path(manifest_path).resolve(strict=True)
        if manifest_path.name != "assets.json":
            raise ConfigurationError("--assets must be the staged root/assets.json")
        root = manifest_path.parent
        manifest = verify_assets(root)
        by_name = {row["name"]: row for row in manifest["files"]}
        mapping_name = _relative(mapping_name, "asset_map")
        if mapping_name not in by_name:
            raise IntegrityError("The native asset mapping must itself be a verified staged asset")
        mapping = read_json(root / mapping_name)
        required = {"schema", "cell_id", "source_root", "base_model", "checkpoint", "config", "fps"}
        if not isinstance(mapping, dict) or required - set(mapping) or set(mapping) - required - {"ffmpeg"}:
            raise ConfigurationError("CausVid asset mapping fields differ from the worker schema")
        if mapping["schema"] != ASSET_MAP_SCHEMA or mapping["cell_id"] != "causvid480":
            raise ConfigurationError("Native asset mapping is not for causvid480")
        paths = {}
        for role in ("source_root", "base_model", "checkpoint", "config", "ffmpeg"):
            if role not in mapping:
                continue
            name = _relative(mapping[role], f"asset_map.{role}")
            path = (root / name).resolve(strict=True)
            if not path.is_relative_to(root):
                raise IntegrityError("A native asset path escapes the verified local asset root")
            if role in ("source_root", "base_model"):
                if not path.is_dir():
                    raise ConfigurationError(f"{role} must be a verified local directory")
                for child in path.rglob("*"):
                    if child.is_symlink() and child.is_dir():
                        raise IntegrityError("Native asset directories may not hide unenumerated symlink trees")
                    if child.is_file() and str(child.relative_to(root)) not in by_name:
                        raise IntegrityError(f"Unmanifested file in native import/model directory: {child.name}")
            elif name not in by_name or not path.is_file():
                raise IntegrityError(f"The explicit native {role} was not verified")
            paths[role] = path
        for name in ("causal_model.py", "causal_inference.py", "wan_wrapper.py", "attention.py", "vae.py"):
            path = paths["source_root"] / "third_party/causvid/modeling" / name
            if not path.is_file() or str(path.relative_to(root)) not in by_name:
                raise IntegrityError(f"The native source set is incomplete: {name}")
        approved_hashes = {
            row["sha256"] for row in launch_receipt["source_closure"]["files"].values()
        }
        for path in paths["source_root"].rglob("*.py"):
            if by_name[str(path.relative_to(root))]["sha256"] not in approved_hashes:
                raise IntegrityError("The launch review did not cover every staged native Python source")
        base_config_path = paths["base_model"] / "config.json"
        if str(base_config_path.relative_to(root)) not in by_name:
            raise IntegrityError("The native model config was not explicitly staged")
        config = read_json(base_config_path)
        expected = {"model_type": "t2v", "dim": 1536, "num_heads": 12, "num_layers": 30, "in_dim": 16, "out_dim": 16}
        if any(config.get(key) != item for key, item in expected.items()):
            raise ConfigurationError("The model configuration is not the qualified native CausVid 1.3B geometry")
        fps = rgb_windows(1, mapping["fps"])["fps"]
        return cls(root, manifest["asset_id"], paths, fps, manifest)


def _executed_closure(receipt):
    records = {str(Path(row["path"]).resolve()): row for row in receipt["source_closure"]["files"].values()}
    required = [
        Path(__file__), ROOT / "examples/native_causvid.py", ROOT / "examples/causvid_native_reference.py",
        *(ROOT / "reelcache").rglob("*.py"),
    ]
    for path in required:
        if str(path.resolve()) not in records or records[str(path.resolve())]["sha256"] != file_sha256(path):
            raise IntegrityError("The launch receipt does not cover this executed worker and package closure")


def _study_request(row, assets, worker_budgets, phase):
    """Translate the controller envelope without widening any declared grant."""
    for name in ("case_id", "trajectory_id", "arm_id"):
        if not isinstance(row.get(name), str) or not row[name].strip():
            raise ConfigurationError(f"The controller must explicitly bind {name} before computing request_id")
    budget_document = row.get("causvid_memory_budgets", worker_budgets)
    if budget_document is None:
        raise ConfigurationError("CausVid batches need --worker-budgets or per-request causvid_memory_budgets")
    detailed = CausVidMemoryBudgets.from_dict(budget_document)
    envelope = row["budgets"]
    if (
        detailed.device_bytes > envelope["device_bytes_per_rank"]
        or detailed.host_bytes > envelope["host_bytes"]
        or detailed.resident_bytes > envelope["history_bytes_per_rank"]
        or detailed.quality_history_bytes > envelope["history_bytes_per_rank"]
        or detailed.current_bytes + detailed.attention_bytes + detailed.model_workspace_bytes
        > envelope["workspace_bytes_per_rank"]
        or detailed.store_bytes + detailed.clean_disk_bytes + detailed.source_archive_bytes
        + envelope["disk_reserve_bytes"] > envelope["disk_bytes"]
    ):
        raise CapacityError("CausVid detailed allocations exceed the source-bound controller resource grant")
    rgb_windows(row["chunks"], assets.fps)
    if row["source_fps"] != assets.fps or isinstance(row["source_fps"], bool):
        raise ConfigurationError("The controller FPS differs from the verified native asset mapping")
    representation = row["representation"]
    policy = representation.get("rank_policy")
    layer_ranks = representation.get("layer_ranks")
    kind = representation["kind"]
    if layer_ranks is not None and (
        policy is not None or kind == "raw" or not isinstance(layer_ranks, Mapping)
        or set(layer_ranks) != {"K", "V"}
        or any(name in representation for name in ("rank_k", "rank_v"))
    ):
        raise ConfigurationError("Static layer ranks require only complete K/V vectors, without policy or uniform ranks")
    if policy is not None:
        _validate_policy_codec(representation, policy)
        hints = {name for name in ("rank_k", "rank_v") if name in representation}
        if hints and hints != {"rank_k", "rank_v"}:
            raise ConfigurationError("Legacy adaptive scalar hints must be supplied as both K and V, never one")
    common = {name: representation[name] for name in (
        "normalization", "pix_fmt", "qp", "preset", "protected_fraction", "inter_prediction",
    ) if name in representation}
    if kind == "svd_hevc":
        common.setdefault("inter_prediction", False)
    if layer_ranks is not None:
        cell = study_cell("causvid480")
        for target, values in layer_ranks.items():
            if not isinstance(values, list) or len(values) != cell.layers:
                raise ConfigurationError("Static rank vectors must cover every native CausVid layer")
            for rank in values:
                if _positive(rank, f"layer_ranks.{target}") > min(cell.tokens, cell.channels):
                    raise ConfigurationError("A static layer rank exceeds the native full-channel domain")
        specs = [
            {"key": {"layer": layer, "target": target},
             "spec": {"kind": kind, **common, "rank": layer_ranks[target][layer]}}
            for layer in range(cell.layers) for target in ("K", "V")
        ]
    else:
        specs = None if policy is not None else {
            target: {"kind": kind, **common, **({} if kind == "raw" else {"rank": representation["rank_" + target.lower()]})}
            for target in ("K", "V")
        }
    prompt_group = row.get("group_id") or "development-" + identity(row["prompt"])[:16]
    result = {
        "schema": REQUEST_SCHEMA, "request_id": row["request_id"], "cell_id": row["cell_id"],
        "case_id": row["case_id"], "trajectory_id": row["trajectory_id"], "arm_id": row["arm_id"],
        "prompt_group_id": prompt_group, "text": row["prompt"], "seed": row["seed"],
        "chunks": row["chunks"], "conditioning_identity": row.get(
            "conditioning_identity", conditioning_identity(row["prompt"], assets.asset_id),
        ),
        "split": row["split"], "history_mode": row["history_mode"],
        "spec_config": specs, "rank_policy": policy, "memory_budgets": asdict(detailed),
        "disk_budget": {name: envelope[name] for name in ("disk_bytes", "disk_reserve_bytes")},
        "output_namespace": row["output_namespace"],
        "capture": row.get("capture", {
            "spectra": phase in ("calibration_census", "heldout") and kind == "raw",
        }),
        "study_request": row, "study_phase": phase,
    }
    for name in ("latent_source", "reference_source"):
        if name in row:
            result[name] = row[name]
    return result


def _validate_phase(document, protocol, phase):
    expected = {
        "native_preflight": "development", "calibration_census": "calibration",
        "calibration_frontier": "calibration", "heldout": "heldout",
    }
    if phase not in expected or document.get("split") != expected[phase]:
        raise StateError("This launch phase must not inspect another prompt split")
    if not 2 <= document["chunks"] <= len(protocol["clean_boundaries"]):
        raise ConfigurationError("The admitted native study horizon is two or three clean chunks")


def _validate_thread_admission(requests, threads):
    _positive(threads, "admitted CPU threads")
    available = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
    if available is not None and threads > available:
        raise CapacityError("Admitted CausVid CPU threads exceed the process CPU allocation")
    for request in requests:
        document = request.document
        detector = document["capture"].get("detector")
        recipes = [] if detector is None else [detector["spec"]]
        if document["rank_policy"] is not None:
            recipes.append(document["rank_policy"].get("detector_spec"))
        for spec in recipes:
            if not isinstance(spec, Mapping) or _positive(spec.get("cpu_threads"), "detector CPU threads") != threads:
                raise ConfigurationError("Every bound detector recipe must match the admitted CPU thread count")


def verify_worker_inputs(*, protocol_path, assets_path, asset_map, launch_receipt_path, requests_path,
                         phase="native_preflight", worker_budgets=None):
    """All launch and local-asset gates precede any native import or allocation."""
    protocol, receipt = read_json(protocol_path), read_json(launch_receipt_path)
    validate_protocol(protocol)
    verify_launch_receipt(receipt, protocol, phase=phase)
    _executed_closure(receipt)
    assets = CausVidAssets.verified(assets_path, asset_map, launch_receipt=receipt)

    if worker_budgets is not None and not isinstance(worker_budgets, Mapping):
        worker_budgets = read_json(worker_budgets)
    batch = read_json(requests_path)
    rows = validate_request_batch(batch, protocol, receipt, cell_id="causvid480", phase=phase)
    requests = [
        CausVidRankRequest.from_dict(
            _study_request(row, assets, worker_budgets, phase), protocol=protocol, asset_id=assets.asset_id,
        )
        for row in rows
    ]
    for request in requests:
        _validate_phase(request.document, protocol, phase)
    return protocol, receipt, assets, requests


def _inside(root, relative, *, exists=False):
    path = root / _relative(relative, "output/input path")
    resolved = path.resolve(strict=exists)
    if not resolved.is_relative_to(root.resolve()):
        raise IntegrityError("An output/input namespace escapes the explicit worker root")
    return path


def _bytes_new(path, value, *, disk=None, category="artifacts"):
    if disk is not None:
        disk.charge(len(value), category=category)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())


def _json_new(path, value, *, disk=None, failure=False):
    if disk is not None:
        disk.charge(len(canonical_json(value)) + 1, failure=failure)
    write_json_new(path, value)


def _persist_spectrum(result, *, identity, spectrum_path, observation_path=None, disk=None):
    from reelcache.spectra import write_spectrum_json

    if identity is None:
        if disk is not None:
            disk.charge(len(canonical_json(result.to_dict())))
        return {"spectrum_sha256": write_spectrum_json(result, spectrum_path)}
    from reelcache.rank_analysis import observation_from_spectrum, write_spectrum_observation

    if disk is not None:
        raw = canonical_json(result.to_dict())
        row = observation_from_spectrum(
            result, identity=identity, spectrum_sha256=hashlib.sha256(raw).hexdigest(),
        )
        try:
            row["spectrum_path"] = str(spectrum_path.resolve().relative_to(observation_path.parent.resolve()))
        except ValueError:
            row["spectrum_path"] = str(spectrum_path.resolve())
        numeric = {"schema": "reelcache_rank_observation_v1", "observation": row}
        disk.charge(len(raw) + len(canonical_json(numeric)) + 1)
    return write_spectrum_observation(
        result, identity=identity, spectrum_path=spectrum_path, observation_path=observation_path,
    )


def _file_record(path, root):
    return {"path": str(path.relative_to(root)), "sha256": file_sha256(path), "bytes": path.stat().st_size}


def _hook_signature(pipeline):
    generator = pipeline.generator
    return tuple(
        (
            id(module), tuple(module._forward_hooks.items()), tuple(module._forward_pre_hooks.items()),
            "forward" in module.__dict__, module.__dict__.get("forward"),
        )
        for module in generator.modules()
    )


def _parameter_signature(pipeline):
    modules = [pipeline.generator]
    text = getattr(pipeline, "text_encoder", None)
    if hasattr(text, "parameters"):
        modules.append(text)
    return tuple(
        (id(parameter), parameter.data_ptr(), parameter._version, parameter.dtype, tuple(parameter.shape))
        for module in modules for parameter in module.parameters()
    )


def _generation_value(value, active=None):
    import torch
    from reelcache.adapters.causvid import tensor_digest

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else ("float", repr(value))
    if isinstance(value, torch.Tensor):
        return ("tensor", str(value.dtype), tuple(value.shape), tensor_digest(value))
    if isinstance(value, (torch.dtype, torch.device)):
        return str(value)
    if callable(value):
        function = getattr(value, "__func__", value)
        return ("callable", id(function), id(getattr(function, "__code__", None)))
    active = set() if active is None else active
    if id(value) in active:
        return ("cycle", id(value))
    active.add(id(value))
    try:
        if isinstance(value, Mapping):
            return tuple((str(key), _generation_value(item, active)) for key, item in sorted(
                value.items(), key=lambda pair: str(pair[0]),
            ))
        if isinstance(value, (list, tuple)):
            return tuple(_generation_value(item, active) for item in value)
        if is_dataclass(value):
            return tuple((item.name, _generation_value(getattr(value, item.name), active)) for item in fields(value))
        if hasattr(value, "__dict__") and not isinstance(value, torch.nn.Module):
            return (type(value).__module__, type(value).__qualname__, _generation_value(vars(value), active))
        return (type(value).__module__, type(value).__qualname__, id(value))
    finally:
        active.remove(id(value))


def _generation_signature(pipeline, config):
    generator, model = pipeline.generator, pipeline.generator.model
    text = getattr(pipeline, "text_encoder", None)
    tokenizer = getattr(text, "tokenizer", None)

    def selected(obj, names):
        return {name: getattr(obj, name) for name in names if hasattr(obj, name)}

    schedulers = []
    for owner in (pipeline, generator):
        scheduler = getattr(owner, "scheduler", None)
        if scheduler is not None:
            schedulers.append((
                id(scheduler), type(scheduler).__module__, type(scheduler).__qualname__,
                _generation_value(vars(scheduler)),
                _generation_value(getattr(scheduler, "add_noise", None)),
            ))
    controls = {
        "adapter": config,
        "pipeline": selected(pipeline, (
            "args", "config", "generator_model_name", "denoising_step_list",
            "num_transformer_blocks", "frame_seq_length", "num_frame_per_block",
        )),
        "generator": selected(generator, ("config", "seq_len", "uniform_timestep")),
        "model": selected(model, (
            "model_type", "patch_size", "num_heads", "dim", "num_layers", "num_frame_per_block",
            "freqs", "freq_dim", "text_len", "window_size", "eps",
        )),
        "text": selected(text, ("config", "text_len", "max_length", "dtype")),
        "tokenizer": selected(tokenizer, ("name", "seq_len", "clean", "vocab_size")),
        "tokenizer_settings": selected(getattr(tokenizer, "tokenizer", None), (
            "padding_side", "truncation_side", "model_max_length", "add_bos_token", "add_eos_token",
            "clean_up_tokenization_spaces", "init_kwargs",
        )),
        "modules": [
            selected(module, ("training", "eps", "p", "num_heads", "head_dim", "window_size", "qk_norm"))
            for owner in (generator, text) if hasattr(owner, "modules") for module in owner.modules()
        ],
    }
    return _generation_value(controls), tuple(schedulers)


def _native_cache_empty(pipeline):
    return (
        all(getattr(pipeline, name, None) is None for name in ("kv_cache1", "kv_cache2", "crossattn_cache"))
        and getattr(pipeline, "_reelcache_causvid_owner", None) is None
    )


def _reset_native_cache(pipeline):
    if getattr(pipeline, "_reelcache_causvid_owner", None) is not None:
        raise StateError("A CausVid adapter still owns the native cache during request cleanup")
    for name in ("kv_cache1", "kv_cache2", "crossattn_cache"):
        setattr(pipeline, name, None)
    model = pipeline.generator.model
    if hasattr(model, "block_mask"):
        model.block_mask = None


class CausVidRankWorker:
    """Serial resident-model engine with injectable CPU-fake dependencies.

    Constructing an engine is not authorization to launch native code. The CLI
    verifies source, protocol and asset receipts before constructing its native
    pipeline. An in-flight request failure aborts this entire model epoch.
    """

    def __init__(
        self, pipeline, model, *, config, output_root, asset_id, fps,
        protocol=None, provenance=None, conditioning_fn=None, attention_scope=None,
        observer_factory=None, rank_policy_factory=None, ffmpeg=None, revalidate=None,
        observation_evidence_kind=None,
    ):
        self.pipeline, self.model, self.config = pipeline, model, config
        rgb_windows(1, fps)
        self.output_root = Path(output_root).resolve()
        self.asset_id, self.fps, self.protocol = asset_id, fps, protocol
        self.provenance = json_copy({} if provenance is None else provenance)
        self.conditioning_fn = conditioning_fn or pipeline.text_encoder
        self.attention_scope = attention_scope or nullcontext
        self.observer_factory, self.rank_policy_factory = observer_factory, rank_policy_factory
        self.ffmpeg, self.revalidate = ffmpeg, revalidate
        if observation_evidence_kind not in (None, "native", "synthetic_fixture"):
            raise ConfigurationError("Numeric observations must declare native versus synthetic_fixture evidence")
        self.observation_evidence_kind = observation_evidence_kind
        self._lock = threading.Lock()
        self._poisoned, self._closed = False, False
        self._requests, self._namespaces = set(), set()
        if not _native_cache_empty(pipeline):
            raise StateError("A resident worker requires a pristine pipeline, not an adopted native history")
        if getattr(pipeline, "vae", None) is not None:
            raise ConfigurationError("Latent generation must not retain a VAE from another execution epoch")
        self._hooks = _hook_signature(pipeline)
        self._parameters = _parameter_signature(pipeline)
        self._generation = _generation_signature(pipeline, config)

    def _check_generation(self):
        if _generation_signature(self.pipeline, self.config) != self._generation:
            self._poisoned = True
            raise StateError("abort_epoch: generation schedule, scheduler or configuration drifted")

    def _acquire_census_detector(self, document, directory, context, tensor, observation, row, *, disk=None):
        from reelcache.factorization import DetectorSpec, acquire_anchor_directions
        from reelcache.rank_analysis import detector_records_from_acquisition

        declaration = document["capture"]["detector"]
        spec = DetectorSpec.from_dict(declaration["spec"])
        root = directory / "detectors"
        root.mkdir(mode=0o700, exist_ok=True)
        name = f"chunk-{observation['boundary']:04d}-layer-{context.key.layer:02d}-{context.key.target}"
        started = time.perf_counter()
        try:
            acquisition = acquire_anchor_directions(
                tensor, context, spec=spec, max_workspace_bytes=declaration["workspace_bytes"],
                solver_workspace_bytes=declaration["solver_workspace_bytes"],
            )
            payload = acquisition.to_dict()
            del acquisition
            if payload["source_sha256"] != row["source_provenance"]["original_bytes_sha256"]:
                raise IntegrityError("Fixed detector did not acquire the same original raw source bytes")
            records = detector_records_from_acquisition(payload, observation=observation)
            if any(record["detector_sha256"] != spec.fingerprint for record in records):
                raise IntegrityError("Fixed acquisition exporter differs from the admitted detector recipe")
            digest = records[0]["acquisition_sha256"]
            path = root / f"{name}.json"
            _json_new(path, {
                "schema": "reelcache_rank_detector_observations_v1",
                "detector_spec": spec.to_dict(), "detector_sha256": spec.fingerprint, "observations": records,
            }, disk=disk)
            row["detector_records"] = {
                **_file_record(path, self.output_root), "detector_sha256": spec.fingerprint,
                "acquisition_sha256": digest,
            }
            work = {
                "schema": "reelcache.causvid.detector_work.v1", "status": "complete",
                "request_id": document["request_id"], "source_identity": observation["source_identity"],
                "detector_sha256": spec.fingerprint, "acquisition_sha256": digest,
                "outer_wall_seconds": time.perf_counter() - started,
                "acquisition_metrics": payload["metrics"],
                "included_in_online_selector_cost": False, "included_in_full_request_wall_time": True,
                "timing_scope": "offline_RAW_calibration_fixed_QR_q2_acquisition_and_record_persistence",
                "timing_relationship": "one acquisition serves both thresholds; nested times are not extra charges",
            }
            work_path = root / f"{name}.work.json"
            _json_new(work_path, work, disk=disk)
            row["detector_telemetry"] = _file_record(work_path, self.output_root)
        except BaseException as exc:
            metrics = getattr(exc, "metrics", None)
            path = root / f"{name}.failure.json"
            _json_new(path, {
                "schema": "reelcache.causvid.detector_work.v1", "status": "failed", "error": repr(exc),
                "request_id": document["request_id"], "context": context.to_dict(),
                "source_provenance": row["source_provenance"], "detector_sha256": spec.fingerprint,
                "outer_wall_seconds": time.perf_counter() - started,
                "acquisition_metrics": json_copy(metrics) if isinstance(metrics, Mapping) else None,
                "included_in_online_selector_cost": False, "included_in_full_request_wall_time": True,
            }, disk=disk, failure=True)
            row["detector_failure"] = _file_record(path, self.output_root)
            raise

    def _observer(self, document, directory, runtime, rows, *, disk=None):
        from reelcache.adapters.causvid import tensor_digest
        from reelcache.serialization import serialize_tensors

        capture = document["capture"]
        rank_metadata = rank_control_metadata(document)
        if capture.get("detector") is not None and self.observation_evidence_kind is None:
            raise ConfigurationError("Detector calibration requires a prebound canonical study source identity")
        retain = {(row["chunk_id"], row["layer"], row["target"]) for row in capture["retain"]}
        archive_bytes = 0
        extra = None if self.observer_factory is None else self.observer_factory(document, directory, runtime)
        spectrum_directory = directory / "spectra"
        telemetry_directory = directory / "spectrum_work"
        if capture["spectra"]:
            spectrum_directory.mkdir(mode=0o700)
            telemetry_directory.mkdir(mode=0o700)
        observation_directory = directory / "observations"
        if capture["spectra"] and self.observation_evidence_kind is not None:
            observation_directory.mkdir(mode=0o700)

        def observe(context, tensor):
            nonlocal archive_bytes
            chunk = context.token_start // context.token_count
            name = f"chunk-{chunk:04d}-layer-{context.key.layer:02d}-{context.key.target}"
            runtime_head = runtime.branch_info("request")["head_id"]
            parent_head = runtime_head if chunk else None
            native_execution = self.observation_evidence_kind == "native"
            original_bytes_sha256 = tensor_digest(tensor)
            provenance = {
                **{
                    key: value for key, value in self.provenance.items()
                    if key not in ("epoch_model_load_seconds", "resident_model_parameter_and_buffer_bytes")
                },
                "request_id": document["request_id"], "cell_id": document["cell_id"],
                "case_id": document["case_id"], "trajectory_id": document["trajectory_id"],
                "arm_id": document["arm_id"], "split": document["split"],
                "group_id": document["prompt_group_id"],
                "prompt_group_id": document["prompt_group_id"], "text": document["text"],
                "seed": document["seed"], "conditioning_identity": document["conditioning_identity"],
                "history_mode": document["history_mode"], "branch_id": "request",
                "history_head_id": parent_head, "parent_head": parent_head, "runtime_head_id": runtime_head,
                "clean_forward": "separate_native_t0", "context": context.to_dict(),
                "raw_native_tensor_sha256": original_bytes_sha256,
                "original_bytes_sha256": original_bytes_sha256,
                "evidence_kind": self.observation_evidence_kind or "synthetic_fixture",
                "native_execution": native_execution, "fixture": not native_execution,
                **rank_metadata,
            }
            observation_identity = None
            if self.observation_evidence_kind is not None:
                from reelcache.rank_analysis import rank_observation_identity

                observation_identity = rank_observation_identity(
                    case_id=document["case_id"], group_id=document["prompt_group_id"],
                    split=document["split"], seed=document["seed"], cell_id=document["cell_id"],
                    layer=context.key.layer, target=context.key.target, boundary=chunk,
                    trajectory_id=document["trajectory_id"], evidence_kind=self.observation_evidence_kind,
                    source_bindings={
                        "protocol_id": self.provenance.get("protocol_id"), "request_id": document["request_id"],
                        "group_id": document["prompt_group_id"],
                        "arm_id": document["arm_id"], "branch_id": "request", "parent_head": parent_head,
                        "history_head_id": parent_head, "runtime_head_id": runtime_head,
                        "native_execution": native_execution, "fixture": not native_execution,
                        "original_bytes_sha256": original_bytes_sha256,
                        "original_bytes_hash_format": "sha256-little-endian-row-major-bytes-no-header-v1",
                        "original_bytes_byte_order": sys.byteorder,
                        **{
                            name: self.provenance[name]
                            for name in ("asset_id", "launch_closure_id", "source_closure_id")
                            if name in self.provenance
                        },
                    },
                )
                cell = study_cell(document["cell_id"])
                if (
                    context.model != self.model or context.shape != (cell.tokens, cell.channels)
                    or context.model.layers != cell.layers or context.token_start % cell.tokens
                    or (context.frames, context.height, context.width) != (
                        cell.frames, cell.token_height, cell.token_width,
                    )
                    or context.dtype != observation_identity["source_dtype"]
                    or context.key_space != ("pre_rope" if context.key.target == "K" else "none")
                ):
                    raise IntegrityError("Numeric observation identity must match the original native context before capture")
                provenance.update(observation_identity)
                source_identity = identity(provenance)
                observation_identity["source_identity"] = source_identity
                provenance["source_identity"] = source_identity
            row = {"context": context.to_dict(), "source_provenance": provenance}
            if observation_identity is not None:
                row["observation_identity"] = json_copy(observation_identity)
            selected = (chunk, context.key.layer, context.key.target) in retain
            if selected:
                raw = serialize_tensors({"source": tensor.detach().cpu()})
                if archive_bytes + len(raw) > document["memory_budgets"]["source_archive_bytes"]:
                    raise CapacityError("Pristine-source archive exceeded its admitted bytes")
                path = directory / "sources" / f"{name}.reelcache"
                _bytes_new(path, raw, disk=disk, category="archive")
                archive_bytes += len(raw)
                row["retained_source"] = _file_record(path, self.output_root)
            if capture["spectra"]:
                from reelcache.spectra import SpectrumSource, exact_spectrum

                source = SpectrumSource(
                    identity=provenance.get("source_identity", identity(provenance)),
                    shape=tuple(tensor.shape), dtype=context.dtype,
                    target=context.key.target, key_space=context.key_space,
                    retention="exact-input-retained" if selected else "spectrum-only", provenance=provenance,
                )
                tick = time.perf_counter()
                try:
                    result = exact_spectrum(
                        tensor, source=source, tile_rows=capture["tile_rows"],
                        max_workspace_bytes=document["memory_budgets"]["spectrum_workspace_bytes"],
                    )
                except BaseException as exc:
                    wall_seconds = time.perf_counter() - tick
                    metrics = getattr(exc, "metrics", None)
                    metrics = json_copy(metrics) if isinstance(metrics, Mapping) else None
                    row["spectrum_work"] = {
                        "wall_seconds": wall_seconds, "tile_rows": capture["tile_rows"],
                        "workspace_budget_bytes": document["memory_budgets"]["spectrum_workspace_bytes"],
                        "timing_scope": (
                            "failed_offline_source_callback_outer_host_wall_only" if metrics is None else
                            "failed_offline_source_callback_outer_host_wall_with_recorded_internal_work"
                        ),
                        "oracle_failure_metrics": metrics,
                        "oracle_timings_seconds": None if metrics is None else {
                            name: value for name, value in metrics.items() if name.endswith("_seconds")
                        },
                        "oracle_work_counts": None if metrics is None else {
                            name: value for name, value in metrics.items() if name.endswith("_count")
                        },
                        "timing_relationship": (
                            "failed-call internal timings/counts are unavailable" if metrics is None else
                            "internal total_seconds=exclusive_pipeline_seconds+eigensolve_seconds; "
                            "nested phase diagnostics are not additional charges; outer wall includes caller overhead"
                        ),
                    }
                    telemetry_path = telemetry_directory / f"{name}.failure.json"
                    _json_new(telemetry_path, {
                        "schema": "reelcache.causvid.spectrum_work.v1", "status": "failed",
                        "request_id": document["request_id"], "case_id": document["case_id"],
                        "boundary": chunk, "layer": context.key.layer, "target": context.key.target,
                        "context": context.to_dict(), "source_provenance": provenance,
                        "error": repr(exc), "spectrum_sha256": None, "work": row["spectrum_work"],
                    }, disk=disk, failure=True)
                    row["spectrum_failure"] = _file_record(telemetry_path, self.output_root)
                    rows.append(row)
                    raise
                wall_seconds = time.perf_counter() - tick
                path = spectrum_directory / f"{name}.json"
                if observation_identity is None:
                    artifact_digest = _persist_spectrum(
                        result, identity=None, spectrum_path=path, disk=disk,
                    )["spectrum_sha256"]
                else:
                    observation_path = observation_directory / f"{name}.json"
                    receipt = _persist_spectrum(
                        result, identity=observation_identity, spectrum_path=path,
                        observation_path=observation_path, disk=disk,
                    )
                    artifact_digest = receipt["spectrum_sha256"]
                    row["numeric_observation"] = {
                        "path": str(observation_path.relative_to(self.output_root)),
                        "sha256": receipt["observation_sha256"], "bytes": observation_path.stat().st_size,
                    }
                row["spectrum"] = {
                    "path": str(path.relative_to(self.output_root)),
                    "sha256": artifact_digest, "bytes": path.stat().st_size,
                }
                row["spectrum_work"] = {
                    "wall_seconds": wall_seconds, "tile_rows": capture["tile_rows"],
                    "workspace_budget_bytes": document["memory_budgets"]["spectrum_workspace_bytes"],
                    "timing_scope": "offline_source_callback_Gram_eigh_and_native_tile_transfers_not_online_selection",
                    "oracle_timings_seconds": result.timings.to_dict(),
                    "oracle_work_counts": json_copy(result.work_counts()),
                    "timing_relationship": "oracle subphases are nested; do not add them to wall_seconds",
                }
                telemetry_path = telemetry_directory / f"{name}.json"
                _json_new(telemetry_path, {
                    "schema": "reelcache.causvid.spectrum_work.v1", "status": "complete",
                    "request_id": document["request_id"],
                    "case_id": document["case_id"], "boundary": chunk,
                    "layer": context.key.layer, "target": context.key.target,
                    "spectrum_sha256": artifact_digest, "work": row["spectrum_work"],
                }, disk=disk)
                row["spectrum_telemetry"] = _file_record(telemetry_path, self.output_root)
                if capture.get("detector") is not None:
                    try:
                        self._acquire_census_detector(
                            document, directory, context, tensor, receipt["observation"], row, disk=disk,
                        )
                    except BaseException:
                        rows.append(row)
                        raise
            if extra is not None:
                outcome = extra(context, tensor)
                if inspect.isawaitable(outcome):
                    if inspect.iscoroutine(outcome):
                        outcome.close()
                    raise ConfigurationError("Worker clean observers must be synchronous")
            rows.append(row)

        return observe

    def _rank_source_provenance(self, context, captures):
        matches = [row for row in captures if row["context"] == context.to_dict()]
        if len(matches) != 1 or "observation_identity" not in matches[0]:
            raise IntegrityError("Adaptive study provenance requires one prebound pristine source observation")
        captured = matches[0]
        if any(
            captured["source_provenance"].get(name) != value
            for name, value in captured["observation_identity"].items()
        ):
            raise IntegrityError("Prebound source identity changed after the pristine observation")
        return json_copy(captured["source_provenance"])

    def _export_policy_oracles(self, document, directory, runtime, policy, chunk, captures, *, disk=None):
        """Persist the core's original single-solve results without adding labels."""
        from reelcache.spectra import SpectrumResult

        root = directory / "policy_oracle"
        root.mkdir(mode=0o700, exist_ok=True)
        observation_root = directory / "observations"
        token_start = chunk * self.config.chunk_tokens
        current = {
            MatrixKey.from_dict(row["context"]["key"]): row for row in captures
            if row["context"]["token_start"] == token_start
        }
        if set(current) != set(policy.keys):
            raise IntegrityError("Exact-policy export needs every actual pre-compression capture binding")
        records = []
        expected_chunk_id = f"chunk-{chunk:06d}"
        for key in policy.keys:
            decisions = [
                item for item in runtime.history_rank_decisions("request", key)
                if item.chunk_id == expected_chunk_id
            ]
            if len(decisions) != 1:
                raise IntegrityError("Exact-policy publication needs one committed decision for its exact chunk ID")
            decision = decisions[0]
            captured = current[key]
            context = TensorContext.from_dict(captured["context"])
            bound = decision.context
            if (
                decision.policy_hash != policy.fingerprint or bound.key != key
                or bound.model != context.model or bound.shape != context.shape
                or bound.dtype != context.dtype or bound.key_space != context.key_space
                or (bound.token_start, bound.frames, bound.height, bound.width) != (
                    context.token_start, context.frames, context.height, context.width,
                )
                or decision.input_sha256 != captured["source_provenance"]["raw_native_tensor_sha256"]
            ):
                raise IntegrityError("Committed exact-policy result differs from its original native capture")
            if decision.oracle_observation is None:
                raise IntegrityError("Exact-policy decision omitted its complete single-solve spectrum")
            result = SpectrumResult.from_dict(json_copy(decision.oracle_observation))
            if (
                result.source.shape != context.shape or result.source.dtype != context.dtype
                or result.source.target != key.target or result.source.key_space != context.key_space
                or result.source.provenance.get("original_bytes_sha256") != decision.input_sha256
            ):
                raise IntegrityError("The policy's oracle declaration differs from its bound original bytes")
            path = root / f"chunk-{chunk:04d}-layer-{key.layer:02d}-{key.target}.json"
            observation_identity = captured.get("observation_identity")
            numeric_observation = None
            if self.observation_evidence_kind is not None:
                expected = {
                    "case_id": document["case_id"], "prompt_group": document["prompt_group_id"],
                    "split": document["split"], "seed": document["seed"], "cell": document["cell_id"],
                    "trajectory_id": document["trajectory_id"], "request_id": document["request_id"],
                    "arm_id": document["arm_id"], "layer": key.layer, "target": key.target, "boundary": chunk,
                    "evidence_kind": self.observation_evidence_kind,
                }
                if not isinstance(observation_identity, Mapping) or any(
                    observation_identity.get(name) != value for name, value in expected.items()
                ):
                    raise IntegrityError("The prebound exact-policy observation differs from the declared request")
                observation_path = observation_root / path.name
                receipt = _persist_spectrum(
                    result, identity=observation_identity, spectrum_path=path,
                    observation_path=observation_path, disk=disk,
                )
                digest = receipt["spectrum_sha256"]
                numeric_observation = {
                    "path": str(observation_path.relative_to(self.output_root)),
                    "sha256": receipt["observation_sha256"], "bytes": observation_path.stat().st_size,
                }
            else:
                if observation_identity is not None:
                    raise IntegrityError("A bound study observation needs an explicit worker evidence kind")
                digest = _persist_spectrum(
                    result, identity=None, spectrum_path=path, disk=disk,
                )["spectrum_sha256"]
            record = {
                "key": key.to_dict(), "chunk_id": decision.chunk_id, "policy_hash": decision.policy_hash,
                "original_bytes_sha256": decision.input_sha256,
                "oracle_source_sha256": result.source.tensor_sha256,
                "spectrum": {
                    "path": str(path.relative_to(self.output_root)), "sha256": digest,
                    "bytes": path.stat().st_size,
                },
                "prebound_capture": captured["source_provenance"],
                "study_observation_eligible": numeric_observation is not None,
                "scope": (
                    "original_single_solve_with_prebound_study_identity"
                    if numeric_observation is not None else
                    "original_core_numerics; cohort labels were not bound inside the core oracle source"
                ),
            }
            captured["spectrum"] = record["spectrum"]
            if numeric_observation is not None:
                record["numeric_observation"] = numeric_observation
                captured["numeric_observation"] = numeric_observation
            records.append(record)
        index = directory / f"policy-oracle-index-{chunk:04d}.json"
        _json_new(index, {
            "schema": "reelcache.causvid.policy_oracle_index.v1",
            "request_id": document["request_id"], "case_id": document["case_id"],
            "trajectory_id": document["trajectory_id"], "arm_id": document["arm_id"],
            "records": records, "additional_eigensolves": 0,
            "study_observation_eligible": all(row["study_observation_eligible"] for row in records),
            "evidence_kind": self.observation_evidence_kind,
            "source_provenance_rewritten": False,
        }, disk=disk)
        return _file_record(index, self.output_root)

    def run(self, request):
        if self._closed or self._poisoned:
            raise StateError("abort_epoch: this resident CausVid worker cannot accept another request")
        if not self._lock.acquire(blocking=False):
            raise StateError("CausVid model epochs accept only isolated serial requests")
        try:
            document = request.document if isinstance(request, CausVidRankRequest) else request
            request = CausVidRankRequest.from_dict(document, protocol=self.protocol, asset_id=self.asset_id)
            document = request.document
            if "latent_source" in document or "reference_source" in document:
                raise ConfigurationError("Saved input references belong to a separate decode/native-reference epoch")
            if document["request_id"] in self._requests or document["output_namespace"] in self._namespaces:
                raise StateError("Request IDs and output namespaces cannot be reused within an epoch")
            directory = _inside(self.output_root, document["output_namespace"])
            if directory.exists():
                raise StateError("A request output namespace must be new; prior evidence is immutable")
            specs = (
                codec_specs(document["spec_config"], self.model.layers, ffmpeg=self.ffmpeg)
                if document["rank_policy"] is None else {}
            )
            if any(spec.kind == "svd_hevc" for spec in specs.values()) and self.ffmpeg is None:
                raise ConfigurationError("HEVC requests require a verified explicit local FFmpeg asset")
            policy = None
            if document["rank_policy"] is not None:
                if self.rank_policy_factory is None:
                    raise ConfigurationError("This worker has no bound adaptive-policy factory")
                policy = self.rank_policy_factory(document["rank_policy"], self.model, self.config)
                _admit_policy(policy, document, self.ffmpeg)
            if self.revalidate is not None:
                self.revalidate()
            if document["capture"].get("native_history_audit", False) and len(self.pipeline.denoising_step_list) > 4:
                raise CapacityError("Native-history diagnostics admit at most four noisy passes per chunk")
            if document["capture"].get("detector") is not None:
                from reelcache.factorization import DetectorSpec

                DetectorSpec.from_dict(document["capture"]["detector"]["spec"]).validate_runtime()
            self._check_generation()
            if (
                not _native_cache_empty(self.pipeline) or _hook_signature(self.pipeline) != self._hooks
                or _parameter_signature(self.pipeline) != self._parameters
            ):
                self._poisoned = True
                raise StateError("abort_epoch: native caches/hooks/parameters changed across requests")
            return self._run_admitted(request, directory, specs, policy)
        finally:
            self._lock.release()

    def _run_admitted(self, request, directory, specs, policy):
        import torch
        from reelcache.adapters.causvid import CausVidAdapter, tensor_digest, unique_tensor_bytes
        from reelcache.manager import KVRuntime
        from reelcache.serialization import serialize_tensors

        document = request.document
        budget = CausVidMemoryBudgets.from_dict(document["memory_budgets"])
        disk = _RequestOutputBudget(document, self.output_root, "generate")
        directory.mkdir(parents=True, exist_ok=False, mode=0o700)
        self._requests.add(document["request_id"])
        self._namespaces.add(document["output_namespace"])
        _json_new(directory / "request.json", document, disk=disk)
        runtime = adapter = native_history_stream = None
        noise = conditioning = state = None
        committed_head_before_step = None
        rows, captures = [], []
        native_history_count = 0
        started = time.perf_counter()
        try:
            _reset_native_cache(self.pipeline)
            runtime = KVRuntime(
                directory / "runtime", self.model, max_resident_bytes=budget.resident_bytes,
                max_store_bytes=budget.store_bytes,
            )
            kwargs = {} if policy is None else {"rank_policy": policy}
            if policy is not None and self.observation_evidence_kind is not None:
                kwargs["rank_source_provenance"] = lambda context: self._rank_source_provenance(context, captures)
            if document["capture"].get("native_history_audit", False):
                native_history_stream = (directory / "native-history.jsonl").open("xb")

                def observe_native(record, _tensor):
                    nonlocal native_history_count
                    raw = canonical_json(record) + b"\n"
                    if len(raw) > 65536:
                        raise CapacityError("Native-history scalar event exceeds its bounded metadata workspace")
                    disk.charge(len(raw))
                    native_history_stream.write(raw)
                    native_history_stream.flush()
                    native_history_count += 1

                kwargs["native_history_observer"] = observe_native
            adapter = CausVidAdapter(
                self.pipeline, runtime, "request", config=self.config, specs=None if policy is not None else specs,
                history_mode=document["history_mode"], max_quality_history_bytes=budget.quality_history_bytes,
                max_attention_bytes=budget.attention_bytes, max_current_bytes=budget.current_bytes,
                max_clean_host_bytes=budget.clean_host_bytes, max_clean_disk_bytes=budget.clean_disk_bytes,
                clean_staging_root=directory / "clean_staging" if budget.clean_disk_bytes else None,
                clean_observer=self._observer(document, directory, runtime, captures, disk=disk),
                source_provenance={**self.provenance, "request_id": document["request_id"],
                                   "history_mode": document["history_mode"]},
                **kwargs,
            )
            from reelcache.codecs import using_ffmpeg

            codec_scope = nullcontext() if self.ffmpeg is None else using_ffmpeg(self.ffmpeg)
            with torch.no_grad(), self.attention_scope(), codec_scope, adapter:
                if adapter.device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(adapter.device)
                conditioning = self.conditioning_fn(text_prompts=[document["text"]])
                self._check_generation()
                rng = torch.Generator(device=adapter.device).manual_seed(document["seed"])
                initial_rng = rng.get_state().clone()
                noise = torch.empty(
                    (1, document["chunks"] * self.config.frames_per_chunk, *self.config.latent_tail),
                    device=adapter.device, dtype=adapter.dtype,
                ).normal_(generator=rng)
                denoising_rng = rng.get_state().clone()
                adapter.start(
                    noise, conditioning, rng, conditioning_id=document["conditioning_identity"],
                )
                noise = noise.cpu()
                for chunk in range(document["chunks"]):
                    committed_head_before_step = runtime.branch_info("request")["head_id"]
                    tick = time.perf_counter()
                    sealed = adapter.step()
                    self._check_generation()
                    memory = adapter.memory_report()
                    if any(memory[name] for name in (
                        "live_mutable_kv_bytes", "live_clean_capture_cpu_bytes",
                        "live_clean_capture_disk_bytes", "live_attention_read_bytes",
                    )) or runtime.active_leases or runtime.resident_bytes:
                        raise StateError("Clean publication retained native scratch or history leases")
                    if document["history_mode"] == "reload" and memory["retained_reconstruction_dictionary"]:
                        raise StateError("Bounded reload execution retained a dense history")
                    if adapter.device.type == "cuda":
                        torch.cuda.synchronize(adapter.device)
                        memory["device_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(adapter.device)
                        memory["device_peak_reserved_bytes"] = torch.cuda.max_memory_reserved(adapter.device)
                        if memory["device_peak_allocated_bytes"] > budget.device_bytes:
                            raise CapacityError("Measured native allocation peak exceeded the explicit device grant")
                    policy_oracle_index = (
                        self._export_policy_oracles(
                            document, directory, runtime, policy, chunk, captures, disk=disk,
                        )
                        if policy is not None and policy.solver == "exact_oracle" else None
                    )
                    rows.append({
                        "chunk_id": chunk, "manifest_id": sealed, "elapsed_seconds": time.perf_counter() - tick,
                        "memory": memory, "clean_audit": adapter.audit[-1],
                        "policy_oracle_index": policy_oracle_index,
                    })
                state = {
                    "latents": adapter.latent_state, "noise": noise,
                    "initial_rng_state": initial_rng, "denoising_rng_state": denoising_rng,
                    "rng_state": adapter.rng_state, "prompt_embeds": conditioning["prompt_embeds"].detach().cpu(),
                }
                if 4 * unique_tensor_bytes(state.values()) > budget.host_bytes - budget.model_host_bytes:
                    raise CapacityError("Saved request state exceeds its host serialization admission")
                latent_path = directory / "latents.reelcache"
                _bytes_new(latent_path, serialize_tensors(state), disk=disk)
                result = {
                    "schema": "reelcache.causvid.rank_result.v1", "status": "complete",
                    "operation": "generate", "request": document, "model": self.model.to_dict(),
                    **rank_control_metadata(document),
                    "asset_id": self.asset_id, "provenance": self.provenance,
                    "generation_from_scratch": True, "supplied_prefix": False,
                    "initial_noise_sha256": tensor_digest(noise),
                    "conditioning_tensor_sha256": tensor_digest(state["prompt_embeds"]),
                    "initial_rng_sha256": tensor_digest(initial_rng),
                    "denoising_rng_sha256": tensor_digest(denoising_rng),
                    "final_rng_sha256": tensor_digest(state["rng_state"]),
                    "latents_sha256": tensor_digest(state["latents"]),
                    "latent_file": _file_record(latent_path, self.output_root),
                    "rgb": {**rgb_windows(document["chunks"], self.fps), "decoded": False},
                    "chunks": rows, "clean_captures": captures,
                    "history_head": runtime.branch_info("request"),
                    "quality_only_decoded_history": document["history_mode"] == "quality_hot",
                    "native_parity_claim": False, "metrics_loaded": False, "vae_loaded": False,
                    "native_execution": self.observation_evidence_kind == "native",
                    "fixture": self.observation_evidence_kind != "native",
                    "disk_admission": disk.plan,
                    "elapsed_seconds": time.perf_counter() - started,
                }
                if native_history_stream is not None:
                    expected = (
                        document["chunks"] * 2 * self.model.layers * (len(self.pipeline.denoising_step_list) + 2)
                    )
                    if native_history_count != expected:
                        raise IntegrityError("Native-history diagnostic did not cover every native read and publication")
                    native_history_stream.flush()
                    os.fsync(native_history_stream.fileno())
                    native_history_stream.close()
                    result["native_history_audit"] = {
                        "schema": "reelcache.causvid.native_history_audit.v1",
                        "events": _file_record(directory / "native-history.jsonl", self.output_root),
                        "event_count": native_history_count, "expected_event_count": expected,
                        "diagnostic_only": True, "native_parity_claim": False,
                        "native_execution": self.observation_evidence_kind == "native",
                        "fixture": self.observation_evidence_kind != "native",
                        "overhead": adapter.memory_report()["native_history_diagnostic"],
                        "timing_scope": "request_and_chunk_times_include_diagnostic_readback_and_reload_decode",
                        "hash_domain": "actual_attention_ready_native_bytes; not_canonical_pre_rope",
                    }
                    result["elapsed_seconds"] = time.perf_counter() - started
            if runtime.active_leases or runtime.resident_bytes or not _native_cache_empty(self.pipeline):
                raise StateError("Request teardown retained caches or history leases")
            adapter = None
            runtime.close()
            runtime = None
            _reset_native_cache(self.pipeline)
            staging_root = directory / "clean_staging"
            if staging_root.exists():
                staging_root.rmdir()
            if _hook_signature(self.pipeline) != self._hooks or _parameter_signature(self.pipeline) != self._parameters:
                raise StateError("Native request hooks or resident parameters were not preserved exactly")
            if self.revalidate is not None:
                self.revalidate()
            self._check_generation()
            result["request_reset_verified"] = True
            _json_new(directory / "result.json", result, disk=disk)
            return result
        except BaseException as exc:
            self._poisoned = True
            from reelcache.online_rank import RankAbstention

            failure = {
                "schema": "reelcache.causvid.epoch_abort.v1", "status": "abort_epoch",
                "request_id": document["request_id"], "error": repr(exc),
                "reason": "An admitted request failed; do not reuse this resident model epoch.",
                "elapsed_seconds": time.perf_counter() - started,
                "timing_scope": "request_execution_through_abort_before_final_cleanup",
                "chunks": rows, "clean_captures": captures,
            }
            if native_history_stream is not None:
                failure["native_history_audit"] = {
                    "events": _file_record(directory / "native-history.jsonl", self.output_root),
                    "event_count": native_history_count, "incomplete": True, "native_parity_claim": False,
                }
            if isinstance(exc, RankAbstention):
                failure["rank_decision"] = exc.decision.to_dict()
                try:
                    current_head = runtime.branch_info("request")["head_id"] if runtime is not None else None
                except BaseException:
                    current_head = None
                failure["committed_head_before_step"] = committed_head_before_step
                failure["committed_head_after_failure"] = current_head
                failure["committed_head_preserved"] = (
                    current_head == committed_head_before_step
                    if current_head is not None and committed_head_before_step is not None else None
                )
            _json_new(directory / "failure.json", failure, disk=disk, failure=True)
            raise
        finally:
            cleanup_errors = []
            try:
                for action in (
                    None if native_history_stream is None else native_history_stream.close,
                    None if adapter is None else adapter.close,
                    None if runtime is None else runtime.close,
                    lambda: _reset_native_cache(self.pipeline),
                ):
                    if action is not None:
                        try:
                            action()
                        except BaseException as exc:
                            cleanup_errors.append(exc)
                try:
                    staging_root = directory / "clean_staging"
                    if staging_root.exists():
                        staging_root.rmdir()
                    if (
                        _hook_signature(self.pipeline) != self._hooks
                        or _parameter_signature(self.pipeline) != self._parameters
                        or (runtime is not None and (runtime.active_leases or runtime.resident_bytes))
                    ):
                        raise StateError("abort_epoch: native hook/parameter/lease cleanup could not be verified")
                except BaseException as exc:
                    cleanup_errors.append(exc)
                if cleanup_errors:
                    self._poisoned = True
                    raise cleanup_errors[0]
            finally:
                noise = conditioning = state = adapter = runtime = None
                gc.collect()

    def close(self):
        if self._lock.locked():
            raise StateError("Cannot close a worker while a serial request is active")
        if not self._poisoned:
            self._check_generation()
        if (
            not _native_cache_empty(self.pipeline) or _hook_signature(self.pipeline) != self._hooks
            or _parameter_signature(self.pipeline) != self._parameters
        ):
            self._poisoned = True
            raise StateError("abort_epoch: resident worker cleanup is uncertain")
        self._closed = True


def _copy_function(function, globals_update):
    copied = types.FunctionType(
        function.__code__, dict(function.__globals__, **globals_update),
        function.__name__, function.__defaults__, function.__closure__,
    )
    copied.__kwdefaults__ = function.__kwdefaults__
    return copied


def build_latent_pipeline(native, config, *, device):
    """Use native construction with an instance-local no-VAE constructor seam."""
    cls = native.InferencePipeline
    function = cls.__init__
    if not {"get_diffusion_wrapper", "get_text_encoder_wrapper", "get_vae_wrapper"} <= set(function.__code__.co_names):
        raise ConfigurationError("The approved native pipeline constructor seam changed")
    init = _copy_function(function, {"get_vae_wrapper": lambda **_kwargs: lambda *_args: None})
    pipeline = cls.__new__(cls)
    init(pipeline, config, device)
    if pipeline.vae is not None:
        raise IntegrityError("A latent-generation epoch unexpectedly loaded a VAE")
    return pipeline


def _check_native_components(assets, operation):
    base = assets.paths["base_model"]
    required = (
        ("models_t5_umt5-xxl-enc-bf16.pth", "google/umt5-xxl/tokenizer_config.json")
        if operation in ("generate", "native-reference") else ("Wan2.1_VAE.pth",)
    )
    for relative in required:
        if not (base / relative).is_file():
            raise IntegrityError(f"The verified native {operation} asset set is incomplete: {relative}")
    if operation in ("generate", "native-reference"):
        if not any((base / "google/umt5-xxl" / name).is_file() for name in ("tokenizer.json", "spiece.model")):
            raise IntegrityError("No explicitly staged native tokenizer vocabulary is available")
        if not any(
            path.is_file() and (path.suffix == ".safetensors" or path.name.startswith("diffusion_pytorch_model"))
            for path in base.iterdir()
        ):
            raise IntegrityError("No explicitly staged native base-model weights are available")
        manifest_names = {row["name"] for row in assets.manifest["files"]}
        for index in base.glob("*.index.json"):
            document = read_json(index)
            weights = document.get("weight_map")
            if not isinstance(weights, dict) or not weights:
                raise IntegrityError("The native base-model shard index is incomplete")
            for name in weights.values():
                path = base / _relative(name, "native base-model shard")
                if (
                    not path.resolve(strict=True).is_relative_to(base)
                    or str(path.relative_to(assets.root)) not in manifest_names
                ):
                    raise IntegrityError("An index-listed native weight shard bypasses verified local assets")


def construction_host_floor(assets, operation):
    """Conservative source-size admission floor, not a measured peak-RSS bound."""
    base = assets.paths["base_model"]
    if operation == "decode":
        return 4 * (base / "Wan2.1_VAE.pth").stat().st_size + (64 << 20)
    base_bytes = sum(
        path.stat().st_size for path in base.iterdir()
        if path.is_file() and (path.suffix == ".safetensors" or path.name.startswith("diffusion_pytorch_model"))
    )
    text_bytes = (base / "models_t5_umt5-xxl-enc-bf16.pth").stat().st_size
    return 4 * base_bytes + 3 * text_bytes + 2 * assets.paths["checkpoint"].stat().st_size + (64 << 20)


def _verified_import_seam(source_root):
    from importlib.machinery import PathFinder

    search = sys.path
    for name in ("third_party", "third_party.causvid", "third_party.causvid.modeling",
                 "third_party.causvid.modeling.causal_inference"):
        loaded = sys.modules.get(name)
        if loaded is not None and (
            (
                getattr(loaded, "__file__", None) is not None
                and not Path(loaded.__file__).resolve().is_relative_to(source_root)
            )
            or any(not Path(path).resolve().is_relative_to(source_root) for path in getattr(loaded, "__path__", ()))
        ):
            raise IntegrityError("A pre-imported namespace bypasses the verified native source root")
        # Inspect each explicit parent path without importing namespace parents.
        spec = PathFinder.find_spec(name.rsplit(".", 1)[-1], search)
        if spec is None:
            raise IntegrityError("The explicit native source tree does not resolve its import seam")
        if spec.origin is not None and not Path(spec.origin).resolve().is_relative_to(source_root):
            raise IntegrityError("Native import resolution would execute an unverified source path")
        locations = spec.submodule_search_locations
        if locations is not None:
            if any(not Path(path).resolve().is_relative_to(source_root) for path in locations):
                raise IntegrityError("A native namespace would also search unverified source roots")
            search = list(locations)


class NativeCausVidEpoch:
    """Only instantiated after the pure launch gate; no fallback model paths."""

    def __init__(self, assets, requests, *, operation, threads):
        started = time.perf_counter()
        _check_native_components(assets, operation)
        host_floor = construction_host_floor(assets, operation)
        if any(item.document["memory_budgets"]["model_host_bytes"] < host_floor for item in requests):
            raise CapacityError("The explicit native construction host budget is below its conservative source-size floor")
        if any(
            name.startswith("third_party.causvid")
            for name in sys.modules
        ):
            raise StateError("A new native model epoch must not inherit imported CausVid modules")
        import torch
        from omegaconf import OmegaConf
        from reelcache.adapters.causvid import unique_tensor_bytes
        from examples.native_causvid import instance_cross_attention_sdpa

        if torch.cuda.device_count() != 1:
            raise ConfigurationError("Native CausVid epochs require exactly one explicitly visible GPU")
        torch.set_num_threads(threads)
        torch.set_num_interop_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.pipeline = self.vae = None
        self._poisoned = False
        self.operation, self.assets = operation, assets
        self._cross_attention_scope = instance_cross_attention_sdpa
        self.device = torch.device("cuda:0")
        self.dtype = torch.bfloat16
        budgets = [CausVidMemoryBudgets.from_dict(item.document["memory_budgets"]) for item in requests]
        self.model_device_budget = min(item.model_device_bytes for item in budgets)
        source_root = assets.paths["source_root"]
        sys.path.insert(0, str(source_root))
        _verified_import_seam(source_root)
        native = importlib.import_module("third_party.causvid.modeling.causal_inference")
        for name, module in list(sys.modules.items()):
            if name.startswith("third_party.causvid") and getattr(module, "__file__", None):
                if not Path(module.__file__).resolve().is_relative_to(source_root):
                    raise IntegrityError("An imported CausVid module bypassed the verified local source mapping")
        with torch.no_grad():
            if operation in ("generate", "native-reference"):
                config = OmegaConf.load(assets.paths["config"])
                if (
                    getattr(config, "generator_name", None) != "causal_wan"
                    or getattr(config, "num_frame_per_block", None) != 3
                    or list(config.image_or_video_shape)[2:] != [16, 60, 104]
                ):
                    raise ConfigurationError("Native YAML differs from the CausVid480 geometry and causal model")
                config.model_path = str(assets.paths["base_model"])
                config.checkpoint_folder = str(assets.paths["checkpoint"].parent)
                self.pipeline = build_latent_pipeline(native, config, device=self.device)
                state = torch.load(assets.paths["checkpoint"], map_location="cpu", weights_only=True)
                if not isinstance(state, dict) or "generator" not in state:
                    raise IntegrityError("The explicit native CausVid checkpoint has no generator")
                self.pipeline.generator.load_state_dict(state["generator"], strict=True)
                del state
                self.pipeline.eval().requires_grad_(False).to(dtype=self.dtype)
                modules = self.pipeline
            else:
                self.vae = native.get_vae_wrapper(model_name="causal_wan")(str(assets.paths["base_model"]))
                self.vae.eval().requires_grad_(False).to(dtype=self.dtype)
                modules = self.vae
            self.model_bytes = unique_tensor_bytes([*modules.parameters(), *modules.buffers()])
            if self.model_bytes > self.model_device_budget:
                raise CapacityError("Native model/text or VAE allocations exceed the explicit device-model budget")
            modules.to(self.device)
        self.model = model_identity(assets.asset_id)
        gc.collect()
        self.load_seconds = time.perf_counter() - started

    def attention_scope(self):
        from contextlib import ExitStack
        from torch.nn.attention import SDPBackend, sdpa_kernel

        stack = ExitStack()
        try:
            stack.enter_context(self._cross_attention_scope(self.pipeline))
            stack.enter_context(sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]))
            return stack
        except BaseException:
            stack.close()
            raise

    def close(self):
        import torch

        if self.pipeline is not None and not _native_cache_empty(self.pipeline):
            raise StateError("Cannot release a native model epoch that still owns request caches")
        self.pipeline = self.vae = None
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def verify_latent_source(request, output_root, assets):
    document = request.document
    rgb_windows(document["chunks"], assets.fps)
    source = document.get("latent_source")
    if source is None:
        raise ConfigurationError("Decode requests require a hashed local latent_source result.json")
    path = _inside(output_root, source["path"], exists=True)
    if file_sha256(path) != source["sha256"]:
        raise IntegrityError("Generation result changed before separate VAE admission")
    result = read_json(path)
    if (
        result.get("schema") != "reelcache.causvid.rank_result.v1" or result.get("status") != "complete"
        or result.get("operation") not in ("generate", "native-reference") or result.get("asset_id") != assets.asset_id
        or result.get("generation_from_scratch") is not True or result.get("supplied_prefix") is not False
        or result.get("model") != model_identity(assets.asset_id).to_dict()
    ):
        raise IntegrityError("VAE input is not a completed from-scratch result from these exact assets")
    if result["operation"] == "native-reference" and (
        document.get("study_phase") != "native_preflight" or document["split"] != "development"
        or result["request"].get("study_phase") != "native_preflight"
        or result.get("native_pipeline_inference_called") is not True
        or result.get("adapter_instantiated") is not False
    ):
        raise ConfigurationError("Native-reference VAE input is development/native_preflight only")
    if result["operation"] == "native-reference":
        _validate_reference_request(result["request"])
    for name in (
        "case_id", "trajectory_id", "arm_id", "cell_id", "prompt_group_id", "text", "seed", "chunks", "conditioning_identity",
        "history_mode", "spec_config", "rank_policy",
    ):
        if result["request"].get(name) != document[name]:
            raise ConfigurationError("VAE input does not belong to this exact requested generation case")
    record = result["latent_file"]
    latent = _inside(output_root, record["path"], exists=True)
    if (
        latent.stat().st_size != record["bytes"] or file_sha256(latent) != record["sha256"]
        or latent.stat().st_size * 4 > document["memory_budgets"]["host_bytes"]
    ):
        raise IntegrityError("Latent state failed hash, size or host admission before VAE loading")
    _RequestOutputBudget(
        document, output_root, "decode", input_bytes=latent.stat().st_size + path.stat().st_size,
    )
    return result


def decode_request(epoch, request, output_root, source_result, *, provenance, revalidate=None):
    import numpy as np
    import torch
    from reelcache.adapters.causvid import tensor_digest
    from reelcache.serialization import deserialize_tensors

    if getattr(epoch, "_poisoned", False):
        raise StateError("abort_epoch: this VAE epoch cannot accept another request")
    document = request.document
    directory = _inside(output_root, document["output_namespace"])
    if directory.exists():
        raise StateError("Decode output namespace must be new")
    timeline = rgb_windows(document["chunks"], epoch.assets.fps)
    clear = getattr(getattr(epoch.vae, "model", None), "clear_cache", None)
    if not callable(clear):
        raise ConfigurationError("Native VAE must provide the explicit full-prefix cache reset")
    latent_path = _inside(output_root, source_result["latent_file"]["path"], exists=True)
    source_path = _inside(output_root, document["latent_source"]["path"], exists=True)
    disk = _RequestOutputBudget(
        document, output_root, "decode", input_bytes=latent_path.stat().st_size + source_path.stat().st_size,
    )
    directory.mkdir(parents=True, exist_ok=False, mode=0o700)
    started = time.perf_counter()
    try:
        _json_new(directory / "request.json", document, disk=disk)
        if latent_path.stat().st_size != source_result["latent_file"]["bytes"]:
            raise IntegrityError("Latent input changed after VAE admission")
        raw = latent_path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != source_result["latent_file"]["sha256"]:
            raise IntegrityError("Latent input failed its hash at VAE execution")
        state = deserialize_tensors(raw, max_bytes=document["memory_budgets"]["host_bytes"])
        if set(state) != {"latents", "noise", "initial_rng_state", "denoising_rng_state", "rng_state", "prompt_embeds"}:
            raise IntegrityError("The saved from-scratch native request state is incomplete")
        latents = state["latents"]
        if (
            tuple(latents.shape) != (1, document["chunks"] * 3, 16, 60, 104)
            or latents.dtype != torch.bfloat16 or tensor_digest(latents) != source_result["latents_sha256"]
            or not bool(torch.isfinite(latents).all())
        ):
            raise IntegrityError("Saved latent geometry/dtype/hash differs from native CausVid480")
        try:
            clear()
            with torch.no_grad():
                decoded = epoch.vae.decode_to_pixel(latents.to(epoch.device))
                if tuple(decoded.shape) != (1, timeline["prefix_frame_counts"][-1], 3, 480, 832):
                    raise IntegrityError("Native VAE did not return the genuine 4F-3 CausVid480 prefix")
                if not bool(torch.isfinite(decoded).all()):
                    raise IntegrityError("Native VAE returned nonfinite pixels before quantization")
                video = (decoded.float() * 0.5 + 0.5).clamp(0, 1)
            pixels = np.ascontiguousarray(
                (video[0].permute(0, 2, 3, 1).cpu().numpy() * 255).astype("uint8"),
            )
            rgb_path = directory / "rgb.npy"
            disk.charge(pixels.nbytes + (64 << 10))
            with rgb_path.open("xb") as stream:
                np.save(stream, pixels, allow_pickle=False)
                stream.flush()
                os.fsync(stream.fileno())
        finally:
            clear()
        if revalidate is not None:
            revalidate()
        result = {
            "schema": "reelcache.causvid.rank_result.v1", "status": "complete", "operation": "decode",
            "request": document, "asset_id": epoch.assets.asset_id, "provenance": provenance,
            **rank_control_metadata(document),
            "latent_source": document["latent_source"], "latents_sha256": source_result["latents_sha256"],
            "source_operation": source_result["operation"],
            "native_execution": source_result.get("native_execution") is True,
            "fixture": source_result.get("native_execution") is not True,
            "rgb_file": _file_record(rgb_path, output_root), "rgb_sha256": hashlib.sha256(pixels.tobytes()).hexdigest(),
            "rgb": {**timeline, "decoded": True, "shape": list(pixels.shape), "resize_or_crop": False},
            "native_parity_claim": False, "metrics_loaded": False, "dit_loaded": False,
            "disk_admission": disk.plan, "vae_cache_reset_verified": True,
            "elapsed_seconds": time.perf_counter() - started,
        }
        _json_new(directory / "result.json", result, disk=disk)
        return result
    except BaseException as exc:
        epoch._poisoned = True
        _json_new(directory / "failure.json", {
            "schema": "reelcache.causvid.epoch_abort.v1", "status": "abort_epoch", "error": repr(exc),
            "request_id": document["request_id"],
            "elapsed_seconds": time.perf_counter() - started,
            "timing_scope": "decode_execution_through_abort_before_final_cache_reset",
        }, disk=disk, failure=True)
        raise


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--protocol", type=Path, required=True)
    cli.add_argument("--assets", type=Path, required=True, help="Verified staged ROOT/assets.json")
    cli.add_argument("--asset-map", required=True, help="Manifested relative CausVid asset-map JSON")
    cli.add_argument("--launch-receipt", type=Path, required=True)
    cli.add_argument("--launch-phase", default="native_preflight", help="Must match the approved receipt exactly")
    cli.add_argument("--requests", type=Path, required=True, help="Source-bound reelcache.rank_requests.v1 JSON batch")
    cli.add_argument(
        "--worker-budgets", type=Path,
        help="Explicit CausVidMemoryBudgets for controller batches; must fit every bound request grant",
    )
    cli.add_argument("--output-root", type=Path, required=True)
    cli.add_argument("--operation", choices=("generate", "decode", "native-reference"), default="generate")
    cli.add_argument(
        "--threads", type=int, choices=range(1, 9), default=1,
        help="Admitted single-process CPU threads; must match every bound detector recipe",
    )
    cli.add_argument("--allow-cuda", action="store_true", help="Acknowledge the coordinator's separate GPU grant")
    return cli


def main(argv=None):
    args = parser().parse_args(argv)
    if not args.allow_cuda:
        raise ConfigurationError("Worker launch requires --allow-cuda and a separate coordinator resource grant")
    protocol, receipt, assets, requests = verify_worker_inputs(
        protocol_path=args.protocol, assets_path=args.assets, asset_map=args.asset_map,
        launch_receipt_path=args.launch_receipt, requests_path=args.requests, phase=args.launch_phase,
        worker_budgets=args.worker_budgets,
    )
    output_root = args.output_root.resolve()
    for request in requests:
        destination = _inside(output_root, request.document["output_namespace"])
        if destination.is_relative_to(assets.root) or assets.root.is_relative_to(destination):
            raise ConfigurationError("Worker output namespaces must not modify the verified asset tree")
        if destination.exists():
            raise StateError("A worker output namespace already exists; do not overwrite prior evidence")
    sources = (
        [verify_latent_source(request, output_root, assets) for request in requests]
        if args.operation == "decode" else []
    )
    if args.operation == "native-reference":
        from examples.causvid_native_reference import verify_reference_source

        for request in requests:
            verify_reference_source(
                request, output_root, asset_id=assets.asset_id, model=model_identity(assets.asset_id),
                provenance={
                    "protocol_id": protocol["protocol_id"],
                    "launch_closure_id": receipt["source_closure"]["closure_id"],
                }, native_execution=True,
            )
    if args.operation == "generate" and any(
        "latent_source" in request.document or "reference_source" in request.document for request in requests
    ):
        raise ConfigurationError("Generation requests must start from their own prompt and seed, not saved state")
    _validate_thread_admission(requests, args.threads)
    if args.operation == "generate":
        for request in requests:
            _RequestOutputBudget(request.document, output_root, "generate")
    _check_native_components(assets, args.operation)
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = str(args.threads)
    for name in (
        "TMPDIR", "TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR", "TORCH_EXTENSIONS_DIR",
        "CUDA_CACHE_PATH", "XDG_CACHE_HOME", "HF_HOME",
    ):
        path = output_root / "build_cache" / name.lower()
        path.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(path)
    os.environ.update(
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1",
        PYTHONDONTWRITEBYTECODE="1", TOKENIZERS_PARALLELISM="false",
    )
    sys.dont_write_bytecode = True
    for request in requests:
        document = request.document
        if document["rank_policy"] is None:
            specs = codec_specs(document["spec_config"], 30, ffmpeg=assets.paths.get("ffmpeg"))
            uses_hevc = any(spec.kind == "svd_hevc" for spec in specs.values())
        else:
            from reelcache.adapters.causvid import CausVidConfig

            policy = build_rank_policy(document["rank_policy"], model_identity(assets.asset_id), CausVidConfig())
            _admit_policy(policy, document, assets.paths.get("ffmpeg"))
            uses_hevc = policy.codec.kind == "svd_hevc"
            if uses_hevc and policy.codec.ffmpeg_path not in (None, str(assets.paths.get("ffmpeg"))):
                raise ConfigurationError("Adaptive policy refers to an unverified FFmpeg path")
        if uses_hevc and (
            "ffmpeg" not in assets.paths or not os.access(assets.paths["ffmpeg"], os.X_OK)
        ):
            raise ConfigurationError("HEVC needs an executable, explicitly verified local FFmpeg asset")
    provenance = {
        "protocol_id": protocol["protocol_id"], "receipt_id": receipt["receipt_id"],
        "source_closure_id": receipt["source_closure"]["closure_id"], "asset_id": assets.asset_id,
        "launch_closure_id": receipt["source_closure"]["closure_id"],
    }
    epoch = worker = None
    try:
        epoch = NativeCausVidEpoch(assets, requests, operation=args.operation, threads=args.threads)
        provenance = {
            **provenance, "epoch_model_load_seconds": epoch.load_seconds,
            "resident_model_parameter_and_buffer_bytes": epoch.model_bytes,
        }
        if args.operation in ("generate", "native-reference"):
            from reelcache.adapters.causvid import CausVidConfig

            worker_type = CausVidRankWorker
            if args.operation == "native-reference":
                from examples.causvid_native_reference import CausVidNativeReferenceWorker

                worker_type = CausVidNativeReferenceWorker
            worker = worker_type(
                epoch.pipeline, epoch.model, config=CausVidConfig(), output_root=output_root,
                asset_id=assets.asset_id, fps=assets.fps, protocol=protocol, provenance=provenance,
                attention_scope=epoch.attention_scope, ffmpeg=assets.paths.get("ffmpeg"),
                rank_policy_factory=build_rank_policy,
                observation_evidence_kind="native",
                revalidate=lambda: verify_launch_receipt(receipt, protocol, phase=args.launch_phase),
            )
        for index, request in enumerate(requests):
            if worker is not None:
                result = worker.run(request)
            else:
                verify_launch_receipt(receipt, protocol, phase=args.launch_phase)
                result = decode_request(
                    epoch, request, output_root, sources[index], provenance=provenance,
                    revalidate=lambda: verify_launch_receipt(receipt, protocol, phase=args.launch_phase),
                )
            print(canonical_json({
                "request_id": request.document["request_id"], "status": result["status"],
                "result": f"{request.document['output_namespace']}/result.json",
            }).decode(), flush=True)
    finally:
        try:
            if worker is not None:
                worker.close()
                worker = None
        finally:
            if epoch is not None:
                epoch.close()


if __name__ == "__main__":
    main()
