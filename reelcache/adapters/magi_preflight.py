"""Model-free input bundles and read-only attention records for MAGI preflight."""

from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path, PurePosixPath
import sys
import time

from ..errors import CapacityError, ConfigurationError, IntegrityError
from ..study import canonical_json, file_sha256, identity, read_json, write_json_new


INPUT_SCHEMA = "reelcache.magi_preflight_inputs.v1"
DIGEST_SCHEMA = "reelcache.magi_attention_digest.v1"
CONDITIONING_NAMES = {"y", "masks", "null_caption", "null_mask"}
NOISE_NAMES = {"noise", "postdraw_generator_state"}
FINAL_NOISE_RNG_MAX_BYTES = 65536


def final_noise_rng_record(generator, *, rank, binding_id):
    import torch

    if (type(rank) is not int or rank < 0 or not isinstance(binding_id, str)
            or len(binding_id) != 64 or any(value not in "0123456789abcdef" for value in binding_id)):
        raise IntegrityError("Final private RNG needs its rank and verified input binding")
    state = generator.get_state()
    if (not isinstance(state, torch.Tensor) or state.dtype != torch.uint8
            or state.device.type != "cpu" or state.ndim != 1):
        raise IntegrityError("Final private RNG state must be a one-dimensional CPU uint8 tensor")
    if not 0 < state.numel() <= FINAL_NOISE_RNG_MAX_BYTES:
        raise CapacityError("Final private RNG exceeds the admitted state-byte bound")
    raw = state.contiguous().numpy().tobytes()
    return {
        "rank": rank, "dtype": "uint8", "shape": [len(raw)], "state_hex": raw.hex(),
        "sha256": hashlib.sha256(raw).hexdigest(), "binding_id": binding_id,
    }


def input_binding(request, *, protocol_id, asset_id, config_id, launch_closure_id):
    from .magi_rank_worker import serial_sampler_contract

    return {
        "protocol_id": protocol_id, "case_id": request.case_id,
        "cell_id": request.cell_id, "group_id": request.group_id,
        "prompt_id": request.prompt_id, "prompt": request.prompt, "seed": request.seed,
        "chunks": request.chunks, "latent_shape": list(request.latent_shape),
        "source_fps": request.source_fps, "asset_id": asset_id,
        "config_id": config_id, "launch_closure_id": launch_closure_id,
        "sampler_contract": serial_sampler_contract(),
    }


def preflight_allowances(request, cell):
    latent = math.prod(request.latent_shape) * 4
    conditioning = 3 * 800 * 4096 * 4 + 3 * 800 * 8
    calls = cell.layers * request.chunks * 33
    kv = (request.chunks + 1) * cell.tokens * cell.channels // cell.world_size * 2
    rng_rows = cell.world_size * (2 * FINAL_NOISE_RNG_MAX_BYTES + 4096)
    return {
        "host_bytes": cell.world_size * (4 * conditioning + 4 * latent + 4 * kv) + 8 * rng_rows,
        "device_bytes_per_rank": 2 * latent,
        "input_bundle_bytes": conditioning + (64 << 10) + request.chunks * (latent + (64 << 10)),
        "attention_digest_bytes": cell.world_size * calls * (16 << 10),
        "max_attention_records_per_rank": calls,
        "final_noise_rng_metadata_bytes": 2 * rng_rows,  # Request result and complete batch manifest.
    }


def _artifact(path):
    path = Path(path)
    return {"path": str(path), "sha256": file_sha256(path), "bytes": path.stat().st_size}


def _record(tensors):
    from .magi import _tensor_hash

    return {
        name: {"shape": list(value.shape), "dtype": str(value.dtype).removeprefix("torch."),
               "sha256": _tensor_hash(value)}
        for name, value in tensors.items()
    }


def _bundle_path(root, name):
    if not isinstance(name, str):
        raise IntegrityError("An input-bundle tensor file needs an explicit relative path")
    relative = PurePosixPath(name)
    if relative.is_absolute() or len(relative.parts) != 1 or relative.name in ("", ".", ".."):
        raise IntegrityError("An input bundle cannot escape its admitted directory")
    path = Path(root) / name
    if path.is_symlink() or not path.is_file():
        raise IntegrityError("An input-bundle tensor file is missing or is a symlink")
    return path


def _verify_tensor_file(root, record, names, max_bytes=None):
    if not isinstance(record, dict) or set(record.get("tensors", {})) != names:
        raise IntegrityError("An input bundle contains missing or unexpected tensor names")
    if type(record.get("bytes")) is not int or record["bytes"] < 1 or (
        max_bytes is not None and record["bytes"] > max_bytes
    ):
        raise CapacityError("An input-bundle tensor exceeds its admitted byte limit")
    path = _bundle_path(root, record.get("path"))
    if path.stat().st_size != record.get("bytes") or file_sha256(path) != record.get("sha256"):
        raise IntegrityError("An input-bundle tensor file changed")
    return path


def verify_input_bundle(path, digest, binding, max_bytes=None):
    path = Path(path).resolve(strict=True)
    if path.stat().st_size > (128 << 10) + 2 * len(canonical_json(binding)):
        raise CapacityError("Input-bundle manifest exceeds its bounded metadata allowance")
    if file_sha256(path) != digest:
        raise IntegrityError("The admitted input-bundle manifest changed")
    document = read_json(path)
    if (not isinstance(document, dict) or document.get("schema") != INPUT_SCHEMA
            or document.get("binding") != binding
            or document.get("bundle_id") != identity({k: v for k, v in document.items() if k != "bundle_id"})):
        raise IntegrityError("Input bundle prompt/seed/cell/assets/config/source/horizon binding differs")
    _verify_tensor_file(path.parent, document.get("conditioning"), CONDITIONING_NAMES, max_bytes)
    chunks = document.get("chunks")
    if not isinstance(chunks, list) or len(chunks) != binding["chunks"]:
        raise IntegrityError("An input bundle must contain every chunk, without prefix or noise skipping")
    for index, record in enumerate(chunks):
        if type(record.get("chunk_index")) is not int or record["chunk_index"] != index:
            raise IntegrityError("Input-bundle chunk order changed")
        _verify_tensor_file(path.parent, record, NOISE_NAMES, max_bytes)
        noise = record["tensors"]["noise"]
        rng = record["tensors"]["postdraw_generator_state"]
        if noise.get("shape") != binding["latent_shape"] or noise.get("dtype") != "float32":
            raise IntegrityError("Input-bundle noise must be the complete FP32 latent input")
        if (rng.get("dtype") != "uint8" or not isinstance(rng.get("shape"), list) or len(rng["shape"]) != 1
                or type(rng["shape"][0]) is not int or not 0 < rng["shape"][0] <= 65536):
            raise IntegrityError("Each noise needs its actual postdraw private generator state")
    total = path.stat().st_size + document["conditioning"]["bytes"] + sum(row["bytes"] for row in chunks)
    if max_bytes is not None and total > max_bytes:
        raise CapacityError("The complete input bundle exceeds its admitted bytes")
    return document


class MagiInputBundle:
    def __init__(self, *, mode, binding, directory=None, path=None, sha256=None, max_bytes=None):
        self.mode, self.binding = mode, binding
        self.max_bytes, self.written_bytes = max_bytes, 0
        self.next_chunk = 0
        self.conditioning_checked = False
        self.original_path = Path(path).resolve(strict=True) if path is not None else None
        self.original_sha256 = sha256
        if mode == "capture":
            self.root = Path(directory)
            self.root.mkdir(parents=True, exist_ok=False)
            self.document = {"schema": INPUT_SCHEMA, "binding": binding, "conditioning": None, "chunks": []}
        elif mode == "replay":
            self.document = verify_input_bundle(self.original_path, sha256, binding, max_bytes)
            self.root = self.original_path.parent
        else:
            raise ConfigurationError("Input-bundle mode must be capture or replay")

    def _save(self, label, tensors):
        from ..serialization import serialize_tensors

        raw = serialize_tensors(tensors)
        if self.max_bytes is not None and self.written_bytes + len(raw) > self.max_bytes:
            raise CapacityError("Input capture exceeds its admitted output bytes")
        path = self.root / (label + ".reelcache")
        with path.open("xb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        self.written_bytes += len(raw)
        return {"path": path.name, "sha256": hashlib.sha256(raw).hexdigest(),
                "bytes": len(raw), "tensors": _record(tensors)}

    def _load_checked(self, record, tensors, names):
        from ..serialization import deserialize_tensors

        path = _verify_tensor_file(self.root, record, names, self.max_bytes)
        with path.open("rb") as stream:
            raw = stream.read(record["bytes"] + 1)
        if len(raw) != record["bytes"] or hashlib.sha256(raw).hexdigest() != record["sha256"]:
            raise IntegrityError("An input-bundle tensor changed during its bounded read")
        restored = deserialize_tensors(raw)
        if _record(restored) != record["tensors"] or _record(tensors) != record["tensors"]:
            raise IntegrityError("Re-encoded conditioning or per-chunk noise/postdraw generator state differs")
        return restored

    def conditioning(self, tensors):
        if self.conditioning_checked or set(tensors) != CONDITIONING_NAMES:
            raise IntegrityError("Every trial must check its newly encoded conditioning exactly once")
        if self.mode == "capture":
            self.document["conditioning"] = self._save("conditioning", tensors)
        else:
            self._load_checked(self.document["conditioning"], tensors, CONDITIONING_NAMES)
        self.conditioning_checked = True
        return self.document["conditioning"]

    def noise(self, index, noise, postdraw_generator_state):
        if not self.conditioning_checked or index != self.next_chunk or index >= self.binding["chunks"]:
            raise IntegrityError("Input replay must follow the actual draw for each contiguous chunk")
        tensors = {"noise": noise.detach().cpu(), "postdraw_generator_state": postdraw_generator_state.cpu()}
        if list(noise.shape) != self.binding["latent_shape"] or str(noise.dtype) != "torch.float32":
            raise IntegrityError("The actual drawn noise differs from the bound complete latent shape")
        if self.mode == "capture":
            record = {"chunk_index": index, **self._save(f"chunk{index}", tensors)}
            self.document["chunks"].append(record)
            explicit = tensors["noise"]
        else:
            record = self.document["chunks"][index]
            explicit = self._load_checked(record, tensors, NOISE_NAMES)["noise"]
        self.next_chunk += 1
        return record, explicit

    def finish(self):
        if not self.conditioning_checked or self.next_chunk != self.binding["chunks"]:
            raise IntegrityError("An incomplete input bundle cannot be published")
        if self.mode == "capture":
            self.document["bundle_id"] = identity(self.document)
            path = self.root / "bundle.json"
            if self.max_bytes is not None and self.written_bytes + len(canonical_json(self.document)) + 1 > self.max_bytes:
                raise CapacityError("The complete input manifest exceeds its admitted bytes")
            write_json_new(path, self.document)
        else:
            path = self.original_path
            verify_input_bundle(path, self.original_sha256, self.binding, self.max_bytes)
        return {"schema": INPUT_SCHEMA, "mode": self.mode, "binding_id": identity(self.binding),
                "bundle_id": self.document["bundle_id"], "artifact": _artifact(path)}


class AttentionDigestRecorder:
    def __init__(self, path, *, geometry, rank, binding_id, max_chunks, max_records, max_bytes):
        self.path, self.geometry, self.rank = Path(path), geometry, rank
        self.binding_id, self.max_chunks = binding_id, max_chunks
        self.max_records, self.max_bytes = max_records, max_bytes
        self.count = self.bytes = self.host_tensor_bytes = 0
        self.seconds = 0.0
        self.calls = {}
        self.stream = self.path.open("xb")

    def record(self, point, layer, key, value, *, query_ranges, key_ranges, near_clean):
        from .magi import _tensor_hash, NATIVE_TENSOR_HASH_FORMAT

        started = time.perf_counter()
        if set(point) != {"chunk_index", "phase", "step_index", "input_noise_sha256"}:
            raise IntegrityError("Attention digest point must use the declared scalar input/step schema")
        digest = point["input_noise_sha256"]
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise IntegrityError("Attention digest point lacks the actual input-noise hash")
        chunk, step, phase = point["chunk_index"], point["step_index"], point["phase"]
        if (type(chunk) is not int or not 0 <= chunk < self.max_chunks
                or type(layer) is not int or not 0 <= layer < self.geometry.layers
                or phase not in ("denoise", "clean")
                or (phase == "denoise" and (type(step) is not int or not 0 <= step < 16))
                or (phase == "clean" and step is not None)):
            raise IntegrityError("Attention digest point is outside the bound native schedule")
        expected_near = phase == "denoise" and step >= 12
        rows = (chunk + 1 + int(expected_near)) * self.geometry.tokens
        shape = (rows, self.geometry.layout.local_heads, self.geometry.head_dim)
        if (type(near_clean) is not bool or near_clean != expected_near
                or tuple(key.shape) != shape or tuple(value.shape) != shape
                or str(key.dtype) != "torch.bfloat16" or str(value.dtype) != "torch.bfloat16"):
            raise IntegrityError("Digest source must be actual full-token post-RoPE BF16 K/direct V")
        query_rows = self.geometry.tokens * (1 + int(expected_near))
        for ranges, extent in ((query_ranges, query_rows), (key_ranges, rows)):
            if (not isinstance(ranges, list) or len(ranges) != 1 + int(expected_near)
                    or any(not isinstance(pair, list) or len(pair) != 2
                           or any(type(number) is not int for number in pair)
                           or not 0 <= pair[0] < pair[1] <= extent for pair in ranges)):
                raise IntegrityError("Native attention visibility ranges are not complete bounded token ranges")
        slot = (chunk, phase, step, layer)
        call = self.calls.get(slot, 0)
        if call >= (2 if phase == "denoise" else 1) or self.count >= self.max_records:
            raise CapacityError("Attention instrumentation exceeded its admitted invocation count")
        archived = chunk * self.geometry.tokens
        record = {
            "schema": DIGEST_SCHEMA, "binding_id": self.binding_id, "rank": self.rank,
            **point, "layer": layer, "attention_call": call,
            "near_clean": near_clean, "query_ranges": query_ranges, "key_ranges": key_ranges,
            "shape": list(shape), "archived_rows": archived, "dtype": "bfloat16",
            "key_space": "post_rope", "value_space": "none",
            "hash_format": NATIVE_TENSOR_HASH_FORMAT, "hash_byte_order": sys.byteorder,
            "key_sha256": _tensor_hash(key), "value_sha256": _tensor_hash(value),
            "archived_key_sha256": _tensor_hash(key[:archived]),
            "archived_value_sha256": _tensor_hash(value[:archived]),
        }
        raw = canonical_json(record) + b"\n"
        if self.bytes + len(raw) > self.max_bytes:
            raise CapacityError("Attention instrumentation exceeded its admitted record bytes")
        self.stream.write(raw)
        self.calls[slot] = call + 1
        self.count += 1
        self.bytes += len(raw)
        self.host_tensor_bytes += 2 * (rows + archived) * self.geometry.layout.local_channels * 2
        self.seconds += time.perf_counter() - started
        return record

    def finish(self):
        expected = {
            (chunk, phase, step, layer)
            for chunk in range(self.max_chunks) for layer in range(self.geometry.layers)
            for phase, step in [*(("denoise", index) for index in range(16)), ("clean", None)]
        }
        if set(self.calls) != expected:
            raise IntegrityError("Attention digest coverage omitted native layers, timesteps or clean recaptures")
        self.stream.flush()
        os.fsync(self.stream.fileno())
        self.close()
        return {"schema": "reelcache.magi_attention_digest_receipt.v1", "artifact": _artifact(self.path),
                "records": self.count, "local_record_seconds": self.seconds,
                "host_tensor_bytes": self.host_tensor_bytes, "offline_instrumentation": True}

    def close(self):
        if not self.stream.closed:
            self.stream.close()
