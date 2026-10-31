#!/usr/bin/env python3
"""Bounded CPU comparison of existing, hash-bound native-preflight artifacts.

Run ``python scripts/rank_preflight_compare.py --example`` for a request template,
then supply --request, --request-sha256 and a NEW --output receipt path.
Library API: compare_preflight(path, request_sha256=..., output_path=None).
No model, native inference, GPU, staging or gate-advancement API is called.

MAGI kinds are raw_native_noop (roles raw/native_noop), hot_reload (hot/reload),
and request_isolation (a_before/b/a_after/fresh_a/fresh_b). Isolation requires
one ordered A/B/A epoch and two separate single-request fresh epochs. Every
trial supplies generation, result, decode and launch_receipt file references.
The approved cfg_number=1/overlap_degree=1 path emits ONE digest per denoise
step and one per clean forward: 17 records per layer/chunk/rank. Near-clean
has two visibility rows inside that same recorded attention invocation.
CausVid raw_native_reference delegates its four generation/decode references to
examples.causvid_native_reference.compare_reference_outputs.

All references contain path, sha256 and bytes and stay below artifact_root.
Native geometry is fixed by the study cell. Tiny fixtures require the explicit
synthetic_fixture evidence kind and synthetic_geometry; they cannot qualify a
native cell. Byte allowances cover conservative owned comparison allocations,
not measured process RSS or model admission. Native MAGI requires actual final
private generator-state records from both ranks; these are never inferred from
the last postdraw input-bundle state. Synthetic fixtures may explicitly omit
them unless require_final_rng is true, with that missing evidence reported.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import struct
import sys
import time
from types import SimpleNamespace
import zipfile
import zlib


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reelcache.adapters.magi_preflight import (  # noqa: E402
    CONDITIONING_NAMES, DIGEST_SCHEMA, FINAL_NOISE_RNG_MAX_BYTES, INPUT_SCHEMA, NOISE_NAMES,
    input_binding, verify_input_bundle,
)
from reelcache.adapters.magi import NATIVE_TENSOR_HASH_FORMAT  # noqa: E402
from reelcache.errors import ReelCacheError  # noqa: E402
from reelcache.study import CELLS, LAUNCH_SCHEMA, canonical_json, identity  # noqa: E402


REQUEST_SCHEMA = "reelcache.rank_preflight_comparison_request.v1"
RECEIPT_SCHEMA = "reelcache.rank_preflight_comparison.v1"
FIXTURE_RECEIPT_SCHEMA = "reelcache.rank_preflight_fixture_receipt.v1"
HASH_FORMAT = NATIVE_TENSOR_HASH_FORMAT
GENERATION_SCHEMA = "reelcache.magi_generation_batch.v1"
DECODE_SCHEMA = "reelcache.magi_decode_batch.v1"
TRACE_RECEIPT_SCHEMA = "reelcache.magi_attention_digest_receipt.v1"
DENOISE_ATTENTION_CALLS = 1
ROLES = {
    "raw_native_noop": ("raw", "native_noop"),
    "hot_reload": ("hot", "reload"),
    "request_isolation": ("a_before", "b", "a_after", "fresh_a", "fresh_b"),
}
CAUSVID_ROLES = ("raw_generation", "native_reference", "raw_decode", "native_decode")
IO_BYTES = 1 << 20
METADATA_MEMORY_FACTOR = 16
TRACE_LINE_BYTES = 16 << 10
EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
ZIP_END = struct.Struct("<4s4H2LH")
ZIP64_LOCATOR = struct.Struct("<4sLQL")
ZIP64_END = struct.Struct("<4sQ2H2L4Q")
ZIP_CENTRAL = struct.Struct("<4s6H3L5H2L")


class ComparisonError(ValueError):
    """Invalid, changed, incomplete or unadmitted comparison evidence."""


def _require(condition, message):
    if not condition:
        raise ComparisonError(message)


def _integer(value, name, minimum=0):
    _require(type(value) is int and value >= minimum, f"{name} must be an integer >= {minimum}")
    return value


def _text(value, name):
    _require(isinstance(value, str) and bool(value), f"{name} must be nonempty text")
    return value


def _start_ticks(value):
    _require(isinstance(value, str) and 0 < len(value) <= 20 and value[0] in "123456789"
             and all("0" <= character <= "9" for character in value),
             "worker start ticks must be a positive canonical ASCII decimal string")
    result = int(value)
    _require(result <= (1 << 64) - 1, "worker start ticks exceed the proc unsigned 64-bit field")
    return result


def _shape(value, length):
    _require(isinstance(value, list) and len(value) == length
             and all(type(size) is int and size > 0 for size in value),
             "shape must contain exactly the declared positive integer dimensions")
    return value


def _sha(value):
    _require(isinstance(value, str) and len(value) == 64
             and all(c in "0123456789abcdef" for c in value), "SHA256 must be lowercase hexadecimal")
    return value


def _object_pairs(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, "duplicate JSON object key")
        result[key] = value
    return result


def _number(value):
    number = float(value)
    _require(math.isfinite(number), "nonfinite JSON number")
    return number


def _json(raw):
    try:
        return json.loads(raw, object_pairs_hook=_object_pairs, parse_float=_number, parse_constant=_number)
    except (UnicodeError, json.JSONDecodeError, OverflowError) as error:
        raise ComparisonError(f"invalid JSON: {error}") from error


def _file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(IO_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


class _Budget:
    def __init__(self, root, limits):
        _require(isinstance(limits, dict) and set(limits) == {"host_bytes", "metadata_bytes"},
                 "explicit host_bytes and metadata_bytes limits are required")
        self.root = Path(root).resolve(strict=True)
        _require(self.root.is_dir(), "artifact_root must be an existing directory")
        self.host = _integer(limits["host_bytes"], "host_bytes", 1)
        self.metadata_limit = _integer(limits["metadata_bytes"], "metadata_bytes", 1)
        self.metadata_bytes = self.peak_admission = 0
        self.tensor_bytes = self.rgb_bytes = self.trace_records = self.final_rng_bytes = 0
        self.source_evidence_validated = False
        self.references, self.documents, self.metadata_paths = {}, {}, set()
        self.admit(0)

    def admit(self, transient):
        needed = IO_BYTES + self.metadata_bytes * METADATA_MEMORY_FACTOR + transient
        _require(needed <= self.host, "comparison host byte budget exceeded before allocation")
        self.peak_admission = max(self.peak_admission, needed)

    def metadata(self, path, size):
        if path not in self.metadata_paths:
            _require(self.metadata_bytes + size <= self.metadata_limit, "comparison metadata byte budget exceeded")
            self.metadata_bytes += size
            self.metadata_paths.add(path)
            self.admit(0)

    def artifact(self, record):
        _require(isinstance(record, dict) and set(record) == {"path", "sha256", "bytes"},
                 "artifact references need exactly path, sha256 and bytes")
        name = _text(record["path"], "artifact path")
        digest = _sha(record["sha256"])
        size = _integer(record["bytes"], "artifact bytes", 1)
        original = Path(name) if Path(name).is_absolute() else self.root / name
        path = original.resolve(strict=True)
        _require(path.is_relative_to(self.root) and path != self.root,
                 "artifact escaped its declared root")
        cursor = original
        while cursor != self.root and cursor != cursor.parent:
            _require(not cursor.is_symlink(), "artifact paths must not contain symlinks")
            cursor = cursor.parent
        _require(path.is_file() and path.stat().st_size == size, "artifact size or file type changed")
        _require(_file_hash(path) == digest, "artifact file SHA256 changed")
        canonical = {"path": str(path), "sha256": digest, "bytes": size}
        previous = self.references.setdefault(str(path), canonical)
        _require(previous == canonical, "one artifact path has conflicting references")
        return path

    def document(self, record):
        path = self.artifact(record)
        self.metadata(path, record["bytes"])
        if path not in self.documents:
            with path.open("rb") as stream:
                raw = stream.read(record["bytes"] + 1)
            _require(len(raw) == record["bytes"] and hashlib.sha256(raw).hexdigest() == record["sha256"],
                     "metadata artifact changed during read")
            self.documents[path] = _json(raw)
        return self.documents[path]

    def measurements(self):
        return {
            "metadata_input_bytes": self.metadata_bytes,
            "maximum_owned_host_admission_bytes": self.peak_admission,
            "host_admission_is_measured_peak": False,
            "metadata_memory_factor": METADATA_MEMORY_FACTOR,
            "input_tensor_bytes_verified": self.tensor_bytes,
            "rgb_bytes_verified": self.rgb_bytes,
            "attention_records_verified": self.trace_records,
            "final_noise_rng_bytes_verified": self.final_rng_bytes,
            "unique_artifact_bytes_verified": sum(row["bytes"] for row in self.references.values()),
        }


def _geometry(request):
    cell_id, kind = request["expected"]["cell_id"], request["evidence_kind"]
    if kind == "native":
        _require("synthetic_geometry" not in request, "native evidence cannot use synthetic geometry")
        _require(cell_id in CELLS and CELLS[cell_id].model == "magi_distill",
                 "native MAGI comparisons require an approved MAGI cell")
        cell = CELLS[cell_id]
        return {
            "layers": cell.layers, "world_size": cell.world_size, "tokens": cell.tokens,
            "local_heads": 4, "head_dim": 128,
            "latent_shape": [1, 16, cell.frames, cell.token_height * 2, cell.token_width * 2],
            "rgb_shape": [24, cell.height, cell.width, 3],
            "crop_rows": [cell.crop_top, cell.crop_top + cell.height], "fps": 24,
            "source_geometry": cell.to_dict(),
        }
    _require(kind == "synthetic_fixture" and cell_id not in CELLS,
             "tiny comparison fixtures must use an explicitly synthetic cell")
    value = request.get("synthetic_geometry")
    names = {"layers", "world_size", "tokens", "local_heads", "head_dim",
             "latent_shape", "rgb_shape", "crop_rows", "fps"}
    _require(isinstance(value, dict) and set(value) == names, "complete synthetic geometry is required")
    for name in ("layers", "world_size", "tokens", "local_heads", "head_dim", "fps"):
        _integer(value[name], name, 1)
    _require(value["world_size"] == 2 and value["fps"] == 24, "fixtures preserve SP2 and true 24 FPS")
    for name, length in (("latent_shape", 5), ("rgb_shape", 4)):
        _shape(value[name], length)
    _require(value["latent_shape"][0] == 1 and value["rgb_shape"][-1] == 3,
             "fixture shapes must be batch-one latents and THWC RGB")
    crop = value["crop_rows"]
    _require(isinstance(crop, list) and len(crop) == 2, "crop_rows needs two bounds")
    _integer(crop[0], "crop start")
    _integer(crop[1], "crop end", 1)
    _require(crop[1] - crop[0] == value["rgb_shape"][1], "crop differs from visible RGB height")
    return {**value, "source_geometry": value}


def _flags(value, kind):
    _require(value.get("fixture") is (kind == "synthetic_fixture")
             and value.get("native_execution") is (kind == "native"),
             "source fixture/native flags do not match the declared evidence kind")


def _final_noise_rng(value, *, binding_id, world_size, required, budget):
    if value is None:
        _require(not required, "comparison requires actual final_noise_rng after generation/clean recapture")
        return None
    _require(isinstance(value, list) and len(value) == world_size,
             "final_noise_rng must cover every rank exactly once in rank order")
    states = []
    for rank, row in enumerate(value):
        _require(isinstance(row, dict)
                 and set(row) == {"rank", "dtype", "shape", "state_hex", "sha256", "binding_id"},
                 "final_noise_rng record fields differ from the producer contract")
        _require(type(row["rank"]) is int and row["rank"] == rank and row["dtype"] == "uint8"
                 and row["binding_id"] == binding_id, "final_noise_rng rank/dtype/input binding differs")
        shape = _shape(row["shape"], 1)
        _require(shape[0] <= FINAL_NOISE_RNG_MAX_BYTES, "final_noise_rng exceeds the fixed state-byte bound")
        encoded = row["state_hex"]
        _require(isinstance(encoded, str) and len(encoded) == 2 * shape[0]
                 and all(character in "0123456789abcdef" for character in encoded),
                 "final_noise_rng needs exact-length lowercase raw-byte hex")
        budget.admit(sum(len(state) for state in states) + shape[0])
        raw = bytes.fromhex(encoded)
        _require(hashlib.sha256(raw).hexdigest() == _sha(row["sha256"]),
                 "final_noise_rng raw-byte hash differs")
        states.append(raw)
        budget.final_rng_bytes += len(raw)
    return tuple(states)


def _receipt(record, expected, kind, budget):
    receipt = budget.document(record)
    schema = LAUNCH_SCHEMA if kind == "native" else FIXTURE_RECEIPT_SCHEMA
    _require(isinstance(receipt, dict) and receipt.get("schema") == schema,
             "comparison needs its predeclared native or explicitly synthetic receipt")
    if kind == "synthetic_fixture":
        _flags(receipt, kind)
    _require(receipt.get("protocol_id") == expected["protocol_id"]
             and receipt.get("phase") == "native_preflight"
             and receipt.get("source_closure", {}).get("closure_id") == expected["launch_closure_id"],
             "launch receipt protocol/closure/phase binding differs")
    _require(receipt.get("receipt_id") == identity({k: v for k, v in receipt.items() if k != "receipt_id"}),
             "launch receipt content identity differs")


def _tensor_hash(tensor):
    import torch

    _require(tensor.device.type == "cpu" and tensor.layout == torch.strided and tensor.is_contiguous(),
             "only actual contiguous CPU tensors may be compared")
    if tensor.is_floating_point() or tensor.is_complex():
        _require(bool(torch.isfinite(tensor).all()), "nonfinite tensor input")
    return hashlib.sha256(memoryview(tensor.detach().reshape(-1).view(torch.uint8).numpy())).hexdigest()


def _same_tensor(left, right):
    import torch

    return (left.shape == right.shape and left.dtype == right.dtype
            and torch.equal(left.reshape(-1).view(torch.uint8), right.reshape(-1).view(torch.uint8)))


def _bundle_tensor(root, record, names, budget, *, live_bytes=0):
    from reelcache.serialization import deserialize_tensors

    _require(isinstance(record, dict) and set(record.get("tensors", {})) == names,
             "bundle tensor names are incomplete")
    reference = {name: record[name] for name in ("path", "sha256", "bytes")}
    reference["path"] = str(root / reference["path"])
    path = budget.artifact(reference)
    budget.admit(live_bytes + 8 * record["bytes"])
    with path.open("rb") as stream:
        raw = stream.read(record["bytes"] + 1)
    _require(len(raw) == record["bytes"] and hashlib.sha256(raw).hexdigest() == record["sha256"],
             "bundle tensor changed during read")
    tensors = deserialize_tensors(raw, device="cpu", max_bytes=record["bytes"])
    _require(set(tensors) == names, "actual bundle tensors differ from declared names")
    actual = {
        name: {"shape": list(value.shape), "dtype": str(value.dtype).removeprefix("torch."),
               "sha256": _tensor_hash(value)}
        for name, value in tensors.items()
    }
    _require(actual == record["tensors"], "actual bundle tensor metadata/hash differs")
    budget.tensor_bytes += sum(value.numel() * value.element_size() for value in tensors.values())
    return tensors


def _workers(generation, budget, world_size):
    records = generation.get("worker_execution_records")
    _require(isinstance(records, list) and len(records) == world_size, "missing worker execution records")
    processes = []
    for rank, reference in enumerate(records):
        item = budget.document(reference)
        _require(isinstance(item, dict), "worker execution metadata must be an object")
        _require(item.get("plan_identity") == generation["batch_id"], "worker belongs to another generation epoch")
        _require(item.get("environment", {}).get("RANK") == str(rank)
                 and item["environment"].get("WORLD_SIZE") == str(world_size),
                 "worker rank/world execution identity differs")
        processes.append((_text(item.get("host"), "worker host"),
                          _integer(item.get("pid"), "worker PID", 1),
                          _start_ticks(item.get("process_start_ticks"))))
    _require(len(set(processes)) == world_size, "worker process identities must be distinct")
    return tuple(processes)


def _trace_index(receipts, binding_id, noises, geometry, calls, budget):
    _require(isinstance(receipts, list) and len(receipts) == geometry["world_size"],
             "attention receipts must cover every SP rank")
    chunks, layers, tokens = len(noises), geometry["layers"], geometry["tokens"]
    index, clean_hashes, archived_hashes = {}, {}, {}
    expected_count = chunks * layers * (16 * calls + 1)
    fields = {
        "schema", "binding_id", "rank", "chunk_index", "phase", "step_index", "input_noise_sha256",
        "layer", "attention_call", "near_clean", "query_ranges", "key_ranges", "shape", "archived_rows",
        "dtype", "key_space", "value_space", "hash_format", "hash_byte_order",
        "key_sha256", "value_sha256", "archived_key_sha256", "archived_value_sha256",
    }
    for rank, receipt in enumerate(receipts):
        _require(isinstance(receipt, dict), "attention receipt must be an object")
        _require(receipt.get("schema") == TRACE_RECEIPT_SCHEMA and receipt.get("offline_instrumentation") is True,
                 "attention receipt schema/scope differs")
        _require(_integer(receipt.get("records"), "attention record count", 1) == expected_count,
                 "attention call count omits or adds native events")
        path = budget.artifact(receipt["artifact"])
        budget.metadata(path, receipt["artifact"]["bytes"])
        count, digest = 0, hashlib.sha256()
        with path.open("rb") as stream:
            while True:
                raw = stream.readline(TRACE_LINE_BYTES + 1)
                if not raw:
                    break
                _require(len(raw) <= TRACE_LINE_BYTES and raw.endswith(b"\n"),
                         "attention record exceeds its bounded line or is incomplete")
                digest.update(raw)
                row = _json(raw)
                _require(isinstance(row, dict) and set(row) == fields, "unexpected attention record fields")
                _require(row["schema"] == DIGEST_SCHEMA and row["binding_id"] == binding_id
                         and type(row["rank"]) is int and row["rank"] == rank,
                         "attention source binding/rank differs")
                chunk = _integer(row["chunk_index"], "attention chunk")
                layer = _integer(row["layer"], "attention layer")
                call = _integer(row["attention_call"], "attention call")
                _require(chunk < chunks and layer < layers and row["phase"] in ("denoise", "clean"),
                         "attention event outside declared horizon/layers")
                step = row["step_index"]
                if row["phase"] == "denoise":
                    _require(type(step) is int and 0 <= step < 16 and call < calls,
                             "denoise attention step/call differs")
                else:
                    _require(step is None and call == 0, "clean recapture must occur exactly once")
                near = row["phase"] == "denoise" and step >= 12
                rows = (chunk + 1 + int(near)) * tokens
                _shape(row["shape"], 3)
                _require(row["near_clean"] is near
                         and row["shape"] == [rows, geometry["local_heads"], geometry["head_dim"]]
                         and type(row["archived_rows"]) is int and row["archived_rows"] == chunk * tokens,
                         "attention shape/archive/near-clean geometry differs")
                _require(row["dtype"] == "bfloat16" and row["key_space"] == "post_rope"
                         and row["value_space"] == "none" and row["hash_format"] == HASH_FORMAT
                         and row["hash_byte_order"] == "little" and sys.byteorder == "little",
                         "attention tensor dtype/domain/hash format differs")
                _require(row["input_noise_sha256"] == noises[chunk], "attention used a different input noise")
                queries = row["query_ranges"]
                _require(isinstance(queries, list)
                         and all(isinstance(pair, list) and len(pair) == 2
                                 and all(type(value) is int for value in pair) for pair in queries)
                         and queries == [[i * tokens, (i + 1) * tokens] for i in range(1 + near)],
                         "attention query visibility is not the complete current token view")
                ranges = row["key_ranges"]
                _require(isinstance(ranges, list) and len(ranges) == 1 + near,
                         "attention key visibility count differs")
                for pair in ranges:
                    _require(isinstance(pair, list) and len(pair) == 2
                             and all(type(x) is int for x in pair)
                             and 0 <= pair[0] < pair[1] <= rows
                             and pair[0] % tokens == pair[1] % tokens == 0,
                             "attention key visibility is outside complete token blocks")
                hashes = tuple(_sha(row[name]) for name in (
                    "key_sha256", "value_sha256", "archived_key_sha256", "archived_value_sha256",
                ))
                event = (rank, chunk, row["phase"], -1 if step is None else step, layer, call)
                _require(event not in index, "duplicate attention event")
                index[event] = identity(row)
                archive_key = (rank, chunk, layer)
                previous = archived_hashes.setdefault(archive_key, hashes[2:])
                _require(previous == hashes[2:], "archived prefix changed across attention steps/calls")
                if chunk == 0:
                    _require(hashes[2:] == (EMPTY_SHA256, EMPTY_SHA256), "empty archive digest differs")
                if row["phase"] == "clean":
                    clean_hashes[archive_key] = hashes[:2]
                count += 1
        _require(count == expected_count and digest.hexdigest() == receipt["artifact"]["sha256"],
                 "attention artifact count/hash changed during read")
        budget.trace_records += count
    expected = {
        (rank, chunk, phase, step, layer, call)
        for rank in range(geometry["world_size"]) for chunk in range(chunks) for layer in range(layers)
        for phase, step in [*(("denoise", i) for i in range(16)), ("clean", -1)]
        for call in range(calls if phase == "denoise" else 1)
    }
    _require(set(index) == expected, "attention coverage omits a rank/layer/step/clean event")
    for (rank, chunk, layer), value in archived_hashes.items():
        if chunk:
            _require(value == clean_hashes[(rank, chunk - 1, layer)],
                     "archived prefix differs from the preceding committed clean attention")
    return index


@dataclass
class _Trial:
    role: str
    source: dict
    generation: dict
    result: dict
    decode: dict
    sequence: int
    processes: tuple
    binding: dict
    bundle: dict
    bundle_root: Path
    trace: dict
    media: list
    pairing_sha256: str
    final_rng: tuple[bytes, ...] | None


def _load_trial(source, request, geometry, budget):
    _require(isinstance(source, dict)
             and set(source) == {"role", "generation", "result", "decode", "launch_receipt"},
             "each MAGI trial needs role and four complete artifact references")
    expected, kind = request["expected"], request["evidence_kind"]
    _receipt(source["launch_receipt"], expected, kind, budget)
    generation = budget.document(source["generation"])
    result = budget.document(source["result"])
    decode = budget.document(source["decode"])
    _require(all(isinstance(value, dict) for value in (generation, result, decode)),
             "generation, result and decode metadata must be objects")
    _require(generation.get("schema") == GENERATION_SCHEMA and decode.get("schema") == DECODE_SCHEMA,
             "generation/decode batch schema differs")
    _require(generation.get("protocol_id") == expected["protocol_id"]
             and generation.get("cell_id") == decode.get("cell_id") == expected["cell_id"]
             and decode.get("batch_id") == generation.get("batch_id"),
             "generation/decode protocol, cell or epoch differs")
    _sha(generation["batch_id"])
    _require(generation.get("models_unloaded") is True and generation.get("vae_loaded") is False
             and decode.get("models_unloaded_before_VAE_load") is True,
             "comparison needs separately decoded, completed generation epochs")
    manifest = budget.artifact(decode["generation_manifest"])
    _require(manifest == budget.artifact(source["generation"]), "decode is bound to another generation manifest")
    namespace = _text(result.get("namespace"), "result namespace")
    entries = generation.get("results")
    _require(isinstance(entries, list) and bool(entries), "generation has no completed requests")
    _require(all(isinstance(row, dict) and row.get("sequence") == index
                 and type(row.get("sequence")) is int for index, row in enumerate(entries)),
             "generation request ordering is incomplete")
    matches = [row for row in entries if row.get("namespace") == namespace]
    _require(len(matches) == 1 and matches[0].get("result") == result,
             "per-case result does not match exactly one published generation entry")
    context = result.get("request")
    _require(isinstance(context, dict), "result lacks original source context")
    _flags(context, kind)
    for name in ("protocol_id", "launch_closure_id", "cell_id"):
        _require(context.get(name) == expected[name], f"source {name} differs from expected binding")
    _require(context.get("split") == "development" and context.get("from_scratch") is True
             and context.get("skipped_initial_noise") is False, "preflight requires complete development inputs")
    _require(context.get("geometry") == geometry["source_geometry"], "result source geometry differs")
    _require(context.get("preflight_backend") == result.get("preflight_backend")
             and context["preflight_backend"] in ("reelcache", "native_noop"),
             "predeclared source backend differs from the result")
    _require(context.get("request_hash") == matches[0].get("request_hash"), "generation request hash differs")
    _sha(context["request_hash"])
    _require(context.get("asset_id") == generation.get("asset_id"), "generation assets differ")
    _sha(context["asset_id"])
    _sha(context.get("config_id"))
    for name in ("case_id", "group_id", "prompt_id", "prompt"):
        _text(context.get(name), name)
    _integer(context.get("seed"), "generation seed")
    chunks = _integer(context.get("chunks"), "generation chunks", 2)
    _require(chunks <= 3 and context.get("source_fps") == geometry["fps"], "horizon/FPS differs")
    _require(request["frame_window"] == [0, chunks * geometry["rgb_shape"][0]],
             "comparison frame window is not the complete genuine decoded horizon")
    binding = input_binding(
        SimpleNamespace(**{name: context[name] for name in (
            "case_id", "cell_id", "group_id", "prompt_id", "prompt", "seed", "chunks", "source_fps",
        )}, latent_shape=geometry["latent_shape"]),
        protocol_id=expected["protocol_id"], asset_id=context["asset_id"],
        config_id=context["config_id"], launch_closure_id=expected["launch_closure_id"],
    )
    for name, value in binding["sampler_contract"].items():
        _require(context.get(name) == value and generation.get(name) == value,
                 "source-bound serial sampler/config identity differs")
    bundle_receipt = result.get("input_bundle")
    _require(isinstance(bundle_receipt, dict) and bundle_receipt.get("schema") == INPUT_SCHEMA
             and bundle_receipt.get("mode") in ("capture", "replay")
             and bundle_receipt.get("binding_id") == identity(binding), "input bundle receipt binding differs")
    final_rng = _final_noise_rng(
        result.get("final_noise_rng"), binding_id=identity(binding), world_size=geometry["world_size"],
        required=kind == "native" or request.get("require_final_rng", False), budget=budget,
    )
    document = budget.document(bundle_receipt["artifact"])
    path = budget.artifact(bundle_receipt["artifact"])
    verify_input_bundle(path, bundle_receipt["artifact"]["sha256"], binding, max_bytes=budget.host)
    _require(bundle_receipt.get("bundle_id") == document.get("bundle_id"),
             "input bundle content identity differs")
    _require(result.get("conditioning_input") == document["conditioning"]
             and result.get("chunk_inputs") == document["chunks"], "actual checked input records differ from bundle")
    for record, names in [(document["conditioning"], CONDITIONING_NAMES),
                          *((row, NOISE_NAMES) for row in document["chunks"])]:
        tensors = _bundle_tensor(path.parent, record, names, budget)
        del tensors
    noises = [row["tensors"]["noise"]["sha256"] for row in document["chunks"]]
    latents = result.get("latents")
    _require(isinstance(latents, list) and len(latents) == chunks, "missing/extra generated latent chunks")
    for index, row in enumerate(latents):
        _require(isinstance(row, dict), "latent record must be an object")
        _shape(row.get("shape"), 5)
        _require(type(row.get("chunk_index")) is int and row["chunk_index"] == index
                 and row.get("shape") == geometry["latent_shape"] and row.get("dtype") == "float32"
                 and row.get("noise_sha256") == noises[index], "latent geometry/order/input noise differs")
        _sha(row.get("tensor_sha256"))
    media = decode.get("media")
    _require(isinstance(media, list), "decode media must be explicit")
    expected_media = {(item["namespace"], index) for item in entries
                      for index in range(item["result"]["request"]["chunks"])}
    actual_media = []
    for row in media:
        _require(isinstance(row, dict), "RGB record must be an object")
        _integer(row.get("chunk_index"), "RGB chunk index")
        actual_media.append((row.get("namespace"), row["chunk_index"]))
    _require(len(actual_media) == len(set(actual_media)) and set(actual_media) == expected_media,
             "decode media coverage has duplicate/missing/extra chunks or namespaces")
    selected = sorted((row for row in media if row["namespace"] == namespace), key=lambda row: row["chunk_index"])
    for index, row in enumerate(selected):
        _shape(row.get("shape"), 4)
        _require(isinstance(row.get("crop_rows"), list) and len(row["crop_rows"]) == 2
                 and all(type(value) is int for value in row["crop_rows"])
                 and row.get("shape") == geometry["rgb_shape"]
                 and type(row.get("fps")) is int and row["fps"] == geometry["fps"]
                 and row.get("crop_rows") == geometry["crop_rows"]
                 and row.get("VAE_state") == "stateless_chunk_decode"
                 and row.get("streaming_seam_parity_verified") is False
                 and row.get("latent_sha256") == latents[index]["tensor_sha256"],
                 "RGB crop/FPS/shape/window/latent/stateless-decode binding differs")
        _sha(row.get("rgb_sha256"))
    trace = _trace_index(result.get("attention_digests"), identity(binding), noises, geometry,
                         request["denoise_attention_calls"], budget)
    pairing = identity({
        "binding": binding, "bundle_id": document["bundle_id"], "frame_window": request["frame_window"],
        "fps": geometry["fps"], "crop_rows": geometry["crop_rows"], "VAE_state": "stateless_chunk_decode",
    })
    return _Trial(source["role"], source, generation, result, decode, matches[0]["sequence"],
                  _workers(generation, budget, geometry["world_size"]), binding, document, path.parent,
                  trace, selected, pairing, final_rng)


def _zip_directory_admission(path, budget, live_bytes, max_members):
    size = path.stat().st_size
    _require(size >= ZIP_END.size, "truncated ZIP end record")
    tail_bytes = min(size, ZIP_END.size + 65535)
    read_allowance = 3 * tail_bytes + 2048
    budget.admit(live_bytes + read_allowance)
    with path.open("rb") as stream:
        stream.seek(size - tail_bytes)
        tail = stream.read(tail_bytes)
        _require(len(tail) == tail_bytes, "ZIP changed during bounded footer read")
        position = tail.rfind(b"PK\x05\x06")
        _require(position >= 0 and len(tail) - position >= ZIP_END.size, "missing or truncated ZIP end record")
        end = ZIP_END.unpack(tail[position:position + ZIP_END.size])
        end_offset = size - tail_bytes + position
        _require(end_offset + ZIP_END.size + end[7] == size,
                 "unsupported ZIP trailing data or ambiguous end record")
        _require(end[1] == end[2] == 0 and end[3] == end[4], "multi-disk ZIP archives are unsupported")
        footer_start = end_offset
        count, directory_bytes, directory_offset = end[4:7]
        if end_offset >= ZIP64_LOCATOR.size:
            stream.seek(end_offset - ZIP64_LOCATOR.size)
            locator_raw = stream.read(ZIP64_LOCATOR.size)
        else:
            locator_raw = b""
        if locator_raw.startswith(b"PK\x06\x07"):
            _require(len(locator_raw) == ZIP64_LOCATOR.size, "truncated ZIP64 locator")
            locator = ZIP64_LOCATOR.unpack(locator_raw)
            _require(locator[1] == 0 and locator[3] == 1, "multi-disk ZIP64 archives are unsupported")
            footer_start = locator[2]
            _require(footer_start + ZIP64_END.size + ZIP64_LOCATOR.size == end_offset,
                     "unsupported ZIP64 end-record location or extension")
            stream.seek(footer_start)
            raw = stream.read(ZIP64_END.size)
            _require(len(raw) == ZIP64_END.size, "truncated ZIP64 end record")
            extended = ZIP64_END.unpack(raw)
            _require(extended[0] == b"PK\x06\x06" and extended[1] == ZIP64_END.size - 12
                     and extended[4] == extended[5] == 0 and extended[6] == extended[7],
                     "invalid or unsupported ZIP64 end record")
            count, directory_bytes, directory_offset = extended[7:10]
            _require(all(old == sentinel or old == new for old, new, sentinel in (
                (end[4], count, 0xffff), (end[5], directory_bytes, 0xffffffff),
                (end[6], directory_offset, 0xffffffff),
            )), "ZIP and ZIP64 directory declarations disagree")
        else:
            _require(count != 0xffff and directory_bytes != 0xffffffff and directory_offset != 0xffffffff,
                     "ZIP64 directory requires its actual locator and end record")
        _require(0 < count <= max_members, "ZIP archive member count exceeds its admitted limit")
        _require(directory_bytes >= count * ZIP_CENTRAL.size
                 and directory_offset + directory_bytes == footer_start,
                 "ZIP central-directory length or offset is inconsistent")
        _require(directory_bytes + budget.metadata_bytes <= budget.metadata_limit,
                 "ZIP directory exceeds the comparison metadata byte budget")
        # ZipFile materializes the whole directory in its constructor, before infolist().
        directory_allowance = read_allowance + 16 * directory_bytes + 8192 * count
        budget.admit(live_bytes + directory_allowance)
        stream.seek(directory_offset)
        for _ in range(count):
            raw = stream.read(ZIP_CENTRAL.size)
            _require(len(raw) == ZIP_CENTRAL.size, "truncated ZIP central-directory entry")
            entry = ZIP_CENTRAL.unpack(raw)
            length = sum(entry[10:13])
            _require(entry[0] == b"PK\x01\x02" and entry[10] > 0 and entry[13] == 0,
                     "invalid or unsupported ZIP central-directory entry")
            _require(stream.tell() + length <= footer_start, "ZIP central-directory member length exceeds its bounds")
            stream.seek(length, 1)
        _require(stream.tell() == footer_start, "ZIP central-directory member count or length differs")
    return count, directory_allowance


def _zip_admission(path, expected_data_bytes, budget, live_bytes=0, *, max_members=32):
    count, directory_allowance = _zip_directory_admission(path, budget, live_bytes, max_members)
    with zipfile.ZipFile(path) as archive:
        entries = archive.infolist()
        _require(len(entries) == count and len({row.filename for row in entries}) == len(entries),
                 "duplicate/excess archive members")
        total = sum(row.file_size for row in entries)
        _require(total <= expected_data_bytes + IO_BYTES, "archive uncompressed size exceeds declared geometry")
        budget.admit(live_bytes + directory_allowance + 8 * (total + path.stat().st_size))
        return entries


def _latent(record, budget, live_bytes=0):
    import torch

    path = budget.artifact(record["artifact"])
    size = math.prod(record["shape"]) * 4
    _zip_admission(path, size, budget, live_bytes)
    value = torch.load(path, map_location="cpu", weights_only=True)
    _require(isinstance(value, dict) and set(value) == {"latent"}
             and isinstance(value["latent"], torch.Tensor), "latent artifact must contain exactly its actual tensor")
    tensor = value["latent"]
    _require(list(tensor.shape) == record["shape"] and tensor.dtype == torch.float32
             and _tensor_hash(tensor) == record["tensor_sha256"], "actual latent metadata/hash differs")
    _require(_file_hash(path) == record["artifact"]["sha256"], "latent artifact changed during comparison")
    budget.tensor_bytes += tensor.numel() * tensor.element_size()
    return tensor


def _rgb(record, budget, live_bytes=0):
    import numpy as np

    path = budget.artifact(record["artifact"])
    size = math.prod(record["shape"])
    entries = _zip_admission(path, size, budget, live_bytes, max_members=1)
    _require(len(entries) == 1 and entries[0].filename == "frames.npy",
             "lossless RGB NPZ must contain only frames")
    with zipfile.ZipFile(path) as archive, archive.open("frames.npy") as stream:
        version = np.lib.format.read_magic(stream)
        _require(version in ((1, 0), (2, 0)), "unsupported RGB NPY header")
        shape, fortran, dtype = (np.lib.format.read_array_header_1_0(stream) if version == (1, 0)
                                  else np.lib.format.read_array_header_2_0(stream))
        _require(list(shape) == record["shape"] and dtype == np.uint8 and not fortran,
                 "RGB NPZ header differs from declared genuine THWC geometry")
        _require(entries[0].file_size == stream.tell() + size, "RGB payload is truncated or padded")
        raw = stream.read(size + 1)
        _require(len(raw) == size, "RGB payload changed during its bounded read")
    frames = np.frombuffer(raw, dtype=np.uint8).reshape(shape)
    _require(frames.dtype == np.uint8 and list(frames.shape) == record["shape"] and frames.flags.c_contiguous
             and hashlib.sha256(memoryview(frames).cast("B")).hexdigest() == record["rgb_sha256"],
             "actual RGB bytes/hash differ")
    _require(_file_hash(path) == record["artifact"]["sha256"], "RGB artifact changed during comparison")
    budget.rgb_bytes += frames.nbytes
    return frames


def _compare_pair(left, right, budget):
    import numpy as np

    _require(left.binding == right.binding and left.bundle["bundle_id"] == right.bundle["bundle_id"]
             and left.pairing_sha256 == right.pairing_sha256, "paired generation inputs/bundle/window differ")
    context_names = ("arm", "rank_selection_mode", "rank_k", "rank_v", "layer_ranks", "rank_policy_hash")
    _require(all(left.result["request"].get(name) == right.result["request"].get(name) for name in context_names),
             "paired representation/policy differs")
    checks = {}
    _require((left.final_rng is None) == (right.final_rng is None),
             "paired final_noise_rng coverage differs")
    final_rng = []
    if left.final_rng is not None:
        final_rng = [
            {"rank": rank, "bitwise_equal": a == b, "left_bytes": len(a), "right_bytes": len(b),
             "left_sha256": hashlib.sha256(a).hexdigest(), "right_sha256": hashlib.sha256(b).hexdigest()}
            for rank, (a, b) in enumerate(zip(left.final_rng, right.final_rng))
        ]
        checks["final_noise_rng_bitwise_equal"] = all(row["bitwise_equal"] for row in final_rng)
    records = [("conditioning", left.bundle["conditioning"], right.bundle["conditioning"], CONDITIONING_NAMES)]
    records += [(f"input_chunk_{i}", a, b, NOISE_NAMES)
                for i, (a, b) in enumerate(zip(left.bundle["chunks"], right.bundle["chunks"]))]
    for label, a, b, names in records:
        first = _bundle_tensor(left.bundle_root, a, names, budget)
        size = sum(item.numel() * item.element_size() for item in first.values())
        second = _bundle_tensor(right.bundle_root, b, names, budget, live_bytes=size)
        checks[label] = all(_same_tensor(first[name], second[name]) for name in names)
        del first, second
    checks["attention_events_values_and_archive_preservation"] = left.trace == right.trace
    for index, (a, b) in enumerate(zip(left.result["latents"], right.result["latents"])):
        first = _latent(a, budget)
        second = _latent(b, budget, live_bytes=first.numel() * first.element_size())
        checks[f"latent_chunk_{index}"] = _same_tensor(first, second)
        del first, second
        first = _rgb(left.media[index], budget)
        second = _rgb(right.media[index], budget, live_bytes=first.nbytes)
        checks[f"rgb_chunk_{index}"] = bool(np.array_equal(first, second))
        del first, second
    return {
        "left": left.role, "right": right.role, "pairing_sha256": left.pairing_sha256,
        "checks": checks, "passed": all(checks.values()),
        "pairing_excludes_output_pixels_latents_namespaces_and_timings": True,
        "final_generator_state_compared": bool(final_rng), "final_noise_rng": final_rng,
    }


def _magi(request, budget):
    geometry = _geometry(request)
    kind = request["comparison_kind"]
    _require(kind in ROLES, "unsupported MAGI comparison kind")
    sources = request.get("trials")
    _require(isinstance(sources, list) and all(isinstance(row, dict) for row in sources)
             and [row.get("role") for row in sources] == list(ROLES[kind]),
             "comparison needs exactly its declared ordered trial roles")
    calls = _integer(request.get("denoise_attention_calls"), "denoise_attention_calls", 1)
    _require(calls == DENOISE_ATTENTION_CALLS,
             "approved cfg_number=1 requires exactly one attention digest per denoise step")
    trials = {row["role"]: _load_trial(row, request, geometry, budget) for row in sources}
    budget.source_evidence_validated = True
    if kind == "raw_native_noop":
        raw, native = trials["raw"], trials["native_noop"]
        _require(raw.result.get("preflight_backend") == "reelcache" and raw.result.get("runtime_created") is True
                 and native.result.get("preflight_backend") == "native_noop"
                 and native.result.get("runtime_created") is False
                 and all(t.result["request"].get("arm") == "raw" for t in (raw, native)),
                 "RAW/native-noop comparison requires genuinely distinct declared backends")
        pairs = [(raw, native)]
    elif kind == "hot_reload":
        hot, reload = trials["hot"], trials["reload"]
        _require(hot.result.get("mode") == "quality_hot" and reload.result.get("mode") == "runtime"
                 and all(t.result.get("preflight_backend") == "reelcache"
                         and t.result.get("runtime_created") is True for t in (hot, reload)),
                 "hot/reload comparison requires the actual distinct history modes")
        pairs = [(hot, reload)]
    else:
        before, middle, after, fresh_a, fresh_b = (trials[name] for name in ROLES[kind])
        _require(before.generation == middle.generation == after.generation
                 and [before.sequence, middle.sequence, after.sequence] == [0, 1, 2]
                 and len(before.generation["results"]) == 3, "resident trial must be the complete ordered A/B/A epoch")
        _require(before.binding == after.binding == fresh_a.binding and middle.binding == fresh_b.binding,
                 "A/B/A and fresh cases differ")
        _require((before.binding["prompt"], before.binding["seed"]) !=
                 (middle.binding["prompt"], middle.binding["seed"]), "B must change the prompt or seed")
        _require(len(fresh_a.generation["results"]) == len(fresh_b.generation["results"]) == 1
                 and len({before.generation["batch_id"], fresh_a.generation["batch_id"],
                          fresh_b.generation["batch_id"]}) == 3, "fresh controls need separate single-request epochs")
        groups = [set(t.processes) for t in (before, fresh_a, fresh_b)]
        _require(all(not groups[i] & groups[j] for i in range(3) for j in range(i)),
                 "fresh trials reused resident/fresh worker process identities")
        _require(all(t.result.get("preflight_backend") == "reelcache" and t.result.get("runtime_created") is True
                     for t in trials.values()), "request-isolation trials require the actual runtime backend")
        pairs = [(before, after), (before, fresh_a), (middle, fresh_b)]
    compared = [_compare_pair(left, right, budget) for left, right in pairs]
    final_complete = all(row["final_generator_state_compared"] for row in compared)
    return {
        "pairs": compared,
        "missing_evidence": [] if final_complete else ["MAGI explicit final_noise_rng after generation/clean recapture"],
        "final_rng_scope": (
            "Actual final private noise-generator bytes are verified and compared for every rank; not postdraw substitutes."
            if final_complete else "Synthetic fixture omits final states; only actual per-chunk postdraw states are compared."
        ),
        "trial_binding_ids": {role: identity(trial.binding) for role, trial in trials.items()},
    }


def _causvid(request, budget):
    from examples.causvid_native_reference import compare_reference_outputs
    from examples.causvid_rank_worker import NATIVE_FPS, rgb_windows

    _require(request["comparison_kind"] == "raw_native_reference",
             "CausVid supports the existing four-artifact native-reference comparator")
    _require(request["expected"]["cell_id"] == "causvid480", "CausVid comparison cell differs")
    references = request.get("causvid_inputs")
    _require(isinstance(references, dict) and set(references) == set(CAUSVID_ROLES),
             "CausVid needs four explicit generation/decode result references")
    _receipt(request["launch_receipt"], request["expected"], request["evidence_kind"], budget)
    passed = {}
    for name, reference in references.items():
        document = budget.document(reference)
        _flags(document, request["evidence_kind"])
        _require(document.get("request", {}).get("cell_id") == "causvid480",
                 "CausVid source cell differs")
        for field in ("protocol_id", "launch_closure_id"):
            _require(document.get("provenance", {}).get(field) == request["expected"][field],
                     "CausVid source receipt binding differs")
        passed[name] = {key: reference[key] for key in ("path", "sha256")}
    raw = budget.document(references["raw_generation"])
    _require(request.get("frame_window") ==
             [0, rgb_windows(raw["request"]["chunks"], NATIVE_FPS)["prefix_frame_counts"][-1]],
             "CausVid comparison window must contain the full genuine native timeline")
    available = budget.host - IO_BYTES - budget.metadata_bytes * METADATA_MEMORY_FACTOR
    budget.admit(available)
    result = compare_reference_outputs(str(budget.root), **passed, max_host_bytes=available)
    _require(result.get("fixture") is (request["evidence_kind"] == "synthetic_fixture")
             and result.get("native_execution") is (request["evidence_kind"] == "native"),
             "delegated CausVid evidence flags differ")
    budget.source_evidence_validated = True
    return {
        "pairs": [{"left": "raw", "right": "native_reference", "checks": result["comparisons"],
                   "passed": result["all_compared_outputs_equal"]}],
        "causvid_comparison": result, "missing_evidence": [],
        "final_rng_scope": "Existing CausVid comparator verifies its explicit final RNG tensor.",
    }


def compare_preflight(request_path, *, request_sha256, output_path=None):
    """Return/persist a comparison receipt; never authorize native execution.

    Invalid request hashes/limits raise ComparisonError before publication.
    Artifact, coverage, admission or equality failures produce status='failed'.
    Output uses exclusive creation, including for failed comparisons.
    """
    started = time.perf_counter()
    path = Path(request_path).resolve(strict=True)
    _sha(request_sha256)
    _require(path.is_file() and path.stat().st_size <= 1 << 20, "comparison request exceeds its fixed metadata cap")
    _require(_file_hash(path) == request_sha256, "comparison request SHA256 changed")
    if output_path is not None:
        _require(not Path(output_path).exists(), "comparison output already exists")
    request = _json(path.read_bytes())
    _require(isinstance(request, dict) and request.get("schema") == REQUEST_SCHEMA, "unsupported comparison request")
    required = {"schema", "model", "comparison_kind", "evidence_kind", "artifact_root", "expected", "limits",
                "frame_window"}
    allowed = required | {"require_final_rng"} | (
        {"trials", "denoise_attention_calls", "synthetic_geometry"} if request.get("model") == "magi"
        else {"causvid_inputs", "launch_receipt"}
    )
    _require(required <= set(request) <= allowed, "missing or unexpected comparison request fields")
    _require(request.get("evidence_kind") in ("native", "synthetic_fixture"), "explicit evidence kind is required")
    _text(request.get("comparison_kind"), "comparison_kind")
    expected = request.get("expected")
    _require(isinstance(expected, dict) and set(expected) == {"protocol_id", "launch_closure_id", "cell_id"},
             "expected protocol, source closure and cell are required")
    _sha(expected["protocol_id"])
    _sha(expected["launch_closure_id"])
    _text(expected["cell_id"], "cell_id")
    _require(type(request.get("require_final_rng", False)) is bool, "require_final_rng must be boolean")
    _require(request.get("model") in ("magi", "causvid"), "model must be magi or causvid")
    budget = _Budget(request["artifact_root"], request["limits"])
    request_ref = {"path": str(path), "sha256": request_sha256, "bytes": path.stat().st_size}
    budget.document(request_ref)
    result, error = {}, None
    try:
        window = request.get("frame_window")
        _require(isinstance(window, list) and len(window) == 2
                 and all(type(value) is int for value in window), "explicit integer frame_window is required")
        result = (_magi if request["model"] == "magi" else _causvid)(request, budget)
        status = "passed" if all(row["passed"] for row in result["pairs"]) else "failed"
    except (ReelCacheError, ValueError, TypeError, KeyError, OSError, RuntimeError, EOFError,
            pickle.UnpicklingError, zipfile.BadZipFile, zlib.error) as exc:
        status, error = "failed", {"type": type(exc).__name__, "message": str(exc)}
    receipt = {
        "schema": RECEIPT_SCHEMA, "status": status, "comparison_kind": request["comparison_kind"],
        "evidence_kind": (request["evidence_kind"] if budget.source_evidence_validated
                          or request["evidence_kind"] == "synthetic_fixture" else "unverified"),
        "requested_evidence_kind": request["evidence_kind"],
        "source_evidence_validated": budget.source_evidence_validated,
        "request": request_ref, "expected": expected,
        "limits": request["limits"], "measurements": budget.measurements(),
        "verified_artifacts": list(budget.references.values()), **result, "error": error,
        "wall_seconds": time.perf_counter() - started,
        "scope": "CPU comparison of declared artifact evidence; not a complete G3 proof or launch authorization",
        "launch_receipt_binding_only": True, "launch_gate_reverified": False,
        "g3_qualification": False, "gate_advanced": False, "native_run_performed": False,
        "compares_result_namespace_or_timing": False,
    }
    receipt["comparison_id"] = identity(receipt)
    if output_path is not None:
        with Path(output_path).open("x", encoding="utf-8") as stream:
            json.dump(receipt, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
    return receipt


def sample_request():
    def reference(path):
        return {"path": path, "sha256": "<actual-file-sha256>", "bytes": 1}

    return {
        "schema": REQUEST_SCHEMA, "model": "magi", "comparison_kind": "raw_native_noop",
        "evidence_kind": "native", "artifact_root": "/owned/preflight-artifacts",
        "expected": {"protocol_id": "<frozen-protocol-sha256>", "launch_closure_id": "<frozen-closure-sha256>",
                     "cell_id": "magi720"},
        "limits": {"host_bytes": 2 << 30, "metadata_bytes": 64 << 20},
        "frame_window": [0, 48], "denoise_attention_calls": DENOISE_ATTENTION_CALLS, "require_final_rng": True,
        "trials": [
            {"role": role, "generation": reference(f"{role}/generation_complete.json"),
             "result": reference(f"{role}/case/result.json"),
             "decode": reference(f"{role}/decode_batch_complete.json"),
             "launch_receipt": reference("launch_receipt.json")}
            for role in ROLES["raw_native_noop"]
        ],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path)
    parser.add_argument("--request-sha256")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--example", action="store_true", help="print a template; replace all paths/hashes/byte counts")
    args = parser.parse_args(argv)
    if args.example:
        print(json.dumps(sample_request(), sort_keys=True, indent=2))
        return 0
    if args.request is None or args.request_sha256 is None or args.output is None:
        parser.error("--request, --request-sha256 and --output are required")
    try:
        receipt = compare_preflight(args.request, request_sha256=args.request_sha256, output_path=args.output)
    except (ComparisonError, OSError) as error:
        print(f"preflight comparison rejected: {error}", file=sys.stderr)
        return 2
    print(json.dumps({"status": receipt["status"], "comparison_id": receipt["comparison_id"],
                      "output": str(args.output), "g3_qualification": False}, sort_keys=True))
    return 0 if receipt["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
