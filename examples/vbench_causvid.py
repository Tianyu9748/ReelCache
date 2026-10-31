#!/usr/bin/env python3
"""Fresh official-prompt CausVid paired generation for a coordinator-run VBench study.

This is not the official VBench evaluator. It emits source-bound continuation
RGB, lossless evaluator videos, latent states, paired SSIM and two explicitly
separate compression denominators. The coordinator freezes cases/cells and runs
the official evaluator. No pilot conditioning or prefix files are accepted.

``schema`` and ``prepare`` never import a model or initialize CUDA. ``run``
requires both a byte-bound case manifest and a separate GPU0 grant. Active data
stays under an explicitly supplied local-SSD root; ``export`` copies only
verified compact outputs, never an entire RAW store.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "reelcache.causvid.vbench.case.v1"
OFFICIAL_REPOSITORY = "https://github.com/Vchitect/VBench"
NATIVE_REQUIRED = {
    "third_party/causvid/modeling/" + name
    for name in ("causal_model.py", "causal_inference.py", "wan_wrapper.py", "attention.py", "vae.py")
}
FILE_FIELDS = {"config", "checkpoint", "base_config", "text_encoder", "vae"}
FIXED_CODEC = {
    "K_rank": 256, "V_rank": 384, "normalization": "per_channel",
    "protected_fraction": 1.0, "pix_fmt": "yuv444p", "qp": 8, "preset": "ultrafast",
}


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            value.update(block)
    return value.hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()


def write_json(path, value, *, mutable=False):
    path = Path(path)
    if path.exists() and not mutable:
        raise FileExistsError(f"Refusing to overwrite evidence: {path}")
    staged = path.with_name(path.name + ".pending")
    with staged.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    staged.replace(path)


def _pairs(items):
    value = {}
    for key, item in items:
        if key in value:
            raise ValueError(f"Duplicate manifest key: {key}")
        value[key] = item
    return value


def _json_bytes(data):
    def reject(value):
        raise ValueError(f"Nonfinite JSON number: {value}")
    return json.loads(data, object_pairs_hook=_pairs, parse_constant=reject)


def _slug(value, name):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,120}", value):
        raise ValueError(f"{name} must be a nonempty safe identifier")


def _digest(value):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("Expected lowercase SHA256")


def validate_case(case):
    required = {
        "schema", "case_id", "cohort_id", "protocol_id", "master_protocol_sha256", "protocol_sha256", "protocol",
        "prompt", "seed", "model", "implementation", "ffmpeg",
    }
    if not isinstance(case, dict) or set(case) != required or case["schema"] != SCHEMA:
        raise ValueError("Use the fresh CausVid case schema; pilot/latent/conditioning paths are not accepted")
    for name in ("case_id", "cohort_id", "protocol_id"):
        _slug(case[name], name)
    _digest(case["master_protocol_sha256"])
    if type(case["seed"]) is not int or case["seed"] < 0:
        raise ValueError("seed must be a nonnegative integer")
    protocol = case["protocol"]
    if set(protocol) != {
        "chunks", "fps", "prefix_rgb_frames", "continuation_frames", "arms", "codec",
        "hot_history_bytes", "lease_budget_bytes", "attention_budget_bytes",
        "replay_chunks", "fork_seed", "write_lossless_video",
    }:
        raise ValueError("Protocol fields must be explicit and complete")
    for name in ("chunks", "fps", "prefix_rgb_frames", "continuation_frames"):
        if type(protocol[name]) is not int:
            raise ValueError(f"Protocol {name} must be an integer")
    if protocol["chunks"] < 2 or protocol["fps"] <= 0 or protocol["prefix_rgb_frames"] != 9:
        raise ValueError("Use at least two native 3-latent-frame chunks and the nine-RGB-frame prefix")
    if protocol["continuation_frames"] != 12 * (protocol["chunks"] - 1):
        raise ValueError("CausVid continuation is 12 RGB frames per subsequent native chunk")
    if protocol["arms"] not in (["raw", "svd_hevc"], ["raw", "svd", "svd_hevc"]):
        raise ValueError("Freeze RAW and SVD_HEVC, optionally with a separate SVD diagnostic")
    if protocol["codec"] != FIXED_CODEC:
        raise ValueError("This driver implements fixed per-channel yuv444p/QP8/ultrafast K256/V384")
    for name in ("hot_history_bytes", "lease_budget_bytes", "attention_budget_bytes", "replay_chunks", "fork_seed"):
        if type(protocol[name]) is not int or protocol[name] < 0:
            raise ValueError(f"Protocol {name} must be a nonnegative integer")
    if not protocol["lease_budget_bytes"] or not protocol["attention_budget_bytes"]:
        raise ValueError("Lease and attention budgets must be positive")
    maximum_history_tokens = (protocol["chunks"] - 1) * 4680
    complete_layer_bytes = maximum_history_tokens * 1536 * 2 * 2
    history_factor = 36 if protocol["hot_history_bytes"] >= complete_layer_bytes else 32
    attention_floor = maximum_history_tokens * 1536 * history_factor + 4680 * 1536 * 32
    if protocol["attention_budget_bytes"] < attention_floor:
        raise ValueError("Attention budget cannot admit the complete requested native history")
    if protocol["replay_chunks"] >= protocol["chunks"] or type(protocol["write_lossless_video"]) is not bool:
        raise ValueError("Invalid replay scope or evaluator-video selection")
    if protocol["replay_chunks"] and protocol["fork_seed"] == case["seed"]:
        raise ValueError("The changed-seed fork check requires a different seed")
    if case["protocol_sha256"] != identity(protocol):
        raise ValueError("Protocol ID is not bound to the supplied protocol content")
    prompt = case["prompt"]
    if set(prompt) != {"id", "text", "catalog", "catalog_index", "catalog_revision", "repository", "dimensions"}:
        raise ValueError("An exact official catalog row and source binding are required")
    _slug(prompt["id"], "prompt.id")
    if not isinstance(prompt["text"], str) or not prompt["text"].strip():
        raise ValueError("Fresh prompt text must be nonempty")
    if prompt["repository"] != OFFICIAL_REPOSITORY:
        raise ValueError("This schema expects the official Vchitect/VBench prompt catalog")
    if not isinstance(prompt["catalog_revision"], str) or not re.fullmatch(r"[0-9a-f]{40}", prompt["catalog_revision"]):
        raise ValueError("Pin an official catalog repository commit")
    if type(prompt["catalog_index"]) is not int or prompt["catalog_index"] < 0:
        raise ValueError("catalog_index is an explicit zero-based row")
    if not isinstance(prompt["dimensions"], list) or not all(isinstance(item, str) for item in prompt["dimensions"]):
        raise ValueError("Official catalog dimensions must be a list of names")
    model = case["model"]
    if set(model) != {"name", "model_code_root", "base_model", "files", "native_sources"} or model["name"] != "causvid-480x832":
        raise ValueError("Unsupported model binding")
    if any(not isinstance(model[name], str) or not Path(model[name]).is_absolute()
           for name in ("model_code_root", "base_model")):
        raise ValueError("Native code and base-model roots must be explicit absolute paths")
    if set(model["files"]) != FILE_FIELDS or not NATIVE_REQUIRED <= set(model["native_sources"]):
        raise ValueError("Complete source/config/checkpoint/T5/VAE bindings are required")
    if not isinstance(case["implementation"], dict) or not case["implementation"]:
        raise ValueError("An immutable implementation hash table is required")
    for name, value in {**model["native_sources"], **case["implementation"]}.items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or relative.suffix != ".py":
            raise ValueError("Source hash keys must be relative Python paths without traversal")
        _digest(value)
    for item in [prompt["catalog"], case["ffmpeg"], *model["files"].values()]:
        if not isinstance(item, dict) or set(item) != {"path", "sha256", "bytes"}:
            raise ValueError("File bindings require path, sha256 and bytes")
        _digest(item["sha256"])
        if not isinstance(item["path"], str) or not Path(item["path"]).is_absolute():
            raise ValueError("External prerequisite paths must be explicit and absolute")
        if type(item["bytes"]) is not int or item["bytes"] <= 0:
            raise ValueError("External prerequisite length must be positive")
    return case


def read_case(path, expected):
    _digest(expected)
    data = Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError("Parent-supplied immutable case manifest hash differs")
    return validate_case(_json_bytes(data))


def implementation_files():
    return sorted(
        [path.relative_to(ROOT).as_posix() for path in (ROOT / "reelcache").rglob("*.py")]
        + ["examples/native_causvid.py", "examples/vbench_causvid.py"]
    )


def _verify_bound_file(entry):
    path = Path(entry["path"])
    if not path.is_file() or path.stat().st_size != entry["bytes"] or sha256(path) != entry["sha256"]:
        raise ValueError(f"Prerequisite hash or length differs: {path}")
    return path


def preflight(case, manifest_path, manifest_sha256):
    if any(name.startswith("third_party.causvid") for name in sys.modules):
        raise RuntimeError("Fresh-case preflight must precede all native model imports")
    if "torch" in sys.modules and sys.modules["torch"].cuda.is_initialized():
        raise RuntimeError("Fresh-case CPU preflight must not initialize CUDA")
    validate_case(case)
    if sha256(manifest_path) != manifest_sha256:
        raise ValueError("Case manifest changed after loading")
    catalog_path = _verify_bound_file(case["prompt"]["catalog"])
    catalog = _json_bytes(catalog_path.read_bytes())
    index = case["prompt"]["catalog_index"]
    if not isinstance(catalog, list) or index >= len(catalog):
        raise ValueError("Official prompt catalog index is out of range")
    record = catalog[index]
    if record.get("prompt_en") != case["prompt"]["text"] or record.get("dimension") != case["prompt"]["dimensions"]:
        raise ValueError("Case text/dimensions are not the exact bound official prompt row")
    model = case["model"]
    code_root, base = Path(model["model_code_root"]).resolve(), Path(model["base_model"]).resolve()
    files = {name: _verify_bound_file(entry) for name, entry in model["files"].items()}
    _verify_bound_file(case["ffmpeg"])
    if not os.access(case["ffmpeg"]["path"], os.X_OK):
        raise ValueError("Explicit FFmpeg prerequisite is not executable")
    for name, expected_path in {
        "base_config": base / "config.json",
        "text_encoder": base / "models_t5_umt5-xxl-enc-bf16.pth",
        "vae": base / "Wan2.1_VAE.pth",
    }.items():
        if files[name].resolve() != expected_path:
            raise ValueError(f"{name} does not belong to the explicitly configured base model")
    config = _json_bytes(files["base_config"].read_bytes())
    if any(config.get(key) != value for key, value in {
        "model_type": "t2v", "dim": 1536, "num_heads": 12, "num_layers": 30, "in_dim": 16, "out_dim": 16,
    }.items()):
        raise ValueError("Base model is not the qualified native 480x832 CausVid geometry")
    if set(case["implementation"]) != set(implementation_files()):
        raise ValueError("Implementation manifest does not cover exactly the executing package and both drivers")
    native_paths = {
        path.relative_to(code_root).as_posix()
        for path in (code_root / "third_party/causvid/modeling").rglob("*.py")
    }
    if not native_paths <= set(model["native_sources"]):
        raise ValueError("Native modeling source coverage is incomplete")
    source_files = {
        "case_manifest": {"path": str(Path(manifest_path).resolve()), "sha256": manifest_sha256},
        "official_catalog": {"path": str(catalog_path), "sha256": case["prompt"]["catalog"]["sha256"]},
    }
    for prefix, root, table in (
        ("implementation", ROOT, case["implementation"]),
        ("native", code_root, model["native_sources"]),
    ):
        for relative, expected in table.items():
            path = (root / relative).resolve()
            path.relative_to(root)
            if sha256(path) != expected:
                raise ValueError(f"Bound {prefix} source differs: {relative}")
            source_files[prefix + "/" + relative] = {"path": str(path), "sha256": expected}
    for name in ("config", "base_config"):
        source_files[name] = {"path": str(files[name]), "sha256": model["files"][name]["sha256"]}
    return {
        "schema": "reelcache.causvid.vbench.preflight.v1", "status": "CPU_preflight_passed",
        "case_manifest_sha256": manifest_sha256, "protocol_id": case["protocol_id"],
        "master_protocol_sha256": case["master_protocol_sha256"],
        "protocol_sha256": case["protocol_sha256"], "case_id": case["case_id"],
        "source_files": source_files, "model_files": model["files"], "ffmpeg": case["ffmpeg"],
        "model_fingerprint": identity({"model": model, "native_dtype": "bfloat16"}),
        "official_catalog_row_verified": True,
        "official_repository_provenance": "coordinator-pinned catalog commit/hash; exact row verified locally",
        "model_loaded": False, "CUDA_initialized": False,
        "common_prefix": "must be generated from actual case prompt and seed; no saved prefix inputs",
        "geometry": {"height": 480, "width": 832, "layers": 30, "heads": 12, "head_dim": 128,
                     "chunk_tokens": 4680, "latent_frames_per_chunk": 3, "source_dtype": "bfloat16"},
        "requested_horizon": {
            "chunks": case["protocol"]["chunks"],
            "total_latent_frames": 3 * case["protocol"]["chunks"],
            "total_RGB_frames": 12 * case["protocol"]["chunks"] - 3,
            "continuation_RGB_frames": case["protocol"]["continuation_frames"],
            "maximum_past_tokens": 4680 * (case["protocol"]["chunks"] - 1),
            "native_dense_KV_bytes_at_final_seal": case["protocol"]["chunks"] * 862617600,
            "full_attention_ready_history_bytes_before_last_chunk": (case["protocol"]["chunks"] - 1) * 862617600,
            "native_32760_token_preallocation_used": False,
            "native_seq_len_check": "per-forward current 4680-token chunk; absolute RoPE and full-prefix history are preserved",
        },
        "official_VBench_scoring": "separate coordinator evaluation; no full-benchmark claim for a small cohort",
    }


def verify_sources(provenance):
    for name, entry in provenance["source_files"].items():
        if sha256(entry["path"]) != entry["sha256"]:
            raise RuntimeError(f"Source-bound benchmark input changed: {name}")


def storage_placement(root):
    root = Path(root).resolve()
    if not root.is_dir():
        raise ValueError("The coordinator must create the active local-SSD root first")
    information = root.stat()
    if information.st_uid != os.geteuid() or information.st_mode & 0o077:
        raise ValueError("Use the coordinator-created, user-owned private local-SSD root")
    result = subprocess.run(
        ["findmnt", "-n", "-r", "-T", str(root), "-o", "SOURCE,FSTYPE,TARGET"],
        capture_output=True, text=True, check=True, timeout=30,
    ).stdout.strip().split(maxsplit=2)
    if len(result) != 3:
        raise ValueError("Cannot establish active filesystem placement")
    source, filesystem, mount = result
    if filesystem not in {"ext4", "xfs", "btrfs"} or not source.startswith("/dev/nvme"):
        raise ValueError("Active native stores require the explicit local NVMe path, not NFS or shared memory")
    return {
        "root": str(root), "source": source, "filesystem": filesystem, "mount": mount,
        "owner_uid": information.st_uid, "mode": oct(information.st_mode & 0o777),
        "device_id": root.stat().st_dev, "free_bytes": shutil.disk_usage(root).free,
        "model_weights_copied": False, "NFS_speed_comparison": False,
    }


def native_driver():
    spec = importlib.util.spec_from_file_location("reelcache_fresh_causvid_builder", ROOT / "examples/native_causvid.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(case, provenance, directory, *, threads):
    verify_sources(provenance)
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from reelcache.adapters.causvid import CausVidConfig, tensor_digest
    from reelcache.adapters.causvid_benchmark import (
        allocator_sample, filesystem_bytes, fresh_inputs, generate_matched_case, preserve_global_rng,
    )
    from reelcache.codecs import hevc_capabilities, using_ffmpeg
    from reelcache.manager import KVRuntime
    from reelcache.types import ModelIdentity

    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Fresh native case requires exactly one explicitly authorized visible GPU")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    config, protocol = CausVidConfig(), case["protocol"]
    model = case["model"]
    args = SimpleNamespace(
        model_code_root=Path(model["model_code_root"]), base_model=Path(model["base_model"]),
        checkpoint=Path(model["files"]["checkpoint"]["path"]), config=Path(model["files"]["config"]["path"]),
    )
    builder = native_driver()
    ffmpeg = Path(case["ffmpeg"]["path"])
    capabilities = hevc_capabilities(ffmpeg)
    if "yuv444p" not in capabilities["pixel_formats"]:
        raise RuntimeError("Configured FFmpeg lacks required HEVC yuv444p")
    start = time.perf_counter()

    def progress(phase, detail):
        verify_sources(provenance)
        row = {
            "case_id": case["case_id"], "phase": phase, **detail,
            "elapsed_seconds": time.perf_counter() - start, "utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        }
        write_json(directory / "progress.json", row, mutable=True)
        with (directory / "progress.jsonl").open("a") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        print(f"[CausVid fresh] {case['case_id']} {phase}", flush=True)

    with preserve_global_rng("cuda:0"), torch.no_grad():
        progress("model-loading", {})
        pipeline = builder.build_pipeline(args)
        load_seconds = time.perf_counter() - start
        progress("fresh-text-encoding", {"prompt_id": case["prompt"]["id"]})
        noise, conditioning, rng = fresh_inputs(
            pipeline, case["prompt"]["text"], case["seed"], config, protocol["chunks"],
        )
        if conditioning["prompt_embeds"].dtype != torch.bfloat16:
            raise RuntimeError("Qualified CausVid conditioning must remain BF16")
        conditioning_id = identity({
            "prompt": case["prompt"], "embeddings_sha256": tensor_digest(conditioning["prompt_embeds"]),
        })
        initialization_memory = allocator_sample("cuda:0")
        native_identity = ModelIdentity("causvid", provenance["model_fingerprint"], 30, 1536, "tokens_channels", "bfloat16")
        with (
            KVRuntime(directory / "runtime", native_identity, max_resident_bytes=protocol["lease_budget_bytes"]) as runtime,
            builder.instance_cross_attention_sdpa(pipeline),
            sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]),
            using_ffmpeg(ffmpeg),
        ):
            result = generate_matched_case(
                pipeline, runtime, config=config, noise=noise, conditioning=conditioning, rng=rng,
                conditioning_id=conditioning_id, arms=protocol["arms"], directory=directory,
                prefix_rgb_frames=protocol["prefix_rgb_frames"], fps=protocol["fps"],
                history_attention_bytes=protocol["attention_budget_bytes"],
                hot_history_bytes=protocol["hot_history_bytes"], replay_chunks=protocol["replay_chunks"],
                fork_seed=protocol["fork_seed"], ffmpeg=ffmpeg if protocol["write_lossless_video"] else None,
                progress=progress, rgb_size=(480, 832),
            )
            result["model_identity"] = native_identity.to_dict()
        result["filesystem_after_store_close"] = filesystem_bytes(directory / "runtime")
        result["initialization_memory"] = initialization_memory
        for name, value in allocator_sample("cuda:0").items():
            if isinstance(value, int):
                result["peak_observations"][name] = max(
                    value, initialization_memory.get(name, 0), result["peak_observations"].get(name, 0),
                )
        result.update(
            schema="reelcache.causvid.vbench.result.v1", case_id=case["case_id"], cohort_id=case["cohort_id"],
            protocol_id=case["protocol_id"], protocol_sha256=case["protocol_sha256"],
            master_protocol_sha256=case["master_protocol_sha256"],
            protocol=protocol,
            prompt=case["prompt"], seed=case["seed"], provenance=provenance,
            fresh_conditioning_sha256=tensor_digest(conditioning["prompt_embeds"]),
            model_load_seconds=load_seconds, seconds=time.perf_counter() - start,
            hardware={"GPU": torch.cuda.get_device_name(), "torch": torch.__version__, "cuda": torch.version.cuda},
            execution_mode="serial closed-loop clean history; optional bounded decoded-history reuse, no pipeline overlap claim",
            completed_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        )
    if result["continuation_frames"] != protocol["continuation_frames"]:
        raise RuntimeError("Generated continuation does not match the frozen protocol")
    verify_sources(provenance)
    return result


def compact_export(result_path, destination, *, max_bytes, checkpoints=()):
    """Export only verified outputs; keep every active/raw store off NFS."""
    result_path = Path(result_path).resolve()
    result = _json_bytes(result_path.read_bytes())
    if result.get("schema") != "reelcache.causvid.vbench.result.v1" or result.get("status") != "passed":
        raise ValueError("Only a completed verified fresh-case result can be exported")
    destination = Path(destination)
    selected = {"result.json": _file_binding(result_path)}
    for name in ("case.json", "preflight.json", "execution.json", "progress.jsonl"):
        path = result_path.parent / name
        if path.is_file():
            selected[name] = _file_binding(path)
    if result.get("manifest_ledger") is not None:
        ledger = result["manifest_ledger"]
        Path(ledger["path"]).resolve().relative_to(result_path.parent)
        selected["manifest_ledger.jsonl"] = ledger
    for arm, value in result["arms"].items():
        if arm not in ("raw", "svd", "svd_hevc"):
            raise ValueError("Unknown arm in completed export")
        for label in ("RGB", "latent_state", "video"):
            entry = value["outputs"].get(label)
            if entry is not None:
                path = Path(entry["path"]).resolve()
                path.relative_to(result_path.parent)
                selected[f"{arm}/{path.name}"] = entry
    checkpoint_ids = []
    for selection in checkpoints:
        if not isinstance(selection, str) or selection.count(":") != 1:
            raise ValueError("Checkpoint selection must be arm:prefix or arm:final")
        arm, boundary = selection.split(":")
        if arm not in result["arms"] or boundary not in ("prefix", "final"):
            raise ValueError("Checkpoint selection is outside the completed run")
        checkpoint_ids.append(result["arms"][arm][boundary + "_snapshot"])
    checkpoint_bytes = 0
    if checkpoint_ids:
        from reelcache.adapters.causvid_benchmark import object_graph
        from reelcache.manager import KVRuntime
        from reelcache.types import ModelIdentity
        model = ModelIdentity.from_dict(result["model_identity"])
        with KVRuntime(result_path.parent / "runtime", model) as source:
            checkpoint_bytes = object_graph(source.store, checkpoint_ids)["unique_file_bytes"]
    total = sum(entry["bytes"] for entry in selected.values()) + checkpoint_bytes
    if type(max_bytes) is not int or max_bytes <= 0 or total > max_bytes:
        raise ValueError("Compact export exceeds its explicit byte budget")
    for entry in selected.values():
        _verify_bound_file(entry)
    destination.mkdir(parents=True, exist_ok=False)
    records = {}
    for relative, entry in selected.items():
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        with Path(entry["path"]).open("rb") as source, target.open("xb") as output:
            shutil.copyfileobj(source, output, length=4 * 1024**2)
            output.flush()
            os.fsync(output.fileno())
        if target.stat().st_size != entry["bytes"] or sha256(target) != entry["sha256"]:
            raise RuntimeError("Exported result did not verify byte-for-byte")
        records[relative] = {"source": entry["path"], "bytes": entry["bytes"], "sha256": entry["sha256"]}
    checkpoint_record = None
    if checkpoint_ids:
        from reelcache.adapters.causvid_benchmark import export_snapshot_objects
        with (
            KVRuntime(result_path.parent / "runtime", model) as source,
            KVRuntime(destination / "selected_checkpoints", model) as target,
        ):
            checkpoint_record = export_snapshot_objects(source, target, checkpoint_ids)
        for path in (destination / "selected_checkpoints").rglob("*"):
            if path.is_file():
                record = _file_binding(path)
                records[path.relative_to(destination).as_posix()] = {
                    "source": "selected immutable snapshot closure", "bytes": record["bytes"], "sha256": record["sha256"],
                }
    write_json(destination / "copy_manifest.json", {
        "schema": "reelcache.causvid.vbench.compact_export.v1", "status": "verified",
        "files": records, "total_copied_bytes": sum(record["bytes"] for record in records.values()),
        "budgeted_content_bytes": total, "checkpoint_object_bytes": checkpoint_bytes,
        "runtime_store_exported": False,
        "RAW_store_deleted": False, "selected_checkpoints": list(checkpoints),
        "checkpoint_export": checkpoint_record,
        "budget_scope": "output files and unique snapshot objects; database/index and copy-manifest overhead reported separately",
        "note": "Only explicitly selected snapshot closures are copied; no original runtime index or unrelated RAW branches.",
    })
    return records


def _file_binding(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}


def schema_example():
    protocol = {
        "chunks": 9, "fps": 24, "prefix_rgb_frames": 9, "continuation_frames": 96,
        "arms": ["raw", "svd_hevc"], "codec": dict(FIXED_CODEC),
        "hot_history_bytes": 0, "lease_budget_bytes": 1024**3, "attention_budget_bytes": 4 * 1024**3,
        "replay_chunks": 1, "fork_seed": 1, "write_lossless_video": True,
    }
    unbound = {"path": "/EXPLICIT/PARENT/SUPPLIED/FILE", "sha256": "REQUIRED_SHA256", "bytes": 1}
    return {
        "schema": SCHEMA, "case_id": "official-prompt-000-seed0", "cohort_id": "PARENT_FROZEN_COHORT",
        "protocol_id": "PARENT_FROZEN_PROTOCOL", "master_protocol_sha256": "REQUIRED_MASTER_PROTOCOL_SHA256",
        "protocol_sha256": identity(protocol), "protocol": protocol,
        "prompt": {
            "id": "official-prompt-000", "text": "EXACT_OFFICIAL_PROMPT_TEXT",
            "catalog": dict(unbound), "catalog_index": 0, "catalog_revision": "REQUIRED_40_HEX_COMMIT",
            "repository": OFFICIAL_REPOSITORY, "dimensions": ["OFFICIAL_ROW_DIMENSIONS"],
        },
        "seed": 0, "ffmpeg": dict(unbound),
        "model": {
            "name": "causvid-480x832", "model_code_root": "/EXPLICIT/RAVEN", "base_model": "/EXPLICIT/WAN",
            "files": {name: dict(unbound) for name in sorted(FILE_FIELDS)},
            "native_sources": {name: "REQUIRED_SHA256" for name in sorted(NATIVE_REQUIRED)},
        },
        "implementation": {relative: sha256(ROOT / relative) for relative in implementation_files()},
    }


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    commands = cli.add_subparsers(dest="command", required=True)
    commands.add_parser("schema", help="Print a manifest template and current implementation hashes")
    for operation in ("prepare", "run"):
        sub = commands.add_parser(operation)
        sub.add_argument("--case-manifest", type=Path, required=True)
        sub.add_argument("--case-sha256", required=True)
        if operation == "prepare":
            sub.add_argument("--output", type=Path, required=True, help="New compact CPU preflight JSON")
        else:
            sub.add_argument("--work-root", type=Path, required=True, help="Existing coordinator-created local NVMe root")
            sub.add_argument("--run-id", required=True, help="New run identifier; never reuses an existing output")
            sub.add_argument("--allow-cuda", action="store_true")
            sub.add_argument("--gpu", type=int, choices=(0,), default=0)
            sub.add_argument("--threads", type=int, choices=range(1, 9), default=4)
    export = commands.add_parser("export")
    export.add_argument("--result", type=Path, required=True)
    export.add_argument("--destination", type=Path, required=True)
    export.add_argument("--max-bytes", type=int, required=True)
    export.add_argument("--checkpoint", action="append", default=[], help="Explicit arm:prefix or arm:final; otherwise no store")
    return cli


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command == "schema":
        print(json.dumps(schema_example(), indent=2))
        return
    if args.command == "export":
        if args.checkpoint:
            sys.path.insert(0, str(ROOT))
        records = compact_export(args.result, args.destination, max_bytes=args.max_bytes, checkpoints=args.checkpoint)
        print(json.dumps({"status": "verified", "copied_files": len(records)}))
        return
    case = read_case(args.case_manifest, args.case_sha256)
    if args.command == "prepare":
        result = preflight(case, args.case_manifest, args.case_sha256)
        write_json(args.output, result)
        print(json.dumps({"status": result["status"], "path": str(args.output), "model_loaded": False}))
        return
    if not args.allow_cuda:
        raise ValueError("Run requires --allow-cuda and a separate current coordinator GPU0 grant")
    _slug(args.run_id, "run-id")
    placement = storage_placement(args.work_root)
    directory = Path(placement["root"]) / args.run_id
    directory.mkdir(exist_ok=False)
    builder = native_driver()
    builder.environment(directory, gpu=args.gpu, threads=args.threads)
    write_json(directory / "execution.json", {
        "command": shlex.join([sys.executable, "-B", str(Path(__file__).resolve()), *(sys.argv[1:] if argv is None else argv)]),
        "pid": os.getpid(), "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "storage": placement, "case_sha256": args.case_sha256, "model_weights_copied": False,
        "GPU0_grant": "externally coordinated; flag alone is not authorization",
        "environment": {key: os.environ[key] for key in (
            "CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "TMPDIR", "TRITON_CACHE_DIR",
        )},
    })
    try:
        with (directory / "case.json").open("xb") as stream:
            stream.write(args.case_manifest.read_bytes())
        provenance = preflight(case, args.case_manifest, args.case_sha256)
        write_json(directory / "preflight.json", provenance)
        result = run(case, provenance, directory, threads=args.threads)
        write_json(directory / "result.json", result)
        print(json.dumps({"status": result["status"], "result": str(directory / "result.json")}))
    except BaseException as exc:
        write_json(directory / "failure.json", {
            "schema": "reelcache.causvid.vbench.failure.v1", "status": "failed", "case_id": case["case_id"],
            "error": repr(exc), "traceback": traceback.format_exc(), "case_sha256": args.case_sha256,
            "utc": dt.datetime.now(dt.timezone.utc).isoformat(), "partial_outputs_are_not_completed_evidence": True,
        })
        raise


if __name__ == "__main__":
    main()
