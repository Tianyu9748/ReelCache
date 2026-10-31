"""Fresh, explicitly selected official-VBench cases on serial native MAGI SP2.

This executable helper reuses only the qualified builder/configuration provenance,
never a pilot's prompt, embedding, noise or prefix. ``prepare`` is model-free;
``condition`` explicitly loads native FP32 T5 on CPU. Native generation and VAE
decoding have independent GPU grants. The four-prompt subset is not a full VBench
score. ``schema`` prints the immutable case/protocol contract.

All outputs are confined to this source checkout. Freeze/copy the checkout onto
the parent's approved SSD before preparing a native run there. Completed stores
are intentionally NOT copied by this driver; compact_artifacts.json enumerates
the persistent evidence separately from local canonical stores and RGB scratch.
"""

from __future__ import annotations

from contextlib import contextmanager, ExitStack, nullcontext
from datetime import timedelta
import ast
import gc
import hashlib
import importlib
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys
import time
from unittest.mock import patch

from . import magi_native as native


GIB = 1024**3
CASE_SCHEMA = "reelcache_vbench_case_v1"
PROTOCOL_SCHEMA = "reelcache_vbench_protocol_v1"
PLAN_SCHEMA = "reelcache_fresh_magi_vbench_v1"
FIXED_MAGI = {
    "K_rank": 256, "V_rank": 384, "normalization": "per_channel",
    "pix_fmt": "yuv444p", "qp": 8, "preset": "ultrafast", "reuse_threshold": None,
    "hot_window_bytes": 24 * GIB, "hot_headroom_bytes": 12 * GIB,
    "resident_mib": 1536, "special_tokens": [],
}


def schema(args=None):
    value = {
        "case": {
            "schema": CASE_SCHEMA, "protocol_id": "PARENT_FROZEN_PROTOCOL_ID",
            "case_id": "vbench_0000_seed0", "prompt_id": 0,
            "prompt": "EXACT catalogue[0].prompt_en; this is a placeholder, not a selected prompt", "seed": 0,
        },
        "protocol": {
            "schema": PROTOCOL_SCHEMA, "protocol_id": "PARENT_FROZEN_PROTOCOL_ID",
            "weight_identity": {
                "path": "/configured/study/weight_identity.json", "sha256": "MANIFEST_FILE_SHA256",
                "content_manifest_sha256": "PARENT_COMBINED_WEIGHT_FINGERPRINT",
            },
            "magi_auxiliary_identity": {
                "path": "/configured/study/magi_auxiliary_identity.json", "sha256": "MANIFEST_FILE_SHA256",
                "content_manifest_sha256": "MAGI_T5_VAE_SOURCE_CONTENT_FINGERPRINT",
            },
            "catalogue": {
                "path": "/configured/official/VBench_full_info.json", "sha256": "SHA256_OF_THAT_FILE",
                "origin": "https://github.com/Vchitect/VBench/blob/COMMIT/vbench/VBench_full_info.json",
                "id_kind": "zero_based_index",
            },
            "prompt_ids": [0, 1, 2, 3], "seed": 0, "prefix_frames": 24,
            "continuation_frames": 96, "fps": 24,
            "diagnostic_svd_prompt_ids": [0],
            "magi": {**FIXED_MAGI, "arms": ["raw", "svd_hevc"]},
        },
        "notes": [
            "Example IDs/text are placeholders; the parent freezes four actual official catalogue rows.",
            "The case text must exactly equal its hash-bound official prompt_en field.",
            "All arms use the SAME attention-ready hot budget and owner CPU budget.",
            "hot_window_bytes may be 0 (old path) or explicitly set up to 24 GiB; positive budgets must fit all four histories.",
            "diagnostic_svd_prompt_ids is [] or only the FIRST cohort ID; SVD is excluded from four-case primary means.",
            "Primary bytes are native logical BF16 KV / unique decoder closure; snapshots and physical files are separate.",
            "Weight and auxiliary manifests are fully hashed once per study; per-case original stat guards fail closed.",
            "Optional SVD separates projection loss; fixed K256/V384 is NOT a quality guarantee.",
            "Native generation is source-bound local-SSD work; do not reuse old conditioning or relabel old videos.",
        ],
    }
    print(json.dumps(value, indent=2))
    return value


def _safe_id(value, name):
    if (not isinstance(value, str) or not value or len(value) > 120
            or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-" for character in value)):
        raise ValueError(f"{name} must be a bounded filename-safe identifier")
    return value


def validate_case(case, protocol):
    if case.get("schema") != CASE_SCHEMA or protocol.get("schema") != PROTOCOL_SCHEMA:
        raise ValueError("Unknown immutable VBench case/protocol schema")
    _safe_id(case["case_id"], "case_id")
    _safe_id(protocol["protocol_id"], "protocol_id")
    if case["protocol_id"] != protocol["protocol_id"]:
        raise ValueError("Case belongs to a different protocol")
    for key, expected in (("seed", 0), ("prefix_frames", 24), ("continuation_frames", 96), ("fps", 24)):
        if type(protocol.get(key)) is not int or protocol[key] != expected:
            raise ValueError(f"Only the fixed seed0/24-prefix/96-continuation/24fps protocol is supported: {key}")
    if type(case.get("seed")) is not int or case["seed"] != 0:
        raise ValueError("This cohort fixes the fresh CUDA noise seed to zero")
    ids = protocol["prompt_ids"]
    if not isinstance(ids, list) or len(ids) != 4 or any(type(i) is not int or i < 0 for i in ids) or len(set(ids)) != 4:
        raise ValueError("Exactly four distinct official zero-based prompt indices are required")
    diagnostic_ids = protocol.get("diagnostic_svd_prompt_ids", [])
    if diagnostic_ids not in ([], [ids[0]]) or any(type(i) is not int for i in diagnostic_ids):
        raise ValueError("Optional SVD diagnostics are restricted to the first official cohort case")
    if type(case.get("prompt_id")) is not int or case["prompt_id"] not in ids:
        raise ValueError("Case prompt is not a member of the frozen four-prompt cohort")
    catalogue = protocol["catalogue"]
    if catalogue["id_kind"] != "zero_based_index":
        raise ValueError("Prompt IDs must be explicit zero-based official full-info indices")
    if not catalogue["origin"].startswith((
        "https://github.com/Vchitect/VBench/", "https://raw.githubusercontent.com/Vchitect/VBench/",
    )):
        raise ValueError("Record the official Vchitect/VBench catalogue origin")
    if native._sha(catalogue["path"]) != catalogue["sha256"]:
        raise ValueError("Official prompt catalogue content changed")
    rows = native._read(catalogue["path"])
    if not isinstance(rows, list) or max(ids) >= len(rows):
        raise ValueError("Official prompt indices exceed the supplied catalogue")
    if len({rows[index]["prompt_en"] for index in ids}) != 4:
        raise ValueError("The four-case cohort must contain four distinct official prompt texts")
    row = rows[case["prompt_id"]]
    prompt = row["prompt_en"]
    if not isinstance(prompt, str) or not prompt.strip() or case["prompt"] != prompt:
        raise ValueError("Case text must exactly equal its official prompt_en row, without substitutions")
    profile = protocol["magi"]
    if set(profile) != {*FIXED_MAGI, "arms"}:
        raise ValueError("Unknown or missing MAGI profile fields must not be silently adopted or ignored")
    for name, expected in FIXED_MAGI.items():
        if name in ("hot_window_bytes", "hot_headroom_bytes", "resident_mib"):
            if type(profile.get(name)) is not int or profile[name] < 0:
                raise ValueError(f"Invalid explicitly fixed MAGI memory budget: {name}")
        elif profile.get(name) != expected:
            raise ValueError(f"Unsupported fixed MAGI profile: {name}")
    required = 4 * native.N * (native.C // 2) * native.LAYERS * 2 * 2
    cap = profile["hot_window_bytes"]
    if cap and not required <= cap <= 24 * GIB:
        raise ValueError("An enabled hot window must hold all four BF16 head-local histories and be <=24 GiB")
    if profile["resident_mib"] < 1 or profile["hot_headroom_bytes"] < 2 * GIB:
        raise ValueError("Require positive CPU residency and at least 2 GiB explicit CUDA headroom")
    arms = profile["arms"]
    if arms != ["raw", "svd_hevc"]:
        raise ValueError("Primary arms must be RAW/HEVC; select first-case SVD through diagnostic_svd_prompt_ids")
    return {"catalogue_row": row, "max_hot_required_bytes_per_rank": required}


def _case_arms(case, protocol):
    return (["raw", "svd", "svd_hevc"] if case["prompt_id"] in protocol.get("diagnostic_svd_prompt_ids", [])
            else ["raw", "svd_hevc"])


def _t5_inventory(directory):
    directory = Path(directory).resolve() / "t5-v1_1-xxl"
    small = ["config.json", "tokenizer_config.json", "special_tokens_map.json",
             "spiece.model", "pytorch_model.bin.index.json"]
    rows = {name: native._file_row(directory / name) for name in small}
    index = native._read(directory / "pytorch_model.bin.index.json")
    shards = []
    for name in sorted(set(index["weight_map"].values())):
        path = (directory / name).resolve()
        path.relative_to(directory)
        stat = path.stat()
        shards.append({"path": str(path), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    return {
        "directory": str(directory.parent), "files": rows, "shards": shards,
        "weight_verification": "per-case index/config/tokenizer SHA256 and shard sizes/mtime; full contents bound by study auxiliary identity",
    }


def _check_rows(rows):
    for row in rows:
        if Path(row["path"]).stat().st_size != row["bytes"] or native._sha(row["path"]) != row["sha256"]:
            raise ValueError(f"Bound file changed; prepare a new immutable run: {row['path']}")


def _storage_location(output):
    existing = Path(output).resolve()
    while not existing.exists():
        existing = existing.parent
    probe = subprocess.run(
        ["stat", "-f", "-c", "%T", str(existing)], check=True, capture_output=True, text=True, timeout=15,
    )
    return {"filesystem_type": probe.stdout.strip(), "device_id": existing.stat().st_dev}


def prepare(args):
    output = native._output(args.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Fresh benchmark preparation requires an empty NEW output")
    if not args.allow_lossy:
        raise ValueError("Explicit --allow-lossy is required; fixed K256/V384 has no quality guarantee")
    case, protocol = native._read(args.case), native._read(args.protocol)
    checked = validate_case(case, protocol)
    placement = _storage_location(output)
    if args.storage_placement == "local_ssd" and placement["filesystem_type"] in (
        "nfs", "nfs4", "cifs", "smb2", "fuse.sshfs",
    ):
        raise ValueError("The declared local-SSD preparation is on a network filesystem")
    if any(type(value) is not int or value < 1 for value in
           (args.cpu_threads, args.max_seconds, args.stage_seconds, args.store_gib, args.condition_seconds)):
        raise ValueError("Execution/thread/storage limits must be positive integers")
    if args.max_seconds > 6 * 3600 or args.stage_seconds > args.max_seconds or args.store_gib > 160:
        raise ValueError("This bounded case supports <=6h, a smaller per-stage limit and <=160GiB local store")
    source_project, native_source = args.source_project.resolve(), args.native_source.resolve()
    from .magi_provenance import verify_auxiliary, verify_weights

    if not {"weight_identity", "magi_auxiliary_identity"} <= set(protocol):
        raise ValueError("Fresh cases require the study's explicit full-weight and T5/VAE/source provenance bindings")
    study_weights = verify_weights(protocol["weight_identity"], args.weights)
    auxiliary = verify_auxiliary(
        protocol["magi_auxiliary_identity"], t5=args.t5, vae=args.vae, source=native_source,
    )
    reference = native._read(args.qualified_plan)
    identity = reference.pop("identity")
    if native._identity(reference) != identity:
        raise ValueError("Qualified builder reference plan identity changed")
    reference["identity"] = identity
    native._validate_config(reference["config"])
    external = {}
    for name, row in reference["external_sources"].items():
        if name.startswith("native/"):
            path = native_source / name.removeprefix("native/")
            if name == "native/special_tokens":
                path = native_source / "example/assets/special_tokens.npz"
            if native._sha(path) != row["sha256"]:
                raise ValueError(f"Qualified native source differs: {name}")
            external[name] = native._file_row(path)
    builder = source_project / "kv_compression/harness/neurico_magi_sweep.py"
    if native._sha(builder) != reference["builder"]["sha256"]:
        raise ValueError("Configured native builder differs from the proven builder")
    for name in ("native_magi_sp_bench.py", "neurico_magi_sweep.py", "adaptive_rank_magi.py"):
        reference_key = {
            "native_magi_sp_bench.py": "pilot/native",
            "neurico_magi_sweep.py": "pilot/capture_and_model_loader",
            "adaptive_rank_magi.py": "pilot/native_fixes",
        }[name]
        if native._sha(source_project / "kv_compression/harness" / name) != reference["external_sources"][reference_key]["sha256"]:
            raise ValueError(f"Qualified builder dependency differs: {name}")
        external[f"builder/{name}"] = native._file_row(source_project / "kv_compression/harness" / name)
    for path in sorted((native_source / "inference/model/t5").glob("*.py")):
        external[f"fresh_t5/{path.name}"] = native._file_row(path)
    checkpoint = native._weights(args.weights, reference["checkpoint_reference"])
    vae_path = args.vae.resolve()
    if native._sha(vae_path / "config.json") != reference["vae"]["config_sha256"]:
        raise ValueError("Native stateless VAE configuration differs from the qualified path")
    vae = {key: reference["vae"][key] for key in
           ("config_sha256", "expected_checkpoint_sha256", "mode", "temporal_tile_overlap_factor")}
    vae["path"] = str(vae_path)
    t5 = _t5_inventory(args.t5)
    import torch
    import transformers
    from ..codecs import hevc_capabilities

    if torch.cuda.is_initialized() or not torch.__version__.startswith("2.8.0") or transformers.__version__ != "4.42.3":
        raise RuntimeError("Prepare CPU-only in the existing Torch2.8 / transformers4.42.3 environment")
    capabilities = hevc_capabilities()
    if "yuv444p" not in capabilities["pixel_formats"]:
        raise RuntimeError("Configured HEVC binary must support yuv444p")
    encoders = subprocess.run(
        [capabilities["binary"], "-hide_banner", "-encoders"], check=True, capture_output=True, text=True, timeout=15,
    )
    if "libx264rgb" not in encoders.stdout:
        raise RuntimeError("The bound FFmpeg must support lossless RGB H.264 VBench output")
    for path in (native.PROJECT / "reelcache").rglob("*.py"):
        ast.parse(path.read_text(), filename=str(path))
    own = {str(path.relative_to(native.PROJECT)): native._file_row(path)
           for path in sorted((native.PROJECT / "reelcache").rglob("*.py"))}
    own["examples/vbench_magi.py"] = native._file_row(native.PROJECT / "examples/vbench_magi.py")
    config = json.loads(json.dumps(reference["config"]))
    config["runtime_config"].update(
        load=str(args.weights.resolve().parent), vae_pretrained=str(vae_path),
        t5_pretrained=str(args.t5.resolve()),
    )
    plan = {
        "schema": PLAN_SCHEMA, "case": case, "protocol": protocol,
        "case_file": native._file_row(args.case), "protocol_file": native._file_row(args.protocol),
        "catalogue_file": native._file_row(protocol["catalogue"]["path"]),
        "official_dimensions_for_prompt": checked["catalogue_row"].get("dimension", []),
        "qualified_builder_reference": native._file_row(args.qualified_plan),
        "reference_use": "builder/config/weights/VAE provenance ONLY; no saved prompt, embedding, prefix or noise",
        "fresh_prompt_inputs": True, "historical_replay_claim": False,
        "source_project": str(source_project), "native_source": str(native_source),
        "builder": native._file_row(builder), "external_sources": external, "reelcache_sources": own,
        "weights": str(args.weights.resolve()), "checkpoint_verification": checkpoint,
        "study_weight_identity": study_weights, "auxiliary_identity": auxiliary,
        "checkpoint_reference": reference["checkpoint_reference"], "t5": t5, "vae": vae,
        "config": config, "arms": _case_arms(case, protocol), "primary_arms": ["raw", "svd_hevc"],
        "diagnostic_arms": ["svd"] if "svd" in _case_arms(case, protocol) else [], "continuations": 4,
        "cpu_threads": args.cpu_threads, "resident_mib": protocol["magi"]["resident_mib"],
        "ffmpeg": native._file_row(capabilities["binary"]),
        "qualified_environment": {"python": sys.executable, "torch": torch.__version__,
                                  "transformers": transformers.__version__},
        "native_precision_contract": native._precision_preflight(protocol["magi"]["arms"]),
        "budgets": {
            "max_case_seconds": args.max_seconds, "max_stage_seconds": args.stage_seconds,
            "condition_seconds": args.condition_seconds, "max_store_bytes": args.store_gib * GIB,
            "max_hot_required_bytes_per_rank": checked["max_hot_required_bytes_per_rank"],
            "staging_reserve_bytes": 2 * native.N * native.C * native.LAYERS * (4 + 2),
            "disk_headroom_bytes": 8 * GIB,
        },
        "storage": {
            "declared_placement": args.storage_placement, "output": str(output),
            **placement,
            "copy_full_store_per_case": False,
        },
        "qualification": {
            "native_raw_first_chunk_exact": "required in every case",
            "compressed_cache_hit_bridge": "first native denoise epoch: two real hit tensors compared bitwise with independent packet redecodes",
            "cold_vs_hot_full_video_parity": "not claimed by the tensor bridge",
            "same_representation_only": True,
        },
        "limitations": [
            "Four official prompts x seed0 is a fixed subset, not the official complete VBench aggregate.",
            "All arms use fresh common T5/prefix/noise and each arm's own closed-loop recaptured histories.",
            "K256/V384 is experimental; poor prior SVD quality is not repaired or hidden by caching.",
            "SVD+HEVC uses FP32 normalized pre-RoPE K, BF16 V, global C1024, local four-KV-head restoration.",
            "Hot values are exact decoded native BF16 attention-ready tensors, never RAW originals or coefficients.",
            "No visibility filtering, async/factor-wire transport, full-KV replication claim or performance guarantee.",
        ],
    }
    plan["identity"] = native._identity(plan)
    native._write(output / "plan.json", plan)
    native._write(output / "config.json", config)
    (output / "build_cache" / "tmpdir").mkdir(parents=True, exist_ok=True)
    env = ["env", f"TMPDIR={output / 'build_cache/tmpdir'}", "PYTHONDONTWRITEBYTECODE=1",
           f"OMP_NUM_THREADS={args.cpu_threads}", f"MKL_NUM_THREADS={args.cpu_threads}",
           f"OPENBLAS_NUM_THREADS={args.cpu_threads}"]
    script = str(native.PROJECT / "examples/vbench_magi.py")
    prefix = f"cd {shlex.quote(str(native.PROJECT))} && "
    commands = {
        "condition_CPU_model_load_requires_approval": prefix + shlex.join([
            *env, "CUDA_VISIBLE_DEVICES=", "timeout", "--signal=TERM", "--kill-after=60s",
            str(args.condition_seconds), sys.executable, "-B", script, "condition",
            "--output", str(output), "--cpu-model-grant",
        ]),
        "run_REQUIRES_BOTH_GPU_GRANT": prefix + shlex.join([
            *env, "CUDA_VISIBLE_DEVICES=0,1", "timeout", "--signal=TERM", "--kill-after=60s",
            str(args.max_seconds), sys.executable, "-B", "-m", "torch.distributed.run",
            "--standalone", "--nproc_per_node=2", script, "run", "--output", str(output), "--gpu-grant",
        ]),
        "decode_REQUIRES_SEPARATE_GPU0_GRANT_AFTER_SP2_EXIT": prefix + shlex.join([
            *env, "CUDA_VISIBLE_DEVICES=0", sys.executable, "-B", script, "decode",
            "--output", str(output), "--gpu-grant", "--device", "0",
        ]),
        "report_CPU": prefix + shlex.join([
            *env, "CUDA_VISIBLE_DEVICES=", sys.executable, "-B", script, "report", "--output", str(output),
        ]),
    }
    native._write(output / "commands.json", commands)
    print(json.dumps({"status": "CPU_prepared_NOT_model_or_GPU_authorized", "identity": plan["identity"],
                      "commands": commands}))
    return plan


def verify_plan(output):
    plan = native._read(output / "plan.json")
    identity = plan.pop("identity")
    if native._identity(plan) != identity or plan["schema"] != PLAN_SCHEMA:
        raise ValueError("Immutable fresh benchmark plan changed")
    plan["identity"] = identity
    if str(output.resolve()) != plan["storage"]["output"] or _storage_location(output) != {
        key: plan["storage"][key] for key in ("filesystem_type", "device_id")
    }:
        raise ValueError("Run storage location changed; prepare anew at the explicitly selected placement")
    _check_rows([
        plan["case_file"], plan["protocol_file"], plan["catalogue_file"], plan["qualified_builder_reference"],
        plan["builder"], plan["ffmpeg"], *plan["external_sources"].values(), *plan["reelcache_sources"].values(),
    ])
    if native._read(output / "config.json") != plan["config"]:
        raise ValueError("Native execution config differs from the prepared protocol")
    validate_case(plan["case"], plan["protocol"])
    from .magi_provenance import verify_auxiliary, verify_weights

    if verify_weights(plan["protocol"]["weight_identity"], plan["weights"]) != plan["study_weight_identity"]:
        raise ValueError("Shared study full-weight provenance changed")
    if verify_auxiliary(
        plan["protocol"]["magi_auxiliary_identity"], t5=plan["t5"]["directory"],
        vae=plan["vae"]["path"], source=plan["native_source"],
    ) != plan["auxiliary_identity"]:
        raise ValueError("Study T5/VAE/source provenance changed")
    native._weights(plan["weights"], plan["checkpoint_reference"])
    if _t5_inventory(plan["t5"]["directory"]) != plan["t5"]:
        raise ValueError("Native T5 checkpoint/tokenizer inventory changed")
    return plan


def auxiliary_identity(args):
    from .magi_provenance import capture_auxiliary

    with _deadline(args.max_seconds):
        result = capture_auxiliary(
            output=args.output, t5=args.t5, vae=args.vae, source=args.native_source,
            qualified_plan=args.qualified_plan, max_seconds=args.max_seconds,
        )
    print(json.dumps({
        "status": result["status"], "output": str(args.output),
        "content_manifest_sha256": result["content_manifest_sha256"],
        "full_content_bytes": result["hashing"]["bytes"], "model_or_GPU_loaded": False,
    }))


@contextmanager
def _deadline(seconds):
    def expired(*args):
        raise TimeoutError("Explicit native benchmark time budget exhausted")
    previous = signal.signal(signal.SIGALRM, expired)
    prior = signal.setitimer(signal.ITIMER_REAL, seconds)
    started = time.monotonic()
    try:
        yield
    finally:
        remaining = max(0.000001, prior[0] - (time.monotonic() - started)) if prior[0] else 0
        signal.setitimer(signal.ITIMER_REAL, remaining, prior[1])
        signal.signal(signal.SIGALRM, previous)


def _check_text(text, case):
    import torch

    if (text["prompt"] != case["prompt"] or text["case_id"] != case["case_id"]
            or tuple(text["caption"].shape) != (1, 800, 4096)
            or text["caption"].dtype != torch.float32
            or tuple(text["mask"].shape) != (1, 800)
            or not torch.isfinite(text["caption"]).all()
            or not torch.all((text["mask"] == 0) | (text["mask"] == 1))):
        raise ValueError("Fresh native T5 conditioning does not match its official case")


def condition(args):
    if not args.cpu_model_grant:
        raise RuntimeError("Native FP32 T5 loading requires explicit --cpu-model-grant")
    output = native._output(args.output)
    plan = verify_plan(output)
    if (output / "conditioning.pt").exists():
        raise FileExistsError("Never replace a prepared case's fresh conditioning")
    native._environment(output, plan["cpu_threads"])
    import torch
    import transformers

    if torch.cuda.is_initialized() or os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("Run CPU conditioning with CUDA_VISIBLE_DEVICES= and no initialized CUDA")
    if (torch.__version__ != plan["qualified_environment"]["torch"]
            or transformers.__version__ != plan["qualified_environment"]["transformers"]):
        raise RuntimeError("Conditioning environment differs from the qualified preparation")
    torch.set_num_threads(plan["cpu_threads"])
    sys.path.insert(0, plan["native_source"])
    from inference.model.t5.t5_model import T5Embedder

    if native._file_row(sys.modules[T5Embedder.__module__].__file__) != plan["external_sources"]["fresh_t5/t5_model.py"]:
        raise RuntimeError("An unbound pre-imported native T5 source shadows the prepared conditioning code")
    started = time.perf_counter()
    with _deadline(plan["budgets"]["condition_seconds"]), torch.inference_mode():
        embedder = T5Embedder(
            device="cpu", local_cache=True, cache_dir=plan["t5"]["directory"],
            torch_dtype=torch.float32, model_max_length=800,
        )
        caption, mask = embedder.get_text_embeddings([plan["case"]["prompt"]])
        text = {"caption": caption.float().cpu(), "mask": mask.cpu(),
                "prompt": plan["case"]["prompt"], "case_id": plan["case"]["case_id"]}
        _check_text(text, plan["case"])
        with (output / "conditioning.pt").open("xb") as handle:
            torch.save(text, handle)
        del embedder
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU T5 unexpectedly initialized CUDA")
    native._write(output / "conditioning.json", {
        "plan_identity": plan["identity"], "case": plan["case"],
        "artifact": native._file_row(output / "conditioning.pt"),
        "caption_tensor_sha256": native._tensor_hash(text["caption"]),
        "mask_tensor_sha256": native._tensor_hash(text["mask"]),
        "native_T5_dtype": "float32", "native_preprocessing": True,
        "special_tokens": [], "cuda_initialized": False, "seconds": time.perf_counter() - started,
        "t5_weight_verification": plan["auxiliary_identity"],
    })


def _conditioning(plan, output):
    import torch

    row = native._read(output / "conditioning.json")
    if row["plan_identity"] != plan["identity"] or row["case"] != plan["case"]:
        raise ValueError("Conditioning belongs to another fresh case or source plan")
    _check_rows([row["artifact"]])
    text = torch.load(row["artifact"]["path"], map_location="cpu", weights_only=True)
    _check_text(text, plan["case"])
    for key in ("caption", "mask"):
        if native._tensor_hash(text[key]) != row[f"{key}_tensor_sha256"]:
            raise ValueError("Fresh T5 tensor identity changed")
    return text, row


class _FreshSession(native._Session):
    """Shared proven native calls, but fresh conditioning/noise/prefix construction."""

    def __init__(self, plan, output):
        import torch
        import torch.distributed as dist
        from ..distributed import DistributedOwner

        self.torch, self.dist, self.plan, self.output = torch, dist, plan, output
        text, self.conditioning_record = _conditioning(plan, output)
        sys.path[:0] = [plan["native_source"], str(Path(plan["source_project"]) / "kv_compression/harness"),
                       plan["source_project"]]
        os.environ["SPECIAL_TOKEN_PATH"] = str(Path(plan["native_source"]) / "example/assets/special_tokens.npz")
        driver = importlib.import_module("native_magi_sp_bench")
        loader = importlib.import_module("neurico_magi_sweep")
        if Path(loader.__file__).resolve() != Path(plan["builder"]["path"]):
            raise RuntimeError("An unbound pre-imported native builder shadows the qualified source")
        driver.SOURCE, driver.WEIGHTS, driver.OUT = Path(plan["native_source"]), Path(plan["weights"]), output
        from inference.common import MagiConfig, InferenceParams
        from inference.infra.distributed.dist_utils import dist_init
        from inference.pipeline.video_generate import init_intervel, init_t
        from inference.model.dit.dit_module import flash_apply_rotary_emb

        self.config = MagiConfig.from_json(str(output / "config.json"))
        dist_init(self.config)
        self.device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
        self.rank = dist.get_rank()
        torch.set_num_threads(plan["cpu_threads"])
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.manual_seed(plan["case"]["seed"])
        torch.cuda.manual_seed_all(plan["case"]["seed"])
        self.storage_group = dist.new_group(backend="gloo", timeout=timedelta(
            seconds=plan["budgets"]["max_stage_seconds"] + 60,
        ))
        self.transport = DistributedOwner(self.storage_group, expected_world_size=2)
        started = time.perf_counter()
        self.model = loader.load_model(self.config, self.device)
        torch.cuda.synchronize(self.device)
        self.model_load_seconds = time.perf_counter() - started
        self.executed_model_source = native._file_row(sys.modules[self.model.__class__.__module__].__file__)
        if self.executed_model_source != plan["external_sources"]["native/inference/model/dit/dit_model.py"]:
            raise RuntimeError("Executed native model differs from the bound source")
        self.rotary, self.fallback = flash_apply_rotary_emb, ExitStack()
        self.blackwell_silu = torch.cuda.get_device_capability(self.device)[0] >= 12
        if self.blackwell_silu:
            import flashinfer
            import torch.nn.functional as functional

            def silu(value, out=None, enable_pdl=None):
                gate, up = value.chunk(2, dim=-1)
                result = functional.silu(gate) * up
                if out is not None:
                    out.copy_(result)
                    return out
                return result
            self.fallback.enter_context(patch.object(flashinfer.activation, "silu_and_mul", silu))
        self.cache = InferenceParams(max_batch_size=1, max_sequence_length=5 * native.N)
        caption, mask = text["caption"].to(self.device), text["mask"].to(self.device)
        self.null_caption = self.model.y_embedder.null_caption_embedding.detach().unsqueeze(0)
        self.null_mask = torch.zeros_like(mask)
        self.null_mask[:, :50] = 1
        self.y = torch.cat((caption, self.null_caption)).unsqueeze(1)
        self.masks = torch.cat((mask, self.null_mask)).unsqueeze(1)
        self.generator = torch.Generator(device=self.device).manual_seed(plan["case"]["seed"])
        self.prefix_noise = torch.randn(native.SHAPE, generator=self.generator, device=self.device)
        self.prefix_noise_sha256 = native._tensor_hash(self.prefix_noise)
        self.noises = [torch.randn(native.SHAPE, generator=self.generator, device=self.device) for _ in range(4)]
        self.schedule = init_t({}, 16, self.device, shortcut_mode=self.config.engine_config.shortcut_mode)
        self.intervals = init_intervel(16, self.device, shortcut_mode=self.config.engine_config.shortcut_mode)
        self.near = [value > 0.3 for value in self.schedule[:-1].cpu().tolist()]
        if [i for i, near in enumerate(self.near) if near] != [12, 13, 14, 15]:
            raise RuntimeError("Native distilled schedule changed")
        hashes = self.transport.checkpoint("fresh_inputs", details={
            "caption": native._tensor_hash(self.y), "mask": native._tensor_hash(self.masks),
            "noise": [self.prefix_noise_sha256, *[native._tensor_hash(value) for value in self.noises]],
        })
        if any(value != hashes[0] for value in hashes):
            raise RuntimeError("SP2 ranks disagree on fresh conditioning or noise")
        self.started = time.perf_counter()

    def timed(self, label, operation):
        remaining = self.plan["budgets"]["max_case_seconds"] - (time.perf_counter() - self.started)
        error = "Case deadline exhausted" if remaining <= 0 else None
        self.transport.checkpoint(f"budget:{label}", error=error)
        with _deadline(min(remaining, self.plan["budgets"]["max_stage_seconds"])):
            return super().timed(label, operation)

    def state(self, latent):
        state = {
            "metadata": {
                "conditioning_identity": self.conditioning_record["caption_tensor_sha256"],
                "execution_config": {
                    "config": self.plan["config"], "schedule": self.schedule.cpu().tolist(),
                    "intervals": self.intervals.cpu().tolist(), "near_clean_steps": [12, 13, 14, 15],
                    "case": self.plan["case"], "protocol_identity": native._identity(self.plan["protocol"]),
                },
                "vae_contract": self.plan["vae"],
            },
            "tensors": {
                "latent": latent, "conditioning": self.y, "conditioning_mask": self.masks,
                "null_caption": self.null_caption, "null_mask": self.null_mask,
                **{f"noise/{i + 1}": noise for i, noise in enumerate(self.noises)},
            },
            "generators": {"noise": self.generator},
        }
        return state


def _closure(runtime, roots):
    pending, seen, sizes, ledger = list(roots), set(), {"blob": 0, "basis": 0, "manifest": 0}, []
    while pending:
        object_id = pending.pop()
        if object_id in seen:
            continue
        seen.add(object_id)
        kind = object_id.split(":", 1)[0]
        path = runtime.store.path_for(object_id)
        stat = path.stat()
        sizes[kind] += stat.st_size
        ledger.append({
            "object_id": object_id, "kind": kind, "relative_path": str(path.relative_to(runtime.store.root)),
            "sha256": object_id.split(":", 1)[1], "bytes": stat.st_size,
            "allocated_file_bytes": stat.st_blocks * 512,
        })
        if kind == "manifest":
            pending.extend(runtime.store.references(object_id))
    return {
        "roots": roots, "unique_bytes": sum(sizes.values()), "byte_breakdown": sizes,
        "unique_objects": len(seen), "ledger": sorted(ledger, key=lambda row: row["object_id"]),
    }


def _physical_store(runtime):
    rows = []
    seen = set()
    for path in sorted(runtime.store.root.rglob("*")):
        if not path.is_file():
            continue
        stat = path.stat()
        inode = (stat.st_dev, stat.st_ino)
        if inode in seen:
            continue
        seen.add(inode)
        rows.append({
            "path": str(path.relative_to(runtime.store.root)), "bytes": stat.st_size,
            "allocated_file_bytes": stat.st_blocks * 512,
            "index_or_control_file": path.relative_to(runtime.store.root).parts[0] != "objects",
        })
    return {
        "logical_file_bytes": sum(row["bytes"] for row in rows),
        "allocated_file_bytes": sum(row["allocated_file_bytes"] for row in rows),
        "index_and_control_bytes": sum(row["bytes"] for row in rows if row["index_or_control_file"]),
        "index_and_control_allocated_bytes": sum(row["allocated_file_bytes"] for row in rows if row["index_or_control_file"]),
        "file_count_unique_inodes": len(rows),
        "scope": "whole shared owner store including other arms; st_size and st_blocks*512, not filesystem metadata/replication",
        "index_and_control_files": [row for row in rows if row["index_or_control_file"]],
    }


def _storage_audit(runtime, branch, snapshot):
    """Separate decoder, resumable-state and physical-store closures."""
    info = runtime.branch_info(branch)
    decoder = _closure(runtime, [info["head_id"]])
    resume = _closure(runtime, [snapshot])
    snapshot_manifest = runtime.store.get_manifest(snapshot)
    if snapshot_manifest["head"] != info["head_id"]:
        raise ValueError("The selected resumable snapshot and measured decoder history must match")
    matrices = []
    for manifest in info["chunk_manifests"]:
        chunk = runtime.store.get_manifest(manifest)
        for entry in chunk["matrices"]:
            packet = runtime.store.get_manifest(entry["tensor"])
            matrices.append({
                "chunk_id": chunk["chunk_id"], "key": entry["key"], "context": packet["context"],
                "spec": packet["spec"], "basis": packet["encoded"]["basis"], "metrics": packet["encoded"]["metrics"],
            })
    native_bytes = info["token_end"] * runtime.model.layers * (
        runtime.model.channels_for("K") + runtime.model.channels_for("V")
    ) * 2
    return {
        "branch": branch, "head_id": info["head_id"], "snapshot_id": snapshot,
        "history_chunks": len(info["chunk_manifests"]),
        "decoder_closure": decoder, "resumable_snapshot_closure": resume,
        "native_history_bytes_across_SP2": native_bytes,
        "primary_ratio_native_BF16_to_decoder_closure": native_bytes / decoder["unique_bytes"],
        "secondary_ratio_native_BF16_to_resumable_snapshot": native_bytes / resume["unique_bytes"],
        "numerator": "native BF16 K+V already head-sharded across SP2, NOT FP32 preK factor inputs",
        "primary_denominator": "unique payloads, all stored basis bytes and required head/chunk/tensor manifests",
        "decoder_boundary": "decoded canonical matrices in their recorded key spaces; native execution/RoPE/RNG/conditioning state is in the separate resumable snapshot",
        "size_method": "actual unique object file sizes; payload integrity validated on publication/acquisition",
        "physical_store": _physical_store(runtime), "whole_store": runtime.store.stats(), "packets": matrices,
    }


def _export_checkpoint(runtime, snapshot, output, arm, plan):
    import torch

    saved = runtime.load_snapshot(snapshot)
    path = output / "checkpoints" / f"{arm}_before_chunk4.pt"
    path.parent.mkdir(exist_ok=True)
    with path.open("xb") as handle:
        torch.save({
            "snapshot_id": snapshot, "metadata": dict(saved.metadata), "tensors": dict(saved.tensors),
            "plan_identity": plan["identity"], "case": plan["case"],
        }, handle)
    return {
        "artifact": native._file_row(path), "snapshot_id": snapshot, "boundary": "completed clean chunk3",
        "role": "selected execution state, including latent/conditioning/RNG/RoPE; canonical decoder objects listed separately",
        "standalone_full_resume": False,
    }


def _storage_summary(measured):
    result = {key: value for key, value in measured.items() if key != "packets"}
    for name in ("decoder_closure", "resumable_snapshot_closure"):
        result[name] = {key: value for key, value in measured[name].items() if key != "ledger"}
    return result


def run(args):
    if not args.gpu_grant:
        raise RuntimeError("Native MAGI generation requires the parent's explicit BOTH-GPU grant")
    if int(os.environ.get("WORLD_SIZE", "0")) != 2:
        raise RuntimeError("Use torchrun with exactly two native SP workers")
    output = native._output(args.output)
    plan = verify_plan(output)
    if (output / f"execution_rank{os.environ['RANK']}.json").exists():
        raise FileExistsError("Never rerun or relabel a used native output; prepare a new case attempt")
    native._environment(output, plan["cpu_threads"])
    import torch
    import transformers
    from .magi import MagiAdapter
    from ..codecs import using_ffmpeg
    from ..manager import KVRuntime
    from ..types import ModelIdentity

    if (torch.__version__ != plan["qualified_environment"]["torch"]
            or transformers.__version__ != plan["qualified_environment"]["transformers"]):
        raise RuntimeError("Native runtime environment differs from preparation")
    if plan["storage"]["declared_placement"] != "local_ssd":
        raise RuntimeError("This bounded cohort requires a NEW parent-approved local-SSD preparation")
    required_disk = (plan["budgets"]["max_store_bytes"] + plan["budgets"]["staging_reserve_bytes"]
                     + plan["budgets"]["disk_headroom_bytes"])
    if shutil.disk_usage(output).free < required_disk:
        raise RuntimeError("Local case store plus explicit disk headroom does not fit")
    native._write(output / f"execution_rank{os.environ['RANK']}.json", native._execution(plan))
    session, runtime, active = None, None, None
    try:
        with _deadline(plan["budgets"]["max_case_seconds"]), torch.inference_mode(), using_ffmpeg(plan["ffmpeg"]["path"]):
            session = _FreshSession(plan, output)
            identity = ModelIdentity(
                "MAGI-1-24B-distill", native._identity({
                    "weights": plan["study_weight_identity"]["magi_model_content_sha256"],
                    "native_sources": {name: row["sha256"] for name, row in plan["external_sources"].items()},
                    "config": plan["config"],
                }), native.LAYERS, native.C, "tokens_global_channels_ulysses_sp2", "bfloat16",
            )
            records, audits, memory, gates, checkpoints = {}, {}, {}, {}, {}
            prefix = session.timed("fresh_common_prefix", lambda: session.denoise(session.prefix_noise, 0))
            records["common_prefix"] = session.save("common_prefix", prefix)
            del session.prefix_noise
            session.timed("native_clean_prefix", lambda: session.clean(prefix, 0))
            reference = session.timed("native_raw_first", lambda: session.denoise(session.noises[0], 1))
            records["native_raw_first"] = session.save("native_raw_first", reference)
            dense_before = session.transport.checkpoint("native_release_before", details=native._memory(session.device, session.cache))
            session.cache.key_value_memory_dict.clear()
            gc.collect()
            torch.cuda.empty_cache()
            dense_after = session.transport.checkpoint("native_release_after", details=native._memory(session.device, session.cache))
            runtime = session.transport.owner_call("open_store", lambda: KVRuntime(
                output / "stores/native", identity,
                max_resident_bytes=plan["resident_mib"] * 1024**2,
                max_store_bytes=plan["budgets"]["max_store_bytes"],
            ))
            profile = plan["protocol"]["magi"]
            for arm in plan["arms"]:
                def adapter(branch, verify_hits=0):
                    return MagiAdapter(
                        session.model, session.cache, runtime, model_identity=identity, branch_id=branch,
                        staging_dir=output / "staging", coordinator=session.transport,
                        specs=native._arm_specs(arm), reuse_threshold=None,
                        rotary_apply=session.rotary, device=session.device, native_dtype="bfloat16",
                        hot_window_bytes=profile["hot_window_bytes"],
                        hot_window_headroom_bytes=profile["hot_headroom_bytes"],
                        hot_window_require_full=True, hot_window_verify_hits=verify_hits,
                    )
                active = adapter(arm)
                active.attach()
                session.timed(f"{arm}_clean_prefix", lambda: session.clean(prefix, 0, active))
                session.timed(f"{arm}_publish_prefix", active.seal_chunk)
                snapshot = session.timed(f"{arm}_snapshot", lambda: active.snapshot(**session.state(prefix)))
                active.detach()
                active = adapter(f"{arm}_resume", verify_hits=2 if arm == "svd_hevc" else 0)
                active.attach()
                restored = session.timed(f"{arm}_restore", lambda: active.restore(
                    snapshot, f"{arm}_resume", generators={"noise": session.generator},
                ))
                session.reinstate(restored)
                del restored
                for index in range(1, 5):
                    latent = session.timed(f"{arm}_chunk{index}", lambda: session.denoise(
                        session.noises[index - 1], index, active,
                    ))
                    records[f"{arm}_chunk{index}"] = session.save(
                        f"{arm}_chunk{index}", latent, reference=reference if index == 1 else None,
                    )
                    if arm == "raw" and index == 1:
                        if not torch.equal(latent, reference):
                            raise RuntimeError("Fresh RAW restored/hot continuation differs from the native reference")
                        gates["fresh_raw_native_first_exact"] = True
                    if arm == "svd_hevc" and index == 1 and profile["hot_window_bytes"]:
                        stats = active.memory_report()["decoded_hot_window"]
                        if stats["verified_hits"] < 2 or stats["hits"] <= 0:
                            raise RuntimeError("The real compressed native hot-hit parity bridge did not execute")
                        gates["compressed_hit_tensor_parity"] = {
                            "verified_hits_per_rank": stats["verified_hits"], "kind": "same decoded packet",
                            "full_video_cold_vs_hot_parity_claimed": False,
                        }
                        active.hot_window_verify_hits = 0
                    rank_memory = session.transport.checkpoint(f"{arm}_chunk{index}_memory", details=active.memory_report())
                    memory[f"{arm}_chunk{index}"] = rank_memory
                    if any(row["live_native_cache_storage_bytes"] or
                           row["decoded_hot_window"]["resident_bytes"] for row in rank_memory):
                        raise RuntimeError("Native or hot allocations survived their completed denoise epoch")
                    if index < 4:
                        session.timed(f"{arm}_clean_chunk{index}", lambda: session.clean(latent, index, active))
                        session.timed(f"{arm}_publish_chunk{index}", active.seal_chunk)
                        if index == 3:
                            snapshot = session.timed(f"{arm}_later_snapshot", lambda: active.snapshot(**session.state(latent)))
                            session.reinstate(session.timed(
                                f"{arm}_later_restore", lambda: active.restore(
                                    snapshot, f"{arm}_later_resume", generators={"noise": session.generator},
                                ),
                            ))
                    del latent
                session.timed(f"{arm}_final_snapshot_hash_verification", lambda: session.transport.owner_call(
                    f"{arm}_verify_selected_closure", lambda: runtime.store.verify(snapshot),
                ))
                measured = session.transport.owner_call(f"{arm}_storage_audit", lambda: _storage_audit(
                    runtime, active.branch_id, snapshot,
                ))
                session.transport.owner_call(f"{arm}_storage_record", lambda: native._write(output / f"{arm}_storage.json", measured))
                summary = _storage_summary(measured) if session.rank == 0 else None
                audits[arm] = session.transport.broadcast_json(summary, operation=f"{arm}_storage_summary")
                selected = session.transport.owner_call(f"{arm}_export_selected_checkpoint", lambda: _export_checkpoint(
                    runtime, snapshot, output, arm, plan,
                ))
                checkpoints[arm] = session.transport.broadcast_json(selected, operation=f"{arm}_checkpoint_export")
                gates[f"{arm}_final_snapshot_closure_hash_verified"] = True
                gates[f"{arm}_later_clean_restore_executed"] = True
                active.detach()
                active = None
                gc.collect()
                torch.cuda.empty_cache()
            result = {
                "schema": PLAN_SCHEMA, "status": "native_complete_RGB_pending",
                "plan_identity": plan["identity"], "case": plan["case"], "gates": gates,
                "latents": records, "profile_storage": audits, "profile_memory": memory,
                "selected_checkpoints": checkpoints, "primary_arms": plan["primary_arms"],
                "diagnostic_arms": plan["diagnostic_arms"],
                "native_dense_release": {"before": dense_before, "after": dense_after},
                "fresh_noise_hashes": [session.prefix_noise_sha256, *[native._tensor_hash(noise) for noise in session.noises]],
                "conditioning": session.conditioning_record, "blackwell_silu_fallback": session.blackwell_silu,
                "executed_model_source": session.executed_model_source,
                "mode": "serial_clean_history", "generated_frames": 120, "evaluated_frames": 96,
                "storage_placement": plan["storage"], "final_chunk_is_noisy_complete_not_snapshotted": True,
                "study_weight_identity": plan["study_weight_identity"], "auxiliary_identity": plan["auxiliary_identity"],
                "worker_execution_records": [native._file_row(output / f"execution_rank{rank}.json") for rank in range(2)],
                "quality_guarantee": None, "limitations": plan["limitations"],
            }
            session.transport.owner_call("fresh_native_complete", lambda: native._write(output / "native_complete.json", result))
    finally:
        if active is not None and not active.phase.startswith("active_"):
            active.detach()
        if runtime is not None:
            runtime.close()
        if session is not None:
            session.fallback.close()
            if session.dist.is_initialized():
                session.dist.destroy_process_group()


class _VideoWriter:
    """Lossless RGB H.264 avoids an unaccounted RGB-to-YUV quality confound."""

    def __init__(self, binary, path, *, width=1920, height=1080, frames=96):
        self.path, self.frames = Path(path), frames
        if self.path.exists():
            raise FileExistsError("Do not replace a completed or partial benchmark video")
        self.shape = (height, width, 3)
        self.count = 0
        self.digest = hashlib.sha256()
        self.command = [
            binary, "-hide_banner", "-loglevel", "error", "-n",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", "24",
            "-i", "pipe:0", "-an", "-c:v", "libx264rgb", "-qp", "0", "-preset", "ultrafast",
            "-threads", "1", "-pix_fmt", "rgb24", "-movflags", "+faststart", str(path),
        ]
        self.process = subprocess.Popen(self.command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    def write(self, frames):
        import numpy as np

        if frames.dtype != np.uint8 or tuple(frames.shape[1:]) != self.shape or self.count + len(frames) > self.frames:
            raise ValueError("Unexpected RGB frames for the fixed video contract")
        raw = memoryview(np.ascontiguousarray(frames)).cast("B")
        self.digest.update(raw)
        self.process.stdin.write(raw)
        self.count += len(frames)

    def close(self):
        if self.process.stdin is not None:
            self.process.stdin.close()
            self.process.stdin = None
        _, error = self.process.communicate(timeout=120)
        if self.process.returncode or self.count != self.frames:
            raise RuntimeError(f"Lossless RGB video encoding failed or frame count changed: {error.decode(errors='replace')[-2000:]}")
        return {"artifact": native._file_row(self.path), "frames": self.count, "fps": 24,
                "shape": [self.frames, *self.shape], "rgb_sha256": self.digest.hexdigest(),
                "encoder": "libx264rgb QP0 ultrafast, lossless RGB", "command": self.command}

    def abort(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        for handle in (self.process.stdin, self.process.stderr):
            if handle is not None:
                handle.close()


def _verify_video_rgb(binary, row):
    _check_rows([row["artifact"]])
    process = subprocess.Popen([
        binary, "-hide_banner", "-loglevel", "error", "-i", row["artifact"]["path"],
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-threads", "1", "pipe:1",
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    digest, count = hashlib.sha256(), 0
    try:
        for data in iter(lambda: process.stdout.read(4 * 1024**2), b""):
            digest.update(data)
            count += len(data)
        error = process.stderr.read()
        process.wait(timeout=120)
        expected_bytes = 1
        for size in row["shape"]:
            expected_bytes *= size
        if process.returncode or count != expected_bytes or digest.hexdigest() != row["rgb_sha256"]:
            raise RuntimeError(f"Encoded VBench video is not exact native RGB: {error.decode(errors='replace')[-1000:]}")
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        process.stdout.close()
        process.stderr.close()


def decode(args):
    if not args.gpu_grant or int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise RuntimeError("Native VAE requires a separate one-GPU grant after BOTH SP2 workers exit")
    output = native._output(args.output)
    plan = verify_plan(output)
    result = native._read(output / "native_complete.json")
    if result["plan_identity"] != plan["identity"]:
        raise ValueError("Fresh native latents belong to another plan")
    native._require_workers_exited(result)
    if (output / "execution_decode.json").exists():
        raise FileExistsError("Do not relabel/restart a partially used decode directory")
    native._environment(output, plan["cpu_threads"])
    sys.path.insert(0, plan["native_source"])
    os.environ["SPECIAL_TOKEN_PATH"] = str(Path(plan["native_source"]) / "example/assets/special_tokens.npz")
    import numpy as np
    import torch
    from safetensors.torch import load_file
    from inference.model.vae.vae_model import ViTVAE
    from inference.pipeline.video_process import VaeHelper

    weight = Path(plan["vae"]["path"]) / "diffusion_pytorch_model.safetensors"
    vae_evidence = next(
        row for row in plan["auxiliary_identity"]["components"]["vae"]
        if row["relative_path"] == "diffusion_pytorch_model.safetensors"
    )
    if vae_evidence["sha256"] != plan["vae"]["expected_checkpoint_sha256"]:
        raise ValueError("Full VAE study provenance differs from the qualified VAE")
    config_path = Path(plan["vae"]["path"]) / "config.json"
    if native._sha(config_path) != plan["vae"]["config_sha256"]:
        raise ValueError("Qualified native VAE config changed")
    config = native._read(config_path)
    original_linspace = torch.linspace

    def cpu_linspace(*args, **kwargs):
        return original_linspace(*args, **{**kwargs, "device": "cpu"})
    with patch.object(torch, "linspace", cpu_linspace), torch.device("meta"):
        vae = ViTVAE(ddconfig=config["ddconfig"], model_type=config["model_type"])
    vae.load_state_dict(load_file(str(weight), device="cpu"), strict=True, assign=True)
    device = torch.device("cuda", args.device)
    torch.cuda.set_device(device)
    torch.set_num_threads(plan["cpu_threads"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.nn.Module.to(vae.eval(), device=device, dtype=torch.bfloat16)
    native._write(output / "execution_decode.json", native._execution(plan))
    (output / "rgb").mkdir(exist_ok=True)
    (output / "videos").mkdir(exist_ok=True)
    media, videos = {}, {}

    def rgb(name):
        record = result["latents"][name]
        _check_rows([record["artifact"]])
        latent = torch.load(record["artifact"]["path"], map_location=device, weights_only=True)["latent"]
        if native._tensor_hash(latent) != record["tensor_sha256"]:
            raise ValueError("Saved fresh latent tensor hash changed")
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        with torch.inference_mode(), patch.object(torch.distributed, "get_rank", return_value=0):
            value = VaeHelper.decode(
                latent / 0.18215, vae, tile_sample_min_height=256, tile_sample_min_width=256,
                spatial_tile_overlap_factor=0.25, temporal_tile_overlap_factor=0,
                tile_sample_min_length=12, allow_spatial_tiling=True,
                uint8_output=False, parallel_group=None,
            )
        torch.cuda.synchronize(device)
        if tuple(value.shape) != (24, 3, 1088, 1920) or not torch.isfinite(value).all():
            raise RuntimeError("Qualified native VAE geometry/finiteness changed")
        pixels = (value.float() * 127.5 + 127.5).clamp(0, 255).to(torch.uint8)
        frames = pixels[:, :, 4:1084].permute(0, 2, 3, 1).contiguous().cpu().numpy()
        path = output / "rgb" / f"{name}.npz"
        with path.open("xb") as handle:
            np.savez_compressed(handle, frames=frames)
        media[name] = {
            "artifact": native._file_row(path), "shape": list(frames.shape), "fps": 24,
            "rgb_sha256": hashlib.sha256(memoryview(frames)).hexdigest(),
            "native_decode_seconds": time.perf_counter() - started, "crop_rows": [4, 1084],
            "latent_sha256": record["tensor_sha256"], "state": "stateless_chunk_decode",
            "memory": native._memory(device),
        }
        native._write(output / "rgb" / f"{name}.json", media[name])
        return frames

    with _deadline(plan["budgets"]["max_case_seconds"]):
        for name in ("common_prefix", "native_raw_first"):
            frames = rgb(name)
            del frames
        for arm in plan["arms"]:
            writer = _VideoWriter(
                plan["ffmpeg"]["path"], output / "videos" / f"{plan['case']['case_id']}__{arm}.mp4",
            )
            try:
                for index in range(1, 5):
                    frames = rgb(f"{arm}_chunk{index}")
                    writer.write(frames)
                    del frames
                videos[arm] = writer.close()
            finally:
                writer.abort()
            _verify_video_rgb(plan["ffmpeg"]["path"], videos[arm])
            native._write(output / "videos" / f"{arm}.json", videos[arm])
    native._write(output / "decode_complete.json", {
        "plan_identity": plan["identity"], "media": media, "videos": videos,
        "VAE_weight_content_sha256": vae_evidence["sha256"],
        "VAE_full_hash_evidence": plan["auxiliary_identity"]["artifact"],
        "VAE_original_stat_guard_checked_this_stage": True,
        "video_rgb_lossless_verified": True, "prefix_excluded_from_videos": True,
        "paired_fps": 24, "continuation_frames": 96,
    })


def _read_rgb(row):
    import numpy as np

    _check_rows([row["artifact"]])
    with np.load(row["artifact"]["path"], allow_pickle=False) as data:
        frames = data["frames"]
    if (frames.dtype != np.uint8 or list(frames.shape) != [24, 1080, 1920, 3]
            or hashlib.sha256(memoryview(np.ascontiguousarray(frames))).hexdigest() != row["rgb_sha256"]):
        raise ValueError("Native RGB count/geometry/hash differs from the decoding record")
    return frames


def report(args):
    import inspect
    import numpy as np
    import skimage
    from skimage.metrics import structural_similarity

    output = native._output(args.output)
    plan = verify_plan(output)
    result = native._read(output / "native_complete.json")
    decoded = native._read(output / "decode_complete.json")
    if any(row["plan_identity"] != plan["identity"] for row in (result, decoded)):
        raise ValueError("Fresh generation, decode and plan identities differ")
    for row in decoded["videos"].values():
        _verify_video_rgb(plan["ffmpeg"]["path"], row)
    reference, raw = (_read_rgb(decoded["media"][name]) for name in ("native_raw_first", "raw_chunk1"))
    if not np.array_equal(reference, raw):
        raise RuntimeError("Fresh RAW native/restored RGB parity failed")
    del reference, raw
    pairs = [("svd_hevc_vs_raw", "raw", "svd_hevc")]
    if "svd" in plan["arms"]:
        pairs += [("svd_vs_raw", "raw", "svd"), ("svd_hevc_vs_svd", "svd", "svd_hevc")]
    comparisons = {}
    for label, first, second in pairs:
        scores, equal = [], True
        for index in range(1, 5):
            a, b = (_read_rgb(decoded["media"][f"{arm}_chunk{index}"]) for arm in (first, second))
            equal = equal and bool(np.array_equal(a, b))
            scores.extend(float(structural_similarity(x, y, channel_axis=-1, data_range=255))
                          for x, y in zip(a, b))
            del a, b
        comparisons[label] = {
            "reference": first, "candidate": second, "frames": 96, "ssim_mean": float(np.mean(scores)),
            "per_frame_ssim": scores, "rgb_bitwise_equal": equal,
            "scope": "matched closed-loop continuation; losses are NOT additive",
        }
    report_value = {
        "schema": "reelcache_fresh_magi_report_v1", "plan_identity": plan["identity"],
        "case": plan["case"], "status": "paired_RGB_storage_verified_VBench_pending",
        "comparisons": comparisons, "raw_native_RGB_exact": True,
        "profile_storage": result["profile_storage"], "profile_memory": result["profile_memory"],
        "primary_arms": plan["primary_arms"], "diagnostic_arms": plan["diagnostic_arms"],
        "selected_checkpoints": result["selected_checkpoints"],
        "native_gates": result["gates"], "videos": decoded["videos"],
        "timings": {path.stem: native._read(path) for path in sorted((output / "timings").glob("*.json"))},
        "ssim": {
            "method": "mean96frames, native uint8 RGB; skimage structural_similarity(channel_axis=-1,data_range=255)",
            "version": skimage.__version__, "source": native._file_row(inspect.getsourcefile(structural_similarity)),
            "video_RGB_lossless_verified": True,
        },
        "official_vbench_evaluation": "pending separately granted parent evaluator; no invented score",
        "limitations": plan["limitations"],
    }
    native._write(output / "report.json", report_value)
    native._write(output / "vbench_input.json", {
        "schema": "reelcache_vbench_video_handoff_v1", "case": plan["case"], "plan_identity": plan["identity"],
        "catalogue": plan["protocol"]["catalogue"], "dimensions": plan["official_dimensions_for_prompt"],
        "videos": decoded["videos"], "frames": 96, "fps": 24, "resolution": [1920, 1080],
        "primary_arms": plan["primary_arms"], "diagnostic_arms": plan["diagnostic_arms"],
        "evaluation_scope": "four official prompt rows x seed0, NOT full official VBench aggregate",
        "official_custom_input_prompt_file": "vbench_prompts.json",
        "evaluator_video_loader_check": {
            "required_before_scoring": True,
            "compare_to": "videos[arm].rgb_sha256 and videos[arm].shape in RGB uint8 order",
            "already_verified": "bound FFmpeg CPU decode",
            "not_yet_verified": "the parent's selected official VBench video-loading backend",
        },
    })
    native._write(output / "vbench_prompts.json", {
        Path(row["artifact"]["path"]).name: plan["case"]["prompt"] for row in decoded["videos"].values()
    })
    compact = []
    for path in sorted(output.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(output)
        if relative.parts[0] in {"stores", "staging", "build_cache", "rgb"}:
            continue
        compact.append({"relative_path": str(relative), **native._file_row(path)})
    native._write(output / "compact_artifacts.json", {
        "schema": "reelcache_fresh_magi_compact_v1", "plan_identity": plan["identity"],
        "files": compact, "bytes": sum(row["bytes"] for row in compact),
        "retain": "source plans, conditioning, selected execution/latent checkpoints, immutable decoder/resume ledgers, videos, native/SSIM/storage/timing/evaluator metadata",
        "local_only": ["stores", "staging", "build_cache", "rgb"],
        "full_snapshot_resume_without_local_store": False,
        "external_frozen_source_requirements": plan["reelcache_sources"],
        "external_study_provenance_requirements": {
            "weights": plan["study_weight_identity"]["artifact"],
            "T5_VAE_sources": plan["auxiliary_identity"]["artifact"],
        },
        "cleanup": "parent may remove local output ONLY after all listed compact files are copied and hash-verified",
    })
    print(json.dumps({"status": report_value["status"], "case": plan["case"]["case_id"],
                      "ssim": {name: row["ssim_mean"] for name, row in comparisons.items()},
                      "compact_manifest": str(output / "compact_artifacts.json")}))
