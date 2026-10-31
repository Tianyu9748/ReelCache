"""Source-bound MAGI legacy verification and isolated rank-study workers.

Legacy prepare/run/decode/report retain the qualified development builder.
The study worker never imports that builder: it consumes reviewed native
source, exact locally verified indexed shards, and immutable study requests.
It generates every chunk from the request's first noise, saves latents while
DiT is resident, and hands off to decode_batch only after both SP2 ranks exit.
No native module, model or checkpoint is loaded when importing this helper.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import ExitStack, contextmanager, nullcontext
from datetime import datetime, timedelta, timezone
import gc
import hashlib
import importlib
import json
import os
from pathlib import Path
import resource
import shlex
import struct
import sys
import time
import uuid
from unittest.mock import patch


PROJECT = Path(__file__).resolve().parents[2]
N, C, LAYERS = 48960, 1024, 48
SHAPE = (1, 16, 6, 136, 240)


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def _identity(value):
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()


def _read(path):
    return json.loads(Path(path).read_text(), parse_constant=lambda value: (
        (_ for _ in ()).throw(ValueError(f"Nonfinite JSON: {value}"))
    ))


def _output(path):
    path = Path(path).resolve()
    path.relative_to(PROJECT)
    relative = path.relative_to(PROJECT)
    if not relative.parts or relative.parts[0] in {"reelcache", "tests", "examples", "docs"}:
        raise ValueError("Native outputs must be new ReelCache artifact directories, not source files")
    return path


def _write(path, value):
    path = _output(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(f".{path.name}.{uuid.uuid4().hex}.pending")
    try:
        with pending.open("x") as handle:
            json.dump(value, handle, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(pending, path)
    finally:
        pending.unlink(missing_ok=True)


def _file_row(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": _sha(path), "bytes": path.stat().st_size}


def _tensor_hash(tensor):
    from .magi import _tensor_hash as digest

    return digest(tensor)


def _validate_config(config, *, cell_id="magi1080", study_worker=False):
    from ..study import study_cell

    cell = study_cell(cell_id)
    if cell.model != "magi_distill":
        raise ValueError("MAGI native configuration requires a MAGI study geometry")
    expected = {
        "model_config": {
            "num_layers": 48, "hidden_size": 6144, "num_attention_heads": 48,
            "num_query_groups": 8, "kv_channels": 128, "params_dtype": "torch.bfloat16",
        },
        "runtime_config": {
            "cfg_number": 1, "num_steps": 16, "chunk_width": 6,
            "video_size_h": cell.working_height, "video_size_w": cell.width, "clean_t": 0.9999,
        },
        "engine_config": {
            "cp_size": 2, "cp_strategy": "cp_ulysses", "pp_size": 1,
            "ulysses_overlap_degree": 1, "distill": True, "fp8_quant": False,
            "enable_cuda_graph": False, "kv_offload": False,
        },
    }
    for section, fields in expected.items():
        for key, value in fields.items():
            if config[section].get(key) != value:
                raise ValueError(f"Unsupported native MAGI contract: {section}.{key}")
    if config["model_config"].get("patch_size", 2) != 2:
        raise ValueError("MAGI native token grids require the checkpoint's spatial patch size two")
    if config["engine_config"].get("distill_nearly_clean_chunk_threshold", 0.3) != 0.3:
        raise ValueError("The qualified native distilled duplicate threshold is 0.3")
    for key in ("compress_kv", "save_kvcache_every_forward", "fwd_extra_1st_chunk"):
        if config["runtime_config"].get(key, False) or config["engine_config"].get(key, False):
            raise ValueError("Study workers require the original serial full-KV write path")
    if study_worker:
        window = config["runtime_config"].get("window_size")
        if type(window) is not int or window != 1:
            raise ValueError("Study serial_clean_history requires explicit integer runtime_config.window_size=1")
        spans = config["runtime_config"].get("noise2clean_kvrange")
        clean_span = config["runtime_config"].get("clean_chunk_kvrange")
        if (not isinstance(spans, list) or len(spans) != 4
                or any(type(value) is not int or value <= 0 for value in spans)
                or type(clean_span) is not int or (clean_span != -1 and clean_span <= 0)):
            raise ValueError("Study workers require explicit native noisy/clean visibility ranges")


def _validate_study_timetable(schedule, intervals):
    import torch

    if (schedule.dtype != torch.float32 or intervals.dtype != torch.float32
            or tuple(schedule.shape) != (17,) or tuple(intervals.shape) != (16,)
            or intervals.device != schedule.device):
        raise ValueError("Serial MAGI requires the native sixteen-step FP32 timetable and intervals")
    squared = torch.linspace(0, 1, 17, device=schedule.device, dtype=torch.float32).square()
    expected = (1.0 / 3.0) * squared / (1 + (1.0 / 3.0 - 1) * squared)
    if (not torch.equal(schedule, expected)
            or not torch.equal(intervals, torch.ones_like(intervals))
            or [i for i, value in enumerate(schedule[:-1].cpu().tolist()) if value > 0.3]
            != [12, 13, 14, 15]):
        raise ValueError("The native sixteen-step uniform-square-SD3 shift-three timetable changed")


def _precision_preflight(arms):
    import torch
    from .magi import MagiGeometry, _tensor_context
    from ..codecs import decode_tensor, encode_tensor
    from ..types import CodecSpec, MatrixKey, ModelIdentity

    geometry = MagiGeometry(layers=1, kv_heads=2, head_dim=4, frames=1, height=2, width=4, world_size=2)
    model = ModelIdentity("magi_precision_probe", "synthetic_CPU_contract", 1, 8, "global_KV_channels", "bfloat16")
    native = (torch.arange(64).float().reshape(8, 8) / 17).to(torch.bfloat16)
    context = _tensor_context(model, MatrixKey(0, "K"), 0, geometry, "post_rope", native.dtype)
    encoded = encode_tensor(native, context, CodecSpec())
    raw = decode_tensor(encoded, columns=slice(4, 8), dtype=torch.bfloat16)
    if not torch.equal(raw, native[:, 4:8]):
        raise RuntimeError("The RAW provider does not preserve local native head-column bytes")
    result = {"raw_local_column_exact": True, "native_dtype": "bfloat16",
              "synthetic_contract_check_not_calibration": True}
    if any(arm != "raw" for arm in arms):
        source = torch.arange(64).float().reshape(8, 8) / 17
        context = _tensor_context(model, MatrixKey(0, "K"), 0, geometry, "pre_rope", source.dtype)
        try:
            encoded = encode_tensor(source, context, CodecSpec(kind="svd", rank=2))
        except Exception as exc:
            raise RuntimeError(
                "The explicit source_dtype must preserve FP32 pre-RoPE K under the native BF16 identity; "
                "do not round K to BF16 or launch before fixing this precision contract."
            ) from exc
        decoded = decode_tensor(encoded, columns=slice(4, 8), dtype=torch.float32)
        if decoded.shape != (8, 4) or encoded.metadata["dtype"] != "float32":
            raise RuntimeError("Pre-RoPE source precision or local-head reconstruction changed")
        result["pre_rope_source_dtype"] = "float32"
        result["factor_source_dtypes"] = {"K": "float32", "V": "bfloat16"}
        result["source_dtype_override"] = context.source_dtype
        result["native_identity_unchanged_across_representations"] = context.model is model
        result["pre_rope_local_column_shape"] = [8, 4]
    return result


def _weights(directory, expected):
    directory = Path(directory).resolve()
    index = directory / "model.safetensors.index.json"
    if _sha(index) != expected["index_sha256"]:
        raise ValueError("Checkpoint index differs from the proven source-bound pilot")
    rows = []
    for reference in expected["shards"]:
        path = directory / Path(reference["path"]).name
        if path.stat().st_size != reference["bytes"]:
            raise ValueError(f"Checkpoint shard size changed: {path}")
        with path.open("rb") as handle:
            size = struct.unpack("<Q", handle.read(8))[0]
            if size > 128 * 1024**2:
                raise ValueError("Implausible safetensors header size")
            digest = hashlib.sha256(handle.read(size)).hexdigest()
        if digest != reference["header_sha256"]:
            raise ValueError(f"Checkpoint shard header changed: {path}")
        rows.append({"path": str(path), "bytes": path.stat().st_size, "header_sha256": digest})
    return {
        "index": _file_row(index), "shards": rows,
        "verification": "exact index, shard sizes and safetensors headers; full shard content not rehashed",
        "full_weight_content_rehashed": False,
    }


def _environment(output, threads):
    for name in ("TMPDIR", "TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR", "TORCH_EXTENSIONS_DIR",
                 "FLASHINFER_WORKSPACE_BASE", "TVM_FFI_CACHE_DIR", "CUDA_CACHE_PATH",
                 "XDG_CACHE_HOME", "HF_HOME"):
        path = output / "build_cache" / name.lower()
        path.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(path)
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = str(threads)
    os.environ.update({
        "PYTHONDONTWRITEBYTECODE": "1", "HF_HUB_OFFLINE": "1", "HF_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
        "TVM_FFI_DISABLE_TORCH_C_DLPACK": "1", "FLASHINFER_CUDA_ARCH_LIST": "9.0",
        "prev_chunks_scale": "0.7",
    })
    sys.dont_write_bytecode = True


def prepare(args):
    output = _output(args.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Refusing to alter an existing native verification run")
    if args.cpu_threads < 1 or args.resident_mib < 1:
        raise ValueError("CPU threads and reconstruction budget must be positive")
    if len(set(args.arms)) != len(args.arms) or args.arms[0] != "raw":
        raise ValueError("Arms must be unique and RAW must be first")
    if any(arm != "raw" for arm in args.arms) and not args.allow_lossy:
        raise ValueError("Lossy arms require explicit --allow-lossy; no qualified profile is implied")
    if "svd_hevc" in args.arms and "svd" not in args.arms:
        raise ValueError("Include SVD-only to distinguish projection loss from HEVC loss")
    binding_path = Path(args.binding).resolve()
    binding = _read(binding_path)
    claimed = binding.pop("experiment_identity_sha256")
    if _identity(binding) != claimed:
        raise ValueError("The source pilot binding's content identity changed")
    binding["experiment_identity_sha256"] = claimed
    _validate_config(binding["config"])
    if binding["case_id"] not in ("p00_seed0", "p01_seed0") or binding["seed"] != 0:
        raise ValueError("This bounded driver supports only the existing p00/p01 seed-zero inputs")
    source_project, native_source = args.source_project.resolve(), args.native_source.resolve()
    source_plan = _read(binding["actual_source_plan"])
    if _sha(binding["actual_source_plan"]) != binding["actual_source_plan_sha256"]:
        raise ValueError("Current source plan no longer matches its pilot binding")
    external = {}
    for name, row in binding["source_files"].items():
        if _sha(row["path"]) != row["sha256"]:
            raise ValueError(f"A source-bound pilot helper changed: {name}")
        external[f"pilot/{name}"] = _file_row(row["path"])
    builder = source_project / "kv_compression/harness/neurico_magi_sweep.py"
    if _sha(builder) != binding["source_files"]["capture_and_model_loader"]["sha256"]:
        raise ValueError("Configured model builder is not the proven narrowly qualified builder")
    for relative, expected in source_plan["source_hashes"].items():
        path = native_source / relative
        if _sha(path) != expected:
            raise ValueError(f"Configured native source differs from the source-bound plan: {relative}")
        external[f"native/{relative}"] = _file_row(path)
    external["native/special_tokens"] = _file_row(native_source / "example/assets/special_tokens.npz")
    checkpoint = _weights(args.weights, binding["checkpoint_verification"])
    for name in ("embedding", "prefix"):
        if _sha(binding[name]["path"]) != binding[name]["file_sha256"]:
            raise ValueError(f"Existing VBench {name} changed")
    import torch
    import transformers

    if torch.cuda.is_initialized():
        raise RuntimeError("Preparation must not run in a CUDA-initialized process")
    if not torch.__version__.startswith("2.8.0") or transformers.__version__ != "4.42.3":
        raise RuntimeError("Use the existing qualified Torch 2.8 / transformers 4.42.3 MAGI environment")
    precision = _precision_preflight(args.arms)
    capabilities = None
    if "svd_hevc" in args.arms:
        from ..codecs import hevc_capabilities

        capabilities = hevc_capabilities()
        if "yuv444p" not in capabilities["pixel_formats"]:
            raise RuntimeError("The existing HEVC binary does not support the requested yuv444p format")
    prefix = torch.load(binding["prefix"]["path"], map_location="cpu", weights_only=True)["latent"]
    text = torch.load(binding["embedding"]["path"], map_location="cpu", weights_only=True)
    if (tuple(prefix.shape) != SHAPE or prefix.dtype != torch.float32 or not torch.isfinite(prefix).all()
            or _tensor_hash(prefix) != binding["prefix"]["tensor_sha256"]):
        raise ValueError("Saved native prefix geometry or tensor identity changed")
    if (text["prompt"] != binding["prompt"] or tuple(text["caption"].shape) != (1, 800, 4096)
            or tuple(text["mask"].shape) != (1, 800)
            or _tensor_hash(text["caption"]) != binding["embedding"]["caption_tensor_sha256"]
            or _tensor_hash(text["mask"]) != binding["embedding"]["mask_tensor_sha256"]):
        raise ValueError("Saved native conditioning identity changed")
    vae = args.vae.resolve()
    if not (vae / "diffusion_pytorch_model.safetensors").is_file():
        raise FileNotFoundError("Configure the existing native MAGI VAE directory")
    vae_reference_path = binding_path.parent / binding["case_id"] / "arms/RAW/media/manifest.json"
    vae_media = _read(vae_reference_path)
    generation_reference_path = binding_path.parent / binding["case_id"] / "arms/RAW/generation.json"
    generation_reference = _read(generation_reference_path)
    if _sha(vae / "config.json") != vae_media["VAE_config_sha256"]:
        raise ValueError("Configured VAE config differs from the native reference")
    source_names = (
        "__init__.py", "errors.py", "types.py", "serialization.py", "store.py", "residency.py",
        "manager.py", "codecs.py", "factorization.py", "reuse.py", "_hevc.py", "distributed.py",
        "adapters/__init__.py", "adapters/magi.py", "adapters/magi_native.py",
    )
    own_sources = {f"reelcache/{name}": _file_row(PROJECT / "reelcache" / name) for name in source_names}
    own_sources["examples/native_magi.py"] = _file_row(PROJECT / "examples/native_magi.py")
    config = json.loads(json.dumps(binding["config"]))
    config["runtime_config"]["load"] = str(args.weights.resolve().parent)
    config["runtime_config"]["vae_pretrained"] = str(vae)
    plan = {
        "schema": "reelcache_native_magi_verification_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_type": "new_ReelCache_serial_clean_history_native_verification",
        "historical_replay_claim": False, "pipelined_performance_claim": False,
        "quality_guarantee": None, "calibration_expansion": False,
        "source_binding": _file_row(binding_path),
        "source_plan": _file_row(binding["actual_source_plan"]),
        "external_sources": external, "reelcache_sources": own_sources,
        "builder": _file_row(builder), "source_project": str(source_project),
        "native_source": str(native_source), "weights": str(args.weights.resolve()),
        "checkpoint_verification": checkpoint,
        "checkpoint_reference": binding["checkpoint_verification"],
        "config": config, "case_id": binding["case_id"], "prompt": binding["prompt"],
        "seed": 0, "prefix": binding["prefix"], "embedding": binding["embedding"],
        "noise_reference": {
            "artifact": _file_row(generation_reference_path),
            "prefix_noise_sha256": generation_reference["prefix_noise_sha256"],
            "first_continuation_noise_sha256": generation_reference["next_noise_sha256"],
            "later_continuation_is_new": True,
        },
        "transport": {"canonical_storage": "owner rank 0", "storage_backend": "gloo",
                      "native_attention_backend": "nccl",
                      "restoration_payload": "rank-local reconstructed dense head columns"},
        "canonical_domains": {
            "raw": {"K": "bfloat16", "V": "bfloat16", "key_space": "post_rope"},
            "factors": {"K": "float32", "V": "bfloat16", "key_space": "pre_rope"},
            "native_cache_dtype_for_both": "bfloat16",
            "native_model_identity_dtype": "bfloat16",
            "explicit_context_source_dtype_for_factor_K": "float32",
            "all_arms_share_native_model_identity_and_owner_store": True,
            "SVD_and_HEVC_share_canonical_bases": True,
        },
        "vae": {
            "path": str(vae), "config_sha256": vae_media["VAE_config_sha256"],
            "source_reference": _file_row(vae_reference_path),
            "expected_checkpoint_sha256": vae_media["VAE_weight_sha256"],
            "checkpoint_full_hash_verified_during_decode": False,
            "mode": "stateless_chunk_decode", "temporal_tile_overlap_factor": 0,
        },
        "arms": args.arms, "lossy_explicit_opt_in": args.allow_lossy,
        "hevc_capabilities": capabilities,
        "ffmpeg": _file_row(capabilities["binary"]) if capabilities is not None else None,
        "continuations": args.continuations, "ranks": {"K": 256, "V": 384},
        "cpu_threads": args.cpu_threads, "resident_mib": args.resident_mib,
        "qualified_environment": {
            "python": sys.executable, "torch": torch.__version__,
            "transformers": transformers.__version__, "cuda_initialized": False,
        },
        "native_precision_contract": precision,
        "compatibility": {
            "blackwell_silu": "existing qualified PyTorch F.silu(gate) * up fallback, scoped to this driver",
            "flashinfer_arch_for_import_only": "9.0", "SM90_kernels_executed": False,
            "native_attention_fix": "qualified completed Ulysses receive and row-local FlashAttention",
        },
        "limitations": [
            "Serial clean-history, batch one, SP2; not pipelined MAGI throughput.",
            "RAW is native post-RoPE BF16; lossy K factorization uses FP32 normalized pre-RoPE K.",
            "No HBM saving is attributed to duplicated full KV: native KV is already head-sharded.",
            "CPU/Gloo canonical gather and per-layer local-column restore are included in timings.",
            "Owner coefficient/basis storage is canonical once; native layer buffers and allocator HBM are measured.",
            "Existing p00 PC SSIM about 0.803 is not a quality guarantee for ReelCache or other prompts.",
            "Full MAGI weight contents are not rehashed; exact shard headers, sizes and index are checked.",
        ],
    }
    plan["identity"] = _identity(plan)
    _write(output / "plan.json", plan)
    _write(output / "config.json", config)
    launch_directory = output / "build_cache" / "tmpdir"
    launch_directory.mkdir(parents=True, exist_ok=True)
    interpreter = sys.executable
    launch_environment = [
        "env", f"TMPDIR={launch_directory}", "PYTHONDONTWRITEBYTECODE=1",
        f"OMP_NUM_THREADS={args.cpu_threads}", f"MKL_NUM_THREADS={args.cpu_threads}",
        f"OPENBLAS_NUM_THREADS={args.cpu_threads}",
    ]
    working_directory = f"cd {shlex.quote(str(PROJECT))} && "
    commands = {
        "cwd": str(PROJECT),
        "run_AFTER_EXPLICIT_BOTH_GPU_GRANT": working_directory + shlex.join([
            *launch_environment, "CUDA_VISIBLE_DEVICES=0,1",
            interpreter, "-B", "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2",
            "examples/native_magi.py", "run", "--output", str(output), "--gpu-grant",
        ]),
        "decode_AFTER_SP2_EXIT_AND_SEPARATE_GPU_GRANT": working_directory + shlex.join([
            *launch_environment, "CUDA_VISIBLE_DEVICES=0",
            interpreter, "-B", "examples/native_magi.py", "decode", "--output", str(output),
            "--device", "0", "--gpu-grant",
        ]),
        "report_CPU": working_directory + shlex.join([
            *launch_environment, "CUDA_VISIBLE_DEVICES=",
            interpreter, "-B", "examples/native_magi.py", "report", "--output", str(output),
        ]),
    }
    _write(output / "commands.json", commands)
    print(json.dumps({"status": "CPU_ready_NOT_GPU_authorized", "output": str(output),
                      "identity": plan["identity"], "case": plan["case_id"], "commands": commands}))


def _verify_plan(output):
    plan = _read(output / "plan.json")
    claimed = plan.pop("identity")
    if _identity(plan) != claimed:
        raise ValueError("ReelCache verification plan changed")
    plan["identity"] = claimed
    for row in [plan["source_binding"], plan["source_plan"], plan["builder"], plan["vae"]["source_reference"],
                plan["noise_reference"]["artifact"],
                *plan["external_sources"].values(), *plan["reelcache_sources"].values()]:
        if _sha(row["path"]) != row["sha256"]:
            raise ValueError(f"A bound source changed; prepare a NEW verification run: {row['path']}")
    if plan["ffmpeg"] is not None and _sha(plan["ffmpeg"]["path"]) != plan["ffmpeg"]["sha256"]:
        raise ValueError("The bound software HEVC executable changed")
    for name in ("embedding", "prefix"):
        if _sha(plan[name]["path"]) != plan[name]["file_sha256"]:
            raise ValueError(f"Bound {name} changed")
    if _read(output / "config.json") != plan["config"]:
        raise ValueError("Bound native configuration changed")
    _weights(plan["weights"], plan["checkpoint_reference"])
    _validate_config(plan["config"])
    return plan


def _execution(plan):
    return {
        "plan_identity": plan["identity"], "pid": os.getpid(),
        "process_start_ticks": _process_start_ticks(os.getpid()),
        "host": os.uname().nodename, "python": sys.executable,
        "command": shlex.join([sys.executable, "-B", *sys.argv]),
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "environment": {key: os.environ.get(key) for key in (
            "CUDA_VISIBLE_DEVICES", "RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR",
            "MASTER_PORT", "TMPDIR", "FLASHINFER_CUDA_ARCH_LIST",
        )},
    }


def _process_start_ticks(pid):
    try:
        value = Path(f"/proc/{pid}/stat").read_text()
    except FileNotFoundError:
        return None
    return value[value.rfind(")") + 2:].split()[19]


def _require_workers_exited(result):
    for row in result["worker_execution_records"]:
        if _sha(row["path"]) != row["sha256"]:
            raise ValueError("Native worker execution metadata changed")
        execution = _read(row["path"])
        if execution["host"] != os.uname().nodename:
            raise RuntimeError("Initial VAE handoff requires the native worker host for exit verification")
        current = _process_start_ticks(execution["pid"])
        if current is not None and current == execution["process_start_ticks"]:
            raise RuntimeError("SP2 workers are still alive; wait for BOTH workers to exit before VAE loading")


def _memory(device, cache=None):
    import torch

    torch.cuda.synchronize(device)
    free, total = torch.cuda.mem_get_info(device)
    buffers = {} if cache is None else cache.key_value_memory_dict
    allocations = {value.untyped_storage().data_ptr(): value.untyped_storage().nbytes()
                   for value in buffers.values()}
    return {
        "torch_allocated_bytes": torch.cuda.memory_allocated(device),
        "torch_reserved_bytes": torch.cuda.memory_reserved(device),
        "torch_peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "device_used_bytes": total - free, "device_total_bytes": total,
        "device_used_includes_other_processes_and_driver": True,
        "native_cache_storage_bytes": sum(allocations.values()),
        "native_cache_entries": len(buffers),
        "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
    }


class _Session:
    def __init__(self, plan, output):
        import torch
        import torch.distributed as dist
        from ..distributed import DistributedOwner

        self.torch, self.dist, self.plan, self.output = torch, dist, plan, output
        sys.path[:0] = [plan["native_source"], str(Path(plan["source_project"]) / "kv_compression/harness"),
                       plan["source_project"]]
        os.environ["SPECIAL_TOKEN_PATH"] = str(Path(plan["native_source"]) / "example/assets/special_tokens.npz")
        native = importlib.import_module("native_magi_sp_bench")
        loader = importlib.import_module("neurico_magi_sweep")
        if Path(loader.__file__).resolve() != Path(plan["builder"]["path"]):
            raise RuntimeError("A different pre-imported native builder shadows the bound source")
        native.SOURCE = Path(plan["native_source"])
        native.WEIGHTS = Path(plan["weights"])
        native.OUT = output
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
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        self.storage_group = dist.new_group(backend="gloo", timeout=timedelta(minutes=60))
        self.transport = DistributedOwner(self.storage_group, expected_world_size=2)
        load_started = time.perf_counter()
        self.model = loader.load_model(self.config, self.device)
        torch.cuda.synchronize(self.device)
        self.model_load_seconds = time.perf_counter() - load_started
        self.executed_model_source = _file_row(sys.modules[self.model.__class__.__module__].__file__)
        if self.executed_model_source != plan["external_sources"]["native/inference/model/dit/dit_model.py"]:
            raise RuntimeError("The executed native model is not the bound model source")
        self.rotary = flash_apply_rotary_emb
        self.fallback = ExitStack()
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
        self.cache = InferenceParams(max_batch_size=1, max_sequence_length=(plan["continuations"] + 1) * N)
        text = torch.load(plan["embedding"]["path"], map_location="cpu", weights_only=True)
        caption, mask = text["caption"].to(self.device), text["mask"].to(self.device)
        self.null_caption = self.model.y_embedder.null_caption_embedding.detach().unsqueeze(0)
        self.null_mask = torch.zeros_like(mask)
        self.null_mask[:, :50] = 1
        self.y = torch.cat((caption, self.null_caption)).unsqueeze(1)
        self.masks = torch.cat((mask, self.null_mask)).unsqueeze(1)
        self.prefix = torch.load(plan["prefix"]["path"], map_location=self.device, weights_only=True)["latent"]
        self.generator = torch.Generator(device=self.device).manual_seed(0)
        first_noise = torch.randn(SHAPE, generator=self.generator, device=self.device)
        self.prefix_noise_sha256 = _tensor_hash(first_noise)
        del first_noise
        self.noises = [torch.randn(SHAPE, generator=self.generator, device=self.device)
                       for _ in range(plan["continuations"])]
        if (self.prefix_noise_sha256 != plan["noise_reference"]["prefix_noise_sha256"]
                or _tensor_hash(self.noises[0]) != plan["noise_reference"]["first_continuation_noise_sha256"]):
            raise RuntimeError("Qualified seed-zero CUDA noise differs from the source-bound pilot")
        self.schedule = init_t({}, 16, self.device, shortcut_mode=self.config.engine_config.shortcut_mode)
        self.intervals = init_intervel(16, self.device, shortcut_mode=self.config.engine_config.shortcut_mode)
        self.near = [value > 0.3 for value in self.schedule[:-1].cpu().tolist()]
        if [index for index, near in enumerate(self.near) if near] != [12, 13, 14, 15]:
            raise RuntimeError("Native distilled schedule changed")
        hashes = self.transport.checkpoint("noise_identity", details=[_tensor_hash(value) for value in self.noises])
        if any(value != hashes[0] for value in hashes):
            raise RuntimeError("SP ranks received different continuation noise")

    def kwargs(self, index, *, clean=False, step=0):
        return {
            "chunk_width": 6, "num_steps": 16, "slice_point": index, "range_num": index + 1,
            "denoising_range_num": 1, "fwd_extra_1st_chunk": False, "start_chunk_id": index,
            "end_chunk_id": index + 1 + int(not clean and self.near[step]),
            "chunk_num": self.plan["continuations"] + 1, "extract_prefix_video_feature": clean,
            "distill_nearly_clean_chunk": not clean and self.near[step],
            "distill_interval": self.intervals[step % len(self.intervals)],
        }

    def clean(self, latent, index, adapter=None):
        t = self.torch
        first = max(0, index + 1 - self.config.runtime_config.clean_chunk_kvrange)
        guard = nullcontext() if adapter is None else adapter.execution(
            chunk_index=index, clean=True, clean_timestep=0.9999,
        )
        with guard:
            self.cache.update_kv_cache = True
            value = self.model.forward(
                latent, t.full((1, 1), 0.9999, device=self.device), self.null_caption.unsqueeze(1),
                caption_dropout_mask=t.zeros((1,), dtype=t.bool, device=self.device),
                xattn_mask=self.null_mask.unsqueeze(1),
                kv_range=t.tensor([[first * N, (index + 1) * N]], dtype=t.int32, device=self.device),
                inference_params=self.cache, **self.kwargs(index, clean=True),
            )
            if not t.isfinite(value).all():
                raise RuntimeError("Native clean feature forward became nonfinite")

    def denoise(self, noise, index, adapter=None):
        t = self.torch
        guard = nullcontext() if adapter is None else adapter.execution(chunk_index=index, clean=False)
        with guard:
            x = t.cat((noise, noise))
            for step in range(16):
                context = self.config.runtime_config.noise2clean_kvrange[step // 4]
                first = max(0, index + 1 - context)
                velocity = self.model.forward_dispatcher(
                    x, self.schedule[step].expand(2, 1), self.y, self.masks,
                    t.tensor([[first * N, (index + 1) * N]], dtype=t.int32, device=self.device),
                    self.cache, **self.kwargs(index, step=step),
                )
                x = x + velocity * (self.schedule[step + 1] - self.schedule[step])
            latent = x[:1].clone()
            if tuple(latent.shape) != SHAPE or latent.dtype != t.float32 or not t.isfinite(latent).all():
                raise RuntimeError("Native continuation latent contract failed")
        return latent

    def timed(self, label, operation):
        self.torch.cuda.synchronize(self.device)
        self.transport.checkpoint(f"timing:{label}:begin")
        self.torch.cuda.reset_peak_memory_stats(self.device)
        before = _memory(self.device, self.cache)
        started = time.perf_counter()
        result = operation()
        self.torch.cuda.synchronize(self.device)
        row = {"rank": self.rank, "seconds": time.perf_counter() - started,
               "before": before, "after": _memory(self.device, self.cache)}
        rows = self.transport.checkpoint(f"timing:{label}:end", details=row)
        self.transport.owner_call(f"timing:{label}:publish", lambda: _write(
            self.output / "timings" / f"{label}.json", {
                "label": label, "max_rank_seconds": max(item["seconds"] for item in rows),
                "per_rank": rows, "scope": "serial operation including synchronous storage/transport when active",
            },
        ))
        return result

    def save(self, name, latent, *, reference=None):
        digest = _tensor_hash(latent)
        hashes = self.transport.checkpoint(f"latent:{name}", details=digest)
        if len(set(hashes)) != 1:
            raise RuntimeError("SP2 ranks disagree on the native continuation")
        delta = None
        if reference is not None:
            diff = latent.double() - reference.double()
            delta = {
                "bitwise_equal": self.torch.equal(latent, reference),
                "relative_frobenius": float(diff.norm() / reference.double().norm()),
                "maximum_absolute_error": float(diff.abs().max()),
            }
        def publish():
            directory = self.output / "latents"
            directory.mkdir(exist_ok=True)
            path = directory / f"{name}.pt"
            with path.open("xb") as handle:
                self.torch.save({"latent": latent.detach().cpu()}, handle)
            row = {
                "name": name, "artifact": _file_row(path), "tensor_sha256": digest,
                "shape": list(latent.shape), "dtype": str(latent.dtype),
                "per_rank_hashes": hashes, "paired_reference_delta": delta,
                "plan_identity": self.plan["identity"],
            }
            _write(directory / f"{name}.json", row)
            return row
        row = self.transport.owner_call(f"save:{name}", publish)
        return self.transport.broadcast_json(row, operation=f"saved:{name}")

    def state(self, latent):
        return {
            "metadata": {
                "conditioning_identity": self.plan["embedding"]["caption_tensor_sha256"],
                "execution_config": {
                    "config": self.plan["config"], "schedule": self.schedule.cpu().tolist(),
                    "intervals": self.intervals.cpu().tolist(), "near_clean_steps": [12, 13, 14, 15],
                },
                "vae_contract": self.plan["vae"],
            },
            "tensors": {
                "latent": latent, "conditioning": self.y, "conditioning_mask": self.masks,
                "null_caption": self.null_caption, "null_mask": self.null_mask,
                **{f"noise/{index + 1}": noise for index, noise in enumerate(self.noises)},
            },
            "generators": {"noise": self.generator},
        }

    def reinstate(self, restored):
        _, tensors = restored
        self.y = tensors["conditioning"].to(self.device)
        self.masks = tensors["conditioning_mask"].to(self.device)
        self.null_caption = tensors["null_caption"].to(self.device)
        self.null_mask = tensors["null_mask"].to(self.device)
        self.noises = [tensors[f"noise/{index + 1}"].to(self.device)
                       for index in range(self.plan["continuations"])]
        return tensors["latent"].to(self.device)


def _storage(runtime, branch, snapshot):
    from ..serialization import tensor_nbytes
    from ..types import TensorContext

    info = runtime.branch_info(branch)
    seen, rows = set(), []
    canonical_input_bytes = 0
    pending = [info["head_id"]] + ([snapshot] if snapshot is not None else [])
    counts = {"basis": 0, "blob": 0, "manifest": 0}
    while pending:
        object_id = pending.pop()
        if object_id in seen:
            continue
        seen.add(object_id)
        payload = runtime.store.get(object_id)
        kind = object_id.split(":", 1)[0]
        counts[kind] += len(payload)
        if kind == "manifest":
            pending.extend(runtime.store.references(object_id))
    for chunk_id in info["chunk_manifests"]:
        chunk = runtime.store.get_manifest(chunk_id)
        for matrix in chunk["matrices"]:
            packet = runtime.store.get_manifest(matrix["tensor"])
            context = TensorContext.from_dict(packet["context"])
            input_bytes = tensor_nbytes(context.shape, context.dtype)
            canonical_input_bytes += input_bytes
            rows.append({
                "chunk_id": chunk["chunk_id"], "key": matrix["key"],
                "basis_object": packet["encoded"]["basis"], "metrics": packet["encoded"]["metrics"],
                "spec": packet["spec"], "context": packet["context"],
                "canonical_input_dtype": context.dtype, "canonical_input_bytes": input_bytes,
                **({"rank_decision": matrix["decision"]} if "decision" in matrix else {}),
            })
    native_bytes = (
        info["token_end"] * 2 * runtime.model.layers
        * (runtime.model.channels_for("K") + runtime.model.channels_for("V"))
    )
    return {
        "branch": info, "reachable_unique_bytes": sum(counts.values()),
        "store_root": str(runtime.store.root),
        "snapshot_id": snapshot, "execution_snapshot_bytes_included": snapshot is not None,
        "byte_breakdown": counts, "native_history_bytes_across_SP2": native_bytes,
        "native_history_bytes_per_rank": native_bytes // 2,
        "native_reference_domain": "BF16 post-RoPE K and BF16 V; already head-sharded in native SP2",
        "canonical_matrix_input_bytes": canonical_input_bytes,
        "canonical_input_precision_note": (
            "Factor K is captured native FP32 normalization output before RoPE, not an adapter promotion; "
            "V is native BF16. These input bytes are not used as the native compression-ratio numerator."
        ),
        "ratio_native_history_to_unique_objects": native_bytes / sum(counts.values()),
        "canonical_owner": 0, "canonical_bases_stored_once": True,
        "no_duplicate_full_KV_savings_claim": True, "packets": rows,
        "whole_store": runtime.store.stats(),
    }


def _arm_specs(arm):
    from ..types import CodecSpec, MatrixKey

    if arm not in ("raw", "svd", "svd_hevc"):
        raise ValueError("Unknown fixed native MAGI arm")
    return {
        MatrixKey(layer, target): CodecSpec() if arm == "raw" else CodecSpec(
            kind=arm, rank=256 if target == "K" else 384,
            normalization="per_channel", protected_fraction=1.0,
        )
        for layer in range(LAYERS) for target in ("K", "V")
    }


def run(args):
    if not args.gpu_grant:
        raise RuntimeError("An explicit grant of BOTH GPUs is required")
    if int(os.environ.get("WORLD_SIZE", "0")) != 2:
        raise RuntimeError("Launch this stage using native torchrun --nproc_per_node=2")
    output = _output(args.output)
    plan = _verify_plan(output)
    if (output / "native_complete.json").exists():
        raise FileExistsError("This verification run already completed; prepare a new run")
    _environment(output, plan["cpu_threads"])
    import torch
    import transformers
    from .magi import MagiAdapter
    from ..codecs import using_ffmpeg
    from ..manager import KVRuntime
    from ..types import ModelIdentity

    if (torch.__version__ != plan["qualified_environment"]["torch"]
            or transformers.__version__ != plan["qualified_environment"]["transformers"]):
        raise RuntimeError("The native launch environment differs from the CPU-qualified environment")
    _write(output / f"execution_rank{os.environ['RANK']}.json", _execution(plan))
    session = _Session(plan, output)
    model_load = session.transport.checkpoint(
        "model_load", details={"rank": session.rank, "seconds": session.model_load_seconds,
                               "memory": _memory(session.device, session.cache)},
    )
    session.transport.owner_call("model_load_record", lambda: _write(output / "model_load.json", model_load))
    raw_identity = ModelIdentity(
        "MAGI-1-24B-distill", _identity({
            "checkpoint": plan["checkpoint_verification"]["index"]["sha256"],
            "source": {name: row["sha256"] for name, row in plan["external_sources"].items()
                       if name.startswith("native/")},
            "config": plan["config"],
        }), LAYERS, C, "tokens_global_channels_ulysses_sp2", "bfloat16",
    )
    runtimes = {}
    runtime = None
    records, gates, audits, later_audits, memory_audits, source_identities = {}, {}, {}, {}, {}, {}
    active_adapter = None
    try:
        with torch.inference_mode(), (
            using_ffmpeg(plan["ffmpeg"]["path"]) if plan["ffmpeg"] is not None else nullcontext()
        ):
            session.timed("noop_clean_prefix", lambda: session.clean(session.prefix, 0))
            reference = session.timed("noop_chunk1", lambda: session.denoise(session.noises[0], 1))
            records["noop_chunk1"] = session.save("noop_chunk1", reference)
            repeated = session.timed("noop_repeat_chunk1", lambda: session.denoise(session.noises[0], 1))
            if not torch.equal(reference, repeated):
                raise RuntimeError("Native no-op/repeat RAW continuation is not exact")
            gates["native_noop_repeat_exact"] = True
            del repeated
            later_reference = None
            if plan["continuations"] == 2:
                session.timed("noop_clean_chunk1", lambda: session.clean(reference, 1))
                later_reference = session.timed("noop_chunk2", lambda: session.denoise(session.noises[1], 2))
                records["noop_chunk2"] = session.save("noop_chunk2", later_reference)
            raw_allocations = session.transport.checkpoint(
                "native_dense_before_release", details=_memory(session.device, session.cache),
            )
            session.cache.key_value_memory_dict.clear()
            gc.collect()
            torch.cuda.empty_cache()
            released = session.transport.checkpoint(
                "native_dense_after_release", details=_memory(session.device, session.cache),
            )
            gates["native_dense_release"] = {"before": raw_allocations, "after": released}
            raw_snapshot = None
            for arm in plan["arms"]:
                family = "native"
                identity = raw_identity

                def open_runtime():
                    if family not in runtimes:
                        runtimes[family] = KVRuntime(
                            output / "stores" / family, identity,
                            max_resident_bytes=plan["resident_mib"] * 1024**2,
                        )
                    return runtimes[family]
                runtime = session.transport.owner_call(f"{arm}_open_canonical_store", open_runtime)
                def new_adapter(branch):
                    return MagiAdapter(
                        session.model, session.cache, runtime, model_identity=identity,
                        branch_id=branch, staging_dir=output / "staging", coordinator=session.transport,
                        specs=_arm_specs(arm), reuse_threshold=None,
                        rotary_apply=session.rotary, device=session.device,
                        native_dtype="bfloat16",
                    )

                adapter = new_adapter(arm)
                active_adapter = adapter
                adapter.attach()
                session.timed(f"{arm}_clean_prefix", lambda: session.clean(session.prefix, 0, adapter))
                session.timed(f"{arm}_publish", adapter.seal_chunk)
                source_identities[arm] = dict(adapter.canonical_source_hashes["0"])
                snapshot = session.timed(f"{arm}_snapshot", lambda: adapter.snapshot(**session.state(session.prefix)))
                if arm == "raw":
                    raw_snapshot = snapshot
                def audit_storage():
                    measured = _storage(runtime, arm, snapshot)
                    _write(output / f"{arm}_storage.json", measured)
                    return {
                        **{key: value for key, value in measured.items() if key != "packets"},
                        "basis_objects": {
                            f"{row['key']['layer']}:{row['key']['target']}": row["basis_object"]
                            for row in measured["packets"]
                        },
                    }
                measured = session.transport.owner_call(f"{arm}_storage", audit_storage)
                audits[arm] = session.transport.broadcast_json(measured, operation=f"{arm}_storage_audit")
                adapter.detach()
                adapter = new_adapter(f"{arm}_resume")
                active_adapter = adapter
                adapter.attach()
                restored = session.timed(
                    f"{arm}_restore", lambda: adapter.restore(snapshot, f"{arm}_resume", generators={"noise": session.generator}),
                )
                if adapter.specs != _arm_specs(arm) or adapter.reuse_threshold is not None:
                    raise RuntimeError("Native restoration changed the requested fixed arm/profile")
                gates[f"{arm}_restored_profile_preserved"] = True
                session.reinstate(restored)
                latent = session.timed(f"{arm}_resume_chunk1", lambda: session.denoise(session.noises[0], 1, adapter))
                records[f"{arm}_resume_chunk1"] = session.save(f"{arm}_resume_chunk1", latent, reference=reference)
                if arm == "raw" and not torch.equal(reference, latent):
                    raise RuntimeError("ReelCache RAW snapshot/restore differs from the native no-op reference")
                if arm == "raw":
                    gates["raw_snapshot_restore_exact"] = True
                    from ..errors import StateError

                    try:
                        adapter.snapshot(**session.state(latent))
                    except StateError:
                        gates["completed_noisy_chunk_snapshot_rejected"] = True
                    else:
                        raise RuntimeError("A noisy, not clean-recaptured chunk was snapshotted")
                    parent_before = session.transport.owner_call("raw_parent_before", lambda: runtime.branch_info("raw"))
                    restored = session.timed(
                        "raw_fork", lambda: adapter.fork(raw_snapshot, "raw_fork", generators={"noise": session.generator}),
                    )
                    state = session.reinstate(restored)
                    forked = session.timed("raw_fork_chunk1", lambda: session.denoise(session.noises[0], 1, adapter))
                    if not torch.equal(reference, forked):
                        raise RuntimeError("RAW fork continuation is not exact")
                    records["raw_fork_chunk1"] = session.save("raw_fork_chunk1", forked, reference=reference)
                    gates["raw_fork_exact"] = True
                    if plan["continuations"] == 2:
                        session.timed("raw_fork_clean_chunk1", lambda: session.clean(forked, 1, adapter))
                        session.timed("raw_fork_publish_chunk1", adapter.seal_chunk)
                        later_snapshot = session.timed("raw_later_snapshot", lambda: adapter.snapshot(**session.state(forked)))
                        session.reinstate(session.timed(
                            "raw_later_restore",
                            lambda: adapter.restore(later_snapshot, "raw_later_resume", generators={"noise": session.generator}),
                        ))
                        later = session.timed("raw_fork_chunk2", lambda: session.denoise(session.noises[1], 2, adapter))
                        if not torch.equal(later_reference, later):
                            raise RuntimeError("Later-chunk RAW resume is not exact")
                        records["raw_fork_chunk2"] = session.save("raw_fork_chunk2", later, reference=later_reference)
                        gates["later_clean_boundary_and_resume_exact"] = True
                        later_storage = session.transport.owner_call(
                            "raw_later_storage",
                            lambda: _storage(runtime, "raw_later_resume", later_snapshot),
                        )
                        if session.rank == 0:
                            _write(output / "raw_later_storage.json", later_storage)
                        later_audits["raw"] = session.transport.broadcast_json(
                            {key: value for key, value in later_storage.items() if key != "packets"}
                            if session.rank == 0 else None,
                            operation="raw_later_storage_summary",
                        )
                        del later
                    parent_after = session.transport.owner_call("raw_parent_after", lambda: runtime.branch_info("raw"))
                    session.transport.checkpoint(
                        "raw_parent_unchanged",
                        error=("Child branch mutation changed the parent"
                               if session.rank == 0 and parent_before != parent_after else None),
                    )
                    gates["fork_parent_unchanged"] = True
                    del forked, state
                elif plan["continuations"] == 2:
                    session.timed(f"{arm}_clean_chunk1", lambda: session.clean(latent, 1, adapter))
                    session.timed(f"{arm}_publish_chunk1", adapter.seal_chunk)
                    later_snapshot = session.timed(
                        f"{arm}_later_snapshot", lambda: adapter.snapshot(**session.state(latent)),
                    )
                    session.reinstate(session.timed(
                        f"{arm}_later_restore",
                        lambda: adapter.restore(
                            later_snapshot, f"{arm}_later_resume", generators={"noise": session.generator},
                        ),
                    ))
                    later = session.timed(
                        f"{arm}_resume_chunk2", lambda: session.denoise(session.noises[1], 2, adapter),
                    )
                    records[f"{arm}_resume_chunk2"] = session.save(
                        f"{arm}_resume_chunk2", later, reference=later_reference,
                    )
                    gates[f"{arm}_later_clean_resume_completed"] = True
                    later_storage = session.transport.owner_call(
                        f"{arm}_later_storage",
                        lambda: _storage(runtime, f"{arm}_later_resume", later_snapshot),
                    )
                    if session.rank == 0:
                        _write(output / f"{arm}_later_storage.json", later_storage)
                    later_audits[arm] = session.transport.broadcast_json(
                        {key: value for key, value in later_storage.items() if key != "packets"}
                        if session.rank == 0 else None,
                        operation=f"{arm}_later_storage_summary",
                    )
                    del later
                measurements = session.transport.checkpoint(f"{arm}_physical_memory", details=adapter.memory_report())
                if any(row["live_native_cache_storage_bytes"] for row in measurements):
                    raise RuntimeError("Native raw buffers survived bounded layer acquisition")
                session.transport.owner_call(
                    f"{arm}_memory_publish", lambda: _write(output / f"{arm}_memory.json", measurements),
                )
                adapter.detach()
                active_adapter = None
                del latent, restored, adapter
                gc.collect()
                torch.cuda.empty_cache()
                allocator_released = session.transport.checkpoint(
                    f"{arm}_allocator_release", details=_memory(session.device, session.cache),
                )
                memory_audits[arm] = {"before_allocator_release": measurements,
                                      "after_allocator_release": allocator_released}
                session.transport.owner_call(
                    f"{arm}_allocator_release_publish",
                    lambda: _write(output / f"{arm}_allocator_release.json", allocator_released),
                )
            if "svd" in audits and "svd_hevc" in audits:
                gates["svd_and_hevc_same_clean_sources"] = source_identities["svd"] == source_identities["svd_hevc"]
                if not gates["svd_and_hevc_same_clean_sources"]:
                    raise RuntimeError("SVD-only and HEVC consumed different clean pre-RoPE/value sources")
                gates["svd_and_hevc_same_canonical_bases"] = (
                    audits["svd"]["basis_objects"] == audits["svd_hevc"]["basis_objects"]
                )
                if not gates["svd_and_hevc_same_canonical_bases"]:
                    raise RuntimeError("SVD-only and HEVC used different rounded canonical bases")
            result = {
                "status": "native_latent_verified_RGB_pending", "plan_identity": plan["identity"],
                "case": plan["case_id"], "mode": "serial_clean_history", "gates": gates,
                "latents": records, "profile_storage": {name: {
                    key: value for key, value in audit.items() if key != "packets"
                } for name, audit in audits.items()},
                "later_profile_storage": later_audits,
                "first_chunk_canonical_source_hashes": source_identities,
                "prefix_noise_sha256": session.prefix_noise_sha256,
                "blackwell_silu_fallback": session.blackwell_silu,
                "executed_model_source": session.executed_model_source,
                "profile_memory": memory_audits,
                "worker_execution_records": [
                    _file_row(output / f"execution_rank{rank}.json") for rank in range(2)
                ],
                "quality_guarantee": None, "pipelined_performance_claim": False,
                "limitations": plan["limitations"],
            }
            session.transport.owner_call("native_complete", lambda: _write(output / "native_complete.json", result))
    finally:
        if active_adapter is not None and not active_adapter.phase.startswith("active_"):
            active_adapter.detach()
        for owned_runtime in runtimes.values():
            owned_runtime.close()
        session.fallback.close()
        if session.dist.is_initialized():
            session.dist.destroy_process_group()


def decode(args):
    if not args.gpu_grant or int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise RuntimeError("VAE decoding needs a separate one-GPU grant after both SP2 workers exit")
    output = _output(args.output)
    plan = _verify_plan(output)
    result = _read(output / "native_complete.json")
    if result["plan_identity"] != plan["identity"]:
        raise ValueError("Latents belong to a different source-bound native run")
    _require_workers_exited(result)
    _environment(output, plan["cpu_threads"])
    sys.path.insert(0, plan["native_source"])
    os.environ["SPECIAL_TOKEN_PATH"] = str(Path(plan["native_source"]) / "example/assets/special_tokens.npz")
    import numpy as np
    import torch
    from safetensors.torch import load_file
    from inference.model.vae.vae_model import ViTVAE
    from inference.pipeline.video_process import VaeHelper

    weight = Path(plan["vae"]["path"]) / "diffusion_pytorch_model.safetensors"
    if _sha(weight) != plan["vae"]["expected_checkpoint_sha256"]:
        raise ValueError("Protected native VAE weights changed")
    config_path = Path(plan["vae"]["path"]) / "config.json"
    if _sha(config_path) != plan["vae"]["config_sha256"]:
        raise ValueError("Protected native VAE configuration changed")
    config = _read(config_path)
    original_linspace = torch.linspace

    def cpu_linspace(*positional, **keywords):
        keywords["device"] = "cpu"
        return original_linspace(*positional, **keywords)

    with patch.object(torch, "linspace", cpu_linspace), torch.device("meta"):
        vae = ViTVAE(ddconfig=config["ddconfig"], model_type=config["model_type"])
    vae.load_state_dict(load_file(str(weight), device="cpu"), strict=True, assign=True)
    device = torch.device("cuda", args.device)
    torch.cuda.set_device(device)
    torch.set_num_threads(plan["cpu_threads"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.nn.Module.to(vae.eval(), device=device, dtype=torch.bfloat16)
    _write(output / "execution_decode.json", _execution(plan))
    media = {}
    for name, record in result["latents"].items():
        if name == "raw_fork_chunk1":
            continue
        if _sha(record["artifact"]["path"]) != record["artifact"]["sha256"]:
            raise ValueError("A saved continuation latent changed")
        latent = torch.load(record["artifact"]["path"], map_location=device, weights_only=True)["latent"]
        if _tensor_hash(latent) != record["tensor_sha256"]:
            raise ValueError("A saved continuation tensor changed")
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
        seconds = time.perf_counter() - started
        if tuple(value.shape) != (24, 3, 1088, 1920) or not torch.isfinite(value).all():
            raise RuntimeError("Native per-chunk VAE returned unexpected geometry or nonfinite RGB")
        pixels = (value.float() * 127.5 + 127.5).clamp(0, 255).to(torch.uint8)
        frames = pixels[:, :, 4:1084].permute(0, 2, 3, 1).contiguous().cpu().numpy()
        path = output / "rgb" / f"{name}.npz"
        path.parent.mkdir(exist_ok=True)
        with path.open("xb") as handle:
            np.savez_compressed(handle, frames=frames)
        media[name] = {
            "artifact": _file_row(path), "shape": list(frames.shape),
            "rgb_sha256": hashlib.sha256(memoryview(frames)).hexdigest(),
            "native_decode_seconds": seconds, "crop_rows": [4, 1084], "fps": 24,
            "latent_sha256": record["tensor_sha256"], "VAE_state": "stateless_chunk_decode",
            "memory": _memory(device),
        }
        _write(output / "rgb" / f"{name}.json", media[name])
        del latent, value, pixels, frames
    _write(output / "decode_complete.json", {
        "plan_identity": plan["identity"], "media": media,
        "vae_weight_sha256_verified": _sha(weight), "quality_guarantee": None,
    })


def report(args):
    import inspect
    import numpy as np
    import skimage
    from skimage.metrics import structural_similarity

    output = _output(args.output)
    plan = _verify_plan(output)
    native = _read(output / "native_complete.json")
    decoded = _read(output / "decode_complete.json")
    if any(row["plan_identity"] != plan["identity"] for row in (native, decoded)):
        raise ValueError("Native and VAE artifacts belong to different verification plans")
    artifacts = {}
    for name, row in decoded["media"].items():
        if _sha(row["artifact"]["path"]) != row["artifact"]["sha256"]:
            raise ValueError("A decoded RGB artifact changed")
        with np.load(row["artifact"]["path"], allow_pickle=False) as data:
            frames = data["frames"]
            if frames.dtype != np.uint8 or list(frames.shape) != [24, 1080, 1920, 3]:
                raise ValueError("Unexpected RGB shape, dtype or frame count")
            if hashlib.sha256(memoryview(np.ascontiguousarray(frames))).hexdigest() != row["rgb_sha256"]:
                raise ValueError("Decoded RGB tensor hash changed")
        artifacts[name] = row["artifact"]["path"]
    pairs = [
        ("raw_resume_vs_noop", "noop_chunk1", "raw_resume_chunk1"),
        ("raw_later_vs_noop", "noop_chunk2", "raw_fork_chunk2"),
        ("svd_projection_vs_raw", "raw_resume_chunk1", "svd_resume_chunk1"),
        ("svd_plus_hevc_vs_raw", "raw_resume_chunk1", "svd_hevc_resume_chunk1"),
        ("incremental_hevc_vs_svd", "svd_resume_chunk1", "svd_hevc_resume_chunk1"),
        ("later_svd_vs_raw", "raw_fork_chunk2", "svd_resume_chunk2"),
        ("later_svd_plus_hevc_vs_raw", "raw_fork_chunk2", "svd_hevc_resume_chunk2"),
        ("later_hevc_vs_svd", "svd_resume_chunk2", "svd_hevc_resume_chunk2"),
    ]
    comparisons = {}
    for label, reference, candidate in pairs:
        if reference not in artifacts or candidate not in artifacts:
            continue
        with np.load(artifacts[reference], allow_pickle=False) as first, np.load(
            artifacts[candidate], allow_pickle=False,
        ) as second:
            a, b = first["frames"], second["frames"]
            scores = [float(structural_similarity(x, y, channel_axis=-1, data_range=255)) for x, y in zip(a, b)]
            comparisons[label] = {
                "reference": reference, "candidate": candidate, "rgb_bitwise_equal": bool(np.array_equal(a, b)),
                "ssim_mean": float(np.mean(scores)), "ssim_min": min(scores), "per_frame_ssim": scores,
                "interpretation": "paired diagnostic; SSIM differences are not additive loss estimates",
            }
    result = {
        "schema": "reelcache_native_magi_verified_report_v1", "plan_identity": plan["identity"],
        "native_gates": native["gates"], "comparisons": comparisons,
        "profile_storage": native["profile_storage"], "quality_guarantee": None,
        "later_profile_storage": native["later_profile_storage"],
        "later_comparison_scope": "Includes propagated latent differences and each arm's own clean recapture; not same-source codec isolation",
        "profile_memory": native["profile_memory"],
        "timings": {path.stem: _read(path) for path in sorted((output / "timings").glob("*.json"))},
        "mode": "serial_clean_history", "pipelined_performance_claim": False,
        "ssim_method": "skimage structural_similarity uint8 RGB, channel_axis=-1, data_range=255; mean24frames",
        "ssim_implementation": {
            "scikit_image_version": skimage.__version__,
            "source": _file_row(inspect.getsourcefile(structural_similarity)),
            "function": f"{structural_similarity.__module__}.{structural_similarity.__name__}",
        },
        "limitations": plan["limitations"],
    }
    _write(output / "report.json", result)
    print(json.dumps({"status": "complete", "output": str(output / "report.json"),
                      "ssim": {key: row["ssim_mean"] for key, row in comparisons.items()},
                      "quality_guarantee": None}))


@contextmanager
def _study_native_fixes(model, attention_observer=None):
    """Scope the already-qualified completed-receive/row-visible attention seam."""
    import torch
    from types import MethodType
    from einops import rearrange
    from flash_attn.flash_attn_interface import flash_attn_func
    from inference.infra.parallelism.context_parallel import UlyssesScheduler, all_to_all_output_split

    def completed_receive(query, key, value, core_attn_func, cross_attn_func,
                          overlap_degree, batch_size, cp_size, cp_split_sizes=None):
        if overlap_degree != 1:
            raise ValueError("The study supports only the bound overlap-degree-one Ulysses path")
        output, handle = all_to_all_output_split(core_attn_func(query, key, value), cp_split_sizes)
        cross = cross_attn_func()
        handle.wait()
        return rearrange(output, "(cp sq b) hn hd -> sq b (cp hn hd)", cp=cp_size, b=batch_size), cross

    def attention(self, query, key, value, bs, meta_args, *, layer):
        if bs != 1:
            raise ValueError("The study's native attention contract is batch one")
        if attention_observer is not None:
            attention_observer(layer, key, value, meta_args)
        values = [item.reshape(-1, bs, item.shape[1], item.shape[2]).transpose(0, 1).contiguous()
                  for item in (query, key, value)]
        query, key, value = values
        outputs = []
        for row in range(meta_args.denoising_range_num):
            q0, q1 = meta_args.core_attn_params.np_q_range[row]
            k0, k1 = meta_args.core_attn_params.np_k_range[row]
            if not 0 <= q0 < q1 <= query.shape[1] or not 0 <= k0 < k1 <= key.shape[1]:
                raise ValueError("Native attention visibility ranges changed")
            result = flash_attn_func(
                q=query[:, q0:q1], k=key[:, k0:k1], v=value[:, k0:k1],
                deterministic=torch.are_deterministic_algorithms_enabled(),
            )
            outputs.append(result.transpose(0, 1).reshape(-1, result.shape[2], result.shape[3]))
        return torch.cat(outputs, dim=0).contiguous()

    with ExitStack() as stack:
        stack.enter_context(patch.object(UlyssesScheduler, "get_attn_and_xattn_base",
                                         staticmethod(completed_receive)))
        for index, layer in enumerate(model.videodit_blocks.layers):
            obj = layer.self_attention

            def bound_attention(self, query, key, value, bs, meta_args, _layer=index):
                return attention(self, query, key, value, bs, meta_args, layer=_layer)

            stack.enter_context(patch.object(obj, "core_attention", MethodType(bound_attention, obj)))
        if torch.cuda.get_device_capability()[0] >= 12:
            import flashinfer
            import torch.nn.functional as functional

            def silu(value, out=None, enable_pdl=None):
                gate, up = value.chunk(2, dim=-1)
                result = functional.silu(gate) * up
                if out is None:
                    return result
                out.copy_(result)
                return out

            stack.enter_context(patch.object(flashinfer.activation, "silu_and_mul", silu))
        yield


def _align_promoted_timestep_input(model):
    """Match this instance's explicit input cast to its promoted timestep weights."""
    import torch

    embedder = model.t_embedder
    if not hasattr(embedder, "data_type"):
        raise ValueError("The native timestep embedder no longer declares its input dtype")
    dtypes = {parameter.dtype for parameter in embedder.parameters() if parameter.is_floating_point()}
    if len(dtypes) != 1 or next(iter(dtypes)) not in (torch.float32, torch.bfloat16):
        raise ValueError("The native promoted timestep weights must have one supported dtype")
    embedder.data_type = next(iter(dtypes))


class _StudyBackend:
    """Fresh generation with original observations bound to the admitted launch receipt."""

    def __init__(self, verified, output, config, transport, device):
        import torch
        from safetensors.torch import load_file
        from inference.model.dit.dit_model import VideoDiTModel, _high_precision_promoter
        from inference.model.dit.dit_module import LearnableRotaryEmbeddingCat, flash_apply_rotary_emb
        from inference.model.t5.t5_model import T5Embedder
        from inference.pipeline.video_generate import init_t, init_intervel
        from ..study import study_cell
        from ..errors import CapacityError

        free, _ = torch.cuda.mem_get_info(device)
        required = max(row["device_bytes"] for row in verified["admissions"])
        admission_error = None if required <= free else CapacityError(
            "Physical free HBM cannot admit the complete model/workspace/history path before model loading",
        )
        transport.checkpoint("magi_model_request_capacity", error=admission_error)

        self.torch, self.verified, self.output, self.config = torch, verified, output, config
        self.transport, self.device = transport, device
        self.cell = study_cell(verified["batch"]["cell_id"])
        self.active = None
        self.observation_evidence_kind = "native"
        self.model = self.encoder = None
        self.fixes = ExitStack()
        self.rotary = flash_apply_rotary_emb
        self.schedule = init_t({}, 16, device, shortcut_mode=config.engine_config.shortcut_mode)
        self.intervals = init_intervel(16, device, shortcut_mode=config.engine_config.shortcut_mode)
        _validate_study_timetable(self.schedule, self.intervals)
        try:
            with torch.device("meta"), patch.object(
                LearnableRotaryEmbeddingCat, "get_default_bands",
                lambda self: torch.empty(self.dim // 8, device="meta"),
            ):
                model = VideoDiTModel(config)
            state = {}
            for name in verified["checkpoint"]["shards"]:
                part = load_file(str(verified["asset_root"] / name), device="cpu")
                if state.keys() & part.keys():
                    raise ValueError("Duplicate parameters in the exact indexed native shards")
                for key, value in part.items():
                    expected = verified["checkpoint"]["tensors"][key]
                    dtype = torch.float32 if expected["dtype"] == "F32" else torch.bfloat16
                    if value.dtype != dtype or list(value.shape) != expected["shape"]:
                        raise ValueError("Loaded native tensor differs from its verified mixed-precision header")
                state.update(part)
                del part
            if set(state) != set(verified["checkpoint"]["tensors"]):
                raise ValueError("Loaded native parameters differ from the exact checkpoint index")
            model.load_state_dict(state, strict=True, assign=True)
            if any(value.is_meta for value in model.parameters()):
                raise ValueError("Unloaded meta parameters survived native checkpoint assignment")
            del state
            self.model = _high_precision_promoter(model.to(device=device)).eval()
            _align_promoted_timestep_input(self.model)
            self.fixes.enter_context(_study_native_fixes(
                self.model, self._observe_attention if verified.get("preflight_resources") else None,
            ))
            if verified.get("ffmpeg") is not None:
                from ..codecs import using_ffmpeg

                self.fixes.enter_context(using_ffmpeg(verified["ffmpeg"]))
            if transport.is_owner:
                directory = verified["asset_root"] / verified["batch"]["asset_layout"]["t5_directory"]
                self.encoder = T5Embedder(
                    "cpu", dir_or_name=directory.name, local_cache=True, cache_dir=str(directory.parent),
                    torch_dtype=torch.float32, model_max_length=800,
                    t5_model_kwargs={
                        "low_cpu_mem_usage": True, "torch_dtype": torch.float32,
                        "device_map": {"shared": "cpu", "encoder": "cpu"}, "local_files_only": True,
                    },
                )
        except BaseException:
            self.close()
            raise

    def assert_clean(self):
        if self.active is not None or getattr(self.model, "_reelcache_magi_adapter", None) is not None:
            raise RuntimeError("MAGI request hooks/caches survived cleanup; abort both ranks")
        if self.model is not None and any(
            layer.self_attention.attn_weights_history for layer in self.model.videodit_blocks.layers
        ):
            raise RuntimeError("MAGI attention history survived request cleanup")

    def new_request(self, request, namespace):
        from inference.common import InferenceParams
        from .magi_rank_worker import magi_model_identity, serial_sampler_contract

        self.assert_clean()
        t = self.torch
        if self.device.type == "cuda":
            t.cuda.reset_peak_memory_stats(self.device)
        directory = self.output / (request.output_namespace or f"requests/{namespace}")
        if not directory.resolve().is_relative_to(self.output):
            raise ValueError("The request output namespace escapes its new admitted epoch")
        self.active = state = {"request": request, "namespace": namespace, "directory": directory,
                               "runtime": None, "adapter": None, "latents": [],
                               "trajectory_id": request.trajectory_id or namespace,
                               "deferred_observations": {}}
        self.transport.owner_call("magi_request_directory", lambda: directory.mkdir(parents=True, exist_ok=False))
        caption = mask = None

        def encode_prompt():
            with t.inference_mode():
                return self.encoder.get_text_embeddings([request.prompt])

        text = self.transport.owner_call("magi_request_encode_official_prompt", encode_prompt)
        for name, item in zip(("caption", "mask"), text if text is not None else (None, None)):
            local = None
            for rank in range(self.transport.world_size):
                received = self.transport.transfer_tensor(
                    item, source=self.transport.owner_rank, destination=rank,
                    operation=f"magi_request_conditioning:{name}:{rank}",
                )
                if rank == self.transport.rank:
                    local = received.to(self.device)
            if name == "caption":
                caption = local
            else:
                mask = local
        if (tuple(caption.shape) != (1, 800, 4096) or tuple(mask.shape) != (1, 800)
                or caption.dtype != t.float32 or not t.isfinite(caption).all()):
            raise ValueError("Native T5 conditioning differs from its checkpoint contract")
        null_caption = self.model.y_embedder.null_caption_embedding.detach().unsqueeze(0).clone()
        null_mask = t.zeros_like(mask)
        null_mask[:, :50] = 1
        state.update(
            y=t.cat((caption, null_caption)).unsqueeze(1), masks=t.cat((mask, null_mask)).unsqueeze(1),
            null_caption=null_caption, null_mask=null_mask,
            generator=t.Generator(device=self.device).manual_seed(request.seed),
            cache=InferenceParams(max_batch_size=1, max_sequence_length=request.chunks * self.cell.tokens),
            noise_hashes=[],
        )
        policy = None
        if request.arm == "adaptive":
            from ..online_rank import RankPolicy

            policy = RankPolicy.from_dict(request.policy)
        context = {
            **serial_sampler_contract(),
            "protocol_id": self.verified["protocol"]["protocol_id"], "cell_id": self.cell.cell_id,
            "request_id": request.request_id, "request_namespace": namespace,
            "case_id": request.case_id, "trajectory_id": state["trajectory_id"],
            "prompt_id": request.prompt_id, "group_id": request.group_id, "split": request.split,
            "prompt": request.prompt, "seed": request.seed, "arm": request.arm,
            "arm_id": request.arm_id, "source_fps": request.source_fps,
            "rank_selection_mode": request.rank_selection_mode,
            "uniform_rank_fields_used": request.arm in ("svd", "svd_hevc") and request.layer_ranks is None,
            "rank_k": request.rank_k, "rank_v": request.rank_v,
            "layer_ranks": request.layer_rank_map,
            "rank_policy_hash": policy.fingerprint if policy is not None else None,
            "source_observation_mode": (
                "native_attention_digests_only" if self.verified.get("preflight_backend") == "native_noop" else
                "disabled" if not request.observe_sources else
                "exact_policy_reuse" if policy is not None and policy.solver == "exact_oracle"
                else "raw_source_spectrum" if request.arm == "raw"
                else "explicit_offline_spectrum"
            ),
            "output_namespace": request.output_namespace,
            "conditioning_sha256": _tensor_hash(caption), "conditioning_mask_sha256": _tensor_hash(mask),
            "asset_id": self.verified["assets"]["asset_id"],
            "launch_closure_id": self.verified["receipt"]["source_closure"]["closure_id"],
            "config_id": _identity(self.verified["config"]), "geometry": self.cell.to_dict(),
            "from_scratch": True, "skipped_initial_noise": False,
            "fixture": False, "native_execution": True,
            "preflight_backend": self.verified.get("preflight_backend", "reelcache"),
            "collective_timeout_seconds": self.verified["collective_timeout_seconds"],
            "collective_timeout_is_latency_proof": False,
        }
        state["source_context"] = context
        model_identity = magi_model_identity(
            cell_id=self.cell.cell_id, asset_id=context["asset_id"], config_id=context["config_id"],
            source_closure_id=context["launch_closure_id"],
        )
        self._install_request_adapter(state, request, model_identity, policy)
        self._prepare_input_trial(state)
        return state

    def _install_request_adapter(self, state, request, model_identity, policy):
        if self.verified.get("preflight_backend", "reelcache") == "native_noop":
            if request.arm != "raw" or self.verified["receipt"]["phase"] != "native_preflight":
                raise ValueError("Independent native cache is restricted to RAW native preflight")
            return
        from ..manager import KVRuntime
        from ..types import CodecSpec
        from .magi import MagiAdapter, MagiGeometry

        directory, namespace = state["directory"], state["namespace"]
        state["runtime"] = self.transport.owner_call(
            "magi_request_owner_runtime", lambda: KVRuntime(
                directory / "store", model_identity,
                max_resident_bytes=self.verified["batch"]["reconstruction_bytes"],
                max_store_bytes=self.verified["batch"]["store_bytes"],
            ),
        )
        specs = CodecSpec() if request.arm in ("raw", "adaptive") else request.static_specs()
        state["adapter"] = MagiAdapter(
            self.model, state["cache"], state["runtime"], model_identity=model_identity,
            branch_id=namespace, staging_dir=directory / "capture",
            geometry=MagiGeometry.for_cell(self.cell.cell_id), coordinator=self.transport,
            rotary_apply=self.rotary, device=self.device, specs=specs,
            reuse_threshold=None, source_observer=self._observe if request.observe_sources else None,
            source_context=state["source_context"], resource_budget=self.verified["budget"],
            cache_mode=self.verified["batch"].get("cache_mode", "runtime"),
            quality_hot_bytes=self.verified["batch"].get("quality_hot_bytes", 0),
            max_layer_buffer_bytes=self.verified["batch"]["native_layer_bytes"],
            rank_policy=policy,
        )
        state["adapter"].attach()

    def _prepare_input_trial(self, state):
        from .magi import MagiGeometry
        from .magi_preflight import MagiInputBundle, AttentionDigestRecorder

        plans = self.verified.get("input_bundles", {})
        if not plans:
            return
        started = time.perf_counter()
        plan = plans[state["request"].request_id]
        overhead = self.verified["preflight_resources"][state["request"].request_id]

        def prepare():
            bundle = MagiInputBundle(
                mode=plan["mode"], binding=plan["binding"],
                directory=state["directory"] / "inputs",
                path=plan.get("path"), sha256=plan.get("sha256"),
                max_bytes=overhead["input_bundle_bytes"],
            )
            state["input_bundle"] = bundle
            return bundle.conditioning({name: state[name] for name in (
                "y", "masks", "null_caption", "null_mask",
            )})

        record = self.transport.owner_call("magi_reencoded_input_conditioning", prepare)
        state["conditioning_input"] = self.transport.broadcast_json(record, operation="magi_conditioning_input_record")
        state["input_records"] = []
        state["input_binding_id"] = _identity(plan["binding"])
        error = None
        try:
            state["attention_recorder"] = AttentionDigestRecorder(
                state["directory"] / f"attention_rank{self.transport.rank}.jsonl",
                geometry=MagiGeometry.for_cell(self.cell.cell_id), rank=self.transport.rank,
                binding_id=state["input_binding_id"], max_chunks=state["request"].chunks,
                max_records=overhead["max_attention_records_per_rank"],
                max_bytes=overhead["attention_digest_bytes"] // self.transport.world_size,
            )
        except BaseException as exc:
            error = exc
        self.transport.checkpoint("magi_attention_record_admission", error=error)
        state["input_instrumentation_seconds"] = time.perf_counter() - started

    def _preflight_noise(self, state, index, noise):
        if "input_records" not in state:
            return noise
        started = time.perf_counter()
        postdraw = state["generator"].get_state()
        states = self.transport.checkpoint("magi_postdraw_private_rng", details=_tensor_hash(postdraw))
        if len(set(states)) != 1:
            raise RuntimeError("SP ranks disagree on the actual postdraw private generator state")
        pair = self.transport.owner_call(
            "magi_explicit_chunk_input",
            lambda: state["input_bundle"].noise(index, noise, postdraw),
        )
        record = self.transport.broadcast_json(
            pair[0] if self.transport.is_owner else None, operation="magi_chunk_input_record",
        )
        explicit = None
        for rank in range(self.transport.world_size):
            received = self.transport.transfer_tensor(
                pair[1] if self.transport.is_owner else None,
                source=self.transport.owner_rank, destination=rank,
                operation=f"magi_explicit_input:{index}:{rank}",
            )
            if rank == self.transport.rank:
                explicit = received.to(self.device)
        state["input_records"].append(record)
        state["input_instrumentation_seconds"] = state.get("input_instrumentation_seconds", 0.0) + (
            time.perf_counter() - started
        )
        return explicit

    def _observe_attention(self, layer, key, value, meta_args):
        started = time.perf_counter()
        state = self.active
        if state is None or "attention_recorder" not in state:
            raise RuntimeError("Native attention entered before its admitted preflight record was prepared")
        point = state["attention_point"]
        error = None
        try:
            def ranges(value):
                return value.tolist() if hasattr(value, "tolist") else [list(pair) for pair in value]

            if meta_args.slice_point != point["chunk_index"]:
                raise ValueError("Native attention cursor differs from the input-bundle chunk")
            state["attention_recorder"].record(
                point, layer, key, value,
                query_ranges=ranges(meta_args.core_attn_params.np_q_range),
                key_ranges=ranges(meta_args.core_attn_params.np_k_range),
                near_clean=bool(meta_args.distill_nearly_clean_chunk),
            )
        except BaseException as exc:
            error = exc
        self.transport.checkpoint(
            f"magi_attention_digest:{point['chunk_index']}:{point['phase']}:{point['step_index']}:{layer}",
            error=error,
        )
        state["attention_instrumentation_seconds"] = state.get("attention_instrumentation_seconds", 0.0) + (
            time.perf_counter() - started
        )

    def _observe(self, description, matrix):
        """Bind original-source identity; exact-policy arms return metadata without solving."""
        from ..spectra import exact_spectrum
        from ..rank_analysis import native_magi_source
        from ..types import TensorContext, canonical_dtype
        from ..errors import IntegrityError
        from .magi import NATIVE_TENSOR_HASH_FORMAT, _tensor_hash_format

        request = self.active["request"]
        receipt = self.verified.get("receipt")
        census = isinstance(receipt, Mapping) and receipt.get("phase") == "calibration_census"
        if census:
            if request.split != "calibration" or description.get("split") != "calibration":
                raise IntegrityError("RAW detector-bank acquisition is restricted to the calibration split")
            if request.arm != "raw" or self.verified.get("detector_spec") is None:
                raise IntegrityError("RAW census requires its admitted fixed detector before source inspection")
            threads = self.verified["batch"].get("cpu_threads")
            if type(threads) is not int or threads != self.verified["detector_spec"].cpu_threads:
                raise IntegrityError("RAW detector CPU threads differ from the admitted worker count")
            self.verified["detector_spec"].validate_runtime()
        native = self.observation_evidence_kind == "native"
        if native or receipt is not None:
            try:
                admitted_closure = receipt["source_closure"]["closure_id"]
            except (KeyError, TypeError) as exc:
                raise IntegrityError("Original MAGI observation requires its admitted launch closure") from exc
            if (type(admitted_closure) is not str or len(admitted_closure) != 64
                    or any(char not in "0123456789abcdef" for char in admitted_closure)
                    or description.get("launch_closure_id") != admitted_closure):
                raise IntegrityError("Original MAGI observation differs from its admitted launch closure")
        context = TensorContext.from_dict(description["tensor_context"])
        expected_space = "pre_rope" if context.key.target == "K" else "none"
        if (context.key_space != expected_space or context.dtype != description["actual_dtype"]
                or context.shape != tuple(matrix.shape) or description["shape"] != list(matrix.shape)
                or canonical_dtype(matrix.dtype) != context.dtype or matrix.device.type != "cpu"
                or description.get("source_hash_format") != NATIVE_TENSOR_HASH_FORMAT
                or description.get("source_hash_descriptor") != _tensor_hash_format(matrix.dtype)
                or description.get("fixture") is not (not native)
                or description.get("native_execution") is not native
                or type(description.get("seed")) is not int
                or any(name in description and type(description[name]) is not int for name in ("boundary", "layer"))
                or description["source_sha256"] != _tensor_hash(matrix)):
            raise IntegrityError("Original MAGI observation context, shape, dtype, domain or byte hash disagrees")
        if (description.get("case_id") != request.case_id
                or description.get("trajectory_id") != self.active["trajectory_id"]):
            raise IntegrityError("A MAGI observation cannot relabel its bound paired case or current trajectory")
        selector = f"{request.request_id}:{description['chunk_id']}:{context.key.layer}:{context.key.target}"
        retain = selector in self.verified["batch"].get("retain_originals", [])
        source, observation_identity = native_magi_source(
            description, case_id=request.case_id, trajectory_id=self.active["trajectory_id"],
            evidence_kind=self.observation_evidence_kind,
            retention="exact-input-retained" if retain else "spectrum-only",
        )
        prefix = self.active["directory"] / "spectra" / f"chunk{description['chunk_id']}"
        label = f"layer{context.key.layer:03d}_{context.key.target}"
        original = None
        if retain:
            path = prefix / f"{label}.pt"
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as stream:
                self.torch.save({"original": matrix.detach().cpu()}, stream)
            original = _file_row(path)
        binding = {
            **source.to_dict()["provenance"], "source_identity": source.identity,
            "source_retention": source.retention,
        }
        if sys.byteorder == "little":
            binding["original_bytes_sha256"] = description["source_sha256"]
        if original is not None:
            binding["retained_original_artifact"] = original
        captured = {
            "description": description, "identity": observation_identity, "source": source.to_dict(),
            "binding": binding, "original": original, "prefix": str(prefix), "label": label,
        }
        policy = getattr(request, "policy", None)
        if policy is not None and policy.get("solver") == "exact_oracle":
            deferred = self.active.setdefault("deferred_observations", {})
            if context.key in deferred:
                raise IntegrityError("An exact-policy source was captured twice before its atomic commit")
            deferred[context.key] = captured
            return binding
        started = time.perf_counter()
        result = exact_spectrum(
            matrix, source=source, tile_rows=self.verified["batch"].get("spectrum_tile_rows", 1024),
            max_workspace_bytes=self.verified["batch"]["spectrum_workspace_bytes"],
        )
        oracle_seconds = time.perf_counter() - started
        detector, detector_seconds = None, 0.0
        if census:
            from ..factorization import acquire_anchor_directions
            from ..study import write_json_new
            from ..types import json_copy

            started = time.perf_counter()
            try:
                acquisition = acquire_anchor_directions(
                    matrix, context, spec=self.verified["detector_spec"],
                    max_workspace_bytes=self.verified["batch"]["detector_workspace_bytes"],
                    solver_workspace_bytes=self.verified["batch"]["detector_solver_workspace_bytes"],
                )
                detector = acquisition.to_dict()
                del acquisition
            except BaseException as exc:
                prefix.mkdir(parents=True, exist_ok=True)
                write_json_new(prefix / f"{label}.detector-failed.json", {
                    "schema": "reelcache.magi_detector_failure.v1", "status": "failed",
                    "capture": description, "observation_identity": observation_identity,
                    "detector_spec": self.verified["detector_spec"].to_dict(),
                    "outer_seconds": time.perf_counter() - started,
                    "actual_failure_metrics": (
                        json_copy(exc.metrics) if getattr(exc, "metrics", None) is not None else None
                    ),
                    "error": str(exc), "included_in_full_request_wall_time": True,
                })
                raise
            detector_seconds = time.perf_counter() - started
        self._persist_observation(
            result, captured, oracle_seconds=oracle_seconds,
            detector=detector, detector_seconds=detector_seconds,
        )
        return binding

    def _persist_observation(self, result, captured, *, oracle_seconds, policy_metrics=None,
                             detector=None, detector_seconds=0.0):
        from ..rank_analysis import write_spectrum_observation

        description, observation_identity = captured["description"], captured["identity"]
        prefix, label, original = Path(captured["prefix"]), captured["label"], captured["original"]
        shared = policy_metrics is not None
        prefix.mkdir(parents=True, exist_ok=True)
        spectrum_path = prefix / f"{label}.spectrum.json"
        observation_path = prefix / f"{label}.rank.json"
        persisted = time.perf_counter()
        receipt = write_spectrum_observation(
            result, identity=observation_identity,
            spectrum_path=spectrum_path, observation_path=observation_path,
        )
        persistence_seconds = time.perf_counter() - persisted
        detector_artifact = None
        detector_work_record = None
        detector_persistence_seconds = 0.0
        if detector is not None:
            from ..rank_analysis import detector_records_from_acquisition
            from ..study import write_json_new

            persisted = time.perf_counter()
            rows = detector_records_from_acquisition(detector, observation=receipt["observation"])
            path = prefix / f"{label}.detector.json"
            write_json_new(path, {
                "schema": "reelcache_rank_detector_observations_v1",
                "detector_spec": detector["detector_spec"],
                "detector_sha256": detector["detector_fingerprint"], "observations": rows,
            })
            detector_artifact = _file_row(path)
            detector_persistence_seconds = time.perf_counter() - persisted
            detector_work_record = {
                "acquisition_sha256": rows[0]["acquisition_sha256"],
                "acquisition_count": 1, "thresholds": [.95, .99],
                "metrics": detector["metrics"], "outer_seconds": detector_seconds,
                "scope": "one acquisition for this original matrix and both threshold rows",
                "count_once_for_both_threshold_rows": True,
            }
        _write(prefix / f"{label}.json", {
            "schema": "reelcache.magi_clean_observation.v2", "capture": description,
            "spectrum_artifact": {
                "path": str(spectrum_path), "sha256": receipt["spectrum_sha256"],
                "bytes": spectrum_path.stat().st_size,
            },
            "rank_observation_artifact": {
                "path": str(observation_path), "sha256": receipt["observation_sha256"],
                "bytes": observation_path.stat().st_size,
            },
            "observation_identity": observation_identity,
            "original_artifact": original, "offline_oracle_seconds": oracle_seconds,
            "source_retention": result.source.retention,
            "separately_retained_original": original is not None,
            "spectrum_persistence_seconds": persistence_seconds,
            "persistence_scope": "spectrum and numeric observation validation plus atomic publication",
            "included_in_online_selector_cost": shared,
            "artifact_export_included_in_online_selector_cost": False,
            "included_in_full_request_wall_time": True,
            "standalone_offline_decompositions": 0 if shared else result.solver.decomposition_count,
            "oracle_work_counts": result.work_counts(),
            "oracle_solver_work_counts": result.solver.work_counts(),
            "oracle_timings": (
                policy_metrics.get("oracle_timings") if shared
                else result.timings.to_dict() if result.timings is not None else None
            ),
            "oracle_timing_scope": "committed policy solve, not export work" if shared else "standalone offline observer",
            "oracle_timing_nesting": (
                "hash/Gram/direction subphases are nested in oracle/finalization totals; do not add them again"
            ),
            "shares_policy_decomposition": shared,
            "exact_policy_decomposition_is_additional": False,
            "policy_oracle_seconds": policy_metrics.get("oracle_seconds") if shared else None,
            "detector_artifact": detector_artifact,
            "detector_work_record": detector_work_record,
            "offline_detector_seconds": detector_seconds,
            "detector_persistence_seconds": detector_persistence_seconds,
            "detector_cost_included_in_online_RAW_compression": False,
            "detector_cost_included_in_full_request_wall": detector is not None,
        })
        return receipt

    def _publish_policy_observations(self, chunk_object):
        from ..errors import IntegrityError
        from ..online_rank import RankDecision
        from ..spectra import SpectrumResult
        from ..types import MatrixKey, TensorContext

        state = self.active
        captured = state["deferred_observations"]
        chunk = state["runtime"].store.get_manifest(chunk_object)
        if chunk.get("version") != 2:
            raise IntegrityError("Exact policy export requires an atomically committed adaptive chunk")
        entries = {MatrixKey.from_dict(row["key"]): row for row in chunk["matrices"]}
        if len(entries) != len(chunk["matrices"]) or set(entries) != set(captured):
            raise IntegrityError("Committed exact decisions differ from the complete original capture")
        for key, entry in entries.items():
            capture = captured[key]
            decision = RankDecision.from_dict(entry["decision"])
            observation = decision.to_dict()["metrics"].get("oracle_observation")
            if observation is None or decision.solver != "exact_oracle":
                raise IntegrityError("The committed policy did not supply its own complete exact spectrum")
            context = TensorContext.from_dict(capture["description"]["tensor_context"])
            if decision.context != context or decision.chunk_id != capture["description"]["chunk_id"]:
                raise IntegrityError("An exact observation belongs to another native clean matrix")
            result = SpectrumResult.from_dict(observation)
            actual = result.source.to_dict()
            expected = capture["source"]
            if any(actual[name] != expected[name] for name in (
                "identity", "shape", "dtype", "target", "key_space", "representation",
            )) or any(actual["provenance"].get(name) != value for name, value in capture["binding"].items()):
                raise IntegrityError("The committed exact spectrum lacks its prebound original-source provenance")
            if sys.byteorder == "little" and decision.input_sha256 != capture["description"]["source_sha256"]:
                raise IntegrityError("The committed exact decision did not use the observed original bytes")
            self._persist_observation(result, capture, oracle_seconds=0.0, policy_metrics=entry["decision"]["metrics"])
        captured.clear()

    def _visible_range(self, index, span):
        if type(span) is not int or span <= 0:
            raise ValueError("Native history visibility must be a positive chunk range")
        return self.torch.tensor(
            [[max(0, index + 1 - span) * self.cell.tokens, (index + 1) * self.cell.tokens]],
            dtype=self.torch.int32, device=self.device,
        )

    def generate(self, state):
        t = self.torch
        request, adapter = state["request"], state["adapter"]
        native_noop = self.verified.get("preflight_backend", "reelcache") == "native_noop"
        if (adapter is None) != native_noop:
            raise RuntimeError("The requested backend differs from the installed request capsule")
        with t.inference_mode():
            for index in range(request.chunks):
                noise = t.randn(request.latent_shape, generator=state["generator"],
                                device=self.device, dtype=t.float32)
                noise = self._preflight_noise(state, index, noise)
                noise_hash = _tensor_hash(noise)
                noises = self.transport.checkpoint("magi_fresh_chunk_noise", details={
                    "namespace": state["namespace"], "chunk": index, "sha256": noise_hash,
                })
                if any(row != noises[0] for row in noises):
                    raise RuntimeError("SP2 disagrees on the request's fresh chunk noise")
                state["noise_hashes"].append(noise_hash)
                provenance = dict(
                    noise_sha256=noise_hash, prior_noise_sha256=list(state["noise_hashes"]),
                    noise_generator_state_sha256=_tensor_hash(state["generator"].get_state()),
                )
                state["source_context"].update(provenance)
                if adapter is not None:
                    adapter.source_context.update(provenance)
                with (adapter.execution(chunk_index=index, clean=False) if adapter is not None else nullcontext()):
                    x = t.cat((noise, noise))
                    del noise
                    for step in range(16):
                        span = self.config.runtime_config.noise2clean_kvrange[step // 4]
                        state["attention_point"] = {
                            "chunk_index": index, "phase": "denoise", "step_index": step,
                            "input_noise_sha256": noise_hash,
                        }
                        velocity = self.model.forward_dispatcher(
                            x, self.schedule[step].expand(2, 1), state["y"], state["masks"],
                            self._visible_range(index, span), state["cache"],
                            **request.native_kwargs(index, clean=False, step=step, intervals=self.intervals),
                        )
                        x = x + velocity * (self.schedule[step + 1] - self.schedule[step])
                    latent = x[:1].clone()
                    del x, velocity
                    if tuple(latent.shape) != request.latent_shape or latent.dtype != t.float32 or not t.isfinite(latent).all():
                        raise ValueError("Native generated latent changed geometry, precision or finiteness")
                clean_span = self.config.runtime_config.clean_chunk_kvrange
                if clean_span == -1:
                    clean_span = self.config.runtime_config.noise2clean_kvrange[-1]
                with (adapter.execution(chunk_index=index, clean=True, clean_timestep=0.9999)
                      if adapter is not None else nullcontext()):
                    state["attention_point"] = {
                        "chunk_index": index, "phase": "clean", "step_index": None,
                        "input_noise_sha256": noise_hash,
                    }
                    state["cache"].update_kv_cache = True
                    value = self.model.forward(
                        latent, t.full((1, 1), 0.9999, device=self.device, dtype=t.float32),
                        state["null_caption"].unsqueeze(1),
                        caption_dropout_mask=t.zeros((1,), dtype=t.bool, device=self.device),
                        xattn_mask=state["null_mask"].unsqueeze(1),
                        kv_range=self._visible_range(index, clean_span), inference_params=state["cache"],
                        **request.native_kwargs(index, clean=True, step=0, intervals=self.intervals),
                    )
                    if not t.isfinite(value).all():
                        raise ValueError("Native clean recapture became nonfinite")
                    del value
                committed = adapter.seal_chunk(str(index)) if adapter is not None else None
                if adapter is not None and getattr(request, "policy", None) is not None and request.policy.get("solver") == "exact_oracle" and (
                    request.observe_sources
                ):
                    self.transport.owner_call(
                        "magi_committed_policy_spectrum_export",
                        lambda: self._publish_policy_observations(committed),
                    )
                digest = _tensor_hash(latent)
                hashes = self.transport.checkpoint("magi_generated_latent", details=digest)
                if len(set(hashes)) != 1:
                    raise RuntimeError("SP2 generated different latent outputs")

                def save():
                    path = state["directory"] / "latents" / f"chunk{index}.pt"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with path.open("xb") as stream:
                        t.save({"latent": latent.detach().cpu()}, stream)
                    return {"chunk_index": index, "artifact": _file_row(path),
                            "tensor_sha256": digest, "noise_sha256": noise_hash,
                            "shape": list(latent.shape), "dtype": "float32"}

                record = self.transport.owner_call("magi_latent_publish", save)
                state["latents"].append(self.transport.broadcast_json(record, operation="magi_latent_record"))
                if adapter is not None and index == request.chunks - 1:
                    state["snapshot_id"] = adapter.snapshot(
                        metadata={
                            "study_request": dict(state["source_context"]),
                            "conditioning_identity": state["source_context"]["conditioning_sha256"],
                            "execution_config": {
                                "config": self.verified["config"], "schedule": self.schedule.cpu().tolist(),
                                "intervals": self.intervals.cpu().tolist(), "near_clean_steps": [12, 13, 14, 15],
                                "request_namespace": state["namespace"], "request_hash": request.request_hash,
                            },
                            "vae_contract": {"mode": "stateless_chunk_decode", "fps": 24,
                                             "crop_rows": [self.cell.crop_top, self.cell.crop_top + self.cell.height]},
                        },
                        tensors={
                            "latent": latent, "conditioning": state["y"], "conditioning_mask": state["masks"],
                            "null_caption": state["null_caption"], "null_mask": state["null_mask"],
                        },
                        generators={"noise": state["generator"]},
                    )
                del latent
        inputs, digests, final_noise_rng = None, None, None
        if "input_records" in state:
            persisted = time.perf_counter()
            inputs = self.transport.owner_call("magi_input_bundle_complete", lambda: state["input_bundle"].finish())
            inputs = self.transport.broadcast_json(inputs, operation="magi_completed_input_bundle")
            state["input_instrumentation_seconds"] = state.get("input_instrumentation_seconds", 0.0) + (
                time.perf_counter() - persisted
            )
            persisted = time.perf_counter()
            record, error = None, None
            try:
                record = state["attention_recorder"].finish()
            except BaseException as exc:
                error = exc
            digests = self.transport.checkpoint("magi_attention_records_complete", details=record, error=error)
            state["attention_instrumentation_seconds"] = state.get("attention_instrumentation_seconds", 0.0) + (
                time.perf_counter() - persisted
            )
            persisted = time.perf_counter()
            record, error = None, None
            try:
                from .magi_preflight import final_noise_rng_record

                record = final_noise_rng_record(
                    state["generator"], rank=self.transport.rank, binding_id=inputs["binding_id"],
                )
            except BaseException as exc:
                error = exc
            final_noise_rng = self.transport.checkpoint("magi_final_noise_rng", details=record, error=error)
            state["input_instrumentation_seconds"] = state.get("input_instrumentation_seconds", 0.0) + (
                time.perf_counter() - persisted
            )
        instrumentation = self.transport.checkpoint("magi_preflight_instrumentation", details={
            "rank": self.transport.rank,
            "input_seconds": state.get("input_instrumentation_seconds", 0.0),
            "attention_seconds": state.get("attention_instrumentation_seconds", 0.0),
            "scope": "host wall including copies, hashing, local writes and rank agreement",
            "included_in_full_request_wall": True,
            "local_record_seconds_are_nested": True,
        }) if inputs is not None else None
        memory = adapter.memory_report() if adapter is not None else {
            "native_cache_bytes": sum(value.numel() * value.element_size()
                                     for value in state["cache"].key_value_memory_dict.values()),
            "cache_owner": "unmodified_native_inference_params",
        }
        memories = self.transport.checkpoint("magi_request_memory", details=memory)
        storage = self.transport.owner_call(
            "magi_request_storage_accounting",
            lambda: {
                "decoder_closure": _storage(state["runtime"], state["namespace"], None),
                "snapshot_closure": _storage(state["runtime"], state["namespace"], state["snapshot_id"]),
            } if state.get("runtime") is not None else None,
        ) if adapter is not None else None
        result = {
            "namespace": state["namespace"], "request": {
                **state["source_context"], "chunks": request.chunks, "request_hash": request.request_hash,
            },
            "latents": state["latents"], "per_rank_memory": memories,
            "snapshot_id": state.get("snapshot_id"),
            "storage_owner_record": storage,
            "preflight_backend": "native_noop" if native_noop else "reelcache",
            "runtime_created": not native_noop,
            "input_bundle": inputs, "conditioning_input": state.get("conditioning_input"),
            "chunk_inputs": state.get("input_records"), "attention_digests": digests,
            "final_noise_rng": final_noise_rng,
            "preflight_instrumentation": instrumentation,
            "native_parity_verified": False, "quality_hot_parity_verified": False,
            "mode": "native_dense" if native_noop else self.verified["batch"].get("cache_mode", "runtime"),
        }
        if self.device.type == "cuda":
            peak = t.cuda.max_memory_allocated(self.device)
            error = None if peak <= self.verified["budget"].device_bytes else RuntimeError(
                "Observed native HBM exceeded the frozen request allowance; reject the result",
            )
            self.transport.checkpoint("magi_observed_device_budget", error=error)
        self.transport.owner_call("magi_request_result", lambda: _write(state["directory"] / "result.json", result))
        return result

    def clear_request(self, state):
        state = self.active if state is None else state
        if state is None:
            return
        failures = []
        recorder = state.get("attention_recorder")
        if recorder is not None:
            try:
                recorder.close()
            except BaseException as exc:
                failures.append(exc)
        adapter = state.get("adapter")
        if adapter is not None:
            try:
                adapter.detach()
            except BaseException as exc:
                failures.append(exc)
        runtime = state.get("runtime")
        if runtime is not None:
            try:
                if runtime.active_leases:
                    raise RuntimeError("A request lease survived native cleanup")
                runtime.close()
            except BaseException as exc:
                failures.append(exc)
        cache = state.get("cache")
        if cache is not None:
            cache.key_value_memory_dict.clear()
        if self.model is not None:
            for layer in self.model.videodit_blocks.layers:
                layer.self_attention.attn_weights_history.clear()
        state.clear()
        self.active = None
        gc.collect()
        if self.device.type == "cuda":
            self.torch.cuda.synchronize(self.device)
        if failures:
            raise RuntimeError("MAGI request cleanup failed; the whole SP2 epoch is poisoned") from failures[0]

    def close(self):
        try:
            self.clear_request(self.active)
        finally:
            self.fixes.close()
            self.encoder = self.model = self.schedule = self.intervals = None
            gc.collect()
            self.torch.cuda.empty_cache()


def _initialize_study_collectives(distributed, timeout_seconds):
    deadline = timedelta(seconds=timeout_seconds)
    distributed.init_process_group("nccl", timeout=deadline)
    return distributed.new_group(backend="gloo", timeout=deadline)


def _bind_native_collective_timeout(config, timeout_seconds):
    config.engine_config.distributed_timeout_minutes = timeout_seconds / 60.0


def worker(args):
    """Run a verified, finite latent-generation batch; never load the VAE here."""
    from .magi_rank_worker import MagiWorkerEpoch, verify_worker_inputs

    epoch_started = time.perf_counter()
    verified = verify_worker_inputs(args.protocol, args.assets, args.launch_receipt, args.requests)
    verification_seconds = time.perf_counter() - epoch_started
    if not args.gpu_grant:
        raise RuntimeError("Verified code/assets are not a GPU grant; both MAGI ranks require --gpu-grant")
    import torch
    import torch.distributed as dist
    from ..distributed import DistributedOwner

    if int(os.environ.get("WORLD_SIZE", "0")) != 2:
        raise RuntimeError("Launch the MAGI study worker with torchrun --nproc_per_node=2")
    batch = verified["batch"]
    timeout_seconds = verified["collective_timeout_seconds"]
    output = _output(batch["output"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    group = _initialize_study_collectives(dist, timeout_seconds)
    transport = DistributedOwner(group, expected_world_size=2)
    epoch = None
    try:
        rows = transport.checkpoint("magi_verified_batch", details={
            "batch_id": verified["batch_id"], "asset_id": verified["assets"]["asset_id"],
            "closure_id": verified["receipt"]["source_closure"]["closure_id"],
            "collective_timeout_seconds": timeout_seconds,
        })
        if any(row != rows[0] for row in rows):
            raise RuntimeError("SP2 launch manifests disagree before any model load")
        transport.owner_call("magi_new_epoch_output", lambda: output.mkdir(parents=True, exist_ok=False))
        _environment(output, batch.get("cpu_threads", 1))
        os.environ["SPECIAL_TOKEN_PATH"] = str(verified["asset_root"] / batch["asset_layout"]["special_tokens"])
        source = Path(batch["native_source"]).resolve()
        sys.path.insert(0, str(source))
        for name, module in tuple(sys.modules.items()):
            if name == "inference" or name.startswith("inference."):
                path = Path(getattr(module, "__file__", "")).resolve()
                if not path.is_relative_to(source):
                    raise RuntimeError("A pre-imported native module shadows the reviewed MAGI source")
        from inference.common import MagiConfig
        from inference.infra.distributed.dist_utils import dist_init

        config = MagiConfig.from_json(str(verified["asset_root"] / batch["asset_layout"]["config"]))
        loaded_config = {name: dict(vars(getattr(config, name)))
                         for name in ("model_config", "runtime_config", "engine_config")}
        loaded_config["model_config"]["params_dtype"] = str(config.model_config.params_dtype)
        _validate_config(loaded_config, cell_id=batch["cell_id"], study_worker=True)
        config.runtime_config.load = str((verified["asset_root"] / batch["asset_layout"]["dit_index"]).parent.parent)
        config.runtime_config.t5_pretrained = str((verified["asset_root"] / batch["asset_layout"]["t5_directory"]).parent)
        config.runtime_config.vae_pretrained = str((verified["asset_root"] / batch["asset_layout"]["vae_config"]).parent)
        _bind_native_collective_timeout(config, timeout_seconds)
        dist_init(config)
        torch.set_num_threads(batch.get("cpu_threads", 1))
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        execution = _execution({"identity": verified["batch_id"]})
        execution["source_asset_request_verification_seconds"] = verification_seconds
        execution["collective_timeout_seconds"] = timeout_seconds
        execution["native_distributed_timeout_minutes"] = config.engine_config.distributed_timeout_minutes
        execution["collective_timeout_is_latency_proof"] = False
        execution_path = output / f"worker_execution_rank{transport.rank}.json"
        _write(execution_path, execution)
        workers = transport.checkpoint("magi_worker_process_identity", details=_file_row(execution_path))
        epoch = MagiWorkerEpoch(
            cell_id=batch["cell_id"], config_id=_identity(verified["config"]),
            epoch_id=verified["batch_id"], transport=transport,
            backend_factory=lambda: _StudyBackend(verified, output, config, transport, device),
            budget=verified["budget"], cache_mode=batch.get("cache_mode", "runtime"),
            quality_hot_bytes=batch.get("quality_hot_bytes", 0),
            preflight_backend=verified["preflight_backend"], phase=verified["receipt"]["phase"],
            preflight_resources=verified["preflight_resources"],
        )
        results = [epoch.run(request) for request in verified["requests"]]
        epoch.close()
        wall_times = transport.checkpoint("magi_full_epoch_wall", details={
            "rank": transport.rank, "seconds": time.perf_counter() - epoch_started,
            "source_asset_request_verification_seconds": verification_seconds,
            "model_startup_seconds": epoch.startup_seconds,
            "collective_timeout_seconds": timeout_seconds,
        })
        transport.owner_call("magi_epoch_complete", lambda: _write(output / "generation_complete.json", {
            "schema": "reelcache.magi_generation_batch.v1", "batch_id": verified["batch_id"],
            "protocol_id": verified["protocol"]["protocol_id"], "cell_id": batch["cell_id"],
            "asset_id": verified["assets"]["asset_id"], "worker_execution_records": workers,
            **verified["sampler_contract"],
            "collective_timeout_seconds": timeout_seconds,
            "collective_timeout_is_latency_proof": False,
            "batch_disk_admission": verified["batch_disk_admission"],
            "request_disk_admissions": verified["disk_admissions"],
            "results": results, "model_startup_seconds": epoch.startup_seconds,
            "per_rank_epoch_wall": wall_times,
            "full_epoch_wall_includes_verification_load_generation_offline_capture_and_unload": True,
            "models_unloaded": True, "vae_loaded": False, "native_parity_verified": False,
            "quality_hot_parity_verified": False, "requires_separate_decode_after_both_processes_exit": True,
        }))
    except BaseException as exc:
        try:
            if output.is_dir():
                _write(output / f"epoch_failed_rank{transport.rank}.json", {
                    "schema": "reelcache.magi_failed_epoch.v1", "batch_id": verified["batch_id"],
                    "rank": transport.rank, "error": str(exc), "restart_both_ranks": True,
                    "collective_timeout_seconds": timeout_seconds,
                    "attempt": epoch.failure if epoch is not None else None,
                    "completed_requests": epoch.completed if epoch is not None else [],
                    "total_failed_epoch_seconds": time.perf_counter() - epoch_started,
                    "native_parity_verified": False,
                })
        except (OSError, ValueError):
            pass
        raise
    finally:
        if epoch is not None and not epoch.closed and not epoch.poisoned:
            epoch.close()
        dist.destroy_process_group()


def decode_batch(args):
    """A single VAE instance, separate from the completed two-rank DiT epoch."""
    from .magi_rank_worker import _decode_disk_admission, verify_worker_inputs
    from ..study import study_cell

    verified = verify_worker_inputs(
        args.protocol, args.assets, args.launch_receipt, args.requests, check_storage=False,
        execution_world_size=1,
    )
    if not args.gpu_grant:
        raise RuntimeError("Batch VAE decoding requires its separate one-GPU grant after SP2 exits")
    batch = verified["batch"]
    output = _output(batch["output"])
    generation_path = Path(args.generation_manifest).resolve(strict=True)
    if generation_path != output / "generation_complete.json":
        raise ValueError("VAE input must be this admitted epoch's complete generation manifest")
    generation = _read(generation_path)
    if (generation.get("schema") != "reelcache.magi_generation_batch.v1"
            or generation.get("batch_id") != verified["batch_id"]
            or generation.get("asset_id") != verified["assets"]["asset_id"]
            or generation.get("models_unloaded") is not True
            or generation.get("vae_loaded") is not False
            or len(generation.get("worker_execution_records", [])) != 2):
        raise ValueError("VAE handoff requires both unloaded source-bound MAGI workers")
    _require_workers_exited(generation)
    if output.joinpath("rgb-batch").exists() or output.joinpath("decode_batch_complete.json").exists():
        raise FileExistsError("Refusing to overwrite an earlier/partial batch VAE result")
    if len(generation["results"]) != len(verified["requests"]):
        raise ValueError("VAE handoff omitted a generation request")
    cell = study_cell(batch["cell_id"])
    pairs = []
    for request, result in zip(verified["requests"], generation["results"]):
        item = result["result"]
        if item["request"]["request_hash"] != request.request_hash or len(item["latents"]) != request.chunks:
            raise ValueError("VAE request order or latent coverage differs from the frozen batch")
        for index, latent in enumerate(item["latents"]):
            if (latent["chunk_index"] != index or latent["shape"] != list(request.latent_shape)
                    or latent["dtype"] != "float32"
                    or _sha(latent["artifact"]["path"]) != latent["artifact"]["sha256"]):
                raise ValueError("A complete generated latent changed before VAE decoding")
            pairs.append((item["namespace"], latent))
    import shutil

    decode_disk = _decode_disk_admission(cell, len(pairs))
    disk_need = decode_disk["required_bytes"]
    reserve = max(row["budgets"]["disk_reserve_bytes"] for row in batch["requests"])
    if shutil.disk_usage(output).free < disk_need + reserve:
        raise ValueError("Insufficient admitted storage for lossless RGB batch and reserve")
    _environment(output, batch.get("cpu_threads", 1))
    source = Path(batch["native_source"]).resolve()
    for name, module in tuple(sys.modules.items()):
        if name == "inference" or name.startswith("inference."):
            if not Path(getattr(module, "__file__", "")).resolve().is_relative_to(source):
                raise RuntimeError("A different pre-imported native source shadows the verified VAE")
    sys.path.insert(0, str(source))
    os.environ["SPECIAL_TOKEN_PATH"] = str(verified["asset_root"] / batch["asset_layout"]["special_tokens"])
    import numpy as np
    import torch
    from safetensors.torch import load_file
    from inference.model.vae.vae_model import ViTVAE
    from inference.pipeline.video_process import VaeHelper

    vae_config = _read(verified["asset_root"] / batch["asset_layout"]["vae_config"])
    linspace = torch.linspace

    def cpu_linspace(*args, **kwargs):
        kwargs["device"] = "cpu"
        return linspace(*args, **kwargs)

    with patch.object(torch, "linspace", cpu_linspace), torch.device("meta"):
        vae = ViTVAE(ddconfig=vae_config["ddconfig"], model_type=vae_config["model_type"])
    vae.load_state_dict(
        load_file(str(verified["asset_root"] / batch["asset_layout"]["vae_weights"]), device="cpu"),
        strict=True, assign=True,
    )
    device = torch.device("cuda", args.device)
    torch.cuda.set_device(device)
    torch.set_num_threads(batch.get("cpu_threads", 1))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.nn.Module.to(vae.eval(), device=device, dtype=torch.bfloat16)
    records = []
    try:
        for namespace, record in pairs:
            latent = torch.load(record["artifact"]["path"], map_location=device, weights_only=True)["latent"]
            if tuple(latent.shape) != tuple(record["shape"]) or _tensor_hash(latent) != record["tensor_sha256"]:
                raise ValueError("VAE latent tensor hash or geometry changed")
            started = time.perf_counter()
            with torch.inference_mode(), patch.object(torch.distributed, "get_rank", return_value=0):
                value = VaeHelper.decode(
                    latent / 0.18215, vae, tile_sample_min_height=256, tile_sample_min_width=256,
                    spatial_tile_overlap_factor=0.25, temporal_tile_overlap_factor=0,
                    tile_sample_min_length=12, allow_spatial_tiling=True,
                    uint8_output=False, parallel_group=None,
                )
            torch.cuda.synchronize(device)
            seconds = time.perf_counter() - started
            if tuple(value.shape) != (24, 3, cell.working_height, cell.width) or not torch.isfinite(value).all():
                raise ValueError("Native batch VAE output changed its exact working geometry")
            pixels = (value.float() * 127.5 + 127.5).clamp(0, 255).to(torch.uint8)
            frames = pixels[:, :, cell.crop_top:cell.crop_top + cell.height].permute(0, 2, 3, 1).contiguous().cpu().numpy()
            path = output / "rgb-batch" / namespace / f"chunk{record['chunk_index']}.npz"
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as stream:
                np.savez_compressed(stream, frames=frames)
            media = {
                "namespace": namespace, "chunk_index": record["chunk_index"], "artifact": _file_row(path),
                "shape": list(frames.shape), "rgb_sha256": hashlib.sha256(memoryview(frames)).hexdigest(),
                "native_decode_seconds": seconds, "fps": 24,
                "crop_rows": [cell.crop_top, cell.crop_top + cell.height],
                "VAE_state": "stateless_chunk_decode", "streaming_seam_parity_verified": False,
                "latent_sha256": record["tensor_sha256"],
            }
            _write(path.with_suffix(".json"), media)
            records.append(media)
            del latent, value, pixels, frames
    finally:
        del vae
        gc.collect()
        torch.cuda.empty_cache()
    _write(output / "decode_batch_complete.json", {
        "schema": "reelcache.magi_decode_batch.v1", "batch_id": verified["batch_id"], "media": records,
        "decode_disk_admission": decode_disk,
        "generation_manifest": _file_row(generation_path), "cell_id": cell.cell_id,
        "models_unloaded_before_VAE_load": True, "native_parity_verified": False,
    })
