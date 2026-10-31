#!/usr/bin/env python3
"""Source-bound CausVid runtime verification, not a held-out quality study.

``worker`` is a separate, launch-gated CausVid480 rank-study entry point:
``worker --help`` documents exact-prompt requests, explicit staged assets and
separate persistent latent-generation / VAE-decode epochs. The historical
``preflight`` and ``run`` commands below retain their reused-pilot contract.

No model imports occur on import, --help, or preflight. A run requires explicit
external model paths, a completed reused-prompt pilot manifest, a new run ID,
and --allow-cuda. GPU access must separately be authorized by the coordinator.
RAW is the default. ``--arms raw svd svd_hevc`` explicitly requests unqualified
lossy diagnostics; it does not select or qualify an automatic quality policy.
``--output-root`` keeps run artifacts in the project when executing a frozen
copy of the package and this driver.

The integration builder imports the configured native InferencePipeline
directly. It does not import historical harnesses or their output globals.
Only cross-attention instances receive a reversible SDPA compatibility binding;
the original native self-attention forward/kernel remains in use.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
import datetime as dt
import gc
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import shlex
import sys
import time
import traceback
import types


ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def identity(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def verify_sources(provenance):
    changed = [
        name for name, entry in provenance["source_files"].items()
        if sha256(entry["path"]) != entry["sha256"]
    ]
    if changed:
        raise RuntimeError(f"Executed source files changed during measurement: {changed}")


def write_json(path, value, *, mutable=False):
    path = Path(path)
    if path.exists() and not mutable:
        raise FileExistsError(f"Refusing to overwrite verification evidence: {path}")
    pending = path.with_name(path.name + ".pending")
    with pending.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    pending.replace(path)


def environment(directory, *, gpu, threads):
    if threads < 1 or threads > 8:
        raise ValueError("Use a bounded CPU thread count between one and eight")
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = str(threads)
    for name in (
        "TMPDIR", "TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR", "TORCH_EXTENSIONS_DIR",
        "CUDA_CACHE_PATH", "XDG_CACHE_HOME", "HF_HOME",
    ):
        path = directory / "build_cache" / name.lower()
        path.mkdir(parents=True)
        os.environ[name] = str(path)
    os.environ.update(
        CUDA_VISIBLE_DEVICES="" if gpu is None else str(gpu),
        PYTHONDONTWRITEBYTECODE="1", TOKENIZERS_PARALLELISM="false",
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1",
    )
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(ROOT))


def preflight(args):
    if any(name.startswith("third_party.causvid") for name in sys.modules):
        raise RuntimeError("Run preflight before importing external native model modules")
    if "torch" in sys.modules and sys.modules["torch"].cuda.is_initialized():
        raise RuntimeError("CPU preflight must not initialize CUDA")
    paths = {
        "model_code_root": args.model_code_root.resolve(),
        "base_model": args.base_model.resolve(),
        "checkpoint": args.checkpoint.resolve(),
        "config": args.config.resolve(),
        "case_manifest": args.case_manifest.resolve(),
    }
    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing explicit {name}: {path}")
    case = json.loads(paths["case_manifest"].read_text())
    if case.get("status") != "complete" or case.get("seed") != 0 or case.get("prompt_id") not in ("t00", "t01"):
        raise ValueError("Use one completed pilot t00/t01 seed0 manifest; this is reused-prompt correctness only")
    if case.get("native_clean_geometry") != {"frames": 3, "hw": [30, 52], "tokens": 4680, "channels": 1536}:
        raise ValueError("Pilot geometry is not native 480x832 CausVid")
    source = case["source"]
    prefix = Path(source["prefix_inputs_path"]).resolve()
    if sha256(prefix) != source["prefix_inputs_sha256"]:
        raise ValueError("Explicit pilot prefix source failed its recorded hash")
    checkpoint_hash = sha256(paths["checkpoint"])
    if checkpoint_hash != source["checkpoint_sha256"]:
        raise ValueError("Checkpoint differs from the completed pilot's checkpoint")
    native_root = paths["model_code_root"] / "third_party/causvid/modeling"
    files = {
        "driver": Path(__file__).resolve(), "config": paths["config"],
        "adapter": ROOT / "reelcache/adapters/causvid.py",
        "base_config": paths["base_model"] / "config.json",
    }
    base_config = json.loads(files["base_config"].read_text())
    expected_config = {"model_type": "t2v", "dim": 1536, "num_heads": 12, "num_layers": 30, "in_dim": 16, "out_dim": 16}
    if any(base_config.get(key) != value for key, value in expected_config.items()):
        raise ValueError("Base configuration does not match qualified CausVid 1.3B geometry")
    files.update({f"native/{path.name}": path for path in sorted(native_root.glob("*.py"))})
    files.update({
        f"reelcache/{path.relative_to(ROOT / 'reelcache')}": path
        for path in sorted((ROOT / "reelcache").rglob("*.py"))
    })
    for required in ("causal_model.py", "causal_inference.py", "wan_wrapper.py", "attention.py", "vae.py"):
        if f"native/{required}" not in files:
            raise FileNotFoundError(f"Missing native model source: {native_root / required}")
    sources = {name: {"path": str(path), "sha256": sha256(path)} for name, path in files.items()}
    prior = case["provenance"]["source_verified_model_runtime"]
    if (
        sources["native/causal_model.py"]["sha256"] != prior["native_model_source_sha256"]
        or sources["native/causal_inference.py"]["sha256"] != prior["native_pipeline_source_sha256"]
    ):
        raise ValueError("Native attention/pipeline source is not the qualified completed-pilot source")
    assets = {}
    for name in ("models_t5_umt5-xxl-enc-bf16.pth", "Wan2.1_VAE.pth"):
        path = paths["base_model"] / name
        if not path.is_file():
            raise FileNotFoundError(path)
        assets[name] = {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}
    return {
        "schema": "reelcache.causvid.preflight.v1", "status": "cpu_preflight_passed",
        "model_loaded": False, "cuda_initialized": False, "torch_imported": "torch" in sys.modules,
        "paths": {key: str(value) for key, value in paths.items()},
        "source_files": sources,
        "weights": {
            "checkpoint": {"path": str(paths["checkpoint"]), "sha256": checkpoint_hash,
                           "bytes": paths["checkpoint"].stat().st_size},
            **assets,
        },
        "case_manifest_sha256": sha256(paths["case_manifest"]),
        "case_id": case["case_id"], "prompt": case["prompt"], "seed": 0,
        "prefix_inputs": {"path": str(prefix), "sha256": source["prefix_inputs_sha256"]},
        "scope": "single reused prompt/seed; correctness and diagnostic distortion, not held-out quality",
        "integration_dependency": "explicit native RAVEN/third_party/causvid source + weights; no historical harness imports",
    }


def build_pipeline(args):
    import torch
    from omegaconf import OmegaConf

    root = args.model_code_root.resolve()
    sys.path.insert(0, str(root))
    native = importlib.import_module("third_party.causvid.modeling.causal_inference")
    Path(native.__file__).resolve().relative_to(root)
    config = OmegaConf.load(args.config)
    config.model_path = str(args.base_model.resolve())
    config.checkpoint_folder = str(args.checkpoint.resolve().parent)
    config.num_frame_per_block = 3
    pipeline = native.InferencePipeline(config, device="cuda")
    pipeline.to(device="cuda", dtype=torch.bfloat16)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    pipeline.generator.load_state_dict(state["generator"], strict=True)
    del state
    pipeline.generator.model.num_frame_per_block = 3
    pipeline.eval()
    return pipeline


def _full_sdpa(q, k, v, q_lens=None, k_lens=None, dropout_p=0.0,
               softmax_scale=None, q_scale=None, causal=False, window_size=(-1, -1),
               deterministic=False, dtype=None, version=None):
    import torch

    if q_lens is not None or k_lens is not None or tuple(window_size) != (-1, -1) or causal:
        raise ValueError("The compatibility binding supports only native full unmasked text cross-attention")
    if dropout_p:
        raise ValueError("Native verification does not permit stochastic attention dropout")
    output_dtype = q.dtype
    dtype = torch.bfloat16 if dtype is None else dtype
    q, k, v = (item.to(dtype) for item in (q, k, v))
    if q_scale is not None:
        q = q * q_scale
    return torch.nn.functional.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
        dropout_p=0.0, is_causal=False, scale=softmax_scale,
    ).transpose(1, 2).contiguous().to(output_dtype)


@contextmanager
def instance_cross_attention_sdpa(pipeline):
    """Bind copied function globals to instances; never mutate module/class state."""
    changed = []
    try:
        for block in pipeline.generator.model.blocks:
            instance = block.cross_attn
            bound = instance.forward
            function = bound.__func__
            if "flash_attention" not in function.__code__.co_names:
                raise ValueError("Unrecognized native cross-attention compatibility seam")
            globals_for_instance = dict(function.__globals__, flash_attention=_full_sdpa)
            local = types.FunctionType(
                function.__code__, globals_for_instance, function.__name__,
                function.__defaults__, function.__closure__,
            )
            local.__kwdefaults__ = function.__kwdefaults__
            changed.append((instance, "forward" in instance.__dict__, instance.__dict__.get("forward")))
            instance.forward = types.MethodType(local, instance)
        yield
    finally:
        for instance, had, previous in reversed(changed):
            if had:
                instance.forward = previous
            else:
                instance.__dict__.pop("forward", None)


def memory_sample(label):
    import torch

    torch.cuda.synchronize()
    return {
        "label": label, "time": time.time(),
        "allocated_bytes": torch.cuda.memory_allocated(),
        "reserved_bytes": torch.cuda.memory_reserved(),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }


class DenseNativeAudit:
    """Read-only native hooks; clean writes must not modify preceding chunks."""

    def __init__(self, pipeline, prefix, chunks, *, conditioning_observer=None, clean_observer=None):
        from reelcache.adapters.causvid import tensor_digest

        self.pipeline, self.chunks = pipeline, chunks
        self.prefix_hash = None if prefix is None else tensor_digest(prefix)
        self.chunk_tokens = pipeline.num_frame_per_block * pipeline.frame_seq_length
        self.conditioning_observer, self.clean_observer = conditioning_observer, clean_observer
        self.calls, self.clean = [], {}
        self.handles = []
        self.conditioning_hash = None

    def __enter__(self):
        generator = self.pipeline.generator
        try:
            self.handles.append(generator.register_forward_pre_hook(self.before, with_kwargs=True))
            self.handles.append(generator.register_forward_hook(self.after, with_kwargs=True))
            if self.clean_observer is not None:
                for layer, block in enumerate(generator.model.blocks):
                    for target, module in (("K", block.self_attn.norm_k), ("V", block.self_attn.v)):
                        self.handles.append(module.register_forward_hook(self.source_hook(layer, target)))
            return self
        except BaseException:
            self.__exit__()
            raise

    def __exit__(self, *_):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def before(self, _module, args, kwargs):
        import torch
        from reelcache.adapters.causvid import tensor_digest

        if args or kwargs["kv_cache"] is not self.pipeline.kv_cache1:
            raise RuntimeError("Native dense generator/cache seam changed")
        start, end = int(kwargs["current_start"]), int(kwargs["current_end"])
        values = torch.unique(kwargs["timestep"]).cpu().tolist()
        if len(values) != 1 or end != start + self.chunk_tokens or start % self.chunk_tokens:
            raise RuntimeError("Native dense absolute position/timestep changed")
        condition = kwargs["conditional_dict"]
        if set(condition) != {"prompt_embeds"}:
            raise RuntimeError("Native dense conditioning contract changed")
        condition_hash = tensor_digest(condition["prompt_embeds"])
        if self.conditioning_hash is None:
            self.conditioning_hash = condition_hash
            if self.conditioning_observer is not None:
                self.conditioning_observer(condition["prompt_embeds"])
        elif condition_hash != self.conditioning_hash:
            raise RuntimeError("Native conditioning changed during the denoising trajectory")
        current = {
            "token_start": start, "token_end": end, "timestep": float(values[0]),
            "input_sha256": tensor_digest(kwargs["noisy_image_or_video"]),
        }
        if current["timestep"] == 0:
            expected = self.prefix_hash if start == 0 and self.prefix_hash is not None else (
                self.calls[-1]["output_sha256"] if self.calls and self.calls[-1]["token_start"] == start else None
            )
            if current["input_sha256"] != expected:
                raise RuntimeError("Native t=0 write did not use the independently denoised clean latents")
        self.current = current
        self.versions = [(row["k"]._version, row["v"]._version) for row in self.pipeline.kv_cache1]

    def source_hook(self, layer, target):
        def capture(_module, _args, output):
            if self.current["timestep"] != 0:
                return
            from reelcache.adapters.causvid import tensor_digest

            value = output[0].detach()
            before, version = tensor_digest(value), value._version
            self.clean_observer(self.current["token_start"], layer, target, value)
            if value._version != version or tensor_digest(value) != before:
                raise RuntimeError("Dense native source observer mutated an original clean matrix")
        return capture

    def segment_hashes(self, start):
        from reelcache.adapters.causvid import tensor_digest

        return [
            {"layer": layer, "post_rope_K": tensor_digest(row["k"][:, start:start + self.chunk_tokens]),
             "V": tensor_digest(row["v"][:, start:start + self.chunk_tokens])}
            for layer, row in enumerate(self.pipeline.kv_cache1)
        ]

    def after(self, _module, _args, _kwargs, output):
        from reelcache.adapters.causvid import tensor_digest

        if not all(
            row["k"]._version == key_version + 1 and row["v"]._version == value_version + 1
            for row, (key_version, value_version) in zip(self.pipeline.kv_cache1, self.versions)
        ):
            raise RuntimeError("Native dense forward did not write every K/V exactly once")
        self.current["output_sha256"] = tensor_digest(output)
        self.calls.append(self.current)
        if self.current["timestep"] == 0:
            start = self.current["token_start"]
            self.clean[start] = self.segment_hashes(start)

    def finish(self):
        expected = [] if self.prefix_hash is None else [(0, 0.0)]
        for chunk in range(0 if self.prefix_hash is None else 1, self.chunks):
            expected.extend(
                (chunk * self.chunk_tokens, float(step))
                for step in self.pipeline.denoising_step_list.cpu().tolist() + [0]
            )
        if [(row["token_start"], row["timestep"]) for row in self.calls] != expected:
            raise RuntimeError("Native dense denoising/clean-forward lifecycle changed")
        if set(self.clean) != set(range(0, self.chunks * self.chunk_tokens, self.chunk_tokens)):
            raise RuntimeError("Native dense clean chunks are incomplete")
        for start, hashes in self.clean.items():
            if self.segment_hashes(start) != hashes:
                raise RuntimeError("Later native denoising overwrote a clean parent segment")
        return {
            "calls": self.calls, "clean_segment_hashes": self.clean,
            "all_native_clean_segments_unchanged": True, "all_KV_writes_verified": True,
            "separate_clean_forward_verified": True,
            "first_chunk_denoised_from_scratch": self.prefix_hash is None,
            "conditioning_tensor_sha256": self.conditioning_hash,
        }


def video_rgb(video):
    import numpy as np

    return np.ascontiguousarray(
        (video[0].float().clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy() * 255).astype("uint8")
    )


def save_output(directory, latents, frames, rng_state):
    import numpy as np
    from reelcache.adapters.causvid import tensor_digest
    from reelcache.serialization import serialize_tensors

    directory.mkdir()
    latent_path, rgb_path = directory / "state.reelcache", directory / "rgb.npy"
    with latent_path.open("xb") as stream:
        stream.write(serialize_tensors({"latents": latents.cpu(), "rng_state": rng_state.cpu()}))
    with rgb_path.open("xb") as stream:
        np.save(stream, frames, allow_pickle=False)
    return {
        "latents_sha256": tensor_digest(latents), "rng_sha256": tensor_digest(rng_state),
        "rgb_sha256": hashlib.sha256(frames.tobytes()).hexdigest(),
        "state_file": {"path": str(latent_path), "sha256": sha256(latent_path), "bytes": latent_path.stat().st_size},
        "rgb_file": {"path": str(rgb_path), "sha256": sha256(rgb_path), "bytes": rgb_path.stat().st_size},
    }


def branch_bytes(runtime, branch):
    """Unique persistent decoder bytes, including every canonical basis row."""
    root = runtime.branch_info(branch)["head_id"]
    pending, seen = [root], set()
    sizes = {"blob": 0, "basis": 0, "manifest": 0}
    while pending:
        object_id = pending.pop()
        if object_id in seen:
            continue
        seen.add(object_id)
        kind, _digest = object_id.split(":")
        path = runtime.store.path_for(object_id)
        sizes[kind] += path.stat().st_size
        if kind == "manifest":
            pending.extend(runtime.store.references(object_id))
    return {
        "unique_object_count": len(seen), "payload_bytes": sizes["blob"],
        "full_canonical_basis_bytes": sizes["basis"], "metadata_bytes": sizes["manifest"],
        "total_decoder_bytes": sum(sizes.values()),
        "counts_snapshot_latents_or_rng": False,
        "includes_embedded_codec_diagnostics_in_manifests": True,
        "excludes_external_RGB_audits_and_store_index": True,
        "counts_unreachable_objects": False, "canonical_basis_counted_once_per_object": True,
    }


def verify_raw_chunk(runtime, chunk_id, expected):
    chunk = runtime.store.get_manifest(chunk_id)
    by_layer = {row["layer"]: row for row in expected}
    for entry in chunk["matrices"]:
        key = entry["key"]
        tensor = runtime.store.get_manifest(entry["tensor"])
        metadata = tensor["encoded"]["metadata"]
        target = "post_rope_K" if key["target"] == "K" else "V"
        if (
            metadata["kind"] != "raw"
            or metadata["payloads"]["tensor.bin"]["sha256"] != by_layer[key["layer"]][target]
        ):
            raise RuntimeError(f"Native RAW clean-cache parity failed at layer {key['layer']}/{target}")
    if len(chunk["matrices"]) != 2 * len(expected):
        raise RuntimeError("Native RAW clean-cache layer coverage is incomplete")


def compare(reference_latents, reference_frames, latents, frames, *, exact):
    import numpy as np
    import torch
    from skimage.metrics import structural_similarity

    if reference_frames.shape != frames.shape or reference_latents.shape != latents.shape:
        raise RuntimeError("Continuation geometry changed")
    latent_equal, rgb_equal = torch.equal(reference_latents, latents), np.array_equal(reference_frames, frames)
    prefix_equal = np.array_equal(reference_frames[:9], frames[:9]) and torch.equal(reference_latents[:, :3], latents[:, :3])
    if exact and (not latent_equal or not rgb_equal):
        raise RuntimeError("Native RAW identity/restore parity failed")
    if not prefix_equal:
        raise RuntimeError("Clean prefix changed")
    scores = [
        float(structural_similarity(left, right, channel_axis=2, data_range=255))
        for left, right in zip(reference_frames[9:], frames[9:])
    ]
    return {
        "full_latents_bitwise_equal": latent_equal, "full_RGB_bitwise_equal": rgb_equal,
        "clean_prefix_latents_and_RGB_bitwise_equal": prefix_equal,
        "continuation_ssim": float(np.mean(scores)), "per_frame_ssim": scores,
        "continuation_frame_count": len(scores), "prefix_pixel_frames": 9,
        "quality_qualification_claim": False,
    }


def run_verification(args, directory, provenance):
    verify_sources(provenance)
    import numpy as np
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from reelcache.adapters.causvid import CausVidAdapter, CausVidConfig, tensor_digest, unique_tensor_bytes
    from reelcache.codecs import hevc_capabilities, using_ffmpeg
    from reelcache.manager import KVRuntime
    from reelcache.types import CodecSpec, MatrixKey, ModelIdentity

    if not callable(getattr(KVRuntime, "load_snapshot", None)):
        raise RuntimeError("KVRuntime requires read-only load_snapshot before native model loading is permitted")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    codec_environment = None
    if "svd_hevc" in args.arms:
        codec_environment = hevc_capabilities(args.ffmpeg)
        codec_environment["binary_sha256"] = sha256(codec_environment["binary"])
        if "yuv444p" not in codec_environment["pixel_formats"]:
            raise RuntimeError("Requested per-channel HEVC yuv444p is unavailable")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Verification requires exactly one explicitly visible GPU")
    started = time.perf_counter()
    pipe = build_pipeline(args)
    model_load_seconds = time.perf_counter() - started
    cfg = CausVidConfig()
    inputs = torch.load(provenance["prefix_inputs"]["path"], map_location="cpu", weights_only=True)
    if set(inputs) != {"latents", "prompt_embeds"} or tuple(inputs["latents"].shape) != (1, 3, 16, 60, 104):
        raise ValueError("Invalid explicit pilot prefix inputs")
    total_frames = args.chunks * 3
    rng = torch.Generator(device="cuda").manual_seed(0)
    noise = torch.empty((1, total_frames, 16, 60, 104), device="cuda", dtype=torch.bfloat16).normal_(generator=rng)
    initial_rng = rng.get_state().clone()
    noise_cpu = noise.cpu()
    prompt, conditioning_capture = provenance["prompt"], {}
    result = {
        "schema": "reelcache.causvid.native_verification.v1", "status": "running", "provenance": provenance,
        "geometry": {"layers": 30, "heads": 12, "head_dim": 128, "tokens_per_chunk": 4680,
                     "latent_frames_per_chunk": 3, "height": 480, "width": 832},
        "chunks": args.chunks, "uniform_lossy_ranks": {"K": 256, "V": 384},
        "model_load_seconds": model_load_seconds, "native_gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "initial_noise_sha256": tensor_digest(noise_cpu), "initial_rng_sha256": tensor_digest(initial_rng),
        "cross_attention_backend": "instance-local reversible full SDPA; same for dense and runtime arms",
        "codec_environment": codec_environment,
        "attention_visibility": "all tokens [0,current_end); original native self-attention kernel/forward",
        "execution_mode": "serial clean-history restoration, not fully pipelined native performance",
        "memory": [], "arms": {}, "fork": {}, "native_proof": True,
        "held_out_quality_claim": False, "historical_replay_claim": False,
    }

    def progress(phase):
        verify_sources(provenance)
        write_json(directory / "progress.json", {
            "phase": phase, "elapsed_seconds": time.perf_counter() - started,
            "completed_arms": list(result["arms"]), "utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        }, mutable=True)
        print(f"[CausVid] {phase}", flush=True)

    def capture_conditioning(_module, _inputs, output):
        conditioning_capture["prompt_embeds"] = output["prompt_embeds"].detach().to("cpu", copy=True)

    with (
        instance_cross_attention_sdpa(pipe),
        sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]),
        using_ffmpeg(args.ffmpeg) if codec_environment is not None else nullcontext(),
    ):
        progress("native dense reference")
        pipe.kv_cache1 = [
            {name: torch.zeros((1, total_frames * cfg.frame_tokens, cfg.heads, cfg.head_dim),
                               device="cuda", dtype=torch.bfloat16) for name in ("k", "v")}
            for _ in range(30)
        ]
        pipe._initialize_crossattn_cache(batch_size=1, dtype=torch.bfloat16, device="cuda")
        torch.cuda.reset_peak_memory_stats()
        result["memory"].append(memory_sample("dense_preallocated"))
        dense_bytes = unique_tensor_bytes(value for row in pipe.kv_cache1 for value in row.values())
        handle = pipe.text_encoder.register_forward_hook(capture_conditioning)
        dense_started = time.perf_counter()
        try:
            with DenseNativeAudit(pipe, inputs["latents"], args.chunks) as dense_audit:
                video, baseline = pipe.inference(
                    noise=noise, text_prompts=[prompt], start_latents=inputs["latents"].to("cuda"),
                    return_latents=True, rng=rng,
                )
            native_audit = dense_audit.finish()
        finally:
            handle.remove()
        result["memory"].append(memory_sample("dense_after_native_generation_and_VAE"))
        baseline, reference_frames, reference_rng = baseline.cpu(), video_rgb(video), rng.get_state().clone()
        result["dense"] = {
            "elapsed_seconds": time.perf_counter() - dense_started, "dense_kv_backing_bytes": dense_bytes,
            "artifacts": save_output(directory / "dense_reference", baseline, reference_frames, reference_rng),
            "conditioning_sha256": tensor_digest(conditioning_capture["prompt_embeds"]),
            "pilot_saved_conditioning_equal": torch.equal(conditioning_capture["prompt_embeds"], inputs["prompt_embeds"]),
            "native_audit": native_audit,
        }
        del video, noise
        pipe.kv_cache1 = pipe.crossattn_cache = None
        gc.collect()
        result["memory"].append(memory_sample("dense_allocations_released"))
        binding = {
            "sources": {key: value["sha256"] for key, value in provenance["source_files"].items()
                        if key.startswith("native/") or key in ("config", "base_config")},
            "weights": {
                name: {"sha256": row["sha256"], "bytes": row["bytes"]}
                for name, row in provenance["weights"].items()
            },
            "native_shape": result["geometry"],
            "attention_backend": result["cross_attention_backend"],
        }
        model = ModelIdentity("causvid", identity(binding), 30, 1536, "tokens_channels", "bfloat16")
        prompt_id = identity({"prompt": prompt, "conditioning_sha256": result["dense"]["conditioning_sha256"]})
        result["model_identity"] = model.to_dict()
        for kind in args.arms:
            arm_dir = directory / kind
            arm_dir.mkdir()
            specs = {
                MatrixKey(layer, target): (
                    CodecSpec() if kind == "raw" else CodecSpec(
                        kind, rank=256 if target == "K" else 384,
                        inter_prediction=args.hevc_inter_prediction and kind == "svd_hevc",
                    )
                ) for layer in range(30) for target in ("K", "V")
            }
            arm = {
                "kind": kind, "chunks": [], "memory": [], "lossy_opt_in": kind != "raw",
                "codec_specs": {target: specs[MatrixKey(0, target)].to_dict() for target in ("K", "V")},
                "history_protocol": "archive every completed clean chunk; later chunks see all prior archived history",
                "later_chunks_are_closed_loop_not_matched_source_codec_injections": True,
            }
            arm_started = time.perf_counter()
            with KVRuntime(arm_dir / "runtime", model, max_resident_bytes=args.history_mib * 1024**2) as runtime:
                with CausVidAdapter(
                    pipe, runtime, "parent", config=cfg, specs=specs,
                    max_attention_bytes=args.attention_mib * 1024**2,
                ) as adapter:
                    rng = torch.Generator(device="cuda").set_state(initial_rng)
                    adapter.start(noise_cpu, conditioning_capture, rng, conditioning_id=prompt_id)

                    def step(label, *, prefix=None):
                        progress(label)
                        torch.cuda.reset_peak_memory_stats()
                        before = memory_sample(label + "/before")
                        tick = time.perf_counter()
                        sealed = adapter.step(clean_prefix=prefix)
                        after = memory_sample(label + "/after")
                        report = adapter.memory_report()
                        if (
                            report["live_mutable_kv_bytes"] or report["live_clean_capture_cpu_bytes"]
                            or report["live_attention_read_bytes"] or runtime.active_leases
                        ):
                            raise RuntimeError("A sealed layer/chunk retained dense allocations or history leases")
                        row = {
                            "label": label, "elapsed_seconds": time.perf_counter() - tick,
                            "before": before, "after": after, "owned_memory": report,
                            "runtime_residency": runtime.residency.stats(),
                            "chunk_manifest": sealed,
                        }
                        if kind == "raw" and adapter.branch_id in ("parent", "restored"):
                            start = adapter.token_cursor - cfg.chunk_tokens
                            verify_raw_chunk(runtime, sealed, native_audit["clean_segment_hashes"][start])
                            row["native_clean_KV_bitwise_equal"] = True
                        arm["chunks"].append(row)
                        return row

                    step(f"{kind}/clean-prefix", prefix=inputs["latents"])
                    snapshot = adapter.snapshot()
                    arm["prefix_snapshot"] = snapshot
                    while not adapter.finished:
                        step(f"{kind}/parent/chunk{adapter.frame_cursor // 3}")
                    latents = adapter.latent_state
                    generated = adapter.decode()
                    frames = video_rgb(generated)
                    del generated
                    arm["parent_audit"] = list(adapter.audit)
                    arm["comparison"] = compare(baseline, reference_frames, latents, frames, exact=kind == "raw")
                    if not torch.equal(reference_rng, adapter.rng_state):
                        raise RuntimeError("Codec/restoration changed the explicit denoising RNG")
                    arm["artifacts"] = save_output(arm_dir / "parent", latents, frames, adapter.rng_state)
                    arm["parent_unique_decoder_bytes"] = branch_bytes(runtime, "parent")
                    arm["parent_head"] = runtime.branch_info("parent")
                    arm["parent_snapshot"] = adapter.snapshot()
                    if kind == "raw":
                        progress("raw/restore snapshot into new branch")
                        adapter.restore(snapshot, "restored", conditioning_id=prompt_id)
                        while not adapter.finished:
                            step(f"raw/restored/chunk{adapter.frame_cursor // 3}")
                        restored = adapter.latent_state
                        generated = adapter.decode()
                        restored_frames = video_rgb(generated)
                        del generated
                        arm["restore_comparison"] = compare(baseline, reference_frames, restored, restored_frames, exact=True)
                        if not torch.equal(reference_rng, adapter.rng_state):
                            raise RuntimeError("RAW resume changed the RNG")
                        arm["restore_artifacts"] = save_output(arm_dir / "restored", restored, restored_frames, adapter.rng_state)
                        parent_before = runtime.branch_info("parent")
                        child_latents = child_frames = child_rng = None
                        for branch in ("fork", "fork-replay"):
                            adapter.fork(snapshot, branch, seed=args.fork_seed)
                            for index in range(args.fork_chunks):
                                step(f"raw/{branch}/chunk{index + 1}")
                            generated = adapter.decode()
                            branch_frames = video_rgb(generated)
                            del generated
                            branch_latents, branch_rng = adapter.latent_state, adapter.rng_state
                            if not torch.equal(branch_latents[:, :3], baseline[:, :3]):
                                raise RuntimeError("A new fork seed mutated its clean prefix")
                            if torch.equal(branch_latents[:, 3:], baseline[:, 3:branch_latents.shape[1]]):
                                raise RuntimeError("Fork seed did not produce a divergent continuation")
                            if branch == "fork":
                                child_latents, child_frames, child_rng = branch_latents, branch_frames, branch_rng
                            elif (
                                not torch.equal(child_latents, branch_latents)
                                or not np.array_equal(child_frames, branch_frames)
                                or not torch.equal(child_rng, branch_rng)
                            ):
                                raise RuntimeError("Fork replay changed latents, RGB or RNG")
                            if runtime.branch_info("parent") != parent_before:
                                raise RuntimeError("Fork mutated the committed parent branch")
                            result["fork"][branch] = save_output(
                                arm_dir / branch, branch_latents, branch_frames, branch_rng,
                            )
                        result["fork"].update(
                            seed=args.fork_seed, continuation_chunks=args.fork_chunks,
                            replay_latents_RGB_rng_bitwise_equal=True, parent_branch_head_unchanged=True,
                            parent_snapshot_immutable=True,
                            seed_semantics="redraw only remaining upfront noise with the child Generator",
                        )
                        runtime.store.verify(arm["parent_snapshot"])
                        result["fork"]["parent_snapshot_and_KV_hashes_verified_after_replay"] = True
                    arm["memory"].append(memory_sample(f"{kind}/after-verification"))
                    arm["final_adapter_owned_memory"] = adapter.memory_report()
                runtime.collect_garbage()
                arm["all_branches_and_snapshots_unique_store_bytes"] = runtime.store.stats()
                if runtime.resident_bytes or runtime.active_leases:
                    raise RuntimeError("History allocations survived adapter uninstall")
            arm["elapsed_seconds"] = time.perf_counter() - arm_started
            write_json(arm_dir / "result.json", arm)
            result["arms"][kind] = arm
        result["memory"].append(memory_sample("all_adapters_uninstalled"))
    verify_sources(provenance)
    result.update(
        status="passed", elapsed_seconds=time.perf_counter() - started,
        source_hashes_stable_during_run=True,
        completed_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
    )
    write_json(directory / "result.json", result)
    return result


def parser():
    cli = argparse.ArgumentParser(
        description=__doc__,
        epilog="For gated new-prompt rank-study requests, use: native_causvid.py worker --help",
    )
    cli.add_argument("command", choices=("preflight", "run"))
    cli.add_argument("--model-code-root", type=Path, required=True, help="Explicit RAVEN native source root")
    cli.add_argument("--base-model", type=Path, required=True, help="Explicit Wan2.1-T2V-1.3B component directory")
    cli.add_argument("--checkpoint", type=Path, required=True, help="Explicit CausVid model.pt")
    cli.add_argument("--config", type=Path, required=True, help="Explicit native CausVid YAML")
    cli.add_argument("--case-manifest", type=Path, required=True, help="Completed pilot t00/t01 results.json, read-only")
    cli.add_argument("--run-id", required=True, help="New immutable artifact directory name")
    cli.add_argument("--output-root", type=Path, help="Explicit artifact parent, useful with a frozen source tree")
    cli.add_argument("--threads", type=int, default=4)
    cli.add_argument("--gpu", type=int, choices=(0,), default=0)
    cli.add_argument("--allow-cuda", action="store_true", help="Acknowledge separate coordinator GPU authorization")
    cli.add_argument("--chunks", type=int, choices=(3, 4), default=3)
    cli.add_argument(
        "--arms", nargs="+", choices=("raw", "svd", "svd_hevc"), default=["raw"],
        help="RAW by default; lossy arms are explicit experiments, not quality-qualified policy selections",
    )
    cli.add_argument("--fork-seed", type=int, default=1)
    cli.add_argument("--fork-chunks", type=int, choices=(1, 2), default=1)
    cli.add_argument("--history-mib", type=int, default=512, help="Shared lease admission including decoder workspace")
    cli.add_argument("--attention-mib", type=int, default=1024, help="Conservative complete per-layer visible-history bound")
    cli.add_argument("--ffmpeg", type=Path, help="Explicit FFmpeg binary; otherwise use provider discovery")
    cli.add_argument(
        "--hevc-inter-prediction", action=argparse.BooleanOptionalAction, default=False,
        help="svd_hevc GOP: all-intra by default; enable the documented inter GOP explicitly",
    )
    return cli


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments[:1] == ["worker"]:
        sys.path.insert(0, str(ROOT))
        from examples.causvid_rank_worker import main as worker_main

        return worker_main(arguments[1:])
    args = parser().parse_args(arguments)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,100}", args.run_id):
        raise ValueError("run-id must be a new simple alphanumeric identifier")
    if args.command == "run" and not args.allow_cuda:
        raise ValueError("Native launch requires --allow-cuda after explicit coordinator authorization")
    if args.arms[0] != "raw" or len(args.arms) != len(set(args.arms)):
        raise ValueError("Run RAW first, followed by explicitly requested unique lossy arms")
    if args.fork_seed < 0 or min(args.history_mib, args.attention_mib) <= 0:
        raise ValueError("Invalid fork seed or residency budget")
    output_root = ROOT / "artifacts/native_causvid" if args.output_root is None else args.output_root.resolve()
    directory = output_root / args.run_id
    directory.mkdir(parents=True, exist_ok=False)
    environment(directory, gpu=args.gpu if args.command == "run" else None, threads=args.threads)
    record = {
        "command": shlex.join([
            sys.executable, "-B", str(Path(__file__).resolve()),
            *(sys.argv[1:] if argv is None else argv),
        ]),
        "python": sys.executable, "pid": os.getpid(), "host": os.uname().nodename,
        "cwd": str(Path.cwd()), "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "environment": {
            key: value for key, value in os.environ.items()
            if key in ("CUDA_VISIBLE_DEVICES", "TRITON_CACHE_DIR", "TMPDIR", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
        },
    }
    write_json(directory / "execution.json", record)
    try:
        provenance = preflight(args)
        write_json(directory / "preflight.json", provenance)
        if args.command == "preflight":
            print(json.dumps({"status": provenance["status"], "path": str(directory), "model_loaded": False}))
        else:
            result = run_verification(args, directory, provenance)
            print(json.dumps({"status": result["status"], "path": str(directory / "result.json")}))
    except BaseException as exc:
        write_json(directory / "failure.json", {
            "status": "failed", "error": repr(exc), "traceback": traceback.format_exc(),
            "utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        })
        raise


if __name__ == "__main__":
    main()
