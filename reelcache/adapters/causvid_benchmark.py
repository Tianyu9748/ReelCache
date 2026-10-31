"""Fresh CausVid execution and accounting helpers, not an official VBench scorer.

The common prefix is denoised from the case's own noise and conditioning.
Cross-codec execution forks explicitly recapture that clean prefix; they never
reinterpret post-RoPE RAW packets as pre-RoPE factorization inputs.
"""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import time

import torch

from .causvid import CausVidAdapter, tensor_digest
from reelcache.errors import IntegrityError, StateError
from reelcache.serialization import serialize_tensors
from reelcache.types import CodecSpec, MatrixKey


def object_graph(store, roots):
    """Count each complete backing object once; blob and basis IDs are leaves."""
    pending, seen = list(roots), set()
    sizes = {"blob": 0, "basis": 0, "manifest": 0}
    allocated = 0
    while pending:
        object_id = pending.pop()
        if object_id in seen:
            continue
        seen.add(object_id)
        kind = object_id.split(":", 1)[0]
        stat = store.path_for(object_id).stat()
        sizes[kind] += stat.st_size
        allocated += stat.st_blocks * 512
        if kind == "manifest":
            pending.extend(store.references(object_id))
    return {
        "unique_object_count": len(seen), "payload_bytes": sizes["blob"],
        "full_canonical_basis_bytes": sizes["basis"], "metadata_bytes": sizes["manifest"],
        "unique_file_bytes": sum(sizes.values()), "allocated_filesystem_bytes": allocated,
    }


def checkpoint_bytes(runtime, branch, snapshot, config):
    head = runtime.branch_info(branch)
    record = runtime.store.get_manifest(snapshot)
    if record["head"] != head["head_id"]:
        raise IntegrityError("Byte accounting snapshot and branch refer to different clean heads")
    dtype = getattr(torch, runtime.model.dtype)
    numerator = (
        head["token_end"] * config.channels * runtime.model.layers
        * 2 * torch.empty((), dtype=dtype).element_size()
    )
    decoder = object_graph(runtime.store, [head["head_id"]])
    complete = object_graph(runtime.store, [snapshot])
    return {
        "native_dense_KV_bytes": numerator,
        "native_dtype": runtime.model.dtype,
        "token_start": 0, "token_end": head["token_end"],
        "decoder": decoder, "snapshot": complete,
        "snapshot_execution_and_root_extra_bytes": complete["unique_file_bytes"] - decoder["unique_file_bytes"],
        "native_KV_over_decoder": numerator / decoder["unique_file_bytes"],
        "native_KV_over_snapshot": numerator / complete["unique_file_bytes"],
        "numerator_convention": "native K+V element bytes across all KV heads/layers and all sealed chunks; prefix once",
        "denominator_convention": "actual unique reachable object file bytes, full canonical bases and manifests",
        "decoder_excludes_execution_state": True,
        "snapshot_includes_execution_state": True,
        "store_index_and_external_outputs_excluded": True,
    }


def write_manifest_ledger(runtime, roots, path):
    pending, seen = list(roots), set()
    with Path(path).open("x") as stream:
        while pending:
            object_id = pending.pop()
            if object_id in seen:
                continue
            seen.add(object_id)
            kind = object_id.split(":", 1)[0]
            row = {"object_id": object_id, "kind": kind, "bytes": runtime.store.object_size(object_id)}
            if kind == "manifest":
                data, references = runtime.store.get_manifest_with_references(object_id)
                row.update(manifest=data, references=list(references))
                pending.extend(references)
            else:
                runtime.store.get(object_id)
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    return {
        **_file_record(Path(path)), "unique_object_count": len(seen),
        "roots": sorted(set(roots)), "every_reachable_object_hash_verified": True,
        "large_payload_or_basis_bytes_exported": False,
    }


def filesystem_bytes(root):
    root = Path(root)
    groups = {name: {"file_bytes": 0, "allocated_bytes": 0, "unique_inodes": 0}
              for name in ("objects", "index", "other")}
    seen = set()
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        info = path.stat()
        inode = (info.st_dev, info.st_ino)
        if inode in seen:
            continue
        seen.add(inode)
        relative = path.relative_to(root)
        group = "objects" if relative.parts[0] == "objects" else (
            "index" if relative.name.startswith("index.sqlite3") else "other"
        )
        groups[group]["file_bytes"] += info.st_size
        groups[group]["allocated_bytes"] += info.st_blocks * 512
        groups[group]["unique_inodes"] += 1
    return {
        "root": str(root), "groups": groups,
        "file_bytes": sum(row["file_bytes"] for row in groups.values()),
        "allocated_bytes": sum(row["allocated_bytes"] for row in groups.values()),
        "scope": "runtime directory only; inode-deduplicated stat sizes and st_blocks*512; index separate",
        "physical_blocks_are_not_an_allocator_or_compression_ratio": True,
    }


def allocator_sample(device):
    value = {
        "host_process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "completed_child_max_rss_bytes": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss * 1024,
        "host_peak_scope": "process lifetime; child maximum is separate, not a simultaneous tree peak",
    }
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)
        value.update(
            GPU_allocated_bytes=torch.cuda.memory_allocated(device),
            GPU_reserved_bytes=torch.cuda.memory_reserved(device),
            GPU_peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
            GPU_peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
        )
    return value


def rgb_uint8(video):
    import numpy as np

    if (
        not isinstance(video, torch.Tensor) or video.ndim != 5 or video.shape[0] != 1
        or video.shape[2] != 3 or not video.dtype.is_floating_point or not bool(torch.isfinite(video).all())
    ):
        raise IntegrityError("Expected finite native VAE [1,F,3,H,W] floating-point RGB output")
    return np.ascontiguousarray(
        (video[0].float().clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy() * 255).astype("uint8")
    )


def paired_ssim(reference, candidate):
    import numpy as np
    from skimage.metrics import structural_similarity

    if (
        reference.dtype != np.uint8 or candidate.dtype != np.uint8
        or reference.shape != candidate.shape or reference.ndim != 4
        or reference.shape[-1] != 3 or not reference.shape[0]
        or min(reference.shape[1:3]) < 7
    ):
        raise IntegrityError("Paired SSIM requires matching nonempty uint8 F,H,W,3 frames, H/W >= 7")
    scores = [
        float(structural_similarity(left, right, channel_axis=2, data_range=255))
        for left, right in zip(reference, candidate)
    ]
    return {
        "mean": float(np.mean(scores)), "minimum": min(scores), "per_frame": scores,
        "frame_count": len(scores), "data_range": 255, "channel_axis": 2,
        "implementation": "skimage.metrics.structural_similarity defaults; win_size=7; uniform weights",
        "scope": "paired same-case continuation uint8 RGB; not an official VBench score",
    }


def encode_lossless_video(frames, path, *, ffmpeg, fps):
    height, width = frames.shape[1:3]
    command = [
        str(ffmpeg), "-nostdin", "-hide_banner", "-loglevel", "error", "-n",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
        "-r", str(fps), "-i", "pipe:0", "-an", "-c:v", "libx264rgb", "-crf", "0",
        "-preset", "ultrafast", "-threads", "1", str(path),
    ]
    encoded = subprocess.run(command, input=frames.tobytes(), capture_output=True, timeout=600)
    if encoded.returncode:
        raise RuntimeError("Lossless evaluator-video encode failed: " + encoded.stderr.decode(errors="replace"))
    decoded = subprocess.run(
        [str(ffmpeg), "-nostdin", "-hide_banner", "-loglevel", "error", "-i", str(path),
         "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"],
        capture_output=True, timeout=600,
    )
    expected = hashlib.sha256(frames.tobytes()).hexdigest()
    if decoded.returncode or len(decoded.stdout) != frames.nbytes or hashlib.sha256(decoded.stdout).hexdigest() != expected:
        raise IntegrityError("Evaluator video did not roundtrip the measured uint8 RGB exactly")
    return {
        "command": command, "codec": "libx264rgb", "crf": 0, "fps": fps,
        "decoded_RGB_sha256": expected, "lossless_RGB_roundtrip_verified": True,
        "verified_decoder": "explicit FFmpeg rgb24; official VBench loader validation is separate",
    }


def _file_record(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(4 * 1024**2), b""):
            digest.update(part)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def save_evaluator_output(directory, latents, frames, rng_state, *, prefix_frames, fps, ffmpeg=None):
    import numpy as np

    directory.mkdir()
    continuation = np.ascontiguousarray(frames[prefix_frames:])
    rgb_path, state_path = directory / "continuation_rgb.npy", directory / "latents.reelcache"
    with rgb_path.open("xb") as stream:
        np.save(stream, continuation, allow_pickle=False)
    with state_path.open("xb") as stream:
        stream.write(serialize_tensors({"latents": latents.cpu(), "rng_state": rng_state.cpu()}))
    result = {
        "RGB": _file_record(rgb_path), "latent_state": _file_record(state_path),
        "continuation_shape": list(continuation.shape), "RGB_dtype": "uint8", "RGB_layout": "F,H,W,C",
        "RGB_quantization": "native [0,1] clamp; multiply by 255; uint8 truncation, no lossy transcode",
        "full_latents_sha256": tensor_digest(latents), "rng_sha256": tensor_digest(rng_state),
        "prefix_RGB_sha256": hashlib.sha256(frames[:prefix_frames].tobytes()).hexdigest(),
        "continuation_RGB_sha256": hashlib.sha256(continuation.tobytes()).hexdigest(),
        "fps": fps, "prefix_frames_excluded_from_evaluation": prefix_frames,
    }
    if ffmpeg is not None:
        video = directory / "continuation.mp4"
        result["video_encoding"] = encode_lossless_video(continuation, video, ffmpeg=ffmpeg, fps=fps)
        result["video"] = _file_record(video)
    return result


@contextmanager
def preserve_global_rng(device):
    devices = [torch.device(device).index or 0] if torch.device(device).type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        yield


@torch.no_grad()
def fresh_inputs(pipeline, prompt, seed, config, chunks):
    if not isinstance(prompt, str) or not prompt.strip() or type(seed) is not int or seed < 0:
        raise ValueError("Fresh input requires actual prompt text and a nonnegative seed")
    if type(chunks) is not int or chunks < 2 or getattr(pipeline.text_encoder, "training", False):
        raise ValueError("Use at least two chunks and an evaluation-mode native text encoder")
    parameter = next(pipeline.generator.model.parameters())
    encoded = pipeline.text_encoder(text_prompts=[prompt])
    if (
        not isinstance(encoded, dict) or set(encoded) != {"prompt_embeds"}
        or not isinstance(encoded["prompt_embeds"], torch.Tensor)
        or encoded["prompt_embeds"].dtype != parameter.dtype
        or not bool(torch.isfinite(encoded["prompt_embeds"]).all())
    ):
        raise IntegrityError("Native text encoder changed conditioning schema or source dtype")
    conditioning = {"prompt_embeds": encoded["prompt_embeds"].detach().to("cpu", copy=True)}
    rng = torch.Generator(device=parameter.device).manual_seed(seed)
    noise = torch.empty(
        (1, chunks * config.frames_per_chunk, *config.latent_tail),
        device=parameter.device, dtype=parameter.dtype,
    ).normal_(generator=rng).cpu()
    return noise, conditioning, rng


def export_snapshot_objects(source, destination, snapshot_ids):
    """Copy only the selected immutable snapshot closure through public APIs."""
    copied = set()

    def copy_object(object_id):
        if object_id in copied:
            return
        kind = object_id.split(":", 1)[0]
        if kind == "manifest":
            data, refs = source.store.get_manifest_with_references(object_id)
            for child in refs:
                copy_object(child)
            created = destination.store.put_manifest(data, references=refs)
        else:
            created = destination.store.put(source.store.get(object_id), kind=kind)
        if created != object_id:
            raise IntegrityError("A selected checkpoint changed content identity during export")
        copied.add(object_id)

    for snapshot in snapshot_ids:
        source.load_snapshot(snapshot)
        copy_object(snapshot)
        destination.store.set_ref("snapshots/" + snapshot.split(":", 1)[1], snapshot)
        destination.load_snapshot(snapshot)
    return {"snapshot_ids": list(snapshot_ids), "unique_copied_objects": len(copied)}


@torch.no_grad()
def generate_matched_case(
    pipeline, runtime, *, config, noise, conditioning, rng, conditioning_id,
    arms, directory, prefix_rgb_frames=9, fps=24, history_attention_bytes=4 * 1024**3,
    hot_history_bytes=0, replay_chunks=0, fork_seed=1, ffmpeg=None,
    lossy_ranks=None, rgb_size=None,
    progress=lambda _phase, _detail: None,
):
    """Run one fresh prefix, fixed-profile execution forks, and optional replay.

    Callers own model loading, source/protocol validation and disk placement.
    The production driver passes only actual fresh text-encoder conditioning.
    CPU tests use fake native layers and are never native/scientific evidence.
    """
    import numpy as np

    directory = Path(directory)
    if arms not in (["raw", "svd_hevc"], ["raw", "svd", "svd_hevc"]):
        raise ValueError("Use RAW and SVD_HEVC, with optional intervening SVD diagnostic")
    total_chunks = noise.shape[1] // config.frames_per_chunk
    if not 0 <= replay_chunks < total_chunks or fork_seed < 0:
        raise ValueError("Invalid bounded replay/fork scope")
    device = next(pipeline.generator.model.parameters()).device
    lossy_ranks = {"K": 256, "V": 384} if lossy_ranks is None else dict(lossy_ranks)
    common_branch = "common-prefix"
    reports = {"arms": {}, "checks": {}, "prefix_generation": "fresh native denoising from this prompt/noise"}
    snapshots = []
    maximums = {}
    cold_raw_specs = CodecSpec()

    def sample(label):
        row = allocator_sample(device)
        for name, value in row.items():
            if isinstance(value, int):
                maximums[name] = max(maximums.get(name, 0), value)
        progress(label, {"memory": row})
        return row

    def step(adapter, label, *, prefix=None):
        progress(label, {"phase": "before_clean_chunk", "branch": adapter.branch_id})
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        before = sample(label + "/before")
        start = time.perf_counter()
        chunk = adapter.step(clean_prefix=prefix)
        duration = time.perf_counter() - start
        after = sample(label + "/after")
        owned = adapter.memory_report()
        if (
            runtime.active_leases or owned["live_mutable_kv_bytes"] or owned["live_attention_read_bytes"]
            or owned["live_clean_capture_cpu_bytes"] or owned["live_hot_history_bytes"]
        ):
            raise StateError("A clean sealed boundary retained native/history allocations")
        snapshot_start = time.perf_counter()
        snapshot = adapter.snapshot()
        snapshots.append(snapshot)
        accounting = checkpoint_bytes(runtime, adapter.branch_id, snapshot, config)
        row = {
            "chunk_id": adapter.frame_cursor // config.frames_per_chunk - 1,
            "chunk_manifest": chunk, "snapshot": snapshot, "seconds": duration,
            "snapshot_and_accounting_seconds": time.perf_counter() - snapshot_start,
            "memory_before": before, "memory_after": after, "owned_memory": owned,
            "runtime_residency": runtime.residency.stats(), "bytes": accounting,
            "rng_sha256": tensor_digest(adapter.rng_state), "clean_audit": adapter.audit[-1],
        }
        progress(label, {"phase": "clean_chunk_sealed", "record": row})
        return row

    with CausVidAdapter(
        pipeline, runtime, common_branch, config=config, specs=cold_raw_specs,
        max_attention_bytes=history_attention_bytes, history_mode="reload",
    ) as adapter:
        adapter.start(noise, conditioning, rng, conditioning_id=conditioning_id)
        common = step(adapter, "common-prefix/fresh-denoising")
        common_snapshot = common["snapshot"]
        common_latents, common_rng = adapter.latent_state, adapter.rng_state
        common_head = runtime.branch_info(common_branch)
        if adapter.audit[-1]["supplied_clean_prefix"] or not adapter.audit[-1]["denoising_steps"]:
            raise IntegrityError("The benchmark common prefix was not freshly denoised")
    reports["common_prefix"] = {
        **common, "head": common_head, "latent_sha256": tensor_digest(common_latents),
        "conditioning_sha256": tensor_digest(conditioning["prompt_embeds"]),
        "initial_noise_sha256": tensor_digest(noise), "rng_after_prefix_sha256": tensor_digest(common_rng),
        "historical_prefix_inputs_used": False,
    }
    baseline_latents = baseline_frames = None
    baseline_rows = None
    svd_frames = None
    for kind in arms:
        specs = {
            MatrixKey(layer, target): (
                CodecSpec() if kind == "raw" else
                CodecSpec(kind, rank=lossy_ranks[target], normalization="per_channel",
                          protected_fraction=1.0, pix_fmt="yuv444p", qp=8, preset="ultrafast")
            ) for layer in range(runtime.model.layers) for target in ("K", "V")
        }
        arm = {
            "kind": kind, "chunks": [], "reuse_threshold": None, "closed_loop": True,
            "primary_arm": kind in ("raw", "svd_hevc"), "diagnostic_only": kind == "svd",
            "codec_specs": {target: specs[MatrixKey(0, target)].to_dict() for target in ("K", "V")},
        }
        tick = time.perf_counter()
        with CausVidAdapter(
            pipeline, runtime, kind + "-prefix", config=config, specs=specs,
            max_attention_bytes=history_attention_bytes, max_hot_history_bytes=hot_history_bytes,
            history_mode="reload",
        ) as adapter:
            if kind == "raw":
                prefix_snapshot = common_snapshot
                arm["chunks"].append(common)
                arm["prefix_initialization"] = "exact snapshot fork"
            else:
                prefix_rng = torch.Generator(device=device).set_state(common_rng)
                adapter.start(noise, conditioning, prefix_rng, conditioning_id=conditioning_id)
                prefix_row = step(adapter, kind + "/native-clean-prefix-recapture", prefix=common_latents)
                if not torch.equal(adapter.latent_state, common_latents) or not torch.equal(adapter.rng_state, common_rng):
                    raise IntegrityError("Cross-profile prefix recapture changed clean latents or RNG")
                prefix_snapshot = prefix_row["snapshot"]
                arm["chunks"].append(prefix_row)
                arm["prefix_initialization"] = "native clean recapture from fresh common latents; never RAW packet recoding"
            adapter.fork(prefix_snapshot, kind, conditioning_id=conditioning_id)
            arm["prefix_snapshot"] = prefix_snapshot
            while not adapter.finished:
                arm["chunks"].append(step(adapter, f"{kind}/chunk{adapter.frame_cursor // config.frames_per_chunk}"))
            latents = adapter.latent_state
            progress(kind + "/full-prefix-VAE", {"phase": "decoding"})
            frames = rgb_uint8(adapter.decode())
            if rgb_size is not None and tuple(frames.shape[1:3]) != tuple(rgb_size):
                raise IntegrityError("Native VAE spatial geometry differs from the frozen protocol")
            sample(kind + "/after-VAE")
            expected_total_rgb = 4 * (noise.shape[1] - 1) + 1
            if len(frames) != expected_total_rgb or not torch.equal(latents[:, :config.frames_per_chunk], common_latents):
                raise IntegrityError("Native temporal geometry or common prefix changed")
            if kind == "raw":
                baseline_latents, baseline_frames, baseline_rows = latents, frames, arm["chunks"]
            else:
                if not np.array_equal(frames[:prefix_rgb_frames], baseline_frames[:prefix_rgb_frames]):
                    raise IntegrityError("Full-prefix VAE changed the common prefix RGB")
                if not torch.equal(adapter.rng_state, reference_rng):
                    raise IntegrityError("The representation changed the matched denoising RNG")
            reference_rng = adapter.rng_state if kind == "raw" else reference_rng
            arm["paired_RAW_ssim"] = paired_ssim(baseline_frames[prefix_rgb_frames:], frames[prefix_rgb_frames:])
            if kind == "svd":
                svd_frames = frames
            elif kind == "svd_hevc" and svd_frames is not None:
                arm["paired_SVD_ssim"] = paired_ssim(svd_frames[prefix_rgb_frames:], frames[prefix_rgb_frames:])
            arm["outputs"] = save_evaluator_output(
                directory / kind, latents, frames, adapter.rng_state,
                prefix_frames=prefix_rgb_frames, fps=fps, ffmpeg=ffmpeg,
            )
            sample(kind + "/after-evaluator-output")
            arm["final_head"] = runtime.branch_info(kind)
            arm["final_snapshot"] = arm["chunks"][-1]["snapshot"]
            arm["final_bytes"] = arm["chunks"][-1]["bytes"]
            arm["closed_loop_clean_chunks"] = len(arm["chunks"])
            if len(arm["chunks"]) != total_chunks or arm["final_head"]["token_end"] != total_chunks * config.chunk_tokens:
                raise IntegrityError("Not every selected clean chunk was sealed")
        arm["seconds"] = time.perf_counter() - tick
        reports["arms"][kind] = arm
        if runtime.branch_info(common_branch) != common_head:
            raise IntegrityError("An execution fork mutated the common-prefix parent")
    if replay_chunks:
        parent = runtime.branch_info("raw")
        children = []
        with CausVidAdapter(
            pipeline, runtime, "checks", config=config, specs=cold_raw_specs,
            max_attention_bytes=history_attention_bytes, max_hot_history_bytes=0,
            history_mode="reload",
        ) as adapter:
            adapter.restore(common_snapshot, "raw-uncached-replay")
            for index in range(replay_chunks):
                step(adapter, f"checks/raw-uncached-replay/{index}")
            end = adapter.frame_cursor
            if (
                not torch.equal(adapter.latent_state, baseline_latents[:, :end])
                or tensor_digest(adapter.rng_state) != baseline_rows[replay_chunks]["rng_sha256"]
            ):
                raise IntegrityError("Uncached RAW snapshot replay differs from the matched RAW arm")
            replay_frames = rgb_uint8(adapter.decode())
            if not np.array_equal(replay_frames, baseline_frames[:len(replay_frames)]):
                raise IntegrityError("Uncached RAW replay RGB differs at the same absolute prefix positions")
            for name in ("seed-fork", "seed-fork-replay"):
                adapter.fork(common_snapshot, name, seed=fork_seed)
                for index in range(replay_chunks):
                    step(adapter, f"checks/{name}/{index}")
                current = (adapter.latent_state, rgb_uint8(adapter.decode()), adapter.rng_state)
                if torch.equal(current[0][:, config.frames_per_chunk:], baseline_latents[:, config.frames_per_chunk:end]):
                    raise IntegrityError("Changed-seed fork did not change the continuation")
                children.append(current)
            left, right = children
            if not torch.equal(left[0], right[0]) or not np.array_equal(left[1], right[1]) or not torch.equal(left[2], right[2]):
                raise IntegrityError("Changed-seed fork replay differs in latents, RGB or RNG")
        if runtime.branch_info("raw") != parent or runtime.branch_info(common_branch) != common_head:
            raise IntegrityError("Replay/fork changed an immutable parent")
        runtime.store.verify(common_snapshot)
        reports["checks"] = {
            "uncached_RAW_replay_latents_and_rng_equal": True,
            "uncached_RAW_replay_RGB_equal": True,
            "changed_seed_fork_latents_RGB_rng_replay_equal": True,
            "parent_and_common_prefix_unchanged": True, "common_snapshot_hashes_verified": True,
            "replay_chunks": replay_chunks, "fork_seed": fork_seed,
            "checks_are_not_additional_quality_samples": True,
        }
    for kind, arm in reports["arms"].items():
        arm["RAW_archive_over_decoder"] = (
            reports["arms"]["raw"]["final_bytes"]["decoder"]["unique_file_bytes"]
            / arm["final_bytes"]["decoder"]["unique_file_bytes"]
        )
    progress("case/final-manifest-ledger", {"phase": "verifying_all_reachable_object_hashes"})
    verification_start = time.perf_counter()
    reports["manifest_ledger"] = write_manifest_ledger(runtime, snapshots, directory / "manifest_ledger.jsonl")
    reports["final_integrity_verification_seconds"] = time.perf_counter() - verification_start
    sample("case/after-all-checks")
    maximums["owned_hot_history_bytes"] = max(
        row["owned_memory"]["peak_hot_history_bytes"]
        for arm in reports["arms"].values() for row in arm["chunks"]
    )
    reports.update(
        status="passed", selected_chunks=total_chunks, continuation_frames=len(baseline_frames) - prefix_rgb_frames,
        fps=fps, peak_observations=maximums, global_store_stats=runtime.store.stats(),
        official_VBench_status="pending_parent_official_evaluator",
        store_retention="active store retained locally; compact export excludes all RAW stores by default",
        hot_history_budget_bytes=hot_history_bytes,
    )
    return reports
