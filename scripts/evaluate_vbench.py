#!/usr/bin/env python3
"""Reproducible, explicitly partial official VBench evaluation.

Commands:
  candidate  Build an UNFROZEN annotation-stratified prompt proposal.
  inventory  Inventory a pinned official checkout, dependencies and local weights.
  prepare    Validate paired RGB, compute SSIM, encode/verify lossless RGB MP4.
  evaluate   Run requested official dimensions, only with explicit model permission.

The input schema is illustrated by ``example-inputs``. All source arrays remain
authoritative for SSIM. VBench receives libx264rgb CRF 0 MP4, decoded identically
by FFmpeg, OpenCV AND Decord before evaluation. Missing dimensions fail or are
explicitly inapplicable; they never silently become a score or a full aggregate.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import copy
import gc
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

try:
    from . import _vbench_official as official
    from . import _vbench_protocol as protocol
except ImportError:
    import _vbench_official as official
    import _vbench_protocol as protocol


def array_sha256(frames):
    result = hashlib.sha256()
    for frame in frames:
        result.update(np.ascontiguousarray(frame).tobytes())
    return result.hexdigest()


@contextmanager
def rgb_window(clip, input_base, model_geometry, *, verify_file=True):
    path = Path(clip["rgb_path"])
    if not path.is_absolute():
        path = Path(input_base) / path
    path = path.resolve()
    protocol.require(path.suffix.lower() in (".npy", ".npz"), "Only original .npy/.npz RGB is accepted")
    if verify_file:
        protocol.require(protocol.sha256_file(path) == clip["rgb_file_sha256"], "Source RGB file hash mismatch")
    archive = None
    frames = None
    try:
        if path.suffix.lower() == ".npy":
            protocol.require(clip.get("array_key") is None, "array_key applies only to NPZ")
            frames = np.load(path, mmap_mode="r", allow_pickle=False)
        else:
            archive = np.load(path, allow_pickle=False)
            key = clip.get("array_key", "frames")
            protocol.require(key in archive.files, f"NPZ has no array {key!r}")
            frames = archive[key]
        protocol.require(frames.dtype == np.uint8 and frames.ndim == 4 and frames.shape[-1] == 3,
                         "RGB must have exact uint8 [T,H,W,3] layout; no implicit conversion")
        protocol.require(tuple(frames.shape[1:3]) == (model_geometry["height"], model_geometry["width"]),
                         "Display RGB geometry differs; crop MAGI to 1080 before metric input")
        start, count = clip["rgb_frame_start"], clip["frame_count"]
        protocol.require(start + count <= frames.shape[0], "Insufficient genuine continuation frames")
        window = frames[start:start + count]
        yield window, {
            "path": str(path), "file_sha256": clip["rgb_file_sha256"],
            "array_key": clip.get("array_key") if path.suffix.lower() == ".npz" else None,
            "source_shape": list(frames.shape), "window_shape": list(window.shape),
            "rgb_sha256": array_sha256(window), "source_indices": list(range(start, start + count)),
            "dtype": "uint8", "color_order": "RGB", "resampling": False,
        }
    finally:
        if archive is not None:
            archive.close()
        if isinstance(frames, np.memmap):
            frames._mmap.close()


def ssim_frames(reference, candidate, workers=1):
    from skimage.metrics import structural_similarity

    protocol.integer(workers, "ssim_workers", 1)
    protocol.require(reference.shape == candidate.shape and reference.dtype == candidate.dtype == np.uint8,
                     "SSIM requires matched uint8 RGB arrays")
    protocol.require(reference.ndim == 4 and reference.shape[-1] == 3
                     and reference.shape[0] > 0 and min(reference.shape[1:3]) >= 7, "Invalid SSIM geometry")

    def score(index):
        value = float(structural_similarity(reference[index], candidate[index], **protocol.SSIM_PARAMETERS))
        protocol.require(math.isfinite(value), "Nonfinite SSIM")
        return value

    with ThreadPoolExecutor(max_workers=workers) as pool:
        scores = list(pool.map(score, range(len(reference))))
    return {
        "mean": math.fsum(scores) / len(scores), "minimum_frame": min(scores), "per_frame": scores,
        "parameters": protocol.SSIM_PARAMETERS.copy(),
        "implementation": "skimage.metrics.structural_similarity",
        "implementation_sha256": protocol.sha256_file(inspect.getfile(structural_similarity)),
        "versions": {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "scikit-image")},
        "window": "continuation only; native common prefix excluded",
    }


def ffmpeg_executable(path=None):
    if path is None:
        import imageio_ffmpeg
        path = imageio_ffmpeg.get_ffmpeg_exe()
    result = shutil.which(str(path))
    protocol.require(result is not None, f"FFmpeg is unavailable: {path}")
    return str(Path(result).resolve())


def _stop_owned(process):
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)


def encode_lossless_rgb(frames, fps, path, ffmpeg):
    path = Path(path).resolve()
    protocol.require(path.suffix == ".mp4" and not path.exists(), "Use a new .mp4 output path")
    protocol.require(frames.dtype == np.uint8 and frames.ndim == 4 and frames.shape[-1] == 3,
                     "Lossless transport requires RGB uint8")
    _, height, width, _ = frames.shape
    path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-n",
        "-f", "rawvideo", "-pixel_format", "rgb24", "-video_size", f"{width}x{height}",
        "-framerate", str(fps), "-i", "pipe:0", "-an", "-c:v", "libx264rgb",
        "-crf", "0", "-preset", "ultrafast", "-pix_fmt", "rgb24",
        "-color_range", "pc", "-threads", "1", "-movflags", "+faststart", str(path),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                               stderr=subprocess.PIPE)
    try:
        for frame in frames:
            process.stdin.write(np.ascontiguousarray(frame).tobytes())
        process.stdin.close()
        error = process.stderr.read().decode(errors="replace")
        code = process.wait(timeout=120)
        protocol.require(code == 0 and path.is_file(), f"FFmpeg RGB encode failed ({code}): {error}")
    except Exception:
        _stop_owned(process)
        raise
    finally:
        process.stderr.close()
        if not process.stdin.closed:
            process.stdin.close()
    return command


def verify_video(frames, fps, path, ffmpeg):
    """Verify the actual three decoders used by transport/official dimensions."""
    import cv2
    import decord

    path = str(Path(path).resolve())
    frame_bytes = int(np.prod(frames.shape[1:]))
    command = [ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error",
               "-threads", "1", "-i", path, "-map", "0:v:0",
               "-f", "rawvideo", "-pix_fmt", "rgb24", "-threads", "1", "pipe:1"]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        for expected in frames:
            raw = process.stdout.read(frame_bytes)
            protocol.require(raw == np.ascontiguousarray(expected).tobytes(), "FFmpeg decoded RGB differs")
        protocol.require(process.stdout.read(1) == b"", "FFmpeg found extra video frames")
        error = process.stderr.read().decode(errors="replace")
        protocol.require(process.wait(timeout=120) == 0, f"FFmpeg decode failed: {error}")
    except Exception:
        _stop_owned(process)
        raise
    finally:
        process.stdout.close()
        process.stderr.close()
    reader = cv2.VideoCapture(path)
    try:
        protocol.require(reader.isOpened(), "OpenCV cannot decode RGB MP4")
        observed_fps = float(reader.get(cv2.CAP_PROP_FPS))
        protocol.require(math.isclose(observed_fps, fps, rel_tol=0, abs_tol=1e-6), "OpenCV FPS mismatch")
        for expected in frames:
            success, bgr = reader.read()
            protocol.require(success and bgr.shape == expected.shape
                             and np.array_equal(bgr[..., ::-1], expected), "OpenCV decoded RGB differs")
        protocol.require(not reader.read()[0], "OpenCV found extra frames")
    finally:
        reader.release()
    decord.bridge.set_bridge("native")
    reader = decord.VideoReader(path, num_threads=1)
    try:
        protocol.require(len(reader) == len(frames), "Decord frame count mismatch")
        decord_fps = float(reader.get_avg_fps())
        protocol.require(math.isclose(decord_fps, fps, rel_tol=0, abs_tol=1e-6), "Decord FPS mismatch")
        for index, expected in enumerate(frames):
            decoded = reader[index].asnumpy()
            protocol.require(decoded.shape == expected.shape,
                             "Decord RGB geometry differs (0.6.0 can mis-handle non-32-aligned RGB widths); no padding is allowed")
            protocol.require(np.array_equal(decoded, expected), "Decord decoded RGB differs")
        for start in range(0, len(frames), 8):
            indices = list(range(start, min(start + 8, len(frames))))
            batch = reader.get_batch(indices).asnumpy()
            protocol.require(np.array_equal(batch, frames[start:start + len(indices)]),
                             "Decord batch-decoded RGB differs")
    finally:
        del reader
    return {
        "exact_rgb_identity": True, "verified_frames_per_decoder": len(frames),
        "decoders": ["ffmpeg_rgb24", "opencv_bgr_to_rgb", "decord_native_rgb"],
        "opencv_fps": observed_fps, "decord_fps": decord_fps,
        "opencv_version": cv2.__version__, "decord_version": decord.__version__,
        "decord_individual_and_batch_identity": True,
    }


def summarize_ssim(clips):
    groups = {}
    for clip in clips:
        if clip["arm"] == "raw":
            continue
        key = (clip["model"], clip["arm"])
        groups.setdefault(key, []).append(clip)
    return [{
        "model": model, "arm": arm,
        "prompt_ids": [clip["prompt_id"] for clip in rows],
        "case_count": len(rows),
        "mean_case_ssim": math.fsum(clip["ssim"]["mean"] for clip in rows) / len(rows),
        "minimum_frame_ssim": min(clip["ssim"]["minimum_frame"] for clip in rows),
        "case_mean_ssim": {clip["prompt_id"]: clip["ssim"]["mean"] for clip in rows},
        "pooling": "Equal case weights within this model/arm; never pooled across models or unmatched arms.",
    } for (model, arm), rows in sorted(groups.items())]


def prepare(protocol_path, inputs_path, output_dir, *, ffmpeg=None, ssim_workers=1):
    spec, spec_hash = protocol.load_protocol(protocol_path, frozen=True)
    inputs_path = Path(inputs_path).resolve()
    inputs = protocol.validate_inputs(protocol.read_json(inputs_path), spec, spec_hash)
    output_dir = Path(output_dir).resolve()
    protocol.require(not output_dir.exists(), "Prepare requires a new output directory")
    ffmpeg = ffmpeg_executable(ffmpeg)
    required_bytes = sum(
        spec["models"][clip["model"]]["height"] * spec["models"][clip["model"]]["width"]
        * clip["frame_count"] * 3 for clip in inputs["clips"])
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    # RGB video can expand on pathological content; this is an admission reserve,
    # not a promise of peak memory or total native-run storage.
    reserve = required_bytes * 2 + (64 << 20)
    protocol.require(shutil.disk_usage(output_dir.parent).free >= reserve,
                     f"Insufficient media reserve: need {reserve} free bytes")
    output_dir.mkdir()
    prepared = {
        "schema": protocol.SCHEMAS["prepared"], "status": "in_progress",
        "protocol_path": str(Path(protocol_path).resolve()), "protocol_sha256": spec_hash,
        "input_manifest_path": str(inputs_path), "input_manifest_sha256": protocol.sha256_file(inputs_path),
        "input_manifest": inputs, "clips": [],
        "video_codec": {"encoder": "libx264rgb", "crf": 0, "preset": "ultrafast",
                        "container": "mp4", "input_pixel_format": "rgb24",
                        "range": "full", "lossy_conversion": False},
        "ffmpeg": {"path": ffmpeg, "sha256": protocol.sha256_file(ffmpeg),
                   "version": subprocess.check_output([ffmpeg, "-version"], text=True).splitlines()[0]},
        "script_sha256": protocol.sha256_file(__file__),
        "versions": official.version_inventory(),
        "media_admission_reserve_bytes": reserve, "models_executed": False,
        "planned_primary_coverage": protocol.primary_coverage(spec, inputs["clips"]),
    }
    by_id = {clip["id"]: clip for clip in inputs["clips"]}
    try:
        for clip in inputs["clips"]:
            geometry = spec["models"][clip["model"]]
            row = copy.deepcopy(clip)
            path = output_dir / "videos" / f"{clip['id']}.mp4"
            with rgb_window(clip, inputs_path.parent, geometry) as (frames, rgb):
                command = encode_lossless_rgb(frames, clip["fps"], path, ffmpeg)
                verified = verify_video(frames, clip["fps"], path, ffmpeg)
                row.update(rgb=rgb, video_path=str(path.relative_to(output_dir)),
                           video_sha256=protocol.sha256_file(path), video_bytes=path.stat().st_size,
                           encode_command=command, decode_verification=verified)
                raw = by_id[clip["paired_raw_id"]]
                if clip["arm"] == "raw":
                    row["ssim"] = ssim_frames(frames, frames, ssim_workers)
                else:
                    with rgb_window(raw, inputs_path.parent, geometry) as (reference, _):
                        row["ssim"] = ssim_frames(reference, frames, ssim_workers)
                scores = row["ssim"]["per_frame"]
                block = int(clip["fps"])
                row["ssim"]["consecutive_one_second_blocks"] = [
                    {"start": start, "stop": min(start + block, len(scores)),
                     "mean": math.fsum(scores[start:start + block]) / len(scores[start:start + block]),
                     "minimum": min(scores[start:start + block])}
                    for start in range(0, len(scores), block)
                ] if float(clip["fps"]).is_integer() else []
            if "storage" in clip:
                row["compression"] = protocol.compression_ratios(clip["storage"])
                row["compression"]["provenance"] = "Producer-reported store accounting; not inferred from evaluation video bytes."
            prepared["clips"].append(row)
            protocol.write_json(output_dir / "prepared.json", prepared)
        prepared["ssim_summary"] = summarize_ssim(prepared["clips"])
        prepared["status"] = "complete"
        protocol.write_json(output_dir / "prepared.json", prepared)
        return prepared
    except Exception as exc:
        prepared.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        protocol.write_json(output_dir / "prepared.json", prepared)
        raise


def paired_static_eligibility(clips, video_paths, passes):
    groups = {}
    for clip in clips:
        if str(video_paths[clip["id"]]) in passes:
            key = (clip["model"], clip["prompt_id"], clip["seed"])
            groups.setdefault(key, []).append(clip)
    eligible, details = set(), []
    for (model, prompt_id, seed), rows in sorted(groups.items()):
        pair_pass = all(passes[str(video_paths[row["id"]])] for row in rows)
        if pair_pass:
            eligible.update(str(video_paths[row["id"]]) for row in rows)
        details.append({
            "model": model, "prompt_id": prompt_id, "seed": seed,
            "passes_by_arm": {r["arm"]: passes[str(video_paths[r["id"]])] for r in rows},
            "included_for_every_paired_arm": pair_pass,
            "exclusion_is_not_a_success_or_zero_score": not pair_pass,
        })
    return eligible, details


def human_action_aliases(records, output_dir):
    """UMT's official label parser reads the filename, not prompt_en metadata."""
    result = []
    for index, record in enumerate(records):
        prompt = record["prompt_en"]
        protocol.require(prompt.startswith("A person is ") and "/" not in prompt
                         and "\\" not in prompt and "\x00" not in prompt,
                         "Human-action prompt cannot be represented by the official filename convention")
        name = f"{prompt}-0.mp4"
        protocol.require(len(name.encode()) <= 255, "Human-action filename is too long")
        target = Path(output_dir) / "human_action_inputs" / str(index) / name
        target.parent.mkdir(parents=True)
        protocol.require(len(record["video_list"]) == 1, "Use one clip per explicit human-action record")
        target.symlink_to(Path(record["video_list"][0]).resolve())
        row = copy.deepcopy(record)
        row["video_list"] = [str(target.absolute())]
        result.append(row)
    return result


@contextmanager
def model_settings(device):
    import random
    import torch

    numpy_state, python_state = np.random.get_state(), random.getstate()
    threads = torch.get_num_threads()
    flags = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
             torch.backends.cudnn.benchmark)
    devices = []
    if device.startswith("cuda"):
        torch.cuda.set_device(torch.device(device))
        devices = [torch.cuda.current_device()]
    with torch.random.fork_rng(devices=devices):
        try:
            torch.default_generator.manual_seed(0)
            if devices:
                torch.cuda.manual_seed(0)
            np.random.seed(0)
            random.seed(0)
            torch.set_num_threads(min(8, len(os.sched_getaffinity(0))))
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cudnn.benchmark = False
            yield "cuda" if devices else "cpu"
        finally:
            np.random.set_state(numpy_state)
            random.setstate(python_state)
            torch.set_num_threads(threads)
            torch.backends.cuda.matmul.allow_tf32 = flags[0]
            torch.backends.cudnn.allow_tf32 = flags[1]
            torch.backends.cudnn.benchmark = flags[2]


def summarize_vbench(clips, dimensions, results, *, quality_summary=False):
    groups = {}
    by_id = {clip["id"]: clip for clip in clips}
    for clip in clips:
        groups.setdefault((clip["model"], clip["arm"]), []).append(clip)
    summaries = []
    for (model, arm), rows in sorted(groups.items()):
        summary = {"model": model, "arm": arm, "supplied_prompt_ids": [r["prompt_id"] for r in rows],
                   "dimensions": {}, "full_official_aggregate": False}
        complete_quality = {}
        for dimension in dimensions:
            dimension_result = results.get(dimension, {})
            per_video = dimension_result.get("by_clip", {})
            scored = [r for r in rows if r["id"] in per_video]
            if not scored:
                summary["dimensions"][dimension] = {
                    "status": "uncovered_or_static_excluded", "case_count": 0, "score": None,
                }
                continue
            values = [per_video[r["id"]]["dimension_score"] for r in scored]
            entry = {
                "status": "complete_for_listed_cases", "case_count": len(scored),
                "prompt_ids": [r["prompt_id"] for r in scored],
                "score": math.fsum(values) / len(values),
                "per_case": {r["prompt_id"]: per_video[r["id"]]["dimension_score"] for r in scored},
            }
            paired = [r for r in scored if r["paired_raw_id"] in per_video]
            if arm != "raw":
                deltas = {r["prompt_id"]: per_video[r["id"]]["dimension_score"]
                          - per_video[r["paired_raw_id"]]["dimension_score"] for r in paired}
                entry["matched_candidate_minus_raw"] = {
                    "case_count": len(paired), "per_case": deltas,
                    "mean": math.fsum(deltas.values()) / len(deltas) if deltas else None,
                }
                if dimension == "dynamic_degree":
                    entry["raw_motion_classification_preserved"] = {
                        r["prompt_id"]: per_video[r["id"]]["dimension_score"]
                        == per_video[by_id[r["paired_raw_id"]]["id"]]["dimension_score"]
                        for r in paired}
            summary["dimensions"][dimension] = entry
            if dimension in protocol.QUALITY:
                complete_quality[dimension] = entry["score"]
        if quality_summary:
            if set(complete_quality) == set(protocol.QUALITY):
                summary["quality_formula_pilot_only"] = protocol.quality_aggregate(complete_quality)
            else:
                summary["quality_formula_pilot_only"] = {
                    "status": "unavailable", "missing_dimensions": sorted(set(protocol.QUALITY) - set(complete_quality)),
                }
        summaries.append(summary)
    return summaries


def evaluate(prepared_path, assets_path, output_dir, dimensions, *, device="cpu",
             model_execution_authorized=False, gpu_grant=None, quality_mode="generic",
             quality_summary=False, only_prompts=()):
    protocol.require(model_execution_authorized, "Model execution requires explicit authorization")
    dimensions = protocol.dimensions_arg(dimensions)
    if device.startswith("cuda"):
        protocol.require(isinstance(gpu_grant, str) and bool(gpu_grant.strip()),
                         "CUDA evaluation requires the coordinator's metric GPU grant identifier")
    else:
        protocol.require(device == "cpu", "Only cpu or an explicitly granted CUDA device is supported")
    prepared_path = Path(prepared_path).resolve()
    prepared = protocol.read_json(prepared_path)
    protocol.require(prepared.get("schema") == protocol.SCHEMAS["prepared"]
                     and prepared.get("status") == "complete", "Prepared media is incomplete")
    spec, spec_hash = protocol.load_protocol(prepared["protocol_path"], frozen=True)
    protocol.require(prepared["protocol_sha256"] == spec_hash, "Prepared/protocol binding differs")
    protocol.validate_inputs(prepared["input_manifest"], spec, spec_hash)
    protocol.require(protocol.sha256_file(prepared["input_manifest_path"]) == prepared["input_manifest_sha256"],
                     "Original input manifest changed")
    clips = prepared["clips"]
    input_by_id = {r["id"]: r for r in prepared["input_manifest"]["clips"]}
    protocol.require(len(clips) == len(input_by_id)
                     and {r["id"] for r in clips} == set(input_by_id), "Prepared clip coverage differs")
    for clip in clips:
        protocol.require(all(clip.get(k) == v for k, v in input_by_id[clip["id"]].items()),
                         "Prepared input binding changed")
        protocol.require(clip.get("decode_verification", {}).get("exact_rgb_identity") is True,
                         "No exact RGB decoder verification")
    if only_prompts:
        requested = set(only_prompts)
        protocol.require(requested <= {p["id"] for p in spec["prompts"]}, "Unknown selected prompt")
        clips = [r for r in clips if r["prompt_id"] in requested]
    protocol.require(bool(clips), "No selected clips")
    video_paths = {}
    for clip in clips:
        path = (prepared_path.parent / clip["video_path"]).resolve()
        protocol.require(path.is_relative_to(prepared_path.parent), "Prepared video escaped its media root")
        protocol.require(protocol.sha256_file(path) == clip["video_sha256"], "Prepared video hash differs")
        video_paths[clip["id"]] = path
        protocol.frame_policy(clip["frame_count"], clip["fps"], dimensions)
    manifest = protocol.read_json(assets_path)
    records = {d: protocol.dimension_records(spec, clips, d, video_paths, quality_mode) for d in dimensions}
    active = tuple(d for d in dimensions if records[d])
    with official.dependency_paths(manifest.get("dependency_paths", [])):
        checks = official.preflight_assets(manifest, active) if active else {}
    blocked = {d: row["blockers"] for d, row in checks.items() if row["blockers"]}
    protocol.require(not blocked, f"Requested covered dimensions are blocked: {blocked}")
    if "temporal_flickering" in active:
        protocol.require(device.startswith("cuda"), "Official static filtering requires a metric GPU grant")
    verified_assets = official.verify_required_assets(manifest, active)
    output_dir = Path(output_dir).resolve()
    protocol.require(not output_dir.exists(), "Evaluation requires a new output directory; no stale result reuse")
    output_dir.mkdir(parents=True)
    result = {
        "schema": protocol.SCHEMAS["results"], "status": "in_progress",
        "protocol_sha256": spec_hash, "prepared_sha256": protocol.sha256_file(prepared_path),
        "assets_manifest_sha256": protocol.sha256_file(assets_path),
        "official_source": manifest["official_source"], "verified_assets": verified_assets,
        "requested_dimensions": list(dimensions), "quality_mode": quality_mode,
        "scope": "Official per-dimension algorithms on an explicitly partial, matched continuation cohort.",
        "full_official_aggregate": False, "gpu_grant": gpu_grant, "device": device,
        "script_sha256": protocol.sha256_file(__file__),
        "helper_sha256": {
            "protocol": protocol.sha256_file(protocol.__file__),
            "official": protocol.sha256_file(official.__file__),
        },
        "dimensions": {}, "static_filter": None,
        "planned_primary_coverage": protocol.primary_coverage(spec, clips),
    }
    try:
        with official.official_environment(manifest, output_dir / "bound_cache", active), \
                model_settings(device) as execution_device:
            import torch

            if execution_device == "cuda":
                result["gpu_name"] = torch.cuda.get_device_name()
                properties = torch.cuda.get_device_properties(torch.cuda.current_device())
                result["gpu_total_memory_bytes"] = properties.total_memory
                result["gpu_compute_capability"] = [properties.major, properties.minor]
            result["cuda_runtime"] = torch.version.cuda
            result["cudnn_version"] = torch.backends.cudnn.version()
            result["versions"] = official.version_inventory()
            result["numeric_settings"] = {
                "tf32_matmul": False, "tf32_cudnn": False, "cudnn_benchmark": False,
                "autocast": False, "seed": 0, "torch_threads": torch.get_num_threads(),
                "clip_precision": "Official clip.load device-native precision",
                "imaging_quality_preprocessing_mode": "longer",
            }
            if "temporal_flickering" in active:
                paths = [row["video_list"][0] for row in records["temporal_flickering"]]
                with torch.inference_mode():
                    passes = official.static_filter(paths, manifest, execution_device)
                eligible, details = paired_static_eligibility(clips, video_paths, passes)
                result["static_filter"] = {
                    "implementation": "Pinned official static_filter.py::StaticFilter",
                    "passes_by_video": passes, "paired_coverage": details,
                    "policy": "All supplied arms pass, or this model/prompt is excluded for every arm; no resampling.",
                }
                records["temporal_flickering"] = [
                    row for row in records["temporal_flickering"] if row["video_list"][0] in eligible]
            for dimension in dimensions:
                current = records[dimension]
                if not current:
                    result["dimensions"][dimension] = {
                        "status": "uncovered_or_static_excluded", "by_clip": {}, "score": None,
                    }
                    continue
                if dimension == "human_action":
                    current = human_action_aliases(current, output_dir)
                info_path = output_dir / f"{dimension}_full_info.json"
                protocol.write_json(info_path, current)
                if execution_device == "cuda":
                    torch.cuda.synchronize()
                before = time.perf_counter()
                with torch.inference_mode():
                    values = official.compute_dimension(dimension, info_path, manifest, execution_device)
                if execution_device == "cuda":
                    torch.cuda.synchronize()
                elapsed = time.perf_counter() - before
                expected = [r["video_list"][0] for r in current]
                validated = protocol.validate_official_results(dimension, values, expected)
                by_clip = {cid: validated[str(path)] for cid, path in video_paths.items() if str(path) in validated}
                result["dimensions"][dimension] = {
                    "status": "complete_for_listed_inputs", "by_clip": by_clip,
                    "metadata_sha256": protocol.sha256_file(info_path),
                    "elapsed_seconds": elapsed,
                    "upstream_mixed_input_score_not_a_model_comparison": float(values[0]),
                    "frame_policy": protocol.frame_policy(spec["continuation_frames"], spec["fps"], [dimension])[dimension],
                }
                protocol.write_json(output_dir / "result.json", result)
                gc.collect()
                if execution_device == "cuda":
                    torch.cuda.empty_cache()
        result["summary"] = summarize_vbench(
            clips, dimensions, result["dimensions"], quality_summary=quality_summary)
        result["status"] = "complete"
        protocol.write_json(output_dir / "result.json", result)
        return result
    except Exception as exc:
        result.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        protocol.write_json(output_dir / "result.json", result)
        raise


def example_inputs(spec, spec_hash):
    prompt = spec["prompts"][0]["id"]
    model = next(iter(spec["models"]))
    clips = []
    for arm in spec["primary_arms"]:
        clips.append({
            "id": f"{model}-{prompt}-{arm}", "model": model, "prompt_id": prompt,
            "arm": arm, "seed": spec["generation_seed"],
            "fps": spec["fps"], "native_fps": spec["fps"],
            "prefix_frames": 9, "native_frame_start": 9,
            "frame_count": spec["continuation_frames"], "rgb_frame_start": 0,
            "rgb_path": f"REPLACE_WITH_ORIGINAL_{arm}_continuation.npy",
            "rgb_file_sha256": "REPLACE_WITH_FILE_SHA256",
            "generation_binding_sha256": "REPLACE_WITH_MATCHED_NATIVE_GENERATION_BINDING_SHA256",
            "representation": {"kind": "raw"} if arm == "raw" else
                              {"kind": "svd_hevc", **spec["primary_compression"]},
            "representation_evidence_sha256": "REPLACE_WITH_NATIVE_CODEC_REPORT_SHA256",
            "paired_raw_id": f"{model}-{prompt}-raw",
        })
    return {
        "schema": protocol.SCHEMAS["inputs"], "protocol_sha256": spec_hash,
        "note": "Illustration only. Expand to actual cases/models/arms, and use each model's actual common-prefix length.",
        "clips": clips,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dependency-path", action="append", default=[],
                        help="Explicit isolated dependency overlay; never modifies installed packages")
    sub = parser.add_subparsers(dest="command", required=True)
    candidate = sub.add_parser("candidate", help="Create a proposal; never freezes the shared protocol")
    candidate.add_argument("--full-info", type=Path, required=True)
    candidate.add_argument("--prior-manifest", type=Path, action="append", default=[])
    candidate.add_argument("--selection-seed", type=int, default=20260918)
    candidate.add_argument("--strata", nargs=4, choices=protocol.DIMENSIONS)
    candidate.add_argument("--output-dir", type=Path, required=True)
    inventory = sub.add_parser("inventory", help="CPU-only source/dependency/asset inventory")
    inventory.add_argument("--vbench-root", type=Path, required=True)
    inventory.add_argument("--weight-root", type=Path, required=True)
    inventory.add_argument("--hash-weights", action="store_true")
    inventory.add_argument("--output", type=Path, required=True)
    example = sub.add_parser("example-inputs")
    example.add_argument("--protocol", type=Path, required=True)
    example.add_argument("--output", type=Path, required=True)
    prep = sub.add_parser("prepare", help="CPU-only raw SSIM plus verified lossless VBench media")
    prep.add_argument("--protocol", type=Path, required=True)
    prep.add_argument("--inputs", type=Path, required=True)
    prep.add_argument("--output-dir", type=Path, required=True)
    prep.add_argument("--ffmpeg")
    prep.add_argument("--ssim-workers", type=int, default=1)
    scoring = sub.add_parser("evaluate", help="Official models; waits for explicit execution permission")
    scoring.add_argument("--prepared", type=Path, required=True)
    scoring.add_argument("--assets", type=Path, required=True)
    scoring.add_argument("--output-dir", type=Path, required=True)
    scoring.add_argument("--dimensions", nargs="+", choices=protocol.DIMENSIONS,
                         default=list(protocol.GENERIC_QUALITY))
    scoring.add_argument("--device", default="cpu")
    scoring.add_argument("--model-execution-authorized", action="store_true")
    scoring.add_argument("--gpu-grant")
    scoring.add_argument("--quality-mode", choices=("generic", "annotated"), default="generic")
    scoring.add_argument("--quality-summary", action="store_true")
    scoring.add_argument("--only-prompt", action="append", default=[])
    args = parser.parse_args(argv)
    with official.dependency_paths(args.dependency_path):
        if args.command == "candidate":
            protocol.require(not args.output_dir.exists(), "Candidate output directory already exists")
            spec = protocol.make_candidate(args.full_info, args.prior_manifest, args.selection_seed, args.strata)
            protocol.validate_protocol(spec, protocol.source_rows(args.full_info))
            source = args.output_dir / "sources/VBench_full_info.json"
            source.parent.mkdir(parents=True)
            shutil.copyfile(args.full_info, source)
            for license_path in (args.full_info.parent / "VBench_LICENSE",
                                 args.full_info.parent.parent / "LICENSE"):
                if license_path.is_file():
                    shutil.copyfile(license_path, source.parent / "VBench_LICENSE")
                    break
            protocol.write_json(args.output_dir / "candidate_protocol.json", spec)
            print(json.dumps({"status": "candidate_only", "prompts": [
                {"id": p["id"], "text": p["text"], "dimensions": p["official_annotation"]["dimension"]}
                for p in spec["prompts"]]}, indent=2))
        elif args.command == "inventory":
            manifest = official.inventory_assets(args.vbench_root, args.weight_root,
                                                  hash_weights=args.hash_weights,
                                                  dependency_paths=args.dependency_path)
            manifest["dimension_readiness"] = official.preflight_assets(manifest, protocol.DIMENSIONS)
            protocol.write_json(args.output, manifest)
            print(json.dumps({"models_executed": False, "dimension_readiness": manifest["dimension_readiness"]}, indent=2))
        elif args.command == "example-inputs":
            spec, spec_hash = protocol.load_protocol(args.protocol)
            protocol.write_json(args.output, example_inputs(spec, spec_hash))
        elif args.command == "prepare":
            result = prepare(args.protocol, args.inputs, args.output_dir,
                             ffmpeg=args.ffmpeg, ssim_workers=args.ssim_workers)
            print(json.dumps({"status": result["status"], "clips": len(result["clips"]),
                              "models_executed": False}))
        else:
            result = evaluate(
                args.prepared, args.assets, args.output_dir, args.dimensions,
                device=args.device, model_execution_authorized=args.model_execution_authorized,
                gpu_grant=args.gpu_grant, quality_mode=args.quality_mode,
                quality_summary=args.quality_summary, only_prompts=args.only_prompt)
            print(json.dumps({"status": result["status"], "full_official_aggregate": False}))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError, ImportError, RuntimeError) as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        raise SystemExit(2)
