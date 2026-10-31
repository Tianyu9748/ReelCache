#!/usr/bin/env python3
"""Strict paired SSIM and six official VBench dimensions on lossless RGB.

This module imports no array, native-model or metric-model package on import.
The standalone worker accepts an explicit request, output and workspace.  A
request with ``vbench: true`` additionally requires ``--allow-model-load`` and
the parent's independently approved metric resource grant; this flag is not a
resource grant.  Full official 16-dimension benchmarking is not implemented.

``metric_config`` binds vbench_source_root, vbench_commit, source_sha256
(relative Python paths plus pretrained/amt_model/AMT-S.yaml), dino_source_root,
dino_source_sha256, weights (each path/sha256), and environment.  Environment
binds python_executable, prefix, python_version, exact distribution versions,
package_roots and explicit extra_sys_path.  No downloads or implicit paths.
``study_admission`` supplies the parent protocol, launch receipt and phase;
the shared study verifier is called before external source/weight verification.
The parent remains responsible for the separate live resource grant.

Each raw/candidate descriptor contains path, sha256 and metadata.  Metadata
binds the same run/case/group/seed/cell and generation/decode identities, native
FPS, working geometry, crop, genuine contiguous continuation frame window,
and RAW reference clip.  Arrays are uint8 frames[T,H,W,3]; there is no padding,
frame duplication, video/JPEG preview decoding, or frame-count-as-FPS fallback.
Native metadata additionally binds ``chunks`` (2 or 3) to the admitted request.
``source_start_frame`` is the first NPZ frame's absolute decoded-run coordinate;
``window_start`` is NPZ-local; ``prefix_frames`` is the run's unchanged prefix.
Main continuation is CausVid [9,21/33) at 16 FPS or MAGI [24,48/72) at 24 FPS.
Intentional per-window statistics require clip_kind="diagnostic_window" and a
nonempty window_label; they cannot be exported as main continuation outcomes.
Pairing binds common generation inputs and the exact RAW reference/window,
never candidate pixels or a combined RAW/candidate artifact digest.
``sampling_sha256`` identifies the frozen sampler sources AND runtime config
(MAGI study: serial clean history, chunk window 1, 16 steps for every chunk);
``decode_manifest_sha256`` identifies the common decoder source/config manifest,
not a per-candidate output manifest. Per-clip files and paired_metrics_sha256
remain distinct artifact/comparison identities.
Worker scratch must be a private directory below the executing PROJECT_ROOT.
This follows a verified relocated source tree; it does not depend on a mount
name or permit arbitrary directories outside that tree.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import socket
import stat
import subprocess
import sys
import threading
import time


VBENCH_COMMIT = "fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DIMENSIONS = (
    "subject_consistency", "background_consistency", "motion_smoothness",
    "dynamic_degree", "aesthetic_quality", "imaging_quality",
)
REQUIRED_PACKAGES = (
    "numpy", "torch", "torchvision", "pyiqa", "decord", "easydict",
    "openai-clip", "Pillow", "scikit-image", "omegaconf",
)
PAIR_FIELDS = (
    "run_id", "case_id", "prompt_group", "split", "seed", "cell",
    "prompt_sha256", "noise_sha256", "conditioning_sha256", "sampling_sha256",
    "model_source_sha256", "vae_source_sha256", "attention_sha256",
    "decode_manifest_sha256", "decode_mode", "source_fps", "fps_authority",
    "working_hw", "crop", "crop_applied", "source_start_frame",
    "window_start", "window_count", "prefix_frames", "clip_kind", "evidence_kind",
)
_LOADER_LOCK = threading.RLock()


class MetricInputError(ValueError):
    """A missing dependency, identity mismatch or invalid result is a failure."""


def _int(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise MetricInputError(f"{name} must be an integer >= {minimum}")
    return value


def _number(value, name, minimum=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise MetricInputError(f"{name} must be finite")
    if minimum is not None and value < minimum:
        raise MetricInputError(f"{name} must be >= {minimum}")
    return float(value)


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise MetricInputError(f"{name} must be a nonempty string")
    return value


def _hash(value, name):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise MetricInputError(f"{name} must be lowercase SHA256")
    return value


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_bytes(value):
    try:
        return json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":")).encode()
    except (ValueError, TypeError) as error:
        raise MetricInputError("metadata must be finite JSON") from error


def read_json(path):
    def constant(value):
        raise MetricInputError(f"nonfinite JSON constant: {value}")

    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise MetricInputError(f"duplicate JSON field: {key}")
            result[key] = value
        return result

    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream, parse_constant=constant, object_pairs_hook=pairs)


def write_json(path, value):
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(payload)


def frame_policy(frame_count, source_fps):
    """The pinned video's genuine-FPS DD sampling and unchanged AMT partition."""
    _int(frame_count, "frame_count", 3)
    fps = _number(source_fps, "source_fps", 0)
    if fps <= 0:
        raise MetricInputError("source_fps must be the positive native playback rate")
    full = list(range(frame_count))
    interval = max(1, round(fps / 8))
    sampled = full[::interval]
    threshold = round(4 * (len(sampled) / 16))
    if len(sampled) < 2 or threshold < 1 or threshold > len(sampled) - 1:
        raise MetricInputError("insufficient genuine frames for official dynamic-degree sampling")
    policy = {name: {"sampled_indices": full.copy()} for name in DIMENSIONS}
    policy["dynamic_degree"].update({
        "sampled_indices": sampled, "interval": interval, "effective_fps": fps / interval,
        "motion_pair_count_threshold": threshold,
        "policy": "official range(0,T,max(1,round(native_fps/8)))",
    })
    policy["motion_smoothness"].update({
        "anchor_indices": full[::2],
        "target_indices": full[1:2 * (len(full[::2]) - 1):2],
        "unused_terminal_indices": [frame_count - 1] if frame_count % 2 == 0 else [],
        "policy": "official AMT even anchors; odd targets strictly between anchors",
    })
    return policy


def _parent_study():
    root = PROJECT_ROOT
    original = list(sys.path)
    try:
        sys.path.insert(0, str(root))
        study = importlib.import_module("reelcache.study")
        if Path(study.__file__).resolve() != root / "reelcache/study.py":
            raise MetricInputError("another study implementation shadows this metric worker")
        return study
    finally:
        sys.path[:] = original


def _native_frame_limits(metadata, cell):
    chunks = _int(metadata.get("chunks"), "native chunks", 2)
    if chunks not in (2, 3):
        raise MetricInputError("native chunks must match the admitted two- or three-chunk run")
    leading_trim = 3 if cell.cell_id == "causvid480" else 0
    return 4 * cell.frames - leading_trim, 4 * cell.frames * chunks - leading_trim


def _paired_metadata(metadata):
    fields = list(PAIR_FIELDS)
    if metadata["evidence_kind"] == "native":
        fields.append("chunks")
    if metadata["clip_kind"] == "diagnostic_window":
        fields.append("window_label")
    return {name: metadata[name] for name in fields}


def validate_clip_metadata(metadata):
    try:
        for name in PAIR_FIELDS:
            if name not in metadata:
                raise MetricInputError(f"missing paired metadata: {name}")
        for name in ("run_id", "case_id", "prompt_group", "cell", "clip_id",
                     "reference_clip_id", "arm_id", "trajectory_id", "fps_authority"):
            _text(metadata[name], name)
        if metadata["split"] not in ("development", "calibration", "heldout"):
            raise MetricInputError("unknown prompt-group split")
        for name in PAIR_FIELDS:
            if name.endswith("_sha256"):
                _hash(metadata[name], name)
        for name in ("seed", "source_start_frame", "window_start", "prefix_frames"):
            _int(metadata[name], name)
        _int(metadata["window_count"], "window_count", 3)
        fps = _number(metadata["source_fps"], "source_fps", 0)
        if fps == 0:
            raise MetricInputError("source_fps must be positive")
        start = metadata["source_start_frame"] + metadata["window_start"]
        end = start + metadata["window_count"]
        if metadata["clip_kind"] not in ("continuation", "diagnostic_window"):
            raise MetricInputError("declare main continuation or an explicitly labelled diagnostic_window")
        if metadata["clip_kind"] == "diagnostic_window":
            _text(metadata.get("window_label"), "diagnostic window_label")
        elif start < metadata["prefix_frames"]:
            raise MetricInputError("main metrics must exclude the unchanged common prefix")
        if metadata["decode_mode"] not in ("full_prefix", "stateless_chunk", "verified_streaming"):
            raise MetricInputError("declare the actual VAE decoding mode")
        if metadata["decode_mode"] == "verified_streaming":
            _hash(metadata["streaming_verification_sha256"], "streaming_verification_sha256")
        if metadata["evidence_kind"] not in ("native", "synthetic_fixture"):
            raise MetricInputError("evidence_kind must distinguish native output from fixtures")
        if metadata.get("genuine_frames") is not True or metadata.get("padding_frames") != 0 or (
            type(metadata.get("padding_frames")) is not int
        ) or metadata.get("duplicated_frames") != 0 or type(metadata.get("duplicated_frames")) is not int:
            raise MetricInputError("declare genuine frames with no fabricated padding or duplication")
        if type(metadata["crop_applied"]) is not bool:
            raise MetricInputError("crop_applied must be explicit")
        if not isinstance(metadata["working_hw"], (list, tuple)) or len(metadata["working_hw"]) != 2:
            raise MetricInputError("working_hw must contain height,width")
        height, width = (_int(value, "working_hw", 1) for value in metadata["working_hw"])
        if not isinstance(metadata["crop"], (list, tuple)) or len(metadata["crop"]) != 4:
            raise MetricInputError("crop must contain top,bottom,left,right exclusive bounds")
        top, bottom, left, right = (_int(value, "crop") for value in metadata["crop"])
        if not 0 <= top < bottom <= height or not 0 <= left < right <= width:
            raise MetricInputError("crop is outside the actual working canvas")
        if metadata["evidence_kind"] == "native":
            study = _parent_study()
            cell = study.study_cell(metadata["cell"])
            if (height, width) != (cell.working_height, cell.width) or tuple(metadata["crop"]) != (
                cell.crop_top, cell.crop_top + cell.height, 0, cell.width
            ):
                raise MetricInputError("native RGB geometry/crop differs from the bound study cell")
            if fps != study.NATIVE_SOURCE_FPS[cell.cell_id]:
                raise MetricInputError("native source_fps differs from the approved study cell FPS")
            prefix, total = _native_frame_limits(metadata, cell)
            if metadata["prefix_frames"] != prefix:
                raise MetricInputError("native prefix_frames differs from the unchanged first decoded chunk")
            if cell.cell_id == "causvid480" and metadata["decode_mode"] == "stateless_chunk":
                raise MetricInputError("native CausVid requires full-prefix or independently verified streaming decode")
            if cell.model == "magi_distill" and metadata["decode_mode"] != "stateless_chunk":
                raise MetricInputError("approved native MAGI requires stateless_chunk decoding")
            if end > total:
                raise MetricInputError("native window exceeds the admitted decoded run")
            if metadata["clip_kind"] == "continuation" and (start, end) != (prefix, total):
                raise MetricInputError("main continuation must span the entire admitted run after its unchanged prefix")
    except (KeyError, TypeError) as error:
        raise MetricInputError(f"malformed clip metadata: {error}") from error
    _json_bytes(metadata)
    return dict(metadata)


def load_rgb_npz(descriptor):
    """Load the exact hash-bound NPZ, apply at most one declared crop/window."""
    import numpy as np

    metadata = validate_clip_metadata(descriptor["metadata"])
    path = Path(descriptor["path"])
    expected_hash = _hash(descriptor["sha256"], "RGB file sha256")
    if path.suffix.lower() != ".npz" or not path.is_file():
        raise MetricInputError("only an existing lossless RGB NPZ is accepted")
    if sha256(path) != expected_hash:
        raise MetricInputError("RGB NPZ checksum mismatch")
    with np.load(path, allow_pickle=False) as data:
        if set(data.files) != {"frames"}:
            raise MetricInputError("RGB NPZ must contain exactly the frames array")
        frames = data["frames"]
    if frames.dtype != np.uint8 or frames.ndim != 4 or frames.shape[-1] != 3:
        raise MetricInputError(f"expected uint8 [T,H,W,3] RGB, got {frames.dtype}/{frames.shape}")
    top, bottom, left, right = metadata["crop"]
    expected_hw = (bottom - top, right - left) if metadata["crop_applied"] else tuple(metadata["working_hw"])
    if tuple(frames.shape[1:3]) != expected_hw:
        raise MetricInputError("array geometry differs from declared working/cropped geometry")
    total = len(frames)
    start, count = metadata["window_start"], metadata["window_count"]
    if metadata["evidence_kind"] == "native":
        _, run_frames = _native_frame_limits(metadata, _parent_study().study_cell(metadata["cell"]))
        if metadata["source_start_frame"] + total > run_frames:
            raise MetricInputError("RGB source coordinates exceed the admitted decoded run")
    if start + count > total:
        raise MetricInputError("continuation window exceeds genuine source frames")
    frames = frames[start:start + count]
    if not metadata["crop_applied"]:
        frames = frames[:, top:bottom, left:right]
    frames = np.ascontiguousarray(frames)
    indices = list(range(metadata["source_start_frame"] + start,
                         metadata["source_start_frame"] + start + count))
    return frames, {
        "path": str(path.resolve()), "file_sha256": expected_hash, "metadata": metadata,
        "source_total_frames": total, "source_frame_indices": indices,
        "shape": list(frames.shape), "dtype": "uint8", "color_order": "RGB",
        "source_fps": metadata["source_fps"], "duration_seconds": count / metadata["source_fps"],
        "rgb_sha256": hashlib.sha256(frames.tobytes()).hexdigest(),
        "frame_duplication": False, "temporal_resampling": False,
        "origin_verification": "caller-supplied source-bound decode metadata; pixel equality cannot prove native origin",
    }


def validate_pair(raw_info, candidate_info):
    raw = validate_clip_metadata(raw_info["metadata"])
    candidate = validate_clip_metadata(candidate_info["metadata"])
    for info, metadata in ((raw_info, raw), (candidate_info, candidate)):
        start = metadata["source_start_frame"] + metadata["window_start"]
        count = metadata["window_count"]
        top, bottom, left, right = metadata["crop"]
        indices = info["source_frame_indices"]
        if not isinstance(indices, (list, tuple)) or len(indices) != count or any(
            type(value) is not int or value != start + offset for offset, value in enumerate(indices)
        ) or (
            tuple(info["shape"]) != (count, bottom - top, right - left, 3)
        ):
            raise MetricInputError("paired RGB coordinates/shape differ from the declared frame window")
        if info["source_fps"] != metadata["source_fps"] or (
            info["dtype"] != "uint8" or info["color_order"] != "RGB"
        ):
            raise MetricInputError("paired RGB FPS/type differs from its declared native transport")
        total = _int(info["source_total_frames"], "source_total_frames", 3)
        if metadata["window_start"] + count > total:
            raise MetricInputError("paired window exceeds its declared genuine source frames")
        if metadata["evidence_kind"] == "native":
            _, run_frames = _native_frame_limits(metadata, _parent_study().study_cell(metadata["cell"]))
            if metadata["source_start_frame"] + total > run_frames:
                raise MetricInputError("paired RGB source coordinates exceed the admitted decoded run")
    if raw["arm_id"] != "RAW" or raw["reference_clip_id"] != raw["clip_id"]:
        raise MetricInputError("reference must be same-run RAW, not a reconstructed or historical prefix")
    if candidate["reference_clip_id"] != raw["clip_id"] or candidate["clip_id"] == raw["clip_id"]:
        raise MetricInputError("candidate does not identify this distinct RAW reference")
    for name, value in _paired_metadata(raw).items():
        if value != candidate.get(name):
            raise MetricInputError(f"paired RAW/candidate mismatch: {name}")
    if raw.get("streaming_verification_sha256") != candidate.get("streaming_verification_sha256"):
        raise MetricInputError("paired streaming-VAE evidence differs")
    for name in ("source_total_frames", "source_frame_indices", "shape", "dtype", "color_order"):
        if raw_info[name] != candidate_info[name]:
            raise MetricInputError(f"paired window/geometry mismatch: {name}")


def ssim_frames(reference, candidate, *, ssim_function=None):
    import numpy as np

    if reference.dtype != np.uint8 or candidate.dtype != np.uint8 or reference.shape != candidate.shape:
        raise MetricInputError("SSIM requires matching uint8 RAW/candidate arrays")
    if reference.ndim != 4 or reference.shape[-1] != 3 or reference.shape[0] == 0 or min(reference.shape[1:3]) < 7:
        raise MetricInputError("SSIM requires nonempty [T,H,W,3] and H,W >= 7")
    if ssim_function is None:
        from skimage.metrics import structural_similarity
        ssim_function = structural_similarity
    parameters = {
        "data_range": 255, "channel_axis": -1, "win_size": 7,
        "gaussian_weights": False, "use_sample_covariance": True, "K1": .01, "K2": .03,
    }
    values = []
    for raw, compressed in zip(reference, candidate):
        score = _number(float(ssim_function(raw, compressed, **parameters)), "per-frame SSIM")
        if not -1 <= score <= 1:
            raise MetricInputError("SSIM lies outside [-1,1]")
        values.append(score)
    return {
        "per_frame": values, "mean": math.fsum(values) / len(values),
        "minimum": min(values), "frame_count": len(values), "parameters": parameters,
        "implementation": "skimage.metrics.structural_similarity",
        "implementation_version": importlib.metadata.version("scikit-image"),
        "comparison": "paired uint8 RGB frames",
    }


def verify_environment(environment):
    """Fail before model imports if generation and metric environments mix."""
    if str(Path(sys.executable).absolute()) != str(Path(environment["python_executable"]).absolute()):
        raise MetricInputError("metric Python executable differs from its frozen environment")
    if Path(sys.prefix).resolve() != Path(environment["prefix"]).resolve():
        raise MetricInputError("metric Python prefix mismatch")
    if ".".join(map(str, sys.version_info[:3])) != environment["python_version"]:
        raise MetricInputError("metric Python version mismatch")
    versions, roots = environment["versions"], environment["package_roots"]
    if not set(REQUIRED_PACKAGES) <= set(versions) or not set(REQUIRED_PACKAGES) <= set(roots):
        raise MetricInputError("environment manifest must bind every required metric package/version/root")
    observed = {}
    for name in versions:
        distribution = importlib.metadata.distribution(name)
        if distribution.version != versions[name]:
            raise MetricInputError(f"{name} version mismatch: {distribution.version} != {versions[name]}")
        if name not in roots or Path(distribution.locate_file("")).resolve() != Path(roots[name]).resolve():
            raise MetricInputError(f"{name} distribution is shadowed by another environment")
        observed[name] = distribution.version
    for module_name, distribution_name in (("numpy", "numpy"), ("torch", "torch"),
                                            ("torchvision", "torchvision"), ("clip", "openai-clip"),
                                            ("PIL", "Pillow"), ("skimage", "scikit-image")):
        module = sys.modules.get(module_name)
        if module is not None:
            location = getattr(module, "__file__", None)
            if not location or not Path(location).resolve().is_relative_to(Path(roots[distribution_name]).resolve()):
                raise MetricInputError(f"already imported {module_name} contaminates the metric environment")
            if module_name == "numpy" and getattr(module, "__version__", None) != versions["numpy"]:
                raise MetricInputError("already imported NumPy version differs from metric NumPy")
    return {"python": sys.version, "python_executable": sys.executable, "prefix": sys.prefix,
            "versions": observed, "package_roots": roots}


def _verify_source_tree(root, expected, *, extra_files=()):
    root = Path(root).resolve()
    if not root.is_dir() or not isinstance(expected, dict) or not expected:
        raise MetricInputError("explicit nonempty source tree hashes are required")
    inventory = {str(path.relative_to(root)) for path in root.rglob("*.py")} | set(extra_files)
    if set(expected) != inventory:
        raise MetricInputError("source hash inventory is incomplete or contains unexpected source files")
    for name, digest in expected.items():
        _hash(digest, name)
        path = (root / name).resolve()
        if not path.is_relative_to(root) or not path.is_file() or sha256(path) != digest:
            raise MetricInputError(f"source identity mismatch: {name}")
    return str(root)


def verify_metric_sources(config):
    if config["vbench_commit"] != VBENCH_COMMIT:
        raise MetricInputError("only the approved pinned official VBench commit is supported")
    source = Path(config["vbench_source_root"]).resolve()
    actual = subprocess.check_output(["git", "--no-pager", "-C", str(source), "rev-parse", "HEAD"],
                                     text=True).strip()
    if actual != config["vbench_commit"]:
        raise MetricInputError("official VBench checkout commit differs from the request")
    if subprocess.check_output(["git", "--no-pager", "-C", str(source), "diff", "--name-only", "HEAD"],
                               text=True).strip():
        raise MetricInputError("official VBench tracked source has been modified")
    _verify_source_tree(source, config["source_sha256"],
                        extra_files=("pretrained/amt_model/AMT-S.yaml",))
    dino = _verify_source_tree(config["dino_source_root"], config["dino_source_sha256"])
    if not {"hubconf.py", "vision_transformer.py"} <= set(config["dino_source_sha256"]):
        raise MetricInputError("DINO local architecture source is incomplete")
    if set(config["weights"]) != set(DIMENSIONS) | {"aesthetic_head"}:
        raise MetricInputError("all six dimensions and the aesthetic head need explicit local weights")
    for name, entry in config["weights"].items():
        _hash(entry["sha256"], name)
        path = Path(entry["path"])
        if not path.is_file() or sha256(path) != entry["sha256"]:
            raise MetricInputError(f"missing or mismatched metric weight: {name}")
    head = Path(config["weights"]["aesthetic_head"]["path"])
    if head.name != "sa_0_4_vit_l_14_linear.pth":
        raise MetricInputError("official aesthetic loader requires its exact local head filename")
    return {"vbench_source_root": str(source), "vbench_commit": actual,
            "source_sha256": config["source_sha256"], "dino_source_root": dino,
            "dino_source_sha256": config["dino_source_sha256"], "weights": config["weights"]}


def verify_study_admission(admission):
    """Use the parent's gate, not a second receipt or phase-transition implementation."""
    if not isinstance(admission, dict) or set(admission) != {"protocol", "receipt", "phase"}:
        raise MetricInputError("metric models require the parent protocol/receipt/phase admission")
    _text(admission["phase"], "admitted study phase")
    study = _parent_study()
    study.verify_launch_receipt(admission["receipt"], admission["protocol"], phase=admission["phase"])
    return {
        "protocol_id": admission["protocol"]["protocol_id"],
        "receipt_id": admission["receipt"]["receipt_id"], "phase": admission["phase"],
    }


@contextlib.contextmanager
def offline_network():
    """Metric workers must fail rather than fetch missing models or telemetry."""
    def denied(*args, **kwargs):
        raise MetricInputError("network access is disabled in the source-bound metric worker")

    with _LOADER_LOCK:
        original = [(socket, "create_connection", socket.create_connection),
                    (socket, "getaddrinfo", socket.getaddrinfo),
                    (socket.socket, "connect", socket.socket.connect),
                    (socket.socket, "connect_ex", socket.socket.connect_ex)]
        try:
            for owner, name, _ in original:
                setattr(owner, name, denied)
            yield
        finally:
            for owner, name, value in reversed(original):
                setattr(owner, name, value)


@contextlib.contextmanager
def array_loader(module, frames, token):
    """Serialized scoped transport change; numerical VBench functions stay intact."""
    import torch

    def load_video(path, *args, **kwargs):
        if path != token or args or kwargs:
            raise MetricInputError("unexpected official loader call; reverify the pinned source")
        return torch.from_numpy(frames).permute(0, 3, 1, 2).float().contiguous()

    with _LOADER_LOCK:
        original = module.load_video
        module.load_video = load_video
        try:
            yield
        finally:
            module.load_video = original


def read_png_frames(directory):
    import numpy as np
    from PIL import Image

    paths = sorted(Path(directory).iterdir())
    if not paths or any(not path.is_file() or path.suffix != ".png" for path in paths):
        raise MetricInputError("official input directory must contain only the verified PNG sequence")
    frames = []
    for path in paths:
        with Image.open(path) as image:
            if image.mode != "RGB":
                raise MetricInputError("lossless transport is not RGB")
            frames.append(np.array(image, dtype=np.uint8))
    return frames


def verified_png_inputs(frames, policy, directory):
    import numpy as np
    from PIL import Image

    directory = Path(directory)
    full, dynamic = directory / "all_frames", directory / "dynamic_degree_8fps"
    full.mkdir(parents=True, exist_ok=True)
    dynamic.mkdir(parents=True, exist_ok=True)
    selected = set(policy["dynamic_degree"]["sampled_indices"])
    names, dynamic_names, records = [], [], []
    for index, frame in enumerate(frames):
        name = f"{index:06d}.png"
        path = full / name
        if not path.exists():
            with path.open("xb") as stream:
                Image.fromarray(frame).save(stream, format="PNG", compress_level=1)
        with Image.open(path) as image:
            if image.mode != "RGB" or not np.array_equal(np.asarray(image), frame):
                raise MetricInputError("lossless PNG transport changed RGB samples")
        names.append(name)
        if index in selected:
            target = dynamic / name
            if not target.exists():
                os.link(path, target)
            with Image.open(target) as image:
                if image.mode != "RGB" or not np.array_equal(np.asarray(image), frame):
                    raise MetricInputError("dynamic-degree transport changed sampled RGB")
            dynamic_names.append(name)
        records.append({"index": index, "sha256": sha256(path)})
    if sorted(path.name for path in full.iterdir()) != names or (
        sorted(path.name for path in dynamic.iterdir()) != dynamic_names
    ):
        raise MetricInputError("unexpected files in metric transport directories")
    return full, dynamic, {
        "codec": "lossless PNG RGB", "decoder": "Pillow",
        "decode_identity": True, "frame_count": len(frames), "frames": records,
        "adaptation": "input transport only; official numerical routines unchanged",
        "lossy_encoding_performed": False,
    }


def validate_dimension_scores(scores):
    if not isinstance(scores, dict) or set(scores) != set(DIMENSIONS):
        raise MetricInputError("a successful result requires all six named official dimensions")
    for dimension, record in scores.items():
        for name in ("raw_score", "normalized_score", "official_dimension_score",
                     "official_per_video_result", "normalization_divisor"):
            _number(record[name], f"{dimension}.{name}")
        divisor = record["normalization_divisor"]
        wanted_divisor = 100.0 if dimension == "imaging_quality" else (
            10.0 if dimension == "aesthetic_quality" else 1.0
        )
        if divisor != wanted_divisor or not math.isclose(
            record["raw_score"] / divisor, record["normalized_score"], rel_tol=1e-7, abs_tol=1e-8
        ) or not math.isclose(record["normalized_score"], record["official_dimension_score"],
                             rel_tol=1e-7, abs_tol=1e-8):
            raise MetricInputError(f"{dimension}: official raw/normalized units mismatch")
        per_video = record["official_per_video_result"]
        expected_video = record["raw_score"] if dimension == "imaging_quality" else record["normalized_score"]
        if not math.isclose(per_video, expected_video, rel_tol=1e-7, abs_tol=1e-8):
            raise MetricInputError(f"{dimension}: per-video official units mismatch")
        if dimension == "dynamic_degree" and record["normalized_score"] not in (0.0, 1.0):
            raise MetricInputError("official per-clip dynamic_degree must be binary")
    _json_bytes(scores)
    return scores


class StrictVBenchScorer:
    """One independently launched process per scorer worker; never a model-worker plugin."""

    def __init__(self, config, *, device="cpu", allow_model_load=False):
        if allow_model_load is not True or os.environ.get("REELCACHE_RANK_METRIC_WORKER") != "1" or not sys.flags.isolated:
            raise MetricInputError("official models require an explicitly admitted isolated metric worker")
        self.pid = os.getpid()
        self.config = config
        self.models, self.modules, self.load_latency = {}, {}, {}
        self.torch = None
        self.original_sys_path = list(sys.path)
        self.original_torch_flags = None
        try:
            with _LOADER_LOCK, offline_network():
                admission_info = verify_study_admission(config["study_admission"])
                environment = config["environment"]
                for path in reversed(environment["extra_sys_path"]):
                    if not Path(path).is_dir():
                        raise MetricInputError(f"missing explicit dependency path: {path}")
                    sys.path.insert(0, str(Path(path).resolve()))
                env_info = verify_environment(environment)
                source_info = verify_metric_sources(config)
                self.upstream = Path(source_info["vbench_source_root"])
                sys.path.insert(0, str(self.upstream))
                for name, module in list(sys.modules.items()):
                    if name == "vbench" or name.startswith("vbench."):
                        raise MetricInputError("VBench was imported before isolated source verification")
                import torch
                self.torch = torch
                verify_environment(environment)
                self.device = torch.device(device)
                if self.device.type not in ("cpu", "cuda"):
                    raise MetricInputError("metric device must be explicit CPU or CUDA")
                if self.device.type == "cuda":
                    if not torch.cuda.is_available():
                        raise MetricInputError("requested metric CUDA device is unavailable")
                    torch.cuda.set_device(self.device)
                self.original_torch_flags = (
                    torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
                    torch.backends.cudnn.benchmark,
                )
                torch.backends.cuda.matmul.allow_tf32 = False
                torch.backends.cudnn.allow_tf32 = False
                torch.backends.cudnn.benchmark = False
                start = time.perf_counter()
                for dimension in DIMENSIONS:
                    module = importlib.import_module(f"vbench.{dimension}")
                    if not Path(module.__file__).resolve().is_relative_to(self.upstream):
                        raise MetricInputError("metric module escaped pinned VBench source")
                    self.modules[dimension] = module
                    before = time.perf_counter()
                    self.models[dimension] = self._build(dimension, module)
                    self._synchronize()
                    self.load_latency[dimension] = time.perf_counter() - before
                self.provenance = {
                    **source_info, "environment": env_info, "adapter_sha256": sha256(__file__),
                    "study_admission": admission_info,
                    "device": str(self.device), "cuda": torch.version.cuda,
                    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                    "tf32_matmul": False, "tf32_cudnn": False, "cudnn_benchmark": False,
                    "load_latency_seconds": self.load_latency, "load_seconds": time.perf_counter() - start,
                    "official_dimensions_unchanged": True, "network_access": "blocked",
                }
        except BaseException:
            self.close()
            raise

    def _synchronize(self):
        if self.device.type == "cuda":
            self.torch.cuda.synchronize(self.device)

    def _build(self, dimension, module):
        torch = self.torch
        path = str(Path(self.config["weights"][dimension]["path"]).resolve())
        if dimension == "subject_consistency":
            model = torch.hub.load(str(Path(self.config["dino_source_root"]).resolve()),
                                   "dino_vitb16", source="local", pretrained=False)
            model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
            return model.eval().to(self.device)
        if dimension == "background_consistency":
            import clip
            return clip.load(path, device=self.device)
        if dimension == "aesthetic_quality":
            import clip
            head = Path(self.config["weights"]["aesthetic_head"]["path"]).resolve()
            return module.get_aesthetic_model(str(head.parent)).to(self.device), clip.load(path, device=self.device)[0]
        if dimension == "imaging_quality":
            return module.MUSIQ(pretrained_model_path=path).eval().to(self.device)
        if dimension == "motion_smoothness":
            model = module.MotionSmoothness(str(self.upstream / "pretrained/amt_model/AMT-S.yaml"),
                                            path, "cuda" if self.device.type == "cuda" else "cpu")
            model.fp.get_frames_from_img_folder = read_png_frames
            return model
        if dimension == "dynamic_degree":
            model = module.DynamicDegree(module.edict(model=path, small=False, mixed_precision=False,
                                                      alternate_corr=False), self.device)
            model.get_frames_from_img_folder = lambda directory: [
                torch.from_numpy(frame).permute(2, 0, 1).float()[None].to(self.device)
                for frame in read_png_frames(directory)
            ]
            return model
        raise MetricInputError(dimension)

    def score_frames(self, frames, input_info, output_dir):
        import numpy as np

        if os.getpid() != self.pid:
            raise MetricInputError("metric models may not be inherited into another worker process")
        if frames.dtype != np.uint8 or frames.ndim != 4 or frames.shape[-1] != 3 or (
            list(frames.shape) != input_info["shape"]
        ) or hashlib.sha256(frames.tobytes()).hexdigest() != input_info["rgb_sha256"]:
            raise MetricInputError("metric tensor differs from the verified lossless RGB window")
        metadata = validate_clip_metadata(input_info["metadata"])
        start = metadata["source_start_frame"] + metadata["window_start"]
        if input_info["source_frame_indices"] != list(range(start, start + len(frames))) or (
            len(frames) != metadata["window_count"] or input_info["source_fps"] != metadata["source_fps"]
        ):
            raise MetricInputError("metric frames/FPS differ from the bound continuation window")
        if min(frames.shape[1:3]) < 32:
            raise MetricInputError("official visual networks require genuine frames at least 32x32")
        policy = frame_policy(len(frames), input_info["source_fps"])
        for record in policy.values():
            record["sampled_source_indices"] = [input_info["source_frame_indices"][index]
                                                 for index in record["sampled_indices"]]
        with _LOADER_LOCK, offline_network(), self.torch.no_grad():
            if self.torch.is_autocast_enabled():
                raise MetricInputError("metric workers must not inherit generation autocast")
            start = time.perf_counter()
            full, dynamic, transport = verified_png_inputs(
                frames, policy, Path(output_dir) / "inputs" / input_info["rgb_sha256"],
            )
            output = {"status": "in_progress", "requested_dimensions": list(DIMENSIONS),
                      "full_vbench_aggregate": False, "input": input_info,
                      "provenance": self.provenance, "lossless_transport": transport, "scores": {},
                      "transport_latency_seconds": time.perf_counter() - start}
            token = "lossless-rgb:" + input_info["rgb_sha256"]
            for dimension in DIMENSIONS:
                module, model = self.modules[dimension], self.models[dimension]
                self._synchronize()
                before = time.perf_counter()
                if dimension == "motion_smoothness":
                    official, rows = module.motion_smoothness(model, [str(full)])
                elif dimension == "dynamic_degree":
                    official, rows = module.dynamic_degree(model, [str(dynamic)])
                else:
                    with array_loader(module, frames, token):
                        if dimension == "subject_consistency":
                            official, rows = module.subject_consistency(model, [token], self.device, False)
                        elif dimension == "background_consistency":
                            official, rows = module.background_consistency(model[0], model[1], [token],
                                                                           self.device, False)
                        elif dimension == "aesthetic_quality":
                            official, rows = module.laion_aesthetic(*model, [token], self.device)
                        elif dimension == "imaging_quality":
                            official, rows = module.technical_quality(
                                model, [token], self.device, imaging_quality_preprocessing_mode="longer",
                            )
                self._synchronize()
                if len(rows) != 1:
                    raise MetricInputError(f"{dimension}: expected exactly one per-video result")
                normalized, video_result = float(official), float(rows[0]["video_results"])
                divisor = 100.0 if dimension == "imaging_quality" else (
                    10.0 if dimension == "aesthetic_quality" else 1.0
                )
                raw = video_result if dimension == "imaging_quality" else normalized * divisor
                result = {
                    "raw_score": raw, "normalization_divisor": divisor, "normalized_score": normalized,
                    "official_dimension_score": normalized, "official_per_video_result": video_result,
                    "latency_seconds": time.perf_counter() - before,
                    "raw_units": "MUSIQ SPAQ nominal 0..100" if dimension == "imaging_quality" else (
                        "LAION linear predictor nominal 0..10" if dimension == "aesthetic_quality"
                        else "binary motion classification" if dimension == "dynamic_degree"
                        else "unitless official dimension"
                    ),
                    "higher_is_better": dimension != "dynamic_degree", **policy[dimension],
                }
                if dimension == "dynamic_degree":
                    result.update(flow_threshold_pixels=model.params["thres"],
                                  motion_pair_count_threshold=model.params["count_num"],
                                  interpretation="preserve paired RAW motion classification, not maximize")
                if dimension == "imaging_quality":
                    result["preprocessing_mode"] = "longer"
                output["scores"][dimension] = result
            validate_dimension_scores(output["scores"])
            # Imports performed lazily by a numerical routine cannot change NumPy's origin.
            verify_environment(self.config["environment"])
            output["status"] = "complete"
            output["score_latency_seconds"] = sum(row["latency_seconds"] for row in output["scores"].values())
            return output

    def close(self):
        self.models.clear()
        self.modules.clear()
        if self.original_torch_flags is not None and self.torch is not None:
            (self.torch.backends.cuda.matmul.allow_tf32, self.torch.backends.cudnn.allow_tf32,
             self.torch.backends.cudnn.benchmark) = self.original_torch_flags
            self.original_torch_flags = None
        sys.path[:] = self.original_sys_path
        gc.collect()
        if self.torch is not None and getattr(self, "device", None) is not None and self.device.type == "cuda":
            self.torch.cuda.empty_cache()


def evaluate_pair(raw_descriptor, candidate_descriptor, *, output_dir, scorer=None):
    reference, raw_info = load_rgb_npz(raw_descriptor)
    candidate, candidate_info = load_rgb_npz(candidate_descriptor)
    validate_pair(raw_info, candidate_info)
    metadata = raw_info["metadata"]
    main = metadata["clip_kind"] == "continuation"
    result = {
        "schema": "reelcache_rank_paired_metrics_v1", "status": "complete",
        "raw": raw_info, "candidate": candidate_info, "full_vbench_aggregate": False,
        "ssim": ssim_frames(reference, candidate), "vbench_status": "not_requested",
        "ssim_reference_self": 1.0, "metric_scope": "main_continuation" if main else "diagnostic_window",
        "window_label": None if main else metadata["window_label"],
    }
    result["ssim"]["comparison"] = (
        "candidate main continuation versus the same-run RAW continuation" if main else
        f"candidate diagnostic window {metadata['window_label']!r} versus the same-run RAW window"
    )
    result["ssim"]["source_frame_indices"] = raw_info["source_frame_indices"]
    if scorer is not None:
        # Validate temporal sufficiency even when a test substitutes a model-free scorer.
        frame_policy(len(reference), raw_info["source_fps"])
        raw_scores = scorer.score_frames(reference, raw_info, output_dir)
        candidate_scores = scorer.score_frames(candidate, candidate_info, output_dir)
        for scores in (raw_scores, candidate_scores):
            if scores["status"] != "complete":
                raise MetricInputError("one paired official metric evaluation did not complete")
            validate_dimension_scores(scores["scores"])
        result["vbench_status"] = "complete"
        result["vbench"] = {
            "raw": raw_scores, "candidate": candidate_scores,
            "normalized_delta": {
                name: candidate_scores["scores"][name]["normalized_score"] - raw_scores["scores"][name]["normalized_score"]
                for name in DIMENSIONS
            },
            "interpretation": "individual candidate minus RAW dimensions; DD is a motion diagnostic",
        }
    return result


def quality_outcomes(result, *, raw_accounting=None, candidate_accounting=None):
    """Convert a complete paired result to numeric analysis rows, retaining its binding."""
    if result.get("schema") != "reelcache_rank_paired_metrics_v1" or result.get("status") != "complete" or (
        result.get("vbench_status") != "complete"
    ) or result.get("full_vbench_aggregate") is not False:
        raise MetricInputError("quality tables require a successful same-run SSIM and six-dimension pair")
    validate_pair(result["raw"], result["candidate"])
    if result["raw"]["metadata"]["clip_kind"] != "continuation" or (
        result.get("metric_scope", "main_continuation") != "main_continuation"
        or result.get("window_label") is not None
    ):
        raise MetricInputError("diagnostic window scores cannot enter main continuation quality tables")
    for side in ("raw", "candidate"):
        validate_dimension_scores(result["vbench"][side]["scores"])
    scores = result["ssim"]["per_frame"]
    if len(scores) != len(result["raw"]["source_frame_indices"]) or not scores:
        raise MetricInputError("SSIM must retain every paired continuation frame")
    for score in scores:
        if not -1 <= _number(score, "SSIM") <= 1:
            raise MetricInputError("invalid SSIM frame score")
    if not math.isclose(math.fsum(scores) / len(scores), result["ssim"]["mean"], abs_tol=1e-12) or (
        min(scores) != result["ssim"]["minimum"]
    ):
        raise MetricInputError("SSIM summary differs from its actual frame observations")
    binding = {
        "pair_metadata": _paired_metadata(result["raw"]["metadata"]),
        "raw_reference_clip_id": result["raw"]["metadata"]["clip_id"],
        "raw_file_sha256": _hash(result["raw"]["file_sha256"], "RAW reference file_sha256"),
        "raw_rgb_sha256": _hash(result["raw"]["rgb_sha256"], "RAW reference rgb_sha256"),
        "source_frame_indices": result["raw"]["source_frame_indices"],
    }
    pairing_hash = hashlib.sha256(_json_bytes(binding)).hexdigest()
    result_hash = hashlib.sha256(_json_bytes(result)).hexdigest()
    output = []
    for side, accounting in (("raw", raw_accounting), ("candidate", candidate_accounting)):
        info = result[side]["metadata"]
        row = {
            **{name: info[name] for name in ("case_id", "prompt_group", "split", "seed", "cell", "arm_id")},
            "status": "complete", "pairing_sha256": pairing_hash, "paired_metrics_sha256": result_hash,
            "metrics": {
                "ssim": 1.0 if side == "raw" else result["ssim"]["mean"],
                **{name: result["vbench"][side]["scores"][name]["normalized_score"] for name in DIMENSIONS},
            },
        }
        if accounting is not None:
            row["accounting"] = accounting
        output.append(row)
    return output


def _validate_workspace_ancestry(workspace, project_root):
    """Check already resolved paths; this helper performs no filesystem access."""
    if (
        not workspace.is_absolute() or not project_root.is_absolute()
        or ".." in workspace.parts or ".." in project_root.parts
        or workspace == project_root or project_root not in workspace.parents
    ):
        raise MetricInputError("metric workspace must be a private directory inside the executing PROJECT_ROOT")


def _private_directory(path):
    if path.is_symlink():
        raise MetricInputError("metric workspace/cache directories must not be symlinks")
    try:
        path.mkdir(mode=0o700, parents=True, exist_ok=True)
        details = path.lstat()
    except OSError as error:
        raise MetricInputError("metric workspace/cache must be an accessible private directory") from error
    if (
        not stat.S_ISDIR(details.st_mode) or details.st_uid != os.geteuid()
        or stat.S_IMODE(details.st_mode) != 0o700
    ):
        raise MetricInputError("metric workspace/cache must be owned by this user with mode 0700")


def worker_environment(workspace, *, cuda_visible_devices="", threads=1):
    _int(threads, "threads", 1)
    if threads > 8:
        raise MetricInputError("metric CPU threads must be bounded by eight")
    requested = Path(workspace)
    workspace = requested.resolve()
    _validate_workspace_ancestry(workspace, PROJECT_ROOT.resolve())
    if requested.is_symlink():
        raise MetricInputError("metric workspace must not be a symlink")
    _private_directory(workspace)
    environment = {
        name: value for name, value in os.environ.items()
        if name not in ("PYTHONPATH", "PYTHONHOME", "LD_LIBRARY_PATH", "LD_PRELOAD", "VIRTUAL_ENV")
    }
    environment.update({
        "CUDA_VISIBLE_DEVICES": cuda_visible_devices, "REELCACHE_RANK_METRIC_WORKER": "1",
        "PYTHONDONTWRITEBYTECODE": "1", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false", "PYTHONNOUSERSITE": "1",
        "OMP_NUM_THREADS": str(threads), "MKL_NUM_THREADS": str(threads),
        "OPENBLAS_NUM_THREADS": str(threads), "NUMEXPR_NUM_THREADS": str(threads),
    })
    cache_root = workspace / "worker_cache"
    _private_directory(cache_root)
    for name in ("TMPDIR", "MPLCONFIGDIR", "TORCH_HOME", "HF_HOME", "XDG_CACHE_HOME",
                 "CUDA_CACHE_PATH", "TORCH_EXTENSIONS_DIR"):
        path = cache_root / name.lower()
        _private_directory(path)
        environment[name] = str(path)
    return environment


def launch_metric_worker(request_path, output_path, *, python_executable, workspace,
                         allow_model_load=False, cuda_visible_devices="", threads=1):
    """Synchronous isolated subprocess; parent retains all admission/grant ownership."""
    request = read_json(request_path)
    if request.get("vbench") is True:
        if allow_model_load is not True:
            raise MetricInputError("no model/metric-model load before independent approval")
        bound_python = request["metric_config"]["environment"]["python_executable"]
        if Path(python_executable).absolute() != Path(bound_python).absolute():
            raise MetricInputError("worker executable differs from the frozen metric environment")
    command = [str(python_executable), "-I", str(Path(__file__).resolve()),
               "--request", str(Path(request_path).resolve()), "--output", str(Path(output_path).resolve()),
               "--workspace", str(Path(workspace).resolve()), "--threads", str(threads)]
    if allow_model_load:
        command.append("--allow-model-load")
    return subprocess.run(command, env=worker_environment(
        workspace, cuda_visible_devices=cuda_visible_devices, threads=threads,
    ), cwd=str(Path(workspace).resolve()), check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--allow-model-load", action="store_true")
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite paired metric evidence")
    scorer, request = None, None
    try:
        request = read_json(args.request)
        if type(request.get("vbench")) is not bool:
            raise MetricInputError("request must explicitly declare vbench true/false")
        environment = worker_environment(
            args.workspace, cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            threads=args.threads,
        )
        for name in ("PYTHONPATH", "PYTHONHOME", "LD_LIBRARY_PATH", "LD_PRELOAD", "VIRTUAL_ENV"):
            os.environ.pop(name, None)
        os.environ.update(environment)
        if request["vbench"]:
            scorer = StrictVBenchScorer(request["metric_config"], device=request["device"],
                                       allow_model_load=args.allow_model_load)
        result = evaluate_pair(request["raw"], request["candidate"], output_dir=args.workspace, scorer=scorer)
        result["request_sha256"] = sha256(args.request)
        write_json(args.output, result)
    except Exception as error:
        write_json(args.output, {
            "schema": "reelcache_rank_paired_metrics_v1", "status": "failed",
            "error_type": type(error).__name__, "error": str(error), "full_vbench_aggregate": False,
            "requested_dimensions": list(DIMENSIONS) if request and request.get("vbench") else [],
            "request_sha256": sha256(args.request) if args.request.is_file() else None,
        })
        raise
    finally:
        if scorer is not None:
            scorer.close()


if __name__ == "__main__":
    main()
