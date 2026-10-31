"""VBench v1 protocol validation; no models, downloads, or GPU initialization.

Prompt annotations and aggregate constants are from Vchitect/VBench, Apache-2.0,
at the revision below. This pilot is not a full VBench leaderboard submission.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import random
import re


REVISION = "fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490"
SOURCE_SHA256 = "5dd2de80ee43cda750b2b72ea7023657c0b90d3702041c7e4608c65dbe50dccd"
REPOSITORY = "https://github.com/Vchitect/VBench"
QUALITY = (
    "subject_consistency", "background_consistency", "temporal_flickering",
    "motion_smoothness", "dynamic_degree", "aesthetic_quality", "imaging_quality",
)
GENERIC_QUALITY = tuple(d for d in QUALITY if d != "temporal_flickering")
SEMANTIC = (
    "object_class", "multiple_objects", "human_action", "color",
    "spatial_relationship", "scene", "appearance_style", "temporal_style",
    "overall_consistency",
)
DIMENSIONS = QUALITY + SEMANTIC
EXTRA_ANNOTATIONS = {
    "object_class", "multiple_objects", "color", "spatial_relationship",
    "scene", "appearance_style",
}
QUALITY_BOUNDS = {
    "subject_consistency": (0.1462, 1.0),
    "background_consistency": (0.2615, 1.0),
    "temporal_flickering": (0.6293, 1.0),
    "motion_smoothness": (0.706, 0.9975),
    "dynamic_degree": (0.0, 1.0),
    "aesthetic_quality": (0.0, 1.0),
    "imaging_quality": (0.0, 1.0),
}
SCHEMAS = {
    "protocol": "reelcache.vbench_protocol.v1",
    "inputs": "reelcache.vbench_inputs.v1",
    "prepared": "reelcache.vbench_prepared.v1",
    "results": "reelcache.vbench_results.v1",
    "assets": "reelcache.vbench_assets.v1",
}
SSIM_PARAMETERS = {
    "data_range": 255, "channel_axis": -1, "win_size": 7,
    "gaussian_weights": False, "use_sample_covariance": True,
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def integer(value, name, minimum=0):
    require(type(value) is int and value >= minimum, f"{name} must be an integer >= {minimum}")
    return value


def finite(value, name):
    require(isinstance(value, (int, float)) and not isinstance(value, bool),
            f"{name} must be numeric")
    require(math.isfinite(value), f"{name} must be finite")
    return float(value)


def digest(value, name="sha256"):
    require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None,
            f"{name} must be a lowercase SHA256")
    return value


def identifier(value, name="id"):
    require(isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", value),
            f"Invalid {name}")
    return value


def sha256_file(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            result.update(block)
    return result.hexdigest()


def canonical_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def object_sha256(value):
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def read_json(path):
    def reject_constant(value):
        raise ValueError(f"Nonfinite JSON value: {value}")

    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    return json.loads(Path(path).read_text(), parse_constant=reject_constant,
                      object_pairs_hook=unique_keys)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + ".pending")
    with pending.open("x") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")
    pending.replace(path)


def dimensions_arg(values):
    require(bool(values) and len(values) == len(set(values)), "Dimensions must be nonempty and unique")
    require(not (set(values) - set(DIMENSIONS)), f"Unknown dimensions: {set(values) - set(DIMENSIONS)}")
    return tuple(values)


def source_rows(path):
    require(sha256_file(path) == SOURCE_SHA256, "Official prompt source SHA256 differs from pinned source")
    rows = read_json(path)
    require(isinstance(rows, list) and len(rows) == 946, "Unexpected official prompt source")
    return rows


def _prior_texts(value, location=""):
    if isinstance(value, dict):
        for key in ("text", "prompt", "prompt_en"):
            if isinstance(value.get(key), str):
                yield value[key], location
        for key, child in value.items():
            if isinstance(child, (dict, list)):
                yield from _prior_texts(child, f"{location}/{key}")
    elif isinstance(value, list):
        for i, child in enumerate(value):
            yield from _prior_texts(child, f"{location}/{i}")


def make_candidate(full_info, prior_paths=(), selection_seed=20260918, strata=None):
    rows = source_rows(full_info)
    rng = random.Random(selection_seed)
    strata = tuple(strata or ("subject_consistency", "background_consistency",
                            "temporal_flickering", "aesthetic_quality"))
    dimensions_arg(strata)
    require(len(strata) == 4, "The proposal requires exactly four annotation strata")
    priors = []
    for path in prior_paths:
        priors.append({
            "path": str(Path(path).resolve()), "sha256": sha256_file(path),
            "texts": dict(_prior_texts(read_json(path))),
        })
    prompts = []
    for dimension in strata:
        index = rng.choice([i for i, row in enumerate(rows) if dimension in row["dimension"]])
        row = rows[index]
        matches = [{"path": p["path"], "sha256": p["sha256"],
                    "location": p["texts"][row["prompt_en"]]}
                   for p in priors if row["prompt_en"] in p["texts"]]
        prompts.append({
            "id": f"vb{index:04d}", "source_index": index, "text": row["prompt_en"],
            "official_annotation": copy.deepcopy(row), "selection_stratum": dimension,
            "prior_membership": matches,
            "novelty_claim": "Only membership in the listed prior manifests was checked; not certified untouched.",
        })
    return {
        "schema": SCHEMAS["protocol"], "status": "candidate",
        "source": {
            "repository": REPOSITORY, "commit": REVISION, "sha256": SOURCE_SHA256,
            "path": "sources/VBench_full_info.json",
            "url": f"https://raw.githubusercontent.com/Vchitect/VBench/{REVISION}/vbench/VBench_full_info.json",
            "license": "Apache-2.0",
            "license_url": f"https://raw.githubusercontent.com/Vchitect/VBench/{REVISION}/LICENSE",
        },
        "selection": {
            "seed": selection_seed, "strata_in_order": list(strata),
            "method": "One random.Random(seed) generator; choice from source-order indices for each stratum.",
            "output_selection": False, "prior_membership_used_for_selection": False,
            "prior_manifests_checked": [{k: p[k] for k in ("path", "sha256")} for p in priors],
        },
        "prompts": prompts,
        "models": {
            "causvid_480p": {"height": 480, "width": 832},
            "magi_distill_1080p": {"height": 1080, "width": 1920, "working_height": 1088},
        },
        "generation_seed": 0, "continuation_frames": 96, "fps": 24,
        "fps_policy": "Declared metric playback; no frame interpolation, padding or duplication. Record native FPS separately.",
        "primary_arms": ["raw", "svd_hevc"],
        "primary_compression": {
            "K_rank": 256, "V_rank": 384, "normalization": "per_channel",
            "pix_fmt": "yuv444p", "qp": 8, "preset": "ultrafast", "approximate_reuse": False,
        },
        "generic_quality_dimensions": list(GENERIC_QUALITY),
        "annotated_dimensions": [d for d in DIMENSIONS if any(
            d in p["official_annotation"]["dimension"] for p in prompts)],
        "static_filter_policy": "Official StaticFilter; paired intersection of all supplied arms, report every exclusion.",
        "ssim": SSIM_PARAMETERS.copy(),
        "scope": "Four-prompt, one-seed continuation pilot, not the full official prompt/seed suite.",
        "full_official_aggregate": False,
    }


def validate_protocol(protocol, rows, *, frozen=False):
    require(protocol.get("schema") == SCHEMAS["protocol"], "Wrong protocol schema")
    require(protocol.get("status") in ("candidate", "frozen"), "Unknown protocol status")
    if frozen:
        require(protocol["status"] == "frozen" and bool(protocol.get("frozen_by"))
                and bool(protocol.get("frozen_utc")), "Coordinator-frozen protocol is required")
    source = protocol["source"]
    require(source["sha256"] == SOURCE_SHA256 and source["commit"] == REVISION
            and source["repository"] == REPOSITORY, "Unpinned official source")
    prompts = protocol["prompts"]
    require(isinstance(prompts, list) and bool(prompts), "No prompts")
    seen_ids, seen_texts = set(), set()
    for prompt in prompts:
        pid = identifier(prompt["id"], "prompt id")
        index = integer(prompt["source_index"], "source_index")
        require(index < len(rows), "Prompt source index is out of range")
        require(prompt["text"] == rows[index]["prompt_en"], "Prompt text was changed")
        require(prompt["official_annotation"] == rows[index], "Official annotation was changed")
        require(pid not in seen_ids and prompt["text"] not in seen_texts, "Duplicate prompt")
        seen_ids.add(pid)
        seen_texts.add(prompt["text"])
    integer(protocol["generation_seed"], "generation_seed")
    integer(protocol["continuation_frames"], "continuation_frames", 3)
    require(finite(protocol["fps"], "fps") > 0, "FPS must be positive")
    require(protocol["continuation_frames"] / protocol["fps"] < 5.0,
            "Pinned VBench FAQ routes clips >= 5 seconds to VBench-Long, which this runner does not implement")
    require(protocol.get("ssim") == SSIM_PARAMETERS, "SSIM settings differ from the declared metric")
    require(protocol.get("full_official_aggregate") is False, "Pilot cannot claim a full official aggregate")
    require(bool(protocol["models"]), "No model geometries")
    for model, geometry in protocol["models"].items():
        identifier(model, "model")
        integer(geometry["height"], "height", 16)
        integer(geometry["width"], "width", 16)
    require(protocol.get("primary_arms") == ["raw", "svd_hevc"], "Primary arms must remain explicitly paired")
    return protocol


def load_protocol(path, *, frozen=False):
    path = Path(path).resolve()
    protocol = read_json(path)
    source = Path(protocol["source"]["path"])
    if not source.is_absolute():
        source = path.parent / source
    validate_protocol(protocol, source_rows(source), frozen=frozen)
    return protocol, sha256_file(path)


def frame_policy(count, fps, dimensions):
    integer(count, "frame_count", 3)
    require(finite(fps, "fps") > 0, "FPS must be positive")
    dimensions_arg(dimensions)
    result = {}
    for dimension in dimensions:
        indices = list(range(count))
        if dimension == "dynamic_degree":
            interval = max(1, round(fps / 8))
            indices = indices[::interval]
            threshold = round(4 * len(indices) / 16)
            require(len(indices) >= 2 and threshold >= 1,
                    "Clip is too short for meaningful official dynamic_degree sampling")
            result[dimension] = {"indices": indices, "interval": interval,
                                 "motion_pair_count_threshold": threshold}
        elif dimension == "motion_smoothness":
            anchors = indices[::2]
            result[dimension] = {
                "indices": indices, "anchor_indices": anchors,
                "target_indices": indices[1:2 * (len(anchors) - 1):2],
                "unused_terminal_indices": [count - 1] if count % 2 == 0 else [],
            }
        elif dimension in SEMANTIC and dimension != "appearance_style":
            samples = 8 if dimension in ("overall_consistency", "temporal_style") else 16
            require(count >= samples, f"{dimension} needs >= {samples} real frames; no padding")
            # Exact middle-index rule in pinned utils.get_frame_indices.
            indices = [(i * count // samples + (i + 1) * count // samples - 1) // 2
                       for i in range(samples)]
            result[dimension] = {"indices": indices, "sampling": "official_middle"}
        else:
            result[dimension] = {"indices": indices}
    return result


def validate_inputs(data, protocol, protocol_sha256):
    require(data.get("schema") == SCHEMAS["inputs"], "Wrong input manifest schema")
    require(data.get("protocol_sha256") == protocol_sha256, "Input/protocol binding differs")
    require(isinstance(data.get("clips"), list) and bool(data["clips"]), "No input clips")
    prompts = {p["id"]: p for p in protocol["prompts"]}
    clips, identities = {}, set()
    for clip in data["clips"]:
        cid = identifier(clip["id"], "clip id")
        require(cid not in clips, "Duplicate clip id")
        require(clip["model"] in protocol["models"], "Unknown model")
        require(clip["prompt_id"] in prompts, "Unknown prompt id")
        identifier(clip["arm"], "arm")
        representation = clip.get("representation", {})
        if clip["arm"] == "raw":
            require(canonical_bytes(representation) == canonical_bytes({"kind": "raw"}),
                    "RAW representation declaration differs")
        elif clip["arm"] == "svd_hevc":
            require(canonical_bytes(representation) == canonical_bytes(
                {"kind": "svd_hevc", **protocol["primary_compression"]}),
                    "Primary lossy representation differs from frozen protocol")
        else:
            require(clip.get("cohort_role") == "diagnostic" and isinstance(representation, dict)
                    and bool(representation.get("kind")), "Diagnostic arms need an explicit representation and role")
        digest(clip["representation_evidence_sha256"], "representation_evidence_sha256")
        require(clip["seed"] == protocol["generation_seed"] and type(clip["seed"]) is int,
                "Seed differs from protocol")
        require(clip["fps"] == protocol["fps"] and not isinstance(clip["fps"], bool),
                "Metric FPS differs from protocol")
        if clip.get("native_fps") != clip["fps"]:
            require(isinstance(clip.get("timing_note"), str) and bool(clip["timing_note"]),
                    "Unknown/different native FPS requires an explicit timing note")
        if clip.get("native_fps") is not None:
            require(finite(clip["native_fps"], "native_fps") > 0, "native_fps must be positive")
        integer(clip["prefix_frames"], "prefix_frames", 1)
        require(clip["native_frame_start"] == clip["prefix_frames"]
                and type(clip["native_frame_start"]) is int,
                "Window must begin after the entire common prefix")
        require(clip["frame_count"] == protocol["continuation_frames"]
                and type(clip["frame_count"]) is int, "Continuation frame count differs")
        integer(clip["rgb_frame_start"], "rgb_frame_start")
        digest(clip["rgb_file_sha256"], "rgb_file_sha256")
        digest(clip["generation_binding_sha256"], "generation_binding_sha256")
        require(isinstance(clip["rgb_path"], str) and bool(clip["rgb_path"]), "Missing RGB path")
        identity = (clip["model"], clip["prompt_id"], clip["seed"], clip["arm"])
        require(identity not in identities, "Duplicate model/prompt/seed/arm")
        identities.add(identity)
        clips[cid] = clip
    for clip in clips.values():
        raw = clips.get(clip["paired_raw_id"])
        require(raw is not None and raw["arm"] == "raw", "Missing paired RAW clip")
        if clip["arm"] == "raw":
            require(raw["id"] == clip["id"], "RAW must pair to itself")
        for key in ("model", "prompt_id", "seed", "fps", "native_fps", "prefix_frames",
                    "native_frame_start", "frame_count", "generation_binding_sha256"):
            require(clip.get(key) == raw.get(key), f"RAW pairing mismatch: {key}")
    return data


def dimension_records(protocol, clips, dimension, video_paths, quality_mode="generic"):
    dimensions_arg([dimension])
    require(quality_mode in ("generic", "annotated"), "Unknown quality coverage mode")
    prompts = {p["id"]: p for p in protocol["prompts"]}
    result = []
    for clip in clips:
        prompt = prompts[clip["prompt_id"]]
        annotation = prompt["official_annotation"]
        generic = quality_mode == "generic" and dimension in GENERIC_QUALITY
        if not generic and dimension not in annotation["dimension"]:
            continue
        if dimension in EXTRA_ANNOTATIONS:
            require(dimension in annotation.get("auxiliary_info", {}),
                    f"Missing official auxiliary annotation for {dimension}")
        row = copy.deepcopy(annotation)
        row["dimension"] = [dimension]
        row["video_list"] = [str(video_paths[clip["id"]])]
        result.append(row)
    return result


def validate_official_results(dimension, result, expected_paths):
    require(isinstance(result, (tuple, list)) and len(result) == 2, "Malformed official result")
    finite(float(result[0]), "official dimension score")
    require(isinstance(result[1], list), "Missing official per-video results")
    by_path = {}
    for row in result[1]:
        path = str(Path(row["video_path"]).resolve())
        require(path not in by_path, "Duplicate official per-video result")
        value = finite(float(row["video_results"]), "official per-video result")
        normalized = value / 100 if dimension == "imaging_quality" else value
        if dimension == "dynamic_degree":
            require(normalized in (0, 1), "Official dynamic_degree must be binary per video")
        by_path[path] = {"official_per_video_result": value, "dimension_score": normalized}
    require(set(by_path) == {str(Path(p).resolve()) for p in expected_paths},
            "Official evaluation omitted or added input videos")
    return by_path


def quality_aggregate(scores):
    require(set(scores) == set(QUALITY), "Seven-dimension quality summary requires all seven dimensions")
    numerator, denominator = 0.0, 0.0
    normalized = {}
    for dimension in QUALITY:
        lo, hi = QUALITY_BOUNDS[dimension]
        value = finite(scores[dimension], dimension)
        normalized[dimension] = (value - lo) / (hi - lo)
        weight = 0.5 if dimension == "dynamic_degree" else 1.0
        numerator += weight * normalized[dimension]
        denominator += weight
    return {
        "score": numerator / denominator, "normalized_dimensions": normalized,
        "scope": "VBench-compatible quality formula on this pilot's stated per-dimension cohorts; NOT a leaderboard score.",
        "official_full_aggregate": False,
        "formula_source": f"{REPOSITORY}/blob/{REVISION}/scripts/constant.py",
    }


def compression_ratios(storage):
    require(storage.get("denominator") == "native_dense_kv_bytes",
            "Compression denominator must be native dense KV, not FP32 factor inputs or videos")
    require(storage.get("native_dtype") == "bfloat16",
            "This study's numerator is native logical BF16 KV, not canonical FP32 pre-K")
    require(storage.get("unique_native_head_owners") is True,
            "Native logical bytes must count each native KV head owner once, without query-head expansion")
    raw = integer(storage["native_dense_kv_bytes"], "native_dense_kv_bytes", 1)
    decoder = integer(storage["unique_decoder_bytes"], "unique_decoder_bytes", 1)
    require(isinstance(storage.get("scope_id"), str) and bool(storage["scope_id"]),
            "Byte accounting needs a named head/chunk/branch scope")
    require(isinstance(storage.get("includes"), list)
            and {"payloads", "whole_immutable_bases", "required_manifests"} <= set(storage["includes"]),
            "Decoder closure must include payloads, full immutable bases and every required manifest")
    digest(storage["evidence_sha256"], "storage evidence_sha256")
    result = {
        "scope_id": storage["scope_id"], "native_dense_kv_bytes": raw,
        "native_dtype": "bfloat16", "unique_native_head_owners": True,
        "unique_decoder_bytes": decoder, "native_over_decoder_ratio": raw / decoder,
        "decoder_fraction_of_native": decoder / raw,
        "evidence_sha256": storage["evidence_sha256"],
    }
    if "snapshot_closure_bytes" in storage:
        closure = integer(storage["snapshot_closure_bytes"], "snapshot_closure_bytes", decoder)
        result.update(snapshot_closure_bytes=closure, native_over_snapshot_ratio=raw / closure)
    for key in ("physical_index_bytes", "physical_filesystem_bytes"):
        if key in storage:
            result[key] = integer(storage[key], key)
    return result


def primary_coverage(spec, clips):
    expected = {p["id"] for p in spec["prompts"]}
    return [{
        "model": model, "arm": arm, "expected_cases": len(expected),
        "present_prompt_ids": sorted({c["prompt_id"] for c in clips
                                      if c["model"] == model and c["arm"] == arm}),
        "missing_prompt_ids": sorted(expected - {c["prompt_id"] for c in clips
                                                 if c["model"] == model and c["arm"] == arm}),
    } for model in spec["models"] for arm in spec["primary_arms"]]
