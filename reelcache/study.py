"""Source-bound protocols and launch gates for the experimental rank study."""

from __future__ import annotations

import contextlib
import dataclasses
import difflib
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import random
import re
import shutil
import tempfile
import unicodedata
from types import MappingProxyType
from typing import Any, Iterable, Mapping

from .errors import CapacityError, ConfigurationError, IntegrityError, StateError


VBENCH_COMMIT = "fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490"
VBENCH_PROMPTS_SHA256 = "5dd2de80ee43cda750b2b72ea7023657c0b90d3702041c7e4608c65dbe50dccd"
PROTOCOL_SCHEMA = "reelcache.rank_study.v1"
ASSET_SCHEMA = "reelcache.study_assets.v1"
LAUNCH_SCHEMA = "reelcache.study_launch.v1"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DISCOVERY = {
    "directories": ["reelcache", "examples", "scripts", "tests"],
    "pattern": "*.py", "root_files": ["pyproject.toml"],
}


def canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    except (TypeError, ValueError, OverflowError) as exc:
        raise ConfigurationError("study metadata must be finite portable JSON") from exc


def identity(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: str | Path) -> Any:
    def invalid_constant(value: str) -> None:
        raise IntegrityError(f"nonfinite JSON constant: {value}")

    def unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise IntegrityError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    try:
        return json.loads(
            Path(path).read_text(), parse_constant=invalid_constant,
            object_pairs_hook=unique_pairs,
        )
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise IntegrityError(f"invalid JSON document: {path}") from exc


def write_json_new(path: str | Path, value: Any) -> None:
    """Atomically publish a new immutable JSON file without overwriting evidence."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = canonical_json(value) + b"\n"
    descriptor, pending = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(pending, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        Path(pending).unlink()


def _integer(value: Any, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ConfigurationError(f"{name} must be an integer >= {minimum}")
    return value


def _digest(value: Any, name: str) -> str:
    if not isinstance(value, str) or re.fullmatch("[0-9a-f]{64}", value) is None:
        raise ConfigurationError(f"{name} must be a lowercase SHA256")
    return value


def _relative_name(value: Any) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ConfigurationError("asset/output names must be relative POSIX paths")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or str(path) != value or value == ".":
        raise ConfigurationError("asset/output names must be normalized relative paths")
    return value


@dataclasses.dataclass(frozen=True)
class StudyCell:
    cell_id: str
    model: str
    width: int
    height: int
    working_height: int
    crop_top: int
    frames: int
    token_height: int
    token_width: int
    layers: int
    channels: int
    key_dtype: str
    value_dtype: str
    world_size: int

    @property
    def tokens(self) -> int:
        return self.frames * self.token_height * self.token_width

    @property
    def source_bytes(self) -> int:
        sizes = {"float32": 4, "bfloat16": 2}
        return self.tokens * self.channels * self.layers * (
            sizes[self.key_dtype] + sizes[self.value_dtype]
        )

    @property
    def native_kv_bytes(self) -> int:
        return self.tokens * self.channels * self.layers * 4

    def to_dict(self) -> dict[str, Any]:
        return {
            **dataclasses.asdict(self), "tokens": self.tokens,
            "source_bytes_per_chunk": self.source_bytes,
            "native_kv_bytes_per_chunk": self.native_kv_bytes,
        }


CELLS = MappingProxyType({
    "causvid480": StudyCell(
        "causvid480", "causvid", 832, 480, 480, 0, 3, 30, 52, 30, 1536,
        "bfloat16", "bfloat16", 1,
    ),
    "magi720": StudyCell(
        "magi720", "magi_distill", 1280, 720, 720, 0, 6, 45, 80, 48, 1024,
        "float32", "bfloat16", 2,
    ),
    "magi1080": StudyCell(
        "magi1080", "magi_distill", 1920, 1080, 1088, 4, 6, 68, 120, 48, 1024,
        "float32", "bfloat16", 2,
    ),
})

NATIVE_SOURCE_FPS = MappingProxyType({
    "causvid480": 16,
    "magi720": 24,
    "magi1080": 24,
})


def study_cell(cell_id: str) -> StudyCell:
    if not isinstance(cell_id, str) or cell_id not in CELLS:
        raise ConfigurationError("study cell must be causvid480, magi720, or magi1080")
    return CELLS[cell_id]


_STYLE_PREFIXES = (
    "a tranquil tableau of ", "in a still frame ", "a still frame of ",
    "a beautiful ", "an animated painting of ",
)
_STOP_WORDS = frozenset(
    "a an the in on at of and with to from is are as by its it this that "
    "red blue green yellow black white orange purple pink brown grey gray "
    "beautiful small large single view scene frame".split()
)


def prompt_signature(text: str) -> tuple[str, frozenset[str]]:
    if not isinstance(text, str) or not text.strip():
        raise ConfigurationError("prompt text must be nonempty")
    normalized = unicodedata.normalize("NFKC", text).casefold()
    normalized = re.sub(
        r",?\s*(?:pan (?:left|right)|tilt (?:up|down)|zoom (?:in|out)|"
        r"frozen in time|in super slow motion|with an intense shaking effect)\.?\s*$", "", normalized,
    )
    normalized = " ".join(re.findall(r"[a-z0-9]+", normalized))
    for prefix in _STYLE_PREFIXES:
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix):]
    tokens = frozenset(word for word in normalized.split() if word not in _STOP_WORDS)
    return normalized, tokens


def related_prompts(left: str, right: str) -> bool:
    a, tokens_a = prompt_signature(left)
    b, tokens_b = prompt_signature(right)
    if a == b:
        return True
    union = tokens_a | tokens_b
    if tokens_a and tokens_b and len(tokens_a & tokens_b) / len(union) >= 0.75:
        return True
    return difflib.SequenceMatcher(None, a, b, autojunk=False).ratio() >= 0.86


def historical_prompt_texts(documents: Iterable[Mapping[str, Any]]) -> set[str]:
    """Extract only explicit prompt fields, never arbitrary prose or file paths."""
    result: set[str] = set()

    def visit(value: Any, prompt_container: bool = False) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                if key in {"prompt", "prompt_en", "prompt_text"} and isinstance(item, str):
                    if item.strip():
                        result.add(item)
                elif key == "text" and prompt_container and isinstance(item, str):
                    if item.strip():
                        result.add(item)
                else:
                    visit(item, prompt_container or "prompt" in key.casefold())
        elif isinstance(value, list):
            for item in value:
                if prompt_container and isinstance(item, str):
                    if item.strip():
                        result.add(item)
                else:
                    visit(item, prompt_container)

    for document in documents:
        if not isinstance(document, Mapping):
            raise ConfigurationError("historical prompt manifests must be JSON objects")
        visit(document)
    return result


def _stratum(dimensions: Iterable[str]) -> str:
    values = set(dimensions)
    if "temporal_flickering" in values:
        return "static"
    if values & {"human_action", "dynamic_degree", "motion_smoothness", "temporal_style"}:
        return "motion"
    if values & {"object_class", "multiple_objects", "color", "spatial_relationship"}:
        return "objects"
    return "scenes"


def _prompt_family_groups(texts: list[str]) -> list[list[str]]:
    signatures = [prompt_signature(text) for text in texts]
    parents = list(range(len(texts)))

    def root(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    for i, (normalized, tokens) in enumerate(signatures):
        for j in range(i):
            other, other_tokens = signatures[j]
            union = tokens | other_tokens
            overlap = bool(tokens and other_tokens) and len(tokens & other_tokens) / len(union) >= 0.75
            length_bound = 2 * min(len(normalized), len(other)) / max(
                len(normalized) + len(other), 1
            )
            text_match = length_bound >= 0.86 and difflib.SequenceMatcher(
                None, normalized, other, autojunk=False
            ).ratio() >= 0.86
            if normalized == other or overlap or text_match:
                parents[root(i)] = root(j)
    groups: dict[int, list[str]] = {}
    for index, text in enumerate(texts):
        groups.setdefault(root(index), []).append(text)
    return list(groups.values())


def create_protocol(
    catalog_path: str | Path,
    *,
    historical_paths: Iterable[str | Path],
    selection_seed: int = 20260918,
    generation_seeds: tuple[int, int] = (0, 1),
    calibration_groups: int = 12,
    heldout_groups: int = 8,
    expected_catalog_sha256: str = VBENCH_PROMPTS_SHA256,
) -> dict[str, Any]:
    """Freeze outcome-blind, lexically grouped, stratified official prompts."""
    _integer(selection_seed, "selection seed")
    for value, name in (
        (calibration_groups, "calibration groups"), (heldout_groups, "heldout groups")
    ):
        _integer(value, name, 1)
    if len(generation_seeds) != 2 or len(set(generation_seeds)) != 2:
        raise ConfigurationError("exactly two distinct generation seeds are required")
    for seed in generation_seeds:
        _integer(seed, "generation seed")
    catalog_path = Path(catalog_path)
    if file_sha256(catalog_path) != _digest(expected_catalog_sha256, "catalog hash"):
        raise IntegrityError("official prompt catalog hash mismatch")
    catalog = read_json(catalog_path)
    if not isinstance(catalog, list) or not catalog:
        raise IntegrityError("official prompt catalog must be a nonempty list")
    entries: dict[str, dict[str, Any]] = {}
    for index, entry in enumerate(catalog):
        if not isinstance(entry, dict):
            raise IntegrityError("invalid official prompt entry")
        text, dimensions = entry.get("prompt_en"), entry.get("dimension")
        prompt_signature(text)
        if not isinstance(dimensions, list) or not all(
            isinstance(item, str) and item for item in dimensions
        ):
            raise IntegrityError("invalid official dimension list")
        existing = entries.setdefault(text, {"text": text, "source_indices": [], "dimensions": []})
        existing["source_indices"].append(index)
        existing["dimensions"] = sorted(set(existing["dimensions"]) | set(dimensions))
    histories = [Path(path).resolve(strict=True) for path in historical_paths]
    if not histories or len(histories) != len(set(histories)):
        raise ConfigurationError("unique historical exclusion manifests must be explicit")
    excluded = historical_prompt_texts(read_json(path) for path in histories)
    if not excluded:
        raise IntegrityError("historical manifests yielded no prompt exclusions")
    texts = sorted(entries)
    families = _prompt_family_groups(texts)
    candidates: dict[str, list[dict[str, Any]]] = {
        name: [] for name in ("static", "motion", "objects", "scenes")
    }
    for family in families:
        if any(related_prompts(text, old) for text in family for old in excluded):
            continue
        representative = min(family, key=lambda text: entries[text]["source_indices"][0])
        record = dict(entries[representative])
        record["related_source_indices"] = sorted(
            index for text in family for index in entries[text]["source_indices"]
        )
        record["group_id"] = "g-" + identity(sorted(family))[:16]
        record["stratum"] = _stratum(record["dimensions"])
        candidates[record["stratum"]].append(record)
    rng = random.Random(selection_seed)
    for family in candidates.values():
        rng.shuffle(family)
    selected = []
    names = list(candidates)
    for split, count in (("calibration", calibration_groups), ("heldout", heldout_groups)):
        for index in range(count):
            preferred = names[index % len(names)]
            available = [
                preferred, *(name for name in names if name != preferred)
            ]
            source = next((name for name in available if candidates[name]), None)
            if source is None:
                raise ConfigurationError("not enough unexposed prompt groups for the requested split")
            record = candidates[source].pop()
            selected.append({**record, "split": split, "prompt_id": f"{split[:3]}{index:02d}"})
    document = {
        "schema": PROTOCOL_SCHEMA, "study": "calibrated_online_rank",
        "catalog": {
            "path": str(catalog_path.resolve()), "sha256": expected_catalog_sha256,
            "commit": VBENCH_COMMIT, "entry_count": len(catalog), "unique_prompts": len(entries),
        },
        "exclusions": [
            {"path": str(path), "sha256": file_sha256(path)} for path in histories
        ],
        "excluded_prompt_count": len(excluded),
        "selection": {
            "seed": selection_seed, "method": "lexical-family-stratified-v1",
            "grouping": "connected components; token Jaccard >= .75 or text ratio >= .86",
            "semantic_grouping_review_required": True,
            "outcomes_consulted": False,
        },
        "prompts": selected, "seeds": list(generation_seeds),
        "calibration_groups": calibration_groups, "heldout_groups": heldout_groups,
        "clean_boundaries": [0, 1, 2], "energy_targets": [0.95, 0.99],
        "cells": {name: cell.to_dict() for name, cell in CELLS.items()},
        "full_official_vbench_authorized": False,
    }
    return {**document, "protocol_id": identity(document)}


def validate_protocol(document: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(document, Mapping) or document.get("schema") != PROTOCOL_SCHEMA:
        raise IntegrityError("unsupported rank-study protocol")
    copied = dict(document)
    digest = copied.pop("protocol_id", None)
    if digest != identity(copied):
        raise IntegrityError("rank-study protocol identity mismatch")
    if copied.get("cells") != {name: cell.to_dict() for name, cell in CELLS.items()}:
        raise IntegrityError("study geometry does not match the approved three cells")
    if copied.get("energy_targets") != [0.95, 0.99] or copied.get("clean_boundaries") != [0, 1, 2]:
        raise IntegrityError("study energy targets or boundary contract changed")
    catalog = copied.get("catalog")
    if not isinstance(catalog, dict) or catalog.get("commit") != VBENCH_COMMIT:
        raise IntegrityError("protocol must use the pinned official VBench source")
    _digest(catalog.get("sha256"), "catalog hash")
    if copied.get("full_official_vbench_authorized") is not False:
        raise IntegrityError("this study does not authorize full official VBench")
    seeds = copied.get("seeds")
    if not isinstance(seeds, list) or len(seeds) != 2 or len(set(seeds)) != 2:
        raise IntegrityError("protocol requires two distinct seeds")
    for seed in seeds:
        _integer(seed, "generation seed")
    prompts = copied.get("prompts")
    if not isinstance(prompts, list) or not prompts:
        raise IntegrityError("protocol has no prompts")
    seen_groups: set[str] = set()
    seen_texts: list[str] = []
    seen_ids: set[str] = set()
    counts = {"calibration": 0, "heldout": 0}
    for prompt in prompts:
        if not isinstance(prompt, dict) or prompt.get("split") not in counts:
            raise IntegrityError("invalid prompt split")
        text, group, prompt_id = prompt.get("text"), prompt.get("group_id"), prompt.get("prompt_id")
        prompt_signature(text)
        if not isinstance(group, str) or not group or not isinstance(prompt_id, str) or not prompt_id:
            raise IntegrityError("prompt and group identities must be explicit")
        if group in seen_groups or prompt_id in seen_ids or any(
            related_prompts(text, previous) for previous in seen_texts
        ):
            raise IntegrityError("duplicate or related prompts cross study group boundaries")
        seen_groups.add(group)
        seen_ids.add(prompt_id)
        seen_texts.append(text)
        counts[prompt["split"]] += 1
    for split in counts:
        if counts[split] != _integer(copied.get(f"{split}_groups"), f"{split} groups", 1):
            raise IntegrityError("prompt split counts differ from the protocol")
    return dict(document)


def validate_request(
    request: Mapping[str, Any],
    protocol: Mapping[str, Any],
    *,
    phase: str,
) -> dict[str, Any]:
    """Validate an immutable generation request before native model loading."""
    validate_protocol(protocol)
    if not isinstance(request, Mapping) or request.get("schema") != "reelcache.rank_request.v1":
        raise IntegrityError("unsupported study request")
    result = dict(request)
    request_id = result.pop("request_id", None)
    if request_id != identity(result):
        raise IntegrityError("study request identity mismatch")
    if result.get("protocol_id") != protocol["protocol_id"] or result.get("phase") != phase:
        raise IntegrityError("request is not bound to the admitted protocol/phase")
    cell = study_cell(result.get("cell_id"))
    chunks = _integer(result.get("chunks"), "chunks", 2)
    if chunks > len(protocol["clean_boundaries"]):
        raise ConfigurationError("longer horizons require a new frozen study protocol")
    split = result.get("split")
    if phase == "native_preflight":
        if split != "development":
            raise StateError("native preflight must not inspect fresh calibration or held-out cases")
    elif phase in {"calibration_census", "calibration_frontier"}:
        if split != "calibration":
            raise StateError("calibration phase may not open held-out outcomes")
    elif phase == "heldout":
        if split != "heldout":
            raise StateError("held-out run must use the frozen held-out groups")
    else:
        raise StateError("unsupported or separately unauthorized study phase")
    _integer(result.get("seed"), "request seed")
    if result["seed"] not in protocol["seeds"]:
        raise IntegrityError("request seed is outside the frozen seed set")
    if split != "development":
        matches = [
            item for item in protocol["prompts"]
            if item["group_id"] == result.get("group_id")
            and item["prompt_id"] == result.get("prompt_id")
            and item["text"] == result.get("prompt") and item["split"] == split
        ]
        if len(matches) != 1:
            raise IntegrityError("request prompt/group/split differs from the frozen protocol")
    else:
        prompt_signature(result.get("prompt"))
        if any(related_prompts(result["prompt"], item["text"]) for item in protocol["prompts"]):
            raise IntegrityError("development prompt overlaps a fresh study group")
    if result.get("history_mode") not in {"reload", "quality_hot"}:
        raise ConfigurationError("history mode must be reload or quality_hot")
    fps = result.get("source_fps")
    if isinstance(fps, bool) or not isinstance(fps, (int, float)) or not math.isfinite(fps) or fps <= 0:
        raise ConfigurationError("positive finite native source_fps is required")
    if fps != NATIVE_SOURCE_FPS[cell.cell_id]:
        raise ConfigurationError("source_fps differs from the source-bound native export recipe")
    _relative_name(result.get("output_namespace"))
    budgets = result.get("budgets")
    required = {
        "device_bytes_per_rank", "history_bytes_per_rank", "workspace_bytes_per_rank",
        "host_bytes", "source_staging_bytes_global", "disk_bytes", "disk_reserve_bytes",
    }
    if not isinstance(budgets, dict) or set(budgets) != required:
        raise ConfigurationError("request must declare every device/host/staging/disk budget")
    for name, value in budgets.items():
        _integer(value, name, 1)
    if budgets["source_staging_bytes_global"] < cell.source_bytes:
        raise CapacityError("current capture path requires one complete clean source chunk")
    if budgets["host_bytes"] < budgets["source_staging_bytes_global"]:
        raise CapacityError("host budget does not cover declared clean-source staging")
    if budgets["device_bytes_per_rank"] < (
        budgets["history_bytes_per_rank"] + budgets["workspace_bytes_per_rank"]
    ):
        raise CapacityError("device budget does not cover the declared history and workspace")
    if result["history_mode"] == "quality_hot" and budgets["history_bytes_per_rank"] < (
        cell.native_kv_bytes * chunks // cell.world_size
    ):
        raise CapacityError("quality-hot history budget cannot hold the admitted trajectory")
    representation = result.get("representation")
    if not isinstance(representation, dict) or representation.get("kind") not in {"raw", "svd", "svd_hevc"}:
        raise ConfigurationError("request representation must be explicit RAW, SVD, or SVD+HEVC")
    if phase == "calibration_census" and representation["kind"] != "raw":
        raise StateError("the calibration energy census requires native RAW history")
    if representation["kind"] == "raw":
        if (
            representation.get("rank_policy") is not None
            or representation.get("layer_ranks") is not None
            or any(
                representation.get(key) is not None for key in ("rank_k", "rank_v")
            )
        ):
            raise ConfigurationError("native RAW requests may not bind a factor rank/policy")
    else:
        policy = representation.get("rank_policy")
        layer_ranks = representation.get("layer_ranks")
        if policy is not None and layer_ranks is not None:
            raise ConfigurationError("static layer ranks and an adaptive policy are mutually exclusive")
        if policy is not None:
            if not isinstance(policy, dict):
                raise ConfigurationError("rank_policy must be inline portable policy data, not a mutable path")
            codec = policy.get("codec")
            if not isinstance(codec, dict):
                raise ConfigurationError("rank_policy must declare its factor codec")
            for name in ("kind", "normalization", "pix_fmt", "qp", "preset", "inter_prediction"):
                if name in representation and canonical_json(codec.get(name)) != canonical_json(
                    representation[name]
                ):
                    raise ConfigurationError("rank_policy codec differs from the declared study arm")
        if layer_ranks is not None:
            if (
                not isinstance(layer_ranks, dict) or set(layer_ranks) != {"K", "V"}
                or any(key in representation for key in ("rank_k", "rank_v"))
            ):
                raise ConfigurationError("static layer ranks require only complete K/V vectors")
            ranks = []
            for target, values in layer_ranks.items():
                if not isinstance(values, list) or len(values) != cell.layers:
                    raise ConfigurationError("static rank vectors must cover every native layer")
                ranks.extend((f"layer_ranks.{target}[{index}]", value)
                             for index, value in enumerate(values))
        else:
            scalar_keys = [key for key in ("rank_k", "rank_v") if key in representation]
            if policy is not None and len(scalar_keys) == 1:
                raise ConfigurationError("legacy adaptive rank hints must supply both K and V")
            ranks = [(key, representation.get(key)) for key in ("rank_k", "rank_v")
                     if policy is None or key in representation]
        for name, value in ranks:
            rank = _integer(value, name, 1)
            if rank > min(cell.tokens, cell.channels):
                raise ConfigurationError("rank exceeds the native full-channel domain")
        if representation["kind"] == "svd_hevc" and (
            representation.get("normalization") != "per_channel"
            or representation.get("pix_fmt") != "yuv444p"
            or representation.get("qp") != 8 or representation.get("preset") != "ultrafast"
        ):
            raise ConfigurationError("primary study fixes per-channel yuv444p QP8/ultrafast")
    if not isinstance(result.get("arm_id"), str) or not result["arm_id"]:
        raise ConfigurationError("request arm_id must be a nonempty string")
    return dict(request)


def verify_protocol_sources(protocol: Mapping[str, Any]) -> None:
    validate_protocol(protocol)
    catalog_record = protocol["catalog"]
    if file_sha256(catalog_record["path"]) != catalog_record["sha256"]:
        raise IntegrityError("official prompt source changed after selection")
    catalog = read_json(catalog_record["path"])
    if not isinstance(catalog, list) or len(catalog) != catalog_record["entry_count"]:
        raise IntegrityError("prompt source entry count changed")
    for record in protocol["exclusions"]:
        if file_sha256(record["path"]) != record["sha256"]:
            raise IntegrityError("historical exclusions changed after selection")
    for prompt in protocol["prompts"]:
        indices = prompt.get("source_indices")
        if not isinstance(indices, list) or not indices or len(set(indices)) != len(indices):
            raise IntegrityError("official prompt source indices must be unique and nonempty")
        dimensions = set()
        for index in indices:
            if type(index) is not int or not 0 <= index < len(catalog):
                raise IntegrityError("official prompt source index is invalid")
            if catalog[index].get("prompt_en") != prompt["text"]:
                raise IntegrityError("study prompt is not the unchanged official text")
            dimensions.update(catalog[index]["dimension"])
        if dimensions != set(prompt["dimensions"]):
            raise IntegrityError("study dimensions do not match the original prompt")


def verify_production_protocol(protocol: Mapping[str, Any]) -> None:
    """Fixture/custom catalogs can be analyzed, but cannot authorize native runs."""
    verify_protocol_sources(protocol)
    if (
        protocol["catalog"]["sha256"] != VBENCH_PROMPTS_SHA256
        or protocol["catalog"]["entry_count"] != 946
        or protocol["catalog"]["unique_prompts"] != 944
        or protocol["calibration_groups"] != 12 or protocol["heldout_groups"] != 8
        or protocol["seeds"] != [0, 1]
    ):
        raise StateError("native launch requires the pinned official catalog and approved 12/8, seed-0/1 cohort")


def validate_request_batch(
    document: Mapping[str, Any],
    protocol: Mapping[str, Any],
    receipt: Mapping[str, Any],
    *,
    cell_id: str,
    phase: str,
) -> tuple[dict[str, Any], ...]:
    """Admit one homogeneous model/resolution epoch with unique output ownership."""
    study_cell(cell_id)
    verify_launch_receipt(receipt, protocol, phase=phase)
    if not isinstance(document, Mapping):
        raise IntegrityError("request batch must be an object")
    body = dict(document)
    digest = body.pop("requests_id", None)
    if (
        body.get("schema") != "reelcache.rank_requests.v1" or identity(body) != digest
        or body.get("protocol_id") != protocol["protocol_id"]
        or body.get("receipt_id") != receipt["receipt_id"] or body.get("phase") != phase
    ):
        raise IntegrityError("request batch identity, phase, or source authorization mismatch")
    requests = body.get("requests")
    if not isinstance(requests, list) or not requests:
        raise IntegrityError("resident epoch must contain explicit requests")
    result = tuple(validate_request(item, protocol, phase=phase) for item in requests)
    if any(item["cell_id"] != cell_id for item in result):
        raise ConfigurationError("one resident epoch cannot silently switch model or resolution")
    for field in ("request_id", "output_namespace"):
        if len({item[field] for item in result}) != len(result):
            raise IntegrityError(f"resident epoch has duplicate {field} values")
    if phase == "heldout":
        allowed = receipt.get("frozen_arms")
        if not isinstance(allowed, list) or not allowed:
            raise StateError("held-out receipt has no frozen arm definitions")
        allowed_ids = {identity(arm) for arm in allowed}
        if any(identity(arm_contract(item)) not in allowed_ids for item in result):
            raise StateError("held-out request changes an arm or setting frozen on calibration")
    return result


def source_closure(root: str | Path, extra_paths: Iterable[str | Path] = ()) -> dict[str, Any]:
    root = Path(root).resolve(strict=True)
    files = [
        path for directory in ("reelcache", "examples", "scripts", "tests")
        for path in (root / directory).rglob("*.py")
    ]
    files.append(root / "pyproject.toml")
    files.extend(Path(path).resolve(strict=True) for path in extra_paths)
    records = {}
    for path in sorted(set(path.resolve(strict=True) for path in files)):
        key = str(path.relative_to(root)) if path.is_relative_to(root) else str(path)
        records[key] = {"path": str(path), "sha256": file_sha256(path), "bytes": path.stat().st_size}
    body = {
        "schema": "reelcache.launch_sources.v1", "root": str(root),
        "discovery": json.loads(canonical_json(SOURCE_DISCOVERY)), "files": records,
    }
    return {**body, "closure_id": identity(body)}


def verify_source_closure(closure: Mapping[str, Any]) -> None:
    if closure.get("schema") != "reelcache.launch_sources.v1":
        raise IntegrityError("unsupported source closure")
    body = dict(closure)
    digest = body.pop("closure_id", None)
    records = body.get("files")
    if (
        not isinstance(records, dict) or not records or digest != identity(body)
        or body.get("discovery") != SOURCE_DISCOVERY
    ):
        raise IntegrityError("invalid source closure identity")
    root = Path(closure["root"]).resolve(strict=True)
    if str(root) != closure["root"]:
        raise IntegrityError("closure root must be canonical")
    current_paths = {
        str(path.resolve(strict=True))
        for directory in ("reelcache", "examples", "scripts", "tests")
        for path in (root / directory).rglob("*.py")
    } | {str(root / "pyproject.toml")}
    recorded_paths = {record["path"] for record in records.values()}
    if not current_paths <= recorded_paths:
        raise IntegrityError("new source files appeared after the launch closure was frozen")
    for key, record in records.items():
        path = Path(record["path"]).resolve(strict=True)
        expected_key = str(path.relative_to(root)) if path.is_relative_to(root) else str(path)
        if key != expected_key or str(path) != record["path"]:
            raise IntegrityError("closure paths do not match their canonical discovery root")
        if not path.is_file() or path.stat().st_size != record["bytes"]:
            raise IntegrityError(f"launch source disappeared or changed size: {key}")
        if file_sha256(path) != record["sha256"]:
            raise IntegrityError(f"launch source changed after verification: {key}")


def make_launch_receipt(
    protocol: Mapping[str, Any],
    closure: Mapping[str, Any],
    *,
    verification_path: str | Path,
    review_path: str | Path,
) -> dict[str, Any]:
    verify_production_protocol(protocol)
    verify_source_closure(closure)
    if Path(closure["root"]) != PROJECT_ROOT.resolve(strict=True):
        raise IntegrityError("launch closure does not identify the executing runtime project")
    verification, review = read_json(verification_path), read_json(review_path)
    if not isinstance(verification, dict) or verification.get("schema") != "reelcache_cpu_verification_v1":
        raise StateError("a typed CPU verification result is required")
    tests_run = _integer(verification.get("tests_run"), "tests_run", 1)
    if verification.get("status") != "passed":
        raise StateError("a passing nonempty CPU verification suite is required")
    for name in ("skipped", "failures", "errors", "expected_failures", "unexpected_successes"):
        if _integer(verification.get(name), name) != 0:
            raise StateError(f"mandatory prelaunch gate requires zero {name}")
    if verification.get("launch_closure_id") != closure["closure_id"]:
        raise IntegrityError("CPU verification did not test this launch closure")
    if (
        verification.get("source_unchanged_during_tests") is not True
        or verification.get("launch_closure_error") is not None
        or verification.get("require_no_skips") is not True
    ):
        raise IntegrityError("source changed during CPU verification")
    discovered = verification.get("mandatory_inventory")
    selected = verification.get("selected_inventory")
    records = verification.get("records")
    if (
        not isinstance(discovered, list) or not discovered
        or not all(isinstance(item, str) and item for item in discovered)
        or len(set(discovered)) != len(discovered) or selected != discovered
        or verification.get("discovery") != {
            "root": str(PROJECT_ROOT / "tests"), "pattern": "test_*.py", "full_suite": True,
        }
    ):
        raise StateError("only the complete mandatory test inventory can authorize launch")
    if (
        not isinstance(records, list) or len(records) != tests_run or tests_run != len(discovered)
        or any(not isinstance(record, dict) or record.get("status") != "passed" for record in records)
        or sorted(record.get("test", "") for record in records) != sorted(discovered)
    ):
        raise StateError("test records contradict the passing counters or mandatory inventory")
    expected_sources = {
        key: record["sha256"] for key, record in closure["files"].items()
        if key == "pyproject.toml" or (
            not PurePosixPath(key).is_absolute() and PurePosixPath(key).parts[0]
            in SOURCE_DISCOVERY["directories"] and key.endswith(".py")
        )
    }
    if (
        verification.get("source_files") != expected_sources
        or verification.get("source_tree_sha256") != identity(expected_sources)
    ):
        raise IntegrityError("CPU verification source inventory differs from the launch closure")
    expected_test_modules = {
        key.removesuffix(".py").replace("/", ".")
        for key in expected_sources if key.startswith("tests/")
        and PurePosixPath(key).name.startswith("test_")
    }
    if any(
        not any(test.startswith(module + ".") for test in discovered)
        for module in expected_test_modules
    ):
        raise StateError("mandatory inventory omits a discovered test module")
    if (
        review.get("schema") != "reelcache.independent_review.v1"
        or review.get("status") != "passed"
        or review.get("blocking_findings") != []
        or review.get("independent_of_implementation") is not True
        or not review.get("reviewer")
        or review.get("launch_closure_id") != closure["closure_id"]
    ):
        raise StateError("this complete source closure needs a passing independent review")
    if review.get("protocol_id") != protocol["protocol_id"]:
        raise IntegrityError("review is not bound to this protocol")
    if review.get("prompt_grouping_reviewed") is not True:
        raise StateError("selected prompt families require semantic grouping review")
    body = {
        "schema": LAUNCH_SCHEMA, "protocol_id": protocol["protocol_id"],
        "source_closure": dict(closure), "phase": "native_preflight",
        "verification": {
            "path": str(Path(verification_path).resolve()),
            "sha256": file_sha256(verification_path),
        },
        "review": {"path": str(Path(review_path).resolve()), "sha256": file_sha256(review_path)},
    }
    return {**body, "receipt_id": identity(body)}


def verify_launch_receipt(
    receipt: Mapping[str, Any], protocol: Mapping[str, Any], *, phase: str = "native_preflight"
) -> None:
    validate_protocol(protocol)
    body = dict(receipt)
    digest = body.pop("receipt_id", None)
    if (
        body.get("schema") != LAUNCH_SCHEMA or identity(body) != digest
        or body.get("protocol_id") != protocol["protocol_id"] or body.get("phase") != phase
    ):
        raise IntegrityError("launch receipt does not authorize this protocol and phase")
    verify_source_closure(body["source_closure"])
    for name in ("verification", "review"):
        record = body[name]
        if file_sha256(record["path"]) != record["sha256"]:
            raise IntegrityError(f"{name} evidence changed after approval")
    rebuilt = make_launch_receipt(
        protocol, body["source_closure"],
        verification_path=body["verification"]["path"], review_path=body["review"]["path"],
    )
    if phase == "native_preflight" and rebuilt != dict(receipt):
        raise IntegrityError("launch receipt differs from its verified evidence")
    if phase != "native_preflight":
        previous = body.get("previous_receipt")
        evidence = body.get("phase_evidence")
        if not isinstance(previous, dict) or not isinstance(evidence, dict):
            raise IntegrityError("later phases require the complete preceding launch/evidence chain")
        verify_launch_receipt(previous, protocol, phase=previous.get("phase"))
        if file_sha256(evidence["path"]) != evidence["sha256"]:
            raise IntegrityError("phase evidence changed after admission")
        expected = advance_launch_receipt(
            previous, protocol, evidence_path=evidence["path"], next_phase=phase,
            _previous_verified=True,
        )
        if expected != dict(receipt):
            raise IntegrityError("later-phase receipt differs from its evidence")


_PREFLIGHT_CHECKS = frozenset({
    "native_geometry", "original_pre_rope_capture", "raw_replay",
    "resident_aba", "native_history_visibility", "hot_history_parity",
    "staged_memory_admission", "oracle_svd_crosscheck",
})
_PHASE_TRANSITIONS = {
    "native_preflight": "calibration_census",
    "calibration_census": "calibration_frontier",
    "calibration_frontier": "heldout",
}


def arm_contract(request: Mapping[str, Any]) -> dict[str, Any]:
    fields = ("cell_id", "arm_id", "representation", "history_mode", "chunks", "source_fps")
    if any(key not in request for key in fields):
        raise IntegrityError("frozen arm lacks representation or evaluation settings")
    return {key: request[key] for key in fields}


def _artifact(record: Any) -> Any:
    if (
        not isinstance(record, dict) or not isinstance(record.get("path"), str)
        or file_sha256(record["path"]) != record.get("sha256")
    ):
        raise IntegrityError("phase artifact is missing or changed")
    return read_json(record["path"])


def _proof(
    record: Any, *, schema: str, protocol: Mapping[str, Any],
    closure_id: str, cell_id: str, phase: str,
) -> dict[str, Any]:
    result = _artifact(record)
    if (
        not isinstance(result, dict) or result.get("schema") != schema
        or result.get("status") != "passed" or result.get("protocol_id") != protocol["protocol_id"]
        or result.get("launch_closure_id") != closure_id or result.get("cell_id") != cell_id
        or result.get("phase") != phase or result.get("fixture") is not False
        or result.get("native_execution") is not True
    ):
        raise IntegrityError("phase proof is failed, synthetic, unrelated, or not source-bound")
    return result


def _numeric(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise IntegrityError("measurement must be a finite number, not a boolean")
    return float(value)


def _native_preflight_proof(proof: Mapping[str, Any], cell: StudyCell) -> set[str]:
    checks = proof.get("checks")
    if not isinstance(checks, dict) or set(checks) != _PREFLIGHT_CHECKS:
        raise IntegrityError("native proof must record every mandatory preflight check")
    required_measurements = {
        "native_geometry": {"cell_geometry"},
        "original_pre_rope_capture": {"source_contract"},
        "raw_replay": {"latent_sha256", "rgb_sha256"},
        "resident_aba": {"latent_sha256", "rgb_sha256"},
        "native_history_visibility": {"archived_prefix_sha256"},
        "hot_history_parity": {"native_kv_sha256", "latent_sha256", "rgb_sha256"},
        "staged_memory_admission": {
            "source_staging_bytes_global", "host_peak_bytes", "device_peak_bytes_per_rank_max",
        },
        "oracle_svd_crosscheck": {"K_max_energy_error", "V_max_energy_error"},
    }
    source_contract = {
        "layers": cell.layers,
        "K": {"shape": [cell.tokens, cell.channels], "dtype": cell.key_dtype, "key_space": "pre_rope"},
        "V": {"shape": [cell.tokens, cell.channels], "dtype": cell.value_dtype, "key_space": "none"},
    }
    for name, check in checks.items():
        if not isinstance(check, dict) or check.get("status") != "passed":
            raise StateError(f"native preflight check failed: {name}")
        measurements = check.get("measurements")
        if not isinstance(measurements, list) or not measurements:
            raise IntegrityError("native check has no measured observations")
        by_name = {}
        for measurement in measurements:
            if not isinstance(measurement, dict) or measurement.get("name") in by_name:
                raise IntegrityError("invalid or duplicate native measurement")
            measurement_name = measurement.get("name")
            if not isinstance(measurement_name, str) or not measurement_name:
                raise IntegrityError("native measurement needs a name")
            observed, expected = measurement.get("observed"), measurement.get("expected")
            comparison = measurement.get("comparison")
            if measurement_name in required_measurements[name]:
                if measurement_name.endswith("_sha256"):
                    _digest(observed, "observed native hash")
                    _digest(expected, "expected native hash")
                    if comparison != "equal":
                        raise IntegrityError("native identity proofs must compare SHA256 values for equality")
                elif name in {"native_geometry", "original_pre_rope_capture"}:
                    if comparison != "equal" or not isinstance(observed, dict) or not isinstance(expected, dict):
                        raise IntegrityError("native geometry/source proofs require structural equality")
                else:
                    if comparison != "leq":
                        raise IntegrityError("native error and memory bounds require nonnegative upper-bound checks")
                    if _numeric(observed) < 0 or _numeric(expected) < 0:
                        raise IntegrityError("native error and memory measurements cannot be negative")
            if comparison == "equal":
                if canonical_json(observed) != canonical_json(expected):
                    raise StateError(f"native equality check did not pass: {measurement_name}")
                if measurement_name.endswith("_sha256"):
                    _digest(observed, "native measurement hash")
            elif comparison == "leq":
                if not 0 <= _numeric(observed) <= _numeric(expected):
                    raise StateError(f"native bound did not pass: {measurement_name}")
            else:
                raise IntegrityError("native measurement uses an unrecognized comparison")
            by_name[measurement_name] = measurement
        if not required_measurements[name] <= set(by_name):
            raise IntegrityError(f"native proof omits required measurements: {name}")
        if name == "native_geometry" and by_name["cell_geometry"]["observed"] != cell.to_dict():
            raise IntegrityError("measured native geometry differs from the study cell")
        if name == "original_pre_rope_capture" and by_name["source_contract"]["observed"] != source_contract:
            raise IntegrityError("measured original-source contract differs from the cell")
        if name == "staged_memory_admission":
            for key in required_measurements[name]:
                _integer(by_name[key]["observed"], key)
                _integer(by_name[key]["expected"], f"{key} capacity", 1)
            if _numeric(by_name["source_staging_bytes_global"]["observed"]) < cell.source_bytes:
                raise IntegrityError("staging proof does not account for the complete original source")
            if any(_numeric(by_name[key]["expected"]) <= 0 for key in required_measurements[name]):
                raise IntegrityError("memory proof requires positive admitted capacities")
        if name == "oracle_svd_crosscheck" and any(
            _numeric(by_name[key]["expected"]) > 0.01 for key in required_measurements[name]
        ):
            raise IntegrityError("oracle comparison tolerance cannot discard a material energy error")
        artifacts = check.get("artifacts")
        if not isinstance(artifacts, list) or not artifacts:
            raise IntegrityError("native check has no retained comparison artifacts")
        for artifact in artifacts:
            if (
                not isinstance(artifact, dict) or not isinstance(artifact.get("path"), str)
                or file_sha256(artifact["path"]) != artifact.get("sha256")
            ):
                raise IntegrityError("native comparison artifact is missing or changed")
    return set(checks)


def _census_proof(
    proof: Mapping[str, Any], protocol: Mapping[str, Any], cell: StudyCell,
) -> set[tuple[str, int, int, int, str]]:
    from .spectra import SpectrumResult

    observations = proof.get("observations")
    if not isinstance(observations, list) or not observations:
        raise IntegrityError("census proof must contain source-bound matrix observations")
    allowed_groups = {
        item["group_id"] for item in protocol["prompts"] if item["split"] == "calibration"
    }
    seen = set()
    for observation in observations:
        if not isinstance(observation, dict):
            raise IntegrityError("invalid census observation")
        group = observation.get("group_id")
        seed = _integer(observation.get("seed"), "census seed")
        boundary = _integer(observation.get("boundary"), "census boundary")
        layer = _integer(observation.get("layer"), "census layer")
        target = observation.get("target")
        if (
            group not in allowed_groups or seed not in protocol["seeds"]
            or boundary not in protocol["clean_boundaries"] or layer >= cell.layers
            or target not in {"K", "V"} or observation.get("cell_id") != cell.cell_id
        ):
            raise IntegrityError("census observation is outside the declared calibration cell")
        key = (group, seed, boundary, layer, target)
        if key in seen:
            raise IntegrityError("duplicate census matrix observation")
        payload = _artifact(observation.get("spectrum"))
        if isinstance(payload, dict) and "spectrum" in payload:
            payload = payload["spectrum"]
        spectrum = SpectrumResult.from_dict(payload)
        expected_provenance = {
            "protocol_id": protocol["protocol_id"], "cell_id": cell.cell_id,
            "group_id": group, "seed": seed, "boundary": boundary, "layer": layer,
            "target": target, "split": "calibration", "fixture": False, "native_execution": True,
            "launch_closure_id": proof["launch_closure_id"],
        }
        if (
            spectrum.source.shape != (cell.tokens, cell.channels)
            or spectrum.source.target != target
            or spectrum.source.dtype != (cell.key_dtype if target == "K" else cell.value_dtype)
            or any(
                canonical_json(spectrum.source.provenance.get(name)) != canonical_json(value)
                for name, value in expected_provenance.items()
            )
        ):
            raise IntegrityError("spectrum source does not substantiate its census position")
        seen.add(key)
    return seen


def _frontier_proof(
    proof: Mapping[str, Any], protocol: Mapping[str, Any], cell_id: str,
) -> dict[str, dict[str, Any]]:
    arms = proof.get("arms")
    cases = proof.get("cases")
    if not isinstance(arms, list) or not arms or not isinstance(cases, list) or not cases:
        raise IntegrityError("frontier proof must contain arms and per-case measured outcomes")
    contracts = {}
    for arm in arms:
        contract = arm_contract(arm)
        if contract != arm or contract["cell_id"] != cell_id or contract["arm_id"] in contracts:
            raise IntegrityError("frontier arm is duplicated or belongs to a different cell")
        contracts[contract["arm_id"]] = contract
    expected = {
        (prompt["group_id"], seed, arm_id)
        for prompt in protocol["prompts"] if prompt["split"] == "calibration"
        for seed in protocol["seeds"] for arm_id in contracts
    }
    seen = set()
    for case in cases:
        if not isinstance(case, dict):
            raise IntegrityError("frontier case must be a typed outcome")
        key = (case.get("group_id"), case.get("seed"), case.get("arm_id"))
        _integer(case.get("seed"), "frontier seed")
        if key not in expected or key in seen or case.get("cell_id") != cell_id:
            raise IntegrityError("frontier case is duplicated or outside calibration")
        outcome = _artifact(case.get("outcome"))
        if (
            not isinstance(outcome, dict) or outcome.get("schema") != "reelcache.rank_case_result.v1"
            or outcome.get("protocol_id") != protocol["protocol_id"]
            or outcome.get("launch_closure_id") != proof["launch_closure_id"]
            or outcome.get("native_execution") is not True or outcome.get("fixture") is not False
            or outcome.get("split") != "calibration" or outcome.get("cell_id") != cell_id
            or any(outcome.get(name) != case.get(name) for name in ("group_id", "seed", "arm_id"))
            or outcome.get("arm") != contracts[case["arm_id"]]
        ):
            raise IntegrityError("frontier case artifact does not match its bound arm and source")
        if outcome.get("status") == "passed":
            quality = outcome.get("quality")
            if not isinstance(quality, dict) or not -1 <= _numeric(quality.get("mean_ssim")) <= 1:
                raise IntegrityError("successful frontier case has no valid paired SSIM")
            dimensions = {
                "subject_consistency", "background_consistency", "motion_smoothness",
                "dynamic_degree", "aesthetic_quality", "imaging_quality",
            }
            if not isinstance(quality.get("vbench"), dict) or set(quality["vbench"]) != dimensions:
                raise IntegrityError("successful frontier case omits official metric dimensions")
            for value in quality["vbench"].values():
                _numeric(value)
            for name in ("native_logical_bytes", "decoder_closure_bytes", "snapshot_closure_bytes"):
                _integer(outcome.get(name), name, 1)
        elif outcome.get("status") not in {"failed", "abstained"} or not isinstance(
            outcome.get("failure_reason"), str
        ) or not outcome["failure_reason"]:
            raise IntegrityError("failed frontier case has no explicit failure/abstention reason")
        seen.add(key)
    if seen != expected:
        raise StateError("frontier completion cannot omit difficult prompt/seed/arm cases")
    return contracts


def advance_launch_receipt(
    previous: Mapping[str, Any],
    protocol: Mapping[str, Any],
    *,
    evidence_path: str | Path,
    next_phase: str,
    _previous_verified: bool = False,
) -> dict[str, Any]:
    previous_phase = previous.get("phase")
    if _PHASE_TRANSITIONS.get(previous_phase) != next_phase:
        raise StateError("study phases cannot be skipped; full VBench requires separate authorization")
    if not _previous_verified:
        verify_launch_receipt(previous, protocol, phase=previous_phase)
    evidence = read_json(evidence_path)
    if (
        not isinstance(evidence, dict) or evidence.get("schema") != "reelcache.rank_phase_result.v1"
        or evidence.get("status") != "passed" or evidence.get("phase") != previous_phase
        or evidence.get("protocol_id") != protocol["protocol_id"]
        or evidence.get("launch_closure_id") != previous["source_closure"]["closure_id"]
        or evidence.get("heldout_outcomes_opened") is not False
    ):
        raise StateError("phase needs complete, source-bound evidence without held-out leakage")
    cells = evidence.get("cells")
    if not isinstance(cells, dict) or set(cells) != set(CELLS):
        raise StateError("all three cells must pass before expanding the study")
    measured_arms = {}
    for cell_id, result in cells.items():
        if not isinstance(result, dict) or result.get("status") != "passed":
            raise StateError(f"study cell is incomplete: {cell_id}")
        proofs = result.get("evidence")
        if not isinstance(proofs, list) or not proofs:
            raise IntegrityError(f"study cell has no persistent evidence: {cell_id}")
        schema = {
            "native_preflight": "reelcache.rank_native_preflight.v1",
            "calibration_census": "reelcache.rank_census.v1",
            "calibration_frontier": "reelcache.rank_frontier.v1",
        }[previous_phase]
        loaded = [
            _proof(
                record, schema=schema, protocol=protocol,
                closure_id=previous["source_closure"]["closure_id"], cell_id=cell_id,
                phase=previous_phase,
            ) for record in proofs
        ]
        if previous_phase == "native_preflight":
            for proof in loaded:
                _native_preflight_proof(proof, CELLS[cell_id])
        if previous_phase == "calibration_census":
            expected = {
                (prompt["group_id"], seed, boundary, layer, target)
                for prompt in protocol["prompts"] if prompt["split"] == "calibration"
                for seed in protocol["seeds"] for boundary in protocol["clean_boundaries"]
                for layer in range(CELLS[cell_id].layers) for target in ("K", "V")
            }
            observed = set()
            for proof in loaded:
                values = _census_proof(proof, protocol, CELLS[cell_id])
                if observed & values:
                    raise IntegrityError("census files duplicate matrix observations")
                observed.update(values)
            if observed != expected:
                raise StateError(f"calibration census has missing/extra matrix observations: {cell_id}")
        if previous_phase == "calibration_frontier":
            if len(loaded) != 1:
                raise IntegrityError("each cell needs one complete normalized frontier ledger")
            measured_arms[cell_id] = _frontier_proof(loaded[0], protocol, cell_id)
    frozen_arms = None
    if next_phase == "heldout":
        frozen = evidence.get("frozen_policies")
        if not isinstance(frozen, list) or not frozen or evidence.get("settings_frozen") is not True:
            raise StateError("held-out evaluation requires frozen policies and comparison settings")
        frozen_arms, seen_arms = [], set()
        for entry in frozen:
            payload = _artifact(entry)
            if (
                not isinstance(payload, dict) or payload.get("schema") != "reelcache.rank_frozen_arms.v1"
                or payload.get("protocol_id") != protocol["protocol_id"]
                or payload.get("launch_closure_id") != previous["source_closure"]["closure_id"]
                or not isinstance(payload.get("arms"), list) or not payload["arms"]
            ):
                raise IntegrityError("frozen policies need canonical source-bound arm definitions")
            for arm in payload["arms"]:
                contract = arm_contract(arm)
                key = (contract["cell_id"], contract["arm_id"])
                if (
                    contract != arm or key in seen_arms
                    or measured_arms.get(key[0], {}).get(key[1]) != contract
                ):
                    raise IntegrityError("frozen arm was not measured with these exact calibration settings")
                seen_arms.add(key)
                frozen_arms.append(contract)
        if {arm["cell_id"] for arm in frozen_arms} != set(CELLS):
            raise StateError("held-out evaluation must freeze finalists for all three cells")
        frozen_arms.sort(key=lambda arm: (arm["cell_id"], arm["arm_id"]))
    body = {
        key: value for key, value in previous.items()
        if key not in {"receipt_id", "previous_receipt", "phase_evidence"}
    }
    body.update({
        "phase": next_phase, "previous_receipt": dict(previous),
        "phase_evidence": {
            "path": str(Path(evidence_path).resolve()), "sha256": file_sha256(evidence_path),
        },
    })
    if frozen_arms is not None:
        body["frozen_arms"] = frozen_arms
    return {**body, "receipt_id": identity(body)}


def stage_assets(
    assets: Iterable[Mapping[str, Any]],
    destination: str | Path,
    *,
    reserve_bytes: int,
) -> dict[str, Any]:
    """Copy a fully enumerated asset set; publish readiness only after full hashes."""
    reserve_bytes = _integer(reserve_bytes, "disk reserve")
    records = list(assets)
    if not records:
        raise ConfigurationError("asset list is empty")
    names: set[str] = set()
    sources = []
    for record in records:
        name = _relative_name(record.get("name"))
        if name in names or name == "assets.json":
            raise ConfigurationError("duplicate or reserved asset name")
        names.add(name)
        path = Path(record["source"]).resolve(strict=True)
        if not path.is_file():
            raise ConfigurationError(f"asset is not a regular file: {path}")
        size = _integer(record.get("bytes"), "asset bytes")
        digest = _digest(record.get("sha256"), "asset SHA256")
        if path.stat().st_size != size:
            raise IntegrityError(f"source asset size mismatch: {path}")
        sources.append((name, path, size, digest))
    destination = Path(destination).absolute()
    destination.parent.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(destination.parent).free
    required = sum(size for _, _, size, _ in sources)
    if free < required + reserve_bytes:
        raise CapacityError("insufficient storage for declared assets plus reserve")
    try:
        destination.mkdir(mode=0o700)
    except FileExistsError as exc:
        raise StateError("asset staging destination must be new") from exc
    staging = destination.resolve(strict=True)
    completed = False
    try:
        copied = []
        for name, source, size, expected in sources:
            target = staging / name
            target.parent.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256()
            before = source.stat()
            with source.open("rb") as src, target.open("xb") as dst:
                for block in iter(lambda: src.read(8 << 20), b""):
                    dst.write(block)
                    digest.update(block)
                dst.flush()
                os.fsync(dst.fileno())
            after = source.stat()
            if (
                (before.st_size, before.st_mtime_ns, before.st_ino, before.st_mode)
                != (after.st_size, after.st_mtime_ns, after.st_ino, after.st_mode)
                or digest.hexdigest() != expected or target.stat().st_size != size
                or file_sha256(target) != expected
            ):
                raise IntegrityError(f"source/destination asset hash mismatch: {name}")
            mode = 0o500 if before.st_mode & 0o111 else 0o400
            target.chmod(mode)
            copied.append({
                "name": name, "source": str(source), "bytes": size, "sha256": expected, "mode": mode,
            })
        body = {"schema": ASSET_SCHEMA, "files": copied, "total_bytes": required}
        manifest = {**body, "asset_id": identity(body)}
        write_json_new(staging / "assets.json", manifest)
        completed = True
        return manifest
    finally:
        if not completed:
            shutil.rmtree(staging)


def verify_assets(root: str | Path, *, rehash: bool = True) -> dict[str, Any]:
    root = Path(root).resolve(strict=True)
    manifest = read_json(root / "assets.json")
    if not isinstance(manifest, dict):
        raise IntegrityError("asset manifest must be an object")
    body = dict(manifest)
    digest = body.pop("asset_id", None)
    if body.get("schema") != ASSET_SCHEMA or digest != identity(body):
        raise IntegrityError("asset manifest identity mismatch")
    names: set[str] = set()
    total = 0
    for record in body["files"]:
        name = _relative_name(record["name"])
        if name in names:
            raise IntegrityError("duplicate staged asset")
        names.add(name)
        path = root / name
        resolved = path.resolve(strict=True)
        if not resolved.is_relative_to(root) or not resolved.is_file():
            raise IntegrityError("staged asset escapes its private root")
        size = _integer(record["bytes"], "asset bytes")
        _digest(record["sha256"], "asset hash")
        if "mode" in record:
            mode = record["mode"]
            if type(mode) is not int or mode not in (0o400, 0o500):
                raise IntegrityError("staged asset mode must be owner-only read or read/execute")
            if path.stat().st_mode & 0o7777 != mode:
                raise IntegrityError(f"staged asset permissions changed: {name}")
        if path.stat().st_size != size or (rehash and file_sha256(path) != record["sha256"]):
            raise IntegrityError(f"staged asset changed: {name}")
        total += size
    if total != body["total_bytes"] or not names:
        raise IntegrityError("asset byte ledger mismatch")
    return manifest


@contextlib.contextmanager
def resource_grant(lock_directory: str | Path, resources: Iterable[str]):
    """Hold task-wide advisory device grants; file existence is not the lock."""
    names = sorted(set(resources))
    if not names or any(re.fullmatch(r"[A-Za-z0-9_-]+", name) is None for name in names):
        raise ConfigurationError("resource grant names must be explicit nonempty identifiers")
    directory = Path(lock_directory)
    directory.mkdir(parents=True, exist_ok=True)
    handles = []
    try:
        for name in names:
            stream = (directory / f"{name}.lock").open("a+")
            handles.append(stream)
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise CapacityError(f"resource already granted: {name}") from exc
        yield
    finally:
        for stream in reversed(handles):
            fcntl.flock(stream, fcntl.LOCK_UN)
            stream.close()
