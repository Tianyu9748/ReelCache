"""Model-free study provenance for MAGI weights, fresh T5 and native VAE.

Full large-file hashing is a once-per-study operation. Per-case preparation
binds the immutable manifest and checks the verified originals' identity/stat
guards. That fast check is deliberately not described as another content hash.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time

from . import magi_native as native


def canonical_digest(value):
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False,
    ).encode()).hexdigest()


def stat_guard(value):
    return {name: getattr(value, attr) for name, attr in (
        ("device", "st_dev"), ("inode", "st_ino"), ("size", "st_size"),
        ("mtime_ns", "st_mtime_ns"), ("ctime_ns", "st_ctime_ns"),
        ("mode", "st_mode"), ("uid", "st_uid"), ("gid", "st_gid"), ("nlink", "st_nlink"),
    )}


def _current_guard(row):
    path = Path(row["path"])
    if str(path.resolve()) != row.get("resolved_path", str(path)):
        raise ValueError(f"A verified original was replaced or redirected: {path}")
    expected = row["final_manifest_stat"]
    actual = stat_guard(path.stat())
    if any(expected.get(key) != value for key, value in actual.items()) or actual["size"] != row["bytes"]:
        raise ValueError(f"Verified original stat guard changed; repeat explicit study hashing: {path}")


def _read_bound(binding, schema):
    row = native._file_row(binding["path"])
    if row["sha256"] != binding["sha256"]:
        raise ValueError("Study provenance manifest bytes changed")
    manifest = native._read(row["path"])
    original = dict(manifest)
    claimed = original.pop("manifest_integrity_sha256")
    if manifest["schema"] != schema or manifest["status"] != "passed" or canonical_digest(original) != claimed:
        raise ValueError("Study provenance integrity/status/schema failed")
    fingerprint = canonical_digest(manifest["content_manifest"])
    if fingerprint != manifest["content_manifest_sha256"] or fingerprint != binding["content_manifest_sha256"]:
        raise ValueError("Study provenance content fingerprint differs from the frozen protocol")
    return manifest, row


def _relative(path, directory):
    resolved = Path(path).resolve()
    return str(resolved.relative_to(Path(directory).resolve()))


def _check_complete_rows(rows, content_rows):
    if len(rows) != len(content_rows) or len({row["relative_path"] for row in rows}) != len(rows):
        raise ValueError("Study provenance has duplicate or incomplete file coverage")
    actual = sorted(
        ({key: row[key] for key in ("relative_path", "role", "bytes", "sha256")} for row in rows),
        key=lambda row: row["relative_path"],
    )
    if actual != sorted(content_rows, key=lambda row: row["relative_path"]):
        raise ValueError("Study content rows disagree with verified-file evidence")
    for row in rows:
        if not all(row.get(flag) is True for flag in
                   ("full_content_hashed", "stable_during_hash", "stable_through_manifest_validation")):
            raise ValueError("Header-only, incomplete or unstable provenance cannot bind a fresh case")
        _current_guard(row)


def verify_weights(binding, directory):
    manifest, artifact = _read_bound(binding, "reelcache_weight_identity_v1")
    model = manifest["content_manifest"]["models"]["magi"]
    model_digest = canonical_digest(model)
    if model_digest != manifest["model_content_sha256"]["magi"]:
        raise ValueError("MAGI full model-content fingerprint changed")
    rows = [row for row in manifest["files"] if row["model"] == "magi"]
    _check_complete_rows(rows, model["files"])
    root = Path(directory).resolve()
    index = root / "model.safetensors.index.json"
    expected_names = {"model.safetensors.index.json", *native._read(index)["weight_map"].values()}
    if {row["relative_path"] for row in rows} != expected_names:
        raise ValueError("Weight identity must cover the index and EVERY referenced MAGI shard")
    for row in rows:
        if (_relative(row["path"], root) != row["relative_path"]
                or row.get("matches_recorded_reference") is not True):
            raise ValueError("Configured MAGI weights are not the study's verified originals")
        if row["role"] != ("index" if row["relative_path"] == index.name else "weight_shard"):
            raise ValueError("MAGI weight evidence role does not match the index")
    if native._sha(index) != next(row["sha256"] for row in rows if row["role"] == "index"):
        raise ValueError("MAGI weight-map bytes changed after the full study verification")
    return {
        "artifact": artifact, "combined_content_manifest_sha256": manifest["content_manifest_sha256"],
        "magi_model_content_sha256": model_digest,
        "full_content_hash_evidence": "study manifest, complete index and referenced shards",
        "per_case_verification": "manifest SHA256/canonical integrity, exact original path/stat guards and index hash; NO shard reread",
        "shards_rehashed_per_case": False,
    }


def auxiliary_paths(t5, vae, source):
    t5 = Path(t5).resolve() / "t5-v1_1-xxl"
    small = ["config.json", "tokenizer_config.json", "special_tokens_map.json",
             "spiece.model", "pytorch_model.bin.index.json"]
    index = native._read(t5 / "pytorch_model.bin.index.json")
    shards = sorted(set(index["weight_map"].values()))
    paths = []
    for name in [*small, *shards]:
        path = (t5 / name).resolve()
        _relative(path, t5)
        paths.append(("t5", name, "weight_shard" if name in shards else "config_or_tokenizer", path))
    for name in ("config.json", "diffusion_pytorch_model.safetensors"):
        paths.append(("vae", name, "weights" if name.endswith(".safetensors") else "config",
                      Path(vae).resolve() / name))
    source = Path(source).resolve()
    for path in sorted((source / "inference").rglob("*.py")):
        paths.append(("native_source", str(path.relative_to(source)), "python_source", path))
    paths.append(("native_source", "example/assets/special_tokens.npz", "special_tokens",
                  source / "example/assets/special_tokens.npz"))
    return paths


def _full_file(path, deadline):
    path = Path(path).resolve()
    before = stat_guard(path.stat())
    digest, count = hashlib.sha256(), 0
    started = time.perf_counter()
    with path.open("rb") as handle:
        descriptor_before = stat_guard(os.fstat(handle.fileno()))
        for block in iter(lambda: handle.read(4 * 1024**2), b""):
            if time.monotonic() > deadline:
                raise TimeoutError("MAGI auxiliary provenance wall budget exhausted")
            digest.update(block)
            count += len(block)
        descriptor_after = stat_guard(os.fstat(handle.fileno()))
    after = stat_guard(path.stat())
    if before != descriptor_before or before != descriptor_after or before != after or count != before["size"]:
        raise ValueError(f"Auxiliary input changed during complete hashing: {path}")
    return {
        "path": str(path), "resolved_path": str(path), "bytes": count, "sha256": digest.hexdigest(),
        "path_stat_before": before, "descriptor_stat_before": descriptor_before,
        "descriptor_stat_after": descriptor_after, "path_stat_after": after, "final_manifest_stat": after,
        "full_content_hashed": True, "stable_during_hash": True, "stable_through_manifest_validation": True,
        "hash_seconds": time.perf_counter() - started,
    }


def capture_auxiliary(*, output, t5, vae, source, qualified_plan, max_seconds=1800):
    output = native._output(output)
    if output.exists():
        raise FileExistsError("Never overwrite a study auxiliary provenance manifest")
    if type(max_seconds) is not int or not 1 <= max_seconds <= 2700:
        raise ValueError("Auxiliary full hashing requires an explicit <=2700-second budget")
    reference = native._read(qualified_plan)
    claimed = reference.pop("identity")
    if native._identity(reference) != claimed:
        raise ValueError("Qualified native configuration reference identity changed")
    deadline, started = time.monotonic() + max_seconds, time.perf_counter()
    rows = []
    for component, relative, role, path in auxiliary_paths(t5, vae, source):
        row = {**_full_file(path, deadline), "component": component, "relative_path": relative, "role": role}
        rows.append(row)
        if component in ("t5", "vae") and row["bytes"] > 1024**3:
            print(json.dumps({"stage": "auxiliary_full_hash", "component": component,
                              "file": relative, "bytes": row["bytes"], "seconds": row["hash_seconds"]}), flush=True)
    for row in rows:
        _current_guard(row)
    vae_rows = {row["relative_path"]: row for row in rows if row["component"] == "vae"}
    if (vae_rows["config.json"]["sha256"] != reference["vae"]["config_sha256"]
            or vae_rows["diffusion_pytorch_model.safetensors"]["sha256"] != reference["vae"]["expected_checkpoint_sha256"]):
        raise ValueError("Freshly hashed VAE differs from qualified native VAE evidence")
    components = {
        component: sorted(
            [{key: row[key] for key in ("relative_path", "role", "bytes", "sha256")}
             for row in rows if row["component"] == component],
            key=lambda row: row["relative_path"],
        ) for component in ("t5", "vae", "native_source")
    }
    content = {
        "schema": "reelcache_magi_auxiliary_content_v1", "components": components,
        "qualified_native_config_sha256": canonical_digest(reference["config"]),
    }
    value = {
        "schema": "reelcache_magi_auxiliary_identity_v1", "status": "passed",
        "content_manifest": content, "content_manifest_sha256": canonical_digest(content), "files": rows,
        "qualified_native_reference": native._file_row(qualified_plan),
        "hashing": {"algorithm": "sha256", "mode": "sequential complete reads to EOF; once per study",
                    "bytes": sum(row["bytes"] for row in rows), "files": len(rows),
                    "buffer_bytes": 4 * 1024**2, "max_seconds": max_seconds,
                    "seconds": time.perf_counter() - started},
        "native_T5_execution_contract": {
            "dtype": "float32", "model_max_length": 800, "native_text_preprocessing": True,
            "special_tokens": [], "fresh_conditioning_tensor_identity": "captured separately per case after CPU T5 grant",
        },
        "not_covered": ["MAGI DiT/CausVid weights are in parent's separate weight_identity.json",
                        "fresh prompt embeddings, noise and common prefix are generated only under their stage grants"],
        "model_or_GPU_loaded": False, "weights_copied": False,
    }
    value["manifest_integrity_sha256"] = canonical_digest(value)
    native._write(output, value)
    return value


def verify_auxiliary(binding, *, t5, vae, source):
    manifest, artifact = _read_bound(binding, "reelcache_magi_auxiliary_identity_v1")
    expected = auxiliary_paths(t5, vae, source)
    rows = manifest["files"]
    paths = {(component, relative): str(path.resolve()) for component, relative, _, path in expected}
    if {(row["component"], row["relative_path"]) for row in rows} != set(paths):
        raise ValueError("Auxiliary identity must cover all referenced T5/VAE/source files")
    for component, content in manifest["content_manifest"]["components"].items():
        selected = [row for row in rows if row["component"] == component]
        _check_complete_rows(selected, content)
        for row in selected:
            if row["path"] != paths[(component, row["relative_path"])]:
                raise ValueError("Auxiliary inputs are not the study's verified original paths")
    return {
        "artifact": artifact, "content_manifest_sha256": manifest["content_manifest_sha256"],
        "components": manifest["content_manifest"]["components"],
        "per_case_verification": "immutable manifest integrity and exact original path/stat guards; large inputs not rehashed",
        "full_content_hashed_once_per_study": True,
    }
