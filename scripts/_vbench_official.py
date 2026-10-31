"""Local-only loading of pinned official VBench implementations.

No historical harness is imported. DINO's architecture and checkpoint are loaded
explicitly to avoid torch.hub's implicit download; its official scoring function
is unchanged. Other dimensions use their official compute_* entry points.
"""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import importlib
import importlib.metadata
import importlib.util
import os
from pathlib import Path
import socket
import subprocess
import sys
from unittest import mock

try:
    from ._vbench_protocol import (
        DIMENSIONS, REVISION, SCHEMAS, dimensions_arg, require, sha256_file,
    )
except ImportError:
    from _vbench_protocol import (
        DIMENSIONS, REVISION, SCHEMAS, dimensions_arg, require, sha256_file,
    )


# Existing local locations are inventory hints, not installation destinations.
ASSETS = {
    "dino": {
        "relative_path": ".torchcache/hub/checkpoints/dino_vitbase16_pretrain.pth",
        "cache_path": "dino_model/dino_vitbase16_pretrain.pth",
        "url": "https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth",
        "sha256": "bf34ad0f424b9029b593e8dc3ed553bf26e88bcba0d32bf3e62a6209cb64c85e",
    },
    "clip_b": {
        "relative_path": "cache/clip/ViT-B-32.pt", "cache_path": "clip_model/ViT-B-32.pt",
        "url": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
        "sha256": "40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af",
    },
    "clip_l": {
        "relative_path": "cache/clip/ViT-L-14.pt", "cache_path": "clip_model/ViT-L-14.pt",
        "url": "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt",
        "sha256": "b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836",
    },
    "amt": {
        "relative_path": ".vbench_cache/amt_model/amt-s.pth", "cache_path": "amt_model/amt-s.pth",
        "url": "https://huggingface.co/lalala125/AMT/resolve/main/amt-s.pth",
        "sha256": "07e7e03405c213fe4405678db9b42d05671e48c56faf6a441b99bb0047d3cf77",
    },
    "raft": {
        "relative_path": ".vbench_cache/raft_model/models/raft-things.pth",
        "cache_path": "raft_model/models/raft-things.pth",
        "url": "https://dl.dropboxusercontent.com/s/4j4z58wuv8o0mfz/models.zip",
        "sha256": "fcfa4125d6418f4de95d84aec20a3c5f4e205101715a79f193243c186ac9a7e1",
    },
    "aesthetic_head": {
        "relative_path": ".vbench_cache/aesthetic_model/emb_reader/sa_0_4_vit_l_14_linear.pth",
        "cache_path": "aesthetic_model/emb_reader/sa_0_4_vit_l_14_linear.pth",
        "url": "https://raw.githubusercontent.com/LAION-AI/aesthetic-predictor/main/sa_0_4_vit_l_14_linear.pth",
        "sha256": "2cd4e60f4f24ae3bcd57b847b13c1f3ba27edc28cc1a7f9ce74ee9f421243cba",
    },
    "musiq": {
        "relative_path": ".vbench_cache/pyiqa_model/musiq_spaq_ckpt-358bb6af.pth",
        "cache_path": "pyiqa_model/musiq_spaq_ckpt-358bb6af.pth",
        "url": "https://github.com/chaofengc/IQA-PyTorch/releases/download/v0.1-weights/musiq_spaq_ckpt-358bb6af.pth",
        "sha256": "358bb6af275e28ea56821d44fb55c6cb83645db11f394d3ad65b2d149965ab50",
    },
    "umt": {
        "relative_path": ".vbench_cache/umt_model/l16_ptk710_ftk710_ftk400_f16_res224.pth",
        "cache_path": "umt_model/l16_ptk710_ftk710_ftk400_f16_res224.pth",
        "url": "https://huggingface.co/OpenGVLab/VBench_Used_Models/resolve/main/l16_ptk710_ftk710_ftk400_f16_res224.pth",
    },
    "grit": {
        "relative_path": ".vbench_cache/grit_model/grit_b_densecap_objectdet.pth",
        "cache_path": "grit_model/grit_b_densecap_objectdet.pth",
        "url": "https://huggingface.co/OpenGVLab/VBench_Used_Models/resolve/main/grit_b_densecap_objectdet.pth",
    },
    "tag2text": {
        "relative_path": ".vbench_cache/caption_model/tag2text_swin_14m.pth",
        "cache_path": "caption_model/tag2text_swin_14m.pth",
        "url": "https://huggingface.co/spaces/xinyu1205/recognize-anything/resolve/main/tag2text_swin_14m.pth",
    },
    "viclip": {
        "relative_path": ".vbench_cache/ViCLIP/ViClip-InternVid-10M-FLT.pth",
        "cache_path": "ViCLIP/ViClip-InternVid-10M-FLT.pth",
        "url": "https://huggingface.co/OpenGVLab/VBench_Used_Models/resolve/main/ViClip-InternVid-10M-FLT.pth",
    },
    "viclip_bpe": {
        "relative_path": ".vbench_cache/ViCLIP/bpe_simple_vocab_16e6.txt.gz",
        "cache_path": "ViCLIP/bpe_simple_vocab_16e6.txt.gz",
        "url": "https://raw.githubusercontent.com/openai/CLIP/main/clip/bpe_simple_vocab_16e6.txt.gz",
    },
}
REQUIRED_ASSETS = {
    "subject_consistency": ("dino",), "background_consistency": ("clip_b",),
    "temporal_flickering": ("raft",), "motion_smoothness": ("amt",),
    "dynamic_degree": ("raft",), "aesthetic_quality": ("clip_l", "aesthetic_head"),
    "imaging_quality": ("musiq",), "object_class": ("grit",),
    "multiple_objects": ("grit",), "color": ("grit",),
    "spatial_relationship": ("grit",), "human_action": ("umt",),
    "scene": ("tag2text",), "appearance_style": ("clip_b",),
    "temporal_style": ("viclip", "viclip_bpe"),
    "overall_consistency": ("viclip", "viclip_bpe"),
}
COMMON_IMPORTS = ("torch", "torchvision", "numpy", "cv2", "PIL", "decord", "tqdm")
EXTRA_IMPORTS = {
    "background_consistency": ("clip",), "aesthetic_quality": ("clip",),
    "imaging_quality": ("pyiqa",), "motion_smoothness": ("omegaconf",),
    "dynamic_degree": ("easydict",), "human_action": ("timm", "scipy"),
    "object_class": ("detectron2", "fairscale", "fvcore", "lvis"),
    "multiple_objects": ("detectron2", "fairscale", "fvcore", "lvis"),
    "color": ("detectron2", "fairscale", "fvcore", "lvis"),
    "spatial_relationship": ("detectron2", "fairscale", "fvcore", "lvis"),
    "scene": ("transformers", "fairscale", "timm"),
    "appearance_style": ("clip",),
    "overall_consistency": ("clip", "timm", "ftfy", "regex"),
    "temporal_style": ("clip", "timm", "ftfy", "regex"),
}


def version_inventory():
    result = {"python": sys.version}
    for name in ("torch", "torchvision", "numpy", "scipy", "scikit-image", "pyiqa",
                 "openai-clip", "opencv-python", "Pillow", "decord", "easydict",
                 "transformers", "timm", "imageio-ffmpeg"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def verify_checkout(root):
    root = Path(root).resolve()
    command = ["git", "--no-pager", "-C", str(root)]
    head = subprocess.check_output(command + ["rev-parse", "HEAD"], text=True, timeout=120).strip()
    require(head == REVISION, f"Expected VBench {REVISION}, found {head}")
    dirty = subprocess.check_output(command + ["diff", "--name-only", "HEAD"],
                                    text=True, timeout=120).strip()
    require(not dirty, f"Official tracked sources were modified: {dirty}")
    files = sorted((root / "vbench").rglob("*.py"))
    files += [root / "static_filter.py", root / "pretrained/amt_model/AMT-S.yaml",
              root / "scripts/constant.py", root / "vbench/VBench_full_info.json"]
    return {"root": str(root), "commit": head,
            "source_sha256": {str(p.relative_to(root)): sha256_file(p) for p in files}}


def inventory_assets(vbench_root, weight_root, *, hash_weights=False, dependency_paths=()):
    source = verify_checkout(vbench_root)
    weight_root = Path(weight_root).resolve()
    assets = {}
    for name, declaration in ASSETS.items():
        path = weight_root / declaration["relative_path"]
        entry = {
            "path": str(path), "source_url": declaration["url"],
            "exists": path.is_file(), "bytes": path.stat().st_size if path.is_file() else None,
            "expected_sha256": declaration.get("sha256"), "verified_sha256": None,
        }
        if hash_weights and entry["exists"]:
            entry["verified_sha256"] = sha256_file(path)
            if entry["expected_sha256"] is not None:
                require(entry["verified_sha256"] == entry["expected_sha256"],
                        f"Weight hash mismatch: {name}")
            else:
                entry["expected_sha256"] = entry["verified_sha256"]
        assets[name] = entry
    dino = weight_root / "cache/torch/hub/facebookresearch_dino_main"
    dino_sources = {p.name: sha256_file(p) for p in (
        dino / "hubconf.py", dino / "vision_transformer.py", dino / "utils.py") if p.is_file()}
    return {
        "schema": SCHEMAS["assets"], "official_source": source, "assets": assets,
        "dino_source": {"path": str(dino), "sha256": dino_sources},
        "dependency_paths": [str(Path(p).resolve()) for p in dependency_paths],
        "versions": version_inventory(), "models_executed": False,
        "downloads_performed": False,
        "known_present_weight_bytes": sum(a["bytes"] or 0 for a in assets.values()),
        "missing_model_size_policy": "Resolve authoritative sizes and approve storage before acquisition; never download automatically.",
    }


def preflight_assets(manifest, dimensions):
    dimensions_arg(dimensions)
    require(manifest.get("schema") == SCHEMAS["assets"], "Wrong asset schema")
    result = {}
    for dimension in dimensions:
        blockers = []
        for name in REQUIRED_ASSETS[dimension]:
            asset = manifest["assets"].get(name)
            if asset is None or not Path(asset["path"]).is_file():
                blockers.append(f"missing asset: {name}")
            elif not (asset.get("expected_sha256") or asset.get("verified_sha256")):
                blockers.append(f"unbound asset SHA256: {name}")
        if dimension == "subject_consistency":
            dino = manifest.get("dino_source", {})
            if not {"hubconf.py", "vision_transformer.py", "utils.py"} <= set(dino.get("sha256", {})):
                blockers.append("missing DINO source bindings")
        for name in COMMON_IMPORTS + EXTRA_IMPORTS.get(dimension, ()):
            try:
                if importlib.util.find_spec(name) is None:
                    blockers.append(f"missing import: {name}")
            except (ImportError, ValueError):
                blockers.append(f"unavailable import: {name}")
        if dimension == "scene":
            # The pinned Tag2Text uses the old transformers API and also needs
            # a locally populated bert-base-uncased tokenizer cache.
            version = version_inventory()["transformers"]
            if version != "4.33.2":
                blockers.append(f"Tag2Text requires its isolated transformers==4.33.2 environment, found {version}")
            blockers.append("Tag2Text tokenizer/config cache needs explicit offline qualification")
        result[dimension] = {
            "status": "blocked" if blockers else "cpu_prepared_not_gpu_validated",
            "blockers": blockers, "required_assets": list(REQUIRED_ASSETS[dimension]),
        }
    return result


def verify_required_assets(manifest, dimensions):
    require(manifest["official_source"] == verify_checkout(manifest["official_source"]["root"]),
            "Official source hashes changed since inventory")
    required = sorted({name for d in dimensions for name in REQUIRED_ASSETS[d]})
    verified = {}
    for name in required:
        record = manifest["assets"][name]
        path = Path(record["path"])
        expected = record.get("expected_sha256") or record.get("verified_sha256")
        require(expected is not None and path.is_file(), f"Missing/unbound asset {name}")
        actual = sha256_file(path)
        require(actual == expected, f"Asset SHA256 mismatch: {name}")
        verified[name] = {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": actual}
    if "subject_consistency" in dimensions:
        source = manifest["dino_source"]
        for name, expected in source["sha256"].items():
            require(sha256_file(Path(source["path"]) / name) == expected, f"DINO source changed: {name}")
    return verified


@contextmanager
def dependency_paths(paths):
    previous = sys.path[:]
    try:
        for path in reversed(paths):
            path = Path(path).resolve()
            require(path.is_dir(), f"Dependency path is absent: {path}")
            sys.path.insert(0, str(path))
        yield
    finally:
        sys.path[:] = previous


@contextmanager
def no_network_downloads():
    def refused(*args, **kwargs):
        raise RuntimeError("Evaluation is offline; missing assets must be inventoried and acquired separately")

    original_run = subprocess.run
    original_popen = subprocess.Popen

    def checked_call(original, command, *args, **kwargs):
        require(not kwargs.get("shell"), "Metric subprocess shell execution is disabled")
        executable = command[0] if isinstance(command, (tuple, list)) else command.split()[0]
        if Path(executable).name in ("wget", "curl", "git", "pip", "pip3"):
            return refused()
        return original(command, *args, **kwargs)

    with mock.patch.object(socket.socket, "connect", refused), \
            mock.patch.object(socket, "create_connection", refused), \
            mock.patch.object(subprocess, "run",
                              lambda cmd, *a, **k: checked_call(original_run, cmd, *a, **k)), \
            mock.patch.object(subprocess, "Popen",
                              lambda cmd, *a, **k: checked_call(original_popen, cmd, *a, **k)):
        yield


@contextmanager
def official_environment(manifest, cache_root, dimensions):
    root = Path(manifest["official_source"]["root"]).resolve()
    cache_root = Path(cache_root).resolve()
    require(not cache_root.exists(), "Use a new owned VBench cache-binding directory")
    cache_root.mkdir(parents=True)
    for name in sorted({n for d in dimensions for n in REQUIRED_ASSETS[d]}):
        source = Path(manifest["assets"][name]["path"]).resolve()
        target = cache_root / ASSETS[name]["cache_path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(source)
    keys = ("VBENCH_CACHE_DIR", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
    previous = {key: os.environ.get(key) for key in keys}
    try:
        os.environ.update(VBENCH_CACHE_DIR=str(cache_root), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        with dependency_paths([str(root)] + manifest.get("dependency_paths", [])), no_network_downloads():
            module = importlib.import_module("vbench")
            require(Path(module.__file__).resolve().is_relative_to(root), "Another VBench installation is shadowing the pinned source")
            require(Path(importlib.import_module("vbench.utils").CACHE_DIR).resolve() == cache_root,
                    "VBench was imported with a different cache binding; use a fresh worker process")
            yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def compute_dimension(dimension, info_path, manifest, device):
    import torch

    module = importlib.import_module(f"vbench.{dimension}")
    root = Path(manifest["official_source"]["root"]).resolve()
    require(Path(module.__file__).resolve().is_relative_to(root), "Metric module source mismatch")
    paths = {name: record["path"] for name, record in manifest["assets"].items()}
    if dimension == "subject_consistency":
        model = torch.hub.load(manifest["dino_source"]["path"], "dino_vitb16",
                               source="local", pretrained=False)
        model.load_state_dict(torch.load(paths["dino"], map_location="cpu", weights_only=True), strict=True)
        model.eval().to(device)
        videos, _ = importlib.import_module("vbench.utils").load_dimension_info(
            str(info_path), dimension, "en")
        return module.subject_consistency(model, videos, device, False)
    submodules = {
        "background_consistency": [paths["clip_b"], False],
        "temporal_flickering": [],
        "motion_smoothness": {"config": str(root / "pretrained/amt_model/AMT-S.yaml"), "ckpt": paths["amt"]},
        "dynamic_degree": {"model": paths["raft"]},
        "aesthetic_quality": [paths["clip_l"], str(Path(paths["aesthetic_head"]).parent)],
        "imaging_quality": {"model_path": paths["musiq"]},
        "human_action": [paths["umt"]],
        "scene": {"pretrained": paths["tag2text"], "image_size": 384, "vit": "swin_b"},
        "appearance_style": {"name": paths["clip_b"]},
        "temporal_style": {"pretrain": paths["viclip"]},
        "overall_consistency": {"pretrain": paths["viclip"]},
    }
    for name in ("object_class", "multiple_objects", "color", "spatial_relationship"):
        submodules[name] = {"model_weight": paths["grit"]}
    return getattr(module, f"compute_{dimension}")(
        str(info_path), device, submodules[dimension], imaging_quality_preprocessing_mode="longer")


def static_filter(videos, manifest, device):
    require(device == "cuda", "Pinned official StaticFilter uses CUDA; CPU mode must omit temporal_flickering")
    from types import SimpleNamespace

    path = Path(manifest["official_source"]["root"]) / "static_filter.py"
    spec = importlib.util.spec_from_file_location("_reelcache_official_static_filter", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    args = SimpleNamespace(model=manifest["assets"]["raft"]["path"],
                           small=False, mixed_precision=False, alternate_corr=False)
    model = module.StaticFilter(args, device)
    return {str(path): bool(model.infer(str(path))) for path in videos}
