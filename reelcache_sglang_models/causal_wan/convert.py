"""Checkpoint conversion and profile model directories for the causal-Wan profiles (torch + safetensors; no SGLang).

``native_to_diffusers_key`` maps the original Wan/CausVid/LongLive/RAVEN parameter names to the
diffusers ``WanTransformer3DModel`` names that SGLang's ``CausalWanTransformer3DModel`` loads through its own
``param_names_mapping``. The mapping is verified on real weights: converting the native
Wan2.1-T2V-1.3B base safetensors reproduces the Wan2.1-T2V-1.3B-Diffusers transformer bitwise
(``step12_profiles/conversion``). LongLive's PEFT LoRA (rank 256, alpha 256) is merged in float64.

``build_profile_dir`` writes ``<PROFILE_MODEL_ROOT>/<model_dir_name>/``: model_index.json naming the profile
pipeline, the converted ``transformer/`` (config.json + one safetensors file), symlinks to the bound
Diffusers ``text_encoder``/``tokenizer``/``vae``/``scheduler`` directories, and ``reelcache_profile.json``
(profile, fingerprint, source and output hashes). ``verify_profile_dir`` re-checks that binding at load time.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path

from .profiles import COMPONENT_DIRS, DIFFUSERS_SNAPSHOT, FileBinding, ModelProfile

MANIFEST_NAME = "reelcache_profile.json"
TRANSFORMER_FILE = "diffusion_pytorch_model.safetensors"
MANIFEST_SCHEMA = "reelcache.causal_wan.profile_dir.v1"
SMALL_FILE_BYTES = 64 << 20  # always re-hashed; larger HF LFS files are bound through their blob name

_BLOCK_RULES = (
    (r"modulation", "scale_shift_table"),
    (r"self_attn\.q\.(weight|bias)", r"attn1.to_q.\1"),
    (r"self_attn\.k\.(weight|bias)", r"attn1.to_k.\1"),
    (r"self_attn\.v\.(weight|bias)", r"attn1.to_v.\1"),
    (r"self_attn\.o\.(weight|bias)", r"attn1.to_out.0.\1"),
    (r"self_attn\.norm_q\.weight", "attn1.norm_q.weight"),
    (r"self_attn\.norm_k\.weight", "attn1.norm_k.weight"),
    (r"cross_attn\.q\.(weight|bias)", r"attn2.to_q.\1"),
    (r"cross_attn\.k\.(weight|bias)", r"attn2.to_k.\1"),
    (r"cross_attn\.v\.(weight|bias)", r"attn2.to_v.\1"),
    (r"cross_attn\.o\.(weight|bias)", r"attn2.to_out.0.\1"),
    (r"cross_attn\.norm_q\.weight", "attn2.norm_q.weight"),
    (r"cross_attn\.norm_k\.weight", "attn2.norm_k.weight"),
    (r"norm3\.(weight|bias)", r"norm2.\1"),
    (r"ffn\.0\.(weight|bias)", r"ffn.net.0.proj.\1"),
    (r"ffn\.2\.(weight|bias)", r"ffn.net.2.\1"),
)
_TOP_RULES = (
    (r"patch_embedding\.(weight|bias)", r"patch_embedding.\1"),
    (r"text_embedding\.0\.(weight|bias)", r"condition_embedder.text_embedder.linear_1.\1"),
    (r"text_embedding\.2\.(weight|bias)", r"condition_embedder.text_embedder.linear_2.\1"),
    (r"time_embedding\.0\.(weight|bias)", r"condition_embedder.time_embedder.linear_1.\1"),
    (r"time_embedding\.2\.(weight|bias)", r"condition_embedder.time_embedder.linear_2.\1"),
    (r"time_projection\.1\.(weight|bias)", r"condition_embedder.time_proj.\1"),
    (r"head\.head\.(weight|bias)", r"proj_out.\1"),
    (r"head\.modulation", "scale_shift_table"),
)
LORA_TARGETS = ("self_attn.q", "self_attn.k", "self_attn.v", "self_attn.o",
                "cross_attn.q", "cross_attn.k", "cross_attn.v", "cross_attn.o", "ffn.0", "ffn.2")

# Wan2.1-T2V-1.3B transformer config (equal to the Diffusers snapshot's transformer/config.json except _class_name,
# which names the profile's DiT architecture; see transformer_config_for).
TRANSFORMER_CONFIG = {
    "_class_name": "CausalWanTransformer3DModel",
    "_diffusers_version": "0.33.0.dev0",
    "added_kv_proj_dim": None, "attention_head_dim": 128, "cross_attn_norm": True, "eps": 1e-06, "ffn_dim": 8960,
    "freq_dim": 256, "image_dim": None, "in_channels": 16, "num_attention_heads": 12, "num_layers": 30,
    "out_channels": 16, "patch_size": [1, 2, 2], "qk_norm": "rms_norm_across_heads", "rope_max_seq_len": 1024,
    "text_dim": 4096,
}


def transformer_config_for(profile: ModelProfile) -> dict:
    return {**TRANSFORMER_CONFIG, "_class_name": profile.sglang.dit_class}


def native_to_diffusers_key(key: str) -> str:
    """Original Wan parameter name (no ``model.`` prefix) -> diffusers WanTransformer3DModel name."""
    m = re.fullmatch(r"blocks\.(\d+)\.(.+)", key)
    if m:
        idx, rest = m.groups()
        for pattern, repl in _BLOCK_RULES:
            if re.fullmatch(pattern, rest):
                return f"blocks.{idx}." + re.sub(pattern, repl, rest)
        raise KeyError(f"unmapped block parameter {key}")
    for pattern, repl in _TOP_RULES:
        if re.fullmatch(pattern, key):
            return re.sub(pattern, repl, key)
    raise KeyError(f"unmapped parameter {key}")


def strip_prefix(state: dict, prefix: str) -> dict:
    if not prefix:
        return dict(state)
    bad = [k for k in state if not k.startswith(prefix)]
    if bad:
        raise KeyError(f"{len(bad)} keys lack prefix {prefix!r}: {bad[:3]}")
    return {k[len(prefix):]: v for k, v in state.items()}


def merge_lora(base: dict, lora: dict, *, prefix: str, scale: float, num_layers: int) -> tuple[dict, dict]:
    """W <- W + scale * B @ A for every LoRA target (float64 accumulate, original dtype out)."""
    import torch

    lora = strip_prefix(lora, prefix)
    expected = {f"blocks.{i}.{t}.lora_{ab}.weight" for i in range(num_layers) for t in LORA_TARGETS for ab in "AB"}
    if set(lora) != expected:
        missing, extra = sorted(expected - set(lora)), sorted(set(lora) - expected)
        raise KeyError(f"LoRA keys differ: missing {missing[:3]} ({len(missing)}), extra {extra[:3]} ({len(extra)})")
    merged = dict(base)
    max_delta, rel = 0.0, 0.0
    for i in range(num_layers):
        for t in LORA_TARGETS:
            name = f"blocks.{i}.{t}.weight"
            a = lora[f"blocks.{i}.{t}.lora_A.weight"].to(torch.float64)
            b = lora[f"blocks.{i}.{t}.lora_B.weight"].to(torch.float64)
            w = base[name]
            if b.shape[0] != w.shape[0] or a.shape[1] != w.shape[1] or a.shape[0] != b.shape[1]:
                raise ValueError(f"LoRA shape mismatch at {name}: A {tuple(a.shape)} B {tuple(b.shape)} W {tuple(w.shape)}")
            delta = scale * (b @ a)
            merged[name] = (w.to(torch.float64) + delta).to(w.dtype)
            max_delta = max(max_delta, float(delta.abs().max()))
            rel = max(rel, float(delta.norm() / w.to(torch.float64).norm()))
    return merged, {"targets": num_layers * len(LORA_TARGETS), "scale": scale, "max_abs_delta": max_delta,
                    "max_relative_frobenius_delta": rel}


def convert_state_dict(native: dict) -> dict:
    out = {}
    for k, v in native.items():
        nk = native_to_diffusers_key(k)
        if nk in out:
            raise KeyError(f"duplicate target {nk}")
        out[nk] = v
    return out


def sha256_file(path: str | Path, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def verify_binding(binding: FileBinding, *, full_hash: bool = True) -> dict:
    p = Path(binding.path)
    if not p.is_file():
        raise FileNotFoundError(f"{binding.role}: {binding.path} missing")
    size = p.stat().st_size
    if size != binding.bytes:
        raise ValueError(f"{binding.role}: size {size} != bound {binding.bytes}")
    rec = {"role": binding.role, "path": binding.path, "bytes": size}
    target = p.resolve()
    if full_hash or size < SMALL_FILE_BYTES:
        digest = sha256_file(p)
        if digest != binding.sha256:
            raise ValueError(f"{binding.role}: sha256 {digest} != bound {binding.sha256}")
        rec["sha256_verified"] = digest
    elif len(target.name) == 64 and target.parent.name == "blobs":
        if target.name != binding.sha256:
            raise ValueError(f"{binding.role}: HF blob {target.name} != bound {binding.sha256}")
        rec["sha256_from_lfs_blob_name"] = target.name
    else:
        rec["sha256_bound_not_rehashed"] = binding.sha256
    return rec


def load_native_generator(profile: ModelProfile) -> tuple[dict, dict]:
    """Load (and hash-verify) the native generator state, merge LoRA when the profile has one."""
    import torch

    info = {"sources": []}
    role = profile.conversion.source_roles[0]
    gen = profile.file(role)
    info["sources"].append(verify_binding(gen))
    obj = torch.load(gen.path, map_location="cpu", mmap=True, weights_only=True)
    state = obj[gen.state_dict_key] if gen.state_dict_key else obj
    state = strip_prefix(state, gen.key_prefix or "")
    state = {k: v for k, v in state.items()}
    if len(state) != profile.conversion.expected_tensors:
        raise ValueError(f"{profile.profile_id}: {len(state)} tensors, expected {profile.conversion.expected_tensors}")
    lora = profile.conversion.lora
    if lora is not None:
        lb = profile.file(profile.conversion.source_roles[1])
        info["sources"].append(verify_binding(lb))
        lobj = torch.load(lb.path, map_location="cpu", mmap=True, weights_only=True)
        state, info["lora_merge"] = merge_lora(state, lobj[lora.state_dict_key], prefix=lora.key_prefix,
                                               scale=lora.scale, num_layers=int(profile.backbone["num_layers"]))
    dtypes = sorted({str(v.dtype) for v in state.values()})
    info["source_dtypes"] = dtypes
    return state, info


def model_index_for(profile: ModelProfile) -> dict:
    return {
        "_class_name": profile.sglang.pipeline_name,
        "_diffusers_version": "0.33.0.dev0",
        "scheduler": ["diffusers", "UniPCMultistepScheduler"],
        "text_encoder": ["transformers", "UMT5EncoderModel"],
        "tokenizer": ["transformers", "T5TokenizerFast"],
        "transformer": ["diffusers", profile.sglang.dit_class],
        "vae": ["diffusers", "AutoencoderKLWan"],
    }


def build_profile_dir(profile: ModelProfile, *, root: str | Path | None = None, overwrite: bool = False) -> dict:
    """Convert the native generator and write the profile model directory; returns the manifest."""
    import torch
    from safetensors.torch import save_file

    t0 = time.perf_counter()
    target = Path(root) / profile.sglang.model_dir_name if root else Path(profile.sglang.model_path)
    if target.exists() and not overwrite:
        raise FileExistsError(f"{target} exists")
    target.mkdir(parents=True, exist_ok=True)
    native, info = load_native_generator(profile)
    converted = convert_state_dict(native)
    out_dtype = getattr(torch, profile.conversion.output_dtype)
    converted = {k: v.to(out_dtype).contiguous() for k, v in sorted(converted.items())}
    tdir = target / "transformer"
    tdir.mkdir(exist_ok=True)
    wfile = tdir / TRANSFORMER_FILE
    tmp = tdir / (TRANSFORMER_FILE + ".partial")
    save_file(converted, str(tmp), metadata={"format": "pt", "reelcache_profile": profile.profile_id,
                                             "reelcache_binding_fingerprint": profile.binding_fingerprint()})
    os.replace(tmp, wfile)
    (tdir / "config.json").write_text(json.dumps(transformer_config_for(profile), indent=2) + "\n")
    for comp in COMPONENT_DIRS:
        link = target / comp
        if link.is_symlink() or link.exists():
            if link.is_symlink() and os.readlink(link) == f"{DIFFUSERS_SNAPSHOT}/{comp}":
                continue
            raise FileExistsError(f"{link} exists and is not the bound component link")
        link.symlink_to(f"{DIFFUSERS_SNAPSHOT}/{comp}")
    (target / "model_index.json").write_text(json.dumps(model_index_for(profile), indent=2) + "\n")
    manifest = {
        "schema": MANIFEST_SCHEMA, "profile_id": profile.profile_id,
        "binding_fingerprint": profile.binding_fingerprint(), "profile_fingerprint_at_build": profile.fingerprint(),
        "profile_at_build": profile.to_dict(), "model_path": str(target),
        "transformer": {"file": f"transformer/{TRANSFORMER_FILE}", "bytes": wfile.stat().st_size,
                        "sha256": sha256_file(wfile), "tensors": len(converted), "dtype": profile.conversion.output_dtype,
                        "config_sha256": sha256_file(tdir / "config.json")},
        "conversion": {**info, "kind": profile.conversion.kind, "seconds": time.perf_counter() - t0,
                       "torch": torch.__version__},
        "components": {c: f"{DIFFUSERS_SNAPSHOT}/{c}" for c in COMPONENT_DIRS},
        "model_index_sha256": sha256_file(target / "model_index.json"),
    }
    (target / MANIFEST_NAME).write_text(json.dumps(manifest, indent=1) + "\n")
    return manifest


def refresh_profile_metadata(profile: ModelProfile, *, root: str | Path | None = None) -> dict:
    """Rewrite config/model_index/manifest of an existing directory whose converted weights are unchanged.

    Used when only metadata of the binding changed (e.g. the DiT architecture name): the weights file must
    still hash to the old manifest's SHA256 and the old manifest's sources must be the profile's sources.
    """
    target = Path(root) / profile.sglang.model_dir_name if root else Path(profile.sglang.model_path)
    old = json.loads((target / MANIFEST_NAME).read_text())
    wfile = target / old["transformer"]["file"]
    digest = sha256_file(wfile)
    if digest != old["transformer"]["sha256"]:
        raise ValueError(f"{wfile} no longer matches its manifest; rebuild with build_profile_dir")
    srcs = {s["role"]: s.get("sha256_verified") for s in old["conversion"]["sources"]}
    for role in profile.conversion.source_roles:
        if srcs.get(role) != profile.file(role).sha256:
            raise ValueError(f"{role} source changed since conversion; rebuild with build_profile_dir")
    tdir = target / "transformer"
    (tdir / "config.json").write_text(json.dumps(transformer_config_for(profile), indent=2) + "\n")
    (target / "model_index.json").write_text(json.dumps(model_index_for(profile), indent=2) + "\n")
    manifest = dict(old)
    manifest.update({"binding_fingerprint": profile.binding_fingerprint(),
                     "profile_fingerprint_at_build": profile.fingerprint(), "profile_at_build": profile.to_dict(),
                     "model_index_sha256": sha256_file(target / "model_index.json")})
    manifest["transformer"] = {**old["transformer"], "config_sha256": sha256_file(tdir / "config.json")}
    manifest.setdefault("metadata_refreshes", []).append(
        {"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "previous_binding_fingerprint": old["binding_fingerprint"],
         "weights_sha256_reverified": digest})
    (target / MANIFEST_NAME).write_text(json.dumps(manifest, indent=1) + "\n")
    return manifest


def verify_profile_dir(profile: ModelProfile, model_path: str | Path, *, full_hash: bool = True) -> dict:
    """Load-time binding check of a profile directory; raises on any mismatch (no silent fallback)."""
    root = Path(model_path)
    mpath = root / MANIFEST_NAME
    if not mpath.is_file():
        raise FileNotFoundError(f"{root} has no {MANIFEST_NAME}; build it with build_profile_dir()")
    manifest = json.loads(mpath.read_text())
    if manifest.get("schema") != MANIFEST_SCHEMA or manifest.get("profile_id") != profile.profile_id:
        raise ValueError(f"{mpath} is not a {profile.profile_id} profile manifest")
    if manifest["binding_fingerprint"] != profile.binding_fingerprint():
        raise ValueError(f"{root} was built for binding {manifest['binding_fingerprint'][:12]}, code has "
                         f"{profile.binding_fingerprint()[:12]}; rebuild the directory")
    report = {"model_path": str(root), "binding_fingerprint": manifest["binding_fingerprint"],
              "profile_fingerprint": profile.fingerprint(), "components": []}
    if json.loads((root / "model_index.json").read_text()) != model_index_for(profile):
        raise ValueError(f"{root}/model_index.json does not name {profile.sglang.pipeline_name}")
    t = manifest["transformer"]
    wfile = root / t["file"]
    if wfile.stat().st_size != t["bytes"]:
        raise ValueError(f"{wfile} size changed")
    if full_hash:
        digest = sha256_file(wfile)
        if digest != t["sha256"]:
            raise ValueError(f"{wfile} sha256 {digest} != manifest {t['sha256']}")
        report["transformer_sha256_verified"] = digest
    if json.loads((root / "transformer/config.json").read_text()) != transformer_config_for(profile):
        raise ValueError("transformer/config.json differs from the profile's Wan2.1-T2V-1.3B causal config")
    for comp in COMPONENT_DIRS:
        link = root / comp
        if not link.is_symlink() or os.readlink(link) != f"{DIFFUSERS_SNAPSHOT}/{comp}":
            raise ValueError(f"{link} is not the bound Diffusers component {DIFFUSERS_SNAPSHOT}/{comp}")
    for b in profile.components:
        report["components"].append(verify_binding(b, full_hash=False))
    return report
