"""Exact weight binding for the LongCat-Video port (path + size + SHA256).

The hashes are the Step 2 feasibility receipt (item 9,
``artifacts/sglang_integration/source_bindings/feasibility/item9_weights/weights_sha256_table.csv``),
which also checked each value against the HF LFS ETag / git blob SHA of revision ``03b5552``.
``refinement_lora.safetensors`` belongs to the 480p->720p refinement path and is not bound.
"""

from __future__ import annotations

import concurrent.futures as cf
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

from .spec import HF_REPO_ID, HF_REVISION, LONGCAT_VIDEO_COMMIT

#: Placeholder default; pass the checkpoint directory explicitly or edit this for your machine.
DEFAULT_CHECKPOINT_DIR = "/models/LongCat-Video"
VERIFY_ENV = "REELCACHE_LONGCAT_VIDEO_WEIGHT_VERIFY"
SMALL_FILE_BYTES = 1 << 20


class WeightBindingError(RuntimeError):
    """The checkpoint directory does not match the bound LongCat-Video weights."""


@dataclass(frozen=True)
class BoundFile:
    relpath: str
    role: str
    size: int
    sha256: str


LONGCAT_VIDEO_FILES: tuple[BoundFile, ...] = (
    BoundFile("config.json", "repo_metadata", 37, "0121461c0aae6083e23217414ddfdd78c4ff4d088838e9c88c1f483b8c6cefd1"),
    BoundFile("model_index.json", "repo_metadata", 37, "0121461c0aae6083e23217414ddfdd78c4ff4d088838e9c88c1f483b8c6cefd1"),
    BoundFile("dit/config.json", "dit", 593, "89860487de2587e230ac1b0413ba94dd16447b0688d9bdb8fabeb8d5e6228659"),
    BoundFile("dit/diffusion_pytorch_model.safetensors.index.json", "dit", 93772,
              "881a29e97a204be3c77f0ca97db78cc71a726357fc8a39d18d4da4b50bbb40d2"),
    BoundFile("dit/diffusion_pytorch_model-00001-of-00006.safetensors", "dit", 9932990440,
              "4a64ebfe319eaea7144a3e38acaf79e031736c254b53cd3d59643d5673444b7c"),
    BoundFile("dit/diffusion_pytorch_model-00002-of-00006.safetensors", "dit", 9976452384,
              "711a92e5bb38be84508869d62f451e8a1cb14cd903faf34b8291b1b3e299bab7"),
    BoundFile("dit/diffusion_pytorch_model-00003-of-00006.safetensors", "dit", 9955430184,
              "977b90558908fa725ff66995ac72993e4092e9c040092893766acd60ad8e086f"),
    BoundFile("dit/diffusion_pytorch_model-00004-of-00006.safetensors", "dit", 9821179488,
              "902f38deacdb75d97639968c98a2b520d97d268cfde9698c55810ed26e9170e2"),
    BoundFile("dit/diffusion_pytorch_model-00005-of-00006.safetensors", "dit", 9925989240,
              "f3e7ef21120757e658d501fa70fa8db2af3ff95945de72c4904fbe5ccbe0050b"),
    BoundFile("dit/diffusion_pytorch_model-00006-of-00006.safetensors", "dit", 4712351112,
              "dac297b544a030ce961648994ea791f58bab0e3d722092ac8c11a1475b2f4989"),
    BoundFile("lora/cfg_step_lora.safetensors", "distill_lora_cfg_step", 2523077984,
              "1a903b49d62571e343258211300668f4b8d375842388092781bd11e364ae65a9"),
    BoundFile("text_encoder/config.json", "text_encoder", 854,
              "4087b6192155a4643f6d29fd326c4610103130674011ef4d0a53f8bce5de967d"),
    BoundFile("text_encoder/model.safetensors.index.json", "text_encoder", 22476,
              "8af791f24a6447aa30c95786f40adf23d3c9df97470d2f36159c8f1113f120b4"),
    BoundFile("text_encoder/model-00001-of-00005.safetensors", "text_encoder", 4972389712,
              "c0ef3a140898e228a3520c9adec60743d2e8e5b3d229651bb37f1a3921919f99"),
    BoundFile("text_encoder/model-00002-of-00005.safetensors", "text_encoder", 4899225672,
              "481c7b2b39771c44df6dd8d13ee12ed072d731b4a650bd092885d4d52db229ad"),
    BoundFile("text_encoder/model-00003-of-00005.safetensors", "text_encoder", 4966309504,
              "f93148bcc04052a169e1e49bfcf6125df6cf9bf243cb9c627da75266cf8e35c3"),
    BoundFile("text_encoder/model-00004-of-00005.safetensors", "text_encoder", 4999880704,
              "a451792c739c05bca4606190cc2dd16731411bac03b4cf6aacc5767321f857c9"),
    BoundFile("text_encoder/model-00005-of-00005.safetensors", "text_encoder", 2885866152,
              "7e76e18d224531b8197a46231cb53daf7f2f6ca707130252becf933026ac4eea"),
    BoundFile("tokenizer/special_tokens_map.json", "tokenizer", 7079,
              "456b58fd240a06c743a7c2cf8008bec501240d68ebd1fc4018ea569505fea270"),
    BoundFile("tokenizer/spiece.model", "tokenizer", 4548313,
              "e3909a67b780650b35cf529ac782ad2b6b26e6d1f849d3fbb6a872905f452458"),
    BoundFile("tokenizer/tokenizer.json", "tokenizer", 16837459,
              "20a46ac256746594ed7e1e3ef733b83fbc5a6f0922aa7480eda961743de080ef"),
    BoundFile("tokenizer/tokenizer_config.json", "tokenizer", 61758,
              "1d8d2a216bf8e70ac15b7ddcea566c4dd0433c024b39a58ca5e4c66bd78defbd"),
    BoundFile("vae/config.json", "vae", 724, "f0c1cc1d7decb5badc384f54691746a27a9aeff49f7ebca974e583389342d527"),
    BoundFile("vae/diffusion_pytorch_model.safetensors", "vae", 507591892,
              "d6e524b3fffede1787a74e81b30976dce5400c4439ba64222168e607ed19e793"),
    BoundFile("scheduler/scheduler_config.json", "scheduler", 201,
              "3bc8f1aa49f02f0ea23fe02f2c6a6a5a9c037c441a8345fd5938d0bcc61338e9"),
)

LORA_RELPATH = "lora/cfg_step_lora.safetensors"


def binding_fingerprint(files: tuple[BoundFile, ...] = LONGCAT_VIDEO_FILES) -> str:
    """Stable identity of the bound weights (used as the model fingerprint of cache epochs)."""
    payload = {"repo": HF_REPO_ID, "revision": HF_REVISION, "source_commit": LONGCAT_VIDEO_COMMIT,
               "files": sorted((f.relpath, f.size, f.sha256) for f in files)}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


MODEL_FINGERPRINT = binding_fingerprint()


def file_sha256(path: str | os.PathLike, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def resolve_verify_mode(mode: str | None = None) -> str:
    mode = (mode or os.environ.get(VERIFY_ENV) or "full").strip().lower()
    if mode not in ("full", "stat"):
        raise WeightBindingError(f"{VERIFY_ENV} must be 'full' or 'stat', got {mode!r}")
    return mode


def verify_checkpoint(root: str | os.PathLike, mode: str | None = None, *, roles: tuple[str, ...] | None = None,
                      files: tuple[BoundFile, ...] | None = None, workers: int = 8) -> dict:
    """Check every bound file under ``root``.

    ``full`` hashes every file (exact binding). ``stat`` checks sizes and hashes only files below 1 MiB
    (configs, indices, tokenizer metadata); it is an explicitly weaker mode for repeated local loads and is
    recorded as such in the returned report. Raises :class:`WeightBindingError` on any mismatch.
    """
    mode = resolve_verify_mode(mode)
    root = Path(root)
    if not root.is_dir():
        raise WeightBindingError(f"LongCat-Video checkpoint directory not found: {root}")
    selected = [f for f in (LONGCAT_VIDEO_FILES if files is None else files) if roles is None or f.role in roles]
    if not selected:
        raise WeightBindingError(f"no bound files for roles {roles!r}")
    problems: list[str] = []
    to_hash: list[BoundFile] = []
    for bound in selected:
        path = root / bound.relpath
        if not path.is_file():
            problems.append(f"missing {bound.relpath}")
            continue
        size = path.stat().st_size
        if size != bound.size:
            problems.append(f"size mismatch {bound.relpath}: {size} != {bound.size}")
            continue
        if mode == "full" or bound.size < SMALL_FILE_BYTES:
            to_hash.append(bound)
    hashed = {}
    if not problems and to_hash:
        with cf.ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            futures = {pool.submit(file_sha256, root / b.relpath): b for b in to_hash}
            for fut in cf.as_completed(futures):
                b = futures[fut]
                digest = fut.result()
                hashed[b.relpath] = digest
                if digest != b.sha256:
                    problems.append(f"sha256 mismatch {b.relpath}: {digest} != {b.sha256}")
    if problems:
        raise WeightBindingError(f"LongCat-Video weights at {root} do not match the binding: " + "; ".join(sorted(problems)))
    return {"root": str(root.resolve()), "mode": mode, "roles": sorted({f.role for f in selected}),
            "files_checked": len(selected), "files_hashed": len(hashed), "model_fingerprint": MODEL_FINGERPRINT,
            "hf_repo": HF_REPO_ID, "hf_revision": HF_REVISION, "source_commit": LONGCAT_VIDEO_COMMIT,
            "weaker_stat_mode": mode == "stat"}
