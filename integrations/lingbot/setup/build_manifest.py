#!/usr/bin/env python3
"""Weights manifest (path, bytes, sha256) for the LingBot-World-Fast checkpoint dir; cross-checks LFS oids on HF."""
import json, os, subprocess, urllib.request
from pathlib import Path

CKPT = Path(os.environ.get("LB_CKPT", "/models/lingbot-world/lingbot-world-base-cam"))
HERE = Path(__file__).resolve().parent
REVS = {"robbyant/lingbot-world-base-cam": "6fc824ffc338d64c97c77e2eb8c0f4cfc24d82bd",
        "robbyant/lingbot-world-fast": "b7a1db2da9bd3ba675ee01c228f41db0309b981e"}


def hf_tree(repo, rev):
    url = f"https://huggingface.co/api/models/{repo}/tree/{rev}?recursive=true"
    with urllib.request.urlopen(url, timeout=60) as r:
        return {e["path"]: e for e in json.load(r) if e["type"] == "file"}


sums = {}
for line in (HERE / "weights.sha256sum").read_text().splitlines():
    digest, rel = line.split(None, 1)
    sums[rel.strip().lstrip("./")] = digest
trees = {repo: hf_tree(repo, rev) for repo, rev in REVS.items()}
rows, mismatches = [], []
for rel, digest in sorted(sums.items()):
    path = CKPT / rel
    if rel.startswith("lingbot_world_fast/"):
        repo, hf_path = "robbyant/lingbot-world-fast", rel.split("/", 1)[1]
    else:
        repo, hf_path = "robbyant/lingbot-world-base-cam", rel
    entry = trees[repo].get(hf_path)
    size = path.stat().st_size
    hf_size = (entry.get("lfs") or {}).get("size", entry.get("size")) if entry else None
    lfs_oid = (entry.get("lfs") or {}).get("oid") if entry else None
    ok = entry is not None and hf_size == size and (lfs_oid is None or lfs_oid == digest)
    if not ok:
        mismatches.append(rel)
    rows.append({"path": f"$LB_CKPT/{rel}", "relpath": rel, "bytes": size, "sha256": digest, "hf_repo": repo,
                 "hf_revision": REVS[repo], "hf_lfs_oid": lfs_oid, "hf_size": hf_size, "verified_against_hf": ok})
manifest = {"schema": "lingbot.weights_manifest.v1", "ckpt_dir": "$LB_CKPT", "revisions": REVS,
            "files": rows, "total_bytes": sum(r["bytes"] for r in rows), "mismatches": mismatches,
            "not_downloaded": "lingbot-world-base-cam high_noise_model/ and low_noise_model/ (MoE experts, ~150 GB): "
                              "WanI2VFast loads only T5, VAE and WanModelFast.from_pretrained(ckpt_dir, subfolder='lingbot_world_fast')"}
(HERE / "weights_manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
print(len(rows), "files", manifest["total_bytes"], "bytes; mismatches:", mismatches)
