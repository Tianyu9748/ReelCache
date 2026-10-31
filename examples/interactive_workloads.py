"""Audit public inputs, compile trace-mapped video workloads, and validate their DAGs.

No model loading or GPU work occurs here. WebLINX compilation requires explicit
research-use clearance; browsing a public dataset card is not that clearance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from reelcache.errors import ConfigurationError
from reelcache.workloads import (
    WorkloadPolicy, browser_observations, compile_browser_workload, read_narratives,
    replay_cpu_control, validate_workload,
)


def fetch(url, *, max_bytes=8 * 1024**2):
    request = urllib.request.Request(url, headers={"User-Agent": "ReelCache-workload-audit", "Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=45) as response:
        data = response.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise ConfigurationError("source exceeds the bounded audit download limit")
    return data


def write_json(path, value):
    with Path(path).open("x") as output:
        json.dump(value, output, indent=2, allow_nan=False)
        output.write("\n")


def audit_sources(output):
    output.mkdir(parents=True, exist_ok=False)
    sources, narratives = [], {}
    for name, repo, path, license_path in (
        ("memflow", "KlingAIResearch/MemFlow", "prompts/interactive_benchmark.jsonl", "LICENSE"),
        ("longlive-smoke", "NVlabs/LongLive", "LongLive1.0/example/interactive_example.jsonl", "LongLive1.0/LICENSE"),
    ):
        commits = json.loads(fetch(f"https://api.github.com/repos/{repo}/commits?path={path}&per_page=1"))
        if not commits:
            raise ConfigurationError("source has no pin-able commit")
        sha = commits[0]["sha"]
        raw = fetch(f"https://raw.githubusercontent.com/{repo}/{sha}/{path}")
        license_bytes = fetch(f"https://raw.githubusercontent.com/{repo}/{sha}/{license_path}")
        rows = read_narratives(raw.decode())
        narratives[name] = rows
        (output / f"{name}.jsonl").write_bytes(raw)
        (output / f"{name}.license.txt").write_bytes(license_bytes)
        sources.append({
            "source_id": name, "repository": repo, "revision": sha, "path": path,
            "sha256": hashlib.sha256(raw).hexdigest(), "records": len(rows),
            "prompt_counts": sorted(set(len(row["prompts"]) for row in rows)),
            "observed_fields": ["prompts"], "lifecycle_operations_recorded": False,
            "license_file_sha256": hashlib.sha256(license_bytes).hexdigest(),
            "license": "Apache-2.0" if b"Apache License" in license_bytes else "requires_manual_review",
            "scheduled_segment_seconds": 10 if name == "memflow" else None,
            "timing_provenance": "prescribed_benchmark" if name == "memflow" else "not_in_prompt_file",
        })
    overlap = sorted(set(r["narrative_id"] for r in narratives["memflow"])
                     & set(r["narrative_id"] for r in narratives["longlive-smoke"]))
    for name, dataset in (("weblinx", "McGill-NLP/WebLINX"), ("swe-agent", "nebius/SWE-agent-trajectories")):
        metadata = json.loads(fetch(f"https://huggingface.co/api/datasets/{dataset}"))
        revision = metadata["sha"]
        card = fetch(f"https://huggingface.co/datasets/{dataset}/raw/{revision}/README.md")
        match = re.search(r"^license:\s*(\S+)", card.decode(), re.MULTILINE)
        sources.append({
            "source_id": name, "dataset": dataset, "revision": revision,
            "card_sha256": hashlib.sha256(card).hexdigest(), "license": match.group(1) if match else None,
            "data_downloaded": False,
            "research_use_clearance_required": name == "weblinx",
            "trace_evidence": "raw_replay_timestamps_and_tabs" if name == "weblinx" else "sequential_attempts_only",
            "explicit_checkpoint_tree_verified": False,
        })
    manifest = {
        "schema": "reelcache.interactive_source_audit.v1", "sources": sources,
        "narrative_group_overlap": overlap, "native_experiment_run": False,
        "prior_experiment_contamination_audit": "not_yet_verified",
        "blocked_gates": ["WebLINX research-use clearance", "full trace cohort and rewind-target coverage",
                          "native five-model lifecycle qualification"],
    }
    write_json(output / "manifest.json", manifest)
    return manifest


def audit_models(models_root):
    from reelcache.adapters.causvid import CausVidAdapter

    rows = []
    for name, repo_dir, config_path, integrated in (
        ("causvid", "RAVEN", "third_party/causvid/configs/wan_causal_dmd_vbench.yaml", True),
        ("longlive1", "RAVEN", "third_party/longlive/configs/longlive_vbench.yaml", False),
        ("longlive2", "LongLive", "configs/inference.yaml", False),
        ("raven-wan", "RAVEN", "configs", False),
        ("longcat-720p", "LongCat-Video", "run_demo_interactive_video.py", False),
    ):
        root = models_root / repo_dir
        commit = None
        if root.exists():
            result = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True)
            if result.returncode == 0:
                commit = result.stdout.strip()
        config = root / config_path
        rows.append({
            "model": name, "source_root": str(root), "source_present": root.is_dir(), "git_revision": commit,
            "contract_input": str(config),
            "contract_input_sha256": hashlib.sha256(config.read_bytes()).hexdigest() if config.is_file() else None,
            "reelcache_native_adapter_present": integrated,
            "native_lifecycle_qualified_for_this_workload": False,
            "checkpoint_weights_verified": False,
            "blocker": ("trace-mapped native pilot not run" if integrated else
                        "no model-specific ReelCache snapshot/restore/prompt-transition adapter"),
        })
    return {
        "schema": "reelcache.interactive_model_audit.v1", "models": rows,
        "causvid_operations": {name: callable(getattr(CausVidAdapter, name, None))
                              for name in ("snapshot", "restore", "fork", "set_conditioning", "step")},
        "external_harvest_adapters_are_serving_adapters": False,
        "note": "KV harvesting/generate_clip support does not establish clean-boundary serving lifecycle support.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    audit = commands.add_parser("audit")
    audit.add_argument("--output", type=Path, required=True, help="New source-audit directory")
    models = commands.add_parser("models")
    models.add_argument("--models-root", type=Path, required=True)
    models.add_argument("--output", type=Path, required=True)
    compile_ = commands.add_parser("compile")
    compile_.add_argument("--replay", type=Path, required=True, help="Local WebLINX raw replay.json")
    compile_.add_argument("--narratives", type=Path, required=True)
    compile_.add_argument("--narrative-index", type=int, default=0)
    compile_.add_argument("--episode-id", required=True)
    compile_.add_argument("--source-id", required=True, help="Pinned source revision and episode identifier")
    compile_.add_argument("--chunk-seconds", type=float, required=True, help="Audited native chunk media duration")
    compile_.add_argument("--chunks-per-action", type=int, default=1)
    compile_.add_argument("--pause-gap-seconds", type=float, default=30)
    compile_.add_argument("--branch-content", choices=("new_seed", "next_prompt"), default="new_seed")
    compile_.add_argument("--seed", type=int, default=0)
    compile_.add_argument("--infer-url-rewind", action="store_true")
    compile_.add_argument("--research-use-cleared", action="store_true",
                          help="Explicit acknowledgement that the WebLINX research-use terms are cleared")
    compile_.add_argument("--output", type=Path, required=True)
    validate = commands.add_parser("validate")
    validate.add_argument("workload", type=Path)
    control = commands.add_parser("control")
    control.add_argument("workload", type=Path)
    control.add_argument("--output", type=Path, required=True, help="New control output directory")
    args = parser.parse_args()
    if args.command == "audit":
        report = audit_sources(args.output)
        print(json.dumps({s["source_id"]: s.get("records", "card_only") for s in report["sources"]}))
    elif args.command == "models":
        report = audit_models(args.models_root)
        write_json(args.output, report)
        print(json.dumps({r["model"]: r["blocker"] for r in report["models"]}))
    elif args.command == "compile":
        if not args.research_use_cleared:
            parser.error("WebLINX compilation requires --research-use-cleared; public access is not license clearance")
        rows = read_narratives(args.narratives.read_text())
        if not 0 <= args.narrative_index < len(rows):
            parser.error("--narrative-index is outside the input set")
        raw = args.replay.read_bytes()
        observations = browser_observations(json.loads(raw))
        policy = WorkloadPolicy(
            chunk_seconds=args.chunk_seconds, chunks_per_action=args.chunks_per_action,
            pause_gap_seconds=args.pause_gap_seconds, seed=args.seed, branch_content=args.branch_content,
            infer_url_rewind=args.infer_url_rewind,
        )
        workload = compile_browser_workload(observations, rows[args.narrative_index], policy,
                                            source_id=args.source_id, episode_id=args.episode_id)
        workload["source_sha256"] = hashlib.sha256(raw).hexdigest()
        workload["license_clearance_acknowledged"] = True
        write_json(args.output, workload)
        print(json.dumps(workload["summary"]))
    elif args.command == "control":
        workload = json.loads(args.workload.read_text())
        args.output.mkdir(parents=True, exist_ok=False)
        report = replay_cpu_control(workload, args.output / "store")
        write_json(args.output / "result.json", report)
        print(json.dumps({"status": report["status"], "operations": report["operations"],
                          "native_model_executed": False}))
    else:
        workload = json.loads(args.workload.read_text())
        validate_workload(workload)
        print(json.dumps({"status": "passed", "operations": workload["summary"]["operations"],
                          "native_model_executed": False}))


if __name__ == "__main__":
    main()
