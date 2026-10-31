"""Assemble Astra-authored sessions and prepare reproducible synthetic fork DAGs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from reelcache.synthetic_workloads import compile_session, read_sessions, validate_bundle
from reelcache.workloads import read_narratives


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parts", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True, help="Original MemFlow JSONL")
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New final output directory")
    parser.add_argument("--chunks-per-prompt", type=int, default=1,
                        help="Model-free control mapping; real replay must audit the native chunk geometry")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    paths = sorted(args.parts.glob("batch-*.jsonl"))
    if len(paths) != 10:
        raise ValueError(f"expected ten complete Astra-authored parts, got {len(paths)}")
    sessions, provenance = [], []
    for path in paths:
        rows = read_sessions(path)
        if len(rows) != 50:
            raise ValueError(f"{path} has {len(rows)} sessions instead of 50")
        sessions.extend(rows)
        provenance.append({"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                           "sessions": len(rows)})
    raw = args.source.read_bytes()
    sources = read_narratives(raw.decode())
    source_manifest = json.loads(args.source_manifest.read_text())
    pinned_source = next(s for s in source_manifest["sources"] if s["source_id"] == "memflow")
    if hashlib.sha256(raw).hexdigest() != pinned_source["sha256"]:
        raise ValueError("original source differs from the pinned MemFlow audit")
    source_license = args.source.parent / "memflow.license.txt"
    license_bytes = source_license.read_bytes()
    if hashlib.sha256(license_bytes).hexdigest() != pinned_source["license_file_sha256"]:
        raise ValueError("source license differs from the pinned MemFlow audit")
    coverage = validate_bundle(sessions, sources)
    schedules = [compile_session(s, chunks_per_prompt=args.chunks_per_prompt, seed=args.seed) for s in sessions]
    args.output.mkdir(parents=True)
    for name, rows in (("sessions.jsonl", sessions), ("operations.jsonl", schedules)):
        with (args.output / name).open("x") as output:
            for row in rows:
                if name == "sessions.jsonl":
                    index = int(row["session_id"][1:])
                    row = {
                        **row, "schema": "reelcache.synthetic_interactive_session.v1",
                        "provenance": {
                            "kind": "memflow_extension" if index < 100 else "llm_authored_new",
                            "model": "gpt-6-astra", "reasoning_effort": "xhigh",
                            "source_narrative_index": index if index < 100 else None,
                            "source_revision": pinned_source["revision"] if index < 100 else None,
                            "recorded_video_user_trace": False,
                        },
                    }
                output.write(json.dumps(row, ensure_ascii=True, allow_nan=False) + "\n")
    hashes = {name: hashlib.sha256((args.output / name).read_bytes()).hexdigest()
              for name in ("sessions.jsonl", "operations.jsonl")}
    (args.output / "SOURCE_LICENSE.txt").write_bytes(license_bytes)
    hashes["SOURCE_LICENSE.txt"] = hashlib.sha256(license_bytes).hexdigest()
    groups = [s["session_id"] for s in sessions]
    # Keep all branches of one narrative together; a split is not an unseen-data qualification.
    splits = {"development": groups[:100], "evaluation_candidate": groups[100:],
              "unit": "whole_session_including_all_forks", "prior_use_audited": False,
              "note": "The original 100 MemFlow extensions are development-only; verify semantic overlap before held-out claims."}
    manifest = {
        "schema": "reelcache.synthetic_workload_manifest.v1", "coverage": coverage, "files_sha256": hashes,
        "source": pinned_source, "parts": provenance,
        "source_attribution": {
            "name": "MemFlow multi-prompt benchmark",
            "url": "https://github.com/KlingAIResearch/MemFlow",
            "changes": "First 100 six-prompt prefixes preserved; main paths extended and alternative fork continuations authored.",
        },
        "generation": {"model": "gpt-6-astra", "reasoning_effort": "xhigh",
                       "main_path_prefixes_preserved": True, "fork_text_authored_additionally": True},
        "fork_history_unit": "completed_prompt_segments",
        "native_chunk_mapping": {"chunks_per_prompt": args.chunks_per_prompt, "qualified": False},
        "claims": {"recorded_video_user_traces": False, "recorded_agent_traces": False,
                   "native_video_generated": False, "video_quality_measured": False},
        "source_license": "Apache-2.0 (retain original source attribution for the first 100 prefixes)",
    }
    for name, value in (("manifest.json", manifest), ("splits.json", splits)):
        with (args.output / name).open("x") as output:
            json.dump(value, output, indent=2, allow_nan=False)
            output.write("\n")
    print(json.dumps(coverage, indent=2))


if __name__ == "__main__":
    main()
