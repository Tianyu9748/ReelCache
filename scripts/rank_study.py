"""Create immutable rank-study protocols, manifests, and gated worker requests."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from reelcache.errors import CapacityError, ConfigurationError, IntegrityError, StateError
from reelcache.study import (
    CELLS, advance_launch_receipt, create_protocol, file_sha256, identity, make_launch_receipt, read_json,
    source_closure, stage_assets, validate_protocol, validate_request, verify_assets,
    validate_request_batch, verify_launch_receipt, write_json_new,
)


def _protocol(args):
    protocol = create_protocol(
        args.catalog, historical_paths=args.exclude, selection_seed=args.selection_seed,
    )
    validate_protocol(protocol)
    write_json_new(args.output, protocol)
    print(f"Protocol: {args.output}; identity: {protocol['protocol_id']}")
    for prompt in protocol["prompts"]:
        print(f"{prompt['prompt_id']} [{prompt['stratum']}]: {prompt['text']}")
    print("Prompt families still require semantic review before the native launch gate.")


def _freeze(args):
    closure = source_closure(ROOT, args.dependency)
    write_json_new(args.output, closure)
    print(f"Launch closure: {closure['closure_id']}; {len(closure['files'])} files")


def _receipt(args):
    protocol = read_json(args.protocol)
    receipt = make_launch_receipt(
        protocol, read_json(args.closure), verification_path=args.verification, review_path=args.review,
    )
    write_json_new(args.output, receipt)
    print(f"Native-preflight receipt: {receipt['receipt_id']}")


def _advance(args):
    receipt = advance_launch_receipt(
        read_json(args.receipt), read_json(args.protocol),
        evidence_path=args.evidence, next_phase=args.phase,
    )
    write_json_new(args.output, receipt)
    print(f"Admitted phase: {receipt['phase']}; receipt: {receipt['receipt_id']}")


def _assets(args):
    protocol, receipt = read_json(args.protocol), read_json(args.receipt)
    verify_launch_receipt(receipt, protocol)
    destination = args.destination.absolute()
    local_root = args.local_root.resolve(strict=True)
    if (
        not destination.parent.resolve(strict=True).is_relative_to(local_root)
        or str(local_root).startswith("/dev/shm")
        or local_root.stat().st_dev != Path("/tmp").stat().st_dev
    ):
        raise ConfigurationError("assets must use an explicitly owned local-SSD directory, not shared memory")
    if local_root.stat().st_uid != os.getuid():
        raise ConfigurationError("local asset root must be owned by the current user")
    asset_inputs = read_json(args.manifest)
    if not isinstance(asset_inputs, dict) or not isinstance(asset_inputs.get("files"), list):
        raise ConfigurationError("source asset manifest must contain an explicit files list")
    manifest = stage_assets(asset_inputs["files"], destination, reserve_bytes=args.reserve_bytes)
    verify_assets(destination)
    print(f"Verified local assets: {manifest['asset_id']}; {manifest['total_bytes']} bytes")


def _asset_manifest(args):
    records = []
    for item in args.file:
        name, separator, source = item.partition("=")
        if not separator:
            raise ConfigurationError("assets use NAME=ABSOLUTE_SOURCE_PATH")
        path = Path(source).resolve(strict=True)
        records.append({
            "name": name, "source": str(path), "sha256": file_sha256(path),
            "bytes": path.stat().st_size,
        })
    write_json_new(args.output, {"schema": "reelcache.source_assets.v1", "files": records})
    print(f"Asset inventory: {len(records)} fully hashed files")


def _requests(args):
    if len(args.cell) != 1:
        raise ConfigurationError("emit one cell per invocation; model/resolution epochs cannot be mixed")
    protocol, receipt = read_json(args.protocol), read_json(args.receipt)
    verify_launch_receipt(receipt, protocol, phase=args.phase)
    budgets = read_json(args.budgets)
    representation = read_json(args.representation)
    if args.phase == "native_preflight":
        if not args.development_prompts:
            raise ConfigurationError("preflight needs explicit historical development prompts")
        prompt_document = read_json(args.development_prompts)
        prompts = prompt_document.get("prompts")
        if not isinstance(prompts, list) or not prompts:
            raise ConfigurationError("development manifest must contain explicit prompt records")
        prompts = [{**prompt, "split": "development"} for prompt in prompts]
    else:
        split = "heldout" if args.phase == "heldout" else "calibration"
        prompts = [prompt for prompt in protocol["prompts"] if prompt["split"] == split]
    fps = read_json(args.source_fps)
    cases = []
    for cell_id in args.cell:
        if cell_id not in budgets or cell_id not in fps:
            raise ConfigurationError(f"missing explicit budget/native FPS for {cell_id}")
        for prompt in prompts:
            for seed in ([protocol["seeds"][0]] if args.first_seed_only else protocol["seeds"]):
                body = {
                    "schema": "reelcache.rank_request.v1", "protocol_id": protocol["protocol_id"],
                    "cell_id": cell_id, "phase": args.phase, "group_id": prompt["group_id"],
                    "prompt_id": prompt["prompt_id"], "prompt": prompt["text"], "split": prompt["split"],
                    "seed": seed, "chunks": args.chunks, "source_fps": fps[cell_id],
                    "history_mode": args.history_mode, "budgets": budgets[cell_id],
                    "representation": representation, "arm_id": args.arm,
                }
                body["case_id"] = identity({
                    "schema": "reelcache.rank_case_identity.v1",
                    **{name: body[name] for name in (
                        "protocol_id", "cell_id", "group_id", "prompt_id", "prompt", "seed",
                    )},
                })
                body["trajectory_id"] = identity({
                    "schema": "reelcache.rank_trajectory_identity.v1", "request": body,
                })
                body["output_namespace"] = (
                    f"{protocol['protocol_id']}/{args.phase}/{cell_id}/"
                    f"{prompt['prompt_id']}/seed{seed}/{identity(body)}"
                )
                request = {**body, "request_id": identity(body)}
                validate_request(request, protocol, phase=args.phase)
                cases.append(request)
    if len({case["request_id"] for case in cases}) != len(cases):
        raise IntegrityError("duplicate generation requests")
    document = {
        "schema": "reelcache.rank_requests.v1", "protocol_id": protocol["protocol_id"],
        "receipt_id": receipt["receipt_id"], "phase": args.phase, "requests": cases,
    }
    document = {**document, "requests_id": identity(document)}
    validate_request_batch(
        document, protocol, receipt, cell_id=args.cell[0], phase=args.phase,
    )
    write_json_new(args.output, document)
    print(f"Frozen {len(cases)} requests; no model loaded or generation launched")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("protocol")
    command.add_argument("--catalog", type=Path, required=True)
    command.add_argument("--exclude", type=Path, action="append", required=True)
    command.add_argument("--selection-seed", type=int, default=20260918)
    command.add_argument("--output", type=Path, required=True)
    command.set_defaults(run=_protocol)
    command = commands.add_parser("freeze")
    command.add_argument("--dependency", type=Path, action="append", default=[])
    command.add_argument("--output", type=Path, required=True)
    command.set_defaults(run=_freeze)
    command = commands.add_parser("receipt")
    for name in ("protocol", "closure", "verification", "review", "output"):
        command.add_argument(f"--{name}", type=Path, required=True)
    command.set_defaults(run=_receipt)
    command = commands.add_parser("advance")
    for name in ("protocol", "receipt", "evidence", "output"):
        command.add_argument(f"--{name}", type=Path, required=True)
    command.add_argument(
        "--phase", choices=("calibration_census", "calibration_frontier", "heldout"), required=True,
    )
    command.set_defaults(run=_advance)
    command = commands.add_parser("asset-manifest")
    command.add_argument("--file", action="append", required=True)
    command.add_argument("--output", type=Path, required=True)
    command.set_defaults(run=_asset_manifest)
    command = commands.add_parser("stage")
    for name in ("protocol", "receipt", "manifest", "destination", "local-root"):
        command.add_argument(f"--{name}", type=Path, required=True)
    command.add_argument("--reserve-bytes", type=int, required=True)
    command.set_defaults(run=_assets)
    command = commands.add_parser("requests")
    for name in ("protocol", "receipt", "budgets", "representation", "source-fps", "output"):
        command.add_argument(f"--{name}", type=Path, required=True)
    command.add_argument("--development-prompts", type=Path)
    command.add_argument("--cell", choices=tuple(CELLS), action="append", required=True)
    command.add_argument(
        "--phase", choices=("native_preflight", "calibration_census", "calibration_frontier", "heldout"),
        default="native_preflight",
    )
    command.add_argument("--history-mode", choices=("reload", "quality_hot"), default="reload")
    command.add_argument("--arm", required=True)
    command.add_argument("--chunks", type=int, default=2)
    command.add_argument("--first-seed-only", action="store_true")
    command.set_defaults(run=_requests)
    args = parser.parse_args()
    try:
        args.run(args)
    except (CapacityError, ConfigurationError, IntegrityError, StateError, OSError) as exc:
        parser.exit(1, f"Rank study blocked: {exc}\n")


if __name__ == "__main__":
    main()
