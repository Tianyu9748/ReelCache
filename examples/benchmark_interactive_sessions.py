"""Session-tree workload v2: sweep expansion, feasibility, frozen bundles, content slots and the native runner.

Subcommands (generator: Python standard library + numpy, no GPU):

* ``expand SWEEP.json``: the settings of a sweep document (one-factor-at-a-time around the anchor, explicit blocks).
* ``check SWEEP.json [--bank BANK] [--report OUT.json]``: feasibility of every setting (validation, structural
  rules, full in-memory generation), the feasible region and the content-slot counts; writes nothing else.
  Exit 0 when every setting is feasible, 3 otherwise.
* ``generate SWEEP.json --out DIR [--setting ID ...] [--bank BANK] [--feasible-only]``: checks every selected
  setting first; writes the bundles (``DIR/<setting>/``) only when all of them are feasible (or, with
  ``--feasible-only``, the feasible ones), plus ``DIR/feasibility.json``.
* ``slots SWEEP.json --out SLOTS.jsonl [--bank BANK]``: the union of the content slots of every feasible setting
  (authoring input) and the per-setting counts.
* ``validate-bank --bank BANK [--slots SLOTS.jsonl] [--dataset DIR] [--report OUT.json]``: authored-entry checks.
* ``run ...``: the open/closed-loop native runner (``wl`` package under
  ``artifacts/sglang_integration/benchmarks/interactive_native_v2``; ``--runner-root`` or ``WL_RUNNER_ROOT``);
  every argument after ``run`` is passed to it (``run --help`` for its options).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reelcache import session_workloads as sw  # noqa: E402

#: The evaluation runner (a ``wl`` package) is not part of this repository: ``run`` needs ``--runner-root`` or
#: ``WL_RUNNER_ROOT``.
DEFAULT_RUNNER_ROOT = None


def setting_dirname(setting_id: str) -> str:
    """A directory name for a setting id (``:`` -> ``__``, ``,`` -> ``+``, other unsafe characters -> ``_``)."""
    name = setting_id.replace(":", "__").replace(",", "+")
    return re.sub(r"[^A-Za-z0-9._=+-]", "_", name)


def read_sweep(path: str) -> tuple[dict, dict]:
    text = Path(path).read_text()
    document = json.loads(text)
    return document, {"name": document.get("name"), "path": str(Path(path).resolve()),
                      "sha256": sw.sha256_text(text)}


def _print(value) -> None:
    print(json.dumps(value, indent=1, sort_keys=True, default=str))


def _summary(report: dict) -> dict:
    return {"count": report["count"], "feasible": report["feasible"], "infeasible": report["infeasible"],
            "region": report["region"], "content_slots_union": report["content_slots_union"],
            "settings": [{key: row.get(key) for key in ("setting_id", "feasible", "requests", "slots",
                                                        "slots_unbound", "slots_by_kind", "frozen", "problems")}
                         for row in report["settings"]]}


def cmd_expand(args) -> int:
    document, _ = read_sweep(args.sweep)
    _print([{"setting_id": item["setting_id"], "origin": item["origin"], "aliases": item.get("aliases", [])}
            for item in sw.expand_sweep(document)])
    return 0


def cmd_check(args) -> int:
    document, info = read_sweep(args.sweep)
    report = sw.check_settings(sw.expand_sweep(document), bank=args.bank, sweep=info)
    if args.report:
        Path(args.report).write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    _print(_summary(report))
    return 0 if not report["infeasible"] else 3


def cmd_generate(args) -> int:
    document, info = read_sweep(args.sweep)
    expanded = sw.expand_sweep(document)
    if args.setting:
        wanted = set(args.setting)
        expanded = [item for item in expanded if item["setting_id"] in wanted
                    or wanted & set(item.get("aliases", []))]
        missing = wanted - {item["setting_id"] for item in expanded} - {a for item in expanded
                                                                       for a in item.get("aliases", [])}
        if missing:
            print(f"unknown settings: {sorted(missing)}", file=sys.stderr)
            return 2
    out = Path(args.out)
    targets = {item["setting_id"]: out / setting_dirname(item["setting_id"]) for item in expanded}
    existing = [str(path) for path in targets.values() if path.exists()]
    if existing:
        print(f"bundle directories exist (bundles are immutable): {existing}", file=sys.stderr)
        return 2
    # every combination is checked before anything is written
    report = sw.check_settings(expanded, bank=args.bank, sweep=info, keep=True)
    workloads = report.pop("workloads")
    if report["infeasible"] and not args.feasible_only:
        _print(_summary(report))
        print(f"infeasible settings {report['infeasible']}: nothing written (use --feasible-only to write the "
              "feasible ones)", file=sys.stderr)
        return 3
    out.mkdir(parents=True, exist_ok=True)
    written = {}
    for setting_id, workload in workloads.items():
        written[setting_id] = sw.write_bundle(workload, targets[setting_id])
    report["bundles"] = written
    (out / "feasibility.json").write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    _print({"written": {key: {k: v[k] for k in ("path", "bundle_id", "frozen", "requests")}
                        for key, v in written.items()}, "infeasible": report["infeasible"]})
    return 0


def cmd_slots(args) -> int:
    document, info = read_sweep(args.sweep)
    report = sw.check_settings(sw.expand_sweep(document), bank=args.bank, sweep=info, keep=True)
    workloads = report.pop("workloads")
    slots = sw.union_slots(workloads.values())
    if args.out:
        with open(args.out, "x") as handle:
            for slot in slots:
                handle.write(sw.canonical_json(slot) + "\n")
    per_setting = {row["setting_id"]: {"feasible": row["feasible"], "slots": row.get("slots"),
                                       "unbound": row.get("slots_unbound"), "by_kind": row.get("slots_by_kind"),
                                       "requests": row.get("requests")} for row in report["settings"]}
    _print({"settings": per_setting, "union": report["content_slots_union"], "union_written": args.out,
            "infeasible": report["infeasible"]})
    return 0


def cmd_validate_bank(args) -> int:
    slots = None
    if args.slots:
        slots = [json.loads(line) for line in Path(args.slots).read_text().splitlines() if line.strip()]
    dataset = sw.load_dataset(args.dataset) if args.dataset else None
    report = sw.validate_bank(args.bank, slots, dataset=dataset)
    if args.report:
        Path(args.report).write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    _print({key: report[key] for key in ("entries", "slots", "ok", "complete")}
           | {"errors": report["errors"][:50], "missing_keys": len(report["missing_keys"])})
    return 0 if report["ok"] else 4


def cmd_run(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="benchmark_interactive_sessions.py run", add_help=False)
    parser.add_argument("--runner-root", default=os.environ.get("WL_RUNNER_ROOT", DEFAULT_RUNNER_ROOT))
    known, rest = parser.parse_known_args(argv)
    if not known.runner_root:
        print("run needs --runner-root or WL_RUNNER_ROOT (the evaluation runner is not included)", file=sys.stderr)
        return 2
    root = Path(known.runner_root)
    if not (root / "wl" / "__init__.py").is_file():
        print(f"no runner package at {root}/wl", file=sys.stderr)
        return 2
    sys.path.insert(0, str(root))
    from wl import cli  # noqa: E402 - the runner package (benchmarks/interactive_native_v2)

    return cli.main(rest)


def parser() -> argparse.ArgumentParser:
    top = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = top.add_subparsers(dest="command", required=True)
    expand = sub.add_parser("expand", help="list the settings of a sweep document")
    expand.add_argument("sweep")
    check = sub.add_parser("check", help="feasibility region and content-slot counts (writes nothing else)")
    check.add_argument("sweep")
    check.add_argument("--bank")
    check.add_argument("--report")
    generate = sub.add_parser("generate", help="check every setting, then write the frozen bundles")
    generate.add_argument("sweep")
    generate.add_argument("--out", required=True)
    generate.add_argument("--setting", action="append", help="only this setting id (repeatable)")
    generate.add_argument("--bank")
    generate.add_argument("--feasible-only", action="store_true")
    slots = sub.add_parser("slots", help="union of the content slots of a sweep (authoring input)")
    slots.add_argument("sweep")
    slots.add_argument("--out")
    slots.add_argument("--bank")
    bank = sub.add_parser("validate-bank", help="check authored continuations")
    bank.add_argument("--bank", required=True)
    bank.add_argument("--slots")
    bank.add_argument("--dataset")
    bank.add_argument("--report")
    sub.add_parser("run", help="the native runner (arguments are passed to the wl package; run --help)",
                   add_help=False)
    return top


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["run"]:
        return cmd_run(argv[1:])
    args = parser().parse_args(argv)
    return {"expand": cmd_expand, "check": cmd_check, "generate": cmd_generate, "slots": cmd_slots,
            "validate-bank": cmd_validate_bank}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
