#!/usr/bin/env python3
"""Plot validated full-spectrum summaries; never launch generation or metrics.

Input JSON: {schema: reelcache_rank_observations_v1, expected: [...],
observations: [...]}.  Alternatively use observation_files: [{path, sha256}]
from rank_analysis.write_spectrum_observation; full spectra and compact
observation files are hash checked.  The parent-owned protocol supplies the complete expected
identity/geometry inventory.  Calibration and heldout outputs are never pooled.
Each split gets all/first/later-boundary PNG/PDFs and r/C variants.  Optional
quality JSON contains expected_cases, outcomes, reference_arm and candidate_arms;
it is reported with complete-group coverage, failures and paired bootstrap CIs.
Every completed outcome requires a valid per-case matching pairing_sha256,
even when expected_cases does not repeat it.  Longform CSV retains both CDF
bound pairs and complete_endpoint_used, including ambiguous full-rank proofs.
Optional uniform_arms names measured reference ladder arms for actual-byte
bracketing; outcomes must then bind representation as well as byte accounting.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reelcache.rank_analysis import (  # noqa: E402
    AnalysisError, OBSERVATION_SCHEMA, SPLITS, aggregate_ranks,
    canonical_sha256, file_sha256, load_observation_document, longform_rows, matched_uniform_bytes,
    paired_group_bootstrap,
    paired_accounting_tables, plot_rank_panels, read_json, validate_census,
    validate_quality_outcomes, write_csv, write_json,
)


def build_report(input_path, output_dir, *, cell_order, plots=True, quality_path=None,
                 bootstrap_seed=0, bootstrap_resamples=2000):
    data = load_observation_document(input_path)
    rows = validate_census(data["observations"], data["expected"])
    if len(cell_order) != 3 or set(cell_order) != {row["cell"] for row in rows}:
        raise AnalysisError("declare exactly the three observed cells in row order")
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError("report output must be a new, task-owned directory")
    output.mkdir(parents=True)
    provenance = {
        "schema": "reelcache_rank_report_v1", "input_sha256": file_sha256(input_path),
        "expected_sha256": canonical_sha256(data["expected"]),
        "script_sha256": file_sha256(__file__),
        "analysis_sha256": file_sha256(ROOT / "reelcache/rank_analysis.py"),
        "study_contract_sha256": file_sha256(ROOT / "reelcache/study.py"),
        "experiment_result_claim": False,
        "meaning": "numeric analysis of supplied records; this script does not establish native provenance",
        "figures": {},
        "source_observation_artifacts": data.get("source_observation_artifacts", []),
    }
    for split in SPLITS:
        subset = [row for row in rows if row["split"] == split]
        if not subset:
            continue
        directory = output / split
        write_json(directory / "observations.json", {
            "schema": OBSERVATION_SCHEMA,
            "expected": [row for row in data["expected"] if row["split"] == split],
            "observations": subset,
        })
        write_csv(directory / "observations.csv", longform_rows(subset))
        for boundaries in ("all", "first", "later"):
            for normalized in (False, True):
                name = boundaries + ("_normalized" if normalized else "_rank")
                aggregates = aggregate_ranks(rows, data["expected"], split=split,
                                             boundaries=boundaries, normalized=normalized)
                write_json(directory / f"{name}.json", aggregates)
                write_csv(directory / f"{name}.csv", aggregates)
                if plots:
                    provenance["figures"][f"{split}/{name}"] = plot_rank_panels(
                        aggregates, cell_order=cell_order, png_path=directory / f"{name}.png",
                        pdf_path=directory / f"{name}.pdf",
                    )
    if quality_path is not None:
        quality = read_json(quality_path)
        validate_quality_outcomes(quality["expected_cases"], quality["outcomes"])
        arms = quality["candidate_arms"]
        if not isinstance(arms, list) or not arms or any(not isinstance(arm, str) or not arm for arm in arms) or (
            len(set(arms)) != len(arms) or quality["reference_arm"] in arms
        ):
            raise AnalysisError("quality reports need distinct declared candidate/reference arms")
        uniform_arms = quality.get("uniform_arms", [])
        if not isinstance(uniform_arms, list) or any(not isinstance(arm, str) or not arm for arm in uniform_arms) or (
            len(set(uniform_arms)) != len(uniform_arms)
        ):
            raise AnalysisError("uniform ladder IDs must be an explicit unique list")
        known_arms = {quality["reference_arm"], *arms, *uniform_arms}
        if any(row["arm_id"] not in known_arms for row in quality["outcomes"]):
            raise AnalysisError("quality outcomes contain an undeclared arm")
        declared_groups = {(row["cell"], row["prompt_group"], row["split"]) for row in rows}
        if any((row["cell"], row["prompt_group"], row["split"]) not in declared_groups
               for row in quality["expected_cases"]):
            raise AnalysisError("quality cases differ from the declared census cell/group/split inventory")
        provenance["quality_input_sha256"] = file_sha256(quality_path)
        for split in SPLITS:
            for cell in cell_order:
                expected = [row for row in quality["expected_cases"]
                            if row["split"] == split and row["cell"] == cell]
                if not expected:
                    continue
                case_ids = {row["case_id"] for row in expected}
                outcomes = [row for row in quality["outcomes"] if row["case_id"] in case_ids]
                for arm in quality["candidate_arms"]:
                    report = paired_group_bootstrap(
                        expected, outcomes, reference_arm=quality["reference_arm"],
                        candidate_arm=arm, seed=bootstrap_seed, resamples=bootstrap_resamples,
                    )
                    # IDs are data, not user-controlled directory components.
                    token = canonical_sha256({"cell": cell, "arm": arm})[:16]
                    directory = output / split / "quality" / token
                    write_json(directory / "paired.json", {"cell": cell, "arm_id": arm, **report})
                    write_csv(directory / "paired_metrics.csv", report["metrics"])
                    if report["groups"]:
                        write_csv(directory / "paired_groups.csv", report["groups"])
                    write_json(directory / "accounting.json", paired_accounting_tables(
                        expected, outcomes, reference_arm=quality["reference_arm"], candidate_arm=arm,
                    ))
                    if quality.get("uniform_arms") and arm not in quality["uniform_arms"]:
                        for denominator in ("decoder_closure_bytes", "snapshot_closure_bytes", "filesystem_bytes"):
                            matched = matched_uniform_bytes(
                                expected, outcomes, adaptive_arm=arm,
                                uniform_arms=quality["uniform_arms"], denominator=denominator,
                            )
                            write_json(directory / f"matched_{denominator}.json", matched)
                            if matched["per_case"]:
                                write_csv(directory / f"matched_{denominator}.csv", matched["per_case"])
    write_json(output / "provenance.json", provenance)
    return provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cells", nargs=3, required=True)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--quality", type=Path)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    args = parser.parse_args()
    build_report(args.input, args.output, cell_order=args.cells, plots=not args.no_plots,
                 quality_path=args.quality, bootstrap_seed=args.bootstrap_seed,
                 bootstrap_resamples=args.bootstrap_resamples)


if __name__ == "__main__":
    main()
