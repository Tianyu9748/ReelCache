#!/usr/bin/env python3
"""Fit source-bound numeric calibration data with the actual online predictor.

Inputs are calibration-only numeric census JSON (inline observations or
hash-bound observation_files) and detector JSON
{schema: reelcache_rank_detector_observations_v1, detector_spec: {...},
detector_sha256: "<full recipe fingerprint>", observations: [...]}. Detector
records must bind the same fixed, group-independent CPU QR/q=2 acquisition
recipe and both thresholds from its SAME final fit, not oracle labels or
policy/template-dependent detector settings. Each row retains the full
AnchorAcquisition.to_dict() payload and its canonical acquisition_sha256.
The full recipe binds the acquisition's declared CPU thread count, which must
match every record's work telemetry. Numeric replay neither reacquires features
nor changes its host thread count; mixed capture recipes are rejected.

This offline CPU entrypoint does not capture data, load a model, stage assets,
reserve resources or launch native experiments.  It writes LOPO-CV tables and
small online RankPredictorProfile records for one cell/target/energy/history
regime.  The parent can bind complete K/V startup/later reports using
rank_analysis.bind_calibration_reports and its admitted base RankPolicy.
For a STATIC layerwise control, explicitly choose frozen fixed_layerwise K/V
profiles from one declared regime and compile them with rank_analysis.static_layer_ranks.
Use that result as representation.layer_ranks with NO rank_policy; an adaptive
zero-anchor RankPolicy remains guarded and is not a fixed-rank codec control.
Unresolved required acquisitions abort the current and later trajectory chunks.
CV retains their failure/work records and selects complete-trajectory coverage
before rank/byte proxies; calibration_union_admission replays all selected
startup/later K/V union keys before a policy may be bound.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reelcache.rank_analysis import (  # noqa: E402
    AnalysisError, calibrate_selector, canonical_sha256, file_sha256, freeze_calibration,
    load_observation_document, read_json, write_csv, write_json,
)
from reelcache.study import CELLS  # noqa: E402


def build_calibration_report(input_path, detector_path, output_dir, **settings):
    document = load_observation_document(input_path, required_split="calibration")
    detector_data = read_json(detector_path)
    if detector_data.get("schema") != "reelcache_rank_detector_observations_v1":
        raise AnalysisError("unsupported detector observation schema")
    if not isinstance(detector_data.get("observations"), list):
        raise AnalysisError("detector observations must be an explicit numeric row list")
    if not isinstance(detector_data.get("detector_spec"), dict) or detector_data.get("detector_sha256") != (
        canonical_sha256(detector_data["detector_spec"])
    ):
        raise AnalysisError("detector input must bind its complete frozen recipe and matching SHA256")
    if "detector_spec" in settings:
        raise AnalysisError("detector recipe is bound by the feature bank, not a calibration override")
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError("calibration output must be a new, task-owned directory")
    report = calibrate_selector(
        document["observations"], document["expected"], detector_data["observations"],
        detector_spec=detector_data["detector_spec"], **settings,
    )
    output.mkdir(parents=True)
    write_json(output / "calibration.json", report)
    frozen_hash = freeze_calibration(report, output / "profiles.json")
    summaries = [
        {
            **{name: row[name] for name in ("policy_kind", "anchor_budget", "anchor_layers",
                                           "rank_margin", "uniform_quantile")},
            **row["metrics"],
        }
        for row in report["candidates"]
    ]
    write_csv(output / "cv_candidates.csv", summaries)
    write_csv(output / "selected_cv_predictions.csv", report["selected_validation_predictions"])
    write_json(output / "provenance.json", {
        "schema": "reelcache_rank_calibration_cli_v1", "input_sha256": file_sha256(input_path),
        "detector_input_sha256": file_sha256(detector_path),
        "profiles_sha256": frozen_hash, "script_sha256": file_sha256(__file__),
        "analysis_sha256": file_sha256(ROOT / "reelcache/rank_analysis.py"),
        "online_predictor_sha256": report["predictor_sha256"], "model_loaded": False,
        "detector_spec": report["detector_spec"], "detector_sha256": report["detector_sha256"],
        "quality_guarantee": False, "native_experiment_launched": False,
        "scope": "numeric calibration only; not an experiment result or a launch receipt",
    })
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--detectors", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cell", choices=tuple(CELLS), required=True)
    parser.add_argument("--target", choices=("K", "V"), required=True)
    parser.add_argument("--tau", type=float, choices=(.95, .99), required=True)
    parser.add_argument("--history-regime", choices=("startup", "later"), required=True)
    parser.add_argument("--execution-mode", choices=("causal_lagged", "whole_clean_staged"), required=True)
    parser.add_argument("--rank-buckets", nargs="+", type=int, required=True)
    parser.add_argument("--anchor-candidate-layers", nargs="+", type=int)
    parser.add_argument("--margins", nargs="+", type=int, default=(0, 8, 16))
    parser.add_argument("--uniform-quantiles", nargs="+", type=float, default=(.5, .9, 1.0))
    parser.add_argument("--pair-shortlist", type=int, default=8)
    args = parser.parse_args()
    build_calibration_report(
        args.input, args.detectors, args.output, cell=args.cell, target=args.target, tau=args.tau,
        history_regime=args.history_regime, execution_mode=args.execution_mode,
        rank_buckets=args.rank_buckets, anchor_candidate_layers=args.anchor_candidate_layers,
        margins=args.margins, uniform_quantiles=args.uniform_quantiles,
        pair_shortlist=args.pair_shortlist,
    )


if __name__ == "__main__":
    main()
