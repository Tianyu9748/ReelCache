import copy
import csv
import dataclasses
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import types
import unittest
from unittest import mock
import uuid

from reelcache.rank_analysis import (
    AnalysisError, DIMENSIONS, GEOMETRY_FIELDS, IDENTITY_FIELDS, METRICS,
    OBSERVATION_SCHEMA, POINT_POLICY, RANGE_LABEL, aggregate_ranks,
    accounting_ratios, bracket_actual_bytes, canonical_sha256, file_sha256,
    bind_calibration_reports, calibration_union_admission, calibrate_selector, freeze_calibration, observation_from_spectrum,
    rank_observation_identity,
    load_observation_document, matched_uniform_bytes,
    native_magi_source, observe_magi_spectrum,
    dc_fraction_bounds, detector_records_from_acquisition,
    static_layer_ranks,
    longform_rows, paired_accounting_tables, paired_group_bootstrap, plot_rank_panels, read_json,
    validate_accounting, validate_census, validate_rank_observation, write_json,
    validate_detector_records, write_csv, write_spectrum_observation,
)


def observation(*, group="g0", split="calibration", seed=0, cell="causvid480",
                geometry=(4680, 1536, 30), layer=0, target="K", boundary=0, rank=4):
    n_rows, channels, layers = geometry
    case = f"{cell}/{group}/{seed}"
    return {
        "case_id": case, "prompt_group": group, "split": split, "seed": seed,
        "cell": cell, "layer": layer, "target": target, "boundary": boundary,
        "trajectory_id": "raw/" + case, "n_rows": n_rows, "n_channels": channels,
        "n_layers": layers, "source_dtype": "float32" if target == "K" and n_rows != 4680 else "bfloat16",
        "source_domain": "normalized_pre_rope" if target == "K" else "direct",
        "input_sha256": "1" * 64, "spectrum_sha256": "2" * 64, "oracle_sha256": "3" * 64,
        "full_spectrum_count": channels, "energy_definition": "uncentered_squared_singular_values",
        "full_input_energy": 100.0, "mean_dc_energy_fraction": .1, "zero_energy": False,
        "r95": {"lower": rank, "upper": rank, "cdf_before_lower": 0 if rank == 1 else .94,
                "cdf_at_upper": .96, "cdf_tolerance": 1e-12},
        "r99": {"lower": rank + 3, "upper": rank + 3, "cdf_before_lower": .98,
                "cdf_at_upper": 1.0, "cdf_tolerance": 1e-12},
        "diagnostics": {"solver": "synthetic_fixture", "trace_relative_error": 0.0,
                        "symmetry_relative_error": 0.0, "negative_eigenvalue_mass": 0.0},
        "evidence_kind": "synthetic_fixture",
    }


def census(*, all_cells=False, heldout=False):
    cells = [("causvid480", (4680, 1536, 30))]
    if all_cells:
        cells += [("magi720", (21600, 1024, 48)), ("magi1080", (48960, 1024, 48))]
    rows = []
    groups = [("g0", "calibration", 0), ("g1", "calibration", 1)]
    if heldout:
        groups += [("h0", "heldout", 2), ("h1", "heldout", 3)]
    for cell, geometry in cells:
        for group, split, group_index in groups:
            for seed in (0, 1):
                for boundary in (0, 1):
                    rank = 1 + 8 * group_index + 4 * seed + 2 * boundary
                    for layer in range(geometry[2]):
                        for target in ("K", "V"):
                            rows.append(observation(group=group, split=split, seed=seed, cell=cell,
                                                    geometry=geometry, layer=layer, target=target,
                                                    boundary=boundary, rank=rank))
    expected = [{name: row[name] for name in (*IDENTITY_FIELDS, *GEOMETRY_FIELDS)} for row in rows]
    return rows, expected


def accounting(native=1000, stored=200):
    return {
        "native_logical_bytes": native, "source_bytes": native + 100,
        "decoder_closure_bytes": stored, "snapshot_closure_bytes": stored + 50,
        "filesystem_bytes": stored + 100, "basis_bytes": stored // 2,
        "payload_bytes": stored - stored // 2 - 10, "metadata_bytes": 10,
        "work_counts": {"decompositions": 2, "guard_hits": 1},
        "resource_bytes": {"host_peak_bytes": 4000}, "timings_seconds": {"encode": .1},
    }


def quality_fixture():
    expected, results = [], []
    for group in range(3):
        for seed in (0, 1):
            for window in (0, 1):
                case = f"g{group}-s{seed}-w{window}"
                identity = {"case_id": case, "prompt_group": f"g{group}",
                            "seed": seed, "cell": "causvid480", "split": "heldout"}
                expected.append(identity)
                for arm in ("RAW", "adaptive"):
                    drop = (.1 + .1 * group) if arm == "adaptive" else 0.0
                    metrics = {name: .9 - drop for name in METRICS}
                    metrics["ssim"] = 1.0 - drop
                    metrics["dynamic_degree"] = 1
                    results.append({**identity, "arm_id": arm, "status": "complete",
                                    "pairing_sha256": canonical_sha256({"fixture_pair": identity}),
                                    "metrics": metrics, "accounting": accounting()})
    return expected, results


class OwnedDirectoryTest(unittest.TestCase):
    def setUp(self):
        self.directory = Path(".test-artifacts") / (f"rank-analysis-fixture-{os.getpid()}-" + uuid.uuid4().hex)
        self.directory.mkdir(parents=True)

    def tearDown(self):
        shutil.rmtree(self.directory)


class RankObservationTests(unittest.TestCase):
    def test_valid_original_numeric_summary(self):
        row = observation()
        self.assertEqual(validate_rank_observation(row), row)

    def test_parent_protocol_identity_mapping_binds_cell_geometry(self):
        row = rank_observation_identity(
            case_id="case", group_id="g0", split="calibration", seed=0, cell_id="magi720",
            layer=47, target="K", boundary=0, trajectory_id="raw-case", evidence_kind="synthetic_fixture",
            source_bindings={"parent_head": None, "parent_trajectory_id": "same-run-parent",
                             "protocol_sha256": "a" * 64},
        )
        self.assertEqual((row["n_rows"], row["n_channels"], row["n_layers"]), (21600, 1024, 48))
        self.assertEqual(row["source_dtype"], "float32")
        self.assertEqual(row["parent_trajectory_id"], "same-run-parent")
        self.assertEqual(row["protocol_sha256"], "a" * 64)
        bad = observation()
        bad["cell"] = "magi720"
        with self.assertRaisesRegex(AnalysisError, "geometry differs"):
            validate_rank_observation(bad)

    def test_native_identity_requires_actual_protocol_arm_and_parent_bindings(self):
        arguments = dict(
            case_id="case", group_id="g0", split="calibration", seed=0, cell_id="magi720",
            layer=0, target="K", boundary=0, trajectory_id="current-arm", evidence_kind="native",
        )
        with self.assertRaisesRegex(AnalysisError, "protocol_id"):
            rank_observation_identity(**arguments)
        bindings = {
            "protocol_id": "a" * 64, "request_id": "actual-request", "arm_id": "actual-arm",
            "branch_id": "actual-branch", "parent_head": None,
            "launch_closure_id": "b" * 64, "fixture": False, "native_execution": True,
        }
        identity = rank_observation_identity(**arguments, source_bindings=bindings)
        self.assertEqual(identity["cell_id"], identity["cell"])
        self.assertEqual(identity["group_id"], identity["prompt_group"])
        self.assertEqual(identity["arm_id"], "actual-arm")
        with self.assertRaisesRegex(AnalysisError, "parent head"):
            rank_observation_identity(**{**arguments, "boundary": 1}, source_bindings=bindings)
        with self.assertRaisesRegex(AnalysisError, "history head"):
            rank_observation_identity(
                **arguments, source_bindings={**bindings, "history_head_id": "another-parent"},
            )
        for change in ({"fixture": True}, {"native_execution": False}):
            with self.assertRaisesRegex(AnalysisError, "non-fixture execution"):
                rank_observation_identity(**arguments, source_bindings={**bindings, **change})
        with self.assertRaisesRegex(AnalysisError, "cannot claim native"):
            rank_observation_identity(
                **{**arguments, "evidence_kind": "synthetic_fixture"}, source_bindings=bindings,
            )
    def test_invalid_geometry_domain_dtype_and_nonfinite(self):
        base = observation()
        for change in (
            {"n_rows": 21600}, {"n_channels": 6144}, {"source_domain": "post_rope"},
            {"source_dtype": "float16"}, {"full_input_energy": float("nan")},
            {"mean_dc_energy_fraction": float("inf")}, {"full_spectrum_count": 32},
            {"energy_definition": "centered"}, {"seed": True},
            {"layer": 30}, {"input_sha256": "missing"},
        ):
            with self.subTest(change=change), self.assertRaises(AnalysisError):
                validate_rank_observation({**base, **change})

    def test_invalid_and_ambiguous_rank_brackets(self):
        row = observation()
        row["r95"]["upper"] += 1
        result = validate_rank_observation(row)
        self.assertEqual(result["r95"]["lower"], 4)
        self.assertEqual(result["r95"]["upper"], 5)
        self.assertFalse(longform_rows([row])[0]["rank_is_exact"])
        self.assertIsNone(longform_rows([row])[0]["rank_exact"])
        self.assertEqual(longform_rows([row])[0]["rank_status"], "interval")
        for change in (
            {"lower": 0}, {"lower": 8}, {"upper": 1600},
            {"cdf_before_lower": .96, "cdf_tolerance": 0},
            {"cdf_at_upper": .90, "cdf_tolerance": 0}, {"cdf_tolerance": float("nan")},
        ):
            bad = copy.deepcopy(row)
            bad["r95"].update(change)
            with self.subTest(change=change), self.assertRaises(AnalysisError):
                validate_rank_observation(bad)

    def test_zero_energy_is_rank_zero_not_nan(self):
        row = observation()
        row.update(full_input_energy=0.0, zero_energy=True, mean_dc_energy_fraction=None)
        for level in ("r95", "r99"):
            row[level] = {"lower": 0, "upper": 0, "cdf_before_lower": None,
                          "cdf_at_upper": None, "cdf_tolerance": 0}
        self.assertTrue(validate_rank_observation(row)["zero_energy"])
        row["mean_dc_energy_fraction"] = 0
        with self.assertRaises(AnalysisError):
            validate_rank_observation(row)

    def test_scaled_energy_and_roundoff_dc_fraction_are_not_clamped(self):
        row = observation()
        row.update(energy_scaled=25.0, unscale_energy_exponent=2,
                   mean_dc_energy_fraction=1 + 1e-12, mean_dc_energy_fraction_error_bound=1e-10)
        self.assertGreater(validate_rank_observation(row)["mean_dc_energy_fraction"], 1)
        row["full_input_energy"] = None
        with self.assertRaisesRegex(AnalysisError, "unscaled energy"):
            validate_rank_observation(row)
        row.update(energy_scaled=1.0, unscale_energy_exponent=2046)
        self.assertIsNone(validate_rank_observation(row)["full_input_energy"])

    def test_dc_ratio_bounds_include_independent_norm_uncertainty(self):
        summary = dc_fraction_bounds(mean_energy=10, energy=100, mean_error=1, norm_error=2)
        self.assertEqual(summary["mean_dc_energy_fraction"], .1)
        self.assertEqual(summary["mean_dc_energy_fraction_bounds"], [9 / 102, 11 / 98])
        self.assertGreater(summary["mean_dc_energy_fraction_error_bound"], .01)
        roundoff = dc_fraction_bounds(mean_energy=1.02, energy=1, mean_error=.01, norm_error=.02)
        self.assertEqual(roundoff["mean_dc_energy_fraction"], 1.02)
        self.assertEqual(roundoff["mean_dc_energy_fraction_bounds"], [1.01 / 1.02, 1])
        row = {**observation(), **roundoff}
        self.assertGreater(validate_rank_observation(row)["mean_dc_energy_fraction"], 1)
        self.assertTrue(all(value is None for value in dc_fraction_bounds(
            mean_energy=0, energy=0, mean_error=0, norm_error=0,
        ).values()))
        with self.assertRaisesRegex(AnalysisError, "positive lower norm"):
            dc_fraction_bounds(mean_energy=1, energy=1, mean_error=0, norm_error=1)

    def test_numeric_oracle_dc_roundoff_preserves_mean_and_norm_budgets(self):
        from reelcache.spectra import (
            SpectrumDiagnostics, SpectrumNumerics, SpectrumResult, SpectrumSource,
            _budgets, _solver, _workspace_bytes,
        )

        identity = rank_observation_identity(
            case_id="numeric-fixture", group_id="fixture", split="calibration", seed=0,
            cell_id="causvid480", layer=0, target="K", boundary=0, trajectory_id="fixture-raw",
            evidence_kind="synthetic_fixture", source_bindings={"fixture": True, "native_execution": False},
        )
        numerics = SpectrumNumerics()
        shape, energy, tile = (4680, 1536), 100.0, 64
        tiles = math.ceil(shape[0] / tile)
        budgets = _budgets(shape[1], tile, tiles, 0, energy, numerics)
        workspace = _workspace_bytes(shape, tile)
        diagnostics = SpectrumDiagnostics(
            tile_rows=tile, tiles=tiles, max_tile_rows=tile, rescalings=0,
            gram_trace_scaled=energy, raw_eigenvalue_sum_scaled=energy, symmetry_error_scaled=0,
            minimum_raw_eigenvalue_scaled=0, negative_eigenvalue_mass_scaled=0,
            structural_zero_tail_mass_scaled=0, direction_orthogonality_error=None,
            direction_residual_scaled=None, workspace_estimate_bytes=workspace,
            workspace_limit_bytes=workspace, **budgets,
        )
        source = SpectrumSource(
            identity="numeric-only-dc-fixture", shape=shape, dtype="bfloat16", target="K",
            key_space="pre_rope", provenance=identity, tensor_sha256="1" * 64,
        )
        with mock.patch("torch.linalg.eigh", side_effect=AssertionError("numeric fixture cannot solve")), \
                mock.patch("torch.linalg.eigvalsh", side_effect=AssertionError("numeric fixture cannot solve")):
            spectrum = SpectrumResult(
                source=source, eigenvalues_scaled=(energy, *(0.0 for _ in range(shape[1] - 1))),
                energy_scaled=energy, scale_exponent=0,
                mean_energy_scaled=energy + budgets["norm_error_bound_scaled"] / 2,
                diagnostics=diagnostics, numerics=numerics, solver=_solver("torch.linalg.eigvalsh"),
            )
            row = observation_from_spectrum(spectrum.to_dict(), identity=identity, spectrum_sha256="2" * 64)
        expected = dc_fraction_bounds(
            mean_energy=spectrum.mean_energy_scaled, energy=energy,
            mean_error=budgets["mean_error_bound_scaled"], norm_error=budgets["norm_error_bound_scaled"],
        )
        self.assertGreater(row["mean_dc_energy_fraction"], 1)
        self.assertGreater(row["mean_dc_energy_fraction"], 1 + budgets["mean_error_bound_scaled"] / energy)
        for name, value in expected.items():
            self.assertEqual(row[name], value)
        self.assertEqual(validate_rank_observation(row), row)

    def test_asymmetric_certified_bounds_not_scalar_summary_prove_endpoint(self):
        row = observation()
        row["r99"].update(
            tau=.99, upper=1536, rank=None, status="interval",
            cdf_before_lower_bounds=[.9799999999, .9800000001],
            cdf_at_upper_bounds=[1.0, 1.0], complete_endpoint_used=True,
            cdf_at_upper=1.000000000001, cdf_tolerance=0,
        )
        result = validate_rank_observation(row)
        self.assertGreater(result["r99"]["cdf_at_upper"], 1)
        self.assertEqual(result["r99"]["cdf_at_upper_bounds"], [1, 1])
        self.assertIsNone(result["r99"]["rank"])
        row["r99"]["cdf_tolerance"] = .5
        row["r99"]["cdf_at_upper_bounds"] = [.98, 1]
        with self.assertRaisesRegex(AnalysisError, "certified CDF bounds"):
            validate_rank_observation(row)

    def test_duplicate_missing_extra_and_binding_mismatch(self):
        rows, expected = census()
        for bad_rows, bad_expected in (
            (rows + [rows[0]], expected), (rows[1:], expected),
            (rows, expected + [expected[0]]), (rows, expected[1:]),
        ):
            with self.assertRaises(AnalysisError):
                validate_census(bad_rows, bad_expected)
        expected[0]["input_sha256"] = "f" * 64
        with self.assertRaisesRegex(AnalysisError, "input_sha256"):
            validate_census(rows, expected)
        expected[0].pop("input_sha256")
        expected[0]["parent_head"] = "expected-parent"
        rows[0]["parent_head"] = "different-parent"
        with self.assertRaisesRegex(AnalysisError, "parent_head"):
            validate_census(rows, expected)

    def test_group_cannot_cross_splits(self):
        rows, expected = census(heldout=True)
        for row in rows:
            if row["prompt_group"] == "h0":
                row["prompt_group"] = "g0"
        with self.assertRaisesRegex(AnalysisError, "crosses"):
            validate_census(rows, expected)

    def test_manifest_cannot_hide_one_seed_or_boundary_in_one_group(self):
        rows, _ = census()
        rows = [row for row in rows if not (row["prompt_group"] == "g1" and row["seed"] == 1)]
        expected = [{name: row[name] for name in (*IDENTITY_FIELDS, *GEOMETRY_FIELDS)} for row in rows]
        with self.assertRaisesRegex(AnalysisError, "coverage"):
            validate_census(rows, expected)

    def test_first_boundary_and_identical_seed_windows_are_required(self):
        rows, _ = census()
        missing_first = [row for row in rows if row["boundary"] != 0]
        with self.assertRaisesRegex(AnalysisError, "first clean boundary"):
            validate_census(missing_first, missing_first)
        for row in rows:
            if row["seed"] == 1 and row["boundary"] == 1:
                row["boundary"] = 2
        with self.assertRaisesRegex(AnalysisError, "same selected boundaries"):
            validate_census(rows, rows)


class RankAggregationTests(OwnedDirectoryTest):
    def test_longform_csv_roundtrip_preserves_complete_endpoint_and_unclamped_cdf(self):
        row = observation()
        row["r95"]["complete_endpoint_used"] = False
        row["r99"].update(
            upper=1536, rank=None, status="interval", complete_endpoint_used=True,
            cdf_before_lower_bounds=[.9799999999, .9800000001],
            cdf_at_upper_bounds=[1.0, 1.0], cdf_at_upper=1.000000000001, cdf_tolerance=0,
        )
        exported = longform_rows([row])
        self.assertFalse(exported[0]["complete_endpoint_used"])
        self.assertTrue(exported[1]["complete_endpoint_used"])
        path = self.directory / "endpoint.csv"
        write_csv(path, exported)
        with path.open(newline="") as stream:
            restored = list(csv.DictReader(stream))
        self.assertEqual(restored[0]["complete_endpoint_used"], "False")
        self.assertEqual(restored[1]["complete_endpoint_used"], "True")
        self.assertEqual(restored[1]["rank_status"], "interval")
        self.assertEqual(restored[1]["rank_exact"], "")
        self.assertGreater(float(restored[1]["cdf_at_upper"]), 1)
        self.assertEqual(json.loads(restored[1]["cdf_at_upper_bounds"]), [1, 1])

    def test_hand_calculated_prompt_equal_means_and_dispersion(self):
        rows, expected = census()
        report = aggregate_ranks(rows, expected, split="calibration")
        row = next(row for row in report if row["layer"] == 0 and row["target"] == "K" and row["tau"] == .95)
        self.assertEqual(row["mean"], 8)  # g0=(1,3,5,7)->4; g1=(9,11,13,15)->12.
        self.assertEqual((row["min"], row["max"]), (1, 15))
        self.assertEqual((row["std"], row["var"]), (4, 16))
        self.assertEqual((row["count"], row["observation_count"], row["seed_count"]), (2, 8, 4))
        self.assertIn("not a confidence interval", row["range_label"])

    def test_ambiguous_interval_is_not_silently_made_exact(self):
        rows, expected = census()
        rows[0]["r95"]["upper"] = 2
        row = next(row for row in aggregate_ranks(rows, expected, split="calibration")
                   if row["layer"] == 0 and row["target"] == "K" and row["tau"] == .95)
        self.assertEqual(row["mean_lower"], 8)
        self.assertEqual(row["mean_upper"], 8.125)
        self.assertEqual(row["ambiguous_count"], 1)
        self.assertEqual(row["point_policy"], POINT_POLICY)

    def test_first_later_and_normalized_are_separate(self):
        rows, expected = census(heldout=True)
        first = aggregate_ranks(rows, expected, split="calibration", boundaries="first")[0]
        later = aggregate_ranks(rows, expected, split="calibration", boundaries="later")[0]
        normalized = aggregate_ranks(rows, expected, split="calibration", normalized=True)[0]
        heldout = aggregate_ranks(rows, expected, split="heldout")[0]
        self.assertEqual(first["mean"], 7)
        self.assertEqual(later["mean"], 9)
        self.assertAlmostEqual(normalized["mean"], 8 / 1536)
        self.assertAlmostEqual(normalized["var"], 16 / 1536**2)
        self.assertEqual(heldout["mean"], 24)

    def test_plot_fixture_three_by_two_observed_range(self):
        rows, expected = census(all_cells=True)
        aggregates = aggregate_ranks(rows, expected, split="calibration")
        record = plot_rank_panels(
            aggregates, cell_order=("causvid480", "magi720", "magi1080"),
            png_path=self.directory / "fixture.png", pdf_path=self.directory / "fixture.pdf",
        )
        self.assertEqual(record["range_label"], RANGE_LABEL)
        self.assertEqual(record["font_settings"]["font.family"], "DejaVu Sans")
        self.assertEqual(record["aggregate_sha256"], canonical_sha256(aggregates))
        self.assertGreater((self.directory / "fixture.png").stat().st_size, 1000)
        self.assertEqual(record["pdf_sha256"], file_sha256(self.directory / "fixture.pdf"))
        aggregates[0]["split"] = "heldout"
        with self.assertRaisesRegex(AnalysisError, "mix"):
            plot_rank_panels(aggregates, cell_order=("causvid480", "magi720", "magi1080"),
                             png_path=self.directory / "bad.png", pdf_path=self.directory / "bad.pdf")

    def test_report_exports_calibration_and_heldout_separately(self):
        rows, expected = census(all_cells=True, heldout=True)
        source = self.directory / "fixture.json"
        write_json(source, {"schema": OBSERVATION_SCHEMA, "observations": rows, "expected": expected})
        spec = importlib.util.spec_from_file_location("rank_report_fixture", Path("scripts/rank_report.py"))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        result = module.build_report(source, self.directory / "report",
                                     cell_order=("causvid480", "magi720", "magi1080"), plots=False)
        self.assertFalse(result["experiment_result_claim"])
        for split in ("calibration", "heldout"):
            data = read_json(self.directory / "report" / split / "observations.json")
            self.assertEqual({row["split"] for row in data["observations"]}, {split})
            self.assertTrue((self.directory / "report" / split / "later_normalized.csv").is_file())
        self.assertFalse((self.directory / "report" / "pooled").exists())

    def test_report_quality_tables_include_byte_brackets_and_failure_coverage(self):
        rows, expected = census(all_cells=True, heldout=True)
        source = self.directory / "fixture.json"
        write_json(source, {"schema": OBSERVATION_SCHEMA, "observations": rows, "expected": expected})
        cases, outcomes = quality_fixture()
        cases = [row for row in cases if row["prompt_group"] != "g2"]
        outcomes = [row for row in outcomes if row["prompt_group"] != "g2"]
        for row in (*cases, *outcomes):
            row["prompt_group"] = "h" + row["prompt_group"][1:]
        for row in list(outcomes):
            row["representation"] = "svd_hevc"
            if row["arm_id"] == "adaptive":
                outcomes.append({**copy.deepcopy(row), "arm_id": "uniform-low", "accounting": accounting(stored=150)})
                outcomes.append({**copy.deepcopy(row), "arm_id": "uniform-high", "accounting": accounting(stored=250)})
        failed = next(row for row in outcomes if row["arm_id"] == "adaptive")
        failed.update(status="failed", error="fixture failure; not a native result")
        quality_path = self.directory / "quality.json"
        write_json(quality_path, {"expected_cases": cases, "outcomes": outcomes,
                                  "reference_arm": "RAW", "candidate_arms": ["adaptive"],
                                  "uniform_arms": ["uniform-low", "uniform-high"]})
        spec = importlib.util.spec_from_file_location("rank_report_quality_fixture", Path("scripts/rank_report.py"))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.build_report(source, self.directory / "report",
                            cell_order=("causvid480", "magi720", "magi1080"), plots=False,
                            quality_path=quality_path, bootstrap_resamples=20)
        token = canonical_sha256({"cell": "causvid480", "arm": "adaptive"})[:16]
        output = self.directory / "report" / "heldout/quality" / token
        paired = read_json(output / "paired.json")
        self.assertEqual(paired["coverage"], .5)
        self.assertEqual(len(paired["metrics"]), 7)
        matched = read_json(output / "matched_decoder_closure_bytes.json")
        self.assertEqual(matched["coverage"], .5)
        self.assertTrue(matched["pooled_sum_bytes"]["bracketed"])
        self.assertTrue((output / "matched_decoder_closure_bytes.csv").is_file())

    def test_json_rejects_duplicate_keys_and_nonfinite_and_overwrite(self):
        path = self.directory / "bad.json"
        for value in ('{"a": 1, "a": 2}', '{"x": NaN}'):
            path.write_text(value)
            with self.assertRaises(AnalysisError):
                read_json(path)
        with self.assertRaises(FileExistsError):
            write_json(path, {})
        with self.assertRaises(ValueError):
            write_json(self.directory / "nan.json", {"value": float("nan")})


class AccountingTests(unittest.TestCase):
    def test_invalid_sizes_and_denominator_rejected(self):
        base = accounting()
        for changes in (
            {"decoder_closure_bytes": -1}, {"basis_bytes": True},
            {"payload_bytes": 99}, {"snapshot_closure_bytes": 199},
            {"filesystem_bytes": 0}, {"native_logical_bytes": 0},
            {"source_bytes": float("nan")}, {"work_counts": {"svd": -.5}},
            {"timings_seconds": {"encode": float("inf")}},
        ):
            with self.subTest(changes=changes), self.assertRaises(AnalysisError):
                validate_accounting({**base, **changes})

    def test_pooled_ratio_is_sum_ratio_not_average_ratio(self):
        rows = [
            {"case_id": str(index), "prompt_group": str(index), "arm_id": "svd", "cell": "causvid480",
             "split": "heldout", "accounting": accounting(native, stored)}
            for index, (native, stored) in enumerate(((1000, 100), (3000, 600)))
        ]
        result = accounting_ratios(rows, numerator="native_logical_bytes", denominator="decoder_closure_bytes")
        self.assertAlmostEqual(result["pooled_sum_ratio"], 4000 / 700)
        self.assertEqual(result["mean_per_case_ratio"], 7.5)
        with self.assertRaises(AnalysisError):
            accounting_ratios(rows, numerator="raw_bytes", denominator="stored_bytes")

    def test_actual_hevc_bytes_bracket_and_remaining_mismatch(self):
        adaptive = {"case_id": "c", "cell": "causvid480", "split": "heldout", "representation": "svd_hevc",
                    "arm_id": "adaptive", "accounting": accounting(stored=250)}
        points = [{**adaptive, "arm_id": "uniform-small", "accounting": accounting(stored=200)},
                  {**adaptive, "arm_id": "uniform-large", "accounting": accounting(stored=300)}]
        result = bracket_actual_bytes(adaptive, points, denominator="decoder_closure_bytes")
        self.assertTrue(result["bracketed"])
        self.assertFalse(result["exact_match"])
        self.assertEqual(result["lower"]["mismatch_bytes"], -50)
        self.assertEqual(result["upper"]["relative_mismatch"], .2)
        self.assertFalse(result["interpolated"])
        self.assertFalse(bracket_actual_bytes(adaptive, points[:1], denominator="decoder_closure_bytes")["bracketed"])
        points[0]["representation"] = "svd"
        with self.assertRaises(AnalysisError):
            bracket_actual_bytes(adaptive, points, denominator="decoder_closure_bytes")


class PromptGroupBootstrapTests(unittest.TestCase):
    def test_completed_pairs_require_matching_hashes_without_expected_hashes(self):
        expected, results = quality_fixture()
        self.assertTrue(all("pairing_sha256" not in row for row in expected))
        for function in (paired_group_bootstrap, paired_accounting_tables):
            for invalid in (None, "", "bad", "A" * 64, "b" * 64):
                bad = copy.deepcopy(results)
                for row in bad:
                    row["pairing_sha256"] = "a" * 64
                if invalid is None:
                    bad[1].pop("pairing_sha256")
                else:
                    bad[1]["pairing_sha256"] = invalid
                with self.subTest(function=function.__name__, pairing=invalid), \
                        self.assertRaisesRegex(AnalysisError, "pairing|paired window"):
                    function(expected, bad, reference_arm="RAW", candidate_arm="adaptive")

    def test_pairing_mismatch_cannot_hide_in_an_incomplete_group(self):
        expected, results = quality_fixture()
        results[1]["pairing_sha256"] = "f" * 64
        results[3].update(status="failed", error="independent fixture failure")
        for function in (paired_group_bootstrap, paired_accounting_tables):
            with self.subTest(function=function.__name__), \
                    self.assertRaisesRegex(AnalysisError, "pairing|paired window"):
                function(expected, results, reference_arm="RAW", candidate_arm="adaptive")

    def test_private_rng_deterministic_paired_groups_not_repeated_cases(self):
        import numpy as np
        expected, results = quality_fixture()
        py_state, np_state = random.getstate(), np.random.get_state()
        first = paired_group_bootstrap(expected, results, reference_arm="RAW", candidate_arm="adaptive",
                                       seed=42, resamples=200)
        second = paired_group_bootstrap(expected, results, reference_arm="RAW", candidate_arm="adaptive",
                                        seed=42, resamples=200)
        self.assertEqual(first, second)
        self.assertEqual(random.getstate(), py_state)
        np.testing.assert_equal(np.random.get_state(), np_state)
        self.assertEqual(first["complete_groups"], 3)
        self.assertEqual(first["complete_group_cases"], 12)
        self.assertAlmostEqual(first["metrics"][0]["delta_mean"], -.2)
        self.assertEqual({row["metric"] for row in first["metrics"]}, set(METRICS))
        self.assertFalse(first["vbench_aggregate"])

    def test_failed_or_missing_seed_keeps_entire_group_in_coverage(self):
        expected, results = quality_fixture()
        failed = next(row for row in results if row["case_id"] == "g0-s0-w0" and row["arm_id"] == "adaptive")
        failed.update(status="failed", error="guard budget exceeded")
        failed.pop("metrics")
        results = [row for row in results if not (row["case_id"] == "g1-s1-w1" and row["arm_id"] == "adaptive")]
        result = paired_group_bootstrap(expected, results, reference_arm="RAW", candidate_arm="adaptive",
                                        resamples=20)
        self.assertEqual(result["complete_groups"], 1)
        self.assertEqual(result["excluded_groups"], ["g0", "g1"])
        self.assertEqual(result["coverage"], 1 / 3)
        self.assertEqual({row["status"] for row in result["failures"]}, {"failed", "missing"})
        self.assertEqual(result["failures"][0]["accounting"]["work_counts"]["guard_hits"], 1)

    def test_no_success_is_explicit_none_not_nan(self):
        expected, _ = quality_fixture()
        result = paired_group_bootstrap(expected, [], reference_arm="RAW", candidate_arm="adaptive",
                                        resamples=5)
        self.assertEqual(result["coverage"], 0)
        self.assertTrue(all(row["delta_mean"] is None for row in result["metrics"]))
        json.dumps(result, allow_nan=False)

    def test_accounting_uses_identical_complete_groups_and_preserves_missing(self):
        expected, results = quality_fixture()
        results[1].pop("accounting")
        result = paired_accounting_tables(expected, results, reference_arm="RAW", candidate_arm="adaptive")
        self.assertEqual(result["complete_groups"], ["g1", "g2"])
        self.assertEqual(result["paired_case_count"], 8)
        self.assertEqual(result["missing_or_failed"][0]["reason"], "accounting absent")
        case_sets = [{row["case_id"] for row in ratio["per_case"]} for ratio in result["ratios"]]
        self.assertTrue(all(cases == case_sets[0] for cases in case_sets))
        results[-1]["accounting"]["source_bytes"] += 1
        with self.assertRaisesRegex(AnalysisError, "numerators"):
            paired_accounting_tables(expected, results, reference_arm="RAW", candidate_arm="adaptive")

    def test_actual_uniform_frontier_exports_per_case_and_pooled_mismatch(self):
        expected, results = quality_fixture()
        for row in results:
            row["representation"] = "svd_hevc"
        for row in list(results):
            if row["arm_id"] == "adaptive":
                results.append({**copy.deepcopy(row), "arm_id": "uniform-low", "accounting": accounting(stored=150)})
                results.append({**copy.deepcopy(row), "arm_id": "uniform-high", "accounting": accounting(stored=250)})
        result = matched_uniform_bytes(expected, results, adaptive_arm="adaptive",
                                       uniform_arms=("uniform-low", "uniform-high"),
                                       denominator="decoder_closure_bytes")
        self.assertEqual(result["coverage"], 1)
        self.assertEqual(result["pooled_sum_bytes"]["adaptive_bytes"], 12 * 200)
        self.assertTrue(result["pooled_sum_bytes"]["bracketed"])
        self.assertEqual(result["per_case"][0]["lower"]["mismatch_bytes"], -50)
        self.assertFalse(result["rank_matched_is_byte_matched"])
        failed = next(row for row in results if row["arm_id"] == "uniform-high")
        failed.update(status="failed", error="codec budget failed")
        result = matched_uniform_bytes(expected, results, adaptive_arm="adaptive",
                                       uniform_arms=("uniform-low", "uniform-high"),
                                       denominator="decoder_closure_bytes")
        self.assertEqual(result["coverage"], 2 / 3)
        self.assertEqual(result["complete_cases"], 8)
        self.assertEqual(result["missing_or_failed"][0]["reason"], "codec budget failed")

    def test_missing_dimension_nonfinite_duplicate_or_split_leak_fails(self):
        expected, results = quality_fixture()
        bad = copy.deepcopy(results)
        bad[0]["metrics"].pop("imaging_quality")
        with self.assertRaisesRegex(AnalysisError, "all six"):
            paired_group_bootstrap(expected, bad, reference_arm="RAW", candidate_arm="adaptive")
        bad = copy.deepcopy(results)
        bad[0]["metrics"]["ssim"] = float("nan")
        for rows in (bad, results + [results[0]]):
            with self.assertRaises(AnalysisError):
                paired_group_bootstrap(expected, rows, reference_arm="RAW", candidate_arm="adaptive")
        expected[-1]["split"] = "calibration"
        with self.assertRaisesRegex(AnalysisError, "crosses"):
            paired_group_bootstrap(expected, results, reference_arm="RAW", candidate_arm="adaptive")


def detector_fixture(rows, spec=None):
    from reelcache.factorization import DetectorSpec
    from reelcache.study import CELLS
    from reelcache.types import MatrixKey, ModelIdentity, TensorContext

    spec = DetectorSpec() if spec is None else spec
    records = []
    for row in rows:
        cell = CELLS[row["cell"]]
        model = ModelIdentity("synthetic-detector-fixture", "fixture", row["n_layers"],
                              row["n_channels"], "tokens_channels", "bfloat16")
        context = TensorContext(
            model, MatrixKey(row["layer"], row["target"]), row["boundary"] * row["n_rows"],
            cell.frames, cell.token_height, cell.token_width,
            "pre_rope" if row["target"] == "K" else "none", source_dtype=row["source_dtype"],
        )
        width = min(spec.initial_width, *context.shape)
        seed = spec.seed_for(context)
        acquisition = {
            "schema_version": "reelcache.anchor_acquisition.v1",
            "detector_spec": spec.to_dict(), "detector_fingerprint": spec.fingerprint,
            "context": context.to_dict(), "source_sha256": "6" * 64,
            "source_hash_format": "sha256-little-endian-row-major-bytes-no-header-v1",
            "final_width": width, "features": {"0.95": row["r95"]["upper"], "0.99": row["r99"]["upper"]},
            "status": "zero" if row["zero_energy"] else "resolved",
            "attempts": [{"width": width, "seed": seed, "status": "resolved", "stop_crossing": row["r99"]["upper"]}],
            "metrics": {
                "fixture": True, "operation_counts_complete": True, "fit_attempts": 1,
                "detector_fingerprint": spec.fingerprint, "source_sha256": "6" * 64,
                "factorization_seed": seed, "detector_device": spec.device,
                "compute_dtype": spec.compute_dtype, "source_dtype": context.dtype,
                "cpu_threads": spec.cpu_threads, "final_width": width,
            },
        }
        for level, tau in (("r95", .95), ("r99", .99)):
            records.append({
                **{name: row[name] for name in IDENTITY_FIELDS}, "tau": tau,
                "input_sha256": row["input_sha256"], "detector_sha256": spec.fingerprint,
                "estimate_kind": "online_detector", "available_after_layer": row["layer"],
                "solver": "qr_q2", "rank_statistic": "approximate_spectral_crossing",
                "status": "resolved", "estimated_rank": row[level]["upper"], "maximum_width": width,
                "acquisition": acquisition, "acquisition_sha256": canonical_sha256(acquisition),
            })
    return records


def unresolved_detector_fixture(record):
    from reelcache.factorization import DetectorSpec
    from reelcache.types import TensorContext

    acquisition = record["acquisition"]
    spec = DetectorSpec.from_dict(acquisition["detector_spec"])
    context = TensorContext.from_dict(acquisition["context"])
    cap, seed = min(context.shape), spec.seed_for(context)
    width, attempts = min(spec.initial_width, cap), []
    while True:
        attempts.append({"width": width, "seed": (seed + len(attempts)) % 2**63,
                         "status": "unresolved", "stop_crossing": None})
        if width == cap:
            break
        width = min(cap, width * spec.width_growth)
    acquisition.update(status="unresolved", features={"0.95": None, "0.99": None},
                       final_width=cap, attempts=attempts)
    acquisition["metrics"].update(fit_attempts=len(attempts), final_width=cap)
    record.update(status="unresolved", estimated_rank=None, maximum_width=cap,
                  acquisition_sha256=canonical_sha256(acquisition))


class FixedDetectorBankTests(unittest.TestCase):
    def bank(self):
        from reelcache.factorization import DetectorSpec

        spec = DetectorSpec()
        rows = [observation()]
        return rows, detector_fixture(rows, spec), spec

    def validate(self, rows, records, spec):
        return validate_detector_records(rows, records, detector_spec=spec)

    def test_full_fixed_recipe_required_not_arbitrary_digest_or_learned_width(self):
        rows, records, spec = self.bank()
        self.assertEqual(len(self.validate(rows, records, spec)), 2)
        for value in (None, "4" * 64, {"initial_width": 128},
                      {**spec.to_dict(), "initial_width": 16},
                      {**spec.to_dict(), "base_seed": 7},
                      {**spec.to_dict(), "compute_dtype": "float64"},
                      {**spec.to_dict(), "stop_energy": .95},
                      {**spec.to_dict(), "training_groups": ["g0"]}):
            with self.subTest(recipe=value), self.assertRaisesRegex(AnalysisError, "frozen|recipe"):
                self.validate(rows, records, value)
        records[0]["detector_sha256"] = "4" * 64
        with self.assertRaisesRegex(AnalysisError, "recipe"):
            self.validate(rows, records, spec)

    def test_declared_threads_bind_recipe_and_telemetry_without_changing_replay_threads(self):
        import torch
        from reelcache.factorization import DetectorSpec

        rows, single_thread_records, default = self.bank()
        declared = DetectorSpec(cpu_threads=8)
        records = detector_fixture(rows, declared)
        self.assertNotEqual(declared.fingerprint, default.fingerprint)
        self.assertEqual(declared.fingerprint, canonical_sha256(declared.to_dict()))
        self.assertEqual(records[0]["acquisition"]["attempts"], single_thread_records[0]["acquisition"]["attempts"])
        threads = torch.get_num_threads()
        with mock.patch("torch.set_num_threads", side_effect=AssertionError("numeric replay cannot change threads")), \
                mock.patch("reelcache.factorization.acquire_anchor_directions",
                           side_effect=AssertionError("numeric replay cannot reacquire features")):
            self.assertEqual(len(self.validate(rows, records, declared)), 2)
            converted = detector_records_from_acquisition(records[0]["acquisition"], observation=rows[0])
            self.assertTrue(all(row["detector_sha256"] == declared.fingerprint for row in converted))
            with self.assertRaisesRegex(AnalysisError, "recipe"):
                self.validate(rows, [records[0], single_thread_records[1]], declared)
            mismatched = copy.deepcopy(records)
            mismatched[0]["acquisition"]["metrics"]["cpu_threads"] = default.cpu_threads
            for record in mismatched:
                record["acquisition_sha256"] = canonical_sha256(record["acquisition"])
            with self.assertRaisesRegex(AnalysisError, "telemetry"):
                self.validate(rows, mismatched, declared)
        self.assertEqual(torch.get_num_threads(), threads)

    def test_both_thresholds_must_share_the_same_last_fit(self):
        rows, records, spec = self.bank()
        records[1] = copy.deepcopy(records[1])
        records[1]["acquisition"]["metrics"]["separate_fit_marker"] = True
        records[1]["acquisition_sha256"] = canonical_sha256(records[1]["acquisition"])
        with self.assertRaisesRegex(AnalysisError, "SAME"):
            self.validate(rows, records, spec)
        records[1] = copy.deepcopy(records[0])
        records[1].update(tau=.99, estimated_rank=rows[0]["r99"]["upper"] + 1)
        with self.assertRaisesRegex(AnalysisError, "final fixed fit"):
            self.validate(rows, records, spec)

    def test_fixed_width_seed_and_stop_schedule_cannot_be_replaced_by_encoding_fits(self):
        for change in ("seed", "width", "extra_guard_fit"):
            rows, records, spec = self.bank()
            acquisition = records[0]["acquisition"]
            if change == "seed":
                acquisition["attempts"][0]["seed"] += 17
                acquisition["metrics"]["factorization_seed"] += 17
            elif change == "width":
                acquisition["attempts"][0]["width"] = 32
                acquisition["final_width"] = acquisition["metrics"]["final_width"] = 32
                for record in records:
                    record["maximum_width"] = 32
            else:
                attempt = copy.deepcopy(acquisition["attempts"][0])
                attempt.update(width=256, seed=(attempt["seed"] + 1) % 2**63)
                acquisition["attempts"].append(attempt)
                acquisition["final_width"] = acquisition["metrics"]["final_width"] = 256
                acquisition["metrics"]["fit_attempts"] = 2
                for record in records:
                    record["maximum_width"] = 256
            for record in records:
                record["acquisition_sha256"] = canonical_sha256(acquisition)
            with self.subTest(change=change), self.assertRaisesRegex(AnalysisError, "recipe|first resolved"):
                self.validate(rows, records, spec)

    def test_unresolved_at_cap_is_explicit_and_keeps_a_resolved_lower_threshold(self):
        rows, records, spec = self.bank()
        for record in records:
            unresolved_detector_fixture(record)
        self.assertEqual({row["status"] for row in self.validate(rows, records, spec).values()}, {"unresolved"})
        acquisition = records[0]["acquisition"]
        acquisition["features"]["0.95"] = 4
        records[0].update(status="resolved", estimated_rank=4)
        for record in records:
            record["acquisition_sha256"] = canonical_sha256(acquisition)
        valid = self.validate(rows, records, spec)
        self.assertIsNone(next(row for row in valid.values() if row["tau"] == .99)["estimated_rank"])
        acquisition["attempts"] = acquisition["attempts"][:1]
        acquisition["final_width"] = acquisition["metrics"]["final_width"] = 128
        acquisition["metrics"]["fit_attempts"] = 1
        for record in records:
            record.update(maximum_width=128, acquisition_sha256=canonical_sha256(acquisition))
        with self.assertRaisesRegex(AnalysisError, "geometry cap"):
            self.validate(rows, records, spec)

    def test_recipe_seed_excludes_prompt_group_generation_seed_and_policy(self):
        from reelcache.factorization import DetectorSpec

        spec = DetectorSpec()
        rows = [observation(group="g0", seed=0), observation(group="other-group", seed=123)]
        records = detector_fixture(rows, spec)
        seeds = {record["acquisition"]["attempts"][0]["seed"] for record in records}
        self.assertEqual(len(seeds), 1)
        self.assertEqual(len(self.validate(rows, records, spec)), 4)
        records[0]["acquisition"]["context"]["key_space"] = "post_rope"
        for record in records[:2]:
            record["acquisition_sha256"] = canonical_sha256(record["acquisition"])
        with self.assertRaisesRegex(AnalysisError, "context"):
            self.validate(rows, records, spec)

    def test_actual_shared_acquisition_validates_both_features_without_another_fit(self):
        import torch
        from reelcache.factorization import DetectorSpec, acquire_anchor_directions, tensor_sha256
        from reelcache.spectra import SpectrumSource, exact_spectrum
        from reelcache.study import CELLS
        from reelcache.types import MatrixKey, ModelIdentity, TensorContext

        cell = dataclasses.replace(CELLS["causvid480"], frames=1, token_height=2, token_width=2,
                                   layers=1, channels=8)
        model = ModelIdentity("fixed-acquisition-cpu-fixture", "fixture", 1, 8, "tokens_channels", "bfloat16")
        context = TensorContext(model, MatrixKey(0, "V"), 0, 1, 2, 2, "none", source_dtype="bfloat16")
        matrix = torch.arange(32, dtype=torch.bfloat16).reshape(4, 8)
        spec = DetectorSpec()
        with mock.patch("reelcache.rank_analysis.CELLS", {"causvid480": cell}):
            identity = rank_observation_identity(
                case_id="cpu-fixture", group_id="fixture-group", split="calibration", seed=19,
                cell_id="causvid480", layer=0, target="V", boundary=0, trajectory_id="raw-fixture",
                evidence_kind="synthetic_fixture",
                source_bindings={"original_bytes_sha256": tensor_sha256(matrix)},
            )
            source = SpectrumSource(
                identity="fixed-acquisition-cpu-fixture", shape=(4, 8), dtype="bfloat16",
                target="V", key_space="none", provenance=identity,
            )
            oracle = exact_spectrum(matrix, source=source, tile_rows=2, max_workspace_bytes=1 << 20)
            observation_row = observation_from_spectrum(oracle, identity=identity, spectrum_sha256="2" * 64)
            acquisition = acquire_anchor_directions(
                matrix, context, spec=spec, max_workspace_bytes=1 << 20, solver_workspace_bytes=0,
            ).to_dict()
            with mock.patch("torch.linalg.svd", side_effect=AssertionError("bank validation cannot refit")), \
                    mock.patch("torch.linalg.qr", side_effect=AssertionError("bank validation cannot refit")), \
                    mock.patch("torch.linalg.eigh", side_effect=AssertionError("bank validation cannot solve")), \
                    mock.patch("torch.linalg.eigvalsh", side_effect=AssertionError("bank validation cannot solve")):
                records = detector_records_from_acquisition(acquisition, observation=observation_row)
                validated = self.validate([observation_row], records, spec)
            self.assertEqual(len(validated), 2)
            self.assertEqual(acquisition["source_sha256"], tensor_sha256(matrix))
            self.assertNotEqual(acquisition["source_sha256"], observation_row["input_sha256"])
            acquisition["source_sha256"] = acquisition["metrics"]["source_sha256"] = "7" * 64
            for record in records:
                record["acquisition"] = acquisition
                record["acquisition_sha256"] = canonical_sha256(acquisition)
            with self.assertRaisesRegex(AnalysisError, "raw bytes"):
                self.validate([observation_row], records, spec)


class FixtureOnlineProfile:
    """A model-free callback fixture, not a second production rank policy."""

    def __init__(self, parameters):
        self.parameters = copy.deepcopy(parameters)

    def to_dict(self):
        return {"schema": "mock_online_profile", **self.parameters}


def fixture_predictor(profile, *, layer, current_anchors, previous_anchors, previous_regime):
    data = profile.parameters
    if data["policy_kind"] == "fixed_uniform":
        return data["uniform_rank"]
    ratios = []
    for anchor in data["anchor_layers"]:
        if anchor in current_anchors:
            baseline = data["detector_baselines"][data["history_regime"]][anchor]
            ratios.append(current_anchors[anchor] / baseline)
        elif anchor in previous_anchors and data["detector_baselines"][previous_regime][anchor] is not None:
            ratios.append(previous_anchors[anchor] / data["detector_baselines"][previous_regime][anchor])
    scale = sum(ratios) / len(ratios) if ratios else 1
    value = math.ceil(data["layer_template"][layer] * scale + data["rank_margin"])
    return next(rank for rank in data["rank_buckets"] if rank >= max(1, min(data["rank_buckets"][-1], value)))


class CalibrationSelectorTests(OwnedDirectoryTest):
    def calibrate(self, rows=None, expected=None, detectors=None, **options):
        from reelcache.factorization import DetectorSpec

        if rows is None:
            rows, expected = census()
        detectors = detector_fixture(rows) if detectors is None else detectors
        defaults = {
            "cell": "causvid480", "target": "K", "tau": .95,
            "history_regime": "startup", "execution_mode": "causal_lagged",
            "rank_buckets": (1, 4, 8, 16, 32, 1536), "profile_builder": FixtureOnlineProfile,
            "rank_predictor": fixture_predictor, "predictor_sha256": "5" * 64,
            "detector_spec": DetectorSpec().to_dict(),
            "anchor_candidate_layers": (0, 1, 2, 3), "margins": (0, 4),
            "uniform_quantiles": (.5, 1.0),
        }
        defaults.update(options)
        return calibrate_selector(rows, expected, detectors, **defaults)

    def test_lopo_two_seeds_all_boundaries_stay_together_and_profiles_freeze(self):
        report = self.calibrate()
        self.assertEqual(report["anchor_budgets"], [0, 1, 2, 4])
        self.assertFalse(report["quality_guarantee"])
        self.assertFalse(report["selection_cv_is_holdout"])
        self.assertEqual(
            {row["anchor_budget"] for row in report["profiles"] if row["policy_kind"] != "fixed_uniform"},
            {0, 1, 2, 4},
        )
        for candidate in report["candidates"]:
            self.assertIn(candidate["anchor_budget"], (0, 1, 2, 4))
            for fold in candidate["folds"]:
                self.assertNotIn(fold["validation_group"], fold["training_groups"])
                self.assertEqual(len(fold["training_groups"]), 1)
                self.assertIsNone(fold["metrics"]["measured_guard_frequency"])
                self.assertIsNone(fold["metrics"]["measured_packet_bytes"])
        frozen_path = self.directory / "profiles.json"
        digest = freeze_calibration(report, frozen_path)
        frozen = read_json(frozen_path)
        self.assertEqual(frozen["sha256"], digest)
        self.assertEqual(frozen["detector_spec"], report["detector_spec"])
        self.assertEqual(canonical_sha256(frozen["detector_spec"]), report["detector_sha256"])
        self.assertTrue(all(
            row["online_profile"]["detector_fingerprint"] == report["detector_sha256"]
            for row in frozen["profiles"]
        ))
        self.assertNotIn("selected_validation_predictions", frozen)
        self.assertTrue(all(row["online_profile"]["schema"] == "mock_online_profile" for row in frozen["profiles"]))
        with self.assertRaises(FileExistsError):
            freeze_calibration(report, frozen_path)
        changed = copy.deepcopy(report)
        changed["profiles"][0]["fitted_parameters"]["detector_fingerprint"] = "4" * 64
        with self.assertRaisesRegex(AnalysisError, "frozen profiles"):
            freeze_calibration(changed, self.directory / "changed_recipe.json")
        self.assertFalse((self.directory / "changed_recipe.json").exists())

    def test_declared_eight_thread_recipe_freezes_and_mixed_corpus_fails_before_fitting(self):
        from reelcache.factorization import DetectorSpec

        rows, expected = census()
        declared = DetectorSpec(cpu_threads=8)
        records = detector_fixture(rows, declared)
        report = self.calibrate(
            rows, expected, records, detector_spec=declared.to_dict(),
            profile_builder=None, rank_predictor=None, predictor_sha256=None,
            margins=(0,), uniform_quantiles=(1.0,),
        )
        self.assertTrue(report["using_deployment_predictor"])
        self.assertEqual(report["detector_spec"], declared.to_dict())
        frozen_path = self.directory / "eight_thread_profiles.json"
        freeze_calibration(report, frozen_path)
        frozen = read_json(frozen_path)
        self.assertEqual(frozen["detector_spec"], declared.to_dict())
        self.assertEqual(frozen["detector_sha256"], declared.fingerprint)
        self.assertTrue(all(
            row["online_profile"]["detector_fingerprint"] == declared.fingerprint
            for row in frozen["profiles"]
        ))
        records[0] = detector_fixture([rows[0]], DetectorSpec())[0]
        builder = mock.Mock(side_effect=AssertionError("mixed recipes must reject before profile fitting"))
        with self.assertRaisesRegex(AnalysisError, "recipe"):
            self.calibrate(rows, expected, records, detector_spec=declared.to_dict(), profile_builder=builder)
        builder.assert_not_called()

    def test_no_anchor_is_valid_and_can_win(self):
        rows, expected = census()
        for row in rows:
            row["r95"].update(lower=4, upper=4, cdf_before_lower=.94)
            row["r99"].update(lower=7, upper=7)
        report = self.calibrate(rows, expected)
        selected = next(row for row in report["profiles"] if row["selected"])
        self.assertEqual(selected["anchor_budget"], 0)
        self.assertEqual(selected["policy_kind"], "fixed_layerwise")
        self.assertEqual(selected["selection_cv_metrics"]["possible_underprediction"], 0)

    def test_causal_and_lagged_availability_are_not_staged_lookahead(self):
        calls = []

        def audited(profile, **kwargs):
            data = profile.parameters
            current, previous = kwargs["current_anchors"], kwargs["previous_anchors"]
            if data["execution_mode"] == "causal_lagged":
                self.assertTrue(all(layer < kwargs["layer"] for layer in current))
            if data["history_regime"] == "startup":
                self.assertEqual(previous, {})
                self.assertIsNone(kwargs["previous_regime"])
            calls.append((data["history_regime"], data["execution_mode"], kwargs["layer"],
                          tuple(current), tuple(previous)))
            return fixture_predictor(profile, **kwargs)

        self.calibrate(rank_predictor=audited)
        self.calibrate(rank_predictor=audited, history_regime="later")
        self.calibrate(rank_predictor=audited, execution_mode="whole_clean_staged")
        self.assertTrue(any(regime == "later" and layer == 0 and previous
                            for regime, mode, layer, current, previous in calls))
        self.assertTrue(any(mode == "whole_clean_staged" and layer == 0 and any(a > layer for a in current)
                            for regime, mode, layer, current, previous in calls))

    def test_heldout_feature_or_oracle_detector_leakage_is_rejected(self):
        rows, expected = census(heldout=True)
        builder = mock.Mock()
        with self.assertRaisesRegex(AnalysisError, "heldout"):
            self.calibrate(rows, expected, profile_builder=builder)
        builder.assert_not_called()
        rows, expected = census()
        detectors = detector_fixture(rows)
        detectors[0]["estimate_kind"] = "oracle_rank"
        with self.assertRaisesRegex(AnalysisError, "deployment"):
            self.calibrate(rows, expected, detectors)
        detectors[0]["estimate_kind"] = "online_detector"
        detectors[0]["solver"] = "exact_oracle"
        with self.assertRaisesRegex(AnalysisError, "QR/q=2"):
            self.calibrate(rows, expected, detectors)
        detectors[0]["solver"] = "qr_q2"
        detectors[0]["available_after_layer"] = -1
        with self.assertRaisesRegex(AnalysisError, "available"):
            self.calibrate(rows, expected, detectors)

    def test_unresolved_features_are_unavailable_not_fake_crossings(self):
        rows, expected = census()
        detectors = detector_fixture(rows)
        for row in detectors:
            if row["prompt_group"] == "g0" and row["layer"] == 0:
                unresolved_detector_fixture(row)
        report = self.calibrate(rows, expected, detectors)
        chosen = next(row for row in report["profiles"] if row["selected"])
        self.assertNotIn(0, chosen["anchor_layers"])
        bad = next(row for row in report["candidates"] if row["anchor_layers"] == [0])
        self.assertFalse(bad["metrics"]["selection_eligible"])
        self.assertLess(bad["metrics"]["coverage"], 1)
        self.assertTrue(any(fold["failed_or_incomplete_predictions"] for fold in bad["folds"]))
        detectors[0]["estimated_rank"] = detectors[0]["maximum_width"]
        with self.assertRaisesRegex(AnalysisError, "unresolved sketch"):
            self.calibrate(rows, expected, detectors)

    def test_entirely_unavailable_anchor_candidates_do_not_remove_zero_anchor_control(self):
        rows, expected = census()
        detectors = detector_fixture(rows)
        for record in detectors:
            if record["target"] == "K" and record["layer"] in (0, 1, 2, 3):
                unresolved_detector_fixture(record)
        report = self.calibrate(rows, expected, detectors, margins=(0,), uniform_quantiles=(1.0,))
        chosen = next(row for row in report["profiles"] if row["selected"])
        self.assertEqual(chosen["anchor_budget"], 0)
        self.assertEqual(chosen["selection_cv_metrics"]["coverage"], 1)
        failed_profiles = [row for row in report["profiles"] if row["anchor_budget"]]
        self.assertTrue(all(row["online_profile"] is None for row in failed_profiles))
        self.assertTrue(all(row["selection_cv_metrics"]["coverage"] == 0 for row in failed_profiles))
        freeze_calibration(report, self.directory / "zero_control_with_failures.json")

    def test_unresolved_inactive_union_anchor_aborts_startup_and_all_later_predictions(self):
        from reelcache.study import CELLS

        cell = dataclasses.replace(CELLS["causvid480"], layers=4)
        rows = [
            observation(group=f"g{group}", seed=seed, boundary=boundary, layer=layer, target=target,
                        geometry=(4680, 1536, 4), rank=4)
            for group in range(3) for seed in (0, 1) for boundary in (0, 1, 2)
            for layer in range(4) for target in ("K", "V")
        ]
        expected = [{name: row[name] for name in (*IDENTITY_FIELDS, *GEOMETRY_FIELDS)} for row in rows]
        with mock.patch("reelcache.rank_analysis.CELLS", {"causvid480": cell}), \
                mock.patch("reelcache.study.CELLS", {"causvid480": cell}):
            detectors = detector_fixture(rows)
            for record in detectors:
                if (record["prompt_group"], record["seed"], record["boundary"],
                    record["layer"], record["target"]) == ("g0", 0, 0, 0, "K"):
                    unresolved_detector_fixture(record)
            for history in ("startup", "later"):
                report = self.calibrate(rows, expected, detectors, history_regime=history,
                                        margins=(0,), uniform_quantiles=(1.0,))
                candidate = next(row for row in report["candidates"] if row["anchor_layers"] == [0])
                self.assertEqual(candidate["metrics"]["coverage"], 2 / 3)
                self.assertFalse(candidate["metrics"]["selection_eligible"])
                fold = next(row for row in candidate["folds"] if row["validation_group"] == "g0")
                failures = [row for row in fold["failed_or_incomplete_predictions"] if row["seed"] == 0]
                self.assertEqual(len(failures), 4 if history == "startup" else 8)
                self.assertTrue(all(row["predicted_starting_rank"] is None for row in failures))
                self.assertEqual({row["status"] for row in failures},
                                 {"failed"} if history == "startup" else {"not_executed"})
                admission = fold["acquisition_admission"]
                failure = admission["failures"][0]
                self.assertEqual(failure["failure_reason"], "fixed_anchor_feature_unresolved")
                self.assertEqual(failure["active_in_regime"], history == "startup")
                attempted = [row for row in admission["acquisition_attempts"] if row["seed"] == 0]
                self.assertEqual([row["boundary"] for row in attempted], [0])
                self.assertEqual(attempted[0]["attempted_acquisition_work"]["final_width"], 1536)
                chosen = next(row for row in report["profiles"] if row["selected"])
                self.assertEqual(chosen["selection_cv_metrics"]["coverage"], 1)
                self.assertNotIn(0, chosen["anchor_layers"])
            late_detectors = detector_fixture(rows)
            for record in late_detectors:
                if (record["prompt_group"], record["seed"], record["boundary"],
                    record["layer"], record["target"]) == ("g0", 0, 2, 0, "K"):
                    unresolved_detector_fixture(record)
            report = self.calibrate(rows, expected, late_detectors, history_regime="startup",
                                    margins=(0,), uniform_quantiles=(1.0,))
            candidate = next(row for row in report["candidates"] if row["anchor_layers"] == [0])
            self.assertEqual(candidate["metrics"]["coverage"], 2 / 3)
            self.assertEqual(candidate["metrics"]["scored_observation_count"], 16)
            failed_fold = next(row for row in candidate["folds"] if row["validation_group"] == "g0")
            self.assertEqual(failed_fold["acquisition_admission"]["failures"][0]["boundary"], 2)
            self.assertFalse(failed_fold["acquisition_admission"]["failures"][0]["active_in_regime"])
            self.assertTrue(all(not row["trajectory_complete"]
                                for row in failed_fold["failed_or_incomplete_predictions"]))

    def test_invalid_bucket_prediction_or_missing_detector_fails(self):
        with self.assertRaisesRegex(AnalysisError, "complete frozen"):
            self.calibrate(detector_spec=None)
        with self.assertRaisesRegex(AnalysisError, "buckets"):
            self.calibrate(rank_buckets=(4, 8))
        with self.assertRaisesRegex(AnalysisError, "non-admissible"):
            self.calibrate(rank_predictor=lambda *args, **kwargs: 0)
        rows, expected = census()
        with self.assertRaisesRegex(AnalysisError, "missing online detector"):
            self.calibrate(rows, expected, detector_fixture(rows)[1:])
        for row in (*rows, *expected):
            if row["boundary"] == 1:
                row["boundary"] = 2
        with self.assertRaisesRegex(AnalysisError, "consecutive boundaries"):
            self.calibrate(rows, expected, detector_fixture(rows))

    def test_deterministic_private_rng_and_proxy_brackets(self):
        before = random.getstate()
        first, second = self.calibrate(), self.calibrate()
        self.assertEqual(first, second)
        self.assertEqual(random.getstate(), before)
        for row in first["selected_validation_predictions"]:
            self.assertLessEqual(row["excess_bytes_proxy_lower"], row["excess_bytes_proxy_upper"])
        self.assertFalse(first["proxy_definition"]["actual_packet_bytes"])

    def test_deployment_profiles_bind_startup_later_under_one_policy(self):
        from reelcache.online_rank import RankPolicy, RankPredictorProfile, RankPreparation, RankState, rank_predictor
        from reelcache.types import MatrixKey, ModelIdentity
        from reelcache.errors import ConfigurationError

        rows, expected = census()
        detectors = detector_fixture(rows)
        reports = [
            self.calibrate(rows, expected, detectors, target=target, history_regime=regime,
                           profile_builder=None, rank_predictor=None, predictor_sha256=None,
                           anchor_candidate_layers=(0, 1, 2, 3) if regime == "startup" else (1, 2, 3, 4),
                           margins=(0,), uniform_quantiles=(1.0,))
            for regime in ("startup", "later") for target in ("K", "V")
        ]
        self.assertTrue(all(report["using_deployment_predictor"] for report in reports))
        keys = [MatrixKey(layer, target) for layer in range(30) for target in ("K", "V")]
        base = RankPolicy(
            policy_id="synthetic-fixture", model=ModelIdentity("fixture", "fixture", 30, 1536,
                                                               "tokens_channels", "bfloat16"),
            geometry=(3, 30, 52), source_dtypes={"K": "bfloat16", "V": "bfloat16"},
            source_fingerprint="1" * 64, geometry_fingerprint="2" * 64,
            history_fingerprint="3" * 64, calibration_fingerprint="4" * 64,
            implementation_fingerprint="5" * 64, energy_target=.95,
            template_ranks={key: 4.0 for key in keys},
            allowed_ranks={key: (1, 4, 8, 16, 32, 1536) for key in keys},
        )
        bound = bind_calibration_reports(base, reports, policy_id="calibrated-fixture")
        self.assertEqual(set(bound.predictor_profiles), {"startup:K", "startup:V", "later:K", "later:V"})
        self.assertEqual(RankPolicy.from_dict(bound.to_dict()).fingerprint, bound.fingerprint)
        self.assertEqual(bound.source_limit_bytes, base.source_limit_bytes)
        self.assertEqual(bound.staging_limit_bytes, 0)
        self.assertNotEqual(bound.calibration_fingerprint, base.calibration_fingerprint)
        key = MatrixKey(10, "K")
        current = {anchor: 12 for anchor in bound.anchors if anchor.target == "K"}
        profile = bound.predictor_profiles["startup:K"]
        self.assertIsInstance(profile, RankPredictorProfile)
        online = bound.predict_rank(key, current_anchors=current)
        calibrated = rank_predictor(
            profile, layer=10, current_anchors={key.layer: value for key, value in current.items()},
            previous_anchors={}, previous_regime=None,
        )
        self.assertEqual(online.initial_rank, calibrated)
        preparation = RankPreparation(bound, chunk_id="first", parent_head=None, token_start=0)
        preparation._current_anchors.update(current)
        prepared_prediction = preparation._prediction(key)
        self.assertEqual(prepared_prediction.predicted_rank, online.predicted_rank)
        self.assertEqual(prepared_prediction.initial_rank, online.initial_rank)
        self.assertEqual(list(prepared_prediction.prediction_sources), list(online.prediction_sources))
        preparation.close()
        prior = RankState(
            bound.fingerprint, last_chunk_id="first", token_end=4680, completed_chunks=1,
            anchor_ranks={anchor: 12 for anchor in bound.anchors},
            anchor_chunks={anchor: 1 for anchor in bound.anchors},
            selected_ranks={key: 1 for key in keys},
        )
        later = bound.predict_rank(key, current_anchors={}, prior_state=prior)
        expected_later = rank_predictor(
            bound.predictor_profiles["later:K"], layer=10, current_anchors={},
            previous_anchors={key.layer: value for key, value in prior.anchor_ranks.items() if key.target == "K"},
            previous_regime="startup",
        )
        self.assertEqual(later.initial_rank, expected_later)
        with self.assertRaises((ConfigurationError, AnalysisError)):
            bind_calibration_reports(base, reports[:-1], policy_id="incomplete")
        bad = copy.deepcopy(reports)
        bad[0]["profiles"][0]["profile_sha256"] = "a" * 64
        selected = next(row for row in bad[0]["profiles"] if row["selected"])
        selected["profile_sha256"] = "a" * 64
        with self.assertRaisesRegex(AnalysisError, "checksum"):
            bind_calibration_reports(base, bad, policy_id="tampered")
        admission = calibration_union_admission(reports)
        self.assertEqual(admission["coverage"], 1)
        self.assertEqual(
            {(row["layer"], row["target"]) for row in admission["required_union"]},
            {(key.layer, key.target) for key in bound.anchors},
        )
        self.assertTrue(admission["required_union"], "fixture must exercise actual union acquisition")
        self.assertTrue(any(not row["active_in_regime"] for row in admission["acquisition_attempts"]))
        expected_attempts = len(admission["required_union"]) * sum(
            len(case["boundaries"]) for case in reports[0]["trajectory_inventory"]
        )
        self.assertEqual(len(admission["acquisition_attempts"]), expected_attempts)
        missing = copy.deepcopy(reports)
        with_bank = next(report for report in missing if report["final_acquisition_bank"]["observations"])
        with_bank["final_acquisition_bank"]["observations"].pop()
        with self.assertRaisesRegex(AnalysisError, "every selected key"):
            calibration_union_admission(missing)
        changed_trajectory = copy.deepcopy(reports)
        with_bank = next(report for report in changed_trajectory if report["final_acquisition_bank"]["observations"])
        for row in with_bank["final_acquisition_bank"]["observations"]:
            row["trajectory_id"] = "other/" + row["case_id"]
        with self.assertRaisesRegex(AnalysisError, "original trajectory"):
            calibration_union_admission(changed_trajectory)
        first_key = admission["required_union"][0]
        corrupted = copy.deepcopy(reports)
        for report in corrupted:
            for record in report["final_acquisition_bank"]["detectors"]:
                if (record["layer"], record["target"], record["boundary"], record["seed"],
                    record["prompt_group"]) == (first_key["layer"], first_key["target"], 0, 0, "g0"):
                    unresolved_detector_fixture(record)
        rejected = calibration_union_admission(corrupted)
        self.assertLess(rejected["coverage"], 1)
        self.assertEqual(rejected["failures"][0]["failure_reason"], "fixed_anchor_feature_unresolved")
        with self.assertRaisesRegex(AnalysisError, "union aborts"):
            bind_calibration_reports(base, corrupted, policy_id="must-not-bind-aborting-union")

    def test_frozen_zero_anchor_profiles_compile_to_static_codec_maps_without_policy(self):
        from reelcache.online_rank import RankPredictorProfile, profile_builder
        from reelcache.types import CodecSpec, MatrixKey

        profiles = {}
        for target, value in (("K", 3.25), ("V", .25)):
            profiles[target] = profile_builder({
                "cell": "causvid480", "target": target, "tau": .95,
                "history_regime": "later", "execution_mode": "causal_lagged",
                "layer_template": [value + 7 * (layer % 2) for layer in range(30)],
                "detector_baselines": {"startup": [None] * 30, "later": [None] * 30},
                "anchor_layers": [], "rank_margin": 1, "rank_buckets": [1, 4, 8, 16, 1536],
                "policy_kind": "fixed_layerwise", "uniform_rank": None, "training_groups": ["g0", "g1"],
            })
        frozen = {target: profile.to_dict() for target, profile in profiles.items()}
        before = copy.deepcopy(frozen)
        with mock.patch("reelcache.online_rank.RankPreparation", side_effect=AssertionError("not an adaptive policy")), \
                mock.patch("torch.linalg.svd", side_effect=AssertionError("static compilation cannot fit")):
            ranks = static_layer_ranks(frozen)
        self.assertEqual(frozen, before)
        self.assertEqual(ranks["K"], [8, 16] * 15)
        self.assertEqual(ranks["V"], [4, 16] * 15)
        specs = {MatrixKey(layer, target): CodecSpec(kind="svd", rank=rank)
                 for target, values in ranks.items() for layer, rank in enumerate(values)}
        self.assertEqual(len(specs), 60)
        self.assertTrue(all(spec.rank == ranks[key.target][key.layer] for key, spec in specs.items()))
        self.assertEqual(static_layer_ranks(profiles), ranks)
        with self.assertRaisesRegex(AnalysisError, "exactly"):
            static_layer_ranks({"K": profiles["K"]})
        changed = copy.deepcopy(frozen)
        changed["V"]["history_regime"] = "startup"
        with self.assertRaisesRegex(AnalysisError, "scope"):
            static_layer_ranks(changed)
        changed = copy.deepcopy(frozen)
        changed["K"].update(policy_kind="anchors", anchor_layers=[0],
                            detector_baselines={"startup": [4] * 30, "later": [4] * 30})
        with self.assertRaisesRegex(AnalysisError, "not adaptive anchors"):
            static_layer_ranks(changed)
        self.assertIsInstance(RankPredictorProfile.from_dict(frozen["K"]), RankPredictorProfile)

    def test_calibration_cli_uses_real_deployment_callbacks_and_writes_profiles(self):
        from reelcache.factorization import DetectorSpec

        rows, expected = census()
        recipe = DetectorSpec()
        input_path, detector_path = self.directory / "calibration_input.json", self.directory / "detectors.json"
        write_json(input_path, {"schema": OBSERVATION_SCHEMA, "expected": expected, "observations": rows})
        write_json(detector_path, {"schema": "reelcache_rank_detector_observations_v1",
                                  "detector_spec": recipe.to_dict(), "detector_sha256": recipe.fingerprint,
                                  "observations": detector_fixture(rows)})
        spec = importlib.util.spec_from_file_location("rank_calibrate_fixture", Path("scripts/rank_calibrate.py"))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        report = module.build_calibration_report(
            input_path, detector_path, self.directory / "output", cell="causvid480", target="K", tau=.95,
            history_regime="startup", execution_mode="causal_lagged", rank_buckets=(1, 4, 8, 16, 32, 1536),
            anchor_candidate_layers=(0, 1, 2, 3), margins=(0,), uniform_quantiles=(1.0,),
        )
        self.assertTrue(report["using_deployment_predictor"])
        self.assertEqual(report["detector_spec"], recipe.to_dict())
        frozen = read_json(self.directory / "output/profiles.json")
        self.assertTrue(all(
            row["online_profile"]["schema_version"] == "reelcache.rank_predictor_profile.v1"
            for row in frozen["profiles"]
        ))
        self.assertTrue((self.directory / "output/selected_cv_predictions.csv").is_file())
        bad_document = self.directory / "heldout_inventory.json"
        write_json(bad_document, {
            "schema": OBSERVATION_SCHEMA, "expected": [{**expected[0], "split": "heldout"}],
            "observation_files": [{"path": "must-not-open.json", "sha256": "0" * 64}],
        })
        with self.assertRaisesRegex(AnalysisError, "no spectrum files were opened"):
            module.build_calibration_report(bad_document, detector_path, self.directory / "bad")


class OracleSummaryAdapterTests(OwnedDirectoryTest):
    def test_adapter_revalidates_source_binding_and_preserves_cdf_bounds(self):
        from reelcache.spectra import SpectrumResult

        row = observation()
        identity = {name: row[name] for name in (*IDENTITY_FIELDS, *GEOMETRY_FIELDS, "evidence_kind")}
        source = types.SimpleNamespace(
            shape=(4680, 1536), target="K", dtype="bfloat16", key_space="pre_rope",
            representation="original", provenance=dict(identity), tensor_sha256="1" * 64,
            retention="spectrum-only", identity="fixture-original",
        )
        source.to_dict = lambda: {"provenance": copy.deepcopy(source.provenance)}
        payload = {
            "schema_version": "mock-spectrum",
            "diagnostics": {"gram_trace_scaled": 100.0, "symmetry_error_scaled": 0.0,
                            "negative_eigenvalue_mass_scaled": 0.0, "mean_error_bound_scaled": 0.0,
                            "norm_error_bound_scaled": 0.0},
            "solver": {"algorithm": "mock"},
        }

        def rank_at(tau):
            bracket = copy.deepcopy(row["r95" if tau == .95 else "r99"])
            bracket.update(
                tau=tau, rank=bracket["lower"], status="exact", complete_endpoint_used=False,
                diagnostic="fixture",
                cdf_before_lower_bounds=[bracket["cdf_before_lower"]] * 2,
                cdf_at_upper_bounds=[bracket["cdf_at_upper"]] * 2,
            )
            bracket.pop("cdf_tolerance")
            return types.SimpleNamespace(to_dict=lambda: bracket)

        result = types.SimpleNamespace(
            source=source, energy=100.0, energy_scaled=100.0, scale_exponent=0,
            dc_energy_fraction=.1, mean_energy_scaled=10.0, zero_energy=False,
            eigenvalues_scaled=(0,) * 1536,
            solver=types.SimpleNamespace(algorithm="mock"), rank_at=rank_at,
        )
        def persist(validated, path):
            self.assertIs(validated, result)
            encoded = json.dumps(payload, allow_nan=False, sort_keys=True, separators=(",", ":")).encode()
            with Path(path).open("xb") as stream:
                stream.write(encoded)
            return canonical_sha256(payload)

        with mock.patch.object(SpectrumResult, "from_dict", return_value=result) as validate, \
                mock.patch("reelcache.spectra.write_spectrum_json", side_effect=persist) as persist_spectrum:
            adapted = observation_from_spectrum(payload, identity=identity, spectrum_sha256="2" * 64)
            validate.assert_called_once_with(payload)
            self.assertEqual(adapted["r95"]["cdf_before_lower_bounds"], [.94, .94])
            self.assertEqual(adapted["oracle_sha256"], canonical_sha256(payload))
            receipt = write_spectrum_observation(
                payload, identity=identity, spectrum_path=self.directory / "full-spectrum.json",
                observation_path=self.directory / "observation.json",
            )
            persist_spectrum.assert_called_once_with(result, self.directory / "full-spectrum.json")
            self.assertEqual(receipt["spectrum_sha256"], file_sha256(self.directory / "full-spectrum.json"))
            persisted = read_json(self.directory / "observation.json")
            self.assertEqual(persisted["observation"]["spectrum_sha256"], receipt["spectrum_sha256"])
            self.assertEqual(read_json(self.directory / "full-spectrum.json"), payload)
            self.assertNotIn("eigenvalues_scaled", receipt["observation"])
            document_path = self.directory / "census.json"
            write_json(document_path, {
                "schema": OBSERVATION_SCHEMA, "expected": [identity],
                "observation_files": [{"path": "observation.json", "sha256": receipt["observation_sha256"]}],
            })
            loaded = load_observation_document(document_path)
            self.assertEqual(loaded["observations"][0]["oracle_sha256"], canonical_sha256(payload))
            self.assertEqual(len(loaded["source_observation_artifacts"]), 1)
            tampered = copy.deepcopy(persisted)
            tampered["observation"]["r95"].update(lower=5, upper=5, rank=5)
            write_json(self.directory / "tampered.json", tampered)
            write_json(self.directory / "tampered_census.json", {
                "schema": OBSERVATION_SCHEMA, "expected": [identity],
                "observation_files": [{"path": "tampered.json", "sha256": file_sha256(self.directory / "tampered.json")}],
            })
            with self.assertRaisesRegex(AnalysisError, "contradicts"):
                load_observation_document(self.directory / "tampered_census.json")
            with self.assertRaises(FileExistsError):
                write_spectrum_observation(
                    payload, identity=identity, spectrum_path=self.directory / "full-spectrum.json",
                    observation_path=self.directory / "observation.json",
                )
            source.provenance["seed"] = 99
            with self.assertRaisesRegex(AnalysisError, "seed"):
                observation_from_spectrum(payload, identity=identity, spectrum_sha256="2" * 64)
            (self.directory / "full-spectrum.json").write_text('{"changed": true}')
            with self.assertRaisesRegex(AnalysisError, "spectrum artifact checksum"):
                load_observation_document(document_path)


def magi_description_fixture():
    from reelcache.types import MatrixKey, ModelIdentity, TensorContext

    model = ModelIdentity("magi-fixture", "fixture", 48, 1024, "tokens_channels", "bfloat16")
    context = TensorContext(model, MatrixKey(0, "K"), 0, 6, 45, 80, "pre_rope", source_dtype="float32")
    return {
        "schema": "reelcache.magi_original_clean.v1", "tensor_context": context.to_dict(),
        "cell_id": "magi720", "group_id": "fixture-group", "split": "calibration", "seed": 7,
        "request_id": "fixture-request", "chunk_id": "clean0", "branch_id": "fixture-branch",
        "arm_id": "fixture-raw",
        "fixture": True, "native_execution": False,
        "history": [], "history_head": None, "clean_timestep": .9999,
        "actual_dtype": "float32", "source_sha256": "a" * 64,
        "shape": [21600, 1024], "source_layout": "owner_assembled_original_token_rows",
        "global_channels": 1024, "token_positions": [0, 21600], "inverse_rope_source": False,
        "rotary_sha256": ["b" * 64, "c" * 64],
        "source_hash_format": "sha256-native-contiguous-bytes-no-header-v1",
        "row_shard_hash_format": "sha256-native-contiguous-bytes-no-header-v1",
        "source_hash_byte_order": sys.byteorder, "row_shard_hash_byte_order": sys.byteorder,
        "row_shards": [
            {"rank": rank, "row_start": rank * 10800, "row_end": (rank + 1) * 10800,
             "channels": 1024, "dtype": "float32", "sha256": str(rank + 1) * 64}
            for rank in range(2)
        ],
        "protocol_id": "d" * 64, "asset_id": "e" * 64,
        "launch_closure_id": "f" * 64, "config_id": "0" * 64,
    }


class NativeMagiObserverAdapterTests(OwnedDirectoryTest):
    def test_description_binds_original_domain_and_keeps_hash_algorithms_separate(self):
        description = magi_description_fixture()
        source, identity = native_magi_source(
            description, case_id="paired-case", trajectory_id="current-arm",
            evidence_kind="synthetic_fixture",
        )
        self.assertEqual(source.shape, (21600, 1024))
        self.assertEqual((source.dtype, source.key_space), ("float32", "pre_rope"))
        self.assertIsNone(source.tensor_sha256)
        self.assertEqual(source.provenance["source_sha256"], description["source_sha256"])
        self.assertEqual(identity["native_source_sha256"], description["source_sha256"])
        self.assertEqual(identity["case_id"], "paired-case")
        self.assertEqual(identity["trajectory_id"], "current-arm")
        self.assertEqual(identity["boundary"], 0)
        self.assertEqual(identity["capture_sha256"], canonical_sha256(description))
        self.assertEqual(source.provenance["protocol_id"], description["protocol_id"])
        self.assertEqual(source.provenance["arm_id"], description["arm_id"])
        self.assertEqual(source.provenance["cell_id"], "magi720")
        self.assertEqual(source.provenance["group_id"], "fixture-group")
        self.assertEqual(source.retention, "spectrum-only")

    def test_bad_post_rope_dtype_row_shards_and_unlabelled_hashes_fail(self):
        original = magi_description_fixture()
        variants = []
        post_rope = copy.deepcopy(original)
        post_rope["tensor_context"]["key_space"] = "post_rope"
        variants.append(post_rope)
        wrong_dtype = copy.deepcopy(original)
        wrong_dtype["tensor_context"]["source_dtype"] = "bfloat16"
        variants.append(wrong_dtype)
        for field, value in (
            ("inverse_rope_source", True), ("global_channels", 512), ("shape", [10800, 1024]),
            ("clean_timestep", 1.0), ("source_hash_format", "reelcache.original-row-major.v1"),
            ("token_positions", [1, 21601]), ("history_head", "uncommitted-head"),
        ):
            variants.append({**copy.deepcopy(original), field: value})
        for field, value in (("row_start", 1), ("channels", 512), ("dtype", "bfloat16"), ("rank", 1)):
            row = copy.deepcopy(original)
            row["row_shards"][0][field] = value
            variants.append(row)
        row = copy.deepcopy(original)
        row["row_shards"].pop()
        variants.append(row)
        for description in variants:
            with self.subTest(description=description), self.assertRaises(AnalysisError):
                native_magi_source(description, case_id="case", trajectory_id="trajectory",
                                   evidence_kind="synthetic_fixture")

    def test_callback_solves_once_and_persists_separate_offline_cost(self):
        description = magi_description_fixture()
        matrix = types.SimpleNamespace(shape=(21600, 1024), dtype="torch.float32")
        result = types.SimpleNamespace(
            solver=types.SimpleNamespace(decomposition_count=1,
                                         work_counts=lambda: {"decomposition_count": 1, "eigvalsh_count": 1}),
            work_counts=lambda: {"decomposition_count": 1, "fixture_logical_gram_count": 1},
            timings=None,
        )
        receipt = {"spectrum_sha256": "1" * 64, "observation_sha256": "2" * 64}
        with mock.patch("reelcache.spectra.exact_spectrum", return_value=result) as oracle, \
                mock.patch("reelcache.rank_analysis.write_spectrum_observation", return_value=receipt) as writer:
            output = observe_magi_spectrum(
                description, matrix, case_id="case", trajectory_id="trajectory",
                spectrum_path=self.directory / "spectrum.json", observation_path=self.directory / "observation.json",
                telemetry_path=self.directory / "offline-cost.json", tile_rows=16, max_workspace_bytes=1 << 20,
                evidence_kind="synthetic_fixture",
            )
        oracle.assert_called_once()
        self.assertIs(oracle.call_args.args[0], matrix)
        self.assertEqual(oracle.call_args.kwargs["max_workspace_bytes"], 1 << 20)
        self.assertIsNone(oracle.call_args.kwargs["source"].tensor_sha256)
        writer.assert_called_once()
        self.assertIs(writer.call_args.args[0], result)
        self.assertEqual(writer.call_args.kwargs["identity"]["evidence_kind"], "synthetic_fixture")
        telemetry = read_json(self.directory / "offline-cost.json")
        self.assertEqual(telemetry["new_decomposition_count"], 1)
        self.assertEqual(telemetry["recorded_oracle_work_counts"]["fixture_logical_gram_count"], 1)
        self.assertFalse(telemetry["included_in_online_selector_cost"])
        self.assertEqual(output["telemetry_sha256"], file_sha256(self.directory / "offline-cost.json"))

    def test_bound_existing_result_is_reused_without_second_solve(self):
        from reelcache.spectra import SpectrumResult

        description = magi_description_fixture()
        source, _ = native_magi_source(
            description, case_id="case", trajectory_id="trajectory", evidence_kind="synthetic_fixture",
        )
        result = types.SimpleNamespace(
            source=dataclasses.replace(
                source, tensor_sha256="3" * 64, identity="policy-source-identity",
                provenance={**source.to_dict()["provenance"], "policy_hash": "7" * 64,
                            "original_bytes_sha256": description["source_sha256"]},
            ),
            solver=types.SimpleNamespace(work_counts=lambda: {"decomposition_count": 1}),
            work_counts=lambda: {"decomposition_count": 1, "fixture_logical_gram_count": 1},
            timings=None,
        )
        matrix = types.SimpleNamespace(shape=(21600, 1024), dtype="torch.float32")
        with mock.patch.object(SpectrumResult, "from_dict", return_value=result), \
                mock.patch("reelcache.spectra.exact_spectrum") as oracle, \
                mock.patch("reelcache.rank_analysis.write_spectrum_observation", return_value={
                    "spectrum_sha256": "1" * 64, "observation_sha256": "2" * 64,
                }):
            options = dict(
                trajectory_id="trajectory", spectrum_path=self.directory / "spectrum.json",
                observation_path=self.directory / "observation.json", tile_rows=16,
                max_workspace_bytes=1 << 20, evidence_kind="synthetic_fixture",
                existing_result={"mock": "source-bound full spectrum"},
            )
            output = observe_magi_spectrum(description, matrix, case_id="case", **options)
            self.assertTrue(output["offline_oracle_telemetry"]["reused_existing_result"])
            self.assertEqual(output["offline_oracle_telemetry"]["new_decomposition_count"], 0)
            with self.assertRaisesRegex(AnalysisError, "not bound"):
                observe_magi_spectrum(description, matrix, case_id="different-case", **options)
            with self.assertRaisesRegex(AnalysisError, "actual original"):
                observe_magi_spectrum(
                    description, types.SimpleNamespace(shape=(10800, 1024), dtype="torch.float32"),
                    case_id="case", **options,
                )
            if sys.byteorder == "little":
                result.source = dataclasses.replace(
                    result.source, provenance={**result.source.to_dict()["provenance"],
                                               "original_bytes_sha256": "f" * 64},
                )
                with self.assertRaisesRegex(AnalysisError, "measured raw-byte hash"):
                    observe_magi_spectrum(description, matrix, case_id="case", **options)
        oracle.assert_not_called()


class ExactPolicyStudyExportTests(OwnedDirectoryTest):
    def test_prebound_exact_decision_exports_same_spectrum_without_another_eigh(self):
        import torch
        from reelcache.online_rank import RankPolicy, RankPreparation
        from reelcache.types import MatrixKey, ModelIdentity, TensorContext

        cell = types.SimpleNamespace(tokens=4, channels=2, layers=1,
                                     key_dtype="float32", value_dtype="float32")
        model = ModelIdentity("cpu-fixture", "fixture", 1, 2, "tokens_channels", "float32")
        keys = (MatrixKey(0, "K"), MatrixKey(0, "V"))
        policy = RankPolicy(
            policy_id="exact-cpu-fixture", model=model, geometry=(1, 2, 2),
            source_dtypes={"K": "float32", "V": "float32"},
            source_fingerprint="1" * 64, geometry_fingerprint="2" * 64,
            history_fingerprint="3" * 64, calibration_fingerprint="4" * 64,
            implementation_fingerprint="5" * 64, energy_target=.95,
            template_ranks={key: 2.0 for key in keys}, allowed_ranks={key: (1, 2) for key in keys},
            solver="exact_oracle", tile_rows=2, solver_workspace_bytes=0,
            source_limit_bytes=4096, workspace_limit_bytes=1 << 20,
        )
        # A private tiny test geometry, not a substituted native study cell.
        with mock.patch("reelcache.rank_analysis.CELLS", {"cpu-fixture": cell}):
            identities = {
                key: rank_observation_identity(
                    case_id="same-fixture-case", group_id="fixture-group", split="calibration",
                    seed=0, cell_id="cpu-fixture", layer=key.layer, target=key.target, boundary=0,
                    trajectory_id="fixture-current-trajectory", evidence_kind="synthetic_fixture",
                    source_bindings={"fixture": True, "native_execution": False, "parent_head": None},
                )
                for key in keys
            }
            preparation = RankPreparation(
                policy, chunk_id="fixture-clean0", parent_head=None, token_start=0,
                source_provenance=identities,
            )
            context = TensorContext(model, keys[0], 0, 1, 2, 2, "pre_rope", source_dtype="float32")
            matrix = torch.tensor([[3., 0.], [0., 1.], [0., 0.], [0., 0.]], dtype=torch.float32)
            try:
                with mock.patch("torch.linalg.eigh", wraps=torch.linalg.eigh) as eigh:
                    prepared = preparation.prepare_matrix(matrix, context)
                    observation = prepared.decision.to_dict()["metrics"]["oracle_observation"]
                    self.assertEqual(eigh.call_count, 1)
                    self.assertEqual(observation["source"]["provenance"]["case_id"], "same-fixture-case")
                    self.assertEqual(
                        observation["source"]["provenance"]["original_bytes_sha256"],
                        prepared.decision.input_sha256,
                    )
                    self.assertNotEqual(observation["source"]["tensor_sha256"], prepared.decision.input_sha256)
                    receipt = write_spectrum_observation(
                        observation, identity=identities[keys[0]],
                        spectrum_path=self.directory / "same-solve.json",
                        observation_path=self.directory / "numeric.json",
                    )
                    self.assertEqual(eigh.call_count, 1)
                    self.assertEqual(receipt["observation"]["r95"]["upper"], 2)
                    self.assertEqual(len(read_json(self.directory / "same-solve.json")["eigenvalues_scaled"]), 2)
                    for change in ({"case_id": "different-case"}, {"n_rows": 8}):
                        with self.assertRaises(AnalysisError):
                            write_spectrum_observation(
                                observation, identity={**identities[keys[0]], **change},
                                spectrum_path=self.directory / "must-not-write-spectrum.json",
                                observation_path=self.directory / "must-not-write-observation.json",
                            )
                    self.assertFalse((self.directory / "must-not-write-spectrum.json").exists())
                    self.assertFalse((self.directory / "must-not-write-observation.json").exists())
                    self.assertEqual(eigh.call_count, 1)
            finally:
                preparation.close()


class NativeWorkerRawCaptureTests(OwnedDirectoryTest):
    """Actual worker callbacks with tiny, explicitly non-native CPU contracts."""

    def _check_artifacts(self, directory, cell, group, protocol_id, closure_id):
        from reelcache import study
        from reelcache.spectra import SpectrumResult

        numeric_files = []
        for path in directory.rglob("*.json"):
            value = read_json(path)
            if isinstance(value, dict) and value.get("schema") == "reelcache_rank_observation_v1":
                numeric_files.append((path, value["observation"]))
        self.assertEqual(len(numeric_files), 1)
        path, row = numeric_files[0]
        spectrum_path = (path.parent / row["spectrum_path"]).resolve()
        self.assertTrue(spectrum_path.is_relative_to(directory.resolve()))
        self.assertEqual(file_sha256(spectrum_path), row["spectrum_sha256"])
        result = SpectrumResult.from_dict(read_json(spectrum_path))
        provenance = result.source.to_dict()["provenance"]
        required = {
            "protocol_id": protocol_id, "launch_closure_id": closure_id,
            "cell_id": cell.cell_id, "group_id": group, "seed": 0, "boundary": 0,
            "layer": 0, "target": "K", "split": "calibration",
            "fixture": True, "native_execution": False,
        }
        for name, expected in required.items():
            self.assertEqual(provenance[name], expected, name)
        for name, value in row["observation_identity"].items():
            self.assertEqual(provenance[name], value, name)
        self.assertEqual(row["evidence_kind"], "synthetic_fixture")
        document_path = directory / "numeric-input.json"
        write_json(document_path, {
            "schema": OBSERVATION_SCHEMA, "expected": [row["observation_identity"]],
            "observation_files": [{"path": str(path.relative_to(directory)), "sha256": file_sha256(path)}],
        })
        self.assertEqual(load_observation_document(document_path)["observations"][0]["input_sha256"],
                         result.source.tensor_sha256)
        proof = {
            "launch_closure_id": closure_id,
            "fixture": False, "native_execution": True,
            "observations": [{
                "cell_id": cell.cell_id, "group_id": group, "seed": 0, "boundary": 0,
                "layer": 0, "target": "K",
                "spectrum": {"path": str(spectrum_path), "sha256": row["spectrum_sha256"]},
            }],
        }
        protocol = {"protocol_id": protocol_id, "prompts": [{"group_id": group, "split": "calibration"}],
                    "seeds": [0], "clean_boundaries": [0]}
        with self.assertRaisesRegex(study.IntegrityError, "does not substantiate"):
            study._census_proof(proof, protocol, cell)

    def test_actual_magi_raw_callback_prebinds_then_writes_full_and_numeric_files(self):
        import torch
        from reelcache.adapters.magi import _tensor_hash, _tensor_hash_format, NATIVE_TENSOR_HASH_FORMAT
        from reelcache.adapters.magi_native import _StudyBackend
        from reelcache.spectra import exact_spectrum
        from reelcache.study import CELLS
        from reelcache.types import MatrixKey, ModelIdentity, TensorContext

        cell = dataclasses.replace(CELLS["magi720"], frames=1, token_height=2, token_width=3,
                                   layers=1, channels=8)
        model = ModelIdentity("magi-cpu-fixture", "fixture", 1, 8, "tokens_channels", "bfloat16")
        context = TensorContext(model, MatrixKey(0, "K"), 0, 1, 2, 3, "pre_rope", source_dtype="float32")
        matrix = torch.eye(6, 8, dtype=torch.float32)
        description = magi_description_fixture()
        description.update(
            tensor_context=context.to_dict(), shape=[6, 8], global_channels=8, token_positions=[0, 6],
            case_id="paired-fixture", trajectory_id="current-fixture", seed=0, source_sha256=_tensor_hash(matrix),
            source_hash_format=NATIVE_TENSOR_HASH_FORMAT, source_hash_descriptor=_tensor_hash_format(matrix.dtype),
            row_shard_hash_format=NATIVE_TENSOR_HASH_FORMAT,
            row_shards=[
                {"rank": rank, "row_start": 3 * rank, "row_end": 3 * (rank + 1), "channels": 8,
                 "dtype": "float32", "sha256": _tensor_hash(matrix[3 * rank:3 * (rank + 1)]),
                 "hash_format": _tensor_hash_format(matrix.dtype)}
                for rank in range(2)
            ],
        )
        backend = _StudyBackend.__new__(_StudyBackend)
        backend.torch, backend.cell = torch, cell
        backend.observation_evidence_kind = "synthetic_fixture"
        backend.verified = {"batch": {
            "retain_originals": [], "spectrum_tile_rows": 2, "spectrum_workspace_bytes": 1 << 20,
        }}
        backend.active = {
            "request": types.SimpleNamespace(
                request_id=description["request_id"], case_id="paired-fixture", policy=None,
                arm="raw", arm_id=description["arm_id"], observe_sources=True,
            ),
            "directory": self.directory, "trajectory_id": "current-fixture",
        }

        def assert_prebound(tensor, *, source, **kwargs):
            self.assertIs(tensor, matrix)
            self.assertEqual(source.provenance["case_id"], "paired-fixture")
            self.assertEqual(source.provenance["boundary"], 0)
            self.assertEqual(source.provenance["target"], "K")
            self.assertEqual(source.provenance["layer"], 0)
            self.assertTrue(source.provenance["fixture"])
            self.assertFalse(source.provenance["native_execution"])
            return exact_spectrum(tensor, source=source, **kwargs)

        with mock.patch("reelcache.rank_analysis.CELLS", {"magi720": cell}), \
                mock.patch("reelcache.spectra.exact_spectrum", side_effect=assert_prebound) as oracle:
            backend._observe(description, matrix)
            oracle.assert_called_once()
            self._check_artifacts(self.directory, cell, description["group_id"],
                                  description["protocol_id"], description["launch_closure_id"])

    def test_actual_causvid_raw_callback_prebinds_and_does_not_qualify_fixture_as_native(self):
        import torch
        from examples import causvid_rank_worker as worker
        from reelcache.spectra import exact_spectrum
        from reelcache.study import CELLS
        from reelcache.types import MatrixKey, ModelIdentity, TensorContext

        cell = dataclasses.replace(CELLS["causvid480"], frames=1, token_height=2, token_width=2,
                                   layers=1, channels=8)
        model = ModelIdentity("causvid-cpu-fixture", "fixture", 1, 8, "tokens_channels", "bfloat16")
        context = TensorContext(model, MatrixKey(0, "K"), 0, 1, 2, 2, "pre_rope")
        matrix = torch.eye(4, 8, dtype=torch.bfloat16)
        engine = worker.CausVidRankWorker.__new__(worker.CausVidRankWorker)
        engine.model, engine.output_root = model, self.directory
        engine.provenance = {"protocol_id": "a" * 64, "launch_closure_id": "b" * 64}
        engine.observation_evidence_kind, engine.observer_factory = "synthetic_fixture", None
        document = {
            "capture": {"spectra": True, "tile_rows": 2, "retain": []},
            "memory_budgets": {"source_archive_bytes": 0, "spectrum_workspace_bytes": 1 << 20},
            "request_id": "raw-request", "case_id": "paired-fixture", "trajectory_id": "raw-trajectory",
            "arm_id": "fixture-raw", "split": "calibration", "cell_id": "causvid480",
            "prompt_group_id": "fixture-group", "text": "CPU fixture; not model output",
            "seed": 0, "conditioning_identity": "fixture-conditioning", "history_mode": "reload",
            "rank_policy": None,
        }
        directory = self.directory / "request"
        directory.mkdir()
        runtime = types.SimpleNamespace(branch_info=lambda branch: {"head_id": None, "token_end": 0})
        rows = []

        def assert_prebound(tensor, *, source, **kwargs):
            self.assertIs(tensor, matrix)
            self.assertEqual(source.provenance["case_id"], "paired-fixture")
            self.assertEqual(source.provenance["boundary"], 0)
            self.assertEqual(source.provenance["group_id"], "fixture-group")
            self.assertEqual(source.provenance["target"], "K")
            self.assertTrue(source.provenance["fixture"])
            self.assertFalse(source.provenance["native_execution"])
            return exact_spectrum(tensor, source=source, **kwargs)

        with mock.patch("reelcache.rank_analysis.CELLS", {"causvid480": cell}), \
                mock.patch.object(worker, "study_cell", return_value=cell), \
                mock.patch("reelcache.spectra.exact_spectrum", side_effect=assert_prebound) as oracle:
            engine._observer(document, directory, runtime, rows)(context, matrix)
            oracle.assert_called_once()
            self.assertEqual(len(rows), 1)
            self._check_artifacts(directory, cell, "fixture-group", "a" * 64, "b" * 64)

    def test_actual_causvid_exact_export_uses_committed_decisions_without_an_observer_solve(self):
        import torch
        from examples import causvid_rank_worker as worker
        from reelcache.manager import KVRuntime
        from reelcache.online_rank import RankPolicy
        from reelcache.study import CELLS
        from reelcache.types import MatrixKey, ModelIdentity, TensorContext

        cell = dataclasses.replace(CELLS["causvid480"], frames=1, token_height=2, token_width=2,
                                   layers=1, channels=8)
        model = ModelIdentity("causvid-cpu-fixture", "fixture", 1, 8, "tokens_channels", "bfloat16")
        keys = (MatrixKey(0, "K"), MatrixKey(0, "V"))
        policy = RankPolicy(
            policy_id="exact-worker-fixture", model=model, geometry=(1, 2, 2),
            source_dtypes={"K": "bfloat16", "V": "bfloat16"},
            source_fingerprint="1" * 64, geometry_fingerprint="2" * 64,
            history_fingerprint="3" * 64, calibration_fingerprint="4" * 64,
            implementation_fingerprint="5" * 64, energy_target=.95,
            template_ranks={key: 4.0 for key in keys}, allowed_ranks={key: (1, 2, 4) for key in keys},
            solver="exact_oracle", tile_rows=2, solver_workspace_bytes=0,
            source_limit_bytes=4096, workspace_limit_bytes=1 << 20,
        )
        engine = worker.CausVidRankWorker.__new__(worker.CausVidRankWorker)
        engine.model, engine.output_root = model, self.directory
        engine.provenance = {"protocol_id": "a" * 64, "launch_closure_id": "b" * 64}
        engine.observation_evidence_kind, engine.observer_factory = "synthetic_fixture", None
        engine.config = types.SimpleNamespace(chunk_tokens=4)
        document = {
            "capture": {"spectra": False, "tile_rows": 2, "retain": []},
            "memory_budgets": {"source_archive_bytes": 0, "spectrum_workspace_bytes": 1 << 20},
            "request_id": "exact-request", "case_id": "paired-fixture", "trajectory_id": "exact-trajectory",
            "arm_id": "fixture-exact", "split": "calibration", "cell_id": "causvid480",
            "prompt_group_id": "fixture-group", "text": "CPU fixture; not model output",
            "seed": 0, "conditioning_identity": "fixture-conditioning", "history_mode": "reload",
            "rank_policy": policy.to_dict(),
        }
        directory = self.directory / "request"
        directory.mkdir()
        runtime = KVRuntime(self.directory / "store", model, max_resident_bytes=1 << 20, max_store_bytes=1 << 24)
        try:
            runtime.create_branch("request")
            contexts = {
                key: TensorContext(model, key, 0, 1, 2, 2, "pre_rope" if key.target == "K" else "none")
                for key in keys
            }
            matrices = {key: torch.eye(4, 8, dtype=torch.bfloat16) for key in keys}
            captures = []
            with mock.patch("reelcache.rank_analysis.CELLS", {"causvid480": cell}), \
                    mock.patch.object(worker, "study_cell", return_value=cell):
                observe = engine._observer(document, directory, runtime, captures)
                with mock.patch("reelcache.spectra.exact_spectrum", side_effect=AssertionError("observer duplicated oracle")):
                    for key in keys:
                        observe(contexts[key], matrices[key])
                self.assertFalse(list(directory.rglob("*.json")))
                provenance = {
                    MatrixKey.from_dict(row["context"]["key"]): row["source_provenance"] for row in captures
                }
                previous_head = runtime.branch_info("request")["head_id"]
                with mock.patch("torch.linalg.eigh", wraps=torch.linalg.eigh) as eigh:
                    runtime.seal_chunk(
                        "request", "chunk-000000", matrices, contexts,
                        rank_policy=policy, source_provenance=provenance,
                    )
                    self.assertNotEqual(runtime.branch_info("request")["head_id"], previous_head)
                    self.assertEqual(eigh.call_count, 2)
                    originals = {
                        key: runtime.history_rank_decisions("request", key)[-1].to_dict()["metrics"]["oracle_observation"]
                        for key in keys
                    }
                    engine._export_policy_oracles(document, directory, runtime, policy, 0, captures)
                    self.assertEqual(eigh.call_count, 2)
                numeric = [
                    (path, read_json(path)["observation"]) for path in directory.rglob("*.json")
                    if isinstance(read_json(path), dict) and
                    read_json(path).get("schema") == "reelcache_rank_observation_v1"
                ]
                self.assertEqual(len(numeric), 2, "post-publication export must use the paired analysis writer")
                for path, row in numeric:
                    key = MatrixKey(row["layer"], row["target"])
                    self.assertEqual(row["case_id"], document["case_id"])
                    saved = read_json(path.parent / row["spectrum_path"])
                    self.assertEqual(saved, originals[key])
                    for name, value in row["observation_identity"].items():
                        self.assertEqual(saved["source"]["provenance"][name], value)
        finally:
            runtime.close()

    def test_actual_magi_exact_export_uses_prebound_committed_oracle_without_a_second_solve(self):
        import torch
        from reelcache.adapters.magi import _tensor_hash, _tensor_hash_format, NATIVE_TENSOR_HASH_FORMAT
        from reelcache.adapters.magi_native import _StudyBackend
        from reelcache.manager import KVRuntime
        from reelcache.online_rank import RankPolicy
        from reelcache.study import CELLS
        from reelcache.types import MatrixKey, ModelIdentity, TensorContext

        cell = dataclasses.replace(CELLS["magi720"], frames=1, token_height=2, token_width=3,
                                   layers=1, channels=8)
        model = ModelIdentity("magi-cpu-fixture", "fixture", 1, 8, "tokens_channels", "bfloat16")
        keys = (MatrixKey(0, "K"), MatrixKey(0, "V"))
        policy = RankPolicy(
            policy_id="exact-magi-worker-fixture", model=model, geometry=(1, 2, 3),
            source_dtypes={"K": "float32", "V": "bfloat16"},
            source_fingerprint="1" * 64, geometry_fingerprint="2" * 64,
            history_fingerprint="3" * 64, calibration_fingerprint="4" * 64,
            implementation_fingerprint="5" * 64, energy_target=.95,
            template_ranks={key: 6.0 for key in keys}, allowed_ranks={key: (1, 2, 4, 6) for key in keys},
            solver="exact_oracle", tile_rows=2, solver_workspace_bytes=0,
            source_limit_bytes=4096, workspace_limit_bytes=1 << 20,
        )
        directory = self.directory / "request"
        directory.mkdir()
        backend = _StudyBackend.__new__(_StudyBackend)
        backend.torch, backend.cell = torch, cell
        backend.observation_evidence_kind = "synthetic_fixture"
        backend.verified = {"batch": {
            "retain_originals": [], "spectrum_tile_rows": 2, "spectrum_workspace_bytes": 1 << 20,
        }}
        runtime = KVRuntime(self.directory / "store", model, max_resident_bytes=1 << 20, max_store_bytes=1 << 24)
        try:
            runtime.create_branch("request")
            backend.active = {
                "request": types.SimpleNamespace(
                    request_id="fixture-request", case_id="paired-fixture", policy=policy.to_dict(),
                    arm="adaptive", arm_id="fixture-exact", observe_sources=True,
                ),
                "directory": directory, "trajectory_id": "exact-trajectory", "runtime": runtime,
            }
            contexts, matrices, provenance = {}, {}, {}
            with mock.patch("reelcache.rank_analysis.CELLS", {"magi720": cell}):
                with mock.patch("reelcache.spectra.exact_spectrum", side_effect=AssertionError("observer duplicated oracle")):
                    for key in keys:
                        dtype = "float32" if key.target == "K" else "bfloat16"
                        context = TensorContext(model, key, 0, 1, 2, 3,
                                                "pre_rope" if key.target == "K" else "none", source_dtype=dtype)
                        matrix = torch.eye(6, 8, dtype=getattr(torch, dtype))
                        description = magi_description_fixture()
                        description.update(
                            tensor_context=context.to_dict(), shape=[6, 8], global_channels=8,
                            token_positions=[0, 6], actual_dtype=dtype, seed=0, chunk_id="0", branch_id="request",
                            case_id="paired-fixture", trajectory_id="exact-trajectory", arm_id="fixture-exact",
                            source_sha256=_tensor_hash(matrix), source_hash_format=NATIVE_TENSOR_HASH_FORMAT,
                            source_hash_descriptor=_tensor_hash_format(matrix.dtype),
                            row_shard_hash_format=NATIVE_TENSOR_HASH_FORMAT,
                            row_shards=[
                                {"rank": rank, "row_start": 3 * rank, "row_end": 3 * (rank + 1),
                                 "channels": 8, "dtype": dtype,
                                 "sha256": _tensor_hash(matrix[3 * rank:3 * (rank + 1)]),
                                 "hash_format": _tensor_hash_format(matrix.dtype)}
                                for rank in range(2)
                            ],
                        )
                        contexts[key], matrices[key] = context, matrix
                        provenance[key] = backend._observe(description, matrix)
                        self.assertEqual(provenance[key]["source_dtype"], dtype)
                        self.assertEqual(provenance[key]["target"], key.target)
                        self.assertTrue(provenance[key]["fixture"])
                        self.assertFalse(provenance[key]["native_execution"])
                self.assertEqual(set(backend.active["deferred_observations"]), set(keys))
                self.assertFalse(list(directory.rglob("*.json")), "no completed observation before core publication")
                previous_head = runtime.branch_info("request")["head_id"]
                with mock.patch("torch.linalg.eigh", wraps=torch.linalg.eigh) as eigh:
                    chunk_object = runtime.seal_chunk(
                        "request", "0", matrices, contexts, rank_policy=policy, source_provenance=provenance,
                    )
                    self.assertNotEqual(runtime.branch_info("request")["head_id"], previous_head)
                    self.assertEqual(eigh.call_count, 2)
                    originals = {
                        key: runtime.history_rank_decisions("request", key)[-1].to_dict()["metrics"]["oracle_observation"]
                        for key in keys
                    }
                    backend._publish_policy_observations(chunk_object)
                    self.assertEqual(eigh.call_count, 2)
                self.assertEqual(backend.active["deferred_observations"], {})
                numeric, indexes = [], []
                for path in directory.rglob("*.json"):
                    value = read_json(path)
                    if value.get("schema") == "reelcache_rank_observation_v1":
                        numeric.append((path, value["observation"]))
                    elif value.get("schema") == "reelcache.magi_clean_observation.v2":
                        indexes.append(value)
                self.assertEqual(len(numeric), 2)
                self.assertEqual(len(indexes), 2)
                for path, row in numeric:
                    key = MatrixKey(row["layer"], row["target"])
                    saved = read_json(path.parent / row["spectrum_path"])
                    self.assertEqual(saved, originals[key])
                    for name, value in row["observation_identity"].items():
                        self.assertEqual(saved["source"]["provenance"][name], value)
                for index in indexes:
                    self.assertTrue(index["shares_policy_decomposition"])
                    self.assertEqual(index["standalone_offline_decompositions"], 0)
                    self.assertFalse(index["exact_policy_decomposition_is_additional"])
                    self.assertEqual(index["offline_oracle_seconds"], 0.0)
        finally:
            runtime.close()


if __name__ == "__main__":
    unittest.main()
