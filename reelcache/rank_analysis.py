"""Numeric-only analysis for the source-bound online-rank study.

The caller owns the protocol and supplies an explicit expected observation
inventory.  This module does not capture tensors, load models, choose a corpus,
or authorize experiments.  Rank intervals remain intervals: plotted point
summaries use their conservative upper endpoints and say so explicitly.
"""

from __future__ import annotations

import csv
import dataclasses
import hashlib
import json
import math
import random
import statistics
import sys
import time
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from .study import CELLS, write_json_new


OBSERVATION_SCHEMA = "reelcache_rank_observations_v1"
SINGLE_OBSERVATION_SCHEMA = "reelcache_rank_observation_v1"
DIMENSIONS = (
    "subject_consistency", "background_consistency", "motion_smoothness",
    "dynamic_degree", "aesthetic_quality", "imaging_quality",
)
METRICS = ("ssim", *DIMENSIONS)
SPLITS = ("development", "calibration", "heldout")
IDENTITY_FIELDS = (
    "case_id", "prompt_group", "split", "seed", "cell", "layer", "target",
    "boundary", "trajectory_id",
)
GEOMETRY_FIELDS = ("n_rows", "n_channels", "n_layers", "source_dtype", "source_domain")
BYTE_DENOMINATORS = (
    "decoder_closure_bytes", "snapshot_closure_bytes", "filesystem_bytes",
)
POINT_POLICY = "conservative upper rank endpoint; intervals are not exact ranks"
RANGE_LABEL = "Observed min–max (rank brackets included), not a confidence interval"


class AnalysisError(ValueError):
    """Malformed, incomplete, mismatched or leakage-prone analysis input."""


def _integer(value: Any, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise AnalysisError(f"{name} must be an integer >= {minimum}")
    return value


def _finite(value: Any, name: str, minimum=None, maximum=None) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise AnalysisError(f"{name} must be a finite number")
    if not math.isfinite(value):
        raise AnalysisError(f"{name} must be finite")
    if minimum is not None and value < minimum:
        raise AnalysisError(f"{name} is below {minimum}")
    if maximum is not None and value > maximum:
        raise AnalysisError(f"{name} is above {maximum}")
    return float(value)


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AnalysisError(f"{name} must be a nonempty string")
    return value


def _digest(value: Any, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(
        char not in "0123456789abcdef" for char in value
    ):
        raise AnalysisError(f"{name} must be lowercase SHA256 hex")
    return value


def canonical_sha256(value: Any) -> str:
    try:
        data = json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":"))
    except (ValueError, TypeError) as error:
        raise AnalysisError("records must contain finite JSON values") from error
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: str | Path) -> Any:
    def constant(value):
        raise AnalysisError(f"nonfinite JSON constant: {value}")

    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise AnalysisError(f"duplicate JSON field: {key}")
            result[key] = value
        return result

    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream, parse_constant=constant, object_pairs_hook=pairs)


def write_json(path: str | Path, value: Any) -> None:
    """Create evidence, never overwrite a previously published output."""
    payload = json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(payload)


def write_csv(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise AnalysisError("cannot export an empty numerical table")
    keys = list(rows[0])
    if any(set(row) != set(keys) for row in rows):
        raise AnalysisError("CSV rows have different fields")
    canonical_sha256(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: json.dumps(value, allow_nan=False, sort_keys=True)
                if isinstance(value, (dict, list, tuple)) else value
                for key, value in row.items()
            })


def _identity(row: Mapping[str, Any]) -> tuple:
    try:
        for name in ("case_id", "prompt_group", "cell", "trajectory_id"):
            _text(row[name], name)
        if row["split"] not in SPLITS or row["target"] not in ("K", "V"):
            raise AnalysisError("unknown split or K/V target")
        for name in ("seed", "layer", "boundary"):
            _integer(row[name], name)
        for name in ("n_rows", "n_channels", "n_layers"):
            _integer(row[name], name, 1)
        geometry = tuple(row[name] for name in GEOMETRY_FIELDS[:3])
        if row["cell"] not in CELLS:
            raise AnalysisError("unknown parent-protocol cell ID")
        cell = CELLS[row["cell"]]
        if geometry != (cell.tokens, cell.channels, cell.layers):
            raise AnalysisError(f"geometry differs from parent study cell {row['cell']}: {geometry}")
        if row["layer"] >= row["n_layers"]:
            raise AnalysisError("layer lies outside the declared geometry")
        if row["source_dtype"] != (cell.key_dtype if row["target"] == "K" else cell.value_dtype):
            raise AnalysisError("source dtype differs from canonical study input")
        if row["source_domain"] != ("normalized_pre_rope" if row["target"] == "K" else "direct"):
            raise AnalysisError("only normalized pre-RoPE K and direct V are supported")
        return tuple(row[name] for name in IDENTITY_FIELDS)
    except KeyError as error:
        raise AnalysisError(f"missing identity field: {error.args[0]}") from error


def rank_observation_identity(
    *, case_id: str, group_id: str, split: str, seed: int, cell_id: str,
    layer: int, target: str, boundary: int, trajectory_id: str, evidence_kind: str,
    source_bindings: Mapping[str, Any] | None = None,
) -> dict:
    """Map parent request/group/cell identities; never infer a case from an arm request ID.

    Call this before capture and include the result in SpectrumSource.provenance.
    Native identities also require actual protocol/launch/request/arm and
    branch/parent-head bindings plus explicit native-execution/fixture flags.
    The parent still owns prompt selection,
    request validation, launch gates and corpus splits.
    """
    if cell_id not in CELLS:
        raise AnalysisError("unknown parent-protocol cell ID")
    if evidence_kind not in ("native", "synthetic_fixture"):
        raise AnalysisError("distinguish native observations from synthetic fixtures")
    cell = CELLS[cell_id]
    row = {
        "case_id": case_id, "prompt_group": group_id, "split": split, "seed": seed,
        "cell": cell_id, "cell_id": cell_id, "group_id": group_id,
        "layer": layer, "target": target, "boundary": boundary,
        "trajectory_id": trajectory_id, "evidence_kind": evidence_kind,
        "n_rows": cell.tokens, "n_channels": cell.channels, "n_layers": cell.layers,
        "source_dtype": cell.key_dtype if target == "K" else cell.value_dtype,
        "source_domain": "normalized_pre_rope" if target == "K" else "direct",
    }
    if source_bindings is not None:
        if not isinstance(source_bindings, Mapping) or any(
            name in row and value != row[name] for name, value in source_bindings.items()
        ):
            raise AnalysisError("extra source bindings cannot replace case/geometry identities")
        for name, value in source_bindings.items():
            _text(name, "source binding name")
            if name.endswith("_sha256"):
                _digest(value, name)
            if name in ("parent_head", "parent_trajectory_id") and value is not None:
                _text(value, name)
        row.update(source_bindings)
        canonical_sha256(row)
    _identity(row)
    if evidence_kind == "native":
        _native_capture_binding(row)
    elif ("fixture" in row and row["fixture"] is not True) or (
        "native_execution" in row and row["native_execution"] is not False
    ):
        raise AnalysisError("synthetic observation cannot claim native execution")
    return row


def _native_capture_binding(row: Mapping[str, Any]) -> None:
    try:
        _digest(row["protocol_id"], "protocol_id")
        _digest(row["launch_closure_id"], "launch_closure_id")
        for name in ("request_id", "arm_id", "branch_id"):
            _text(row[name], name)
        if row["fixture"] is not False or row["native_execution"] is not True:
            raise AnalysisError("native observation requires actual non-fixture execution bindings")
        if row["cell_id"] != row["cell"] or row["group_id"] != row["prompt_group"]:
            raise AnalysisError("native census ledger aliases differ from the bound cell/group")
        parent = row["parent_head"]
        if row["boundary"] == 0:
            if parent is not None:
                raise AnalysisError("startup observation cannot claim an existing parent head")
        else:
            _text(parent, "current native parent head")
        for name in ("history_head", "history_head_id"):
            if name in row and row[name] != parent:
                raise AnalysisError("native parent/history head bindings disagree")
    except KeyError as error:
        raise AnalysisError(f"native observation omits actual capture binding: {error.args[0]}") from error


def _check_inventory(rows: Sequence[Mapping[str, Any]], *, complete_layers: bool) -> dict:
    if not rows:
        raise AnalysisError("an explicit nonempty expected observation inventory is required")
    inventory, groups, cases, cells, trajectories = {}, {}, {}, {}, {}
    slices = defaultdict(set)
    case_boundaries = defaultdict(set)
    for row in rows:
        key = _identity(row)
        if key in inventory:
            raise AnalysisError(f"duplicate observation identity: {key}")
        inventory[key] = row
        group, case, cell = row["prompt_group"], row["case_id"], row["cell"]
        if group in groups and groups[group] != row["split"]:
            raise AnalysisError("a prompt group crosses calibration/heldout splits")
        groups[group] = row["split"]
        case_identity = tuple(row[name] for name in
                              ("prompt_group", "split", "seed", "cell", "trajectory_id"))
        if case in cases and cases[case] != case_identity:
            raise AnalysisError("case identity changes across observations")
        cases[case] = case_identity
        if row["trajectory_id"] in trajectories and trajectories[row["trajectory_id"]] != case:
            raise AnalysisError("a trajectory is aliased by different case IDs")
        trajectories[row["trajectory_id"]] = case
        geometry = tuple(row[name] for name in GEOMETRY_FIELDS[:3])
        if cell in cells and cells[cell] != geometry:
            raise AnalysisError("cell geometry changes across observations")
        cells[cell] = geometry
        slices[(case, row["boundary"])].add((row["layer"], row["target"]))
        case_boundaries[case].add(row["boundary"])
    if complete_layers:
        for (case, boundary), keys in slices.items():
            cell = cases[case][3]
            required = {(layer, target) for layer in range(cells[cell][2]) for target in ("K", "V")}
            if keys != required:
                raise AnalysisError(f"expected inventory omits layers/targets: {case}/{boundary}")
        # The numerical contract cannot recover an absent seed/boundary from data.
        # It can, however, reject an inconsistent inventory that weights groups unequally.
        layouts = defaultdict(dict)
        for case, identity in cases.items():
            group, split, seed, cell, _ = identity
            if min(case_boundaries[case]) != 0:
                raise AnalysisError("the first clean boundary (index zero) is missing")
            slot = layouts[(split, cell)].setdefault(group, {})
            if seed in slot:
                raise AnalysisError("multiple trajectories for the same group/cell/seed")
            slot[seed] = tuple(sorted(case_boundaries[case]))
        for groups_in_cell in layouts.values():
            if len({
                boundaries for seeds in groups_in_cell.values() for boundaries in seeds.values()
            }) != 1:
                raise AnalysisError("every prompt/seed must cover the same selected boundaries")
            patterns = [
                tuple(sorted(boundaries.values())) for boundaries in groups_in_cell.values()
            ]
            if any(pattern != patterns[0] for pattern in patterns[1:]):
                raise AnalysisError("groups must have the same declared seed/boundary coverage")
    return inventory


def dc_fraction_bounds(*, mean_energy: float, energy: float, mean_error: float, norm_error: float) -> dict:
    """Preserve m/E; bound the ratio with (m-dm)/(E+dE) and (m+dm)/(E-dE)."""
    mean_energy = _finite(mean_energy, "mean_energy", 0)
    energy = _finite(energy, "energy", 0)
    mean_error = _finite(mean_error, "mean_error", 0)
    norm_error = _finite(norm_error, "norm_error", 0)
    if energy == 0:
        if mean_energy != 0 or mean_error != 0 or norm_error != 0:
            raise AnalysisError("exact zero energy cannot have nonzero mean/energy uncertainty")
        return {"mean_dc_energy_fraction": None, "mean_dc_energy_fraction_bounds": None,
                "mean_dc_energy_fraction_error_bound": None}
    if energy <= norm_error:
        raise AnalysisError("DC ratio needs a strictly positive lower norm bound")
    nominal = mean_energy / energy
    lower = max(0.0, (mean_energy - mean_error) / (energy + norm_error))
    upper = min(1.0, (mean_energy + mean_error) / (energy - norm_error))
    if lower > upper:
        raise AnalysisError("mean/norm uncertainty gives inconsistent physical DC bounds")
    return {
        "mean_dc_energy_fraction": nominal,
        "mean_dc_energy_fraction_bounds": [lower, upper],
        "mean_dc_energy_fraction_error_bound": max(abs(nominal - lower), abs(upper - nominal)),
    }


def validate_rank_observation(row: Mapping[str, Any]) -> dict:
    """Validate a compact, source-bound full-spectrum numerical summary.

    ``r95`` and ``r99`` contain ``lower``, ``upper``, ``cdf_before_lower``,
    ``cdf_at_upper`` and ``cdf_tolerance``.  CDFs use independent full-input
    energy, not a renormalized truncated spectrum.  Full spectra themselves
    remain in the oracle artifact identified by ``spectrum_sha256``.
    """
    _identity(row)
    canonical_sha256(row)
    try:
        if row["evidence_kind"] not in ("native", "synthetic_fixture"):
            raise AnalysisError("distinguish native observations from synthetic fixtures")
        if row["evidence_kind"] == "native":
            _native_capture_binding(row)
        elif ("fixture" in row and row["fixture"] is not True) or (
            "native_execution" in row and row["native_execution"] is not False
        ):
            raise AnalysisError("synthetic observation cannot claim native execution")
        if "observation_identity" in row:
            bound = row["observation_identity"]
            if not isinstance(bound, dict) or any(
                name not in bound for name in (*IDENTITY_FIELDS, *GEOMETRY_FIELDS, "evidence_kind")
            ) or any(name not in row or row[name] != value for name, value in bound.items()):
                raise AnalysisError("flattened observation differs from its capture-bound identity")
        for name in ("input_sha256", "spectrum_sha256", "oracle_sha256"):
            _digest(row[name], name)
        if "original_bytes_sha256" in row:
            _digest(row["original_bytes_sha256"], "original_bytes_sha256")
        if row["full_spectrum_count"] != row["n_channels"] or type(
            row["full_spectrum_count"]
        ) is not int:
            raise AnalysisError("a complete channel spectrum is required")
        if row["energy_definition"] != "uncentered_squared_singular_values":
            raise AnalysisError("unsupported energy definition")
        has_scaling = "energy_scaled" in row or "unscale_energy_exponent" in row
        if has_scaling:
            energy = _finite(row["energy_scaled"], "energy_scaled", 0)
            exponent = row["unscale_energy_exponent"]
            if type(exponent) is not int or not -2148 <= exponent <= 2046:
                raise AnalysisError("scaled energy requires an explicit valid power-of-two exponent")
            try:
                unscaled = math.ldexp(energy, exponent)
                unscaled = None if not math.isfinite(unscaled) or (unscaled == 0 and energy != 0) else unscaled
            except OverflowError:
                unscaled = None
            if row["full_input_energy"] is not None:
                _finite(row["full_input_energy"], "full_input_energy", 0)
            if unscaled != row["full_input_energy"]:
                raise AnalysisError("unscaled energy differs from its finite scaled representation")
        else:
            energy = _finite(row["full_input_energy"], "full_input_energy", 0)
        if type(row["zero_energy"]) is not bool or row["zero_energy"] != (energy == 0):
            raise AnalysisError("zero energy must be explicitly and consistently flagged")
        if energy == 0:
            if row["mean_dc_energy_fraction"] is not None:
                raise AnalysisError("zero-energy DC fraction is undefined, not NaN or zero")
            if row.get("mean_dc_energy_fraction_bounds") is not None or row.get(
                "mean_dc_energy_fraction_error_bound"
            ) is not None:
                raise AnalysisError("zero-energy DC fraction uncertainty is undefined")
        else:
            dc_error = _finite(row.get("mean_dc_energy_fraction_error_bound", 0),
                               "mean_dc_energy_fraction_error_bound", 0)
            nominal = _finite(row["mean_dc_energy_fraction"], "mean_dc_energy_fraction", 0, 1 + dc_error)
            if "mean_dc_energy_fraction_bounds" in row:
                interval = row["mean_dc_energy_fraction_bounds"]
                if not isinstance(interval, (list, tuple)) or len(interval) != 2:
                    raise AnalysisError("nonzero-energy DC fraction requires a numerical interval")
                low, high = (_finite(value, "DC fraction bound", 0, 1) for value in interval)
                if low > high or dc_error < max(abs(nominal - low), abs(high - nominal)):
                    raise AnalysisError("DC fraction scalar uncertainty understates its full-ratio interval")
        diagnostics = row["diagnostics"]
        if not isinstance(diagnostics, dict) or not diagnostics:
            raise AnalysisError("full-spectrum numerical diagnostics are required")
        _text(diagnostics["solver"], "solver")
        for name in ("trace_relative_error", "symmetry_relative_error", "negative_eigenvalue_mass"):
            _finite(diagnostics[name], name, 0)
        for level, threshold in (("r95", .95), ("r99", .99)):
            bracket = row[level]
            if "tau" in bracket and _finite(bracket["tau"], f"{level}.tau") != threshold:
                raise AnalysisError("rank bracket threshold differs from its r95/r99 label")
            low = _integer(bracket["lower"], f"{level}.lower")
            high = _integer(bracket["upper"], f"{level}.upper")
            if low > high or high > min(row["n_rows"], row["n_channels"]):
                raise AnalysisError(f"invalid {level} rank bracket")
            if energy == 0:
                if (low, high) != (0, 0) or any(
                    bracket[name] is not None for name in ("cdf_before_lower", "cdf_at_upper")
                ):
                    raise AnalysisError("zero energy has rank zero and undefined CDF")
                _finite(bracket["cdf_tolerance"], f"{level}.cdf_tolerance", 0, 1)
            else:
                if low < 1:
                    raise AnalysisError("nonzero input cannot have required rank zero")
                tolerance = _finite(bracket["cdf_tolerance"], f"{level}.cdf_tolerance", 0)
                before = _finite(bracket["cdf_before_lower"], f"{level}.cdf_before_lower", 0)
                after = _finite(bracket["cdf_at_upper"], f"{level}.cdf_at_upper", 0)
                if before > after:
                    raise AnalysisError("nominal CDF prefixes must be monotone")
                if low == 1 and before != 0:
                    raise AnalysisError("CDF before rank one must be zero")
                if ("cdf_before_lower_bounds" in bracket) != ("cdf_at_upper_bounds" in bracket):
                    raise AnalysisError("both certified CDF bound pairs must be retained")
                if "cdf_before_lower_bounds" in bracket:
                    bounds = []
                    for name in ("cdf_before_lower_bounds", "cdf_at_upper_bounds"):
                        pair = bracket[name]
                        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
                            raise AnalysisError(f"{level}.{name} must be a CDF interval")
                        a, b = (_finite(value, name, 0, 1) for value in pair)
                        if a > b:
                            raise AnalysisError("reversed CDF uncertainty bounds")
                        bounds.append((a, b))
                    if not bounds[0][1] < threshold <= bounds[1][0]:
                        raise AnalysisError("certified CDF bounds do not enclose a minimal-rank crossing")
                elif before - tolerance >= threshold or after + tolerance < threshold:
                    raise AnalysisError(f"{level} does not bracket the full-energy crossing")
            if "complete_endpoint_used" in bracket:
                if type(bracket["complete_endpoint_used"]) is not bool:
                    raise AnalysisError("complete-spectrum endpoint flag must be boolean")
                if bracket["complete_endpoint_used"] and (
                    high != min(row["n_rows"], row["n_channels"])
                    or bracket.get("cdf_at_upper_bounds") not in ([1, 1], (1, 1))
                ):
                    raise AnalysisError("complete-spectrum endpoint proof requires the full admissible rank")
            if "rank" in bracket:
                if bracket["rank"] != (low if low == high else None) or (
                    bracket["rank"] is not None and type(bracket["rank"]) is not int
                ):
                    raise AnalysisError("ambiguous rank interval cannot carry an exact rank")
            if "status" in bracket and bracket["status"] != (
                "zero" if energy == 0 else "exact" if low == high else "interval"
            ):
                raise AnalysisError("rank status contradicts its interval/zero-energy flag")
        if row["r95"]["lower"] > row["r99"]["lower"] or row["r95"]["upper"] > row["r99"]["upper"]:
            raise AnalysisError("r95 must not exceed r99")
    except (KeyError, TypeError) as error:
        raise AnalysisError(f"malformed rank observation: {error}") from error
    return json.loads(json.dumps(row, allow_nan=False))


def observation_from_spectrum(
    spectrum: Any, *, identity: Mapping[str, Any], spectrum_sha256: str,
) -> dict:
    """Adapt the oracle's validated serialization without computing another spectrum.

    ``identity`` contains the numeric identity/geometry fields plus
    ``evidence_kind``.  Every identity and geometry field must also exist in the oracle source's
    provenance; adding a case label after capture is not source binding.
    ``full_input_energy`` can be null only for an unrepresentable unscaled
    FP64 energy; its exact power-of-two scaling is retained in that case.
    """
    from .spectra import SpectrumResult

    payload = spectrum.to_dict() if isinstance(spectrum, SpectrumResult) else spectrum
    result = SpectrumResult.from_dict(payload)
    _identity(identity)
    source = result.source
    provenance = source.to_dict()["provenance"]
    for name, value in identity.items():
        if name == "source_identity":
            if value != source.identity:
                raise AnalysisError("oracle source identity differs from its explicit binding")
        elif provenance.get(name) != value:
            raise AnalysisError(f"oracle source provenance does not bind {name}")
    if tuple(source.shape) != (identity["n_rows"], identity["n_channels"]) or (
        source.target != identity["target"] or source.dtype != identity["source_dtype"]
    ):
        raise AnalysisError("oracle original source differs from observation geometry/target/dtype")
    if source.key_space != ("pre_rope" if source.target == "K" else "none") or source.representation != "original":
        raise AnalysisError("oracle is not the original pre-RoPE K/direct V input")
    if identity.get("evidence_kind") not in ("native", "synthetic_fixture"):
        raise AnalysisError("oracle adapter requires native versus synthetic_fixture evidence kind")
    if provenance.get("evidence_kind") != identity["evidence_kind"]:
        raise AnalysisError("evidence kind was not bound at source capture")
    diag = payload["diagnostics"]
    energy = result.energy_scaled
    row = {
        **dict(identity), "observation_identity": dict(identity), "input_sha256": source.tensor_sha256,
        "spectrum_sha256": _digest(spectrum_sha256, "spectrum_sha256"),
        "oracle_sha256": canonical_sha256(payload),
        "full_spectrum_count": len(result.eigenvalues_scaled),
        "energy_definition": "uncentered_squared_singular_values",
        "full_input_energy": result.energy, "energy_scaled": energy,
        "unscale_energy_exponent": 2 * result.scale_exponent,
        **dc_fraction_bounds(
            mean_energy=result.mean_energy_scaled, energy=energy,
            mean_error=diag["mean_error_bound_scaled"], norm_error=diag["norm_error_bound_scaled"],
        ),
        "zero_energy": result.zero_energy,
        "diagnostics": {
            "solver": result.solver.algorithm,
            "trace_relative_error": abs(diag["gram_trace_scaled"] - energy) / energy if energy else 0.0,
            "symmetry_relative_error": diag["symmetry_error_scaled"] / energy if energy else 0.0,
            "negative_eigenvalue_mass": diag["negative_eigenvalue_mass_scaled"],
            "negative_eigenvalue_mass_units": "scaled squared singular values",
            "oracle_diagnostics": diag, "solver_identity": payload["solver"],
            "source_retention": source.retention, "source_identity": source.identity,
        },
    }
    if "original_bytes_sha256" in provenance:
        row["original_bytes_sha256"] = _digest(provenance["original_bytes_sha256"], "original_bytes_sha256")
    for name, tau in (("r95", .95), ("r99", .99)):
        bracket = result.rank_at(tau).to_dict()
        before, after = bracket["cdf_before_lower"], bracket["cdf_at_upper"]
        tolerance = max(
            *(abs(value - before) for value in bracket["cdf_before_lower_bounds"]),
            *(abs(value - after) for value in bracket["cdf_at_upper_bounds"]),
        )
        row[name] = {**bracket, "cdf_tolerance": tolerance}
        if result.zero_energy:
            row[name].update(cdf_before_lower=None, cdf_at_upper=None)
    return validate_rank_observation(row)


def write_spectrum_observation(
    spectrum: Any, *, identity: Mapping[str, Any], spectrum_path: str | Path,
    observation_path: str | Path,
) -> dict:
    """Persist one complete spectrum and its compact observation before releasing it.

    This is a per-matrix writer, not a tensor-retention queue.  The caller owns
    paths, storage admission and launch gates.  No original input tensors or
    optional oracle directions are serialized.  Both files use the owners'
    immutable atomic writers.  The returned receipt contains only compact
    numeric metadata and file identities.
    """
    from .spectra import SpectrumResult, write_spectrum_json

    spectrum_path, observation_path = Path(spectrum_path), Path(observation_path)
    if spectrum_path.resolve() == observation_path.resolve() or spectrum_path.exists() or observation_path.exists():
        raise FileExistsError("spectrum and observation need distinct, new artifact paths")
    payload = spectrum.to_dict() if isinstance(spectrum, SpectrumResult) else spectrum
    validated = spectrum if isinstance(spectrum, SpectrumResult) else SpectrumResult.from_dict(payload)
    spectrum_digest = canonical_sha256(payload)
    row = observation_from_spectrum(payload, identity=identity, spectrum_sha256=spectrum_digest)
    try:
        artifact_path = str(spectrum_path.resolve().relative_to(observation_path.parent.resolve()))
    except ValueError:
        artifact_path = str(spectrum_path.resolve())
    row["spectrum_path"] = artifact_path
    spectrum_path.parent.mkdir(parents=True, exist_ok=True)
    written_digest = write_spectrum_json(validated, spectrum_path)
    if written_digest != spectrum_digest or file_sha256(spectrum_path) != spectrum_digest:
        raise AnalysisError("persisted spectrum checksum differs from the validated serialization")
    write_json_new(observation_path, {"schema": SINGLE_OBSERVATION_SCHEMA, "observation": row})
    return {
        "observation": row, "observation_path": str(observation_path),
        "observation_sha256": file_sha256(observation_path),
        "spectrum_path": str(spectrum_path), "spectrum_sha256": spectrum_digest,
    }


def native_magi_source(
    description: Mapping[str, Any], *, case_id: str, trajectory_id: str,
    parent_trajectory_id: str | None = None, evidence_kind: str = "native",
    retention: str = "spectrum-only",
) -> tuple[Any, dict]:
    """Validate the native ``description, original_full_matrix`` callback schema.

    The parent supplies case/trajectory identities, never inferred from an
    arm-specific request ID.  Native headerless hashes remain provenance;
    SpectrumSource.tensor_sha256 is deliberately left unset so the oracle
    computes its independent, header-bound original-row-major digest.
    """
    from .spectra import SpectrumSource
    from .types import TensorContext

    canonical_sha256(description)
    try:
        if description["schema"] != "reelcache.magi_original_clean.v1":
            raise AnalysisError("unsupported original MAGI source-description schema")
        cell = CELLS.get(description["cell_id"])
        if cell is None or cell.cell_id not in ("magi720", "magi1080"):
            raise AnalysisError("MAGI capture requires a parent-approved MAGI cell")
        context = TensorContext.from_dict(description["tensor_context"])
        key = context.key
        dtype = cell.key_dtype if key.target == "K" else cell.value_dtype
        if context.shape != (cell.tokens, cell.channels) or context.model.layers != cell.layers or (
            context.frames, context.height, context.width
        ) != (cell.frames, cell.token_height, cell.token_width):
            raise AnalysisError("MAGI original context differs from full-token/full-channel geometry")
        if context.dtype != dtype or description["actual_dtype"] != dtype:
            raise AnalysisError("MAGI original context/source dtype differs from native pre-K/direct V")
        if context.key_space != ("pre_rope" if key.target == "K" else "none"):
            raise AnalysisError("a RAW post-RoPE storage context is not an original pre-K observation")
        if tuple(description["shape"]) != context.shape or description["global_channels"] != cell.channels:
            raise AnalysisError("MAGI observation is not a complete full-channel original matrix")
        if type(description["global_channels"]) is not int or context.token_start % cell.tokens:
            raise AnalysisError("MAGI token range is not a canonical clean-chunk boundary")
        if description["token_positions"] != [context.token_start, context.token_end]:
            raise AnalysisError("MAGI description/context absolute token positions differ")
        if description["inverse_rope_source"] is not False or description["source_layout"] != (
            "owner_assembled_original_token_rows"
        ):
            raise AnalysisError("MAGI spectra require directly captured, owner-assembled original token rows")
        if _finite(description["clean_timestep"], "clean_timestep") != .9999:
            raise AnalysisError("MAGI observation is not its declared original clean boundary")
        _text(description["branch_id"], "branch_id")
        _text(description["request_id"], "request_id")
        if not isinstance(description["chunk_id"], str) or not description["chunk_id"]:
            raise AnalysisError("MAGI capture must bind an explicit immutable chunk identity")
        if not isinstance(description["history"], list):
            raise AnalysisError("MAGI capture must bind its current history")
        boundary = context.token_start // cell.tokens
        head = description["history_head"]
        if (boundary == 0 and head is not None) or (boundary > 0 and (
            not isinstance(head, str) or not head
        )):
            raise AnalysisError("MAGI current parent head disagrees with startup/later history")
        hash_format = "sha256-native-contiguous-bytes-no-header-v1"
        for name in ("source_hash_format", "row_shard_hash_format"):
            if description[name] != hash_format:
                raise AnalysisError("native source/shard hashes need their explicit headerless format")
        for name in ("source_hash_byte_order", "row_shard_hash_byte_order"):
            if description[name] != sys.byteorder:
                raise AnalysisError("native capture hash byte order differs from the observing process")
        _digest(description["source_sha256"], "native source_sha256")
        if not isinstance(description["rotary_sha256"], list) or not description["rotary_sha256"]:
            raise AnalysisError("native rotary/position identities are required")
        for digest in description["rotary_sha256"]:
            _digest(digest, "rotary_sha256")
        shards = description["row_shards"]
        if not isinstance(shards, list) or len(shards) != cell.world_size:
            raise AnalysisError("MAGI original source needs both declared row shards")
        cursor = 0
        for rank, shard in enumerate(shards):
            if _integer(shard["rank"], "row shard rank") != rank or _integer(
                shard["row_start"], "row_start"
            ) != cursor:
                raise AnalysisError("MAGI row shards must be ordered, nonoverlapping and gap-free")
            end = _integer(shard["row_end"], "row_end", cursor + 1)
            if end > cell.tokens or shard["channels"] != cell.channels or type(shard["channels"]) is not int or (
                shard["dtype"] != dtype
            ):
                raise AnalysisError("row shards must retain all C channels and canonical source dtype")
            _digest(shard["sha256"], "native row-shard sha256")
            cursor = end
        if cursor != cell.tokens:
            raise AnalysisError("MAGI original row shards do not cover all N rows")
        bindings = {
            "request_id": description["request_id"], "chunk_id": description["chunk_id"],
            "arm_id": _text(description["arm_id"], "actual request arm_id"),
            "branch_id": description["branch_id"], "parent_head": head,
            "parent_trajectory_id": parent_trajectory_id,
            "capture_sha256": canonical_sha256(description),
            "native_source_sha256": description["source_sha256"],
            "native_source_hash_format": hash_format,
            "native_source_hash_byte_order": description["source_hash_byte_order"],
            "token_start": context.token_start, "token_end": context.token_end,
        }
        bindings["protocol_id"] = _digest(description["protocol_id"], "protocol_id")
        for name in ("asset_id", "launch_closure_id", "config_id"):
            if name in description:
                _digest(description[name], name)
                bindings[name] = description[name]
        for name in ("fixture", "native_execution"):
            if name in description:
                bindings[name] = description[name]
        identity = rank_observation_identity(
            case_id=case_id, group_id=description["group_id"], split=description["split"],
            seed=description["seed"], cell_id=cell.cell_id, layer=key.layer, target=key.target,
            boundary=boundary, trajectory_id=trajectory_id, evidence_kind=evidence_kind,
            source_bindings=bindings,
        )
        for name, value in identity.items():
            if name in description and description[name] != value:
                raise AnalysisError(f"native capture and bound observation disagree on {name}")
        provenance = {**description, **identity}
        source = SpectrumSource(
            identity=canonical_sha256({"capture": description, "observation_identity": identity}),
            shape=context.shape, dtype=dtype, target=key.target, key_space=context.key_space,
            retention=retention, provenance=provenance,
        )
    except (KeyError, TypeError) as error:
        raise AnalysisError(f"malformed native MAGI capture description: {error}") from error
    return source, identity


def observe_magi_spectrum(
    description: Mapping[str, Any], original_full_matrix: Any, *, case_id: str,
    trajectory_id: str, spectrum_path: str | Path, observation_path: str | Path,
    tile_rows: int, max_workspace_bytes: int, parent_trajectory_id: str | None = None,
    evidence_kind: str = "native", retention: str = "spectrum-only",
    existing_result: Any = None, telemetry_path: str | Path | None = None,
) -> dict:
    """Native callback adapter; parent admission is required before invoking it.

    The native callback retains responsibility for original-source mutation/hash
    checks.  Oracle validation checks the actual tensor's type, dtype, finiteness
    and source identity.  A reusable result must already bind this complete
    capture: mismatches fail instead of silently performing a second eigensolve.
    Telemetry is separate from deterministic numerical spectrum JSON.
    """
    from .spectra import SpectrumResult, exact_spectrum

    source, identity = native_magi_source(
        description, case_id=case_id, trajectory_id=trajectory_id,
        parent_trajectory_id=parent_trajectory_id, evidence_kind=evidence_kind, retention=retention,
    )
    _integer(tile_rows, "tile_rows", 1)
    _integer(max_workspace_bytes, "max_workspace_bytes", 1)
    shape = tuple(getattr(original_full_matrix, "shape", ()))
    dtype = str(getattr(original_full_matrix, "dtype", "")).removeprefix("torch.")
    if shape != tuple(source.shape) or dtype != source.dtype:
        raise AnalysisError("actual original MAGI matrix differs from the bound callback shape/dtype")
    paths = [Path(spectrum_path), Path(observation_path)]
    if telemetry_path is not None:
        paths.append(Path(telemetry_path))
    if len({path.resolve() for path in paths}) != len(paths) or any(path.exists() or path.is_symlink() for path in paths):
        raise FileExistsError("native spectrum/observation/telemetry paths must be distinct new admitted paths")
    reused = existing_result is not None
    if reused:
        payload = existing_result.to_dict() if isinstance(existing_result, SpectrumResult) else existing_result
        result = SpectrumResult.from_dict(payload)
        expected_source, actual_source = source.to_dict(), result.source.to_dict()
        original_provenance = actual_source["provenance"]
        if any(actual_source[name] != expected_source[name] for name in (
            "shape", "dtype", "target", "key_space", "retention", "representation",
        )) or any(original_provenance.get(name) != value for name, value in identity.items()):
            raise AnalysisError("existing full spectrum is not bound to this original source/case/trajectory")
        if sys.byteorder == "little" and "original_bytes_sha256" in original_provenance and (
            original_provenance["original_bytes_sha256"] != description["source_sha256"]
        ):
            raise AnalysisError("existing policy's measured raw-byte hash differs from native capture")
        call_seconds = 0.0
    else:
        started = time.perf_counter()
        result = exact_spectrum(original_full_matrix, source=source, tile_rows=tile_rows,
                                max_workspace_bytes=max_workspace_bytes)
        call_seconds = time.perf_counter() - started
    receipt = write_spectrum_observation(
        result, identity=identity, spectrum_path=spectrum_path, observation_path=observation_path,
    )
    telemetry = {
        "schema": "reelcache_rank_offline_oracle_cost_v1",
        "spectrum_sha256": receipt["spectrum_sha256"], "observation_sha256": receipt["observation_sha256"],
        "reused_existing_result": reused, "new_decomposition_count": 0 if reused else result.solver.decomposition_count,
        "new_oracle_call_seconds": call_seconds,
        "recorded_solver_work_counts": result.solver.work_counts(),
        "recorded_oracle_work_counts": result.work_counts(),
        "oracle_timings": result.timings.to_dict() if result.timings is not None else None,
        "timing_nesting": "eigensolve/direction timings are included in finalization, not added again",
        "included_in_online_selector_cost": False,
        "scope": "recorded counts describe the original oracle; reused results add no solve; "
                 "capture/staging/persistence/failed work charged separately",
    }
    if telemetry_path is not None:
        write_json_new(telemetry_path, telemetry)
        receipt["telemetry_path"] = str(telemetry_path)
        receipt["telemetry_sha256"] = file_sha256(telemetry_path)
    return {**receipt, "offline_oracle_telemetry": telemetry}


def load_observation_document(path: str | Path, *, required_split: str | None = None) -> dict:
    """Load numeric rows directly or from hash-bound per-matrix writer receipts.

    A census document contains ``schema``, ``expected`` and exactly one of
    ``observations`` or ``observation_files``.  File entries are path/sha256
    objects relative to the census document.  Referenced complete spectrum
    artifacts are checksum-checked and revalidated one numeric spectrum at a
    time, never loaded as tensors or retained together.
    """
    path = Path(path)
    document = read_json(path)
    if document.get("schema") != OBSERVATION_SCHEMA or not isinstance(document.get("expected"), list):
        raise AnalysisError("unsupported observation document or absent expected inventory")
    if required_split is not None:
        if required_split not in SPLITS or any(
            row.get("split") != required_split for row in document["expected"]
        ):
            raise AnalysisError("input inventory contains a prohibited split; no spectrum files were opened")
    if ("observations" in document) == ("observation_files" in document):
        raise AnalysisError("supply exactly one inline or file-backed observation collection")

    def verify_spectrum(row, base):
        _text(row.get("spectrum_path"), "bound spectrum path")
        spectrum_path = (base / row["spectrum_path"]).resolve()
        if file_sha256(spectrum_path) != row["spectrum_sha256"]:
            raise AnalysisError("complete spectrum artifact checksum mismatch")
        if "observation_identity" not in row:
            raise AnalysisError("source-backed observation omits the capture-bound identity")
        oracle = read_json(spectrum_path)
        regenerated = observation_from_spectrum(
            oracle, identity=row["observation_identity"], spectrum_sha256=row["spectrum_sha256"],
        )
        if canonical_sha256(regenerated) != canonical_sha256({
            name: value for name, value in row.items() if name != "spectrum_path"
        }):
            raise AnalysisError("numeric observation contradicts the complete original spectrum")
        row["spectrum_path"] = str(spectrum_path)
        return row

    if "observations" in document:
        if not isinstance(document["observations"], list):
            raise AnalysisError("observations must be a JSON row list")
        if required_split is not None and any(
            row.get("split") != required_split for row in document["observations"]
        ):
            raise AnalysisError("inline observations contain a prohibited split")
        rows = [
            verify_spectrum(validate_rank_observation(row), path.parent)
            if row.get("evidence_kind") == "native" else validate_rank_observation(row)
            for row in document["observations"]
        ]
        return {**document, "observations": rows}
    entries = document["observation_files"]
    if not isinstance(entries, list) or not entries:
        raise AnalysisError("observation_files must be a nonempty path/sha256 list")
    rows, artifacts, seen = [], [], set()
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {"path", "sha256"}:
            raise AnalysisError("observation file entries require exactly path and sha256")
        _text(entry["path"], "observation path")
        _digest(entry["sha256"], "observation file sha256")
        numeric_path = (path.parent / entry["path"]).resolve()
        if numeric_path in seen:
            raise AnalysisError("duplicate observation artifact path")
        seen.add(numeric_path)
        if file_sha256(numeric_path) != entry["sha256"]:
            raise AnalysisError("numeric observation file checksum mismatch")
        record = read_json(numeric_path)
        if set(record) != {"schema", "observation"} or record["schema"] != SINGLE_OBSERVATION_SCHEMA:
            raise AnalysisError("numeric observation file has an unsupported schema")
        if required_split is not None and record["observation"].get("split") != required_split:
            raise AnalysisError("numeric observation file contains a prohibited split")
        row = verify_spectrum(validate_rank_observation(record["observation"]), numeric_path.parent)
        rows.append(row)
        artifacts.append({
            "path": str(numeric_path), "sha256": entry["sha256"],
            "spectrum_path": row["spectrum_path"], "spectrum_sha256": row["spectrum_sha256"],
        })
    return {**document, "observations": rows, "source_observation_artifacts": artifacts}


def validate_census(
    observations: Sequence[Mapping[str, Any]], expected: Sequence[Mapping[str, Any]],
) -> list[dict]:
    """Reject missing/extra observations before aggregation or plotting."""
    required = _check_inventory(expected, complete_layers=True)
    actual = _check_inventory(observations, complete_layers=True)
    if set(actual) != set(required):
        raise AnalysisError(
            f"observation coverage mismatch: missing={len(set(required) - set(actual))}, "
            f"unexpected={len(set(actual) - set(required))}"
        )
    result = []
    for key in sorted(actual):
        row = actual[key]
        for name, value in required[key].items():
            if name not in row or row[name] != value:
                raise AnalysisError(f"observation differs from bound {name}")
        result.append(validate_rank_observation(row))
    return result


def longform_rows(observations: Sequence[Mapping[str, Any]]) -> list[dict]:
    rows = []
    for observation in observations:
        row = validate_rank_observation(observation)
        for name, tau in (("r95", .95), ("r99", .99)):
            bracket = row[name]
            rows.append({
                **{key: row[key] for key in (*IDENTITY_FIELDS, *GEOMETRY_FIELDS)},
                "tau": tau, "rank_lower": bracket["lower"], "rank_upper": bracket["upper"],
                "rank_is_exact": bracket["lower"] == bracket["upper"],
                "rank_exact": bracket["lower"] if bracket["lower"] == bracket["upper"] else None,
                "rank_status": "zero" if row["zero_energy"] else (
                    "exact" if bracket["lower"] == bracket["upper"] else "interval"
                ),
                "cdf_before_lower": bracket["cdf_before_lower"],
                "cdf_at_upper": bracket["cdf_at_upper"], "cdf_tolerance": bracket["cdf_tolerance"],
                **{key: row[key] for key in (
                    "input_sha256", "spectrum_sha256", "oracle_sha256", "full_spectrum_count",
                    "full_input_energy", "mean_dc_energy_fraction", "zero_energy",
                )},
                "diagnostics": row["diagnostics"],
                "energy_scaled": row.get("energy_scaled"),
                "original_bytes_sha256": row.get("original_bytes_sha256"),
                "unscale_energy_exponent": row.get("unscale_energy_exponent"),
                "mean_dc_energy_fraction_bounds": row.get("mean_dc_energy_fraction_bounds"),
                "mean_dc_energy_fraction_error_bound": row.get(
                    "mean_dc_energy_fraction_error_bound", None if row["zero_energy"] else 0.0
                ),
                "cdf_before_lower_bounds": bracket.get("cdf_before_lower_bounds"),
                "cdf_at_upper_bounds": bracket.get("cdf_at_upper_bounds"),
                "complete_endpoint_used": bracket.get("complete_endpoint_used"),
            })
    return rows


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        raise AnalysisError("mean of empty observations")
    return math.fsum(values) / len(values)


def aggregate_ranks(
    observations: Sequence[Mapping[str, Any]], expected: Sequence[Mapping[str, Any]],
    *, split: str, boundaries: str = "all", normalized: bool = False,
) -> list[dict]:
    if split not in SPLITS or boundaries not in ("all", "first", "later"):
        raise AnalysisError("declare one split and all/first/later boundaries")
    rows = validate_census(observations, expected)
    first = {}
    for row in rows:
        first[row["case_id"]] = min(first.get(row["case_id"], row["boundary"]), row["boundary"])
    grouped = defaultdict(list)
    for row in rows:
        is_first = row["boundary"] == first[row["case_id"]]
        if row["split"] != split or (boundaries == "first" and not is_first) or (
            boundaries == "later" and is_first
        ):
            continue
        for threshold in (95, 99):
            grouped[(row["cell"], row["layer"], row["target"], threshold)].append(row)
    if not grouped:
        raise AnalysisError("requested split/boundaries has no observations")
    output = []
    for (cell, layer, target, threshold), values in sorted(grouped.items()):
        width = values[0]["n_channels"]
        divisor = width if normalized else 1
        per_prompt = defaultdict(lambda: defaultdict(list))
        lows, highs = [], []
        for row in values:
            bracket = row[f"r{threshold}"]
            low, high = bracket["lower"] / divisor, bracket["upper"] / divisor
            per_prompt[row["prompt_group"]][row["seed"]].append((low, high))
            lows.append(low)
            highs.append(high)
        group_low, group_high = [], []
        for seeds in per_prompt.values():
            group_low.append(_mean(_mean(pair[0] for pair in pairs) for pairs in seeds.values()))
            group_high.append(_mean(_mean(pair[1] for pair in pairs) for pairs in seeds.values()))
        variance = statistics.pvariance(group_high)
        output.append({
            "cell": cell, "split": split, "boundaries": boundaries,
            "layer": layer, "target": target, "tau": threshold / 100,
            "n_channels": width, "normalized": normalized,
            "mean": _mean(group_high), "mean_lower": _mean(group_low),
            "mean_upper": _mean(group_high), "min": min(lows), "max": max(highs),
            "std": math.sqrt(variance), "var": variance,
            "count": len(per_prompt), "observation_count": len(values),
            "seed_count": sum(len(seeds) for seeds in per_prompt.values()),
            "boundary_indices": sorted({row["boundary"] for row in values}),
            "ambiguous_count": sum(low != high for low, high in zip(lows, highs)),
            "point_policy": POINT_POLICY, "range_label": RANGE_LABEL,
            "dispersion_unit": "prompt-group mean upper endpoint; descriptive population variance (ddof=0)",
        })
    return output


def plot_rank_panels(
    aggregates: Sequence[Mapping[str, Any]], *, cell_order: Sequence[str],
    png_path: str | Path, pdf_path: str | Path,
) -> dict:
    """Render exactly three cells by K/V without importing native/model code."""
    if len(cell_order) != 3 or len(set(cell_order)) != 3:
        raise AnalysisError("main figures require three explicit cells")
    canonical_sha256(aggregates)
    if not aggregates or {row["cell"] for row in aggregates} != set(cell_order):
        raise AnalysisError("figure cell coverage differs from the explicit cell order")
    variants = {(row["split"], row["boundaries"], row["normalized"]) for row in aggregates}
    if len(variants) != 1:
        raise AnalysisError("never mix calibration/heldout or boundary variants in a figure")
    split, boundaries, normalized = next(iter(variants))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    font_settings = {"font.family": "DejaVu Sans"}
    if "font.enable_last_resort" in matplotlib.rcParams:
        # All study labels are covered by DejaVu; avoid an unnecessary remote font read.
        font_settings["font.enable_last_resort"] = False
    with matplotlib.rc_context(font_settings):
        figure, axes = plt.subplots(3, 2, figsize=(12, 10), squeeze=False)
        colors = {.95: "#2563a6", .99: "#bd5319"}
        try:
            for i, cell in enumerate(cell_order):
                for j, target in enumerate(("K", "V")):
                    axis = axes[i, j]
                    panel = [row for row in aggregates if row["cell"] == cell and row["target"] == target]
                    if not panel:
                        raise AnalysisError("figure is missing a K/V panel")
                    for tau in (.95, .99):
                        data = sorted((row for row in panel if row["tau"] == tau), key=lambda row: row["layer"])
                        if not data or [row["layer"] for row in data] != list(range(len(data))):
                            raise AnalysisError("figure layers are missing or duplicated")
                        for row in data:
                            for name in ("mean", "mean_lower", "mean_upper", "min", "max", "std", "var"):
                                _finite(row[name], name, 0)
                            if not row["min"] <= row["mean_lower"] <= row["mean"] <= row["max"]:
                                raise AnalysisError("figure numerical bounds are inconsistent")
                        x = [row["layer"] for row in data]
                        suffix = " (upper if ambiguous)" if any(row["ambiguous_count"] for row in data) else ""
                        axis.plot(x, [row["mean"] for row in data], color=colors[tau],
                                  linestyle="-", label=f"Mean r{round(tau * 100)}{suffix}")
                        axis.fill_between(x, [row["min"] for row in data], [row["max"] for row in data],
                                          color=colors[tau], alpha=.18)
                    counts = {(row["count"], row["observation_count"]) for row in panel}
                    if len(counts) != 1:
                        raise AnalysisError("layer-dependent missing observations in figure")
                    groups, count = next(iter(counts))
                    axis.set_title(f"{cell} · {target} · C={panel[0]['n_channels']}\n"
                                   f"{groups} prompt groups; {count} matrices/layer/target")
                    axis.set_xlabel("Layer index")
                    axis.set_ylabel("Required rank / C" if normalized else "Required rank")
                    axis.legend(fontsize=8)
                    axis.grid(alpha=.2)
            figure.suptitle(f"{split} · {boundaries} clean boundaries\n{RANGE_LABEL}", fontsize=11)
            figure.tight_layout(rect=(0, 0, 1, .94))
            for path in (Path(png_path), Path(pdf_path)):
                if path.exists():
                    raise FileExistsError(f"refusing to overwrite figure: {path}")
                path.parent.mkdir(parents=True, exist_ok=True)
                figure.savefig(path, dpi=160)
        finally:
            plt.close(figure)
    return {"aggregate_sha256": canonical_sha256(aggregates), "range_label": RANGE_LABEL,
            "point_policy": POINT_POLICY, "matplotlib_version": matplotlib.__version__,
            "font_settings": font_settings,
            "png_sha256": file_sha256(png_path), "pdf_sha256": file_sha256(pdf_path)}


def validate_accounting(accounting: Mapping[str, Any], *, complete: bool = True) -> dict:
    required = ("native_logical_bytes", "source_bytes", *BYTE_DENOMINATORS,
                "basis_bytes", "payload_bytes", "metadata_bytes")
    try:
        for name in required:
            _integer(accounting[name], name)
        if complete and any(accounting[name] == 0 for name in
                            ("native_logical_bytes", "source_bytes", *BYTE_DENOMINATORS)):
            raise AnalysisError("completed measurements require positive byte denominators")
        if accounting["decoder_closure_bytes"] != sum(
            accounting[name] for name in ("basis_bytes", "payload_bytes", "metadata_bytes")
        ):
            raise AnalysisError("decoder closure must count full unique bases, payloads and metadata")
        if accounting["snapshot_closure_bytes"] < accounting["decoder_closure_bytes"]:
            raise AnalysisError("snapshot closure cannot omit decoder closure")
        for mapping in ("work_counts", "resource_bytes"):
            if not isinstance(accounting[mapping], dict):
                raise AnalysisError(f"{mapping} must be explicit, even when empty")
            for name, value in accounting[mapping].items():
                _text(name, mapping)
                _integer(value, f"{mapping}.{name}")
        if not isinstance(accounting["timings_seconds"], dict):
            raise AnalysisError("timings_seconds must be an explicit mapping")
        for name, value in accounting["timings_seconds"].items():
            _text(name, "timings_seconds")
            _finite(value, f"timings_seconds.{name}", 0)
    except KeyError as error:
        raise AnalysisError(f"missing accounting field: {error.args[0]}") from error
    canonical_sha256(accounting)
    return dict(accounting)


def accounting_ratios(
    records: Sequence[Mapping[str, Any]], *, numerator: str, denominator: str,
) -> dict:
    """Pooled sum ratios are distinct from the mean of per-case ratios."""
    if numerator not in ("native_logical_bytes", "source_bytes") or denominator not in BYTE_DENOMINATORS:
        raise AnalysisError("declare one supported byte numerator and denominator")
    if not records:
        raise AnalysisError("no accounting records")
    keys, rows = set(), []
    arm_cells = set()
    for row in records:
        key = (_text(row["case_id"], "case_id"), _text(row["arm_id"], "arm_id"))
        if key in keys:
            raise AnalysisError("duplicate accounting case/arm")
        keys.add(key)
        arm_cells.add((row["arm_id"], row["cell"], row["split"]))
        accounting = validate_accounting(row["accounting"])
        rows.append({"case_id": row["case_id"], "prompt_group": row["prompt_group"],
                     "numerator_bytes": accounting[numerator],
                     "denominator_bytes": accounting[denominator],
                     "ratio": accounting[numerator] / accounting[denominator]})
    if len(arm_cells) != 1:
        raise AnalysisError("accounting pools must keep arm, cell and split separate")
    total_numerator = sum(row["numerator_bytes"] for row in rows)
    total_denominator = sum(row["denominator_bytes"] for row in rows)
    return {"numerator": numerator, "denominator": denominator, "per_case": rows,
            "sum_numerator_bytes": total_numerator, "sum_denominator_bytes": total_denominator,
            "pooled_sum_ratio": total_numerator / total_denominator,
            "mean_per_case_ratio": _mean(row["ratio"] for row in rows),
            "scope": "supplied completed cases only; consult outcome coverage",
            "case_count": len(rows)}


def bracket_actual_bytes(
    adaptive: Mapping[str, Any], uniform_points: Sequence[Mapping[str, Any]], *, denominator: str,
) -> dict:
    """Bracket measured bytes, never claim equal rank sums imply equal HEVC bytes."""
    if denominator not in BYTE_DENOMINATORS or not uniform_points:
        raise AnalysisError("actual-byte matching requires a denominator and uniform measurements")
    wanted = validate_accounting(adaptive["accounting"])[denominator]
    points, ids = [], set()
    for row in uniform_points:
        for name in ("case_id", "cell", "split", "representation"):
            if row[name] != adaptive[name]:
                raise AnalysisError(f"actual-byte matching differs in {name}")
        if row["arm_id"] in ids:
            raise AnalysisError("duplicate uniform byte point")
        ids.add(row["arm_id"])
        values = validate_accounting(row["accounting"])
        for name in ("native_logical_bytes", "source_bytes"):
            if values[name] != adaptive["accounting"][name]:
                raise AnalysisError("byte comparison has inconsistent numerator definitions")
        points.append({"arm_id": row["arm_id"], "bytes": values[denominator],
                       "mismatch_bytes": values[denominator] - wanted,
                       "relative_mismatch": (values[denominator] - wanted) / wanted})
    points.sort(key=lambda row: (row["bytes"], row["arm_id"]))
    lower = [row for row in points if row["bytes"] <= wanted]
    upper = [row for row in points if row["bytes"] >= wanted]
    return {"denominator": denominator, "adaptive_bytes": wanted,
            "lower": lower[-1] if lower else None, "upper": upper[0] if upper else None,
            "bracketed": bool(lower and upper),
            "exact_match": any(row["bytes"] == wanted for row in points),
            "interpolated": False, "matching_basis": "actual measured bytes, not average rank"}


def _outcome_inventory(
    expected_cases: Sequence[Mapping[str, Any]], outcomes: Sequence[Mapping[str, Any]],
) -> tuple[dict, dict]:
    expected, groups = {}, {}
    for row in expected_cases:
        for name in ("case_id", "prompt_group", "cell"):
            _text(row[name], name)
        _integer(row["seed"], "seed")
        if row["split"] not in SPLITS:
            raise AnalysisError("unknown outcome split")
        if row["case_id"] in expected:
            raise AnalysisError("duplicate expected quality case")
        if row["prompt_group"] in groups and groups[row["prompt_group"]] != row["split"]:
            raise AnalysisError("quality prompt group crosses splits")
        if "pairing_sha256" in row:
            _digest(row["pairing_sha256"], "expected pairing_sha256")
        groups[row["prompt_group"]] = row["split"]
        expected[row["case_id"]] = row
    if not expected:
        raise AnalysisError("explicit expected quality cases are required")
    result, completed_pairings = {}, {}
    for row in outcomes:
        if row["case_id"] not in expected:
            raise AnalysisError("unexpected quality case")
        for name in ("prompt_group", "cell", "seed", "split"):
            if row[name] != expected[row["case_id"]][name]:
                raise AnalysisError("quality outcome does not match expected identity")
        if "pairing_sha256" in expected[row["case_id"]] and (
            row.get("pairing_sha256") != expected[row["case_id"]]["pairing_sha256"]
        ):
            raise AnalysisError("quality outcome differs from the bound same-run paired window")
        arm = _text(row["arm_id"], "arm_id")
        key = (row["case_id"], arm)
        if key in result:
            raise AnalysisError("duplicate quality case/arm")
        if row["status"] not in ("complete", "failed"):
            raise AnalysisError("outcomes must declare complete or failed")
        if row["status"] == "complete":
            pairing = _digest(row.get("pairing_sha256"), "completed pairing_sha256")
            if completed_pairings.setdefault(row["case_id"], pairing) != pairing:
                raise AnalysisError("completed arms differ in their same-run pairing_sha256 / paired window")
            if set(row["metrics"]) != set(METRICS):
                raise AnalysisError("completed quality outcome requires SSIM and all six dimensions")
            for metric, value in row["metrics"].items():
                _finite(value, metric)
            _finite(row["metrics"]["ssim"], "ssim", -1, 1)
            if row["metrics"]["dynamic_degree"] not in (0, 1):
                raise AnalysisError("per-case official dynamic_degree is binary")
            if "accounting" in row:
                validate_accounting(row["accounting"])
        else:
            _text(row["error"], "failure error")
            if "accounting" in row:
                validate_accounting(row["accounting"], complete=False)
        canonical_sha256(row)
        result[key] = row
    return expected, result


def validate_quality_outcomes(expected_cases, outcomes) -> None:
    """Validate globally before splitting tables, so filtering cannot hide bad rows."""
    _outcome_inventory(expected_cases, outcomes)


def paired_accounting_tables(
    expected_cases, outcomes, *, reference_arm: str, candidate_arm: str,
) -> dict:
    """Compare matching, fully accounted prompt groups; expose every missing result."""
    expected, actual = _outcome_inventory(expected_cases, outcomes)
    if reference_arm == candidate_arm:
        raise AnalysisError("accounting comparison arms must differ")
    if len({(row["cell"], row["split"]) for row in expected.values()}) != 1:
        raise AnalysisError("paired accounting must keep cell and split separate")
    by_group = defaultdict(list)
    for case in expected.values():
        by_group[case["prompt_group"]].append(case)
    missing, paired, complete_groups = [], [], []
    for group, cases in sorted(by_group.items()):
        reasons = []
        for case in cases:
            for arm in (reference_arm, candidate_arm):
                row = actual.get((case["case_id"], arm))
                if row is None or row["status"] != "complete" or "accounting" not in row:
                    reasons.append({
                        "prompt_group": group, "case_id": case["case_id"], "arm_id": arm,
                        "status": "missing" if row is None else row["status"],
                        "reason": "result absent" if row is None else (
                            row["error"] if row["status"] == "failed" else "accounting absent"
                        ),
                        "attempt_accounting": row.get("accounting") if row else None,
                    })
        if reasons:
            missing.extend(reasons)
            continue
        complete_groups.append(group)
        for case in cases:
            reference = actual[(case["case_id"], reference_arm)]
            candidate = actual[(case["case_id"], candidate_arm)]
            for name in ("native_logical_bytes", "source_bytes"):
                if reference["accounting"][name] != candidate["accounting"][name]:
                    raise AnalysisError("paired byte numerators differ across the same case/geometry")
            paired.extend((reference, candidate))
    ratios = []
    if paired:
        for arm in (reference_arm, candidate_arm):
            measured = [row for row in paired if row["arm_id"] == arm]
            for numerator in ("native_logical_bytes", "source_bytes"):
                for denominator in BYTE_DENOMINATORS:
                    ratios.append({
                        "arm_id": arm,
                        **accounting_ratios(measured, numerator=numerator, denominator=denominator),
                    })
    return {
        "reference_arm": reference_arm, "candidate_arm": candidate_arm,
        "coverage": len(complete_groups) / len(by_group), "expected_groups": len(by_group),
        "complete_groups": complete_groups, "expected_cases": len(expected),
        "paired_case_count": len(paired) // 2, "missing_or_failed": missing, "ratios": ratios,
        "comparison_scope": "identical complete prompt-group case sets; failed work retained separately",
    }


def matched_uniform_bytes(
    expected_cases, outcomes, *, adaptive_arm: str, uniform_arms: Sequence[str], denominator: str,
) -> dict:
    """Per-case and pooled actual-byte brackets with explicit incomplete-group coverage."""
    expected, actual = _outcome_inventory(expected_cases, outcomes)
    arms = tuple(uniform_arms)
    if not arms or len(set(arms)) != len(arms) or adaptive_arm in arms:
        raise AnalysisError("declare distinct adaptive and uniform arm IDs")
    if denominator not in BYTE_DENOMINATORS:
        raise AnalysisError("declare one actual measured byte denominator")
    if len({(row["cell"], row["split"]) for row in expected.values()}) != 1:
        raise AnalysisError("uniform matching must keep cell and split separate")
    groups = defaultdict(list)
    for case in expected.values():
        groups[case["prompt_group"]].append(case)
    complete, failures = [], []
    for group, cases in sorted(groups.items()):
        reasons = []
        for case in cases:
            for arm in (adaptive_arm, *arms):
                row = actual.get((case["case_id"], arm))
                if row is None or row["status"] != "complete" or "accounting" not in row:
                    reasons.append({
                        "prompt_group": group, "case_id": case["case_id"], "arm_id": arm,
                        "reason": "missing" if row is None else row["error"] if row["status"] == "failed"
                        else "accounting absent",
                        "attempt_accounting": row.get("accounting") if row else None,
                    })
        if reasons:
            failures.extend(reasons)
        else:
            complete.extend(cases)
    per_case = []
    for case in complete:
        selected = [actual[(case["case_id"], arm)] for arm in (adaptive_arm, *arms)]
        if any(not isinstance(row.get("representation"), str) or not row["representation"] for row in selected):
            raise AnalysisError("actual uniform matching requires explicit representation identities")
        per_case.append({
            **{name: case[name] for name in ("case_id", "prompt_group", "seed")},
            **bracket_actual_bytes(selected[0], selected[1:], denominator=denominator),
        })
    pooled = None
    if complete:
        accounting_fields = ("native_logical_bytes", "source_bytes", *BYTE_DENOMINATORS,
                             "basis_bytes", "payload_bytes", "metadata_bytes")
        points = []
        for arm in (adaptive_arm, *arms):
            values = [actual[(case["case_id"], arm)] for case in complete]
            if len({row["representation"] for row in values}) != 1:
                raise AnalysisError("one arm cannot pool changing representations")
            points.append({
                "case_id": "pooled-identical-complete-cases", "arm_id": arm,
                "cell": complete[0]["cell"], "split": complete[0]["split"],
                "representation": values[0]["representation"],
                "accounting": {
                    **{name: sum(row["accounting"][name] for row in values) for name in accounting_fields},
                    "work_counts": {}, "resource_bytes": {}, "timings_seconds": {},
                },
            })
        pooled = bracket_actual_bytes(points[0], points[1:], denominator=denominator)
    complete_groups = sorted({case["prompt_group"] for case in complete})
    return {
        "adaptive_arm": adaptive_arm, "uniform_arms": list(arms), "denominator": denominator,
        "expected_groups": len(groups), "complete_groups": complete_groups,
        "coverage": len(complete_groups) / len(groups), "expected_cases": len(expected),
        "complete_cases": len(complete), "per_case": per_case, "pooled_sum_bytes": pooled,
        "missing_or_failed": failures, "interpolation": False,
        "rank_matched_is_byte_matched": False,
        "scope": "actual closure bytes on identical complete groups; pooled sum bracket may differ from per-case brackets",
    }


def _quantile(values: Sequence[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low = int(math.floor(position))
    high = int(math.ceil(position))
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def paired_group_bootstrap(
    expected_cases: Sequence[Mapping[str, Any]], outcomes: Sequence[Mapping[str, Any]],
    *, reference_arm: str, candidate_arm: str, seed: int = 0,
    resamples: int = 2000, confidence: float = .95,
) -> dict:
    """Resample paired prompt groups, never frames/seeds/chunks as independent prompts.

    A group with any missing/failed expected pair is not imputed or partially
    averaged.  It remains in coverage and failures; intervals explicitly
    describe only complete groups, not an all-prompts success estimate.
    """
    if reference_arm == candidate_arm:
        raise AnalysisError("paired arms must differ")
    _integer(seed, "bootstrap seed")
    _integer(resamples, "resamples", 1)
    _finite(confidence, "confidence", 0, 1)
    if not 0 < confidence < 1:
        raise AnalysisError("confidence must lie strictly between zero and one")
    expected, actual = _outcome_inventory(expected_cases, outcomes)
    if len({(row["cell"], row["split"]) for row in expected.values()}) != 1:
        raise AnalysisError("bootstrap tables must keep cell and split separate")
    group_cases = defaultdict(list)
    failures = []
    for case in expected.values():
        group_cases[case["prompt_group"]].append(case)
    group_rows = []
    complete_cases = 0
    for group, cases in sorted(group_cases.items()):
        group_failures = []
        for case in cases:
            for arm in (reference_arm, candidate_arm):
                outcome = actual.get((case["case_id"], arm))
                if outcome is None or outcome["status"] != "complete":
                    group_failures.append({
                        "prompt_group": group, "case_id": case["case_id"], "arm_id": arm,
                        "status": "missing" if outcome is None else "failed",
                        "error": "expected result absent" if outcome is None else outcome["error"],
                        "accounting": None if outcome is None else outcome.get("accounting"),
                    })
        if group_failures:
            failures.extend(group_failures)
            continue
        complete_cases += len(cases)
        for metric in METRICS:
            arm_means = {}
            for arm in (reference_arm, candidate_arm):
                by_seed = defaultdict(list)
                for case in cases:
                    by_seed[case["seed"]].append(actual[(case["case_id"], arm)]["metrics"][metric])
                arm_means[arm] = _mean(_mean(values) for values in by_seed.values())
            group_rows.append({
                "prompt_group": group, "metric": metric, "case_count": len(cases),
                "seed_count": len({case["seed"] for case in cases}),
                "reference": arm_means[reference_arm], "candidate": arm_means[candidate_arm],
                "delta": arm_means[candidate_arm] - arm_means[reference_arm],
            })
    rng = random.Random(seed)
    complete_groups = sorted({row["prompt_group"] for row in group_rows})
    sampled_indices = [
        [rng.randrange(len(complete_groups)) for _ in complete_groups] for _ in range(resamples)
    ] if complete_groups else []
    summaries = []
    for metric in METRICS:
        rows = [row for row in group_rows if row["metric"] == metric]
        deltas = [row["delta"] for row in rows]
        bootstrap = [_mean(deltas[index] for index in indices) for indices in sampled_indices]
        summaries.append({
            "metric": metric, "reference_mean": _mean(row["reference"] for row in rows) if rows else None,
            "candidate_mean": _mean(row["candidate"] for row in rows) if rows else None,
            "delta_mean": _mean(deltas) if deltas else None,
            "delta_ci_lower": _quantile(bootstrap, (1 - confidence) / 2) if rows else None,
            "delta_ci_upper": _quantile(bootstrap, (1 + confidence) / 2) if rows else None,
            "prompt_group_count": len(rows),
            "interpretation": "motion classification change, not a quality objective"
            if metric == "dynamic_degree" else "candidate minus reference; not additive loss components",
        })
    return {
        "reference_arm": reference_arm, "candidate_arm": candidate_arm,
        "bootstrap_unit": "paired prompt group", "seed": seed, "resamples": resamples,
        "confidence": confidence, "expected_groups": len(group_cases),
        "complete_groups": len(complete_groups), "expected_cases": len(expected),
        "complete_group_cases": complete_cases,
        "excluded_groups": sorted(set(group_cases) - set(complete_groups)),
        "coverage": len(complete_groups) / len(group_cases),
        "inference_scope": "conditional on completely observed groups; no imputation or population guarantee",
        "failures": failures, "groups": group_rows, "metrics": summaries,
        "vbench_aggregate": False,
    }


def _fixed_detector_spec(value):
    from .errors import ConfigurationError
    from .factorization import DetectorSpec

    try:
        spec = value if isinstance(value, DetectorSpec) else DetectorSpec.from_dict(value)
    except (ConfigurationError, TypeError, ValueError) as error:
        raise AnalysisError("a complete frozen fixed DetectorSpec is required, not an arbitrary detector digest") from error
    if spec.implementation_fingerprint != file_sha256(Path(__file__).with_name("factorization.py")):
        raise AnalysisError("fixed detector implementation differs from the current source; rebind before calibration")
    return spec


def _validate_fixed_acquisition(acquisition, source, spec):
    from .errors import ConfigurationError
    from .types import TensorContext

    try:
        if acquisition["schema_version"] != "reelcache.anchor_acquisition.v1":
            raise AnalysisError("unsupported fixed detector acquisition")
        if acquisition["detector_spec"] != spec.to_dict() or acquisition["detector_fingerprint"] != spec.fingerprint:
            raise AnalysisError("acquisition differs from the full frozen detector recipe")
        if acquisition["source_hash_format"] != "sha256-little-endian-row-major-bytes-no-header-v1":
            raise AnalysisError("detector source hash must retain its raw little-endian byte domain")
        raw_hash = _digest(acquisition["source_sha256"], "detector source_sha256")
        expected_raw = source.get("original_bytes_sha256")
        if expected_raw is None and source.get("native_source_hash_format") == (
            "sha256-native-contiguous-bytes-no-header-v1"
        ) and source.get("native_source_hash_byte_order") == "little":
            expected_raw = source.get("native_source_sha256")
        if source.get("evidence_kind") == "native" and expected_raw is None:
            raise AnalysisError("native detector joins require the original raw-byte digest as well as the oracle digest")
        if expected_raw is not None and expected_raw != raw_hash:
            raise AnalysisError("detector source raw bytes differ from the bound original")
        context = TensorContext.from_dict(acquisition["context"])
        cell = CELLS[source["cell"]]
        if (
            context.shape != (source["n_rows"], source["n_channels"])
            or context.model.layers != source["n_layers"] or context.key.layer != source["layer"]
            or context.key.target != source["target"] or context.dtype != source["source_dtype"]
            or context.key_space != ("pre_rope" if source["target"] == "K" else "none")
            or (context.frames, context.height, context.width) != (cell.frames, cell.token_height, cell.token_width)
            or context.token_start != source["boundary"] * source["n_rows"]
        ):
            raise AnalysisError("fixed detector context differs from the original cell/layer/target/clean position")
        cap = min(context.shape)
        final_width = _integer(acquisition["final_width"], "fixed detector final_width", 1)
        if final_width > cap:
            raise AnalysisError("fixed detector final width exceeds the original matrix")
        features = acquisition["features"]
        if not isinstance(features, Mapping) or set(features) != {"0.95", "0.99"}:
            raise AnalysisError("one final detector fit must supply both .95 and .99 features")
        for rank in features.values():
            if rank is not None and (
                _integer(rank, "fixed detector crossing") > final_width
                or (rank == 0 and not source["zero_energy"])
            ):
                raise AnalysisError("invalid fixed-acquisition threshold crossing")
        r95, r99 = features["0.95"], features["0.99"]
        if r99 is not None and (r95 is None or r95 > r99):
            raise AnalysisError("same-fit .95/.99 detector crossings are inconsistent")
        status = "zero" if source["zero_energy"] else "unresolved" if r99 is None else "resolved"
        if acquisition["status"] != status or (source["zero_energy"] and (r95, r99) != (0, 0)):
            raise AnalysisError("fixed detector status contradicts its original energy/crossings")
        attempts = acquisition["attempts"]
        if not isinstance(attempts, (list, tuple)) or not attempts:
            raise AnalysisError("fixed acquisition needs its complete width/seed attempt history")
        width, seed = min(spec.initial_width, cap), spec.seed_for(context)
        for index, attempt in enumerate(attempts):
            if (
                type(attempt["width"]) is not int or attempt["width"] != width
                or type(attempt["seed"]) is not int or attempt["seed"] != (seed + index) % 2**63
            ):
                raise AnalysisError("detector widths/seeds differ from the fixed group-independent acquisition recipe")
            final = index == len(attempts) - 1
            expected_crossing = r99 if final else None
            if attempt["stop_crossing"] is not None:
                _integer(attempt["stop_crossing"], "detector stop_crossing")
            if attempt["stop_crossing"] != expected_crossing or attempt["status"] != (
                "unresolved" if expected_crossing is None else "resolved"
            ):
                raise AnalysisError("acquisition must stop at its first resolved .99 crossing")
            if final:
                if width != final_width or (r99 is None and width != cap):
                    raise AnalysisError("unresolved fixed acquisition must reach its geometry cap")
            else:
                if width == cap:
                    raise AnalysisError("fixed acquisition cannot refit after reaching its geometry cap")
                width = min(cap, width * spec.width_growth)
        metrics = acquisition["metrics"]
        for name in ("factorization_seed", "fit_attempts", "final_width", "cpu_threads"):
            _integer(metrics[name], f"fixed acquisition {name}", 0 if name == "factorization_seed" else 1)
        if (
            metrics["operation_counts_complete"] is not True
            or metrics["detector_fingerprint"] != spec.fingerprint
            or metrics["source_sha256"] != raw_hash or metrics["factorization_seed"] != seed
            or metrics["detector_device"] != spec.device or metrics["compute_dtype"] != spec.compute_dtype
            or metrics["source_dtype"] != context.dtype or metrics["cpu_threads"] != spec.cpu_threads
            or metrics["fit_attempts"] != len(attempts) or metrics["final_width"] != final_width
        ):
            raise AnalysisError("fixed acquisition telemetry differs from its recipe/input/final fit")
        canonical_sha256(acquisition)
        return features, final_width
    except (KeyError, TypeError, ConfigurationError) as error:
        raise AnalysisError(f"malformed fixed detector acquisition: {error}") from error


def validate_detector_records(
    observations: Sequence[Mapping[str, Any]], detectors: Sequence[Mapping[str, Any]], *, detector_spec,
) -> dict:
    """Bind deployment-available estimates to original inputs, not oracle anchor labels.

    Detector records repeat IDENTITY_FIELDS and contain tau, input_sha256,
    detector_sha256, estimate_kind='online_detector', solver='qr_q2',
    rank_statistic='approximate_spectral_crossing', available_after_layer,
    status='resolved'/'unresolved', estimated_rank and maximum_width. The full
    frozen DetectorSpec is required; detector_sha256 is its fingerprint, not an
    arbitrary implementation label. Each row binds acquisition plus its SHA256;
    both thresholds must come from the SAME last fixed-acquisition fit.
    Unresolved sketches remain unavailable features, never a fabricated crossing.
    """
    spec = _fixed_detector_spec(detector_spec)
    inputs = {tuple(row[name] for name in IDENTITY_FIELDS): row for row in observations}
    required = {(key, tau) for key in inputs for tau in (.95, .99)}
    actual, acquisition_ids = {}, {}
    for record in detectors:
        try:
            key = tuple(record[name] for name in IDENTITY_FIELDS)
            if record["split"] != "calibration":
                raise AnalysisError("calibration may not inspect heldout/development detector features")
            tau = _finite(record["tau"], "detector tau")
            joined = (key, tau)
            if joined not in required or joined in actual:
                raise AnalysisError("unexpected or duplicate detector record")
            source = inputs[key]
            if record["input_sha256"] != source["input_sha256"]:
                raise AnalysisError("detector and oracle do not describe the same current original input")
            if _digest(record["detector_sha256"], "detector_sha256") != spec.fingerprint:
                raise AnalysisError("detector record differs from the frozen acquisition recipe")
            if record["estimate_kind"] != "online_detector":
                raise AnalysisError("oracle ranks or offline-only estimates are not deployment features")
            if record["solver"] != "qr_q2" or record["rank_statistic"] != "approximate_spectral_crossing":
                raise AnalysisError("features must be deployed QR/q=2 crossings before guard/bucket expansion")
            if type(record["available_after_layer"]) is not int or (
                record["available_after_layer"] != record["layer"]
            ):
                raise AnalysisError("a detector estimate becomes available only after its actual layer")
            width = _integer(record["maximum_width"], "detector maximum_width", 1)
            if width > min(source["n_rows"], source["n_channels"]):
                raise AnalysisError("detector maximum width exceeds source geometry")
            if record["status"] == "resolved":
                rank = _integer(record["estimated_rank"], "estimated_rank")
                if rank > width or (rank == 0 and not source["zero_energy"]):
                    raise AnalysisError("invalid resolved detector rank")
            elif record["status"] == "unresolved":
                if record["estimated_rank"] is not None:
                    raise AnalysisError("unresolved sketch must not pretend its maximum rank reached tau")
            else:
                raise AnalysisError("unknown detector status")
            acquisition = record["acquisition"]
            acquisition_hash = _digest(record["acquisition_sha256"], "acquisition_sha256")
            if canonical_sha256(acquisition) != acquisition_hash:
                raise AnalysisError("fixed detector acquisition checksum differs from its record")
            if acquisition_ids.setdefault(key, acquisition_hash) != acquisition_hash:
                raise AnalysisError("both detector thresholds must use the SAME final acquisition fit")
            features, final_width = _validate_fixed_acquisition(acquisition, source, spec)
            feature = features[str(tau)]
            if record["estimated_rank"] != feature or width != final_width or record["status"] != (
                "unresolved" if feature is None else "resolved"
            ):
                raise AnalysisError("detector feature differs from its final fixed fit, not a later guard/operational rank")
            canonical_sha256(record)
            actual[joined] = dict(record)
        except (KeyError, TypeError) as error:
            raise AnalysisError(f"malformed detector record: {error}") from error
    if set(actual) != required:
        raise AnalysisError(f"missing online detector records: {len(required - set(actual))}")
    if len({row["detector_sha256"] for row in actual.values()}) != 1:
        raise AnalysisError("a calibration corpus cannot mix detector implementations")
    return actual


def detector_records_from_acquisition(acquisition, *, observation: Mapping[str, Any]) -> list[dict]:
    """Export both features of the shared fixed acquisition, without fitting again.

    ``observation`` is the already source-bound compact original spectrum record.
    The acquisition's raw-byte hash and context must match it; its oracle framed
    input digest is retained separately. No directions, source tensors or
    operational/guard-expanded ranks are copied into the feature bank.
    """
    from .factorization import AnchorAcquisition

    row = validate_rank_observation(observation)
    payload = acquisition.to_dict() if isinstance(acquisition, AnchorAcquisition) else acquisition
    if not isinstance(payload, Mapping):
        raise AnalysisError("feature-bank export requires the shared AnchorAcquisition or its complete JSON")
    spec = _fixed_detector_spec(payload.get("detector_spec"))
    features, width = _validate_fixed_acquisition(payload, row, spec)
    payload = json.loads(json.dumps(payload, allow_nan=False))
    acquisition_hash = canonical_sha256(payload)
    records = [
        {
            **{name: row[name] for name in IDENTITY_FIELDS}, "tau": tau,
            "input_sha256": row["input_sha256"], "detector_sha256": spec.fingerprint,
            "estimate_kind": "online_detector", "solver": spec.solver,
            "rank_statistic": "approximate_spectral_crossing", "available_after_layer": row["layer"],
            "status": "unresolved" if features[str(tau)] is None else "resolved",
            "estimated_rank": features[str(tau)], "maximum_width": width,
            "acquisition": payload, "acquisition_sha256": acquisition_hash,
        }
        for tau in spec.feature_energies
    ]
    validate_detector_records([row], records, detector_spec=spec)
    return records


def _prompt_equal_value(rows, value):
    by_group = defaultdict(lambda: defaultdict(list))
    for row in rows:
        by_group[row["prompt_group"]][row["seed"]].append(value(row))
    return _mean(_mean(_mean(values) for values in seeds.values()) for seeds in by_group.values())


def _trajectory_inventory(rows):
    cases = {}
    for row in rows:
        for name in ("case_id", "prompt_group", "cell", "trajectory_id"):
            _text(row[name], name)
        _integer(row["seed"], "trajectory seed")
        if row["split"] != "calibration" or row["cell"] not in CELLS:
            raise AnalysisError("acquisition admission requires declared calibration trajectories")
        _integer(row["boundary"], "trajectory boundary")
        identity = {name: row[name] for name in
                    ("case_id", "prompt_group", "seed", "cell", "split", "trajectory_id")}
        case = cases.setdefault(row["case_id"], {**identity, "boundaries": set()})
        if any(case[name] != value for name, value in identity.items()):
            raise AnalysisError("one trajectory cannot mix case/group/seed/cell/split bindings")
        case["boundaries"].add(row["boundary"])
    result = []
    for case_id in sorted(cases):
        case = cases[case_id]
        boundaries = sorted(case["boundaries"])
        if boundaries != list(range(len(boundaries))):
            raise AnalysisError("acquisition admission requires consecutive boundaries from startup")
        result.append({**case, "boundaries": boundaries})
    return result


def _acquisition_admission(inventory, records, required, active_by_regime, mode, *, profile_available=True):
    """Replay required union acquisitions, not encoder fits or closed-loop quality."""
    actual = {
        (row["case_id"], row["boundary"], row["layer"], row["target"], row["tau"]): row
        for row in records
    }
    states, events, failures, complete_cases = {}, [], [], []
    union = set(required)
    for case in inventory:
        failed = None
        for boundary in case["boundaries"]:
            marker = (case["case_id"], boundary)
            if not profile_available:
                states[marker] = {"status": "profile_unavailable",
                                  "failure_reason": "no resolved training detector baseline",
                                  "failure_boundary": None}
                continue
            if failed is not None:
                states[marker] = {"status": "not_executed", "failure_reason": "prior trajectory acquisition abort",
                                  "failure_boundary": failed["boundary"]}
                continue
            regime = "startup" if boundary == 0 else "later"
            active = set(active_by_regime.get(regime, ()))
            if not active <= union:
                raise AnalysisError("active anchors must belong to the declared acquisition union")
            ordered = sorted(union)
            if mode == "whole_clean_staged":
                ordered = [*sorted(active), *sorted(union - active)]
            for layer, target in ordered:
                key = (case["case_id"], boundary, layer, target, required[(layer, target)])
                if key not in actual:
                    raise AnalysisError("declared anchor union is missing a source-bound acquisition record")
                record = actual[key]
                event = {
                    **{name: case[name] for name in
                       ("case_id", "prompt_group", "seed", "cell", "split", "trajectory_id")},
                    "boundary": boundary, "layer": layer, "target": target, "tau": key[-1],
                    "active_in_regime": (layer, target) in active, "status": record["status"],
                    "acquisition_sha256": record["acquisition_sha256"],
                    "attempted_acquisition_work": record["acquisition"]["metrics"],
                }
                events.append(event)
                if record["status"] != "resolved":
                    failed = {**event, "failure_reason": "fixed_anchor_feature_unresolved"}
                    failures.append(failed)
                    break
            states[marker] = {
                "status": "complete" if failed is None else "failed",
                "failure_reason": None if failed is None else failed["failure_reason"],
                "failure_boundary": None if failed is None else boundary,
            }
        if profile_available and failed is None:
            complete_cases.append(case["case_id"])
    complete_set = set(complete_cases)
    groups = sorted({case["prompt_group"] for case in inventory})
    complete_groups = [
        group for group in groups if all(case["case_id"] in complete_set
                                        for case in inventory if case["prompt_group"] == group)
    ]
    return {
        "states": states, "complete_cases": complete_cases, "complete_groups": complete_groups,
        "expected_groups": len(groups), "coverage": len(complete_groups) / len(groups) if groups else 0,
        "acquisition_attempts": events, "failures": failures,
        "required_union": [{"layer": layer, "target": target, "tau": required[(layer, target)]}
                           for layer, target in sorted(union)],
        "cost_scope": "RAW-bank acquisition work replayed once per required union key; not new fits or full encoder cost",
    }


def _calibration_metrics(predictions):
    if not predictions:
        raise AnalysisError("calibration fold has no prediction targets")
    names = (
        "definite_underprediction", "possible_underprediction", "shortfall_rank_upper",
        "excess_rank_lower", "excess_rank_upper", "excess_bytes_proxy_lower",
        "excess_bytes_proxy_upper", "predicted_guard_hit_lower", "predicted_guard_hit_upper",
        "unavailable_anchor_count", "current_anchor_count", "lagged_anchor_count",
    )
    groups = {row["prompt_group"] for row in predictions}
    complete = {group for group in groups if all(
        row["trajectory_complete"] and row["status"] == "complete"
        for row in predictions if row["prompt_group"] == group
    )}
    scored = [row for row in predictions if row["prompt_group"] in complete]
    result = {name: _prompt_equal_value(scored, lambda row: row[name]) if scored else None for name in names}
    result.update(
        prompt_group_count=len(groups), complete_prompt_group_count=len(complete),
        coverage=len(complete) / len(groups), selection_eligible=len(complete) == len(groups),
        scored_observation_count=len(scored),
        failed_or_unexecuted_prediction_count=sum(row["status"] != "complete" for row in predictions),
        incomplete_groups=sorted(groups - complete),
        observation_count=len(predictions), measured_guard_frequency=None,
        measured_packet_bytes=None,
        cost_kind="rank and declared-size factor storage proxies, not actual packets or HEVC byte predictions",
        guard_kind="rank-shortfall proxy only; approximate/stored-basis guards can fail at sufficient oracle rank",
    )
    return result


def _calibration_objective(metrics):
    return (-metrics["coverage"], *(math.inf if metrics[name] is None else metrics[name] for name in (
        "possible_underprediction", "shortfall_rank_upper", "excess_bytes_proxy_upper",
    )))


def calibrate_selector(
    observations: Sequence[Mapping[str, Any]], expected: Sequence[Mapping[str, Any]],
    detectors: Sequence[Mapping[str, Any]], *,
    cell: str, target: str, tau: float, history_regime: str, execution_mode: str,
    rank_buckets: Sequence[int], detector_spec=None, profile_builder: Callable | None = None,
    rank_predictor: Callable | None = None, predictor_sha256: str | None = None,
    anchor_candidate_layers: Sequence[int] | None = None,
    margins: Sequence[int] = (0, 8, 16), uniform_quantiles: Sequence[float] = (.5, .9, 1.0),
    basis_itemsize: int = 2, coefficient_itemsize: int = 4, pair_shortlist: int = 8,
) -> dict:
    """LOPO calibration, emitting the online module's own serialized profiles.

    By default this imports online_rank.profile_builder and rank_predictor,
    sharing the actual deployed ratio/bucketing implementation. Explicit
    callbacks are for source-bound alternative predictors or CPU fixtures only.
    ``profile_builder(parameters)`` returns an online-module profile with
    ``to_dict()``.  ``rank_predictor(profile, *, layer, current_anchors,
    previous_anchors, previous_regime)`` returns an admissible integer starting
    rank.  These callbacks must bridge to the *same* deployment predictor; this
    module deliberately does not implement a second rank-policy class or a
    second ratio predictor.

    Parameters supplied to the builder are layer_template, detector_baselines
    (startup/later), anchor_layers, rank_margin, rank_buckets, policy_kind,
    uniform_rank, cell/target/tau/history_regime/execution_mode and training_groups.
    Builder/predictor source identity and the full fixed detector acquisition
    recipe are retained in the frozen report. The feature bank is acquired
    independently of every LOPO template, policy and prompt/generation seed.

    All 0/1/2/4-anchor candidates and <=3 fixed-uniform ladder controls use
    training-group-only fits.  Anchor selection is deterministic forward search
    on LOPO error, with all supplied margins tried.  The 2->4 step tries pairs
    from at most ``pair_shortlist`` remaining single-anchor CV finalists; it
    never constructs an inadmissible three-anchor deployment policy.  The CV is used for
    selection, not an unbiased final holdout estimate or a quality guarantee.
    A component's proposed anchors are required at every trajectory boundary,
    including inactive regimes. An unresolved acquisition aborts that commit and
    all later commits; incomplete prompt groups remain in coverage, not proxy
    means. Selection is coverage-first. Binding all four profiles replays their
    combined union, deduplicates acquisition work and rejects any trajectory abort.
    """
    if any(row.get("split") != "calibration" for row in observations) or any(
        row.get("split") != "calibration" for row in expected
    ):
        raise AnalysisError("heldout/development inputs are prohibited throughout calibration")
    rows = validate_census(observations, expected)
    deployment_callbacks = profile_builder is None and rank_predictor is None
    if deployment_callbacks:
        from .online_rank import profile_builder, rank_predictor

        actual_predictor_hash = file_sha256(Path(__file__).with_name("online_rank.py"))
        if predictor_sha256 is not None and predictor_sha256 != actual_predictor_hash:
            raise AnalysisError("deployed predictor implementation differs from its requested source hash")
        predictor_sha256 = actual_predictor_hash
    elif not callable(profile_builder) or not callable(rank_predictor):
        raise AnalysisError("supply both source-bound callbacks or use both deployment defaults")
    spec = _fixed_detector_spec(detector_spec)
    detector_map = validate_detector_records(rows, detectors, detector_spec=spec)
    if target not in ("K", "V") or tau not in (.95, .99):
        raise AnalysisError("calibrate K or V at r95 or r99 separately")
    if history_regime not in ("startup", "later") or execution_mode not in (
        "causal_lagged", "whole_clean_staged",
    ):
        raise AnalysisError("startup/later and causal_lagged/whole_clean_staged must remain explicit")
    _digest(predictor_sha256, "predictor_sha256")
    _integer(basis_itemsize, "basis_itemsize", 1)
    _integer(coefficient_itemsize, "coefficient_itemsize", 1)
    _integer(pair_shortlist, "pair_shortlist", 2)
    if pair_shortlist > 16:
        raise AnalysisError("bound pair search to at most sixteen shortlisted layers")
    selected = [row for row in rows if row["cell"] == cell and row["target"] == target]
    if not selected:
        raise AnalysisError("requested calibration cell/target is absent")
    layers, channels, n_rows = selected[0]["n_layers"], selected[0]["n_channels"], selected[0]["n_rows"]
    buckets = tuple(rank_buckets)
    if not buckets or any(type(rank) is not int or not 1 <= rank <= channels for rank in buckets) or (
        tuple(sorted(set(buckets))) != buckets or buckets[-1] != min(channels, n_rows)
    ):
        raise AnalysisError("rank buckets must increase uniquely through the full admissible rank")
    margins = tuple(margins)
    if not margins or any(type(value) is not int or value < 0 for value in margins) or len(set(margins)) != len(margins):
        raise AnalysisError("rank margins must be explicit unique nonnegative integers")
    quantiles = tuple(uniform_quantiles)
    if not 1 <= len(quantiles) <= 3 or len(set(quantiles)) != len(quantiles):
        raise AnalysisError("the fixed-uniform ladder must contain one to three unique quantiles")
    for value in quantiles:
        _finite(value, "uniform quantile", 0, 1)
    available_layers = tuple(range(layers)) if anchor_candidate_layers is None else tuple(anchor_candidate_layers)
    if len(available_layers) < 4 or len(set(available_layers)) != len(available_layers) or any(
        type(layer) is not int or not 0 <= layer < layers for layer in available_layers
    ):
        raise AnalysisError("0/1/2/4 anchor search requires at least four distinct valid candidate layers")
    boundaries = defaultdict(set)
    for row in selected:
        boundaries[row["case_id"]].add(row["boundary"])
    order = {case: sorted(values) for case, values in boundaries.items()}
    if execution_mode == "causal_lagged" and any(
        values != list(range(len(values))) for values in order.values()
    ):
        raise AnalysisError("causal/lagged calibration needs consecutive boundaries from startup, not unobserved state")
    inventory = _trajectory_inventory(selected)
    regime = lambda row: "startup" if row["boundary"] == order[row["case_id"]][0] else "later"
    scored = [row for row in selected if regime(row) == history_regime]
    groups = sorted({row["prompt_group"] for row in scored})
    if len(groups) < 2:
        raise AnalysisError("leave-one-prompt-group-out calibration needs at least two groups")
    if history_regime == "later" and any(len(values) < 2 for values in order.values()):
        raise AnalysisError("later calibration needs genuine previous-boundary state for every case")
    level = "r95" if tau == .95 else "r99"
    detector_index = {}
    for row in selected:
        record = detector_map[(tuple(row[name] for name in IDENTITY_FIELDS), tau)]
        detector_index[(row["case_id"], row["boundary"], row["layer"])] = record

    def bucket(value):
        value = max(1, min(channels, math.ceil(value)))
        return next(rank for rank in buckets if rank >= value)

    training_cache = {}

    def fit(training_groups, anchors, margin, uniform_q):
        training_key = tuple(sorted(training_groups))
        if training_key not in training_cache:
            training = [row for row in selected if row["prompt_group"] in training_groups]
            training_targets = [row for row in training if regime(row) == history_regime]
            template = [
                _prompt_equal_value([row for row in training_targets if row["layer"] == layer],
                                    lambda row: row[level]["upper"])
                for layer in range(layers)
            ]
            baselines = {}
            for history in ("startup", "later"):
                baselines[history] = []
                for layer in range(layers):
                    feature_rows = [
                        row for row in training if regime(row) == history and row["layer"] == layer
                        and detector_index[(row["case_id"], row["boundary"], layer)]["status"] == "resolved"
                    ]
                    baselines[history].append(
                        _prompt_equal_value(feature_rows, lambda row: max(
                            1, detector_index[(row["case_id"], row["boundary"], layer)]["estimated_rank"]
                        )) if feature_rows else None
                    )
            maxima = [
                max(row[level]["upper"] for row in training_targets if row["prompt_group"] == group)
                for group in training_key
            ]
            training_cache[training_key] = template, baselines, maxima
        template, baselines, group_maxima = training_cache[training_key]
        uniform = None
        if uniform_q is not None:
            uniform = bucket(_quantile(group_maxima, uniform_q) + margin)
        parameters = {
            "cell": cell, "target": target, "tau": tau, "history_regime": history_regime,
            "execution_mode": execution_mode, "layer_template": list(template),
            "detector_baselines": {name: list(values) for name, values in baselines.items()},
            "anchor_layers": list(anchors), "rank_margin": margin,
            "rank_buckets": list(buckets), "policy_kind": "fixed_uniform" if uniform_q is not None else (
                "fixed_layerwise" if not anchors else "anchors"
            ),
            "uniform_rank": uniform, "training_groups": sorted(training_groups),
            "detector_fingerprint": spec.fingerprint,
        }
        if anchors and any(baselines[history_regime][layer] is None for layer in anchors):
            return None, parameters, None
        profile = profile_builder(parameters)
        if not hasattr(profile, "to_dict"):
            raise AnalysisError("profile builder must return the deployment module's serializable profile")
        serialized = profile.to_dict()
        canonical_sha256(serialized)
        return profile, parameters, serialized

    def predict_rows(profile, validation_rows, anchors):
        predictions = []
        case_ids = {row["case_id"] for row in validation_rows}
        required = {(anchor, target): tau for anchor in anchors}
        admission = _acquisition_admission(
            [case for case in inventory if case["case_id"] in case_ids],
            list(detector_index.values()), required, {history_regime: set(required)}, execution_mode,
            profile_available=profile is not None,
        )
        complete_cases = set(admission["complete_cases"])
        for row in validation_rows:
            case, boundary, layer = row["case_id"], row["boundary"], row["layer"]
            previous_index = order[case].index(boundary) - 1
            previous_boundary = order[case][previous_index] if previous_index >= 0 else None
            previous_regime = None if previous_boundary is None else (
                "startup" if previous_boundary == order[case][0] else "later"
            )
            state = admission["states"][(case, boundary)]
            base = {
                **{name: row[name] for name in IDENTITY_FIELDS}, "tau": tau,
                "history_regime": history_regime, "status": state["status"],
                "trajectory_complete": case in complete_cases,
                "failure_reason": state["failure_reason"], "failure_boundary": state["failure_boundary"],
                "oracle_rank_lower": row[level]["lower"], "oracle_rank_upper": row[level]["upper"],
                "previous_boundary": previous_boundary, "previous_regime": previous_regime,
            }
            if state["status"] != "complete":
                predictions.append({
                    **base, **{name: None for name in (
                        "predicted_starting_rank", "definite_underprediction", "possible_underprediction",
                        "shortfall_rank_upper", "excess_rank_lower", "excess_rank_upper",
                        "excess_bytes_proxy_lower", "excess_bytes_proxy_upper",
                        "predicted_guard_hit_lower", "predicted_guard_hit_upper",
                        "current_anchor_count", "lagged_anchor_count", "unavailable_anchor_count",
                    )}, "current_anchor_layers": [], "lagged_anchor_layers": [],
                })
                continue
            current, previous = {}, {}
            for anchor in anchors:
                record = detector_index[(case, boundary, anchor)]
                if record["status"] == "resolved" and (
                    execution_mode == "whole_clean_staged" or record["available_after_layer"] < layer
                ):
                    current[anchor] = record["estimated_rank"]
                if execution_mode == "causal_lagged" and previous_boundary is not None:
                    record = detector_index[(case, previous_boundary, anchor)]
                    if record["status"] == "resolved":
                        previous[anchor] = record["estimated_rank"]
            rank = rank_predictor(profile, layer=layer, current_anchors=dict(current),
                                  previous_anchors=dict(previous), previous_regime=previous_regime)
            if type(rank) is not int or rank not in buckets:
                raise AnalysisError("deployment predictor returned a non-admissible starting rank")
            low, high = row[level]["lower"], row[level]["upper"]
            factor_bytes_per_rank = channels * basis_itemsize + n_rows * coefficient_itemsize
            predictions.append({
                **base, "predicted_starting_rank": rank,
                "definite_underprediction": int(rank < low), "possible_underprediction": int(rank < high),
                "shortfall_rank_upper": max(0, high - rank),
                "excess_rank_lower": max(0, rank - high), "excess_rank_upper": max(0, rank - low),
                "excess_bytes_proxy_lower": max(0, rank - high) * factor_bytes_per_rank,
                "excess_bytes_proxy_upper": max(0, rank - low) * factor_bytes_per_rank,
                "predicted_guard_hit_lower": int(rank < low), "predicted_guard_hit_upper": int(rank < high),
                "current_anchor_count": len(current),
                "lagged_anchor_count": len(set(previous) - set(current)),
                "unavailable_anchor_count": len(set(anchors) - set(current) - set(previous)),
                "current_anchor_layers": sorted(current), "lagged_anchor_layers": sorted(set(previous) - set(current)),
            })
        return predictions, {name: value for name, value in admission.items() if name != "states"}

    cache = {}

    def evaluate(anchors, margin, uniform_q=None):
        key = (tuple(sorted(anchors)), margin, uniform_q)
        if key in cache:
            return cache[key]
        all_predictions, folds = [], []
        for held_group in groups:
            training_groups = set(groups) - {held_group}
            profile, parameters, serialized = fit(training_groups, key[0], margin, uniform_q)
            validation_rows = [row for row in scored if row["prompt_group"] == held_group]
            predictions, admission = predict_rows(profile, validation_rows, key[0])
            all_predictions.extend(predictions)
            folds.append({
                "validation_group": held_group, "training_groups": sorted(training_groups),
                "fitted_profile_sha256": canonical_sha256(serialized) if serialized is not None else None,
                "fitted_parameters_sha256": canonical_sha256(parameters),
                "metrics": _calibration_metrics(predictions),
                "acquisition_admission": admission,
                "failed_or_incomplete_predictions": [
                    row for row in predictions if not row["trajectory_complete"]
                ],
            })
        value = {
            "anchor_layers": list(key[0]), "anchor_budget": len(key[0]), "rank_margin": margin,
            "uniform_quantile": uniform_q, "policy_kind": "fixed_uniform" if uniform_q is not None else (
                "fixed_layerwise" if not key[0] else "anchors"
            ),
            "metrics": _calibration_metrics(all_predictions), "folds": folds,
        }
        cache[key] = value
        return value

    def best(candidates):
        return min(candidates, key=lambda candidate: (
            *_calibration_objective(candidate["metrics"]), candidate["anchor_budget"],
            candidate["rank_margin"], tuple(candidate["anchor_layers"]),
        ))

    finalists = [best([evaluate((), margin) for margin in sorted(margins)])]
    current_anchors = ()
    single_best = {}
    for size in (1, 2):
        candidates = [
            evaluate(tuple(sorted((*current_anchors, layer))), margin)
            for layer in sorted(available_layers) if layer not in current_anchors
            for margin in sorted(margins)
        ]
        winner = best(candidates)
        if size == 1:
            single_best = {
                layer: best([row for row in candidates if row["anchor_layers"] == [layer]])
                for layer in available_layers
            }
        current_anchors = tuple(winner["anchor_layers"])
        finalists.append(winner)
    remaining = sorted(
        (layer for layer in available_layers if layer not in current_anchors),
        key=lambda layer: (*_calibration_objective(single_best[layer]["metrics"]), layer),
    )[:pair_shortlist]
    finalists.append(best([
        evaluate(tuple(sorted((*current_anchors, *pair))), margin)
        for pair in combinations(remaining, 2) for margin in sorted(margins)
    ]))
    for quantile in sorted(quantiles):
        finalists.append(best([evaluate((), margin, quantile) for margin in sorted(margins)]))
    chosen = best(finalists)
    if not chosen["metrics"]["selection_eligible"]:
        raise AnalysisError("no candidate completes every declared calibration trajectory")
    selected_predictions = []
    for held_group in groups:
        profile, _, _ = fit(set(groups) - {held_group}, tuple(chosen["anchor_layers"]),
                            chosen["rank_margin"], chosen["uniform_quantile"])
        predictions, _ = predict_rows(
            profile, [row for row in scored if row["prompt_group"] == held_group],
            tuple(chosen["anchor_layers"]),
        )
        selected_predictions.extend(predictions)
    frozen = []
    for candidate in finalists:
        _, parameters, serialized = fit(set(groups), tuple(candidate["anchor_layers"]),
                                        candidate["rank_margin"], candidate["uniform_quantile"])
        frozen.append({
            **{name: candidate[name] for name in ("policy_kind", "anchor_budget", "anchor_layers",
                                                 "rank_margin", "uniform_quantile")},
            "online_profile": serialized,
            "profile_sha256": canonical_sha256(serialized) if serialized is not None else None,
            "fitted_parameters": parameters, "selection_cv_metrics": candidate["metrics"],
            "selected": candidate is chosen,
        })
    admitted_sources = [row for row in selected if row["layer"] in chosen["anchor_layers"]]
    admitted_keys = {tuple(row[name] for name in IDENTITY_FIELDS) for row in admitted_sources}
    admission_bank = {
        "observations": admitted_sources,
        "detectors": [record for (key, _), record in detector_map.items() if key in admitted_keys],
    }
    if deployment_callbacks and file_sha256(Path(__file__).with_name("online_rank.py")) != predictor_sha256:
        raise AnalysisError("deployment predictor source changed while calibrating; discard and rerun after code review")
    if file_sha256(Path(__file__).with_name("factorization.py")) != spec.implementation_fingerprint:
        raise AnalysisError("fixed detector source changed while calibrating; discard and rebind the recipe")
    return {
        "schema": "reelcache_rank_calibration_report_v1", "split": "calibration",
        "cell": cell, "target": target, "tau": tau, "history_regime": history_regime,
        "execution_mode": execution_mode, "calibration_groups": groups,
        "predictor_sha256": predictor_sha256,
        "using_deployment_predictor": deployment_callbacks,
        "detector_sha256": spec.fingerprint, "detector_spec": spec.to_dict(),
        "observation_sha256": canonical_sha256(rows), "detector_records_sha256": canonical_sha256(detectors),
        "expected_sha256": canonical_sha256(expected), "anchor_budgets": [0, 1, 2, 4],
        "anchor_candidate_layers": sorted(available_layers), "pair_shortlist": pair_shortlist,
        "selection_rule": "complete-trajectory prompt-group coverage first; then LOPO underprediction, shortfall, bytes proxy",
        "acquisition_scope": "candidate anchors are required union keys at EVERY trajectory boundary, active or inactive",
        "trajectory_inventory": inventory, "final_acquisition_bank": admission_bank,
        "selection_cv_is_holdout": False,
        "proxy_definition": {"basis_itemsize": basis_itemsize, "coefficient_itemsize": coefficient_itemsize,
                             "metadata_included": False, "actual_packet_bytes": False},
        "staging_required": execution_mode == "whole_clean_staged",
        "staging_bytes_measured": None, "quality_guarantee": False,
        "profiles": frozen,
        "selected_validation_predictions": selected_predictions,
        "candidates": [
            cache[key] for key in sorted(cache, key=lambda key: (len(key[0]), key[0], key[1],
                                                                 -1 if key[2] is None else key[2]))
        ],
        "unimplemented_measurement_prerequisites": [
            "source-verified native calibration corpus and deployment detector measurements",
            "actual accepted packets, full-basis/metadata closures and measured current-input guard/work counts",
            "closed-loop same-run RAW quality and matched actual HEVC-byte frontier",
        ],
    }


def freeze_calibration(report: Mapping[str, Any], path: str | Path) -> str:
    """Persist deterministic profiles without blessing them as quality-qualified."""
    if report.get("schema") != "reelcache_rank_calibration_report_v1" or report.get("split") != "calibration":
        raise AnalysisError("only a calibration-only selector report can be frozen")
    if not report.get("profiles") or sum(bool(profile["selected"]) for profile in report["profiles"]) != 1:
        raise AnalysisError("a calibration report must freeze exactly one selected profile")
    if not next(profile for profile in report["profiles"] if profile["selected"])[
        "selection_cv_metrics"
    ].get("selection_eligible"):
        raise AnalysisError("cannot freeze an incomplete/failed selected calibration candidate")
    spec = _fixed_detector_spec(report.get("detector_spec"))
    if report.get("detector_sha256") != spec.fingerprint:
        raise AnalysisError("calibration report detector recipe checksum mismatch")
    if any(entry["fitted_parameters"].get("detector_fingerprint") != spec.fingerprint or (
        entry["online_profile"] is None and (
            entry["selected"] or entry["selection_cv_metrics"]["selection_eligible"]
        )
    ) or (
        entry["online_profile"] is not None
        and entry["online_profile"].get("detector_fingerprint") != spec.fingerprint
    ) for entry in report["profiles"]):
        raise AnalysisError("frozen profiles must retain their complete acquisition recipe fingerprint")
    body = {
        "schema": "reelcache_rank_calibration_freeze_v1",
        **{name: report[name] for name in (
            "split", "cell", "target", "tau", "history_regime", "execution_mode", "calibration_groups",
            "predictor_sha256", "detector_sha256", "observation_sha256", "expected_sha256",
            "using_deployment_predictor", "detector_spec",
        )},
        "calibration_report_sha256": canonical_sha256(report), "profiles": report["profiles"],
        "quality_guarantee": False, "selection_cv_is_holdout": False,
    }
    digest = canonical_sha256(body)
    write_json(path, {"sha256": digest, **body})
    return digest


def static_layer_ranks(profiles: Mapping[str, Any]) -> dict[str, list[int]]:
    """Compile explicitly selected frozen zero-anchor templates to STATIC ranks.

    Insert the result under representation.layer_ranks, without rank_policy or
    uniform rank_k/rank_v. Each vector is constant for the whole trajectory.
    The caller chooses and records the calibration/history-regime provenance;
    this function neither selects profiles nor fits data, grants resources or
    turns a RankPolicy initializer into a fixed-rank encoding guarantee.
    """
    from .errors import ConfigurationError
    from .online_rank import RankPredictorProfile, rank_predictor

    if not isinstance(profiles, Mapping) or set(profiles) != {"K", "V"}:
        raise AnalysisError("static layer_ranks requires exactly one frozen K and V template")
    ranks, scopes = {}, set()
    for target in ("K", "V"):
        value = profiles[target]
        try:
            profile = value if isinstance(value, RankPredictorProfile) else RankPredictorProfile.from_dict(value)
        except (ConfigurationError, TypeError, ValueError) as error:
            raise AnalysisError("static layer_ranks requires complete frozen predictor profiles") from error
        if profile.target != target or profile.policy_kind != "fixed_layerwise" or profile.anchor_layers:
            raise AnalysisError("static layer_ranks must use zero-anchor fixed_layerwise templates, not adaptive anchors")
        cell = CELLS.get(profile.cell)
        if cell is None or len(profile.layer_template) != cell.layers:
            raise AnalysisError("static template length differs from the approved cell's complete layer inventory")
        scopes.add((profile.cell, profile.history_regime, profile.execution_mode,
                    profile.training_groups, profile.detector_fingerprint))
        ranks[target] = [
            rank_predictor(
                profile, layer=layer, current_anchors={}, previous_anchors={},
                previous_regime=None, chunk_ordinal=1 if profile.history_regime == "startup" else 2,
            )
            for layer in range(cell.layers)
        ]
        if any(type(rank) is not int or not 1 <= rank <= min(cell.tokens, cell.channels)
               for rank in ranks[target]):
            raise AnalysisError("static layer rank exceeds its original matrix's admissible range")
    if len(scopes) != 1:
        raise AnalysisError("static K/V templates must share one explicit cell/calibration/regime scope")
    return ranks


def calibration_union_admission(reports: Sequence[Mapping[str, Any]]) -> dict:
    """Replay the selected policy's entire startup/later K/V acquisition union.

    The returned work/failure records refer to measured RAW-bank acquisitions,
    not executed CV fits, total encoder cost, or a closed-loop quality guarantee.
    """
    from .online_rank import RankPredictorProfile

    if not reports:
        raise AnalysisError("union admission requires selected calibration reports")
    inventory, common, required, active, actual = None, set(), {}, defaultdict(set), {}
    labels = set()
    sources = {}
    for report in reports:
        selected = [entry for entry in report["profiles"] if entry["selected"]]
        if len(selected) != 1 or selected[0]["online_profile"] is None:
            raise AnalysisError("union admission requires one available selected profile per report")
        profile = RankPredictorProfile.from_dict(selected[0]["online_profile"])
        if canonical_sha256(selected[0]["online_profile"]) != selected[0]["profile_sha256"] or any(
            getattr(profile, name) != report[name]
            for name in ("cell", "target", "tau", "history_regime", "execution_mode")
        ) or tuple(profile.training_groups) != tuple(sorted(report["calibration_groups"])):
            raise AnalysisError("union profile differs from its frozen grouped-calibration identity")
        label = (profile.history_regime, profile.target)
        if label in labels:
            raise AnalysisError("duplicate target/regime in union admission")
        labels.add(label)
        spec = _fixed_detector_spec(report.get("detector_spec"))
        if profile.detector_fingerprint != spec.fingerprint or report["detector_sha256"] != spec.fingerprint:
            raise AnalysisError("union admission detector recipe differs across selected profiles")
        proposed = report.get("trajectory_inventory")
        if not isinstance(proposed, list) or not proposed:
            raise AnalysisError("union admission requires the full declared trajectory inventory")
        normalized = _trajectory_inventory([
            {**case, "boundary": boundary} for case in proposed for boundary in case["boundaries"]
        ])
        if normalized != proposed:
            raise AnalysisError("trajectory inventory is incomplete, duplicated or noncanonical")
        if {case["prompt_group"] for case in proposed} != set(report["calibration_groups"]) or any(
            case["cell"] != profile.cell for case in proposed
        ):
            raise AnalysisError("union inventory differs from its calibrated cell/groups")
        if inventory is not None and inventory != proposed:
            raise AnalysisError("all startup/later profiles require identical calibration trajectories")
        inventory = proposed
        common.add((profile.cell, profile.execution_mode, spec.fingerprint,
                    report["observation_sha256"], report["expected_sha256"], report["detector_records_sha256"]))
        keys = {(layer, profile.target) for layer in profile.anchor_layers}
        active[profile.history_regime].update(keys)
        for key in keys:
            if key in required and required[key] != profile.tau:
                raise AnalysisError("one union key cannot change detector threshold between regimes")
            required[key] = profile.tau
        bank = report.get("final_acquisition_bank")
        if not isinstance(bank, dict) or set(bank) != {"observations", "detectors"}:
            raise AnalysisError("selected profiles require their source-bound final admission bank")
        if bank["observations"]:
            _check_inventory(bank["observations"], complete_layers=False)
        validated = [validate_rank_observation(row) for row in bank["observations"]]
        if any((row["layer"], row["target"]) not in keys for row in validated):
            raise AnalysisError("admission bank contains keys outside its selected profile")
        expected_bank = {
            (case["case_id"], boundary, layer, target)
            for case in proposed for boundary in case["boundaries"] for layer, target in keys
        }
        if {(row["case_id"], row["boundary"], row["layer"], row["target"]) for row in validated} != expected_bank:
            raise AnalysisError("union admission bank must cover every selected key at every trajectory boundary")
        declared_cases = {case["case_id"]: case for case in proposed}
        if any(any(row[name] != declared_cases[row["case_id"]][name]
                   for name in ("prompt_group", "seed", "cell", "split", "trajectory_id"))
               for row in validated):
            raise AnalysisError("union source differs from its declared original trajectory")
        records = validate_detector_records(validated, bank["detectors"], detector_spec=spec) if validated else {}
        if not validated and (keys or bank["detectors"]):
            raise AnalysisError("selected union anchors have no source-bound acquisition bank")
        for row in validated:
            identity = tuple(row[name] for name in IDENTITY_FIELDS)
            if identity in sources and sources[identity] != row:
                raise AnalysisError("union profiles disagree on their original source observations")
            sources[identity] = row
        for key, row in records.items():
            if key in actual and actual[key] != row:
                raise AnalysisError("union profiles disagree on their fixed detector acquisition")
            actual[key] = row
    if len(common) != 1:
        raise AnalysisError("union admission must use one source-bound census and acquisition corpus")
    regimes = ("startup", "later") if any(len(case["boundaries"]) > 1 for case in inventory) else ("startup",)
    if labels != {(regime, target) for regime in regimes for target in ("K", "V")}:
        raise AnalysisError("union admission requires every supported startup/later K/V profile")
    mode = next(iter(common))[1]
    admission = _acquisition_admission(inventory, list(actual.values()), required, active, mode)
    return {name: value for name, value in admission.items() if name != "states"}


def bind_calibration_reports(base_policy: Any, reports: Sequence[Mapping[str, Any]], *, policy_id: str):
    """Bind selected startup/later K/V profiles into the *existing* online RankPolicy.

    Resource allowances, source identities, codec and solver come unchanged
    from the parent-supplied base policy.  Binding grants no launch permission
    and does not make the profiles quality-qualified.  Static uniform/layerwise
    reference arms still use fixed CodecSpecs, not this adaptive guard.
    """
    from .online_rank import RankPolicy, RankPredictorProfile, bind_predictor_profiles

    if not isinstance(base_policy, RankPolicy) or base_policy.solver != "qr_q2":
        raise AnalysisError("calibrated detector profiles require the parent's bound QR/q=2 RankPolicy")
    _text(policy_id, "policy_id")
    if not reports:
        raise AnalysisError("provide every target/history-regime calibration report")
    profiles, provenance = {}, {}
    predictor_hash = file_sha256(Path(__file__).with_name("online_rank.py"))
    common = set()
    for report in reports:
        if report.get("schema") != "reelcache_rank_calibration_report_v1" or report.get("split") != "calibration" or (
            report.get("using_deployment_predictor") is not True
        ):
            raise AnalysisError("only calibration-only profiles fitted by the deployment predictor can bind")
        if report["predictor_sha256"] != predictor_hash:
            raise AnalysisError("predictor source changed after calibration; refit before binding")
        spec = _fixed_detector_spec(report.get("detector_spec"))
        if report.get("detector_sha256") != spec.fingerprint or base_policy.detector_spec != spec:
            raise AnalysisError("calibration and base policy must bind the SAME full frozen detector acquisition")
        if report["tau"] != base_policy.energy_target or report["cell"] not in CELLS:
            raise AnalysisError("calibration energy/cell differs from the bound policy")
        cell = CELLS[report["cell"]]
        if base_policy.geometry != (cell.frames, cell.token_height, cell.token_width) or (
            base_policy.model.layers != cell.layers
        ) or any(
            base_policy.model.channels_for(key) != cell.channels or
            base_policy.source_dtypes[key] != (cell.key_dtype if key.target == "K" else cell.value_dtype)
            for key in base_policy.keys
        ):
            raise AnalysisError("base policy differs from parent study geometry/source dtypes")
        selected = [entry for entry in report["profiles"] if entry["selected"]]
        if len(selected) != 1:
            raise AnalysisError("calibration report must select exactly one profile")
        entry = selected[0]
        if not entry["selection_cv_metrics"].get("selection_eligible"):
            raise AnalysisError("a failed/incomplete CV candidate cannot become the deployed selector")
        profile = RankPredictorProfile.from_dict(entry["online_profile"])
        if profile.detector_fingerprint != spec.fingerprint:
            raise AnalysisError("predictor profile differs from its frozen acquisition recipe")
        if canonical_sha256(entry["online_profile"]) != entry["profile_sha256"]:
            raise AnalysisError("selected predictor profile checksum mismatch")
        if any(getattr(profile, name) != report[name] for name in (
            "cell", "target", "tau", "history_regime", "execution_mode"
        )) or tuple(profile.training_groups) != tuple(sorted(report["calibration_groups"])):
            raise AnalysisError("selected profile differs from its grouped calibration scope")
        label = f"{profile.history_regime}:{profile.target}"
        if label in profiles:
            raise AnalysisError("duplicate calibrated target/history regime")
        profiles[label], provenance[label] = profile, canonical_sha256(report)
        common.add((profile.cell, profile.tau, profile.execution_mode, tuple(profile.training_groups),
                    report["detector_sha256"]))
    if len(common) != 1:
        raise AnalysisError("cannot mix cells, energy, modes, detector implementations or calibration groups")
    admission = calibration_union_admission(reports)
    if admission["coverage"] != 1 or admission["failures"]:
        raise AnalysisError("selected policy union aborts a declared calibration trajectory")
    bound_base = dataclasses.replace(
        base_policy, policy_id=policy_id,
        calibration_fingerprint=canonical_sha256({
            "base_calibration_fingerprint": base_policy.calibration_fingerprint,
            "reports": provenance, "predictor_sha256": predictor_hash,
            "union_admission_sha256": canonical_sha256(admission),
        }),
    )
    return bind_predictor_profiles(bound_base, profiles)
