import dataclasses
import hashlib
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from reelcache.errors import CapacityError, ConfigurationError, IntegrityError, StateError
from reelcache.study import (
    CELLS, NATIVE_SOURCE_FPS, advance_launch_receipt, create_protocol, file_sha256,
    historical_prompt_texts, identity,
    make_launch_receipt, read_json, related_prompts, resource_grant, source_closure,
    stage_assets, validate_protocol, validate_request, validate_request_batch, verify_assets,
    verify_launch_receipt, verify_source_closure, verify_protocol_sources,
    verify_production_protocol, write_json_new,
)


class StudyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        catalog = [
            {"prompt_en": text, "dimension": [dimension]}
            for text, dimension in (
                ("A quiet observatory beneath a starry sky", "temporal_flickering"),
                ("A blacksmith hammers glowing metal", "human_action"),
                ("a teapot beside a violin", "multiple_objects"),
                ("Wide snowy mountain landscape", "scene"),
                ("Still photograph of an antique compass", "temporal_flickering"),
                ("An otter dives through a river", "dynamic_degree"),
                ("a ladder behind a cactus", "spatial_relationship"),
                ("Bamboo forest beneath gentle rainfall", "scene"),
                ("A forgotten clock tower", "scene"),
            )
        ]
        self.catalog = self.root / "catalog.json"
        self.catalog.write_text(json.dumps(catalog))
        self.history = self.root / "history.json"
        self.history.write_text(json.dumps({"prompts": [{"text": "A forgotten clock tower"}]}))
        self.protocol = create_protocol(
            self.catalog, historical_paths=[self.history], calibration_groups=4, heldout_groups=4,
            expected_catalog_sha256=file_sha256(self.catalog),
        )
        self.production_patch = mock.patch(
            "reelcache.study.verify_production_protocol", side_effect=verify_protocol_sources,
        )
        self.production_patch.start()
        self.addCleanup(self.production_patch.stop)

    def tearDown(self):
        self.temp.cleanup()

    def test_geometry_and_source_ledger(self):
        expected = {
            "causvid480": (4680, 862617600),
            "magi720": (21600, 6370099200),
            "magi1080": (48960, 14438891520),
        }
        for name, (tokens, source_bytes) in expected.items():
            cell = CELLS[name]
            self.assertEqual(cell.tokens, tokens)
            self.assertEqual(cell.source_bytes, source_bytes)
            self.assertEqual(cell.width, cell.token_width * 16)
            self.assertEqual(cell.working_height, cell.token_height * 16)
        self.assertEqual(CELLS["magi1080"].crop_top, 4)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            CELLS["causvid480"].channels = 1024
        with self.assertRaises(TypeError):
            CELLS["new"] = CELLS["causvid480"]

    def test_prompt_groups_repeat_without_global_rng(self):
        import random
        before = random.getstate()
        again = create_protocol(
            self.catalog, historical_paths=[self.history], calibration_groups=4, heldout_groups=4,
            expected_catalog_sha256=file_sha256(self.catalog),
        )
        self.assertEqual(again, self.protocol)
        self.assertEqual(random.getstate(), before)
        self.assertEqual(validate_protocol(again), again)
        self.assertNotIn("A forgotten clock tower", [p["text"] for p in again["prompts"]])
        self.assertEqual({p["stratum"] for p in again["prompts"]}, {"static", "motion", "objects", "scenes"})

    def test_fixture_catalog_cannot_authorize_production(self):
        with self.assertRaises(StateError):
            verify_production_protocol(self.protocol)
        changed = json.loads(json.dumps(self.protocol))
        changed["seeds"] = [2, 3]
        changed.pop("protocol_id")
        changed["protocol_id"] = identity(changed)
        with self.assertRaises(StateError):
            verify_production_protocol(changed)

    def test_prompt_family_excludes_color_and_camera_variants(self):
        self.assertTrue(related_prompts("a red umbrella", "a blue umbrella"))
        self.assertTrue(related_prompts("A panda drinking coffee, pan left", "A panda drinking coffee, tilt up"))
        self.assertTrue(related_prompts(
            "A panda drinking coffee in a cafe in Paris, tilt up",
            "A panda drinking coffee in a cafe in Paris, in super slow motion",
        ))
        self.assertFalse(related_prompts("A blacksmith hammers glowing metal", "a ladder behind a cactus"))

    def test_explicit_history_fields_only(self):
        self.assertEqual(
            historical_prompt_texts([{
                "prompts": [{"text": "one prompt"}], "prompt": "another prompt",
                "description": "not a prompt", "text": "unrelated metadata",
            }]),
            {"one prompt", "another prompt"},
        )

    def test_protocol_tamper_and_related_split_rejected(self):
        altered = json.loads(json.dumps(self.protocol))
        altered["seeds"][0] = 2
        with self.assertRaises(IntegrityError):
            validate_protocol(altered)
        altered = json.loads(json.dumps(self.protocol))
        altered["prompts"][1]["text"] = altered["prompts"][0]["text"]
        altered.pop("protocol_id")
        altered["protocol_id"] = identity(altered)
        with self.assertRaises(IntegrityError):
            validate_protocol(altered)

    def test_strict_json_and_immutable_publication(self):
        path = self.root / "one.json"
        write_json_new(path, {"status": "planned"})
        with self.assertRaises(FileExistsError):
            write_json_new(path, {"status": "passed"})
        self.assertEqual(read_json(path), {"status": "planned"})
        self.assertEqual(list(self.root.glob(".one.json.*")), [])
        path.write_text('{"a": 1, "a": 2}')
        with self.assertRaises(IntegrityError):
            read_json(path)
        path.write_text('{"a": NaN}')
        with self.assertRaises(IntegrityError):
            read_json(path)

    def _closure(self):
        project = self.root / "project"
        (project / "reelcache").mkdir(parents=True)
        (project / "examples").mkdir()
        (project / "tests").mkdir()
        (project / "reelcache/core.py").write_text("ANSWER = 42\n")
        (project / "examples/driver.py").write_text("NATIVE = False\n")
        (project / "pyproject.toml").write_text("[project]\nname = 'fixture'\n")
        (project / "tests/test_fixture.py").write_text(
            "import unittest\nclass FixtureTests(unittest.TestCase):\n"
            "    def test_valid(self):\n        self.assertTrue(True)\n"
        )
        dependency = self.root / "external.py"
        dependency.write_text("DEPENDENCY = 1\n")
        patch = mock.patch("reelcache.study.PROJECT_ROOT", project)
        patch.start()
        self.addCleanup(patch.stop)
        return source_closure(project, [dependency])

    def _receipt(self):
        closure = self._closure()
        sources = {
            key: record["sha256"] for key, record in closure["files"].items()
            if not key.startswith("/")
        }
        inventory = ["tests.test_fixture.FixtureTests.test_valid"]
        verification = {
            "schema": "reelcache_cpu_verification_v1",
            "status": "passed", "tests_run": 1, "skipped": 0, "failures": 0, "errors": 0,
            "expected_failures": 0, "unexpected_successes": 0,
            "source_unchanged_during_tests": True, "launch_closure_id": closure["closure_id"],
            "require_no_skips": True, "launch_closure_error": None,
            "mandatory_inventory": inventory, "selected_inventory": inventory,
            "records": [{"test": inventory[0], "status": "passed"}],
            "source_files": sources, "source_tree_sha256": identity(sources),
            "discovery": {
                "root": str(self.root / "project/tests"), "pattern": "test_*.py", "full_suite": True,
            },
        }
        review = {
            "schema": "reelcache.independent_review.v1", "status": "passed",
            "blocking_findings": [], "independent_of_implementation": True,
            "reviewer": "fixture-independent-reviewer", "launch_closure_id": closure["closure_id"],
            "protocol_id": self.protocol["protocol_id"], "prompt_grouping_reviewed": True,
        }
        verification_path, review_path = self.root / "verified.json", self.root / "review.json"
        write_json_new(verification_path, verification)
        write_json_new(review_path, review)
        return closure, verification_path, review_path

    def test_launch_requires_independent_review_zero_skips_and_exact_sources(self):
        closure, verified, reviewed = self._receipt()
        receipt = make_launch_receipt(
            self.protocol, closure, verification_path=verified, review_path=reviewed,
        )
        verify_launch_receipt(receipt, self.protocol)
        (self.root / "project/examples/driver.py").write_text("NATIVE = True\n")
        with self.assertRaises(IntegrityError):
            verify_launch_receipt(receipt, self.protocol)
        with self.assertRaises(IntegrityError):
            verify_source_closure(closure)

    def test_new_driver_or_modified_prompt_source_invalidates_launch(self):
        closure, verified, reviewed = self._receipt()
        receipt = make_launch_receipt(
            self.protocol, closure, verification_path=verified, review_path=reviewed,
        )
        new_driver = self.root / "project/examples/new_driver.py"
        new_driver.write_text("UNVERIFIED = True\n")
        with self.assertRaises(IntegrityError):
            verify_launch_receipt(receipt, self.protocol)
        new_driver.unlink()
        self.catalog.write_text(self.catalog.read_text() + "\n")
        with self.assertRaises(IntegrityError):
            verify_launch_receipt(receipt, self.protocol)

    def test_skips_or_same_author_cannot_open_launch_gate(self):
        closure, verified, reviewed = self._receipt()
        record = read_json(verified)
        record["skipped"] = 1
        verified.write_text(json.dumps(record))
        with self.assertRaises(StateError):
            make_launch_receipt(self.protocol, closure, verification_path=verified, review_path=reviewed)
        record["skipped"] = 0
        verified.write_text(json.dumps(record))
        record = read_json(reviewed)
        record["independent_of_implementation"] = False
        reviewed.write_text(json.dumps(record))
        with self.assertRaises(StateError):
            make_launch_receipt(self.protocol, closure, verification_path=verified, review_path=reviewed)

    def test_contradictory_counts_records_sources_and_partial_inventory_rejected(self):
        closure, verified, reviewed = self._receipt()
        original = read_json(verified)
        mutations = (
            lambda data: data.update(tests_run=True),
            lambda data: data["records"][0].update(status="failed"),
            lambda data: data.update(source_files={}, source_tree_sha256="0" * 64),
            lambda data: data.update(selected_inventory=[]),
            lambda data: data["discovery"].update(pattern="test_study.py", full_suite=False),
            lambda data: data.update(mandatory_inventory=["tests.unrelated.Test.test_other"]),
        )
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                data = json.loads(json.dumps(original))
                mutate(data)
                verified.write_text(json.dumps(data))
                with self.assertRaises((ConfigurationError, IntegrityError, StateError)):
                    make_launch_receipt(
                        self.protocol, closure, verification_path=verified, review_path=reviewed,
                    )

    def test_root_and_discovery_are_part_of_reviewed_closure(self):
        closure, verified, reviewed = self._receipt()
        changed = json.loads(json.dumps(closure))
        changed["root"] = str(self.root)
        with self.assertRaises(IntegrityError):
            make_launch_receipt(self.protocol, changed, verification_path=verified, review_path=reviewed)
        changed.pop("closure_id")
        changed["closure_id"] = identity(changed)
        with self.assertRaises(IntegrityError):
            make_launch_receipt(self.protocol, changed, verification_path=verified, review_path=reviewed)

    def test_receipt_cannot_be_forged_from_nonpassing_evidence(self):
        closure, verified, reviewed = self._receipt()
        receipt = make_launch_receipt(
            self.protocol, closure, verification_path=verified, review_path=reviewed,
        )
        review = read_json(reviewed)
        review["status"] = "failed"
        reviewed.write_text(json.dumps(review))
        receipt["review"]["sha256"] = file_sha256(reviewed)
        receipt.pop("receipt_id")
        receipt["receipt_id"] = identity(receipt)
        with self.assertRaises(StateError):
            verify_launch_receipt(receipt, self.protocol)

    def test_phase_expansion_requires_all_cells_and_preserves_source_chain(self):
        closure, verified, reviewed = self._receipt()
        receipt = make_launch_receipt(
            self.protocol, closure, verification_path=verified, review_path=reviewed,
        )
        proof = self.root / "native-fixture.json"
        proof.write_text('{"fixture": true}')
        artifact = {"path": str(proof), "sha256": file_sha256(proof)}
        measurements = {
            "raw_replay": ("latent_sha256", "rgb_sha256"),
            "resident_aba": ("latent_sha256", "rgb_sha256"),
            "native_history_visibility": ("archived_prefix_sha256",),
            "hot_history_parity": ("native_kv_sha256", "latent_sha256", "rgb_sha256"),
        }
        cells = {}
        for cell_id, cell in CELLS.items():
            checks = {
                name: {
                    "status": "passed", "artifacts": [artifact],
                    "measurements": [{
                        "name": key, "observed": "a" * 64, "expected": "a" * 64,
                        "comparison": "equal",
                    } for key in keys],
                } for name, keys in measurements.items()
            }
            source_contract = {
                "layers": cell.layers,
                "K": {"shape": [cell.tokens, cell.channels], "dtype": cell.key_dtype, "key_space": "pre_rope"},
                "V": {"shape": [cell.tokens, cell.channels], "dtype": cell.value_dtype, "key_space": "none"},
            }
            for name, key, value in (
                ("native_geometry", "cell_geometry", cell.to_dict()),
                ("original_pre_rope_capture", "source_contract", source_contract),
            ):
                checks[name] = {
                    "status": "passed", "artifacts": [artifact], "measurements": [
                        {"name": key, "observed": value, "expected": value, "comparison": "equal"},
                    ],
                }
            checks["staged_memory_admission"] = {
                "status": "passed", "artifacts": [artifact], "measurements": [
                    {"name": key, "observed": cell.source_bytes, "expected": cell.source_bytes * 2,
                     "comparison": "leq"}
                    for key in ("source_staging_bytes_global", "host_peak_bytes", "device_peak_bytes_per_rank_max")
                ],
            }
            checks["oracle_svd_crosscheck"] = {
                "status": "passed", "artifacts": [artifact], "measurements": [
                    {"name": key, "observed": 1e-15, "expected": 1e-10, "comparison": "leq"}
                    for key in ("K_max_energy_error", "V_max_energy_error")
                ],
            }
            path = self.root / f"{cell_id}-native.json"
            path.write_text(json.dumps({
                "schema": "reelcache.rank_native_preflight.v1", "status": "passed",
                "protocol_id": self.protocol["protocol_id"],
                "launch_closure_id": closure["closure_id"], "phase": "native_preflight",
                "cell_id": cell_id, "fixture": False, "native_execution": True, "checks": checks,
            }))
            cells[cell_id] = {"status": "passed", "evidence": [{"path": str(path), "sha256": file_sha256(path)}]}
        evidence = {
            "schema": "reelcache.rank_phase_result.v1", "status": "passed",
            "phase": "native_preflight", "protocol_id": self.protocol["protocol_id"],
            "launch_closure_id": closure["closure_id"], "heldout_outcomes_opened": False,
            "cells": cells,
        }
        path = self.root / "preflight-fixture.json"
        path.write_text(json.dumps(evidence))
        expanded = advance_launch_receipt(
            receipt, self.protocol, evidence_path=path, next_phase="calibration_census",
        )
        verify_launch_receipt(expanded, self.protocol, phase="calibration_census")
        from reelcache.study import _native_preflight_proof
        original_proof = read_json(self.root / "causvid480-native.json")
        for check_name, changes in (
            ("raw_replay", {"observed": 0, "expected": 1, "comparison": "leq"}),
            ("raw_replay", {"observed": "a" * 64, "expected": "a" * 64, "comparison": "leq"}),
            ("oracle_svd_crosscheck", {"observed": -1, "expected": -1, "comparison": "equal"}),
            ("oracle_svd_crosscheck", {"observed": -1, "expected": 1e-10, "comparison": "leq"}),
        ):
            invalid = json.loads(json.dumps(original_proof))
            invalid["checks"][check_name]["measurements"][0].update(changes)
            with self.subTest(check=check_name, changes=changes), self.assertRaises(
                (ConfigurationError, IntegrityError, StateError)
            ):
                _native_preflight_proof(invalid, CELLS["causvid480"])
        with self.assertRaises(StateError):
            advance_launch_receipt(receipt, self.protocol, evidence_path=path, next_phase="heldout")
        del evidence["cells"]["magi720"]
        path.write_text(json.dumps(evidence))
        with self.assertRaises(StateError):
            advance_launch_receipt(
                receipt, self.protocol, evidence_path=path, next_phase="calibration_census",
            )
        with self.assertRaises(IntegrityError):
            verify_launch_receipt(expanded, self.protocol, phase="calibration_census")

    def test_failed_unrelated_native_file_cannot_substantiate_pass(self):
        closure, verified, reviewed = self._receipt()
        receipt = make_launch_receipt(
            self.protocol, closure, verification_path=verified, review_path=reviewed,
        )
        proof = self.root / "failed-native.json"
        proof.write_text(json.dumps({
            "status": "failed", "fixture": True, "cell_id": "wrong", "protocol_id": "wrong",
        }))
        evidence = {
            "schema": "reelcache.rank_phase_result.v1", "status": "passed",
            "phase": "native_preflight", "protocol_id": self.protocol["protocol_id"],
            "launch_closure_id": closure["closure_id"], "heldout_outcomes_opened": False,
            "cells": {
                cell_id: {
                    "status": "passed",
                    "evidence": [{"path": str(proof), "sha256": file_sha256(proof)}],
                } for cell_id in CELLS
            },
        }
        path = self.root / "forged-summary.json"
        path.write_text(json.dumps(evidence))
        with self.assertRaises(IntegrityError):
            advance_launch_receipt(
                receipt, self.protocol, evidence_path=path, next_phase="calibration_census",
            )

    def test_census_summary_counts_cannot_replace_matrix_records(self):
        closure, verified, reviewed = self._receipt()
        previous = make_launch_receipt(
            self.protocol, closure, verification_path=verified, review_path=reviewed,
        )
        previous["phase"] = "calibration_census"
        cells = {}
        for cell_id in CELLS:
            proof = {
                "schema": "reelcache.rank_census.v1", "status": "passed", "phase": "calibration_census",
                "protocol_id": self.protocol["protocol_id"], "launch_closure_id": closure["closure_id"],
                "cell_id": cell_id, "fixture": False, "native_execution": True, "observations": [],
            }
            path = self.root / f"{cell_id}-empty-census.json"
            path.write_text(json.dumps(proof))
            cells[cell_id] = {
                "status": "passed", "complete_prompt_groups": self.protocol["calibration_groups"],
                "matrix_observations": self.protocol["calibration_groups"] * 2 * 3 * CELLS[cell_id].layers * 2,
                "evidence": [{"path": str(path), "sha256": file_sha256(path)}],
            }
        phase = {
            "schema": "reelcache.rank_phase_result.v1", "status": "passed", "phase": "calibration_census",
            "protocol_id": self.protocol["protocol_id"], "launch_closure_id": closure["closure_id"],
            "heldout_outcomes_opened": False, "cells": cells,
        }
        path = self.root / "census-counts-only.json"
        path.write_text(json.dumps(phase))
        with self.assertRaises(IntegrityError):
            advance_launch_receipt(
                previous, self.protocol, evidence_path=path, next_phase="calibration_frontier",
                _previous_verified=True,
            )

    def test_census_checks_exact_spectrum_provenance_and_duplicates(self):
        import torch
        from reelcache.spectra import SpectrumSource, exact_spectrum
        from reelcache.study import _census_proof

        cell = dataclasses.replace(
            CELLS["causvid480"], frames=1, token_height=2, token_width=2, channels=3, layers=2,
        )
        group = self.protocol["prompts"][0]["group_id"]
        provenance = {
            "protocol_id": self.protocol["protocol_id"], "cell_id": cell.cell_id, "group_id": group,
            "seed": 0, "boundary": 0, "layer": 0, "target": "K", "split": "calibration",
            "native_execution": True, "fixture": False,
            "launch_closure_id": "a" * 64,
        }
        matrix = torch.arange(12, dtype=torch.float32).reshape(4, 3).to(torch.bfloat16)
        source = SpectrumSource(
            identity="unit-test-only", shape=(4, 3), dtype="bfloat16", target="K",
            key_space="pre_rope", provenance=provenance,
        )
        path = self.root / "small-spectrum.json"
        path.write_text(json.dumps(exact_spectrum(matrix, source=source).to_dict()))
        observation = {
            "cell_id": cell.cell_id, "group_id": group, "seed": 0, "boundary": 0,
            "layer": 0, "target": "K", "spectrum": {"path": str(path), "sha256": file_sha256(path)},
        }
        self.assertEqual(
            _census_proof({"observations": [observation], "launch_closure_id": "a" * 64}, self.protocol, cell),
            {(group, 0, 0, 0, "K")},
        )
        with self.assertRaises(IntegrityError):
            _census_proof(
                {"observations": [observation, observation], "launch_closure_id": "a" * 64},
                self.protocol, cell,
            )
        with self.assertRaises(IntegrityError):
            _census_proof(
                {"observations": [{**observation, "layer": 1}], "launch_closure_id": "a" * 64},
                self.protocol, cell,
            )
        with self.assertRaises(IntegrityError):
            _census_proof(
                {"observations": [observation], "launch_closure_id": "b" * 64}, self.protocol, cell,
            )
        for changes in ({"fixture": 0}, {"native_execution": 1}, {"seed": False}):
            bad_source = dataclasses.replace(source, provenance={**provenance, **changes})
            path.write_text(json.dumps(exact_spectrum(matrix, source=bad_source).to_dict()))
            observation["spectrum"]["sha256"] = file_sha256(path)
            with self.subTest(changes=changes), self.assertRaises(IntegrityError):
                _census_proof(
                    {"observations": [observation], "launch_closure_id": "a" * 64},
                    self.protocol, cell,
                )

    def _assets(self):
        source = self.root / "weights.bin"
        source.write_bytes(b"actual fixture weights")
        return [{
            "name": "model/weights.bin", "source": str(source),
            "bytes": source.stat().st_size, "sha256": file_sha256(source),
        }]

    def test_asset_staging_full_hash_and_no_overwrite(self):
        assets = self._assets()
        destination = self.root / "staged"
        manifest = stage_assets(assets, destination, reserve_bytes=1)
        self.assertEqual(verify_assets(destination), manifest)
        self.assertEqual(manifest["files"][0]["mode"], 0o400)
        self.assertEqual((destination / assets[0]["name"]).stat().st_mode & 0o7777, 0o400)
        with self.assertRaises(StateError):
            stage_assets(assets, destination, reserve_bytes=1)
        staged = destination / assets[0]["name"]
        staged.chmod(0o600)
        staged.write_bytes(b"x" * assets[0]["bytes"])
        with self.assertRaises(IntegrityError):
            verify_assets(destination)

    def test_staged_tools_remain_executable_and_modes_are_bound(self):
        source = self.root / "fixture-tool"
        source.write_text("#!/bin/sh\nprintf 'staged-tool-ok\\n'\n")
        source.chmod(0o755)
        destination = self.root / "tool-assets"
        manifest = stage_assets([{
            "name": "bin/fixture-tool", "source": str(source),
            "bytes": source.stat().st_size, "sha256": file_sha256(source),
        }], destination, reserve_bytes=1)
        tool = destination / "bin/fixture-tool"
        self.assertEqual(manifest["files"][0]["mode"], 0o500)
        self.assertEqual(tool.stat().st_mode & 0o7777, 0o500)
        self.assertEqual(source.stat().st_mode & 0o7777, 0o755)
        self.assertEqual(verify_assets(destination), manifest)
        result = subprocess.run([str(tool)], capture_output=True, text=True, check=True, timeout=5)
        self.assertEqual(result.stdout, "staged-tool-ok\n")
        for changed in (0o400, 0o700, 0o550):
            tool.chmod(changed)
            with self.subTest(mode=oct(changed)), self.assertRaises(IntegrityError):
                verify_assets(destination, rehash=False)
        tool.chmod(0o500)
        self.assertEqual(verify_assets(destination), manifest)

    def test_empty_native_modules_are_preserved_without_padding(self):
        source = self.root / "__init__.py"
        source.write_bytes(b"")
        empty = {
            "name": "native/package/__init__.py", "source": str(source),
            "bytes": 0, "sha256": hashlib.sha256(b"").hexdigest(),
        }
        for label, assets in (("empty", [empty]), ("mixed", [*self._assets(), empty])):
            with self.subTest(label=label):
                destination = self.root / label
                manifest = stage_assets(assets, destination, reserve_bytes=1)
                self.assertEqual(verify_assets(destination), manifest)
                self.assertEqual(verify_assets(destination, rehash=False), manifest)
                self.assertEqual(manifest["total_bytes"], sum(row["bytes"] for row in assets))
                staged = destination / empty["name"]
                self.assertEqual(staged.read_bytes(), b"")
                self.assertEqual(staged.stat().st_mode & 0o7777, 0o400)
                staged.chmod(0o600)
                staged.write_bytes(b"\n")
                staged.chmod(0o400)
                with self.assertRaises(IntegrityError):
                    verify_assets(destination, rehash=False)

    def test_empty_assets_still_require_matching_hashes(self):
        source = self.root / "__init__.py"
        source.write_bytes(b"")
        asset = {
            "name": "native/__init__.py", "source": str(source),
            "bytes": 0, "sha256": "0" * 64,
        }
        destination = self.root / "empty-hash"
        with self.assertRaises(IntegrityError):
            stage_assets([asset], destination, reserve_bytes=1)
        self.assertFalse(destination.exists())
        asset["sha256"] = file_sha256(source)
        manifest = stage_assets([asset], destination, reserve_bytes=1)
        manifest["files"][0]["sha256"] = "0" * 64
        body = {key: value for key, value in manifest.items() if key != "asset_id"}
        manifest["asset_id"] = identity(body)
        (destination / "assets.json").write_text(json.dumps(manifest))
        with self.assertRaises(IntegrityError):
            verify_assets(destination)

    def test_asset_sizes_reject_negative_or_noninteger_values(self):
        assets = self._assets()
        destination = self.root / "size-check"
        manifest = stage_assets(assets, destination, reserve_bytes=1)
        for invalid in (-1, False, 0.0, "0", None):
            with self.subTest(size=invalid):
                bad = {**assets[0], "bytes": invalid}
                with self.assertRaises(ConfigurationError):
                    stage_assets([bad], self.root / "invalid-size", reserve_bytes=1)
                self.assertFalse((self.root / "invalid-size").exists())
                manifest["files"][0]["bytes"] = invalid
                body = {key: value for key, value in manifest.items() if key != "asset_id"}
                manifest["asset_id"] = identity(body)
                (destination / "assets.json").write_text(json.dumps(manifest))
                with self.assertRaises(ConfigurationError):
                    verify_assets(destination)

    def test_failed_asset_staging_not_ready(self):
        assets = self._assets()
        assets[0]["sha256"] = "0" * 64
        with self.assertRaises(IntegrityError):
            stage_assets(assets, self.root / "failed", reserve_bytes=1)
        self.assertFalse((self.root / "failed").exists())
        assets[0]["name"] = "../escape"
        with self.assertRaises(ConfigurationError):
            stage_assets(assets, self.root / "escape", reserve_bytes=1)

    def test_device_resource_grants_do_not_overlap(self):
        with resource_grant(self.root / "locks", ["gpu0", "gpu1"]):
            with self.assertRaises(CapacityError):
                with resource_grant(self.root / "locks", ["gpu1"]):
                    self.fail("overlapping grant")
        with resource_grant(self.root / "locks", ["gpu0", "gpu1"]):
            pass

    def _request(self):
        prompt = next(p for p in self.protocol["prompts"] if p["split"] == "calibration")
        body = {
            "schema": "reelcache.rank_request.v1", "protocol_id": self.protocol["protocol_id"],
            "cell_id": "causvid480", "phase": "calibration_census",
            "prompt": prompt["text"], "prompt_id": prompt["prompt_id"],
            "group_id": prompt["group_id"], "split": "calibration", "seed": 0,
            "chunks": 3, "history_mode": "reload", "source_fps": 16,
            "arm_id": "raw",
            "output_namespace": "causvid480/cal00/seed0/raw", "representation": {"kind": "raw"},
            "budgets": {
                "device_bytes_per_rank": 20 << 30, "history_bytes_per_rank": 4 << 30,
                "workspace_bytes_per_rank": 4 << 30, "host_bytes": 4 << 30,
                "source_staging_bytes_global": 1 << 30, "disk_bytes": 20 << 30,
                "disk_reserve_bytes": 1 << 30,
            },
        }
        return {**body, "request_id": identity(body)}

    def test_request_rejects_leakage_future_horizon_and_budget(self):
        request = self._request()
        validate_request(request, self.protocol, phase="calibration_census")
        with self.assertRaises(IntegrityError):
            validate_request(request, self.protocol, phase="heldout")
        for field, value in (("chunks", 4), ("prompt", "changed conditioning"), ("source_fps", False)):
            changed = {**request, field: value}
            changed.pop("request_id")
            changed["request_id"] = identity(changed)
            with self.assertRaises((IntegrityError, ConfigurationError)):
                validate_request(changed, self.protocol, phase="calibration_census")
        changed = json.loads(json.dumps(request))
        changed["budgets"]["source_staging_bytes_global"] = 1
        changed.pop("request_id")
        changed["request_id"] = identity(changed)
        with self.assertRaises(CapacityError):
            validate_request(changed, self.protocol, phase="calibration_census")

    def test_requests_require_the_native_cell_frame_rate(self):
        self.assertEqual(dict(NATIVE_SOURCE_FPS), {
            "causvid480": 16, "magi720": 24, "magi1080": 24,
        })
        with self.assertRaises(TypeError):
            NATIVE_SOURCE_FPS["causvid480"] = 24
        for cell_id, cell in CELLS.items():
            body = self._request()
            body.pop("request_id")
            body["cell_id"] = cell_id
            body["budgets"].update(
                source_staging_bytes_global=cell.source_bytes,
                host_bytes=max(cell.source_bytes, 4 << 30),
            )
            expected = NATIVE_SOURCE_FPS[cell_id]
            for fps in (expected, float(expected)):
                body["source_fps"] = fps
                validate_request(
                    {**body, "request_id": identity(body)}, self.protocol,
                    phase="calibration_census",
                )
            for fps in (False, 0, -1, 16 if expected == 24 else 24, expected + 0.5):
                body["source_fps"] = fps
                with self.subTest(cell=cell_id, fps=fps), self.assertRaises(ConfigurationError):
                    validate_request(
                        {**body, "request_id": identity(body)}, self.protocol,
                        phase="calibration_census",
                    )
        validate_protocol(self.protocol)

    def test_adaptive_codec_must_match_the_declared_representation(self):
        body = self._request()
        body.pop("request_id")
        body["phase"] = "calibration_frontier"
        codec = {
            "kind": "svd_hevc", "normalization": "per_channel", "pix_fmt": "yuv444p",
            "qp": 8, "preset": "ultrafast",
        }
        representation = {
            **codec, "rank_k": 256, "rank_v": 384, "rank_policy": {"codec": dict(codec)},
        }

        def check(value):
            request = {**body, "representation": value}
            return validate_request(
                {**request, "request_id": identity(request)}, self.protocol,
                phase="calibration_frontier",
            )

        check(representation)
        for name, value in (
            ("kind", "svd"), ("normalization", "global"), ("pix_fmt", "yuv420p"),
            ("qp", 12), ("preset", "medium"),
        ):
            changed = json.loads(json.dumps(representation))
            changed["rank_policy"]["codec"][name] = value
            with self.subTest(field=name), self.assertRaises(ConfigurationError):
                check(changed)
        for policy in ({}, {"codec": None}, "mutable-policy.json"):
            with self.subTest(policy=policy), self.assertRaises(ConfigurationError):
                check({**representation, "rank_policy": policy})
        check({
            "kind": "svd", "rank_k": 256, "rank_v": 384,
            "rank_policy": {"codec": {"kind": "svd"}},
        })
        check({"kind": "svd", "rank_policy": {"codec": {"kind": "svd"}}})
        with self.assertRaises(ConfigurationError):
            check({"kind": "svd", "rank_k": 256, "rank_policy": {"codec": {"kind": "svd"}}})
        with self.assertRaises(ConfigurationError):
            check({
                "kind": "svd", "rank_k": 256, "rank_v": 384,
                "rank_policy": {"codec": codec},
            })

    def test_calibration_census_rejects_factor_histories(self):
        body = self._request()
        body.pop("request_id")
        for representation in (
            {"kind": "svd", "rank_k": 256, "rank_v": 384},
            {"kind": "svd_hevc", "rank_k": 256, "rank_v": 384,
             "normalization": "per_channel", "pix_fmt": "yuv444p", "qp": 8, "preset": "ultrafast"},
        ):
            body["representation"] = representation
            with self.subTest(kind=representation["kind"]), self.assertRaises(StateError):
                validate_request(
                    {**body, "request_id": identity(body)}, self.protocol,
                    phase="calibration_census",
                )

    def test_static_layer_ranks_are_complete_and_distinct_from_adaptive_ranks(self):
        for cell_id, cell in CELLS.items():
            body = self._request()
            body.pop("request_id")
            body.update(
                cell_id=cell_id, source_fps=NATIVE_SOURCE_FPS[cell_id],
                phase="calibration_frontier",
            )
            body["budgets"].update(
                source_staging_bytes_global=cell.source_bytes,
                host_bytes=max(cell.source_bytes, 4 << 30),
            )
            representation = {
                "kind": "svd",
                "layer_ranks": {"K": [cell.channels] * cell.layers,
                                "V": list(range(1, cell.layers + 1))},
            }

            def check(value):
                request = {**body, "representation": value}
                return validate_request(
                    {**request, "request_id": identity(request)}, self.protocol,
                    phase="calibration_frontier",
                )

            self.assertEqual(check(representation)["representation"], representation)
            check({
                **representation, "kind": "svd_hevc", "normalization": "per_channel",
                "pix_fmt": "yuv444p", "qp": 8, "preset": "ultrafast",
            })
            invalid = [
                {**representation, "rank_policy": {"codec": {"kind": "svd"}}},
                {**representation, "rank_k": 1},
                {**representation, "kind": "raw"},
                {"kind": "svd", "layer_ranks": {"K": [1] * cell.layers}},
                {"kind": "svd", "layer_ranks": {"K": [1], "V": [1] * cell.layers}},
                {"kind": "svd", "layer_ranks": {"K": "1", "V": [1] * cell.layers}},
            ]
            for rank in (True, 0, -1, cell.channels + 1, 1.5, "1"):
                wrong = json.loads(json.dumps(representation))
                wrong["layer_ranks"]["K"][0] = rank
                invalid.append(wrong)
            for wrong in invalid:
                with self.subTest(cell=cell_id, representation=wrong), self.assertRaises(ConfigurationError):
                    check(wrong)

    def test_request_batch_binds_epoch_and_unique_namespaces(self):
        closure, verified, reviewed = self._receipt()
        receipt = make_launch_receipt(
            self.protocol, closure, verification_path=verified, review_path=reviewed,
        )
        request = self._request()
        request.update({
            "phase": "native_preflight", "split": "development",
            "prompt": "A forgotten clock tower", "prompt_id": "dev00", "group_id": "dev-group",
        })
        request.pop("request_id")
        request["request_id"] = identity(request)
        body = {
            "schema": "reelcache.rank_requests.v1", "protocol_id": self.protocol["protocol_id"],
            "receipt_id": receipt["receipt_id"], "phase": "native_preflight", "requests": [request],
        }
        batch = {**body, "requests_id": identity(body)}
        self.assertEqual(
            len(validate_request_batch(
                batch, self.protocol, receipt, cell_id="causvid480", phase="native_preflight",
            )), 1,
        )
        with self.assertRaises(ConfigurationError):
            validate_request_batch(
                batch, self.protocol, receipt, cell_id="magi720", phase="native_preflight",
            )
        body["requests"] = [request, request]
        batch = {**body, "requests_id": identity(body)}
        with self.assertRaises(IntegrityError):
            validate_request_batch(
                batch, self.protocol, receipt, cell_id="causvid480", phase="native_preflight",
            )

    def test_heldout_requests_must_equal_frozen_arms(self):
        from reelcache.study import arm_contract

        request = self._request()
        prompt = next(item for item in self.protocol["prompts"] if item["split"] == "heldout")
        request.update({
            "phase": "heldout", "split": "heldout", "prompt": prompt["text"],
            "prompt_id": prompt["prompt_id"], "group_id": prompt["group_id"],
        })
        request.pop("request_id")
        request["request_id"] = identity(request)
        receipt = {"receipt_id": "b" * 64, "frozen_arms": [arm_contract(request)]}

        def batch_for(item):
            body = {
                "schema": "reelcache.rank_requests.v1", "protocol_id": self.protocol["protocol_id"],
                "receipt_id": receipt["receipt_id"], "phase": "heldout", "requests": [item],
            }
            return {**body, "requests_id": identity(body)}

        with mock.patch("reelcache.study.verify_launch_receipt"):
            validate_request_batch(
                batch_for(request), self.protocol, receipt, cell_id="causvid480", phase="heldout",
            )
            for change in (
                {"arm_id": "unfrozen"},
                {"history_mode": "quality_hot"},
                {"representation": {"kind": "svd", "rank_k": 1, "rank_v": 1}},
                {"chunks": 2},
            ):
                changed = {**request, **change}
                changed.pop("request_id")
                changed["request_id"] = identity(changed)
                with self.subTest(change=change), self.assertRaises(StateError):
                    validate_request_batch(
                        batch_for(changed), self.protocol, receipt, cell_id="causvid480", phase="heldout",
                    )

    def test_controller_rejects_multi_cell_and_namespaces_configurations(self):
        import argparse
        from scripts.rank_study import _requests

        with self.assertRaises(ConfigurationError):
            _requests(argparse.Namespace(cell=["causvid480", "magi720"]))
        paths = {}
        for name, document in (
            ("protocol", self.protocol), ("receipt", {"receipt_id": "c" * 64}),
            ("budgets", {"causvid480": self._request()["budgets"]}),
            ("representation", {"kind": "raw"}), ("fps", {"causvid480": 16}),
        ):
            path = self.root / f"{name}.json"
            path.write_text(json.dumps(document))
            paths[name] = path
        outputs = []
        with (
            mock.patch("scripts.rank_study.verify_launch_receipt"),
            mock.patch("reelcache.study.verify_launch_receipt"),
        ):
            for index, (phase, representation) in enumerate((
                ("calibration_census", {"kind": "raw"}),
                ("calibration_frontier", {"kind": "raw"}),
                ("calibration_frontier", {"kind": "svd", "rank_k": 256, "rank_v": 384}),
                ("calibration_frontier", {"kind": "svd", "rank_k": 256, "rank_v": 384}),
            )):
                if index == 3:
                    budgets = read_json(paths["budgets"])
                    budgets["causvid480"]["history_bytes_per_rank"] += 2 << 30
                    paths["budgets"].write_text(json.dumps(budgets))
                paths["representation"].write_text(json.dumps(representation))
                output = self.root / f"{phase}-{index}-batch.json"
                args = argparse.Namespace(
                    cell=["causvid480"], protocol=paths["protocol"], receipt=paths["receipt"],
                    budgets=paths["budgets"], representation=paths["representation"],
                    source_fps=paths["fps"], phase=phase, development_prompts=None,
                    first_seed_only=False, chunks=3, history_mode="reload", arm="raw",
                    output=output,
                )
                _requests(args)
                outputs.append(read_json(output))
        namespaces = [{item["output_namespace"] for item in doc["requests"]} for doc in outputs]
        self.assertFalse(namespaces[0] & namespaces[1])
        self.assertFalse(namespaces[1] & namespaces[2])
        self.assertFalse(namespaces[2] & namespaces[3])
        case_ids = [{item["case_id"] for item in doc["requests"]} for doc in outputs]
        self.assertEqual(len(case_ids[0]), len(outputs[0]["requests"]))
        self.assertTrue(all(ids == case_ids[0] for ids in case_ids[1:]))
        trajectories = [{item["trajectory_id"] for item in doc["requests"]} for doc in outputs]
        self.assertFalse(trajectories[0] & trajectories[1])
        self.assertFalse(trajectories[1] & trajectories[2])
        self.assertFalse(trajectories[2] & trajectories[3])


if __name__ == "__main__":
    unittest.main()
