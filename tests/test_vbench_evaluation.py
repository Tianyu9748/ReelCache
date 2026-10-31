import copy
from contextlib import nullcontext
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import shutil
import subprocess
import unittest
from unittest import mock
import uuid

import numpy as np

from scripts import _vbench_official as official
from scripts import _vbench_protocol as protocol
from scripts import evaluate_vbench as runner


def fixture_protocol():
    rows = [
        {"prompt_en": "a bird building a nest from twigs and leaves",
         "dimension": ["subject_consistency", "dynamic_degree", "motion_smoothness"]},
        {"prompt_en": "castle", "dimension": ["scene", "background_consistency"],
         "auxiliary_info": {"scene": {"scene": {"scene": "castle"}}}},
        {"prompt_en": "A tranquil tableau of bathroom", "dimension": ["temporal_flickering"]},
        {"prompt_en": "A modern art museum, with colorful paintings",
         "dimension": ["overall_consistency", "aesthetic_quality", "imaging_quality"]},
    ]
    with mock.patch.object(protocol, "source_rows", return_value=rows):
        spec = protocol.make_candidate("unused")
    spec["models"] = {"model_a": {"height": 16, "width": 24},
                      "model_b": {"height": 16, "width": 24}}
    spec.update(status="frozen", frozen_by="unit-test-only", frozen_utc="fixture")
    return spec, rows


class WorkCase(unittest.TestCase):
    def setUp(self):
        self.work = Path(".test-work") / f"vbench-{uuid.uuid4().hex}"
        self.work.mkdir(parents=True)
        self.addCleanup(shutil.rmtree, self.work)

    def write_json(self, name, value):
        path = self.work / name
        protocol.write_json(path, value)
        return path

    def clip(self, arm="raw", prompt_id="vb0000", model="model_a"):
        cid = f"{model}-{prompt_id}-{arm}"
        spec, _ = fixture_protocol()
        return {
            "id": cid, "model": model, "prompt_id": prompt_id, "arm": arm, "seed": 0,
            "fps": 24, "native_fps": 24, "prefix_frames": 9, "native_frame_start": 9,
            "frame_count": 96, "rgb_frame_start": 0, "rgb_path": f"{cid}.npy",
            "rgb_file_sha256": "a" * 64, "generation_binding_sha256": "b" * 64,
            "representation": ({"kind": "raw"} if arm == "raw" else
                               {"kind": "svd_hevc", **spec["primary_compression"]} if arm == "svd_hevc"
                               else {"kind": arm}),
            "cohort_role": "primary" if arm in ("raw", "svd_hevc") else "diagnostic",
            "representation_evidence_sha256": "e" * 64,
            "paired_raw_id": f"{model}-{prompt_id}-raw",
        }

    def inputs(self, *clips):
        return {"schema": protocol.SCHEMAS["inputs"], "protocol_sha256": "c" * 64,
                "clips": list(clips)}

    def save_rgb(self, clip, frames):
        path = self.work / clip["rgb_path"]
        np.save(path, frames, allow_pickle=False)
        clip["rgb_file_sha256"] = protocol.sha256_file(path)
        return path


class ProtocolTests(WorkCase):
    def test_candidate_is_deterministic_unfrozen_and_preserves_rng(self):
        _, rows = fixture_protocol()
        before = random.getstate()
        with mock.patch.object(protocol, "source_rows", return_value=rows):
            first = protocol.make_candidate("unused")
            second = protocol.make_candidate("unused")
        self.assertEqual(first, second)
        self.assertEqual(before, random.getstate())
        self.assertEqual(first["status"], "candidate")
        self.assertFalse(first["full_official_aggregate"])

    def test_prior_membership_is_disclosed_not_filtered(self):
        _, rows = fixture_protocol()
        prior = self.write_json("prior.json", {"prompts": [{"text": rows[0]["prompt_en"]}]})
        with mock.patch.object(protocol, "source_rows", return_value=rows):
            candidate = protocol.make_candidate("unused", [prior])
        self.assertEqual(candidate["prompts"][0]["text"], rows[0]["prompt_en"])
        self.assertEqual(len(candidate["prompts"][0]["prior_membership"]), 1)
        self.assertFalse(candidate["selection"]["prior_membership_used_for_selection"])

    def test_protocol_requires_unchanged_text_and_annotations(self):
        spec, rows = fixture_protocol()
        protocol.validate_protocol(spec, rows, frozen=True)
        for field, value in (("text", "rewritten"),
                             ("official_annotation", {"prompt_en": "castle", "dimension": ["human_action"]})):
            changed = copy.deepcopy(spec)
            changed["prompts"][0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                protocol.validate_protocol(changed, rows)

    def test_candidate_never_passes_execution_freeze_gate(self):
        spec, rows = fixture_protocol()
        spec["status"] = "candidate"
        with self.assertRaisesRegex(ValueError, "frozen"):
            protocol.validate_protocol(spec, rows, frozen=True)

    def test_wrong_source_aggregate_and_ssim_settings_rejected(self):
        spec, rows = fixture_protocol()
        for key, value in (("full_official_aggregate", True), ("ssim", {"data_range": 1})):
            changed = copy.deepcopy(spec)
            changed[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                protocol.validate_protocol(changed, rows)
        changed = copy.deepcopy(spec)
        changed["source"]["commit"] = "unversioned"
        with self.assertRaises(ValueError):
            protocol.validate_protocol(changed, rows)

    def test_five_second_clips_require_vbench_long_not_retiming(self):
        spec, rows = fixture_protocol()
        spec["continuation_frames"] = 120
        with self.assertRaisesRegex(ValueError, "VBench-Long"):
            protocol.validate_protocol(spec, rows)

    def test_duplicate_and_nonfinite_json_rejected(self):
        path = self.work / "invalid.json"
        for value in ('{"x":1,"x":2}', '{"x":NaN}', '{"x":Infinity}'):
            path.write_text(value)
            with self.subTest(value=value), self.assertRaises(ValueError):
                protocol.read_json(path)

    def test_source_hash_is_verified(self):
        path = self.write_json("unofficial.json", [])
        with self.assertRaisesRegex(ValueError, "SHA256"):
            protocol.source_rows(path)

    def test_pairing_requires_same_native_window_and_generation(self):
        spec, _ = fixture_protocol()
        raw, lossy = self.clip(), self.clip("svd_hevc")
        protocol.validate_inputs(self.inputs(raw, lossy), spec, "c" * 64)
        mutations = {
            "seed": 1, "fps": 16, "native_fps": 16, "native_frame_start": 0,
            "frame_count": 95, "generation_binding_sha256": "d" * 64,
            "model": "model_b", "paired_raw_id": "missing",
        }
        for field, value in mutations.items():
            altered = copy.deepcopy(lossy)
            altered[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                protocol.validate_inputs(self.inputs(raw, altered), spec, "c" * 64)

    def test_unknown_fps_needs_explicit_note(self):
        spec, _ = fixture_protocol()
        raw = self.clip()
        raw["native_fps"] = None
        with self.assertRaisesRegex(ValueError, "timing note"):
            protocol.validate_inputs(self.inputs(raw), spec, "c" * 64)
        raw["timing_note"] = "Native generation has no timing metadata; 24fps is declared metric playback."
        protocol.validate_inputs(self.inputs(raw), spec, "c" * 64)

    def test_primary_rank_codec_and_reuse_declarations_are_bound(self):
        spec, _ = fixture_protocol()
        raw, lossy = self.clip(), self.clip("svd_hevc")
        for field, value in (("K_rank", 128), ("qp", 10), ("pix_fmt", "yuv420p"),
                             ("approximate_reuse", True), ("approximate_reuse", 0)):
            changed = copy.deepcopy(lossy)
            changed["representation"][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "representation"):
                protocol.validate_inputs(self.inputs(raw, changed), spec, "c" * 64)
        diagnostic = self.clip("svd")
        del diagnostic["cohort_role"]
        with self.assertRaisesRegex(ValueError, "Diagnostic"):
            protocol.validate_inputs(self.inputs(raw, diagnostic), spec, "c" * 64)

    def test_duplicate_boolean_and_unbound_inputs_rejected(self):
        spec, _ = fixture_protocol()
        raw = self.clip()
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            protocol.validate_inputs(self.inputs(raw, raw), spec, "c" * 64)
        for field, value in (("prefix_frames", True), ("rgb_frame_start", -1),
                             ("rgb_file_sha256", "invalid"), ("id", "../outside")):
            altered = copy.deepcopy(raw)
            altered[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                protocol.validate_inputs(self.inputs(altered), spec, "c" * 64)
        with self.assertRaisesRegex(ValueError, "binding"):
            protocol.validate_inputs(self.inputs(raw), spec, "d" * 64)

    def test_generic_and_annotation_coverage_are_distinct(self):
        spec, _ = fixture_protocol()
        clips = [self.clip(prompt_id=p["id"]) for p in spec["prompts"]]
        paths = {r["id"]: self.work / f"{r['id']}.mp4" for r in clips}
        self.assertEqual(len(protocol.dimension_records(spec, clips, "imaging_quality", paths)), 4)
        self.assertEqual(len(protocol.dimension_records(spec, clips, "imaging_quality", paths, "annotated")), 1)
        self.assertEqual(len(protocol.dimension_records(spec, clips, "temporal_flickering", paths)), 1)
        self.assertEqual(protocol.dimension_records(spec, clips, "color", paths), [])
        original = copy.deepcopy(spec)
        scene = protocol.dimension_records(spec, clips, "scene", paths)
        self.assertEqual(scene[0]["auxiliary_info"]["scene"], {"scene": {"scene": "castle"}})
        self.assertEqual(spec, original)

    def test_missing_semantic_annotations_are_not_invented(self):
        spec, _ = fixture_protocol()
        del spec["prompts"][1]["official_annotation"]["auxiliary_info"]
        clip = self.clip(prompt_id=spec["prompts"][1]["id"])
        with self.assertRaisesRegex(ValueError, "auxiliary"):
            protocol.dimension_records(spec, [clip], "scene", {clip["id"]: "video.mp4"})

    def test_exact_96_frame_policies(self):
        policies = protocol.frame_policy(96, 24, protocol.DIMENSIONS)
        self.assertEqual(policies["dynamic_degree"]["indices"], list(range(0, 96, 3)))
        self.assertEqual(policies["dynamic_degree"]["motion_pair_count_threshold"], 8)
        self.assertEqual(policies["motion_smoothness"]["target_indices"], list(range(1, 94, 2)))
        self.assertEqual(policies["motion_smoothness"]["unused_terminal_indices"], [95])
        self.assertEqual(policies["overall_consistency"]["indices"], list(range(5, 96, 12)))
        self.assertEqual(len(policies["human_action"]["indices"]), 16)

    def test_short_unknown_and_duplicate_dimension_policies_fail(self):
        for count, fps, dims in (
            (1, 24, ["subject_consistency"]), (3, 240, ["dynamic_degree"]),
            (15, 24, ["human_action"]), (7, 24, ["overall_consistency"]),
            (96, 0, ["dynamic_degree"]), (96, 24, ["CLIP_proxy"]),
            (96, 24, ["dynamic_degree", "dynamic_degree"]),
        ):
            with self.subTest(count=count, fps=fps, dims=dims), self.assertRaises(ValueError):
                protocol.frame_policy(count, fps, dims)

    def test_official_result_units_and_complete_coverage(self):
        path = (self.work / "clip.mp4").resolve()
        normalized = protocol.validate_official_results(
            "imaging_quality", (0.7, [{"video_path": str(path), "video_results": 70}]), [path])
        self.assertEqual(normalized[str(path)]["dimension_score"], 0.7)
        for result in (
            (0.7, []),
            (0.7, [{"video_path": str(path), "video_results": float("nan")}]),
            (0.7, [{"video_path": str(path), "video_results": 0.7}] * 2),
            (float("nan"), [{"video_path": str(path), "video_results": 0.7}]),
        ):
            with self.subTest(result=result), self.assertRaises(ValueError):
                protocol.validate_official_results("imaging_quality", result, [path])
        with self.assertRaisesRegex(ValueError, "binary"):
            protocol.validate_official_results(
                "dynamic_degree", (0.1, [{"video_path": str(path), "video_results": 0.1}]), [path])

    def test_only_complete_seven_dimensional_quality_formula(self):
        values = {name: bounds[1] for name, bounds in protocol.QUALITY_BOUNDS.items()}
        result = protocol.quality_aggregate(values)
        self.assertAlmostEqual(result["score"], 1)
        self.assertFalse(result["official_full_aggregate"])
        del values["temporal_flickering"]
        with self.assertRaisesRegex(ValueError, "Seven"):
            protocol.quality_aggregate(values)

    def test_actual_byte_denominators_and_basis_charge(self):
        storage = {
            "denominator": "native_dense_kv_bytes", "native_dense_kv_bytes": 1000,
            "native_dtype": "bfloat16", "unique_native_head_owners": True,
            "unique_decoder_bytes": 250, "snapshot_closure_bytes": 500, "scope_id": "head-123",
            "includes": ["payloads", "whole_immutable_bases", "required_manifests"],
            "physical_index_bytes": 100, "physical_filesystem_bytes": 4096,
            "evidence_sha256": "a" * 64,
        }
        ratios = protocol.compression_ratios(storage)
        self.assertEqual(ratios["native_over_decoder_ratio"], 4)
        self.assertEqual(ratios["native_over_snapshot_ratio"], 2)
        self.assertEqual(ratios["physical_index_bytes"], 100)
        self.assertEqual(ratios["physical_filesystem_bytes"], 4096)
        for key, value in (("denominator", "fp32_factor_inputs"), ("unique_decoder_bytes", 0),
                           ("includes", ["payloads"]), ("snapshot_closure_bytes", 100),
                           ("native_dtype", "float32"), ("unique_native_head_owners", False),
                           ("unique_native_head_owners", 1), ("physical_index_bytes", -1),
                           ("physical_filesystem_bytes", True)):
            changed = dict(storage, **{key: value})
            with self.subTest(key=key), self.assertRaises(ValueError):
                protocol.compression_ratios(changed)


class MediaAndReportingTests(WorkCase):
    def test_npy_window_exact_bits_and_hash(self):
        clip = self.clip()
        frames = np.arange(105 * 16 * 24 * 3, dtype=np.uint8).reshape(105, 16, 24, 3)
        self.save_rgb(clip, frames)
        clip["rgb_frame_start"] = 9
        with runner.rgb_window(clip, self.work, {"height": 16, "width": 24}) as (window, info):
            self.assertTrue(np.array_equal(window, frames[9:]))
            self.assertEqual(info["source_indices"], list(range(9, 105)))
            self.assertEqual(info["rgb_sha256"], hashlib.sha256(frames[9:].tobytes()).hexdigest())

    def test_npz_window_and_invalid_sources(self):
        clip = self.clip()
        clip.update(rgb_path="frames.npz", array_key="frames")
        path = self.work / "frames.npz"
        np.savez_compressed(path, frames=np.zeros((96, 16, 24, 3), dtype=np.uint8))
        clip["rgb_file_sha256"] = protocol.sha256_file(path)
        with runner.rgb_window(clip, self.work, {"height": 16, "width": 24}) as (window, _):
            self.assertEqual(window.shape, (96, 16, 24, 3))
        for altered, geometry in (
            (dict(clip, rgb_file_sha256="0" * 64), {"height": 16, "width": 24}),
            (dict(clip, rgb_frame_start=1), {"height": 16, "width": 24}),
            (clip, {"height": 24, "width": 16}),
        ):
            with self.subTest(altered=altered), self.assertRaises(ValueError):
                with runner.rgb_window(altered, self.work, geometry):
                    pass

    def test_rgb_rejects_float_and_channel_first(self):
        for shape, dtype in (((96, 16, 24, 3), np.float32), ((96, 3, 16, 24), np.uint8)):
            clip = self.clip()
            self.save_rgb(clip, np.zeros(shape, dtype=dtype))
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                with runner.rgb_window(clip, self.work, {"height": 16, "width": 24}):
                    pass

    def test_ssim_matches_direct_skimage(self):
        from skimage.metrics import structural_similarity

        rng = np.random.default_rng(7)
        raw = rng.integers(0, 256, size=(4, 16, 24, 3), dtype=np.uint8)
        candidate = raw.copy()
        candidate[:, 3:5, 4:6] = 0
        result = runner.ssim_frames(raw, candidate, workers=2)
        direct = [float(structural_similarity(x, y, **protocol.SSIM_PARAMETERS))
                  for x, y in zip(raw, candidate)]
        self.assertEqual(result["per_frame"], direct)
        self.assertEqual(runner.ssim_frames(raw, raw)["mean"], 1)
        with self.assertRaises(ValueError):
            runner.ssim_frames(raw, candidate[:1])

    def test_paired_static_filter_exclusions_are_visible(self):
        clips = [self.clip(arm, model=model) for model in ("model_a", "model_b")
                 for arm in ("raw", "svd_hevc")]
        paths = {r["id"]: Path(f"{r['id']}.mp4") for r in clips}
        passes = {str(path): True for path in paths.values()}
        passes[str(paths[clips[1]["id"]])] = False
        eligible, details = runner.paired_static_eligibility(clips, paths, passes)
        self.assertEqual(eligible, {str(paths[r["id"]]) for r in clips if r["model"] == "model_b"})
        self.assertFalse(details[0]["included_for_every_paired_arm"])
        self.assertTrue(details[0]["exclusion_is_not_a_success_or_zero_score"])

    def test_unmatched_diagnostics_and_models_are_never_pooled(self):
        clips = [self.clip(prompt_id=pid) for pid in ("vb0000", "vb0001")]
        clips += [self.clip("svd", prompt_id="vb0000")]
        clips += [self.clip("raw", model="model_b"), self.clip("svd_hevc", model="model_b")]
        for index, row in enumerate(clips):
            row["ssim"] = {"mean": 1 - index / 10, "minimum_frame": 0.5}
        ssim = runner.summarize_ssim(clips)
        self.assertEqual(len(ssim), 2)
        self.assertTrue(all(row["case_count"] == 1 for row in ssim))
        values = {r["id"]: {"dimension_score": index % 2} for index, r in enumerate(clips)}
        summary = runner.summarize_vbench(clips, ["dynamic_degree"],
                                          {"dynamic_degree": {"by_clip": values}}, quality_summary=True)
        self.assertEqual(len(summary), 4)
        svd = next(r for r in summary if r["arm"] == "svd")
        self.assertEqual(svd["dimensions"]["dynamic_degree"]["case_count"], 1)
        self.assertEqual(svd["dimensions"]["dynamic_degree"]["matched_candidate_minus_raw"]["case_count"], 1)
        self.assertEqual(svd["quality_formula_pilot_only"]["status"], "unavailable")

    def test_model_and_gpu_permission_gates_precede_any_io(self):
        with mock.patch.object(protocol, "read_json", side_effect=AssertionError("unexpected read")):
            with self.assertRaisesRegex(ValueError, "authorization"):
                runner.evaluate("absent", "absent", "absent", ["dynamic_degree"])
            with self.assertRaisesRegex(ValueError, "grant"):
                runner.evaluate("absent", "absent", "absent", ["dynamic_degree"],
                                device="cuda:0", model_execution_authorized=True)

    def test_cpu_prepare_refuses_unfrozen_protocol(self):
        with mock.patch.object(protocol, "load_protocol", side_effect=ValueError("unfrozen")):
            with self.assertRaisesRegex(ValueError, "unfrozen"):
                runner.prepare("protocol", "inputs", self.work / "out")
        self.assertFalse((self.work / "out").exists())


class OfficialToolingTests(WorkCase):
    def test_metric_settings_preserve_rng_and_thread_state_without_cuda(self):
        import torch

        python_state, numpy_state = random.getstate(), np.random.get_state()
        torch_state, threads = torch.get_rng_state().clone(), torch.get_num_threads()
        with runner.model_settings("cpu") as device:
            self.assertEqual(device, "cpu")
            random.random()
            np.random.random()
            torch.rand(3)
        self.assertEqual(random.getstate(), python_state)
        self.assertTrue(np.array_equal(np.random.get_state()[1], numpy_state[1]))
        self.assertTrue(torch.equal(torch.get_rng_state(), torch_state))
        self.assertEqual(torch.get_num_threads(), threads)

    def test_human_action_uses_official_filename_label_without_copying_pixels(self):
        video = self.work / "opaque-id.mp4"
        video.write_bytes(b"bound media")
        records = [{"prompt_en": "A person is writing", "dimension": ["human_action"],
                    "video_list": [str(video.resolve())]}]
        aliases = runner.human_action_aliases(records, self.work)
        path = Path(aliases[0]["video_list"][0])
        self.assertTrue(path.is_symlink())
        self.assertEqual(path.resolve(), video.resolve())
        official_label = path.name.lower().split("-")[0].split("person is ")[-1].split("_")[0]
        self.assertEqual(official_label, "writing")
        self.assertEqual(records[0]["video_list"], [str(video.resolve())])

    def evaluate_fixture(self, compute):
        spec, _ = fixture_protocol()
        raw, candidate = self.clip(), self.clip("svd_hevc")
        inputs = self.inputs(raw, candidate)
        inputs_path = self.write_json("inputs.json", inputs)
        spec_path = self.write_json("protocol.json", spec)
        prepared_clips = []
        for clip in inputs["clips"]:
            video = self.work / f"{clip['id']}.mp4"
            video.write_bytes(b"mocked-media-boundary")
            prepared_clips.append({
                **clip, "video_path": video.name, "video_sha256": protocol.sha256_file(video),
                "decode_verification": {"exact_rgb_identity": True},
            })
        prepared = {
            "schema": protocol.SCHEMAS["prepared"], "status": "complete",
            "protocol_path": str(spec_path.resolve()), "protocol_sha256": "c" * 64,
            "input_manifest_path": str(inputs_path.resolve()),
            "input_manifest_sha256": protocol.sha256_file(inputs_path),
            "input_manifest": inputs, "clips": prepared_clips,
        }
        prepared_path = self.write_json("prepared.json", prepared)
        assets_path = self.write_json("assets.json", {
            "schema": protocol.SCHEMAS["assets"], "official_source": {"root": "mock"},
            "dependency_paths": [], "assets": {},
        })
        with mock.patch.object(protocol, "load_protocol", return_value=(spec, "c" * 64)), \
                mock.patch.object(official, "preflight_assets", return_value={}), \
                mock.patch.object(official, "verify_required_assets", return_value={}), \
                mock.patch.object(official, "official_environment", return_value=nullcontext()), \
                mock.patch.object(official, "compute_dimension", side_effect=compute):
            return runner.evaluate(prepared_path, assets_path, self.work / "result",
                                   ["subject_consistency", "color"], model_execution_authorized=True)

    def test_official_dispatch_and_uncovered_dimensions_are_reported(self):
        def compute(dimension, info_path, manifest, device):
            self.assertEqual(dimension, "subject_consistency")
            self.assertEqual(device, "cpu")
            rows = protocol.read_json(info_path)
            self.assertEqual(len(rows), 2)
            return 0.8, [{"video_path": r["video_list"][0], "video_results": 0.8} for r in rows]

        result = self.evaluate_fixture(compute)
        self.assertEqual(result["status"], "complete")
        self.assertFalse(result["full_official_aggregate"])
        self.assertEqual(result["dimensions"]["color"]["status"], "uncovered_or_static_excluded")
        self.assertEqual(len(result["planned_primary_coverage"][0]["missing_prompt_ids"]), 3)
        self.assertEqual(len(result["summary"]), 2)

    def test_official_silent_omission_is_a_persisted_failure(self):
        with self.assertRaisesRegex(ValueError, "omitted"):
            self.evaluate_fixture(lambda *args: (0.8, []))
        result = protocol.read_json(self.work / "result/result.json")
        self.assertEqual(result["status"], "failed")
        self.assertIn("omitted", result["error"])

    def test_missing_weights_and_imports_are_not_silent(self):
        manifest = {
            "schema": protocol.SCHEMAS["assets"],
            "assets": {name: {"path": str(self.work / name)} for name in official.ASSETS},
            "dino_source": {"sha256": {}},
        }
        with mock.patch.object(importlib.util, "find_spec", return_value=None):
            checks = official.preflight_assets(manifest, ["subject_consistency", "imaging_quality"])
        self.assertTrue(all(row["status"] == "blocked" for row in checks.values()))
        self.assertIn("missing asset: dino", checks["subject_consistency"]["blockers"])

    def test_asset_hash_mismatch_stops_before_model_loading(self):
        path = self.work / "weight.bin"
        path.write_bytes(b"not the expected checkpoint")
        manifest = {
            "official_source": {"root": "fixture"},
            "assets": {"raft": {"path": str(path), "expected_sha256": "0" * 64}},
        }
        with mock.patch.object(official, "verify_checkout", return_value=manifest["official_source"]):
            with self.assertRaisesRegex(ValueError, "SHA256"):
                official.verify_required_assets(manifest, ["dynamic_degree"])

    def test_download_guard_blocks_subprocesses_without_launching(self):
        with mock.patch.object(subprocess, "Popen") as popen:
            with official.no_network_downloads():
                for command in (["wget", "https://example.invalid/weight"],
                                ["curl", "https://example.invalid/weight"],
                                ["git", "clone", "https://example.invalid/repo"]):
                    with self.subTest(command=command), self.assertRaises(RuntimeError):
                        subprocess.run(command)
            popen.assert_not_called()

    def test_dependency_paths_restore_on_failure(self):
        import sys

        before = sys.path[:]
        with self.assertRaisesRegex(ValueError, "fixture"):
            with official.dependency_paths([self.work]):
                self.assertEqual(sys.path[0], str(self.work.resolve()))
                raise ValueError("fixture")
        self.assertEqual(before, sys.path)

    def test_static_filter_cpu_is_explicitly_unsupported(self):
        with self.assertRaisesRegex(ValueError, "CUDA"):
            official.static_filter([], {}, "cpu")


@unittest.skipUnless(importlib.util.find_spec("decord") and importlib.util.find_spec("imageio_ffmpeg"),
                     "Optional real media validation requires decord and imageio-ffmpeg")
class RealLosslessMediaTests(WorkCase):
    def test_actual_lossless_rgb_identity_three_decoders(self):
        frames = np.random.default_rng(91).integers(0, 256, size=(12, 32, 64, 3), dtype=np.uint8)
        frames[0, :, :, 0] = 255
        frames[1, :, :, 1] = 0
        path = self.work / "rgb.mp4"
        ffmpeg = runner.ffmpeg_executable()
        command = runner.encode_lossless_rgb(frames, 24, path, ffmpeg)
        self.assertIn("libx264rgb", command)
        proof = runner.verify_video(frames, 24, path, ffmpeg)
        self.assertTrue(proof["exact_rgb_identity"])
        self.assertEqual(proof["verified_frames_per_decoder"], 12)
        altered = frames.copy()
        altered[0, 0, 0, 0] ^= 1
        with self.assertRaisesRegex(ValueError, "RGB differs"):
            runner.verify_video(altered, 24, path, ffmpeg)
        with self.assertRaisesRegex(ValueError, "FPS mismatch"):
            runner.verify_video(frames, 16, path, ffmpeg)
        with self.assertRaisesRegex(ValueError, "new"):
            runner.encode_lossless_rgb(frames, 24, path, ffmpeg)

    def test_incompatible_decord_geometry_is_rejected_not_padded(self):
        frames = np.random.default_rng(91).integers(0, 256, size=(3, 32, 48, 3), dtype=np.uint8)
        path = self.work / "nonaligned.mp4"
        ffmpeg = runner.ffmpeg_executable()
        runner.encode_lossless_rgb(frames, 24, path, ffmpeg)
        try:
            proof = runner.verify_video(frames, 24, path, ffmpeg)
        except ValueError as error:
            self.assertIn("Decord", str(error))
        else:
            # A future decoder may fix the upstream stride limitation, but only
            # exact pixels and dimensions may pass.
            self.assertTrue(proof["exact_rgb_identity"])

    def test_complete_cpu_prepare_emits_bound_ssim_and_media(self):
        spec, _ = fixture_protocol()
        spec["models"]["model_a"]["width"] = 64
        raw, candidate = self.clip(), self.clip("svd_hevc")
        frames = np.random.default_rng(0).integers(0, 256, size=(96, 16, 64, 3), dtype=np.uint8)
        self.save_rgb(raw, frames)
        self.save_rgb(candidate, frames.copy())
        inputs_path = self.write_json("inputs.json", self.inputs(raw, candidate))
        spec_path = self.write_json("protocol.json", spec)
        with mock.patch.object(protocol, "load_protocol", return_value=(spec, "c" * 64)):
            result = runner.prepare(spec_path, inputs_path, self.work / "prepared", ssim_workers=2)
        self.assertEqual(result["status"], "complete")
        self.assertFalse(result["models_executed"])
        self.assertEqual(len(result["clips"]), 2)
        self.assertTrue(all(r["ssim"]["mean"] == 1 for r in result["clips"]))
        self.assertEqual(len(result["clips"][1]["ssim"]["consecutive_one_second_blocks"]), 4)
        persisted = protocol.read_json(self.work / "prepared/prepared.json")
        self.assertEqual(persisted, result)
        self.assertEqual(result["ssim_summary"][0]["case_count"], 1)


if __name__ == "__main__":
    unittest.main()
