import copy
import hashlib
import importlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import socket
import sys
import threading
import time
import types
import unittest
from unittest import mock
import uuid

import numpy as np

from examples import rank_metrics as metrics


def metadata(*, arm="RAW", clip="raw-clip"):
    result = {
        "run_id": "same-fixture-run", "case_id": "case", "prompt_group": "group",
        "split": "calibration", "seed": 7, "cell": "fixture-cell",
        "decode_mode": "full_prefix", "source_fps": 16.0,
        "fps_authority": "explicit fixture playback rate; not an experiment",
        "working_hw": [34, 36], "crop": [1, 33, 2, 34], "crop_applied": False,
        "source_start_frame": 100, "window_start": 2, "window_count": 6,
        "prefix_frames": 2, "clip_kind": "continuation", "evidence_kind": "synthetic_fixture",
        "clip_id": clip, "reference_clip_id": "raw-clip", "arm_id": arm,
        "trajectory_id": "trajectory-" + arm,
        "genuine_frames": True, "padding_frames": 0, "duplicated_frames": 0,
    }
    for name in metrics.PAIR_FIELDS:
        if name.endswith("_sha256"):
            result[name] = "1" * 64
    return result


def native_metadata(*, cell="causvid480", chunks=3):
    study = metrics._parent_study()
    geometry = study.study_cell(cell)
    prefix = 4 * geometry.frames - (3 if cell == "causvid480" else 0)
    total = 4 * geometry.frames * chunks - (3 if cell == "causvid480" else 0)
    result = metadata()
    result.update(
        evidence_kind="native", cell=cell, chunks=chunks,
        source_fps=study.NATIVE_SOURCE_FPS[cell], source_start_frame=0,
        window_start=prefix, window_count=total - prefix, prefix_frames=prefix,
        decode_mode="full_prefix" if cell == "causvid480" else "stateless_chunk",
        working_hw=[geometry.working_height, geometry.width],
        crop=[geometry.crop_top, geometry.crop_top + geometry.height, 0, geometry.width],
    )
    return result


def dimension_scores(value=.8):
    result = {}
    for name in metrics.DIMENSIONS:
        normalized = 1.0 if name == "dynamic_degree" else value
        divisor = 100.0 if name == "imaging_quality" else 10.0 if name == "aesthetic_quality" else 1.0
        result[name] = {
            "normalized_score": normalized, "raw_score": normalized * divisor,
            "official_dimension_score": normalized,
            "official_per_video_result": normalized * divisor if name == "imaging_quality" else normalized,
            "normalization_divisor": divisor,
        }
    return result


class OwnedMetricTest(unittest.TestCase):
    def setUp(self):
        self.directory = Path(".test-artifacts") / (f"rank-metric-fixture-{os.getpid()}-" + uuid.uuid4().hex)
        self.directory.mkdir(parents=True)

    def tearDown(self):
        shutil.rmtree(self.directory)

    def descriptor(self, filename, *, arm="RAW", clip="raw-clip", frames=None, changes=None):
        path = self.directory / filename
        if frames is None:
            frames = np.arange(8 * 34 * 36 * 3, dtype=np.uint8).reshape(8, 34, 36, 3)
        np.savez(path, frames=frames)
        data = metadata(arm=arm, clip=clip)
        data.update(changes or {})
        return {"path": str(path), "sha256": metrics.sha256(path), "metadata": data}


class TemporalPolicyTests(unittest.TestCase):
    def test_official_fps_sampling_is_not_frame_count(self):
        policy = metrics.frame_policy(24, 16)
        self.assertEqual(policy["dynamic_degree"]["sampled_indices"], list(range(0, 24, 2)))
        self.assertEqual(policy["dynamic_degree"]["effective_fps"], 8)
        self.assertEqual(policy["dynamic_degree"]["motion_pair_count_threshold"], 3)
        self.assertEqual(policy["motion_smoothness"]["anchor_indices"], list(range(0, 24, 2)))
        self.assertEqual(policy["motion_smoothness"]["target_indices"], list(range(1, 23, 2)))
        self.assertEqual(policy["motion_smoothness"]["unused_terminal_indices"], [23])
        self.assertNotEqual(policy["dynamic_degree"]["sampled_indices"],
                            metrics.frame_policy(24, 24)["dynamic_degree"]["sampled_indices"])

    def test_short_or_invalid_frame_rate_is_failure_not_padding(self):
        for count, fps in ((2, 16), (3, 120), (24, 0), (24, float("nan")),
                           (True, 16), (24, True), (3.0, 16), (8, -1)):
            with self.subTest(count=count, fps=fps), self.assertRaises(metrics.MetricInputError):
                metrics.frame_policy(count, fps)
        self.assertEqual(metrics.frame_policy(5, 8)["motion_smoothness"]["unused_terminal_indices"], [])

    def test_native_crop_must_match_parent_geometry_not_a_resized_fixture(self):
        info = native_metadata(cell="magi1080")
        info.update(working_hw=[34, 36], crop=[1, 33, 2, 34])
        with self.assertRaisesRegex(metrics.MetricInputError, "study cell"):
            metrics.validate_clip_metadata(info)
        info.update(working_hw=[1088, 1920], crop=[4, 1084, 0, 1920])
        self.assertEqual(metrics.validate_clip_metadata(info)["crop"], [4, 1084, 0, 1920])

    def test_native_fps_matches_the_approved_cell_not_merely_the_other_arm(self):
        for cell, wrong_fps in (("causvid480", 24), ("magi720", 16), ("magi1080", 16)):
            info = native_metadata(cell=cell, chunks=2)
            good = metrics.validate_clip_metadata(info)
            if cell.startswith("magi"):
                self.assertEqual(metrics.frame_policy(good["window_count"], good["source_fps"])[
                    "dynamic_degree"
                ]["sampled_indices"], list(range(0, 24, 3)))
            info["source_fps"] = wrong_fps
            with self.subTest(cell=cell), self.assertRaisesRegex(metrics.MetricInputError, "FPS|source_fps"):
                metrics.validate_clip_metadata(info)

    def test_native_magi_rejects_full_prefix_and_verified_streaming(self):
        for cell in ("magi720", "magi1080"):
            info = native_metadata(cell=cell, chunks=3)
            self.assertEqual(metrics.validate_clip_metadata(info)["decode_mode"], "stateless_chunk")
            for mode in ("full_prefix", "verified_streaming"):
                bad = {**info, "decode_mode": mode, "streaming_verification_sha256": "2" * 64}
                with self.subTest(cell=cell, mode=mode), self.assertRaisesRegex(
                    metrics.MetricInputError, "MAGI.*stateless_chunk",
                ):
                    metrics.validate_clip_metadata(bad)

    def test_causvid_main_window_excludes_nine_frames_and_binds_the_full_run(self):
        for chunks, count in ((2, 12), (3, 24)):
            info = native_metadata(chunks=chunks)
            self.assertEqual(metrics.validate_clip_metadata(info)["window_count"], count)
            extracted = {**info, "source_start_frame": 9, "window_start": 0}
            self.assertEqual(metrics.validate_clip_metadata(extracted)["prefix_frames"], 9)
        info = native_metadata(chunks=3)
        for change in (
            {"prefix_frames": 0, "window_start": 0, "window_count": 33},
            {"window_start": 21, "window_count": 12},
            {"window_count": 12},
            {"source_start_frame": 0, "window_start": 0},
            {"source_start_frame": 9, "window_start": 9},
            {"decode_mode": "stateless_chunk"},
            {"chunks": 1}, {"chunks": 4}, {"chunks": True}, {"chunks": None},
        ):
            with self.subTest(change=change), self.assertRaises(metrics.MetricInputError):
                metrics.validate_clip_metadata({**info, **change})
        info.pop("chunks")
        with self.assertRaisesRegex(metrics.MetricInputError, "chunks"):
            metrics.validate_clip_metadata(info)

    def test_native_diagnostic_windows_remain_separately_labelled(self):
        info = native_metadata()
        info.update(clip_kind="diagnostic_window", window_label="unchanged_first_chunk",
                    window_start=0, window_count=9)
        self.assertEqual(metrics.validate_clip_metadata(info)["window_label"], "unchanged_first_chunk")
        for change in ({"window_label": ""}, {"window_count": 34}):
            with self.subTest(change=change), self.assertRaises(metrics.MetricInputError):
                metrics.validate_clip_metadata({**info, **change})

    def test_native_pair_binds_chunk_horizon_and_diagnostic_window_label(self):
        data = native_metadata(chunks=2)
        data.update(clip_kind="diagnostic_window", window_label="first_chunk", window_start=0, window_count=9)
        raw = {
            "metadata": data, "source_total_frames": 9, "source_frame_indices": list(range(9)),
            "shape": [9, 480, 832, 3], "dtype": "uint8", "color_order": "RGB", "source_fps": 16,
        }
        candidate = copy.deepcopy(raw)
        candidate["metadata"].update(arm_id="adaptive", clip_id="candidate", trajectory_id="candidate-trajectory")
        metrics.validate_pair(raw, candidate)
        for name, value in (("chunks", 3), ("window_label", "other_window")):
            changed = copy.deepcopy(candidate)
            changed["metadata"][name] = value
            with self.subTest(name=name), self.assertRaisesRegex(metrics.MetricInputError, name):
                metrics.validate_pair(raw, changed)


class RGBPairTests(OwnedMetricTest):
    def test_lossless_crop_window_fps_and_same_raw_metadata(self):
        raw = self.descriptor("raw.npz")
        candidate = self.descriptor("candidate.npz", arm="adaptive", clip="candidate")
        frames, info = metrics.load_rgb_npz(raw)
        self.assertEqual(frames.shape, (6, 32, 32, 3))
        self.assertEqual(info["source_frame_indices"], list(range(102, 108)))
        self.assertEqual(info["source_fps"], 16)
        result = metrics.evaluate_pair(raw, candidate, output_dir=self.directory)
        self.assertEqual(result["ssim"]["per_frame"], [1] * 6)
        self.assertEqual(result["ssim"]["mean"], 1)
        self.assertEqual(result["ssim"]["minimum"], 1)
        self.assertEqual(result["ssim"]["parameters"]["channel_axis"], -1)
        self.assertEqual(result["ssim"]["parameters"]["data_range"], 255)
        self.assertEqual(result["vbench_status"], "not_requested")
        self.assertFalse(result["full_vbench_aggregate"])
        self.assertEqual(result["candidate"]["metadata"]["evidence_kind"], "synthetic_fixture")

    def test_already_cropped_does_not_apply_crop_twice(self):
        frames = np.ones((8, 32, 32, 3), dtype=np.uint8)
        descriptor = self.descriptor("cropped.npz", frames=frames, changes={"crop_applied": True})
        decoded, _ = metrics.load_rgb_npz(descriptor)
        np.testing.assert_array_equal(decoded, frames[2:])

    def test_invalid_dtype_geometry_hash_window_prefix_padding(self):
        descriptors = [
            self.descriptor("float.npz", frames=np.zeros((8, 34, 36, 3), dtype=np.float32)),
            self.descriptor("gray.npz", frames=np.zeros((8, 34, 36), dtype=np.uint8)),
            self.descriptor("size.npz", frames=np.zeros((8, 33, 36, 3), dtype=np.uint8)),
            self.descriptor("window.npz", changes={"window_count": 7}),
            self.descriptor("prefix.npz", changes={"source_start_frame": 0, "window_start": 1}),
            self.descriptor("padding.npz", changes={"padding_frames": 1}),
            self.descriptor("dup.npz", changes={"duplicated_frames": 1}),
            self.descriptor("fps.npz", changes={"source_fps": float("nan")}),
            self.descriptor("crop.npz", changes={"crop": [0, 80, 0, 36]}),
        ]
        bad_hash = self.descriptor("hash.npz")
        bad_hash["sha256"] = "f" * 64
        descriptors.append(bad_hash)
        for descriptor in descriptors:
            with self.subTest(path=descriptor["path"]), self.assertRaises(metrics.MetricInputError):
                metrics.load_rgb_npz(descriptor)

    def test_extracted_clip_uses_absolute_source_coordinates_to_exclude_prefix(self):
        frames = np.arange(6 * 34 * 36 * 3, dtype=np.uint8).reshape(6, 34, 36, 3)
        descriptor = self.descriptor(
            "extracted.npz", frames=frames,
            changes={"source_start_frame": 2, "window_start": 0, "window_count": 6, "prefix_frames": 2},
        )
        loaded, info = metrics.load_rgb_npz(descriptor)
        self.assertEqual(info["source_frame_indices"], list(range(2, 8)))
        self.assertEqual(len(loaded), 6)

    def test_diagnostic_window_scores_cannot_enter_main_continuation_tables(self):
        changes = {"clip_kind": "diagnostic_window", "window_label": "fixture_first_window",
                   "source_start_frame": 0, "window_start": 0}
        raw = self.descriptor("raw-window.npz", changes=changes)
        candidate = self.descriptor("candidate-window.npz", arm="adaptive", clip="window-candidate",
                                    changes=changes)
        scorer = mock.Mock()
        scorer.score_frames.return_value = {"status": "complete", "scores": dimension_scores()}
        result = metrics.evaluate_pair(raw, candidate, output_dir=self.directory, scorer=scorer)
        self.assertEqual(result["metric_scope"], "diagnostic_window")
        self.assertEqual(result["window_label"], changes["window_label"])
        self.assertIn("diagnostic window", result["ssim"]["comparison"])
        with self.assertRaisesRegex(metrics.MetricInputError, "diagnostic|main continuation"):
            metrics.quality_outcomes(result)

    def test_precise_ssim_parameters_and_nonfinite_rejected(self):
        frames = np.zeros((3, 8, 8, 3), dtype=np.uint8)
        observed = []

        def fake_ssim(raw, candidate, **kwargs):
            observed.append(kwargs)
            return [.9, .7, .8][len(observed) - 1]

        result = metrics.ssim_frames(frames, frames, ssim_function=fake_ssim)
        self.assertAlmostEqual(result["mean"], .8)
        self.assertEqual(result["minimum"], .7)
        self.assertTrue(all(row["data_range"] == 255 and row["channel_axis"] == -1 for row in observed))
        with self.assertRaises(metrics.MetricInputError):
            metrics.ssim_frames(frames, frames, ssim_function=lambda *args, **kwargs: float("nan"))
        with self.assertRaises(metrics.MetricInputError):
            metrics.ssim_frames(frames, frames[:, :-1])

    def test_pair_rejects_wrong_same_run_source_seed_crop_fps_and_reference(self):
        raw = self.descriptor("raw.npz")
        candidate = self.descriptor("candidate.npz", arm="adaptive", clip="candidate")
        _, raw_info = metrics.load_rgb_npz(raw)
        _, good_info = metrics.load_rgb_npz(candidate)
        for name, value in (
            ("run_id", "other"), ("seed", 8), ("source_fps", 24.0),
            ("model_source_sha256", "f" * 64), ("reference_clip_id", "other"),
            ("crop", [0, 32, 0, 32]), ("window_start", 0), ("split", "heldout"),
        ):
            bad = copy.deepcopy(good_info)
            bad["metadata"][name] = value
            with self.subTest(name=name), self.assertRaises(metrics.MetricInputError):
                metrics.validate_pair(raw_info, bad)

    def test_matching_false_frame_coordinates_or_fps_cannot_pass_pair_validation(self):
        raw = self.descriptor("raw.npz")
        candidate = self.descriptor("candidate.npz", arm="adaptive", clip="candidate")
        _, raw_info = metrics.load_rgb_npz(raw)
        _, candidate_info = metrics.load_rgb_npz(candidate)
        for name, value in (
            ("source_frame_indices", list(range(6))), ("source_fps", 24),
            ("shape", [6, 31, 32, 3]), ("source_total_frames", 5),
        ):
            changed = [copy.deepcopy(raw_info), copy.deepcopy(candidate_info)]
            for info in changed:
                info[name] = value
            with self.subTest(name=name), self.assertRaises(metrics.MetricInputError):
                metrics.validate_pair(*changed)

    def test_mocked_official_models_keep_exact_six_dimensions_and_units(self):
        raw = self.descriptor("raw.npz")
        candidate = self.descriptor("candidate.npz", arm="adaptive", clip="candidate")
        scorer = mock.Mock()
        scorer.score_frames.side_effect = [
            {"status": "complete", "scores": dimension_scores(.8)},
            {"status": "complete", "scores": dimension_scores(.7)},
        ]
        result = metrics.evaluate_pair(raw, candidate, output_dir=self.directory, scorer=scorer)
        self.assertEqual(result["vbench_status"], "complete")
        self.assertEqual(set(result["vbench"]["normalized_delta"]), set(metrics.DIMENSIONS))
        self.assertAlmostEqual(result["vbench"]["normalized_delta"]["imaging_quality"], -.1)
        self.assertEqual(result["vbench"]["normalized_delta"]["dynamic_degree"], 0)
        self.assertEqual(scorer.score_frames.call_count, 2)
        self.assertEqual(result["vbench"]["raw"]["scores"]["imaging_quality"]["raw_score"], 80)
        outcomes = metrics.quality_outcomes(result)
        self.assertEqual({row["arm_id"] for row in outcomes}, {"RAW", "adaptive"})
        self.assertEqual(outcomes[0]["pairing_sha256"], outcomes[1]["pairing_sha256"])
        self.assertEqual(outcomes[1]["metrics"]["ssim"], 1)
        self.assertEqual(set(outcomes[0]["metrics"]), {"ssim", *metrics.DIMENSIONS})
        result["ssim"]["mean"] = .2
        with self.assertRaisesRegex(metrics.MetricInputError, "SSIM summary"):
            metrics.quality_outcomes(result)

    def test_actual_metric_outputs_pair_different_candidates_but_not_other_raw_window_or_sampler(self):
        from reelcache.rank_analysis import AnalysisError, paired_accounting_tables, paired_group_bootstrap

        frames = np.arange(8 * 34 * 36 * 3, dtype=np.uint8).reshape(8, 34, 36, 3)
        raw = self.descriptor("shared-raw.npz", frames=frames)
        candidates = [
            self.descriptor("candidate-a.npz", arm="candidate-a", clip="candidate-a",
                            frames=((frames.astype(np.uint16) + 7) % 256).astype(np.uint8)),
            self.descriptor("candidate-b.npz", arm="candidate-b", clip="candidate-b", frames=255 - frames),
        ]
        sampler = {
            "fixture_only": True, "execution": "serial_clean_history",
            "clean_history_chunk_window": 1, "steps_per_chunk": [16, 16, 16],
            "source_config_sha256": hashlib.sha256(b"fixture sampler source/config").hexdigest(),
        }
        sampling_hash = hashlib.sha256(metrics._json_bytes(sampler)).hexdigest()
        for descriptor in (raw, *candidates):
            descriptor["metadata"]["sampling_sha256"] = sampling_hash
        scorer = mock.Mock()
        scorer.score_frames.return_value = {"status": "complete", "scores": dimension_scores()}
        accounting = {
            "native_logical_bytes": 1000, "source_bytes": 1000, "decoder_closure_bytes": 100,
            "snapshot_closure_bytes": 120, "filesystem_bytes": 150,
            "basis_bytes": 40, "payload_bytes": 50, "metadata_bytes": 10,
            "work_counts": {}, "resource_bytes": {}, "timings_seconds": {},
        }

        def outcomes(reference, candidate):
            result = metrics.evaluate_pair(reference, candidate, output_dir=self.directory, scorer=scorer)
            self.assertNotEqual(result["candidate"]["rgb_sha256"], result["raw"]["rgb_sha256"])
            self.assertLess(result["ssim"]["mean"], 1)
            return metrics.quality_outcomes(
                result, raw_accounting=accounting, candidate_accounting=accounting,
            )

        first, second = (outcomes(raw, candidate) for candidate in candidates)
        self.assertEqual(first[0]["pairing_sha256"], first[1]["pairing_sha256"])
        self.assertEqual(first[0]["pairing_sha256"], second[1]["pairing_sha256"])
        self.assertNotEqual(first[1]["paired_metrics_sha256"], second[1]["paired_metrics_sha256"])
        expected = [{name: raw["metadata"][name] for name in (
            "case_id", "prompt_group", "cell", "split", "seed",
        )}]
        combined = [first[0], first[1], second[1]]
        for arm in ("candidate-a", "candidate-b"):
            self.assertEqual(paired_group_bootstrap(
                expected, combined, reference_arm="RAW", candidate_arm=arm, resamples=8,
            )["coverage"], 1)
            self.assertEqual(paired_accounting_tables(
                expected, combined, reference_arm="RAW", candidate_arm=arm,
            )["coverage"], 1)

        other_raw = self.descriptor(
            "other-raw.npz", frames=frames, clip="different-raw-reference",
            changes={"reference_clip_id": "different-raw-reference", "sampling_sha256": sampling_hash},
        )
        other_candidate = copy.deepcopy(candidates[0])
        other_candidate["metadata"]["reference_clip_id"] = "different-raw-reference"
        changed_pairs = [(other_raw, other_candidate)]
        for change in (
            {"window_start": 3, "window_count": 5},
            {"sampling_sha256": hashlib.sha256(metrics._json_bytes(
                {**sampler, "clean_history_chunk_window": 4},
            )).hexdigest()},
        ):
            reference, candidate = copy.deepcopy(raw), copy.deepcopy(candidates[0])
            reference["metadata"].update(change)
            candidate["metadata"].update(change)
            changed_pairs.append((reference, candidate))
        for reference, candidate in changed_pairs:
            changed = outcomes(reference, candidate)[1]
            self.assertNotEqual(first[0]["pairing_sha256"], changed["pairing_sha256"])
            for function in (paired_group_bootstrap, paired_accounting_tables):
                with self.subTest(reference=reference["metadata"], report=function.__name__), \
                        self.assertRaisesRegex(AnalysisError, "pairing|paired window"):
                    function(expected, [first[0], changed], reference_arm="RAW", candidate_arm=changed["arm_id"])

    def test_missing_nonfinite_or_incorrectly_normalized_dimension_fails(self):
        good = dimension_scores()
        bad_missing = copy.deepcopy(good)
        bad_missing.pop("imaging_quality")
        bad_nan = copy.deepcopy(good)
        bad_nan["subject_consistency"]["normalized_score"] = float("nan")
        bad_units = copy.deepcopy(good)
        bad_units["imaging_quality"]["official_per_video_result"] = .8
        bad_binary = dimension_scores()
        bad_binary["dynamic_degree"] = {**bad_binary["dynamic_degree"], "normalized_score": .5}
        for scores in (bad_missing, bad_nan, bad_units, bad_binary):
            with self.assertRaises(metrics.MetricInputError):
                metrics.validate_dimension_scores(scores)

    def test_png_transport_preserves_all_and_official_dynamic_subset(self):
        descriptor = self.descriptor("frames.npz")
        frames, _ = metrics.load_rgb_npz(descriptor)
        policy = metrics.frame_policy(len(frames), 16)
        full, dynamic, evidence = metrics.verified_png_inputs(frames, policy, self.directory / "transport")
        np.testing.assert_array_equal(np.asarray(metrics.read_png_frames(full)), frames)
        np.testing.assert_array_equal(np.asarray(metrics.read_png_frames(dynamic)), frames[::2])
        self.assertTrue(evidence["decode_identity"])
        self.assertFalse(evidence["lossy_encoding_performed"])
        (full / "unexpected.txt").write_text("fixture")
        with self.assertRaises(metrics.MetricInputError):
            metrics.verified_png_inputs(frames, policy, self.directory / "transport")

    def test_six_official_dispatches_are_transport_only_with_model_free_mocks(self):
        import torch

        frames, info = metrics.load_rgb_npz(self.descriptor("source.npz"))
        scorer = object.__new__(metrics.StrictVBenchScorer)
        scorer.pid = os.getpid()
        scorer.torch, scorer.device = torch, torch.device("cpu")
        scorer.provenance = {"evidence_kind": "synthetic_fixture", "no_models_loaded": True}
        scorer.config = {"environment": {}}
        scorer.modules, scorer.models = {}, {}
        original_loaders = {}
        entry_names = {
            "subject_consistency": "subject_consistency",
            "background_consistency": "background_consistency",
            "motion_smoothness": "motion_smoothness", "dynamic_degree": "dynamic_degree",
            "aesthetic_quality": "laion_aesthetic", "imaging_quality": "technical_quality",
        }
        for dimension in metrics.DIMENSIONS:
            normalized = 1 if dimension == "dynamic_degree" else .8
            video = 80 if dimension == "imaging_quality" else normalized
            loader = mock.Mock(side_effect=AssertionError("original video decoder must not run"))
            original_loaders[dimension] = loader
            scorer.modules[dimension] = types.SimpleNamespace(
                load_video=loader,
                **{entry_names[dimension]: mock.Mock(return_value=(normalized, [{"video_results": video}]))},
            )
            scorer.models[dimension] = types.SimpleNamespace()
        scorer.models["background_consistency"] = (object(), object())
        scorer.models["aesthetic_quality"] = (object(), object())
        scorer.models["dynamic_degree"] = types.SimpleNamespace(params={"thres": .75, "count_num": 1})
        with mock.patch.object(metrics, "verify_environment", return_value={}) as environment:
            result = scorer.score_frames(frames, info, self.directory / "mock-worker")
        self.assertEqual(result["status"], "complete")
        self.assertEqual(set(result["scores"]), set(metrics.DIMENSIONS))
        self.assertEqual(result["scores"]["dynamic_degree"]["sampled_source_indices"], [102, 104, 106])
        self.assertEqual(result["scores"]["motion_smoothness"]["target_indices"], [1, 3])
        self.assertEqual(result["scores"]["imaging_quality"]["raw_score"], 80)
        for name in metrics.DIMENSIONS:
            self.assertIs(scorer.modules[name].load_video, original_loaders[name])
            getattr(scorer.modules[name], entry_names[name]).assert_called_once()
        kwargs = scorer.modules["imaging_quality"].technical_quality.call_args.kwargs
        self.assertEqual(kwargs, {"imaging_quality_preprocessing_mode": "longer"})
        environment.assert_called_once()


class LoaderAndIsolationTests(OwnedMetricTest):
    def test_workspace_ancestry_is_portable_to_an_opaque_relocated_source_root(self):
        project = PurePosixPath("/tmp/reelcache-online-rank-fixture-x/diagnostic-code-v1")
        metrics._validate_workspace_ancestry(project / ".tmp-owned-fixture" / "worker", project)
        for denied in (
            project, project.parent, project.parent / "unrelated", PurePosixPath("/tmp/arbitrary"),
            PurePosixPath(str(project) + "-lookalike") / "worker",
            project / ".." / "escaped", PurePosixPath("relative/worker"),
        ):
            with self.subTest(path=str(denied)), self.assertRaisesRegex(metrics.MetricInputError, "PROJECT_ROOT"):
                metrics._validate_workspace_ancestry(denied, project)

    def test_worker_environment_uses_the_executing_project_root_not_its_directory_name(self):
        project = (self.directory / "unusual-relocated-project").resolve()
        project.mkdir(mode=0o700)
        workspace = project / ".tmp-owned-fixture" / "worker"
        with mock.patch.object(metrics, "PROJECT_ROOT", project):
            environment = metrics.worker_environment(workspace)
        self.assertEqual(workspace.stat().st_mode & 0o777, 0o700)
        for name in ("TMPDIR", "MPLCONFIGDIR", "TORCH_HOME", "HF_HOME", "XDG_CACHE_HOME",
                     "CUDA_CACHE_PATH", "TORCH_EXTENSIONS_DIR"):
            path = Path(environment[name])
            self.assertTrue(path.is_relative_to(workspace))
            self.assertEqual(path.stat().st_mode & 0o777, 0o700)
        self.assertEqual(environment["CUDA_VISIBLE_DEVICES"], "")
        self.assertEqual(environment["HF_HUB_OFFLINE"], "1")

    def test_worker_environment_rejects_external_shared_and_symlink_workspaces(self):
        project = (self.directory / "project").resolve()
        project.mkdir(mode=0o700)
        outside = (self.directory / "outside").resolve()
        outside.mkdir(mode=0o700)
        shared = project / "shared"
        shared.mkdir(mode=0o700)
        shared.chmod(0o755)
        link = project / "escape"
        link.symlink_to(outside, target_is_directory=True)
        with mock.patch.object(metrics, "PROJECT_ROOT", project):
            for path in (project, outside, shared, link):
                with self.subTest(path=path), self.assertRaises(metrics.MetricInputError):
                    metrics.worker_environment(path)
        self.assertFalse((outside / "worker_cache").exists())
        self.assertFalse((shared / "worker_cache").exists())
        private = project / "private"
        private.mkdir(mode=0o700)
        with mock.patch.object(metrics, "PROJECT_ROOT", project), \
                mock.patch.object(metrics.os, "geteuid", return_value=os.getuid() + 1), \
                self.assertRaisesRegex(metrics.MetricInputError, "owned by this user"):
            metrics.worker_environment(private)
        self.assertFalse((private / "worker_cache").exists())
        (private / "worker_cache").symlink_to(outside, target_is_directory=True)
        with mock.patch.object(metrics, "PROJECT_ROOT", project), self.assertRaisesRegex(
            metrics.MetricInputError, "symlink",
        ):
            metrics.worker_environment(private)

    def test_array_loader_is_float32_tchw_and_restores_on_error(self):
        import torch
        original = lambda path: None
        module = types.SimpleNamespace(load_video=original)
        frames = np.arange(3 * 8 * 8 * 3, dtype=np.uint8).reshape(3, 8, 8, 3)
        with self.assertRaisesRegex(RuntimeError, "fixture"):
            with metrics.array_loader(module, frames, "bound"):
                tensor = module.load_video("bound")
                self.assertEqual(tensor.dtype, torch.float32)
                self.assertEqual(tuple(tensor.shape), (3, 3, 8, 8))
                np.testing.assert_array_equal(tensor.permute(0, 2, 3, 1).numpy(), frames)
                with self.assertRaises(metrics.MetricInputError):
                    module.load_video("another")
                raise RuntimeError("fixture")
        self.assertIs(module.load_video, original)

    def test_scoped_loader_changes_serialize_threads(self):
        module = types.SimpleNamespace(load_video=lambda path: None)
        frames = np.zeros((3, 8, 8, 3), dtype=np.uint8)
        entered = threading.Event()
        completed = threading.Event()

        def other():
            with metrics.array_loader(module, frames, "other"):
                entered.set()
            completed.set()

        with metrics.array_loader(module, frames, "first"):
            worker = threading.Thread(target=other)
            worker.start()
            self.assertFalse(entered.wait(.05))
            self.assertEqual(tuple(module.load_video("first").shape), (3, 3, 8, 8))
        self.assertTrue(completed.wait(5))
        worker.join()

    def test_network_attempt_is_blocked_and_binding_restored(self):
        original = socket.create_connection
        with metrics.offline_network():
            with self.assertRaisesRegex(metrics.MetricInputError, "network"):
                socket.create_connection(("fixture.invalid", 1))
        self.assertIs(socket.create_connection, original)

    def test_model_load_is_denied_before_sources_or_packages_are_touched(self):
        with mock.patch.object(metrics, "verify_metric_sources") as sources:
            with self.assertRaises(metrics.MetricInputError):
                metrics.StrictVBenchScorer({}, allow_model_load=False)
            with mock.patch.dict(os.environ, {}, clear=True):
                with self.assertRaises(metrics.MetricInputError):
                    metrics.StrictVBenchScorer({}, allow_model_load=True)
        sources.assert_not_called()

    def test_model_admission_reuses_parent_gate_without_a_new_gate_schema(self):
        from reelcache import study

        admission = {
            "protocol": {"protocol_id": "fixture-protocol"},
            "receipt": {"receipt_id": "fixture-receipt"}, "phase": "calibration_frontier",
        }
        before = list(sys.path)
        with mock.patch.object(study, "verify_launch_receipt") as gate:
            result = metrics.verify_study_admission(admission)
        gate.assert_called_once_with(admission["receipt"], admission["protocol"], phase="calibration_frontier")
        self.assertEqual(result["receipt_id"], "fixture-receipt")
        self.assertEqual(sys.path, before)
        with self.assertRaises(metrics.MetricInputError):
            metrics.verify_study_admission({"allow_model_load": True})

    def test_environment_numpy_contamination_guard_with_mock_distributions(self):
        environment = {
            "python_executable": sys.executable, "prefix": sys.prefix,
            "python_version": ".".join(map(str, sys.version_info[:3])),
            "versions": {name: "fixture-version" for name in metrics.REQUIRED_PACKAGES},
            "package_roots": {name: str(self.directory.resolve()) for name in metrics.REQUIRED_PACKAGES},
            "extra_sys_path": [],
        }
        distribution = types.SimpleNamespace(version="fixture-version",
                                             locate_file=lambda name: self.directory.resolve())
        with mock.patch.object(metrics.importlib.metadata, "distribution", return_value=distribution):
            with self.assertRaisesRegex(metrics.MetricInputError, "contaminates"):
                metrics.verify_environment(environment)
        wrong = copy.deepcopy(environment)
        wrong["prefix"] = str(self.directory / "other-env")
        with self.assertRaisesRegex(metrics.MetricInputError, "prefix"):
            metrics.verify_environment(wrong)

    def test_worker_launch_isolated_and_no_pythonpath_leak(self):
        request = self.directory / "request.json"
        metrics.write_json(request, {"vbench": False})
        with mock.patch.dict(os.environ, {"PYTHONPATH": "/modelenv/numpy126",
                                          "LD_LIBRARY_PATH": "/modelenv/lib"}):
            with mock.patch.object(metrics.subprocess, "run") as run:
                metrics.launch_metric_worker(request, self.directory / "out.json",
                                              python_executable=sys.executable,
                                              workspace=self.directory / "worker")
        args, kwargs = run.call_args
        self.assertIn("-I", args[0])
        self.assertNotIn("PYTHONPATH", kwargs["env"])
        self.assertNotIn("LD_LIBRARY_PATH", kwargs["env"])
        self.assertEqual(kwargs["env"]["CUDA_VISIBLE_DEVICES"], "")
        self.assertEqual(kwargs["env"]["HF_HUB_OFFLINE"], "1")
        self.assertTrue(kwargs["check"])
        metrics.write_json(self.directory / "denied.json", {"vbench": True})
        with self.assertRaisesRegex(metrics.MetricInputError, "approval"):
            metrics.launch_metric_worker(self.directory / "denied.json", self.directory / "out.json",
                                          python_executable=sys.executable, workspace=self.directory)

    def test_source_hash_mismatch_in_model_free_local_fixture(self):
        root = self.directory / "source"
        root.mkdir()
        (root / "source.py").write_text("FIXTURE = True\n")
        hashes = {"source.py": metrics.sha256(root / "source.py")}
        self.assertEqual(metrics._verify_source_tree(root, hashes), str(root.resolve()))
        (root / "extra.py").write_text("UNEXPECTED = True\n")
        with self.assertRaisesRegex(metrics.MetricInputError, "inventory"):
            metrics._verify_source_tree(root, hashes)
        hashes["extra.py"] = "f" * 64
        with self.assertRaisesRegex(metrics.MetricInputError, "identity"):
            metrics._verify_source_tree(root, hashes)

    def test_wrong_pin_fails_before_external_checkout_probe(self):
        with mock.patch.object(metrics.subprocess, "check_output") as probe:
            with self.assertRaisesRegex(metrics.MetricInputError, "pinned"):
                metrics.verify_metric_sources({"vbench_commit": "f" * 40})
        probe.assert_not_called()

    def test_source_and_weight_manifest_success_then_tamper_uses_no_models(self):
        official, dino = self.directory / "official", self.directory / "dino"
        (official / "pretrained/amt_model").mkdir(parents=True)
        dino.mkdir()
        (official / "metric.py").write_text("FIXTURE = True\n")
        (official / "pretrained/amt_model/AMT-S.yaml").write_text("fixture: true\n")
        for name in ("hubconf.py", "vision_transformer.py"):
            (dino / name).write_text("FIXTURE = True\n")
        weights = {}
        for name in (*metrics.DIMENSIONS, "aesthetic_head"):
            path = self.directory / ("sa_0_4_vit_l_14_linear.pth" if name == "aesthetic_head" else name + ".pth")
            path.write_bytes(b"not a model; fixture bytes only")
            weights[name] = {"path": str(path), "sha256": metrics.sha256(path)}
        config = {
            "vbench_commit": metrics.VBENCH_COMMIT, "vbench_source_root": str(official),
            "source_sha256": {str(path.relative_to(official)): metrics.sha256(path)
                              for path in official.rglob("*") if path.is_file()},
            "dino_source_root": str(dino),
            "dino_source_sha256": {path.name: metrics.sha256(path) for path in dino.iterdir()},
            "weights": weights,
        }
        with mock.patch.object(metrics.subprocess, "check_output", side_effect=[metrics.VBENCH_COMMIT, ""]):
            self.assertEqual(metrics.verify_metric_sources(config)["vbench_commit"], metrics.VBENCH_COMMIT)
        Path(weights["dynamic_degree"]["path"]).write_bytes(b"changed fixture")
        with mock.patch.object(metrics.subprocess, "check_output", side_effect=[metrics.VBENCH_COMMIT, ""]):
            with self.assertRaisesRegex(metrics.MetricInputError, "weight"):
                metrics.verify_metric_sources(config)


if __name__ == "__main__":
    unittest.main()
