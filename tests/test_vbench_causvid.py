"""CPU-only fresh-case contracts; fake models/catalogs are not native VBench proof."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import uuid

import numpy as np
import torch

from reelcache.adapters.causvid import CausVidAdapter, tensor_digest
from reelcache.adapters.causvid_benchmark import (
    checkpoint_bytes, encode_lossless_video, fresh_inputs, generate_matched_case,
    filesystem_bytes, object_graph, paired_ssim, rgb_uint8, save_evaluator_output,
)
from reelcache.errors import IntegrityError
from reelcache.manager import KVRuntime
from reelcache.types import CodecSpec, MatrixKey, ModelIdentity
from tests.test_causvid_adapter import CONFIG, FakePipeline, FakeVAE, initial_state


ROOT = Path(__file__).resolve().parents[1]
DRIVER_PATH = ROOT / "examples/vbench_causvid.py"
SPEC = importlib.util.spec_from_file_location("vbench_causvid_cpu_tests", DRIVER_PATH)
DRIVER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DRIVER)


class TextEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.prompts = []
        self.eval()

    def forward(self, *, text_prompts):
        self.prompts.append(list(text_prompts))
        value = hashlib.sha256(text_prompts[0].encode()).digest()[0] / 255
        return {"prompt_embeds": torch.full((1, 3, 8), value, dtype=torch.float32)}


class TemporalVAE(FakeVAE):
    def decode_to_pixel(self, latents):
        values = torch.nn.functional.interpolate(
            latents[:, :, :3].flatten(0, 1).tanh(), size=(8, 12), mode="nearest",
        ).unflatten(0, (1, latents.shape[1]))
        return torch.cat((values[:, :1], values[:, 1:].repeat_interleave(4, dim=1)), dim=1)


class VBenchCausVidTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        self.directory = ROOT / ".test-work" / ("vbench-causvid-" + uuid.uuid4().hex)
        self.directory.mkdir(parents=True, mode=0o700)
        self.addCleanup(shutil.rmtree, self.directory)

    def bound_file(self, relative, data):
        path = self.directory / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return {"path": str(path), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}

    def case(self):
        value = DRIVER.schema_example()
        value["master_protocol_sha256"] = "2" * 64
        value["prompt"].update(
            id="official-fixture-0", text="In a still frame, a stop sign",
            dimensions=["temporal_flickering"], catalog_revision="1" * 40,
            catalog=self.bound_file("catalog.json", json.dumps([
                {"prompt_en": "In a still frame, a stop sign", "dimension": ["temporal_flickering"]},
            ]).encode()),
        )
        value["model"]["model_code_root"] = str(self.directory / "native")
        value["model"]["base_model"] = str(self.directory / "base")
        config = {
            "model_type": "t2v", "dim": 1536, "num_heads": 12,
            "num_layers": 30, "in_dim": 16, "out_dim": 16,
        }
        value["model"]["files"] = {
            "base_config": self.bound_file("base/config.json", json.dumps(config).encode()),
            "text_encoder": self.bound_file("base/models_t5_umt5-xxl-enc-bf16.pth", b"CPU fixture, not weights"),
            "vae": self.bound_file("base/Wan2.1_VAE.pth", b"CPU fixture, not weights"),
            "config": self.bound_file("config.yaml", b"CPU fixture only\n"),
            "checkpoint": self.bound_file("model.pt", b"CPU fixture, not weights"),
        }
        for name in DRIVER.NATIVE_REQUIRED:
            entry = self.bound_file("native/" + name, b"# Not an executable native model.\n")
            value["model"]["native_sources"][name] = entry["sha256"]
        value["ffmpeg"] = self.bound_file("ffmpeg-fixture", b"#!/bin/sh\nexit 1\n")
        Path(value["ffmpeg"]["path"]).chmod(0o700)
        return value

    def write_case(self, value, name="case.json"):
        path = self.directory / name
        path.write_text(json.dumps(value))
        return path, DRIVER.sha256(path)

    def model(self):
        return ModelIdentity("CPU-fake-not-native", "fake-weights", 2, 8, "tokens_channels", "float32")

    def pipeline(self):
        torch.manual_seed(31)
        pipeline = FakePipeline()
        pipeline.text_encoder, pipeline.vae = TextEncoder(), TemporalVAE()
        return pipeline

    def test_manifest_exact_prompt_protocol_source_and_no_legacy_prefix(self):
        case = self.case()
        path, digest = self.write_case(case)
        self.assertEqual(DRIVER.read_case(path, digest), case)
        ready = DRIVER.preflight(case, path, digest)
        self.assertTrue(ready["official_catalog_row_verified"])
        self.assertFalse(ready["model_loaded"])
        self.assertFalse(ready["CUDA_initialized"])
        self.assertEqual(case["protocol"]["continuation_frames"], 96)
        self.assertEqual(4 * (3 * case["protocol"]["chunks"] - 1) + 1, 105)
        self.assertEqual(ready["requested_horizon"]["maximum_past_tokens"], 37440)
        self.assertFalse(ready["requested_horizon"]["native_32760_token_preallocation_used"])
        for field, value in (
            ("prefix_inputs", {"latents": "old-pilot.pt"}),
            ("conditioning_path", "old-pilot.pt"),
            ("seed", True),
        ):
            changed = copy.deepcopy(case)
            changed[field] = value
            with self.assertRaises(ValueError):
                DRIVER.validate_case(changed)
        changed = copy.deepcopy(case)
        changed["protocol"]["codec"]["normalization"] = "global"
        changed["protocol_sha256"] = DRIVER.identity(changed["protocol"])
        with self.assertRaisesRegex(ValueError, "fixed"):
            DRIVER.validate_case(changed)
        changed = copy.deepcopy(case)
        changed["protocol"]["continuation_frames"] = 95
        changed["protocol_sha256"] = DRIVER.identity(changed["protocol"])
        with self.assertRaisesRegex(ValueError, "12 RGB"):
            DRIVER.validate_case(changed)
        changed = copy.deepcopy(case)
        changed["protocol"]["attention_budget_bytes"] = 1024
        changed["protocol_sha256"] = DRIVER.identity(changed["protocol"])
        with self.assertRaisesRegex(ValueError, "Attention budget"):
            DRIVER.validate_case(changed)
        changed = copy.deepcopy(case)
        changed["prompt"]["text"] = "A different fresh-looking label for old inputs"
        bad_path, bad_digest = self.write_case(changed, "bad-prompt.json")
        with self.assertRaisesRegex(ValueError, "exact bound official"):
            DRIVER.preflight(changed, bad_path, bad_digest)
        changed = copy.deepcopy(case)
        del changed["implementation"]["reelcache/reuse.py"]
        bad_path, bad_digest = self.write_case(changed, "incomplete-source.json")
        with self.assertRaisesRegex(ValueError, "exactly"):
            DRIVER.preflight(changed, bad_path, bad_digest)
        path.write_text("{}")
        with self.assertRaisesRegex(ValueError, "immutable"):
            DRIVER.read_case(path, digest)
        self.assertFalse(torch.cuda.is_initialized())

    def test_model_free_import_help_preflight_and_explicit_grant(self):
        case = self.case()
        path, digest = self.write_case(case)
        output = self.directory / "preflight.json"
        code = f"""
import importlib.util, sys
spec = importlib.util.spec_from_file_location('fresh_driver', {str(DRIVER_PATH)!r})
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
assert 'torch' not in sys.modules
module.main(['prepare', '--case-manifest', {str(path)!r}, '--case-sha256', {digest!r},
             '--output', {str(output)!r}])
assert 'torch' not in sys.modules
assert not any(name.startswith('third_party.causvid') for name in sys.modules)
"""
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1",
                   OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                   TMPDIR=str(self.directory))
        subprocess.run([sys.executable, "-B", "-c", code], cwd=ROOT, env=env,
                       check=True, capture_output=True, timeout=120)
        self.assertFalse(json.loads(output.read_text())["model_loaded"])
        help_result = subprocess.run(
            [sys.executable, "-B", str(DRIVER_PATH), "--help"],
            cwd=ROOT, env=env, check=True, capture_output=True, text=True, timeout=120,
        )
        self.assertIn("prepare", help_result.stdout)
        with self.assertRaisesRegex(ValueError, "separate current coordinator"):
            DRIVER.main([
                "run", "--case-manifest", str(path), "--case-sha256", digest,
                "--run-id", "no-grant", "--work-root", str(self.directory),
            ])
        self.assertFalse((self.directory / "no-grant").exists())
        self.assertFalse(torch.cuda.is_initialized())

    def test_storage_identity_is_observed_not_inferred_and_source_changes_fail(self):
        with patch.object(DRIVER.subprocess, "run", return_value=SimpleNamespace(
            stdout="/dev/nvme0n1p1 ext4 /allowed-local\n",
        )):
            value = DRIVER.storage_placement(self.directory)
            self.assertEqual(value["source"], "/dev/nvme0n1p1")
            self.assertEqual(value["root"], str(self.directory.resolve()))
            self.assertEqual(value["owner_uid"], os.geteuid())
        for observed in ("remote:/pool nfs4 /data\n", "tmpfs tmpfs /dev/shm\n"):
            with patch.object(DRIVER.subprocess, "run", return_value=SimpleNamespace(stdout=observed)):
                with self.assertRaisesRegex(ValueError, "local NVMe"):
                    DRIVER.storage_placement(self.directory)
        source = self.directory / "source.py"
        source.write_text("value = 1\n")
        provenance = {"source_files": {"fake": {"path": str(source), "sha256": DRIVER.sha256(source)}}}
        DRIVER.verify_sources(provenance)
        source.write_text("value = 2\n")
        with self.assertRaisesRegex(RuntimeError, "changed"):
            DRIVER.verify_sources(provenance)

    def test_run_failure_records_case_identity_without_model_or_gpu(self):
        case = self.case()
        path, digest = self.write_case(case)
        work = self.directory / "active"
        work.mkdir(mode=0o700)

        def configure(directory, *, gpu, threads):
            os.environ.update(
                CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                TMPDIR=str(self.directory), TRITON_CACHE_DIR=str(self.directory / "triton"),
            )

        builder = SimpleNamespace(environment=configure)
        with patch.dict(os.environ), patch.object(
            DRIVER, "storage_placement", return_value={"root": str(work), "CPU_test_mock": True},
        ), patch.object(DRIVER, "native_driver", return_value=builder), patch.object(
            DRIVER, "run", side_effect=RuntimeError("injected pre-model native failure"),
        ):
            with self.assertRaisesRegex(RuntimeError, "injected pre-model"):
                DRIVER.main([
                    "run", "--case-manifest", str(path), "--case-sha256", digest,
                    "--work-root", str(work), "--run-id", "failed-case", "--allow-cuda",
                ])
        directory = work / "failed-case"
        failure = json.loads((directory / "failure.json").read_text())
        execution = json.loads((directory / "execution.json").read_text())
        self.assertEqual(failure["case_id"], case["case_id"])
        self.assertEqual(failure["case_sha256"], digest)
        self.assertTrue(failure["partial_outputs_are_not_completed_evidence"])
        self.assertEqual(execution["environment"]["CUDA_VISIBLE_DEVICES"], "")
        self.assertEqual((directory / "case.json").read_bytes(), path.read_bytes())
        self.assertFalse((directory / "result.json").exists())
        self.assertFalse(torch.cuda.is_initialized())

    def test_fresh_text_noise_and_source_dtype_are_not_relabelled(self):
        pipeline = self.pipeline()
        prior = torch.get_rng_state().clone()
        noise, conditioning, rng = fresh_inputs(pipeline, "one actual prompt", 0, CONFIG, 3)
        other_noise, other_conditioning, other_rng = fresh_inputs(pipeline, "another actual prompt", 0, CONFIG, 3)
        self.assertEqual(pipeline.text_encoder.prompts, [["one actual prompt"], ["another actual prompt"]])
        self.assertTrue(torch.equal(noise, other_noise))
        self.assertTrue(torch.equal(rng.get_state(), other_rng.get_state()))
        self.assertFalse(torch.equal(conditioning["prompt_embeds"], other_conditioning["prompt_embeds"]))
        self.assertTrue(torch.equal(prior, torch.get_rng_state()))
        with patch.object(pipeline.text_encoder, "forward", return_value={
            "prompt_embeds": torch.zeros(1, 3, 8, dtype=torch.float64),
        }):
            with self.assertRaisesRegex(IntegrityError, "source dtype"):
                fresh_inputs(pipeline, "same actual prompt", 0, CONFIG, 3)

    def test_shared_basis_bytes_include_snapshot_state_and_prefix_once(self):
        pipeline = self.pipeline()
        noise, conditioning, rng = initial_state()
        with KVRuntime(self.directory / "runtime", self.model()) as runtime:
            with CausVidAdapter(
                pipeline, runtime, "parent", config=CONFIG, specs=CodecSpec("svd", rank=3),
            ) as adapter:
                adapter.start(noise, conditioning, rng, conditioning_id="fake-prompt")
                adapter.step()
                snapshot = adapter.snapshot()
                with patch.object(runtime.store, "references", wraps=runtime.store.references) as references:
                    measured = checkpoint_bytes(runtime, "parent", snapshot, CONFIG)
                self.assertTrue(all(call.args[0].startswith("manifest:") for call in references.call_args_list))
                expected = CONFIG.chunk_tokens * CONFIG.channels * 2 * 2 * 4
                self.assertEqual(measured["native_dense_KV_bytes"], expected)
                self.assertGreater(measured["decoder"]["full_canonical_basis_bytes"], 0)
                self.assertGreater(measured["snapshot_execution_and_root_extra_bytes"], 0)
                self.assertGreater(measured["snapshot"]["unique_file_bytes"], measured["decoder"]["unique_file_bytes"])
                repeated = object_graph(runtime.store, [snapshot, snapshot])
                # NFS can update st_blocks after publication without changing file bytes.
                physical = "allocated_filesystem_bytes"
                self.assertEqual(
                    {key: value for key, value in repeated.items() if key != physical},
                    {key: value for key, value in measured["snapshot"].items() if key != physical},
                )
                for observation in (repeated, measured["snapshot"]):
                    self.assertIsInstance(observation[physical], int)
                    self.assertGreaterEqual(observation[physical], 0)
                    self.assertEqual(observation[physical] % 512, 0)
                with self.assertRaisesRegex(IntegrityError, "different clean heads"):
                    adapter.step()
                    checkpoint_bytes(runtime, "parent", snapshot, CONFIG)

    def test_object_graph_counts_current_allocations_once_without_caching_them(self):
        sizes = {
            "manifest:root": 8, "manifest:left": 8, "manifest:right": 8,
            "basis:shared": 4, "blob:data": 6,
        }
        edges = {
            "manifest:root": ["manifest:left", "manifest:right"],
            "manifest:left": ["basis:shared", "blob:data"],
            "manifest:right": ["basis:shared", "blob:data"],
        }
        reads = []
        allocated_blocks = 1

        def path_for(object_id):
            def stat():
                reads.append(object_id)
                return SimpleNamespace(st_size=sizes[object_id], st_blocks=allocated_blocks)
            return SimpleNamespace(stat=stat)

        store = SimpleNamespace(path_for=path_for, references=edges.__getitem__)
        first = object_graph(store, ["manifest:root", "manifest:root"])
        self.assertEqual(len(reads), len(sizes))
        self.assertEqual(set(reads), set(sizes))
        self.assertEqual(first["unique_file_bytes"], sum(sizes.values()))
        self.assertEqual(first["full_canonical_basis_bytes"], 4)
        self.assertEqual(first["allocated_filesystem_bytes"], len(sizes) * 512)
        reads.clear()
        allocated_blocks = 2
        second = object_graph(store, ["manifest:root"])
        self.assertEqual(len(reads), len(sizes))
        self.assertEqual(second["unique_file_bytes"], first["unique_file_bytes"])
        self.assertEqual(second["allocated_filesystem_bytes"], 2 * first["allocated_filesystem_bytes"])

    def test_fresh_closed_loop_generation_replay_and_selected_export(self):
        pipeline = self.pipeline()
        noise, conditioning, rng = fresh_inputs(pipeline, "actual CPU fake prompt", 0, CONFIG, 3)
        prior_global_rng = torch.get_rng_state().clone()
        events = []
        with KVRuntime(self.directory / "runtime", self.model()) as runtime:
            result = generate_matched_case(
                pipeline, runtime, config=CONFIG, noise=noise, conditioning=conditioning, rng=rng,
                conditioning_id="fresh-case-fixture", arms=["raw", "svd", "svd_hevc"], directory=self.directory,
                hot_history_bytes=4096, replay_chunks=1, fork_seed=1, lossy_ranks={"K": 3, "V": 4},
                progress=lambda phase, data: events.append((phase, data)),
            )
            result.update(schema="reelcache.causvid.vbench.result.v1", model_identity=self.model().to_dict())
            self.assertEqual(result["status"], "passed")
            self.assertEqual(result["continuation_frames"], 24)
            self.assertFalse(result["common_prefix"]["clean_audit"]["supplied_clean_prefix"])
            self.assertEqual(result["common_prefix"]["clean_audit"]["denoising_steps"], 2)
            self.assertTrue(result["checks"]["uncached_RAW_replay_latents_and_rng_equal"])
            self.assertTrue(result["checks"]["uncached_RAW_replay_RGB_equal"])
            self.assertTrue(result["checks"]["changed_seed_fork_latents_RGB_rng_replay_equal"])
            self.assertTrue(result["checks"]["parent_and_common_prefix_unchanged"])
            self.assertEqual(result["arms"]["raw"]["paired_RAW_ssim"]["mean"], 1.0)
            self.assertIn("paired_SVD_ssim", result["arms"]["svd_hevc"])
            self.assertTrue(torch.equal(prior_global_rng, torch.get_rng_state()))
            for arm in result["arms"].values():
                self.assertEqual(arm["closed_loop_clean_chunks"], 3)
                self.assertEqual([row["chunk_id"] for row in arm["chunks"]], [0, 1, 2])
                self.assertEqual(arm["outputs"]["continuation_shape"], [24, 8, 12, 3])
                self.assertTrue(all(row["owned_memory"]["live_hot_history_bytes"] == 0 for row in arm["chunks"]))
                self.assertTrue(all(row["runtime_residency"]["active_leases"] == 0 for row in arm["chunks"]))
            self.assertTrue(events)
            self.assertTrue(result["manifest_ledger"]["every_reachable_object_hash_verified"])
            ledger_rows = [json.loads(line) for line in Path(result["manifest_ledger"]["path"]).read_text().splitlines()]
            self.assertEqual(len(ledger_rows), len({row["object_id"] for row in ledger_rows}))
            self.assertTrue(any(row["kind"] == "basis" for row in ledger_rows))
            self.assertTrue(all("manifest" in row or row["kind"] != "manifest" for row in ledger_rows))
            self.assertTrue(all(set(row) == {"object_id", "kind", "bytes"} for row in ledger_rows if row["kind"] != "manifest"))
        physical = filesystem_bytes(self.directory / "runtime")
        self.assertGreater(physical["groups"]["index"]["file_bytes"], 0)
        self.assertEqual(physical["file_bytes"], sum(row["file_bytes"] for row in physical["groups"].values()))
        result_path = self.directory / "result.json"
        result_path.write_text(json.dumps(result))
        destination = self.directory / "compact"
        records = DRIVER.compact_export(result_path, destination, max_bytes=2 * 1024**2)
        self.assertTrue(records)
        self.assertFalse((destination / "runtime").exists())
        self.assertFalse((destination / "selected_checkpoints").exists())
        self.assertTrue((destination / "manifest_ledger.jsonl").is_file())
        with self.assertRaises(FileExistsError):
            DRIVER.compact_export(result_path, destination, max_bytes=2 * 1024**2)
        checkpoint_export = self.directory / "selected"
        DRIVER.compact_export(result_path, checkpoint_export, max_bytes=4 * 1024**2,
                              checkpoints=["svd_hevc:prefix"])
        with KVRuntime(checkpoint_export / "selected_checkpoints", self.model()) as reopened:
            snapshot = result["arms"]["svd_hevc"]["prefix_snapshot"]
            saved = reopened.load_snapshot(snapshot)
            self.assertEqual(saved.snapshot_id, snapshot)
            self.assertEqual(reopened.store.list_refs("branches/"), {})
            self.assertEqual(len(reopened.store.list_refs("snapshots/")), 1)
        self.assertFalse(torch.cuda.is_initialized())

    def test_uint8_ssim_lossless_evaluator_video_and_export_budget(self):
        import imageio_ffmpeg

        frames = np.arange(12 * 8 * 12 * 3, dtype=np.uint8).reshape(12, 8, 12, 3)
        with self.assertRaisesRegex(IntegrityError, "finite native VAE"):
            rgb_uint8(torch.full((1, 12, 3, 8, 12), float("nan")))
        self.assertEqual(paired_ssim(frames, frames)["mean"], 1.0)
        for wrong in (frames.astype(np.float32), frames[:-1]):
            with self.assertRaises(IntegrityError):
                paired_ssim(frames, wrong)
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        video = self.directory / "lossless.mp4"
        report = encode_lossless_video(frames, video, ffmpeg=ffmpeg, fps=24)
        self.assertTrue(report["lossless_RGB_roundtrip_verified"])
        outputs = save_evaluator_output(
            self.directory / "sample", torch.zeros(1, 6, 8, 2, 4), frames,
            torch.Generator().get_state(), prefix_frames=0, fps=24,
        )
        path = self.directory / "result.json"
        path.write_text(json.dumps({
            "schema": "reelcache.causvid.vbench.result.v1", "status": "passed", "arms": {"raw": {"outputs": outputs}},
        }))
        with self.assertRaisesRegex(ValueError, "budget"):
            DRIVER.compact_export(path, self.directory / "too-large", max_bytes=1)
        self.assertFalse((self.directory / "too-large").exists())
        rgb = Path(outputs["RGB"]["path"])
        rgb.write_bytes(b"corrupted output")
        with self.assertRaisesRegex(ValueError, "hash or length"):
            DRIVER.compact_export(path, self.directory / "corrupt", max_bytes=1024**2)
        self.assertFalse((self.directory / "corrupt").exists())


if __name__ == "__main__":
    unittest.main()
