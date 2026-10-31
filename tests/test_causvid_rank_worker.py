"""Bounded CPU fakes; none of these tests establishes native model/video parity."""

from __future__ import annotations

from contextlib import contextmanager
import copy
from dataclasses import replace
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
import weakref

import torch

from examples import causvid_rank_worker as worker
from examples import causvid_native_reference as reference
from reelcache.adapters.causvid import CausVidAdapter, CausVidCleanStaging, CausVidConfig, tensor_digest
from reelcache.errors import CapacityError, ConfigurationError, IntegrityError, StateError, UnsupportedError
from reelcache.study import ASSET_SCHEMA, PROTOCOL_SCHEMA, CELLS, VBENCH_COMMIT, VBENCH_PROMPTS_SHA256, identity, file_sha256
from reelcache.types import CodecSpec, MatrixKey, ModelIdentity, TensorContext
from tests.test_causvid_adapter import CONFIG, FakePipeline, causal_rope_apply, initial_state


ROOT = Path(__file__).resolve().parents[1]


class MemoryRuntime:
    """Fake packet store with distinct pristine inputs and committed decodes."""

    instances = []

    def __init__(self, root, model, **_kwargs):
        self.root, self.model = Path(root), model
        self.branches, self.packets, self.sources = {}, {}, {}
        self.active_leases = self.resident_bytes = 0
        self.reads, self.lease_refs = [], []
        self.closed, self.fail_target = False, None
        self.instances.append(self)

    def create_branch(self, branch):
        if branch in self.branches:
            raise StateError("duplicate fake branch")
        self.branches[branch] = {"head_id": "manifest:" + "0" * 64, "token_end": 0, "chunks": 0}

    def branch_info(self, branch):
        return dict(self.branches[branch])

    def seal_chunk(self, branch, chunk, matrices, contexts, specs, **_kwargs):
        records = {}
        for key in contexts:
            value = matrices[key].detach().clone()
            self.sources[(branch, chunk, key)] = value
            records[key] = value.clone() if specs[key].kind == "raw" else value * 0.5
        for key, value in records.items():
            self.packets[(branch, chunk, key)] = value
        count = self.branches[branch]["chunks"] + 1
        self.branches[branch] = {
            "head_id": "manifest:" + f"{count:064x}", "token_end": next(iter(contexts.values())).token_end,
            "chunks": count,
        }
        return "manifest:" + f"{count + 100:064x}"

    @contextmanager
    def acquire_history(self, branch, key, *, token_start=0, token_end=None, device="cpu", dtype=None, **_kwargs):
        if self.fail_target == key.target:
            raise CapacityError("injected committed packet reconstruction failure")
        self.reads.append((branch, key, token_start, token_end))
        tensors = [
            self.packets[(branch, chunk, key)] for chunk in range(self.branches[branch]["chunks"])
        ]
        source = torch.cat(tensors, dim=0)[token_start:token_end].to(device=device, dtype=dtype).clone()
        self.active_leases += 1
        self.resident_bytes += source.numel() * source.element_size()
        self.lease_refs.append(weakref.ref(source))
        try:
            yield SimpleNamespace(tensor=source)
        finally:
            self.active_leases -= 1
            self.resident_bytes -= source.numel() * source.element_size()

    def close(self):
        if self.active_leases or self.resident_bytes:
            raise StateError("fake runtime closed with an active lease")
        self.closed = True


class OriginalPipelineFixture(FakePipeline):
    """Independent test-only native entrypoint; never imported native source."""

    def __init__(self, template, text_encoder):
        self.generator, self.scheduler = copy.deepcopy(template.generator), copy.deepcopy(template.scheduler)
        self.text_encoder, self.vae = text_encoder, None
        self.denoising_step_list = template.denoising_step_list.clone()
        self.num_transformer_blocks = template.num_transformer_blocks
        self.num_frame_per_block, self.frame_seq_length = template.num_frame_per_block, template.frame_seq_length
        self.kv_cache1 = self.kv_cache2 = self.crossattn_cache = None
        self.inference_calls, self.fail_forward = [], False

    def _initialize_kv_cache(self, batch_size, dtype, device):
        cfg = self.generator.config
        self.kv_cache1 = [
            {key: torch.zeros(batch_size, 21 * cfg.frame_tokens, cfg.heads, cfg.head_dim, dtype=dtype, device=device)
             for key in ("k", "v")} for _ in range(self.num_transformer_blocks)
        ]

    def _initialize_crossattn_cache(self, batch_size, dtype, device):
        cfg = self.generator.config
        self.crossattn_cache = [
            {**{key: torch.zeros(batch_size, 512, cfg.heads, cfg.head_dim, dtype=dtype, device=device)
                for key in ("k", "v")}, "is_init": False}
            for _ in range(self.num_transformer_blocks)
        ]

    def inference(self, noise, text_prompts, start_latents=None, return_latents=False, rng=None):
        cfg = self.generator.config
        self.inference_calls.append({
            "noise": noise.clone(), "text_prompts": list(text_prompts), "start_latents": start_latents,
            "return_latents": return_latents, "rng_state_at_entry": rng.get_state().clone(),
        })
        condition = self.text_encoder(text_prompts=text_prompts)
        self._initialize_kv_cache(1, noise.dtype, noise.device)
        self._initialize_crossattn_cache(1, noise.dtype, noise.device)
        if self.fail_forward:
            raise RuntimeError("fixture native inference failed before VAE")
        output = torch.zeros_like(noise)
        for start in range(0, noise.shape[1], cfg.frames_per_chunk):
            end = start + cfg.frames_per_chunk

            def forward(value, step):
                return self.generator(
                    noisy_image_or_video=value, conditional_dict=condition,
                    timestep=torch.full((1, cfg.frames_per_chunk), step, dtype=torch.int64),
                    kv_cache=self.kv_cache1, crossattn_cache=self.crossattn_cache,
                    current_start=start * cfg.frame_tokens, current_end=end * cfg.frame_tokens,
                )

            value = noise[:, start:end]
            if start_latents is not None and end <= start_latents.shape[1]:
                clean = start_latents[:, start:end]
            else:
                for index, step in enumerate(self.denoising_step_list):
                    clean = forward(value, step.item())
                    if index + 1 < len(self.denoising_step_list):
                        value = self.scheduler.add_noise(
                            clean.flatten(0, 1), torch.empty_like(clean.flatten(0, 1)).normal_(generator=rng),
                            self.denoising_step_list[index + 1] * torch.ones(cfg.frames_per_chunk, dtype=torch.long),
                        ).unflatten(0, clean.shape[:2])
            output[:, start:end] = clean
            forward(clean, 0)
        pixels = self.vae.decode_to_pixel(output)
        return (pixels, output) if return_latents else pixels


class CausVidRankWorkerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.directory = ROOT / ".test-work" / f"causvid-rank-{uuid.uuid4().hex}"
        self.directory.mkdir(parents=True)
        self.addCleanup(shutil.rmtree, self.directory)
        torch.manual_seed(301)
        self.pipe = FakePipeline()
        self.pipe.vae = None
        self.pipe.generator.model.blocks = torch.nn.ModuleList([self.pipe.generator.model.blocks[0]])
        self.pipe.generator.model.num_layers = self.pipe.num_transformer_blocks = 1
        self.model = ModelIdentity("causvid-cpu-fake", "not-native-evidence", 1, 8, "tokens_channels", "float32")
        self.noise, self.conditioning, self.rng = initial_state()
        self.asset_id = "a" * 64
        self.texts = []
        MemoryRuntime.instances = []

    def budgets(self):
        return {
            "device_bytes": 32 << 30, "model_device_bytes": 8 << 30, "model_workspace_bytes": 2 << 30,
            "host_bytes": 64 << 30, "model_host_bytes": 16 << 30,
            "resident_bytes": 2 << 30, "attention_bytes": 2 << 30,
            "current_bytes": 1 << 30, "clean_host_bytes": 1 << 30, "clean_disk_bytes": 0,
            "quality_history_bytes": 0, "store_bytes": 8 << 30, "spectrum_workspace_bytes": 0,
            "source_archive_bytes": 0,
        }

    def request(self, name="a", *, text="A red fox runs beside a river.", seed=0, chunks=1, mode="reload"):
        budgets = self.budgets()
        if mode == "quality_hot":
            budgets["quality_history_bytes"] = chunks * CELLS["causvid480"].native_kv_bytes
        return {
            "schema": worker.REQUEST_SCHEMA, "request_id": name, "cell_id": "causvid480",
            "case_id": "case-" + identity({"text": text, "seed": seed})[:16],
            "trajectory_id": "trajectory-" + name, "arm_id": "raw", "split": "calibration",
            "prompt_group_id": "g-fox", "text": text, "seed": seed, "chunks": chunks,
            "conditioning_identity": worker.conditioning_identity(text, self.asset_id),
            "history_mode": mode, "spec_config": {"K": {"kind": "raw"}, "V": {"kind": "raw"}},
            "rank_policy": None, "memory_budgets": budgets, "output_namespace": name,
            "disk_budget": {"disk_bytes": 12 << 30, "disk_reserve_bytes": 1 << 30},
        }

    def encode_text(self, *, text_prompts):
        self.texts.append(tuple(text_prompts))
        value = int(hashlib.sha256(text_prompts[0].encode()).hexdigest()[:4], 16) / 65536
        return {"prompt_embeds": torch.full((1, 3, 8), value)}

    def shared_request(self):
        limits = self.budgets()
        return {
            "schema": "reelcache.rank_request.v1", "request_id": "c" * 64, "cell_id": "causvid480",
            "case_id": "explicit-case", "trajectory_id": "explicit-trajectory", "arm_id": "raw",
            "group_id": "g-fox", "prompt_id": "cal00", "prompt": self.request()["text"],
            "seed": 0, "chunks": 3, "split": "calibration", "history_mode": "reload",
            "source_fps": 16, "output_namespace": "controller-case", "representation": {"kind": "raw"},
            "budgets": {
                "device_bytes_per_rank": limits["device_bytes"], "history_bytes_per_rank": 2 << 30,
                "workspace_bytes_per_rank": 6 << 30, "host_bytes": limits["host_bytes"],
                "source_staging_bytes_global": limits["clean_host_bytes"],
                "disk_bytes": 12 << 30, "disk_reserve_bytes": 1 << 30,
            },
        }

    def engine(self, **kwargs):
        result = worker.CausVidRankWorker(
            self.pipe, self.model, config=CONFIG, output_root=self.directory,
            asset_id=self.asset_id, fps=16, conditioning_fn=self.encode_text, **kwargs,
        )
        self.addCleanup(result.close)
        return result

    def adapter(self, name="request", **kwargs):
        runtime = MemoryRuntime(self.directory / name, self.model)
        adapter = CausVidAdapter(self.pipe, runtime, name, config=CONFIG, **kwargs).install()
        self.addCleanup(adapter.close)
        adapter.start(self.noise, self.conditioning, self.rng, conditioning_id="cpu-fake")
        return adapter, runtime

    def reference_fixture(self, name="reference", *, raw_name="reference-raw", text=None, chunks=2):
        self.pipe.denoising_step_list = torch.tensor([1000, 757, 522])
        raw_document = self.request(raw_name, chunks=chunks, **({} if text is None else {"text": text}))
        raw_document.update(split="development", study_phase="native_preflight")
        raw_engine = self.engine()
        with patch("reelcache.manager.KVRuntime", MemoryRuntime):
            raw_result = raw_engine.run(raw_document)
        raw_engine.close()
        document = copy.deepcopy(raw_document)
        document.update(
            request_id=name, trajectory_id="trajectory-" + name, output_namespace=name, arm_id="native-reference",
            reference_source={
                "path": raw_name + "/result.json", "sha256": file_sha256(self.directory / raw_name / "result.json"),
            },
        )
        document["memory_budgets"]["resident_bytes"] = worker.NATIVE_DENSE_KV_BYTES + worker.NATIVE_CROSSATTN_BYTES
        pipeline = OriginalPipelineFixture(self.pipe, self.encode_text)
        engine = reference.CausVidNativeReferenceWorker(
            pipeline, self.model, config=CONFIG, output_root=self.directory, asset_id=self.asset_id,
            fps=16, observation_evidence_kind="synthetic_fixture",
        )
        self.addCleanup(engine.close)
        return engine, document, raw_result

    def test_native_reference_calls_original_from_scratch_with_postdraw_rng_and_no_adapter(self):
        engine, document, raw = self.reference_fixture()
        document["capture"] = {"retain": [{"chunk_id": 0, "layer": 0, "target": "K"}]}
        document["memory_budgets"]["source_archive_bytes"] = 4680 * 1536 * 2 + 65536
        before = torch.get_rng_state().clone()
        with patch("reelcache.adapters.causvid.CausVidAdapter", side_effect=AssertionError("adapter forbidden")), patch(
            "reelcache.manager.KVRuntime", side_effect=AssertionError("store forbidden"),
        ):
            result = engine.run(document)
        self.assertEqual(len(engine.pipeline.inference_calls), 1)
        call = engine.pipeline.inference_calls[0]
        self.assertIsNone(call["start_latents"])
        self.assertTrue(call["return_latents"])
        self.assertEqual(call["text_prompts"], [document["text"]])
        self.assertEqual(tensor_digest(call["rng_state_at_entry"]), raw["denoising_rng_sha256"])
        self.assertNotEqual(raw["initial_rng_sha256"], raw["denoising_rng_sha256"])
        self.assertEqual(tensor_digest(call["noise"]), raw["initial_noise_sha256"])
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertTrue(result["raw_comparison"]["latents_bitwise_equal"])
        self.assertTrue(result["raw_comparison"]["final_rng_bitwise_equal"])
        self.assertFalse(result["raw_comparison"]["rgb_compared"])
        self.assertFalse(result["native_parity_claim"])
        self.assertTrue(result["fixture"])
        self.assertFalse(result["native_execution"])
        self.assertFalse(result["vae_loaded"])
        self.assertFalse(result["rgb"]["decoded"])
        self.assertEqual(result["rgb"]["prefix_frame_counts"], [9, 21])
        self.assertFalse((self.directory / "reference" / "rgb.npy").exists())
        self.assertFalse((self.directory / "reference" / "runtime").exists())
        self.assertIsNone(engine.pipeline.vae)
        self.assertTrue(worker._native_cache_empty(engine.pipeline))
        self.assertEqual(worker._hook_signature(engine.pipeline), engine._hooks)
        self.assertEqual(
            [(row["token_start"], row["timestep"]) for row in result["native_audit"]["calls"]],
            [(start, step) for start in (0, CONFIG.chunk_tokens) for step in (1000, 757, 522, 0)],
        )
        self.assertTrue(result["native_audit"]["first_chunk_denoised_from_scratch"])
        self.assertEqual(sum("original" in row for row in result["clean_captures"]), 1)
        self.assertEqual(len(result["clean_captures"]), 4)
        packets = MemoryRuntime.instances[-1].packets
        for chunk in range(2):
            hashes = result["native_audit"]["clean_segment_hashes"][chunk * CONFIG.chunk_tokens]
            self.assertEqual(hashes[0]["post_rope_K"], tensor_digest(packets["request", chunk, MatrixKey(0, "K")]))
            self.assertEqual(hashes[0]["V"], tensor_digest(packets["request", chunk, MatrixKey(0, "V")]))
        self.assertEqual(
            result["memory"]["native_kv_bytes"], 2 * 21 * CONFIG.frame_tokens * CONFIG.channels * 4,
        )

    def test_native_reference_rejects_wrong_phase_codec_and_dense_budget_before_mutation(self):
        engine, document, _ = self.reference_fixture()
        for name in ("phase", "split", "codec", "census", "resident", "prefix"):
            invalid = copy.deepcopy(document)
            if name == "phase":
                invalid["study_phase"] = "calibration_frontier"
            elif name == "split":
                invalid["split"] = "heldout"
            elif name == "codec":
                invalid["spec_config"]["K"] = {"kind": "svd", "rank": 1}
            elif name == "census":
                invalid["capture"] = {"spectra": True}
                invalid["memory_budgets"]["spectrum_workspace_bytes"] = 1 << 30
            elif name == "resident":
                invalid["memory_budgets"]["resident_bytes"] -= 1
            else:
                invalid["latent_source"] = invalid["reference_source"]
            before = torch.get_rng_state().clone()
            with self.subTest(name=name), self.assertRaises((CapacityError, ConfigurationError)):
                engine.run(invalid)
            self.assertEqual(engine.pipeline.inference_calls, [])
            self.assertFalse((self.directory / document["output_namespace"]).exists())
            self.assertTrue(torch.equal(before, torch.get_rng_state()))
            self.assertTrue(worker._native_cache_empty(engine.pipeline))
        result = engine.run(document)
        self.assertTrue(result["request_reset_verified"])

    def test_native_reference_conditioning_mismatch_and_real_errors_abort_and_remove_boundary(self):
        for name in ("conditioning", "inference"):
            with self.subTest(name=name):
                engine, document, _ = self.reference_fixture(name, raw_name=name + "-raw")
                if name == "conditioning":
                    engine.pipeline.text_encoder = lambda **_: {"prompt_embeds": torch.zeros(1, 3, 8)}
                    expected, message = IntegrityError, "conditioning"
                else:
                    engine.pipeline.fail_forward = True
                    expected, message = RuntimeError, "fixture native inference failed"
                with self.assertRaisesRegex(expected, message):
                    engine.run(document)
                self.assertTrue(engine._poisoned)
                self.assertTrue(worker._native_cache_empty(engine.pipeline))
                self.assertIsNone(engine.pipeline.vae)
                self.assertEqual(worker._hook_signature(engine.pipeline), engine._hooks)
                self.assertFalse((self.directory / name / "result.json").exists())
                self.assertTrue((self.directory / name / "failure.json").is_file())
                with self.assertRaisesRegex(StateError, "abort_epoch"):
                    engine.run(document)

    def test_native_reference_checks_seed_draw_even_for_rehashed_state(self):
        from reelcache.serialization import deserialize_tensors, serialize_tensors

        engine, document, raw = self.reference_fixture()
        path = self.directory / raw["latent_file"]["path"]
        state = deserialize_tensors(path.read_bytes())
        state["denoising_rng_state"] = state["initial_rng_state"].clone()
        path.write_bytes(serialize_tensors(state))
        raw["denoising_rng_sha256"] = tensor_digest(state["denoising_rng_state"])
        raw["latent_file"] = worker._file_record(path, self.directory)
        result_path = self.directory / document["reference_source"]["path"]
        result_path.write_bytes(worker.canonical_json(raw) + b"\n")
        document["reference_source"]["sha256"] = file_sha256(result_path)
        with self.assertRaisesRegex(IntegrityError, "complete upfront draw"):
            engine.run(document)
        self.assertEqual(engine.pipeline.inference_calls, [])
        self.assertIsNone(engine.pipeline.vae)
        self.assertTrue(engine._poisoned)
        self.assertFalse((self.directory / "reference/result.json").exists())

    def test_native_reference_partial_audit_installation_removes_every_hook(self):
        engine, document, _ = self.reference_fixture()
        norm = engine.pipeline.generator.model.blocks[0].self_attn.norm_k
        with patch.object(norm, "register_forward_hook", side_effect=RuntimeError("fixture hook install failed")):
            with self.assertRaisesRegex(RuntimeError, "hook install failed"):
                engine.run(document)
        self.assertTrue(engine._poisoned)
        self.assertEqual(engine.pipeline.inference_calls, [])
        self.assertEqual(worker._hook_signature(engine.pipeline), engine._hooks)
        self.assertIsNone(engine.pipeline.vae)
        self.assertTrue(worker._native_cache_empty(engine.pipeline))

    def test_native_reference_a_b_a_reuses_models_not_conditioning_rng_or_caches(self):
        engine, document, raw_a = self.reference_fixture()
        _, b_document, raw_b = self.reference_fixture("ref-b", raw_name="raw-b", text="A blue bird flies over snow.")
        repeated = copy.deepcopy(document)
        repeated.update(request_id="ref-a-again", trajectory_id="ref-a-again", output_namespace="ref-a-again")
        parameter_pointers = worker._parameter_signature(engine.pipeline)
        first, middle, last = [engine.run(item) for item in (document, b_document, repeated)]
        self.assertEqual(first["latents_sha256"], last["latents_sha256"])
        self.assertNotEqual(first["latents_sha256"], middle["latents_sha256"])
        self.assertEqual(first["latents_sha256"], raw_a["latents_sha256"])
        self.assertEqual(middle["latents_sha256"], raw_b["latents_sha256"])
        self.assertEqual(parameter_pointers, worker._parameter_signature(engine.pipeline))
        self.assertEqual(len(engine.pipeline.inference_calls), 3)
        self.assertTrue(all(item["request_reset_verified"] for item in (first, middle, last)))

    def test_latents_only_boundary_is_instance_local_and_never_fabricates_pixels(self):
        first, second = torch.nn.Module(), torch.nn.Module()
        first.vae = second.vae = None
        modules_before = tuple(first._modules)
        with reference._latent_boundary(first) as boundary:
            self.assertIsNone(second.vae)
            with self.assertRaises(reference._LatentExport) as caught:
                first.vae.decode_to_pixel(torch.zeros(1))
            self.assertIs(caught.exception.boundary, boundary)
        self.assertIsNone(first.vae)
        self.assertEqual(tuple(first._modules), modules_before)
        del first.vae
        first.add_module("vae", None)
        with reference._latent_boundary(first):
            pass
        self.assertIn("vae", first._modules)
        self.assertIsNone(first.vae)

    def test_native_reference_gate_runs_before_native_epoch_and_helper_import_is_model_free(self):
        engine, document, _ = self.reference_fixture()
        invalid = copy.deepcopy(document)
        invalid["split"] = "heldout"
        request = worker.CausVidRankRequest(worker.canonical_json(invalid))
        arguments = [
            "--protocol", "unused", "--assets", "unused", "--asset-map", "causvid480.json",
            "--launch-receipt", "unused", "--requests", "unused", "--output-root", str(self.directory),
            "--operation", "native-reference", "--allow-cuda",
        ]
        with patch.object(worker, "verify_worker_inputs", return_value=(
            {"protocol_id": "b" * 64}, {"source_closure": {"closure_id": "c" * 64}},
            SimpleNamespace(asset_id=self.asset_id, root=self.directory / "assets", paths={}), [request],
        )), patch.object(worker, "NativeCausVidEpoch") as factory:
            with self.assertRaisesRegex(ConfigurationError, "RAW development/native_preflight"):
                worker.main(arguments)
            factory.assert_not_called()
        engine.close()
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1",
                   TMPDIR=str(self.directory), OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
        subprocess.run(
            [sys.executable, "-B", "-c", (
                "import sys; import examples.causvid_native_reference; "
                "assert 'torch' not in sys.modules; "
                "assert not any(k.startswith('third_party.causvid') for k in sys.modules)"
            )], cwd=ROOT, env=env, check=True, capture_output=True,
        )

    def test_reference_comparison_uses_tensors_final_rng_and_rgb_not_result_json(self):
        import numpy as np
        from reelcache.serialization import serialize_tensors

        engine, document, raw = self.reference_fixture(chunks=1)
        native = engine.run(document)
        raw_reference = document["reference_source"]
        native_reference = {
            "path": document["output_namespace"] + "/result.json",
            "sha256": file_sha256(self.directory / document["output_namespace"] / "result.json"),
        }
        decoded_references = []
        for index, (generation, source) in enumerate(((raw, raw_reference), (native, native_reference))):
            directory = self.directory / f"fixture-rgb-{index}"
            directory.mkdir()
            pixels = np.full((9, 480, 832, 3), 127, dtype=np.uint8)
            path = directory / "rgb.npy"
            with path.open("xb") as stream:
                np.save(stream, pixels, allow_pickle=False)
            decoded = {
                "schema": "reelcache.causvid.rank_result.v1", "status": "complete", "operation": "decode",
                "request": generation["request"], "asset_id": generation["asset_id"],
                "latent_source": source, "latents_sha256": generation["latents_sha256"],
                "rgb_file": worker._file_record(path, self.directory),
                "rgb_sha256": hashlib.sha256(pixels.tobytes()).hexdigest(),
                "rgb": {**worker.rgb_windows(1, 16), "decoded": True, "resize_or_crop": False},
                "dit_loaded": False, "vae_cache_reset_verified": True,
                "fixture": True, "native_execution": False, "elapsed_seconds": index + 10,
            }
            result_path = directory / "result.json"
            worker._json_new(result_path, decoded)
            decoded_references.append({"path": str(result_path.relative_to(self.directory)), "sha256": file_sha256(result_path)})
        arguments = {
            "raw_generation": raw_reference, "native_reference": native_reference,
            "raw_decode": decoded_references[0], "native_decode": decoded_references[1], "max_host_bytes": 128 << 20,
        }
        measured = reference.compare_reference_outputs(self.directory, **arguments)
        self.assertTrue(measured["all_compared_outputs_equal"])
        self.assertTrue(measured["comparisons"]["rng_state_bitwise_equal"])
        self.assertTrue(measured["comparisons"]["rgb_bitwise_equal"])
        self.assertFalse(measured["native_execution"])
        self.assertTrue(measured["fixture"])
        self.assertFalse(measured["native_parity_claim"])
        self.assertFalse(measured["compares_result_namespace_or_timing"])
        with self.assertRaises(CapacityError):
            reference.compare_reference_outputs(self.directory, **{**arguments, "max_host_bytes": 1})
        rgb_path = self.directory / "fixture-rgb-1/rgb.npy"
        changed = np.load(rgb_path, allow_pickle=False)
        changed[-1, 0, 0, 0] += 1
        with rgb_path.open("wb") as stream:
            np.save(stream, changed, allow_pickle=False)
        result_path = self.directory / decoded_references[1]["path"]
        changed_result = worker.read_json(result_path)
        changed_result["rgb_file"] = worker._file_record(rgb_path, self.directory)
        changed_result["rgb_sha256"] = hashlib.sha256(changed.tobytes()).hexdigest()
        result_path.write_bytes(worker.canonical_json(changed_result) + b"\n")
        decoded_references[1]["sha256"] = file_sha256(result_path)
        measured = reference.compare_reference_outputs(self.directory, **arguments)
        self.assertFalse(measured["comparisons"]["rgb_bitwise_equal"])
        self.assertFalse(measured["all_compared_outputs_equal"])
        state = reference._load_state(native, self.directory, 1 << 20)
        state["rng_state"][0] ^= 1
        native["final_rng_sha256"] = tensor_digest(state["rng_state"])
        state_path = self.directory / native["latent_file"]["path"]
        state_path.write_bytes(serialize_tensors(state))
        native["latent_file"] = worker._file_record(state_path, self.directory)
        result_path = self.directory / native_reference["path"]
        result_path.write_bytes(worker.canonical_json(native) + b"\n")
        native_reference["sha256"] = file_sha256(result_path)
        changed_result["latent_source"] = native_reference
        result_path = self.directory / decoded_references[1]["path"]
        result_path.write_bytes(worker.canonical_json(changed_result) + b"\n")
        decoded_references[1]["sha256"] = file_sha256(result_path)
        measured = reference.compare_reference_outputs(self.directory, **arguments)
        self.assertFalse(measured["comparisons"]["rng_state_bitwise_equal"])
        self.assertTrue(measured["comparisons"]["latents_bitwise_equal"])

    def test_legacy_prefix_audit_still_accepts_its_original_prefill_lifecycle(self):
        from examples.native_causvid import DenseNativeAudit

        self.pipe.denoising_step_list = torch.tensor([1000, 757, 522])
        pipeline = OriginalPipelineFixture(self.pipe, self.encode_text)
        noise, _, rng = initial_state()
        prefix = noise[:, :3].clone() * 0.25
        with torch.no_grad(), reference._latent_boundary(pipeline), DenseNativeAudit(pipeline, prefix, 3) as audit:
            with self.assertRaises(reference._LatentExport):
                pipeline.inference(noise, [self.request()["text"]], start_latents=prefix, return_latents=True, rng=rng)
            result = audit.finish()
        self.assertFalse(result["first_chunk_denoised_from_scratch"])
        self.assertEqual(result["calls"][0]["timestep"], 0)
        self.assertEqual(result["calls"][0]["input_sha256"], tensor_digest(prefix))
        self.assertEqual(len(result["calls"]), 9)

    def test_request_is_frozen_and_exact_text_conditioning_is_bound(self):
        document = self.request()
        request = worker.CausVidRankRequest.from_dict(document, asset_id=self.asset_id)
        document["text"] = "changed"
        self.assertEqual(request.document["text"], "A red fox runs beside a river.")
        returned = request.document
        returned["spec_config"]["K"]["kind"] = "changed"
        self.assertEqual(request.document["spec_config"]["K"]["kind"], "raw")
        for changes in (
            {"cell_id": "causvid720"}, {"seed": True}, {"seed": -1},
            {"conditioning_identity": "0" * 64}, {"output_namespace": "../escape"},
            {"output_namespace": "/absolute"}, {"output_namespace": "x//y"},
            {"history_mode": "cached_raw"}, {"chunks": False},
        ):
            with self.subTest(changes=changes), self.assertRaises(ConfigurationError):
                worker.CausVidRankRequest.from_dict({**self.request(), **changes}, asset_id=self.asset_id)

    def test_admission_covers_current_clean_capture_and_entire_quality_history(self):
        for mode, field, amount in (
            ("reload", "current_bytes", CELLS["causvid480"].native_kv_bytes - 1),
            ("reload", "clean_host_bytes", CELLS["causvid480"].source_bytes - 1),
            ("quality_hot", "quality_history_bytes", 3 * CELLS["causvid480"].native_kv_bytes - 1),
            ("reload", "device_bytes", 1),
            ("reload", "host_bytes", 1),
        ):
            document = self.request(chunks=3, mode=mode)
            document["memory_budgets"][field] = amount
            with self.subTest(field=field), self.assertRaises(CapacityError):
                worker.CausVidRankRequest.from_dict(document, asset_id=self.asset_id)
        document = self.request()
        document["memory_budgets"]["clean_disk_bytes"] = CELLS["causvid480"].source_bytes + 60 * 65536
        document["memory_budgets"]["clean_host_bytes"] = 4 * 4680 * 1536 * 2 + 65536
        worker.CausVidRankRequest.from_dict(document, asset_id=self.asset_id)
        document["memory_budgets"]["clean_host_bytes"] -= 1
        with self.assertRaises(CapacityError):
            worker.CausVidRankRequest.from_dict(document, asset_id=self.asset_id)

    def test_controller_translation_cannot_expand_any_resource_grant(self):
        limits = self.budgets()
        row = {
            "schema": "reelcache.rank_request.v1", "request_id": "c" * 64, "cell_id": "causvid480",
            "case_id": "explicit-case", "trajectory_id": "explicit-raw-trajectory", "arm_id": "raw",
            "group_id": "g-fox", "prompt_id": "cal00", "prompt": self.request()["text"],
            "seed": 0, "chunks": 3, "split": "calibration", "history_mode": "reload",
            "source_fps": 16, "output_namespace": "controller-case", "representation": {"kind": "raw"},
            "budgets": {
                "device_bytes_per_rank": limits["device_bytes"], "history_bytes_per_rank": 2 << 30,
                "workspace_bytes_per_rank": 6 << 30, "host_bytes": limits["host_bytes"],
                "source_staging_bytes_global": limits["clean_host_bytes"],
                "disk_bytes": 10 << 30, "disk_reserve_bytes": 1 << 30,
            },
        }
        assets = SimpleNamespace(asset_id=self.asset_id, fps=16)
        translated = worker._study_request(row, assets, limits, "calibration_frontier")
        request = worker.CausVidRankRequest.from_dict(translated, asset_id=self.asset_id)
        self.assertEqual(request.document["request_id"], row["request_id"])
        self.assertEqual(request.document["study_request"], row)
        self.assertEqual(request.document["conditioning_identity"],
                         worker.conditioning_identity(row["prompt"], self.asset_id))
        for field in ("device_bytes_per_rank", "history_bytes_per_rank", "workspace_bytes_per_rank",
                      "host_bytes", "disk_bytes"):
            invalid = copy.deepcopy(row)
            invalid["budgets"][field] = 1
            with self.subTest(field=field), self.assertRaises(CapacityError):
                worker._study_request(invalid, assets, limits, "calibration_frontier")
        with self.assertRaisesRegex(ConfigurationError, "worker-budgets"):
            worker._study_request(row, assets, None, "calibration_frontier")
        with self.assertRaisesRegex(ConfigurationError, "FPS"):
            worker._study_request({**row, "source_fps": 24}, assets, limits, "calibration_frontier")
        without_case = dict(row)
        without_case.pop("case_id")
        with self.assertRaisesRegex(ConfigurationError, "explicitly bind case_id"):
            worker._study_request(without_case, assets, limits, "calibration_frontier")

    def test_adaptive_translation_rejects_every_outer_codec_mismatch_before_native(self):
        assets = SimpleNamespace(asset_id=self.asset_id, fps=16)
        policy = self.rank_policy().to_dict()
        for outer in ("raw", "svd", "svd_hevc"):
            for inner in ("svd", "svd_hevc"):
                if outer == inner:
                    continue
                row = self.shared_request()
                row["representation"] = {
                    "kind": outer, "rank_k": 6, "rank_v": 4, "rank_policy": copy.deepcopy(policy),
                }
                row["representation"]["rank_policy"]["codec"]["kind"] = inner
                with self.subTest(outer=outer, inner=inner), self.assertRaisesRegex(
                    ConfigurationError, "Outer representation",
                ):
                    worker._study_request(row, assets, self.budgets(), "calibration_frontier")
        for name, value in (
            ("normalization", "global"), ("protected_fraction", 0.5),
            ("pix_fmt", "yuv420p"), ("qp", 9), ("qp", 8.0), ("preset", "slow"),
            ("ffmpeg_path", "/unverified/ffmpeg"),
        ):
            row = self.shared_request()
            row["representation"] = {
                "kind": "svd", "rank_k": 6, "rank_v": 4,
                "rank_policy": copy.deepcopy(policy), name: value,
            }
            with self.subTest(field=name), self.assertRaisesRegex(ConfigurationError, "Outer representation"):
                worker._study_request(row, assets, self.budgets(), "calibration_frontier")
        row = self.shared_request()
        row["representation"] = {"kind": "svd", "rank_k": 6, "rank_v": 4, "rank_policy": policy}
        translated = worker._study_request(row, assets, self.budgets(), "calibration_frontier")
        self.assertIsNone(translated["spec_config"])
        self.assertEqual(translated["rank_policy"], policy)
        engine = self.engine(rank_policy_factory=worker.build_rank_policy)
        translated["study_request"]["representation"]["kind"] = "svd_hevc"
        before = torch.get_rng_state().clone()
        with self.assertRaises(ConfigurationError):
            engine.run(translated)
        self.assertEqual(self.texts, [])
        self.assertFalse((self.directory / "controller-case").exists())
        self.assertTrue(torch.equal(before, torch.get_rng_state()))

    def test_parent_static_layer_vectors_translate_to_immutable_complete_specs(self):
        row = self.shared_request()
        ranks = {"K": [index + 1 for index in range(30)], "V": [30 - index for index in range(30)]}
        row["representation"] = {"kind": "svd", "layer_ranks": copy.deepcopy(ranks)}
        assets = SimpleNamespace(asset_id=self.asset_id, fps=16)
        translated = worker._study_request(row, assets, self.budgets(), "calibration_frontier")
        request = worker.CausVidRankRequest.from_dict(translated, asset_id=self.asset_id)
        self.assertIsNone(request.document["rank_policy"])
        self.assertEqual(request.document["case_id"], row["case_id"])
        self.assertEqual(request.document["trajectory_id"], row["trajectory_id"])
        specs = worker.codec_specs(request.document["spec_config"], 30)
        self.assertEqual(len(specs), 60)
        for key, spec in specs.items():
            self.assertEqual(spec.kind, "svd")
            self.assertEqual(spec.rank, ranks[key.target][key.layer])
        row["representation"]["layer_ranks"]["K"][0] = 999
        translated["spec_config"][0]["spec"]["rank"] = 999
        self.assertEqual(worker.codec_specs(request.document["spec_config"], 30)[MatrixKey(0, "K")].rank, 1)
        fixed = {MatrixKey(0, "K"): CodecSpec("svd", 2), MatrixKey(0, "V"): CodecSpec("svd", 4)}
        adapter, runtime = self.adapter("static-vector", specs=fixed)
        with patch.object(runtime, "seal_chunk", wraps=runtime.seal_chunk) as seal:
            adapter.step()
            adapter.step()
        self.assertIsNone(adapter.rank_policy)
        for call in seal.call_args_list:
            self.assertEqual(dict(call.args[4]), fixed)
            self.assertNotIn("rank_policy", call.kwargs)
            self.assertNotIn("source_provenance", call.kwargs)
        with self.assertRaises(TypeError):
            adapter.specs[MatrixKey(0, "K")] = CodecSpec("svd", 1)
        self.assertEqual(
            {item["key"]["target"]: item["spec"]["rank"] for item in adapter._metadata()["binding"]["specs"]},
            {"K": 2, "V": 4},
        )

    def test_static_layer_vectors_reject_incomplete_or_ambiguous_controls(self):
        assets = SimpleNamespace(asset_id=self.asset_id, fps=16)
        valid = {"kind": "svd", "layer_ranks": {"K": [1] * 30, "V": [2] * 30}}
        invalid = []
        for name, ranks in (
            ("missing-V", {"K": [1] * 30}),
            ("short", {"K": [1] * 29, "V": [2] * 30}),
            ("boolean", {"K": [True] + [1] * 29, "V": [2] * 30}),
            ("float", {"K": [1.0] + [1] * 29, "V": [2] * 30}),
            ("zero", {"K": [0] + [1] * 29, "V": [2] * 30}),
            ("oversized", {"K": [1537] + [1] * 29, "V": [2] * 30}),
        ):
            invalid.append((name, {"kind": "svd", "layer_ranks": ranks}))
        invalid.extend((
            ("raw", {**valid, "kind": "raw"}),
            ("adaptive", {**valid, "rank_policy": self.rank_policy().to_dict()}),
            ("uniform", {**valid, "rank_k": 1, "rank_v": 2}),
        ))
        for name, representation in invalid:
            row = self.shared_request()
            row["representation"] = representation
            with self.subTest(name=name), self.assertRaises(ConfigurationError):
                worker._study_request(row, assets, self.budgets(), "calibration_frontier")
        row = self.shared_request()
        row["representation"] = {"kind": "svd", "rank_k": 1, "rank_v": 2}
        uniform = worker._study_request(row, assets, self.budgets(), "calibration_frontier")
        self.assertEqual(uniform["spec_config"], {"K": {"kind": "svd", "rank": 1},
                                                "V": {"kind": "svd", "rank": 2}})

    def test_rank_control_metadata_distinguishes_vectors_uniform_policy_and_raw(self):
        assets = SimpleNamespace(asset_id=self.asset_id, fps=16)
        vectors = {"K": [2] * 30, "V": [3] * 30}
        settings = {"normalization": "per_channel", "pix_fmt": "yuv444p", "qp": 8, "preset": "ultrafast"}
        for representation, expected in (
            ({"kind": "raw"}, "raw"),
            ({"kind": "svd", "rank_k": 2, "rank_v": 3}, "uniform"),
            ({"kind": "svd", "layer_ranks": vectors}, "static-layerwise"),
            ({"kind": "svd_hevc", "layer_ranks": vectors, **settings}, "static-layerwise"),
            ({"kind": "svd", "rank_policy": self.rank_policy().to_dict()}, "adaptive"),
        ):
            with self.subTest(mode=expected, kind=representation["kind"]):
                row = self.shared_request()
                row["representation"] = copy.deepcopy(representation)
                document = worker._study_request(row, assets, self.budgets(), "calibration_frontier")
                request = worker.CausVidRankRequest.from_dict(document, asset_id=self.asset_id)
                metadata = worker.rank_control_metadata(request.document)
                self.assertEqual(metadata["rank_selection"], expected)
                self.assertEqual(metadata["rank_configuration_sha256"], identity({
                    "spec_config": request.document["spec_config"], "rank_policy": request.document["rank_policy"],
                }))
                self.assertEqual(request.document["case_id"], row["case_id"])
                self.assertEqual(request.document["study_request"]["representation"], representation)
                if expected == "static-layerwise":
                    self.assertEqual(metadata["layer_ranks"], vectors)
                    self.assertIsNone(document["rank_policy"])
                    if representation["kind"] == "svd_hevc":
                        for spec in worker.codec_specs(document["spec_config"], 30).values():
                            for name, value in settings.items():
                                self.assertEqual(getattr(spec, name), value)
                    before = metadata["rank_configuration_sha256"]
                    changed = request.document
                    changed["spec_config"][0]["spec"]["rank"] += 1
                    self.assertNotEqual(before, worker.rank_control_metadata(changed)["rank_configuration_sha256"])
                else:
                    self.assertNotIn("layer_ranks", metadata)
        engine = self.engine()
        uniform = self.request("metadata-uniform")
        uniform["spec_config"] = {"K": {"kind": "svd", "rank": 2}, "V": {"kind": "svd", "rank": 3}}
        with patch("reelcache.manager.KVRuntime", MemoryRuntime):
            result = engine.run(uniform)
        self.assertEqual(result["rank_selection"], "uniform")
        for captured in result["clean_captures"]:
            self.assertEqual(captured["source_provenance"]["rank_selection"], "uniform")
            self.assertEqual(
                captured["source_provenance"]["rank_configuration_sha256"], result["rank_configuration_sha256"],
            )

    def test_policy_only_and_paired_legacy_hints_never_become_selected_static_ranks(self):
        assets = SimpleNamespace(asset_id=self.asset_id, fps=16)
        policy = self.rank_policy().to_dict()
        row = self.shared_request()
        row["representation"] = {"kind": "svd", "rank_policy": policy}
        baseline = worker._study_request(row, assets, self.budgets(), "calibration_frontier")
        legacy = copy.deepcopy(row)
        legacy["representation"].update(rank_k=1536, rank_v=1)
        hinted = worker._study_request(legacy, assets, self.budgets(), "calibration_frontier")
        self.assertEqual(baseline["rank_policy"], hinted["rank_policy"])
        self.assertIsNone(baseline["spec_config"])
        self.assertIsNone(hinted["spec_config"])
        self.assertEqual(worker.rank_control_metadata(baseline), worker.rank_control_metadata(hinted))
        for target in ("rank_k", "rank_v"):
            invalid = copy.deepcopy(row)
            invalid["representation"][target] = 2
            with self.subTest(target=target), self.assertRaisesRegex(ConfigurationError, "both K and V"):
                worker._study_request(invalid, assets, self.budgets(), "calibration_frontier")

    def test_static_layer_vector_snapshot_restores_exactly_and_rejects_one_changed_rank(self):
        from reelcache.manager import RestoredState

        self.pipe = FakePipeline()
        self.pipe.vae = None
        self.model = replace(self.model, layers=2)
        vector = {
            MatrixKey(0, "K"): CodecSpec("svd", 2), MatrixKey(0, "V"): CodecSpec("svd", 3),
            MatrixKey(1, "K"): CodecSpec("svd", 4), MatrixKey(1, "V"): CodecSpec("svd", 1),
        }
        source, runtime = self.adapter("vector-parent", specs=vector)
        source.step()
        head = runtime.branch_info("vector-parent")
        saved_values = {}

        def snapshot(branch, *, metadata, tensors):
            self.assertEqual(branch, "vector-parent")
            saved_values["snapshot"] = RestoredState(
                "snapshot-vector", branch, copy.deepcopy(metadata),
                {name: value.detach().cpu().clone() for name, value in tensors.items()}, None, None,
            )
            return "snapshot-vector"

        with patch.object(runtime, "snapshot", side_effect=snapshot, create=True):
            snapshot_id = source.snapshot()
        saved = saved_values["snapshot"]
        self.assertEqual(
            {MatrixKey.from_dict(row["key"]): CodecSpec.from_dict(row["spec"]) for row in saved.metadata["binding"]["specs"]},
            vector,
        )
        self.assertNotIn("rank_policy", saved.metadata["binding"])
        source.step()
        expected_latents, expected_rng = source.latent_state, source.rng_state
        source.close()

        def fork(snapshot, branch, *, rank_policy, metadata_validator, validator):
            self.assertEqual(snapshot, snapshot_id)
            self.assertIsNone(rank_policy)
            self.assertNotIn(branch, runtime.branches)
            metadata_validator(replace(saved, tensors={}))
            validator(saved)
            runtime.branches[branch] = dict(head)
            for key in vector:
                runtime.packets[branch, 0, key] = runtime.packets["vector-parent", 0, key].clone()
            return saved

        restored = CausVidAdapter(self.pipe, runtime, "idle", config=CONFIG, specs=vector).install()
        self.addCleanup(restored.close)
        with patch.object(runtime, "fork", side_effect=fork, create=True):
            restored.restore(snapshot_id, "vector-child")
        self.assertIsNone(restored.rank_policy)
        self.assertEqual(dict(restored.specs), vector)
        restored.step()
        self.assertTrue(torch.equal(restored.latent_state, expected_latents))
        self.assertTrue(torch.equal(restored.rng_state, expected_rng))
        restored.close()
        incompatible = dict(vector)
        incompatible[MatrixKey(1, "K")] = CodecSpec("svd", 3)
        rejected = CausVidAdapter(self.pipe, runtime, "unchanged", config=CONFIG, specs=incompatible).install()
        self.addCleanup(rejected.close)
        before_rng, before_branches = torch.get_rng_state().clone(), copy.deepcopy(runtime.branches)
        with patch.object(runtime, "fork", side_effect=fork, create=True), patch.object(
            rejected, "_prepare_conditioning", side_effect=AssertionError("conditioning must not be touched"),
        ):
            with self.assertRaisesRegex(ConfigurationError, "Snapshot model, codec"):
                rejected.restore(snapshot_id, "wrong-vector", seed=17)
        self.assertEqual(runtime.branches, before_branches)
        self.assertEqual(rejected.branch_id, "unchanged")
        self.assertTrue(torch.equal(before_rng, torch.get_rng_state()))

    def test_output_admission_covers_saved_state_spectra_rgb_and_reserve(self):
        document = self.request("disk")
        plain = worker._output_reservation(document, "generate")
        document["capture"] = {"spectra": True}
        instrumented = worker._output_reservation(document, "generate")
        self.assertGreater(plain["latent_file_bytes"], 2 * 3 * 16 * 60 * 104 * 2)
        self.assertGreater(instrumented["metadata_bytes"], plain["metadata_bytes"])
        for chunks, frames in ((2, 21), (3, 33)):
            decoded = {**self.request("decode-budget", chunks=chunks),
                       "latent_source": {"path": "source/result.json", "sha256": "0" * 64}}
            plan = worker._output_reservation(decoded, "decode")
            self.assertGreaterEqual(plan["rgb_file_bytes"], frames * 480 * 832 * 3 + 128)
            self.assertEqual(plan["runtime_bytes"], 0)
            decoded["memory_budgets"]["store_bytes"] = 1
            decoded["disk_budget"] = {"disk_bytes": 2, "disk_reserve_bytes": 1}
            with self.assertRaisesRegex(CapacityError, "Complete CausVid outputs"):
                worker.CausVidRankRequest.from_dict(decoded, asset_id=self.asset_id)
        row = self.shared_request()
        row["budgets"]["disk_bytes"] = 2
        row["budgets"]["disk_reserve_bytes"] = 1
        limits = self.budgets()
        limits["store_bytes"] = 1
        row["latent_source"] = {"path": "source/result.json", "sha256": "0" * 64}
        translated = worker._study_request(
            row, SimpleNamespace(asset_id=self.asset_id, fps=16), limits, "calibration_frontier",
        )
        with self.assertRaises(CapacityError):
            worker.CausVidRankRequest.from_dict(translated, asset_id=self.asset_id)
        with patch.object(worker.shutil, "disk_usage", return_value=SimpleNamespace(free=1)):
            with self.assertRaisesRegex(CapacityError, "preserve the disk reserve"):
                worker._RequestOutputBudget(self.request(), self.directory, "generate")

    def test_artifact_writers_enforce_live_output_capacity_before_publication(self):
        from reelcache.spectra import SpectrumSource, exact_spectrum

        document = self.request("bounded-writes")
        plan = worker._output_reservation(document, "generate")
        document["disk_budget"]["disk_bytes"] = (
            plan["reserved_component_bytes"] + plan["artifact_floor_bytes"] + plan["disk_reserve_bytes"]
        )
        disk = worker._RequestOutputBudget(document, self.directory, "generate")
        disk.charge(disk.artifact_limit - disk.plan["failure_bytes"] - 1)
        with self.assertRaises(CapacityError):
            worker._bytes_new(self.directory / "over-budget.bin", b"ab", disk=disk)
        with self.assertRaises(CapacityError):
            worker._json_new(self.directory / "over-budget.json", {"x": 1}, disk=disk)
        source = SpectrumSource(
            identity="output-budget-fixture", shape=(2, 2), dtype="float32", target="V", key_space="none",
        )
        result = exact_spectrum(torch.eye(2), source=source)
        with self.assertRaises(CapacityError):
            worker._persist_spectrum(
                result, identity=None, spectrum_path=self.directory / "over-budget-spectrum.json", disk=disk,
            )
        self.assertFalse(list(self.directory.glob("over-budget*")))

    def test_a_b_a_uses_fresh_requests_without_prefix_or_global_rng(self):
        engine = self.engine()
        before = torch.get_rng_state().clone()
        with patch("reelcache.manager.KVRuntime", MemoryRuntime):
            a = engine.run(self.request("a-first"))
            b = engine.run(self.request("b", text="A sailing ship crosses a stormy ocean.", seed=1))
            replay = engine.run(self.request("a-replay"))
        self.assertEqual(a["latents_sha256"], replay["latents_sha256"])
        self.assertEqual(a["initial_noise_sha256"], replay["initial_noise_sha256"])
        self.assertEqual(a["final_rng_sha256"], replay["final_rng_sha256"])
        self.assertNotEqual(a["latents_sha256"], b["latents_sha256"])
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertEqual(len(MemoryRuntime.instances), 3)
        self.assertTrue(all(item.closed and item.active_leases == 0 for item in MemoryRuntime.instances))
        self.assertEqual(self.texts, [
            ("A red fox runs beside a river.",), ("A sailing ship crosses a stormy ocean.",),
            ("A red fox runs beside a river.",),
        ])
        for result in (a, b, replay):
            self.assertTrue(result["generation_from_scratch"])
            self.assertFalse(result["supplied_prefix"])
            self.assertTrue(result["request_reset_verified"])
            self.assertEqual(result["chunks"][0]["clean_audit"]["denoising_steps"], 2)
            self.assertEqual(result["chunks"][0]["clean_audit"]["token_start"], 0)
            self.assertFalse(result["vae_loaded"])
            self.assertFalse(result["native_parity_claim"])
            self.assertTrue(result["fixture"])
            self.assertFalse(result["native_execution"])
        self.assertTrue(worker._native_cache_empty(self.pipe))
        self.assertFalse(self.pipe.generator.model.blocks[0].self_attn.norm_k._forward_hooks)
        self.assertTrue((self.directory / "a-replay/result.json").is_file())

    def test_invalid_request_does_not_touch_model_or_rng_and_namespace_cannot_repeat(self):
        engine = self.engine()
        before = torch.get_rng_state().clone()
        bad = self.request("bad")
        bad["memory_budgets"]["current_bytes"] = 1
        with self.assertRaises(CapacityError):
            engine.run(bad)
        self.assertEqual(self.texts, [])
        self.assertFalse((self.directory / "bad").exists())
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        with patch("reelcache.manager.KVRuntime", MemoryRuntime):
            engine.run(self.request("valid"))
        with self.assertRaises(StateError):
            engine.run(self.request("valid"))
        self.assertEqual(len(self.texts), 1)

    def test_generation_rng_is_independent_of_codec_global_rng(self):
        engine = self.engine()
        with patch("reelcache.manager.KVRuntime", MemoryRuntime):
            reference = engine.run(self.request("rng-reference", chunks=2))
            seal = MemoryRuntime.seal_chunk

            def consume_global(runtime, *args, **kwargs):
                torch.rand(123)
                return seal(runtime, *args, **kwargs)

            with patch.object(MemoryRuntime, "seal_chunk", consume_global):
                perturbed = engine.run(self.request("rng-perturbed", chunks=2))
        for name in ("latents_sha256", "initial_noise_sha256", "denoising_rng_sha256", "final_rng_sha256"):
            self.assertEqual(reference[name], perturbed[name])

    def test_failed_request_cleans_hooks_and_explicitly_aborts_epoch(self):
        engine = self.engine()
        with patch("reelcache.manager.KVRuntime", MemoryRuntime), patch.object(
            engine, "conditioning_fn", side_effect=RuntimeError("injected text failure"),
        ):
            with self.assertRaisesRegex(RuntimeError, "text failure"):
                engine.run(self.request("failed"))
        self.assertTrue(worker._native_cache_empty(self.pipe))
        self.assertFalse(self.pipe.generator.model.blocks[0].self_attn.norm_k._forward_hooks)
        self.assertTrue(MemoryRuntime.instances[-1].closed)
        failure = json.loads((self.directory / "failed/failure.json").read_text())
        self.assertEqual(failure["status"], "abort_epoch")
        self.assertGreaterEqual(failure["elapsed_seconds"], 0)
        self.assertIn("before_final_cleanup", failure["timing_scope"])
        with self.assertRaisesRegex(StateError, "abort_epoch"):
            engine.run(self.request("cannot-reuse"))
        self.assertFalse((self.directory / "failed/result.json").exists())

    def test_schedule_drift_aborts_epoch_even_if_later_restored(self):
        engine = self.engine()
        with patch("reelcache.manager.KVRuntime", MemoryRuntime):
            engine.run(self.request("schedule-a"))
            previous = self.pipe.denoising_step_list.clone()
            self.pipe.denoising_step_list = torch.tensor([700, 200])
            before = torch.get_rng_state().clone()
            with self.assertRaisesRegex(StateError, "configuration drifted"):
                engine.run(self.request("schedule-b"))
            self.assertTrue(torch.equal(before, torch.get_rng_state()))
            self.assertFalse((self.directory / "schedule-b").exists())
            self.pipe.denoising_step_list = previous
            with self.assertRaisesRegex(StateError, "abort_epoch"):
                engine.run(self.request("schedule-a-again"))
        self.assertEqual(len(self.texts), 1)

    def test_scheduler_and_generator_configuration_drift_reject_before_conditioning(self):
        for name in ("scheduler-sigmas", "scheduler-scalar", "generator-uniform", "pipeline-args", "tokenizer-config"):
            with self.subTest(name=name):
                self.pipe.args = SimpleNamespace(warp_denoising_step=True)
                self.pipe.text_encoder = SimpleNamespace(tokenizer=SimpleNamespace(seq_len=512, clean="whitespace"))
                engine = self.engine()
                previous_sigmas = self.pipe.scheduler.sigmas.clone()
                previous_uniform = self.pipe.generator.uniform_timestep
                if name == "scheduler-sigmas":
                    self.pipe.scheduler.sigmas[0] = 0.7
                elif name == "scheduler-scalar":
                    self.pipe.scheduler.shift = 4.0
                elif name == "generator-uniform":
                    self.pipe.generator.uniform_timestep = not previous_uniform
                elif name == "pipeline-args":
                    self.pipe.args.warp_denoising_step = False
                else:
                    self.pipe.text_encoder.tokenizer.seq_len = 256
                before = torch.get_rng_state().clone()
                with self.assertRaisesRegex(StateError, "configuration drifted"):
                    engine.run(self.request(name))
                self.assertTrue(torch.equal(before, torch.get_rng_state()))
                self.assertEqual(self.texts, [])
                self.assertFalse((self.directory / name).exists())
                self.pipe.scheduler.sigmas.copy_(previous_sigmas)
                self.pipe.generator.uniform_timestep = previous_uniform
                if hasattr(self.pipe.scheduler, "shift"):
                    del self.pipe.scheduler.shift
                self.pipe.args.warp_denoising_step = True
                self.pipe.text_encoder.tokenizer.seq_len = 512
                engine.close()

    def test_inflight_generation_configuration_drift_cannot_publish_complete_result(self):
        engine = self.engine()
        seal = MemoryRuntime.seal_chunk

        def drift(runtime, *args, **kwargs):
            result = seal(runtime, *args, **kwargs)
            self.pipe.scheduler.sigmas[0] = 0.7
            return result

        with patch("reelcache.manager.KVRuntime", MemoryRuntime), patch.object(MemoryRuntime, "seal_chunk", drift):
            with self.assertRaisesRegex(StateError, "configuration drifted"):
                engine.run(self.request("inflight-config"))
        failure = json.loads((self.directory / "inflight-config/failure.json").read_text())
        self.assertEqual(failure["status"], "abort_epoch")
        self.assertFalse((self.directory / "inflight-config/result.json").exists())
        self.assertFalse((self.directory / "inflight-config/latents.reelcache").exists())
        self.assertTrue(worker._native_cache_empty(self.pipe))
        self.assertTrue(MemoryRuntime.instances[-1].closed)
        with self.assertRaisesRegex(StateError, "abort_epoch"):
            engine.run(self.request("inflight-cannot-reuse"))

    def test_mid_request_source_change_aborts_before_publishing_complete_result(self):
        calls = []

        def verify():
            calls.append(True)
            if len(calls) > 1:
                raise IntegrityError("source changed while native request was running")

        engine = self.engine(revalidate=verify)
        with patch("reelcache.manager.KVRuntime", MemoryRuntime):
            with self.assertRaisesRegex(IntegrityError, "source changed"):
                engine.run(self.request("stale-during-request"))
        self.assertEqual(len(calls), 2)
        self.assertFalse((self.directory / "stale-during-request/result.json").exists())
        self.assertEqual(
            json.loads((self.directory / "stale-during-request/failure.json").read_text())["status"],
            "abort_epoch",
        )
        self.assertTrue(worker._native_cache_empty(self.pipe))
        with self.assertRaisesRegex(StateError, "abort_epoch"):
            engine.run(self.request("stale-epoch"))

    def test_rank_abstention_persists_failed_decision_without_raw_fallback(self):
        from reelcache.online_rank import RankAbstention

        decision_record = {"status": "OVER_BUDGET", "reason": "fake workspace limit", "rank": None}
        decision = SimpleNamespace(
            status=decision_record["status"], reason=decision_record["reason"],
            to_dict=lambda: dict(decision_record),
        )
        engine = self.engine()
        with patch("reelcache.manager.KVRuntime", MemoryRuntime), patch.object(
            MemoryRuntime, "seal_chunk", side_effect=RankAbstention(decision),
        ):
            with self.assertRaises(RankAbstention):
                engine.run(self.request("abstained"))
        failure = json.loads((self.directory / "abstained/failure.json").read_text())
        self.assertEqual(failure["rank_decision"], decision_record)
        self.assertEqual(MemoryRuntime.instances[-1].branch_info("request")["token_end"], 0)
        self.assertFalse(MemoryRuntime.instances[-1].packets)
        self.assertTrue(failure["committed_head_preserved"])
        self.assertFalse((self.directory / "abstained/result.json").exists())
        with self.assertRaisesRegex(StateError, "abort_epoch"):
            engine.run(self.request("no-implicit-raw-retry"))

    def test_cleanup_uncertainty_always_poisoned_even_when_request_started_clean(self):
        from reelcache.adapters.causvid import CausVidAdapter

        engine = self.engine()
        close = CausVidAdapter.close

        def fail_after_cleanup(adapter):
            close(adapter)
            raise StateError("injected uncertain cleanup")

        with patch("reelcache.manager.KVRuntime", MemoryRuntime), patch.object(
            CausVidAdapter, "close", fail_after_cleanup,
        ):
            with self.assertRaisesRegex(StateError, "uncertain cleanup"):
                engine.run(self.request("uncertain-cleanup"))
        self.assertTrue(worker._native_cache_empty(self.pipe))
        self.assertTrue(MemoryRuntime.instances[-1].closed)
        with self.assertRaisesRegex(StateError, "abort_epoch"):
            engine.run(self.request("unsafe-to-reuse"))
        self.assertFalse((self.directory / "uncertain-cleanup/result.json").exists())

    def test_worker_rejects_concurrent_entry(self):
        engine = self.engine()
        engine._lock.acquire()
        try:
            with self.assertRaisesRegex(StateError, "serial"):
                engine.run(self.request())
        finally:
            engine._lock.release()
        self.assertEqual(self.texts, [])

    def test_observer_sees_only_original_clean_pre_key_and_direct_value(self):
        captured, expected = {}, {}
        attention = self.pipe.generator.model.blocks[0].self_attn
        adapter = None

        def expected_hook(target):
            def observe(_module, _inputs, output):
                if adapter._phase == "clean_forward":
                    expected[(adapter.token_cursor, target)] = output[0].detach().clone()
            return observe

        handles = [
            attention.norm_k.register_forward_hook(expected_hook("K")),
            attention.v.register_forward_hook(expected_hook("V")),
        ]
        self.addCleanup(lambda: [handle.remove() for handle in handles])

        def observe(context, tensor):
            self.assertEqual(adapter._phase, "clean_forward")
            self.assertEqual(context.key_space, "pre_rope" if context.key.target == "K" else "none")
            self.assertEqual(context.dtype, "float32")
            captured[(context.token_start, context.key.target)] = tensor.detach().clone()

        adapter, runtime = self.adapter(clean_observer=observe, source_provenance={"case": "fake"})
        adapter.step()
        adapter.step()
        self.assertEqual(len(captured), 4)
        for key, tensor in captured.items():
            self.assertTrue(torch.equal(tensor, expected[key]))
        self.assertFalse(torch.equal(
            captured[(CONFIG.chunk_tokens, "K")], runtime.sources[("request", 1, MatrixKey(0, "K"))],
        ))
        row = adapter.audit[-1]["clean_sources"][0]
        self.assertEqual(row["provenance"], {"case": "fake"})
        self.assertEqual(row["shape"], [CONFIG.chunk_tokens, CONFIG.channels])
        self.assertEqual(row["source"], "native_clean_t0_norm_k_before_rope")
        self.assertEqual(row["available_keys"], [{"layer": 0, "target": "K"}])
        self.assertEqual(row["sha256"], tensor_digest(captured[(CONFIG.chunk_tokens, "K")]))
        self.assertEqual(adapter.memory_report()["live_clean_capture_cpu_bytes"], 0)

    def test_observer_mutation_and_async_callback_fail_without_committing(self):
        def mutate(_context, tensor):
            tensor.add_(1)

        adapter, runtime = self.adapter(clean_observer=mutate)
        rng = adapter.rng_state
        with self.assertRaisesRegex(IntegrityError, "mutated"):
            adapter.step()
        self.assertEqual(runtime.branch_info("request")["token_end"], 0)
        self.assertTrue(torch.equal(rng, adapter.rng_state))
        self.assertFalse(adapter._mutable)
        adapter.close()

        async def asynchronous(_context, _tensor):
            return None

        adapter, runtime = self.adapter("async", clean_observer=asynchronous)
        with self.assertRaisesRegex(UnsupportedError, "synchronously"):
            adapter.step()
        self.assertEqual(runtime.branch_info("async")["token_end"], 0)

    def test_owned_staging_is_bounded_lazy_and_hash_checked(self):
        key = MatrixKey(0, "V")
        source = torch.arange(24, dtype=torch.float32).reshape(3, 8)
        host = CausVidCleanStaging(max_host_bytes=4 * source.numel() * 4 + 65536)
        self.addCleanup(host.close)
        host.put(key, source)
        source.zero_()
        self.assertNotEqual(float(host[key].sum()), 0)
        with self.assertRaises(CapacityError):
            host.put(MatrixKey(0, "K"), source)
        host[key].add_(1)
        with self.assertRaisesRegex(IntegrityError, "mutated"):
            host[key]
        root = self.directory / "owned-clean"
        disk = CausVidCleanStaging(max_host_bytes=65536 + 4 * 96, max_disk_bytes=4096, directory=root)
        self.addCleanup(disk.close)
        disk.put(key, source)
        self.assertEqual(disk.host_bytes, 0)
        one, two = disk[key], disk[key]
        self.assertNotEqual(one.data_ptr(), two.data_ptr())
        path = next(root.iterdir())
        raw = bytearray(path.read_bytes())
        raw[-1] ^= 1
        path.write_bytes(raw)
        with self.assertRaisesRegex(IntegrityError, "hash"):
            disk[key]
        disk.close()
        self.assertFalse(root.exists())

    def test_source_observer_streams_spectra_and_retains_only_selected_pristine_inputs(self):
        from reelcache.serialization import deserialize_tensors

        engine = self.engine(provenance={"epoch_model_load_seconds": 123.45})
        request = self.request("observed")
        request["memory_budgets"]["spectrum_workspace_bytes"] = 16 << 20
        request["memory_budgets"]["source_archive_bytes"] = 32 << 20
        request["capture"] = {
            "spectra": True, "tile_rows": 3, "retain": [{"chunk_id": 0, "layer": 0, "target": "K"}],
        }
        with patch("reelcache.manager.KVRuntime", MemoryRuntime):
            result = engine.run(request)
        self.assertEqual(len(result["clean_captures"]), 2)
        key, value = result["clean_captures"]
        self.assertIn("retained_source", key)
        self.assertNotIn("retained_source", value)
        source = deserialize_tensors((self.directory / key["retained_source"]["path"]).read_bytes())["source"]
        self.assertEqual(tensor_digest(source), key["source_provenance"]["raw_native_tensor_sha256"])
        spectrum_bytes = (self.directory / key["spectrum"]["path"]).read_bytes()
        spectrum = json.loads(spectrum_bytes)
        self.assertFalse(spectrum_bytes.endswith(b"\n"))
        self.assertEqual(hashlib.sha256(spectrum_bytes).hexdigest(), key["spectrum"]["sha256"])
        self.assertEqual(len(spectrum_bytes), key["spectrum"]["bytes"])
        self.assertGreaterEqual(key["spectrum_work"]["wall_seconds"], 0)
        self.assertIn("not_online_selection", key["spectrum_work"]["timing_scope"])
        self.assertIn("hash_update_seconds", key["spectrum_work"]["oracle_timings_seconds"])
        self.assertIn("gram_matmul_add_seconds", key["spectrum_work"]["oracle_timings_seconds"])
        self.assertTrue(key["spectrum_work"]["oracle_work_counts"])
        telemetry_path = self.directory / key["spectrum_telemetry"]["path"]
        telemetry = json.loads(telemetry_path.read_text())
        self.assertEqual(telemetry["work"], key["spectrum_work"])
        self.assertEqual(telemetry["spectrum_sha256"], key["spectrum"]["sha256"])
        self.assertEqual(file_sha256(telemetry_path), key["spectrum_telemetry"]["sha256"])
        self.assertEqual(spectrum["source"]["representation"], "original")
        self.assertEqual(spectrum["source"]["key_space"], "pre_rope")
        self.assertEqual(spectrum["source"]["retention"], "exact-input-retained")
        self.assertEqual(spectrum["source"]["dtype"], "float32")
        self.assertNotIn("epoch_model_load_seconds", spectrum["source"]["provenance"])
        self.assertTrue(spectrum["source"]["provenance"]["fixture"])
        self.assertFalse(spectrum["source"]["provenance"]["native_execution"])

    def detector_declaration(self):
        from reelcache.factorization import DetectorSpec, rank_fit_workspace_bytes

        spec = DetectorSpec()
        cell = CELLS["causvid480"]
        solver = 64 << 20
        workspace = cell.tokens * cell.channels * 2 + rank_fit_workspace_bytes(
            (cell.tokens, cell.channels), min(cell.tokens, cell.channels),
            tile_rows=spec.tile_rows, solver_workspace_bytes=solver,
        )
        return {"spec": spec.to_dict(), "workspace_bytes": workspace, "solver_workspace_bytes": solver}

    def test_raw_census_requires_frozen_detector_and_complete_width_admission(self):
        row = self.shared_request()
        assets = SimpleNamespace(asset_id=self.asset_id, fps=16)
        limits = self.budgets()
        limits["spectrum_workspace_bytes"] = 512 << 20
        translated = worker._study_request(row, assets, limits, "calibration_census")
        with self.assertRaisesRegex(ConfigurationError, "both exact spectra"):
            worker.CausVidRankRequest.from_dict(translated, asset_id=self.asset_id)
        row["capture"] = {"spectra": True, "detector": self.detector_declaration()}
        translated = worker._study_request(row, assets, limits, "calibration_census")
        accepted = worker.CausVidRankRequest.from_dict(translated, asset_id=self.asset_id).document
        self.assertEqual(accepted["capture"]["detector"], row["capture"]["detector"])
        self.assertEqual(accepted["study_phase"], "calibration_census")
        too_small = copy.deepcopy(translated)
        too_small["capture"]["detector"]["workspace_bytes"] -= 1
        with self.assertRaisesRegex(CapacityError, "complete width ladder"):
            worker.CausVidRankRequest.from_dict(too_small, asset_id=self.asset_id)
        wrong = copy.deepcopy(translated)
        wrong["capture"]["detector"]["spec"]["initial_width"] = 64
        with self.assertRaises(ConfigurationError):
            worker.CausVidRankRequest.from_dict(wrong, asset_id=self.asset_id)
        wrong = copy.deepcopy(translated)
        wrong["study_phase"], wrong["split"] = "heldout", "heldout"
        with self.assertRaisesRegex(ConfigurationError, "only admitted for RAW calibration"):
            worker.CausVidRankRequest.from_dict(wrong, asset_id=self.asset_id)
        wrong = copy.deepcopy(translated)
        wrong["rank_policy"], wrong["spec_config"] = self.rank_policy().to_dict(), None
        wrong.pop("study_request")
        with self.assertRaisesRegex(ConfigurationError, "RAW calibration census"):
            worker.CausVidRankRequest.from_dict(wrong, asset_id=self.asset_id)
        plain = worker._output_reservation({**accepted, "capture": {"spectra": True}}, "generate")
        actual = worker._output_reservation(accepted, "generate")
        self.assertGreater(actual["metadata_bytes"], plain["metadata_bytes"])

    def test_declared_detector_threads_match_whole_epoch_admission_before_model_load(self):
        from reelcache.factorization import DetectorSpec

        before_threads, before_rng = torch.get_num_threads(), torch.get_rng_state().clone()
        spec = DetectorSpec(cpu_threads=8)
        self.assertNotEqual(spec.fingerprint, DetectorSpec().fingerprint)
        declaration = self.detector_declaration()
        declaration["spec"] = spec.to_dict()
        document = self.request("thread-eight")
        document["study_phase"] = "calibration_census"
        document["capture"] = {"spectra": True, "detector": declaration}
        document["memory_budgets"]["spectrum_workspace_bytes"] = 512 << 20
        requested = worker.CausVidRankRequest.from_dict(document, asset_id=self.asset_id)
        self.assertEqual(requested.document["capture"]["detector"]["spec"], spec.to_dict())
        one = copy.deepcopy(document)
        one["request_id"], one["output_namespace"] = "thread-one", "thread-one"
        one["capture"]["detector"]["spec"] = DetectorSpec().to_dict()
        one = worker.CausVidRankRequest.from_dict(one, asset_id=self.asset_id)
        with patch.object(worker.os, "sched_getaffinity", return_value=set(range(32))):
            worker._validate_thread_admission([requested], 8)
            worker._validate_thread_admission([one], 1)
            with self.assertRaisesRegex(ConfigurationError, "admitted CPU thread count"):
                worker._validate_thread_admission([requested], 1)
            with self.assertRaisesRegex(ConfigurationError, "admitted CPU thread count"):
                worker._validate_thread_admission([requested, one], 8)
            adaptive = self.request("adaptive-thread-one")
            adaptive["rank_policy"], adaptive["spec_config"] = self.rank_policy().to_dict(), None
            adaptive = worker.CausVidRankRequest.from_dict(adaptive, asset_id=self.asset_id)
            with self.assertRaisesRegex(ConfigurationError, "admitted CPU thread count"):
                worker._validate_thread_admission([requested, adaptive], 8)
        with patch.object(worker.os, "sched_getaffinity", return_value=set(range(4))):
            with self.assertRaisesRegex(CapacityError, "CPU allocation"):
                worker._validate_thread_admission([requested], 8)
        with self.assertRaisesRegex(ConfigurationError, "thread contract"):
            spec.validate_runtime()
        arguments = [
            "--protocol", "unused", "--assets", "unused", "--asset-map", "causvid480.json",
            "--launch-receipt", "unused", "--requests", "unused", "--output-root", str(self.directory),
            "--threads", "1", "--allow-cuda",
        ]
        with patch.object(worker, "verify_worker_inputs", return_value=(
            {}, {}, SimpleNamespace(root=self.directory / "assets"), [requested],
        )), patch.object(worker, "NativeCausVidEpoch") as native:
            with self.assertRaisesRegex(ConfigurationError, "admitted CPU thread count"):
                worker.main(arguments)
            native.assert_not_called()
        self.assertFalse((self.directory / "thread-eight").exists())
        self.assertFalse((self.directory / "build_cache").exists())
        self.assertEqual(torch.get_num_threads(), before_threads)
        self.assertTrue(torch.equal(torch.get_rng_state(), before_rng))

    def test_raw_census_persists_actual_shared_q2_features_and_releases_directions(self):
        from reelcache import factorization, rank_analysis, spectra, study

        cell = study.StudyCell(
            "synthetic-causvid-census", "synthetic", 8, 1, 1, 0,
            CONFIG.frames_per_chunk, CONFIG.grid_height, CONFIG.grid_width,
            self.model.layers, CONFIG.channels, "float32", "float32", 1,
        )
        catalog = {**study.CELLS, cell.cell_id: cell}
        engine = self.engine(
            provenance={"protocol_id": "b" * 64, "launch_closure_id": "c" * 64},
            observation_evidence_kind="synthetic_fixture",
        )
        document = worker.CausVidRankRequest.from_dict(
            self.request("raw-detectors"), asset_id=self.asset_id,
        ).document
        document["cell_id"], document["study_phase"] = cell.cell_id, "calibration_census"
        document["capture"] = worker._capture_config(
            {"spectra": True, "tile_rows": 3, "detector": self.detector_declaration()}, 1,
        )
        document["memory_budgets"]["spectrum_workspace_bytes"] = 16 << 20
        directory = self.directory / document["output_namespace"]
        directory.mkdir()
        runtime = MemoryRuntime(directory / "runtime", self.model)
        runtime.create_branch("request")
        rows, fits, acquired = [], [], []
        actual_acquire = factorization.acquire_anchor_directions

        def acquire(*args, **kwargs):
            result = actual_acquire(*args, **kwargs)
            fits.append(weakref.ref(result.directions))
            acquired.append(result.to_dict())
            return result

        observe = engine._observer(document, directory, runtime, rows)
        context = TensorContext(self.model, MatrixKey(0, "K"), 0, 3, 1, 2, "pre_rope", source_dtype="float32")
        tensor = torch.arange(48, dtype=torch.float32).reshape(context.shape) / 23
        before, original = torch.get_rng_state().clone(), tensor.clone()
        with patch.object(study, "CELLS", catalog), patch.object(rank_analysis, "CELLS", catalog):
            with patch.object(factorization, "acquire_anchor_directions", side_effect=acquire) as detector, patch.object(
                spectra, "exact_spectrum", wraps=spectra.exact_spectrum,
            ) as oracle:
                observe(context, tensor)
                detector.assert_called_once()
                oracle.assert_called_once()
            record = rows[0]
            path = self.directory / record["detector_records"]["path"]
            envelope = json.loads(path.read_text())
            self.assertEqual(envelope["schema"], "reelcache_rank_detector_observations_v1")
            self.assertEqual(envelope["detector_spec"], document["capture"]["detector"]["spec"])
            self.assertEqual(envelope["detector_sha256"], factorization.DetectorSpec.from_dict(
                envelope["detector_spec"],
            ).fingerprint)
            threshold_records = envelope["observations"]
            numeric = json.loads((self.directory / record["numeric_observation"]["path"]).read_text())["observation"]
            validated = rank_analysis.validate_detector_records(
                [numeric], threshold_records, detector_spec=envelope["detector_spec"],
            )
            self.assertEqual(threshold_records, rank_analysis.detector_records_from_acquisition(
                acquired[0], observation=numeric,
            ))
            self.assertEqual(len(validated), 2)
            self.assertEqual({item["tau"] for item in threshold_records}, {0.95, 0.99})
            self.assertEqual(file_sha256(path), record["detector_records"]["sha256"])
            for item in threshold_records:
                self.assertEqual(item["acquisition"], acquired[0])
                self.assertEqual(item["estimated_rank"], acquired[0]["features"][str(item["tau"])])
                self.assertEqual(item["input_sha256"], numeric["input_sha256"])
                self.assertEqual(item["available_after_layer"], context.key.layer)
                self.assertEqual(item["rank_statistic"], "approximate_spectral_crossing")
                self.assertEqual(item["acquisition_sha256"], record["detector_records"]["acquisition_sha256"])
            work = json.loads((self.directory / record["detector_telemetry"]["path"]).read_text())
            self.assertFalse(work["included_in_online_selector_cost"])
            self.assertTrue(work["included_in_full_request_wall_time"])
            self.assertEqual(work["acquisition_metrics"], acquired[0]["metrics"])
            self.assertEqual(acquired[0]["metrics"]["source_copy_count"], 1)
            self.assertTrue(acquired[0]["metrics"]["operation_counts_complete"])
            self.assertEqual(numeric["original_bytes_sha256"], acquired[0]["source_sha256"])
            self.assertTrue(all(reference() is None for reference in fits))
            self.assertTrue(torch.equal(before, torch.get_rng_state()))
            self.assertTrue(torch.equal(original, tensor))
            changed = copy.deepcopy(threshold_records)
            changed[0]["estimated_rank"] = None if changed[0]["estimated_rank"] is not None else 1
            with self.assertRaises(rank_analysis.AnalysisError):
                rank_analysis.validate_detector_records(
                    [numeric], changed, detector_spec=document["capture"]["detector"]["spec"],
                )
            # A synthetic unresolved helper result must remain null, not inherit the oracle rank.
            unresolved = copy.deepcopy(acquired[0])
            unresolved["features"]["0.99"] = None
            unresolved["status"] = "unresolved"
            unresolved["attempts"][-1].update(status="unresolved", stop_crossing=None)
            unresolved["metrics"].update(status="unresolved", finite_width_misses=1)
            other = self.directory / "unresolved-detector"
            other.mkdir()
            partial = []
            with patch.object(
                factorization, "acquire_anchor_directions",
                return_value=SimpleNamespace(to_dict=lambda: copy.deepcopy(unresolved)),
            ):
                engine._observer(document, other, runtime, partial)(context, tensor)
            partial_records = json.loads(
                (self.directory / partial[0]["detector_records"]["path"]).read_text(),
            )["observations"]
            partial99 = next(item for item in partial_records if item["tau"] == 0.99)
            self.assertIsNone(partial99["estimated_rank"])
            self.assertEqual(partial99["status"], "unresolved")
            failed = self.directory / "failed-detector"
            failed.mkdir()
            bad_document = copy.deepcopy(document)
            bad_document["capture"]["detector"]["workspace_bytes"] = 1
            failures = []
            with self.assertRaises(factorization.DetectorAcquisitionError):
                engine._observer(bad_document, failed, runtime, failures)(context, tensor)
            self.assertNotIn("detector_records", failures[0])
            failure = json.loads((self.directory / failures[0]["detector_failure"]["path"]).read_text())
            self.assertEqual(failure["status"], "failed")
            self.assertEqual(failure["acquisition_metrics"]["decomposition_count"], 0)
            self.assertEqual(failure["acquisition_metrics"]["source_copy_count"], 0)
            self.assertTrue(torch.equal(before, torch.get_rng_state()))
            self.assertTrue(torch.equal(original, tensor))

    def test_failed_oracle_persists_only_observable_outer_work_and_aborts_epoch(self):
        engine = self.engine()
        request = self.request("failed-oracle")
        request["memory_budgets"]["spectrum_workspace_bytes"] = 16 << 20
        request["capture"] = {"spectra": True, "tile_rows": 3}
        before_rng = torch.get_rng_state().clone()
        with patch("reelcache.manager.KVRuntime", MemoryRuntime), patch(
            "reelcache.spectra.exact_spectrum", side_effect=RuntimeError("injected oracle failure"),
        ) as oracle:
            with self.assertRaisesRegex(RuntimeError, "injected oracle failure"):
                engine.run(request)
            oracle.assert_called_once()
        failure = json.loads((self.directory / "failed-oracle/failure.json").read_text())
        self.assertEqual(failure["status"], "abort_epoch")
        self.assertEqual(len(failure["clean_captures"]), 1)
        row = failure["clean_captures"][0]
        self.assertNotIn("spectrum", row)
        self.assertNotIn("numeric_observation", row)
        work = row["spectrum_work"]
        self.assertGreaterEqual(work["wall_seconds"], 0)
        self.assertIsNone(work["oracle_failure_metrics"])
        self.assertIsNone(work["oracle_timings_seconds"])
        self.assertIsNone(work["oracle_work_counts"])
        self.assertIn("outer_host_wall_only", work["timing_scope"])
        self.assertGreaterEqual(failure["elapsed_seconds"], work["wall_seconds"])
        receipt = row["spectrum_failure"]
        path = self.directory / receipt["path"]
        telemetry = json.loads(path.read_text())
        self.assertEqual(file_sha256(path), receipt["sha256"])
        self.assertEqual(telemetry["work"], work)
        self.assertEqual(telemetry["source_provenance"], row["source_provenance"])
        self.assertEqual(telemetry["status"], "failed")
        self.assertIsNone(telemetry["spectrum_sha256"])
        self.assertIn("injected oracle failure", telemetry["error"])
        self.assertFalse(list((self.directory / "failed-oracle/spectra").iterdir()))
        self.assertFalse((self.directory / "failed-oracle/result.json").exists())
        self.assertTrue(torch.equal(before_rng, torch.get_rng_state()))
        self.assertTrue(worker._native_cache_empty(self.pipe))
        self.assertEqual(MemoryRuntime.instances[-1].branch_info("request")["token_end"], 0)
        self.assertTrue(MemoryRuntime.instances[-1].closed)
        with self.assertRaisesRegex(StateError, "abort_epoch"):
            engine.run(self.request("cannot-reuse-failed-oracle"))

    def test_failed_oracle_preserves_actual_solver_counts_and_nested_timings(self):
        from reelcache.spectra import SpectrumNumericalError

        engine = self.engine()
        request = self.request("failed-solver")
        request["memory_budgets"]["spectrum_workspace_bytes"] = 16 << 20
        request["capture"] = {"spectra": True, "tile_rows": 3}
        before_rng = torch.get_rng_state().clone()
        with patch("reelcache.manager.KVRuntime", MemoryRuntime), patch(
            "torch.linalg.eigvalsh", side_effect=RuntimeError("injected eigvalsh failure"),
        ) as solver:
            with self.assertRaises(SpectrumNumericalError) as caught:
                engine.run(request)
            solver.assert_called_once()
        failure = json.loads((self.directory / "failed-solver/failure.json").read_text())
        row = failure["clean_captures"][0]
        work = row["spectrum_work"]
        metrics = worker.json_copy(caught.exception.metrics)
        self.assertEqual(work["oracle_failure_metrics"], metrics)
        self.assertTrue(metrics["operation_counts_complete"])
        self.assertEqual(metrics["decomposition_count"], 1)
        self.assertEqual(metrics["eigvalsh_count"], 1)
        self.assertEqual(metrics["eigh_count"], 0)
        self.assertEqual(work["oracle_work_counts"]["decomposition_count"], 1)
        self.assertEqual(work["oracle_timings_seconds"]["total_seconds"], metrics["total_seconds"])
        self.assertAlmostEqual(
            metrics["total_seconds"], metrics["exclusive_pipeline_seconds"] + metrics["eigensolve_seconds"],
        )
        self.assertLessEqual(metrics["total_seconds"], work["wall_seconds"])
        self.assertIn("not additional charges", work["timing_relationship"])
        telemetry = json.loads((self.directory / row["spectrum_failure"]["path"]).read_text())
        self.assertEqual(telemetry["work"], work)
        self.assertIsNone(telemetry["spectrum_sha256"])
        self.assertNotIn("spectrum", row)
        self.assertTrue(torch.equal(before_rng, torch.get_rng_state()))
        self.assertTrue(worker._native_cache_empty(self.pipe))
        self.assertEqual(MemoryRuntime.instances[-1].branch_info("request")["token_end"], 0)

    def test_numeric_identity_is_bound_before_capture_and_persisted_without_relabeling(self):
        model = worker.model_identity(self.asset_id)
        engine = worker.CausVidRankWorker(
            self.pipe, model, config=CausVidConfig(), output_root=self.directory,
            asset_id=self.asset_id, fps=16, conditioning_fn=self.encode_text,
            provenance={"protocol_id": "b" * 64}, observation_evidence_kind="synthetic_fixture",
        )
        self.addCleanup(engine.close)
        document = self.request("numeric")
        document["capture"] = {"spectra": True}
        document["memory_budgets"]["spectrum_workspace_bytes"] = 512 << 20
        document = worker.CausVidRankRequest.from_dict(document, asset_id=self.asset_id).document
        directory = self.directory / "numeric"
        directory.mkdir()
        runtime = MemoryRuntime(directory / "fake-runtime", model)
        runtime.create_branch("request")
        rows = []
        observe = engine._observer(document, directory, runtime, rows)
        context = TensorContext(model, MatrixKey(0, "K"), 0, 3, 30, 52, "pre_rope")
        observe(context, torch.zeros(context.shape, dtype=torch.bfloat16))
        self.assertEqual(len(rows), 1)
        numeric_path = self.directory / rows[0]["numeric_observation"]["path"]
        record = json.loads(numeric_path.read_text())["observation"]
        source = json.loads((self.directory / rows[0]["spectrum"]["path"]).read_text())["source"]
        for name, value in record["observation_identity"].items():
            self.assertEqual(source["provenance"][name], value)
        self.assertEqual(record["case_id"], document["case_id"])
        self.assertEqual(record["group_id"], document["prompt_group_id"])
        self.assertEqual(source["provenance"]["group_id"], document["prompt_group_id"])
        self.assertNotEqual(record["case_id"], document["request_id"])
        self.assertEqual(record["trajectory_id"], document["trajectory_id"])
        self.assertEqual(record["source_dtype"], "bfloat16")
        self.assertEqual(record["source_domain"], "normalized_pre_rope")
        self.assertEqual(record["evidence_kind"], "synthetic_fixture")
        self.assertTrue(record["fixture"])
        self.assertFalse(record["native_execution"])
        self.assertTrue(source["provenance"]["fixture"])
        self.assertFalse(source["provenance"]["native_execution"])
        self.assertIsNone(record["parent_head"])
        self.assertIsNone(source["provenance"]["history_head_id"])
        self.assertEqual(record["runtime_head_id"], runtime.branch_info("request")["head_id"])
        self.assertEqual(record["spectrum_sha256"], rows[0]["spectrum"]["sha256"])
        self.assertTrue((numeric_path.parent / record["spectrum_path"]).is_file())

    def test_native_numeric_observer_rejects_noncanonical_fake_context_before_oracle(self):
        engine = self.engine(
            provenance={"protocol_id": "b" * 64, "launch_closure_id": "c" * 64},
            observation_evidence_kind="native",
        )
        document = self.request("wrong-native-geometry")
        document["capture"] = {"spectra": True}
        document["memory_budgets"]["spectrum_workspace_bytes"] = 512 << 20
        document = worker.CausVidRankRequest.from_dict(document, asset_id=self.asset_id).document
        directory = self.directory / document["output_namespace"]
        directory.mkdir()
        runtime = MemoryRuntime(directory / "fake-runtime", self.model)
        runtime.create_branch("request")
        observe = engine._observer(document, directory, runtime, [])
        context = TensorContext(self.model, MatrixKey(0, "K"), 0, 3, 1, 2, "pre_rope")
        with patch("reelcache.spectra.exact_spectrum") as oracle:
            with self.assertRaisesRegex(IntegrityError, "before capture"):
                observe(context, torch.zeros(context.shape))
            oracle.assert_not_called()

    def test_native_history_hashes_actual_attention_and_same_committed_packets_in_both_modes(self):
        template = copy.deepcopy(self.pipe)
        required = 2 * self.noise.shape[1] * CONFIG.frame_tokens * CONFIG.channels * 4
        results, packet_fixture = {}, {}
        native_attention = torch.nn.functional.scaled_dot_product_attention
        global_rng = torch.get_rng_state().clone()
        plain, plain_runtime = self.adapter("plain", specs=CodecSpec("svd_hevc", rank=2))
        for _ in range(3):
            plain.step()
        expected_latents, expected_rng = plain.latent_state, plain.rng_state
        packet_fixture = {
            (chunk, key): value.clone() for (_branch, chunk, key), value in plain_runtime.packets.items()
        }
        plain.close()
        for mode in ("reload", "quality_hot"):
            with self.subTest(mode=mode):
                self.pipe = copy.deepcopy(template)
                records, operands, borrows = [], [], []

                def observe(record, tensor):
                    records.append(copy.deepcopy(record))
                    borrows.append(weakref.ref(tensor))
                    self.assertEqual(record["native_kv_sha256"], tensor_digest(tensor))
                    self.assertEqual(tuple(record["shape"]), tuple(tensor.shape))

                def attention(query, keys, values, *args, **kwargs):
                    operands.append({
                        target: tensor_digest(value.transpose(1, 2))
                        for target, value in (("K", keys), ("V", values))
                    })
                    return native_attention(query, keys, values, *args, **kwargs)

                adapter, runtime = self.adapter(
                    mode, specs=CodecSpec("svd_hevc", rank=2), history_mode=mode,
                    max_quality_history_bytes=required if mode == "quality_hot" else 0,
                    native_history_observer=observe,
                )
                seal = runtime.seal_chunk
                committed_native = adapter._committed_native
                native_temporaries = []

                @contextmanager
                def one_transient(*args, **kwargs):
                    self.assertTrue(all(ref() is None for ref in native_temporaries))
                    with committed_native(*args, **kwargs) as value:
                        native_temporaries.append(weakref.ref(value))
                        yield value

                def shared_packets(branch, chunk, matrices, contexts, specs, **kwargs):
                    result = seal(branch, chunk, matrices, contexts, specs, **kwargs)
                    for key in contexts:
                        packet = runtime.packets[(branch, chunk, key)]
                        self.assertTrue(torch.equal(packet, packet_fixture[(chunk, key)]))
                        runtime.packets[(branch, chunk, key)] = packet_fixture[(chunk, key)].clone()
                    return result

                with patch.object(runtime, "seal_chunk", side_effect=shared_packets), patch.object(
                    adapter, "_committed_native", one_transient,
                ), patch(
                    "torch.nn.functional.scaled_dot_product_attention", side_effect=attention,
                ):
                    for _ in range(3):
                        adapter.step()
                self.assertEqual(len(records), 24)
                self.assertEqual([row["event_ordinal"] for row in records], list(range(24)))
                reads = [row for row in records if row["event"] == "attention_read"]
                commits = [row for row in records if row["event"] == "clean_publication"]
                self.assertEqual(len(reads), 18)
                self.assertEqual(len(commits), 6)
                for row in reads:
                    self.assertEqual(
                        row["native_kv_sha256"], operands[row["forward_ordinal"] - 1][row["target"]],
                    )
                    self.assertEqual(row["native_kv_sha256"], row["after_attention_sha256"])
                    self.assertEqual(row["committed_head_id"], "manifest:" + f"{row['chunk_id']:064x}")
                    self.assertEqual(row["archived_token_end"], row["chunk_id"] * CONFIG.chunk_tokens)
                    self.assertEqual(row["phase"], "clean_forward" if row["forward_ordinal"] % 3 == 0 else "denoising")
                for row in commits:
                    key = MatrixKey(row["layer"], row["target"])
                    chunk = row["chunk_id"]
                    native = packet_fixture[(chunk, key)].reshape(1, CONFIG.chunk_tokens, CONFIG.heads, CONFIG.head_dim)
                    if key.target == "K":
                        self.assertEqual(row["packet_domain"], "pre_rope")
                        native = causal_rope_apply(
                            native, torch.tensor([[3, 1, 2]]), torch.tensor([0.3, 0.9], dtype=torch.float64),
                            start_frame=chunk * CONFIG.frames_per_chunk,
                        )
                        self.assertNotEqual(tensor_digest(native), tensor_digest(packet_fixture[(chunk, key)]))
                    self.assertEqual(row["decoded_current_sha256"], tensor_digest(native))
                    self.assertEqual(row["newest_token_start"], chunk * CONFIG.chunk_tokens)
                    self.assertEqual(row["newest_token_end"], (chunk + 1) * CONFIG.chunk_tokens)
                    self.assertEqual(row["committed_head_id"], "manifest:" + f"{chunk + 1:064x}")
                    if mode == "quality_hot":
                        self.assertEqual(row["mutable_current_sha256"], row["decoded_current_sha256"])
                        self.assertEqual(row["retained_current_sha256"], row["decoded_current_sha256"])
                memory = adapter.memory_report()
                diagnostic = memory["native_history_diagnostic"]
                self.assertTrue(diagnostic["timing_is_diagnostic_not_benchmark"])
                self.assertEqual(diagnostic["event_count"], 24)
                self.assertGreater(diagnostic["hash_readback_bytes"], 0)
                self.assertEqual(
                    diagnostic["peak_hash_host_workspace_bytes"],
                    2 * CONFIG.chunk_tokens * CONFIG.channels * 4 + 65536,
                )
                self.assertEqual(diagnostic["extra_committed_reload_decodes"], 6 if mode == "reload" else 0)
                self.assertFalse(memory["compressed_hbm_benchmark"])
                self.assertFalse(adapter._mutable)
                self.assertIsNone(adapter._active_layer)
                self.assertIsNone(adapter._clean_staging)
                self.assertEqual(runtime.active_leases, 0)
                self.assertTrue(all(ref() is None for ref in runtime.lease_refs + borrows))
                self.assertTrue(all(ref() is None for ref in native_temporaries))
                results[mode] = (records, adapter.latent_state, adapter.rng_state)
                self.assertTrue(torch.equal(adapter.latent_state, expected_latents))
                self.assertTrue(torch.equal(adapter.rng_state, expected_rng))
                adapter.close()
                self.assertFalse(adapter._native_committed_hashes)
                self.assertFalse(adapter._quality_history)
                self.assertFalse(adapter._quality_freqs)
        comparison_fields = (
            "event", "event_ordinal", "forward_ordinal", "phase", "layer", "target", "chunk_id",
            "token_start", "token_end", "committed_head_id", "native_kv_sha256", "archived_prefix_sha256",
        )
        self.assertEqual(
            [{name: row[name] for name in comparison_fields} for row in results["reload"][0]],
            [{name: row[name] for name in comparison_fields} for row in results["quality_hot"][0]],
        )
        self.assertTrue(torch.equal(results["reload"][1], results["quality_hot"][1]))
        self.assertTrue(torch.equal(results["reload"][2], results["quality_hot"][2]))
        self.assertTrue(torch.equal(global_rng, torch.get_rng_state()))

    def test_native_history_raw_keys_are_not_rotated_again(self):
        records = []
        adapter, runtime = self.adapter(native_history_observer=lambda row, _tensor: records.append(row))
        adapter.step()
        adapter.step()
        for row in records:
            if row["event"] == "clean_publication":
                key = MatrixKey(row["layer"], row["target"])
                self.assertEqual(
                    row["decoded_current_sha256"], tensor_digest(runtime.packets[("request", row["chunk_id"], key)]),
                )
                if key.target == "K":
                    self.assertEqual(row["packet_domain"], "post_rope")
        with self.assertRaisesRegex(UnsupportedError, "fresh run"):
            adapter.restore("unused", "replay")

    def test_native_history_newest_raw_slot_or_retained_corruption_cannot_pass_counters(self):
        required = 2 * self.noise.shape[1] * CONFIG.frame_tokens * CONFIG.channels * 4
        for damage in ("raw_current", "retained_current", "missing_measurement"):
            with self.subTest(damage=damage):
                adapter, runtime = self.adapter(
                    damage, specs=CodecSpec("svd", rank=2), history_mode="quality_hot",
                    max_quality_history_bytes=required, native_history_observer=lambda *_: None,
                )
                original = adapter._copy_committed_quality

                def corrupt(key, start, stop, freqs, *, replace_current):
                    pristine = adapter._mutable[key].clone()
                    measured = original(key, start, stop, freqs, replace_current=replace_current)
                    if damage == "raw_current":
                        adapter._mutable[key].copy_(pristine)
                    elif damage == "retained_current":
                        adapter._quality_history[key].tensor.data[:, :CONFIG.chunk_tokens].add_(1)
                    else:
                        return None
                    return measured

                with patch.object(adapter, "_copy_committed_quality", side_effect=corrupt):
                    with self.assertRaisesRegex(IntegrityError, "freshly decoded"):
                        adapter.step()
                self.assertEqual(adapter.memory_report()["committed_current_replacements"], 1)
                self.assertEqual(runtime.branch_info(damage)["token_end"], CONFIG.chunk_tokens)
                self.assertEqual(adapter._phase, "poisoned")
                self.assertFalse(adapter._mutable)
                self.assertIsNone(adapter._active_layer)
                self.assertIsNone(adapter._clean_staging)
                self.assertEqual(runtime.active_leases, 0)
                with self.assertRaises(StateError):
                    adapter.step()
                adapter.close()

    def test_native_history_detects_archived_data_mutation_before_attention(self):
        required = 2 * self.noise.shape[1] * CONFIG.frame_tokens * CONFIG.channels * 4
        adapter, runtime = self.adapter(
            specs=CodecSpec("svd", rank=2), history_mode="quality_hot",
            max_quality_history_bytes=required, native_history_observer=lambda *_: None,
        )
        adapter.step()
        entry = adapter._quality_history[MatrixKey(0, "K")]
        version = entry.tensor._version
        entry.tensor.data[:, :CONFIG.chunk_tokens].add_(1)
        self.assertEqual(entry.tensor._version, version)
        before, rng = runtime.branch_info("request"), adapter.rng_state
        with self.assertRaisesRegex(IntegrityError, "archived native attention operand"):
            adapter.step()
        self.assertEqual(runtime.branch_info("request"), before)
        self.assertTrue(torch.equal(rng, adapter.rng_state))
        self.assertIsNone(adapter._active_layer)
        self.assertEqual(runtime.active_leases, 0)

    def test_native_history_reload_postcommit_decode_failure_poisoning_and_cleanup(self):
        adapter, runtime = self.adapter(native_history_observer=lambda *_: None)
        runtime.fail_target = "V"
        with self.assertRaisesRegex(CapacityError, "reconstruction failure"):
            adapter.step()
        self.assertEqual(runtime.branch_info("request")["token_end"], CONFIG.chunk_tokens)
        self.assertEqual(adapter._phase, "poisoned")
        self.assertFalse(adapter._mutable)
        self.assertIsNone(adapter._active_layer)
        self.assertIsNone(adapter._clean_staging)
        self.assertEqual(runtime.active_leases, 0)
        adapter.close()
        self.assertFalse(adapter._native_committed_hashes)

    def test_native_history_detects_native_operand_mutation_during_attention(self):
        adapter, runtime = self.adapter(specs=CodecSpec("svd", rank=2), native_history_observer=lambda *_: None)
        adapter.step()
        before, rng = runtime.branch_info("request"), adapter.rng_state
        native = torch.nn.functional.scaled_dot_product_attention

        def corrupt(query, keys, values, *args, **kwargs):
            keys.data[:, :, :CONFIG.chunk_tokens].add_(1)
            return native(query, keys, values, *args, **kwargs)

        with patch("torch.nn.functional.scaled_dot_product_attention", side_effect=corrupt):
            with self.assertRaisesRegex(IntegrityError, "attention mutated"):
                adapter.step()
        self.assertEqual(runtime.branch_info("request"), before)
        self.assertTrue(torch.equal(rng, adapter.rng_state))
        self.assertFalse(adapter._mutable)
        self.assertIsNone(adapter._active_layer)
        self.assertEqual(runtime.active_leases, 0)

    def test_native_history_observer_is_synchronous_and_read_only(self):
        async def asynchronous(_row, _tensor):
            return None

        def mutate(_row, tensor):
            tensor.add_(1)

        def mutate_data(_row, tensor):
            tensor.data.add_(1)

        def mutate_record(row, _tensor):
            row["native_kv_sha256"] = "0" * 64

        for index, observer in enumerate((mutate, mutate_data, mutate_record, asynchronous)):
            with self.subTest(observer=observer.__name__):
                adapter, runtime = self.adapter(str(index), native_history_observer=observer)
                before, rng = runtime.branch_info(str(index)), adapter.rng_state
                with self.assertRaises((IntegrityError, UnsupportedError)):
                    adapter.step()
                self.assertEqual(runtime.branch_info(str(index)), before)
                self.assertTrue(torch.equal(rng, adapter.rng_state))
                self.assertFalse(adapter._mutable)
                self.assertIsNone(adapter._active_layer)
                self.assertIsNone(adapter._clean_staging)
                self.assertEqual(runtime.active_leases, 0)
                adapter.close()

    def test_native_history_worker_opt_in_is_phase_bound_and_disabled_by_default(self):
        plain = worker.CausVidRankRequest.from_dict(self.request(), asset_id=self.asset_id).document
        self.assertNotIn("native_history_audit", plain["capture"])
        accepted = self.request("audit", chunks=2)
        accepted.update(split="development", study_phase="native_preflight", capture={"native_history_audit": True})
        accepted = worker.CausVidRankRequest.from_dict(accepted, asset_id=self.asset_id).document
        plan = worker._output_reservation(accepted, "generate")
        self.assertGreater(plan["metadata_bytes"], worker._output_reservation(
            {**accepted, "capture": {}}, "generate",
        )["metadata_bytes"])
        for changes in (
            {"split": "calibration"}, {"study_phase": "calibration_frontier"}, {"chunks": 4},
            {"study_request": {"phase": "calibration_census"}},
            {"latent_source": {"path": "other/result.json", "sha256": "0" * 64}},
            {"reference_source": {"path": "other/result.json", "sha256": "0" * 64}},
            {"capture": {"native_history_audit": 1}},
        ):
            with self.subTest(changes=changes):
                with self.assertRaises(ConfigurationError):
                    worker.CausVidRankRequest.from_dict({**accepted, **changes}, asset_id=self.asset_id)
        engine = self.engine()
        with patch("reelcache.manager.KVRuntime", MemoryRuntime), patch.object(
            CausVidAdapter, "_measure_native", side_effect=AssertionError("disabled audit must do no readbacks"),
        ):
            result = engine.run(self.request("default"))
        self.assertNotIn("native_history_audit", result)
        self.assertFalse((self.directory / "default/native-history.jsonl").exists())
        self.assertFalse(result["chunks"][0]["memory"]["native_history_diagnostic"]["enabled"])

    def test_native_history_worker_streams_hash_bound_scalar_evidence_and_closes_it(self):
        document = self.request("audited", chunks=2, mode="quality_hot")
        document.update(split="development", study_phase="native_preflight", capture={"native_history_audit": True})
        document["spec_config"] = {target: {"kind": "svd", "rank": 2} for target in ("K", "V")}
        engine, opened = self.engine(), []
        original_open = Path.open

        def track_open(path, *args, **kwargs):
            stream = original_open(path, *args, **kwargs)
            if path.name == "native-history.jsonl":
                opened.append(stream)
            return stream

        with patch("reelcache.manager.KVRuntime", MemoryRuntime), patch.object(Path, "open", track_open):
            result = engine.run(document)
        audit = result["native_history_audit"]
        path = self.directory / audit["events"]["path"]
        records = [json.loads(line) for line in path.read_text().splitlines()]
        self.assertEqual(len(records), audit["event_count"])
        self.assertEqual(audit["event_count"], audit["expected_event_count"])
        self.assertEqual(audit["event_count"], 16)
        self.assertEqual(file_sha256(path), audit["events"]["sha256"])
        self.assertTrue(audit["fixture"])
        self.assertFalse(audit["native_execution"])
        self.assertFalse(audit["native_parity_claim"])
        self.assertIn("diagnostic", audit["timing_scope"])
        self.assertTrue(all(stream.closed for stream in opened))
        self.assertTrue(result["request_reset_verified"])
        self.assertEqual(json.loads((self.directory / "audited/result.json").read_text())["native_history_audit"], audit)
        runtime = MemoryRuntime.instances[-1]
        self.assertTrue(runtime.closed)
        self.assertEqual(runtime.active_leases, 0)
        self.assertTrue(all(ref() is None for ref in runtime.lease_refs))

    def test_native_history_worker_aborts_epoch_and_closes_partial_evidence_on_bad_publication(self):
        document = self.request("bad-audit", mode="quality_hot")
        document.update(split="development", study_phase="native_preflight", capture={"native_history_audit": True})
        document["spec_config"] = {target: {"kind": "svd", "rank": 2} for target in ("K", "V")}
        engine, opened = self.engine(), []
        original, original_open = CausVidAdapter._copy_committed_quality, Path.open

        def leave_raw(adapter, key, start, stop, freqs, *, replace_current):
            pristine = adapter._mutable[key].clone()
            measured = original(adapter, key, start, stop, freqs, replace_current=replace_current)
            adapter._mutable[key].copy_(pristine)
            return measured

        def track_open(path, *args, **kwargs):
            stream = original_open(path, *args, **kwargs)
            if path.name == "native-history.jsonl":
                opened.append(stream)
            return stream

        with patch("reelcache.manager.KVRuntime", MemoryRuntime), patch.object(
            CausVidAdapter, "_copy_committed_quality", leave_raw,
        ), patch.object(Path, "open", track_open):
            with self.assertRaisesRegex(IntegrityError, "freshly decoded"):
                engine.run(document)
        self.assertTrue(engine._poisoned)
        self.assertFalse((self.directory / "bad-audit/result.json").exists())
        failure = json.loads((self.directory / "bad-audit/failure.json").read_text())
        self.assertTrue(failure["native_history_audit"]["incomplete"])
        self.assertTrue(all(stream.closed for stream in opened))
        self.assertEqual(MemoryRuntime.instances[-1].branches["request"]["token_end"], CONFIG.chunk_tokens)
        self.assertTrue(MemoryRuntime.instances[-1].closed)
        self.assertTrue(worker._native_cache_empty(self.pipe))
        with self.assertRaisesRegex(StateError, "abort_epoch"):
            engine.run(self.request("cannot-reuse"))

    def test_quality_hot_decodes_committed_current_then_reuses_complete_history(self):
        original = copy.deepcopy(self.pipe)
        spec = CodecSpec("svd", rank=2)
        cold, cold_runtime = self.adapter("cold", specs=spec)
        cold.step()
        cold.step()
        expected, expected_rng = cold.latent_state, cold.rng_state
        cold.close()
        self.pipe = original
        required = 2 * self.noise.shape[1] * CONFIG.frame_tokens * CONFIG.channels * 4
        hot, runtime = self.adapter(
            "hot", specs=spec, history_mode="quality_hot", max_quality_history_bytes=required,
        )
        replacements = []
        replace = hot._copy_committed_quality

        def inspect_replacement(key, start, stop, freqs, *, replace_current):
            pristine = hot._mutable[key].clone() if replace_current else None
            replace(key, start, stop, freqs, replace_current=replace_current)
            if key.target == "V" and replace_current:
                committed = runtime.packets[("hot", start // CONFIG.frames_per_chunk, key)]
                self.assertTrue(torch.equal(hot._mutable[key][0].flatten(1), committed))
                self.assertFalse(torch.equal(pristine, hot._mutable[key]))
                replacements.append(key)

        with patch.object(hot, "_copy_committed_quality", side_effect=inspect_replacement):
            hot.step()
            hot.step()
        self.assertTrue(torch.equal(expected, hot.latent_state))
        self.assertTrue(torch.equal(expected_rng, hot.rng_state))
        self.assertEqual(len(replacements), 2)
        self.assertEqual(len(runtime.reads), 4)
        self.assertEqual(len(cold_runtime.reads), 6)
        self.assertEqual(runtime.active_leases, 0)
        self.assertTrue(all(reference() is None for reference in runtime.lease_refs))
        self.assertEqual(hot.memory_report()["live_quality_history_bytes"], required)
        self.assertEqual(hot.memory_report()["committed_current_replacements"], 4)
        self.assertEqual(hot.memory_report()["live_mutable_kv_bytes"], 0)
        self.assertFalse(hot.memory_report()["compressed_hbm_benchmark"])
        cache = hot._quality_history[MatrixKey(0, "V")].tensor[:, :2 * CONFIG.chunk_tokens]
        expected_packets = torch.cat([runtime.packets[("hot", chunk, MatrixKey(0, "V"))] for chunk in range(2)])
        self.assertTrue(torch.equal(cache[0].flatten(1), expected_packets))
        hot.close()
        self.assertEqual(hot.memory_report()["live_quality_history_bytes"], 0)

    def test_quality_budget_and_postcommit_decode_failure_are_fail_closed(self):
        required = 2 * self.noise.shape[1] * CONFIG.frame_tokens * CONFIG.channels * 4
        runtime = MemoryRuntime(self.directory / "budget", self.model)
        adapter = CausVidAdapter(
            self.pipe, runtime, "budget", config=CONFIG,
            history_mode="quality_hot", max_quality_history_bytes=required - 1,
        ).install()
        before = self.rng.get_state().clone()
        with self.assertRaises(CapacityError):
            adapter.start(self.noise, self.conditioning, self.rng, conditioning_id="fake")
        self.assertFalse(runtime.branches)
        self.assertTrue(torch.equal(before, self.rng.get_state()))
        adapter.close()
        adapter, runtime = self.adapter(
            "committed-failure", history_mode="quality_hot", max_quality_history_bytes=required,
        )
        runtime.fail_target = "V"
        with self.assertRaisesRegex(CapacityError, "reconstruction"):
            adapter.step()
        self.assertEqual(runtime.branch_info("committed-failure")["token_end"], CONFIG.chunk_tokens)
        self.assertEqual(adapter._phase, "poisoned")
        self.assertEqual(runtime.active_leases, 0)
        self.assertFalse(adapter._mutable)
        with self.assertRaises(StateError):
            adapter.step()

    def test_adapter_current_and_capture_budgets_reject_before_branch_or_rng(self):
        for options in ({"max_current_bytes": 1}, {"max_clean_host_bytes": 1}):
            runtime = MemoryRuntime(self.directory / "admission", self.model)
            adapter = CausVidAdapter(self.pipe, runtime, "admission", config=CONFIG, **options).install()
            before = self.rng.get_state().clone()
            try:
                with self.assertRaises(CapacityError):
                    adapter.start(self.noise, self.conditioning, self.rng, conditioning_id="fake")
                self.assertFalse(runtime.branches)
                self.assertTrue(torch.equal(before, self.rng.get_state()))
                self.assertFalse(adapter._mutable)
            finally:
                adapter.close()

    def test_quality_restore_allocation_does_not_double_its_history_budget(self):
        required = 2 * self.noise.shape[1] * CONFIG.frame_tokens * CONFIG.channels * 4
        adapter, _runtime = self.adapter(
            history_mode="quality_hot", max_quality_history_bytes=required,
        )
        adapter.step()
        prior = {key: (entry.tensor.data_ptr(), entry.valid_tokens) for key, entry in adapter._quality_history.items()}
        prepared = adapter._new_quality_storage(self.noise.shape[1])
        for key, entry in prepared.items():
            self.assertEqual(entry.tensor.data_ptr(), prior[key][0])
            self.assertEqual(entry.valid_tokens, 0)
            self.assertEqual(adapter._quality_history[key].valid_tokens, prior[key][1])
        self.assertLessEqual(adapter.memory_report()["peak_quality_history_bytes"], required)

    def test_quality_history_mutation_cannot_enter_future_attention(self):
        required = 2 * self.noise.shape[1] * CONFIG.frame_tokens * CONFIG.channels * 4
        adapter, runtime = self.adapter(
            history_mode="quality_hot", max_quality_history_bytes=required,
        )
        adapter.step()
        before, rng = runtime.branch_info("request"), adapter.rng_state
        adapter._quality_history[MatrixKey(0, "V")].tensor.add_(1)
        with self.assertRaisesRegex(IntegrityError, "mutated"):
            adapter.step()
        self.assertEqual(before, runtime.branch_info("request"))
        self.assertTrue(torch.equal(rng, adapter.rng_state))

    def rank_policy(self, **changes):
        from reelcache.online_rank import RankPolicy

        keys = tuple(MatrixKey(0, target) for target in ("K", "V"))
        policy = RankPolicy(
            policy_id="cpu-fake-policy", model=self.model,
            geometry=(CONFIG.frames_per_chunk, CONFIG.grid_height, CONFIG.grid_width),
            source_dtypes={"K": "float32", "V": "float32"},
            source_fingerprint="1" * 64, geometry_fingerprint="2" * 64,
            history_fingerprint="3" * 64, calibration_fingerprint="4" * 64,
            implementation_fingerprint="5" * 64, energy_target=0.5,
            template_ranks={key: 2.0 for key in keys},
            allowed_ranks={key: (1, 2, 4, 6) for key in keys},
            codec=CodecSpec("svd", rank=1), history_min=1, history_max=3,
            source_limit_bytes=1 << 20, workspace_limit_bytes=1 << 20,
            solver_workspace_bytes=0, oversample=0, max_fit_width=6, tile_rows=3,
        )
        return replace(policy, **changes)

    def test_wrong_policy_or_context_is_rejected_before_native_state_and_rng(self):
        policy = self.rank_policy()
        runtime = MemoryRuntime(self.directory / "wrong-policy", self.model)
        before = self.rng.get_state().clone()
        for invalid in (
            object(), replace(policy, geometry=(3, 2, 1)),
            replace(policy, model=replace(self.model, fingerprint="another-source")),
            replace(policy, source_dtypes={"K": "bfloat16", "V": "float32"}),
        ):
            with self.subTest(invalid=type(invalid)), self.assertRaises(ConfigurationError):
                CausVidAdapter(self.pipe, runtime, "bad", config=CONFIG, rank_policy=invalid)
        with self.assertRaisesRegex(ConfigurationError, "owns resolved specs"):
            CausVidAdapter(
                self.pipe, runtime, "mixed", config=CONFIG, rank_policy=policy, specs=CodecSpec("svd", 2),
            )
        self.assertTrue(torch.equal(before, self.rng.get_state()))
        self.assertFalse(runtime.branches)
        self.assertTrue(worker._native_cache_empty(self.pipe))

    def test_shared_context_set_validation_runs_before_start_and_step_mutation(self):
        runtime = MemoryRuntime(self.directory / "context-set", self.model)
        adapter = CausVidAdapter(
            self.pipe, runtime, "context-set", config=CONFIG, rank_policy=self.rank_policy(),
        ).install()
        self.addCleanup(adapter.close)
        context = adapter._context

        def inconsistent(key, **kwargs):
            value = context(key, **kwargs)
            return replace(value, token_start=value.token_start + 1) if key.target == "V" else value

        before = self.rng.get_state().clone()
        with patch.object(adapter, "_context", side_effect=inconsistent):
            with self.assertRaisesRegex(ConfigurationError, "one clean token range"):
                adapter.start(self.noise, self.conditioning, self.rng, conditioning_id="fake")
        self.assertFalse(runtime.branches)
        self.assertTrue(torch.equal(before, self.rng.get_state()))
        adapter.start(self.noise, self.conditioning, self.rng, conditioning_id="fake")
        before_rng, before_head = adapter.rng_state, runtime.branch_info("context-set")
        with patch.object(adapter, "_context", side_effect=inconsistent):
            with self.assertRaisesRegex(ConfigurationError, "one clean token range"):
                adapter.step()
        self.assertTrue(torch.equal(before_rng, adapter.rng_state))
        self.assertEqual(before_head, runtime.branch_info("context-set"))
        self.assertFalse(adapter._mutable)

    def test_dynamic_policy_commits_and_forks_with_stable_source_domain(self):
        from reelcache.manager import KVRuntime

        policy = self.rank_policy()
        with KVRuntime(self.directory / "real-adaptive-store", self.model, max_resident_bytes=1 << 20) as runtime:
            with CausVidAdapter(
                self.pipe, runtime, "parent", config=CONFIG, rank_policy=policy,
            ) as adapter:
                self.assertFalse(adapter.specs)
                adapter.start(self.noise, self.conditioning, self.rng, conditioning_id="rank-fake")
                adapter.step()
                snapshot = adapter.snapshot()
                parent = runtime.branch_info("parent")
                adapter.fork(snapshot, "child")
                adapter.step()
                self.assertEqual(parent, runtime.branch_info("parent"))
                self.assertEqual(runtime.history_rank_state("child").completed_chunks, 2)
                for key in adapter.keys:
                    self.assertEqual(
                        {context.key_space for context in runtime.history_contexts("child", key)},
                        {"pre_rope" if key.target == "K" else "none"},
                    )
                    self.assertEqual(len(runtime.history_rank_decisions("child", key)), 2)
                before_rng, before_child = adapter.rng_state, runtime.branch_info("child")
                adapter.rank_policy = replace(policy, policy_id="wrong-requested-policy")
                with patch.object(runtime, "fork", wraps=runtime.fork) as fork:
                    with self.assertRaises(ConfigurationError):
                        adapter.restore(snapshot, "rejected")
                    fork.assert_not_called()
                self.assertTrue(torch.equal(before_rng, adapter.rng_state))
                self.assertEqual(before_child, runtime.branch_info("child"))
                adapter.rank_policy = policy
                with self.assertRaises(TypeError):
                    adapter.specs[MatrixKey(0, "K")] = CodecSpec()
            with CausVidAdapter(self.pipe, runtime, "static", config=CONFIG) as incompatible:
                refs_before = runtime.store.list_refs()
                global_rng_before = torch.get_rng_state().clone()
                with patch.object(runtime, "fork", wraps=runtime.fork) as fork, patch(
                    "reelcache.manager.deserialize_tensors", side_effect=AssertionError("snapshot tensors read too early"),
                ) as deserialize, patch.object(
                    incompatible, "_prepare_restore", side_effect=AssertionError("native validator ran too early"),
                ) as prepare:
                    with self.assertRaises(ConfigurationError):
                        incompatible.restore(snapshot, "static-cannot-adopt-ranks")
                    fork.assert_called_once()
                    deserialize.assert_not_called()
                    prepare.assert_not_called()
                self.assertEqual(runtime.store.list_refs(), refs_before)
                self.assertTrue(torch.equal(global_rng_before, torch.get_rng_state()))
                self.assertEqual(incompatible.branch_id, "static")
                self.assertEqual(incompatible.frame_cursor, 0)
                self.assertIsNone(incompatible._rng)
                self.assertFalse(incompatible._conditioning)
                self.assertFalse(runtime._pending)
                self.assertEqual((runtime.active_leases, runtime.resident_bytes), (0, 0))

    def test_restore_passes_expected_policy_to_core_before_native_preparation(self):
        adapter, runtime = self.adapter("early-restore", rank_policy=self.rank_policy())
        before_rng, before_head = adapter.rng_state, runtime.branch_info("early-restore")
        before_phase = adapter._phase

        def reject(_snapshot, _branch, *, rank_policy, metadata_validator, validator):
            self.assertEqual(rank_policy.fingerprint, adapter.rank_policy.fingerprint)
            self.assertTrue(callable(validator))
            self.assertTrue(callable(metadata_validator))
            raise ConfigurationError("injected actual snapshot policy mismatch")

        with patch.object(runtime, "fork", side_effect=reject, create=True) as fork, patch.object(
            runtime, "load_snapshot", side_effect=AssertionError("must not deserialize before policy check"),
            create=True,
        ) as load, patch.object(adapter, "_prepare_restore", wraps=adapter._prepare_restore) as prepare:
            with self.assertRaisesRegex(ConfigurationError, "actual snapshot policy mismatch"):
                adapter.restore("foreign-snapshot", "rejected", seed=1234)
            fork.assert_called_once()
            load.assert_not_called()
            prepare.assert_not_called()
        self.assertTrue(torch.equal(before_rng, adapter.rng_state))
        self.assertEqual(before_head, runtime.branch_info("early-restore"))
        self.assertEqual(before_phase, adapter._phase)
        self.assertNotIn("rejected", runtime.branches)

    def test_restore_ignores_forged_capsule_policy_and_checks_verified_contract_first(self):
        policy = self.rank_policy()
        foreign = self.rank_policy(policy_id="foreign-committed-policy")
        for name, expected, actual, error in (
            ("static", None, policy, ConfigurationError),
            ("foreign", policy, foreign, ConfigurationError),
            ("missing-state", policy, policy, IntegrityError),
        ):
            with self.subTest(name=name):
                adapter, runtime = self.adapter(name, rank_policy=expected)
                before_rng, before_head = adapter.rng_state, runtime.branch_info(name)
                saved = SimpleNamespace(
                    rank_policy=actual, rank_state=None,
                    metadata={"binding": adapter._binding()}, tensors={},
                )

                def validate_only(_snapshot, _branch, *, rank_policy, metadata_validator, validator):
                    self.assertIs(rank_policy, adapter.rank_policy)
                    metadata_validator(saved)
                    validator(saved)
                    self.fail("A forged or incomplete committed contract reached publication")

                with patch.object(runtime, "fork", side_effect=validate_only, create=True), patch.object(
                    adapter, "_prepare_conditioning", side_effect=AssertionError("native preparation is premature"),
                ) as prepare:
                    with self.assertRaises(error):
                        adapter.restore("forged-capsule", "rejected", seed=1234)
                    prepare.assert_not_called()
                self.assertTrue(torch.equal(before_rng, adapter.rng_state))
                self.assertEqual(before_head, runtime.branch_info(name))
                self.assertNotIn("rejected", runtime.branches)
                adapter.close()

    def test_restore_validates_native_cursor_against_typed_committed_rank_state(self):
        from reelcache.manager import RestoredState
        from reelcache.online_rank import RankState

        source, _runtime = self.adapter("snapshot-source")
        source.step()
        native_metadata = source._metadata()
        tensors = {
            "noise": source._noise.clone(), "latents": source.latent_state,
            "rng_state": source.rng_state, "prompt_embeds": source._conditioning["prompt_embeds"].clone(),
        }
        source.close()
        for policy in (None, self.rank_policy()):
            with self.subTest(adaptive=policy is not None):
                runtime = MemoryRuntime(self.directory / "restore-target", self.model)
                adapter = CausVidAdapter(
                    self.pipe, runtime, "idle", config=CONFIG, rank_policy=policy,
                ).install()
                self.addCleanup(adapter.close)
                metadata = copy.deepcopy(native_metadata)
                metadata["binding"] = adapter._binding()
                metadata["clean_boundary"]["chunk_id"] = adapter._chunk_id(0)
                state = None if policy is None else RankState(
                    policy_hash=policy.fingerprint, last_chunk_id=adapter._chunk_id(0),
                    token_end=CONFIG.chunk_tokens, completed_chunks=1,
                    anchor_ranks={key: 2.0 for key in policy.anchors},
                    anchor_chunks={key: 1 for key in policy.anchors},
                    selected_ranks={key: 2 for key in policy.keys},
                )
                saved = RestoredState("snapshot", "child", metadata, tensors, policy, state)

                def publish(_snapshot, branch, *, rank_policy, metadata_validator, validator):
                    self.assertIs(rank_policy, adapter.rank_policy)
                    self.assertNotIn(branch, runtime.branches)
                    metadata_validator(replace(saved, tensors={}))
                    validator(saved)
                    runtime.branches[branch] = {
                        "head_id": "manifest:" + "1" * 64, "token_end": CONFIG.chunk_tokens, "chunks": 1,
                    }
                    return saved

                with patch.object(runtime, "fork", side_effect=publish, create=True) as fork, patch.object(
                    runtime, "load_snapshot", side_effect=AssertionError("restore should be a single guarded fork"),
                    create=True,
                ) as load:
                    adapter.restore("snapshot", "child")
                    fork.assert_called_once()
                    load.assert_not_called()
                    self.assertEqual(adapter.frame_cursor, CONFIG.frames_per_chunk)
                    self.assertTrue(torch.equal(adapter.rng_state, tensors["rng_state"]))
                    self.assertTrue(torch.equal(adapter._noise, tensors["noise"]))
                    self.assertEqual(adapter.branch_id, "child")
                    if state is not None:
                        saved = replace(saved, rank_state=replace(state, token_end=1))
                        before_rng = adapter.rng_state
                        with patch.object(
                            adapter, "_prepare_conditioning", side_effect=AssertionError("RNG must stay untouched"),
                        ) as prepare:
                            with self.assertRaisesRegex(IntegrityError, "token boundary mismatch"):
                                adapter.restore("snapshot", "wrong-cursor", seed=99)
                            prepare.assert_not_called()
                        self.assertTrue(torch.equal(before_rng, adapter.rng_state))
                        self.assertEqual(adapter.branch_id, "child")
                        self.assertNotIn("wrong-cursor", runtime.branches)
                adapter.close()

    def test_worker_rejects_foreign_policy_before_conditioning_and_output(self):
        engine = self.engine(rank_policy_factory=worker.build_rank_policy)
        request = self.request()
        request["rank_policy"] = self.rank_policy(geometry=(3, 2, 1)).to_dict()
        request["spec_config"] = None
        before = torch.get_rng_state().clone()
        with self.assertRaises(ConfigurationError):
            engine.run(request)
        self.assertEqual(self.texts, [])
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertFalse((self.directory / "a").exists())

    def test_exact_reference_cannot_duplicate_the_oracle_in_its_observer(self):
        engine = self.engine(rank_policy_factory=worker.build_rank_policy)
        request = self.request()
        request["rank_policy"] = self.rank_policy(solver="exact_oracle", anchor_statistic=None).to_dict()
        request["spec_config"] = None
        request["capture"] = {"spectra": True}
        request["memory_budgets"]["spectrum_workspace_bytes"] = 16 << 20
        before = torch.get_rng_state().clone()
        with self.assertRaisesRegex(ConfigurationError, "single spectrum solve"):
            engine.run(request)
        self.assertEqual(self.texts, [])
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertFalse((self.directory / "a").exists())
        self.assertTrue(worker._native_cache_empty(self.pipe))

    def test_committed_policy_oracle_export_reuses_exact_result_without_relabeling(self):
        from reelcache.spectra import SpectrumSource, exact_spectrum

        engine = self.engine()
        policy = self.rank_policy(solver="exact_oracle", anchor_statistic=None, max_fit_width=8)
        decisions, captures, originals = {}, [], {}
        for key in policy.keys:
            context = TensorContext(self.model, key, 0, 3, 1, 2, "pre_rope" if key.target == "K" else "none")
            tensor = torch.arange(48, dtype=torch.float32).reshape(context.shape) / 100
            digest = tensor_digest(tensor)
            source = SpectrumSource(
                identity=f"generic-policy-fixture-{key.target}", shape=context.shape, dtype=context.dtype,
                target=key.target, key_space=context.key_space,
                provenance={
                    "context": context.to_dict(), "policy_hash": policy.fingerprint,
                    "chunk_id": "chunk-000000", "parent_head": None,
                    "original_bytes_sha256": digest, "fixture": True, "native_execution": False,
                },
            )
            result = exact_spectrum(tensor, source=source, return_directions=True, tile_rows=3)
            originals[key] = result.to_dict()
            decisions[key] = SimpleNamespace(
                context=context, policy_hash=policy.fingerprint, input_sha256=digest,
                chunk_id="chunk-000000", oracle_observation=originals[key],
            )
            captures.append({
                "context": replace(context, source_dtype="float32").to_dict(),
                "source_provenance": {
                    "raw_native_tensor_sha256": digest, "case_id": "explicit-fixture-case",
                    "fixture": True, "native_execution": False,
                },
            })
        runtime = SimpleNamespace(
            history_rank_decisions=lambda _branch, key: (
                decisions[key], SimpleNamespace(chunk_id="chunk-000001"),
            ),
        )
        directory = self.directory / "policy-export"
        directory.mkdir()
        with patch("reelcache.spectra.exact_spectrum", side_effect=AssertionError("second eigensolve forbidden")):
            receipt = engine._export_policy_oracles(
                self.request("policy-export"), directory, runtime, policy, 0, captures,
            )
        index_path = self.directory / receipt["path"]
        index = json.loads(index_path.read_text())
        self.assertEqual(index["additional_eigensolves"], 0)
        self.assertFalse(index["study_observation_eligible"])
        self.assertFalse(index["source_provenance_rewritten"])
        for record in index["records"]:
            key = MatrixKey.from_dict(record["key"])
            path = self.directory / record["spectrum"]["path"]
            self.assertEqual(json.loads(path.read_text()), originals[key])
            self.assertEqual(file_sha256(path), record["spectrum"]["sha256"])
            self.assertNotIn("case_id", originals[key]["source"]["provenance"])
            self.assertEqual(record["prebound_capture"]["case_id"], "explicit-fixture-case")
            self.assertFalse(record["study_observation_eligible"])
        self.assertEqual(file_sha256(index_path), receipt["sha256"])
        for name, candidates in (
            ("missing", lambda key: (SimpleNamespace(chunk_id="chunk-000001"),)),
            ("duplicate", lambda key: (decisions[key], decisions[key])),
        ):
            rejected = self.directory / ("policy-" + name)
            rejected.mkdir()
            runtime.history_rank_decisions = lambda _branch, key, choose=candidates: choose(key)
            with self.subTest(name=name), self.assertRaisesRegex(IntegrityError, "exact chunk ID"):
                engine._export_policy_oracles(self.request(name), rejected, runtime, policy, 0, captures)
            self.assertFalse(list(rejected.rglob("*.json")))

    def test_rank_provenance_provider_is_adaptive_only_and_synchronous(self):
        runtime = MemoryRuntime(self.directory / "bad-provenance-provider", self.model)
        with self.assertRaisesRegex(ConfigurationError, "requires an adaptive policy"):
            CausVidAdapter(
                self.pipe, runtime, "static", config=CONFIG, rank_source_provenance=lambda _: {},
            )

        async def asynchronous(_context):
            return {}

        for name, provider, error in (
            ("not-object", lambda _: [], ConfigurationError),
            ("asynchronous", asynchronous, UnsupportedError),
            ("not-finite", lambda _: {"invalid": float("inf")}, ConfigurationError),
        ):
            with self.subTest(name=name):
                adapter, runtime = self.adapter(
                    name, rank_policy=self.rank_policy(), rank_source_provenance=provider,
                )
                before_rng, before_head = adapter.rng_state, runtime.branch_info(name)
                with patch.object(
                    runtime, "seal_chunk", side_effect=AssertionError("invalid provenance reached core"),
                ) as seal:
                    with self.assertRaises(error):
                        adapter.step()
                    seal.assert_not_called()
                self.assertTrue(torch.equal(before_rng, adapter.rng_state))
                self.assertEqual(before_head, runtime.branch_info(name))
                self.assertFalse(adapter._mutable)
                adapter.close()

    def test_prebound_policy_source_reaches_analysis_writer_with_no_second_solve(self):
        from reelcache import rank_analysis, spectra, study
        from reelcache.online_rank import RankPreparation

        cell = study.StudyCell(
            "synthetic-causvid-cpu", "synthetic", 8, 1, 1, 0,
            CONFIG.frames_per_chunk, CONFIG.grid_height, CONFIG.grid_width,
            self.model.layers, CONFIG.channels, "float32", "float32", 1,
        )
        catalog = {**study.CELLS, cell.cell_id: cell}
        engine = self.engine(
            provenance={"protocol_id": "b" * 64, "launch_closure_id": "c" * 64},
            observation_evidence_kind="synthetic_fixture",
        )
        document = worker.CausVidRankRequest.from_dict(
            self.request("prebound-policy"), asset_id=self.asset_id,
        ).document
        document["cell_id"] = cell.cell_id
        directory = self.directory / document["output_namespace"]
        directory.mkdir()
        policy = self.rank_policy(solver="exact_oracle", anchor_statistic=None)
        original_policy = policy.to_dict()
        runtime = MemoryRuntime(directory / "runtime", self.model)
        captures, decisions = [], {}
        observe = engine._observer(document, directory, runtime, captures)
        adapter = CausVidAdapter(
            self.pipe, runtime, "request", config=CONFIG, rank_policy=policy,
            clean_observer=observe,
            rank_source_provenance=lambda context: engine._rank_source_provenance(context, captures),
        ).install()
        self.addCleanup(adapter.close)
        adapter.start(self.noise, self.conditioning, self.rng, conditioning_id=document["conditioning_identity"])
        before_rng = torch.get_rng_state().clone()

        def seal(branch, chunk, matrices, contexts, *, clean, rank_policy, source_provenance):
            self.assertTrue(clean)
            self.assertEqual(rank_policy.fingerprint, policy.fingerprint)
            self.assertEqual(set(source_provenance), set(policy.keys))
            with RankPreparation(
                rank_policy, chunk_id=chunk, parent_head=None, token_start=0,
                source_provenance=source_provenance,
            ) as preparation:
                for key in policy.keys:
                    self.assertEqual(
                        source_provenance[key], engine._rank_source_provenance(contexts[key], captures),
                    )
                    self.assertEqual(source_provenance[key]["group_id"], document["prompt_group_id"])
                    prepared = preparation.prepare_matrix(matrices[key], contexts[key])
                    decisions[key] = prepared.decision
                self.assertEqual(preparation.next_state().completed_chunks, 1)
            runtime.branches[branch] = {
                "head_id": "manifest:" + "1" * 64, "token_end": CONFIG.chunk_tokens, "chunks": 1,
            }
            return "manifest:" + "2" * 64

        with patch.object(study, "CELLS", catalog), patch.object(rank_analysis, "CELLS", catalog):
            with patch.object(runtime, "seal_chunk", side_effect=seal), patch(
                "reelcache.spectra.exact_spectrum", wraps=spectra.exact_spectrum,
            ) as oracle, patch("torch.linalg.eigh", wraps=torch.linalg.eigh) as eigh:
                adapter.step()
                self.assertEqual(oracle.call_count, len(policy.keys))
                self.assertEqual(eigh.call_count, len(policy.keys))
            originals = {key: decisions[key].to_dict()["metrics"]["oracle_observation"] for key in policy.keys}
            with patch.object(
                runtime, "history_rank_decisions", side_effect=lambda _branch, key: (decisions[key],), create=True,
            ), patch("reelcache.spectra.exact_spectrum", side_effect=AssertionError("duplicate oracle")), patch(
                "torch.linalg.eigh", side_effect=AssertionError("duplicate eigh"),
            ), patch("torch.linalg.svd", side_effect=AssertionError("duplicate SVD")):
                receipt = engine._export_policy_oracles(document, directory, runtime, policy, 0, captures)
                index = json.loads((self.directory / receipt["path"]).read_text())
                self.assertTrue(index["study_observation_eligible"])
                self.assertEqual(index["evidence_kind"], "synthetic_fixture")
                self.assertEqual(index["additional_eigensolves"], 0)
                self.assertFalse(index["source_provenance_rewritten"])
                for record, capture in zip(index["records"], captures):
                    key = MatrixKey.from_dict(record["key"])
                    spectrum_path = self.directory / record["spectrum"]["path"]
                    observation_path = self.directory / record["numeric_observation"]["path"]
                    self.assertEqual(json.loads(spectrum_path.read_text()), originals[key])
                    self.assertEqual(file_sha256(spectrum_path), record["spectrum"]["sha256"])
                    self.assertEqual(file_sha256(observation_path), record["numeric_observation"]["sha256"])
                    numeric = json.loads(observation_path.read_text())["observation"]
                    self.assertEqual(numeric["observation_identity"], capture["observation_identity"])
                    self.assertEqual(capture["numeric_observation"], record["numeric_observation"])
                    self.assertEqual(numeric["case_id"], document["case_id"])
                    self.assertEqual(numeric["group_id"], document["prompt_group_id"])
                    self.assertEqual(
                        originals[key]["source"]["provenance"]["group_id"], document["prompt_group_id"],
                    )
                    self.assertTrue(numeric["fixture"])
                    self.assertFalse(numeric["native_execution"])
                    self.assertEqual(
                        originals[key]["source"]["identity"], capture["observation_identity"]["source_identity"],
                    )
                    for name, value in capture["observation_identity"].items():
                        self.assertEqual(originals[key]["source"]["provenance"][name], value)
                for name, changes in (
                    ("wrong-case", {"case_id": "retroactive-case"}),
                    ("wrong-geometry", {"n_channels": CONFIG.channels + 1}),
                ):
                    bad_captures = copy.deepcopy(captures)
                    bad_captures[0]["observation_identity"].update(changes)
                    rejected = self.directory / name
                    rejected.mkdir()
                    with self.assertRaises((IntegrityError, rank_analysis.AnalysisError)):
                        engine._export_policy_oracles(document, rejected, runtime, policy, 0, bad_captures)
                    self.assertFalse(list(rejected.rglob("*.json")))
        self.assertEqual(policy.to_dict(), original_policy)
        self.assertTrue(torch.equal(before_rng, torch.get_rng_state()))

    def test_latent_constructor_never_calls_native_vae_or_changes_globals(self):
        calls = []

        def get_diffusion_wrapper(**_kwargs):
            calls.append("generator")
            return lambda _path: object()

        def get_text_encoder_wrapper(**_kwargs):
            calls.append("text")
            return lambda _path: object()

        def get_vae_wrapper(**_kwargs):
            calls.append("vae")
            return lambda _path: object()

        namespace = dict(
            get_diffusion_wrapper=get_diffusion_wrapper,
            get_text_encoder_wrapper=get_text_encoder_wrapper, get_vae_wrapper=get_vae_wrapper,
        )
        exec(
            "def constructor(self, config, device):\n"
            "    self.generator = get_diffusion_wrapper(model_name='causal_wan')('local')\n"
            "    self.text_encoder = get_text_encoder_wrapper(model_name='causal_wan')('local')\n"
            "    self.vae = get_vae_wrapper(model_name='causal_wan')('local')\n",
            namespace,
        )
        native_class = type("FakeNativeConstructor", (), {"__init__": namespace["constructor"]})
        original = native_class.__init__.__globals__["get_vae_wrapper"]
        pipeline = worker.build_latent_pipeline(SimpleNamespace(InferencePipeline=native_class), None, device="cpu")
        self.assertIsNone(pipeline.vae)
        self.assertEqual(calls, ["generator", "text"])
        self.assertIs(native_class.__init__.__globals__["get_vae_wrapper"], original)

    def test_gates_and_worker_help_are_model_free(self):
        arguments = [
            "--protocol", "unused", "--assets", "unused", "--asset-map", "causvid480.json",
            "--launch-receipt", "unused", "--requests", "unused", "--output-root", str(self.directory),
            "--allow-cuda",
        ]
        with patch.object(worker, "verify_worker_inputs", side_effect=IntegrityError("stale receipt")), patch.object(
            worker, "NativeCausVidEpoch",
        ) as factory:
            with self.assertRaisesRegex(IntegrityError, "stale receipt"):
                worker.main(arguments)
            factory.assert_not_called()
        with self.assertRaisesRegex(IntegrityError, "executed worker"):
            worker._executed_closure({"source_closure": {"files": {}}})
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1",
                   TMPDIR=str(self.directory), OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
        code = (
            "import sys; import examples.causvid_rank_worker; "
            "assert 'torch' not in sys.modules; "
            "assert not any(k.startswith('third_party.causvid') for k in sys.modules)"
        )
        subprocess.run([sys.executable, "-B", "-c", code], cwd=ROOT, env=env, check=True, capture_output=True)
        process = subprocess.run(
            [sys.executable, "-B", "examples/native_causvid.py", "worker", "--help"],
            cwd=ROOT, env=env, check=True, capture_output=True, text=True,
        )
        self.assertIn("--launch-receipt", process.stdout)
        self.assertIn("--asset-map", process.stdout)
        self.assertFalse(torch.cuda.is_initialized())

    def test_every_launch_request_format_must_pass_shared_batch_gate(self):
        internal = self.request()
        with (
            patch.object(worker, "read_json", side_effect=[{}, {}, internal]),
            patch.object(worker, "validate_protocol"),
            patch.object(worker, "verify_launch_receipt"),
            patch.object(worker, "_executed_closure"),
            patch.object(worker.CausVidAssets, "verified", return_value=SimpleNamespace(asset_id=self.asset_id)),
            patch.object(
                worker, "validate_request_batch", side_effect=IntegrityError("shared batch gate rejected"),
            ) as batch_gate,
            patch.object(worker, "NativeCausVidEpoch") as native,
        ):
            with self.assertRaisesRegex(IntegrityError, "shared batch gate"):
                worker.verify_worker_inputs(
                    protocol_path="protocol.v2.json", assets_path="assets.json", asset_map="causvid480.json",
                    launch_receipt_path="receipt.json", requests_path="family-specific-request.json",
                    phase="calibration_census", worker_budgets=self.budgets(),
                )
            batch_gate.assert_called_once_with(internal, {}, {}, cell_id="causvid480", phase="calibration_census")
            native.assert_not_called()

    def test_native_import_seam_inspects_namespace_and_regular_packages_without_importing(self):
        names = (
            "third_party", "third_party.causvid", "third_party.causvid.modeling",
            "third_party.causvid.modeling.causal_inference",
        )
        for package_depths in ((), (0,), (2,), (0, 1, 2)):
            with self.subTest(package_depths=package_depths), patch.dict(sys.modules):
                for name in list(sys.modules):
                    if name == "third_party" or name.startswith("third_party."):
                        del sys.modules[name]
                root = self.directory / ("import-seam-" + str(package_depths))
                leaf = root / "third_party/causvid/modeling/causal_inference.py"
                leaf.parent.mkdir(parents=True)
                leaf.write_text("raise AssertionError('native module must not execute during inspection')\n")
                for depth in package_depths:
                    package = root.joinpath(*names[depth].split("."))
                    (package / "__init__.py").write_text(
                        "raise AssertionError('package must not execute during inspection')\n",
                    )
                with patch.object(sys, "path", [str(root)]):
                    worker._verified_import_seam(root)
                self.assertTrue(all(name not in sys.modules for name in names))

    def test_native_import_seam_rejects_missing_and_foreign_namespace_paths(self):
        root, foreign = self.directory / "seam-root", self.directory / "seam-foreign"
        leaf = root / "third_party/causvid/modeling/causal_inference.py"
        leaf.parent.mkdir(parents=True)
        leaf.write_text("raise AssertionError('inspection must not import')\n")
        (foreign / "third_party").mkdir(parents=True)
        with patch.dict(sys.modules):
            for name in list(sys.modules):
                if name == "third_party" or name.startswith("third_party."):
                    del sys.modules[name]
            with patch.object(sys, "path", [str(root), str(foreign)]):
                with self.assertRaisesRegex(IntegrityError, "unverified source roots"):
                    worker._verified_import_seam(root)
            with patch.object(sys, "path", [str(root)]):
                sys.modules["third_party.causvid"] = SimpleNamespace(__path__=[str(foreign)])
                with self.assertRaisesRegex(IntegrityError, "pre-imported namespace"):
                    worker._verified_import_seam(root)
                del sys.modules["third_party.causvid"]
                sys.modules["third_party.causvid.modeling.causal_inference"] = SimpleNamespace(
                    __file__=str(foreign / "causal_inference.py"),
                )
                with self.assertRaisesRegex(IntegrityError, "pre-imported namespace"):
                    worker._verified_import_seam(root)
                del sys.modules["third_party.causvid.modeling.causal_inference"]
                leaf.unlink()
                with self.assertRaisesRegex(IntegrityError, "does not resolve"):
                    worker._verified_import_seam(root)

    def test_native_import_seam_rejects_a_leaf_resolving_outside_the_verified_root(self):
        root = self.directory / "seam-symlink-root"
        leaf = root / "third_party/causvid/modeling/causal_inference.py"
        leaf.parent.mkdir(parents=True)
        foreign = self.directory / "unverified.py"
        foreign.write_text("raise AssertionError('unverified module must not execute')\n")
        leaf.symlink_to(foreign)
        with patch.dict(sys.modules), patch.object(sys, "path", [str(root)]):
            for name in list(sys.modules):
                if name == "third_party" or name.startswith("third_party."):
                    del sys.modules[name]
            with self.assertRaisesRegex(IntegrityError, "unverified source path"):
                worker._verified_import_seam(root)

    def test_asset_mapping_rejects_unverified_or_original_path_fallback(self):
        root = self.directory / "assets"
        files = {
            f"source/third_party/causvid/modeling/{name}": "# fake source, never imported\n"
            for name in ("causal_model.py", "causal_inference.py", "wan_wrapper.py", "attention.py", "vae.py")
        }
        files["base/config.json"] = json.dumps({
            "model_type": "t2v", "dim": 1536, "num_heads": 12, "num_layers": 30, "in_dim": 16, "out_dim": 16,
        })
        files["config.yaml"], files["checkpoint.bin"] = "fake: true", "fake-not-native-weights"
        mapping = {
            "schema": worker.ASSET_MAP_SCHEMA, "cell_id": "causvid480", "source_root": "source",
            "base_model": "base", "checkpoint": "checkpoint.bin", "config": "config.yaml", "fps": 16,
        }
        files["causvid480.json"] = json.dumps(mapping)
        records = []
        for name, text in files.items():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
            records.append({"name": name, "bytes": path.stat().st_size, "sha256": file_sha256(path)})
        body = {"schema": ASSET_SCHEMA, "files": records, "total_bytes": sum(row["bytes"] for row in records)}
        (root / "assets.json").write_text(json.dumps({**body, "asset_id": identity(body)}))
        closure = {
            "source_closure": {"files": {row["name"]: {"sha256": row["sha256"]} for row in records}}
        }
        assets = worker.CausVidAssets.verified(root / "assets.json", "causvid480.json", launch_receipt=closure)
        self.assertEqual(assets.paths["checkpoint"], root / "checkpoint.bin")
        for fps in (24, 16):
            mapping["fps"] = fps
            path = root / "causvid480.json"
            path.write_text(json.dumps(mapping))
            for record in records:
                if record["name"] == "causvid480.json":
                    record.update(bytes=path.stat().st_size, sha256=file_sha256(path))
            body["total_bytes"] = sum(record["bytes"] for record in records)
            (root / "assets.json").write_text(json.dumps({**body, "asset_id": identity(body)}))
            if fps != 16:
                with self.assertRaisesRegex(ConfigurationError, "FPS must be 16"):
                    worker.CausVidAssets.verified(root / "assets.json", "causvid480.json", launch_receipt=closure)
        with self.assertRaises(IntegrityError):
            worker.CausVidAssets.verified(root / "assets.json", "unverified.json", launch_receipt=closure)
        (root / "base/unverified.safetensors").write_text("must not silently load")
        with self.assertRaisesRegex(IntegrityError, "Unmanifested"):
            worker.CausVidAssets.verified(root / "assets.json", "causvid480.json", launch_receipt=closure)
        with self.assertRaises(ConfigurationError):
            worker.CausVidAssets.verified(root / "assets.json", "/original/assets.json", launch_receipt=closure)

    def test_protocol_prompt_cannot_be_rewritten_or_reassigned(self):
        body = {
            "schema": PROTOCOL_SCHEMA, "cells": {name: cell.to_dict() for name, cell in CELLS.items()},
            "energy_targets": [0.95, 0.99], "clean_boundaries": [0, 1, 2], "seeds": [0, 1],
            "catalog": {"commit": VBENCH_COMMIT, "sha256": VBENCH_PROMPTS_SHA256},
            "full_official_vbench_authorized": False,
            "calibration_groups": 1, "heldout_groups": 1,
            "prompts": [
                {"text": self.request()["text"], "group_id": "g-fox", "prompt_id": "cal00", "split": "calibration"},
                {"text": "An astronaut repairs a satellite in orbit.", "group_id": "g-space",
                 "prompt_id": "hel00", "split": "heldout"},
            ],
        }
        protocol = {**body, "protocol_id": identity(body)}
        worker.CausVidRankRequest.from_dict(self.request(), protocol=protocol, asset_id=self.asset_id)
        wrong = self.request(text="A red fox is running beside a river.")
        with self.assertRaisesRegex(ConfigurationError, "exact frozen protocol"):
            worker.CausVidRankRequest.from_dict(wrong, protocol=protocol, asset_id=self.asset_id)
        wrong = {**self.request(), "prompt_group_id": "g-space"}
        with self.assertRaises(ConfigurationError):
            worker.CausVidRankRequest.from_dict(wrong, protocol=protocol, asset_id=self.asset_id)
        with self.assertRaisesRegex(StateError, "another prompt split"):
            worker._validate_phase({"split": "heldout", "chunks": 3}, protocol, "calibration_census")
        with self.assertRaisesRegex(StateError, "another prompt split"):
            worker._validate_phase({"split": "calibration", "chunks": 3}, protocol, "native_preflight")

    def test_rgb_prefixes_and_continuation_windows_use_native_four_f_minus_three(self):
        result = worker.rgb_windows(3, 16)
        self.assertEqual(result["prefix_frame_counts"], [9, 21, 33])
        self.assertEqual([(row["start_frame"], row["end_frame"]) for row in result["windows"]],
                         [(0, 9), (9, 21), (21, 33)])
        self.assertEqual([row["kind"] for row in result["windows"]],
                         ["first_chunk", "genuine_continuation", "genuine_continuation"])
        self.assertEqual(result["windows"][2]["end_seconds"], 33 / 16)
        with self.assertRaises(ConfigurationError):
            worker.rgb_windows(3, float("nan"))
        for wrong in (15, 24, True, float("inf")):
            with self.subTest(fps=wrong), self.assertRaisesRegex(ConfigurationError, "FPS must be 16"):
                worker.rgb_windows(3, wrong)
        with self.assertRaisesRegex(ConfigurationError, "FPS must be 16"):
            worker.CausVidRankWorker(
                self.pipe, self.model, config=CONFIG, output_root=self.directory,
                asset_id=self.asset_id, fps=24, conditioning_fn=self.encode_text,
            )

    def decode_fixture(self, name, *, chunks):
        from reelcache.serialization import serialize_tensors

        source_request = worker.CausVidRankRequest.from_dict(
            self.request("source-" + name, chunks=chunks), asset_id=self.asset_id,
        )
        source_directory = self.directory / ("source-" + name)
        source_directory.mkdir()
        latents = torch.zeros((1, chunks * 3, 16, 60, 104), dtype=torch.bfloat16)
        rng = torch.Generator().get_state()
        state = {
            "latents": latents, "noise": latents.clone(), "rng_state": rng,
            "initial_rng_state": rng.clone(), "denoising_rng_state": rng.clone(),
            "prompt_embeds": torch.zeros((1, 3, 8), dtype=torch.bfloat16),
        }
        path = source_directory / "latents.reelcache"
        path.write_bytes(serialize_tensors(state))
        result = {
            "schema": "reelcache.causvid.rank_result.v1", "status": "complete", "operation": "generate",
            "request": source_request.document, "asset_id": self.asset_id,
            "model": worker.model_identity(self.asset_id).to_dict(),
            "generation_from_scratch": True, "supplied_prefix": False,
            "native_execution": False, "fixture": True,
            "latents_sha256": tensor_digest(latents), "latent_file": worker._file_record(path, self.directory),
        }
        result_path = source_directory / "result.json"
        result_path.write_text(json.dumps(result))
        document = self.request(name, chunks=chunks)
        document["trajectory_id"] = source_request.document["trajectory_id"]
        document["latent_source"] = {
            "path": str(result_path.relative_to(self.directory)), "sha256": file_sha256(result_path),
        }
        request = worker.CausVidRankRequest.from_dict(document, asset_id=self.asset_id)
        return request, result, SimpleNamespace(asset_id=self.asset_id, fps=16)

    def test_shape_valid_nonfinite_vae_output_aborts_before_rgb_quantization(self):
        for name, bad in (("nan", float("nan")), ("positive-inf", float("inf")), ("negative-inf", -float("inf"))):
            with self.subTest(value=name):
                request, source, assets = self.decode_fixture("nonfinite-" + name, chunks=3)
                clears = []
                vae = SimpleNamespace(
                    model=SimpleNamespace(clear_cache=lambda: clears.append(True)),
                    decode_to_pixel=lambda _latents, value=bad: torch.tensor(value).expand(1, 33, 3, 480, 832),
                )
                epoch = SimpleNamespace(vae=vae, assets=assets, device=torch.device("cpu"))
                before = torch.get_rng_state().clone()
                with self.assertRaisesRegex(IntegrityError, "nonfinite pixels"):
                    worker.decode_request(
                        epoch, request, self.directory, source, provenance={"fixture": True, "native_execution": False},
                    )
                root = self.directory / request.document["output_namespace"]
                self.assertTrue(epoch._poisoned)
                self.assertEqual(len(clears), 2)
                self.assertFalse((root / "rgb.npy").exists())
                self.assertFalse((root / "result.json").exists())
                self.assertEqual(json.loads((root / "failure.json").read_text())["status"], "abort_epoch")
                self.assertTrue(torch.equal(before, torch.get_rng_state()))
                with self.assertRaisesRegex(StateError, "abort_epoch"):
                    worker.decode_request(epoch, request, self.directory, source, provenance={"fixture": True})

    def test_decode_disk_admission_precedes_vae_cache_and_output_mutation(self):
        request, source, assets = self.decode_fixture("disk-rejected", chunks=3)
        document = request.document
        document["disk_budget"] = {"disk_bytes": 2, "disk_reserve_bytes": 1}
        unvalidated = worker.CausVidRankRequest(worker.canonical_json(document))
        vae = SimpleNamespace(
            model=SimpleNamespace(clear_cache=lambda: self.fail("cache touched before output admission")),
            decode_to_pixel=lambda _: self.fail("VAE called before output admission"),
        )
        epoch = SimpleNamespace(vae=vae, assets=assets, device=torch.device("cpu"))
        with self.assertRaisesRegex(CapacityError, "Complete CausVid outputs"):
            worker.decode_request(epoch, unvalidated, self.directory, source, provenance={"fixture": True})
        self.assertFalse((self.directory / "disk-rejected").exists())

    def test_admitted_finite_decode_keeps_native_pixels_windows_and_disk_reserve(self):
        import numpy as np

        request, source, assets = self.decode_fixture("finite-decode", chunks=1)
        clears = []
        vae = SimpleNamespace(
            model=SimpleNamespace(clear_cache=lambda: clears.append(True)),
            decode_to_pixel=lambda _: torch.zeros(()).expand(1, 9, 3, 480, 832),
        )
        epoch = SimpleNamespace(vae=vae, assets=assets, device=torch.device("cpu"))
        result = worker.decode_request(
            epoch, request, self.directory, source, provenance={"fixture": True, "native_execution": False},
        )
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["rgb"]["fps"], 16)
        self.assertEqual(result["rgb"]["prefix_frame_counts"], [9])
        path = self.directory / result["rgb_file"]["path"]
        pixels = np.load(path, allow_pickle=False)
        self.assertEqual(pixels.shape, (9, 480, 832, 3))
        self.assertTrue(bool((pixels == 127).all()))
        self.assertEqual(result["rgb_file"]["bytes"], pixels.nbytes + 128)
        self.assertEqual(len(clears), 2)
        self.assertEqual(result["disk_admission"]["disk_reserve_bytes"], request.document["disk_budget"]["disk_reserve_bytes"])
        self.assertLess(
            sum(child.stat().st_size for child in path.parent.iterdir()),
            request.document["disk_budget"]["disk_bytes"] - request.document["disk_budget"]["disk_reserve_bytes"],
        )

    def test_native_reference_latents_use_separate_vae_epoch_and_development_gate(self):
        request, source, assets = self.decode_fixture("reference-vae", chunks=1)
        source.update(
            operation="native-reference", native_pipeline_inference_called=True, adapter_instantiated=False,
        )
        source["request"].update(
            split="development", study_phase="native_preflight",
            reference_source={"path": "fixture-raw/result.json", "sha256": "d" * 64},
        )
        source["request"]["memory_budgets"]["resident_bytes"] = (
            worker.NATIVE_DENSE_KV_BYTES + worker.NATIVE_CROSSATTN_BYTES
        )
        document = request.document
        document.update(split="development", study_phase="native_preflight")
        result_path = self.directory / document["latent_source"]["path"]
        result_path.write_bytes(worker.canonical_json(source) + b"\n")
        document["latent_source"]["sha256"] = file_sha256(result_path)
        request = worker.CausVidRankRequest.from_dict(document, asset_id=self.asset_id)
        checked = worker.verify_latent_source(request, self.directory, assets)
        forbidden = request.document
        forbidden["study_phase"], forbidden["split"] = "heldout", "heldout"
        with self.assertRaisesRegex(ConfigurationError, "development/native_preflight"):
            worker.verify_latent_source(
                worker.CausVidRankRequest.from_dict(forbidden, asset_id=self.asset_id), self.directory, assets,
            )
        clears = []
        vae = SimpleNamespace(
            model=SimpleNamespace(clear_cache=lambda: clears.append(True)),
            decode_to_pixel=lambda _: torch.zeros(()).expand(1, 9, 3, 480, 832),
        )
        epoch = SimpleNamespace(vae=vae, device=torch.device("cpu"), assets=assets)
        with patch("reelcache.adapters.causvid.CausVidAdapter", side_effect=AssertionError("adapter forbidden")):
            result = worker.decode_request(
                epoch, request, self.directory, checked, provenance={"fixture": True, "native_execution": False},
            )
        self.assertEqual(result["source_operation"], "native-reference")
        self.assertTrue(result["rgb"]["decoded"])
        self.assertFalse(result["dit_loaded"])
        self.assertFalse(result["native_execution"])
        self.assertEqual(len(clears), 2)

    def test_separate_vae_failure_resets_cache_and_rejects_wrong_rgb_count(self):
        from reelcache.serialization import serialize_tensors

        source_request = worker.CausVidRankRequest.from_dict(self.request("source"), asset_id=self.asset_id)
        source_directory = self.directory / "source"
        source_directory.mkdir()
        latents = torch.zeros((1, 3, 16, 60, 104), dtype=torch.bfloat16)
        rng = torch.Generator().get_state()
        state = {
            "latents": latents, "noise": latents.clone(), "rng_state": rng,
            "initial_rng_state": rng.clone(), "denoising_rng_state": rng.clone(),
            "prompt_embeds": torch.zeros((1, 3, 8), dtype=torch.bfloat16),
        }
        latent_path = source_directory / "latents.reelcache"
        latent_path.write_bytes(serialize_tensors(state))
        source_result = {
            "schema": "reelcache.causvid.rank_result.v1", "status": "complete", "operation": "generate",
            "request": source_request.document, "asset_id": self.asset_id,
            "model": worker.model_identity(self.asset_id).to_dict(),
            "generation_from_scratch": True, "supplied_prefix": False,
            "latents_sha256": tensor_digest(latents),
            "latent_file": worker._file_record(latent_path, self.directory),
        }
        result_path = source_directory / "result.json"
        result_path.write_text(json.dumps(source_result))
        document = self.request("decoded")
        document["trajectory_id"] = source_request.document["trajectory_id"]
        document["latent_source"] = {"path": "source/result.json", "sha256": file_sha256(result_path)}
        request = worker.CausVidRankRequest.from_dict(document, asset_id=self.asset_id)
        assets = SimpleNamespace(asset_id=self.asset_id, fps=16)
        checked = worker.verify_latent_source(request, self.directory, assets)
        clears = []
        vae = SimpleNamespace(
            model=SimpleNamespace(clear_cache=lambda: clears.append(True)),
            decode_to_pixel=lambda _latents: torch.zeros((1, 12, 3, 2, 2)),
        )
        epoch = SimpleNamespace(vae=vae, device=torch.device("cpu"), assets=assets)
        with self.assertRaisesRegex(IntegrityError, "4F-3"):
            worker.decode_request(epoch, request, self.directory, checked, provenance={"proof": "CPU fake only"})
        self.assertEqual(len(clears), 2)
        failure = json.loads((self.directory / "decoded/failure.json").read_text())
        self.assertEqual(failure["status"], "abort_epoch")
        self.assertFalse((self.directory / "decoded/result.json").exists())


if __name__ == "__main__":
    unittest.main()
