"""Synthetic CPU fixtures, never an official-prompt generation result."""

import copy
from contextlib import redirect_stdout
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
from types import SimpleNamespace
import unittest
import uuid
from unittest.mock import patch

import numpy as np
import torch

from reelcache.adapters import magi_vbench as benchmark
from reelcache.adapters import magi_native as native


class MagiVBenchTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.root = native.PROJECT / ".test-artifacts" / f"fresh-magi-{uuid.uuid4().hex}"
        self.root.mkdir(parents=True)
        self.catalogue = self.root / "synthetic_catalogue.json"
        self._write(self.catalogue, [{"prompt_en": f"SYNTHETIC UNIT TEST {i}", "dimension": ["subject_consistency"]}
                                     for i in range(4)])
        self.case = {"schema": benchmark.CASE_SCHEMA, "protocol_id": "synthetic_test_v1", "case_id": "synthetic_case",
                     "prompt_id": 0, "prompt": "SYNTHETIC UNIT TEST 0", "seed": 0}
        self.protocol = {
            "schema": benchmark.PROTOCOL_SCHEMA, "protocol_id": "synthetic_test_v1",
            "catalogue": {"path": str(self.catalogue), "sha256": native._sha(self.catalogue),
                          "origin": "https://github.com/Vchitect/VBench/synthetic-test-fixture-only",
                          "id_kind": "zero_based_index"},
            "prompt_ids": [0, 1, 2, 3], "seed": 0, "prefix_frames": 24, "continuation_frames": 96, "fps": 24,
            "magi": {**benchmark.FIXED_MAGI, "arms": ["raw", "svd_hevc"]},
        }

    def tearDown(self):
        shutil.rmtree(self.root)

    def _write(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
        return path

    def _prepare_fixture(self):
        self._write(self.root / "case.json", self.case)
        self._write(self.root / "protocol.json", self.protocol)
        source, model = self.root / "builder_source", self.root / "native_source"
        builder_rows = {}
        for name, key in (("native_magi_sp_bench.py", "native"), ("neurico_magi_sweep.py", "capture_and_model_loader"),
                          ("adaptive_rank_magi.py", "native_fixes")):
            path = source / "kv_compression/harness" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("# Synthetic unit fixture, not executable native source.\n")
            builder_rows[f"pilot/{key}"] = native._file_row(path)
        model_file = model / "inference/model/dit/dit_model.py"
        model_file.parent.mkdir(parents=True)
        model_file.write_text("# Synthetic model source fixture.\n")
        tokens = model / "example/assets/special_tokens.npz"
        tokens.parent.mkdir(parents=True)
        tokens.write_bytes(b"not-a-real-special-token-file")
        (model / "inference/model/t5").mkdir(parents=True)
        weights = self.root / "weights"
        weights.mkdir()
        self._write(weights / "model.safetensors.index.json", {"weight_map": {"not_a_weight": "fixture.safetensors"}})
        header = json.dumps({"__metadata__": {"unit_fixture": "not-a-model"}}).encode()
        (weights / "fixture.safetensors").write_bytes(struct.pack("<Q", len(header)) + header)
        checkpoint = {
            "index_sha256": native._sha(weights / "model.safetensors.index.json"),
            "shards": [{"path": str(weights / "fixture.safetensors"), "bytes": 8 + len(header),
                        "header_sha256": benchmark.hashlib.sha256(header).hexdigest()}],
        }
        vae = self.root / "vae"
        self._write(vae / "config.json", {"unit_fixture": True})
        (vae / "diffusion_pytorch_model.safetensors").write_bytes(b"synthetic VAE bytes, never a model")
        t5 = self.root / "t5"
        for name in ("config.json", "tokenizer_config.json", "special_tokens_map.json"):
            self._write(t5 / "t5-v1_1-xxl" / name, {})
        self._write(t5 / "t5-v1_1-xxl/pytorch_model.bin.index.json", {"weight_map": {"fixture": "fixture.bin"}})
        (t5 / "t5-v1_1-xxl/spiece.model").write_bytes(b"fixture")
        (t5 / "t5-v1_1-xxl/fixture.bin").write_bytes(b"not-a-model")
        config = {
            "model_config": {"num_layers": 48, "hidden_size": 6144, "num_attention_heads": 48,
                             "num_query_groups": 8, "kv_channels": 128, "params_dtype": "torch.bfloat16"},
            "runtime_config": {"cfg_number": 1, "num_steps": 16, "chunk_width": 6,
                               "video_size_h": 1088, "video_size_w": 1920, "clean_t": 0.9999},
            "engine_config": {"cp_size": 2, "cp_strategy": "cp_ulysses", "pp_size": 1,
                              "ulysses_overlap_degree": 1, "distill": True, "fp8_quant": False,
                              "enable_cuda_graph": False, "kv_offload": False},
        }
        reference = {
            "config": config, "external_sources": {
                **builder_rows, "native/inference/model/dit/dit_model.py": native._file_row(model_file),
                "native/special_tokens": native._file_row(tokens),
            }, "builder": builder_rows["pilot/capture_and_model_loader"], "checkpoint_reference": checkpoint,
            "vae": {"config_sha256": native._sha(vae / "config.json"),
                    "expected_checkpoint_sha256": native._sha(vae / "diffusion_pytorch_model.safetensors"),
                    "mode": "stateless_chunk_decode", "temporal_tile_overlap_factor": 0},
            "embedding": {"path": "/DO_NOT_READ_OLD_EMBEDDING"},
            "prefix": {"path": "/DO_NOT_READ_OLD_PREFIX"},
        }
        reference["identity"] = native._identity(reference)
        qualified = self._write(self.root / "synthetic_qualified.json", reference)
        from reelcache.adapters.magi_provenance import canonical_digest, capture_auxiliary, stat_guard

        weight_rows = []
        for name, role in (("model.safetensors.index.json", "index"), ("fixture.safetensors", "weight_shard")):
            path = weights / name
            weight_rows.append({
                **native._file_row(path), "relative_path": name, "resolved_path": str(path), "role": role,
                "model": "magi", "full_content_hashed": True, "stable_during_hash": True,
                "stable_through_manifest_validation": True, "matches_recorded_reference": True,
                "final_manifest_stat": stat_guard(path.stat()),
            })
        model_content = {"files": [{key: row[key] for key in ("relative_path", "role", "bytes", "sha256")}
                                   for row in weight_rows]}
        content = {"schema": "reelcache_weight_content_v1", "models": {"magi": model_content}}
        weight_manifest = {
            "schema": "reelcache_weight_identity_v1", "status": "passed", "content_manifest": content,
            "content_manifest_sha256": canonical_digest(content), "model_content_sha256": {"magi": canonical_digest(model_content)},
            "files": weight_rows,
        }
        weight_manifest["manifest_integrity_sha256"] = canonical_digest(weight_manifest)
        weight_path = self._write(self.root / "weight_identity.json", weight_manifest)
        auxiliary_path = self.root / "auxiliary_identity.json"
        auxiliary = capture_auxiliary(
            output=auxiliary_path, t5=t5, vae=vae, source=model, qualified_plan=qualified, max_seconds=60,
        )
        self.protocol.update({
            "weight_identity": {
                "path": str(weight_path), "sha256": native._sha(weight_path),
                "content_manifest_sha256": weight_manifest["content_manifest_sha256"],
            },
            "magi_auxiliary_identity": {
                "path": str(auxiliary_path), "sha256": native._sha(auxiliary_path),
                "content_manifest_sha256": auxiliary["content_manifest_sha256"],
            },
        })
        self._write(self.root / "protocol.json", self.protocol)
        return SimpleNamespace(
            output=self.root / "run", case=self.root / "case.json", protocol=self.root / "protocol.json",
            qualified_plan=qualified, source_project=source, native_source=model, weights=weights, vae=vae, t5=t5,
            allow_lossy=True, cpu_threads=1, max_seconds=10800, stage_seconds=1800,
            condition_seconds=1800, store_gib=100, storage_placement="nfs_prepare_only",
        )

    def test_case_contract_exact_official_row_and_head_local_budget(self):
        result = benchmark.validate_case(self.case, self.protocol)
        self.assertEqual(result["max_hot_required_bytes_per_rank"], 19251855360)
        self.assertEqual(result["catalogue_row"]["prompt_en"], self.case["prompt"])
        self.assertFalse(torch.cuda.is_initialized())
        broken = copy.deepcopy(self.case)
        broken["prompt"] += " "
        with self.assertRaisesRegex(ValueError, "exactly equal"):
            benchmark.validate_case(broken, self.protocol)

    def test_cohort_geometry_precision_and_profile_fail_closed(self):
        for field, value in (("seed", True), ("prefix_frames", 0), ("continuation_frames", 24), ("fps", 30),
                             ("prompt_ids", [0, 0, 1, 2])):
            with self.subTest(field=field):
                protocol = copy.deepcopy(self.protocol)
                protocol[field] = value
                with self.assertRaises(ValueError):
                    benchmark.validate_case(self.case, protocol)
        for ids in ([1], [0, 1], [False]):
            protocol = copy.deepcopy(self.protocol)
            protocol["diagnostic_svd_prompt_ids"] = ids
            with self.assertRaises(ValueError):
                benchmark.validate_case(self.case, protocol)
        for field, value in (("K_rank", 384), ("qp", 9), ("pix_fmt", "yuv420p"),
                             ("normalization", "global"), ("reuse_threshold", 0.01),
                             ("hot_window_bytes", 4 * benchmark.GIB), ("special_tokens", ["HQ_TOKEN"])):
            with self.subTest(field=field):
                protocol = copy.deepcopy(self.protocol)
                protocol["magi"][field] = value
                with self.assertRaises(ValueError):
                    benchmark.validate_case(self.case, protocol)

    def test_svd_diagnostic_is_only_first_case_not_a_primary_arm(self):
        protocol = copy.deepcopy(self.protocol)
        protocol["diagnostic_svd_prompt_ids"] = [0]
        benchmark.validate_case(self.case, protocol)
        self.assertEqual(benchmark._case_arms(self.case, protocol), ["raw", "svd", "svd_hevc"])
        second = {**self.case, "case_id": "synthetic_second", "prompt_id": 1, "prompt": "SYNTHETIC UNIT TEST 1"}
        benchmark.validate_case(second, protocol)
        self.assertEqual(benchmark._case_arms(second, protocol), ["raw", "svd_hevc"])
        self.assertEqual(protocol["magi"]["arms"], ["raw", "svd_hevc"])

    def test_catalogue_and_case_identity_mutation_rejected(self):
        changed = copy.deepcopy(self.case)
        changed["protocol_id"] = "another_protocol"
        with self.assertRaises(ValueError):
            benchmark.validate_case(changed, self.protocol)
        self.catalogue.write_text("[]")
        with self.assertRaisesRegex(ValueError, "catalogue content changed"):
            benchmark.validate_case(self.case, self.protocol)

    def test_prepare_fresh_binding_never_reads_old_prefix_or_embedding(self):
        args = self._prepare_fixture()
        with (
            patch.dict(sys.modules, {"transformers": SimpleNamespace(__version__="4.42.3")}),
            redirect_stdout(io.StringIO()),
        ):
            plan = benchmark.prepare(args)
        self.assertTrue(plan["fresh_prompt_inputs"])
        self.assertNotIn("embedding", plan)
        self.assertNotIn("prefix", plan)
        self.assertNotIn("noise_reference", plan)
        self.assertEqual(plan, benchmark.verify_plan(args.output))
        commands = native._read(args.output / "commands.json")
        self.assertIn("--nproc_per_node=2", commands["run_REQUIRES_BOTH_GPU_GRANT"])
        self.assertIn("timeout", commands["run_REQUIRES_BOTH_GPU_GRANT"])
        self.assertIn("--cpu-model-grant", commands["condition_CPU_model_load_requires_approval"])
        self.assertFalse(torch.cuda.is_initialized())
        with self.assertRaises(FileExistsError):
            benchmark.prepare(args)
        self._write(args.case, {**self.case, "prompt": "A DIFFERENT SYNTHETIC PROMPT"})
        with self.assertRaisesRegex(ValueError, "Bound file changed"):
            benchmark.verify_plan(args.output)

    def test_prepare_rejects_incompatible_model_environment_before_encoder_probe(self):
        args = self._prepare_fixture()
        for index, (torch_version, transformers_version, cuda_initialized) in enumerate((
            ("2.8.0", "unsupported", False),
            ("2.7.0", "4.42.3", False),
            ("2.8.0", "4.42.3", True),
        )):
            rejected = SimpleNamespace(**{**vars(args), "output": self.root / f"rejected-{index}"})
            with (
                self.subTest(torch=torch_version, transformers=transformers_version,
                             cuda_initialized=cuda_initialized),
                patch.dict(sys.modules, {
                    "transformers": SimpleNamespace(__version__=transformers_version),
                }),
                patch.object(torch, "__version__", torch_version),
                patch.object(torch.cuda, "is_initialized", return_value=cuda_initialized),
                patch("reelcache.codecs.hevc_capabilities") as encoder_probe,
            ):
                with self.assertRaisesRegex(RuntimeError, "Prepare CPU-only"):
                    benchmark.prepare(rejected)
                encoder_probe.assert_not_called()
                self.assertFalse((rejected.output / "plan.json").exists())

    def test_fresh_T5_shape_and_precision_are_not_silently_cast(self):
        text = {"case_id": self.case["case_id"], "prompt": self.case["prompt"],
                "caption": torch.zeros(1, 800, 4096), "mask": torch.ones(1, 800, dtype=torch.int64)}
        benchmark._check_text(text, self.case)
        with self.assertRaises(ValueError):
            benchmark._check_text({**text, "caption": text["caption"].bfloat16()}, self.case)
        with self.assertRaises(ValueError):
            benchmark._check_text({**text, "mask": torch.full((1, 800), 2)}, self.case)

    def test_native_call_sequence_preserves_visibility_positions_and_duplicate_steps(self):
        session = object.__new__(benchmark._FreshSession)
        session.torch, session.device = torch, torch.device("cpu")
        session.plan = {"continuations": 4}
        session.config = SimpleNamespace(runtime_config=SimpleNamespace(
            clean_chunk_kvrange=1, noise2clean_kvrange=[5, 4, 3, 2],
        ))
        session.schedule = torch.linspace(0, 1, 17)
        session.intervals = torch.zeros(16)
        session.near = [False] * 12 + [True] * 4
        session.cache = SimpleNamespace(update_kv_cache=False)
        session.y, session.masks = torch.ones(2, 1, 8, 4), torch.ones(2, 1, 8)
        session.null_caption, session.null_mask = torch.zeros(1, 8, 4), torch.zeros(1, 8)
        seen, clean_seen = [], []

        def forward(x, timestep, y, mask, kv_range, cache, **kwargs):
            self.assertIs(y, session.y)
            self.assertIs(mask, session.masks)
            self.assertEqual(x.shape[0], 2)
            seen.append((kv_range.tolist(), kwargs))
            return torch.zeros_like(x)

        def clean_forward(x, timestep, y, **kwargs):
            clean_seen.append((timestep.clone(), kwargs))
            return torch.zeros_like(x)

        session.model = SimpleNamespace(forward_dispatcher=forward, forward=clean_forward)
        shape = (1, 2, 1, 2, 2)
        with patch.object(native, "SHAPE", shape), patch.object(native, "N", 6):
            for index in range(5):
                noise = torch.full(shape, float(index))
                self.assertTrue(torch.equal(session.denoise(noise, index), noise))
                session.clean(noise, index)
                for step, (visibility, kw) in enumerate(seen[index * 16:(index + 1) * 16]):
                    context = [5, 4, 3, 2][step // 4]
                    self.assertEqual(visibility, [[max(0, index + 1 - context) * 6, (index + 1) * 6]])
                    self.assertEqual(kw["start_chunk_id"], index)
                    self.assertEqual(kw["slice_point"], index)
                    self.assertEqual(kw["chunk_num"], 5)
                    self.assertEqual(kw["end_chunk_id"], index + 1 + int(step >= 12))
                    self.assertEqual(kw["distill_nearly_clean_chunk"], step >= 12)
        for index, (timestep, kw) in enumerate(clean_seen):
            self.assertTrue(torch.equal(timestep, torch.full((1, 1), 0.9999)))
            self.assertEqual(kw["kv_range"].tolist(), [[index * 6, (index + 1) * 6]])
            self.assertFalse(kw["distill_nearly_clean_chunk"])
            self.assertTrue(kw["extract_prefix_video_feature"])

    def test_network_storage_cannot_be_labeled_local_SSD(self):
        args = self._prepare_fixture()
        args.storage_placement = "local_ssd"
        with patch.object(benchmark, "_storage_location", return_value={"filesystem_type": "nfs", "device_id": 0}):
            with self.assertRaisesRegex(ValueError, "network filesystem"):
                benchmark.prepare(args)
        self.assertFalse(args.output.exists())

    def test_storage_ratio_uses_native_BF16_and_never_rereads_blob_payloads(self):
        from reelcache.manager import KVRuntime
        from reelcache.types import CodecSpec, MatrixKey, ModelIdentity, TensorContext

        identity = ModelIdentity("synthetic", "test", 1, 8, "global_channels", "bfloat16")
        runtime = KVRuntime(self.root / "store", identity, max_resident_bytes=1024**2)
        try:
            runtime.create_branch("raw")
            keys = [MatrixKey(0, target) for target in ("K", "V")]
            matrices = {key: torch.arange(48).reshape(6, 8).to(torch.bfloat16) for key in keys}
            contexts = {key: TensorContext(identity, key, 0, 1, 2, 3, "post_rope" if key.target == "K" else "none")
                        for key in keys}
            runtime.seal_chunk("raw", "0", matrices, contexts, CodecSpec())
            snapshot = runtime.snapshot("raw", metadata={
                "unit_fixture": True,
                "clean_boundary": {"chunk_id": "0", "token_end": 6, "clean": True},
                "conditioning": {"mode": "synthetic_fixture"},
                "execution": {"pending": False, "mode": "synthetic_fixture"},
                "rng": {"tensor": "rng"}, "latent": {"tensor": "latent"},
                "vae": {"mode": "synthetic_fixture_no_model"},
            }, tensors={"latent": torch.zeros(4096), "rng": torch.get_rng_state()})
            get = runtime.store.get

            def metadata_only(object_id):
                if not object_id.startswith("manifest:"):
                    raise AssertionError("Storage audit reread a dense payload just to count bytes")
                return get(object_id)
            with patch.object(runtime.store, "get", side_effect=metadata_only):
                measured = benchmark._storage_audit(runtime, "raw", snapshot)
            self.assertEqual(measured["native_history_bytes_across_SP2"], 6 * 8 * 2 * 2)
            decoder, resume = measured["decoder_closure"], measured["resumable_snapshot_closure"]
            self.assertEqual(decoder["unique_bytes"], sum(decoder["byte_breakdown"].values()))
            self.assertLess(decoder["unique_bytes"], resume["unique_bytes"])
            self.assertEqual(measured["primary_ratio_native_BF16_to_decoder_closure"], 192 / decoder["unique_bytes"])
            self.assertGreater(measured["physical_store"]["index_and_control_bytes"], 0)
            self.assertNotIn(snapshot, [row["object_id"] for row in decoder["ledger"]])
            self.assertIn(snapshot, [row["object_id"] for row in resume["ledger"]])
            self.assertNotIn("ledger", benchmark._storage_summary(measured)["decoder_closure"])
            self.assertEqual(len(measured["packets"]), 2)
            exported = benchmark._export_checkpoint(
                runtime, snapshot, self.root, "raw", {"identity": "synthetic_plan", "case": self.case},
            )
            saved = torch.load(exported["artifact"]["path"], map_location="cpu", weights_only=True)
            self.assertEqual(saved["snapshot_id"], snapshot)
            self.assertTrue(torch.equal(saved["tensors"]["latent"], torch.zeros(4096)))
            self.assertFalse(exported["standalone_full_resume"])
        finally:
            runtime.close()

    def test_import_help_and_grant_guards_never_load_models(self):
        script = native.PROJECT / "examples/vbench_magi.py"
        code = f"""
import importlib.util, sys
spec = importlib.util.spec_from_file_location('fresh_cli', {str(script)!r})
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
assert 'torch' not in sys.modules
assert 'transformers' not in sys.modules
assert not any(name.startswith('inference.') for name in sys.modules)
try:
    module.main(['--help'])
except SystemExit as exc:
    assert exc.code == 0
assert 'torch' not in sys.modules
"""
        environment = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1",
                       "TMPDIR": str(self.root)}
        result = subprocess.run([sys.executable, "-B", "-c", code], env=environment, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        spec = importlib.util.spec_from_file_location("fresh_cli", script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for stage in ("run", "decode", "condition"):
            with self.assertRaises(SystemExit):
                module.main([stage, "--output", str(self.root)])
        with self.assertRaises(RuntimeError):
            benchmark.run(SimpleNamespace(gpu_grant=False))
        with self.assertRaises(RuntimeError):
            benchmark.decode(SimpleNamespace(gpu_grant=False))
        with self.assertRaises(RuntimeError):
            benchmark.condition(SimpleNamespace(cpu_model_grant=False))

    def test_lossless_RGB_video_roundtrip_and_frame_contract(self):
        from reelcache.codecs import hevc_capabilities

        binary = hevc_capabilities()["binary"]
        frames = np.arange(4 * 8 * 8 * 3, dtype=np.uint16).astype(np.uint8).reshape(4, 8, 8, 3)
        writer = benchmark._VideoWriter(binary, self.root / "fixture.mp4", width=8, height=8, frames=4)
        try:
            writer.write(frames[:2])
            writer.write(frames[2:])
            row = writer.close()
        finally:
            writer.abort()
        benchmark._verify_video_rgb(binary, row)
        self.assertEqual(row["frames"], 4)
        self.assertEqual(row["fps"], 24)
        with self.assertRaises(FileExistsError):
            benchmark._VideoWriter(binary, self.root / "fixture.mp4", width=8, height=8, frames=4)
        with self.assertRaises(RuntimeError):
            benchmark._verify_video_rgb(binary, {**row, "rgb_sha256": "0" * 64})

    def test_nested_deadline_does_not_extend_outer_budget(self):
        state = [(8.0, 0.0), (0.0, 0.0)]
        with patch.object(benchmark.signal, "signal", return_value=benchmark.signal.SIG_DFL), \
                patch.object(benchmark.signal, "setitimer", side_effect=state) as timer, \
                patch.object(benchmark.time, "monotonic", side_effect=[100.0, 103.0]):
            with benchmark._deadline(2):
                pass
        self.assertEqual(timer.call_args_list[-1].args, (benchmark.signal.ITIMER_REAL, 5.0, 0.0))


if __name__ == "__main__":
    unittest.main()
