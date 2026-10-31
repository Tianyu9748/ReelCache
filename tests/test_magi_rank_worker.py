"""No native imports/weights: request isolation, original rows and hot history."""

from dataclasses import asdict, replace
from datetime import timedelta
from contextlib import contextmanager, nullcontext
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import struct
import time
from types import SimpleNamespace
from types import ModuleType
import unittest
from unittest.mock import Mock, patch
import uuid
import weakref

import torch

from reelcache.adapters.magi import (
    MagiGeometry, MagiResourceBudget, NATIVE_TENSOR_HASH_FORMAT,
    _tensor_context, _tensor_hash, _tensor_hash_format,
)
from reelcache.adapters.magi_native import _validate_config
from reelcache.adapters.magi_rank_worker import (
    MagiEpochAborted, MagiRankRequest, MagiWorkerEpoch, validate_magi_checkpoint, verify_worker_inputs,
    serial_sampler_contract, worker_collective_timeout, worker_thread_allowance,
)
from reelcache.distributed import DistributedOwner, PeerFailure
from reelcache.errors import CapacityError, ConfigurationError, IntegrityError, StateError
from reelcache.study import file_sha256, identity, study_cell
from reelcache.types import CodecSpec, MatrixKey, ModelIdentity
from tests import test_magi_adapter as fixtures


def budget(value=1 << 45):
    return MagiResourceBudget(
        host_bytes=value, device_bytes=value, staging_bytes=value,
        workspace_host_bytes=1 << 26, workspace_device_bytes=1024, packet_bytes=1024,
    )


def request(name="A", seed=11, cell="magi720"):
    return MagiRankRequest(name, cell, name, "group-" + name, "calibration",
                           "Unchanged official prompt " + name, seed, 2, case_id="case-" + name)


def serial_timetable():
    squared = torch.linspace(0, 1, 17, dtype=torch.float32).square()
    return (1.0 / 3.0) * squared / (1 + (1.0 / 3.0 - 1) * squared), torch.ones(16)


class FakeTransport:
    world_size = 2

    def __init__(self, mismatch=False):
        self.mismatch = mismatch
        self.operations = []

    def checkpoint(self, operation, *, details=None, error=None):
        self.operations.append(operation)
        if error is not None:
            raise PeerFailure(str(error))
        peer = copy.deepcopy(details)
        if self.mismatch and operation == "magi_request_agreement":
            peer["sequence"] += 1
        return [details, peer]


class FakeBackend:
    device = torch.device("cpu")

    def __init__(self):
        self.active = None
        self.hooks = []
        self.namespaces = []
        self.first_noises = []
        self.closed = False
        self.fail_generate = False
        self.fail_cleanup = False

    def assert_clean(self):
        if self.active is not None or self.hooks:
            raise StateError("request state survived")

    def new_request(self, value, namespace):
        self.assert_clean()
        self.namespaces.append(namespace)
        self.hooks.append(namespace)
        self.active = {
            "request": value, "generator": torch.Generator().manual_seed(value.seed),
            "cache": [], "conditioning": value.prompt, "cursor": 0,
            "leases": [], "policy_state": {}, "vae_state": {}, "counter": 0,
        }
        return self.active

    def generate(self, state):
        if self.fail_generate:
            raise RuntimeError("simulated poisoned CUDA execution")
        result = []
        for chunk in range(state["request"].chunks):
            noise = torch.randn((2, 3), generator=state["generator"])
            if chunk == 0:
                self.first_noises.append(noise.clone())
            result.append(_tensor_hash(noise))
            state["cache"].append(noise)
            state["cursor"] += 1
            state["counter"] += 1
        return {"noise_hashes": result, "ambient": [float(torch.rand(())), random.random()],
                "prompt": state["conditioning"], "cursor": state["cursor"]}

    def clear_request(self, state):
        if self.active is not None:
            self.active.clear()
        self.active = None
        self.hooks.clear()
        if self.fail_cleanup:
            raise RuntimeError("simulated cleanup failure")

    def close(self):
        self.closed = True
        self.active = None
        self.hooks.clear()


class RankWorkerTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.root = Path(__file__).resolve().parents[1] / ".test-artifacts" / ("magi-rank-" + uuid.uuid4().hex)
        self.root.mkdir(parents=True)

    def tearDown(self):
        shutil.rmtree(self.root)

    def epoch(self, backend, **options):
        return MagiWorkerEpoch(
            cell_id="magi720", config_id="verified-config", epoch_id="fresh-epoch",
            transport=options.pop("transport", FakeTransport()), backend_factory=lambda: backend,
            budget=options.pop("budget", budget()), **options,
        )

    def test_final_geometries_and_native_dtype_contract(self):
        low, high = study_cell("magi720"), study_cell("magi1080")
        self.assertEqual((low.width, low.working_height, low.crop_top, low.tokens), (1280, 720, 0, 21600))
        self.assertEqual((high.width, high.working_height, high.crop_top, high.tokens), (1920, 1088, 4, 48960))
        self.assertEqual(request().latent_shape, (1, 16, 6, 90, 160))
        self.assertEqual(request(cell="magi1080").latent_shape, (1, 16, 6, 136, 240))
        for cell in (low, high):
            geometry = MagiGeometry.for_cell(cell.cell_id)
            self.assertEqual(geometry.channels, 1024)
            self.assertEqual(geometry.layout.local_channels, 512)
            self.assertEqual(cell.source_bytes, geometry.tokens * 1024 * 48 * 6)
        self.assertEqual((low.native_kv_bytes // 2, high.native_kv_bytes // 2),
                         (2123366400, 4812963840))

    def test_admission_counts_both_rank_captures_global_staging_and_rotary(self):
        cell = study_cell("magi720")
        geometry = MagiGeometry.for_cell(cell.cell_id)
        ledger = budget().admit(
            geometry, chunks=2, factor_keys=True, observer=False, quality_hot=True,
        )
        self.assertEqual(ledger["global_capture_bytes"], cell.source_bytes)
        self.assertEqual(ledger["staging_bytes"], 2 * cell.source_bytes)
        self.assertEqual(ledger["native_history_current_bytes"], cell.native_kv_bytes)
        self.assertGreaterEqual(
            ledger["host_bytes"],
            ledger["staging_bytes"] + ledger["shard_transfer_bytes"] + ledger["retained_cpu_rotary_bytes"],
        )
        owner_only = replace(budget(), staging_bytes=cell.source_bytes * 3 // 2)
        with self.assertRaisesRegex(CapacityError, "staging_bytes"):
            owner_only.admit(geometry, chunks=2, factor_keys=True, observer=False, quality_hot=True)

    def test_cpu_threads_are_explicit_and_bounded_across_the_worker_allocation(self):
        with patch("reelcache.adapters.magi_rank_worker.os.sched_getaffinity", return_value=set(range(8))), patch.dict(
            os.environ, {"SLURM_CPUS_PER_TASK": "4"},
        ):
            self.assertEqual(worker_thread_allowance({"cpu_threads": 2}), 2)
            self.assertEqual(worker_thread_allowance({"cpu_threads": 4}, world_size=1), 4)
            for value in ({}, {"cpu_threads": 3}, {"cpu_threads": 128}, {"cpu_threads": True}):
                with self.assertRaises(ConfigurationError):
                    worker_thread_allowance(value)

    def test_collective_deadlines_are_bounded_integers_and_bind_both_worker_and_native_groups(self):
        from reelcache.adapters.magi_native import _bind_native_collective_timeout, _initialize_study_collectives

        self.assertEqual(worker_collective_timeout({}), 600)
        for value in (None, True, False, 0, -1, 21601, 600.0, "600"):
            with self.subTest(value=value), self.assertRaises(ConfigurationError):
                worker_collective_timeout({"collective_timeout_seconds": value})
        for seconds in (1, 600, 3600, 21600):
            with self.subTest(seconds=seconds):
                self.assertEqual(worker_collective_timeout({"collective_timeout_seconds": seconds}), seconds)
                distributed = Mock()
                group = _initialize_study_collectives(distributed, seconds)
                self.assertIs(group, distributed.new_group.return_value)
                distributed.init_process_group.assert_called_once_with("nccl", timeout=timedelta(seconds=seconds))
                distributed.new_group.assert_called_once_with(backend="gloo", timeout=timedelta(seconds=seconds))
                config = SimpleNamespace(engine_config=SimpleNamespace(distributed_timeout_minutes=10))
                _bind_native_collective_timeout(config, seconds)
                self.assertEqual(timedelta(minutes=config.engine_config.distributed_timeout_minutes),
                                 timedelta(seconds=seconds))

    def test_no_736_fallback_and_legacy_default_config(self):
        config = {
            "model_config": {
                "num_layers": 48, "hidden_size": 6144, "num_attention_heads": 48,
                "num_query_groups": 8, "kv_channels": 128, "params_dtype": "torch.bfloat16",
            },
            "runtime_config": {
                "cfg_number": 1, "num_steps": 16, "chunk_width": 6, "video_size_h": 1088,
                "video_size_w": 1920, "clean_t": 0.9999,
                "window_size": 4,
            },
            "engine_config": {
                "cp_size": 2, "cp_strategy": "cp_ulysses", "pp_size": 1, "ulysses_overlap_degree": 1,
                "distill": True, "fp8_quant": False, "enable_cuda_graph": False, "kv_offload": False,
            },
        }
        _validate_config(config)
        config["runtime_config"].update(video_size_h=720, video_size_w=1280)
        _validate_config(config, cell_id="magi720")
        config["runtime_config"]["window_size"] = 1
        with self.assertRaisesRegex(ValueError, "visibility"):
            _validate_config(config, cell_id="magi720", study_worker=True)
        config["runtime_config"].update(noise2clean_kvrange=[8, 8, 4, 4], clean_chunk_kvrange=4)
        _validate_config(config, cell_id="magi720", study_worker=True)
        config["runtime_config"]["video_size_h"] = 736
        with self.assertRaises(ValueError):
            _validate_config(config, cell_id="magi720")

    def test_serial_study_window_is_explicit_typed_and_does_not_change_legacy_validation(self):
        with self.verified_inputs() as (_, assets, _, _):
            config = json.loads((assets / "config.json").read_text())
        for cell_id in ("magi720", "magi1080"):
            cell = study_cell(cell_id)
            config["runtime_config"].update(
                video_size_h=cell.working_height, video_size_w=cell.width, window_size=1,
            )
            _validate_config(config, cell_id=cell_id, study_worker=True)
            for invalid in (None, False, True, 0, 2, 4, 1.0, "1"):
                with self.subTest(cell=cell_id, window=invalid):
                    config["runtime_config"]["window_size"] = invalid
                    if invalid is None:
                        config["runtime_config"].pop("window_size")
                    with self.assertRaisesRegex(ValueError, "window_size=1"):
                        _validate_config(config, cell_id=cell_id, study_worker=True)
            config["runtime_config"]["window_size"] = 4
            _validate_config(config, cell_id=cell_id)

    def test_actual_sixteen_step_timetable_requires_squared_sd3_and_unit_intervals(self):
        from reelcache.adapters.magi_native import _validate_study_timetable

        timetable, intervals = serial_timetable()
        _validate_study_timetable(timetable, intervals)
        self.assertEqual(timetable.shape, (17,))
        torch.testing.assert_close(timetable[[0, -1]], torch.tensor([0.0, 1.0]), rtol=0, atol=2e-7)
        self.assertTrue(torch.all(timetable[1:] > timetable[:-1]))
        self.assertEqual([index for index, value in enumerate(timetable[:-1].tolist()) if value > 0.3],
                         [12, 13, 14, 15])
        for altered, changed_intervals in (
            (torch.linspace(0, 1, 17), intervals),
            (timetable[:9], intervals[:8]),
            (timetable.double(), intervals),
            (timetable.clamp(0, 1), intervals),
            (timetable, torch.full((16,), 2.0)),
            (timetable, intervals.double()),
        ):
            with self.assertRaises(ValueError):
                _validate_study_timetable(altered, changed_intervals)

    def test_promoted_timestep_input_dtype_changes_only_the_loaded_instance(self):
        from reelcache.adapters.magi_native import _align_promoted_timestep_input

        class Timestep(torch.nn.Module):
            data_type = torch.bfloat16

            def __init__(self):
                super().__init__()
                self.mlp = torch.nn.Sequential(
                    torch.nn.Linear(4, 8, dtype=torch.bfloat16), torch.nn.SiLU(),
                    torch.nn.Linear(8, 4, dtype=torch.bfloat16),
                )

            def forward(self, value):
                return self.mlp(value.to(self.data_type))

        promoted, untouched = Timestep().float(), Timestep()
        model = SimpleNamespace(t_embedder=promoted, other=torch.nn.Linear(4, 4, dtype=torch.bfloat16))
        weights = {name: value.clone() for name, value in promoted.state_dict().items()}
        value = torch.ones((2, 4), dtype=torch.float32)
        with self.assertRaisesRegex(RuntimeError, "same dtype"):
            promoted(value)
        _align_promoted_timestep_input(model)
        self.assertEqual(promoted(value).dtype, torch.float32)
        self.assertEqual(promoted.data_type, torch.float32)
        self.assertEqual(untouched.data_type, torch.bfloat16)
        self.assertEqual(Timestep.data_type, torch.bfloat16)
        self.assertEqual(model.other.weight.dtype, torch.bfloat16)
        for name, original in weights.items():
            self.assertTrue(torch.equal(promoted.state_dict()[name], original))
            self.assertEqual(promoted.state_dict()[name].dtype, torch.float32)
        promoted.mlp[-1].to(dtype=torch.bfloat16)
        with self.assertRaisesRegex(ValueError, "one supported dtype"):
            _align_promoted_timestep_input(model)

    def test_every_request_starts_at_first_noise_and_A_B_A_is_fresh(self):
        backend = FakeBackend()
        worker = self.epoch(backend)
        before = torch.get_rng_state().clone()
        python_before = random.getstate()
        first = worker.run(request())
        second = worker.run(request("B", 29))
        repeated = worker.run(request())
        fresh = self.epoch(FakeBackend())
        reference = fresh.run(request())
        self.assertEqual(first["result"], repeated["result"])
        self.assertEqual(first["result"], reference["result"])
        self.assertNotEqual(first["result"], second["result"])
        self.assertEqual(len(set(backend.namespaces)), 3)
        expected_noise = torch.randn((2, 3), generator=torch.Generator().manual_seed(11))
        self.assertTrue(torch.equal(backend.first_noises[0], expected_noise))
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertEqual(python_before, random.getstate())
        self.assertIsNone(backend.active)
        self.assertEqual(backend.hooks, [])
        worker.close()
        fresh.close()

    def test_rank_disagreement_precedes_backend_load(self):
        calls = []
        worker = MagiWorkerEpoch(
            cell_id="magi720", config_id="cfg", epoch_id="epoch", transport=FakeTransport(True),
            backend_factory=lambda: calls.append("load"), budget=budget(),
        )
        with self.assertRaisesRegex(MagiEpochAborted, "BOTH ranks"):
            worker.run(request())
        self.assertTrue(worker.poisoned)
        self.assertEqual(calls, [])
        with self.assertRaises(MagiEpochAborted):
            worker.run(request())

    def test_large_oracle_ladders_use_bounded_hash_agreement(self):
        control_limit = 4096

        class BoundedTransport(FakeTransport):
            def checkpoint(self, operation, *, details=None, error=None):
                if len(json.dumps(details).encode()) > control_limit:
                    raise AssertionError("request agreement broadcast a full large rank policy")
                return super().checkpoint(operation, details=details, error=error)

        large = {"allowed_ranks": {f"{layer}:{target}": list(range(1, 1025))
                                   for layer in range(48) for target in ("K", "V")}}
        self.assertGreater(len(json.dumps(large).encode()), control_limit)
        value = replace(request(), arm="adaptive", policy=large)
        worker = self.epoch(FakeBackend(), transport=BoundedTransport())
        worker.run(value)
        worker.close()

    def test_whole_request_admission_and_resolution_precede_load(self):
        for value, limits in ((request(), budget(1)), (request(cell="magi1080"), budget())):
            backend = FakeBackend()
            worker = self.epoch(backend, budget=limits)
            with self.assertRaises(MagiEpochAborted):
                worker.run(value)
            self.assertFalse(backend.namespaces)
            self.assertIsNone(worker.backend)

    def test_execution_or_cleanup_failure_poison_whole_epoch(self):
        for name in ("fail_generate", "fail_cleanup"):
            backend = FakeBackend()
            setattr(backend, name, True)
            worker = self.epoch(backend)
            with self.assertRaises(MagiEpochAborted):
                worker.run(request())
            self.assertTrue(worker.poisoned)
            self.assertTrue(backend.closed)
            self.assertIsNone(backend.active)
            self.assertFalse(backend.hooks)
            with self.assertRaises(MagiEpochAborted):
                worker.run(request("B", 4))

    def test_coordinated_peer_failure_never_enters_completion_collectives(self):
        backend = FakeBackend()
        transport = FakeTransport()
        worker = self.epoch(backend, transport=transport)
        with patch.object(backend, "generate", side_effect=PeerFailure("agreed capture failure")):
            with self.assertRaisesRegex(MagiEpochAborted, "BOTH ranks"):
                worker.run(request())
        self.assertNotIn("magi_request_completed_and_clean", transport.operations)
        self.assertNotIn("magi_request_rank_timing", transport.operations)
        self.assertTrue(worker.poisoned)
        self.assertTrue(backend.closed)
        self.assertIsNone(backend.active)

    def test_non_execution_thread_cannot_issue_collectives(self):
        import threading

        transport = FakeTransport()
        worker = self.epoch(FakeBackend(), transport=transport)
        errors = []

        def attempt():
            try:
                worker.run(request())
            except StateError as exc:
                errors.append(str(exc))

        thread = threading.Thread(target=attempt)
        thread.start()
        thread.join()
        self.assertEqual(len(errors), 1)
        self.assertIn("execution thread", errors[0])
        self.assertEqual(transport.operations, [])
        worker.close()

    def test_near_clean_duplicate_never_changes_stored_chunk_cursor(self):
        value = request()
        for step in range(16):
            kwargs = value.native_kwargs(1, clean=False, step=step, intervals=[0.1])
            self.assertEqual(kwargs["slice_point"], 1)
            self.assertEqual(kwargs["start_chunk_id"], 1)
            self.assertEqual(kwargs["end_chunk_id"], 2 + int(step >= 12))
            self.assertEqual(kwargs["denoising_range_num"], 1)
            self.assertEqual(kwargs["distill_nearly_clean_chunk"], step >= 12)
        clean = value.native_kwargs(1, clean=True, step=15, intervals=[0.1])
        self.assertEqual(clean["end_chunk_id"], 2)
        self.assertFalse(clean["distill_nearly_clean_chunk"])

    def test_exact_indexed_mixed_precision_shards_and_missing_assets(self):
        files, weight_map = {}, {}
        for name, dtype, count in (("a", "F32", 4), ("b", "BF16", 2)):
            shard = f"{name}.safetensors"
            header = json.dumps({name: {"dtype": dtype, "shape": [1], "data_offsets": [0, count]}}).encode()
            path = self.root / shard
            path.write_bytes(struct.pack("<Q", len(header)) + header + bytes(count))
            files[shard] = {"bytes": path.stat().st_size}
            weight_map[name] = shard
        index = self.root / "model.safetensors.index.json"
        index.write_text(json.dumps({"weight_map": weight_map}))
        files[index.name] = {"bytes": index.stat().st_size}
        result = validate_magi_checkpoint(self.root, files, index.name)
        self.assertEqual(result["shards"], ["a.safetensors", "b.safetensors"])
        self.assertFalse(result["blanket_dtype_conversion"])
        missing = dict(files)
        missing.pop("a.safetensors")
        with self.assertRaises(IntegrityError):
            validate_magi_checkpoint(self.root, missing, index.name)
        index.write_text(json.dumps({"weight_map": {"a": "../escape.safetensors"}}))
        with self.assertRaises(ConfigurationError):
            validate_magi_checkpoint(self.root, files, index.name)

    @contextmanager
    def verified_inputs(self, *, chunks=2, phase="native_preflight"):
        assets = self.root / "assets"
        assets.mkdir()
        config = {
            "model_config": {
                "num_layers": 48, "hidden_size": 6144, "num_attention_heads": 48,
                "num_query_groups": 8, "kv_channels": 128, "params_dtype": "torch.bfloat16",
            },
            "runtime_config": {
                "cfg_number": 1, "num_steps": 16, "chunk_width": 6, "video_size_h": 720,
                "video_size_w": 1280, "clean_t": 0.9999,
                "window_size": 1,
                "noise2clean_kvrange": [8, 8, 4, 4], "clean_chunk_kvrange": 4,
            },
            "engine_config": {
                "cp_size": 2, "cp_strategy": "cp_ulysses", "pp_size": 1, "ulysses_overlap_degree": 1,
                "distill": True, "fp8_quant": False, "enable_cuda_graph": False, "kv_offload": False,
            },
        }
        (assets / "config.json").write_text(json.dumps(config))
        (assets / "dit").mkdir()
        weights = {}
        for name, dtype, count in (("a", "F32", 4), ("b", "BF16", 2)):
            header = json.dumps({name: {"dtype": dtype, "shape": [1], "data_offsets": [0, count]}}).encode()
            (assets / "dit" / f"{name}.safetensors").write_bytes(struct.pack("<Q", len(header)) + header + bytes(count))
            weights[name] = f"{name}.safetensors"
        (assets / "dit" / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weights}))
        (assets / "t5").mkdir()
        for name in ("config.json", "spiece.model", "tokenizer_config.json", "special_tokens_map.json",
                     "model.safetensors"):
            (assets / "t5" / name).write_bytes(b"fake verified local asset")
        (assets / "vae.json").write_text("{}")
        (assets / "vae.safetensors").write_bytes(b"not loaded")
        (assets / "special_tokens.npz").write_bytes(b"not loaded")
        records = [{"name": str(path.relative_to(assets)), "sha256": file_sha256(path),
                    "bytes": path.stat().st_size} for path in assets.rglob("*") if path.is_file()]
        body = {"schema": "reelcache.study_assets.v1", "files": records,
                "total_bytes": sum(row["bytes"] for row in records)}
        manifest = {**body, "asset_id": identity(body)}
        (assets / "assets.json").write_text(json.dumps(manifest))
        source = self.root / "native_source" / "inference"
        source.mkdir(parents=True)
        native_file = source / "__init__.py"
        native_file.write_text("# CPU-only source closure fixture; never imported.\n")
        protocol = {"protocol_id": "p"}
        receipt = {"phase": phase, "source_closure": {
            "closure_id": "c", "files": {"inference/__init__.py": {"path": str(native_file)}}}}
        row = {
            "request_id": "request-a", "cell_id": "magi720", "prompt_id": "development-a",
            "case_id": "paired-case-a",
            "group_id": "development", "split": "development", "prompt": "Fixture prompt",
            "seed": 3, "chunks": chunks, "representation": {"kind": "raw"},
            "source_fps": 24, "history_mode": "reload", "output_namespace": "outputs/a", "arm_id": "raw",
            "budgets": {name: 1 << 45 for name in (
                "host_bytes", "device_bytes_per_rank", "disk_bytes", "history_bytes_per_rank",
                "workspace_bytes_per_rank", "source_staging_bytes_global",
            )} | {"disk_reserve_bytes": 1},
        }
        row["budgets"]["disk_bytes"] = 1 << 46
        batch = {
            "schema": "reelcache.rank_requests.v1", "requests_id": "batch", "protocol_id": "p",
            "cell_id": "magi720", "asset_id": manifest["asset_id"], "native_source": str(source.parent),
            "output": str(self.root / "new_output"), "resources": asdict(budget()),
            "requests": [row], "reconstruction_bytes": 1024, "store_bytes": 1024,
            "native_layer_bytes": 1 << 30, "spectrum_workspace_bytes": 1024,
            "cpu_threads": 1,
            "sampler_mode": "serial_clean_history",
            "asset_layout": {
                "config": "config.json", "dit_index": "dit/model.safetensors.index.json", "t5_directory": "t5",
                "vae_config": "vae.json", "vae_weights": "vae.safetensors", "special_tokens": "special_tokens.npz",
            },
        }
        for name, document in (("protocol", protocol), ("receipt", receipt), ("requests", batch)):
            (self.root / f"{name}.json").write_text(json.dumps(document))
        args = (self.root / "protocol.json", assets / "assets.json",
                self.root / "receipt.json", self.root / "requests.json")
        with patch("reelcache.adapters.magi_rank_worker.validate_protocol", return_value=protocol), patch(
            "reelcache.adapters.magi_rank_worker.verify_launch_receipt",
        ), patch("reelcache.adapters.magi_rank_worker.validate_request_batch", return_value=[row]), patch.object(
            torch.cuda, "set_device", side_effect=AssertionError("native load before CPU verification"),
        ):
            yield args, assets, batch, row

    def test_verified_asset_and_batch_seam_is_cpu_only_and_has_no_codec_fallback(self):
        with self.verified_inputs() as (args, assets, batch, row):
            checked = verify_worker_inputs(*args, check_storage=False)
            self.assertEqual(checked["requests"][0].latent_shape, (1, 16, 6, 90, 160))
            self.assertEqual(checked["collective_timeout_seconds"], 600)
            self.assertEqual(checked["sampler_contract"]["sampler_mode"], "serial_clean_history")
            self.assertEqual(checked["sampler_contract"]["sampler_window_size"], 1)
            self.assertTrue(checked["sampler_contract"]["standalone_clean_recapture"])
            self.assertFalse(checked["sampler_contract"]["native_window4_schedule_parity_claimed"])
            self.assertEqual(checked["input_bundles"][row["request_id"]]["binding"]["sampler_contract"],
                             checked["sampler_contract"])
            self.assertEqual(checked["checkpoint"]["shards"], ["dit/a.safetensors", "dit/b.safetensors"])
            for name, too_small in (("workspace_bytes_per_rank", 1), ("source_staging_bytes_global", 1)):
                previous = row["budgets"][name]
                row["budgets"][name] = too_small
                with self.assertRaisesRegex(ConfigurationError, "immutable request resource"):
                    verify_worker_inputs(*args, check_storage=False)
                row["budgets"][name] = previous
            row["representation"] = {"kind": "svd_hevc", "rank_k": 1, "rank_v": 1}
            with self.assertRaisesRegex(ConfigurationError, "verified local.*ffmpeg"):
                verify_worker_inputs(*args, check_storage=False)
            row["representation"] = {"kind": "raw"}
            (assets / "dit" / "a.safetensors").write_bytes(b"modified staged file")
            with self.assertRaisesRegex(IntegrityError, "staged asset changed"):
                verify_worker_inputs(*args, check_storage=False)
        self.assertFalse(torch.cuda.is_initialized())

    def test_wrong_or_omitted_sampler_mode_rejects_before_assets_or_native_loading(self):
        with self.verified_inputs() as (args, _, batch, _):
            for invalid in (None, True, 1, "window4", "serial", ""):
                if invalid is None:
                    batch.pop("sampler_mode", None)
                else:
                    batch["sampler_mode"] = invalid
                args[-1].write_text(json.dumps(batch))
                with self.subTest(mode=invalid), patch(
                    "reelcache.adapters.magi_rank_worker.verify_assets",
                    side_effect=AssertionError("assets inspected before sampler rejection"),
                ) as assets:
                    with self.assertRaisesRegex(ConfigurationError, "sampler_mode"):
                        verify_worker_inputs(*args, check_storage=False)
                    assets.assert_not_called()
        self.assertFalse(torch.cuda.is_initialized())

    def test_bad_collective_timeout_rejects_before_asset_or_model_work(self):
        with self.verified_inputs() as (args, _, batch, _):
            for invalid in (None, True, False, 0, -1, 21601, 600.0, "600"):
                batch["collective_timeout_seconds"] = invalid
                args[-1].write_text(json.dumps(batch))
                with self.subTest(timeout=invalid), patch(
                    "reelcache.adapters.magi_rank_worker.verify_assets",
                    side_effect=AssertionError("assets inspected before timeout rejection"),
                ) as assets:
                    with self.assertRaisesRegex(ConfigurationError, "collective_timeout_seconds"):
                        verify_worker_inputs(*args, check_storage=False)
                    assets.assert_not_called()

    def test_collective_timeout_is_covered_by_actual_shared_batch_hash_gate(self):
        from reelcache.study import validate_request_batch

        with self.verified_inputs() as (args, _, batch, row):
            receipt = json.loads(args[2].read_text())
            receipt["receipt_id"] = "fixture-receipt"
            args[2].write_text(json.dumps(receipt))
            batch.update(receipt_id=receipt["receipt_id"], phase="native_preflight",
                         collective_timeout_seconds=1800)
            batch["requests_id"] = identity({name: value for name, value in batch.items() if name != "requests_id"})
            args[-1].write_text(json.dumps(batch))
            with (
                patch("reelcache.adapters.magi_rank_worker.validate_request_batch", wraps=validate_request_batch),
                patch("reelcache.study.verify_launch_receipt"),
                patch("reelcache.study.validate_request", return_value=row),
            ):
                checked = verify_worker_inputs(*args, check_storage=False)
                self.assertEqual(checked["collective_timeout_seconds"], 1800)
                batch["collective_timeout_seconds"] = 3600
                args[-1].write_text(json.dumps(batch))
                with patch(
                    "reelcache.adapters.magi_rank_worker.verify_assets",
                    side_effect=AssertionError("assets inspected before batch-hash rejection"),
                ) as assets:
                    with self.assertRaisesRegex(IntegrityError, "batch identity"):
                        verify_worker_inputs(*args, check_storage=False)
                    assets.assert_not_called()

    def test_verified_window4_config_rejects_before_model_identity_or_execution(self):
        with self.verified_inputs() as (args, assets, batch, _):
            config_path = assets / "config.json"
            config = json.loads(config_path.read_text())
            config["runtime_config"]["window_size"] = 4
            config_path.write_text(json.dumps(config))
            manifest = json.loads((assets / "assets.json").read_text())
            manifest.pop("asset_id")
            for row in manifest["files"]:
                if row["name"] == "config.json":
                    row.update(sha256=file_sha256(config_path), bytes=config_path.stat().st_size)
            manifest["total_bytes"] = sum(row["bytes"] for row in manifest["files"])
            manifest["asset_id"] = identity(manifest)
            (assets / "assets.json").write_text(json.dumps(manifest))
            batch["asset_id"] = manifest["asset_id"]
            args[-1].write_text(json.dumps(batch))
            with patch(
                "reelcache.adapters.magi_rank_worker.magi_model_identity",
                side_effect=AssertionError("model preparation preceded sampler validation"),
            ) as model:
                with self.assertRaisesRegex(ValueError, "window_size=1"):
                    verify_worker_inputs(*args, check_storage=False)
                model.assert_not_called()
        self.assertFalse(torch.cuda.is_initialized())

    def test_complete_frozen_disk_grant_includes_retention_latents_and_json_before_filesystem(self):
        with self.verified_inputs(chunks=3) as (args, assets, batch, row):
            cell = study_cell("magi720")
            ledger = budget().admit(MagiGeometry.for_cell(cell.cell_id), chunks=3, factor_keys=False,
                                    observer=True, quality_hot=False)
            batch["resources"]["staging_bytes"] = ledger["staging_bytes"]
            batch["store_bytes"] = 16 << 30
            batch["retain_originals"] = [
                f"{row['request_id']}:{chunk}:{layer}:{target}"
                for chunk in range(3) for layer in range(cell.layers) for target in ("K", "V")
            ]
            old_incomplete_bound = ledger["staging_bytes"] + batch["store_bytes"]
            self.assertEqual(old_incomplete_bound, 32_043_433_984)
            row["budgets"]["disk_bytes"] = old_incomplete_bound
            args[-1].write_text(json.dumps(batch))
            with patch("reelcache.adapters.magi_rank_worker.shutil.disk_usage") as filesystem:
                with self.assertRaisesRegex(ConfigurationError, "complete MAGI request disk bound"):
                    verify_worker_inputs(*args)
                filesystem.assert_not_called()
            self.assertFalse(Path(batch["output"]).exists())
            row["budgets"]["disk_bytes"] = 1 << 46
            checked = verify_worker_inputs(*args, check_storage=False)
            disk = checked["disk_admissions"][0]
            self.assertEqual(disk["retained_original_payload_bytes"], cell.source_bytes * 3)
            payload_bound = old_incomplete_bound + disk["retained_original_payload_bytes"] + disk["latent_payload_bytes"]
            self.assertEqual(payload_bound, 51_170_320_384)
            self.assertGreater(disk["required_bytes"], payload_bound)
            self.assertEqual(disk["observation_count"], 288)
            self.assertGreater(disk["spectrum_bytes"], 0)
            self.assertGreater(disk["metadata_bytes"], 0)
            row["budgets"]["disk_bytes"] = disk["required_bytes"] - 1
            with self.assertRaisesRegex(ConfigurationError, "immutable request disk_bytes"):
                verify_worker_inputs(*args, check_storage=False)
            row["budgets"]["disk_bytes"] = disk["required_bytes"]
            self.assertEqual(
                verify_worker_inputs(*args, check_storage=False)["disk_admissions"][0]["required_bytes"],
                disk["required_bytes"],
            )

    def test_free_filesystem_is_checked_separately_from_frozen_disk_grant(self):
        with self.verified_inputs() as (args, assets, batch, row):
            checked = verify_worker_inputs(*args, check_storage=False)
            disk = checked["disk_admissions"][0]
            self.assertLess(disk["required_bytes"], row["budgets"]["disk_bytes"])
            required_free = checked["storage_admission_bytes"] + row["budgets"]["disk_reserve_bytes"]
            with patch("reelcache.adapters.magi_rank_worker.shutil.disk_usage",
                       return_value=SimpleNamespace(free=required_free - 1)):
                with self.assertRaisesRegex(CapacityError, "Insufficient storage for the complete MAGI batch"):
                    verify_worker_inputs(*args)
            self.assertFalse(Path(batch["output"]).exists())
            with patch("reelcache.adapters.magi_rank_worker.shutil.disk_usage",
                       return_value=SimpleNamespace(free=required_free)):
                admitted = verify_worker_inputs(*args)
            self.assertEqual(admitted["storage_admission_bytes"], checked["storage_admission_bytes"])
            self.assertFalse(Path(batch["output"]).exists())

    def test_eighty_case_RGB_admission_uses_aggregate_decode_peak_without_allocating_tensors(self):
        from reelcache.adapters.magi_rank_worker import _batch_disk_admission, _request_disk_admission

        cell = study_cell("magi720")
        staging = 2 * cell.source_bytes
        with patch.object(torch, "empty", side_effect=AssertionError("disk arithmetic allocated")), patch.object(
            torch, "zeros", side_effect=AssertionError("disk arithmetic allocated"),
        ), patch.object(torch, "randn", side_effect=AssertionError("disk arithmetic generated sources")):
            cases = [
                replace(request(f"rgb-{index}"), chunks=3, arm="svd", rank_k=256, rank_v=384,
                        observe_sources=False)
                for index in range(80)
            ]
            disks = [
                _request_disk_admission(
                    case, cell, staging_bytes=staging, store_bytes=1 << 30, retained={"K": 0, "V": 0},
                )
                for case in cases
            ]
            admitted = _batch_disk_admission(disks, staging_bytes=staging)
        persistent = sum(row["persistent_bytes"] for row in disks)
        rgb = 80 * 3 * 24 * 3 * 1280 * 720
        decoder_metadata = 80 * 3 * (1 << 20)
        self.assertEqual(admitted["retained_rgb_bytes"], rgb)
        self.assertEqual(admitted["decoder_metadata_bytes"], decoder_metadata)
        self.assertEqual(admitted["generation_peak_bytes"], persistent + staging)
        self.assertEqual(admitted["decode_peak_bytes"], persistent + rgb + decoder_metadata)
        self.assertEqual(admitted["required_bytes"], persistent + max(staging, rgb + decoder_metadata))
        self.assertEqual(admitted["required_bytes"] - (persistent + staging), 3_436_707_840)

    def test_RGB_and_decoder_metadata_are_inside_each_immutable_request_requirement(self):
        from reelcache.adapters.magi_rank_worker import _decode_disk_admission, _request_disk_admission

        for cell_id in ("magi720", "magi1080"):
            cell = study_cell(cell_id)
            value = replace(request(cell=cell_id), chunks=3, arm="svd", rank_k=256, rank_v=384,
                            observe_sources=False)
            disk = _request_disk_admission(
                value, cell, staging_bytes=1 << 20, store_bytes=1 << 20, retained={"K": 0, "V": 0},
            )
            decoded = _decode_disk_admission(cell, value.chunks)
            expected = 3 * (24 * 3 * cell.width * cell.height + (1 << 20))
            self.assertEqual(decoded["required_bytes"], expected)
            self.assertEqual(disk["retained_rgb_bytes"] + disk["decoder_metadata_bytes"], expected)
            self.assertEqual(disk["required_bytes"], disk["persistent_bytes"] + expected)
            self.assertGreater(disk["required_bytes"], disk["persistent_bytes"] + disk["staging_bytes"])

    def test_lossy_ranks_cannot_default_or_expand_to_query_width(self):
        for options in ({"arm": "svd"}, {"arm": "svd", "rank_k": 6144, "rank_v": 384},
                        {"arm": "raw", "rank_k": 128}, {"arm": "adaptive"}, {"source_fps": 30},
                        {"source_fps": 16},
                        {"case_id": None}, {"case_id": ""}):
            with self.assertRaises(ConfigurationError):
                replace(request(), **options)
        valid = replace(request(), arm="svd", rank_k=256, rank_v=384)
        self.assertEqual((valid.rank_k, valid.rank_v), (256, 384))
        self.assertEqual(valid.static_specs()[MatrixKey(47, "K")].rank, 256)
        self.assertEqual(valid.static_specs()[MatrixKey(47, "V")].rank, 384)

    def test_nonreference_observer_is_opt_in_but_raw_and_exact_defaults_preserve_capture(self):
        row = {
            "request_id": "source-mode-fixture", "cell_id": "magi720", "case_id": "paired-case",
            "trajectory_id": "declared-trajectory", "prompt": "Fixture prompt", "seed": 1,
            "split": "development", "chunks": 2, "source_fps": 24, "output_namespace": "mode-fixture",
        }
        for representation, expected in (
            ({"kind": "raw"}, True),
            ({"kind": "svd", "rank_k": 1, "rank_v": 2}, False),
            ({"kind": "svd", "rank_policy": {"solver": "qr_q2"}}, False),
            ({"kind": "svd", "rank_policy": {"solver": "exact_oracle"}}, True),
        ):
            with self.subTest(representation=representation):
                document = {**row, "representation": representation}
                self.assertIs(MagiRankRequest.from_study_request(document).observe_sources, expected)
                self.assertIs(MagiRankRequest.from_study_request(
                    {**document, "observe_sources": True},
                ).observe_sources, True)
                self.assertIs(MagiRankRequest.from_study_request(
                    {**document, "observe_sources": False},
                ).observe_sources, False)
                with self.assertRaises(ConfigurationError):
                    MagiRankRequest.from_study_request({**document, "observe_sources": None})
        direct = MagiRankRequest(
            "direct-mode", "magi720", "prompt", "group", "calibration", "prompt text", 1, 2, "paired",
            arm="adaptive", policy={"solver": "qr_q2"},
        )
        self.assertIs(direct.observe_sources, False)

    def test_default_approximate_policy_installs_no_source_observer(self):
        from reelcache.adapters.magi_native import _StudyBackend

        backend = object.__new__(_StudyBackend)
        backend.cell = study_cell("magi720")
        backend.model = object()
        backend.rotary = object()
        backend.device = "cpu"
        backend.transport = SimpleNamespace(owner_call=lambda operation, action: action())
        backend.verified = {
            "batch": {"reconstruction_bytes": 1024, "store_bytes": 4096, "native_layer_bytes": 1024},
            "budget": budget(),
        }
        base = MagiRankRequest(
            "approximate-mode", "magi720", "prompt", "group", "calibration", "prompt text", 1, 2, "paired",
            arm="adaptive", policy={"solver": "qr_q2"},
        )
        for capture in (False, True):
            with self.subTest(capture=capture), patch(
                "reelcache.manager.KVRuntime",
            ) as runtime, patch("reelcache.adapters.magi.MagiAdapter") as adapter:
                value = base if not capture else replace(base, observe_sources=True)
                state = {
                    "directory": self.root, "namespace": "observer-mode", "cache": object(),
                    "source_context": {},
                }
                backend._install_request_adapter(state, value, object(), object())
                runtime.assert_called_once()
                adapter.assert_called_once()
                observer = adapter.call_args.kwargs["source_observer"]
                self.assertEqual(observer, backend._observe if capture else None)
                adapter.return_value.attach.assert_called_once()

    def test_shared_static_layer_vectors_are_complete_frozen_and_parent_bound(self):
        with self.verified_inputs(phase="calibration_frontier") as (args, assets, batch, row):
            vectors = {"K": [1 + layer % 2 for layer in range(48)],
                       "V": [3 + layer % 2 for layer in range(48)]}
            row["representation"] = {"kind": "svd", "layer_ranks": vectors}
            row["trajectory_id"] = "parent-declared-static-trajectory"
            value = verify_worker_inputs(*args, check_storage=False)["requests"][0]
            self.assertEqual(value.case_id, row["case_id"])
            self.assertEqual(value.trajectory_id, row["trajectory_id"])
            self.assertEqual(value.arm, "svd")
            self.assertEqual(value.rank_selection_mode, "static_layerwise")
            self.assertIsNone(value.policy)
            self.assertIsNone(value.rank_k)
            self.assertIsNone(value.rank_v)
            before = value.request_hash
            expected = copy.deepcopy(vectors)
            vectors["K"][0] = 7
            self.assertEqual(value.layer_rank_map, expected)
            self.assertEqual(value.request_hash, before)
            changed = replace(value, layer_ranks=vectors)
            self.assertNotEqual(changed.request_hash, before)
            self.assertEqual(changed.case_id, value.case_id)
            with self.assertRaises(TypeError):
                value.layer_ranks[0][0] = 7
            exported = value.layer_rank_map
            exported["V"][0] = 8
            self.assertEqual(value.layer_rank_map, expected)
            specs = value.static_specs()
            self.assertEqual(len(specs), 96)
            for key, spec in specs.items():
                self.assertEqual(spec.rank, expected[key.target][key.layer])
                self.assertEqual(spec.kind, "svd")
            hevc = replace(value, arm="svd_hevc").static_specs()[MatrixKey(1, "V")]
            self.assertEqual((hevc.rank, hevc.kind, hevc.qp, hevc.pix_fmt), (4, "svd_hevc", 8, "yuv444p"))

    def test_static_layer_vectors_reject_ambiguous_incomplete_and_invalid_profiles(self):
        vectors = {"K": [1] * 48, "V": [2] * 48}
        for invalid in (
            {"K": [1] * 48}, {**vectors, "Q": [1] * 48},
            {"K": [1] * 47, "V": [2] * 48},
            *({"K": [value] + [1] * 47, "V": [2] * 48} for value in (0, -1, 1025, True, 1.0)),
        ):
            with self.subTest(profile=invalid):
                with self.assertRaises(ConfigurationError):
                    replace(request(), arm="svd", layer_ranks=invalid)
        for overrides in (
            {"arm": "raw"},
            {"arm": "adaptive", "policy": {"anchor_mode": "none"}},
            {"arm": "svd", "rank_k": 1, "rank_v": 2},
        ):
            with self.assertRaises(ConfigurationError):
                replace(request(), layer_ranks=vectors, **overrides)

    def test_shared_rank_modes_reject_mixed_presence_and_ignore_only_paired_adaptive_hints(self):
        row = {
            "request_id": "rank-mode", "cell_id": "magi720", "case_id": "same-paired-case",
            "prompt": "Same heldout prompt", "split": "heldout", "seed": 7, "chunks": 2,
            "source_fps": 24, "output_namespace": "mode-output",
        }
        vectors = {"K": [1] * 48, "V": [2] * 48}
        for extra in (
            {"rank_k": None}, {"rank_v": None}, {"rank_policy": None},
            {"rank_k": 1, "rank_v": 2}, {"rank_policy": {"solver": "qr_q2"}},
        ):
            with self.subTest(extra=extra), self.assertRaises(ConfigurationError):
                MagiRankRequest.from_study_request({
                    **row, "representation": {"kind": "svd", "layer_ranks": vectors, **extra},
                })
        for representation, mode in (
            ({"kind": "raw"}, "raw"),
            ({"kind": "svd", "rank_k": 1, "rank_v": 2}, "uniform"),
            ({"kind": "svd", "layer_ranks": vectors}, "static_layerwise"),
            ({"kind": "svd", "rank_policy": {"solver": "qr_q2"}}, "adaptive"),
            ({"kind": "svd", "rank_policy": {"solver": "qr_q2"}, "rank_k": 3, "rank_v": 4}, "adaptive"),
        ):
            with self.subTest(mode=mode):
                value = MagiRankRequest.from_study_request({**row, "representation": representation})
                self.assertEqual(value.rank_selection_mode, mode)
                self.assertEqual(value.case_id, row["case_id"])
                if mode == "adaptive":
                    self.assertIsNone(value.rank_k)
                    self.assertIsNone(value.rank_v)
                    with self.assertRaises(ConfigurationError):
                        value.static_specs()
        for hint in ("rank_k", "rank_v"):
            with self.assertRaises(ConfigurationError):
                MagiRankRequest.from_study_request({
                    **row, "representation": {"kind": "svd", "rank_policy": {"solver": "qr_q2"}, hint: 1},
                })

    def test_native_noop_admits_full_dense_history_and_preflight_instrumentation(self):
        with self.verified_inputs(chunks=3) as (args, assets, batch, row):
            batch["preflight_backend"] = "native_noop"
            args[-1].write_text(json.dumps(batch))
            checked = verify_worker_inputs(*args, check_storage=False)
            full = study_cell("magi720").native_kv_bytes * 3 // 2
            self.assertEqual(checked["admissions"][0]["native_history_current_bytes"], full)
            self.assertGreater(checked["admissions"][0]["preflight_host_bytes"], 0)
            self.assertGreater(checked["disk_admissions"][0]["input_bundle_bytes"], 0)
            self.assertGreater(checked["disk_admissions"][0]["attention_digest_bytes"], 0)
            self.assertEqual(checked["disk_admissions"][0]["store_bytes"], 0)
            row["budgets"]["history_bytes_per_rank"] = full - 1
            with self.assertRaisesRegex(ConfigurationError, "entire native dense history"):
                verify_worker_inputs(*args, check_storage=False)
            row["budgets"]["history_bytes_per_rank"] = full
            row["representation"] = {"kind": "svd", "rank_k": 1, "rank_v": 1}
            with self.assertRaisesRegex(ConfigurationError, "native_preflight RAW"):
                verify_worker_inputs(*args, check_storage=False)

    def test_native_noop_rejects_nonpreflight_before_assets_or_models(self):
        with self.verified_inputs(phase="calibration_census") as (args, assets, batch, row):
            batch["preflight_backend"] = "native_noop"
            args[-1].write_text(json.dumps(batch))
            with patch("reelcache.adapters.magi_rank_worker.verify_assets",
                       side_effect=AssertionError("assets inspected before phase rejection")):
                with self.assertRaisesRegex(ConfigurationError, "native_preflight RAW"):
                    verify_worker_inputs(*args, check_storage=False)

    def test_raw_census_requires_frozen_detector_and_complete_workspace_output_admission(self):
        from reelcache.factorization import DetectorSpec, rank_fit_workspace_bytes

        with self.verified_inputs(phase="calibration_census") as (args, assets, batch, row):
            row["split"] = "calibration"
            with self.assertRaisesRegex(ConfigurationError, "complete frozen detector"):
                verify_worker_inputs(*args, check_storage=False)
            cell = study_cell("magi720")
            needed = cell.tokens * cell.channels * 4 + rank_fit_workspace_bytes(
                (cell.tokens, cell.channels), cell.channels, solver_workspace_bytes=1 << 20,
            )
            batch.update(detector_spec=DetectorSpec(cpu_threads=1).to_dict(),
                         detector_workspace_bytes=needed - 1, detector_solver_workspace_bytes=1 << 20)
            batch["resources"]["workspace_host_bytes"] = 1 << 34
            args[-1].write_text(json.dumps(batch))
            with self.assertRaisesRegex(ConfigurationError, "full-width fallback"):
                verify_worker_inputs(*args, check_storage=False)
            batch["detector_workspace_bytes"] = needed
            args[-1].write_text(json.dumps(batch))
            checked = verify_worker_inputs(*args, check_storage=False)
            self.assertEqual(checked["detector_admission_bytes"], needed)
            self.assertEqual(checked["detector_spec"].initial_width, 128)
            self.assertEqual(checked["detector_spec"].feature_energies, (.95, .99))
            self.assertGreater(checked["disk_admissions"][0]["detector_record_bytes"], 0)
            self.assertFalse(checked["preflight_resources"])
            for split in ("development", "heldout"):
                row["split"] = split
                with self.subTest(split=split), self.assertRaisesRegex(ConfigurationError, "calibration split"):
                    verify_worker_inputs(*args, check_storage=False)
            row["split"] = "calibration"
            batch["cpu_threads"] = 2
            args[-1].write_text(json.dumps(batch))
            with patch("reelcache.adapters.magi_rank_worker.worker_thread_allowance", return_value=2):
                with self.assertRaisesRegex(ConfigurationError, "CPU threads differ"):
                    verify_worker_inputs(*args, check_storage=False)
            batch.update(cpu_threads=8, detector_spec=DetectorSpec(cpu_threads=8).to_dict())
            args[-1].write_text(json.dumps(batch))
            with patch("reelcache.adapters.magi_rank_worker.os.sched_getaffinity",
                       return_value=set(range(32))), patch.dict(os.environ, {"SLURM_CPUS_PER_TASK": "32"}):
                eight_threads = verify_worker_inputs(*args, check_storage=False)
            self.assertEqual(eight_threads["detector_spec"].cpu_threads, 8)
            self.assertNotEqual(eight_threads["detector_spec"].fingerprint,
                                checked["detector_spec"].fingerprint)
            self.assertEqual(torch.get_num_threads(), 1)
            with patch("reelcache.adapters.magi_rank_worker.os.sched_getaffinity",
                       return_value=set(range(32))), patch.dict(os.environ, {"SLURM_CPUS_PER_TASK": "8"}):
                with self.assertRaisesRegex(ConfigurationError, "worker threads exceed the CPU allocation"):
                    verify_worker_inputs(*args, check_storage=False)
            batch["cpu_threads"] = 1
            args[-1].write_text(json.dumps(batch))
            with self.assertRaisesRegex(ConfigurationError, "CPU threads differ"):
                verify_worker_inputs(*args, check_storage=False)
            batch["detector_spec"] = DetectorSpec(cpu_threads=1).to_dict()
            args[-1].write_text(json.dumps(batch))
            row["observe_sources"] = False
            with self.assertRaisesRegex(ConfigurationError, "observe every original"):
                verify_worker_inputs(*args, check_storage=False)

    def test_raw_census_observer_rejects_heldout_before_any_source_inspection_or_fit(self):
        from reelcache.adapters.magi_native import _StudyBackend

        backend = _StudyBackend.__new__(_StudyBackend)
        backend.verified = {"receipt": {"phase": "calibration_census"}, "detector_spec": object(),
                            "batch": {"cpu_threads": 1}}
        for split, declared_split in (("heldout", "heldout"), ("heldout", "calibration"),
                                      ("calibration", "heldout"), ("development", "development")):
            backend.active = {"request": replace(request(), split=split)}
            with self.subTest(split=split, declared_split=declared_split), patch(
                "reelcache.spectra.exact_spectrum", side_effect=AssertionError("heldout source solved"),
            ) as oracle, patch(
                "reelcache.factorization.acquire_anchor_directions",
                side_effect=AssertionError("heldout detector acquired"),
            ) as detector:
                with self.assertRaisesRegex(IntegrityError, "calibration split"):
                    backend._observe({"split": declared_split}, object())
                oracle.assert_not_called()
                detector.assert_not_called()
        backend.active = {"request": request()}
        backend.verified["detector_spec"] = SimpleNamespace(cpu_threads=2)
        with self.assertRaisesRegex(IntegrityError, "CPU threads differ"):
            backend._observe({"split": "calibration"}, object())
        self.assertFalse(any(self.root.iterdir()))

    def test_adaptive_detector_thread_mismatch_rejects_before_native_context_admission(self):
        from reelcache.factorization import DetectorSpec

        with self.verified_inputs(phase="calibration_frontier") as (args, _, batch, row):
            row["representation"] = {"kind": "svd", "rank_policy": {"solver": "qr_q2"}}
            batch["cpu_threads"] = 8
            args[-1].write_text(json.dumps(batch))
            model = object()
            policy = SimpleNamespace(
                model=model, codec=CodecSpec(kind="svd", rank=1), solver="qr_q2",
                detector_spec=DetectorSpec(cpu_threads=1), validate_context=Mock(),
            )
            with (
                patch("reelcache.adapters.magi_rank_worker.os.sched_getaffinity", return_value=set(range(32))),
                patch.dict(os.environ, {"SLURM_CPUS_PER_TASK": "32"}),
                patch("reelcache.adapters.magi_rank_worker.magi_model_identity", return_value=model),
                patch("reelcache.online_rank.RankPolicy.from_dict", return_value=policy),
            ):
                with self.assertRaisesRegex(ConfigurationError, "Adaptive detector CPU threads differ"):
                    verify_worker_inputs(*args, check_storage=False)
            policy.validate_context.assert_not_called()
        self.assertFalse(torch.cuda.is_initialized())

    def test_input_bundle_reencodes_conditioning_and_replays_each_actual_postdraw_state(self):
        from reelcache.adapters.magi_preflight import MagiInputBundle, verify_input_bundle

        binding = {"prompt": "A", "seed": 13, "cell_id": "CPU-fixture", "chunks": 3,
                   "latent_shape": [1, 1, 1, 2, 2], "asset_id": "a" * 64,
                   "config_id": "b" * 64, "launch_closure_id": "c" * 64,
                   "sampler_contract": serial_sampler_contract()}
        conditioning = {"y": torch.ones(2, 1, 2, 2), "masks": torch.ones(2, 1, 2),
                        "null_caption": torch.zeros(1, 2, 2), "null_mask": torch.ones(1, 2)}
        capture = MagiInputBundle(mode="capture", binding=binding, directory=self.root / "inputs", max_bytes=1 << 20)
        capture.conditioning(conditioning)
        generator = torch.Generator().manual_seed(13)
        draws, states = [], []
        for index in range(3):
            noise = torch.randn(binding["latent_shape"], generator=generator)
            after = generator.get_state().clone()
            capture.noise(index, noise, after)
            draws.append(noise.clone())
            states.append(after)
        receipt = capture.finish()
        path, sha = receipt["artifact"]["path"], receipt["artifact"]["sha256"]
        replay = MagiInputBundle(mode="replay", binding=binding, path=path, sha256=sha, max_bytes=1 << 20)
        replay.conditioning({name: tensor.clone() for name, tensor in conditioning.items()})
        repeated = torch.Generator().manual_seed(13)
        for index in range(3):
            noise = torch.randn(binding["latent_shape"], generator=repeated)
            record, explicit = replay.noise(index, noise, repeated.get_state())
            self.assertTrue(torch.equal(explicit, draws[index]))
            self.assertTrue(torch.equal(repeated.get_state(), states[index]))
            self.assertEqual(record["tensors"]["postdraw_generator_state"]["sha256"], _tensor_hash(states[index]))
        self.assertEqual(replay.finish()["bundle_id"], receipt["bundle_id"])
        with self.assertRaisesRegex(IntegrityError, "binding differs"):
            verify_input_bundle(path, sha, {**binding, "prompt": "B"}, 1 << 20)
        changed_sampler = copy.deepcopy(binding)
        changed_sampler["sampler_contract"]["sampler_window_size"] = 4
        with self.assertRaisesRegex(IntegrityError, "binding differs"):
            verify_input_bundle(path, sha, changed_sampler, 1 << 20)
        wrong = MagiInputBundle(mode="replay", binding=binding, path=path, sha256=sha)
        with self.assertRaisesRegex(IntegrityError, "Re-encoded conditioning"):
            wrong.conditioning({**conditioning, "y": conditioning["y"] + 1})
        wrong = MagiInputBundle(mode="replay", binding=binding, path=path, sha256=sha)
        wrong.conditioning(conditioning)
        with self.assertRaisesRegex(IntegrityError, "postdraw generator state"):
            wrong.noise(0, draws[0], states[-1])
        first = Path(path).parent / "chunk0.reelcache"
        first.write_bytes(first.read_bytes() + b"changed")
        with self.assertRaises(IntegrityError):
            verify_input_bundle(path, sha, binding)

    def test_native_noop_installs_neither_runtime_nor_adapter_and_cleans_fresh_cache(self):
        from reelcache.adapters.magi_native import _StudyBackend

        backend = _StudyBackend.__new__(_StudyBackend)
        backend.verified = {"preflight_backend": "native_noop", "receipt": {"phase": "native_preflight"}}
        backend.torch, backend.device = torch, torch.device("cpu")
        attention = SimpleNamespace(attn_weights_history=[1])
        backend.model = SimpleNamespace(videodit_blocks=SimpleNamespace(
            layers=[SimpleNamespace(self_attention=attention)],
        ))
        state = {"adapter": None, "runtime": None,
                 "cache": SimpleNamespace(key_value_memory_dict={0: torch.ones(2)})}
        backend.active = state
        with patch("reelcache.manager.KVRuntime", side_effect=AssertionError("baseline created a runtime")), patch(
            "reelcache.adapters.magi.MagiAdapter", side_effect=AssertionError("baseline created an adapter"),
        ):
            backend._install_request_adapter(state, request(), None, None)
        self.assertIsNone(state["runtime"])
        self.assertIsNone(state["adapter"])
        cache = state["cache"]
        backend.clear_request(state)
        self.assertFalse(cache.key_value_memory_dict)
        self.assertFalse(attention.attn_weights_history)
        self.assertIsNone(backend.active)

    def test_native_noop_new_request_reencodes_A_B_A_and_never_installs_runtime(self):
        from reelcache.adapters.magi_native import _StudyBackend

        module = ModuleType("inference.common")

        class InferenceParams:
            def __init__(self, **values):
                self.__dict__.update(values)
                self.key_value_memory_dict = {}

        module.InferenceParams = InferenceParams
        backend = _StudyBackend.__new__(_StudyBackend)
        backend.torch, backend.device = torch, torch.device("cpu")
        backend.output, backend.active = self.root, None
        backend.cell = study_cell("magi720")
        backend.transport = DistributedOwner(expected_world_size=1)
        backend.verified = {
            "preflight_backend": "native_noop",
            "collective_timeout_seconds": 3600,
            "receipt": {"phase": "native_preflight", "source_closure": {"closure_id": "c" * 64}},
            "protocol": {"protocol_id": "a" * 64}, "assets": {"asset_id": "b" * 64},
            "config": {"fixture": True},
        }
        backend.model = SimpleNamespace(
            y_embedder=SimpleNamespace(null_caption_embedding=torch.zeros(800, 4096)),
            videodit_blocks=SimpleNamespace(layers=[]),
        )
        texts = []

        def encode(prompts):
            texts.extend(prompts)
            scalar = sum(map(ord, prompts[0])) % 37
            return torch.full((1, 800, 4096), float(scalar)), torch.ones(1, 800, dtype=torch.int64)

        backend.encoder = SimpleNamespace(get_text_embeddings=encode)
        histories, states = [], []
        with patch.dict("sys.modules", {"inference.common": module}), patch(
            "reelcache.manager.KVRuntime", side_effect=AssertionError("native baseline created runtime"),
        ), patch("reelcache.adapters.magi.MagiAdapter", side_effect=AssertionError("native baseline created adapter")):
            for index, value in enumerate((request(), request("B", 29), request())):
                state = backend.new_request(value, f"native-fixture-{index}")
                histories.append(state["cache"])
                states.append((_tensor_hash(state["y"]), _tensor_hash(state["generator"].get_state())))
                self.assertIsNone(state["runtime"])
                self.assertIsNone(state["adapter"])
                for name, expected in serial_sampler_contract().items():
                    self.assertEqual(state["source_context"][name], expected)
                self.assertEqual(state["source_context"]["collective_timeout_seconds"], 3600)
                self.assertFalse(state["source_context"]["collective_timeout_is_latency_proof"])
                backend.clear_request(state)
        self.assertEqual(texts, [request().prompt, request("B", 29).prompt, request().prompt])
        self.assertEqual(states[0], states[2])
        self.assertNotEqual(states[0], states[1])
        self.assertEqual(len({id(cache) for cache in histories}), 3)
        self.assertTrue(all(not cache.key_value_memory_dict for cache in histories))

    def test_common_generation_loop_replays_bundle_without_baseline_runtime_claims(self):
        from reelcache.adapters.magi_native import _StudyBackend
        from reelcache.adapters.magi_preflight import MagiInputBundle, AttentionDigestRecorder

        geometry = MagiGeometry(layers=1, kv_heads=2, head_dim=4, frames=1, height=2, width=2, world_size=1)
        original = request()
        value = SimpleNamespace(
            chunks=2, latent_shape=(1, 1, 1, 2, 2), request_hash=original.request_hash,
            native_kwargs=original.native_kwargs,
        )
        binding = {"prompt": "CPU fixture", "seed": 13, "chunks": 2, "latent_shape": list(value.latent_shape)}
        outputs, receipts = [], []

        class Adapter:
            source_context = {}

            def execution(self, **kwargs):
                return nullcontext()

            def seal_chunk(self, chunk):
                return "fixture-only"

            def snapshot(self, **kwargs):
                return "fixture-only-snapshot"

            def memory_report(self):
                return {"fixture": True}

        for trial, mode in (
            ("reelcache", "reelcache"), ("native_noop", "native_noop"),
            ("changed-final-rng", "native_noop"), ("invalid-final-rng", "native_noop"),
        ):
            directory = self.root / trial
            directory.mkdir()
            backend = _StudyBackend.__new__(_StudyBackend)
            backend.torch, backend.device = torch, torch.device("cpu")
            backend.transport = DistributedOwner(expected_world_size=1)
            backend.cell = SimpleNamespace(tokens=4, crop_top=0, height=4)
            backend.config = SimpleNamespace(runtime_config=SimpleNamespace(noise2clean_kvrange=[2] * 4,
                                                                          clean_chunk_kvrange=2, num_frames=120))
            backend.verified = {"config": {"fixture": True}, "batch": {}, "preflight_backend": mode}
            backend.schedule, backend.intervals = serial_timetable()
            state = {
                "request": value, "adapter": Adapter() if mode == "reelcache" else None,
                "runtime": None, "namespace": trial, "directory": directory,
                "generator": torch.Generator().manual_seed(13), "noise_hashes": [], "latents": [],
                "y": torch.ones(2, 1, 2, 2), "masks": torch.ones(2, 1, 2),
                "null_caption": torch.zeros(1, 2, 2), "null_mask": torch.ones(1, 2),
                "cache": SimpleNamespace(update_kv_cache=True, key_value_memory_dict={}),
                "source_context": {"conditioning_sha256": "fixture-only", "fixture": True, "native_execution": False},
                "input_records": [], "attention_recorder": AttentionDigestRecorder(
                    directory / "attention.jsonl", geometry=geometry, rank=0, binding_id=identity(binding),
                    max_chunks=2, max_records=66, max_bytes=1 << 20,
                ),
            }
            if mode == "reelcache":
                bundle = MagiInputBundle(mode="capture", binding=binding, directory=directory / "inputs")
            else:
                bundle = MagiInputBundle(mode="replay", binding=binding,
                                         path=receipts[0]["artifact"]["path"],
                                         sha256=receipts[0]["artifact"]["sha256"])
            state["input_bundle"] = bundle
            state["conditioning_input"] = bundle.conditioning(
                {name: state[name] for name in ("y", "masks", "null_caption", "null_mask")},
            )
            backend.active = state

            def forward(tensor, *args, **kwargs):
                chunk, near = kwargs["slice_point"], kwargs["distill_nearly_clean_chunk"]
                rows = (chunk + 1 + int(near)) * geometry.tokens
                k = torch.ones((rows, 2, 4), dtype=torch.bfloat16)
                meta = SimpleNamespace(
                    slice_point=chunk, distill_nearly_clean_chunk=near,
                    core_attn_params=SimpleNamespace(
                        np_q_range=[[i * 4, (i + 1) * 4] for i in range(1 + near)],
                        np_k_range=[[0, rows] for _ in range(1 + near)],
                    ),
                )
                backend._observe_attention(0, k, k + 1, meta)
                if chunk == value.chunks - 1 and state["attention_point"]["phase"] == "clean":
                    if trial == "changed-final-rng":
                        torch.rand(1, generator=state["generator"])
                    elif trial == "invalid-final-rng":
                        state["generator"] = SimpleNamespace(
                            get_state=lambda: torch.zeros(65537, dtype=torch.uint8),
                        )
                return torch.zeros_like(tensor)

            backend.model = SimpleNamespace(forward_dispatcher=forward, forward=forward)
            if trial == "invalid-final-rng":
                with patch.object(backend.transport, "checkpoint", wraps=backend.transport.checkpoint) as checkpoint:
                    with self.assertRaises(PeerFailure):
                        backend.generate(state)
                final = [call for call in checkpoint.call_args_list if call.args[0] == "magi_final_noise_rng"]
                self.assertEqual(len(final), 1)
                self.assertIsInstance(final[0].kwargs["error"], CapacityError)
                self.assertIsNone(final[0].kwargs["details"])
                self.assertFalse((directory / "result.json").exists())
                continue
            result = backend.generate(state)
            outputs.append(result)
            receipts.append(result["input_bundle"])
            self.assertEqual(len(result["latents"]), 2)
            self.assertEqual(backend.config.runtime_config.num_frames, 120)
            self.assertEqual(result["attention_digests"][0]["records"], 34)
            raw = state["generator"].get_state().numpy().tobytes()
            self.assertEqual(result["final_noise_rng"], [{
                "rank": 0, "dtype": "uint8", "shape": [len(raw)], "state_hex": raw.hex(),
                "sha256": hashlib.sha256(raw).hexdigest(), "binding_id": result["input_bundle"]["binding_id"],
            }])
            self.assertEqual(json.loads((directory / "result.json").read_text())["final_noise_rng"],
                             result["final_noise_rng"])
        for other in (1, 2):
            self.assertEqual(receipts[0]["bundle_id"], receipts[other]["bundle_id"])
            self.assertEqual(outputs[0]["chunk_inputs"], outputs[other]["chunk_inputs"])
            self.assertEqual([row["tensor_sha256"] for row in outputs[0]["latents"]],
                             [row["tensor_sha256"] for row in outputs[other]["latents"]])
            self.assertEqual(Path(outputs[0]["attention_digests"][0]["artifact"]["path"]).read_bytes(),
                             Path(outputs[other]["attention_digests"][0]["artifact"]["path"]).read_bytes())
        self.assertEqual(outputs[0]["final_noise_rng"], outputs[1]["final_noise_rng"])
        self.assertNotEqual(outputs[0]["final_noise_rng"], outputs[2]["final_noise_rng"])
        self.assertFalse(outputs[1]["runtime_created"])
        self.assertIsNone(outputs[1]["snapshot_id"])
        self.assertIsNone(outputs[1]["storage_owner_record"])
        self.assertFalse(outputs[1]["native_parity_verified"])

    def test_final_private_rng_record_is_read_only_bounded_and_disk_admitted(self):
        from reelcache.adapters.magi_preflight import (
            FINAL_NOISE_RNG_MAX_BYTES, final_noise_rng_record, preflight_allowances,
        )
        from reelcache.adapters.magi_rank_worker import _request_disk_admission

        generator = torch.Generator().manual_seed(13)
        torch.rand(5, generator=generator)
        before = generator.get_state().clone()
        row = final_noise_rng_record(generator, rank=0, binding_id="a" * 64)
        self.assertTrue(torch.equal(before, generator.get_state()))
        self.assertEqual(bytes.fromhex(row["state_hex"]), before.numpy().tobytes())
        self.assertEqual(row["sha256"], hashlib.sha256(bytes.fromhex(row["state_hex"])).hexdigest())
        for state, error in (
            (torch.zeros(0, dtype=torch.uint8), CapacityError),
            (torch.zeros(FINAL_NOISE_RNG_MAX_BYTES + 1, dtype=torch.uint8), CapacityError),
            (torch.zeros((1, 1), dtype=torch.uint8), IntegrityError),
            (torch.zeros(1, dtype=torch.int64), IntegrityError),
        ):
            with self.subTest(shape=state.shape, dtype=state.dtype), self.assertRaises(error):
                final_noise_rng_record(SimpleNamespace(get_state=lambda: state), rank=0, binding_id="a" * 64)
        maximum = final_noise_rng_record(
            SimpleNamespace(get_state=lambda: torch.zeros(FINAL_NOISE_RNG_MAX_BYTES, dtype=torch.uint8)),
            rank=1, binding_id="a" * 64,
        )
        value, cell = request(), study_cell("magi720")
        allowance = preflight_allowances(value, cell)
        metadata = 2 * cell.world_size * (2 * FINAL_NOISE_RNG_MAX_BYTES + 4096)
        self.assertEqual(allowance["final_noise_rng_metadata_bytes"], metadata)
        self.assertLess(2 * len(json.dumps({"final_noise_rng": [maximum] * cell.world_size}, indent=2)), metadata)
        arguments = dict(staging_bytes=1024, store_bytes=0, retained={"K": 0, "V": 0})
        admitted = _request_disk_admission(value, cell, preflight=allowance, **arguments)
        old = _request_disk_admission(
            value, cell, preflight={**allowance, "final_noise_rng_metadata_bytes": 0}, **arguments,
        )
        for name in ("metadata_bytes", "persistent_bytes", "required_bytes"):
            self.assertEqual(admitted[name] - old[name], metadata)

    def test_attention_records_match_actual_post_rope_buffers_all_steps_and_visibility(self):
        from reelcache.adapters.magi_preflight import AttentionDigestRecorder

        geometry = MagiGeometry(layers=1, kv_heads=2, head_dim=4, frames=1, height=2, width=2, world_size=2)
        records = []
        for trial in ("reelcache", "native-noop"):
            recorder = AttentionDigestRecorder(
                self.root / f"{trial}.jsonl", geometry=geometry, rank=0,
                binding_id="a" * 64, max_chunks=2, max_records=66, max_bytes=1 << 20,
            )
            for chunk in range(2):
                for phase, step in [*(("denoise", step) for step in range(16)), ("clean", None)]:
                    near = phase == "denoise" and step >= 12
                    rows = (chunk + 1 + int(near)) * geometry.tokens
                    k = torch.ones((rows, 1, 4), dtype=torch.bfloat16)
                    v = k * 2
                    point = {"chunk_index": chunk, "phase": phase, "step_index": step,
                             "input_noise_sha256": "b" * 64}
                    before = (_tensor_hash(k), _tensor_hash(v))
                    row = recorder.record(point, 0, k, v, near_clean=near,
                                          query_ranges=[[i * 4, (i + 1) * 4] for i in range(1 + near)],
                                          key_ranges=[[0, rows] for _ in range(1 + near)])
                    self.assertEqual(before, (_tensor_hash(k), _tensor_hash(v)))
                    self.assertEqual(row["archived_rows"], chunk * 4)
                    self.assertEqual(row["archived_key_sha256"], _tensor_hash(k[:chunk * 4]))
                    self.assertEqual(row["key_space"], "post_rope")
            receipt = recorder.finish()
            self.assertEqual(receipt["records"], 34)
            self.assertEqual(file_sha256(receipt["artifact"]["path"]), receipt["artifact"]["sha256"])
            records.append(Path(receipt["artifact"]["path"]).read_bytes())
        self.assertEqual(records[0], records[1])
        recorder = AttentionDigestRecorder(self.root / "invalid.jsonl", geometry=geometry, rank=0,
                                          binding_id="a" * 64, max_chunks=1, max_records=33, max_bytes=1024)
        try:
            with self.assertRaisesRegex(IntegrityError, "post-RoPE BF16"):
                recorder.record({"chunk_index": 0, "phase": "clean", "step_index": None,
                                 "input_noise_sha256": "b" * 64}, 0,
                                torch.ones(4, 1, 4), torch.ones(4, 1, 4),
                                query_ranges=[[0, 4]], key_ranges=[[0, 4]], near_clean=False)
            with self.assertRaisesRegex(IntegrityError, "coverage omitted"):
                recorder.finish()
        finally:
            recorder.close()

    def test_worker_receipt_failure_precedes_cuda_and_native_imports(self):
        from reelcache.adapters.magi_native import worker

        with patch("reelcache.adapters.magi_rank_worker.verify_worker_inputs",
                   side_effect=IntegrityError("not independently reviewed")), patch.object(
            torch.cuda, "set_device", side_effect=AssertionError("CUDA must not be touched"),
        ):
            with self.assertRaisesRegex(IntegrityError, "independently reviewed"):
                worker(SimpleNamespace(protocol="none", assets="none", launch_receipt="none",
                                       requests="none", gpu_grant=True))
        self.assertFalse(torch.cuda.is_initialized())

    def test_native_loop_mock_generates_first_chunk_instead_of_loading_a_prefix(self):
        from reelcache.adapters.magi_native import _StudyBackend

        calls = []

        class Adapter:
            source_context = {}

            def execution(self, **kwargs):
                calls.append(("boundary", kwargs))
                return nullcontext()

            def seal_chunk(self, chunk):
                calls.append(("seal", chunk))

            def snapshot(self, **kwargs):
                calls.append(("snapshot", kwargs["metadata"]["execution_config"]["near_clean_steps"]))
                return "snapshot:test"

            def memory_report(self):
                return {"mock": True}

        backend = _StudyBackend.__new__(_StudyBackend)
        backend.torch = torch
        backend.device = torch.device("cpu")
        backend.transport = DistributedOwner(expected_world_size=1)
        backend.cell = SimpleNamespace(tokens=4, crop_top=0, height=4)
        backend.config = SimpleNamespace(runtime_config=SimpleNamespace(noise2clean_kvrange=[2] * 4,
                                                                      clean_chunk_kvrange=2, num_frames=120))
        backend.verified = {"config": {"fake_native": True}, "batch": {}}
        backend.schedule, backend.intervals = serial_timetable()

        def denoise(x, *args, **kwargs):
            calls.append(("step", kwargs["slice_point"], kwargs["distill_nearly_clean_chunk"],
                          args[0][0, 0].item(), kwargs["fwd_extra_1st_chunk"]))
            return torch.zeros_like(x)

        def clean(latent, *args, **kwargs):
            calls.append(("clean", kwargs["slice_point"], args[0][0, 0].item(),
                          kwargs["fwd_extra_1st_chunk"]))
            return torch.zeros_like(latent)

        backend.model = SimpleNamespace(
            forward_dispatcher=denoise, forward=clean,
        )
        value = replace(request(), chunks=3)
        tiny = SimpleNamespace(
            chunks=3, latent_shape=(1, 1, 1, 2, 2), request_hash=value.request_hash,
            native_kwargs=value.native_kwargs,
        )
        state = {
            "request": tiny, "adapter": Adapter(), "namespace": "native-loop-fake",
            "generator": torch.Generator().manual_seed(13), "noise_hashes": [],
            "y": torch.ones(2, 1, 2, 2), "masks": torch.ones(2, 1, 2),
            "null_caption": torch.zeros(1, 2, 2), "null_mask": torch.ones(1, 2),
            "cache": SimpleNamespace(update_kv_cache=True), "latents": [],
            "source_context": {"conditioning_sha256": "fake-conditioning", **serial_sampler_contract()},
            "directory": self.root,
        }
        result = backend.generate(state)
        generator = torch.Generator().manual_seed(13)
        for record in result["latents"]:
            expected = torch.randn(tiny.latent_shape, generator=generator)
            actual = torch.load(record["artifact"]["path"], weights_only=True)["latent"]
            self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(len(result["latents"]), tiny.chunks)
        self.assertEqual(backend.config.runtime_config.num_frames, 120)
        self.assertEqual(len([row for row in calls if row[0] == "step"]), tiny.chunks * 16)
        self.assertEqual([row[1] for row in calls if row[0] == "seal"], list(map(str, range(tiny.chunks))))
        self.assertEqual(len([row for row in calls if row[0] == "step" and row[2]]), tiny.chunks * 4)
        forwards = [row for row in calls if row[0] in ("step", "clean")]
        self.assertEqual([(row[0], row[1]) for row in forwards],
                         [(kind, chunk) for chunk in range(tiny.chunks) for kind in ["step"] * 16 + ["clean"]])
        for chunk in range(tiny.chunks):
            steps = [row for row in forwards if row[0] == "step" and row[1] == chunk]
            self.assertEqual([row[3] for row in steps], backend.schedule[:-1].tolist())
            self.assertFalse(any(row[4] for row in steps))
        self.assertTrue(all(abs(row[2] - 0.9999) < 1e-7 and not row[3]
                            for row in forwards if row[0] == "clean"))
        self.assertFalse(result["native_parity_verified"])
        self.assertEqual(result["request"]["sampler_mode"], "serial_clean_history")
        self.assertEqual(result["request"]["sampler_window_size"], 1)
        self.assertTrue(result["request"]["standalone_clean_recapture"])
        self.assertFalse(result["request"]["native_window4_schedule_parity_claimed"])
        self.assertEqual(result["request"]["horizon_source"], "request.chunks")

    def test_observer_persists_oracle_schema_distinct_hashes_and_explicit_retention(self):
        from reelcache.adapters.magi_native import _StudyBackend
        from reelcache.study import _census_proof

        geometry = MagiGeometry(layers=1, kv_heads=2, head_dim=4, frames=1, height=2, width=3, world_size=1)
        model = ModelIdentity("synthetic_magi", "CPU_fixture", 1, 8, "global_KV_channels", "bfloat16")
        fixture_cell = replace(study_cell("magi720"), frames=1, token_height=2, token_width=3,
                               layers=1, channels=8, world_size=1)
        cells = patch("reelcache.rank_analysis.CELLS", {"magi720": fixture_cell})
        cells.start()
        self.addCleanup(cells.stop)
        backend = _StudyBackend.__new__(_StudyBackend)
        backend.torch = torch
        backend.observation_evidence_kind = "synthetic_fixture"
        backend.verified = {"batch": {
            "retain_originals": ["fixture:0:0:K"], "spectrum_tile_rows": 2,
            "spectrum_workspace_bytes": 1 << 28,
        }, "receipt": {"fixture": True, "source_closure": {"closure_id": "d" * 64}}}
        backend.active = {
            "request": SimpleNamespace(request_id="fixture", case_id="paired-fixture"),
            "directory": self.root, "trajectory_id": "fixture-trajectory",
        }
        for target, dtype in (("K", torch.float32), ("V", torch.bfloat16)):
            backend.active["request"].policy = {"solver": "qr_q2"} if target == "K" else None
            matrix = torch.eye(6, 8, dtype=dtype)
            context = _tensor_context(
                model, MatrixKey(0, target), 0, geometry, "pre_rope" if target == "K" else "none", dtype,
            )
            description = {
                "schema": "reelcache.magi_original_clean.v1", "tensor_context": context.to_dict(),
                "chunk_id": "0", "shape": [6, 8], "actual_dtype": context.dtype,
                "boundary": 0, "layer": 0, "target": target,
                "source_sha256": _tensor_hash(matrix), "synthetic_cpu_fixture": True,
                "fixture": True, "native_execution": False,
                "source_hash_format": NATIVE_TENSOR_HASH_FORMAT,
                "source_hash_descriptor": _tensor_hash_format(dtype),
                "source_hash_byte_order": _tensor_hash_format(dtype)["byte_order"],
                "row_shard_hash_format": NATIVE_TENSOR_HASH_FORMAT,
                "row_shard_hash_byte_order": _tensor_hash_format(dtype)["byte_order"],
                "cell_id": "magi720", "group_id": "fixture-group", "split": "calibration", "seed": 0,
                "case_id": "paired-fixture", "trajectory_id": "fixture-trajectory",
                "global_channels": 8, "token_positions": [0, 6], "inverse_rope_source": False,
                "source_layout": "owner_assembled_original_token_rows", "request_id": "fixture",
                "history": [], "history_head": None, "clean_timestep": .9999,
                "branch_id": "fixture-branch", "arm_id": "fixture-arm", "protocol_id": "a" * 64,
                "launch_closure_id": backend.verified["receipt"]["source_closure"]["closure_id"],
                "rotary_sha256": ["b" * 64, "c" * 64],
                "row_shards": [{
                    "rank": 0, "row_start": 0, "row_end": 6, "channels": 8,
                    "dtype": context.dtype, "sha256": _tensor_hash(matrix),
                }],
            }
            before_rng, before_python_rng = torch.get_rng_state().clone(), random.getstate()
            for name, invalid in (("source_sha256", "0" * 64), ("shape", [3, 16]),
                                  ("source_hash_format", {"algorithm": "sha256", "header": "unknown"}),
                                  ("case_id", "incorrect-post-hoc-case"), ("fixture", False),
                                  ("fixture", 1), ("native_execution", True), ("native_execution", 0),
                                  ("seed", False), ("boundary", False), ("layer", False),
                                  ("launch_closure_id", "e" * 64)):
                with self.assertRaises(IntegrityError):
                    backend._observe({**description, name: invalid}, matrix)
            missing_closure = {name: value for name, value in description.items() if name != "launch_closure_id"}
            with self.assertRaisesRegex(IntegrityError, "admitted launch closure"):
                backend._observe(missing_closure, matrix)
            if target == "K":
                bad = copy.deepcopy(description)
                bad["tensor_context"]["key_space"] = "post_rope"
                with self.assertRaises(IntegrityError):
                    backend._observe(bad, matrix)
                for unadmitted in (None, {}, {"source_closure": {}},
                                   {"source_closure": {"closure_id": True}},
                                   {"source_closure": {"closure_id": "unknown"}}):
                    with (
                        patch.object(backend, "observation_evidence_kind", "native"),
                        patch.dict(backend.verified, {"receipt": unadmitted}),
                        self.assertRaisesRegex(IntegrityError, "admitted launch closure"),
                    ):
                        backend._observe({**description, "fixture": False, "native_execution": True}, matrix)
                self.assertFalse((self.root / "spectra").exists())
            self.assertFalse((self.root / "spectra" / "chunk0" / f"layer000_{target}.spectrum.json").exists())
            backend._observe(description, matrix)
            self.assertTrue(torch.equal(before_rng, torch.get_rng_state()))
            self.assertEqual(before_python_rng, random.getstate())
            path = self.root / "spectra" / "chunk0" / f"layer000_{target}.json"
            saved = json.loads(path.read_text())
            self.assertEqual(saved["schema"], "reelcache.magi_clean_observation.v2")
            self.assertEqual(saved["capture"], description)
            self.assertFalse(saved["included_in_online_selector_cost"])
            self.assertTrue(saved["included_in_full_request_wall_time"])
            self.assertEqual(saved["standalone_offline_decompositions"], 1)
            self.assertIsInstance(saved["oracle_work_counts"], dict)
            self.assertIsInstance(saved["oracle_solver_work_counts"], dict)
            self.assertIn("hash_update_seconds", saved["oracle_timings"])
            self.assertIn("gram_matmul_add_seconds", saved["oracle_timings"])
            self.assertIn("nested", saved["oracle_timing_nesting"])
            self.assertFalse(saved["shares_policy_decomposition"])
            self.assertFalse(saved["exact_policy_decomposition_is_additional"])
            self.assertNotIn("spectrum", saved)
            spectrum_artifact = saved["spectrum_artifact"]
            self.assertEqual(file_sha256(spectrum_artifact["path"]), spectrum_artifact["sha256"])
            spectrum = json.loads(Path(spectrum_artifact["path"]).read_text())
            source = spectrum["source"]
            self.assertEqual(source["shape"], [6, 8])
            self.assertEqual(source["dtype"], context.dtype)
            for name, value in description.items():
                self.assertEqual(source["provenance"][name], value)
            self.assertIs(source["provenance"]["fixture"], True)
            self.assertIs(source["provenance"]["native_execution"], False)
            for name, expected in {
                "protocol_id": "a" * 64, "cell_id": "magi720", "group_id": "fixture-group",
                "seed": 0, "boundary": 0, "layer": 0, "target": target, "split": "calibration",
                "launch_closure_id": "d" * 64,
            }.items():
                self.assertIs(type(source["provenance"][name]), type(expected))
                self.assertEqual(source["provenance"][name], expected)
            for name, value in saved["observation_identity"].items():
                self.assertEqual(source["provenance"][name], value)
            with self.assertRaisesRegex(IntegrityError, "does not substantiate its census position"):
                _census_proof(
                    {"launch_closure_id": "d" * 64, "observations": [{
                        "group_id": "fixture-group", "seed": 0, "boundary": 0, "layer": 0,
                        "target": target, "cell_id": "magi720", "spectrum": spectrum_artifact,
                    }]},
                    {"protocol_id": "a" * 64, "prompts": [
                        {"group_id": "fixture-group", "split": "calibration"},
                    ], "seeds": [0], "clean_boundaries": [0]},
                    fixture_cell,
                )
            numeric_artifact = saved["rank_observation_artifact"]
            self.assertEqual(file_sha256(numeric_artifact["path"]), numeric_artifact["sha256"])
            numeric = json.loads(Path(numeric_artifact["path"]).read_text())
            self.assertEqual(numeric["observation"]["case_id"], "paired-fixture")
            self.assertEqual(numeric["observation"]["trajectory_id"], "fixture-trajectory")
            self.assertEqual(numeric["observation"]["evidence_kind"], "synthetic_fixture")
            self.assertEqual(numeric["observation"]["spectrum_sha256"], spectrum_artifact["sha256"])
            self.assertNotEqual(source["tensor_sha256"], description["source_sha256"])
            self.assertEqual(_tensor_hash(matrix), description["source_sha256"])
            if target == "K":
                self.assertEqual(source["retention"], "exact-input-retained")
                original = saved["original_artifact"]
                self.assertEqual(file_sha256(original["path"]), original["sha256"])
                self.assertTrue(torch.equal(torch.load(original["path"], weights_only=True)["original"], matrix))
            else:
                self.assertEqual(source["retention"], "spectrum-only")
                self.assertIsNone(saved["original_artifact"])
                self.assertFalse(path.with_suffix(".pt").exists())


class PacketRuntime(fixtures.FakeRuntime):
    def seal_chunk(self, *args, **kwargs):
        result = super().seal_chunk(*args, **kwargs)
        stored = self.branches[args[0]][-1]
        stored["values"] = {key: (value / 2).round() * 2 for key, value in stored["values"].items()}
        return result


class QualityHotAndObserverTests(unittest.TestCase):
    setUp = fixtures.MagiAdapterTests.setUp
    tearDown = fixtures.MagiAdapterTests.tearDown
    _adapter = fixtures.MagiAdapterTests._adapter
    _forward = fixtures.MagiAdapterTests._forward
    _state = fixtures.MagiAdapterTests._state

    def start(self, *, observer=None, specs=CodecSpec(kind="svd", rank=2), **options):
        self.runtime = PacketRuntime(self.identity)
        self.adapter = self._adapter(
            specs=specs, cache_mode="quality_hot", quality_hot_bytes=4096,
            source_observer=observer, source_context={"case_id": "case-A", "seed": 3},
            resource_budget=budget(), **options,
        )
        self.adapter.attach()
        self._forward(0, clean=True)
        self.adapter.seal_chunk()

    def test_original_observer_sees_full_preK_not_inverse_rounded_postK(self):
        observed = []
        self.start(observer=lambda context, tensor: observed.append((context, tensor.clone())),
                   specs=CodecSpec())
        self.assertEqual(len(observed), 4)
        for context, value in observed:
            self.assertEqual(value.shape, (6, 8))
            self.assertEqual(context["shape"], [6, 8])
            self.assertEqual(context["source_layout"], "owner_assembled_original_token_rows")
            self.assertEqual(context["global_channels"], 8)
            self.assertEqual(context["boundary"], 0)
            self.assertEqual(context["layer"], context["tensor_context"]["key"]["layer"])
            self.assertEqual(context["target"], context["tensor_context"]["key"]["target"])
            self.assertIs(type(context["boundary"]), int)
            self.assertIs(type(context["layer"]), int)
            self.assertEqual(context["case_id"], "case-A")
            self.assertFalse(context["inverse_rope_source"])
            self.assertEqual(context["source_sha256"], _tensor_hash(value))
            self.assertEqual(context["source_hash_format"], NATIVE_TENSOR_HASH_FORMAT)
            self.assertEqual(context["source_hash_descriptor"]["algorithm"], "sha256")
            self.assertEqual(context["source_hash_descriptor"]["header"], "none")
            self.assertEqual(context["source_hash_descriptor"]["dtype"], context["actual_dtype"])
            self.assertEqual(context["row_shards"][0]["hash_format"], context["source_hash_descriptor"])
            self.assertTrue(all(row["hash_format"]["header"] == "none" for row in context["rotary_sources"]))
            if context["tensor_context"]["key"]["target"] == "K":
                self.assertEqual(value.dtype, torch.float32)
                self.assertEqual(context["tensor_context"]["key_space"], "pre_rope")
                expected = torch.arange(48).float().reshape(6, 8) / 17
                self.assertTrue(torch.equal(value, expected))
            else:
                self.assertEqual(value.dtype, torch.bfloat16)
        self.assertEqual(len(self.adapter.observed_source_hashes["0"]), 4)

    def test_observer_mutation_cannot_change_committed_history(self):
        self.adapter = self._adapter(
            source_observer=lambda context, tensor: tensor.zero_(), resource_budget=budget(),
        )
        self.adapter.attach()
        self._forward(0, clean=True)
        with self.assertRaisesRegex(PeerFailure, "observer mutated"):
            self.adapter.seal_chunk()
        self.assertEqual(self.adapter.cursor, 0)
        self.assertEqual(self.runtime.branches, {})

    def test_missing_original_row_metadata_cannot_publish_a_spectrum_or_packet(self):
        observed = []
        self.adapter = self._adapter(
            source_observer=lambda context, tensor: observed.append(context), resource_budget=budget(),
        )
        self.adapter.attach()
        self._forward(0, clean=True)
        checkpoint = self.adapter.transport.checkpoint

        def missing_row(operation, **kwargs):
            result = checkpoint(operation, **kwargs)
            return [] if operation.startswith("magi_original_shards:") else result

        with patch.object(self.adapter.transport, "checkpoint", side_effect=missing_row):
            with self.assertRaisesRegex(PeerFailure, "row-shard coverage is incomplete"):
                self.adapter.seal_chunk()
        self.assertEqual(observed, [])
        self.assertEqual(self.adapter.cursor, 0)
        self.assertEqual(self.runtime.branches, {})

    def test_actual_packet_replaces_newest_clean_scratch_before_cursor_advances(self):
        self.start()
        self.assertEqual(self.adapter.cursor, 1)
        self.assertEqual(self.adapter._quality_stats["clean_slot_replacements"], 4)
        for layer in range(2):
            buffer = self.adapter._quality_buffers[layer]
            chunk = self.adapter._chunks[0]
            for target, offset in (("K", 0), ("V", 4)):
                with self.adapter._decoded_history(MatrixKey(layer, target), chunk) as decoded:
                    self.assertTrue(torch.equal(buffer[:6, :, offset:offset + 4], decoded))
        self.assertEqual(self.runtime.pins, 1)
        self.assertEqual(self.adapter._quality_stats["packet_decodes_per_recipient"], 4)

    def test_quiescent_quality_snapshot_keeps_only_packet_backed_native_views(self):
        self.start()
        pointers = [value.data_ptr() for value in self.adapter._quality_buffers.values()]
        snapshot = self.adapter.snapshot(**self._state())
        self.assertEqual(pointers, [value.data_ptr() for value in self.adapter._quality_buffers.values()])
        self.assertEqual(self.runtime.snapshots[snapshot]["metadata"]["residency_mode"], "quality_hot")
        self.assertEqual(self.runtime.pins, 1)

    def test_archived_hashes_survive_all_steps_duplicate_12_15_and_clean_recapture(self):
        self.start()
        protected = {layer: _tensor_hash(value[:6]) for layer, value in self.adapter._quality_buffers.items()}
        initial_decodes = self.adapter._quality_stats["packet_decodes_per_recipient"]
        with self.adapter.execution(chunk_index=1, clean=False):
            for step in range(16):
                outputs, meta = self._forward(1, clean=False, near=step >= 12, guarded=False)
                for layer, value in self.adapter._quality_buffers.items():
                    self.assertEqual(value.shape[0], 12)
                    self.assertEqual(_tensor_hash(value[:6]), protected[layer])
                    self.assertEqual(outputs[layer][0].shape[0], 12 + (6 if step >= 12 else 0))
        self._forward(1, clean=True)
        for layer, value in self.adapter._quality_buffers.items():
            self.assertEqual(_tensor_hash(value[:6]), protected[layer])
        self.assertEqual(self.adapter._quality_stats["packet_decodes_per_recipient"], initial_decodes)
        self.adapter.seal_chunk()
        self.assertEqual(self.adapter.cursor, 2)
        self.assertEqual(self.adapter._quality_stats["packet_decodes_per_recipient"], initial_decodes + 4)
        self.assertEqual(self.adapter._quality_stats["clean_slot_replacements"], 8)
        self.assertIn("blocked", self.adapter.memory_report()["quality_hot"]["native_parity_gate"])

    def test_archived_corruption_aborts_without_reusing_dense_buffers(self):
        self.start()
        with self.assertRaisesRegex(PeerFailure, "protected archived"):
            with self.adapter.execution(chunk_index=1, clean=False):
                self.adapter._quality_buffers[0][0].add_(1)
                self._forward(1, clean=False, guarded=False)
        self.assertEqual(self.adapter.phase, "failed")
        self.assertFalse(self.adapter._quality_buffers)
        self.assertFalse(self.cache.key_value_memory_dict)
        self.assertEqual(self.runtime.pins, 0)

    def test_fork_invalidates_native_shards_and_preserves_parent(self):
        self.start()
        self.adapter.release_quality_history()
        snapshot = self.adapter.snapshot(**self._state())
        self._forward(1, clean=False)
        old_buffers = tuple(self.adapter._quality_buffers.values())
        self.adapter.fork(snapshot, "child")
        self.assertFalse(self.adapter._quality_buffers)
        self.assertFalse(self.cache.key_value_memory_dict)
        self.assertEqual(self.runtime.pins, 0)
        self._forward(1, clean=True)
        self.adapter.seal_chunk()
        self.assertEqual(len(self.runtime.branches["parent"]), 1)
        self.assertEqual(len(self.runtime.branches["child"]), 2)
        self.assertTrue(all(value is not old for value in self.adapter._quality_buffers.values() for old in old_buffers))

    def test_complete_budget_rejects_before_capture_or_native_allocation(self):
        self.adapter = self._adapter(
            source_observer=lambda context, tensor: None, resource_budget=budget(1),
        )
        self.adapter.attach()
        with patch.object(torch, "empty", side_effect=AssertionError("allocation before admission")):
            with self.assertRaisesRegex(PeerFailure, "admitted host_bytes"):
                with self.adapter.execution(chunk_index=0, clean=True, clean_timestep=0.9999):
                    self.fail("insufficient source budget reached a forward")
        self.assertIsNone(self.adapter._pending)
        self.assertEqual(self.adapter.cursor, 0)
        self.assertFalse(self.cache.key_value_memory_dict)

    def test_static_profile_mismatch_still_precedes_branch_native_rng_mutation(self):
        self.start()
        self.adapter.release_quality_history()
        snapshot = self.adapter.snapshot(**self._state())
        self.adapter.detach()
        self.adapter = self._adapter(specs=CodecSpec())
        self.adapter.attach()
        before = torch.get_rng_state().clone()
        with self.assertRaisesRegex(PeerFailure, "codec profile"):
            self.adapter.fork(snapshot, "incompatible")
        self.assertNotIn("incompatible", self.runtime.branches)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertFalse(self.cache.key_value_memory_dict)


class AdaptiveMagiTests(unittest.TestCase):
    setUp = fixtures.MagiAdapterTests.setUp
    _adapter = fixtures.MagiAdapterTests._adapter
    _forward = fixtures.MagiAdapterTests._forward
    _state = fixtures.MagiAdapterTests._state

    def tearDown(self):
        self.adapter.detach()
        if hasattr(self.runtime, "close"):
            self.runtime.close()
        shutil.rmtree(self.root)

    def policy(self):
        from reelcache.online_rank import RankPolicy

        keys = [MatrixKey(layer, target) for layer in range(2) for target in ("K", "V")]
        return RankPolicy(
            policy_id="fake-magi-staged", model=self.identity,
            geometry=(self.geometry.frames, self.geometry.height, self.geometry.width),
            source_dtypes={"K": "float32", "V": "bfloat16"},
            source_fingerprint="1" * 64, geometry_fingerprint="2" * 64,
            history_fingerprint="3" * 64, calibration_fingerprint="4" * 64,
            implementation_fingerprint="5" * 64, energy_target=0.95,
            template_ranks={key: 2 for key in keys}, allowed_ranks={key: (1, 2, 4, 6) for key in keys},
            codec=CodecSpec(kind="svd", rank=1), anchors=(MatrixKey(1, "K"),),
            anchor_mode="staged", seed=4, solver="qr_q2", oversample=0,
            source_limit_bytes=1 << 20, workspace_limit_bytes=1 << 24,
            solver_workspace_bytes=1 << 20, staging_limit_bytes=1 << 20,
        )

    def initialize(self):
        from reelcache.manager import KVRuntime

        self.runtime = KVRuntime(self.root / "adaptive-store", self.identity, max_resident_bytes=1 << 24)
        self.bound_policy = self.policy()
        self.adapter = self._adapter(rank_policy=self.bound_policy, resource_budget=budget(),
                                     cache_mode="quality_hot", quality_hot_bytes=4096)
        self.adapter.attach()

    def initialize_static_layer_template(self, *, cache_mode="runtime"):
        from reelcache.manager import KVRuntime

        value = replace(request(), arm="svd", layer_ranks={
            "K": [1 + layer % 2 for layer in range(48)],
            "V": [3 + layer % 2 for layer in range(48)],
        })
        specs = {key: spec for key, spec in value.static_specs().items() if key.layer < self.geometry.layers}
        self.runtime = KVRuntime(self.root / "static-layer-store", self.identity, max_resident_bytes=1 << 24)
        self.adapter = self._adapter(
            specs=specs, resource_budget=budget(), cache_mode=cache_mode,
            quality_hot_bytes=4096 if cache_mode == "quality_hot" else 0,
        )
        self.adapter.attach()
        return value, specs

    @contextmanager
    def reject_native_state_reads(self, snapshot):
        state_blob = self.runtime.store.get_manifest(snapshot)["tensor_state"]
        get = self.runtime.store._get

        def checked(object_id):
            self.assertNotEqual(object_id, state_blob, "metadata rejection read native state")
            return get(object_id)

        with (
            patch.object(self.runtime.store, "_get", side_effect=checked),
            patch.object(self.runtime.store, "verify",
                         side_effect=AssertionError("full payload verification preceded metadata rejection")) as verify,
            patch("reelcache.manager.deserialize_tensors",
                  side_effect=AssertionError("native state deserialized before metadata rejection")) as deserialize,
            patch.object(self.adapter, "_validate_restored_tensors",
                         side_effect=AssertionError("full native validator ran before metadata rejection")) as full,
        ):
            yield
        verify.assert_not_called()
        deserialize.assert_not_called()
        full.assert_not_called()

    def test_static_layer_template_commits_fixed_ranks_without_adaptive_state(self):
        _, specs = self.initialize_static_layer_template()
        for index in range(2):
            self._forward(index, clean=True)
            committed = self.adapter.seal_chunk()
            chunk = self.runtime.store.get_manifest(committed)
            self.assertEqual(chunk["version"], 1)
            self.assertNotIn("rank_policy", chunk)
            for matrix in chunk["matrices"]:
                key = MatrixKey.from_dict(matrix["key"])
                packet = self.runtime.store.get_manifest(matrix["tensor"])
                self.assertEqual(CodecSpec.from_dict(packet["spec"]), specs[key])
            self.assertIsNone(self.runtime.history_rank_policy("parent"))
            self.assertIsNone(self.runtime.history_rank_state("parent"))
            self.assertEqual(self.adapter.specs, specs)

    def test_static_layer_restore_rejects_forged_matching_capsule_before_branch_or_rng_mutation(self):
        _, specs = self.initialize_static_layer_template(cache_mode="quality_hot")
        self._forward(0, clean=True)
        self.adapter.seal_chunk()
        saved = self.runtime.load_snapshot(self.adapter.snapshot(**self._state()))
        wrong = dict(specs)
        wrong[MatrixKey(0, "K")] = replace(wrong[MatrixKey(0, "K")], rank=2)
        metadata = copy.deepcopy(saved.metadata)
        metadata["specs"] = {f"{key.layer}:{key.target}": spec.to_dict() for key, spec in wrong.items()}
        forged = self.runtime.snapshot("parent", metadata=metadata, tensors=saved.tensors)
        self.adapter.specs = wrong
        rng, cursor = torch.get_rng_state().clone(), self.adapter._cursor
        buffers = {key: (value.data_ptr(), _tensor_hash(value))
                   for key, value in self.adapter._quality_buffers.items()}
        head = self.runtime.branch_info("parent")
        leases = self.runtime.store.stats()["leases"]
        with self.reject_native_state_reads(forged), self.assertRaisesRegex(PeerFailure, "packet spec"):
            self.adapter.fork(forged, "forged-layer-template")
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual((self.adapter.branch_id, self.adapter._cursor), ("parent", cursor))
        self.assertNotIn("branches/forged-layer-template", self.runtime.store.list_refs("branches/"))
        self.assertEqual(self.runtime.branch_info("parent"), head)
        self.assertEqual(self.runtime.store.stats()["leases"], leases)
        self.assertEqual(buffers, {key: (value.data_ptr(), _tensor_hash(value))
                                   for key, value in self.adapter._quality_buffers.items()})

    def test_static_layer_template_snapshot_restores_same_profile_and_rejects_wrong_vector(self):
        value, specs = self.initialize_static_layer_template(cache_mode="quality_hot")
        with patch("reelcache.online_rank.RankPreparation", side_effect=AssertionError("static is not adaptive")):
            self._forward(0, clean=True)
            self.adapter.seal_chunk()
            state = self._state()
            state["metadata"]["study_request"] = {
                "case_id": value.case_id, "rank_selection_mode": value.rank_selection_mode,
                "layer_ranks": {target: ranks[:2] for target, ranks in value.layer_rank_map.items()},
            }
            snapshot = self.adapter.snapshot(**state)
            saved = self.runtime.load_snapshot(snapshot)
            self.assertIsNone(saved.rank_policy)
            self.assertIsNone(saved.rank_state)
            self.assertIsNone(saved.metadata["rank_policy"])
            self.assertEqual(saved.metadata["study_request"], state["metadata"]["study_request"])
            self.assertEqual(saved.metadata["specs"], {
                f"{key.layer}:{key.target}": spec.to_dict() for key, spec in specs.items()
            })
            wrong = dict(specs)
            wrong[MatrixKey(1, "V")] = replace(wrong[MatrixKey(1, "V")], rank=3)
            wrong_codec = dict(specs)
            wrong_codec[MatrixKey(0, "V")] = replace(wrong_codec[MatrixKey(0, "V")], normalization="global")
            rng = torch.get_rng_state().clone()
            head = self.runtime.branch_info("parent")
            buffers = {key: (value.data_ptr(), _tensor_hash(value))
                       for key, value in self.adapter._quality_buffers.items()}
            for name, profile in (("wrong-layer-vector", wrong), ("wrong-codec", wrong_codec)):
                self.adapter.specs = profile
                with self.reject_native_state_reads(snapshot), self.assertRaisesRegex(PeerFailure, "codec profile"):
                    self.adapter.restore(snapshot, name)
                self.assertTrue(torch.equal(rng, torch.get_rng_state()))
                self.assertNotIn(f"branches/{name}", self.runtime.store.list_refs("branches/"))
                self.assertEqual(self.runtime.branch_info("parent"), head)
                self.assertEqual(buffers, {key: (value.data_ptr(), _tensor_hash(value))
                                           for key, value in self.adapter._quality_buffers.items()})
            self.adapter.specs = specs
            events = []
            metadata_check = self.adapter._validate_restored_profile
            tensor_check = self.adapter._validate_restored_tensors

            def preview(state, record, requested):
                self.assertEqual(dict(state.tensors), {})
                events.append("metadata")
                return metadata_check(state, record, requested)

            def full(state, metadata):
                self.assertTrue(state.tensors)
                events.append("full")
                return tensor_check(state, metadata)

            with patch.object(self.adapter, "_validate_restored_profile", side_effect=preview), patch.object(
                self.adapter, "_validate_restored_tensors", side_effect=full,
            ):
                self.adapter.fork(snapshot, "fixed-layer-child")
            self.assertEqual(events, ["metadata", "full"])
            self.assertEqual(self.adapter.branch_id, "fixed-layer-child")
            self.assertEqual(self.adapter.specs, specs)
            self._forward(1, clean=True)
            committed = self.adapter.seal_chunk()
            for matrix in self.runtime.store.get_manifest(committed)["matrices"]:
                key = MatrixKey.from_dict(matrix["key"])
                packet = self.runtime.store.get_manifest(matrix["tensor"])
                self.assertEqual(CodecSpec.from_dict(packet["spec"]), specs[key])
            self.assertIsNone(self.runtime.history_rank_policy("fixed-layer-child"))
            self.assertIsNone(self.runtime.history_rank_state("fixed-layer-child"))

    def test_raw_census_acquires_shared_features_before_release_with_unresolved_nulls(self):
        from reelcache.adapters.magi_native import _StudyBackend
        from reelcache.factorization import DetectorSpec, DirectionFit, acquire_anchor_directions
        from reelcache.manager import KVRuntime
        from reelcache.rank_analysis import validate_detector_records
        from reelcache.spectra import StreamingGramAccumulator

        cell = replace(study_cell("magi720"), frames=1, token_height=2, token_width=3,
                       layers=2, channels=8, world_size=1)
        backend = _StudyBackend.__new__(_StudyBackend)
        backend.torch = torch
        backend.observation_evidence_kind = "synthetic_fixture"
        spec = DetectorSpec(cpu_threads=1)
        backend.verified = {
            "batch": {"retain_originals": [], "spectrum_tile_rows": 2, "cpu_threads": 1,
                      "spectrum_workspace_bytes": 1 << 28, "detector_workspace_bytes": 1 << 26,
                      "detector_solver_workspace_bytes": 1 << 20},
            "receipt": {"phase": "calibration_census", "source_closure": {"closure_id": "b" * 64}},
            "detector_spec": spec,
        }
        value = request()
        backend.active = {"request": value, "directory": self.root, "trajectory_id": "raw-feature-fixture"}
        context = {
            "case_id": value.case_id, "trajectory_id": "raw-feature-fixture", "cell_id": value.cell_id,
            "group_id": value.group_id, "seed": value.seed, "split": "calibration",
            "request_id": value.request_id, "arm_id": "raw", "protocol_id": "a" * 64,
            "launch_closure_id": "b" * 64, "fixture": True, "native_execution": False,
        }
        self.runtime = KVRuntime(self.root / "raw-feature-store", self.identity, max_resident_bytes=1 << 24)
        self.adapter = self._adapter(source_context=context, source_observer=backend._observe,
                                     resource_budget=budget())
        self.adapter.attach()
        acquisitions = []
        released_grams, directions = [], []
        closed_accumulators = set()
        required_rank = DirectionFit.required_rank
        close_gram = StreamingGramAccumulator.close
        persist = backend._persist_observation

        def close(accumulator):
            close_gram(accumulator)
            if accumulator not in closed_accumulators:
                closed_accumulators.add(accumulator)
                released_grams.append(accumulator._closed and accumulator._gram is None
                                      and accumulator._channel_sum is None)

        def acquire(matrix, observed_context, **kwargs):
            self.assertEqual(len(released_grams), len(acquisitions) + 1)
            self.assertTrue(released_grams[-1])
            self.assertEqual(matrix.dtype, torch.float32 if observed_context.key.target == "K" else torch.bfloat16)
            self.assertEqual(observed_context.key_space, "pre_rope" if observed_context.key.target == "K" else "none")
            self.assertEqual(tuple(matrix.shape), (6, 8))
            result = acquire_anchor_directions(matrix, observed_context, **kwargs)
            acquisitions.append((observed_context.key, result.source_sha256))
            directions.append(weakref.ref(result.directions))
            return result

        def publish(*args, **kwargs):
            self.assertTrue(all(reference() is None for reference in directions))
            return persist(*args, **kwargs)

        with patch("reelcache.rank_analysis.CELLS", {"magi720": cell}), patch(
            "reelcache.factorization.acquire_anchor_directions", new=acquire,
        ), patch.object(
            StreamingGramAccumulator, "close", new=close,
        ), patch.object(
            backend, "_persist_observation", side_effect=publish,
        ), patch.object(
            DirectionFit, "required_rank",
            new=lambda fit, tau: None if tau == .99 else required_rank(fit, tau),
        ):
            before = torch.get_rng_state().clone()
            self._forward(0, clean=True)
            self.adapter.seal_chunk()
            self.assertTrue(torch.equal(before, torch.get_rng_state()))
            self.assertEqual(len(acquisitions), 4)
            all_observations, all_features = [], []
            for key, source_hash in acquisitions:
                stem = self.root / "spectra" / "chunk0" / f"layer{key.layer:03d}_{key.target}"
                wrapper = json.loads(stem.with_suffix(".json").read_text())
                artifact = wrapper["detector_artifact"]
                self.assertEqual(file_sha256(artifact["path"]), artifact["sha256"])
                document = json.loads(Path(artifact["path"]).read_text())
                observation = json.loads(stem.with_suffix(".rank.json").read_text())["observation"]
                self.assertEqual(document["schema"], "reelcache_rank_detector_observations_v1")
                self.assertEqual(document["detector_sha256"], spec.fingerprint)
                self.assertEqual(len(document["observations"]), 2)
                work = wrapper["detector_work_record"]
                self.assertEqual(work["acquisition_count"], 1)
                self.assertEqual(work["thresholds"], [.95, .99])
                self.assertTrue(work["count_once_for_both_threshold_rows"])
                self.assertEqual(work["metrics"]["source_copy_count"], 1)
                for feature in document["observations"]:
                    self.assertEqual(feature["acquisition_sha256"], work["acquisition_sha256"])
                    self.assertEqual(feature["acquisition"]["source_sha256"], source_hash)
                    self.assertEqual(feature["maximum_width"], 6)
                    self.assertEqual(feature["available_after_layer"], key.layer)
                    self.assertEqual(feature["rank_statistic"], "approximate_spectral_crossing")
                    self.assertEqual(feature["acquisition"]["metrics"]["source_copy_count"], 1)
                    if feature["tau"] == .99:
                        self.assertIsNone(feature["estimated_rank"])
                        self.assertEqual(feature["status"], "unresolved")
                        self.assertGreater(observation["r99"]["lower"], 0)
                self.assertTrue(wrapper["detector_cost_included_in_full_request_wall"])
                self.assertFalse(wrapper["detector_cost_included_in_online_RAW_compression"])
                all_observations.append(observation)
                all_features.extend(document["observations"])
            self.assertEqual(len(validate_detector_records(
                all_observations, all_features, detector_spec=spec,
            )), 8)

    def test_only_core_advances_policy_state_with_atomic_clean_packet_publication(self):
        self.initialize()
        self._forward(0, clean=True)
        self.adapter.seal_chunk()
        before = self.runtime.history_rank_state("parent")
        self.assertEqual(before.completed_chunks, 1)
        self.assertEqual(self.adapter.specs, {})
        self.assertEqual(self.runtime.history_rank_policy("parent").fingerprint, self.bound_policy.fingerprint)
        snapshot = self.adapter.snapshot(**self._state())
        saved = self.runtime.load_snapshot(snapshot)
        self.assertEqual(saved.rank_policy.fingerprint, self.bound_policy.fingerprint)
        self.assertEqual(saved.rank_state.to_dict(), before.to_dict())
        self._forward(1, clean=False)
        self.assertEqual(self.runtime.history_rank_state("parent"), before)
        self.adapter.fork(snapshot, "child")
        self._forward(1, clean=True)
        self.adapter.seal_chunk()
        self.assertEqual(self.runtime.history_rank_state("parent").completed_chunks, 1)
        self.assertEqual(self.runtime.history_rank_state("child").completed_chunks, 2)
        for key in self.bound_policy.keys:
            self.assertEqual(len(self.runtime.history_rank_decisions("child", key)), 2)
            contexts = self.runtime.history_contexts("child", key)
            self.assertEqual(contexts[0].dtype, "float32" if key.target == "K" else "bfloat16")

    def test_wrong_adaptive_policy_rejected_before_branch_or_rng_mutation(self):
        self.initialize()
        self._forward(0, clean=True)
        self.adapter.seal_chunk()
        snapshot = self.adapter.snapshot(**self._state())
        self.adapter.detach()
        self.adapter = self._adapter(
            rank_policy=replace(self.bound_policy, policy_id="wrong-policy"), resource_budget=budget(),
        )
        self.adapter.attach()
        before = torch.get_rng_state().clone()
        with (
            patch("reelcache.manager.deserialize_tensors",
                  side_effect=AssertionError("native tensors deserialized before the policy gate")) as deserialize,
            patch.object(self.adapter, "_validate_restored_profile",
                         side_effect=AssertionError("native validator ran before the policy gate")) as validate,
        ):
            with self.assertRaisesRegex(PeerFailure, "rank policy"):
                self.adapter.fork(snapshot, "wrong")
            deserialize.assert_not_called()
            validate.assert_not_called()
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertFalse(self.cache.key_value_memory_dict)
        self.assertNotIn("branches/wrong", self.runtime.store.list_refs("branches/"))

    def test_forged_policy_capsule_cannot_bypass_pre_deserialization_head_binding(self):
        self.initialize()
        self._forward(0, clean=True)
        self.adapter.seal_chunk()
        saved = self.runtime.load_snapshot(self.adapter.snapshot(**self._state()))
        wrong_policy = replace(self.bound_policy, policy_id="forged-capsule-policy")
        metadata = copy.deepcopy(saved.metadata)
        metadata["rank_policy"] = wrong_policy.to_dict()
        metadata["rank_policy_state"]["policy_hash"] = wrong_policy.fingerprint
        forged = self.runtime.snapshot("parent", metadata=metadata, tensors=saved.tensors)
        self.adapter.detach()
        self.adapter = self._adapter(rank_policy=wrong_policy, resource_budget=budget())
        self.adapter.attach()
        before = torch.get_rng_state().clone()
        with (
            patch("reelcache.manager.deserialize_tensors",
                  side_effect=AssertionError("native tensors deserialized before the policy gate")) as deserialize,
            patch.object(self.adapter, "_validate_restored_profile",
                         side_effect=AssertionError("native validator ran before the policy gate")) as validate,
        ):
            with self.assertRaisesRegex(PeerFailure, "rank policy"):
                self.adapter.restore(forged, "forged-policy")
            deserialize.assert_not_called()
            validate.assert_not_called()
        self.assertEqual(self.adapter.cursor, 0)
        self.assertEqual(self.adapter.branch_id, "parent")
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertFalse(self.cache.key_value_memory_dict)
        self.assertNotIn("branches/forged-policy", self.runtime.store.list_refs("branches/"))

    def test_rank_state_capsule_must_match_typed_verified_state_before_native_mutation(self):
        self.initialize()
        self._forward(0, clean=True)
        self.adapter.seal_chunk()
        saved = self.runtime.load_snapshot(self.adapter.snapshot(**self._state()))
        metadata = copy.deepcopy(saved.metadata)
        metadata["rank_policy_state"]["completed_chunks"] += 1
        forged = self.runtime.snapshot("parent", metadata=metadata, tensors=saved.tensors)
        before_rng = torch.get_rng_state().clone()
        before_buffers = {
            layer: (value.data_ptr(), _tensor_hash(value))
            for layer, value in self.adapter._quality_buffers.items()
        }
        with self.assertRaisesRegex(PeerFailure, "verified committed rank state"):
            self.adapter.fork(forged, "forged-state")
        self.assertEqual(self.adapter.cursor, 1)
        self.assertEqual(self.adapter.branch_id, "parent")
        self.assertEqual(before_buffers, {
            layer: (value.data_ptr(), _tensor_hash(value))
            for layer, value in self.adapter._quality_buffers.items()
        })
        self.assertTrue(torch.equal(before_rng, torch.get_rng_state()))
        self.assertEqual(self.runtime.history_rank_state("parent").to_dict(), saved.rank_state.to_dict())
        self.assertNotIn("branches/forged-state", self.runtime.store.list_refs("branches/"))

    def test_static_capsule_cannot_hide_verified_adaptive_history(self):
        self.initialize()
        self._forward(0, clean=True)
        self.adapter.seal_chunk()
        saved = self.runtime.load_snapshot(self.adapter.snapshot(**self._state()))
        static_spec = CodecSpec(kind="svd", rank=2)
        metadata = copy.deepcopy(saved.metadata)
        metadata["rank_policy"] = metadata["rank_policy_state"] = None
        metadata["specs"] = {
            f"{key.layer}:{key.target}": static_spec.to_dict() for key in self.bound_policy.keys
        }
        forged = self.runtime.snapshot("parent", metadata=metadata, tensors=saved.tensors)
        self.adapter.detach()
        self.adapter = self._adapter(specs=static_spec)
        self.adapter.attach()
        before = torch.get_rng_state().clone()
        with self.reject_native_state_reads(forged), self.assertRaisesRegex(
            PeerFailure, "static MAGI profile cannot restore adaptive",
        ):
            self.adapter.fork(forged, "hidden-adaptive")
        self.assertEqual(self.adapter.cursor, 0)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertFalse(self.cache.key_value_memory_dict)
        self.assertNotIn("branches/hidden-adaptive", self.runtime.store.list_refs("branches/"))

    def test_adaptive_spec_override_and_postK_domain_are_forbidden(self):
        value = self.policy()
        with self.assertRaises(ConfigurationError):
            self._adapter(rank_policy=value, specs=CodecSpec(kind="svd", rank=2), resource_budget=budget())
        with self.assertRaises(ConfigurationError):
            self._adapter(rank_policy=replace(value, source_dtypes={"K": "bfloat16", "V": "bfloat16"}),
                          resource_budget=budget())

    def test_policy_rebinding_fails_before_request_or_next_native_cache_write(self):
        self.initialize()
        original = self.adapter.rank_policy
        changed = replace(self.bound_policy, policy_id="changed-between-steps")
        before = torch.get_rng_state().clone()
        self.adapter.rank_policy = changed
        with self.assertRaisesRegex(PeerFailure, "binding is immutable"):
            with self.adapter.execution(chunk_index=0, clean=False):
                self.fail("changed policy entered a native forward")
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertEqual(self.adapter.cursor, 0)
        self.assertFalse(self.cache.key_value_memory_dict)
        self.assertEqual(self.runtime.store.list_refs("branches/"), {})
        self.adapter.rank_policy = original
        with self.assertRaisesRegex(PeerFailure, "binding is immutable"):
            with self.adapter.execution(chunk_index=0, clean=False):
                self.adapter.rank_policy = changed
                self._forward(0, clean=False, guarded=False)
        self.assertTrue(all(not layer.self_attention.observed_meta for layer in self.model.videodit_blocks.layers))
        self.assertFalse(self.cache.key_value_memory_dict)
        self.assertEqual(self.runtime.store.list_refs("branches/"), {})

    def test_exact_policy_export_reuses_one_prebound_eigensolve_per_matrix(self):
        from reelcache.adapters.magi_native import _StudyBackend
        from reelcache.manager import KVRuntime
        from reelcache.rank_analysis import AnalysisError, write_spectrum_observation
        from reelcache.spectra import exact_spectrum

        fixture_cell = replace(study_cell("magi720"), frames=1, token_height=2, token_width=3,
                               layers=2, channels=8, world_size=1)
        cells = patch("reelcache.rank_analysis.CELLS", {"magi720": fixture_cell})
        cells.start()
        self.addCleanup(cells.stop)
        self.runtime = KVRuntime(self.root / "exact-store", self.identity, max_resident_bytes=1 << 24)
        policy = replace(self.policy(), solver="exact_oracle", anchor_statistic=None)
        value = replace(request(), arm="adaptive", policy=policy.to_dict())
        backend = _StudyBackend.__new__(_StudyBackend)
        backend.torch = torch
        backend.observation_evidence_kind = "synthetic_fixture"
        backend.verified = {"batch": {"retain_originals": [], "spectrum_tile_rows": 2,
                                      "spectrum_workspace_bytes": 1 << 28}}
        backend.active = {
            "request": value, "directory": self.root, "trajectory_id": "fixture-exact-trajectory",
            "runtime": self.runtime, "deferred_observations": {},
        }
        source_context = {
            "case_id": value.case_id, "trajectory_id": "fixture-exact-trajectory",
            "cell_id": value.cell_id, "group_id": value.group_id, "seed": value.seed,
            "split": value.split, "request_id": value.request_id, "arm_id": "fixture-exact",
            "protocol_id": "a" * 64, "launch_closure_id": "b" * 64,
            "fixture": True, "native_execution": False,
        }
        self.adapter = self._adapter(
            rank_policy=policy, resource_budget=budget(), source_context=source_context,
            source_observer=backend._observe,
        )
        self.adapter.attach()
        solves = []

        def count_solve(*args, **kwargs):
            source = kwargs["source"]
            provenance = source.to_dict()["provenance"]
            for name, expected in {
                **source_context, "boundary": 0, "parent_head": None,
                "source_identity": source.identity, "n_rows": fixture_cell.tokens,
                "n_channels": fixture_cell.channels, "n_layers": fixture_cell.layers,
            }.items():
                self.assertEqual(provenance[name], expected)
            self.assertIs(provenance["fixture"], True)
            self.assertIs(provenance["native_execution"], False)
            for name in ("seed", "boundary", "layer"):
                self.assertIs(type(provenance[name]), int)
            self.assertEqual(provenance["target"], source.target)
            self.assertEqual(provenance["source_dtype"], source.dtype)
            solves.append(kwargs["return_directions"])
            return exact_spectrum(*args, **kwargs)

        with patch("reelcache.spectra.exact_spectrum", new=count_solve):
            self._forward(0, clean=True)
            committed = self.adapter.seal_chunk()
            self.assertEqual(solves, [True] * 4)
            self.assertEqual(len(backend.active["deferred_observations"]), 4)
            self.assertFalse((self.root / "spectra").exists())
            key, capture = next(iter(backend.active["deferred_observations"].items()))
            observation = self.runtime.history_rank_decisions("parent", key)[0].to_dict()["metrics"]["oracle_observation"]
            for name, expected in capture["identity"].items():
                self.assertEqual(observation["source"]["provenance"][name], expected)
            self.assertEqual(observation["source"]["identity"], capture["binding"]["source_identity"])
            rejected = self.root / "rejected-spectra"
            for name, wrong in {
                "case_id": "wrong-paired-case", "n_rows": fixture_cell.tokens + 1,
                "capture_sha256": "0" * 64, "source_identity": "0" * 64,
            }.items():
                with self.subTest(mismatched_binding=name):
                    with self.assertRaises(AnalysisError):
                        write_spectrum_observation(
                            observation, identity={**capture["identity"], name: wrong},
                            spectrum_path=rejected / f"{name}.spectrum.json",
                            observation_path=rejected / f"{name}.rank.json",
                        )
                    self.assertFalse(rejected.exists())
                    self.assertEqual(solves, [True] * 4)
            backend._publish_policy_observations(committed)
            self.assertEqual(solves, [True] * 4)
        self.assertFalse(backend.active["deferred_observations"])
        for key in policy.keys:
            prefix = self.root / "spectra" / "chunk0" / f"layer{key.layer:03d}_{key.target}"
            wrapper = json.loads(prefix.with_suffix(".json").read_text())
            spectrum = json.loads(Path(wrapper["spectrum_artifact"]["path"]).read_text())
            decision = self.runtime.history_rank_decisions("parent", key)[0]
            self.assertEqual(spectrum, decision.to_dict()["metrics"]["oracle_observation"])
            self.assertEqual(spectrum["source"]["provenance"]["case_id"], value.case_id)
            self.assertEqual(spectrum["source"]["provenance"]["trajectory_id"], "fixture-exact-trajectory")
            self.assertEqual(spectrum["source"]["provenance"]["original_bytes_sha256"],
                             wrapper["capture"]["source_sha256"])
            self.assertEqual(wrapper["standalone_offline_decompositions"], 0)
            self.assertEqual(wrapper["offline_oracle_seconds"], 0)
            self.assertTrue(wrapper["shares_policy_decomposition"])
            self.assertTrue(wrapper["included_in_online_selector_cost"])
            self.assertFalse(wrapper["artifact_export_included_in_online_selector_cost"])
            self.assertFalse(wrapper["exact_policy_decomposition_is_additional"])
            self.assertIsNotNone(wrapper["oracle_timings"])
            self.assertEqual(self.runtime.history_rank_state("parent").completed_chunks, 1)


def _gloo_epoch(rank, root):
    import torch.distributed as dist

    root = Path(root)
    dist.init_process_group("gloo", init_method=(root / "rendezvous").as_uri(), rank=rank, world_size=2)
    try:
        transport = DistributedOwner(expected_world_size=2)
        backend = FakeBackend()
        worker = MagiWorkerEpoch(
            cell_id="magi720", config_id="cfg", epoch_id="epoch", transport=transport,
            backend_factory=lambda: backend, budget=budget(),
        )
        a = worker.run(request())
        worker.run(request("B", 29))
        again = worker.run(request())
        if a["result"] != again["result"]:
            raise AssertionError("Gloo A/B/A contaminated a request")
        worker.close()
        never_loaded = []
        bad = MagiWorkerEpoch(
            cell_id="magi720", config_id="cfg", epoch_id="next", transport=DistributedOwner(expected_world_size=2),
            backend_factory=lambda: never_loaded.append(True), budget=budget(),
        )
        try:
            bad.run(request("A" if rank == 0 else "B"))
        except MagiEpochAborted:
            pass
        else:
            raise AssertionError("rank disagreement entered a model")
        if never_loaded or not bad.poisoned:
            raise AssertionError("rank disagreement did not poison both ranks before loading")
    finally:
        dist.destroy_process_group()


def _gloo_post_ulysses_failures(rank, address, report):
    import torch.distributed as dist
    from reelcache.adapters.magi import MagiAdapter

    torch.set_num_threads(1)
    rows = []
    try:
        dist.init_process_group("gloo", init_method=address, rank=rank, world_size=2,
                                timeout=timedelta(seconds=30))
        payload = dist.new_group(backend="gloo", timeout=timedelta(seconds=2))
        control = dist.new_group(backend="gloo", timeout=timedelta(seconds=2))
        geometry = MagiGeometry(layers=1, kv_heads=2, head_dim=4, frames=1, height=2, width=3, world_size=2)
        identity = ModelIdentity("magi-post-ulysses-fixture", "CPU-only", 1, 8, "global_KV_channels", "bfloat16")
        for fault in ("K-write", "V-write", "K-shape", "V-shape", "KV-shape", "native-adjust"):
            model = fixtures.FakeModel(geometry)
            attention = model.videodit_blocks.layers[0].self_attention
            cache = SimpleNamespace(max_batch_size=1, max_sequence_length=6,
                                    key_value_memory_dict={}, update_kv_cache=True, kv_compressed=False)
            adapter = MagiAdapter(
                model, cache, fixtures.FakeRuntime(identity) if rank == 0 else None,
                model_identity=identity, branch_id="file-free", staging_dir=".",
                geometry=geometry, coordinator=DistributedOwner(control, expected_world_size=2),
            )
            adapter._phase = "active_clean"

            def put(key, value):
                if rank == 0 and fault.startswith(key.target + "-"):
                    if fault.endswith("write"):
                        raise OSError(f"injected post-Ulysses {fault}")
                    value = value[:-1]
                if tuple(value.shape) != (geometry.tokens, geometry.layout.local_channels):
                    raise ValueError(f"injected post-Ulysses {key.target}-shape")

            adapter._pending = SimpleNamespace(put=put)
            original = attention.adjust_key_and_value_for_inference

            def adjust(value, state, metadata):
                if rank == 0 and fault == "native-adjust":
                    raise RuntimeError("injected native-adjust failure")
                return original(value, state, metadata)

            meta = SimpleNamespace(
                slice_point=0, clip_token_nums=6, start_chunk_id=0, end_chunk_id=1,
                denoising_range_num=1, extract_prefix_video_feature=True,
                distill_nearly_clean_chunk=False, timestep=torch.full((1, 1), .9999),
            )
            values = torch.ones((6, 1, 8), dtype=torch.bfloat16)
            wire = torch.ones(4)
            dist.all_to_all_single(torch.empty_like(wire), wire, group=payload)
            entered_next = False
            started = time.monotonic()
            try:
                adapter._adjust(0, attention, adjust,
                                values[:-1] if rank == 0 and fault == "KV-shape" else values, meta)
                entered_next = True
                dist.all_to_all_single(torch.empty_like(wire), wire, group=payload)
            except PeerFailure as error:
                elapsed = time.monotonic() - started
                expected = "magi_layer:0" if fault == "KV-shape" else "magi_post_ulysses_adjust:0"
                if entered_next or elapsed >= 1.5 or "timed out" in str(error).lower() or expected not in str(error):
                    raise AssertionError(f"{fault} lacked prompt matched agreement: {error}")
                if cache.key_value_memory_dict or adapter._live_layer is not None:
                    raise AssertionError("failed adjustment retained a live native layer")
                rows.append({"fault": fault, "seconds": elapsed, "next_native_collective": False})
            else:
                raise AssertionError(f"{fault} did not reject on every rank")
        report.send({"rank": rank, "rows": rows})
    except BaseException as exc:
        report.send({"rank": rank, "error": f"{type(exc).__name__}: {exc}"})
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
        report.close()


class GlooPostUlyssesTests(unittest.TestCase):
    def test_file_free_RAW_failures_agree_before_the_next_native_collective(self):
        import multiprocessing
        import socket

        with socket.socket() as reservation:
            reservation.bind(("127.0.0.1", 0))
            address = f"tcp://127.0.0.1:{reservation.getsockname()[1]}"
        context = multiprocessing.get_context("spawn")
        processes, receivers = [], []
        try:
            for rank in range(2):
                receiver, sender = context.Pipe(duplex=False)
                process = context.Process(target=_gloo_post_ulysses_failures, args=(rank, address, sender))
                process.start()
                sender.close()
                processes.append(process)
                receivers.append(receiver)
            for receiver in receivers:
                self.assertTrue(receiver.poll(120), "file-free two-rank regression exceeded its outer deadline")
                result = receiver.recv()
                self.assertNotIn("error", result, result)
                self.assertEqual(len(result["rows"]), 6)
                self.assertTrue(all(not row["next_native_collective"] for row in result["rows"]))
            for process in processes:
                process.join(30)
                self.assertEqual(process.exitcode, 0)
        finally:
            for process in processes:
                if process.is_alive():
                    process.terminate()
                process.join(10)
            for receiver in receivers:
                receiver.close()


def _gloo_capture_failures(rank, root):
    import torch.distributed as dist
    from reelcache.adapters.magi import MagiAdapter

    torch.set_num_threads(1)
    root = Path(root)
    timeout = timedelta(seconds=6)
    dist.init_process_group("gloo", init_method=(root / "rendezvous").as_uri(),
                            rank=rank, world_size=2, timeout=timeout)
    control = dist.new_group(backend="gloo", timeout=timeout)
    try:
        geometry = MagiGeometry(layers=1, kv_heads=2, head_dim=4, frames=1, height=2, width=3, world_size=2)
        model_identity = ModelIdentity("magi-capture-failure-fixture", "CPU-only", 1, 8,
                                       "global_KV_channels", "bfloat16")
        for target in ("K", "V", "K-shape"):
            model = fixtures.FakeModel(geometry)
            cache = SimpleNamespace(
                max_batch_size=1, max_sequence_length=6, key_value_memory_dict={},
                kv_compressed=False, update_kv_cache=True,
            )
            transport = DistributedOwner(control, expected_world_size=2)
            operations = []
            checkpoint = transport.checkpoint

            def record(operation, **kwargs):
                operations.append(operation)
                return checkpoint(operation, **kwargs)

            transport.checkpoint = record
            adapter = MagiAdapter(
                model, cache, fixtures.FakeRuntime(model_identity) if rank == 0 else None,
                model_identity=model_identity, branch_id="capture", staging_dir=root / target,
                geometry=geometry, coordinator=transport, rotary_apply=fixtures.fake_rotary,
                source_observer=lambda description, matrix: None, resource_budget=budget(),
            ).attach()
            native_entered = False
            started = None
            try:
                try:
                    with adapter.execution(chunk_index=0, clean=True, clean_timestep=.9999):
                        def put(key, value):
                            if rank == 0 and key.target == target:
                                raise OSError(f"injected {target} capture failure")

                        adapter._source_pending.put = put
                        rows = geometry.layout.shape(rank, "rows")[0]
                        mixed = torch.ones((rows, geometry.channels))
                        cos = torch.ones((rows, geometry.head_dim // 2))
                        sin = torch.zeros_like(cos)
                        started = time.monotonic()
                        attention = model.videodit_blocks.layers[0].self_attention
                        attention.get_k(mixed[:-1] if rank == 0 and target == "K-shape" else mixed, cos, sin)
                        value = attention.get_v(mixed)
                        native_entered = True
                        native_input = value.reshape(-1).contiguous()
                        dist.all_to_all_single(torch.empty_like(native_input), native_input)
                except PeerFailure as error:
                    elapsed = time.monotonic() - started if started is not None else float("inf")
                    expected = f"magi_pre_ulysses_capture:0:{target[0]}"
                    if expected not in str(error) or elapsed >= 3:
                        raise AssertionError(f"Capture failure was not promptly coordinated: {elapsed}s, {error}")
                else:
                    raise AssertionError("A one-rank capture failure escaped the rank agreement")
                if native_entered or "magi_execution_complete" in operations or adapter.phase != "failed":
                    raise AssertionError("A poisoned capture entered native or cleanup collectives")
            finally:
                adapter.detach()
            dist.barrier(group=control)
    finally:
        dist.destroy_process_group(control)
        dist.destroy_process_group()


class GlooRankWorkerTests(unittest.TestCase):
    def test_two_rank_agreement_A_B_A_and_mismatch_abort(self):
        root = Path(__file__).resolve().parents[1] / ".test-artifacts" / ("magi-rank-gloo-" + uuid.uuid4().hex)
        root.mkdir(parents=True)
        try:
            torch.multiprocessing.start_processes(_gloo_epoch, args=(str(root),),
                                                  nprocs=2, join=True, start_method="spawn")
        finally:
            shutil.rmtree(root)

    def test_two_groups_coordinate_K_and_V_capture_failures_before_all_to_all(self):
        root = Path(__file__).resolve().parents[1] / ".test-artifacts" / ("magi-capture-gloo-" + uuid.uuid4().hex)
        root.mkdir(parents=True)
        try:
            torch.multiprocessing.start_processes(_gloo_capture_failures, args=(str(root),),
                                                  nprocs=2, join=True, start_method="spawn")
        finally:
            shutil.rmtree(root)
