import builtins
import copy
import hashlib
import json
from pathlib import Path
import random
import shutil
import subprocess
import sys
import tracemalloc
from types import SimpleNamespace
import unittest
from unittest import mock
import uuid
import zipfile

import numpy as np
import torch

from reelcache.adapters.magi import _tensor_hash
from reelcache.adapters.magi_preflight import (
    AttentionDigestRecorder, MagiInputBundle, final_noise_rng_record, input_binding,
)
from reelcache.adapters.magi_rank_worker import serial_sampler_contract
from reelcache.serialization import deserialize_tensors, serialize_tensors
from reelcache.study import canonical_json, identity
from scripts import rank_preflight_compare as compare


def file_ref(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "bytes": path.stat().st_size}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    return file_ref(path)


def reader_start_ticks():
    value = Path("/proc/self/stat").read_text()
    return value[value.rfind(")") + 2:].split()[19]


GEOMETRY = {
    "layers": 2, "world_size": 2, "tokens": 2, "local_heads": 1, "head_dim": 2,
    "latent_shape": [1, 1, 1, 2, 2], "rgb_shape": [2, 2, 2, 3],
    "crop_rows": [0, 2], "fps": 24,
}
EXPECTED = {"protocol_id": "a" * 64, "launch_closure_id": "b" * 64, "cell_id": "cpu-magi-fixture"}


class ArtifactFixture:
    """Real tiny serialized tensors, NPZ pixels and native-format trace records."""

    def __init__(self, root, kind="raw_native_noop"):
        self.root, self.kind = Path(root).resolve(), kind
        self.root.mkdir()
        self.cases, self.epochs = {}, []
        receipt = {
            "schema": compare.FIXTURE_RECEIPT_SCHEMA, "protocol_id": EXPECTED["protocol_id"],
            "source_closure": {"closure_id": EXPECTED["launch_closure_id"]}, "phase": "native_preflight",
            "fixture": True, "native_execution": False,
        }
        receipt["receipt_id"] = identity(receipt)
        self.receipt = write_json(self.root / "launch.json", receipt)
        if kind == "raw_native_noop":
            self.epoch([("raw", "A", "reelcache", "runtime")])
            self.epoch([("native_noop", "A", "native_noop", "native_dense")])
        elif kind == "hot_reload":
            self.epoch([("hot", "A", "reelcache", "quality_hot")])
            self.epoch([("reload", "A", "reelcache", "runtime")])
        else:
            self.epoch([("a_before", "A", "reelcache", "runtime"), ("b", "B", "reelcache", "runtime"),
                        ("a_after", "A", "reelcache", "runtime")])
            self.epoch([("fresh_a", "A", "reelcache", "runtime")])
            self.epoch([("fresh_b", "B", "reelcache", "runtime")])
        self.request = {
            "schema": compare.REQUEST_SCHEMA, "model": "magi", "comparison_kind": kind,
            "evidence_kind": "synthetic_fixture", "artifact_root": str(self.root),
            "expected": dict(EXPECTED), "synthetic_geometry": copy.deepcopy(GEOMETRY),
            "limits": {"host_bytes": 64 << 20, "metadata_bytes": 4 << 20},
            "frame_window": [0, 4], "denoise_attention_calls": 1, "require_final_rng": False,
        }
        self.refresh()

    def epoch(self, descriptions):
        number = len(self.epochs)
        directory = self.root / f"epoch-{number}"
        directory.mkdir()
        batch_id = identity({"fixture_epoch": number})
        workers = []
        for rank in range(2):
            workers.append(write_json(directory / f"worker{rank}.json", {
                "fixture": True, "native_execution": False, "plan_identity": batch_id,
                "host": "synthetic-host", "pid": 1000 + 2 * number + rank,
                "process_start_ticks": reader_start_ticks(),
                "environment": {"RANK": str(rank), "WORLD_SIZE": "2"},
            }))
        epoch = {
            "directory": directory, "cases": [], "generation": {
                "schema": compare.GENERATION_SCHEMA, "batch_id": batch_id,
                "protocol_id": EXPECTED["protocol_id"], "cell_id": EXPECTED["cell_id"],
                "asset_id": "c" * 64, "models_unloaded": True, "vae_loaded": False,
                "worker_execution_records": workers, "model_startup_seconds": number + .125,
                **serial_sampler_contract(),
            },
            "decode": {"schema": compare.DECODE_SCHEMA, "batch_id": batch_id,
                       "cell_id": EXPECTED["cell_id"], "models_unloaded_before_VAE_load": True},
        }
        for role, letter, backend, mode in descriptions:
            case = self.case(directory, role, letter, backend, mode)
            case["epoch"] = epoch
            epoch["cases"].append(case)
            self.cases[role] = case
        self.epochs.append(epoch)

    def case(self, directory, role, letter, backend, mode):
        directory = directory / role
        directory.mkdir()
        seed = 11 if letter == "A" else 29
        context = {
            **EXPECTED, **serial_sampler_contract(), "case_id": f"fixture-case-{letter}",
            "group_id": f"fixture-group-{letter}", "prompt_id": f"fixture-prompt-{letter}",
            "prompt": f"CPU fixture {letter}, never model output", "seed": seed, "chunks": 2,
            "source_fps": 24, "asset_id": "c" * 64, "config_id": "d" * 64, "split": "development",
            "from_scratch": True, "skipped_initial_noise": False, "fixture": True, "native_execution": False,
            "geometry": copy.deepcopy(GEOMETRY), "request_hash": identity({"role": role, "seed": seed}),
            "arm": "raw", "rank_selection_mode": "raw", "rank_k": None, "rank_v": None,
            "layer_ranks": None, "rank_policy_hash": None, "preflight_backend": backend,
        }
        binding = input_binding(
            SimpleNamespace(**{name: context[name] for name in (
                "case_id", "cell_id", "group_id", "prompt_id", "prompt", "seed", "chunks", "source_fps",
            )}, latent_shape=GEOMETRY["latent_shape"]),
            protocol_id=EXPECTED["protocol_id"], asset_id="c" * 64, config_id="d" * 64,
            launch_closure_id=EXPECTED["launch_closure_id"],
        )
        bundle = MagiInputBundle(mode="capture", binding=binding, directory=directory / "inputs", max_bytes=1 << 20)
        bundle.conditioning({
            "y": torch.full((2, 1, 2, 2), float(seed)), "masks": torch.ones(2, 1, 2, dtype=torch.int64),
            "null_caption": torch.zeros(1, 2, 2), "null_mask": torch.ones(1, 2, dtype=torch.int64),
        })
        generators = [torch.Generator().manual_seed(seed) for _ in range(2)]
        generator, noises, states = generators[0], [], []
        for chunk in range(2):
            noise = torch.randn(GEOMETRY["latent_shape"], generator=generator)
            repeated = torch.randn(GEOMETRY["latent_shape"], generator=generators[1])
            assert torch.equal(noise, repeated)
            state = generator.get_state().clone()
            bundle.noise(chunk, noise, state)
            noises.append(noise)
            states.append(state)
        bundle_receipt = bundle.finish()
        latents, media = [], []
        namespace = f"synthetic-{role}-{uuid.uuid4().hex}"
        for chunk, noise in enumerate(noises):
            latent = noise + chunk + .5
            path = directory / f"latent{chunk}.pt"
            with path.open("wb") as stream:
                torch.save({"latent": latent}, stream)
            latents.append({"chunk_index": chunk, "artifact": file_ref(path), "tensor_sha256": _tensor_hash(latent),
                            "noise_sha256": _tensor_hash(noise), "shape": list(latent.shape), "dtype": "float32"})
            pixels = (np.arange(24, dtype=np.uint8) + seed + chunk).reshape(GEOMETRY["rgb_shape"])
            path = directory / f"rgb{chunk}.npz"
            with path.open("wb") as stream:
                np.savez_compressed(stream, frames=pixels)
            media.append({
                "namespace": namespace, "chunk_index": chunk, "artifact": file_ref(path),
                "shape": list(pixels.shape), "rgb_sha256": hashlib.sha256(memoryview(pixels)).hexdigest(),
                "latent_sha256": _tensor_hash(latent), "fps": 24, "crop_rows": [0, 2],
                "VAE_state": "stateless_chunk_decode", "streaming_seam_parity_verified": False,
                "native_decode_seconds": .125 + len(role),
            })
        traces = []
        geometry = SimpleNamespace(
            layers=2, tokens=2, head_dim=2, layout=SimpleNamespace(local_heads=1, local_channels=2),
        )
        for rank in range(2):
            recorder = AttentionDigestRecorder(
                directory / f"attention{rank}.jsonl", geometry=geometry, rank=rank,
                binding_id=identity(binding), max_chunks=2, max_records=68, max_bytes=1 << 20,
            )
            for chunk in range(2):
                for phase, step in [*(("denoise", index) for index in range(16)), ("clean", None)]:
                    near = phase == "denoise" and step >= 12
                    for layer in range(2):
                        prefix = [
                            torch.full((2, 1, 2), seed + 40 * rank + 5 * layer + i, dtype=torch.bfloat16)
                            for i in range(chunk)
                        ]
                        value = seed + 40 * rank + 5 * layer + chunk
                        if phase == "denoise":
                            value += .5 * step
                        current = torch.full((2, 1, 2), value, dtype=torch.bfloat16)
                        key = torch.cat(prefix + [current] + ([current + 1] if near else []))
                        point = {"chunk_index": chunk, "phase": phase, "step_index": step,
                                 "input_noise_sha256": _tensor_hash(noises[chunk])}
                        recorder.record(
                            point, layer, key, key * 2, near_clean=near,
                            query_ranges=[[i * 2, (i + 1) * 2] for i in range(1 + near)],
                            key_ranges=[[0, len(key)] for _ in range(1 + near)],
                        )
            traces.append(recorder.finish())
        return {
            "role": role, "directory": directory, "bundle": bundle.document, "media": media,
            "generators": generators,
            "result": {
                "namespace": namespace, "request": context, "latents": latents,
                "preflight_backend": backend, "runtime_created": backend != "native_noop", "mode": mode,
                "input_bundle": bundle_receipt, "conditioning_input": bundle.document["conditioning"],
                "chunk_inputs": bundle.document["chunks"], "attention_digests": traces,
                "final_noise_rng": [
                    final_noise_rng_record(generator, rank=rank, binding_id=bundle_receipt["binding_id"])
                    for rank, generator in enumerate(generators)
                ],
                "storage_owner_record": None, "snapshot_id": None,
            },
        }

    def refresh(self):
        for case in self.cases.values():
            case["result_ref"] = write_json(case["directory"] / "result.json", case["result"])
        for epoch in self.epochs:
            epoch["generation"]["results"] = [
                {"namespace": case["result"]["namespace"], "sequence": sequence,
                 "request_hash": case["result"]["request"]["request_hash"],
                 "seconds_generation_and_cleanup": 123 + sequence, "result": case["result"]}
                for sequence, case in enumerate(epoch["cases"])
            ]
            epoch["generation_ref"] = write_json(epoch["directory"] / "generation.json", epoch["generation"])
            epoch["decode"].update(
                generation_manifest=epoch["generation_ref"],
                media=[row for case in epoch["cases"] for row in case["media"]],
            )
            epoch["decode_ref"] = write_json(epoch["directory"] / "decode.json", epoch["decode"])
        self.request["trials"] = [
            {"role": role, "result": self.cases[role]["result_ref"],
             "generation": self.cases[role]["epoch"]["generation_ref"],
             "decode": self.cases[role]["epoch"]["decode_ref"], "launch_receipt": self.receipt}
            for role in compare.ROLES[self.kind]
        ]
        return self.save_request()

    def save_request(self):
        self.request_ref = write_json(self.root / "comparison_request.json", self.request)
        return self.request_ref

    def run(self, output=None):
        return compare.compare_preflight(
            self.request_ref["path"], request_sha256=self.request_ref["sha256"], output_path=output,
        )

    def trace_change(self, role, mutator, rank=0):
        receipt = self.cases[role]["result"]["attention_digests"][rank]
        path = Path(receipt["artifact"]["path"])
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        mutator(rows)
        path.write_bytes(b"".join(canonical_json(row) + b"\n" for row in rows))
        receipt.update(artifact=file_ref(path), records=len(rows))
        self.refresh()

    def input_change(self, role, name):
        case = self.cases[role]
        document = case["bundle"]
        record = document["chunks"][0]
        path = case["directory"] / "inputs" / record["path"]
        tensors = deserialize_tensors(path.read_bytes(), device="cpu")
        tensors[name].reshape(-1)[0] += 1
        path.write_bytes(serialize_tensors(tensors))
        record.update(sha256=file_ref(path)["sha256"], bytes=path.stat().st_size)
        record["tensors"][name]["sha256"] = _tensor_hash(tensors[name])
        document["bundle_id"] = identity({key: value for key, value in document.items() if key != "bundle_id"})
        reference = write_json(case["directory"] / "inputs/bundle.json", document)
        case["result"]["input_bundle"].update(bundle_id=document["bundle_id"], artifact=reference)
        self.refresh()


class PreflightComparatorTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(".test-artifacts").resolve() / ("preflight-compare-" + uuid.uuid4().hex)
        self.root.mkdir(parents=True, mode=0o700)

    def tearDown(self):
        shutil.rmtree(self.root)

    def fixture(self, kind="raw_native_noop"):
        return ArtifactFixture(self.root / (kind + "-" + uuid.uuid4().hex), kind)

    def assert_failed(self, fixture, message=None):
        result = fixture.run()
        self.assertEqual(result["status"], "failed", result)
        self.assertFalse(result["g3_qualification"])
        if message is not None:
            self.assertIsNotNone(result["error"])
            self.assertIn(message, result["error"]["message"])
        return result

    def test_real_artifacts_match_without_namespace_timing_or_global_rng_comparison(self):
        before_python, before_torch = random.getstate(), torch.get_rng_state().clone()
        fixture = self.fixture()
        real_import = builtins.__import__

        def guarded_import(name, *args, **kwargs):
            if name == "reelcache.adapters.magi_native" or name.split(".")[0] in ("inference", "transformers", "diffusers"):
                raise AssertionError("comparison must not import native/model implementations")
            return real_import(name, *args, **kwargs)

        with mock.patch("builtins.__import__", side_effect=guarded_import), \
                mock.patch("torch.cuda.init", side_effect=AssertionError("no GPU")), \
                mock.patch("torch.set_num_threads", side_effect=AssertionError("no global thread changes")), \
                mock.patch("torch.load", wraps=torch.load) as tensor_loader:
            result = fixture.run(self.root / "receipt.json")
        self.assertEqual(result["status"], "passed", result)
        self.assertEqual(result["evidence_kind"], "synthetic_fixture")
        self.assertTrue(result["source_evidence_validated"])
        self.assertFalse(result["g3_qualification"])
        self.assertFalse(result["gate_advanced"])
        self.assertTrue(result["pairs"][0]["final_generator_state_compared"])
        self.assertTrue(result["pairs"][0]["checks"]["final_noise_rng_bitwise_equal"])
        self.assertEqual(result["missing_evidence"], [])
        self.assertGreater(result["measurements"]["final_noise_rng_bytes_verified"], 0)
        self.assertEqual(result["measurements"]["attention_records_verified"], 272)
        self.assertGreater(result["measurements"]["input_tensor_bytes_verified"], 0)
        self.assertGreater(result["measurements"]["rgb_bytes_verified"], 0)
        self.assertEqual(random.getstate(), before_python)
        self.assertTrue(torch.equal(torch.get_rng_state(), before_torch))
        self.assertEqual(tensor_loader.call_count, 4)
        self.assertTrue(all(call.kwargs == {"map_location": "cpu", "weights_only": True}
                            for call in tensor_loader.call_args_list))
        self.assertEqual(json.loads((self.root / "receipt.json").read_text()), result)
        self.assertEqual(result["comparison_id"], identity({k: v for k, v in result.items() if k != "comparison_id"}))
        with self.assertRaisesRegex(compare.ComparisonError, "already exists"):
            fixture.run(self.root / "receipt.json")

    def test_hot_reload_and_aba_with_fresh_pair_compare_actual_outputs(self):
        for kind in ("hot_reload", "request_isolation"):
            with self.subTest(kind=kind):
                result = self.fixture(kind).run()
                self.assertEqual(result["status"], "passed", result)
                self.assertEqual(len(result["pairs"]), 3 if kind == "request_isolation" else 1)

    def test_worker_start_ticks_use_actual_proc_reader_strings_and_reject_aliases(self):
        ticks = reader_start_ticks()
        self.assertIsInstance(ticks, str)
        self.assertEqual(compare._start_ticks(ticks), int(ticks))
        fixture = self.fixture()
        generation = fixture.epochs[0]["generation"]
        for reference in generation["worker_execution_records"]:
            self.assertEqual(json.loads(Path(reference["path"]).read_text())["process_start_ticks"], ticks)
        budget = compare._Budget(fixture.root, fixture.request["limits"])
        self.assertTrue(all(item[2] == int(ticks) for item in compare._workers(generation, budget, 2)))
        for value in (1, True, 1.0, None, "", "0", "01", "+1", "-1", "1 ", " 1", "1.0", "١", "１",
                      str(1 << 64), "9" * 4096):
            with self.subTest(ticks=value), self.assertRaisesRegex(compare.ComparisonError, "start ticks"):
                compare._start_ticks(value)

    def test_cfg1_requires_one_digest_even_when_near_clean_has_two_visibility_rows(self):
        fixture = self.fixture()
        for case in fixture.cases.values():
            for receipt in case["result"]["attention_digests"]:
                rows = [json.loads(line) for line in Path(receipt["artifact"]["path"]).read_text().splitlines()]
                self.assertEqual(receipt["records"], 2 * 2 * 17)
                self.assertTrue(all(row["attention_call"] == 0 for row in rows))
                self.assertTrue(all(len(row["query_ranges"]) == 2 for row in rows if row["near_clean"]))
        self.assertEqual(compare.sample_request()["denoise_attention_calls"], 1)
        fixture.request["denoise_attention_calls"] = 2
        fixture.save_request()
        with mock.patch.object(compare, "_load_trial", side_effect=AssertionError("reject before artifact loading")):
            self.assert_failed(fixture, "exactly one attention digest")

    def test_one_changed_pixel_fails_but_common_input_pairing_hash_stays_same(self):
        fixture = self.fixture()
        before = fixture.run()["pairs"][0]["pairing_sha256"]
        record = fixture.cases["native_noop"]["media"][0]
        path = Path(record["artifact"]["path"])
        with np.load(path, allow_pickle=False) as archive:
            pixels = archive["frames"].copy()
        pixels[0, 0, 0, 0] ^= 1
        with path.open("wb") as stream:
            np.savez_compressed(stream, frames=pixels)
        record.update(artifact=file_ref(path), rgb_sha256=hashlib.sha256(memoryview(pixels)).hexdigest())
        fixture.refresh()
        result = self.assert_failed(fixture)
        self.assertIsNone(result["error"])
        self.assertFalse(result["pairs"][0]["checks"]["rgb_chunk_0"])
        self.assertEqual(result["pairs"][0]["pairing_sha256"], before)

    def test_changed_latent_bytes_fail_even_when_all_file_and_tensor_hashes_are_updated(self):
        fixture = self.fixture()
        case = fixture.cases["native_noop"]
        record = case["result"]["latents"][0]
        path = Path(record["artifact"]["path"])
        latent = torch.load(path, map_location="cpu", weights_only=True)["latent"]
        latent.reshape(-1)[0] += 1
        with path.open("wb") as stream:
            torch.save({"latent": latent}, stream)
        record.update(artifact=file_ref(path), tensor_sha256=_tensor_hash(latent))
        case["media"][0]["latent_sha256"] = record["tensor_sha256"]
        fixture.refresh()
        result = self.assert_failed(fixture)
        self.assertFalse(result["pairs"][0]["checks"]["latent_chunk_0"])

    def test_actual_input_noise_and_postdraw_rng_changes_are_rejected(self):
        for name in ("noise", "postdraw_generator_state"):
            with self.subTest(tensor=name):
                fixture = self.fixture()
                fixture.input_change("native_noop", name)
                self.assert_failed(fixture)

    def test_bundle_tensor_metadata_is_reverified_not_trusted_from_file_hash(self):
        fixture = self.fixture()
        case = fixture.cases["native_noop"]
        case["bundle"]["conditioning"]["tensors"]["y"]["sha256"] = "f" * 64
        body = case["bundle"]
        body["bundle_id"] = identity({k: v for k, v in body.items() if k != "bundle_id"})
        case["result"]["input_bundle"].update(
            artifact=write_json(case["directory"] / "inputs/bundle.json", body), bundle_id=body["bundle_id"],
        )
        fixture.refresh()
        self.assert_failed(fixture, "actual bundle tensor metadata")

    def test_changed_attention_visibility_and_actual_digest_value_fail_pair(self):
        for field in ("key_ranges", "key_sha256"):
            with self.subTest(field=field):
                fixture = self.fixture()

                def change(rows):
                    row = next(row for row in rows if row["chunk_index"] == 1 and row["phase"] == "denoise")
                    row[field] = [[2, 4]] if field == "key_ranges" else "f" * 64

                fixture.trace_change("native_noop", change)
                result = self.assert_failed(fixture)
                self.assertFalse(result["pairs"][0]["checks"]["attention_events_values_and_archive_preservation"])

    def test_archive_must_match_every_step_and_the_preceding_clean_capture(self):
        for all_steps in (False, True):
            with self.subTest(all_steps=all_steps):
                fixture = self.fixture()

                def change(rows):
                    selected = [row for row in rows if row["chunk_index"] == 1 and row["layer"] == 0]
                    for row in selected if all_steps else selected[:1]:
                        row["archived_key_sha256"] = "f" * 64

                fixture.trace_change("native_noop", change)
                self.assert_failed(fixture, "archived prefix")

    def test_duplicate_missing_calls_ranks_and_layers_are_rejected(self):
        changes = {
            "duplicate": lambda rows: rows.__setitem__(1, copy.deepcopy(rows[0])),
            "missing": lambda rows: rows.pop(),
            "rank": lambda rows: rows[0].__setitem__("rank", 1),
            "layer": lambda rows: rows[0].__setitem__("layer", 2),
            "call": lambda rows: rows[0].__setitem__("attention_call", 1),
        }
        for name, change in changes.items():
            with self.subTest(change=name):
                fixture = self.fixture()
                fixture.trace_change("native_noop", change)
                self.assert_failed(fixture)

    def test_decode_manifest_binding_and_missing_rgb_chunks_are_rejected(self):
        for change in ("binding", "missing"):
            with self.subTest(change=change):
                fixture = self.fixture()
                epoch = fixture.cases["native_noop"]["epoch"]
                if change == "binding":
                    epoch["decode"]["generation_manifest"] = fixture.cases["raw"]["epoch"]["generation_ref"]
                else:
                    epoch["decode"]["media"].pop()
                reference = write_json(epoch["directory"] / "decode.json", epoch["decode"])
                fixture.request["trials"][1]["decode"] = reference
                fixture.save_request()
                self.assert_failed(fixture, "decode")

    def test_fps_crop_and_whole_frame_window_must_match(self):
        for name, value in (("fps", 16), ("crop_rows", [1, 3]), ("VAE_state", "verified_streaming")):
            with self.subTest(field=name):
                fixture = self.fixture()
                fixture.cases["native_noop"]["media"][0][name] = value
                fixture.refresh()
                self.assert_failed(fixture)
        fixture = self.fixture()
        fixture.request["frame_window"] = [1, 4]
        fixture.save_request()
        self.assert_failed(fixture, "frame window")

    def test_mislabeled_native_fixture_receipt_closure_and_boolean_aliases_fail(self):
        changes = (
            ("fixture", False), ("native_execution", True), ("launch_closure_id", "e" * 64),
            ("protocol_id", "e" * 64), ("chunks", True),
        )
        for name, value in changes:
            with self.subTest(field=name):
                fixture = self.fixture()
                fixture.cases["native_noop"]["result"]["request"][name] = value
                fixture.refresh()
                self.assert_failed(fixture)
        fixture = self.fixture()
        fixture.request["evidence_kind"] = "native"
        fixture.save_request()
        result = self.assert_failed(fixture, "synthetic geometry")
        self.assertEqual(result["evidence_kind"], "unverified")

    def test_nonfinite_latents_and_nonfinite_unused_json_fail(self):
        fixture = self.fixture()
        case = fixture.cases["native_noop"]
        record = case["result"]["latents"][0]
        path = Path(record["artifact"]["path"])
        latent = torch.load(path, map_location="cpu", weights_only=True)["latent"]
        latent.reshape(-1)[0] = float("nan")
        with path.open("wb") as stream:
            torch.save({"latent": latent}, stream)
        record.update(artifact=file_ref(path), tensor_sha256=_tensor_hash(latent))
        case["media"][0]["latent_sha256"] = record["tensor_sha256"]
        fixture.refresh()
        self.assert_failed(fixture, "nonfinite tensor")
        fixture = self.fixture()
        fixture.epochs[0]["generation"]["model_startup_seconds"] = float("nan")
        fixture.refresh()
        self.assert_failed(fixture, "nonfinite JSON")

    def test_changed_artifact_hash_and_tensor_hash_are_not_interchangeable(self):
        fixture = self.fixture()
        case = fixture.cases["native_noop"]
        row = case["result"]["latents"][0]
        self.assertNotEqual(row["tensor_sha256"], row["artifact"]["sha256"])
        row["tensor_sha256"] = row["artifact"]["sha256"]
        case["media"][0]["latent_sha256"] = row["tensor_sha256"]
        fixture.refresh()
        self.assert_failed(fixture, "actual latent metadata/hash")
        fixture = self.fixture()
        path = Path(fixture.cases["native_noop"]["media"][0]["artifact"]["path"])
        path.write_bytes(path.read_bytes() + b"unadmitted trailing bytes")
        self.assert_failed(fixture, "artifact size")

    def test_declared_host_metadata_budgets_and_hash_bound_request_reject(self):
        fixture = self.fixture()
        with self.assertRaisesRegex(compare.ComparisonError, "request SHA256"):
            compare.compare_preflight(fixture.request_ref["path"], request_sha256="0" * 64)
        fixture.request["limits"]["host_bytes"] = 1
        fixture.save_request()
        with self.assertRaisesRegex(compare.ComparisonError, "host byte budget"):
            fixture.run()
        fixture.request["limits"] = {"host_bytes": 64 << 20, "metadata_bytes": 1}
        fixture.save_request()
        with self.assertRaisesRegex(compare.ComparisonError, "metadata byte budget"):
            fixture.run()
        fixture.request["limits"] = {"host_bytes": 2 << 20, "metadata_bytes": 4 << 20}
        fixture.save_request()
        self.assert_failed(fixture, "host byte budget")

    def test_invalid_shape_and_compressed_expansion_reject_before_tensor_or_rgb_loading(self):
        fixture = self.fixture()
        fixture.cases["native_noop"]["result"]["latents"][0]["shape"][0] = True
        fixture.refresh()
        with mock.patch("torch.load", side_effect=AssertionError("invalid geometry must reject before load")):
            self.assert_failed(fixture, "positive integer dimensions")
        fixture = self.fixture()
        record = fixture.cases["raw"]["media"][0]
        path = Path(record["artifact"]["path"])
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("frames.npy", b"x" * ((1 << 20) + 1024))
        record["artifact"] = file_ref(path)
        fixture.refresh()
        with mock.patch("zipfile.ZipFile.open", side_effect=AssertionError("unadmitted expansion must not load")):
            self.assert_failed(fixture, "uncompressed size")

    def test_zip_4096_member_counterexample_is_rejected_before_constructor_with_prior_admission(self):
        path = self.root / "too-many-members.zip"
        with zipfile.ZipFile(path, "w") as archive:
            for index in range(4096):
                archive.writestr(f"{index:04d}", b"")
        del archive
        for host_bytes in (1_114_112, 64 << 20):
            budget = compare._Budget(self.root, {"host_bytes": host_bytes, "metadata_bytes": 1 << 20})
            with self.subTest(host_bytes=host_bytes), \
                    mock.patch.object(budget, "admit", wraps=budget.admit) as admitted, \
                    mock.patch("zipfile.ZipFile", side_effect=AssertionError("directory must reject before parsing")) as ctor:
                with self.assertRaises(compare.ComparisonError):
                    compare._zip_admission(path, 0, budget)
                self.assertTrue(admitted.called)
                ctor.assert_not_called()

    def test_zip_count_directory_length_and_rgb_multi_entry_fail_before_constructor(self):
        original = self.root / "two-members.zip"
        with zipfile.ZipFile(original, "w") as archive:
            archive.writestr("a", b"")
            archive.writestr("b", b"")
        content = original.read_bytes()
        offset = content.rfind(b"PK\x05\x06")
        footer = list(compare.ZIP_END.unpack(content[offset:offset + compare.ZIP_END.size]))
        variants = {}
        changed = footer.copy()
        changed[3] = changed[4] = 1
        variants["understated-count"] = content[:offset] + compare.ZIP_END.pack(*changed)
        changed = footer.copy()
        changed[5] += 1
        variants["bad-directory-length"] = content[:offset] + compare.ZIP_END.pack(*changed)
        changed = footer.copy()
        changed[5], changed[6] = 0xfffffffe, 0
        variants["oversized-directory"] = content[:offset] + compare.ZIP_END.pack(*changed)
        entry_offset = footer[6]
        entry = list(compare.ZIP_CENTRAL.unpack(content[entry_offset:entry_offset + compare.ZIP_CENTRAL.size]))
        entry[10] = 65535
        variants["oversized-member-name"] = (
            content[:entry_offset] + compare.ZIP_CENTRAL.pack(*entry)
            + content[entry_offset + compare.ZIP_CENTRAL.size:]
        )
        variants["rgb-two-members"] = content
        for name, value in variants.items():
            path = self.root / (name + ".zip")
            path.write_bytes(value)
            budget = compare._Budget(self.root, {"host_bytes": 64 << 20, "metadata_bytes": 1 << 20})
            with self.subTest(variant=name), \
                    mock.patch("zipfile.ZipFile", side_effect=AssertionError("must reject before ZipFile")) as ctor:
                with self.assertRaises(compare.ComparisonError):
                    compare._zip_admission(path, 0, budget, max_members=1 if name == "rgb-two-members" else 32)
                ctor.assert_not_called()

    def test_actual_torch_zip64_and_numpy_npz_fit_the_admitted_and_measured_directory_budget(self):
        tensor_path, rgb_path = self.root / "actual-torch.pt", self.root / "actual-numpy.npz"
        with tensor_path.open("wb") as stream:
            torch.save({"latent": torch.arange(16, dtype=torch.float32)}, stream)
        with rgb_path.open("wb") as stream:
            np.savez_compressed(stream, frames=np.arange(24, dtype=np.uint8).reshape(2, 2, 2, 3))
        self.assertIn(b"PK\x06\x06", tensor_path.read_bytes())
        self.assertIn(b"PK\x06\x07", tensor_path.read_bytes())
        real_constructor = zipfile.ZipFile
        for path, data_bytes, members in ((tensor_path, 64, 32), (rgb_path, 24, 1)):
            budget = compare._Budget(self.root, {"host_bytes": 1_200_000, "metadata_bytes": 1 << 20})
            with self.subTest(producer=path.name), mock.patch.object(budget, "admit", wraps=budget.admit) as admitted:
                def construct(*args, **kwargs):
                    self.assertGreaterEqual(admitted.call_count, 2)
                    return real_constructor(*args, **kwargs)

                with mock.patch("zipfile.ZipFile", side_effect=construct):
                    tracemalloc.start()
                    try:
                        entries = compare._zip_admission(path, data_bytes, budget, max_members=members)
                        _, peak = tracemalloc.get_traced_memory()
                    finally:
                        tracemalloc.stop()
                self.assertGreater(len(entries), 0)
                self.assertLessEqual(peak, budget.host)
                self.assertLessEqual(budget.peak_admission, budget.host)
        content = tensor_path.read_bytes()
        locator_offset = content.rfind(b"PK\x06\x07")
        locator = list(compare.ZIP64_LOCATOR.unpack(
            content[locator_offset:locator_offset + compare.ZIP64_LOCATOR.size],
        ))
        locator[2] += 1
        malformed = self.root / "bad-zip64-locator.pt"
        malformed.write_bytes(
            content[:locator_offset] + compare.ZIP64_LOCATOR.pack(*locator)
            + content[locator_offset + compare.ZIP64_LOCATOR.size:]
        )
        budget = compare._Budget(self.root, {"host_bytes": 64 << 20, "metadata_bytes": 1 << 20})
        with mock.patch("zipfile.ZipFile", side_effect=AssertionError("bad ZIP64 must reject before parsing")) as ctor:
            with self.assertRaisesRegex(compare.ComparisonError, "ZIP64"):
                compare._zip_admission(malformed, 64, budget)
            ctor.assert_not_called()

    def test_zip_directory_allowance_does_not_scale_with_large_payload_position(self):
        source = self.root / "small-producer.npz"
        with source.open("wb") as stream:
            np.savez_compressed(stream, frames=np.arange(24, dtype=np.uint8).reshape(2, 2, 2, 3))
        content = source.read_bytes()
        end_offset = content.rfind(b"PK\x05\x06")
        footer = list(compare.ZIP_END.unpack(content[end_offset:end_offset + compare.ZIP_END.size]))
        directory = content[footer[6]:end_offset]
        path = self.root / "large-sparse-fixture.npz"
        with path.open("wb") as stream:
            stream.write(content[:footer[6]])
            stream.seek(128 << 20)
            footer[6] = stream.tell()
            stream.write(directory)
            stream.write(compare.ZIP_END.pack(*footer))
        budget = compare._Budget(self.root, {"host_bytes": 1_500_000, "metadata_bytes": 1 << 20})
        count, allowance = compare._zip_directory_admission(path, budget, 0, 1)
        self.assertEqual(count, 1)
        self.assertLess(allowance, 256 << 10)
        self.assertLessEqual(budget.peak_admission, budget.host)
        with np.load(path, allow_pickle=False) as archive:
            np.testing.assert_array_equal(archive["frames"], np.arange(24, dtype=np.uint8).reshape(2, 2, 2, 3))

    def test_malformed_tensor_archive_becomes_a_failed_receipt(self):
        fixture = self.fixture()
        record = fixture.cases["native_noop"]["result"]["latents"][0]
        path = Path(record["artifact"]["path"])
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("archive/data.pkl", b"not a valid weights-only tensor pickle")
            archive.writestr("archive/version", b"3\n")
        record["artifact"] = file_ref(path)
        fixture.refresh()
        self.assert_failed(fixture)

    def test_missing_symlinked_artifacts_and_nonobject_metadata_fail(self):
        fixture = self.fixture()
        record = fixture.cases["native_noop"]["result"]["latents"][0]
        Path(record["artifact"]["path"]).unlink()
        self.assert_failed(fixture)
        fixture = self.fixture()
        record = fixture.cases["native_noop"]["result"]["latents"][0]
        alias = fixture.root / "latent-symlink.pt"
        alias.symlink_to(record["artifact"]["path"])
        record["artifact"]["path"] = str(alias)
        fixture.refresh()
        self.assert_failed(fixture, "symlinks")
        fixture = self.fixture()
        reference = write_json(fixture.root / "not-a-batch.json", [])
        fixture.request["trials"][1]["generation"] = reference
        fixture.save_request()
        self.assert_failed(fixture, "must be objects")

    def test_native_geometry_is_fixed_without_allocating_native_sized_artifacts(self):
        for cell, tokens, crop in (("magi720", 21600, [0, 720]), ("magi1080", 48960, [4, 1084])):
            with self.subTest(cell=cell):
                geometry = compare._geometry({"evidence_kind": "native", "expected": {"cell_id": cell}})
                self.assertEqual(geometry["layers"], 48)
                self.assertEqual(geometry["world_size"], 2)
                self.assertEqual(geometry["tokens"], tokens)
                self.assertEqual(geometry["crop_rows"], crop)
                self.assertEqual(geometry["rgb_shape"][0], 24)

    def test_boolean_aliases_cannot_stand_in_for_crop_visibility_or_window_coordinates(self):
        fixture = self.fixture()
        fixture.cases["native_noop"]["media"][0]["crop_rows"][0] = False
        fixture.refresh()
        self.assert_failed(fixture, "crop")
        fixture = self.fixture()
        fixture.trace_change("native_noop", lambda rows: rows[0]["query_ranges"][0].__setitem__(0, False))
        self.assert_failed(fixture, "query visibility")
        fixture = self.fixture()
        fixture.request["frame_window"][0] = False
        fixture.save_request()
        self.assert_failed(fixture, "integer frame_window")

    def test_unavailable_final_rng_is_never_promoted_to_a_passed_check(self):
        fixture = self.fixture()
        for case in fixture.cases.values():
            del case["result"]["final_noise_rng"]
        fixture.refresh()
        fixture.request["require_final_rng"] = True
        fixture.save_request()
        self.assert_failed(fixture, "final_noise_rng")
        fixture.request["require_final_rng"] = False
        fixture.save_request()
        result = fixture.run()
        self.assertEqual(result["status"], "passed", result)
        self.assertEqual(result["evidence_kind"], "synthetic_fixture")
        self.assertFalse(result["pairs"][0]["final_generator_state_compared"])
        self.assertIn("final_noise_rng", result["missing_evidence"][0])

    def test_extra_final_private_draw_fails_while_all_earlier_artifacts_remain_identical(self):
        fixture = self.fixture()
        before = fixture.run()
        self.assertEqual(before["status"], "passed", before)
        case = fixture.cases["native_noop"]
        saved = copy.deepcopy({key: value for key, value in case["result"].items() if key != "final_noise_rng"})
        torch.rand(1, generator=case["generators"][0])
        case["result"]["final_noise_rng"] = [
            final_noise_rng_record(generator, rank=rank, binding_id=case["result"]["input_bundle"]["binding_id"])
            for rank, generator in enumerate(case["generators"])
        ]
        self.assertEqual(saved, {key: value for key, value in case["result"].items() if key != "final_noise_rng"})
        fixture.refresh()
        result = self.assert_failed(fixture)
        self.assertIsNone(result["error"])
        pair = result["pairs"][0]
        self.assertTrue(pair["final_generator_state_compared"])
        self.assertFalse(pair["checks"]["final_noise_rng_bitwise_equal"])
        self.assertTrue(all(value for key, value in pair["checks"].items() if key != "final_noise_rng_bitwise_equal"))
        self.assertFalse(pair["final_noise_rng"][0]["bitwise_equal"])
        self.assertTrue(pair["final_noise_rng"][1]["bitwise_equal"])
        self.assertEqual(pair["pairing_sha256"], before["pairs"][0]["pairing_sha256"])

    def test_final_rng_parser_rechecks_rank_shape_hex_hash_and_binding(self):
        generator = torch.Generator().manual_seed(123)
        before = generator.get_state().clone()
        rows = [final_noise_rng_record(generator, rank=rank, binding_id="a" * 64) for rank in range(2)]
        budget = compare._Budget(self.root, {"host_bytes": 8 << 20, "metadata_bytes": 1 << 20})
        arguments = {"binding_id": "a" * 64, "world_size": 2, "required": True, "budget": budget}
        states = compare._final_noise_rng(rows, **arguments)
        self.assertEqual(states, (before.numpy().tobytes(), before.numpy().tobytes()))
        self.assertTrue(torch.equal(generator.get_state(), before))
        changes = {
            "missing-rank": lambda value: value.pop(),
            "duplicate-rank": lambda value: value[1].__setitem__("rank", 0),
            "rank-bool": lambda value: value[0].__setitem__("rank", False),
            "rank-order": lambda value: value.reverse(),
            "dtype": lambda value: value[0].__setitem__("dtype", "float32"),
            "shape-empty": lambda value: value[0].__setitem__("shape", [0]),
            "shape-bool": lambda value: value[0].__setitem__("shape", [True]),
            "shape-over-cap": lambda value: value[0].__setitem__("shape", [65537]),
            "shape-rank": lambda value: value[0].__setitem__("shape", [1, len(before)]),
            "truncated-hex": lambda value: value[0].__setitem__("state_hex", value[0]["state_hex"][:-2]),
            "uppercase-hex": lambda value: value[0].__setitem__("state_hex", "AA" + value[0]["state_hex"][2:]),
            "invalid-hex": lambda value: value[0].__setitem__("state_hex", "gg" + value[0]["state_hex"][2:]),
            "raw-hash": lambda value: value[0].__setitem__("sha256", "b" * 64),
            "input-binding": lambda value: value[0].__setitem__("binding_id", "b" * 64),
        }
        for name, change in changes.items():
            with self.subTest(change=name):
                changed = copy.deepcopy(rows)
                change(changed)
                with self.assertRaises(compare.ComparisonError):
                    compare._final_noise_rng(changed, **arguments)
        with self.assertRaisesRegex(compare.ComparisonError, "requires actual final_noise_rng"):
            compare._final_noise_rng(None, **arguments)
        self.assertIsNone(compare._final_noise_rng(None, **{**arguments, "required": False}))

    def test_one_sided_final_rng_coverage_is_a_failure_not_a_silently_dropped_check(self):
        fixture = self.fixture()
        del fixture.cases["native_noop"]["result"]["final_noise_rng"]
        fixture.refresh()
        self.assert_failed(fixture, "final_noise_rng coverage differs")

    def test_fresh_comparisons_cannot_reuse_resident_process_identity(self):
        fixture = self.fixture("request_isolation")
        resident = fixture.epochs[0]
        fresh = fixture.epochs[1]
        for rank in range(2):
            row = json.loads(Path(resident["generation"]["worker_execution_records"][rank]["path"]).read_text())
            row["plan_identity"] = fresh["generation"]["batch_id"]
            fresh["generation"]["worker_execution_records"][rank] = write_json(
                fresh["directory"] / f"worker{rank}.json", row,
            )
        fixture.refresh()
        self.assert_failed(fixture, "worker process identities")

    def test_cli_example_and_real_fixture_comparison(self):
        fixture = self.fixture()
        script = str(Path(compare.__file__).resolve())
        template = subprocess.run([sys.executable, script, "--example"], check=True, capture_output=True, text=True)
        self.assertEqual(json.loads(template.stdout)["schema"], compare.REQUEST_SCHEMA)
        output = self.root / "cli-receipt.json"
        command = [sys.executable, script, "--request", fixture.request_ref["path"],
                   "--request-sha256", fixture.request_ref["sha256"], "--output", str(output)]
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
        self.assertEqual(json.loads(completed.stdout)["status"], "passed")
        self.assertFalse(json.loads(output.read_text())["g3_qualification"])
        repeated = subprocess.run(command, check=False, capture_output=True, text=True)
        self.assertEqual(repeated.returncode, 2)

    def test_causvid_delegates_exact_four_references_to_existing_comparator(self):
        root = self.root / "causvid"
        root.mkdir()
        expected = {**EXPECTED, "cell_id": "causvid480"}
        receipt = {
            "schema": compare.FIXTURE_RECEIPT_SCHEMA, "protocol_id": expected["protocol_id"],
            "source_closure": {"closure_id": expected["launch_closure_id"]}, "phase": "native_preflight",
            "fixture": True, "native_execution": False,
        }
        receipt["receipt_id"] = identity(receipt)
        sources = {}
        for name in compare.CAUSVID_ROLES:
            sources[name] = write_json(root / f"{name}.json", {
                "fixture": True, "native_execution": False, "request": {"cell_id": "causvid480", "chunks": 2},
                "provenance": {name: expected[name] for name in ("protocol_id", "launch_closure_id")},
            })
        request = {
            "schema": compare.REQUEST_SCHEMA, "model": "causvid", "comparison_kind": "raw_native_reference",
            "evidence_kind": "synthetic_fixture", "artifact_root": str(root), "expected": expected,
            "limits": {"host_bytes": 64 << 20, "metadata_bytes": 1 << 20},
            "frame_window": [0, 21], "launch_receipt": write_json(root / "launch.json", receipt),
            "causvid_inputs": sources,
        }
        reference = write_json(root / "request.json", request)
        with mock.patch("examples.causvid_native_reference.compare_reference_outputs", return_value={
            "fixture": True, "native_execution": False, "comparisons": {"rgb_bitwise_equal": True},
            "all_compared_outputs_equal": True,
        }) as delegated:
            result = compare.compare_preflight(reference["path"], request_sha256=reference["sha256"])
        self.assertEqual(result["status"], "passed", result)
        delegated.assert_called_once()
        self.assertEqual(delegated.call_args.args, (str(root),))
        self.assertLess(delegated.call_args.kwargs["max_host_bytes"], request["limits"]["host_bytes"])
        for name in compare.CAUSVID_ROLES:
            self.assertEqual(delegated.call_args.kwargs[name],
                             {key: sources[name][key] for key in ("path", "sha256")})


if __name__ == "__main__":
    unittest.main()
