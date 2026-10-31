"""CPU-only opaque-file provenance tests; never model checkpoints."""

import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from reelcache.adapters import magi_provenance as provenance
from reelcache.adapters import magi_native as native
from tests import test_magi_vbench as fixtures


class MagiProvenanceTests(unittest.TestCase):
    setUp = fixtures.MagiVBenchTests.setUp
    tearDown = fixtures.MagiVBenchTests.tearDown
    _write = fixtures.MagiVBenchTests._write
    _prepare_fixture = fixtures.MagiVBenchTests._prepare_fixture

    def test_full_weight_binding_reuses_evidence_without_shard_reads(self):
        args = self._prepare_fixture()
        original = native._sha
        opened = []

        def permitted(path):
            path = Path(path)
            opened.append(path.name)
            if path.name == "fixture.safetensors":
                self.fail("Per-case binding reread a fully verified weight shard")
            return original(path)
        with patch.object(native, "_sha", side_effect=permitted):
            binding = provenance.verify_weights(self.protocol["weight_identity"], args.weights)
        self.assertFalse(binding["shards_rehashed_per_case"])
        self.assertEqual(opened, ["weight_identity.json", "model.safetensors.index.json"])
        self.assertTrue(binding["magi_model_content_sha256"])

    def test_same_size_weight_rewrite_invalidates_recorded_original(self):
        args = self._prepare_fixture()
        path = args.weights / "fixture.safetensors"
        original = path.read_bytes()
        path.write_bytes(original)
        with self.assertRaisesRegex(ValueError, "stat guard changed"):
            provenance.verify_weights(self.protocol["weight_identity"], args.weights)

    def test_manifest_file_and_canonical_integrity_both_fail_closed(self):
        args = self._prepare_fixture()
        binding = copy.deepcopy(self.protocol["weight_identity"])
        path = Path(binding["path"])
        manifest = native._read(path)
        manifest["status"] = "not_passed"
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "manifest bytes changed"):
            provenance.verify_weights(binding, args.weights)
        binding["sha256"] = native._sha(path)
        with self.assertRaisesRegex(ValueError, "integrity/status/schema"):
            provenance.verify_weights(binding, args.weights)

    def test_canonical_but_incomplete_weight_coverage_is_rejected(self):
        args = self._prepare_fixture()
        binding = copy.deepcopy(self.protocol["weight_identity"])
        path = Path(binding["path"])
        manifest = native._read(path)
        manifest["files"] = [row for row in manifest["files"] if row["role"] == "index"]
        model = manifest["content_manifest"]["models"]["magi"]
        model["files"] = [row for row in model["files"] if row["role"] == "index"]
        manifest["model_content_sha256"]["magi"] = provenance.canonical_digest(model)
        manifest["content_manifest_sha256"] = provenance.canonical_digest(manifest["content_manifest"])
        manifest.pop("manifest_integrity_sha256")
        manifest["manifest_integrity_sha256"] = provenance.canonical_digest(manifest)
        path.write_text(json.dumps(manifest))
        binding.update(sha256=native._sha(path), content_manifest_sha256=manifest["content_manifest_sha256"])
        with self.assertRaisesRegex(ValueError, "EVERY referenced"):
            provenance.verify_weights(binding, args.weights)

    def test_auxiliary_captures_full_T5_VAE_source_then_uses_stat_guards(self):
        args = self._prepare_fixture()
        binding = self.protocol["magi_auxiliary_identity"]
        full = native._read(binding["path"])
        self.assertTrue(all(row["full_content_hashed"] for row in full["files"]))
        self.assertFalse(full["model_or_GPU_loaded"])
        observed = []
        original = native._sha

        def no_large_rehash(path):
            observed.append(Path(path).name)
            if str(path).endswith((".bin", ".safetensors")):
                self.fail("Per-case auxiliary verification reread a large checkpoint")
            return original(path)
        with patch.object(native, "_sha", side_effect=no_large_rehash):
            result = provenance.verify_auxiliary(binding, t5=args.t5, vae=args.vae, source=args.native_source)
        self.assertTrue(result["full_content_hashed_once_per_study"])
        self.assertEqual(observed, ["auxiliary_identity.json"])
        (args.t5 / "t5-v1_1-xxl/spiece.model").write_bytes(b"altered tokenizer")
        with self.assertRaisesRegex(ValueError, "stat guard changed"):
            provenance.verify_auxiliary(binding, t5=args.t5, vae=args.vae, source=args.native_source)

    def test_auxiliary_VAE_must_match_qualified_reference(self):
        args = self._prepare_fixture()
        (args.vae / "diffusion_pytorch_model.safetensors").write_bytes(b"different VAE")
        output = self.root / "new_auxiliary_identity.json"
        with self.assertRaisesRegex(ValueError, "differs from qualified native VAE"):
            provenance.capture_auxiliary(
                output=output, t5=args.t5, vae=args.vae, source=args.native_source,
                qualified_plan=args.qualified_plan, max_seconds=60,
            )
        self.assertFalse(output.exists())

    def test_auxiliary_cannot_overwrite_or_ignore_a_missing_component(self):
        args = self._prepare_fixture()
        with self.assertRaises(FileExistsError):
            provenance.capture_auxiliary(
                output=self.protocol["magi_auxiliary_identity"]["path"], t5=args.t5, vae=args.vae,
                source=args.native_source, qualified_plan=args.qualified_plan,
            )
        (args.native_source / "inference/extra.py").write_text("value = 1\n")
        with self.assertRaisesRegex(ValueError, "all referenced"):
            provenance.verify_auxiliary(
                self.protocol["magi_auxiliary_identity"], t5=args.t5, vae=args.vae, source=args.native_source,
            )


if __name__ == "__main__":
    unittest.main()
