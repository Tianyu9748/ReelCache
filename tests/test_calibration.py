import dataclasses
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from reelcache.calibration import (
    Observation, configuration_fingerprint, fit_profile,
    load_registry, package_fingerprint, save_registry,
)
from reelcache.errors import ConfigurationError, IntegrityError, StateError
from tests.test_policy import profile, select


def observation(template, case, split, **changes):
    values = dict(
        profile_id=template.profile_id, case_id=case, prompt_id=case,
        split=split, configuration_sha256=configuration_fingerprint(template),
        source_sha256=hashlib.sha256(case.encode()).hexdigest(),
        reference_id="per_channel", ssim=(0.98, 0.96),
        reference_ssim=(0.985, 0.965), raw_bytes=1000, stored_bytes=200,
        kv_relative_error=0.01, encode_seconds=0.2,
        features={"K_residual_max": 0.02}, native_validated=True,
        held_out_verified=split == "validation",
    )
    values.update(changes)
    return Observation(**values)


class CalibrationTests(unittest.TestCase):
    def test_fit_uses_only_calibration_costs(self):
        template = profile(evidence=None)
        rows = [
            observation(template, f"cal{i}", "calibration", stored_bytes=180 + i)
            for i in range(6)
        ] + [
            observation(template, f"val{i}-s{s}", "validation",
                        prompt_id=f"val{i}", stored_bytes=999)
            for i in range(4) for s in range(2)
        ]
        fitted = fit_profile(template, rows)
        self.assertEqual(fitted.measured_stored_bytes, 185)
        self.assertAlmostEqual(fitted.evidence.mean_ssim, 0.97)
        self.assertEqual(select([fitted]).profile_id, fitted.profile_id)

    def test_development_only_stays_unqualified(self):
        template = profile(evidence=None)
        fitted = fit_profile(
            template, [observation(template, "used", "development")]
        )
        self.assertIsNone(fitted.evidence)
        self.assertEqual(select([fitted]).profile_id, "RAW")

    def test_split_labels_alone_do_not_prove_held_out(self):
        template = profile(evidence=None)
        rows = [observation(template, "cal", "calibration"),
                observation(template, "val", "validation", held_out_verified=False)]
        fitted = fit_profile(template, rows)
        self.assertFalse(fitted.evidence.held_out)

    def test_leakage_duplicates_and_config_changes_fail(self):
        template = profile(evidence=None)
        cal = observation(template, "cal", "calibration")
        val = observation(template, "val", "validation")
        for rows in (
            [cal, cal],
            [val],
            [cal, dataclasses.replace(val, prompt_id="cal")],
            [cal, dataclasses.replace(val, configuration_sha256="0" * 64)],
            [cal, dataclasses.replace(val, history_chunks=9)],
        ):
            with self.assertRaises(ConfigurationError):
                fit_profile(template, rows)

    def test_validation_does_not_widen_feature_support(self):
        template = profile(evidence=None, feature_bounds={})
        cal = observation(template, "cal", "calibration")
        val = observation(template, "val", "validation",
                          features={"K_residual_max": 0.5})
        with self.assertRaisesRegex(ConfigurationError, "out-of-support"):
            fit_profile(template, [cal, val])

    def test_immutable_registry_and_tamper_detection(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profiles.json"
            original = profile()
            digest = save_registry(path, [original])
            self.assertEqual(save_registry(path, [original]), digest)
            loaded = load_registry(path, expected_sha256=digest)
            self.assertEqual(select(loaded).to_dict(), select([original]).to_dict())
            with self.assertRaises(StateError):
                save_registry(path, [profile("another")])
            data = json.loads(path.read_text())
            data["profiles"][0]["measured_stored_bytes"] = 1
            path.write_text(json.dumps(data))
            with self.assertRaises(IntegrityError):
                load_registry(path)

    def test_source_fingerprint_changes_and_includes_extra_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "codec.py").write_text("VERSION = 1\n")
            original = package_fingerprint(package_root=root)
            self.assertNotEqual(original, package_fingerprint(
                package_root=root, extra_identity={"ffmpeg": "test-version"},
            ))
            (root / "codec.py").write_text("VERSION = 2\n")
            self.assertNotEqual(original, package_fingerprint(package_root=root))

    def test_invalid_measurements_rejected(self):
        template = profile()
        for changes in (
            {"ssim": ()}, {"reference_ssim": (0.99,)},
            {"ssim": (float("nan"), 0.9)}, {"stored_bytes": -1},
            {"configuration_sha256": "invalid"},
            {"native_validated": "true"},
        ):
            with self.assertRaises(ConfigurationError):
                observation(template, "bad", "development", **changes)


if __name__ == "__main__":
    unittest.main()
