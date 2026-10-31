import dataclasses
import hashlib
import json
import unittest

from reelcache.errors import CapacityError, ConfigurationError, StateError
from reelcache.policy import Profile, QualityBudget, QualityEvidence, select_profile
from reelcache.types import CodecSpec, MatrixKey, ModelIdentity


MODEL = ModelIdentity("unit-toy", "a" * 64, 2, 8, "canonical", "float32")
IMPLEMENTATION = "b" * 64


def evidence(**changes):
    values = dict(
        reference_id="per_channel", mean_ssim=0.98,
        minimum_frame_ssim=0.96, mean_ssim_drop=0.002,
        calibration_prompts=tuple(f"cal{i}" for i in range(6)),
        validation_prompts=tuple(f"val{i}" for i in range(4)),
        validation_cases=tuple(f"val{i}-seed{s}" for i in range(4) for s in range(2)),
        source_sha256=tuple(hashlib.sha256(f"case{i}".encode()).hexdigest() for i in range(8)),
        held_out=True, native_validated=True,
    )
    values.update(changes)
    return QualityEvidence(**values)


def profile(name="qualified", **changes):
    values = dict(
        profile_id=name, model=MODEL,
        specs={MatrixKey(layer, target): CodecSpec(kind="svd_hevc", rank=4)
               for layer in range(2) for target in ("K", "V")},
        geometry=(1, 2, 2), measured_stored_bytes=200,
        predicted_distortion=0.01, encode_seconds=0.2,
        history_min=1, history_max=3,
        feature_bounds={"K_residual_max": (0.0, 0.05)},
        evidence=evidence(), implementation_fingerprint=IMPLEMENTATION,
    )
    values.update(changes)
    return Profile(**values)


def select(profiles=(), **changes):
    values = dict(
        model=MODEL, geometry=(1, 2, 2), history_chunks=1,
        features={"K_residual_max": 0.02}, raw_bytes=1000,
        implementation_fingerprint=IMPLEMENTATION,
        budget=QualityBudget(
            min_mean_ssim=0.95, min_frame_ssim=0.90, max_mean_ssim_drop=0.01,
        ),
    )
    values.update(changes)
    return select_profile(profiles, **values)


class PolicyTests(unittest.TestCase):
    def test_no_service_quality_threshold_is_chosen_by_default(self):
        self.assertEqual(select([profile()], budget=None).profile_id, "RAW")
        with self.assertRaises(ConfigurationError):
            QualityBudget(min_mean_ssim=0.95)

    def test_unqualified_default_is_raw(self):
        result = select([profile("pilot", evidence=None)])
        self.assertEqual(result.profile_id, "RAW")
        self.assertTrue(all(spec.kind == "raw" for spec in result.specs.values()))
        self.assertFalse(result.experimental)
        self.assertFalse(result.to_dict()["per_request_SSIM_guarantee"])

    def test_absolute_quality_not_just_small_drop(self):
        bad = profile(evidence=evidence(
            mean_ssim=0.803, minimum_frame_ssim=0.62, mean_ssim_drop=0.0001,
        ))
        self.assertEqual(select([bad]).profile_id, "RAW")

    def test_two_prompt_pilot_cannot_qualify(self):
        small = evidence(
            calibration_prompts=("old0", "old1"),
            validation_prompts=("new0", "new1"),
            validation_cases=("new0-s0", "new1-s0"),
        )
        self.assertEqual(select([profile(evidence=small)]).profile_id, "RAW")

    def test_joint_vector_selection_uses_actual_measured_costs(self):
        mixed = dict(profile().specs)
        mixed[MatrixKey(0, "K")] = CodecSpec(
            kind="svd_hevc", rank=2, normalization="global",
        )
        chosen = profile("joint", specs=mixed, measured_stored_bytes=120)
        result = select([profile(), chosen])
        self.assertEqual(result.profile_id, "joint")
        self.assertEqual(result.specs[MatrixKey(0, "K")].rank, 2)
        self.assertEqual(result.specs[MatrixKey(0, "K")].normalization, "global")
        self.assertTrue(result.evidence_qualified)

    def test_no_unmeasured_layerwise_mix_is_constructed(self):
        first = profile("A", measured_stored_bytes=200)
        second = profile("B", measured_stored_bytes=200)
        result = select([second, first])
        self.assertEqual(result.profile_id, "A")
        self.assertEqual(dict(result.specs), dict(first.specs))

    def test_out_of_support_and_missing_feature_abstain(self):
        for changes in (
            {"history_chunks": 4}, {"geometry": (2, 2, 2)},
            {"features": {}}, {"features": {"K_residual_max": 0.051}},
            {"model": dataclasses.replace(MODEL, fingerprint="c" * 64)},
            {"implementation_fingerprint": "c" * 64},
            {"implementation_fingerprint": None},
        ):
            with self.subTest(changes=changes):
                self.assertEqual(select([profile()], **changes).profile_id, "RAW")

    def test_explicit_experiment_does_not_claim_qualification(self):
        candidate = profile("pilot", evidence=None)
        with self.assertRaises(StateError):
            select([candidate], explicit_profile="pilot")
        result = select([candidate], explicit_profile="pilot", allow_experimental=True)
        self.assertEqual(result.profile_id, "pilot")
        self.assertTrue(result.experimental)
        self.assertFalse(result.evidence_qualified)
        with self.assertRaises(ConfigurationError):
            select([candidate], allow_experimental=True)

    def test_capacity_and_deadline_are_not_silent_fallbacks(self):
        with self.assertRaises(CapacityError):
            select([], budget=QualityBudget(max_predicted_bytes=999))
        with self.assertRaises(CapacityError):
            select([], budget=QualityBudget(max_encode_seconds=1))
        result = select([profile()], budget=QualityBudget(max_predicted_distortion=0.005))
        self.assertEqual(result.profile_id, "RAW")
        with self.assertRaises(StateError):
            select([profile()], explicit_profile="qualified",
                   budget=QualityBudget(max_predicted_bytes=199))

    def test_raw_is_preferred_to_negative_compression(self):
        self.assertEqual(
            select([profile(measured_stored_bytes=1001)]).profile_id, "RAW"
        )

    def test_registry_roundtrip_and_frozen_inputs(self):
        original = profile()
        reloaded = Profile.from_dict(json.loads(json.dumps(original.to_dict())))
        self.assertEqual(select([original]).to_dict(), select([reloaded]).to_dict())
        with self.assertRaises(TypeError):
            original.specs[MatrixKey(0, "K")] = CodecSpec()
        with self.assertRaises(TypeError):
            original.feature_bounds["basis_age"] = (0, 1)

    def test_invalid_or_leaky_evidence_is_rejected(self):
        for changes in (
            {"calibration_prompts": ("val0",)},
            {"validation_cases": ("duplicate", "duplicate")},
            {"mean_ssim": float("nan")}, {"native_validated": "yes"},
            {"source_sha256": ("not-a-hash",)},
        ):
            with self.subTest(changes=changes):
                with self.assertRaises(ConfigurationError):
                    evidence(**changes)
        for changes in (
            {"specs": {}}, {"history_max": 0},
            {"feature_bounds": {"prompt_id": (0, 1)}},
            {"measured_stored_bytes": 0},
        ):
            with self.assertRaises(ConfigurationError):
                profile(**changes)

    def test_selection_rejects_bad_features_and_duplicates(self):
        with self.assertRaises(ConfigurationError):
            select([profile(), profile()])
        with self.assertRaises(ConfigurationError):
            select(features={"K_residual_max": float("nan")})
        with self.assertRaises(ConfigurationError):
            select(features={"prompt_id": 1})
        with self.assertRaises(ConfigurationError):
            select(explicit_profile="absent")


if __name__ == "__main__":
    unittest.main()
