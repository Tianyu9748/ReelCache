"""Default-off HEVC inter prediction: spec identity, x265 contract and real bitstreams.

Every stream assertion parses real libx265 output with FFmpeg's ``trace_headers``
bitstream filter. The filter is a hard dependency here: ``_hevc.trace_headers``
raises UnsupportedError when the selected FFmpeg lacks it, so these tests fail
loudly instead of skipping (the bundled imageio-ffmpeg 7.0.2 build provides it).
"""

import dataclasses
import hashlib
import json
import subprocess
import sys
from pathlib import Path
import unittest
from unittest import mock

import numpy as np
import torch

from reelcache import _hevc
from reelcache.codecs import HEVCProvider, decode_tensor, encode_tensor
from reelcache.errors import ConfigurationError, IntegrityError
from reelcache.policy import Profile
from reelcache.types import CodecSpec, EncodedTensor, MatrixKey, ModelIdentity, TensorContext
from tests.core_support import CoreWorkspaceTest, metadata
from tests.test_adaptive_manager import rank_policy

ROOT = Path(__file__).resolve().parents[1]
# Runtime chunk lengths: single picture, CausVid (3 latent frames), LongLive 2.0 (8).
CHUNK_FRAMES = (1, 3, 8)


def identity(layers=1, channels=12):
    return ModelIdentity("inter-control", "fixture-v2", layers, channels, "tokens_channels", "float32")


def context(frames, target="K", *, height=6, width=10, model=None, layer=0):
    return TensorContext(
        model or identity(), MatrixKey(layer, target), 0, frames, height, width,
        "pre_rope" if target == "K" else "none",
    )


def moving_matrix(frames, channels=12, *, height=6, width=10, seed=0):
    """Coefficients that drift smoothly over frames, so inter coding has real motion."""
    generator = torch.Generator().manual_seed(seed)
    y = torch.linspace(-1, 1, height)[:, None]
    x = torch.linspace(-1, 1, width)[None, :]
    fields = []
    for frame in range(frames):
        shift = 0.15 * frame
        fields.append(torch.stack([
            torch.sin(3 * (x + shift) + channel) * torch.cos(2 * y - 0.5 * channel)
            for channel in range(4)
        ], dim=-1))
    tokens = torch.stack(fields).reshape(-1, 4)
    return tokens @ torch.randn(4, channels, generator=generator) + 0.01 * torch.randn(
        frames * height * width, channels, generator=generator,
    )


def moving_pictures(frames, planes, *, height=64, width=96, seed=0):
    rng = np.random.default_rng(seed)
    base = (rng.random((height, width)) * 255).astype(np.uint8)
    return np.stack([np.stack([np.roll(base, 2 * frame, axis=1)] * planes) for frame in range(frames)])


class CodecSpecInterFieldTests(unittest.TestCase):
    def test_default_is_explicit_all_intra_for_hevc_only(self):
        self.assertIs(CodecSpec().inter_prediction, False)
        self.assertIs(CodecSpec("svd_hevc", 4).inter_prediction, False)
        self.assertEqual(CodecSpec("svd_hevc", 4).to_dict()["inter_prediction"], False)
        self.assertEqual(CodecSpec("svd_hevc", 4, inter_prediction=True).to_dict()["inter_prediction"], True)
        for spec in (CodecSpec(), CodecSpec("svd", 4)):
            self.assertNotIn("inter_prediction", spec.to_dict())
            self.assertEqual(CodecSpec.from_dict(spec.to_dict()), spec)
        for kind, rank in (("raw", None), ("svd", 2)):
            with self.assertRaises(ConfigurationError):
                CodecSpec(kind, rank, inter_prediction=True)
            self.assertIs(CodecSpec(kind, rank, inter_prediction=None).inter_prediction, False)
        for value in (1, 0, "false", 1.0):
            with self.assertRaises(ConfigurationError):
                CodecSpec("svd_hevc", 2, inter_prediction=value)

    def test_serialized_versions_are_never_reinterpreted(self):
        current = CodecSpec("svd_hevc", 3, inter_prediction=False)
        self.assertEqual(CodecSpec.from_dict(current.to_dict()), current)
        legacy_record = {key: value for key, value in current.to_dict().items() if key != "inter_prediction"}
        legacy = CodecSpec.from_dict(legacy_record)
        self.assertIsNone(legacy.inter_prediction)
        self.assertTrue(legacy.legacy_gop)
        self.assertFalse(current.legacy_gop)
        self.assertNotEqual(legacy, current)
        self.assertEqual(legacy.to_dict(), legacy_record)
        for value in (None, 0, "true"):
            with self.assertRaises(ConfigurationError):
                CodecSpec.from_dict({**current.to_dict(), "inter_prediction": value})
        with self.assertRaises(ConfigurationError):
            CodecSpec.from_dict({"kind": "raw", "inter_prediction": True})
        self.assertEqual(dataclasses.replace(CodecSpec("svd_hevc", 2, inter_prediction=True), rank=3).inter_prediction, True)

    def test_legacy_specs_are_decode_only(self):
        legacy = CodecSpec.from_dict({"kind": "svd_hevc", "rank": 2})
        with self.assertRaisesRegex(ConfigurationError, "decode-only"):
            encode_tensor(moving_matrix(3), context(3), legacy)
        with self.assertRaises(ConfigurationError):
            _hevc.encoder_record(None)


class EncoderContractTests(unittest.TestCase):
    def test_records_pin_gop_and_qp_controls_explicitly(self):
        intra, inter = _hevc.encoder_record(False), _hevc.encoder_record(True)
        for record in (intra, inter):
            self.assertEqual(record["schema"], "reelcache.hevc_encoder.v2")
            self.assertEqual((record["preset"], record["qp"]), ("ultrafast", 8))
            for name, value in (("ipratio", 1), ("pbratio", 1), ("info", 0), ("b-adapt", 0), ("scenecut", 0),
                                ("open-gop", 0)):
                self.assertEqual(record["x265_params"][name], value)
            self.assertNotIn("min-cu-size", record["x265_params"])
            self.assertNotIn("ctu", record["x265_params"])
            json.dumps(record, allow_nan=False)
        self.assertEqual(intra["picture_structure"], "all_intra_idr")
        self.assertEqual(
            {name: intra["x265_params"][name] for name in ("keyint", "min-keyint", "bframes")},
            {"keyint": 1, "min-keyint": 1, "bframes": 0},
        )
        self.assertEqual(
            {name: inter["x265_params"][name] for name in ("keyint", "bframes", "ref", "b-pyramid")},
            {"keyint": 250, "bframes": 3, "ref": 1, "b-pyramid": 1},
        )
        text = _hevc.x265_parameter_string(False, 4, 3)
        self.assertTrue(text.startswith("qp=8:ipratio=1:pbratio=1:info=0:keyint=1:min-keyint=1:bframes=0"))
        self.assertTrue(text.endswith("log-level=error:pools=4:frame-threads=3"))
        self.assertIn("keyint=250:min-keyint=25:bframes=3", _hevc.x265_parameter_string(True, 1, 8))
        self.assertIsNone(_hevc.LEGACY_ENCODER["inter_prediction"])


class RealBitstreamTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.binary = _hevc.capabilities()["binary"]
        if "trace_headers" not in _hevc._bitstream_filters(cls.binary):
            raise AssertionError(f"{cls.binary} lacks trace_headers; the inter-control tests require it")

    def encode(self, frames, pix_fmt, inter, threads=1, seed=0):
        planes = 3 if pix_fmt == "yuv444p" else 1
        return _hevc._encode_segment(moving_pictures(frames, planes, seed=seed), pix_fmt, self.binary, threads, inter)

    def test_all_intra_streams_contain_only_idr_i_slices_at_qp8(self):
        for pix_fmt in _hevc.PIXEL_FORMATS:
            for frames in CHUNK_FRAMES:
                with self.subTest(pix_fmt=pix_fmt, frames=frames):
                    stream = self.encode(frames, pix_fmt, False)
                    trace = _hevc.trace_headers(stream)
                    self.assertEqual(trace["slice_type_counts"], {"I": frames, "P": 0, "B": 0})
                    self.assertEqual(len(trace["pictures"]), frames)
                    self.assertTrue(all(picture["nal_unit_type"] in (19, 20) for picture in trace["pictures"]))
                    self.assertEqual(trace["slice_qps"], [8])
                    self.assertEqual(trace["chroma_format_idc"], [3 if pix_fmt == "yuv444p" else 1])
                    self.assertEqual(trace["general_intra_constraint_flag"], [1])
                    units = _hevc.all_intra_access_units(stream, frames)
                    self.assertTrue(all(unit["self_contained"] for unit in units))

    def test_inter_streams_contain_real_p_or_b_slices(self):
        for pix_fmt in _hevc.PIXEL_FORMATS:
            for frames in (3, 8):
                with self.subTest(pix_fmt=pix_fmt, frames=frames):
                    stream = self.encode(frames, pix_fmt, True)
                    trace = _hevc.trace_headers(stream)
                    counts = trace["slice_type_counts"]
                    self.assertEqual(counts["I"], 1)
                    self.assertGreaterEqual(counts["P"] + counts["B"], 1)
                    self.assertEqual(sum(counts.values()), frames)
                    self.assertEqual(trace["pictures"][0]["nal_unit_type"], 20)
                    self.assertTrue(any(picture["nal_unit_type"] not in range(16, 22)
                                        for picture in trace["pictures"]))
                    self.assertEqual(trace["slice_qps"], [8])
                    with self.assertRaises(IntegrityError):
                        _hevc.all_intra_access_units(stream, frames)
            eight = _hevc.trace_headers(self.encode(8, pix_fmt, True))["slice_type_counts"]
            self.assertGreater(eight["P"], 0)
            self.assertGreater(eight["B"], 0)
        single = _hevc.trace_headers(self.encode(1, "yuv444p", True))
        self.assertEqual(single["slice_type_counts"], {"I": 1, "P": 0, "B": 0})

    def test_streams_are_deterministic_and_thread_independent(self):
        for pix_fmt in _hevc.PIXEL_FORMATS:
            for inter in (False, True):
                with self.subTest(pix_fmt=pix_fmt, inter=inter):
                    reference = self.encode(3, pix_fmt, inter, threads=1)
                    self.assertEqual(self.encode(3, pix_fmt, inter, threads=1), reference)
                    self.assertEqual(self.encode(3, pix_fmt, inter, threads=4), reference)

    def test_trace_headers_fails_loudly_without_the_filter(self):
        stream = self.encode(1, "yuv420p", False)
        with mock.patch("reelcache._hevc._bitstream_filters", return_value=frozenset()):
            with self.assertRaisesRegex(Exception, "trace_headers"):
                _hevc.trace_headers(stream)


class PacketInterControlTests(unittest.TestCase):
    def test_tensor_packets_record_and_honour_the_gop_choice(self):
        for pix_fmt in _hevc.PIXEL_FORMATS:
            for frames in CHUNK_FRAMES:
                matrix = moving_matrix(frames)
                for inter in (False, True):
                    with self.subTest(pix_fmt=pix_fmt, frames=frames, inter=inter):
                        spec = CodecSpec("svd_hevc", 6, pix_fmt=pix_fmt, inter_prediction=inter)
                        packet = encode_tensor(matrix, context(frames), spec, validate_hevc=True)
                        self.assertEqual(packet.metadata["schema"], "reelcache.tensor.v2")
                        self.assertIs(packet.metadata["codec"]["inter_prediction"], inter)
                        self.assertEqual(packet.metadata["hevc"], _hevc.encoder_record(inter))
                        self.assertIs(packet.metrics["hevc_inter_prediction"], inter)
                        counts = _hevc.trace_headers(packet.payloads["coefficients.000.hevc"])["slice_type_counts"]
                        if inter and frames > 1:
                            self.assertGreaterEqual(counts["P"] + counts["B"], 1)
                        else:
                            self.assertEqual(counts, {"I": frames, "P": 0, "B": 0})
                        decoded = decode_tensor(packet, verify_packet=True)
                        self.assertLess(((decoded - matrix).norm() / matrix.norm()).item(), 0.05)

    def test_default_spec_produces_all_intra_and_is_deterministic(self):
        matrix = moving_matrix(3, seed=4)
        first = encode_tensor(matrix, context(3), CodecSpec("svd_hevc", 5))
        second = encode_tensor(matrix, context(3), CodecSpec("svd_hevc", 5), hevc_threads=4)
        self.assertEqual(first.payloads, second.payloads)
        self.assertEqual(first.metadata, second.metadata)
        trace = _hevc.trace_headers(first.payloads["coefficients.000.hevc"])
        self.assertEqual(trace["slice_type_counts"], {"I": 3, "P": 0, "B": 0})

    def test_verified_decode_rejects_inter_stream_under_all_intra_label(self):
        matrix = moving_matrix(3, seed=2)
        intra = encode_tensor(matrix, context(3), CodecSpec("svd_hevc", 5))
        inter = encode_tensor(matrix, context(3), CodecSpec("svd_hevc", 5, inter_prediction=True))
        forged_metadata = json.loads(json.dumps(intra.metadata))
        stream = inter.payloads["coefficients.000.hevc"]
        forged_metadata["payloads"]["coefficients.000.hevc"].update(
            bytes=len(stream), shape=[len(stream)], sha256=hashlib.sha256(stream).hexdigest(),
        )
        forged = EncodedTensor(forged_metadata, {**intra.payloads, "coefficients.000.hevc": stream}, intra.basis)
        with self.assertRaisesRegex(IntegrityError, "non-IRAP"):
            decode_tensor(forged, verify_packet=True)
        for field, value in (("hevc", _hevc.encoder_record(True)), ("hevc", None)):
            metadata = json.loads(json.dumps(intra.metadata))
            metadata[field] = value
            with self.assertRaises(IntegrityError):
                decode_tensor(EncodedTensor(metadata, intra.payloads, intra.basis), verify_packet=True)
        metadata = json.loads(json.dumps(intra.metadata))
        del metadata["codec"]["inter_prediction"]
        with self.assertRaises(IntegrityError):
            decode_tensor(EncodedTensor(metadata, intra.payloads, intra.basis), verify_packet=True)


class PropagationTests(CoreWorkspaceTest):
    def test_provider_policy_and_profile_identity_carry_the_flag(self):
        matrix = moving_matrix(3, seed=7)
        for inter in (False, True):
            packet = HEVCProvider(x265_threads=1).encode_tensor(
                matrix, context(3), CodecSpec("svd_hevc", 4, inter_prediction=inter),
            )
            self.assertIs(packet.metadata["codec"]["inter_prediction"], inter)
        model = identity()
        policies = {
            inter: rank_policy(model, codec=CodecSpec("svd_hevc", 1, inter_prediction=inter))
            for inter in (False, True)
        }
        self.assertNotEqual(policies[False].fingerprint, policies[True].fingerprint)
        for inter, policy in policies.items():
            self.assertIs(policy.spec_for(MatrixKey(0, "K"), 2).inter_prediction, inter)
            self.assertIs(policy.to_dict()["codec"]["inter_prediction"], inter)
            self.assertEqual(type(policy).from_dict(policy.to_dict()).fingerprint, policy.fingerprint)
        legacy_document = policies[False].to_dict()
        del legacy_document["codec"]["inter_prediction"]
        legacy = type(policies[False]).from_dict(legacy_document)
        self.assertIsNone(legacy.codec.inter_prediction)
        self.assertEqual(legacy.to_dict(), legacy_document)
        specs = {MatrixKey(0, target): CodecSpec("svd_hevc", 2, inter_prediction=True) for target in ("K", "V")}
        profile = Profile("inter-profile", model, specs, (3, 6, 10), 100, 0.1, 1.0)
        restored = Profile.from_dict(profile.to_dict())
        self.assertEqual(restored, profile)
        self.assertIs(restored.specs[MatrixKey(0, "K")].inter_prediction, True)
        self.assertNotEqual(
            Profile.from_dict({**profile.to_dict(), "specs": {
                label: {**spec, "inter_prediction": False} for label, spec in profile.to_dict()["specs"].items()
            }}).to_dict(),
            profile.to_dict(),
        )

    def test_runtime_commit_restore_and_snapshot_record_the_flag(self):
        from reelcache.manager import KVRuntime

        model = identity(layers=1)
        for inter in (False, True):
            runtime = KVRuntime(self.workspace / f"runtime-{inter}", model)
            self.addCleanup(runtime.close)
            runtime.create_branch("main")
            matrices = {MatrixKey(0, target): moving_matrix(3, seed=11) for target in ("K", "V")}
            contexts = {key: context(3, key.target, model=model) for key in matrices}
            spec = CodecSpec("svd_hevc", 4, inter_prediction=inter)
            runtime.seal_chunk("main", "c0", matrices, contexts, spec)
            chunk = runtime.store.get_manifest(runtime.branch_info("main")["chunk_manifests"][0])
            for entry in chunk["matrices"]:
                record = runtime.store.get_manifest(entry["tensor"])
                self.assertEqual(record["spec"], spec.to_dict())
                self.assertEqual(record["encoded"]["metadata"]["hevc"], _hevc.encoder_record(inter))
                stream = runtime.store.get(record["encoded"]["payloads"]["coefficients.000.hevc"])
                counts = _hevc.trace_headers(stream)["slice_type_counts"]
                if inter:
                    self.assertGreaterEqual(counts["P"] + counts["B"], 1)
                else:
                    self.assertEqual(counts, {"I": 3, "P": 0, "B": 0})
            restored = {}
            for key in matrices:
                with runtime.acquire_history("main", key) as lease:
                    self.assertLess(((lease.tensor - matrices[key]).norm() / matrices[key].norm()).item(), 0.05)
                    restored[key] = lease.tensor.clone()
            state = {name: torch.zeros(1) for name in ("rng", "latent", "vae")}
            snapshot = runtime.snapshot("main", metadata=metadata(end=180), tensors=state)
            runtime.fork(snapshot, "child")
            child_head = runtime.store.get_manifest(runtime.branch_info("child")["head_id"])
            self.assertEqual(child_head["chunks"], runtime.store.get_manifest(
                runtime.branch_info("main")["head_id"])["chunks"])
            for key in matrices:
                with runtime.acquire_history("child", key) as lease:
                    self.assertTrue(torch.equal(lease.tensor, restored[key]))

    def test_cli_and_worker_configuration_expose_the_flag(self):
        result = subprocess.run(
            [sys.executable, "-B", str(ROOT / "examples/runtime_demo.py"), "--help"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=600, check=True,
        )
        self.assertIn("--hevc-inter-prediction", result.stdout.decode())
        self.assertIn("--no-hevc-inter-prediction", result.stdout.decode())
        sys.path.insert(0, str(ROOT))
        try:
            from examples.causvid_rank_worker import codec_specs
        finally:
            sys.path.remove(str(ROOT))
        base = {"kind": "svd_hevc", "rank": 2, "normalization": "per_channel", "pix_fmt": "yuv444p",
                "qp": 8, "preset": "ultrafast"}
        default = codec_specs({"K": base, "V": base}, 1)
        self.assertTrue(all(spec.inter_prediction is False for spec in default.values()))
        enabled = codec_specs({"K": {**base, "inter_prediction": True}, "V": base}, 1)
        self.assertIs(enabled[MatrixKey(0, "K")].inter_prediction, True)
        self.assertIs(enabled[MatrixKey(0, "V")].inter_prediction, False)
        svd = codec_specs({"K": {"kind": "svd", "rank": 1}, "V": {"kind": "svd", "rank": 1}}, 1)
        self.assertNotIn("inter_prediction", svd[MatrixKey(0, "K")].to_dict())


if __name__ == "__main__":
    unittest.main()
