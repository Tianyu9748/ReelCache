import base64
import copy
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest import mock

import numpy as np
import torch

from reelcache import _hevc
from reelcache.codecs import (
    HEVCProvider, decode_tensor, encode_tensor, encoded_nbytes, hevc_capabilities, using_ffmpeg,
)
from reelcache.errors import ConfigurationError, IntegrityError, UnsupportedError
from reelcache.factorization import basis_condition_bound
from reelcache.types import CodecSpec, EncodedTensor, MatrixKey, ModelIdentity, TensorContext


def context(rows=24, channels=4, *, dtype="float32", target="K", key_space=None, geometry=None):
    frames, height, width = geometry or (2, 3, rows // 6)
    return TensorContext(
        ModelIdentity("codec-test", "source-bound-fixture", 1, channels, "native_kv", dtype),
        MatrixKey(0, target), 12, frames, height, width,
        key_space if key_space is not None else ("pre_rope" if target == "K" else "none"),
    )


def checked_decode(packet, **kwargs):
    return decode_tensor(packet, verify_packet=True, **kwargs)


def raw_bytes(tensor):
    return tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()


def reloaded(packet):
    return EncodedTensor(
        json.loads(json.dumps(packet.metadata, allow_nan=False)),
        {name: bytes(bytearray(raw)) for name, raw in packet.payloads.items()},
        bytes(bytearray(packet.basis)) if packet.basis is not None else None,
    )


def with_fraction_header(packet, fraction):
    metadata = copy.deepcopy(packet.metadata)
    metadata["codec"]["protected_fraction"] = fraction
    return EncodedTensor(metadata, dict(packet.payloads), packet.basis)


def independent_process_decode(packet):
    portable = {
        "metadata": packet.metadata,
        "payloads": {name: base64.b64encode(raw).decode("ascii") for name, raw in packet.payloads.items()},
        "basis": base64.b64encode(packet.basis).decode("ascii") if packet.basis is not None else None,
    }
    script = """
import base64, json, sys, torch
from reelcache.types import EncodedTensor
from reelcache.codecs import decode_tensor
d = json.load(sys.stdin)
p = EncodedTensor(d['metadata'], {k: base64.b64decode(v) for k, v in d['payloads'].items()},
                  base64.b64decode(d['basis']) if d['basis'] is not None else None)
x = decode_tensor(p)
sys.stdout.buffer.write(x.contiguous().view(torch.uint8).numpy().tobytes())
"""
    environment = dict(os.environ)
    environment.update(
        CUDA_VISIBLE_DEVICES="", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1",
    )
    result = subprocess.run(
        [sys.executable, "-B", "-c", script], input=json.dumps(portable).encode(),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=environment,
        # The shared model environment can spend minutes importing Torch over NFS.
        timeout=600, check=False,
    )
    if result.returncode:
        raise AssertionError(result.stderr.decode(errors="replace"))
    return result.stdout


class RawAndSVDCodecTests(unittest.TestCase):
    def test_default_injectable_provider_keeps_raw_dependency_free(self):
        matrix = torch.arange(96, dtype=torch.float32).reshape(24, 4)
        with mock.patch("reelcache._hevc.discover_ffmpeg", side_effect=AssertionError("RAW needs no FFmpeg")):
            provider = HEVCProvider()
            packet = provider.encode_tensor(matrix, context(), CodecSpec())
            self.assertEqual(raw_bytes(provider.decode_tensor(packet)), raw_bytes(matrix))

    def test_raw_exact_bits_all_dtypes_and_post_rope(self):
        values = np.array([
            0x0000, 0x8000, 0x3F80, 0xBF80, 0x7F80, 0xFF80, 0x7FC1, 0xFFC2,
        ], dtype=np.uint16)
        bf16 = torch.from_numpy(values.view(np.uint8).copy()).view(torch.bfloat16)
        cases = [
            torch.arange(96, dtype=dtype).reshape(24, 4).T.contiguous().T
            for dtype in (torch.float16, torch.float32, torch.float64)
        ]
        cases.append(bf16.repeat(12).reshape(24, 4))
        for matrix in cases:
            with self.subTest(dtype=matrix.dtype):
                ctx = context(dtype=str(matrix.dtype), key_space="post_rope")
                packet = encode_tensor(matrix, ctx, CodecSpec())
                decoded = decode_tensor(reloaded(packet))
                self.assertEqual(decoded.dtype, matrix.dtype)
                self.assertEqual(raw_bytes(decoded), raw_bytes(matrix))
                self.assertIsNone(packet.basis)
                self.assertEqual(packet.metrics["decomposition_count"], 0)
                self.assertEqual(packet.metrics["source_bytes"], matrix.numel() * matrix.element_size())

    def test_raw_float_nan_payloads_infinities_and_signed_zero(self):
        bits = np.array([
            0, 0x80000000, 0x7FC00001, 0xFFC00002, 0x7F800000, 0xFF800000,
            0x00000001, 0x3F800000,
        ], dtype=np.uint32)
        matrix = torch.from_numpy(bits.view(np.uint8).copy()).view(torch.float32).repeat(12).reshape(24, 4)
        packet = encode_tensor(matrix, context(key_space="post_rope"), CodecSpec())
        self.assertEqual(raw_bytes(decode_tensor(packet)), raw_bytes(matrix))
        self.assertEqual(independent_process_decode(packet), raw_bytes(matrix))

    def test_exact_svd_packet_reconstruction_and_independent_reload(self):
        generator = torch.Generator().manual_seed(66)
        matrix = torch.randn(24, 4, generator=generator)
        packet = encode_tensor(matrix, context(), CodecSpec("svd", 3))
        basis = torch.frombuffer(bytearray(packet.basis), dtype=torch.float16).reshape(3, 4)
        coefficients = torch.frombuffer(
            bytearray(packet.payloads["coefficients.f32"]), dtype=torch.float32,
        ).reshape(24, 3)
        expected = coefficients @ basis.float()
        self.assertTrue(torch.equal(coefficients, matrix @ basis.float().T))
        self.assertEqual(raw_bytes(decode_tensor(packet)), raw_bytes(expected))
        self.assertEqual(raw_bytes(decode_tensor(reloaded(packet))), raw_bytes(expected))
        self.assertEqual(independent_process_decode(packet), raw_bytes(expected))
        self.assertEqual(packet.metadata["basis"]["dtype"], "float16")
        self.assertEqual(packet.metadata["basis"]["shape"], [3, 4])
        self.assertEqual(packet.metadata["basis"]["view_rank"], 3)
        self.assertEqual(packet.metadata["factorization_domain"], "normalized_pre_rope_k")

    def test_svd_full_rank_error_is_only_rounded_basis_and_dtype(self):
        generator = torch.Generator().manual_seed(19)
        for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                matrix = torch.randn(24, 4, generator=generator).to(dtype)
                packet = encode_tensor(matrix, context(dtype=str(dtype)), CodecSpec("svd", 4))
                decoded = decode_tensor(packet, dtype="float32")
                relative = (decoded - matrix.float()).norm() / matrix.float().norm()
                self.assertLess(relative.item(), 0.002)
                self.assertEqual(decode_tensor(packet).dtype, dtype)

    def test_explicit_target_and_layer_dtypes_round_trip_without_casting(self):
        declarations = [
            {"K": "float32", "V": "bfloat16"},
            {
                "0:K": "bfloat16", "0:V": "bfloat16",
                "1:K": "float32", "1:V": "bfloat16",
            },
        ]
        for declared in declarations:
            model = ModelIdentity("declared-dtype-fixture", "synthetic", 2, 4, "native_kv", declared)
            for layer in range(2):
                for target in ("K", "V"):
                    key = MatrixKey(layer, target)
                    dtype = getattr(torch, model.dtype_for(key))
                    lossy = target == "K" and dtype == torch.float32
                    key_space = ("pre_rope" if lossy else "post_rope") if target == "K" else "none"
                    ctx = TensorContext(model, key, 0, 2, 3, 4, key_space)
                    matrix = torch.arange(96).reshape(24, 4).to(dtype)
                    spec = CodecSpec("svd", 2) if lossy else CodecSpec()
                    packet = encode_tensor(matrix, ctx, spec)
                    decoded = decode_tensor(reloaded(packet))
                    self.assertEqual(decoded.dtype, dtype)
                    self.assertEqual(packet.metadata["dtype"], ctx.dtype)
                    self.assertEqual(packet.metadata["context"], ctx.to_dict())
                    if not lossy:
                        self.assertEqual(raw_bytes(decoded), raw_bytes(matrix))
                    with self.assertRaises(ConfigurationError):
                        encode_tensor(matrix.double(), ctx, spec)
                    metadata = copy.deepcopy(packet.metadata)
                    metadata["dtype"] = "float64"
                    with self.assertRaises(IntegrityError):
                        checked_decode(EncodedTensor(metadata, packet.payloads, packet.basis))

    def test_exact_byte_accounting_includes_metadata_and_entire_basis(self):
        matrix = torch.arange(96, dtype=torch.float32).reshape(24, 4)
        for spec in (CodecSpec(), CodecSpec("svd", 3)):
            packet = encode_tensor(matrix, context(), spec)
            metadata_size = len(json.dumps(
                packet.metadata, sort_keys=True, separators=(",", ":"), allow_nan=False,
            ).encode())
            expected = metadata_size + sum(map(len, packet.payloads.values())) + len(packet.basis or b"")
            self.assertEqual(encoded_nbytes(packet), expected)
            self.assertEqual(packet.metrics["metadata_bytes"], metadata_size)
            self.assertEqual(packet.metrics["representation_bytes"], expected)
            self.assertGreater(encoded_nbytes(packet, include_metrics=True), expected)
            for name, data in packet.payloads.items():
                self.assertEqual(packet.metadata["payloads"][name]["bytes"], len(data))
                self.assertEqual(packet.metadata["payloads"][name]["sha256"], hashlib.sha256(data).hexdigest())

    def test_legacy_ignored_fraction_headers_reload_without_rewriting(self):
        matrix = torch.arange(96, dtype=torch.float32).reshape(24, 4)
        for kind, rank in (("raw", None), ("svd", 2)):
            for normalization in ("global", "per_channel"):
                packet = encode_tensor(
                    matrix, context(), CodecSpec(kind, rank, normalization),
                )
                expected = raw_bytes(decode_tensor(packet))
                for fraction in (0, 0.375, 1):
                    with self.subTest(kind=kind, normalization=normalization, fraction=fraction):
                        legacy = with_fraction_header(packet, fraction)
                        before = json.dumps(legacy.metadata, sort_keys=True, separators=(",", ":"))
                        before_size = encoded_nbytes(legacy)
                        self.assertEqual(raw_bytes(decode_tensor(legacy)), expected)
                        self.assertEqual(raw_bytes(decode_tensor(reloaded(legacy))), expected)
                        self.assertEqual(
                            json.dumps(legacy.metadata, sort_keys=True, separators=(",", ":")), before,
                        )
                        self.assertEqual(encoded_nbytes(legacy), before_size)

    def test_ignored_fraction_compatibility_keeps_header_validation_strict(self):
        matrix = torch.arange(96, dtype=torch.float32).reshape(24, 4)
        for normalization in ("global", "per_channel"):
            packet = encode_tensor(matrix, context(), CodecSpec("svd", 2, normalization))
            for fraction in (-0.1, 1.1, True, False, None, "0.5", float("nan"), float("inf")):
                invalid = reloaded(packet)
                invalid.metadata["codec"]["protected_fraction"] = fraction
                with self.subTest(normalization=normalization, fraction=fraction), self.assertRaises(IntegrityError):
                    checked_decode(invalid)
            for missing in ("protected_fraction", "normalization", "preset", "qp", "rank"):
                invalid = reloaded(packet)
                del invalid.metadata["codec"][missing]
                with self.subTest(missing=missing), self.assertRaises(IntegrityError):
                    checked_decode(invalid)
            for field, value in (
                ("kind", "raw"), ("rank", 1), ("qp", 9),
                ("preset", "medium"), ("pix_fmt", "yuv422p"),
                ("ffmpeg_path", None), ("unexpected", True),
            ):
                invalid = reloaded(packet)
                invalid.metadata["codec"][field] = value
                with self.subTest(field=field), self.assertRaises(IntegrityError):
                    checked_decode(invalid)
            invalid = reloaded(packet)
            invalid.metadata["codec"] = list(invalid.metadata["codec"].items())
            with self.assertRaises(IntegrityError):
                checked_decode(invalid)

    def test_native_owner_columns_never_expand_query_heads(self):
        matrix = torch.arange(96, dtype=torch.float32).reshape(24, 4)
        for spec in (CodecSpec(), CodecSpec("svd", 4)):
            packet = encode_tensor(matrix, context(), spec)
            full = decode_tensor(packet)
            for columns in (slice(1, 3), slice(0, 4, 2), [3, 0], (2, 1), torch.tensor([0, 2])):
                with self.subTest(kind=spec.kind, columns=columns):
                    owned = decode_tensor(packet, columns=columns)
                    torch.testing.assert_close(owned, full[:, columns])
                    self.assertEqual(owned.shape[0], 24)
                    self.assertEqual(owned.shape[1], 2)
            for columns in ([], [1, 1], [-1], [4], [True], slice(-1, 2), slice(0, 5), slice(2, 2), slice(0, 4, -1)):
                with self.subTest(columns=columns), self.assertRaises(ConfigurationError):
                    decode_tensor(packet, columns=columns)

    def test_lossy_post_rope_nonfinite_geometry_and_unsupported_options(self):
        matrix = torch.ones(24, 4)
        for kind in ("svd", "svd_hevc"):
            with self.assertRaises(ConfigurationError):
                encode_tensor(matrix, context(key_space="post_rope"), CodecSpec(kind, 2))
        with self.assertRaises(ConfigurationError):
            encode_tensor(matrix, context(key_space="none"), CodecSpec())
        with self.assertRaises(ConfigurationError):
            encode_tensor(matrix[:12], context(), CodecSpec())
        with self.assertRaises(ConfigurationError):
            encode_tensor(matrix, context(dtype="bfloat16"), CodecSpec())
        for rank in (5, 25):
            with self.assertRaises(ConfigurationError):
                encode_tensor(matrix, context(), CodecSpec("svd", rank))
        invalid = matrix.clone()
        invalid[0, 0] = float("nan")
        with self.assertRaises(ConfigurationError):
            encode_tensor(invalid, context(), CodecSpec("svd", 2))
        for options in (
            {"pix_fmt": "yuv422p"}, {"qp": 9}, {"preset": "medium"},
        ):
            with self.subTest(options=options), self.assertRaises(UnsupportedError):
                encode_tensor(matrix, context(), CodecSpec("svd_hevc", 2, **options))
        with self.assertRaises(ConfigurationError):
            CodecSpec("svd_hevc", 2, normalization="greedy")
        with self.assertRaises(ConfigurationError):
            encode_tensor(
                matrix, context(), CodecSpec(),
                reuse_basis=torch.eye(4, dtype=torch.float16), reuse_threshold=0.1,
            )

    def test_raw_and_svd_corruption_rejected_before_reconstruction(self):
        matrix = torch.arange(96, dtype=torch.float32).reshape(24, 4)
        for spec in (CodecSpec(), CodecSpec("svd", 2)):
            packet = encode_tensor(matrix, context(), spec)
            name = next(iter(packet.payloads))
            modified = dict(packet.payloads)
            data = bytearray(modified[name])
            data[0] ^= 1
            modified[name] = bytes(data)
            with self.assertRaises(IntegrityError):
                decode_tensor(EncodedTensor(packet.metadata, modified, packet.basis), verify_packet=True)
            # Default restoration skips payload re-hashing; stores hash-verify on read.
            decode_tensor(EncodedTensor(packet.metadata, modified, packet.basis))
            extra = dict(packet.payloads, uncharged=b"hidden")
            with self.assertRaises(IntegrityError):
                checked_decode(EncodedTensor(packet.metadata, extra, packet.basis))
            for field, value in (
                ("shape", [12, 8]), ("dtype", "float16"), ("byte_order", "big"),
                ("schema", "unknown"), ("unexpected", 1),
            ):
                metadata = copy.deepcopy(packet.metadata)
                metadata[field] = value
                with self.subTest(kind=spec.kind, field=field), self.assertRaises(IntegrityError):
                    checked_decode(EncodedTensor(metadata, packet.payloads, packet.basis))
            metadata = copy.deepcopy(packet.metadata)
            metadata["payloads"][name]["bytes"] = True
            with self.assertRaises(IntegrityError):
                checked_decode(EncodedTensor(metadata, packet.payloads, packet.basis))
            if packet.basis is not None:
                with self.assertRaises(IntegrityError):
                    checked_decode(EncodedTensor(packet.metadata, packet.payloads, packet.basis[:-1]))
                metadata = copy.deepcopy(packet.metadata)
                metadata["basis"]["view_rank"] = 1
                with self.assertRaises(IntegrityError):
                    checked_decode(EncodedTensor(metadata, packet.payloads, packet.basis))

    def test_rehashed_nonfinite_coefficients_or_basis_are_rejected(self):
        matrix = torch.ones(24, 4)
        packet = encode_tensor(matrix, context(), CodecSpec("svd", 2))
        payloads = dict(packet.payloads)
        data = torch.full((24, 2), float("nan")).numpy().tobytes()
        payloads["coefficients.f32"] = data
        metadata = copy.deepcopy(packet.metadata)
        metadata["payloads"]["coefficients.f32"]["sha256"] = hashlib.sha256(data).hexdigest()
        metadata["decoded_coefficients_sha256"] = hashlib.sha256(data).hexdigest()
        with self.assertRaises(IntegrityError):
            decode_tensor(EncodedTensor(metadata, payloads, packet.basis), verify_packet=True)
        for basis in (
            torch.full((2, 4), float("nan"), dtype=torch.float16),
            torch.zeros(2, 4, dtype=torch.float16),
        ):
            data = raw_bytes(basis)
            metadata = copy.deepcopy(packet.metadata)
            metadata["basis"]["sha256"] = hashlib.sha256(data).hexdigest()
            with self.assertRaises(IntegrityError):
                decode_tensor(EncodedTensor(metadata, packet.payloads, data), verify_packet=True)
            # Default: conditioning/finite checks are skipped, structure still accepted.
            decode_tensor(EncodedTensor(metadata, packet.payloads, data))

    def test_restore_verification_is_default_off_and_explicitly_enabled(self):
        packet = encode_tensor(torch.randn(24, 4, generator=torch.Generator().manual_seed(3)),
                               context(), CodecSpec("svd", 2))
        with mock.patch("reelcache.codecs.basis_condition_bound", side_effect=AssertionError("default ran check")), \
                mock.patch("reelcache.codecs._sha", side_effect=AssertionError("default re-hashed")):
            fast = decode_tensor(packet)
        checked = decode_tensor(packet, verify_packet=True)
        self.assertTrue(torch.equal(fast, checked))
        self.assertTrue(torch.equal(HEVCProvider().decode_tensor(packet), fast))
        with mock.patch("reelcache.codecs.basis_condition_bound", wraps=basis_condition_bound) as check:
            HEVCProvider(verify_restored_packets=True).decode_tensor(packet)
        check.assert_called_once()
        truncated = EncodedTensor(packet.metadata, packet.payloads, packet.basis[:-2])
        with self.assertRaises(IntegrityError):
            decode_tensor(truncated, verify_packet=True)
        with self.assertRaises(Exception):
            decode_tensor(truncated)
        for value in (1, None, "on"):
            with self.assertRaises(ConfigurationError):
                decode_tensor(packet, verify_packet=value)
            with self.assertRaises(ConfigurationError):
                HEVCProvider(verify_restored_packets=value)


class NormalizationTests(unittest.TestCase):
    def test_leading_removes_protected_outliers_from_shared_range(self):
        coefficients = np.array([
            [-1000, -100, 1, 2], [1000, 100, 3, 4], [0, 0, 2, 3],
        ], dtype=np.float32)
        quantized, affine, layout = _hevc.quantize_coefficients(coefficients, "leading_columns", 0.5)
        self.assertEqual(layout["protected_count"], 2)
        self.assertEqual(layout["affine_shape"], [2, 3])
        self.assertEqual(float(affine[1, 0]), 1)
        self.assertAlmostEqual(float(affine[0, 0]), 3 / 255, places=7)
        reconstructed = _hevc.dequantize_coefficients(quantized, affine, layout)
        self.assertLess(np.abs(reconstructed[:, 2:] - coefficients[:, 2:]).max(), 0.01)
        _, global_affine, _ = _hevc.quantize_coefficients(coefficients, "global")
        self.assertGreater(global_affine[0, 0], affine[0, 0] * 100)

    def test_all_and_zero_protection_endpoints(self):
        coefficients = np.arange(24, dtype=np.float32).reshape(6, 4)
        q_all, a_all, l_all = _hevc.quantize_coefficients(coefficients, "leading_columns", 1)
        q_pc, a_pc, l_pc = _hevc.quantize_coefficients(coefficients, "per_channel")
        np.testing.assert_array_equal(a_all[:, 0], [0, 0])
        np.testing.assert_array_equal(q_all, q_pc)
        np.testing.assert_array_equal(a_all[:, 1:], a_pc)
        q_none, a_none, _ = _hevc.quantize_coefficients(coefficients, "leading_columns", 0)
        q_global, a_global, _ = _hevc.quantize_coefficients(coefficients, "global")
        np.testing.assert_array_equal(q_none, q_global)
        np.testing.assert_array_equal(a_none, a_global)
        np.testing.assert_array_equal(
            _hevc.dequantize_coefficients(q_all, a_all, l_all),
            _hevc.dequantize_coefficients(q_pc, a_pc, l_pc),
        )

    def test_constant_channels_ignore_codec_pixel_noise(self):
        coefficients = np.tile(np.array([-10, 3, 6, 6], dtype=np.float32), (24, 1))
        for normalization, fraction in (("per_channel", 1), ("leading_columns", 0.5)):
            q, affine, layout = _hevc.quantize_coefficients(coefficients, normalization, fraction)
            self.assertTrue(np.all(affine[0] == 0))
            np.testing.assert_array_equal(
                _hevc.dequantize_coefficients(np.full_like(q, 255), affine, layout),
                coefficients,
            )
        uniform = np.full((24, 4), -7.0, dtype=np.float32)
        q, affine, layout = _hevc.quantize_coefficients(uniform, "global")
        self.assertEqual(affine[0, 0], 0)
        np.testing.assert_array_equal(_hevc.dequantize_coefficients(q + 13, affine, layout), uniform)

    def test_invalid_affine_parameters_and_finite_ranges(self):
        values = np.arange(24, dtype=np.float32).reshape(6, 4)
        q, affine, layout = _hevc.quantize_coefficients(values, "per_channel")
        for value in (-1, float("nan"), float("inf")):
            corrupt = affine.copy()
            corrupt[0, 0] = value
            with self.assertRaises(IntegrityError):
                _hevc.dequantize_coefficients(q, corrupt, layout)
        with self.assertRaises(ConfigurationError):
            _hevc.quantize_coefficients(values, "leading_columns", float("nan"))
        huge = np.array([[-np.finfo(np.float32).max], [np.finfo(np.float32).max]])
        with self.assertRaises(ConfigurationError):
            _hevc.quantize_coefficients(huge, "global")


class RealHEVCTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.capabilities = hevc_capabilities()
        except UnsupportedError as exc:
            raise unittest.SkipTest(str(exc)) from exc
        if set(cls.capabilities["pixel_formats"]) != {"yuv444p", "yuv420p"}:
            raise unittest.SkipTest("both real libx265 pixel formats are required for these tests")

    def identity_packet(self, matrix, pix_fmt, normalization="per_channel", fraction=1, validate=True):
        return encode_tensor(
            matrix, context(rows=matrix.shape[0], channels=matrix.shape[1]),
            CodecSpec("svd_hevc", matrix.shape[1], normalization, fraction, pix_fmt),
            reuse_basis=torch.eye(matrix.shape[1], dtype=torch.float16), reuse_threshold=0,
            validate_hevc=validate,
        )

    def test_encode_validation_decode_is_default_off_and_explicit(self):
        matrix = torch.arange(96, dtype=torch.float32).reshape(24, 4) / 7
        threaded = encode_tensor(
            matrix, context(), CodecSpec("svd_hevc", 3), hevc_threads=4,
        )
        single = encode_tensor(matrix, context(), CodecSpec("svd_hevc", 3))
        self.assertEqual(threaded.metrics["hevc_encode_threads"], 4)
        self.assertEqual(raw_bytes(decode_tensor(threaded)), raw_bytes(decode_tensor(single)))
        self.assertEqual(_hevc.x265_thread_params(1, 3), "pools=1:frame-threads=1")
        self.assertEqual(_hevc.x265_thread_params(8, 3), "pools=8:frame-threads=3")
        for value in (0, 65, 1.5):
            with self.assertRaises(ConfigurationError):
                encode_tensor(matrix, context(), CodecSpec("svd_hevc", 3), hevc_threads=value)
        with self.assertRaises(ConfigurationError):
            HEVCProvider(x265_threads=0)
        with mock.patch("reelcache._hevc.decode_pixels", side_effect=AssertionError("validation decode ran")):
            fast = self.identity_packet(matrix, "yuv444p", validate=False)
        self.assertIsNone(fast.metadata["decoded_coefficients_sha256"])
        self.assertFalse(fast.metrics["hevc_validation_decode"])
        self.assertEqual(fast.metrics["hevc_validation_decode_seconds"], 0.0)
        checked = self.identity_packet(matrix, "yuv444p")
        self.assertTrue(checked.metrics["hevc_validation_decode"])
        self.assertEqual(fast.payloads, checked.payloads)
        self.assertEqual(raw_bytes(decode_tensor(fast)), raw_bytes(decode_tensor(checked)))
        self.assertEqual(raw_bytes(checked_decode(fast)), raw_bytes(decode_tensor(checked)))
        self.assertEqual(raw_bytes(decode_tensor(reloaded(fast))), raw_bytes(decode_tensor(checked)))
        default = encode_tensor(matrix, context(), CodecSpec("svd_hevc", 2))
        self.assertIsNone(default.metadata["decoded_coefficients_sha256"])
        with self.assertRaises(ConfigurationError):
            encode_tensor(matrix, context(), CodecSpec("svd_hevc", 2), validate_hevc=1)
        metadata = copy.deepcopy(encode_tensor(matrix, context(), CodecSpec("svd", 2)).metadata)
        metadata["decoded_coefficients_sha256"] = None
        svd = encode_tensor(matrix, context(), CodecSpec("svd", 2))
        with self.assertRaises(IntegrityError):
            checked_decode(EncodedTensor(metadata, svd.payloads, svd.basis))

    def test_real_both_formats_and_all_supported_normalizations(self):
        rows = torch.arange(24, dtype=torch.float32)
        matrix = torch.stack((rows, rows.remainder(5) * 3, rows.sin(), rows.cos() * 2), dim=1)
        for fmt in ("yuv444p", "yuv420p"):
            for normalization in ("per_channel", "global", "leading_columns"):
                with self.subTest(fmt=fmt, normalization=normalization):
                    packet = self.identity_packet(matrix, fmt, normalization, 0.5)
                    decoded = decode_tensor(packet)
                    self.assertEqual(packet.metadata["video"]["pix_fmt"], fmt)
                    self.assertEqual(packet.metadata["codec"]["qp"], 8)
                    self.assertEqual(packet.metadata["codec"]["preset"], "ultrafast")
                    self.assertEqual(packet.metadata["video"]["planes"], 3 if fmt == "yuv444p" else 1)
                    self.assertTrue(any(name.endswith(".hevc") for name in packet.payloads))
                    self.assertEqual(raw_bytes(decoded), raw_bytes(decode_tensor(reloaded(packet))))
                    self.assertLess(((decoded - matrix).norm() / matrix.norm()).item(), 0.1)
                    self.assertEqual(packet.metrics["representation_bytes"], encoded_nbytes(packet))
                    self.assertGreater(packet.metrics["hevc_encode_seconds"], 0)
                    self.assertGreater(packet.metrics["hevc_validation_decode_seconds"], 0)
                    torch.testing.assert_close(decode_tensor(packet, columns=slice(2, 4)), decoded[:, 2:])
        self.assertEqual(independent_process_decode(packet), raw_bytes(decoded))

    def test_real_constant_channels_and_zero_tensors(self):
        constants = torch.tensor([-10, 3, 6, 6], dtype=torch.float32).repeat(24, 1)
        for fmt in ("yuv444p", "yuv420p"):
            for normalization, values, fraction in (
                ("per_channel", constants, 1),
                ("leading_columns", constants, 0.5),
                ("global", torch.full_like(constants, -7), 1),
                ("per_channel", torch.zeros_like(constants), 1),
            ):
                with self.subTest(fmt=fmt, normalization=normalization):
                    packet = self.identity_packet(values, fmt, normalization, fraction)
                    self.assertEqual(raw_bytes(decode_tensor(packet)), raw_bytes(values))

    def test_public_fraction_headers_and_effective_counts_are_canonical(self):
        matrix = torch.arange(96, dtype=torch.float32).reshape(24, 4)
        for fmt in ("yuv444p", "yuv420p"):
            decoded = {}
            for normalization in ("global", "per_channel", "leading_columns"):
                for fraction in (0.0, 1.0):
                    with self.subTest(fmt=fmt, normalization=normalization, fraction=fraction):
                        packet = self.identity_packet(matrix, fmt, normalization, fraction)
                        expected_count = (
                            0 if normalization == "global"
                            else 4 if normalization == "per_channel" else int(4 * fraction)
                        )
                        expected_fraction = (
                            0.0 if normalization == "global"
                            else 1.0 if normalization == "per_channel" else fraction
                        )
                        self.assertEqual(packet.metadata["codec"]["protected_fraction"], expected_fraction)
                        self.assertEqual(packet.metadata["normalization"]["protected_count"], expected_count)
                        affine_shape = packet.metadata["normalization"]["affine_shape"]
                        self.assertEqual(len(packet.payloads["affine.f32"]), int(np.prod(affine_shape)) * 4)
                        decoded[normalization, fraction] = decode_tensor(reloaded(packet))
            for normalization in ("global", "per_channel"):
                self.assertTrue(torch.equal(decoded[normalization, 0.0], decoded[normalization, 1.0]))
            self.assertTrue(torch.equal(decoded["global", 0.0], decoded["leading_columns", 0.0]))
            self.assertTrue(torch.equal(decoded["per_channel", 1.0], decoded["leading_columns", 1.0]))

    def test_legacy_hevc_headers_and_normalization_descriptors(self):
        matrix = torch.arange(96, dtype=torch.float32).reshape(24, 4)
        for fmt in ("yuv444p", "yuv420p"):
            for normalization in ("global", "per_channel"):
                packet = self.identity_packet(matrix, fmt, normalization)
                expected = raw_bytes(decode_tensor(packet))
                for fraction in (0, 0.375, 1):
                    with self.subTest(fmt=fmt, normalization=normalization, fraction=fraction):
                        legacy = with_fraction_header(packet, fraction)
                        before = json.dumps(legacy.metadata, sort_keys=True, separators=(",", ":"))
                        self.assertEqual(raw_bytes(decode_tensor(reloaded(legacy))), expected)
                        self.assertEqual(
                            json.dumps(legacy.metadata, sort_keys=True, separators=(",", ":")), before,
                        )
                invalid = with_fraction_header(packet, 1 if normalization == "global" else 0)
                invalid.metadata["normalization"]["protected_count"] = (
                    4 if normalization == "global" else 0
                )
                with mock.patch("reelcache._hevc.decode_pixels", side_effect=AssertionError("must not decode")):
                    with self.assertRaises(IntegrityError):
                        checked_decode(invalid)
            leading = self.identity_packet(matrix, fmt, "leading_columns", 0.5)
            for fraction in (0.0, 0.25, 1.0):
                invalid = with_fraction_header(leading, fraction)
                with mock.patch("reelcache._hevc.decode_pixels", side_effect=AssertionError("must not decode")):
                    with self.assertRaises(IntegrityError):
                        checked_decode(invalid)

    def test_hevc_corruption_and_mismatched_native_format(self):
        matrix = torch.arange(96, dtype=torch.float32).reshape(24, 4)
        packet = self.identity_packet(matrix, "yuv444p")
        name = next(name for name in packet.payloads if name.endswith(".hevc"))
        payloads = dict(packet.payloads)
        payloads[name] = payloads[name][:-10]
        with self.assertRaises(IntegrityError):
            decode_tensor(EncodedTensor(packet.metadata, payloads, packet.basis))
        payloads = dict(packet.payloads)
        payloads[name] = b"not an HEVC stream"
        metadata = copy.deepcopy(packet.metadata)
        descriptor = metadata["payloads"][name]
        descriptor.update(
            bytes=len(payloads[name]), shape=[len(payloads[name])],
            sha256=hashlib.sha256(payloads[name]).hexdigest(),
        )
        with self.assertRaises(IntegrityError):
            decode_tensor(EncodedTensor(metadata, payloads, packet.basis))
        metadata = copy.deepcopy(packet.metadata)
        metadata["codec"]["pix_fmt"] = "yuv420p"
        metadata["video"] = _hevc.video_layout(2, 3, 4, 4, "yuv420p")
        with self.assertRaises(IntegrityError):
            decode_tensor(EncodedTensor(metadata, packet.payloads, packet.basis))

    def test_corrupt_affine_and_geometry_rejected_before_decoder(self):
        packet = self.identity_packet(torch.ones(24, 4), "yuv420p")
        metadata = copy.deepcopy(packet.metadata)
        metadata["video"]["image_height"] += 2
        with mock.patch("reelcache._hevc.decode_pixels", side_effect=AssertionError("must not decode")):
            with self.assertRaises(IntegrityError):
                checked_decode(EncodedTensor(metadata, packet.payloads, packet.basis))
        affine = np.full((2, 4), np.nan, dtype="<f4").tobytes()
        payloads = dict(packet.payloads)
        payloads["affine.f32"] = affine
        metadata = copy.deepcopy(packet.metadata)
        metadata["payloads"]["affine.f32"]["sha256"] = hashlib.sha256(affine).hexdigest()
        with mock.patch("reelcache._hevc.decode_pixels", side_effect=AssertionError("must not decode")):
            with self.assertRaises(IntegrityError):
                checked_decode(EncodedTensor(metadata, payloads, packet.basis))

    def test_explicit_binary_rng_environment_and_temp_configuration_preserved(self):
        matrix = torch.arange(96, dtype=torch.float32).reshape(24, 4)
        environment, tempdir = dict(os.environ), tempfile.tempdir
        rng = torch.get_rng_state().clone()
        with using_ffmpeg(self.capabilities["binary"]) as binary:
            self.assertEqual(binary, self.capabilities["binary"])
            packet = self.identity_packet(matrix, "yuv444p")
            decode_tensor(packet)
        self.assertEqual(environment, dict(os.environ))
        self.assertEqual(tempdir, tempfile.tempdir)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertFalse(torch.cuda.is_initialized())
        with self.assertRaises(UnsupportedError):
            with using_ffmpeg("tests/nonexistent-ffmpeg-executable"):
                pass
        with mock.patch("reelcache._hevc._capabilities", return_value=(frozenset({"yuv420p"}), "test")):
            with self.assertRaises(UnsupportedError):
                self.identity_packet(matrix, "yuv444p")

    def test_injected_provider_reestablishes_explicit_binary_on_worker(self):
        provider = HEVCProvider(self.capabilities["binary"])
        original_discover = _hevc.discover_ffmpeg

        def require_explicit(path=None):
            if path is None and _hevc._BINARY.get() is None:
                raise AssertionError("worker lost its explicit FFmpeg selection")
            return original_discover(path)

        matrix = torch.arange(96, dtype=torch.float32).reshape(24, 4)
        with (
            mock.patch("reelcache._hevc.discover_ffmpeg", side_effect=require_explicit),
            ThreadPoolExecutor(max_workers=1) as workers,
        ):
            packet = workers.submit(
                provider.encode_tensor, matrix, context(), CodecSpec("svd_hevc", 3),
            ).result(timeout=30)
            restored = workers.submit(
                provider.decode_tensor, reloaded(packet), columns=slice(1, 3),
            ).result(timeout=30)
            self.assertEqual(restored.shape, (24, 2))
            self.assertIsNone(workers.submit(_hevc._BINARY.get).result(timeout=30))
        self.assertIsNone(_hevc._BINARY.get())

    def test_layout_padding_is_explicit_and_never_changes_requested_format(self):
        for fmt in ("yuv444p", "yuv420p"):
            layout = _hevc.video_layout(2, 3, 5, 4, fmt)
            pixels = np.arange(120, dtype=np.uint8).reshape(30, 4)
            streams, _, _ = _hevc.encode_pixels(pixels, layout)
            decoded, _ = _hevc.decode_pixels(streams, layout)
            self.assertEqual(decoded.shape, pixels.shape)
            self.assertEqual(layout["pix_fmt"], fmt)
            self.assertEqual(layout["frame_padding"], [13, 6] if fmt == "yuv444p" else [10, 6])
        with self.assertRaises(UnsupportedError):
            _hevc.video_layout(1, 8193, 16, 4, "yuv420p")


if __name__ == "__main__":
    unittest.main()
