import copy
from pathlib import Path
import tempfile
import threading
import unittest
from unittest import mock

import numpy as np
import torch

from reelcache import _hevc
from reelcache.codecs import HEVCProvider, dequantize_device_coefficients, encode_tensor
from reelcache.errors import CapacityError, ConfigurationError, IntegrityError, StateError, UnsupportedError
from reelcache.factorization import fit_basis
from reelcache.gpu_fitting import GPUFitter, GPU_FIT_METHODS
from reelcache.nvdec import NVDECDecoder, unpack_device_frame
from reelcache.prefetch import LayerPrefetcher
from reelcache.types import CodecSpec, MatrixKey
from reelcache.telemetry import RuntimeTelemetry, cuda_span
from tests.test_codecs import context
from tests import test_causvid_adapter as caus


class GPUFittingContractTests(unittest.TestCase):
    def test_constructor_and_cpu_rejection_do_not_initialize_cuda(self):
        with mock.patch("torch.cuda._lazy_init", side_effect=AssertionError("CUDA init")):
            for method in GPU_FIT_METHODS:
                GPUFitter(gpu_id=0, gpu_uuid="GPU-test", method=method)
            for method in ("gesvd", "gesvdj", "gesvda", "cupy", "torch_lowrank", "gram"):
                with self.assertRaises(UnsupportedError):
                    fit_basis(torch.ones(24, 4), context(), 2, method=method)
        for kwargs in ({"method": "bad"}, {"gpu_id": -1}, {"max_workspace_bytes": 0}):
            with self.assertRaises(ConfigurationError):
                GPUFitter(**{"gpu_id": 0, "gpu_uuid": "GPU-test", **kwargs})

    def test_workspace_and_post_rope_fail_before_gpu(self):
        fitter = GPUFitter(gpu_id=0, gpu_uuid="GPU-test", max_workspace_bytes=1)
        with mock.patch("torch.cuda.device_count", side_effect=AssertionError("early CUDA")):
            with self.assertRaises(CapacityError):
                fitter.fit(torch.ones(24, 4), context(), 2)
            with self.assertRaises(ConfigurationError):
                fitter.fit(torch.ones(24, 4), context(key_space="post_rope"), 2)

    def test_gpu_fitter_rejects_foreign_thread(self):
        fitter = GPUFitter(gpu_id=0, gpu_uuid="GPU-test")
        errors = []
        def worker():
            try:
                fitter.fit(torch.ones(24, 4), context(), 2)
            except StateError:
                errors.append(True)
        thread = threading.Thread(target=worker)
        thread.start()
        thread.join(10)
        self.assertEqual(errors, [True])


class DeviceCoefficientTests(unittest.TestCase):
    def test_cuda_timing_is_explicit_and_cpu_spans_never_initialize_cuda(self):
        telemetry = RuntimeTelemetry(cuda_timing=True)
        with mock.patch("torch.cuda._lazy_init", side_effect=AssertionError("CUDA initialized")):
            with telemetry.capture(), cuda_span("decode.dequantize", torch.device("cpu")):
                pass
        self.assertFalse(telemetry.snapshot()["cuda_synchronized"])
        for value in (1, None, "yes"):
            with self.assertRaises(ConfigurationError):
                RuntimeTelemetry(cuda_timing=value)

    def test_vectorized_unpack_matches_all_cpu_plane_and_padding_rules(self):
        for fmt in ("yuv444p", "yuv420p"):
            for rank in (1, 4, 17, 320):
                layout = _hevc.video_layout(3, 3, 5, rank, fmt)
                image = np.arange(layout["planes"] * layout["image_height"] * layout["image_width"],
                                  dtype=np.uint8).reshape(layout["planes"], layout["image_height"], layout["image_width"])
                expected = np.empty((3, 5, rank), dtype=np.uint8)
                for index, plane, y, x in _hevc._cells(layout, rank):
                    expected[:, :, index] = image[plane, y:y + 3, x:x + 5]
                actual = unpack_device_frame(torch.from_numpy(image), layout, rank)
                self.assertTrue(np.array_equal(expected, actual.numpy()))

    def test_tensor_dequantization_matches_exact_numpy_rounding(self):
        rng = np.random.default_rng(841)
        values = rng.normal(size=(30, 16)).astype(np.float32)
        for mode, fraction in (("per_channel", 1.), ("global", 0.), ("leading_columns", .25)):
            pixels, affine, layout = _hevc.quantize_coefficients(values, mode, fraction)
            expected = _hevc.dequantize_coefficients(pixels, affine, layout)
            actual = dequantize_device_coefficients(torch.from_numpy(pixels), affine, layout)
            self.assertTrue(np.array_equal(expected, actual.numpy()))

    def test_decoded_checksum_is_explicit_opt_in_only_for_tensor_output(self):
        source = torch.randn(24, 4, generator=torch.Generator().manual_seed(434))
        packet = encode_tensor(source, context(), CodecSpec("svd_hevc", 3), validate_hevc=True)
        hardware = NVDECDecoder(gpu_id=0, gpu_uuid="GPU-fixture", device_output=True)
        streams = [packet.payloads["coefficients.000.hevc"]]
        pixels, _ = _hevc.decode_pixels(streams, packet.metadata["video"])
        altered = torch.from_numpy(pixels.copy())
        altered[0, 0] ^= 1
        with mock.patch.object(hardware, "decode_pixels", return_value=(altered, 0.)):
            with self.assertRaisesRegex(IntegrityError, "authoritative checksum"):
                HEVCProvider(nvdec=hardware, verify_decoded_coefficients=True).decode_tensor(packet)
            fast = HEVCProvider(nvdec=hardware).decode_tensor(packet)
            self.assertEqual(tuple(fast.shape), (24, 4))
            unchecked = encode_tensor(source, context(), CodecSpec("svd_hevc", 3))
            with self.assertRaisesRegex(IntegrityError, "without validation decode"):
                HEVCProvider(nvdec=hardware, verify_decoded_coefficients=True).decode_tensor(unchecked)
        self.assertFalse(torch.cuda.is_initialized())

    def test_checksum_option_must_be_boolean(self):
        for value in (1, None, "off"):
            with self.assertRaises(ConfigurationError):
                HEVCProvider(verify_decoded_coefficients=value)


class LayerPrefetchTests(unittest.TestCase):
    setUp = caus.CausVidAdapterTests.setUp

    def test_prefetch_raw_svd_hevc_matches_serial_and_releases_leases(self):
        for kind in ("raw", "svd", "svd_hevc"):
            with self.subTest(kind=kind):
                spec = CodecSpec(kind, rank=None if kind == "raw" else 3)
                with caus.CausVidAdapter(self.pipe, self.runtime, kind, config=caus.CONFIG, specs=spec) as adapter:
                    adapter.start(self.noise, self.conditioning, self.rng, conditioning_id="fixed")
                    adapter.step(clean_prefix=self.prefix)
                    snapshot = adapter.snapshot()
                    adapter.step()
                    adapter.step()
                    expected, rng = adapter.latent_state, adapter.rng_state
                    adapter.restore(snapshot, kind + "-prefetch", conditioning_id="fixed")
                    adapter.prefetch_history_bytes = 1 << 20
                    adapter.step()
                    adapter.step()
                    self.assertTrue(torch.equal(expected, adapter.latent_state))
                    self.assertTrue(torch.equal(rng, adapter.rng_state))
                    self.assertEqual(self.runtime.active_leases, 0)
                    self.assertEqual(self.runtime.residency.reserved_bytes, 0)
                    self.assertFalse(adapter._prefetch.stats()["active"])

    def test_failed_prefetch_preserves_head_and_cleans_resources(self):
        with caus.CausVidAdapter(self.pipe, self.runtime, "fail", config=caus.CONFIG,
                                prefetch_history_bytes=1 << 20) as adapter:
            adapter.start(self.noise, self.conditioning, self.rng, conditioning_id="fixed")
            adapter.step(clean_prefix=self.prefix)
            head = self.runtime.branch_info("fail")
            before_rng = adapter.rng_state
            decoder = self.runtime.decoder
            def fail(encoded, **kwargs):
                if encoded.metadata["context"]["key"]["target"] == "V":
                    raise IntegrityError("injected decode failure")
                return decoder(encoded, **kwargs)
            with mock.patch.object(self.runtime, "decoder", side_effect=fail):
                with self.assertRaisesRegex(IntegrityError, "injected"):
                    adapter.step()
            self.assertEqual(self.runtime.branch_info("fail"), head)
            self.assertTrue(torch.equal(before_rng, adapter.rng_state))
            self.assertEqual(self.runtime.active_leases, 0)

    def test_prefetch_budget_checked_before_decode(self):
        with caus.CausVidAdapter(self.pipe, self.runtime, "budget", config=caus.CONFIG,
                                prefetch_history_bytes=1) as adapter:
            adapter.start(self.noise, self.conditioning, self.rng, conditioning_id="fixed")
            adapter.step(clean_prefix=self.prefix)
            with mock.patch.object(self.runtime, "decoder", side_effect=AssertionError("early decode")):
                with self.assertRaises(CapacityError):
                    adapter.step()
            self.assertEqual(self.runtime.active_leases, 0)


if __name__ == "__main__":
    unittest.main()
