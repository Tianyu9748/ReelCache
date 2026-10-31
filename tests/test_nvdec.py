import io
import os
from types import SimpleNamespace
import threading
import unittest
from unittest import mock

import numpy as np
import torch

from reelcache import _hevc
from reelcache.codecs import HEVCProvider, decode_tensor, encode_tensor
from reelcache.errors import CapacityError, ConfigurationError, IntegrityError, StateError, UnsupportedError
from reelcache.nvdec import NVDECDecoder
from reelcache.types import CodecSpec
from tests.test_codecs import context


UUID = "GPU-bf621a31-0ece-332e-d43d-d20e0d4a42be"


class FakeFrame:
    def __init__(self, array, fmt, height):
        self.array, self.format, self.height = array, fmt, height
        self.shape, self.strides, self.dtype = array.shape, array.strides, array.dtype

    def framesize(self):
        return self.array.nbytes

    def GetPtrToPlane(self, index):
        return self.array.ctypes.data + index * self.height * self.array.shape[1]


class FakeDecoder:
    def __init__(self, layout):
        self.layout = layout
        self.frames = []
        height, width = layout["image_height"], layout["image_width"]
        for i in range(layout["frames"]):
            array = np.full((3 * height, width), i + 1, dtype=np.uint8)
            self.frames.append(FakeFrame(array, "YUV444", height))

    def Decode(self, packet):
        return self.frames if packet == "EOS" else []

    def SyncOnCUStream(self):
        pass

    def GetWidth(self):
        return self.layout["image_width"]

    def GetHeight(self):
        return self.layout["image_height"]

    def GetPixelFormat(self):
        return "YUV444"

    def GetFrameSize(self):
        return self.layout["image_width"] * self.layout["image_height"] * 3


class FakeDemux:
    def __init__(self, layout):
        self.layout = layout

    def __iter__(self):
        return iter(("packet",))

    def GetNvCodecId(self):
        return "HEVC"

    def Width(self):
        return self.layout["image_width"]

    def Height(self):
        return self.layout["image_height"]

    def BitDepth(self):
        return 8


def fake_backend(layout, **options):
    backend = NVDECDecoder(gpu_id=0, gpu_uuid=UUID, **options)
    decoder = FakeDecoder(layout)
    api = SimpleNamespace(
        __version__="test-only",
        cudaVideoChromaFormat=SimpleNamespace(__members__={"444": 3, "420": 1}),
        cudaVideoCodec=SimpleNamespace(HEVC="HEVC"),
        Pixel_Format=SimpleNamespace(YUV444="YUV444", NV12="NV12"),
        OutputColorType=SimpleNamespace(NATIVE="NATIVE"),
        GetDecoderCaps=mock.Mock(return_value={
            "supported": 1, "width_min": 1, "height_min": 1,
            "width_max": 8192, "height_max": 8192, "mb_num_max": 262144,
        }),
        CreateDecoder=mock.Mock(return_value=decoder),
        CreateDemuxer=mock.Mock(side_effect=lambda callback: FakeDemux(layout)),
        PacketData=lambda: "EOS",
    )
    backend._api, backend._owner = api, (os.getpid(), threading.get_ident())
    return backend, api, decoder


class NVDECTests(unittest.TestCase):
    def layout(self):
        return _hevc.video_layout(3, 30, 52, 320, "yuv444p")

    def test_construction_is_cpu_only_and_options_are_explicit(self):
        with mock.patch("torch.cuda._lazy_init", side_effect=AssertionError("unexpected GPU")):
            with NVDECDecoder(gpu_id=0, gpu_uuid=UUID) as backend:
                self.assertFalse(backend.stats()["initialized"])
            self.assertTrue(backend.stats()["closed"])
        for change in ({"gpu_id": -1}, {"gpu_id": True}, {"gpu_uuid": ""},
                       {"max_sessions": 0}, {"max_output_bytes": -1}, {"max_bitstream_bytes": True}):
            with self.assertRaises(ConfigurationError):
                NVDECDecoder(**{"gpu_id": 0, "gpu_uuid": UUID, **change})
        with self.assertRaises(ConfigurationError):
            HEVCProvider(nvdec=object())

    def test_real_host_plane_copy_shape_and_order(self):
        layout = self.layout()
        backend, _, decoder = fake_backend(layout)
        output, duration = backend.decode_pixels([b"fake-hevc"], layout)
        self.assertEqual(output.shape, (4680, 320))
        for index in range(3):
            self.assertTrue(np.all(output[index * 1560:(index + 1) * 1560] == index + 1))
        decoder.frames[0].array.fill(99)
        self.assertTrue(np.all(output[:1560] == 1))
        self.assertGreaterEqual(duration, 0)
        self.assertEqual(backend.stats()["packet_calls"], 1)
        self.assertEqual(backend.stats()["decoded_frames"], 3)
        backend.close()

    def test_sessions_reuse_and_evict_with_explicit_bound(self):
        layout = self.layout()
        backend, api, _ = fake_backend(layout, max_sessions=1)
        for _ in range(3):
            backend.decode_pixels([b"fake-hevc"], layout)
        self.assertEqual(api.CreateDecoder.call_count, 1)
        self.assertEqual(backend.stats()["session_hits"], 2)
        different = _hevc.video_layout(3, 30, 52, 576, "yuv444p")
        api.CreateDecoder.return_value = FakeDecoder(different)
        api.CreateDemuxer.side_effect = lambda callback: FakeDemux(different)
        backend.decode_pixels([b"fake-hevc"], different)
        self.assertEqual(backend.stats()["live_sessions"], 1)
        self.assertEqual(backend.stats()["evictions"], 1)
        backend.close()
        self.assertEqual(backend.stats()["live_sessions"], 0)

    def test_capability_failure_never_falls_back_or_constructs_decoder(self):
        layout = self.layout()
        backend, api, _ = fake_backend(layout)
        api.GetDecoderCaps.return_value["supported"] = 0
        with mock.patch.object(_hevc, "decode_pixels", side_effect=AssertionError("fallback")):
            with self.assertRaises(UnsupportedError):
                backend.decode_pixels([b"fake-hevc"], layout)
        api.CreateDecoder.assert_not_called()

    def test_incomplete_or_extra_frames_fail_and_discard_session(self):
        for length in (2, 4):
            layout = self.layout()
            backend, _, decoder = fake_backend(layout)
            decoder.frames = (decoder.frames * 2)[:length]
            with self.assertRaises(IntegrityError):
                backend.decode_pixels([b"fake-hevc"], layout)
            self.assertEqual(backend.stats()["live_sessions"], 0)
            self.assertEqual(backend.stats()["failures"], 1)

    def test_wrong_pitch_geometry_format_and_planes_reject_before_pointer_read(self):
        changes = (
            {"shape": (1, 2)}, {"strides": (999, 1)}, {"dtype": np.dtype("uint16")},
            {"format": "NV12"},
        )
        for change in changes:
            layout = self.layout()
            backend, _, decoder = fake_backend(layout)
            for name, value in change.items():
                setattr(decoder.frames[0], name, value)
            with mock.patch("reelcache.nvdec.ctypes.string_at", side_effect=AssertionError("unsafe read")):
                with self.assertRaises(IntegrityError):
                    backend.decode_pixels([b"fake-hevc"], layout)
        backend, _, decoder = fake_backend(self.layout())
        decoder.frames[0].GetPtrToPlane = lambda plane: 0
        with mock.patch("reelcache.nvdec.ctypes.string_at", side_effect=AssertionError("unsafe read")):
            with self.assertRaises(IntegrityError):
                backend.decode_pixels([b"fake-hevc"], self.layout())

    def test_input_and_output_admission_precede_hardware(self):
        for options in ({"max_output_bytes": 1}, {"max_bitstream_bytes": 1}):
            backend = NVDECDecoder(gpu_id=0, gpu_uuid=UUID, **options)
            with mock.patch.object(backend, "_initialize", side_effect=AssertionError("early hardware")):
                with self.assertRaises(CapacityError):
                    backend.decode_pixels([b"fake-hevc"], self.layout())
        backend = NVDECDecoder(gpu_id=0, gpu_uuid=UUID)
        for streams, layout in (([], self.layout()), ([b""], self.layout()),
                                ([b"x"], {**self.layout(), "planes": 1})):
            with self.assertRaises(IntegrityError):
                backend.decode_pixels(streams, layout)

    def test_decoder_thread_process_and_closed_lifetimes_are_enforced(self):
        backend, _, _ = fake_backend(self.layout())
        backend._owner = (os.getpid(), -1)
        with self.assertRaises(StateError):
            backend.decode_pixels([b"x"], self.layout())
        with self.assertRaises(StateError):
            backend.close()
        backend._owner = (os.getpid(), threading.get_ident())
        backend.close()
        with self.assertRaises(StateError):
            backend.decode_pixels([b"x"], self.layout())

    def test_backend_error_propagates_without_software_fallback(self):
        backend, _, decoder = fake_backend(self.layout())
        decoder.Decode = mock.Mock(side_effect=RuntimeError("NVDEC failed"))
        with mock.patch.object(_hevc, "decode_pixels", side_effect=AssertionError("fallback")):
            with self.assertRaisesRegex(RuntimeError, "NVDEC failed"):
                backend.decode_pixels([b"fake-hevc"], self.layout())
        self.assertEqual(backend.stats()["live_sessions"], 0)

    def test_provider_uses_nvdec_only_for_reload_not_worker_encoding_validation(self):
        source = torch.randn(24, 4, generator=torch.Generator().manual_seed(611))
        backend = NVDECDecoder(gpu_id=0, gpu_uuid=UUID)
        provider = HEVCProvider(nvdec=backend, validate_hevc_encode=True)
        with mock.patch.object(backend, "decode_pixels", wraps=_hevc.decode_pixels) as decode:
            packet = provider.encode_tensor(source, context(), CodecSpec("svd_hevc", 3))
            decode.assert_not_called()
            actual = provider.decode_tensor(packet)
            self.assertTrue(torch.equal(actual, decode_tensor(packet)))
            decode.assert_called_once()
            groups = provider.decode_column_views(packet, column_groups=[slice(0, 2), slice(2, 4)])
            self.assertTrue(torch.equal(groups, torch.stack([actual[:, :2], actual[:, 2:]])))
        self.assertFalse(backend.stats()["initialized"])
        with mock.patch.object(backend, "decode_pixels", side_effect=lambda streams, layout: (
            np.zeros((24, 3), dtype=np.uint8), 0.,
        )):
            with self.assertRaisesRegex(IntegrityError, "authoritative checksum"):
                provider.decode_tensor(packet)
        for kind in ("raw", "svd"):
            packet = encode_tensor(source, context(), CodecSpec(kind, rank=None if kind == "raw" else 3))
            with mock.patch.object(backend, "decode_pixels", side_effect=AssertionError("not HEVC")):
                provider.decode_tensor(packet)
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == "__main__":
    unittest.main()
