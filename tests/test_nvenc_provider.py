"""Step 15: explicit NVENC provider contract, tested on CPU with a software stand-in for NVENC.

The stand-in encodes each picture with libx265 (QP8, all-intra) and returns PyNvVideoCodec-shaped
packets, so the packed-stream plumbing, the encoder record, decode-side validation and every explicit
error can run in the no-skip CPU suite. Real NVENC behaviour (RTX PRO 6000 encode, NVDEC identity,
A100/H200 refusal) is measured by the GPU harness in
artifacts/sglang_integration/benchmarks/codec_hardware/ and by NVENCGPUTests below, which exists only
when REELCACHE_NVENC_GPU_TESTS=1 (so the CPU suite never collects or skips it).
"""

import copy
import os
import unittest

import numpy as np
import torch

from reelcache import _hevc
from reelcache import codecs
from reelcache import nvenc
from reelcache.errors import ConfigurationError, IntegrityError, UnsupportedError
from reelcache.packed_decode import prepare_packet
from reelcache.types import CodecSpec, EncodedTensor, MatrixKey, ModelIdentity, TensorContext

RTX_CAPS = {
    "width_min": 129, "width_max": 8192, "height_min": 33, "height_max": 8192, "mb_num_max": 262144,
    "support_yuv444_encode": 1, "support_lossless_encode": 1, "num_encoder_engines": 4, "level_max": 186,
}
DEVICE = {"gpu": "Fake NVENC GPU", "compute_capability": "12.0", "driver_version": "595.91.07"}


def timing_sei(counter):
    """A prefix SEI NAL with one pic_timing message (payload type 1) holding a per-session picture counter."""
    payload = bytes([0x11, 0x10 | (counter >> 8) & 0x0F, (counter & 0xFF) | 0x01])
    return b"\x00\x00\x01\x4e\x01" + bytes([1, len(payload)]) + payload + b"\x80"


def with_sei(access_unit, sei):
    """Insert an SEI NAL before the first slice of an access unit, as NVENC places its pic_timing SEI."""
    first_vcl = next(offset for offset, _, kind, _ in _hevc.nal_units(access_unit) if kind < 32)
    return access_unit[:first_vcl] + sei + access_unit[first_vcl:]


class FakeSession:
    """PyNvVideoCodec-shaped encoder session: one libx265 all-intra access unit per picture, prefixed (like
    NVENC) by a pic_timing SEI whose counter continues across EndEncode within the session."""

    def __init__(self, api, width, height, fmt, options):
        self.api, self.width, self.height, self.fmt, self.options = api, width, height, fmt, options
        self.pending, self.counter = [], 0

    def _picture(self, frame):
        frame = np.asarray(frame, dtype=np.uint8).reshape(-1)
        if self.fmt == "YUV444":
            return frame.reshape(1, 3, self.height, self.width), "yuv444p"
        return frame[:self.height * self.width].reshape(1, 1, self.height, self.width), "yuv420p"

    def Encode(self, frame):  # noqa: N802 - PyNvVideoCodec API
        picture, pix_fmt = self._picture(frame)
        key = (pix_fmt, picture.tobytes())
        if key not in self.api.cache:
            self.api.cache[key] = _hevc.encode_pictures(picture, pix_fmt)[0]
        data = self.api.mutate(with_sei(self.api.cache[key], timing_sei(self.counter)))
        self.counter += 1
        self.pending.append({"data": data, "timestamp": 0, "picture_type": 0})
        # NVENC returns packets with a delay; emulate a one-picture output queue.
        return [self.pending.pop(0)] if len(self.pending) > 1 else []

    def EndEncode(self):  # noqa: N802 - PyNvVideoCodec API
        out, self.pending = self.pending, []
        return out


class FakeAPI:
    __version__ = "fake-2.2"
    nvenc_api_version = 208

    def __init__(self, caps=None, *, fail=None, mutate=None):
        self.caps = dict(RTX_CAPS if caps is None else caps)
        self.fail, self.cache, self.sessions = fail, {}, []
        self.mutate = mutate or (lambda data: data)

    def GetEncoderCaps(self, *, gpuid, codec):  # noqa: N802 - PyNvVideoCodec API
        if self.fail is not None:
            raise RuntimeError(self.fail)
        assert codec == "hevc"
        return dict(self.caps)

    def CreateEncoder(self, width, height, fmt, usecpuinputbuffer, **options):  # noqa: N802
        assert usecpuinputbuffer is True
        session = FakeSession(self, width, height, fmt, options)
        self.sessions.append(session)
        return session


def encoder(api=None, **kwargs):
    return nvenc.NVENCEncoder(api=api or FakeAPI(), device=DEVICE, **kwargs)


def node(layers=2, ranks=(5, 7), spec=None, seed=0):
    model = ModelIdentity("nvenc-fixture", "fixture-15", layers, {"K": 12, "V": 16}, "tokens_channels", "float32")
    generator = torch.Generator().manual_seed(seed)
    matrices, contexts, specs = {}, {}, {}
    frames, height, width = 2, 6, 10
    for layer in range(layers):
        for target, rank in zip(("K", "V"), ranks):
            key = MatrixKey(layer, target)
            channels = model.channels_for(target)
            tokens = frames * height * width
            ramp = torch.linspace(-1, 1, tokens)[:, None]
            matrices[key] = (ramp * torch.randn(1, channels, generator=generator)
                             + torch.randn(tokens, 4, generator=generator) @ torch.randn(4, channels,
                                                                                          generator=generator))
            contexts[key] = TensorContext(model, key, 60, frames, height, width,
                                          "pre_rope" if target == "K" else "none")
            specs[key] = spec(rank + layer) if spec is not None else nvenc.nvenc_spec(rank + layer)
    return matrices, contexts, specs


class SpecContractTests(unittest.TestCase):
    def test_labels_round_trip_and_validation(self):
        spec = nvenc.nvenc_spec(12, qp=10, preset="p4", tuning="ull", pix_fmt="yuv420p")
        self.assertEqual((spec.preset, spec.qp, spec.inter_prediction), ("nvenc-p4-ull", 10, False))
        self.assertEqual(nvenc.spec_settings(spec), nvenc.NVENCSettings("p4", "ull", 10))
        codecs._validate_spec(spec)
        self.assertEqual(CodecSpec.from_dict(spec.to_dict()), spec)
        options = nvenc.NVENCSettings("p4", "ull", 10).options("yuv420p")
        self.assertEqual((options["rc"], options["constqp"], options["gop"], options["idrperiod"], options["bf"]),
                         ("constqp", "10", "1", "1", "0"))
        with self.assertRaises(ConfigurationError):
            nvenc.NVENCSettings("p8", "hq", 8)
        with self.assertRaises(ConfigurationError):
            nvenc.NVENCSettings("p1", "fast", 8)
        with self.assertRaises(ConfigurationError):
            nvenc.NVENCSettings("p1", "hq", 52)
        with self.assertRaises(ConfigurationError):
            nvenc.NVENCSettings("p1", "lossless", 8)
        nvenc.nvenc_spec(4, qp=0, tuning="lossless")
        with self.assertRaises(UnsupportedError):
            nvenc.spec_settings(CodecSpec("svd_hevc", 4))
        for bad in (CodecSpec("svd_hevc", 4, preset="nvenc-p9-hq"), CodecSpec("svd", 4, preset="nvenc-p1-hq"),
                    CodecSpec("svd_hevc", 4, preset="nvenc-p1-hq", inter_prediction=True),
                    CodecSpec("svd_hevc", 4, preset="nvenc-p1-hq", pix_fmt="yuv422p")):
            with self.assertRaises(UnsupportedError):
                codecs._validate_spec(bad)

    def test_libx265_paths_never_encode_nvenc_specifications(self):
        matrices, contexts, specs = node(layers=1)
        key = MatrixKey(0, "K")
        with self.assertRaisesRegex(UnsupportedError, "libx265 never substitutes"):
            codecs.encode_tensor(matrices[key], contexts[key], specs[key])
        for packing in ("node", "per_matrix"):
            with self.assertRaisesRegex(UnsupportedError, "libx265 never substitutes"):
                codecs.encode_node(matrices, contexts, specs, packing=packing)
        with self.assertRaisesRegex(UnsupportedError, "libx265 never substitutes"):
            codecs.HEVCProvider().encode_node(matrices, contexts, specs)

    def test_nvenc_encoder_never_encodes_libx265_or_other_settings(self):
        fake = encoder(qualify=False)
        x265 = node(layers=1, spec=lambda rank: CodecSpec("svd_hevc", rank))
        with self.assertRaises(UnsupportedError):
            codecs.encode_node(*x265, encoder=fake)
        other = node(layers=1, spec=lambda rank: nvenc.nvenc_spec(rank, qp=12))
        with self.assertRaisesRegex(ConfigurationError, "this encoder is"):
            codecs.encode_node(*other, encoder=fake)
        with self.assertRaisesRegex(ConfigurationError, "node or layer_group"):
            codecs.encode_node(*node(layers=1), packing="per_matrix", encoder=fake)
        only444 = encoder(qualify=False, pix_fmts=("yuv444p",))
        with self.assertRaises(UnsupportedError):
            codecs.encode_node(*node(layers=1, spec=lambda rank: nvenc.nvenc_spec(rank, pix_fmt="yuv420p")),
                               encoder=only444)


class DeviceTests(unittest.TestCase):
    def test_devices_without_nvenc_are_explicit_errors(self):
        a100 = FakeAPI(fail="PyNvEncoderCaps : Error code : 2 nvEncOpenEncodeSessionEx returned error 2")
        with self.assertRaisesRegex(UnsupportedError, "unavailable.*libx265 HEVCProvider"):
            encoder(a100)
        with self.assertRaisesRegex(UnsupportedError, "4:4:4"):
            encoder(FakeAPI({**RTX_CAPS, "support_yuv444_encode": 0}))
        encoder(FakeAPI({**RTX_CAPS, "support_yuv444_encode": 0}), pix_fmts=("yuv420p",), qualify=False)
        with self.assertRaisesRegex(UnsupportedError, "lossless"):
            encoder(FakeAPI({**RTX_CAPS, "support_lossless_encode": 0}), qp=0, tuning="lossless")
        with self.assertRaises(UnsupportedError):
            encoder(pix_fmts=("yuv422p",))

    def test_real_construction_without_a_usable_encoder_raises(self):
        # The CPU suite hides CUDA (and smp nodes have no NVIDIA driver): construction must refuse explicitly.
        # With a visible GPU this is covered by NVENCGPUTests (RTX encodes, A100/H200 refuse).
        if not torch.cuda.is_available():
            with self.assertRaises(UnsupportedError):
                nvenc.NVENCEncoder()

    def test_geometry_limits_follow_the_device_caps(self):
        fake = encoder(qualify=False)
        fake.check_geometry(160, 160, "yuv444p")
        for width, height, pix_fmt in ((128, 160, "yuv444p"), (160, 32, "yuv444p"), (8224, 160, "yuv444p"),
                                       (161, 160, "yuv420p")):
            with self.assertRaises(UnsupportedError):
                fake.check_geometry(width, height, pix_fmt)
        tight = encoder(FakeAPI({**RTX_CAPS, "mb_num_max": 99}), qualify=False)
        with self.assertRaisesRegex(UnsupportedError, "macroblocks"):
            tight.check_geometry(160, 160, "yuv444p")
        with self.assertRaises(UnsupportedError):
            fake.encode_pictures(np.zeros((1, 3, 160, 128), dtype=np.uint8), "yuv444p")
        with self.assertRaises(ConfigurationError):
            fake.encode_pictures(np.zeros((1, 1, 160, 160), dtype=np.uint8), "yuv444p")


class QualificationTests(unittest.TestCase):
    def test_qualification_records_the_real_bitstream_contract(self):
        fake = encoder()
        report = fake.qualification
        self.assertEqual(sorted(report["formats"]), ["yuv420p", "yuv444p"])
        for row in report["formats"].values():
            self.assertEqual(row["slice_qps"], [8])
            self.assertTrue(row["deterministic"])
        self.assertEqual(report["formats"]["yuv444p"]["chroma_format_idc"], [3])

    def test_qualification_rejects_a_qp_the_device_does_not_honour(self):
        with self.assertRaisesRegex(UnsupportedError, "constant-QP contract"):
            encoder(qp=12)  # the stand-in always emits QP8 slices

    def test_qualification_rejects_nondeterministic_or_non_self_contained_streams(self):
        counter = {"n": 0}

        def flaky(data):
            counter["n"] += 1
            return data + bytes([0, 0, 1, 0x50, 0x01, 5, 1, counter["n"] % 251 + 1, 0x80])  # changing suffix SEI (type 40)

        with self.assertRaisesRegex(UnsupportedError, "deterministic"):
            encoder(FakeAPI(mutate=flaky))

        def no_parameter_sets(data):
            units = _hevc.nal_units(data)
            return b"".join(data[start:start + length] for start, length, kind, _ in units if kind not in (32, 33, 34))

        with self.assertRaises(IntegrityError):
            encoder(FakeAPI(mutate=no_parameter_sets))

    def test_timing_sei_is_dropped_and_sessions_persist(self):
        au = _hevc.encode_pictures(np.full((1, 3, 160, 160), 77, dtype=np.uint8), "yuv444p")[0]
        sei = timing_sei(5)
        self.assertEqual(nvenc.sei_payload_types(sei), [1])
        stripped, dropped = nvenc.drop_timing_sei(with_sei(au, sei))
        self.assertEqual((stripped, dropped), (au, 1))
        user = b"\x00\x00\x01\x4e\x01\x05\x01\x42\x80"  # user_data_unregistered-like message: kept
        self.assertEqual(nvenc.sei_payload_types(user), [5])
        self.assertEqual(nvenc.drop_timing_sei(with_sei(au, user)), (with_sei(au, user), 0))
        api = FakeAPI()
        fake = encoder(api, qualify=False)
        pictures = np.random.default_rng(3).integers(0, 255, size=(3, 3, 160, 160), dtype=np.uint8)
        first = fake.encode_pictures(pictures, "yuv444p")[0]
        second = fake.encode_pictures(pictures, "yuv444p")[0]
        self.assertEqual(first, second)  # the reused session's counters do not leak into the bytes
        self.assertEqual(len(api.sessions), 1)
        self.assertEqual(fake.stats["dropped_timing_sei"], 6)
        fake.close()
        self.assertEqual(fake.encode_pictures(pictures, "yuv444p")[0], first)
        self.assertEqual(len(api.sessions), 2)

    def test_split_sessions_are_checked_for_byte_identity(self):
        fake = encoder(sessions=2)
        self.assertTrue(fake.qualification["multi_session_identical"])
        pictures = np.random.default_rng(1).integers(0, 255, size=(5, 3, 160, 192), dtype=np.uint8)
        split = fake.encode_pictures(pictures, "yuv444p")[0]
        fake.sessions = 1
        self.assertEqual(split, fake.encode_pictures(pictures, "yuv444p")[0])


class PackedRoundTripTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.encoder = encoder(qualify=False)
        cls.provider = nvenc.NVENCProvider(cls.encoder)
        cls.matrices, cls.contexts, cls.specs = node()
        cls.encoding = cls.provider.encode_node(cls.matrices, cls.contexts, cls.specs, node_id="n15")
        cls.packet = cls.encoding.packets[0]

    def test_packet_records_the_encoder_identity(self):
        metadata = self.packet.metadata
        self.assertEqual(len(self.encoding.packets), 1)
        self.assertEqual(metadata["schema"], codecs.PACK_SCHEMA)
        self.assertEqual(metadata["codec"], {"kind": "svd_hevc", "pix_fmt": "yuv444p", "qp": 8,
                                             "preset": "nvenc-p1-hq", "inter_prediction": False})
        record = metadata["hevc"]
        self.assertEqual((record["schema"], record["encoder"], record["rate_control"], record["picture_structure"]),
                         (nvenc.ENCODER_SCHEMA, "nvenc", "constqp", "all_intra_idr"))
        self.assertEqual(record["implementation"]["gpu"], DEVICE["gpu"])
        self.assertEqual(record["implementation"]["library_version"], "fake-2.2")
        self.assertEqual(record["options"], nvenc.NVENCSettings().options("yuv444p"))
        self.assertEqual(record["post_processing"], "drop_hrd_timing_sei")
        self.assertEqual(metadata["packing"]["random_access"], "tile_byte_ranges")
        self.assertFalse(self.packet.metrics["hevc_software_only"])
        self.assertEqual(self.packet.metrics["hevc_encoder"], "nvenc")
        self.assertIsNone(self.packet.metrics["ffmpeg_version"])

    def test_nvenc_packets_decode_on_the_ordinary_paths(self):
        pack = codecs._validate_pack(self.packet, verify=True)
        canvas = pack["canvas"]
        self.assertGreaterEqual(min(canvas["width"], canvas["height"]), _hevc.MIN_CANVAS)
        decoded = self.provider.decode_packed(self.packet)
        images, _ = _hevc.decode_pictures(pack["stream"], pictures=canvas["pictures"], height=canvas["height"],
                                          width=canvas["width"], pix_fmt=canvas["pix_fmt"])
        for index, key in enumerate(pack["keys"]):
            tile = pack["tiles"][index][1]
            pixels = codecs._unpack_tile(images, tile, tile["pictures"][0], tile["video"])
            affine = np.frombuffer(self.packet.payloads[tile["affine"]], dtype="<f4").reshape(
                tile["normalization"]["affine_shape"])
            expected = _hevc.dequantize_coefficients(pixels, affine, tile["normalization"])
            basis, coefficients = codecs._pack_factors(self.packet, pack, [index], verify_decoded_coefficients=False,
                                                       verify_packet=True)[0]
            np.testing.assert_array_equal(coefficients.numpy(), expected)
            self.assertEqual(tuple(decoded[key[0]].shape), tuple(self.matrices[key[0]].shape))
        prepared = prepare_packet(self.packet, verify=True)
        self.assertEqual(prepared.pictures, canvas["pictures"])
        self.assertTrue(prepared.random_access)

    def test_validation_decode_checksums_hold(self):
        provider = nvenc.NVENCProvider(self.encoder, validate_hevc_encode=True, verify_decoded_coefficients=True)
        packet = provider.encode_node(self.matrices, self.contexts, self.specs).packets[0]
        self.assertTrue(all(tile["decoded_coefficients_sha256"] for tile in packet.metadata["tiles"]))
        provider.decode_packed(packet)

    def test_tampered_encoder_records_are_rejected(self):
        def tamper(edit):
            metadata = copy.deepcopy(dict(self.packet.metadata))
            edit(metadata)
            return EncodedTensor(metadata, dict(self.packet.payloads), None, None)

        edits = {
            "qp": lambda m: m["hevc"].__setitem__("qp", 9),
            "options": lambda m: m["hevc"]["options"].__setitem__("gop", "250"),
            "implementation": lambda m: m["hevc"]["implementation"].pop("driver_version"),
            "schema": lambda m: m["hevc"].__setitem__("schema", "reelcache.hevc_encoder.v2"),
            "codec_preset": lambda m: m["codec"].__setitem__("preset", "ultrafast"),
            "x265_record": lambda m: m.__setitem__("hevc", _hevc.encoder_record(False)),
            "inter": lambda m: m["hevc"].__setitem__("inter_prediction", True),
            "post_processing": lambda m: m["hevc"].__setitem__("post_processing", "none"),
        }
        for name, edit in edits.items():
            with self.subTest(name), self.assertRaises(IntegrityError):
                codecs._validate_pack(tamper(edit), verify=True)
        x265 = codecs.encode_node(*node(layers=1, spec=lambda rank: CodecSpec("svd_hevc", rank))).packets[0]
        metadata = copy.deepcopy(dict(x265.metadata))
        metadata["hevc"] = self.packet.metadata["hevc"]
        with self.assertRaises(IntegrityError):
            codecs._validate_pack(EncodedTensor(metadata, dict(x265.payloads), None, None), verify=True)

    def test_provider_rules(self):
        key = MatrixKey(0, "K")
        with self.assertRaisesRegex(UnsupportedError, "packed node streams"):
            self.provider.encode_tensor(self.matrices[key], self.contexts[key], self.specs[key])
        raw = self.provider.encode_tensor(self.matrices[key], self.contexts[key], CodecSpec("raw"))
        self.assertEqual(raw.metadata["kind"], "raw")
        with self.assertRaises(ConfigurationError):
            nvenc.NVENCProvider(self.encoder, packing="per_matrix")
        with self.assertRaises(ConfigurationError):
            nvenc.NVENCProvider(object())
        groups = nvenc.NVENCProvider(self.encoder, packing="layer_group", layer_group_size=1).encode_node(
            self.matrices, self.contexts, self.specs)
        self.assertEqual(len(groups.packets), 2)
        self.assertEqual(self.provider.spec(9), nvenc.nvenc_spec(9))


if os.environ.get("REELCACHE_NVENC_GPU_TESTS") == "1":
    class NVENCGPUTests(unittest.TestCase):
        """Real device behaviour; run inside GPU jobs only (collected only with REELCACHE_NVENC_GPU_TESTS=1)."""

        def test_device_contract(self):
            import ctypes

            name = torch.cuda.get_device_name(0)
            torch.zeros(1, device="cuda")
            library = ctypes.CDLL("libcuda.so.1")
            before, after = ctypes.c_void_p(), ctypes.c_void_p()
            library.cuCtxGetCurrent(ctypes.byref(before))
            if "A100" in name or "H200" in name or "H100" in name:
                with self.assertRaisesRegex(UnsupportedError, "unavailable"):
                    nvenc.NVENCEncoder()
                # The failed probe must not leave another CUDA context current (it broke NVDEC sessions before).
                library.cuCtxGetCurrent(ctypes.byref(after))
                self.assertEqual(before.value, after.value)
                from reelcache.nvdec import CanvasSession
                stream = torch.cuda.Stream()
                with torch.cuda.stream(stream):
                    CanvasSession(gpu_id=0, width=256, height=256, pix_fmt="yuv444p").close()
                self.assertEqual(float(torch.ones(4, device="cuda").sum()), 4.0)
                return
            real = nvenc.NVENCEncoder()
            self.assertEqual(real.qualification["formats"]["yuv444p"]["slice_qps"], [8])
            provider = nvenc.NVENCProvider(real, validate_hevc_encode=True, verify_decoded_coefficients=True)
            matrices, contexts, specs = node(layers=3)
            packet = provider.encode_node(matrices, contexts, specs).packets[0]
            self.assertEqual(packet.metadata["hevc"]["implementation"]["gpu"], name)
            provider.decode_packed(packet)
            from reelcache.nvdec import NVDECDecoder
            uuid = "GPU-" + str(torch.cuda.get_device_properties(0).uuid).removeprefix("GPU-")
            device = codecs.decode_packed(packet, device="cuda", nvdec=NVDECDecoder(gpu_id=0, gpu_uuid=uuid,
                                                                                    device_output=True))
            software = codecs.decode_packed(packet)
            for key in software:
                torch.testing.assert_close(device[key].cpu(), software[key], rtol=0, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
