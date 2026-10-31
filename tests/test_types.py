from dataclasses import FrozenInstanceError
import unittest

from reelcache.errors import ConfigurationError
from reelcache.types import CodecSpec, EncodedTensor, MatrixKey, ModelIdentity, TensorContext
from tests.core_support import model


class TypesTests(unittest.TestCase):
    def test_identity_supports_asymmetric_full_channels_and_roundtrip(self):
        identity = model(channels={"K": 8, "V": 12}, dtype="torch.bfloat16")
        self.assertEqual(identity.channels_for("K"), 8)
        self.assertEqual(identity.channels_for(MatrixKey(1, "V")), 12)
        self.assertEqual(identity.dtype, "bfloat16")
        self.assertEqual(ModelIdentity.from_dict(identity.to_dict()), identity)
        self.assertEqual(hash(ModelIdentity.from_dict(identity.to_dict())), hash(identity))
        with self.assertRaises(TypeError):
            identity.kv_channels["K"] = 1
        with self.assertRaises(FrozenInstanceError):
            identity.layers = 3

    def test_identity_validation(self):
        base = model().to_dict()
        for changes in (
            {"name": ""}, {"fingerprint": ""}, {"layers": 0}, {"layers": True},
            {"kv_channels": 0}, {"kv_channels": {"K": 4}}, {"kv_channels": {"K": 4, "V": -1}},
            {"layout": ""}, {"dtype": "int8"},
        ):
            with self.subTest(changes=changes), self.assertRaises(ConfigurationError):
                ModelIdentity(**{**base, **changes})

    def test_explicit_target_and_per_matrix_dtype_contracts(self):
        identity = model(dtype={"K": "float32", "V": "bf16"})
        self.assertEqual(identity.dtype_for("K"), "float32")
        self.assertEqual(identity.dtype_for(MatrixKey(1, "V")), "bfloat16")
        self.assertEqual(ModelIdentity.from_dict(identity.to_dict()), identity)
        self.assertEqual(hash(ModelIdentity.from_dict(identity.to_dict())), hash(identity))
        with self.assertRaises(TypeError):
            identity.dtype["K"] = "float16"
        mixed = model(dtype={"0:K": "bfloat16", "0:V": "bfloat16",
                             "1:K": "float32", "1:V": "bfloat16"})
        self.assertEqual(mixed.dtype_for(MatrixKey(0, "K")), "bfloat16")
        self.assertEqual(mixed.dtype_for(MatrixKey(1, "K")), "float32")
        context = TensorContext(mixed, MatrixKey(1, "K"), 0, 1, 2, 2, "pre_rope")
        self.assertEqual(context.dtype, "float32")
        with self.assertRaises(ConfigurationError):
            mixed.dtype_for("K")
        for dtype in ({"K": "float32"}, {"K": "float32", "V": "int8"}, {"0:K": "float32"}):
            with self.subTest(dtype=dtype), self.assertRaises(ConfigurationError):
                model(dtype=dtype)

    def test_matrix_key_validation_and_ordering(self):
        self.assertLess(MatrixKey(0, "K"), MatrixKey(0, "V"))
        self.assertEqual(MatrixKey.from_dict({"layer": 1, "target": "K"}), MatrixKey(1, "K"))
        for args in ((-1, "K"), (True, "K"), (0, "k"), (0, "Q")):
            with self.subTest(args=args), self.assertRaises(ConfigurationError):
                MatrixKey(*args)

    def test_context_shape_positions_and_roundtrip(self):
        context = TensorContext(model(), MatrixKey(0, "K"), 12, 2, 3, 4, "pre_rope")
        self.assertEqual(context.shape, (24, 4))
        self.assertEqual(context.token_end, 36)
        self.assertEqual(TensorContext.from_dict(context.to_dict()), context)

    def test_explicit_source_dtype_preserves_native_identity_and_legacy_headers(self):
        identity = model(dtype="bfloat16")
        native = TensorContext(identity, MatrixKey(0, "K"), 0, 1, 2, 2, "post_rope")
        source = TensorContext(identity, MatrixKey(0, "K"), 0, 1, 2, 2, "pre_rope",
                               source_dtype="torch.float32")
        self.assertEqual(native.dtype, "bfloat16")
        self.assertNotIn("source_dtype", native.to_dict())
        self.assertEqual(source.dtype, "float32")
        self.assertEqual(source.model.dtype, "bfloat16")
        self.assertEqual(source.to_dict()["source_dtype"], "float32")
        self.assertEqual(TensorContext.from_dict(source.to_dict()), source)
        self.assertEqual(TensorContext.from_dict(native.to_dict()), native)
        with self.assertRaises(ConfigurationError):
            TensorContext(identity, MatrixKey(0, "K"), 0, 1, 2, 2, "pre_rope", source_dtype="int8")

    def test_context_rejects_invalid_domains(self):
        identity = model()
        args = [identity, MatrixKey(0, "K"), 0, 1, 2, 2, "pre_rope"]
        for index, value in ((1, MatrixKey(2, "K")), (2, -1), (3, 0), (4, True), (6, "arbitrary")):
            changed = args.copy()
            changed[index] = value
            with self.subTest(index=index), self.assertRaises(ConfigurationError):
                TensorContext(*changed)
        with self.assertRaises(ConfigurationError):
            TensorContext(identity, MatrixKey(0, "V"), 0, 1, 2, 2, "post_rope")

    def test_codec_defaults_and_validation(self):
        self.assertEqual(CodecSpec().kind, "raw")
        spec = CodecSpec("svd_hevc", rank=2, normalization="leading_columns", protected_fraction=.5,
                         ffmpeg_path="tools/ffmpeg")
        self.assertEqual(CodecSpec.from_dict(spec.to_dict()), spec)
        for kwargs in (
            {"kind": "unknown"}, {"kind": "svd"}, {"kind": "svd", "rank": 0},
            {"rank": 2}, {"normalization": "greedy"}, {"protected_fraction": -.01},
            {"protected_fraction": 1.01}, {"protected_fraction": False},
            {"protected_fraction": float("nan")}, {"qp": 52}, {"qp": True}, {"preset": ""},
            {"ffmpeg_path": ""}, {"ffmpeg_path": 123}, {"ffmpeg_path": "bad\x00path"},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ConfigurationError):
                CodecSpec(**kwargs)

    def test_zero_protected_fraction_is_an_explicit_supported_endpoint(self):
        for normalization in ("global", "per_channel", "leading_columns"):
            spec = CodecSpec("svd_hevc", rank=2, normalization=normalization, protected_fraction=0)
            self.assertEqual(spec.protected_fraction, 1 if normalization == "per_channel" else 0)
            self.assertEqual(CodecSpec.from_dict(spec.to_dict()), spec)
        self.assertEqual(CodecSpec(normalization="global").protected_fraction, 0.0)
        self.assertEqual(CodecSpec(normalization="per_channel", protected_fraction=.2).protected_fraction, 1.0)
        self.assertEqual(CodecSpec(normalization="leading_columns", protected_fraction=.2).protected_fraction, .2)
        for mode in ("global", "per_channel"):
            with self.subTest(mode=mode), self.assertRaises(ConfigurationError):
                CodecSpec(normalization=mode, protected_fraction=-1)

    def test_encoded_copies_metadata_and_freezes_payloads(self):
        metadata = {"shape": [2, 4]}
        payloads = {"data": b"bytes"}
        encoded = EncodedTensor(metadata, payloads, metrics={"stored_bytes": 5})
        metadata["shape"][0] = 99
        payloads.clear()
        self.assertEqual(encoded.metadata["shape"], [2, 4])
        self.assertEqual(encoded.payloads["data"], b"bytes")
        with self.assertRaises(TypeError):
            encoded.payloads["data"] = b"changed"
        with self.assertRaises(FrozenInstanceError):
            encoded.basis = b"changed"

    def test_encoded_rejects_pickle_objects_and_nonfinite_json(self):
        for metadata in ({1: "bad"}, {"bad": object()}, {"bad": float("inf")}):
            with self.subTest(metadata=metadata), self.assertRaises(ConfigurationError):
                EncodedTensor(metadata, {"data": b""})
        for payloads in ({}, {"": b"data"}, {"data": bytearray(b"data")}):
            with self.subTest(payloads=payloads), self.assertRaises(ConfigurationError):
                EncodedTensor({}, payloads)


if __name__ == "__main__":
    unittest.main()
