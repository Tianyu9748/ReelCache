import hashlib
import struct
import unittest
from unittest.mock import patch

import torch

from reelcache.errors import CapacityError, ConfigurationError, IntegrityError
from reelcache.serialization import (
    canonical_json, deserialize_tensors, parse_json, serialize_tensors,
    tensor_bytes, tensor_from_bytes,
)


class SerializationTests(unittest.TestCase):
    def test_raw_bits_including_bfloat16_nan_and_signed_zero(self):
        words = torch.tensor([0x0000, 0x8000, 0x7FC1, 0x7F81, 0x3F80, 0xFF80], dtype=torch.uint16)
        values = {"bf16": words.view(torch.bfloat16), "rng": torch.arange(32, dtype=torch.uint8)}
        encoded = serialize_tensors(values)
        decoded = deserialize_tensors(encoded)
        for name in values:
            self.assertEqual(decoded[name].dtype, values[name].dtype)
            self.assertEqual(tensor_bytes(decoded[name]), tensor_bytes(values[name]))

    def test_scalars_empty_noncontiguous_and_complex(self):
        tensors = {
            "scalar": torch.tensor(3, dtype=torch.int64),
            "empty": torch.empty((2, 0, 3), dtype=torch.bfloat16),
            "transpose": torch.arange(12, dtype=torch.float64).reshape(3, 4).T,
            "complex": torch.tensor([1 + 2j, 3 - 4j]).conj(),
            "bool": torch.tensor([True, False]),
        }
        result = deserialize_tensors(serialize_tensors(tensors))
        for name, tensor in tensors.items():
            self.assertTrue(torch.equal(result[name], tensor), name)
            self.assertEqual(result[name].shape, tensor.shape)
            self.assertEqual(result[name].dtype, tensor.dtype)

    def test_serialization_is_deterministic_and_non_aliasing(self):
        tensor = torch.arange(4, dtype=torch.float32)
        first = serialize_tensors({"b": tensor, "a": tensor})
        second = serialize_tensors({"a": tensor, "b": tensor})
        self.assertEqual(first, second)
        restored = deserialize_tensors(first)
        restored["a"][0] = 100
        self.assertEqual(restored["b"][0].item(), 0)
        self.assertEqual(tensor[0].item(), 0)

    def test_roundtrip_all_native_state_dtypes(self):
        for dtype in (torch.int8, torch.uint8, torch.int16, torch.int32, torch.int64,
                      torch.float16, torch.bfloat16, torch.float32, torch.float64,
                      torch.complex64, torch.complex128):
            with self.subTest(dtype=dtype):
                value = torch.tensor([0, 1, 2], dtype=dtype)
                self.assertEqual(
                    tensor_bytes(value),
                    tensor_bytes(deserialize_tensors(serialize_tensors({"v": value}))["v"]),
                )

    def test_rejects_truncated_corrupt_and_pickle_state(self):
        data = serialize_tensors({"x": torch.arange(5)})
        for broken in (b"\x80\x04cposix\nsystem\n.", data[:10], data[:-1], data + b"extra",
                       data[:-1] + bytes([data[-1] ^ 1])):
            with self.subTest(size=len(broken)), self.assertRaises(IntegrityError):
                deserialize_tensors(broken)

    def test_budget_checked_before_tensor_allocation(self):
        data = serialize_tensors({"x": torch.zeros(100)})
        with patch("reelcache.serialization.tensor_from_bytes") as allocate:
            with self.assertRaises(CapacityError):
                deserialize_tensors(data, max_bytes=399)
            allocate.assert_not_called()

    def test_malformed_header_does_not_allocate(self):
        magic = b"REELCACHE_TENSORS\x00\x01"
        entry = {"dtype": "float32", "shape": [10**12], "offset": 0,
                 "nbytes": 4, "sha256": hashlib.sha256(b"1234").hexdigest()}
        for changes in ({}, {"offset": 1}, {"shape": [-1]}, {"shape": [0, 10**100]},
                        {"dtype": "object"}, {"nbytes": True}):
            header = canonical_json({"version": 1, "byteorder": "little",
                                     "tensors": {"x": {**entry, **changes}}})
            data = magic + struct.pack("<Q", len(header)) + header + b"1234"
            with self.subTest(changes=changes), self.assertRaises(IntegrityError):
                deserialize_tensors(data)

    def test_rejects_non_tensor_state_and_non_json(self):
        with self.assertRaises(ConfigurationError):
            serialize_tensors({"object": object()})
        with self.assertRaises(ConfigurationError):
            serialize_tensors({"": torch.tensor(1)})
        with self.assertRaises(ConfigurationError):
            canonical_json({"x": float("nan")})
        for data in (b'{"x":1,"x":2}', b'{"x":NaN}', b"\xff"):
            with self.assertRaises(IntegrityError):
                parse_json(data)

    def test_direct_tensor_decoder_validates_exact_length(self):
        with self.assertRaises(IntegrityError):
            tensor_from_bytes(b"123", dtype="float32", shape=(1,))
        self.assertEqual(deserialize_tensors(serialize_tensors({})), {})


if __name__ == "__main__":
    unittest.main()
