"""Accel track C (C2 ``gil_fastpaths``): every fast path returns exactly what the base function returns (or raises
the same error), on synthetic edge cases and on real libx265 streams/packs encoded here."""

from __future__ import annotations

import hashlib
import json
import math
import random
import unittest
from types import MappingProxyType

import numpy as np

from reelcache import _hevc, accel_c, nvenc, types
from reelcache.codecs import _unpack_tile
from reelcache.errors import ConfigurationError, IntegrityError

FAST = accel_c.AccelOptions(gil_fastpaths=True)


def both(function, *args, **kwargs):
    """``(base, fast)`` outcomes of ``function``: ``("ok", value)`` or ``("raise", type, message)``."""
    outcomes = []
    for options in (accel_c.OFF, FAST):
        with accel_c.scope(options):
            try:
                outcomes.append(("ok", function(*args, **kwargs)))
            except Exception as exc:  # noqa: BLE001 - compared
                outcomes.append(("raise", type(exc), str(exc)))
    return outcomes


def nal(kind, payload, *, long_start=False, layer=0, tid=1):
    header = bytes(((kind << 1) | (layer >> 5), ((layer & 0x1F) << 3) | tid))
    return (b"\x00\x00\x00\x01" if long_start else b"\x00\x00\x01") + header + payload


def escape(payload):
    """H.265 emulation prevention (00 00 0x, x <= 3 -> 00 00 03 0x)."""
    out, zeros = bytearray(), 0
    for byte in payload:
        if zeros >= 2 and byte <= 3:
            out.append(3)
            zeros = 0
        out.append(byte)
        zeros = zeros + 1 if byte == 0 else 0
    return bytes(out)


def sei(messages):
    body = bytearray()
    for kind, size_payload in messages:
        value = kind
        while value >= 255:
            body.append(255)
            value -= 255
        body.append(value)
        size = len(size_payload)
        while size >= 255:
            body.append(255)
            size -= 255
        body.append(size)
        body += size_payload
    body.append(0x80)
    return nal(39, escape(bytes(body)))


def random_stream(rng, pictures):
    units = []
    for index in range(pictures):
        units.append(nal(32, escape(bytes(rng.randrange(4) for _ in range(rng.randrange(4, 40)))), long_start=True))
        units.append(nal(33, escape(bytes(rng.randrange(256) for _ in range(rng.randrange(8, 60))))))
        units.append(nal(34, escape(bytes(rng.randrange(256) for _ in range(rng.randrange(4, 20))))))
        if index % 2:
            units.append(sei([(1, bytes(rng.randrange(3) for _ in range(rng.randrange(1, 9))))]))
        slice_body = bytes([0x80 | rng.randrange(128)]) + bytes(
            rng.choice((0, 0, 0, 1, 2, 3, rng.randrange(256))) for _ in range(rng.randrange(20, 400)))
        units.append(nal(19, escape(slice_body)))
    return b"".join(units)


class NalParsingTests(unittest.TestCase):
    def test_nal_units_and_access_units_match_on_synthetic_streams(self):
        rng = random.Random(5)
        for trial in range(40):
            stream = random_stream(rng, rng.randrange(1, 12))
            base, fast = both(_hevc.nal_units, stream)
            self.assertEqual(base, fast, trial)
            self.assertEqual(both(_hevc.access_units, stream)[0], both(_hevc.access_units, stream)[1])

    def test_errors_are_identical(self):
        rng = random.Random(9)
        good = random_stream(rng, 2)
        cases = [b"", bytearray(good), b"\x01\x02" + good, good[:-1] + b"\x00\x00\x01\x40",
                 b"\x00\x00\x01\xc0\x01" + good, b"\x00\x00\x01\x40\x00" + good, b"\x00\x00\x01\x26\x01",
                 b"\x00\x00\x01\x02", "text"]
        for case in cases:
            base, fast = both(_hevc.nal_units, case)
            self.assertEqual(base, fast, case[:12])
            self.assertEqual(base[0], "raise", case[:12])

    def test_real_libx265_stream(self):
        rng = np.random.default_rng(3)
        pictures = rng.integers(0, 256, size=(5, 3, 64, 96), dtype=np.uint8)
        stream, _, _ = _hevc.encode_pictures(pictures, "yuv444p")
        for function, args in ((_hevc.nal_units, (stream,)), (_hevc.all_intra_access_units, (stream, 5)),
                               (nvenc.verify_stream, (stream, 5))):
            base, fast = both(function, *args)
            self.assertEqual(base, fast, function.__name__)
            self.assertEqual(base[0], "ok")

    def test_start_code_positions_find_every_occurrence(self):
        rng = random.Random(1)
        for _ in range(50):
            data = bytes(rng.choice((0, 0, 0, 1, 3, 7)) for _ in range(rng.randrange(0, 300)))
            expected = [match.start() for match in _hevc._START_CODE.finditer(data)]
            self.assertEqual(accel_c.start_code_positions(data), expected)


class SeiTests(unittest.TestCase):
    def test_rbsp_matches_the_byte_loop(self):
        rng = random.Random(11)
        samples = [b"", b"\x03", b"\x00\x00\x03", b"\x00\x00\x00\x03\x03", b"\x00\x00\x03\x00\x00\x03",
                   b"\x00\x03\x00\x00\x03\x01", b"\x00\x00\x03\x03\x00\x00\x03"]
        samples += [bytes(rng.choice((0, 0, 0, 3, 3, 1, 255)) for _ in range(rng.randrange(0, 64)))
                    for _ in range(2000)]
        for sample in samples:
            base, fast = both(nvenc._rbsp, sample)
            self.assertEqual(base, fast, sample)

    def test_sei_payload_types_and_timing_drop(self):
        rng = random.Random(4)
        for _ in range(200):
            messages = [(rng.choice((0, 1, 5, 137, 300)), bytes(rng.choice((0, 0, 3, 9)) for _ in range(
                rng.randrange(0, 300)))) for _ in range(rng.randrange(1, 4))]
            unit = (nal(35, b"\x10", long_start=True) + sei(messages) + nal(19, escape(b"\x80" + bytes(
                rng.randrange(256) for _ in range(50)))))
            self.assertEqual(*both(nvenc.sei_payload_types, sei(messages)))
            base, fast = both(nvenc.drop_timing_sei, unit)
            self.assertEqual(base, fast)
            self.assertEqual(base[0], "ok")
        broken = nal(39, b"\x01\x05\x00")
        self.assertEqual(*both(nvenc.sei_payload_types, broken))


class PackingTests(unittest.TestCase):
    def test_unpack_tile_matches_per_cell_copies(self):
        rng = np.random.default_rng(7)
        for pix_fmt in ("yuv444p", "yuv420p"):
            for rank in (1, 2, 5, 17, 36, 37, 64, 100):
                layout = _hevc.video_layout(3, 6, 8, rank, pix_fmt)
                planes = layout["planes"]
                pictures = layout["frames"] * layout["segments"]
                canvas = rng.integers(0, 256, size=(pictures + 2, planes, layout["image_height"] + 5,
                                                    layout["image_width"] + 7), dtype=np.uint8)
                tile = {"rect": [3, 2, layout["image_width"], layout["image_height"]]}
                base, fast = both(_unpack_tile, canvas, tile, 1, layout)
                self.assertEqual(base[0], "ok")
                self.assertTrue(np.array_equal(base[1], fast[1]), (pix_fmt, rank))
                # the index-map reference (C3's model): equal to the per-cell copies, segment by segment
                images = canvas[1:, :, 2:2 + layout["image_height"], 3:3 + layout["image_width"]]
                for segment in range(layout["segments"]):
                    expected = np.zeros((3, 6, 8, rank), dtype=np.uint8)
                    got = np.zeros_like(expected)
                    first = segment * layout["frames"]
                    _hevc._unmosaic(images[first:first + 3], layout, segment, expected)
                    accel_c.unmosaic(images[first:first + 3], layout, segment, got)
                    self.assertTrue(np.array_equal(expected, got), (pix_fmt, rank, segment))

    def test_raw_frame_bytes(self):
        rng = np.random.default_rng(8)
        for pix_fmt, planes in (("yuv444p", 3), ("yuv420p", 1)):
            images = rng.integers(0, 256, size=(4, planes, 16, 24), dtype=np.uint8)
            self.assertEqual(*both(_hevc._raw_frame_bytes, images, pix_fmt))

    def test_coefficient_digest_without_copy(self):
        rng = np.random.default_rng(2)
        values = rng.standard_normal((120, 33)).astype(np.float32)
        for array in (values, np.asfortranarray(values), values[:, ::2], values.astype(np.float64)):
            self.assertEqual(accel_c.coefficient_digest(array),
                             hashlib.sha256(array.astype("<f4", copy=False).tobytes()).hexdigest())


def random_json(rng, depth=0):
    choice = rng.randrange(9 if depth < 4 else 6)
    if choice == 0:
        return None
    if choice == 1:
        return rng.random() < 0.5
    if choice == 2:
        return rng.randrange(-10**20, 10**20)
    if choice == 3:
        return rng.choice((0.0, -0.0, 1e-308, 5e-324, 1.5, -2.25e10, rng.random()))
    if choice in (4, 5):
        return "".join(rng.choice("ab\u00e9\u4e2d\ud800\"\\\n") for _ in range(rng.randrange(0, 6)))
    if choice in (6, 7):
        return {f"k{rng.randrange(20)}": random_json(rng, depth + 1) for _ in range(rng.randrange(0, 5))}
    items = [random_json(rng, depth + 1) for _ in range(rng.randrange(0, 5))]
    return tuple(items) if rng.random() < 0.3 else items


class JsonTests(unittest.TestCase):
    def test_json_copy_equal_values_and_types(self):
        rng = random.Random(12)
        for trial in range(300):
            value = random_json(rng)
            base, fast = both(types.json_copy, value)
            self.assertEqual(base[0], "ok")
            self.assertEqual(json.dumps(base[1], sort_keys=True), json.dumps(fast[1], sort_keys=True), trial)
            self.assertEqual(repr(base[1]), repr(fast[1]), trial)

    def test_json_copy_rejections_are_identical(self):
        class Opaque:
            pass

        deep = []
        cursor = deep
        for _ in range(5000):
            cursor.append([])
            cursor = cursor[0]
        cases = [{"a": math.nan}, {"a": [math.inf]}, {1: 2}, {"a": Opaque()}, {"a": np.float64(1.0)},
                 {"a": 10 ** 5000}, deep, {"a": MappingProxyType({"b": (1, 2)})}, {"a": True, "b": [None, 1.5]},
                 {"a": np.int64(3)}, {"a": {"b": float("-inf")}}]
        for case in cases:
            base, fast = both(types.json_copy, case, name="probe")
            self.assertEqual(base[0], fast[0])
            if base[0] == "ok":
                self.assertEqual(repr(base[1]), repr(fast[1]))
            else:
                self.assertEqual(base[1:], fast[1:])

    def test_thaw_and_canonical_digest(self):
        rng = random.Random(14)
        for _ in range(100):
            value = {"x": random_json(rng), "y": [random_json(rng)]}
            frozen = types.frozen_json(value)
            self.assertEqual(*both(types.thaw_json, frozen))
            self.assertEqual(*both(types.canonical_digest, frozen))

    def test_scope_defaults_to_the_base(self):
        self.assertIs(accel_c.current(), accel_c.OFF)
        self.assertFalse(accel_c.fast())
        with accel_c.scope(FAST):
            self.assertTrue(accel_c.fast())
        self.assertFalse(accel_c.fast())
        with self.assertRaises(ConfigurationError):
            accel_c.AccelOptions(validation_decoder="x")
        with self.assertRaises(ConfigurationError):
            accel_c.AccelOptions(hash_once=1)


class UnitsMemoTests(unittest.TestCase):
    def test_verify_stream_parse_is_reused_only_for_the_same_stream(self):
        rng = np.random.default_rng(4)
        stream, _, _ = _hevc.encode_pictures(rng.integers(0, 256, size=(3, 3, 32, 32), dtype=np.uint8), "yuv444p")
        with accel_c.scope(FAST):
            units = nvenc.verify_stream(stream, 3)
            self.assertIs(accel_c.cached_units(bytes(bytearray(stream)), 3), None)  # equal bytes, other object
            nvenc.verify_stream(stream, 3)
            self.assertIs(accel_c.cached_units(stream, 2), None)
            nvenc.verify_stream(stream, 3)
            cached = accel_c.cached_units(stream, 3)
            self.assertEqual(cached, units)
            self.assertIs(accel_c.cached_units(stream, 3), None)  # one use
        nvenc.verify_stream(stream, 3)
        self.assertIs(accel_c.cached_units(stream, 3), None)  # off: nothing remembered


if __name__ == "__main__":
    unittest.main()
