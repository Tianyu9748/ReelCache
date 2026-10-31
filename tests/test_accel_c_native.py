"""Accel track C (C3 ``native_ext``): the Rust extension ``reelcache._native.reelcache_native`` (built into the tree, cp312)
returns exactly what the Python implementations return, on synthetic edge cases and real libx265 streams/canvases."""

from __future__ import annotations

import random
import unittest

import numpy as np

from reelcache import _hevc, accel_c, nvenc
from reelcache.codecs import _unpack_tile
from reelcache.errors import IntegrityError
from tests.test_accel_c_fastpaths import random_stream

NATIVE = accel_c.AccelOptions(native_ext=True)


def outcome(function, options, *args):
    with accel_c.scope(options):
        try:
            return "ok", function(*args)
        except Exception as exc:  # noqa: BLE001 - compared
            return "raise", type(exc), str(exc)


class NativeExtensionTests(unittest.TestCase):
    def test_extension_loads(self):
        self.assertTrue(accel_c.native_available())
        from reelcache._native import reelcache_native
        self.assertEqual(reelcache_native.__version__, "0.1.0")

    def test_start_codes_match_the_regex(self):
        from reelcache._native import reelcache_native
        rng = random.Random(3)
        for _ in range(300):
            data = bytes(rng.choice((0, 0, 0, 1, 3, 9)) for _ in range(rng.randrange(0, 400)))
            self.assertEqual(reelcache_native.start_codes(data),
                             [match.start() for match in _hevc._START_CODE.finditer(data)])

    def test_nal_units_identical_including_errors(self):
        rng = random.Random(21)
        cases = [random_stream(rng, rng.randrange(1, 10)) for _ in range(60)]
        good = cases[0]
        cases += [b"", bytearray(good), b"\x01\x02" + good, good[:-1] + b"\x00\x00\x01\x40",
                  b"\x00\x00\x01\xc0\x01" + good, b"\x00\x00\x01\x40\x00" + good, b"\x00\x00\x01\x26\x01",
                  b"\x00\x00\x01\x02", "text", b"\x00\x00\x00\x01\x40\x01" * 3]
        for case in cases:
            base = outcome(_hevc.nal_units, accel_c.OFF, case)
            native = outcome(_hevc.nal_units, NATIVE, case)
            self.assertEqual(base, native, case[:16])
        stats = accel_c.stats()
        self.assertGreater(stats["native_calls"], 0)

    def test_real_stream_parsing(self):
        rng = np.random.default_rng(5)
        stream, _, _ = _hevc.encode_pictures(rng.integers(0, 256, size=(4, 3, 48, 80), dtype=np.uint8), "yuv444p")
        for function, args in ((_hevc.nal_units, (stream,)), (_hevc.all_intra_access_units, (stream, 4)),
                               (nvenc.verify_stream, (stream, 4)), (nvenc.drop_timing_sei, (stream,))):
            self.assertEqual(outcome(function, accel_c.OFF, *args), outcome(function, NATIVE, *args))

    def test_unpack_tile_identical(self):
        rng = np.random.default_rng(17)
        for pix_fmt in ("yuv444p", "yuv420p"):
            for rank in (1, 2, 5, 17, 36, 37, 64, 100, 151):
                layout = _hevc.video_layout(3, 6, 8, rank, pix_fmt)
                planes = layout["planes"]
                pictures = layout["frames"] * layout["segments"]
                canvas = rng.integers(0, 256, size=(pictures + 3, planes, layout["image_height"] + 6,
                                                    layout["image_width"] + 9), dtype=np.uint8)
                for first, rect in ((0, [0, 0]), (2, [9, 6]), (3, [4, 1])):
                    tile = {"rect": rect + [layout["image_width"], layout["image_height"]]}
                    base = outcome(_unpack_tile, accel_c.OFF, canvas, tile, first, layout)
                    native = outcome(_unpack_tile, NATIVE, canvas, tile, first, layout)
                    self.assertEqual(base[0], "ok")
                    self.assertTrue(np.array_equal(base[1], native[1]), (pix_fmt, rank, first, rect))
                    self.assertEqual(native[1].shape, base[1].shape)

    def test_unpack_tile_falls_back_where_not_applicable(self):
        layout = _hevc.video_layout(3, 6, 8, 20, "yuv444p")
        canvas = np.zeros((3, 3, layout["image_height"], layout["image_width"]), dtype=np.uint8)
        tile = {"rect": [0, 0, layout["image_width"], layout["image_height"]]}
        with accel_c.scope(NATIVE):
            self.assertIsNone(accel_c.native_unpack_tile(canvas[:, :, :, ::-1], tile, 0, layout))  # not contiguous
            self.assertIsNone(accel_c.native_unpack_tile(canvas.astype(np.int16), tile, 0, layout))
            self.assertIsNone(accel_c.native_unpack_tile(canvas, tile, 1, layout))  # pictures outside the canvas
            self.assertIsNotNone(accel_c.native_unpack_tile(canvas, tile, 0, layout))
        self.assertIsNone(accel_c.native_unpack_tile(canvas, tile, 0, layout))  # flag off

    def test_real_canvas_round_trip(self):
        rng = np.random.default_rng(23)
        layout = _hevc.video_layout(3, 6, 8, 40, "yuv444p")
        values = rng.integers(0, 256, size=(3, 6, 8, 40), dtype=np.uint8)
        images = np.concatenate([_hevc._mosaic(values, layout, segment) for segment in range(layout["segments"])])
        stream, _, _ = _hevc.encode_pictures(images, "yuv444p")
        decoded, _ = _hevc.decode_pictures(stream, pictures=images.shape[0], height=images.shape[2],
                                           width=images.shape[3], pix_fmt="yuv444p")
        tile = {"rect": [0, 0, layout["image_width"], layout["image_height"]]}
        self.assertTrue(np.array_equal(outcome(_unpack_tile, accel_c.OFF, decoded, tile, 0, layout)[1],
                                       outcome(_unpack_tile, NATIVE, np.ascontiguousarray(decoded), tile, 0,
                                               layout)[1]))


if __name__ == "__main__":
    unittest.main()
