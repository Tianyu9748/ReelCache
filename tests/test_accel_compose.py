"""accel composition (2026-10-30): levers of different tracks together (CPU, software decoder pool).

* A1 decoded tier (route "all") + L ``release_behind``: the routed restore releases its reconstruction handles early
  and still equals the decoder path bitwise; the tier's leases are released with the restore.
* the configuration: every accel field at its default keeps the base's records (no new key in ``to_dict``), and the
  recommended non-default set validates and round-trips.
"""

from __future__ import annotations

import dataclasses
import unittest

from reelcache.integrations.sglang.config import ReelCacheRuntimeConfig
from tests import test_accel_a_decoded_tier as ta

ACCEL_FIELDS = (
    "decoded_tier_bytes", "decoded_tier_route", "decoded_tier_layer_ms",  # A
    "fitter_green_context_sms", "fitter_green_workqueue", "fitter_cuda_graphs", "fitter_graph_memory_bytes",
    "fused_fingerprints", "finalize_on_fit_device", "fitter_fp64",  # B
    "validation_decoder", "hash_once", "seal_prepare_after_publish", "nvenc_device_input", "gil_fastpaths",
    "native_ext",  # C
)
RECOMMENDED = {"decoded_tier_bytes": 8 << 30, "decoded_tier_route": "all", "fitter_fp64": "off",
               "fused_fingerprints": True, "validation_decoder": "nvdec", "hash_once": True,
               "seal_prepare_after_publish": True, "gil_fastpaths": True}


class TierWithReleaseBehindTests(ta.TierCase):
    def test_routed_restores_with_release_behind_equal_the_decoder_path(self):
        tier = self.make_tier(route="all")
        for packet in self._packs():
            self._fill(tier, packet)
        restorer, pool = self.restorer(tier)
        for depth in (0, 1, 2):
            with self.subTest(release_behind=depth):
                caches, destination = ta._destination(4 * ta.s6.TOKENS)
                report = restorer.restore(self.heads["a"], destination, release_behind=depth)
                ta._same(self, caches, self.reference["a"])
                self.assertEqual(report["decoded_tier"]["routed_nodes"], 4)
                layers = len(report["handles"])  # one reconstruction per layer (the tiny model has 2)
                early = max(0, layers - depth - 1)
                self.assertEqual(report["release_behind"], {"depth": depth, "released_early": early})
                self.assertEqual([h.released for h in report["handles"]], [i < early for i in range(layers)])
                self.assertEqual(tier.stats()["leased"], 0)
                for handle in report["handles"]:
                    handle.release()
        self.assertEqual(pool.jobs, [])  # every node came from the tier

    def _packs(self):
        return ta.RoutedRestoreTests._packs(self)

    def _fill(self, tier, packet):
        return ta.RoutedRestoreTests._fill(self, tier, packet)


class ComposedConfigTests(unittest.TestCase):
    def base(self, **extra):
        return ReelCacheRuntimeConfig.from_dict({"store_root": "/tmp/accel-compose-config", **extra})

    def test_every_accel_field_exists_and_its_default_stays_out_of_the_records(self):
        names = {field.name for field in dataclasses.fields(ReelCacheRuntimeConfig)}
        self.assertEqual(set(ACCEL_FIELDS) - names, set())
        record = self.base().to_dict()
        self.assertEqual(set(ACCEL_FIELDS) & set(record), set())
        self.assertNotIn("pipelined_restore", record)
        self.assertNotIn("verify_bases", record)

    def test_the_recommended_set_validates_and_round_trips(self):
        config = self.base(**RECOMMENDED)
        record = config.to_dict()
        for name, value in RECOMMENDED.items():
            self.assertEqual(record[name], value, name)
        again = ReelCacheRuntimeConfig.from_dict({key: value for key, value in record.items() if key != "schema"})
        self.assertEqual(again.to_dict(), record)


if __name__ == "__main__":
    unittest.main()
