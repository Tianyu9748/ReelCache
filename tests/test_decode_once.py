import unittest
from unittest import mock

import torch

from reelcache.codecs import HEVCProvider
from reelcache.types import CodecSpec
from tests import test_causvid_adapter as fixtures


class DecodeOnceHistoryTests(unittest.TestCase):
    setUp = fixtures.CausVidAdapterTests.setUp

    def run_mode(self, branch, kind, **options):
        provider = HEVCProvider()
        calls = []
        original = provider.decode_tensor

        def counted(encoded, **kwargs):
            calls.append(encoded.metadata["context"]["token_start"])
            return original(encoded, **kwargs)

        self.runtime.decoder = counted
        spec = CodecSpec(kind, rank=None if kind == "raw" else 3)
        with fixtures.CausVidAdapter(self.pipe, self.runtime, branch, config=fixtures.CONFIG,
                                     specs=spec, **options) as adapter:
            adapter.start(self.noise, self.conditioning, self.rng, conditioning_id="fixed")
            adapter.step(clean_prefix=self.prefix)
            adapter.step()
            adapter.step()
            return adapter.latent_state, adapter.rng_state, adapter.decode(), calls

    def test_quality_hot_decodes_each_committed_packet_once_with_reload_parity(self):
        keys = 2 * self.model.layers
        for kind in ("raw", "svd", "svd_hevc"):
            with self.subTest(kind=kind):
                reload = self.run_mode(kind + "-reload", kind)
                once = self.run_mode(kind + "-once", kind, history_mode="quality_hot",
                                     max_quality_history_bytes=1 << 24)
                self.assertTrue(torch.equal(reload[0], once[0]))
                self.assertTrue(torch.equal(reload[1], once[1]))
                self.assertTrue(torch.equal(reload[2], once[2]))
                self.assertEqual(len(once[3]), 3 * keys)
                chunk = fixtures.CONFIG.chunk_tokens
                self.assertEqual(sorted(set(once[3])), [0, chunk, 2 * chunk])
                self.assertTrue(all(once[3].count(start) == keys for start in set(once[3])))
                self.assertGreater(len(reload[3]), len(once[3]))
                self.assertEqual(self.runtime.active_leases, 0)

    def test_default_restoration_skips_conditioning_checks(self):
        with mock.patch("reelcache.codecs.basis_condition_bound", side_effect=AssertionError("check ran")):
            self.run_mode("svd-default", "svd", history_mode="quality_hot", max_quality_history_bytes=1 << 24)


if __name__ == "__main__":
    unittest.main()
