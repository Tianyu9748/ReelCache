import threading
import unittest
from unittest import mock

import torch

from reelcache.codecs import HEVCProvider
from reelcache.errors import CapacityError, ConfigurationError, IntegrityError, StateError, UnsupportedError
from reelcache.paired_decode import PairedDecoder
from reelcache.types import CodecSpec
from tests import test_causvid_adapter as fixtures


class PairedDecoderTests(unittest.TestCase):
    def test_two_workers_really_overlap_and_keep_results_ordered(self):
        barrier = threading.Barrier(2)
        owners = []
        def operation(index):
            def run(provider):
                owners.append(threading.get_ident())
                barrier.wait(timeout=5)
                return index
            return run
        with PairedDecoder(HEVCProvider(), device="cpu", max_output_bytes=100) as pair:
            with pair._pair((operation(0), operation(1)), output_bytes=64) as values:
                self.assertEqual(values, (0, 1))
                with self.assertRaises(CapacityError):
                    with pair._pair((operation(0), operation(1)), output_bytes=1):
                        pass
                with self.assertRaises(StateError):
                    pair.close()
            self.assertEqual(len(set(owners)), 2)
            self.assertGreater(pair.stats()["worker_interval_overlap_seconds"], 0)
            self.assertEqual(pair.stats()["peak_output_bytes"], 64)
        with self.assertRaises(StateError):
            with pair._pair((operation(0), operation(1)), output_bytes=1):
                pass

    def test_failed_peer_drains_and_releases_successful_result(self):
        barrier = threading.Barrier(2)
        released = []
        def good(provider):
            barrier.wait(timeout=5)
            return "owned"
        def fail(provider):
            barrier.wait(timeout=5)
            raise IntegrityError("V failed")
        with PairedDecoder(HEVCProvider(), device="cpu", max_output_bytes=100) as pair:
            with self.assertRaisesRegex(IntegrityError, "V failed"):
                with pair._pair((good, fail), output_bytes=1, cleanup=released.append):
                    self.fail("partial K/V pair escaped")
            self.assertEqual(released, ["owned"])
            self.assertEqual(pair.stats()["failures"], 1)
            with pair._pair((lambda p: "K", lambda p: "V"), output_bytes=1) as values:
                self.assertEqual(values, ("K", "V"))

    def test_cleanup_attempts_both_values_on_failure(self):
        calls = []
        def cleanup(value):
            calls.append(value)
            if value == "K":
                raise RuntimeError("K release")
        with PairedDecoder(HEVCProvider(), device="cpu", max_output_bytes=10) as pair:
            with self.assertRaisesRegex(StateError, "K release"):
                with pair._pair((lambda p: "K", lambda p: "V"), output_bytes=1, cleanup=cleanup):
                    pass
            self.assertEqual(calls, ["K", "V"])

    def test_budget_precedes_work_and_configuration_is_strict(self):
        with PairedDecoder(HEVCProvider(), device="cpu", max_output_bytes=1) as pair:
            work = mock.Mock(side_effect=AssertionError("early decoding"))
            with self.assertRaises(CapacityError):
                with pair._pair((work, work), output_bytes=2):
                    pass
            work.assert_not_called()
            for value in ([], [1], None):
                with self.assertRaises(ConfigurationError):
                    pair.decode_pixels(value)
        with self.assertRaises(ConfigurationError):
            PairedDecoder(object(), device="cpu", max_output_bytes=100)

    def test_two_borrowed_pairs_share_an_aggregate_budget(self):
        operations = (lambda p: "K", lambda p: "V")
        with PairedDecoder(HEVCProvider(), device="cpu", max_output_bytes=100, max_pairs=2) as pair:
            with pair._pair(operations, output_bytes=60):
                with self.assertRaises(CapacityError):
                    with pair._pair(operations, output_bytes=41):
                        pass
                with pair._pair(operations, output_bytes=40):
                    self.assertEqual(pair.stats()["active_pairs"], 2)
                    self.assertEqual(pair.stats()["output_bytes"], 100)
                    with self.assertRaises(CapacityError):
                        with pair._pair(operations, output_bytes=0):
                            pass
                    with self.assertRaises(StateError):
                        pair.close()
                self.assertEqual(pair.stats()["output_bytes"], 60)
            self.assertEqual(pair.stats()["active_pairs"], 0)
            self.assertEqual(pair.stats()["output_bytes"], 0)
            self.assertEqual(pair.stats()["peak_output_bytes"], 100)
        for count in (0, 3, True):
            with self.assertRaises(ConfigurationError):
                PairedDecoder(HEVCProvider(), device="cpu", max_output_bytes=100, max_pairs=count)

    def test_failed_second_pair_does_not_release_first_reservation(self):
        with PairedDecoder(HEVCProvider(), device="cpu", max_output_bytes=100, max_pairs=2) as pair:
            with pair._pair((lambda p: "K", lambda p: "V"), output_bytes=60):
                def fail(p):
                    raise IntegrityError("second pair failure")
                with self.assertRaises(IntegrityError):
                    with pair._pair((fail, lambda p: "V"), output_bytes=40):
                        pass
                self.assertEqual(pair.stats()["active_pairs"], 1)
                self.assertEqual(pair.stats()["output_bytes"], 60)
            self.assertEqual(pair.stats()["output_bytes"], 0)

class CausVidPairedDecodeTests(unittest.TestCase):
    setUp = fixtures.CausVidAdapterTests.setUp

    def test_raw_svd_hevc_native_seam_parity_and_lifetime(self):
        for kind in ("raw", "svd", "svd_hevc"):
            with self.subTest(kind=kind):
                provider = HEVCProvider()
                self.runtime.decoder = provider.decode_tensor
                spec = CodecSpec(kind, rank=None if kind == "raw" else 3)
                with fixtures.CausVidAdapter(
                    self.pipe, self.runtime, kind, config=fixtures.CONFIG, specs=spec,
                ) as adapter:
                    adapter.start(self.noise, self.conditioning, self.rng, conditioning_id="fixed")
                    adapter.step(clean_prefix=self.prefix)
                    snapshot = adapter.snapshot()
                    adapter.step()
                    adapter.step()
                    expected, rng, video = adapter.latent_state, adapter.rng_state, adapter.decode()
                    adapter.restore(snapshot, kind + "-paired", conditioning_id="fixed")
                    adapter.parallel_kv_bytes = 1 << 20
                    adapter.step()
                    adapter.step()
                    self.assertTrue(torch.equal(expected, adapter.latent_state))
                    self.assertTrue(torch.equal(rng, adapter.rng_state))
                    self.assertTrue(torch.equal(video, adapter.decode()))
                    self.assertEqual(self.runtime.resident_bytes, 0)
                    self.assertEqual(self.runtime.residency.reserved_bytes, 0)
                    self.assertGreater(adapter.memory_report()["parallel_kv"]["pairs"], 0)

    def test_failed_pair_keeps_clean_head_and_rng(self):
        provider = HEVCProvider()
        self.runtime.decoder = provider.decode_tensor
        with fixtures.CausVidAdapter(
            self.pipe, self.runtime, "failure", config=fixtures.CONFIG, parallel_kv_bytes=1 << 20,
        ) as adapter:
            adapter.start(self.noise, self.conditioning, self.rng, conditioning_id="fixed")
            adapter.step(clean_prefix=self.prefix)
            before = self.runtime.branch_info("failure")
            rng = adapter.rng_state
            original = HEVCProvider.decode_tensor
            def fail(provider, encoded, **kwargs):
                if encoded.metadata["context"]["key"]["target"] == "V":
                    raise IntegrityError("paired V corruption")
                return original(provider, encoded, **kwargs)
            with mock.patch.object(HEVCProvider, "decode_tensor", fail):
                with self.assertRaisesRegex(IntegrityError, "paired V corruption"):
                    adapter.step()
            self.assertEqual(self.runtime.branch_info("failure"), before)
            self.assertTrue(torch.equal(rng, adapter.rng_state))
            self.assertEqual(self.runtime.active_leases, 0)

    def test_combining_unqualified_cache_modes_is_rejected(self):
        for options in ({"max_hot_history_bytes": 1},):
            with self.assertRaises(UnsupportedError):
                fixtures.CausVidAdapter(
                    self.pipe, self.runtime, "bad", config=fixtures.CONFIG,
                    parallel_kv_bytes=1024, **options,
                )

    def test_combined_paired_lookahead_matches_paired_only(self):
        provider = HEVCProvider()
        self.runtime.decoder = provider.decode_tensor
        for kind in ("raw", "svd", "svd_hevc"):
            with self.subTest(kind=kind), fixtures.CausVidAdapter(
                self.pipe, self.runtime, kind + "-combined", config=fixtures.CONFIG,
                specs=CodecSpec(kind, rank=None if kind == "raw" else 3), parallel_kv_bytes=1 << 20,
            ) as adapter:
                adapter.start(self.noise, self.conditioning, self.rng, conditioning_id="fixed")
                adapter.step(clean_prefix=self.prefix)
                snapshot = adapter.snapshot()
                adapter.step()
                adapter.step()
                expected, rng, video = adapter.latent_state, adapter.rng_state, adapter.decode()
                adapter.restore(snapshot, kind + "-lookahead", conditioning_id="fixed")
                adapter.prefetch_history_bytes = 1 << 20
                adapter.step()
                adapter.step()
                self.assertTrue(torch.equal(expected, adapter.latent_state))
                self.assertTrue(torch.equal(rng, adapter.rng_state))
                self.assertTrue(torch.equal(video, adapter.decode()))
                stats = adapter.memory_report()["layer_prefetch"]
                self.assertEqual(adapter.memory_report()["history_leases_scope"], "active_and_next_layer")
                self.assertFalse(adapter.memory_report()["retained_reconstruction_dictionary"])
                self.assertFalse(stats["active"])
                self.assertEqual(stats["paired_reconstruction"]["max_pairs"], 2)
                self.assertEqual(stats["paired_reconstruction"]["active_pairs"], 0)
                self.assertEqual(stats["paired_reconstruction"]["output_bytes"], 0)
                self.assertEqual(self.runtime.residency.reserved_bytes, 0)

    def test_combined_lookahead_failure_drains_both_layers(self):
        provider = HEVCProvider()
        self.runtime.decoder = provider.decode_tensor
        with fixtures.CausVidAdapter(
            self.pipe, self.runtime, "combined-fail", config=fixtures.CONFIG,
            prefetch_history_bytes=1 << 20, parallel_kv_bytes=1 << 20,
        ) as adapter:
            adapter.start(self.noise, self.conditioning, self.rng, conditioning_id="fixed")
            adapter.step(clean_prefix=self.prefix)
            head, rng = self.runtime.branch_info(adapter.branch_id), adapter.rng_state
            original = HEVCProvider.decode_tensor
            def fail(provider, encoded, **kwargs):
                key = encoded.metadata["context"]["key"]
                if key == {"layer": 1, "target": "V"}:
                    raise IntegrityError("next-layer V failure")
                return original(provider, encoded, **kwargs)
            with mock.patch.object(HEVCProvider, "decode_tensor", fail):
                with self.assertRaisesRegex(IntegrityError, "next-layer V failure"):
                    adapter.step()
            self.assertEqual(head, self.runtime.branch_info(adapter.branch_id))
            self.assertTrue(torch.equal(rng, adapter.rng_state))
            self.assertEqual(self.runtime.residency.reserved_bytes, 0)
            self.assertEqual(adapter._prefetch.stats()["paired_reconstruction"]["active_pairs"], 0)

    def test_combined_budget_rejection_precedes_restoration(self):
        self.runtime.decoder = HEVCProvider().decode_tensor
        with fixtures.CausVidAdapter(
            self.pipe, self.runtime, "combined-budget", config=fixtures.CONFIG,
            prefetch_history_bytes=1 << 20, parallel_kv_bytes=1,
        ) as adapter:
            adapter.start(self.noise, self.conditioning, self.rng, conditioning_id="fixed")
            adapter.step(clean_prefix=self.prefix)
            with mock.patch.object(HEVCProvider, "decode_tensor", side_effect=AssertionError("unadmitted decode")):
                with self.assertRaisesRegex(CapacityError, "aggregate paired"):
                    adapter.step()
            self.assertEqual(self.runtime.active_leases, 0)

    def test_bound_head_check_avoids_duplicate_metadata_reads(self):
        self.runtime.create_branch("head-check")
        head = self.runtime.branch_info("head-check")["head_id"]
        with mock.patch.object(self.runtime, "_load_head", side_effect=AssertionError("metadata reread")):
            self.runtime.require_head("head-check", head)
            with self.assertRaises(StateError):
                self.runtime.require_head("head-check", "wrong-head")


if __name__ == "__main__":
    unittest.main()
