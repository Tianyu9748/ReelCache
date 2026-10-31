"""F4 (dev22j5): content-addressed ids depend on content only, and the SGLang archival defaults.

* Two lineages that share a predictor registry (a bridge's sessions) publish in different interleavings. The
  registry's revision at planning time then differs, but their nodes and records keep identical ids: predictor
  provenance and planner work statistics are run telemetry in unhashed sidecars, which
  ``load_record(ref, telemetry=True)`` and ``node_commit_report`` merge back.
* Encoder thread counts never change ids (they never change the stream bytes).
* SGLang defaults: ``rate_lag`` 3; ``encode_workers`` and ``x265_threads`` derived from the process's CPUs, and
  recorded by the archive with the affinity-mask size they came from.
"""

from __future__ import annotations

import unittest
from unittest import mock

from reelcache.archive_pipeline import ArchivePipeline
from reelcache.lineage_commit import ArchiveOptions
from reelcache.predictor_registry import RankPredictorRegistry
from tests.step6_support import RuntimeFactory, chunk, method, node_content
import tests.test_sglang_bridge as t


class ContentIdTests(unittest.TestCase):
    def setUp(self):
        self.runtimes = RuntimeFactory(self)

    def _two_lineages(self, order, *, encode_threads=1):
        runtime = self.runtimes("content-ids")
        registry = RankPredictorRegistry("content-ids")
        pipeline = ArchivePipeline(runtime, method=method(1), registry=registry,
                                   options=ArchiveOptions(encode_threads=encode_threads), encode_workers=2,
                                   max_source_chunks=8, max_unpublished=8)
        self.addCleanup(pipeline.close, drain=False, timeout=120)
        for branch in ("a", "b"):
            runtime.create_lineage(branch)
        for branch, index in order:
            matrices, contexts = chunk(index, scale=1.0 + 0.1 * (branch == "b"))
            pipeline.submit(branch, f"{branch}-{index:03d}", matrices, contexts)
            if branch == "a":
                pipeline.drain("a")  # a publishes before b plans in one order, after it in the other
        refs = {}
        for branch in ("a", "b"):
            head = pipeline.drain(branch)
            refs[branch] = [ref for ref, _, _ in runtime.node_path(head)[1:]]
        return runtime, registry, refs

    def test_shared_registry_interleaving_keeps_node_and_record_ids(self):
        first = [("a", 0), ("b", 0), ("a", 1), ("b", 1), ("a", 2), ("b", 2)]
        second = [("b", 0), ("b", 1), ("b", 2), ("a", 0), ("a", 1), ("a", 2)]
        runtime_x, _, refs_x = self._two_lineages(first)
        runtime_y, _, refs_y = self._two_lineages(second)
        for branch in ("a", "b"):
            contents_x = [node_content(runtime_x, ref) for ref in refs_x[branch]]
            contents_y = [node_content(runtime_y, ref) for ref in refs_y[branch]]
            self.assertEqual(contents_x, contents_y)  # same stored content and decisions...
            self.assertEqual(refs_x[branch], refs_y[branch])  # ...hence the same content addresses
        # The provenance still differs between the runs, and stays readable from the sidecars.
        node = runtime_y.load_node(refs_y["b"][0])
        record_ref = node.matrix_records[0][1]
        plain = runtime_y.load_record(record_ref)
        merged = runtime_y.load_record(record_ref, telemetry=True)
        self.assertIsNone(plain.predictor_revision_ref)
        self.assertNotIn("predictions", plain.decision)
        self.assertNotIn("work", plain.decision)
        self.assertIn("predictions", merged.decision)
        self.assertIn("counters", merged.decision["work"])
        if merged.decision["predictions"] and merged.decision["predictions"][0].get("predictor"):
            self.assertTrue(merged.predictor_revision_ref.startswith("reelcache-rank-predictor:content-ids:"))
        report = runtime_y.node_commit_report(refs_y["b"][0])
        self.assertIn("pinned_sequence", report["registry"])
        self.assertNotIn("pinned_sequence", runtime_y.store.get_manifest(refs_y["b"][0])["commit"]["registry"])

    def test_encoder_threads_do_not_change_ids(self):
        order = [("a", 0), ("a", 1), ("a", 2)]
        runtime_one, _, one = self._two_lineages(order, encode_threads=1)
        runtime_two, _, two = self._two_lineages(order, encode_threads=2)
        self.assertEqual(one["a"], two["a"])
        # Not vacuous: the settings differ, in the sidecars only (commit options; the pack's x265 parameters).
        for runtime, threads in ((runtime_one, 1), (runtime_two, 2)):
            node_ref = one["a"][0]
            pack_ref = runtime.load_node(node_ref).packet_ref
            for ref in (node_ref, pack_ref):
                hashed = repr(runtime.store.get_manifest(ref))
                for field in ("encode_threads", "hevc_encode_threads", "x265_thread_params"):
                    self.assertNotIn(field, hashed)
            self.assertEqual(runtime.store.telemetry(node_ref)["commit"]["options"]["encode_threads"], threads)
            metrics = runtime.store.telemetry(pack_ref)["metrics"]
            self.assertEqual(metrics["hevc_encode_threads"], threads)
            self.assertIn(f"pools={threads}", metrics["x265_thread_params"])


class SGLangArchivalDefaultTests(unittest.TestCase):
    def test_f4_defaults_and_derived_threads(self):
        from reelcache.integrations.sglang import ReelCacheRuntimeConfig
        from reelcache.integrations.sglang import config as config_module
        config = ReelCacheRuntimeConfig.from_dict({"store_root": "f4-defaults"})
        self.assertEqual((config.rate_lag, config.encode_workers, config.x265_threads), (3, None, None))
        # dev22j8v6 (F6 a): the default derives from the job's CPU set; "inherit" from the calling thread's mask
        inherit = ReelCacheRuntimeConfig.from_dict({"store_root": "f4-defaults", "archival_cpus": "inherit"})
        for cpus, expected in ((1, (2, 2)), (8, (2, 4)), (12, (2, 6)), (16, (2, 6)), (24, (3, 6)), (32, (4, 6)),
                               (64, (4, 6))):
            with self.subTest(cpus=cpus), mock.patch.object(config_module, "job_cpu_set",
                                                            return_value=(frozenset(range(cpus)), "cgroup")):
                self.assertEqual((config.resolved_encode_workers(), config.resolved_x265_threads()), expected)
            with self.subTest(cpus=cpus, archival_cpus="inherit"), mock.patch.object(
                    config_module, "available_cpus", return_value=cpus):
                self.assertEqual((inherit.resolved_encode_workers(), inherit.resolved_x265_threads()), expected)
        with mock.patch.object(config_module, "job_cpu_set", return_value=(frozenset(range(16)), "cgroup")):
            four = ReelCacheRuntimeConfig.from_dict({"store_root": "f4-defaults", "encode_workers": 4})
            self.assertEqual((four.resolved_encode_workers(), four.resolved_x265_threads()), (4, 4))
        explicit = ReelCacheRuntimeConfig.from_dict({"store_root": "f4-defaults", "x265_threads": 6,
                                                  "encode_workers": 2, "rate_lag": 1})
        self.assertEqual((explicit.resolved_encode_workers(), explicit.resolved_x265_threads(), explicit.rate_lag),
                         (2, 6, 1))
        for bad in ({"encode_workers": 0}, {"x265_threads": 0}, {"encode_workers": 65}):
            with self.subTest(config=bad), self.assertRaises(Exception):
                ReelCacheRuntimeConfig.from_dict({"store_root": "f4-defaults", **bad})
        self.assertEqual(ReelCacheRuntimeConfig.from_dict(config.to_dict()), config)


class SGLangEncoderRecordTests(t.BridgeTestCase):
    representation = "svd_hevc"

    def test_the_archive_records_the_cpu_mask_behind_derived_encoders(self):
        """dev22j8v6 (F6 a): the derived counts split the job's CPU set (the gate's RAVEN d32 worker saw 8 of its
        32 CPUs after SGLang's NCCL init); the record keeps both masks and the pipeline gets the job's."""
        from reelcache.integrations.sglang import config as config_module
        job = (frozenset(range(32)), "cgroup")
        with mock.patch("os.sched_getaffinity", return_value=set(range(8))), \
                mock.patch.object(config_module, "job_cpu_set", return_value=job):
            derived = self.bridge("derived", x265_threads=None, encode_workers=None)
        record = derived.archive.encoder_record
        self.assertEqual((record["cpus_available"], record["encode_workers"], record["x265_threads"]), (8, 4, 6))
        self.assertEqual((record["encode_workers_configured"], record["x265_threads_configured"]), (None, None))
        self.assertEqual((record["archival_cpus"], record["job_cpus"], record["job_cpus_count"],
                          record["job_cpus_source"], record["attach_cpus"]), ("job", "0-31", 32, "cgroup", "0-7"))
        self.assertEqual(derived.archive.pipeline.cpu_set, frozenset(range(32)))
        with mock.patch("os.sched_getaffinity", return_value=set(range(16))), \
                mock.patch.object(config_module, "job_cpu_set", return_value=job):
            explicit = self.bridge("explicit")  # the bridge tests' explicit 2 x 2
        record = explicit.archive.encoder_record
        self.assertEqual((record["cpus_available"], record["encode_workers"], record["x265_threads"]), (16, 2, 2))
        self.assertEqual((record["encode_workers_configured"], record["x265_threads_configured"]), (2, 2))


if __name__ == "__main__":
    unittest.main()
