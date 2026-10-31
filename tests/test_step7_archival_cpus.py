"""dev22j8v6 (F6 a, b; coordinator 2026-10-18 18:4x): archival's CPUs and the fitter's stream.

(a) SGLang's NCCL init pins the worker to its GPU's NUMA node, so F4's derived encoder counts saw 8 or 24 of a 32-CPU
job (the F5.2 gate's RAVEN d32 worker 4444406: 8 CPUs, 0.087 vs 0.216 nodes/s). The counts now split the job's CPU set
(the cgroup cpuset, which a thread's affinity does not narrow), and archival's fitter and encoder threads -- with the
FFmpeg children they start -- get that mask; the generation thread keeps its own. (b) The fitter's CUDA stream is its
own by default (the CUDA half is in ``test_step7_raw_tier.CUDARawTierTests``)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest import mock

from reelcache import archive_pipeline as ap
from reelcache.archive_pipeline import ArchivePipeline
from reelcache.errors import ConfigurationError
from reelcache.integrations.sglang import ReelCacheRuntimeConfig
from reelcache.integrations.sglang import config as config_module
from reelcache.lineage_commit import ArchiveOptions
from tests import step6_support as s6
from tests import test_sglang_bridge as tb


class CpuListTests(unittest.TestCase):
    def test_cpu_lists_round_trip(self):
        cpus = config_module.parse_cpu_list("0-3,8,10-11\n")
        self.assertEqual(cpus, frozenset({0, 1, 2, 3, 8, 10, 11}))
        self.assertEqual(config_module.format_cpu_list(cpus), "0-3,8,10-11")
        self.assertEqual(ap._cpu_list(cpus), "0-3,8,10-11")
        self.assertEqual((config_module.parse_cpu_list(" "), config_module.format_cpu_list(())), (frozenset(), ""))
        for bad in ("3-1", "a", "-2"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                config_module.parse_cpu_list(bad)


class JobCpuSetTests(unittest.TestCase):
    """The job's CPUs from the process's cgroup (v2: ``cpuset.cpus.effective``; v1: ``cpuset.effective_cpus`` /
    ``cpuset.cpus``), the nearest ancestor below the hierarchy's root that has them; else the affinity at import;
    else the calling thread's now."""

    def layout(self, cgroup: str, files: dict) -> tuple[str, str]:
        root = tempfile.mkdtemp(prefix="cpuset-")
        self.addCleanup(lambda: __import__("shutil").rmtree(root, ignore_errors=True))
        proc = os.path.join(root, "proc_cgroup")
        with open(proc, "w") as handle:
            handle.write(cgroup)
        for path, text in files.items():
            os.makedirs(os.path.join(root, "fs", os.path.dirname(path)), exist_ok=True)
            with open(os.path.join(root, "fs", path), "w") as handle:
                handle.write(text)
        return proc, os.path.join(root, "fs")

    def test_the_task_cgroup_s_effective_cpuset_is_the_job_s(self):  # this cluster (Slurm, cgroup v2)
        task = "system.slice/slurmstepd.scope/job_7/step_0/user/task_0"
        proc, root = self.layout(f"0::/{task}\n", {f"{task}/cpuset.cpus.effective": "231-238\n",
                                                   f"{task}/cpuset.cpus": "\n",
                                                   "cpuset.cpus.effective": "0-255\n"})
        self.assertEqual(config_module.job_cpu_set(proc_cgroup=proc, cgroup_root=root),
                         (frozenset(range(231, 239)), "cgroup"))

    def test_the_nearest_ancestor_with_a_cpuset_counts_never_the_root(self):
        task = "system.slice/slurmstepd.scope/job_7/step_0/user/task_0"
        proc, root = self.layout(f"0::/{task}\n", {"system.slice/slurmstepd.scope/job_7/cpuset.cpus.effective": "0-31",
                                                   "cpuset.cpus.effective": "0-255"})
        self.assertEqual(config_module.job_cpu_set(proc_cgroup=proc, cgroup_root=root),
                         (frozenset(range(32)), "cgroup"))
        proc, root = self.layout(f"0::/{task}\n", {"cpuset.cpus.effective": "0-255"})  # not confined by a cpuset
        with mock.patch.object(config_module, "_IMPORT_AFFINITY", frozenset({1, 2})):
            self.assertEqual(config_module.job_cpu_set(proc_cgroup=proc, cgroup_root=root),
                             (frozenset({1, 2}), "import"))

    def test_cgroup_v1_s_cpuset_hierarchy(self):
        proc, root = self.layout("12:cpuset:/slurm/uid_1/job_2/step_0\n3:memory:/slurm/uid_1/job_2\n",
                                 {"cpuset/slurm/uid_1/job_2/step_0/cpuset.effective_cpus": "4-7"})
        self.assertEqual(config_module.job_cpu_set(proc_cgroup=proc, cgroup_root=root),
                         (frozenset({4, 5, 6, 7}), "cgroup"))
        proc, root = self.layout("12:cpuset:/slurm/uid_1/job_2\n", {"cpuset/slurm/uid_1/job_2/cpuset.cpus": "0,2"})
        self.assertEqual(config_module.job_cpu_set(proc_cgroup=proc, cgroup_root=root), (frozenset({0, 2}), "cgroup"))

    def test_without_a_cgroup_the_affinity_at_import_then_now(self):
        missing = os.path.join(tempfile.gettempdir(), "no-such-proc-cgroup")
        with mock.patch.object(config_module, "_IMPORT_AFFINITY", frozenset({3})):
            self.assertEqual(config_module.job_cpu_set(proc_cgroup=missing), (frozenset({3}), "import"))
        with mock.patch.object(config_module, "_IMPORT_AFFINITY", None):
            self.assertEqual(config_module.job_cpu_set(proc_cgroup=missing),
                             (frozenset(os.sched_getaffinity(0)), "affinity"))


class ArchivalCpuConfigTests(unittest.TestCase):
    def test_the_derived_encoders_split_the_job_s_cpus_and_the_switches_validate(self):
        config = ReelCacheRuntimeConfig.from_dict({"store_root": "archival-cpus"})
        self.assertEqual((config.archival_cpus, config.fitter_stream), ("job", "dedicated"))
        inherit = ReelCacheRuntimeConfig.from_dict({"store_root": "archival-cpus", "archival_cpus": "inherit",
                                                 "fitter_stream": "pooled"})
        # the gate's RAVEN d32 worker: 8 of the job's 32 CPUs after the NCCL init
        with mock.patch.object(config_module, "job_cpu_set", return_value=(frozenset(range(32)), "cgroup")), \
                mock.patch.object(config_module, "available_cpus", return_value=8):
            self.assertEqual((config.archival_cpu_count(), config.resolved_encode_workers(),
                              config.resolved_x265_threads()), (32, 4, 6))
            self.assertEqual((inherit.archival_cpu_count(), inherit.resolved_encode_workers(),
                              inherit.resolved_x265_threads()), (8, 2, 4))
        self.assertEqual(ReelCacheRuntimeConfig.from_dict(inherit.to_dict()), inherit)
        for bad in ({"archival_cpus": "all"}, {"archival_cpus": None}, {"fitter_stream": "default"},
                    {"fitter_stream": None}):
            with self.subTest(config=bad), self.assertRaises(ConfigurationError):
                ReelCacheRuntimeConfig.from_dict({"store_root": "archival-cpus", **bad})
        for name in ("archival_cpus", "fitter_stream"):  # scheduling only: no content identity
            self.assertNotIn(name, config.content_dict())


class PipelineCpuSetTests(unittest.TestCase):
    def pipeline(self, name: str, **kwargs):
        runtime = s6.RuntimeFactory(self)(name)
        runtime.create_lineage("main")
        pipeline = ArchivePipeline(runtime, method=s6.method(), registry=s6.registry(),
                                   options=ArchiveOptions(encode_threads=1), **kwargs)
        self.addCleanup(lambda: pipeline.close(drain=False, timeout=60))
        return runtime, pipeline

    def test_the_fitter_and_encoder_threads_get_the_job_s_cpus_and_the_generation_thread_keeps_its_own(self):
        full = frozenset(os.sched_getaffinity(0))
        narrow = frozenset({min(full)})
        out: dict = {}

        def generation():  # a SGLang worker after its NCCL init: pinned to its GPU's NUMA node
            os.sched_setaffinity(0, narrow)
            _, masked = self.pipeline("cpus-job", cpu_set=full)
            _, inherited = self.pipeline("cpus-inherit")
            for pipeline in (masked, inherited):
                matrices, contexts = s6.chunk(0)
                pipeline.submit("main", "chunk-000000", matrices, contexts)
                out.setdefault("refs", []).append(pipeline.drain("main", timeout=120))
            out["fitters"] = [frozenset(os.sched_getaffinity(pipeline._thread.native_id))
                              for pipeline in (masked, inherited)]
            out["stats"] = [pipeline.stats()["cpu_sets"] for pipeline in (masked, inherited)]

            def encoder():  # an encoder thread this narrow thread starts, and the FFmpeg-like child it runs
                masked._archival_thread("encoder")
                out["encoder"] = frozenset(os.sched_getaffinity(0))
                child = subprocess.run([sys.executable, "-c", "import os; print(sorted(os.sched_getaffinity(0)))"],
                                       capture_output=True, text=True, timeout=120)
                out["child"] = frozenset(json.loads(child.stdout))

            thread = threading.Thread(target=encoder)
            thread.start()
            thread.join(180)
            out["generation"] = frozenset(os.sched_getaffinity(0))

        thread = threading.Thread(target=generation)
        thread.start()
        thread.join(600)
        self.assertFalse(thread.is_alive())
        self.assertEqual(out["refs"][0], out["refs"][1])  # the mask changes scheduling only
        self.assertEqual(out["generation"], narrow)  # never touched
        self.assertEqual(out["fitters"], [full, narrow])
        self.assertEqual((out["encoder"], out["child"]), (full, full))
        listed = config_module.format_cpu_list(full)
        masked, inherited = out["stats"]
        self.assertEqual((masked["requested"], masked["fitter"], masked["errors"]), (listed, listed, 0))
        self.assertEqual(set(masked["encoders"]), {listed})  # every encoder thread that ran a node
        self.assertEqual(inherited, {"requested": None, "fitter": None, "encoders": {}, "errors": 0,
                                     "last_error": None})

    def test_the_pipeline_validates_its_cpu_set_and_fitter_stream(self):
        for bad in ({"cpu_set": set()}, {"cpu_set": {-1}}, {"cpu_set": {"0"}}, {"fitter_stream": "default"}):
            with self.subTest(bad=bad), self.assertRaises(ConfigurationError):
                self.pipeline("cpus-bad", **bad)
        _, pipeline = self.pipeline("cpus-stream", fitter_stream="dedicated")  # a CPU pipeline has no stream
        self.assertEqual(pipeline.stats()["fitter_stream"], {"mode": "dedicated", "note": None, "dedicated": None})
        _, pipeline = self.pipeline("cpus-default")  # the class's default (other users, e.g. LongCat): F5.5's
        self.assertEqual(pipeline.stats()["fitter_stream"], {"mode": "pooled", "note": None, "dedicated": None})


class BridgeArchivalCpuTests(tb.BridgeTestCase):
    representation = "svd_hevc"  # the lineage archive (its pipeline)

    def test_the_bridge_s_archive_passes_its_cpus_and_dedicated_stream(self):
        job = frozenset(os.sched_getaffinity(0))
        with mock.patch.object(config_module, "job_cpu_set", return_value=(job, "cgroup")):
            bridge = self.bridge("cpus-bridge")
        pipeline = bridge.archive.pipeline
        self.assertEqual((pipeline.fitter_stream_mode, pipeline.cpu_set), ("dedicated", job))
        self.assertEqual(bridge.archive.encoder_record["job_cpus"], config_module.format_cpu_list(job))
        inherit = self.bridge("cpus-bridge-inherit", archival_cpus="inherit", fitter_stream="pooled")
        self.assertEqual((inherit.archive.pipeline.fitter_stream_mode, inherit.archive.pipeline.cpu_set),
                         ("pooled", None))


if __name__ == "__main__":
    unittest.main()
