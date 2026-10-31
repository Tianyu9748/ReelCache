"""Accel study track B (2026-10-30): the fitter's opt-in GPU-footprint options (``reelcache.accel_b``) on the CPU.

Configuration fields, validation and to_dict defaults, plumbing into the archive pipeline, the thread-local scope,
and that every option leaves CPU tensors (and the defaults leave everything) on the base path. The GPU behaviour --
green-context confinement and the allocator hazard, graphs bitwise, fused fingerprints exact, the FP32 passes'
accuracy and the pipeline identity -- is tested in ``tests/test_accel_b_gpu.py`` inside GPU jobs.
"""

from __future__ import annotations

import dataclasses
import threading
import unittest

import torch

from reelcache import accel_b
from reelcache import graph_capture
from reelcache.archive_pipeline import ArchivePipeline
from reelcache.deferred_reads import projection_routine, run_one, source_energy_routine, span_energy_routine
from reelcache.errors import ConfigurationError
from reelcache.integrations.sglang import ReelCacheRuntimeConfig
from reelcache.lineage_commit import ArchiveOptions, _fingerprint_terms, source_fingerprints
from tests.step6_support import RuntimeFactory, chunk, method, node_content, registry

ACCEL_FIELDS = ("fitter_green_context_sms", "fitter_green_workqueue", "fitter_cuda_graphs", "fitter_graph_memory_bytes",
                "fused_fingerprints", "finalize_on_fit_device", "fitter_fp64")


class AccelConfigTests(unittest.TestCase):
    base = {"store_root": "accel-b-config"}

    def test_defaults_are_the_base_records(self):
        config = ReelCacheRuntimeConfig.from_dict(self.base)
        self.assertIsNone(accel_b.FitterAccel.from_config(config))
        data = config.to_dict()
        self.assertFalse(set(ACCEL_FIELDS) & set(data))  # the base tree's telemetry record, unchanged
        self.assertEqual(ReelCacheRuntimeConfig.from_dict(data), config)

    def test_options_round_trip_and_are_execution_options(self):
        values = {"fitter_green_context_sms": 48, "fitter_cuda_graphs": True, "fitter_graph_memory_bytes": 1 << 28,
                  "fused_fingerprints": True, "finalize_on_fit_device": True, "fitter_fp64": "fp32:span+residual"}
        config = ReelCacheRuntimeConfig.from_dict({**self.base, **values})
        data = config.to_dict()
        self.assertEqual({name: data[name] for name in values}, values)
        self.assertEqual(ReelCacheRuntimeConfig.from_dict(data), config)
        self.assertEqual(config.content_dict(), ReelCacheRuntimeConfig.from_dict(self.base).content_dict())
        accel = accel_b.FitterAccel.from_config(config)
        self.assertEqual((accel.green_sms, accel.cuda_graphs, accel.graph_memory_bytes, accel.fused_fingerprints,
                          accel.finalize_on_fit_device, accel.fp32_passes),
                         (48, True, 1 << 28, True, True, frozenset({"span", "residual"})))

    def test_invalid_values_and_combinations_are_refused(self):
        bad = ({"fitter_green_context_sms": 0}, {"fitter_green_context_sms": 4.0}, {"fitter_green_context_sms": True},
               {"fitter_green_workqueue": "balanced"}, {"fitter_green_context_sms": 48, "fitter_green_workqueue": "x"},
               {"fitter_green_context_sms": 48, "stream_priority": -1},
               {"fitter_green_context_sms": 48, "fitter_stream": "pooled"},
               {"fitter_cuda_graphs": 1}, {"fitter_graph_memory_bytes": 0}, {"fused_fingerprints": "yes"},
               {"finalize_on_fit_device": 1}, {"fitter_fp64": "OFF"}, {"fitter_fp64": "fp32:"},
               {"fitter_fp64": "fp32:span+quantize"}, {"fitter_gemm_precision": "tf32"},  # B6 not implemented
               {"fitter_fp64": "off", "fit_device": "cpu"}, {"fitter_cuda_graphs": True, "fit_device": "cpu"},
               {"finalize_on_fit_device": True, "fit_device": "cpu"},
               {"fitter_green_context_sms": 48, "fit_device": "cpu"})
        for values in bad:
            with self.subTest(**{key: str(value) for key, value in values.items()}), \
                    self.assertRaises(ConfigurationError):
                ReelCacheRuntimeConfig.from_dict({**self.base, **values})
        # fused fingerprints alone are a no-op for CPU tensors, so a CPU fitter accepts them
        config = ReelCacheRuntimeConfig.from_dict({**self.base, "fused_fingerprints": True, "fit_device": "cpu"})
        self.assertTrue(accel_b.FitterAccel.from_config(config).fused_fingerprints)

    def test_fitter_fp64_values(self):
        self.assertEqual(accel_b.parse_fitter_fp64("on"), frozenset())
        self.assertEqual(accel_b.parse_fitter_fp64("off"), frozenset(accel_b.OFF_PASSES))
        self.assertEqual(accel_b.parse_fitter_fp64("fp32:products+planner"), frozenset({"products", "planner"}))
        self.assertEqual(accel_b.parse_fitter_fp64("fp32:products_verify+products_recon+products_df"),
                         frozenset({"products_verify", "products_recon", "products_df"}))
        self.assertTrue(set(accel_b.OFF_PASSES) <= set(accel_b.FP32_PASSES))


class AccelPipelineTests(unittest.TestCase):
    def setUp(self):
        self.runtimes = RuntimeFactory(self)

    def _pipeline(self, runtime, **kwargs):
        pipeline = ArchivePipeline(runtime, method=method(), registry=registry(),
                                   options=ArchiveOptions(encode_threads=1), **kwargs)
        self.addCleanup(pipeline.close, drain=False, timeout=60)
        return pipeline

    def _chain(self, count=3, **kwargs):
        runtime = self.runtimes("accel-b")
        runtime.create_lineage("main")
        pipeline = self._pipeline(runtime, **kwargs)
        tickets = [pipeline.submit("main", f"chunk-{index:03d}", *chunk(index)) for index in range(count)]
        pipeline.drain("main", timeout=300)
        return [node_content(runtime, ticket.result(0)) for ticket in tickets], pipeline.stats()

    def test_cuda_options_are_refused_on_a_cpu_pipeline(self):
        runtime = self.runtimes("accel-b-refuse")
        for accel in (accel_b.FitterAccel(green_sms=48), accel_b.FitterAccel(cuda_graphs=True),
                      accel_b.FitterAccel(finalize_on_fit_device=True),
                      accel_b.FitterAccel(fp32_passes=frozenset({"span"}))):
            with self.subTest(accel=accel), self.assertRaises(ConfigurationError):
                ArchivePipeline(runtime, method=method(), registry=registry(), fitter_accel=accel)
        with self.assertRaises(ConfigurationError):
            ArchivePipeline(runtime, method=method(), registry=registry(), fitter_accel={"fused_fingerprints": True})

    def test_defaults_and_cpu_tensors_keep_the_base_nodes(self):
        base, stats = self._chain()
        self.assertNotIn("accel_b", stats)
        same, stats_default = self._chain(fitter_accel=accel_b.FitterAccel())  # the default options are no options
        self.assertEqual(same, base)
        self.assertNotIn("accel_b", stats_default)
        fused, stats_fused = self._chain(fitter_accel=accel_b.FitterAccel(fused_fingerprints=True))
        self.assertEqual(fused, base)  # CPU tensors: the base fingerprint path
        self.assertTrue(stats_fused["accel_b"]["fused_fingerprints"])
        self.assertIsNone(stats_fused["accel_b"]["cuda_graphs"])

    def test_lineage_archive_passes_the_options(self):
        from reelcache.integrations.sglang import lineage_archive
        source = open(lineage_archive.__file__).read()
        self.assertIn('yields["fitter_accel"] = fitter_accel', source)
        config = ReelCacheRuntimeConfig.from_dict({"store_root": "accel-b", "fused_fingerprints": True,
                                                "fit_device": "cpu"})
        self.assertEqual(accel_b.FitterAccel.from_config(config), accel_b.FitterAccel(fused_fingerprints=True))


class AccelScopeTests(unittest.TestCase):
    def test_scope_is_thread_local_and_cpu_tensors_stay_on_the_base_path(self):
        accel = accel_b.FitterAccel(fused_fingerprints=True, finalize_on_fit_device=True,
                                    fp32_passes=frozenset(accel_b.FP32_PASSES))
        generator = torch.Generator().manual_seed(4)
        source = torch.randn(96, 24, generator=generator).to(torch.bfloat16)
        basis = torch.linalg.qr(torch.randn(24, 6, generator=generator)).Q.T.contiguous().to(torch.float16)
        seen = {}
        outside = (run_one(projection_routine(source, basis, rank=6, tile_rows=32, products="fp64")),
                   run_one(source_energy_routine(source, tile_rows=32)), source_fingerprints([source]),
                   _fingerprint_terms(source).tolist())
        with accel_b.scope(accel, fit_device="cpu"):
            self.assertIs(accel_b.current(), accel)
            self.assertTrue(accel_b.fp32("span"))
            self.assertFalse(accel_b.fp32("span", source))  # CPU tensor: FP64 as the base
            self.assertEqual(accel_b.products("fp64", source), "fp64")
            self.assertIs(accel_b.fingerprint_source(source), source)  # the fit device is the CPU
            thread = threading.Thread(target=lambda: seen.setdefault("other", accel_b.current()))
            thread.start()
            thread.join()
            inside = (run_one(projection_routine(source, basis, rank=6, tile_rows=32, products="fp64")),
                      run_one(source_energy_routine(source, tile_rows=32)), source_fingerprints([source]),
                      _fingerprint_terms(source).tolist())
            projection = inside[0]
            span = run_one(span_energy_routine(projection.coefficients, basis, projection.source_norm ** 2))
        self.assertIsNone(seen["other"])
        self.assertIsNone(accel_b.current())
        self.assertEqual(dataclasses.astuple(inside[1])[:5], dataclasses.astuple(outside[1])[:5])  # not seconds
        self.assertEqual((inside[0].relative_residual, inside[0].source_norm, inside[0].condition_bound),
                         (outside[0].relative_residual, outside[0].source_norm, outside[0].condition_bound))
        self.assertEqual(inside[2:], outside[2:])
        reference = run_one(span_energy_routine(outside[0].coefficients, basis, outside[0].source_norm ** 2))
        self.assertEqual(span, reference)

    def test_graph_scope_ignores_cpu_tensors(self):
        class _Fail:
            enabled = True
            device = torch.device("cuda", 0)

            def run(self, *args, **kwargs):  # pragma: no cover - must never be reached with CPU tensors
                raise AssertionError("a CPU tensor took the graph path")

        generator = torch.Generator().manual_seed(3)
        source = torch.randn(64, 16, generator=generator)
        basis = torch.linalg.qr(torch.randn(16, 4, generator=generator)).Q.T.contiguous().to(torch.float16)
        with graph_capture.graph_scope(_Fail()):
            inside = run_one(projection_routine(source, basis, rank=4, tile_rows=16))
            energy = run_one(source_energy_routine(source, tile_rows=16))
        outside = run_one(projection_routine(source, basis, rank=4, tile_rows=16))
        self.assertEqual((inside.relative_residual, inside.source_norm), (outside.relative_residual,
                                                                          outside.source_norm))
        self.assertEqual(energy.scaled_energy, run_one(source_energy_routine(source, tile_rows=16)).scaled_energy)

    def test_span_neumann_matches_the_fp64_routine_on_near_orthonormal_rows(self):
        # the FP32 span routine's arithmetic (exact_blas.mm falls back to torch.mm on the CPU) on a host matrix
        generator = torch.Generator().manual_seed(9)
        rows16 = torch.linalg.qr(torch.randn(256, 40, generator=generator, dtype=torch.float64)).Q.T.contiguous()
        rows16 = rows16.to(torch.float16)
        coefficients = torch.randn(300, 40, generator=generator) * torch.logspace(0, -2, 40)
        source_energy = float(coefficients.double().square().sum()) * 1.05
        reference = run_one(span_energy_routine(coefficients, rows16, source_energy))
        value = run_one(accel_b.span_energy_fp32(coefficients, rows16, source_energy))
        self.assertIsNotNone(value)
        self.assertLess(abs(value - reference), 1e-7)
        wide = rows16.clone()
        wide[1] = wide[0]  # dependent rows: eta >= 1, the routine declines (the FP64 path raises as before)
        self.assertIsNone(run_one(accel_b.span_energy_fp32(coefficients, wide, source_energy)))


if __name__ == "__main__":
    unittest.main()
