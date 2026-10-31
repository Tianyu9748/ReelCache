"""Accel study track B (2026-10-30): GPU tests of ``reelcache.accel_b`` (collected only with REELCACHE_ACCEL_B_GPU_TESTS=1,
inside GPU jobs; the CPU receipt never sees them, so it has no skips).

* B1 green contexts: the stream confines cuBLAS work (time scales with the SM count); allocations made on a green
  stream from another thread can be freed and ``torch.cuda.empty_cache()`` / the device trims / graph drops run
  afterwards without error (the contexts live for the process);
* B3 graphs (dev22l port): graphed routines bitwise the eager ones (energy; projections with known or unknown scale,
  fp32 / fp64 products), lockstep interleaving, budget and trim; a pipeline's nodes identical with graphs;
* B4 fused fingerprints: the same integers as ``lineage_commit._fingerprint_terms`` (16- and 32-bit, odd shapes,
  non-contiguous, several tiles, arbitrary bit patterns); a pipeline's nodes identical;
* B2 FP32 passes: exact IEEE division bitwise the FP64 quotient rounded to FP32; fused residual / energy sums and the
  Neumann span energy within 1e-7 of the FP64 routines; a pipeline's decisions and ranks identical, |dE| <= 1e-6.
"""

from __future__ import annotations

import os
import threading
import time
import unittest

import torch

if os.environ.get("REELCACHE_ACCEL_B_GPU_TESTS") == "1":
    from reelcache import accel_b
    from reelcache import graph_capture
    from reelcache import lineage_commit
    from reelcache.archive_pipeline import ArchivePipeline
    from reelcache.deferred_reads import (projection_routine, run_lockstep, run_one, source_energy_routine,
                                       span_energy_routine)
    from reelcache.exact_blas import mm
    from reelcache.lineage_commit import ArchiveOptions
    from tests.step6_support import RuntimeFactory, chunk, method, node_content, registry

    def _fields(projection):
        return (projection.relative_residual, projection.source_norm, projection.residual_norm,
                projection.condition_bound, projection.zero_input)

    def _energy_fields(energy):
        return (energy.scale, energy.scaled_energy, energy.source_norm, energy.source_energy, energy.zero_input)

    def _data(seed, rows=1100, channels=192, rank=24, dtype=torch.bfloat16):
        generator = torch.Generator().manual_seed(seed)
        source = (torch.randn(rows, 48, generator=generator) @ torch.randn(48, channels, generator=generator)
                  + 0.05 * torch.randn(rows, channels, generator=generator)).to(dtype).cuda()
        basis = torch.linalg.qr(torch.randn(channels, rank, generator=generator)).Q.T.contiguous()
        return source, basis.to(torch.float16).cuda()

    class GreenContextTests(unittest.TestCase):
        def test_confinement_and_the_allocator_hazard(self):
            stream = accel_b.green_stream("cuda:0", 24)
            self.assertIs(accel_b.green_stream("cuda:0", 24), stream)  # one context per process and size
            a = torch.randn(4096, 4096, device="cuda")

            def timed(target):
                with torch.cuda.stream(target):
                    mm(a, a)
                    target.synchronize()
                    started = time.perf_counter()
                    for _ in range(5):
                        mm(a, a)
                    target.synchronize()
                    return time.perf_counter() - started

            box = {}
            thread = threading.Thread(target=lambda: box.update(green=timed(stream), plain=timed(torch.cuda.Stream())))
            thread.start()
            thread.join()
            self.assertGreater(box["green"] / box["plain"], 3.0)  # 24 of 188 SMs: ~7.8x when confined

            def alloc_free():
                with torch.cuda.stream(stream):
                    tensors = [torch.randn(2048, 2048, device="cuda") for _ in range(4)]
                    mm(tensors[0], tensors[1])
                    stream.synchronize()
                    del tensors

            thread = threading.Thread(target=alloc_free)
            thread.start()
            thread.join()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            from reelcache.device_pressure import trim_device_caches
            trim_device_caches("test")
            cache = graph_capture.GraphCache(torch.device("cuda", 0))
            source, basis = _data(5)
            with torch.cuda.stream(stream), graph_capture.graph_scope(cache):
                for _ in range(3):
                    run_one(projection_routine(source, basis, rank=20, tile_rows=256, scale=1.0))
                stream.synchronize()
            self.assertGreater(cache.stats()["replays"], 0)
            cache.trim("test")
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    class GraphTests(unittest.TestCase):
        def setUp(self):
            self.stream = torch.cuda.Stream()

        def _call(self, function, cache=None, repeats=1):
            results = []
            with torch.cuda.stream(self.stream), graph_capture.graph_scope(cache):
                for _ in range(repeats):
                    results.append(function())
            torch.cuda.synchronize()
            return results

        def test_graphed_routines_are_bitwise_eager(self):
            source, basis = _data(1)
            for name, tensor in (("contiguous", source), ("strided", source.T.contiguous().T)):
                energy = self._call(lambda: run_one(source_energy_routine(tensor, tile_rows=256)))[0]
                for products in ("fp32", "fp64"):
                    for scale in (energy.scale, None):
                        kwargs = dict(rank=20, tile_rows=256, products=products, scale=scale)
                        with self.subTest(layout=name, products=products, scale=scale is not None):
                            eager = self._call(lambda: run_one(projection_routine(tensor, basis, **kwargs)))[0]
                            cache = graph_capture.GraphCache(source.device)
                            graphed = self._call(lambda: run_one(projection_routine(tensor, basis, **kwargs)), cache,
                                                 4)
                            stats = cache.stats()
                            self.assertEqual(stats["capture_failures"], 0, stats["last_capture_failure"])
                            self.assertGreaterEqual(stats["replays"], 3)
                            for result in graphed:
                                self.assertEqual(_fields(result), _fields(eager))
                                self.assertTrue(torch.equal(result.coefficients, eager.coefficients))
                cache = graph_capture.GraphCache(source.device)
                graphed = self._call(lambda: run_one(source_energy_routine(tensor, tile_rows=256)), cache, 4)
                self.assertEqual(cache.stats()["replays"], 3)
                for result in graphed:
                    self.assertEqual(_energy_fields(result), _energy_fields(energy))

        def test_lockstep_interleaving_budget_and_trim(self):
            items = [_data(10 + index) for index in range(6)]
            energies = self._call(lambda: run_lockstep(source_energy_routine(s, tile_rows=256) for s, _ in items))[0]

            def projections():
                return run_lockstep(projection_routine(s, b, rank=20, tile_rows=256, scale=e.scale)
                                    for (s, b), e in zip(items, energies))

            eager = self._call(projections)[0]
            cache = graph_capture.GraphCache(items[0][0].device)
            for result in self._call(projections, cache, 3):
                for got, want in zip(result, eager):
                    self.assertEqual(_fields(got), _fields(want))
                    self.assertTrue(torch.equal(got.coefficients, want.coefficients))
            self.assertGreater(cache.stats()["replays"], 0)
            tight = graph_capture.GraphCache(items[0][0].device, budget_bytes=1024)
            for result in self._call(projections, tight, 3):
                self.assertEqual([_fields(item) for item in result], [_fields(item) for item in eager])
            self.assertEqual(tight.stats()["captures"], 0)
            self.assertGreater(cache.trim("oom"), 0)
            for result in self._call(projections, cache, 3):
                self.assertEqual([_fields(item) for item in result], [_fields(item) for item in eager])

    class FusedKernelTests(unittest.TestCase):
        def test_fused_fingerprints_are_the_same_integers(self):
            generator = torch.Generator(device="cuda").manual_seed(2)
            bits = torch.randint(-2 ** 31, 2 ** 31 - 1, (777, 333), device="cuda", generator=generator,
                                 dtype=torch.int64).to(torch.int32)
            cases = {"fp32_bits": bits.view(torch.float32), "bf16": bits[:, :300].to(torch.int16).view(torch.bfloat16),
                     "fp16": bits.to(torch.int16).view(torch.float16),
                     "noncontiguous": bits.view(torch.float32)[5:700:2, 3:300],
                     "wide_rows": torch.randn(4096, 1536, device="cuda", generator=generator).to(torch.bfloat16)}
            for name, tensor in cases.items():
                for elements in (lineage_commit.FINGERPRINT_TILE_ELEMENTS, 1 << 14, 97):
                    with self.subTest(case=name, tile_elements=elements):
                        expected = lineage_commit._fingerprint_terms(tensor).tolist()
                        got = accel_b.fingerprint_terms_fused(tensor, lineage_commit._FINGERPRINT_PRIME,
                                                              elements).tolist()
                        self.assertEqual(got, expected)
            with accel_b.scope(accel_b.FitterAccel(fused_fingerprints=True)):
                fused = lineage_commit.source_fingerprints(list(cases.values()))
            self.assertEqual(fused, lineage_commit.source_fingerprints(list(cases.values())))

        def test_exact_division_is_the_fp64_quotient_rounded(self):
            all_bf16 = torch.arange(-32768, 32768, dtype=torch.int32, device="cuda").to(torch.int16)
            values = all_bf16.view(torch.bfloat16)
            values = values[torch.isfinite(values)]
            generator = torch.Generator(device="cuda").manual_seed(3)
            fp32 = torch.randint(-2 ** 31, 2 ** 31 - 1, (1 << 20,), device="cuda", generator=generator,
                                 dtype=torch.int64).to(torch.int32).view(torch.float32)
            fp32 = fp32[torch.isfinite(fp32)]
            for tensor in (values, fp32):  # FP32-representable scales: the planner pass's only case (planner_fp32)
                for scale in (float(tensor.abs().max()), 1.0, 3.0, 1.5, 2.0 ** -120, 2.0 ** 100,
                              float(tensor.abs()[tensor.abs() > 0].min())):
                    expected = (tensor.double() / scale).float()
                    got = accel_b.exact_div_fp32(tensor, scale)
                    self.assertTrue(torch.equal(got.view(torch.int32), expected.view(torch.int32)), (tensor.dtype,
                                                                                                     scale))

        def test_dot2_coefficients_round_like_fp64(self):
            generator = torch.Generator(device="cuda").manual_seed(8)
            for dtype, scale in ((torch.float32, 30.0), (torch.bfloat16, 1.0)):
                x = (torch.randn(1500, 1536, device="cuda", generator=generator) * scale).to(dtype)
                rows = torch.linalg.qr(torch.randn(1536, 300, device="cuda", generator=generator)).Q.T
                rows = rows.contiguous().to(torch.float16)
                exact = (x.double() @ rows.double().T).float()
                got = accel_b.dot2_gemm(x, rows)
                mismatch = float((got != exact).double().mean())
                ulps = (got.view(torch.int32).long() - exact.view(torch.int32).long()).abs().max()
                self.assertLess(mismatch, 1e-3, (dtype, mismatch))
                self.assertLessEqual(int(ulps), 1)
                self.assertTrue(torch.equal(got, accel_b.dot2_gemm(x, rows)))  # deterministic

        def test_fp32_passes_are_within_tolerance(self):
            source, basis = _data(7, rows=4680, channels=1536, rank=256)
            source32 = (source.float() * 30).contiguous()
            for tensor in (source, source32):
                base = run_one(source_energy_routine(tensor, tile_rows=1024))
                with accel_b.scope(accel_b.FitterAccel(fp32_passes=frozenset(accel_b.FP32_PASSES))):
                    fused = run_one(source_energy_routine(tensor, tile_rows=1024))
                    projection = run_one(projection_routine(tensor, basis, rank=200, tile_rows=1024,
                                                            products="fp64"))
                    span = run_one(span_energy_routine(projection.coefficients, basis[:200],
                                                       projection.source_norm ** 2))
                self.assertEqual(fused.scale, base.scale)
                self.assertLess(abs(fused.scaled_energy / base.scaled_energy - 1), 1e-7)
                reference = run_one(projection_routine(tensor, basis, rank=200, tile_rows=1024, products="fp64"))
                self.assertLess(abs(projection.relative_residual ** 2 - reference.relative_residual ** 2), 1e-7)
                ideal = run_one(span_energy_routine(reference.coefficients, basis[:200],
                                                    reference.source_norm ** 2))
                self.assertLess(abs(span - ideal), 1e-6)
            self.assertGreater(accel_b.stats()["span_neumann"], 0)

    class PipelineIdentityTests(unittest.TestCase):
        def setUp(self):
            self.runtimes = RuntimeFactory(self)

        def _chain(self, accel=None, count=4):
            runtime = self.runtimes("accel-b-cuda")
            runtime.create_lineage("main")
            pipeline = ArchivePipeline(runtime, method=method(1), registry=registry(),
                                       options=ArchiveOptions(encode_threads=1), device="cuda", encode_workers=2,
                                       max_source_chunks=4, max_unpublished=5, fitter_stream="dedicated",
                                       fitter_accel=accel)
            self.addCleanup(pipeline.close, drain=False, timeout=120)
            tickets = []
            for index in range(count):
                matrices, contexts = chunk(index)
                tickets.append(pipeline.submit("main", f"chunk-{index:03d}", {key: value.cuda() for key, value in
                                                                              matrices.items()}, contexts))
            pipeline.drain("main", timeout=600)
            return [node_content(self._runtime(pipeline), ticket.result(0)) for ticket in tickets], pipeline.stats()

        @staticmethod
        def _runtime(pipeline):
            return pipeline.runtime

        def test_bitwise_levers(self):
            base, _ = self._chain()
            for accel in (accel_b.FitterAccel(cuda_graphs=True), accel_b.FitterAccel(fused_fingerprints=True)):
                with self.subTest(accel=accel):
                    nodes, stats = self._chain(accel)
                    self.assertEqual(nodes, base)
                    self.assertIn("accel_b", stats)

        def test_numerics_levers_keep_decisions(self):
            base, _ = self._chain()
            for accel in (accel_b.FitterAccel(fp32_passes=frozenset(accel_b.OFF_PASSES)),
                          accel_b.FitterAccel(green_sms=48)):
                with self.subTest(accel=accel):
                    nodes, stats = self._chain(accel)
                    for want, got in zip(base, nodes):
                        for entry, row in want["rows"].items():
                            other = got["rows"][entry]
                            self.assertEqual((row[0], row[1]), (other[0], other[1]))  # decision, rank view
                            self.assertLessEqual(abs(row[4]["stored_factor_energy"]
                                                     - other[4]["stored_factor_energy"]), 1e-6)
                    if accel.green_sms:
                        self.assertEqual(stats["fitter_stream"]["note"], "green context: 48 SMs")
            torch.cuda.synchronize()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    unittest.main()
