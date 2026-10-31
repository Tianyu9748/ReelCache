"""SGLang Step 7 Part C: archival products independent of the process-global matmul flags (reelcache.exact_blas).

SGLang's RAVEN profile toggles ``torch.backends.cuda.matmul.allow_tf32`` around the model's FP32 regions while
archive_native plans concurrently. Every FP32 product/solve of planning, verification and finalization therefore
goes through :mod:`reelcache.exact_blas` (cuBLAS on Torch's thread-local handle with an explicit IEEE math mode).
CPU tests pin the argument replication and fallbacks; the CUDA tests (collected only with
``REELCACHE_STEP7_GPU_TESTS=1`` inside GPU jobs) check bitwise parity with Torch's untoggled products, immunity to the
flags, that planning/finalization never reach a flag-dependent Torch product, and that archival under a thread
toggling the flags publishes the same content (decisions, ranks, bytes) as an untoggled run.
"""

from __future__ import annotations

import contextlib
import os
import threading
import unittest
from unittest import mock

import torch

from reelcache import exact_blas
from reelcache.errors import ConfigurationError


class ExactBlasCPUTests(unittest.TestCase):
    def test_cpu_and_non_fp32_products_are_torch_products(self):
        generator = torch.Generator().manual_seed(4)
        a = torch.randn(37, 19, generator=generator)
        b = torch.randn(19, 23, generator=generator)
        self.assertTrue(torch.equal(exact_blas.mm(a, b), a @ b))
        self.assertTrue(torch.equal(exact_blas.mm(a.T.contiguous().T, b), a @ b))
        self.assertTrue(torch.equal(exact_blas.mm(a.double(), b.double()), a.double() @ b.double()))
        factor = torch.linalg.cholesky(a.T @ a + torch.eye(19), upper=True)
        self.assertTrue(torch.equal(exact_blas.trsm_right_upper(factor, a),
                                    torch.linalg.solve_triangular(factor, a, upper=True, left=False)))

    def test_shapes_are_validated(self):
        with self.assertRaises(ConfigurationError):
            exact_blas.mm(torch.ones(3, 4), torch.ones(5, 2))
        with self.assertRaises(ConfigurationError):
            exact_blas.mm(torch.ones(2, 3, 4), torch.ones(4, 2))
        with self.assertRaises(ConfigurationError):
            exact_blas.trsm_right_upper(torch.eye(3), torch.ones(5, 4))

    def test_cublas_operand_preparation_replicates_torch(self):
        prepare, dense = exact_blas._prepare, exact_blas._non_overlapping_and_dense
        base = torch.zeros(6, 8)
        self.assertTrue(dense((6, 8), (8, 1)) and dense((6, 8), (1, 6)) and dense((1, 8), (99, 1)))
        self.assertFalse(dense((6, 4), (8, 1)))
        tensor, transposed = prepare(base)
        self.assertTrue(tensor is base and transposed)
        tensor, transposed = prepare(base.T)
        self.assertFalse(transposed)
        self.assertEqual(tensor.data_ptr(), base.data_ptr())
        tensor, transposed = prepare(base[:, :5])  # row-major with a wider leading dimension
        self.assertTrue(transposed)
        self.assertEqual(tensor.data_ptr(), base.data_ptr())
        tensor, transposed = prepare(base.T[:5])  # column-major slice
        self.assertFalse(transposed)
        tensor, transposed = prepare(base[::2, ::2])  # neither layout: a contiguous copy
        self.assertTrue(transposed and tensor.is_contiguous())
        self.assertNotEqual(tensor.data_ptr(), base.data_ptr())


def _forbid_flag_dependent_torch_products():
    """Patch Torch's Python-level FP32 products to fail on CUDA FP32 operands (exact_blas must be used)."""
    def guard(name, original):
        def checked(*args, **kwargs):
            if any(isinstance(value, torch.Tensor) and value.is_cuda and value.dtype == torch.float32
                   for value in list(args) + list(kwargs.values())):
                raise AssertionError(f"flag-dependent CUDA FP32 product: {name}")
            return original(*args, **kwargs)
        return checked

    stack = contextlib.ExitStack()
    for owner, name in ((torch, "matmul"), (torch, "mm"), (torch, "bmm"), (torch, "addmm"), (torch, "baddbmm"),
                        (torch, "einsum"), (torch.Tensor, "__matmul__"), (torch.Tensor, "__rmatmul__"),
                        (torch.Tensor, "mm"), (torch.Tensor, "bmm"), (torch.Tensor, "matmul"),
                        (torch.linalg, "solve_triangular"), (torch.nn.functional, "linear")):
        stack.enter_context(mock.patch.object(owner, name, guard(f"{getattr(owner, '__name__', owner)}.{name}",
                                                                getattr(owner, name))))
    return stack


if os.environ.get("REELCACHE_STEP7_GPU_TESTS") == "1":
    from tests import step6_support as s6

    class ExactBlasCUDATests(unittest.TestCase):
        """Inside GPU jobs only (collected only with REELCACHE_STEP7_GPU_TESTS=1)."""

        @classmethod
        def setUpClass(cls):
            if not torch.cuda.is_available():
                raise AssertionError("REELCACHE_STEP7_GPU_TESTS=1 requires a CUDA device")

        def setUp(self):
            self.flags = (torch.backends.cuda.matmul.allow_tf32, torch.get_float32_matmul_precision())
            self.addCleanup(self.restore_flags)

        def restore_flags(self):
            torch.backends.cuda.matmul.allow_tf32 = self.flags[0]
            torch.set_float32_matmul_precision(self.flags[1])

        def test_products_equal_untoggled_torch_bitwise_and_ignore_the_flags(self):
            generator = torch.Generator(device="cuda").manual_seed(7)
            x = torch.randn(4680, 1536, device="cuda", generator=generator)
            w = torch.randn(1536, 160, device="cuda", generator=generator)
            rows = torch.randn(200, 1536, device="cuda", generator=generator)
            cases = [(x.T, x), (x, w), (x.T, x @ w), (x, rows.T), (x[:1024] @ rows.T, rows), (rows, rows.T),
                     (x[:, :700], w[:700]), (w.T, w), (x, w[:, :1]), (x[:1], w), (x[:2000].T, x[:2000, :64])]
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.set_float32_matmul_precision("highest")
            expected = [a @ b for a, b in cases]
            for allow, precision in ((False, "highest"), (True, "high"), (True, "medium"), (False, "high")):
                torch.backends.cuda.matmul.allow_tf32 = allow
                torch.set_float32_matmul_precision(precision)
                for (a, b), reference in zip(cases, expected):
                    self.assertTrue(torch.equal(exact_blas.mm(a, b), reference), (allow, precision, a.shape))
            factor = torch.linalg.cholesky_ex(rows @ rows.T + 200 * torch.eye(200, device="cuda"), upper=True)[0]
            block = torch.randn(1536, 200, device="cuda", generator=generator)
            torch.backends.cuda.matmul.allow_tf32 = False
            solved = exact_blas.trsm_right_upper(factor, block)
            torch.backends.cuda.matmul.allow_tf32 = True
            self.assertTrue(torch.equal(exact_blas.trsm_right_upper(factor, block), solved))
            reference = torch.linalg.solve_triangular(factor.double(), block.double(), upper=True, left=False)
            self.assertLess(float((solved.double() - reference).abs().max() / reference.abs().max()), 1e-5)

        def test_planning_and_finalization_never_reach_a_flag_dependent_product(self):
            from reelcache.gpu_fitting import select_eigh_solver
            from reelcache.lineage_commit import ArchiveOptions
            from reelcache.rank_method import RankMethodConfig
            label, eigh = select_eigh_solver("auto")
            base = s6.method()
            exact = s6.exact_method()
            for name, method, orthogonalizer in (
                    ("exact", exact, "batched"), ("reference", exact, "householder"),
                    ("topk", RankMethodConfig(base.targets, base.codec, solver="topk_rr"), "batched"),
                    ("warm", RankMethodConfig(base.targets, base.codec, solver="warm_start"), "batched"),
                    ("fp64", RankMethodConfig(base.targets, base.codec, matmul_precision="fp32_verified_fp64"),
                     "batched")):
                with self.subTest(name=name):
                    runtime = s6.RuntimeFactory(self)(f"flagfree-{name}")
                    runtime.create_lineage("main")
                    registry = s6.registry()
                    with _forbid_flag_dependent_torch_products():
                        for index in range(4):
                            matrices, contexts = s6.chunk(index)
                            matrices = {key: value.cuda() for key, value in matrices.items()}
                            runtime.commit_node("main", f"chunk-{index:03d}", matrices, contexts, method=method,
                                                registry=registry, device="cuda", eigh=eigh, eigh_label=label,
                                                options=ArchiveOptions(encode_threads=1,
                                                                       orthogonalizer=orthogonalizer))

        def test_archival_under_a_flag_toggling_thread_publishes_identical_content(self):
            from reelcache.archive_pipeline import ArchivePipeline
            from reelcache.gpu_fitting import GPUFitter
            from reelcache.lineage_commit import ArchiveOptions
            from reelcache.rank_method import RankMethodConfig
            uuid = "GPU-" + str(torch.cuda.get_device_properties(0).uuid).removeprefix("GPU-")
            chunks = [s6.chunk(index) for index in range(6)]
            base = s6.method(rate_lag=1)

            def archive(name, method, toggling):
                runtime = s6.RuntimeFactory(self)(name)
                runtime.create_lineage("main")
                pipeline = ArchivePipeline(runtime, method=method, registry=s6.registry(),
                                           options=ArchiveOptions(encode_threads=1), device="cuda",
                                           fitter=GPUFitter(gpu_id=0, gpu_uuid=uuid, method="gram_batched",
                                                            eigh_solver="auto"),
                                           encode_workers=2, max_source_chunks=2, max_unpublished=4)
                stop, toggles = threading.Event(), [0]

                def toggle():
                    settings = ((True, "high"), (False, "highest"), (True, "medium"), (False, "high"))
                    while not stop.is_set():
                        allow, precision = settings[toggles[0] % len(settings)]
                        torch.backends.cuda.matmul.allow_tf32 = allow
                        torch.set_float32_matmul_precision(precision)
                        toggles[0] += 1
                        stop.wait(0.0005)
                thread = threading.Thread(target=toggle, daemon=True)
                if toggling:
                    thread.start()
                try:
                    tickets = [pipeline.submit("main", f"chunk-{index:03d}",
                                               {key: value.cuda() for key, value in matrices.items()}, contexts)
                               for index, (matrices, contexts) in enumerate(chunks)]
                    head = pipeline.drain("main", timeout=600)
                finally:
                    stop.set()
                    if toggling:
                        thread.join()
                    pipeline.close(drain=False, timeout=60)
                    self.restore_flags()
                nodes = [ticket.result(timeout=0) for ticket in tickets]
                return nodes, head, toggles[0], runtime

            for name, method in (("exact", s6.exact_method(rate_lag=1)),
                                 ("topk", RankMethodConfig(base.targets, base.codec, rate_lag=1, solver="topk_rr")),
                                 ("fp64", RankMethodConfig(base.targets, base.codec, rate_lag=1,
                                                           matmul_precision="fp32_verified_fp64"))):
                with self.subTest(name=name):
                    torch.backends.cuda.matmul.allow_tf32 = False
                    torch.set_float32_matmul_precision("highest")
                    reference, head, _, runtime = archive(f"toggle-off-{name}", method, False)
                    toggled, toggled_head, toggles, other = archive(f"toggle-on-{name}", method, True)
                    self.assertGreater(toggles, 100)
                    # Content-addressed node ids: identical decisions, ranks, rows, coefficients and bytes.
                    self.assertEqual(toggled, reference)
                    self.assertEqual(toggled_head, head)
                    self.assertEqual(s6.node_content(other, toggled_head), s6.node_content(runtime, head))


if __name__ == "__main__":
    unittest.main()
