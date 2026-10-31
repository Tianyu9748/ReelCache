import tempfile
import threading
import unittest
from contextlib import contextmanager
from unittest import mock

import torch

from reelcache import cusolver_eigh
from reelcache.codecs import HEVCProvider
from reelcache.errors import ConfigurationError, StateError
from reelcache.factorization import basis_condition_bound, fast_condition_bound
from reelcache.gpu_fitting import GPUFitter, gram_bases
from reelcache.manager import KVRuntime
from reelcache.types import CodecSpec, MatrixKey, ModelIdentity, TensorContext
from tests.test_codecs import context


def _subspace_gap(a, b):
    """1 - mean squared cosine between two orthonormal row bases."""
    return 1.0 - float(torch.linalg.matrix_norm(a @ b.T) ** 2) / a.shape[0]


class GramBasesTests(unittest.TestCase):
    def test_batched_gram_bases_match_svd_subspaces_in_descending_order(self):
        generator = torch.Generator().manual_seed(7)
        sources, ranks = [], [3, 5, 2]
        for rank in ranks:
            left = torch.linalg.qr(torch.randn(60, 8, generator=generator))[0]
            right = torch.linalg.qr(torch.randn(8, 8, generator=generator))[0]
            spectrum = torch.logspace(0, -3, 8)
            sources.append((left * spectrum) @ right.T)
        bases = gram_bases(sources, ranks)
        for source, rank, basis in zip(sources, ranks, bases):
            _, _, vt = torch.linalg.svd(source.double(), full_matrices=False)
            self.assertEqual(tuple(basis.shape), (rank, 8))
            self.assertLess(_subspace_gap(basis.double(), vt[:rank]), 1e-6)
            energy = (source @ basis.T).square().sum(0)
            self.assertTrue(bool((energy[:-1] >= energy[1:]).all()))

    def test_custom_row_eigensolver_is_used(self):
        source = torch.randn(20, 4, generator=torch.Generator().manual_seed(3))

        def rows(grams):
            values, columns = torch.linalg.eigh(grams)
            return values, columns.transpose(1, 2)

        expected = gram_bases([source], [2])[0]
        actual = gram_bases([source], [2], eigh=rows)[0]
        self.assertTrue(torch.equal(expected, actual))

    def test_invalid_batches_are_rejected(self):
        with self.assertRaises(ConfigurationError):
            gram_bases([], [])
        with self.assertRaises(ConfigurationError):
            gram_bases([torch.ones(4, 3), torch.ones(4, 2)], [1, 1])

    def test_cusolver_wrapper_rejects_cpu_input_without_cuda(self):
        with mock.patch("torch.cuda._lazy_init", side_effect=AssertionError("CUDA init")):
            with self.assertRaises(ConfigurationError):
                cusolver_eigh.eigh_batched(torch.eye(3).unsqueeze(0))


class FastConditionBoundTests(unittest.TestCase):
    def test_upper_bounds_exact_for_rounded_orthonormal_bases(self):
        generator = torch.Generator().manual_seed(11)
        for rank, channels in ((4, 16), (32, 96), (64, 128)):
            q = torch.linalg.qr(torch.randn(channels, rank, generator=generator))[0]
            basis = q.T.contiguous().half()
            exact, fast = basis_condition_bound(basis), fast_condition_bound(basis)
            self.assertGreaterEqual(fast, exact)
            self.assertLess(fast, 1.5)

    def test_non_orthonormal_falls_back_to_exact(self):
        basis = torch.tensor([[1.0, 0.0], [0.9, 0.1]], dtype=torch.float16)
        self.assertEqual(fast_condition_bound(basis), basis_condition_bound(basis))


class DeferredValueCheckTests(unittest.TestCase):
    def test_deferred_value_scan_keeps_structural_and_key_space_rules(self):
        from reelcache.factorization import validate_matrix
        bad = torch.full((24, 4), float("nan"))
        with self.assertRaises(ConfigurationError):
            validate_matrix(bad, context())
        self.assertEqual(tuple(validate_matrix(bad, context(), check_values=False).shape), (24, 4))
        with self.assertRaises(ConfigurationError):
            validate_matrix(torch.ones(24, 4), context(key_space="post_rope"), check_values=False)
        with self.assertRaises(ConfigurationError):
            validate_matrix(torch.ones(24, 3), context(), check_values=False)

    def test_cpu_encoder_still_rejects_nonfinite_lossy_input(self):
        from reelcache.codecs import encode_tensor
        with self.assertRaises(ConfigurationError):
            encode_tensor(torch.full((24, 4), float("inf")), context(), CodecSpec("svd", rank=2))


class GPUFitterBatchedContractTests(unittest.TestCase):
    def test_constructor_and_non_batched_prefit_do_not_touch_cuda(self):
        with mock.patch("torch.cuda._lazy_init", side_effect=AssertionError("CUDA init")):
            GPUFitter(gpu_id=0, gpu_uuid="GPU-test", method="gram_batched", batch_size=8)
            fitter = GPUFitter(gpu_id=0, gpu_uuid="GPU-test", method="gram")
            fitter.prefit({MatrixKey(0, "K"): (torch.ones(24, 4), context(), 2)})
            self.assertEqual(fitter.prefit_stats["matrices"], 0)
        for size in (0, 257, 1.5):
            with self.assertRaises(ConfigurationError):
                GPUFitter(gpu_id=0, gpu_uuid="GPU-test", method="gram_batched", batch_size=size)

    def test_prefit_rejects_post_rope_key_before_gpu(self):
        fitter = GPUFitter(gpu_id=0, gpu_uuid="GPU-test", method="gram_batched")
        with mock.patch("torch.cuda.device_count", side_effect=AssertionError("early CUDA")):
            with self.assertRaises(ConfigurationError):
                fitter.prefit({MatrixKey(0, "K"): (torch.ones(24, 4), context(key_space="post_rope"), 2)})

    def test_prefit_rejects_foreign_thread(self):
        fitter = GPUFitter(gpu_id=0, gpu_uuid="GPU-test", method="gram_batched")
        errors = []

        def worker():
            try:
                fitter.prefit({})
            except StateError:
                errors.append(True)

        thread = threading.Thread(target=worker)
        thread.start()
        thread.join(10)
        self.assertEqual(errors, [True])

    def test_shared_fitter_admits_pool_threads_and_owner_can_move(self):
        fitter = GPUFitter(gpu_id=0, gpu_uuid="GPU-test", shared=True)
        self.assertTrue(fitter.side_stream)
        errors = []

        def worker():
            try:
                with mock.patch("torch.cuda.device_count", return_value=0):
                    fitter.fit(torch.ones(24, 4), context(), 2)
            except BaseException as exc:
                errors.append(type(exc).__name__)

        thread = threading.Thread(target=worker)
        thread.start()
        thread.join(10)
        self.assertEqual(errors, ["UnsupportedError"])
        private = GPUFitter(gpu_id=0, gpu_uuid="GPU-test")
        moved = []

        def adopt():
            private.adopt_current_thread()
            try:
                with mock.patch("torch.cuda.device_count", return_value=0):
                    private.fit(torch.ones(24, 4), context(), 2)
            except BaseException as exc:
                moved.append(type(exc).__name__)

        thread = threading.Thread(target=adopt)
        thread.start()
        thread.join(10)
        self.assertEqual(moved, ["UnsupportedError"])
        with self.assertRaises(StateError):
            private.fit(torch.ones(24, 4), context(), 2)
        for value in (1, None):
            with self.assertRaises(ConfigurationError):
                GPUFitter(gpu_id=0, gpu_uuid="GPU-test", shared=value)

    def test_cache_identity_tracks_storage_view_version_and_request(self):
        tensor = torch.ones(24, 4)
        base = GPUFitter._identity(tensor, context(), 2)
        self.assertEqual(base, GPUFitter._identity(tensor.detach(), context(), 2))
        self.assertNotEqual(base, GPUFitter._identity(tensor, context(), 3))
        self.assertNotEqual(base, GPUFitter._identity(tensor.clone(), context(), 2))
        tensor.add_(1)
        self.assertNotEqual(base, GPUFitter._identity(tensor, context(), 2))

    def test_provider_prefit_is_noop_without_batched_fitter(self):
        with HEVCProvider().prefit({}, {}, {}):
            pass
        fitter = GPUFitter(gpu_id=0, gpu_uuid="GPU-test", method="randomized")
        with mock.patch.object(fitter, "prefit", side_effect=AssertionError("unexpected prefit")):
            with HEVCProvider(fitter=fitter).prefit({}, {}, {}):
                pass


class BatchedEighDispatchTests(unittest.TestCase):
    """The batched eigensolver is chosen explicitly and never silently degraded (Step 4)."""

    def test_auto_prefers_torch_batched_xsyev_then_ctypes_then_fails_loudly(self):
        from reelcache import gpu_fitting
        from reelcache.errors import UnsupportedError
        with mock.patch.object(gpu_fitting, "torch_batched_eigh_is_xsyevbatched", return_value=True):
            label, solver = gpu_fitting.select_eigh_solver("auto")
            self.assertEqual(label, "torch_linalg_eigh_batched_xsyevbatched")
            self.assertEqual(gpu_fitting.select_eigh_solver("torch_batched")[0], label)
        with mock.patch.object(gpu_fitting, "torch_batched_eigh_is_xsyevbatched", return_value=False), \
                mock.patch.object(cusolver_eigh, "available", return_value=True):
            self.assertEqual(gpu_fitting.select_eigh_solver("auto")[0], "cusolver_xsyevbatched_ctypes")
            with self.assertRaises(UnsupportedError):
                gpu_fitting.select_eigh_solver("torch_batched")
        with mock.patch.object(gpu_fitting, "torch_batched_eigh_is_xsyevbatched", return_value=False), \
                mock.patch.object(cusolver_eigh, "available", return_value=False):
            for name in ("auto", "cusolver_ctypes"):
                with self.assertRaisesRegex(UnsupportedError, "batched eigensolver was requested"):
                    gpu_fitting.select_eigh_solver(name)
            label, solver = gpu_fitting.select_eigh_solver("torch_loop")
            self.assertEqual(label, "torch_eigh_loop_explicit")
        with self.assertRaises(ConfigurationError):
            gpu_fitting.select_eigh_solver("magma")

    def test_version_gate_and_row_conventions(self):
        from reelcache import gpu_fitting
        with mock.patch.object(torch, "__version__", "2.8.0+cu128"):
            self.assertEqual(gpu_fitting._torch_version(), (2, 8))
            self.assertFalse(gpu_fitting.torch_batched_eigh_is_xsyevbatched())
        with mock.patch.object(torch, "__version__", "2.13.0+cu130"), mock.patch.object(torch.version, "cuda", "13.0"):
            self.assertTrue(gpu_fitting.torch_batched_eigh_is_xsyevbatched())
        with mock.patch.object(torch, "__version__", "2.13.0+cpu"), mock.patch.object(torch.version, "cuda", None):
            self.assertFalse(gpu_fitting.torch_batched_eigh_is_xsyevbatched())
        grams = torch.randn(3, 6, 6, generator=torch.Generator().manual_seed(1))
        grams = grams @ grams.transpose(1, 2)
        values, rows = gpu_fitting._torch_rows_eigh(grams)
        loop_values, loop_rows = gpu_fitting._torch_loop_eigh(grams)
        self.assertTrue(torch.allclose(values, loop_values))
        for index in range(3):
            self.assertTrue(torch.allclose(grams[index] @ rows[index, -1], values[index, -1] * rows[index, -1],
                                           atol=1e-4))
            self.assertTrue(torch.allclose(rows[index].abs(), loop_rows[index].abs(), atol=1e-5))

    def test_prefit_fails_loudly_instead_of_a_silent_loop(self):
        from reelcache import gpu_fitting
        from reelcache.errors import UnsupportedError
        fitter = GPUFitter(gpu_id=0, gpu_uuid="GPU-test", method="gram_batched")
        self.assertEqual(fitter.eigh_solver, "auto")
        with mock.patch.object(fitter, "_device", return_value=torch.device("cpu")), \
                mock.patch.object(gpu_fitting, "torch_batched_eigh_is_xsyevbatched", return_value=False), \
                mock.patch.object(cusolver_eigh, "available", return_value=False):
            with self.assertRaises(UnsupportedError):
                fitter.prefit({MatrixKey(0, "K"): (torch.ones(24, 4), context(), 2)})
        self.assertIsNone(fitter.prefit_stats["solver"])
        with self.assertRaises(ConfigurationError):
            GPUFitter(gpu_id=0, gpu_uuid="GPU-test", eigh_solver="loop")


class RuntimePrefitHookTests(unittest.TestCase):
    identity = ModelIdentity("prefit", "test-v1", 2, 4, "canonical", "float32")

    def inputs(self):
        tensors = {MatrixKey(layer, t): torch.randn(24, 4, generator=torch.Generator().manual_seed(layer))
                   for layer in range(2) for t in ("K", "V")}
        contexts = {k: TensorContext(self.identity, k, 0, 2, 3, 4, "pre_rope" if k.target == "K" else "none")
                    for k in tensors}
        return tensors, contexts

    def test_hook_sees_lossy_matrices_once_and_always_exits(self):
        calls, exits = [], []

        @contextmanager
        def prefit(matrices, contexts, specs):
            calls.append((sorted(matrices), sorted(specs[k].kind for k in specs)))
            try:
                yield
            finally:
                exits.append(True)

        tensors, contexts = self.inputs()
        specs = {k: CodecSpec() if k.layer == 0 else CodecSpec("svd", rank=2) for k in tensors}
        with tempfile.TemporaryDirectory() as directory, KVRuntime(directory, self.identity, prefit=prefit) as runtime:
            runtime.create_branch("main")
            runtime.seal_chunk("main", "c0", tensors, contexts, specs)
            self.assertEqual(calls, [([MatrixKey(1, "K"), MatrixKey(1, "V")], ["svd", "svd"])])
            self.assertEqual(exits, [True])
            runtime.seal_chunk("main", "c1", tensors, {k: TensorContext(self.identity, k, 24, 2, 3, 4, c.key_space)
                                                       for k, c in contexts.items()}, CodecSpec())
            self.assertEqual(len(calls), 1)
            encoder = runtime.encoder
            with mock.patch.object(runtime, "encoder", side_effect=RuntimeError("encode failed")):
                with self.assertRaisesRegex(RuntimeError, "encode failed"):
                    runtime.seal_chunk("main", "c2", tensors, {
                        k: TensorContext(self.identity, k, 48, 2, 3, 4, c.key_space) for k, c in contexts.items()
                    }, specs)
            self.assertEqual(len(calls), 2)
            self.assertEqual(exits, [True, True])
            self.assertIs(runtime.encoder, encoder)

    def test_prefit_must_be_callable(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ConfigurationError):
                KVRuntime(directory, self.identity, prefit=object())


if __name__ == "__main__":
    unittest.main()
