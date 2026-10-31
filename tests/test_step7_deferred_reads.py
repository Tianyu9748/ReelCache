"""Step 7: lockstep routines with batched host reads equal the sequential reference bit for bit.

``reelcache.deferred_reads`` re-expresses the planner's and finalization's per-matrix measurements as
routines whose host reads are batched across matrices. These tests compare every routine with the
function it mirrors (same kernels, same host scalars: identical values and exceptions), the
lockstep driver's sequential exception semantics, the batched transfers, and the batched tile
finalization against the per-tile reference. CUDA variants run only inside GPU jobs
(``REELCACHE_STEP7_GPU_TESTS=1``), so the no-skip CPU suite never collects or skips them.
"""

from __future__ import annotations

import dataclasses
import os
import unittest

import torch

from reelcache import deferred_reads as dr
from reelcache import node_packets as npk
from reelcache.errors import ConfigurationError, IntegrityError
from reelcache.factorization import basis_condition_bound, fast_condition_bound, measure_source_energy, project_onto_basis
from reelcache.rank_method import span_energy, stored_energy
from reelcache.types import CodecSpec, MatrixKey, ModelIdentity, TensorContext


def _basis(channels: int, rank: int, seed: int, *, device="cpu") -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    q, _ = torch.linalg.qr(torch.randn(channels, rank, generator=generator, dtype=torch.float64))
    return q.T.to(torch.float16).contiguous().to(device)


def _matrix(rows: int, channels: int, seed: int, *, dtype=torch.bfloat16, device="cpu", rank=24) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    low = torch.randn(rows, rank, generator=generator) @ torch.randn(rank, channels, generator=generator)
    noise = 0.05 * torch.randn(rows, channels, generator=generator)
    return (low + noise).to(dtype).to(device)


def _same_projection(test, got, want):
    test.assertTrue(torch.equal(got.coefficients, want.coefficients))
    for name in ("relative_residual", "source_norm", "residual_norm", "zero_input", "condition_bound"):
        test.assertEqual(getattr(got, name), getattr(want, name), name)


class _RoutineChecks:
    device = "cpu"

    def test_projection_matches_reference(self):
        for seed, dtype in ((1, torch.bfloat16), (2, torch.float32), (3, torch.float16)):
            matrix = _matrix(300, 64, seed, dtype=dtype, device=self.device)
            basis = _basis(64, 40, seed + 10, device=self.device)
            for rank in (8, 40):
                view = basis[:rank]
                want = project_onto_basis(matrix, view, rank=rank, tile_rows=128,
                                          _condition_bound=fast_condition_bound(view))
                got = dr.run_one(dr.projection_routine(matrix, view, rank=rank, tile_rows=128, condition="fast"))
                _same_projection(self, got, want)
                scale = float(torch.maximum(matrix.amax().abs(), matrix.amin().abs()))
                again = dr.run_one(dr.projection_routine(matrix, view, rank=rank, tile_rows=128, scale=scale))
                _same_projection(self, again, want)
            exact = project_onto_basis(matrix, basis, tile_rows=100)
            _same_projection(self, dr.run_one(dr.projection_routine(matrix, basis, tile_rows=100, condition=None)),
                             exact)

    def test_zero_input_and_errors_match_reference(self):
        zero = torch.zeros(50, 32, dtype=torch.bfloat16, device=self.device)
        basis = _basis(32, 8, 4, device=self.device)
        want = project_onto_basis(zero, basis, _condition_bound=fast_condition_bound(basis))
        got = dr.run_one(dr.projection_routine(zero, basis, condition="fast"))
        _same_projection(self, got, want)
        bad = _matrix(50, 32, 5, dtype=torch.float32, device=self.device)
        bad[3, 4] = float("nan")
        for function in (lambda: project_onto_basis(bad, basis), lambda: dr.run_one(dr.projection_routine(bad, basis))):
            with self.assertRaisesRegex(ConfigurationError, "nonempty finite"):
                function()
        nonfinite = basis.clone()
        nonfinite[0, 0] = float("inf")
        good = _matrix(50, 32, 6, device=self.device)
        with self.assertRaisesRegex(ConfigurationError, "stored basis must be finite"):
            dr.run_one(dr.projection_routine(good, nonfinite))
        with self.assertRaisesRegex(ConfigurationError, "stored basis must be finite"):
            project_onto_basis(good, nonfinite, _condition_bound=fast_condition_bound(nonfinite))

    def test_source_energy_matches_reference(self):
        for seed, dtype in ((7, torch.bfloat16), (8, torch.float32)):
            matrix = _matrix(1030, 48, seed, dtype=dtype, device=self.device)
            want = measure_source_energy(matrix, tile_rows=256)
            got = dr.run_one(dr.source_energy_routine(matrix, tile_rows=256))
            for name in ("scale", "scaled_energy", "source_norm", "source_energy", "zero_input"):
                self.assertEqual(getattr(got, name), getattr(want, name), name)
        zero = dr.run_one(dr.source_energy_routine(torch.zeros(4, 4, device=self.device)))
        self.assertTrue(zero.zero_input)
        self.assertEqual(zero.scaled_energy, 0.0)

    def test_condition_and_span_energy_match_reference(self):
        near = _basis(96, 64, 9, device=self.device)
        self.assertEqual(dr.run_one(dr.condition_routine(near)), fast_condition_bound(near))
        skewed = near.clone()
        skewed[1] = (near[0].float() * 0.9 + near[1].float() * 0.1).to(torch.float16)
        self.assertEqual(dr.run_one(dr.condition_routine(skewed)), fast_condition_bound(skewed))
        self.assertEqual(dr.run_one(dr.condition_routine(skewed)), basis_condition_bound(skewed))
        matrix = _matrix(200, 96, 10, device=self.device)
        projection = project_onto_basis(matrix, near[:32], rank=32)
        energy = float(projection.source_norm) ** 2
        self.assertEqual(dr.run_one(dr.span_energy_routine(projection.coefficients, near[:32], energy)),
                         span_energy(projection.coefficients, near[:32], energy))
        self.assertEqual(dr.run_one(dr.stored_energy_routine(matrix, near, 32, 64)), stored_energy(matrix, near, 32, 64))

    def test_lockstep_batches_many_routines_identically(self):
        matrices = [_matrix(257, 64, seed, device=self.device) for seed in range(20, 26)]
        basis = _basis(64, 32, 30, device=self.device)
        results = dr.run_lockstep(dr.projection_routine(m, basis, rank=16 + 2 * i, condition="fast")
                                  for i, m in enumerate(matrices))
        for i, (matrix, got) in enumerate(zip(matrices, results)):
            view = basis[:16 + 2 * i]
            _same_projection(self, got, project_onto_basis(matrix, view, rank=16 + 2 * i,
                                                           _condition_bound=fast_condition_bound(basis)))

    def test_upload_and_download_preserve_values(self):
        host = [torch.randn(5, 7).to(torch.float16), torch.randn(3).to(torch.float16), torch.arange(4.0)]
        placed = dr.upload(host, self.device)
        for source, target in zip(host, placed):
            self.assertEqual(target.device.type, torch.device(self.device).type)
            self.assertTrue(torch.equal(source, target.cpu()))
            if target.device.type == "cuda":
                self.assertEqual(target.data_ptr() % dr.UPLOAD_ALIGNMENT, 0)
                self.assertTrue(target.is_contiguous())
        back = dr.download(placed)
        for source, target in zip(host, back):
            self.assertEqual(target.device.type, "cpu")
            self.assertTrue(torch.equal(source, target))

    def test_batched_finalization_matches_reference_composition(self):
        model = ModelIdentity("toy", "fp", 2, 64, "tokens_channels", "bfloat16")
        items = []
        for layer in range(2):
            for target in ("K", "V"):
                key = MatrixKey(layer, target)
                context = TensorContext(model, key, 0, 2, 4, 6, "pre_rope" if target == "K" else "none")
                matrix = _matrix(48, 64, 40 + 2 * layer + (target == "V"), device=self.device)
                rank = 16 if target == "K" else 24
                rows = _basis(64, rank, 50 + layer, device=self.device)
                spec = CodecSpec(kind="svd_hevc", rank=rank, inter_prediction=False)
                items.append(npk.FinalizeItem(matrix, rows, context, spec, key, 0.0, False, rows.cpu()))
        tiles = npk.finalize_tiles(items, tile_rows=20)
        for item, (tile, pixels) in zip(items, tiles):
            projection = project_onto_basis(item.matrix, item.rows16, rank=item.spec.rank, tile_rows=20,
                                            _condition_bound=fast_condition_bound(item.rows16))
            reference_pixels, affine, layout = npk.quantize_device(projection.coefficients.float(), item.spec)
            self.assertTrue(torch.equal(pixels, reference_pixels))
            self.assertEqual(tile.affine, affine.tobytes())
            self.assertEqual(dict(tile.normalization), layout)
            self.assertEqual(tile.stored_energy, 1.0 - projection.relative_residual ** 2)
            self.assertEqual(tile.condition_bound, projection.condition_bound)
            energy = float(projection.source_norm) ** 2
            self.assertEqual(tile.ideal_energy, min(1.0, span_energy(projection.coefficients, item.rows16, energy)))
            self.assertEqual(tile.basis, item.rows16.cpu().numpy().tobytes())
        # Errors follow item order: the second item misses its target and is reported, not the third
        # (whose rows do not match its spec and fail before any read).
        failing = list(items)
        failing[1] = dataclasses.replace(items[1], target=0.999999)
        failing[2] = dataclasses.replace(items[2], rows16=items[2].rows16[:8], host_rows=None)
        with self.assertRaisesRegex(IntegrityError, "below 0.999999"):
            npk.finalize_tiles(failing, tile_rows=20)


class DeferredReadsCPUTests(_RoutineChecks, unittest.TestCase):
    device = "cpu"

    def test_lockstep_exceptions_follow_sequential_order(self):
        closed = []

        def routine(index, fail_round):
            try:
                for round_ in range(4):
                    if round_ == fail_round:
                        raise ValueError(f"routine {index} failed in round {round_}")
                    yield (torch.tensor(float(index)),)
                return index
            finally:
                closed.append(index)

        # Routine 3 fails in round 0, routine 1 later in round 2: the sequential order reports routine 1.
        with self.assertRaisesRegex(ValueError, "routine 1 failed in round 2"):
            dr.run_lockstep([routine(0, None), routine(1, 2), routine(2, None), routine(3, 0), routine(4, None)])
        # Routine 4 never starts (as in a sequential run); every started routine is finished or closed.
        self.assertEqual(sorted(closed), [0, 1, 2, 3])
        self.assertEqual(dr.run_lockstep([routine(i, None) for i in range(3)]), [0, 1, 2])

    def test_read_host_types(self):
        values = dr.read_host([(torch.tensor(True), torch.tensor(3), torch.tensor(1.5, dtype=torch.bfloat16)),
                               (torch.arange(6, dtype=torch.float32).reshape(2, 3),)])
        self.assertEqual(values[0], (True, 3, 1.5))
        self.assertIsInstance(values[0][0], bool)
        self.assertIsInstance(values[0][1], int)
        self.assertEqual(values[1][0].shape, (2, 3))
        self.assertEqual(values[1][0][1, 2], 5.0)


if os.environ.get("REELCACHE_STEP7_GPU_TESTS") == "1":
    class DeferredReadsCUDATests(_RoutineChecks, unittest.TestCase):
        """Bitwise identity on CUDA (inside GPU jobs only; collected only with REELCACHE_STEP7_GPU_TESTS=1)."""

        device = "cuda"

        def setUp(self):
            if not torch.cuda.is_available():
                raise AssertionError("REELCACHE_STEP7_GPU_TESTS=1 requires a CUDA device")
            torch.backends.cuda.matmul.allow_tf32 = False


if __name__ == "__main__":
    unittest.main()
