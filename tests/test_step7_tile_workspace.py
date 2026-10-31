"""Host-stage temporaries in reused buffers (``reelcache.deferred_reads.TileWorkspace``) change no result.

Source fingerprints and finalization's projections and quantization keep their large per-tile host temporaries in
workspace buffers, so their speed no longer depends on how fast the host faults in fresh pages (huge-page
availability on a fragmented host). These tests check that every fingerprint, projection and finalized tile is
bitwise identical to the fresh-temporary arithmetic: fingerprints against freeze D's implementation (kept below
verbatim), projections and tiles with pooling switched off (``POOL_HOST_TEMPORARIES``) and against the per-tile
reference composition, with workspaces shared across matrices, shapes, dtypes and calls as the archive pipeline does.
"""
from __future__ import annotations

import unittest
from unittest import mock

import torch

from reelcache import deferred_reads as dr
from reelcache import lineage_commit as lc
from reelcache import node_packets as npk
from reelcache.factorization import fast_condition_bound, project_onto_basis
from reelcache.rank_method import span_energy
from reelcache.types import CodecSpec, MatrixKey, ModelIdentity, TensorContext


@torch.no_grad()
def _reference_fingerprint_terms(tensor: torch.Tensor) -> torch.Tensor:
    """Freeze D's ``lineage_commit._fingerprint_terms`` (fresh temporaries per tile), verbatim apart from names."""
    size = tensor.element_size()
    rows, columns = tensor.shape
    source = tensor.detach()
    device = source.device
    column = torch.arange(columns, device=device, dtype=torch.int64)
    weights = [(column * a + 7) % 1021 + 1 for a, _, _ in lc._FINGERPRINT_COMBINATIONS]
    totals = torch.zeros((1 if size == 2 else 2) * len(lc._FINGERPRINT_COMBINATIONS), dtype=torch.int64, device=device)
    tile_rows = max(1, lc.FINGERPRINT_TILE_ELEMENTS // columns)
    for start in range(0, rows, tile_rows):
        stop = min(rows, start + tile_rows)
        bits = source[start:stop].contiguous().view(torch.int16 if size == 2 else torch.int32).to(torch.int64)
        parts = [bits] if size == 2 else [bits & 0xFFFF, bits >> 16]
        row = torch.arange(start, stop, device=device, dtype=torch.int64)
        index = 0
        for part in parts:
            for (_, b, square), weight in zip(lc._FINGERPRINT_COMBINATIONS, weights):
                row_weights = (row * b + 3) % 1019 + 1
                term = part * part if square else part
                per_row = (term * weight).sum(1).remainder(lc._FINGERPRINT_PRIME)
                totals[index] += (per_row * row_weights).sum()
                index += 1
    return totals.remainder(lc._FINGERPRINT_PRIME)


def _reference_fingerprint(tensor: torch.Tensor) -> str:
    return lc._fingerprint_text(tensor, _reference_fingerprint_terms(tensor).tolist())


def _bits_matrix(rows: int, columns: int, dtype: torch.dtype, seed: int) -> torch.Tensor:
    """Arbitrary bit patterns (signs, extremes, NaN payloads) of a 16- or 32-bit floating dtype."""
    generator = torch.Generator().manual_seed(seed)
    if torch.empty((), dtype=dtype).element_size() == 2:
        bits = torch.randint(-(1 << 15), 1 << 15, (rows, columns), generator=generator, dtype=torch.int32)
        return bits.to(torch.int16).view(dtype)
    bits = torch.randint(-(1 << 31), (1 << 31) - 1, (rows, columns), generator=generator, dtype=torch.int64)
    return bits.to(torch.int32).view(dtype)


def _basis(channels: int, rank: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    q, _ = torch.linalg.qr(torch.randn(channels, rank, generator=generator, dtype=torch.float64))
    return q.T.to(torch.float16).contiguous()


def _matrix(rows: int, channels: int, seed: int, *, dtype=torch.bfloat16, rank=24) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    low = torch.randn(rows, rank, generator=generator) @ torch.randn(rank, channels, generator=generator)
    return (low + 0.05 * torch.randn(rows, channels, generator=generator)).to(dtype)


def _same_projection(test, got, want):
    test.assertTrue(torch.equal(got.coefficients, want.coefficients))
    for name in ("relative_residual", "source_norm", "residual_norm", "zero_input", "condition_bound"):
        test.assertEqual(getattr(got, name), getattr(want, name), name)


class TileWorkspaceTests(unittest.TestCase):
    def test_buffers_grow_and_views_are_contiguous(self):
        work = dr.TileWorkspace()
        small = work.get("a", (3, 4), torch.float64, "cpu")
        self.assertEqual(tuple(small.shape), (3, 4))
        self.assertTrue(small.is_contiguous())
        big = work.get("a", (10, 7), torch.float64, "cpu")
        self.assertEqual(tuple(big.shape), (10, 7))
        again = work.get("a", (2, 5), torch.float64, "cpu")
        self.assertEqual(again.data_ptr(), big.data_ptr())  # the grown buffer is reused for smaller requests
        other = work.get("a", (2, 5), torch.int64, "cpu")  # names are per dtype
        self.assertEqual(other.dtype, torch.int64)
        self.assertNotEqual(other.data_ptr(), again.data_ptr())
        self.assertEqual(tuple(work.get("b", (0, 4), torch.float32, "cpu").shape), (0, 4))

    def test_fingerprints_equal_freeze_d_implementation(self):
        wide = _bits_matrix(64, 300, torch.bfloat16, 9)
        matrices = [
            _bits_matrix(300, 64, torch.bfloat16, 1), _bits_matrix(97, 33, torch.float16, 2),
            _bits_matrix(301, 64, torch.float32, 3), _bits_matrix(1, 7, torch.bfloat16, 4),
            _bits_matrix(150, 50, torch.float32, 5),
            wide.T,  # non-contiguous
            _bits_matrix(40, 80, torch.bfloat16, 6)[:, 5:69],  # column slice (non-contiguous rows)
            _matrix(200, 64, 7, dtype=torch.bfloat16),
        ]
        for tile_elements in (lc.FINGERPRINT_TILE_ELEMENTS, 640, 64):  # one tile; many tiles with a partial last one
            with mock.patch.object(lc, "FINGERPRINT_TILE_ELEMENTS", tile_elements):
                want = [_reference_fingerprint(matrix) for matrix in matrices]
                self.assertEqual(lc.source_fingerprints(matrices), want)  # one workspace across shapes and dtypes
                self.assertEqual([lc.source_fingerprint(matrix) for matrix in matrices], want)
                self.assertEqual(lc.source_fingerprints(list(reversed(matrices))), list(reversed(want)))

    def test_projection_pooled_equals_fresh_temporaries(self):
        cases = [(_matrix(300, 64, seed, dtype=dtype), _basis(64, 40, seed + 10), rank)
                 for seed, dtype in ((1, torch.bfloat16), (2, torch.float32), (3, torch.float16))
                 for rank in (8, 40)]
        cases.append((_matrix(37, 48, 4), _basis(48, 12, 14), 12))
        for products in ("fp32", "fp64"):
            work = dr.TileWorkspace()  # shared across matrices of different shapes and dtypes
            for matrix, basis, rank in cases:
                view = basis[:rank]
                for scale in (None, float(torch.maximum(matrix.amax().abs(), matrix.amin().abs()))):
                    kwargs = dict(rank=rank, tile_rows=128, condition="fast", scale=scale, products=products)
                    with mock.patch.object(dr, "POOL_HOST_TEMPORARIES", False):
                        want = dr.run_one(dr.projection_routine(matrix, view, **kwargs))
                    _same_projection(self, dr.run_one(dr.projection_routine(matrix, view, workspace=work, **kwargs)),
                                     want)
                    _same_projection(self, dr.run_one(dr.projection_routine(matrix, view, **kwargs)), want)
                    if products == "fp32":
                        reference = project_onto_basis(matrix, view, rank=rank, tile_rows=128,
                                                       _condition_bound=fast_condition_bound(view))
                        _same_projection(self, want, reference)

    def test_lockstep_routines_sharing_one_workspace(self):
        basis = _basis(64, 40, 21)
        matrices = [_matrix(120 + 30 * i, 64, 30 + i) for i in range(4)]
        work = dr.TileWorkspace()
        got = dr.run_lockstep(dr.projection_routine(m, basis, rank=16 + 4 * i, tile_rows=50, workspace=work)
                              for i, m in enumerate(matrices))
        with mock.patch.object(dr, "POOL_HOST_TEMPORARIES", False):
            fresh = dr.run_lockstep(dr.projection_routine(m, basis, rank=16 + 4 * i, tile_rows=50)
                                    for i, m in enumerate(matrices))
        for i, (matrix, projection, want) in enumerate(zip(matrices, got, fresh)):
            _same_projection(self, projection, want)
            view = basis[:16 + 4 * i]  # the condition bound is the passed basis's; the projection the view's
            reference = project_onto_basis(matrix, view, rank=16 + 4 * i, tile_rows=50,
                                           _condition_bound=projection.condition_bound)
            _same_projection(self, projection, reference)

    def test_finalize_with_a_shared_workspace_matches_reference_composition(self):
        model = ModelIdentity("toy", "fp", 3, 64, "tokens_channels", "bfloat16")
        items = []
        for layer in range(3):
            for target in ("K", "V"):
                key = MatrixKey(layer, target)
                frames = 2 if layer < 2 else 3  # two geometries: the workspace buffers grow
                context = TensorContext(model, key, 0, frames, 4, 6, "pre_rope" if target == "K" else "none")
                matrix = _matrix(frames * 24, 64, 60 + 2 * layer + (target == "V"))
                rank = 16 if target == "K" else 24 + 4 * layer
                rows = _basis(64, rank, 70 + layer)
                spec = CodecSpec(kind="svd_hevc", rank=rank, inter_prediction=False)
                items.append(npk.FinalizeItem(matrix, rows, context, spec, key, 0.0, False, rows))
        work = dr.TileWorkspace()
        tiles = npk.finalize_tiles(items[:3], tile_rows=20, workspace=work) + npk.finalize_tiles(
            items[3:], tile_rows=20, workspace=work)
        with mock.patch.object(dr, "POOL_HOST_TEMPORARIES", False):
            fresh = npk.finalize_tiles(items, tile_rows=20)
        for item, (tile, pixels), (fresh_tile, fresh_pixels) in zip(items, tiles, fresh):
            self.assertTrue(torch.equal(pixels, fresh_pixels))
            self.assertEqual(tile, fresh_tile)
            projection = project_onto_basis(item.matrix, item.rows16, rank=item.spec.rank, tile_rows=20,
                                            _condition_bound=fast_condition_bound(item.rows16))
            reference_pixels, affine, layout = npk.quantize_device(projection.coefficients.float(), item.spec)
            self.assertTrue(torch.equal(pixels, reference_pixels))
            self.assertEqual(tile.affine, affine.tobytes())
            self.assertEqual(dict(tile.normalization), layout)
            self.assertEqual(tile.stored_energy, 1.0 - projection.relative_residual ** 2)
            energy = float(projection.source_norm) ** 2
            self.assertEqual(tile.ideal_energy, min(1.0, span_energy(projection.coefficients, item.rows16, energy)))

    def test_quantization_of_degenerate_scales_matches(self):
        """Columns whose expanded affine scale is zero, negative or NaN quantize identically on both paths (the
        pooled path masks them with ``masked_fill_``, the fresh one with ``torch.where``)."""
        model = ModelIdentity("toy", "fp", 1, 64, "tokens_channels", "bfloat16")
        key = MatrixKey(0, "K")
        context = TensorContext(model, key, 0, 2, 4, 6, "pre_rope")
        rows = _basis(64, 16, 81)
        items = [npk.FinalizeItem(matrix, rows, context, CodecSpec(kind="svd_hevc", rank=16, inter_prediction=False),
                                  key, 0.0, True, rows)
                 for matrix in (_matrix(48, 64, 82), torch.zeros(48, 64, dtype=torch.bfloat16))]
        real = npk._hevc._expanded_affine

        def degenerate(affine, layout, columns):
            scale, offset = real(affine, layout, columns)
            scale = scale.copy()
            scale[1], scale[3], scale[5] = 0.0, -2.0, float("nan")
            return scale, offset

        with mock.patch.object(npk._hevc, "_expanded_affine", degenerate):
            got = npk.finalize_tiles(items, tile_rows=20)
            with mock.patch.object(dr, "POOL_HOST_TEMPORARIES", False):
                want = npk.finalize_tiles(items, tile_rows=20)
        for (tile, pixels), (want_tile, want_pixels) in zip(got, want):
            self.assertTrue(torch.equal(pixels, want_pixels))
            self.assertEqual(tile, want_tile)
            self.assertTrue(bool((pixels[:, 1] == 0).all()) and bool((pixels[:, 5] == 0).all()))


class WorkspaceLifetimeTests(unittest.TestCase):
    """Step 7 review (freeze E): workspace bytes are accounted and released, routine-owned workspaces hold no buffers
    across a yield (N routines in lockstep never hold N sets of buffers), and the archive pipeline records them."""

    def test_accounting_release_and_collection(self):
        import gc
        base = dr.workspace_stats()["live_bytes"]
        work = dr.TileWorkspace()
        work.get("a", (4, 8), torch.float64, "cpu")
        work.get("b", (16,), torch.int64, "cpu")
        self.assertEqual((work.nbytes, work.peak_bytes), (4 * 8 * 8 + 16 * 8, 4 * 8 * 8 + 16 * 8))
        work.get("a", (8, 8), torch.float64, "cpu")  # grows: the smaller buffer is released first
        self.assertEqual(work.nbytes, 8 * 8 * 8 + 16 * 8)
        self.assertEqual(dr.workspace_stats()["live_bytes"] - base, work.nbytes)
        self.assertEqual(work.release("b"), 16 * 8)
        self.assertEqual(work.nbytes, 8 * 8 * 8)
        if torch.cuda.is_available():  # device requests are fresh tensors: not pooled, not counted
            self.assertEqual(work.get("c", (3,), torch.float32, "cuda").device.type, "cuda")
            self.assertEqual(work.nbytes, 8 * 8 * 8)
        del work
        gc.collect()
        self.assertEqual(dr.workspace_stats()["live_bytes"], base)  # collected workspaces leave the process total

    def test_lockstep_routines_with_own_workspaces_hold_no_buffers_at_yields(self):
        basis = _basis(64, 40, 41)
        matrices = [_matrix(400, 64, 50 + i) for i in range(6)]
        base = dr.workspace_stats()["live_bytes"]
        held = []
        with dr.step_hook(lambda: held.append(dr.workspace_stats()["live_bytes"] - base)):
            got = dr.run_lockstep(dr.projection_routine(m, basis, rank=24, tile_rows=64) for m in matrices)
        self.assertTrue(held)
        self.assertEqual(max(held), 0)  # every routine is suspended at a yield whenever the hook runs
        with mock.patch.object(dr, "POOL_HOST_TEMPORARIES", False):
            fresh = dr.run_lockstep(dr.projection_routine(m, basis, rank=24, tile_rows=64) for m in matrices)
        for projection, want in zip(got, fresh):
            _same_projection(self, projection, want)
        shared = dr.TileWorkspace()  # a shared workspace keeps one set of buffers for every routine (by design)
        dr.run_lockstep(dr.projection_routine(m, basis, rank=24, tile_rows=64, workspace=shared) for m in matrices)
        one = dr.TileWorkspace()
        dr.run_one(dr.projection_routine(matrices[0], basis, rank=24, tile_rows=64, workspace=one))
        self.assertEqual(shared.peak_bytes, one.peak_bytes)

    def test_the_archive_pipeline_records_host_workspace_bytes(self):
        from tests.test_step7_raw_tier import _Pipelines

        class Probe(_Pipelines, unittest.TestCase):
            def runTest(self):
                pass

        probe = Probe()
        try:
            _, pipeline = probe.pipeline(name="workspace-stats")
            stats = pipeline.stats()["host_workspace"]
            self.assertEqual(set(stats), {"live_bytes", "peak_bytes", "allocations", "allocated_bytes"})
        finally:
            probe.doCleanups()


if __name__ == "__main__":
    unittest.main()
