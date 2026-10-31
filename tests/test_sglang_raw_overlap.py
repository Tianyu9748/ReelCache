"""Layer-overlapped RAW install through the SGLang bridge on CPU (Step 5c's O4f 158abd9b, adapted by Step 11 for the
F5 candidate: the overlap is on by default and the session handle's ``overlap_restore`` is a bool, so the plain
reference turns it off; F5 has no early start, so O4's early-restore test is left out. O4f's ``restore_allocates``
test is carried over unchanged; 3-way merge of O4d d79a29fe -> O4f into the O4d adaptation f1f93ad3, no conflicts).
Freeze F6 (overlay v7): Step 5c's F6 tests (c277d3bc) 3-way merged onto v6 (a711594d), no textual conflict; the F6
bridge test adapted to the same F5 conventions. F6 cut 9: the attach warm-up of the engine RoPE covers the in-place
(``out=``) variant.

With ``raw_restore="prefer"``, the ranges the raw tier serves (GPU raw, host raw) install layer by layer on their own
thread, through the same per-layer calls as the synchronous RAW path (``restore.DenseInstall``), inside the readiness of
their call group. A history mixing raw and compressed ranges has ONE readiness per layer covering both sources, so
every cache layer is awaited once. Latents, caches and chunk digests are bitwise those of the plain restore.

``DenseInstall``'s fast path (O4 cut 2: device rows read in place, host rows copied straight in, no residency staging,
no per-layer host wait; ONE engine RoPE call per layer over the gathered K, or one per span from its rows) writes
bitwise what the staged path (freeze F4's ``restore_dense``: per-span staging and reconstruction) writes, for one or
several spans, offset rows, reordered slots, CFG streams and a key dtype other than the cache's; rows in other dtypes
take the staged path. The plain restore runs the synchronous path with one call per layer, the overlapped install one
call per span.

F6 (in place): an engine RoPE that takes ``out=`` rotates each span's K straight into its cache slot. Bitwise the staged
path's, also with LongLive 2.0's multi-shot RoPE offsets, with no scratch tensor; a slot of another dtype or an engine
without ``out=`` keeps the output-tensor path; an engine that does not write its ``out=`` is an IntegrityError.
"""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from reelcache.errors import ConfigurationError, IntegrityError
from reelcache.integrations.sglang.lineage_archive import LineageSessionArchive
from reelcache.integrations.sglang.restore import DenseInstall, warm_engine_rope
from reelcache.residency import ResidencyPool
from tests.test_sglang_bridge import (
    CHANNELS,
    DIM,
    FRAMES,
    GRID,
    HEADS,
    LAYERS,
    TPF,
    handle,
    request,
    rope_cos_sin,
    rotate_key,
    wait_until,
)
from tests.test_sglang_bridge_overlap import OverlapStage, OverlapTestCase, drive


def tiers_of(summary):
    return {(item["stream"], item["chunk"]): item["tier"]
            for item in summary["restore"]["lease"]["tiers"]["per_chunk"]}


class Span:
    """The attended-span fields the RAW install reads. ``rope_start_frame``: default the chunk's first frame;
    LongLive 2.0 shifts it by ``shot_index * multi_shot_rope_offset``."""

    def __init__(self, chunk, slot, *, node_offset=0, length=FRAMES * TPF, rope_start_frame=None):
        self.token_start, self.slot_start = chunk * FRAMES * TPF, slot * FRAMES * TPF
        self.node_offset, self.length, self.frames = node_offset, length, FRAMES
        self.rope_start_frame = float(chunk * FRAMES if rope_start_frame is None else rope_start_frame)

    def to_dict(self):
        return dict(vars(self))


class OutRotation:
    """The toy engine rotation with ``out=`` (SGLang's ``causal_wan_rotate_key`` contract under Step 5c's F6 patch):
    the result written into ``out`` and returned. Records every ``out``."""

    def __init__(self, *, ignore_out=False):
        self.outs, self.ignore_out = [], ignore_out

    def __call__(self, key, cos, sin, out=None):
        result = rotate_key(key, cos, sin)
        if out is None or self.ignore_out:
            return result
        self.outs.append(out)
        out.copy_(result)
        return out


class DenseInstallTests(unittest.TestCase):
    """The fast path against the staged path (freeze F4's restore_dense), directly on the install."""

    def setUp(self):
        generator = torch.Generator().manual_seed(11)
        self.sources = {}  # (stream, chunk, layer, target) -> rows; chunk 1's node holds 8 leading rows of another range
        for stream in ("cond", "uncond"):
            for chunk in range(3):
                extra = 8 if chunk == 1 else 0
                for layer in range(LAYERS):
                    for target in ("K", "V"):
                        self.sources[(stream, chunk, layer, target)] = torch.randn(
                            extra + FRAMES * TPF, CHANNELS, generator=generator).to(torch.bfloat16)

    def install(self, fast, *, streams=("cond",), cache_k=torch.float32, value_dtype=None, residency=None,
                rope_per="layer", rotation=rotate_key, in_place=None, spans=None, key_dtype=torch.bfloat16):
        spans = spans or [Span(2, 0), Span(0, 1), Span(1, 2, node_offset=8)]  # slots out of chunk order, offset rows
        plans = {stream: SimpleNamespace(spans=tuple(spans)) for stream in streams}
        caches = {stream: [SimpleNamespace(k=torch.full((1, 4 * FRAMES * TPF, HEADS, DIM), float("nan"), dtype=cache_k),
                                           v=torch.full((1, 4 * FRAMES * TPF, HEADS, DIM), float("nan"),
                                                        dtype=torch.bfloat16))
                           for _ in range(LAYERS)] for stream in streams}

        def source(span, stream, key):
            rows = self.sources[(stream, span.token_start // (FRAMES * TPF), key.layer, key.target)]
            if key.target == "K":
                return rows.to(key_dtype)
            return rows.to(value_dtype) if value_dtype is not None else rows

        install = DenseInstall(plans=plans, caches=caches, rope_cos_sin=rope_cos_sin, rotate_key=rotation,
                               device=torch.device("cpu"), residency=residency or ResidencyPool(1 << 30),
                               grid=GRID, source=source, key_dtype=key_dtype, fast=fast, rope_per=rope_per,
                               in_place=in_place)
        for layer in range(LAYERS):
            install.layer(layer).release()
        self.assertIsNone(install.fence())  # CPU: the writes are complete
        return install, caches

    def assertSameCaches(self, left, right):
        def bits(tensor):  # bitwise, NaN payloads of the unattended slots included
            return tensor.view(torch.int16 if tensor.element_size() == 2 else torch.int32)

        for stream in right:
            for a, b in zip(left[stream], right[stream]):
                self.assertTrue(torch.equal(bits(a.k), bits(b.k)) and torch.equal(bits(a.v), bits(b.v)), stream)

    def test_the_fast_path_writes_the_staged_paths_bits(self):
        for streams in (("cond",), ("cond", "uncond")):
            with self.subTest(streams=streams):
                residency = ResidencyPool(1 << 30)
                fast, fast_caches = self.install(True, streams=streams, residency=residency)
                spanwise, span_caches = self.install(True, streams=streams, residency=residency, rope_per="span")
                staged, staged_caches = self.install(False, streams=streams)
                self.assertSameCaches(fast_caches, staged_caches)
                self.assertSameCaches(span_caches, staged_caches)
                self.assertEqual((spanwise.fast_layers, spanwise.rope_calls, spanwise.staged_rows),
                                 (LAYERS, 3 * len(streams) * LAYERS, 0))
                spans = 3 * len(streams)
                self.assertEqual((fast.fast_layers, fast.staged_layers, fast.rope_calls), (LAYERS, 0, LAYERS))
                self.assertEqual((staged.fast_layers, staged.staged_layers, staged.rope_calls),
                                 (0, LAYERS, spans * LAYERS))
                self.assertEqual((fast.staged_rows, staged.staged_rows), (0, 2 * spans * LAYERS))
                self.assertEqual(fast.jobs, staged.jobs)
                self.assertEqual(residency.stats()["peak_reserved_bytes"], 0)  # no staging on the fast path
                for caches in (fast_caches,):  # attended slots written, the free slot untouched
                    cache = caches["cond"][0]
                    self.assertFalse(cache.k[:, :3 * FRAMES * TPF].isnan().any())
                    self.assertTrue(cache.k[:, 3 * FRAMES * TPF:].isnan().all())

    def test_a_single_span_is_rotated_in_place(self):
        spans = [Span(0, 0)]
        plans = {"cond": SimpleNamespace(spans=tuple(spans))}
        results = []
        for fast in (True, False):
            caches = {"cond": [SimpleNamespace(k=torch.zeros((1, FRAMES * TPF, HEADS, DIM), dtype=torch.bfloat16),
                                               v=torch.zeros((1, FRAMES * TPF, HEADS, DIM), dtype=torch.bfloat16))
                               for _ in range(LAYERS)]}
            install = DenseInstall(plans=plans, caches=caches, rope_cos_sin=rope_cos_sin, rotate_key=rotate_key,
                                   device=torch.device("cpu"), residency=ResidencyPool(1 << 30), grid=GRID,
                                   source=lambda span, stream, key: self.sources[(stream, 0, key.layer, key.target)],
                                   fast=fast)
            for layer in range(LAYERS):
                install.layer(layer).release()
            results.append((install, caches))
        (fast, fast_caches), (_, staged_caches) = results
        self.assertSameCaches(fast_caches, staged_caches)
        self.assertEqual((fast.fast_layers, fast.in_place_rows, fast.staged_rows), (LAYERS, 2 * LAYERS, 0))

    def test_rows_in_other_dtypes_take_the_staged_path(self):
        fast, fast_caches = self.install(True, value_dtype=torch.float32)  # V rows float32, the cache bfloat16
        _, staged_caches = self.install(False, value_dtype=torch.float32)
        self.assertSameCaches(fast_caches, staged_caches)
        self.assertEqual((fast.fast_layers, fast.staged_layers), (0, LAYERS))

    # -- F6: the engine's RoPE straight into the cache slots ----------------------------------------------------
    def test_an_engine_rope_with_out_rotates_each_span_into_its_slot_bitwise(self):
        spans = len([Span(2, 0), Span(0, 1), Span(1, 2, node_offset=8)])
        for streams in (("cond",), ("cond", "uncond")):
            for rope_per in ("layer", "span"):
                with self.subTest(streams=streams, rope_per=rope_per):
                    rotation = OutRotation()
                    fast, fast_caches = self.install(True, streams=streams, cache_k=torch.bfloat16,
                                                     rope_per=rope_per, rotation=rotation)
                    _, staged_caches = self.install(False, streams=streams, cache_k=torch.bfloat16)
                    self.assertSameCaches(fast_caches, staged_caches)
                    calls = spans * len(streams) * LAYERS
                    self.assertTrue(fast.in_place)
                    self.assertEqual((fast.in_place_ropes, fast.rope_calls, fast.scratch_bytes), (calls, calls, 0))
                    caches = [tensor for items in fast_caches.values() for cache in items for tensor in (cache.k,)]
                    for out in rotation.outs:  # every rotation went to a view of a cache slot
                        self.assertTrue(any(out.untyped_storage().data_ptr() == cache.untyped_storage().data_ptr()
                                            for cache in caches))
                        self.assertEqual(tuple(out.shape), (1, FRAMES * TPF, HEADS, DIM))

    def test_longlive2_multi_shot_offsets_rotate_in_place_bitwise(self):
        """LongLive 2.0: rope_start_frame = start_frame + shot_index * multi_shot_rope_offset (here 3 shots, offset
        10 frames): each span's tables carry its offset; in place and staged write the same bits."""
        offset = 10.0
        spans = [Span(0, 0, rope_start_frame=0.0), Span(1, 1, rope_start_frame=FRAMES + offset, node_offset=8),
                 Span(2, 2, rope_start_frame=2 * FRAMES + 2 * offset)]
        fast, fast_caches = self.install(True, cache_k=torch.bfloat16, rotation=OutRotation(), spans=spans)
        _, staged_caches = self.install(False, cache_k=torch.bfloat16, spans=spans)
        _, unshifted = self.install(False, cache_k=torch.bfloat16)  # the same rows without the shot offsets
        self.assertSameCaches(fast_caches, staged_caches)
        self.assertEqual(fast.in_place_ropes, 3 * LAYERS)
        shifted_k = staged_caches["cond"][0].k[:, FRAMES * TPF:2 * FRAMES * TPF]
        plain_k = unshifted["cond"][0].k[:, FRAMES * TPF:2 * FRAMES * TPF]
        self.assertFalse(torch.equal(shifted_k, plain_k))  # the offsets reach the rotation

    def test_without_out_or_with_another_slot_dtype_the_output_tensor_path_is_kept(self):
        _, staged_caches = self.install(False)  # float32 K slots: the rotation is cast on store
        for kwargs, in_place in (({"rotation": OutRotation()}, True), ({"rotation": rotate_key}, False),
                                 ({"rotation": OutRotation(), "in_place": False}, False)):
            with self.subTest(**{key: str(value) for key, value in kwargs.items()}):
                fast, fast_caches = self.install(True, **kwargs)
                self.assertSameCaches(fast_caches, staged_caches)
                self.assertEqual((fast.in_place, fast.in_place_ropes), (in_place, 0))
                self.assertGreater(fast.scratch_bytes, 0)
        with self.assertRaises(ConfigurationError):
            self.install(True, in_place="yes")

    def test_float32_keys_rotate_in_place_into_a_bfloat16_cache_bitwise(self):
        """RAVEN: float32 pre-RoPE keys, bfloat16 cache. The engine rounds the rotation once into the slot, as the
        cast on store does; a wider slot (bfloat16 keys, float32 cache) keeps the output-tensor path."""
        fast, fast_caches = self.install(True, cache_k=torch.bfloat16, key_dtype=torch.float32, rotation=OutRotation())
        _, staged_caches = self.install(False, cache_k=torch.bfloat16, key_dtype=torch.float32)
        self.assertSameCaches(fast_caches, staged_caches)
        self.assertEqual((fast.in_place_ropes, fast.scratch_bytes), (3 * LAYERS, 0))
        wide, wide_caches = self.install(True, cache_k=torch.float32, rotation=OutRotation())
        _, staged_wide = self.install(False, cache_k=torch.float32)
        self.assertSameCaches(wide_caches, staged_wide)
        self.assertEqual(wide.in_place_ropes, 0)

    def test_an_engine_that_does_not_write_its_out_is_an_integrity_error(self):
        with self.assertRaisesRegex(IntegrityError, "did not write its out="):
            self.install(True, cache_k=torch.bfloat16, rotation=OutRotation(ignore_out=True))

    # -- F6 cut 9 with RoPE-into-slots: the attach warm-up covers the in-place launches ----------------------------
    def test_the_engine_rope_warm_up_covers_every_launch_of_an_in_place_restore(self):
        """Freeze F6 (overlay v7): SGLang's Triton RoPE kernel is compiled per argument dtypes and ``num_tokens % 16``
        and autotuned per argument dtypes, so the attach warm-up (``warm_engine_rope``) must have launched every
        variant (key dtype, written dtype, token class) a restore launches. With an engine rotation that takes
        ``out=`` it also rotates into an output of the cache dtype (``in_place_dtype``) at both classes. RAVEN's
        float32 keys into the bfloat16 cache are a variant of their own (cut 9 alone misses it); for same-dtype keys
        the in-place launch is the output-tensor variant. A cache wider than the keys keeps the output-tensor path,
        and an engine without ``out=`` has only output-tensor launches: neither warms in place. The restore here has
        spans of both classes (48 and 24 tokens)."""
        def tables(frames, height, width, *, start_frame, device):
            return rope_cos_sin(start_frame, frames)

        def recording(launches, *, takes_out):
            def launch(key, out):  # the variant: argument dtypes (without out=, the output is the key's) and class
                launches.append((key.dtype, key.dtype if out is None else out.dtype, key.shape[-3] % 16 == 0))

            if not takes_out:
                def rotate(key, cos, sin):
                    launch(key, None)
                    return rotate_key(key, cos, sin)
                return rotate
            rotation = OutRotation()

            def rotate(key, cos, sin, out=None):
                launch(key, out)
                return rotation(key, cos, sin, out=out)
            return rotate

        one_frame = Span(0, 1, length=TPF)
        one_frame.frames = 1
        spans = [Span(2, 0), one_frame, Span(1, 2, node_offset=8)]
        self.assertEqual([span.length % 16 == 0 for span in spans], [True, False, True])
        cases = ((torch.float32, "bfloat16", True, "torch.bfloat16"),  # RAVEN: float32 keys, bfloat16 cache
                 (torch.bfloat16, "bfloat16", True, "torch.bfloat16"),  # CausVid, LongLive
                 (torch.bfloat16, "float32", True, None),  # a wider cache: the output-tensor path
                 (torch.float32, "bfloat16", False, None))  # no out=: SGLang without Step 5c's patch
        for key_dtype, native, takes_out, in_place in cases:
            with self.subTest(key=key_dtype, cache=native, takes_out=takes_out):
                warmed, restored = [], []
                profile = SimpleNamespace(default_grid=lambda: GRID, chunk_frames=FRAMES, num_heads=HEADS,
                                          head_dim=DIM, native_dtype=native)
                model = SimpleNamespace(dtype_for=lambda target, dtype=key_dtype: str(dtype).removeprefix("torch."))
                transformer = SimpleNamespace(causal_rotary_cos_sin=tables,
                                              causal_rotate_key=recording(warmed, takes_out=takes_out))
                report = {"components": {}}
                warm_engine_rope(report, transformer, profile, model, torch.device("cpu"))
                rope = report["components"]["engine_rope"]
                self.assertNotIn("error", rope)
                self.assertEqual(rope["detail"].get("in_place_dtype"), in_place)
                self.assertEqual(rope["detail"]["dtype"], str(key_dtype))
                install, _ = self.install(True, cache_k=getattr(torch, native), key_dtype=key_dtype, spans=spans,
                                          rotation=recording(restored, takes_out=takes_out))
                self.assertEqual(install.in_place_ropes, LAYERS * len(spans) if in_place else 0)
                written = {key_dtype if in_place is None else getattr(torch, native)}
                self.assertEqual({out for _, out, _ in restored}, written)
                if in_place:  # one launch per span: both classes
                    self.assertEqual({divisible for _, _, divisible in restored}, {True, False})
                self.assertLessEqual(set(restored), set(warmed))  # every launch of the restore was warmed

    def test_a_short_range_is_an_integrity_error(self):
        plans = {"cond": SimpleNamespace(spans=(Span(0, 0, length=FRAMES * TPF + 1),))}
        caches = {"cond": [SimpleNamespace(k=torch.zeros((1, 2 * FRAMES * TPF, HEADS, DIM)),
                                           v=torch.zeros((1, 2 * FRAMES * TPF, HEADS, DIM)))] * LAYERS}
        with self.assertRaisesRegex(IntegrityError, "shorter than the attended span"):
            DenseInstall(plans=plans, caches=caches, rope_cos_sin=lambda start, frames: rope_cos_sin(start, frames),
                         rotate_key=rotate_key, device=torch.device("cpu"), residency=ResidencyPool(1 << 30),
                         grid=GRID, source=lambda span, stream, key: self.sources[("cond", 0, 0, "K")]).layer(0)


class RawOverlapTests(OverlapTestCase):
    def raw_bridge(self, store="archive"):
        return self.bridge(store, raw_restore="prefer")

    def placed(self, bridge, stage, placement, *, chunks=3, streams=("cond",)):
        """Session L with ``chunks`` chunks, its raw entries placed per chunk (``raw_tier`` control; at most the
        GPU raw budget, 2 chunks here, as ``gpu_raw``)."""
        drive(bridge, request(handle("L", pause_after_chunks=chunks), streams=streams), stage)
        bridge.drain_all()
        bridge.archive.raw_tier.flush(30)
        bridge.control("raw_tier", {"session_id": "L", "action": "demote"})  # the GPU raw budget for the placement
        bridge.control("raw_tier", {"session_id": "L", "action": "place", "placement": placement})
        raw = sum(1 for tier in placement.values() if tier != "compressed")
        wait_until(lambda: bridge.control("describe", {"session_id": "L"})["raw_tier"]["entries"]
                   == chunks - (len(placement) - raw))

    def forks(self, bridge, stage, *, checkpoint=2, streams=("cond",), **overlap):
        """The plain fork, then the overlapped one, of L at ``checkpoint``: (plain, snapshot, overlapped, latents,
        caches)."""
        def fork(name, **extra):
            return drive(bridge, request(handle(name, "fork", from_session="L", checkpoint=checkpoint,
                                                pause_after_chunks=checkpoint + 1, **extra), streams=streams), stage)

        plain, latents_p, caches_p, _ = fork("p", overlap_restore=False)
        expected = self.snapshot(latents_p, caches_p)
        summary, latents, caches, _ = fork("o", overlap_restore=True, **overlap)
        return plain, expected, summary, latents, caches

    def assertBitwise(self, plain, expected, summary, latents, caches):
        self.assertTrue(torch.equal(latents, expected[0]))
        self.assertSameCaches(caches, expected[1])
        self.assertEqual(summary["chunk_digests"], plain["chunk_digests"])
        self.assertTrue(all(cache.awaited == 1 for items in caches.values() for cache in items))  # one wait each
        self.assertNothingPending(caches)

    def test_an_all_raw_history_installs_layer_overlapped_bitwise(self):
        """GPU raw and host raw alone: a raw-only readiness (source ``raw``); step 0 waits per layer; bitwise the
        synchronous RAW path."""
        for tier in ("gpu_raw", "host_raw"):
            with self.subTest(tier=tier):
                bridge, stage = self.raw_bridge(f"archive-{tier}"), OverlapStage()
                self.placed(bridge, stage, {"0": tier, "1": tier})
                plain, expected, summary, latents, caches = self.forks(bridge, stage)
                self.assertEqual(set(tiers_of(plain).values()), {tier})
                self.assertEqual(tiers_of(summary), tiers_of(plain))
                record = summary["restore"]["overlap"]
                self.assertEqual((record["calls"], record["fallback"], record["error"]), (1, None, None))
                self.assertEqual([item["layer"] for item in record["waits"]], list(range(LAYERS)))
                [producer] = record["producer"]
                self.assertEqual(producer["sources"], ["raw"])
                lease = summary["restore"]["lease"]
                self.assertEqual(lease["overlap"]["raw_calls"], 1)
                [raw] = lease["overlap"]["raw_install"]
                self.assertEqual((raw["spans"], raw["jobs"], raw["layers"]), (2, 2 * 2 * LAYERS, LAYERS))
                self.assertEqual((raw["fast_layers"], raw["staged_layers"], raw["rope_per"], raw["rope_calls"]),
                                 (LAYERS, 0, "span", 2 * LAYERS))  # overlapped: one RoPE call per span
                self.assertEqual((raw["in_place_rows"], raw["staged_rows"]), (2 * 2 * LAYERS, 0))
                synchronous = plain["restore"]["lease"]["raw_install"]  # the plain path: one RoPE call per layer
                self.assertEqual((synchronous["fast_layers"], synchronous["rope_calls"], synchronous["staged_rows"]),
                                 (LAYERS, LAYERS, 0))
                self.assertTrue(raw["non_blocking"])
                self.assertEqual(lease["jobs"], plain["restore"]["lease"]["jobs"])  # the plain path's records
                self.assertEqual(lease["spans"], plain["restore"]["lease"]["spans"])
                self.assertGreaterEqual(lease["tiers"]["raw_install_seconds"], 0.0)
                self.assertEqual(lease["verification"]["verified_payloads"], 0)  # raw tiers read no payload
                self.assertTrue(lease["closed"])
                self.assertBitwise(plain, expected, summary, latents, caches)
                self.assertLifetimesRetired(bridge, summary)

    def allocates(self, bridge, stage, name, *, checkpoint=2):
        """``restore_allocates`` of the plans the bridge restores, queried as its restore starts (a plain fork)."""
        seen, real = [], bridge.archive.restore

        def spy(**kwargs):
            seen.append(bridge.archive.restore_allocates(kwargs["plans"]))
            return real(**kwargs)

        with mock.patch.object(bridge.archive, "restore", spy):
            drive(bridge, request(handle(name, "fork", from_session="L", checkpoint=checkpoint,
                                         pause_after_chunks=checkpoint + 1)), stage)
        return seen

    def test_only_an_all_gpu_raw_history_restores_without_allocating(self):
        """Step 11's cut 4 (``restore_allocates``): False only when every attended range is a GPU raw copy and the
        RAW install reads it in place (``raw_install_in_place``, O4's fast path); host raw, compressed and mixed
        histories allocate."""
        self.assertTrue(LineageSessionArchive.raw_install_in_place)
        cases = (({"0": "gpu_raw", "1": "gpu_raw"}, False), ({"0": "host_raw", "1": "host_raw"}, True),
                 ({"0": "gpu_raw", "1": "host_raw"}, True), ({"0": "gpu_raw", "1": "compressed"}, True))
        for index, (placement, expected) in enumerate(cases):
            with self.subTest(placement=placement):
                bridge, stage = self.raw_bridge(f"archive-{index}"), OverlapStage()
                self.placed(bridge, stage, placement)
                self.assertEqual(self.allocates(bridge, stage, "q"), [expected])
                if not expected:  # a staging RAW install (F5 as composed) allocates for the same history
                    bridge.archive.raw_install_in_place = False
                    self.assertEqual(self.allocates(bridge, stage, "q2"), [True])
        bridge, stage = self.bridge("archive-off"), OverlapStage()  # raw_restore "off": compressed restores
        self.session_with_history(bridge, stage)
        self.assertEqual(self.allocates(bridge, stage, "q"), [True])
        self.assertTrue(bridge.archive.restore_allocates({}))

    def test_a_mixed_history_has_one_readiness_per_layer_covering_both_sources(self):
        """Host raw, compressed and GPU raw chunks: one overlapped call whose readiness has the sources ``restore``
        and ``raw``; every layer is released once both marked it, and each cache layer is awaited once."""
        bridge, stage = self.raw_bridge(), OverlapStage()
        self.placed(bridge, stage, {"0": "host_raw", "1": "compressed", "2": "gpu_raw"}, chunks=4)
        plain, expected, summary, latents, caches = self.forks(bridge, stage, checkpoint=3)
        tiers = tiers_of(summary)
        self.assertEqual((tiers[("cond", 0)], tiers[("cond", 2)]), ("host_raw", "gpu_raw"))
        self.assertIn(tiers[("cond", 1)], ("compressed_prepared", "compressed_cold"))
        self.assertEqual(tiers, tiers_of(plain))
        record = summary["restore"]["overlap"]
        self.assertEqual((record["calls"], record["fallback"], record["error"]), (1, None, None))
        self.assertEqual(record["layers_waited"], LAYERS)
        [producer] = record["producer"]
        self.assertEqual(producer["sources"], ["restore", "raw"])
        self.assertEqual(producer["raw_install"]["spans"], 2)
        [timeline] = record["timelines"]
        for layer, ready in enumerate(timeline["marked_at"]):  # released when the later source marked it
            self.assertEqual(ready, max(timeline["sources"]["restore"][layer], timeline["sources"]["raw"][layer]))
        lease = summary["restore"]["lease"]
        self.assertEqual(lease["overlap"]["raw_calls"], 1)
        self.assertEqual([item["stream"] for item in lease["restorer"]], [None])  # the compressed report
        self.assertEqual(lease["jobs"], plain["restore"]["lease"]["jobs"])
        self.assertBitwise(plain, expected, summary, latents, caches)
        self.assertLifetimesRetired(bridge, summary)

    def test_cfg_streams_chain_one_mixed_readiness_per_stream(self):
        """Two CFG streams: one call per stream (cond, then uncond, chained), each readiness covering that stream's
        raw and compressed ranges."""
        bridge, stage = self.raw_bridge(), OverlapStage()
        streams = ("cond", "uncond")
        self.placed(bridge, stage, {"0": "gpu_raw", "1": "compressed"}, streams=streams)
        plain, expected, summary, latents, caches = self.forks(bridge, stage, streams=streams)
        self.assertEqual(tiers_of(summary)[("uncond", 0)], "gpu_raw")
        record = summary["restore"]["overlap"]
        self.assertEqual((record["calls"], record["error"]), (2, None))
        self.assertEqual([producer["sources"] for producer in record["producer"]], [["restore", "raw"]] * 2)
        self.assertEqual([item["stream"] for item in record["waits"]], ["cond"] * LAYERS + ["uncond"] * LAYERS)
        raw = summary["restore"]["lease"]["overlap"]["raw_install"]
        self.assertEqual([(item["stream"], item["spans"]) for item in raw], [("cond", 1), ("uncond", 1)])
        self.assertBitwise(plain, expected, summary, latents, caches)
        self.assertLifetimesRetired(bridge, summary)

    def test_an_unprepared_history_overlaps_its_raw_ranges_and_runs_the_compressed_ones_plain(self):
        bridge, stage = self.raw_bridge(), OverlapStage()
        bridge.overlap_require_prepared = True  # the software restorer has no prepared packs: StateError
        self.placed(bridge, stage, {"0": "gpu_raw", "1": "compressed"})
        plain, expected, summary, latents, caches = self.forks(bridge, stage)
        record = summary["restore"]["overlap"]
        self.assertIn("prepared", record["fallback"])
        self.assertEqual((record["calls"], record["error"], record["layers_waited"]), (1, None, LAYERS))
        [producer] = record["producer"]
        self.assertEqual(producer["sources"], ["raw"])
        lease = summary["restore"]["lease"]
        self.assertEqual([item["stream"] for item in lease["restorer"]], [None])  # the plain compressed call
        self.assertEqual(lease["jobs"], plain["restore"]["lease"]["jobs"])
        self.assertBitwise(plain, expected, summary, latents, caches)
        self.assertLifetimesRetired(bridge, summary)

    def test_a_failing_raw_layer_fails_the_request_typed_and_releases_its_entries(self):
        """A raw layer that cannot be read (IntegrityError at layer 1): step 0 reads layer 0, its wait on layer 1
        raises the failure typed, the request fails, nothing stays pending, the restore's raw pins are released and
        the worker continues."""
        bridge, stage = self.raw_bridge(), OverlapStage()
        self.placed(bridge, stage, {"0": "gpu_raw", "1": "host_raw"})
        real = LineageSessionArchive._raw_matrix

        def failing(leases, span, stream, key):
            if key.layer == 1:
                raise IntegrityError("raw tier entry lacks a matrix (test)")
            return real(leases, span, stream, key)

        with mock.patch.object(LineageSessionArchive, "_raw_matrix", staticmethod(failing)), \
                self.assertRaises(IntegrityError):
            drive(bridge, request(handle("o", "fork", from_session="L", checkpoint=2, pause_after_chunks=3,
                                         overlap_restore=True)), stage)
        failed = bridge.request_summary(bridge.requests[-1]["request_key"])
        self.assertIn("IntegrityError", failed["error"])
        self.assertIn("IntegrityError", failed["restore"]["overlap"]["error"])
        self.assertTrue(failed["restore"]["lease"]["closed"])
        self.assertNothingPending(stage.caches)
        self.assertFalse(bridge.active_request)
        pins = [entry.pins.get("restore", 0) for entry in bridge.archive.raw_tier.entries()]
        self.assertEqual(sum(pins), 0)
        summary, _, _, _ = drive(bridge, request(handle("again", "fork", from_session="L", checkpoint=2,
                                                        pause_after_chunks=3, overlap_restore=True)),
                                 stage)
        self.assertEqual((summary["status"], summary["restore"]["overlap"]["error"]), ("paused", None))

    def test_an_engine_rope_with_out_installs_gpu_raw_in_place_bitwise(self):
        """F6 end to end: the stage's ``rotate_key`` takes ``out=`` (SGLang's ``causal_wan_rotate_key`` with Step 5c's
        patch). Plain and overlapped GPU-raw restores rotate straight into the caches, with no scratch tensor, bitwise
        the output-tensor path."""
        bridge, stage = self.raw_bridge(), OverlapStage()
        self.placed(bridge, stage, {"0": "gpu_raw", "1": "gpu_raw"})
        # F5 conventions (Step 11's adaptation, as above): the overlap is on by default and the handle's
        # overlap_restore is a bool, so the plain reference and the plain in-place restore turn it off
        plain, latents_p, caches_p, _ = drive(bridge, request(handle(
            "p", "fork", from_session="L", checkpoint=2, pause_after_chunks=3, overlap_restore=False)),
            stage)  # no out=: output tensors
        expected = self.snapshot(latents_p, caches_p)
        reference = plain["restore"]["lease"]["raw_install"]
        self.assertEqual(reference["in_place_ropes"], 0)
        self.assertGreater(reference["scratch_bytes"], 0)
        with mock.patch("tests.test_sglang_bridge_overlap.rotate_key", OutRotation()):
            for name, overlapped in (("q", False), ("o", True)):
                summary, latents, caches, _ = drive(bridge, request(handle(
                    name, "fork", from_session="L", checkpoint=2, pause_after_chunks=3,
                    overlap_restore=overlapped)), stage)
                lease = summary["restore"]["lease"]
                record = lease["overlap"]["raw_install"][0] if overlapped else lease["raw_install"]
                self.assertEqual((record["in_place_ropes"], record["rope_calls"], record["scratch_bytes"]),
                                 (2 * LAYERS, 2 * LAYERS, 0), name)
                self.assertTrue(torch.equal(latents, expected[0]))
                self.assertSameCaches(caches, expected[1])
                self.assertEqual(summary["chunk_digests"], plain["chunk_digests"])
        self.assertLifetimesRetired(bridge, summary)

    def test_the_raw_install_writes_inference_tensor_caches_in_the_callers_mode(self):
        bridge, stage = self.raw_bridge(), OverlapStage()
        with torch.inference_mode():
            self.placed(bridge, stage, {"0": "gpu_raw", "1": "host_raw"})
            plain, expected, summary, latents, caches = self.forks(bridge, stage)
        self.assertTrue(caches["cond"][0].k.is_inference())
        record = summary["restore"]["overlap"]
        self.assertEqual((record["calls"], record["error"], record["layers_waited"]), (1, None, LAYERS))
        self.assertBitwise(plain, expected, summary, latents, caches)



if __name__ == "__main__":
    unittest.main()
