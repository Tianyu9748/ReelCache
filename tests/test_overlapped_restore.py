"""Layer-overlapped restore (reelcache.overlapped_restore) on CPU: a real LineageRestorer over a software decoder pool.

Covered here:
- the overlapped restore of a prepared history writes exactly what the plain restore writes;
- earlier layers are released while later ones are still being restored;
- the last layer waits for the restore's integrity gate;
- a failure reaches every later wait and ``join``, typed, and no layer is released after it;
- unprepared histories are refused unless explicitly allowed;
- a restorer without layer order marks every layer at the end;
- layers issued out of order fail the restore;
- per-layer views, cancel / quiesce, and chained restores (cond before uncond) inheriting failures;
- the restore runs in the caller's inference / grad mode, writing inference-tensor caches as SGLang's;
- O4: several sources per readiness (a layer is released once every source marked it), the layer-by-layer RAW install
  (alone or beside the compressed restore: bounded layers in flight, failures and cancels reaching both sources,
  chaining, the caller's modes). The bridge's RAW install through these calls is covered by
  ``tests/test_sglang_raw_overlap.py``.

The CUDA path (events on a private stream, device-side waits, bitwise outputs while a forward reads the layers) is
validated by ``artifacts/sglang_integration/benchmarks/restore_e2e/scripts/overlap_bench.py``.
"""

from __future__ import annotations

import contextlib
import threading
import time
from types import SimpleNamespace
import unittest
from unittest import mock

import torch

from reelcache import restore_kernels as rk
from reelcache.archive_restore import LineageRestorer, SoftwareLineageRestorer
from reelcache.errors import ConfigurationError, IntegrityError, QuarantinedError, StateError
from reelcache.overlapped_restore import LayerReadiness, start_overlapped_install, start_overlapped_restore
from reelcache.types import MatrixKey
from tests import step6_support as s6
from tests.test_packed_nvdec import SoftwarePool
from tests.test_step7_cold_hooks import _commit_chain

LAYERS = s6.MODEL.layers


def destination_factory(chunks=3):
    caches = {}

    def destination(node_id, key, tile):
        key = key if isinstance(key, MatrixKey) else key[0]
        if key not in caches:
            caches[key] = torch.zeros((1, chunks * s6.TOKENS, 1, 128), dtype=torch.float32)
        return rk.NativeView(caches[key], "BTHD", token_offset=tile.token_start)

    return caches, destination


def preallocated(chunks=3):
    """Caches allocated up front by the caller in its own mode (SGLang's KV caches: inference tensors)."""
    caches = {MatrixKey(layer, target): torch.zeros((1, chunks * s6.TOKENS, 1, 128), dtype=torch.float32)
              for layer in range(LAYERS) for target in ("K", "V")}

    def destination(node_id, key, tile):
        key = key if isinstance(key, MatrixKey) else key[0]
        return rk.NativeView(caches[key], "BTHD", token_offset=tile.token_start)

    return caches, destination


def layer_of(entry):
    return (entry if isinstance(entry, MatrixKey) else entry[0]).layer


class LayerReadinessTests(unittest.TestCase):
    def test_wait_blocks_until_the_layer_is_marked(self):
        readiness = LayerReadiness(2, timeout=10)
        released = []
        waiter = threading.Thread(target=lambda: (readiness.wait(1), released.append(time.perf_counter())))
        waiter.start()
        time.sleep(0.1)
        self.assertEqual(released, [])
        marked = time.perf_counter()
        readiness.mark(1)
        waiter.join(10)
        self.assertEqual(len(released), 1)
        self.assertGreaterEqual(released[0], marked)
        self.assertTrue(readiness.ready(1))
        self.assertFalse(readiness.ready(0))
        self.assertGreater(readiness.waited_seconds[1], 0.05)

    def test_failures_are_typed_and_release_no_layer(self):
        for error, expected in ((IntegrityError("payload hash mismatch"), IntegrityError),
                                (QuarantinedError("quarantined"), QuarantinedError),
                                (RuntimeError("CUDA error: an illegal memory access"), StateError)):
            readiness = LayerReadiness(3, timeout=10)
            readiness.mark(0)  # completely written before the failure
            readiness.fail(error)
            readiness.mark(2)  # ignored after a failure
            for layer in range(3):
                with self.assertRaises(expected) as caught:
                    readiness.wait(layer)
                self.assertIs(caught.exception.__cause__, error)
                self.assertFalse(readiness.ready(layer))
            with self.assertRaises(expected):
                readiness.join()

    def test_a_layer_never_issued_times_out_as_a_state_error(self):
        readiness = LayerReadiness(1, timeout=0.05)
        with self.assertRaisesRegex(StateError, "not issued within"):
            readiness.wait(0)

    def test_a_layer_with_two_sources_is_released_once_both_marked_it(self):
        readiness = LayerReadiness(2, timeout=10, sources=("restore", "raw"))
        released = []
        waiter = threading.Thread(target=lambda: (readiness.wait(1), released.append(time.perf_counter())))
        waiter.start()
        readiness.mark(1, source="raw")
        time.sleep(0.1)
        self.assertEqual(released, [])  # one source is not enough
        self.assertTrue(readiness.is_marked(1, "raw"))
        self.assertFalse(readiness.is_marked(1, "restore"))
        self.assertFalse(readiness.is_marked(1))
        self.assertFalse(readiness.ready(1))
        readiness.mark(1, source="restore")
        waiter.join(10)
        self.assertEqual(len(released), 1)
        self.assertTrue(readiness.ready(1) and readiness.is_marked(1))
        self.assertEqual(readiness.marked_at[1], readiness.source_marked_at["restore"][1])
        timeline = readiness.timeline()
        self.assertEqual(set(timeline["sources"]), {"restore", "raw"})
        self.assertIsNone(timeline["marked_at"][0])
        for bad in ({"source": None}, {"source": "nvme"}):
            with self.assertRaises(ConfigurationError):
                readiness.mark(0, **bad)
        for sources in ((), ("raw", "raw"), ("raw", "")):
            with self.assertRaises(ConfigurationError):
                LayerReadiness(2, sources=sources)
        readiness.fail(IntegrityError("raw tier entry lacks a matrix"))
        readiness.mark(0, source="raw")  # ignored after a failure
        self.assertTrue(readiness.stopping)
        self.assertFalse(readiness.cancelled)
        with self.assertRaises(IntegrityError):
            readiness.wait(0)


class FakeRaw:
    """A layer-by-layer RAW install on CPU tensors (``DenseInstall.layer``'s contract): ``layer(L)`` writes layer
    ``L``'s cache and returns its pending work, released later by the producer. Records order, modes and the layers
    in flight; ``hold[L]`` blocks before layer ``L`` is issued, ``fail_at`` raises there."""

    def __init__(self, layers=LAYERS, *, fail_at=None, caches=None):
        self.device = torch.device("cpu")
        generator = torch.Generator().manual_seed(7)
        self.sources = [torch.randn(8, 4, generator=generator) for _ in range(layers)]
        self.caches = caches if caches is not None else [torch.zeros(8, 4) for _ in range(layers)]
        self.fail_at, self.hold, self.entered = fail_at, {}, {}
        self.issued, self.released, self.modes = [], [], []
        self.live = self.max_live = 0
        self.lock = threading.Lock()

    def expected(self, layer):
        return self.sources[layer] * 2  # stands for the engine's RoPE

    def layer(self, layer):
        if layer in self.hold:
            self.entered.setdefault(layer, threading.Event()).set()
            assert self.hold[layer].wait(30), "the test never released the held layer"
        if layer == self.fail_at:
            raise IntegrityError(f"raw tier entry lacks layer {layer} (test)")
        self.modes.append((torch.is_inference_mode_enabled(), torch.is_grad_enabled()))
        self.caches[layer].copy_(self.expected(layer))
        with self.lock:
            self.issued.append(layer)
            self.live += 1
            self.max_live = max(self.max_live, self.live)
        return SimpleNamespace(release=lambda layer=layer: self._release(layer))

    def _release(self, layer):
        with self.lock:
            self.released.append(layer)
            self.live -= 1

    def stats(self):
        return {"jobs": len(self.issued)}

    def wait_entered(self, layer, test):
        test.assertTrue(self.entered.setdefault(layer, threading.Event()).wait(30))

    def assert_written(self, test, layers):
        for layer in layers:
            test.assertTrue(torch.equal(self.caches[layer], self.expected(layer)), layer)


class OverlappedRestoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):  # one committed chain for the class (read-only below; every test builds its own restorer)
        cls.runtime, cls.heads = _commit_chain(SimpleNamespace(addCleanup=cls.addClassCleanup), "overlapped", count=3)
        cls.head = cls.heads[-1]
        cls.reference, destination = destination_factory()
        LineageRestorer(cls.runtime, SoftwarePool(), arena_bytes=1 << 30).restore(cls.head, destination)
        assert len(cls.reference) == 2 * LAYERS

    def prepared_restorer(self):
        restorer = LineageRestorer(self.runtime, SoftwarePool(), arena_bytes=1 << 30)
        self.addCleanup(restorer.close)
        restorer.prewarm(restorer.history(self.head))
        self.assertTrue(all(restorer.prepared.contains(item["packet"]) for item in restorer.history(self.head)))
        return restorer

    def assert_equal_to_plain(self, caches):
        self.assertEqual(caches.keys(), self.reference.keys())
        for key, value in self.reference.items():
            self.assertTrue(torch.equal(caches[key], value), key)

    def test_the_overlapped_restore_of_a_prepared_history_equals_the_plain_restore(self):
        caches, destination = destination_factory()
        readiness = start_overlapped_restore(self.prepared_restorer(), self.head, destination, layers=LAYERS)
        for layer in range(LAYERS):
            readiness.wait(layer)
        info = readiness.join()
        self.assert_equal_to_plain(caches)
        self.assertTrue(info["progressive"])
        self.assertEqual(info["entries"], info["bound_entries"])
        self.assertEqual(info["entries"], 3 * 2 * LAYERS)
        self.assertLessEqual(readiness.marked_at[0], readiness.marked_at[LAYERS - 1])

    def test_earlier_layers_are_released_while_later_ones_are_restored(self):
        caches, inner = destination_factory()
        held = threading.Event()
        holding = threading.Event()

        def destination(node_id, entry, tile):
            if layer_of(entry) == LAYERS - 1 and not held.is_set():
                holding.set()
                self.assertTrue(held.wait(30), "the test never released the last layer")
            return inner(node_id, entry, tile)

        readiness = start_overlapped_restore(self.prepared_restorer(), self.head, destination, layers=LAYERS)
        self.assertTrue(holding.wait(30))
        readiness.wait(0)  # layer 0 is complete and released while the last layer is still held
        self.assertTrue(readiness.ready(0))
        self.assertFalse(readiness.ready(LAYERS - 1))
        written = {key: value.clone() for key, value in caches.items() if key.layer == 0}
        for key, value in written.items():
            self.assertTrue(torch.equal(value, self.reference[key]), key)
        held.set()
        readiness.wait(LAYERS - 1)
        readiness.join()
        self.assert_equal_to_plain(caches)

    def test_the_last_layer_waits_for_the_restores_integrity_gate(self):
        restorer = self.prepared_restorer()
        gate, entered = threading.Event(), threading.Event()
        original = restorer.finish

        def finish(pins):
            entered.set()
            self.assertTrue(gate.wait(30), "the test never opened the gate")
            return original(pins)

        caches, destination = destination_factory()
        with mock.patch.object(restorer, "finish", finish):
            readiness = start_overlapped_restore(restorer, self.head, destination, layers=LAYERS)
            self.assertTrue(entered.wait(30))
            readiness.wait(0)
            self.assertFalse(readiness.ready(LAYERS - 1))  # every write issued, but the gate has not passed
            gate.set()
            readiness.wait(LAYERS - 1)
            readiness.join()
        self.assert_equal_to_plain(caches)

    def test_a_failure_reaches_every_later_wait_typed(self):
        caches, inner = destination_factory()
        damage = IntegrityError("payload hash mismatch: blob:feed")

        def destination(node_id, entry, tile):
            if layer_of(entry) == LAYERS - 1:
                raise damage
            return inner(node_id, entry, tile)

        readiness = start_overlapped_restore(self.prepared_restorer(), self.head, destination, layers=LAYERS)
        with self.assertRaises(IntegrityError) as caught:
            readiness.join()
        self.assertIs(caught.exception.__cause__, damage)
        for layer in range(LAYERS):  # layer 0 was marked, but nothing is released after the failure
            with self.assertRaises(IntegrityError):
                readiness.wait(layer)

    def test_unprepared_histories_are_refused_unless_allowed(self):
        restorer = LineageRestorer(self.runtime, SoftwarePool(), arena_bytes=1 << 30)
        self.addCleanup(restorer.close)
        caches, destination = destination_factory()
        with self.assertRaisesRegex(StateError, "not prepared"):
            start_overlapped_restore(restorer, self.head, destination, layers=LAYERS)
        readiness = start_overlapped_restore(restorer, self.head, destination, layers=LAYERS, require_prepared=False)
        readiness.join()
        self.assert_equal_to_plain(caches)

    def test_a_restorer_without_layer_order_marks_every_layer_at_the_end(self):
        restorer = SoftwareLineageRestorer(self.runtime)
        caches, destination = destination_factory()
        with self.assertRaisesRegex(StateError, "prepared-pack cache"):
            start_overlapped_restore(restorer, self.head, destination, layers=LAYERS)
        readiness = start_overlapped_restore(restorer, self.head, destination, layers=LAYERS, require_prepared=False)
        info = readiness.join()
        self.assertFalse(info["progressive"])
        plain, reference = destination_factory()
        SoftwareLineageRestorer(self.runtime).restore(self.head, reference)
        for key, value in plain.items():
            self.assertTrue(torch.equal(caches[key], value), key)

    def test_per_layer_views_wait_like_the_readiness(self):
        caches, destination = destination_factory()
        readiness = start_overlapped_restore(self.prepared_restorer(), self.head, destination, layers=LAYERS)
        views = [readiness.layer(layer) for layer in range(LAYERS)]
        for view in views:
            view.wait()
            self.assertTrue(view.ready())
        readiness.join()
        self.assert_equal_to_plain(caches)
        with self.assertRaises(ConfigurationError):
            readiness.layer(LAYERS)

    def test_cancel_stops_the_restore_and_quiesce_leaves_nothing_in_flight(self):
        caches, inner = destination_factory()
        held, holding = threading.Event(), threading.Event()

        def destination(node_id, entry, tile):
            if layer_of(entry) == LAYERS - 1 and not held.is_set():
                holding.set()
                self.assertTrue(held.wait(30), "the test never released the last layer")
            return inner(node_id, entry, tile)

        readiness = start_overlapped_restore(self.prepared_restorer(), self.head, destination, layers=LAYERS)
        self.assertTrue(holding.wait(30))
        readiness.wait(0)  # released before the cancel
        readiness.cancel()
        held.set()  # the held bind completes; the next one sees the cancel
        self.assertFalse(readiness.quiesce())
        self.assertTrue(readiness.done)
        for layer in range(LAYERS):
            with self.assertRaisesRegex(StateError, "cancelled"):
                readiness.wait(layer)
        with self.assertRaisesRegex(StateError, "cancelled"):
            readiness.join()

    def test_quiesce_after_a_completed_restore(self):
        caches, destination = destination_factory()
        readiness = start_overlapped_restore(self.prepared_restorer(), self.head, destination, layers=LAYERS)
        readiness.join()
        self.assertTrue(readiness.quiesce())
        self.assertFalse(readiness.cancelled)
        self.assert_equal_to_plain(caches)

    def test_chained_restores_run_in_order_and_inherit_failures(self):
        restorer = self.prepared_restorer()
        order, gate = [], threading.Event()
        first_caches, first_inner = destination_factory()
        second_caches, second_inner = destination_factory()

        def first(node_id, entry, tile):
            order.append("cond")
            self.assertTrue(gate.wait(30), "the test never released the first restore")
            return first_inner(node_id, entry, tile)

        def second(node_id, entry, tile):
            order.append("uncond")
            return second_inner(node_id, entry, tile)

        cond = start_overlapped_restore(restorer, self.head, first, layers=LAYERS)
        uncond = start_overlapped_restore(restorer, self.head, second, layers=LAYERS, after=cond)
        time.sleep(0.2)
        self.assertNotIn("uncond", order)  # the chained restore waits for the first one
        gate.set()
        cond.join()
        uncond.join()
        self.assertEqual(order.index("uncond"), order.count("cond"))  # every cond bind precedes every uncond bind
        self.assert_equal_to_plain(first_caches)
        self.assert_equal_to_plain(second_caches)
        damage = IntegrityError("payload hash mismatch: blob:feed")

        def failing(node_id, entry, tile):
            raise damage

        failed = start_overlapped_restore(restorer, self.head, failing, layers=LAYERS)
        skipped_caches, skipped = destination_factory()
        chained = start_overlapped_restore(restorer, self.head, skipped, layers=LAYERS, after=failed)
        for readiness in (failed, chained):
            with self.assertRaises(IntegrityError):
                readiness.join()
        self.assertEqual(skipped_caches, {})  # the chained restore never ran
        with self.assertRaises(ConfigurationError):
            start_overlapped_restore(restorer, self.head, skipped, layers=LAYERS, after=object())

    def test_the_restore_runs_in_the_callers_inference_and_grad_modes(self):
        restorer = self.prepared_restorer()
        contexts = {"inference": torch.inference_mode, "no_grad": torch.no_grad, "default": contextlib.nullcontext}
        expected = {"inference": (True, False), "no_grad": (False, False), "default": (False, True)}
        for mode, context in contexts.items():
            seen = []
            with context():
                caches, inner = preallocated()

                def destination(node_id, entry, tile, inner=inner, seen=seen):
                    seen.append((torch.is_inference_mode_enabled(), torch.is_grad_enabled()))
                    return inner(node_id, entry, tile)

                readiness = start_overlapped_restore(restorer, self.head, destination, layers=LAYERS)
            readiness.join()
            self.assertEqual(set(seen), {expected[mode]}, mode)
            self.assertEqual(all(tensor.is_inference() for tensor in caches.values()), mode == "inference", mode)
            self.assert_equal_to_plain(caches)

    def test_inference_tensor_caches_are_written_only_from_inference_mode(self):
        # As for the synchronous restore: the caller's mode is mirrored, not forced.
        restorer = self.prepared_restorer()
        with torch.inference_mode():
            caches, destination = preallocated()
        readiness = start_overlapped_restore(restorer, self.head, destination, layers=LAYERS)
        with self.assertRaises(StateError) as caught:
            readiness.join()
        self.assertIn("inference", str(caught.exception.__cause__).lower())

    def test_layers_issued_out_of_order_fail_the_restore(self):
        class Backwards:
            device = torch.device("cpu")

            def restore(self, node_ref, destination, rope=None, **_):
                for layer in (1, 0):
                    destination("n", MatrixKey(layer, "K"), None)
                return {"entries": 2}

        readiness = start_overlapped_restore(Backwards(), "n", lambda *_: None, layers=2, require_prepared=False,
                                             progressive=True)
        with self.assertRaisesRegex(StateError, "out of order"):
            readiness.join()

    # -- O4: the layer-by-layer RAW install, alone and beside the compressed restore ---------------------------------
    def test_a_raw_install_alone_marks_layers_in_order_with_bounded_layers_in_flight(self):
        for depth, live in ((0, 1), (1, 2), (3, 4)):
            raw = FakeRaw(layers=5)
            readiness = start_overlapped_install(raw.layer, layers=5, raw_depth=depth)
            for layer in range(5):
                readiness.wait(layer)
            info = readiness.join()
            self.assertEqual(raw.issued, list(range(5)))
            self.assertEqual(raw.released, list(range(5)))  # oldest first, all before join returned
            self.assertEqual(raw.max_live, live, depth)
            raw.assert_written(self, range(5))
            self.assertEqual(readiness.sources, ("raw",))
            self.assertEqual(info["sources"], ["raw"])
            self.assertEqual((info["raw_install"]["layers"], info["raw_install"]["depth"]), (5, depth))
            self.assertEqual(info["raw_install"]["jobs"], 5)  # the installer's own stats
            marks = readiness.marked_at
            self.assertEqual(marks, sorted(marks))
            self.assertLessEqual(info["first_layer_ready_seconds_from_call"],
                                 info["last_layer_ready_seconds_from_call"])
            self.assertTrue(readiness.done)

    def test_a_mixed_restore_releases_each_layer_once_both_sources_wrote_it(self):
        raw = FakeRaw()
        raw.hold[1] = threading.Event()
        caches, destination = destination_factory()
        readiness = start_overlapped_restore(self.prepared_restorer(), self.head, destination, layers=LAYERS,
                                             raw=raw.layer)
        self.assertEqual(readiness.sources, ("restore", "raw"))
        readiness.wait(0)  # both sources issued layer 0
        raw.wait_entered(1, self)
        deadline = time.perf_counter() + 30
        while not readiness.is_marked(1, "restore") and time.perf_counter() < deadline:
            time.sleep(0.01)
        self.assertTrue(readiness.is_marked(1, "restore"))  # the compressed restore finished every layer
        self.assertFalse(readiness.ready(1))  # but the RAW source has not issued layer 1
        raw.hold[1].set()
        readiness.wait(1)
        info = readiness.join()
        self.assert_equal_to_plain(caches)
        raw.assert_written(self, range(LAYERS))
        self.assertEqual(info["entries"], 3 * 2 * LAYERS)  # the restorer's report, as for a compressed-only restore
        self.assertEqual(info["raw_install"]["layers"], LAYERS)
        self.assertEqual(info["sources"], ["restore", "raw"])
        timeline = readiness.timeline()
        for layer in range(LAYERS):
            self.assertEqual(timeline["marked_at"][layer], max(timeline["sources"]["restore"][layer],
                                                               timeline["sources"]["raw"][layer]))

    def test_a_raw_failure_stops_the_compressed_source_and_reaches_every_wait_typed(self):
        _, inner = destination_factory()
        held, holding = threading.Event(), threading.Event()

        def destination(node_id, entry, tile):
            if layer_of(entry) == LAYERS - 1 and not held.is_set():
                holding.set()
                self.assertTrue(held.wait(30), "the test never released the last layer")
            return inner(node_id, entry, tile)

        raw = FakeRaw(fail_at=1)
        raw.hold[1] = threading.Event()  # fails once the compressed source is inside its last layer
        readiness = start_overlapped_restore(self.prepared_restorer(), self.head, destination, layers=LAYERS,
                                             raw=raw.layer)
        self.assertTrue(holding.wait(30))
        raw.hold[1].set()
        deadline = time.perf_counter() + 30
        while readiness.error is None and time.perf_counter() < deadline:
            time.sleep(0.01)
        self.assertIsInstance(readiness.error, IntegrityError)
        self.assertTrue(readiness.stopping)
        held.set()  # the compressed source sees the stop at its next bind
        with self.assertRaises(IntegrityError) as caught:
            readiness.join()
        self.assertIs(caught.exception.__cause__, readiness.error)
        for layer in range(LAYERS):
            with self.assertRaises(IntegrityError):
                readiness.wait(layer)
        self.assertEqual(raw.released, raw.issued)  # its issued layers were released (waited for, freed)
        self.assertTrue(readiness.done)
        self.assertFalse(readiness.quiesce())

    def test_cancel_stops_a_raw_install_at_its_next_layer(self):
        raw = FakeRaw(layers=4)
        raw.hold[2] = threading.Event()
        readiness = start_overlapped_install(raw.layer, layers=4)
        raw.wait_entered(2, self)
        readiness.wait(1)  # released before the cancel
        readiness.cancel()
        raw.hold[2].set()
        self.assertFalse(readiness.quiesce())
        self.assertTrue(readiness.done and readiness.cancelled)
        self.assertNotIn(3, raw.issued)
        self.assertEqual(raw.released, raw.issued)
        for layer in range(4):
            with self.assertRaisesRegex(StateError, "cancelled"):
                readiness.wait(layer)

    def test_the_raw_install_runs_in_the_callers_modes(self):
        for mode, context, expected in (("inference", torch.inference_mode, (True, False)),
                                        ("no_grad", torch.no_grad, (False, False)),
                                        ("default", contextlib.nullcontext, (False, True))):
            with context():
                raw = FakeRaw(caches=[torch.zeros(8, 4) for _ in range(LAYERS)])
                readiness = start_overlapped_install(raw.layer, layers=LAYERS)
            readiness.join()
            self.assertEqual(set(raw.modes), {expected}, mode)
            self.assertEqual(all(cache.is_inference() for cache in raw.caches), mode == "inference", mode)
            raw.assert_written(self, range(LAYERS))
        with torch.inference_mode():
            caches = [torch.zeros(8, 4) for _ in range(LAYERS)]
        readiness = start_overlapped_install(FakeRaw(caches=caches).layer, layers=LAYERS)
        with self.assertRaises(StateError) as caught:  # inference caches, normal-mode caller: as a synchronous write
            readiness.join()
        self.assertIn("inference", str(caught.exception.__cause__).lower())

    def test_a_chained_restore_waits_for_every_source_of_the_previous_one(self):
        restorer = self.prepared_restorer()
        first_raw, second_raw = FakeRaw(), FakeRaw()
        first_raw.hold[LAYERS - 1] = threading.Event()
        caches, destination = destination_factory()
        cond = start_overlapped_restore(restorer, self.head, destination, layers=LAYERS, raw=first_raw.layer)
        uncond = start_overlapped_install(second_raw.layer, layers=LAYERS, after=cond)
        first_raw.wait_entered(LAYERS - 1, self)
        time.sleep(0.2)
        self.assertEqual(second_raw.issued, [])  # the cond restore's RAW source is still running
        first_raw.hold[LAYERS - 1].set()
        cond.join()
        uncond.join()
        self.assertEqual(second_raw.issued, list(range(LAYERS)))
        self.assert_equal_to_plain(caches)
        failed = start_overlapped_install(FakeRaw(fail_at=0).layer, layers=LAYERS)
        skipped = FakeRaw()
        chained = start_overlapped_restore(restorer, self.head, destination_factory()[1], layers=LAYERS,
                                           raw=skipped.layer, after=failed)
        for readiness in (failed, chained):
            with self.assertRaises(IntegrityError):
                readiness.join()
        self.assertEqual(skipped.issued, [])

    def test_raw_install_arguments_are_checked(self):
        raw = FakeRaw()
        for kwargs in ({}, {"raw": "not callable"}, {"raw": raw.layer, "raw_depth": -1}):
            with self.assertRaises(ConfigurationError):
                start_overlapped_restore(None, layers=LAYERS, **kwargs)
        with self.assertRaises(ConfigurationError):  # a compressed source needs its node and destination
            start_overlapped_restore(self.prepared_restorer(), layers=LAYERS, raw=raw.layer)
        self.assertEqual(raw.issued, [])


if __name__ == "__main__":
    unittest.main()
