#!/usr/bin/env python3
"""accel track L (2026-10-30): the overlapped LingBot restore's harness plumbing on the CPU.

1. ``reelcache_lingbot.lb_overlap.OverlapSession``: the forward pre-hooks make each block's self-attention wait for its layer (it
   never reads a layer before the producer wrote and marked it), the hooks go after the last layer (later forwards do
   not wait), channel tiles wait for every tile, a restore failure reaches the forward typed and the session quiesces.
2. ``reelcache_lingbot.lb_archive.ReelCacheArchive`` on the real tree (software decoder, libx265): the overlapped restore (attached to a
   model stand-in, joined after its "forward") writes bitwise the rows of the synchronous restore, so do the grouped
   restore (the evaluation's g = 2 grouping) and ``release_behind``; the harness keys never reach the ReelCache config; overlap with
   ``pipelined_restore`` is refused.

The CUDA path (side stream, per-layer events, device-side waits during LingBot's first forward, bitwise latents) is
measured on GPU outside this repository. Run from the repository root: ``python integrations/lingbot/tests/test_lb_overlap.py``.
"""
from __future__ import annotations

import os
import shutil
import sys
import threading
import time
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent  # integrations/lingbot: the reelcache_lingbot package
REPO = HERE.parents[2]  # the ReelCache root: the reelcache package
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(ROOT))
WORK = Path(os.environ.get("LBQ_TEST_WORK", str(HERE / "_work" / "test_lb_overlap")))


def _ffmpeg() -> str:
    """The FFmpeg the archive will run (reelcache's own discovery: REELCACHE_FFMPEG_BIN, PATH, imageio-ffmpeg); ""
    when there is none."""
    from reelcache._hevc import discover_ffmpeg

    try:
        return discover_ffmpeg()
    except Exception:  # noqa: BLE001 - no FFmpeg: the real-archive tests are skipped
        return ""


FFMPEG = _ffmpeg()

import torch  # noqa: E402

from reelcache.errors import IntegrityError  # noqa: E402
from reelcache.overlapped_restore import LayerReadiness  # noqa: E402
from reelcache_lingbot.lb_archive import ArchiveError, ReelCacheArchive, split_extra  # noqa: E402
from reelcache_lingbot.lb_overlap import OverlapSession, StallEvents  # noqa: E402
from tests.assets import asset  # noqa: E402

LAYERS, HEAD_DIM = 3, 128


class Attn(torch.nn.Module):
    def __init__(self, net, layer):
        super().__init__()
        self.net, self.layer = net, layer

    def forward(self, x):
        self.net.seen.append((self.layer, float(self.net.cache[self.layer].sum())))
        return x + self.net.cache[self.layer].sum()


class Block(torch.nn.Module):
    def __init__(self, net, layer):
        super().__init__()
        self.self_attn = Attn(net, layer)

    def forward(self, x):
        return self.self_attn(x)


class Net(torch.nn.Module):
    """A stand-in DiT: block l's self-attention reads cache[l]."""

    def __init__(self, layers):
        super().__init__()
        self.cache = [torch.zeros(4) for _ in range(layers)]
        self.seen: list = []
        self.blocks = torch.nn.ModuleList(Block(self, layer) for layer in range(layers))

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


def producer(net, readiness, *, tiles=1, delay=0.03, fail_at=None):
    def run():
        try:
            for layer in range(len(net.cache)):
                time.sleep(delay)
                if layer == fail_at:
                    raise IntegrityError(f"payload of layer {layer} failed verification")
                net.cache[layer].fill_(layer + 1.0)
                for tile in range(tiles):
                    readiness.mark(layer * tiles + tile)
        except BaseException as exc:  # noqa: BLE001 - delivered to every waiter
            readiness.fail(exc)
        finally:
            readiness._finish("restore")
    thread = threading.Thread(target=run, daemon=True)
    readiness.attach(thread)
    readiness.info.update(nodes=1, entries=len(net.cache))
    thread.start()
    return thread


class Session(unittest.TestCase):
    def test_each_layer_is_read_after_its_mark_and_later_forwards_do_not_wait(self):
        for tiles in (1, 2):
            with self.subTest(tiles=tiles):
                net = Net(LAYERS)
                readiness = LayerReadiness(LAYERS * tiles, "cpu", timeout=30)
                readiness.stall_recorder = recorder = StallEvents(torch, torch.device("cpu"))  # as telemetry=
                record = {"overlapped": True}
                session = OverlapSession(readiness, layers=LAYERS, tiles=tiles, record=record, recorder=recorder)
                session.attach([block.self_attn for block in net.blocks])
                producer(net, readiness, tiles=tiles)
                net(torch.zeros(1))
                self.assertEqual(net.seen, [(layer, 4.0 * (layer + 1)) for layer in range(LAYERS)])
                self.assertEqual(session.hooks, [])  # removed after the last layer's wait
                self.assertEqual(session.waited_layers, LAYERS)
                net(torch.zeros(1))  # a later forward: no wait, no hook
                self.assertEqual(session.waited_layers, LAYERS)
                out = session.join()
                self.assertIs(out, record)
                self.assertEqual((out["nodes"], out["entries"], out["overlap"]["layers_waited"]), (1, LAYERS, LAYERS))
                self.assertGreater(out["overlap"]["host_wait_seconds"], 0.0)
                self.assertEqual(session.gpu_summary()["layers_waited"], LAYERS * tiles)

    def test_a_restore_failure_reaches_the_forward_typed_and_the_session_quiesces(self):
        net = Net(LAYERS)
        readiness = LayerReadiness(LAYERS, "cpu", timeout=30)
        session = OverlapSession(readiness, layers=LAYERS, record={})
        session.attach([block.self_attn for block in net.blocks])
        producer(net, readiness, fail_at=1)
        with self.assertRaises(IntegrityError):
            net(torch.zeros(1))
        self.assertEqual([layer for layer, _ in net.seen], [0])  # layer 1 never read
        self.assertFalse(session.abort())
        self.assertEqual(session.hooks, [])
        self.assertTrue(session.record["aborted"])
        self.assertEqual(session.join(), session.record)  # idempotent after the abort

    def test_attach_checks_the_module_count(self):
        net = Net(LAYERS)
        session = OverlapSession(LayerReadiness(LAYERS, "cpu"), layers=LAYERS + 1)
        with self.assertRaises(ValueError):
            session.attach([block.self_attn for block in net.blocks])


class Extra(unittest.TestCase):
    def test_harness_keys_are_split_from_the_reelcache_overrides(self):
        options, rest = split_extra({"lb_overlap_restore": True, "verify_reads": False})
        self.assertEqual(options, {"lb_overlap_restore": True, "lb_overlap_release_behind": 2,
                                   "lb_restore_group_chunks": 0, "lb_native_rope": "model",
                                   "lb_overlap_stream_priority": None})
        self.assertEqual(rest, {"verify_reads": False})
        self.assertEqual(split_extra(None)[0]["lb_overlap_restore"], False)
        for bad in ({"lb_overlap_restore": 1}, {"lb_overlap_release_behind": -1},
                    {"lb_restore_group_chunks": 1.0}, {"lb_native_rope": "fused"},
                    {"lb_overlap_stream_priority": "high"}):
            with self.subTest(bad=bad), self.assertRaises(ArchiveError):
                split_extra(bad)


def lingbot_functions():
    """causal_rope_apply and rope_params, verbatim from the (read-only) LingBot source: the wan package itself needs
    a GPU at import (t5.py)."""
    import ast

    base = asset("lingbot", "repo", override="LB_REPO") / "wan" / "modules"  # the official repo @ a43bec7
    ns = {"torch": torch}
    for name, file in (("causal_rope_apply", "model_fast.py"), ("rope_params", "model.py")):
        src = (base / file).read_text()
        node = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == name)
        exec(ast.get_source_segment(src, node), ns)  # noqa: S102 - the pinned model source
    return ns["causal_rope_apply"], ns["rope_params"]


class LeanRope(unittest.TestCase):
    def test_rope_lean_is_bitwise_the_models_rope(self):
        from reelcache_lingbot.lb_model import LingBotModel

        causal_rope_apply, rope_params = lingbot_functions()
        model = LingBotModel.__new__(LingBotModel)
        d = 128
        freqs = torch.cat([rope_params(1024, d - 4 * (d // 6)), rope_params(1024, 2 * (d // 6)),
                           rope_params(1024, 2 * (d // 6))], dim=1)
        model.__dict__.update(resolution="480x832", window_latents=-1, sink_latents=0, world=1,
                              device=torch.device("cpu"), dtype=torch.bfloat16, _rope=causal_rope_apply,
                              model=type("Dit", (), {"freqs": freqs})())
        n = model.geometry()["chunk_tokens"]
        g = torch.Generator().manual_seed(3)
        torch.cuda.current_stream = lambda device=None: type("S", (), {"synchronize": lambda self: None})()
        try:
            for chunk, heads in ((0, 40), (4, 40), (7, 20)):  # 20 heads: a channel tile of ablation (b)
                with self.subTest(chunk=chunk, heads=heads):
                    k = (torch.randn(n, heads * d, generator=g) * 4).to(torch.bfloat16)
                    ref, lean = model.rope(k, chunk), model.rope_lean(k, chunk)
                    self.assertEqual((ref.shape, ref.dtype), (lean.shape, lean.dtype))
                    self.assertTrue(torch.equal(ref.view(torch.int16), lean.view(torch.int16)))
        finally:
            del torch.cuda.current_stream


def toy_chain(tokens: int, channels: int, chunks: int = 3, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    basis = torch.linalg.qr(torch.randn(channels, 48, generator=g)).Q.T
    out = []
    for _ in range(chunks):
        ks, vs = [], []
        for _ in range(LAYERS):
            ks.append(((torch.randn(tokens, 48, generator=g) * torch.linspace(2, .05, 48)) @ basis
                       + 0.05 * torch.randn(tokens, channels, generator=g)).to(torch.bfloat16))
            vs.append(((torch.randn(tokens, 48, generator=g) * torch.linspace(2, .05, 48)) @ basis
                       + 0.05 * torch.randn(tokens, channels, generator=g)).to(torch.bfloat16))
        out.append((ks, vs))
    return out


class StandIn:
    """The model side of an overlapped restore: its caches, ``destination_tensors`` and the session hand-over."""

    def __init__(self, layers, rows, heads):
        self.kc = [torch.zeros(1, rows, heads, HEAD_DIM, dtype=torch.bfloat16) for _ in range(layers)]
        self.vc = [torch.zeros(1, rows, heads, HEAD_DIM, dtype=torch.bfloat16) for _ in range(layers)]
        self.session = None

    def destination_tensors(self, layer):
        return self.kc[layer], self.vc[layer]

    def attach_overlapped_restore(self, session):
        self.session = session


@unittest.skipUnless(FFMPEG, "needs an FFmpeg with libx265 (REELCACHE_FFMPEG_BIN, PATH or imageio-ffmpeg)")
class RealArchive(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.grid, cls.channels = (3, 12, 16), 256  # 576 tokens per chunk
        cls.tokens = cls.grid[0] * cls.grid[1] * cls.grid[2]
        cls.root = WORK / "archive"
        shutil.rmtree(cls.root, ignore_errors=True)
        cls.arch = ReelCacheArchive(str(cls.root), arm="reelcache", targets=(0.98, 0.95), provisional_targets=True,
                                 device="cpu", layers=LAYERS, channels=cls.channels, token_grid=cls.grid,
                                 prepared_budget_bytes=1 << 26,
                                 rope=lambda values, ci: values.reshape(values.shape[0], -1, HEAD_DIM),
                                 extra={"lb_overlap_restore": False, "lb_overlap_release_behind": 1})
        cls.keys = [("s", c + 1) for c in range(3)]
        for c, (ks, vs) in enumerate(toy_chain(cls.tokens, cls.channels)):
            cls.arch.commit("s", cls.keys[c], cls.keys[c - 1] if c else None, c, ks, vs, torch.zeros(16, 3, 2, 2))
        cls.arch.drain()
        cls.reference = cls.restore_into(lambda model, jobs: cls.arch.restore(
            cls.keys[-1], jobs, model.destination_tensors, torch.bfloat16))

    @classmethod
    def tearDownClass(cls):
        cls.arch.close()
        shutil.rmtree(cls.root, ignore_errors=True)

    @classmethod
    def restore_into(cls, call):
        model = StandIn(LAYERS, 3 * cls.tokens, cls.channels // HEAD_DIM)
        jobs = [(c, c * cls.tokens, cls.keys[c]) for c in range(3)]
        model.record = call(model, jobs)
        return model

    def assertSameRows(self, model):
        for layer in range(LAYERS):
            self.assertTrue(torch.equal(model.kc[layer], self.reference.kc[layer]), layer)
            self.assertTrue(torch.equal(model.vc[layer], self.reference.vc[layer]), layer)
        self.assertGreater(float(model.vc[0].float().abs().sum()), 0.0)

    def test_the_harness_keys_stay_out_of_the_reelcache_config(self):
        self.assertFalse(any(key.startswith("lb_") for key in self.arch.config_dict))
        self.assertEqual(self.arch.describe()["harness_restore"]["lb_overlap_release_behind"], 1)

    def test_the_overlapped_restore_writes_the_synchronous_rows(self):
        model = self.restore_into(lambda model, jobs: self.arch.restore(
            self.keys[-1], jobs, model.destination_tensors, torch.bfloat16, overlap=True))
        record, session = model.record, model.session
        self.assertTrue(record["overlapped"])
        self.assertIsNone(record["seconds"])  # filled when the model joins
        self.assertEqual(record["release_behind_requested"], 1)
        for layer in range(LAYERS):  # the model's first forward
            session.wait(layer)
        self.assertIs(session.join(), record)
        self.assertSameRows(model)
        self.assertIsNotNone(record["seconds"])
        self.assertEqual(record["overlap"]["layers_waited"], LAYERS)
        self.assertEqual(record["nodes"], 3)

    def test_grouped_and_release_behind_restores_write_the_same_rows(self):
        for kwargs in ({"group_chunks": 1}, {"group_chunks": 2}, {"release_behind": 0},
                       {"group_chunks": 2, "release_behind": 1}, {"overlap": True, "stream_priority": -1}):
            with self.subTest(**kwargs):
                model = self.restore_into(lambda model, jobs: self.arch.restore(
                    self.keys[-1], jobs, model.destination_tensors, torch.bfloat16, **kwargs))
                if kwargs.get("overlap"):  # CPU: no CUDA stream (the priority applies on CUDA only)
                    for layer in range(LAYERS):
                        model.session.wait(layer)
                    model.session.join()
                    self.assertIsNone(model.record["stream_priority"])
                self.assertSameRows(model)
                if kwargs.get("group_chunks"):
                    self.assertEqual(model.record["group_chunks"], kwargs["group_chunks"])

    def test_overlap_needs_a_model_and_layer_ordered_restores(self):
        with self.assertRaises(ArchiveError):
            self.arch.restore(self.keys[-1], [(0, 0, self.keys[0])], lambda layer: (None, None), torch.bfloat16,
                              overlap=True)
        with self.assertRaises(ArchiveError):
            ReelCacheArchive(str(WORK / "refused"), arm="reelcache", targets=(0.98, 0.95), provisional_targets=True,
                          device="cpu", layers=LAYERS, channels=self.channels, token_grid=self.grid,
                          prepared_budget_bytes=1 << 26, rope=lambda values, ci: values,
                          extra={"lb_overlap_restore": True, "pipelined_restore": True})
        shutil.rmtree(WORK / "refused", ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
