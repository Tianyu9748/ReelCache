"""Shared Step 6 test fixtures: a tiny model, synthetic clean chunks and runtime factories."""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path

import torch

from reelcache.manager import KVRuntime
from reelcache.predictor_registry import RankPredictorRegistry
from reelcache.rank_method import RankMethodConfig
from reelcache.types import CodecSpec, EnergyTargets, MatrixKey, ModelIdentity, TensorContext

MODEL = ModelIdentity("step6-tiny", "step6-tiny-source", 2, 128, "tokens_channels", "bfloat16")
CHANNELS = 128
# Per-key drift between chunks: chunk 1 shares, augments and refits (fresh in chunk 0).
DRIFT = (0.05, 0.1, 0.15, 0.25)
FRAMES, HEIGHT, WIDTH = 3, 6, 8
TOKENS = FRAMES * HEIGHT * WIDTH
KEYS = tuple(MatrixKey(layer, target) for layer in range(MODEL.layers) for target in ("K", "V"))
ROOT = Path(__file__).resolve().parents[1] / ".test-work"


def method(rate_lag=0, solver=None):
    """The test method; ``solver=None`` is the runtime default (top-k Rayleigh-Ritz since Step 7 Part C)."""
    extra = {} if solver is None else {"solver": solver}
    return RankMethodConfig(EnergyTargets("step6-test", 0.95, 0.9, True),
                            CodecSpec(kind="svd_hevc", rank=1, inter_prediction=False), rate_lag=rate_lag, **extra)


def exact_method(rate_lag=0):
    """The explicit exact solver (full Gram eigendecomposition; the default before Step 7 Part C)."""
    return method(rate_lag, solver="gram_eigh")


def _bases():
    generator = torch.Generator().manual_seed(7)
    return {key: torch.linalg.qr(torch.randn(CHANNELS, 64, generator=generator)).Q.T for key in KEYS}


BASES = _bases()


def chunk(index, *, streams=(), token_start=None, scale=1.0):
    """One complete clean chunk; drifting content makes later chunks share, augment or refit."""
    generator = torch.Generator().manual_seed(1000 + index)
    start = index * TOKENS if token_start is None else token_start
    matrices, contexts = {}, {}
    for stream in (streams or (None,)):
        for position, key in enumerate(KEYS):
            value = (torch.randn(TOKENS, 64, generator=generator) * torch.linspace(2, 0.02, 64)) @ BASES[key]
            value = value + 0.01 * torch.randn(TOKENS, CHANNELS, generator=generator)
            if index:
                value = value + DRIFT[position] * torch.randn(TOKENS, 6, generator=generator) @ torch.randn(
                    6, CHANNELS, generator=generator)
            if key.target == "K":
                value = torch.nn.functional.layer_norm(value, (CHANNELS,))
            entry = key if stream is None else (key, stream)
            matrices[entry] = (scale * value).to(torch.bfloat16)
            contexts[entry] = TensorContext(MODEL, key, start, FRAMES, HEIGHT, WIDTH,
                                            "pre_rope" if key.target == "K" else "none")
    return matrices, contexts


class RuntimeFactory:
    """Creates runtimes in .test-work and removes them at cleanup."""

    def __init__(self, test):
        self.test = test
        self.roots = []

    def __call__(self, name="store"):
        root = ROOT / f"step6-{name}-{uuid.uuid4().hex}"
        self.roots.append(root)
        runtime = KVRuntime(root, MODEL)
        self.test.addCleanup(self._close, runtime, root)
        return runtime

    @staticmethod
    def _close(runtime, root):
        try:
            if not runtime._closed:
                runtime.close()
        finally:
            shutil.rmtree(root, ignore_errors=True)


def registry(name="step6"):
    return RankPredictorRegistry(name)


def node_content(runtime, node_ref):
    """Everything a commit stores except timing-bearing manifests (packets carry encode metrics)."""
    node = runtime.load_node(node_ref)
    rows = {}
    for entry, record_ref in node.matrix_records:
        record = runtime.load_record(record_ref)
        packet = runtime.load_packet(record.packet_ref)
        rows[str(entry)] = (record.decision["decision"], record.rank_view, record.basis_version_ref,
                            record.packet_tile, record.energy.to_dict(), sorted(packet["payloads"].values()),
                            packet["metadata"], packet["tile_bases"], record.stream)
    bindings = {str(entry): binding.to_dict() for entry, binding in node.active_basis}
    return {"kind": node.node_kind, "chunk": node.chunk_id, "frontier": node.frontier, "streams": node.streams,
            "rows": rows, "bindings": bindings}
