from __future__ import annotations

from pathlib import Path
import hashlib
import shutil
import unittest
import uuid

import torch

from reelcache.serialization import tensor_bytes, tensor_from_bytes
from reelcache.types import EncodedTensor, MatrixKey, ModelIdentity, TensorContext


class CoreWorkspaceTest(unittest.TestCase):
    def setUp(self):
        self.workspace = Path(".core-test-artifacts") / uuid.uuid4().hex
        self.workspace.mkdir(parents=True)
        self.addCleanup(self._clean_workspace)

    def _clean_workspace(self):
        shutil.rmtree(self.workspace, ignore_errors=True)
        try:
            self.workspace.parent.rmdir()
        except OSError:
            pass


def model(*, layers=2, channels=4, dtype="float32", fingerprint="fixture-v1"):
    return ModelIdentity("fixture", fingerprint, layers, channels, "tokens_channels", dtype)


def chunk_inputs(identity, *, start=0, frames=1, height=2, width=2, offset=0, key_space="pre_rope"):
    matrices, contexts = {}, {}
    for layer in range(identity.layers):
        for target in ("K", "V"):
            key = MatrixKey(layer, target)
            context = TensorContext(
                identity, key, start, frames, height, width,
                key_space if target == "K" else "none",
            )
            tensor = torch.arange(context.token_count * context.channels, dtype=torch.float32)
            matrices[key] = (tensor.reshape(context.shape) + offset + layer * 100
                             + (20 if target == "V" else 0)).to(getattr(torch, context.dtype))
            contexts[key] = context
    return matrices, contexts


def metadata(*, end=4, chunk_id="c0"):
    return {
        "clean_boundary": {"token_end": end, "chunk_id": chunk_id, "clean": True},
        "conditioning": {"identity": "prompt-and-negative-sha256"},
        "execution": {"cursor": end, "scheduler": "fixture", "pending": False},
        "rng": {"contract": "torch_cpu_state", "tensor": "rng"},
        "latent": {"contract": "prefix", "tensor": "latent"},
        "vae": {"contract": "explicit-cache", "tensor": "vae"},
    }


def raw_encoder(tensor, context, spec, *, reuse_basis=None, reuse_threshold=None):
    if spec.kind != "raw":
        raise AssertionError("fixture provider only encodes RAW")
    payloads = {"matrix": tensor_bytes(tensor)}
    return EncodedTensor(encoded_metadata(context, spec, payloads), payloads)


def encoded_metadata(context, spec, payloads, *, basis=None):
    portable_spec = spec.to_dict()
    portable_spec.pop("ffmpeg_path", None)
    metadata = {
        "kind": spec.kind, "context": context.to_dict(), "codec": portable_spec,
        "shape": list(context.shape), "dtype": context.dtype,
        "payloads": {name: {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
                     for name, payload in payloads.items()},
        "basis": None,
    }
    if basis is not None:
        data = tensor_bytes(basis)
        metadata["basis"] = {
            "dtype": "float16", "shape": list(basis.shape), "view_rank": spec.rank,
            "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
        }
    return metadata


def raw_decoder(encoded, *, device="cpu", dtype=None, columns=None):
    value = tensor_from_bytes(
        encoded.payloads["matrix"], dtype=encoded.metadata["dtype"],
        shape=encoded.metadata["shape"], device=device,
    )
    if columns is not None:
        value = value[:, columns]
    return value if dtype is None else value.to(dtype=dtype)
