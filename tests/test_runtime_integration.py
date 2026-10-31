import gc
import tempfile
import unittest
import weakref
from unittest import mock

import torch

from reelcache.codecs import encode_tensor
from reelcache.execution import BoundedExecutor
from reelcache.manager import KVRuntime
from reelcache.policy import select_profile
from reelcache.types import CodecSpec, MatrixKey, ModelIdentity, TensorContext


MODEL = ModelIdentity("integration-toy", "d" * 64, 1, 8, "canonical", "float32")
KEYS = (MatrixKey(0, "K"), MatrixKey(0, "V"))


def contexts(start, key_space="post_rope"):
    return {
        key: TensorContext(
            MODEL, key, start, 2, 2, 4, key_space if key.target == "K" else "none"
        )
        for key in KEYS
    }


def snapshot_metadata(chunk, token_end):
    return {
        "clean_boundary": {"chunk_id": chunk, "token_end": token_end, "clean": True},
        "conditioning": {"id": "unit-conditioning"},
        "execution": {"next_chunk": 1, "pending": False},
        "rng": {"generator_tensor": "rng"},
        "latent": {"tensor": "latent"},
        "vae": {"strategy": "stateless-whole-prefix"},
    }


class RuntimeIntegrationTests(unittest.TestCase):
    def test_layer_lease_reads_only_selected_raw_payload(self):
        model = ModelIdentity(
            "read-amplification", "e" * 64, 8, 8, "canonical", "float32"
        )
        keys = [MatrixKey(layer, target) for layer in range(8) for target in ("K", "V")]
        matrices = {
            key: torch.full((16, 8), float(index + 1)) for index, key in enumerate(keys)
        }
        metadata = {
            key: TensorContext(
                model, key, 0, 2, 2, 4,
                "post_rope" if key.target == "K" else "none",
            )
            for key in keys
        }
        with tempfile.TemporaryDirectory() as directory:
            runtime = KVRuntime(directory, model, max_resident_bytes=65536)
            try:
                runtime.create_branch("main")
                runtime.seal_chunk("main", "first", matrices, metadata, CodecSpec())
                reads = []
                original = runtime.store._get

                def measured_read(object_id):
                    value = original(object_id)
                    if object_id.startswith("blob:"):
                        reads.append((object_id, len(value)))
                    return value

                with mock.patch.object(runtime.store, "_get", side_effect=measured_read):
                    with runtime.acquire_history("main", keys[0]) as lease:
                        self.assertTrue(torch.equal(lease.tensor, matrices[keys[0]]))
                self.assertEqual(len({object_id for object_id, _ in reads}), 1)
                requested_bytes = matrices[keys[0]].numel() * matrices[keys[0]].element_size()
                self.assertLessEqual(sum(size for _, size in reads), 3 * requested_bytes)
                reads.clear()
                with mock.patch.object(runtime.store, "_get", side_effect=measured_read):
                    with runtime.acquire_history("main", keys[0], token_end=0) as lease:
                        self.assertEqual(tuple(lease.tensor.shape), (0, 8))
                self.assertEqual(reads, [])
            finally:
                runtime.close()

    def test_policy_executor_snapshot_fork_and_restart(self):
        generator = torch.Generator().manual_seed(73)
        matrices = {
            key: torch.randn(16, 8, generator=generator) for key in KEYS
        }
        latent = torch.randn(4, generator=generator)
        rng = generator.get_state()
        selection = select_profile(
            [], model=MODEL, geometry=(2, 2, 4), history_chunks=1,
            features={}, raw_bytes=sum(t.numel() * t.element_size() for t in matrices.values()),
        )
        self.assertEqual(selection.profile_id, "RAW")
        with tempfile.TemporaryDirectory() as directory:
            runtime = KVRuntime(directory, MODEL, max_resident_bytes=65536)
            try:
                runtime.create_branch("main")
                with BoundedExecutor(max_staging_bytes=4096) as executor:
                    executor.submit(
                        runtime.seal_chunk, "main", "chunk-0", matrices,
                        contexts(0), selection.specs, staging_bytes=1024,
                    ).result(timeout=30)
                snapshot = runtime.snapshot(
                    "main", metadata=snapshot_metadata("chunk-0", 16),
                    tensors={"rng": rng, "latent": latent},
                )
                restored = runtime.fork(snapshot, "child")
                self.assertTrue(torch.equal(restored.tensors["latent"], latent))
                child_generator = torch.Generator()
                child_generator.set_state(restored.tensors["rng"])
                self.assertTrue(torch.equal(
                    torch.randn(8, generator=generator),
                    torch.randn(8, generator=child_generator),
                ))
                runtime.seal_chunk(
                    "child", "chunk-1",
                    {key: value + 1 for key, value in matrices.items()},
                    contexts(16), CodecSpec(),
                )
                with runtime.acquire_history("main", KEYS[0]) as lease:
                    self.assertTrue(torch.equal(lease.tensor, matrices[KEYS[0]]))
                    reference = weakref.ref(lease.tensor)
                gc.collect()
                self.assertIsNone(reference())
                with runtime.acquire_history(
                    "child", KEYS[0], token_start=16, token_end=32
                ) as lease:
                    self.assertTrue(torch.equal(lease.tensor, matrices[KEYS[0]] + 1))
            finally:
                runtime.close()
            reopened = KVRuntime(directory, MODEL, max_resident_bytes=65536)
            try:
                reopened.restore(snapshot, "resumed")
                with reopened.acquire_history("resumed", KEYS[1]) as lease:
                    self.assertTrue(torch.equal(lease.tensor, matrices[KEYS[1]]))
            finally:
                reopened.close()

    def test_real_hevc_basis_reuse_and_independent_reload(self):
        generator = torch.Generator().manual_seed(91)
        matrices = {
            key: torch.randn(16, 2, generator=generator)
            @ torch.randn(2, 8, generator=generator)
            for key in KEYS
        }
        spec = CodecSpec(
            kind="svd_hevc", rank=2, normalization="leading_columns",
            protected_fraction=0.5, pix_fmt="yuv444p",
        )
        prior_supplied, bases = [], []

        def recording_encoder(tensor, context, selected, **kwargs):
            prior_supplied.append(kwargs.get("reuse_basis") is not None)
            packet = encode_tensor(tensor, context, selected, **kwargs)
            bases.append(packet.basis)
            return packet

        with tempfile.TemporaryDirectory() as directory:
            runtime = KVRuntime(
                directory, MODEL, max_resident_bytes=65536,
                encoder=recording_encoder,
            )
            try:
                runtime.create_branch("main")
                runtime.seal_chunk("main", "first", matrices, contexts(0, "pre_rope"), spec)
                runtime.seal_chunk(
                    "main", "second", matrices, contexts(16, "pre_rope"),
                    spec, reuse_threshold=0.01,
                )
                self.assertEqual(prior_supplied, [False, False, True, True])
                self.assertEqual(bases[:2], bases[2:])
            finally:
                runtime.close()
            reopened = KVRuntime(directory, MODEL, max_resident_bytes=65536)
            try:
                for key in KEYS:
                    with reopened.acquire_history(
                        "main", key, token_start=16, token_end=32, columns=(0, 4)
                    ) as lease:
                        expected = matrices[key][:, :4]
                        self.assertEqual(tuple(lease.tensor.shape), (16, 4))
                        relative = torch.linalg.vector_norm(
                            lease.tensor - expected
                        ) / torch.linalg.vector_norm(expected)
                        self.assertLess(float(relative), 0.15)
            finally:
                reopened.close()


if __name__ == "__main__":
    unittest.main()
