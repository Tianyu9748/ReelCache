import random
import tempfile
import threading
import unittest
from unittest import mock

import torch

from reelcache.errors import ConfigurationError, PendingError, StateError
from reelcache.execution import BoundedExecutor
from reelcache.manager import KVRuntime
from reelcache.types import CodecSpec, MatrixKey, ModelIdentity, TensorContext

IDENTITY = ModelIdentity("stream", "test-v1", 3, 4, "canonical", "float32")


def inputs(token_start=0, seed=0):
    generator = torch.Generator().manual_seed(seed)
    tensors = {MatrixKey(layer, t): torch.randn(24, 4, generator=generator)
               for layer in range(3) for t in ("K", "V")}
    contexts = {k: TensorContext(IDENTITY, k, token_start, 2, 3, 4, "pre_rope" if k.target == "K" else "none")
                for k in tensors}
    return tensors, contexts


SPECS = {MatrixKey(layer, t): (CodecSpec() if layer == 0 else CodecSpec("svd", rank=2) if layer == 1
                                else CodecSpec("svd_hevc", rank=3))
         for layer in range(3) for t in ("K", "V")}


def portable(runtime, chunk_object):
    """Chunk manifest with every packet expanded, minus timing diagnostics (never deterministic)."""
    chunk = runtime.store.get_manifest(chunk_object)
    rows = []
    for entry in chunk["matrices"]:
        packet = runtime.store.get_manifest(entry["tensor"])
        packet["encoded"].pop("metrics", None)
        rows.append((entry["key"], packet))
    return {k: v for k, v in chunk.items() if k != "matrices"}, rows


class StreamingChunkTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)

    def runtime(self, name):
        runtime = KVRuntime(f"{self.directory.name}/{name}", IDENTITY)
        self.addCleanup(runtime.close)
        runtime.create_branch("main")
        return runtime

    def test_streamed_publication_is_identical_to_serial_seal(self):
        serial = self.runtime("serial")
        expected = []
        for chunk in range(2):
            tensors, contexts = inputs(24 * chunk, seed=chunk)
            expected.append(portable(serial, serial.seal_chunk("main", f"c{chunk}", tensors, contexts, SPECS)))
        for label, workers in (("inline", 0), ("pool", 3)):
            with self.subTest(label), BoundedExecutor(max_workers=max(1, workers), max_pending=max(1, workers),
                                                     max_staging_bytes=1 << 26) as executor:
                runtime = self.runtime(label)
                for chunk in range(2):
                    tensors, contexts = inputs(24 * chunk, seed=chunk)
                    keys = list(tensors)
                    random.Random(chunk).shuffle(keys)
                    before = runtime.store.get_ref("branches/main")
                    with runtime.open_chunk("main", f"c{chunk}", contexts, SPECS,
                                            executor=executor if workers else None) as stream:
                        for key in keys:
                            stream.add(key, tensors[key])
                        self.assertEqual(runtime.branch_info("main")["token_end"], 24 * chunk)
                        self.assertEqual(runtime.store.get_ref("branches/main"), before)
                        chunk_object = stream.commit()
                    self.assertEqual(portable(runtime, chunk_object), expected[chunk])
                    self.assertEqual(stream.stats["encoded"], 6)
                for key in SPECS:
                    with serial.acquire_history("main", key) as a, runtime.acquire_history("main", key) as b:
                        self.assertTrue(torch.equal(a.tensor, b.tensor))

    def test_producer_thread_and_reservation(self):
        runtime = self.runtime("thread")
        tensors, contexts = inputs()
        stream = runtime.open_chunk("main", "c0", contexts, SPECS)
        with self.assertRaises(PendingError):
            runtime.open_chunk("main", "c0", contexts, SPECS)
        with self.assertRaises(PendingError):
            runtime.seal_chunk("main", "c0", tensors, contexts, SPECS)
        errors = []

        def produce():
            try:
                for key in tensors:
                    stream.add(key, tensors[key])
            except BaseException as exc:
                errors.append(exc)

        thread = threading.Thread(target=produce)
        thread.start()
        thread.join(60)
        self.assertEqual(errors, [])
        stream.commit()
        self.assertEqual(runtime.branch_info("main")["token_end"], 24)
        with self.assertRaises(StateError):
            stream.add(MatrixKey(0, "K"), tensors[MatrixKey(0, "K")])

    def test_incomplete_failed_or_aborted_chunks_leave_no_trace(self):
        runtime = self.runtime("fail")
        tensors, contexts = inputs()
        before_head = runtime.branch_info("main")["head_id"]
        before_objects = runtime.store.stats()["object_count"]
        stream = runtime.open_chunk("main", "c0", contexts, SPECS)
        stream.add(MatrixKey(0, "K"), tensors[MatrixKey(0, "K")])
        with self.assertRaisesRegex(StateError, "missing matrices"):
            stream.commit()
        self.assertEqual(stream.state, "failed")
        self.assertEqual(runtime.branch_info("main")["head_id"], before_head)
        self.assertEqual(runtime.store.stats()["object_count"], before_objects)
        with BoundedExecutor(max_workers=2, max_pending=2, max_staging_bytes=1 << 26) as executor:
            encoder = runtime.encoder

            def flaky(tensor, context, spec, **kwargs):
                if context.key == MatrixKey(2, "V"):
                    raise RuntimeError("injected encode failure")
                return encoder(tensor, context, spec, **kwargs)

            with mock.patch.object(runtime, "encoder", side_effect=flaky):
                stream = runtime.open_chunk("main", "c0", contexts, SPECS, executor=executor)
                for key in tensors:
                    stream.add(key, tensors[key])
                with self.assertRaisesRegex(RuntimeError, "injected"):
                    stream.commit()
        self.assertEqual(runtime.branch_info("main")["head_id"], before_head)
        self.assertEqual(runtime.store.stats()["object_count"], before_objects)
        with runtime.open_chunk("main", "c0", contexts, SPECS) as stream:
            stream.add(MatrixKey(1, "V"), tensors[MatrixKey(1, "V")])
        self.assertEqual(stream.state, "aborted")
        self.assertEqual(runtime.store.stats()["object_count"], before_objects)
        runtime.seal_chunk("main", "c0", tensors, contexts, SPECS)

    def test_invalid_additions_are_rejected(self):
        runtime = self.runtime("invalid")
        tensors, contexts = inputs()
        with runtime.open_chunk("main", "c0", contexts, SPECS) as stream:
            stream.add(MatrixKey(0, "K"), tensors[MatrixKey(0, "K")])
            with self.assertRaises(StateError):
                stream.add(MatrixKey(0, "K"), tensors[MatrixKey(0, "K")])
            with self.assertRaises(ConfigurationError):
                stream.add(MatrixKey(9, "K"), tensors[MatrixKey(0, "K")])
            with self.assertRaises(ConfigurationError):
                stream.add(MatrixKey(0, "V"), torch.ones(24, 3))
        with self.assertRaises(ConfigurationError):
            runtime.open_chunk("main", "c0", {}, SPECS)
        with self.assertRaises(ConfigurationError):
            runtime.open_chunk("main", "c0", contexts, SPECS, executor=object())


if __name__ == "__main__":
    unittest.main()
