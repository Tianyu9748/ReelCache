from collections.abc import Mapping
import gc
from pathlib import Path
import tempfile
import threading
import unittest
import weakref

import torch

from reelcache.codecs import encode_tensor
from reelcache.errors import CapacityError, ConfigurationError, StateError, UnsupportedError
from reelcache.execution import BoundedExecutor
from reelcache.manager import KVRuntime
from reelcache.types import CodecSpec, MatrixKey, TensorContext
from tests.test_grouped_history import MODEL


KEYS = (MatrixKey(0, "K"), MatrixKey(0, "V"))


def contexts():
    return {
        key: TensorContext(MODEL, key, 0, 2, 2, 4, "pre_rope" if key.target == "K" else "none")
        for key in KEYS
    }


class Sources(Mapping):
    def __init__(self):
        self.calls = []
        self.references = []

    def __len__(self):
        return len(KEYS)

    def __iter__(self):
        return iter(KEYS)

    def __getitem__(self, key):
        if key not in KEYS:
            raise KeyError(key)
        self.calls.append(key)
        value = torch.arange(128, dtype=torch.float32).reshape(16, 8).clone()
        self.references.append(weakref.ref(value))
        return value


class ParallelSealTests(unittest.TestCase):
    def test_overlapping_encodes_persist_small_results_and_release_sources(self):
        barrier = threading.Barrier(2)

        def encoder(*args, **kwargs):
            barrier.wait(timeout=10)
            return encode_tensor(*args, **kwargs)

        sources = Sources()
        with tempfile.TemporaryDirectory() as root, BoundedExecutor(
            max_workers=2, max_pending=2, max_staging_bytes=32768,
        ) as executor, KVRuntime(
            root, MODEL, encoder=encoder, executor=executor,
            encode_concurrency=2, encode_workspace_bytes=8192,
        ) as runtime:
            runtime.create_branch("main")
            runtime.seal_chunk("main", "c0", sources, contexts())
            gc.collect()
            self.assertEqual(sources.calls, list(KEYS))
            self.assertTrue(all(ref() is None for ref in sources.references))
            stats = executor.stats()
            self.assertEqual(stats["peak_pending_jobs"], 2)
            self.assertEqual(stats["peak_staging_bytes"], 2 * (512 + 8192))
            self.assertEqual(stats["pending_jobs"], stats["staging_bytes"])
            self.assertEqual(stats["pending_jobs"], 0)
            for key in KEYS:
                with runtime.acquire_history("main", key) as lease:
                    self.assertTrue(torch.equal(lease.tensor, torch.arange(128).reshape(16, 8).float()))

    def test_parallel_raw_svd_hevc_match_serial_representation_and_rng(self):
        generator = torch.Generator().manual_seed(235)
        sources = {key: torch.randn(16, 8, generator=generator) for key in KEYS}
        rng = torch.random.get_rng_state()
        for kind in ("raw", "svd", "svd_hevc"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as root:
                with BoundedExecutor(max_workers=2) as executor, KVRuntime(
                    Path(root) / "parallel", MODEL, executor=executor,
                    encode_concurrency=2, encode_workspace_bytes=65536,
                ) as parallel, KVRuntime(Path(root) / "serial", MODEL) as serial:
                    spec = CodecSpec(kind=kind, rank=None if kind == "raw" else 3)
                    for runtime in (serial, parallel):
                        runtime.create_branch("main")
                        runtime.seal_chunk("main", "c0", sources, contexts(), spec)
                    for key in KEYS:
                        with serial.acquire_history("main", key) as a, parallel.acquire_history("main", key) as b:
                            self.assertTrue(torch.equal(a.tensor, b.tensor))
                    self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))

    def test_failure_drains_other_workers_before_releasing_branch(self):
        entered, release = threading.Event(), threading.Event()
        errors = []

        def encoder(tensor, context, spec, **kwargs):
            if context.key.target == "V":
                entered.set()
                if not release.wait(10):
                    raise RuntimeError("test synchronization timeout")
                return encode_tensor(tensor, context, spec, **kwargs)
            if not entered.wait(10):
                raise RuntimeError("second worker did not start")
            raise ValueError("injected K failure")

        with tempfile.TemporaryDirectory() as root, BoundedExecutor(max_workers=2) as executor, KVRuntime(
            root, MODEL, executor=executor, encoder=encoder,
            encode_concurrency=2, encode_workspace_bytes=8192,
        ) as runtime:
            runtime.create_branch("main")
            before = runtime.branch_info("main")
            sources = Sources()

            def seal():
                try:
                    runtime.seal_chunk("main", "c0", sources, contexts())
                except Exception as exc:
                    errors.append(exc)

            thread = threading.Thread(target=seal)
            thread.start()
            try:
                self.assertTrue(entered.wait(10))
                self.assertTrue(thread.is_alive())
                with KVRuntime(root, MODEL) as reader:
                    self.assertEqual(reader.branch_info("main"), before)
            finally:
                release.set()
                thread.join(15)
            self.assertFalse(thread.is_alive())
            self.assertEqual(len(errors), 1)
            self.assertIsInstance(errors[0], ValueError)
            self.assertEqual(runtime.branch_info("main"), before)
            self.assertEqual(executor.stats()["pending_jobs"], 0)
            self.assertEqual(executor.stats()["staging_bytes"], 0)
            self.assertEqual(executor.stats()["unobserved_failures"], {})
            gc.collect()
            self.assertTrue(all(ref() is None for ref in sources.references))

    def test_oversize_packet_does_not_publish_or_keep_jobs(self):
        with tempfile.TemporaryDirectory() as root, BoundedExecutor(max_workers=2) as executor, KVRuntime(
            root, MODEL, executor=executor, encode_concurrency=2, encode_workspace_bytes=1,
        ) as runtime:
            runtime.create_branch("main")
            before = runtime.branch_info("main")
            with self.assertRaisesRegex((CapacityError, StateError), "workspace reservation"):
                runtime.seal_chunk("main", "c0", Sources(), contexts())
            self.assertEqual(runtime.branch_info("main"), before)
            self.assertEqual(executor.stats()["pending_jobs"], 0)
            self.assertEqual(executor.stats()["unobserved_failures"], {})

    def test_parallel_configuration_and_adaptive_ordering_are_explicit(self):
        with tempfile.TemporaryDirectory() as root, BoundedExecutor(max_workers=2) as executor:
            for options in (
                {"encode_concurrency": 0}, {"encode_concurrency": True},
                {"encode_workspace_bytes": -1}, {"encode_concurrency": 2},
                {"encode_concurrency": 3, "encode_workspace_bytes": 8192},
            ):
                with self.subTest(options=options), self.assertRaises(ConfigurationError):
                    KVRuntime(root, MODEL, executor=executor, **options)
            with KVRuntime(
                root, MODEL, executor=executor, encode_concurrency=2, encode_workspace_bytes=8192,
            ) as runtime:
                runtime.create_branch("main")
                with self.assertRaisesRegex(UnsupportedError, "ordered serial"):
                    runtime.seal_chunk("main", "c0", Sources(), contexts(), rank_policy=object())


if __name__ == "__main__":
    unittest.main()
