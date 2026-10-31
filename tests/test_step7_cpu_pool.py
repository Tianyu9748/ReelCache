"""CR-17-2 (SGLang Step 17): the CPU side of session-open prewarm and cold restores is independent of the intra-op pool.

Step 11's diagnosis (an A100-40GB node, a 16-CPU allocation straddling NUMA nodes 2 and 3): the session-open prewarm
took 41-56 s and the first restore was 3.5-5.6x slower, because assembling basis prefixes ran thousands of small
PyTorch CPU ops (``frombuffer().clone()``, ``cat``, ``isfinite().all()``) on the 16-thread OpenMP pool, whose per-op
synchronization cost up to 33.6 ms per 512-row block (79 us - 0.2 ms with 4-8 threads). Row blocks are now copied
and checked with NumPy on the calling thread, ``view_bytes`` joins checked block bytes without tensors, finiteness is
checked once per immutable block (memo), and the prefix digest is computed only where it is checked. The outputs are
bitwise those of the former implementations (kept below as references), and the restore path issues no PyTorch CPU op.
"""

from __future__ import annotations

import hashlib
import shutil
import sys
import unittest
import uuid
from unittest import mock

import numpy as np
import torch

from reelcache import basis_versions as bv
from reelcache.errors import CapacityError, ConfigurationError, IntegrityError
from reelcache.lineage_commit import ArchiveOptions
from reelcache.manager import KVRuntime
from reelcache.store import ObjectStore
from reelcache.types import MatrixKey, ModelIdentity
from tests import step6_support as s6
from tests.test_packed_nvdec import SoftwarePool


def former_rows_to_bytes(rows):
    if not isinstance(rows, torch.Tensor) or rows.dtype != torch.float16 or rows.ndim != 2 or min(rows.shape) < 1:
        raise ConfigurationError("basis row blocks are nonempty FP16 [rows, channels] tensors")
    if not bool(torch.isfinite(rows).all()):
        raise ConfigurationError("basis row blocks must be finite")
    if sys.byteorder != "little":
        raise ConfigurationError("basis row serialization requires a little-endian host")
    value = rows.detach().to("cpu").contiguous()
    return value.view(torch.uint8).numpy().tobytes()


def former_rows_from_bytes(data, rows, channels):
    if not isinstance(data, (bytes, bytearray)) or len(data) != rows * channels * 2:
        raise IntegrityError("basis row block length disagrees with its declared geometry")
    return torch.frombuffer(bytearray(data), dtype=torch.float16).reshape(rows, channels).clone()


def former_assemble(store, ref, rows=None, *, max_bytes=None, block_cache=None):
    version = store.load(ref)
    rows = version.total_rank if rows is None else rows
    if type(rows) is not int or not 1 <= rows <= version.total_rank:
        raise ConfigurationError("a basis view must be a nonempty legal prefix of its version")
    nbytes = rows * version.channels * 2
    if max_bytes is not None and nbytes > max_bytes:
        raise CapacityError(f"basis view needs {nbytes} bytes; workspace bound is {max_bytes}")
    parts, remaining = [], rows
    digest = hashlib.sha256()
    for block, count in store._blocks(ref):
        if remaining <= 0:
            break
        data = None if block_cache is None else block_cache.get(block)
        if data is None:
            data = store.store.get(block)
            if hashlib.sha256(data).hexdigest() != block.split(":", 1)[1]:
                raise IntegrityError("basis row block bytes differ from their content address")
            if block_cache is not None:
                block_cache[block] = data
        take = min(count, remaining)
        chunk = data[:take * version.channels * 2]
        digest.update(chunk)
        parts.append(former_rows_from_bytes(chunk, take, version.channels))
        remaining -= take
    view = torch.cat(parts, 0).contiguous()
    if rows == version.total_rank and digest.hexdigest() != version.view_sha256:
        raise IntegrityError("assembled basis view differs from its version digest")
    return view


SPECIALS = torch.tensor([0.0, -0.0, 6.1e-5, -6.1e-5, 5.96e-8, 65504.0, -65504.0, 1.0, -2.5], dtype=torch.float16)


def rows(count, channels, seed, *, specials=True):
    generator = torch.Generator().manual_seed(seed)
    value = (torch.randn(count, channels, generator=generator) * 3).to(torch.float16)
    if specials:
        flat = value.view(-1)
        flat[:SPECIALS.numel()] = SPECIALS  # subnormals, signed zeros, the largest finite values
    return value


class RowCodecEquivalenceTests(unittest.TestCase):
    def test_bytes_and_tensors_are_bitwise_the_former_ones(self):
        for count, channels in ((1, 1), (3, 7), (64, 128), (512, 1536)):
            value = rows(count, channels, count + channels, specials=count * channels >= SPECIALS.numel())
            data = bv.rows_to_bytes(value)
            self.assertEqual(data, former_rows_to_bytes(value))
            for source in (data, bytearray(data)):
                restored = bv.rows_from_bytes(source, count, channels)
                former = former_rows_from_bytes(source, count, channels)
                self.assertEqual((restored.dtype, restored.shape, restored.is_contiguous()),
                                 (former.dtype, former.shape, True))
                self.assertTrue(torch.equal(restored.view(torch.int16), former.view(torch.int16)))
                restored[0, 0] = 1  # owned and writable, like the former clone
            transposed = value.t()  # non-contiguous input
            self.assertEqual(bv.rows_to_bytes(transposed), former_rows_to_bytes(transposed))

    def test_rejections_are_the_former_ones(self):
        bad_inputs = [torch.zeros(2, 2), torch.zeros(4, dtype=torch.float16), torch.zeros(0, 3, dtype=torch.float16),
                      "rows"]
        for special in (float("inf"), float("-inf"), float("nan")):
            for position in ((0, 0), (5, 3), (63, 127)):
                value = rows(64, 128, 3)
                value[position] = special
                bad_inputs.append(value)
        for value in bad_inputs:
            with self.subTest(value=type(value)):
                with self.assertRaises(ConfigurationError) as former:
                    former_rows_to_bytes(value)
                with self.assertRaises(ConfigurationError) as current:
                    bv.rows_to_bytes(value)
                self.assertEqual(str(current.exception), str(former.exception))
        for data, count, channels in ((b"\x00" * 6, 2, 2), ("text", 1, 1), (b"\x00" * 10, 2, 2)):
            for function in (former_rows_from_bytes, bv.rows_from_bytes):
                with self.subTest(data=data, function=function), self.assertRaises(IntegrityError):
                    function(data, count, channels)
        bits = np.arange(1 << 16, dtype=np.uint16)  # every FP16 bit pattern
        values = torch.from_numpy(bits.view(np.float16).copy())
        self.assertEqual(bv._finite_fp16(bits.tobytes()), False)
        finite = torch.isfinite(values).numpy()
        per_value = np.array([bv._finite_fp16(item.tobytes()) for item in bits])
        self.assertTrue(np.array_equal(per_value, finite))


class AssemblyEquivalenceTests(unittest.TestCase):
    """``assemble`` / ``view_bytes`` on a stored version chain equal the former tensor-based implementations."""

    def setUp(self):
        self.root = s6.ROOT / f"cr172-{uuid.uuid4().hex}"
        self.addCleanup(shutil.rmtree, self.root, True)
        self.store = ObjectStore(self.root)
        self.addCleanup(self.store.close)
        self.model = ModelIdentity("cr172", "cr172-source", 2, 96, "tokens_channels", "float16")
        self.versions = bv.BasisVersionStore(self.store, self.model)
        key = MatrixKey(1, "V")
        with self.store.transaction():
            self.root_ref, _ = self.versions.put_root(key, rows(5, 96, 1))
            self.mid_ref, _ = self.versions.put_extension(self.root_ref, rows(3, 96, 2))
            self.tip_ref, _ = self.versions.put_extension(self.mid_ref, rows(7, 96, 3))

    def test_every_prefix_equals_the_former_assembly(self):
        for ref, total in ((self.root_ref, 5), (self.mid_ref, 8), (self.tip_ref, 15)):
            for count in range(1, total + 1):
                for cache in (None, {}):
                    with self.subTest(ref=ref[-6:], rows=count, cache=cache is not None):
                        former = former_assemble(self.versions, ref, count, block_cache=cache)
                        current = self.versions.assemble(ref, count, block_cache=cache)
                        self.assertEqual((current.dtype, current.shape, current.is_contiguous()),
                                         (former.dtype, former.shape, True))
                        self.assertTrue(torch.equal(current.view(torch.int16), former.view(torch.int16)))
                        self.assertEqual(self.versions.view_bytes(ref, count, block_cache=cache),
                                         former_rows_to_bytes(former))
            self.assertTrue(torch.equal(self.versions.assemble(ref).view(torch.int16),
                                        former_assemble(self.versions, ref).view(torch.int16)))

    def test_failures_are_the_former_ones(self):
        cases = [((self.tip_ref, 0), ConfigurationError), ((self.tip_ref, 16), ConfigurationError),
                 ((self.tip_ref, 1.0), ConfigurationError)]
        for (ref, count), error in cases:
            for function in (self.versions.assemble, self.versions.view_bytes,
                             lambda *a, **k: former_assemble(self.versions, *a, **k)):
                with self.subTest(rows=count), self.assertRaises(error):
                    function(ref, count)
        with self.assertRaises(CapacityError):
            self.versions.view_bytes(self.tip_ref, 15, max_bytes=15 * 96 * 2 - 1)
        self.assertEqual(len(self.versions.view_bytes(self.tip_ref, 15, max_bytes=15 * 96 * 2)), 15 * 96 * 2)
        # A block whose cached bytes are shorter than its declared rows, or a wrong full-view digest.
        blocks = self.versions.blocks(self.tip_ref)
        short = {blocks[0]: self.store.get(blocks[0])[:-2]}
        for function in (self.versions.assemble, self.versions.view_bytes):
            with self.assertRaisesRegex(IntegrityError, "length disagrees"):
                function(self.tip_ref, 5, block_cache=dict(short))
        swapped = {blocks[2]: bytes(reversed(self.store.get(blocks[2])))}
        for function in (self.versions.assemble, self.versions.view_bytes):
            with self.assertRaisesRegex(IntegrityError, "version digest"):
                function(self.tip_ref, 15, block_cache=dict(swapped))
            function(self.tip_ref, 14, block_cache=dict(swapped))  # a prefix is checked by its caller (as before)

    def test_non_finite_rows_fail_like_rows_to_bytes_and_only_where_used(self):
        blocks = self.versions.blocks(self.tip_ref)
        cache = {block: self.store.get(block) for block in blocks}
        for row, value in ((6, np.inf), (2, np.nan)):  # rows of the tip block (view rows 14 and 10)
            data = bytearray(cache[blocks[2]])
            data[row * 192:(row + 1) * 192] = np.full(96, value, dtype=np.float16).tobytes()
            bad = {**cache, blocks[2]: bytes(data)}
            self.versions.drop_caches()
            if row == 6:  # outside the 14-row prefix: accepted, as before, and the block is not memoized finite
                self.assertEqual(self.versions.view_bytes(self.tip_ref, 14, block_cache=dict(bad)),
                                 former_rows_to_bytes(former_assemble(self.versions, self.tip_ref, 14,
                                                                      block_cache=dict(bad))))
                self.assertNotIn(blocks[2], self.versions._finite_blocks)
            else:
                with self.assertRaisesRegex(ConfigurationError, "must be finite"):
                    former_rows_to_bytes(former_assemble(self.versions, self.tip_ref, 14, block_cache=dict(bad)))
                with self.assertRaisesRegex(ConfigurationError, "must be finite"):
                    self.versions.view_bytes(self.tip_ref, 14, block_cache=dict(bad))
            with self.assertRaisesRegex(IntegrityError, "version digest"):  # the full view: digest first, as before
                self.versions.view_bytes(self.tip_ref, 15, block_cache=dict(bad))

    def test_view_bytes_and_assemble_issue_no_torch_cpu_ops(self):
        cache = {}
        forbidden = {name: getattr(torch, name) for name in ("cat", "isfinite", "frombuffer", "stack")}

        def refuse(*_args, **_kwargs):
            raise AssertionError("a PyTorch CPU op on the restore path")
        with mock.patch.multiple(torch, **{name: refuse for name in forbidden}), \
                mock.patch.object(torch.Tensor, "clone", refuse), mock.patch.object(torch.Tensor, "contiguous", refuse):
            for ref in (self.root_ref, self.mid_ref, self.tip_ref):
                self.versions.view_bytes(ref, self.versions.load(ref).total_rank, block_cache=cache)
                self.versions.assemble(ref, block_cache=cache)
        self.assertEqual(self.versions._finite_blocks, set(self.versions.blocks(self.tip_ref)))
        self.versions.drop_caches()
        self.assertEqual(self.versions._finite_blocks, set())

    def test_extensions_continue_the_parents_digest_without_reading_it(self):
        extra = rows(4, 96, 9)
        with self.store.transaction(), mock.patch.object(self.store, "get",
                                                         side_effect=AssertionError("a parent block was read")):
            cached = self.versions.put_extension(self.tip_ref, extra)
        fresh = bv.BasisVersionStore(self.store, self.model)  # no digest states: reads the parent's blocks
        with self.store.transaction():
            read = fresh.put_extension(self.tip_ref, extra)
        self.assertEqual(cached, read)
        view = b"".join(self.store.get(block) for block in self.versions.blocks(cached[0]))
        self.assertEqual(cached[1].view_sha256, hashlib.sha256(view).hexdigest())
        self.versions.drop_caches()
        self.assertEqual(len(self.versions._view_digests), 0)


class NodePackEquivalenceTests(unittest.TestCase):
    """Whole node packs (every tile prefix through ``node_pack``) and restores are bitwise unchanged."""

    def test_node_packs_and_restores_equal_the_former_path(self):
        from reelcache.archive_restore import LineageRestorer
        from reelcache import restore_kernels as rk
        runtime = s6.RuntimeFactory(self)("cr172-pack")
        runtime.create_lineage("main")
        registry = s6.registry()
        heads = [runtime.commit_node("main", f"chunk-{index:03d}", *s6.chunk(index), method=s6.method(),
                                     registry=registry, options=ArchiveOptions(encode_threads=1)).node_ref
                 for index in range(3)]
        packs = [runtime.load_node(head).packet_ref for head in heads]
        current = [runtime.node_pack(pack) for pack in packs]

        def former_view_bytes(store, ref, rows, *, max_bytes=None, block_cache=None):
            return former_rows_to_bytes(former_assemble(store, ref, rows, max_bytes=max_bytes,
                                                        block_cache=block_cache))
        runtime.drop_manifest_caches()
        with mock.patch.object(bv.BasisVersionStore, "view_bytes", former_view_bytes):
            former = [runtime.node_pack(pack) for pack in packs]
        for new, old in zip(current, former):
            self.assertEqual(new.metadata, old.metadata)
            self.assertEqual(dict(new.payloads), dict(old.payloads))

        def restore():
            caches = {}

            def destination(node_id, key, tile):
                key = key if isinstance(key, MatrixKey) else key[0]
                if key not in caches:
                    caches[key] = torch.zeros((1, 3 * s6.TOKENS, 1, 128), dtype=torch.float32)
                return rk.NativeView(caches[key], "BTHD", token_offset=tile.token_start)
            LineageRestorer(runtime, SoftwarePool(), arena_bytes=1 << 30).restore(heads[-1], destination)
            return caches
        runtime.drop_manifest_caches()
        restored = restore()
        runtime.drop_manifest_caches()
        with mock.patch.object(bv.BasisVersionStore, "view_bytes", former_view_bytes), \
                mock.patch.object(bv.BasisVersionStore, "assemble", former_assemble):
            reference = restore()
        self.assertEqual(set(restored), set(reference))
        for key in restored:
            self.assertTrue(torch.equal(restored[key], reference[key]), key)


if __name__ == "__main__":
    unittest.main()
