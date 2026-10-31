"""L1 (SGLang freeze C): each stored basis byte is hashed once per read on the store path.

The store path (``LineageRestorer`` without a staging engine, ``node_pack``, ``matrix_packet``, the device basis
cache) hashed basis bytes up to three times per read: the row block against its content address (the store's
verified read), the complete view against the version's ``view_sha256`` and every tile prefix against its pack
descriptor (``matrix_packet``: the block again, the view, the packet descriptor). With ``verify_reads`` the version
and pack manifests holding those digests are hash-checked when read too, so the digests only re-checked
authenticated bytes: they are skipped there. Without ``verify_reads`` manifests are not verified and every check
stays. These tests pin the hashing, the unchanged detection of corrupted bytes in both modes and bitwise outputs.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import sys
import unittest
from unittest import mock

import torch

from reelcache import restore_kernels as rk
from reelcache.archive_restore import LineageRestorer
from reelcache.errors import IntegrityError
from reelcache.store import ObjectStore
from reelcache.types import MatrixKey
from tests import step6_support as s6
from tests.test_packed_nvdec import SoftwarePool
from tests.test_step4_lineage import KEYS as STEP4_KEYS, LineageTestCase as Step4Case, chunk, orthonormal
from tests.test_step7_store_faults import SERVING, StoreFaultCase, _destination

_REAL_SHA256 = hashlib.sha256


class _CountingHash:
    """A sha256 object logging ``(caller module, hexdigest, bytes hashed)`` when its digest is taken."""

    def __init__(self, log, module, data=b""):
        self._hash, self._log, self._module, self._bytes = _REAL_SHA256(), log, module, 0
        if data:
            self.update(data)

    def update(self, data):
        self._bytes += memoryview(data).nbytes
        self._hash.update(data)

    def hexdigest(self):
        value = self._hash.hexdigest()
        self._log.append((self._module, value, self._bytes))
        return value

    def digest(self):
        return self._hash.digest()

    def copy(self):
        clone = _CountingHash(self._log, self._module)
        clone._hash, clone._bytes = self._hash.copy(), self._bytes
        return clone


@contextlib.contextmanager
def counting_sha256():
    log = []

    def factory(data=b"", **_kwargs):
        return _CountingHash(log, sys._getframe(1).f_globals.get("__name__", "?"), data)
    with mock.patch.object(hashlib, "sha256", factory):
        yield log


def _rewrite_manifest(root, object_id, edit):
    """Corrupt a stored manifest in place (valid JSON, same object id): ``edit(data)`` changes its ``data``."""
    with ObjectStore(root) as store:
        path = store.path_for(object_id)
        value = json.loads(path.read_bytes())
        edit(value["data"])
        path.write_bytes(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())


def _flip_hex(text):
    return ("1" if text[0] != "1" else "2") + text[1:]


class NodePackHashOnceTests(StoreFaultCase):
    """``a``'s newest node pack: its tile prefixes are assembled from row blocks only ``a`` reaches."""

    def blocks(self, runtime):
        packet = runtime.load_packet(self.pack)
        refs = [item["basis_version"] for item in packet["tile_bases"]]
        return packet, {block for ref in refs for block in runtime.basis_versions.blocks(ref)}

    def test_verified_reads_hash_each_basis_block_once_and_nothing_else(self):
        runtime = self.open(**SERVING)
        packet, blocks = self.blocks(runtime)
        digests = {block.split(":", 1)[1] for block in blocks}
        with counting_sha256() as log:
            runtime.node_pack(self.pack)
        by_block = {digest: [entry for entry in log if entry[1] == digest] for digest in digests}
        self.assertTrue(all(len(entries) == 1 and entries[0][0] == "reelcache.store" for entries in by_block.values()),
                        by_block)
        self.assertEqual([entry for entry in log if entry[0] in ("reelcache.basis_versions", "reelcache.node_packets")], [])
        # The store-path restore: no basis digest is recomputed outside the store's verified reads.
        descriptors = {packet["metadata"]["payloads"][tile["basis"]["payload"]]["sha256"]
                       for tile in packet["metadata"]["tiles"]}
        with counting_sha256() as log:
            restored = self.restore(runtime, self.heads["a"])
        self.assertEqual([entry for entry in log if entry[0] == "reelcache.basis_versions"], [])
        # (A prefix that is one whole root block has the block's address as its descriptor digest: store reads.)
        self.assertEqual([entry for entry in log if entry[1] in descriptors and entry[0] != "reelcache.store"], [])
        self.assertSameCaches(restored, self.reference["a"])

    def test_unverified_reads_keep_every_digest_check(self):
        runtime = self.open()
        packet, blocks = self.blocks(runtime)
        with counting_sha256() as log:
            runtime.node_pack(self.pack)
        descriptors = {packet["metadata"]["payloads"][tile["basis"]["payload"]]["sha256"]
                       for tile in packet["metadata"]["tiles"]}
        # Blocks against their addresses (always), tile prefixes against their (unverified) descriptors.
        self.assertTrue({block.split(":", 1)[1] for block in blocks} <= {entry[1] for entry in log})
        self.assertTrue(descriptors <= {entry[1] for entry in log if entry[0] == "reelcache.node_packets"})

    def test_outputs_are_bitwise_identical_in_both_modes(self):
        verified = self.open(**SERVING)
        expected = verified.node_pack(self.pack)
        verified.close()
        plain = self.open()
        actual = plain.node_pack(self.pack)
        self.assertEqual(actual.metadata, expected.metadata)
        self.assertEqual(actual.payloads, expected.payloads)
        self.assertSameCaches(self.restore(plain, self.heads["a"]), self.reference["a"])
        plain.close()
        self.assertSameCaches(self.restore(self.open(**SERVING), self.heads["a"]), self.reference["a"])

    def check_detected(self, flags, *, message=None):
        runtime = self.open(**flags)
        with self.assertRaisesRegex(IntegrityError, message or ""):
            runtime.node_pack(self.pack)
        caches, destination = _destination(2 * s6.TOKENS)
        with self.assertRaises(IntegrityError):
            LineageRestorer(runtime, SoftwarePool(), arena_bytes=1 << 30).restore(self.heads["a"], destination)
        self.assertTrue(all(not value.any() for value in caches.values()))
        self.assertSameCaches(self.restore(runtime, self.heads["b"]), self.reference["b"])
        runtime.close()

    def test_a_corrupted_basis_block_is_detected_in_both_modes(self):
        runtime = self.open()
        _, blocks = self.blocks(runtime)
        only_a = sorted(blocks - self.reachable_blocks(runtime, self.heads["b"]))
        runtime.close()
        self.assertTrue(only_a)
        self.corrupt(only_a[-1])
        self.check_detected(SERVING, message="hash or length mismatch")
        self.check_detected({})

    def reachable_blocks(self, runtime, head):
        with runtime.store.transaction():
            return {item for item in runtime.store._walk([head], verify=False) if item.startswith("basis:")}

    def test_a_corrupted_tile_descriptor_is_detected_in_both_modes(self):
        runtime = self.open()
        packet = runtime.load_packet(self.pack)
        name = packet["metadata"]["tiles"][0]["basis"]["payload"]
        runtime.close()

        def edit(data):
            descriptor = data["metadata"]["payloads"][name]
            descriptor["sha256"] = _flip_hex(descriptor["sha256"])
        _rewrite_manifest(self.root, self.pack, edit)
        self.check_detected(SERVING, message="hash or length mismatch")  # the manifest's own address
        self.check_detected({}, message="differs from the packed tile descriptor")

    def test_a_corrupted_version_digest_is_detected_in_both_modes(self):
        runtime = self.open()
        packet = runtime.load_packet(self.pack)
        version = packet["tile_bases"][0]["basis_version"]
        runtime.close()
        _rewrite_manifest(self.root, version, lambda data: data["record"].update(
            view_sha256=_flip_hex(data["record"]["view_sha256"])))
        verified = self.open(**SERVING)
        with self.assertRaisesRegex(IntegrityError, "hash or length mismatch"):
            verified.basis_versions.assemble(version)
        verified.close()
        plain = self.open()
        with self.assertRaisesRegex(IntegrityError, "differs from its version digest"):
            plain.basis_versions.assemble(version)


class MatrixPacketHashOnceTests(Step4Case):
    """Per-matrix (Step 4) records: ``matrix_packet`` assembles the prefix from blocks it reads itself."""

    def setUp(self):
        super().setUp()
        self.runtime.create_lineage("main")
        self.result = self.commit("main", "c1", chunk(1, 0, orthonormal(1)))
        self.key = STEP4_KEYS[0]
        self.record_ref = self.result.records[0][1][self.key]
        self.expected = self.runtime.matrix_packet(self.record_ref)
        self.root = self.runtime.store.root
        self.runtime.close()

    def reopen(self, **flags):
        from reelcache.manager import KVRuntime
        from tests.test_step4_lineage import MODEL
        self.runtime = KVRuntime(self.root, MODEL, max_resident_bytes=1 << 28, **flags)
        return self.runtime

    def test_verified_matrix_packets_hash_blocks_once(self):
        runtime = self.reopen(**SERVING)
        record = runtime.load_record(self.record_ref)
        blocks = runtime.basis_versions.blocks(record.basis_version_ref)
        with counting_sha256() as log:
            packet = runtime.matrix_packet(self.record_ref)
        self.assertEqual((packet.metadata, packet.payloads, packet.basis),
                         (self.expected.metadata, self.expected.payloads, self.expected.basis))
        for block in blocks:
            self.assertEqual([entry[0] for entry in log if entry[1] == block.split(":", 1)[1]], ["reelcache.store"])
        self.assertEqual([entry for entry in log if entry[0] == "reelcache.basis_versions"], [])
        self.assertEqual([entry for entry in log
                          if entry[1] == packet.metadata["basis"]["sha256"] and entry[0] != "reelcache.store"], [])

    def test_a_corrupted_block_is_detected_in_both_modes(self):
        runtime = self.reopen()
        block = runtime.basis_versions.blocks(runtime.load_record(self.record_ref).basis_version_ref)[-1]
        runtime.close()
        with ObjectStore(self.root) as store:
            path = store.path_for(block)
            data = bytearray(path.read_bytes())
            data[len(data) // 2] ^= 0x40
            path.write_bytes(bytes(data))
        for flags, message in ((SERVING, "hash or length mismatch"), ({}, "differ from their content address")):
            runtime = self.reopen(**flags)
            with self.assertRaisesRegex(IntegrityError, message):
                runtime.matrix_packet(self.record_ref)
            runtime.close()

    def test_a_corrupted_packet_descriptor_is_detected_in_both_modes(self):
        runtime = self.reopen()
        packet_ref = runtime.load_record(self.record_ref).packet_ref
        runtime.close()
        _rewrite_manifest(self.root, packet_ref, lambda data: data["metadata"]["basis"].update(
            sha256=_flip_hex(data["metadata"]["basis"]["sha256"])))
        for flags, message in ((SERVING, "hash or length mismatch"), ({}, "bound prefix")):
            runtime = self.reopen(**flags)
            with self.assertRaisesRegex(IntegrityError, message):
                runtime.matrix_packet(self.record_ref)
            runtime.close()


if os.environ.get("REELCACHE_STEP7_GPU_TESTS") == "1":
    class CUDAHashOnceTests(StoreFaultCase):
        """NVDEC store-path restores (no staging engine) and session-open prewarms of a ``verify_reads`` store:
        bitwise those of an unverified store, without recomputing basis digests (inside GPU jobs)."""

        def setUp(self):
            if not torch.cuda.is_available():
                raise AssertionError("REELCACHE_STEP7_GPU_TESTS=1 requires a CUDA device")
            super().setUp()

        def restorer(self, runtime):
            from reelcache.decoder_pool import DecoderPool
            uuid_ = "GPU-" + str(torch.cuda.get_device_properties(0).uuid).removeprefix("GPU-")
            pool = DecoderPool(gpu_id=0, gpu_uuid=uuid_, workers=2, max_sessions_per_worker=16)
            restorer = LineageRestorer(runtime, pool, arena_bytes=1 << 28)
            self.addCleanup(pool.close)
            self.addCleanup(restorer.close)
            return restorer

        def cuda_restore(self, restorer, head):
            caches = {}

            def destination(node_id, key, tile):
                key = key if isinstance(key, MatrixKey) else key[0]
                if key not in caches:
                    caches[key] = torch.zeros((1, 2 * s6.TOKENS, 1, s6.CHANNELS), dtype=torch.float32,
                                              device="cuda")
                return rk.NativeView(caches[key], "BTHD", token_offset=tile.token_start)
            try:
                restorer.restore(head, destination)
            finally:
                torch.cuda.synchronize()
            return {key: value.cpu() for key, value in caches.items()}

        def test_store_path_restores_hash_once_and_restore_bitwise(self):
            plain = self.open()
            reference = {name: self.cuda_restore(self.restorer(plain), head) for name, head in self.heads.items()}
            plain.close()
            verified = self.open(**SERVING)
            restorer = self.restorer(verified)
            packet = verified.load_packet(self.pack)
            descriptors = {packet["metadata"]["payloads"][tile["basis"]["payload"]]["sha256"]
                           for tile in packet["metadata"]["tiles"]}
            with counting_sha256() as log:
                restorer.prewarm(restorer.history(self.heads["b"]))  # the bridge's session open (store path)
                restored = self.cuda_restore(restorer, self.heads["a"])
            self.assertSameCaches(restored, reference["a"])
            self.assertSameCaches(self.cuda_restore(restorer, self.heads["b"]), reference["b"])
            self.assertEqual([entry for entry in log if entry[0] == "reelcache.basis_versions"], [])
            self.assertEqual([entry for entry in log if entry[1] in descriptors and entry[0] != "reelcache.store"], [])


if __name__ == "__main__":
    unittest.main()
