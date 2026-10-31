"""SGLang Step 7 hooks for Step 5c's cold-session staging (CPU packing; user decision 2026-10-15, option A).

1. Packed commits record each committed range's node pack (``ChunkNode.packet_ref`` / ``TokenRangeRecords.packet_ref``)
   so :meth:`LineageRestorer.history` finds a range's pack without loading every matrix record; ``audit=True`` keeps the
   full record validation. Older nodes (no recorded pack) resolve through their records.
2. ``LineageRestorer(staging_engine=...)`` stages histories whose packs are not prepared and, when the restore
   finishes, adopts the staged packs and complete versions (``DeviceBasisCache.adopt_device`` with readiness).
3. ``ObjectStore.get_many`` (one index query, parallel reads/hash checks) and ``object_sizes``.
4. Batched manifests: ``ObjectStore.get_manifests_with_references``, ``BasisVersionStore.load_many`` /
   ``prefetch_blocks``; ``load_packet`` checks every payload size with one index query;
   ``KVRuntime.drop_manifest_caches`` (cold-session measurements).
5. Node packs record each picture's access-unit length per tile (``node_packets.recorded_access_units``).
"""

from __future__ import annotations

import unittest
from unittest import mock

import torch

from reelcache import lineage_commit
from reelcache.archive_restore import LineageRestorer
from reelcache.errors import ConfigurationError, IntegrityError
from reelcache.lineage_commit import ArchiveOptions
from reelcache.types import ChunkNode, TokenRangeRecords
from tests import step6_support as s6
from tests.test_packed_nvdec import SoftwarePool


def _commit_chain(test, name, count=3, *, legacy=False, wrong_packet=False):
    """Commit ``count`` chunks; ``legacy`` drops the recorded pack (pre-Step-7 nodes), ``wrong_packet`` records the
    previous chunk's pack instead of the node's own."""
    runtime = s6.RuntimeFactory(test)(name)
    runtime.create_lineage("main")
    registry = s6.registry()
    original = lineage_commit.ChunkNode
    previous = {}

    def node(*args, packet_ref=None, **kwargs):
        if legacy:
            return original(*args, **kwargs)
        if wrong_packet and previous.get("packet") is not None:
            packet_ref, previous["packet"] = previous["packet"], packet_ref
        else:
            previous["packet"] = packet_ref
        return original(*args, packet_ref=packet_ref, **kwargs)
    heads = []
    with mock.patch.object(lineage_commit, "ChunkNode", node):
        for index in range(count):
            heads.append(runtime.commit_node("main", f"chunk-{index:03d}", *s6.chunk(index), method=s6.method(),
                                             registry=registry, options=ArchiveOptions(encode_threads=1)).node_ref)
    return runtime, heads


class RecordedPackTests(unittest.TestCase):
    def test_packed_commits_record_their_pack_and_history_skips_the_records(self):
        runtime, heads = _commit_chain(self, "pack-ref")
        for head in heads:
            node = runtime.load_node(head)
            packets = {runtime.load_record(ref).packet_ref for _, ref in node.matrix_records}
            self.assertEqual(packets, {node.packet_ref})
            self.assertEqual(node.to_dict()["packet"], node.packet_ref)
            self.assertIn(node.packet_ref, runtime.store.references(head))
        restorer = LineageRestorer(runtime, SoftwarePool(), arena_bytes=1 << 26)
        with mock.patch.object(type(runtime), "load_record", side_effect=AssertionError("record loaded")):
            fast = restorer.history(heads[-1])
        audited = restorer.history(heads[-1], audit=True)
        self.assertEqual(fast, audited)
        self.assertEqual([item["packet"] for item in fast],
                         [runtime.load_node(head).packet_ref for head in heads])

    def test_nodes_without_a_recorded_pack_resolve_through_their_records(self):
        runtime, heads = _commit_chain(self, "pack-legacy", legacy=True)
        node = runtime.load_node(heads[-1])
        self.assertIsNone(node.packet_ref)
        self.assertNotIn("packet", node.to_dict())
        restorer = LineageRestorer(runtime, SoftwarePool(), arena_bytes=1 << 26)
        calls = []
        original = runtime.load_record

        def counting(ref):
            calls.append(ref)
            return original(ref)
        with mock.patch.object(runtime, "load_record", side_effect=counting):
            items = restorer.history(heads[-1])
        self.assertEqual(len(calls), sum(len(runtime.load_node(head).matrix_records) for head in heads))
        self.assertEqual([item["packet"] for item in items],
                         [runtime.load_record(runtime.load_node(head).matrix_records[0][1]).packet_ref
                          for head in heads])

    def test_audit_rejects_a_recorded_pack_that_its_records_do_not_name(self):
        runtime, heads = _commit_chain(self, "pack-wrong", wrong_packet=True)
        restorer = LineageRestorer(runtime, SoftwarePool(), arena_bytes=1 << 26)
        restorer.history(heads[-1])  # the fast path trusts the content-addressed node
        with self.assertRaises(IntegrityError):
            restorer.history(heads[-1], audit=True)
        with self.assertRaises(ConfigurationError):
            restorer.history(heads[-1], audit=1)

    def test_epoch_ranges_record_their_packs(self):
        runtime = s6.RuntimeFactory(self)("pack-epochs")
        runtime.create_lineage("segments")
        registry = s6.registry()
        segment = runtime.commit_segment("segments", "seg-0", *s6.chunk(0), method=s6.method(), registry=registry,
                                         epoch_id="s0", options=ArchiveOptions(encode_threads=1))
        epoch = runtime.load_epoch(segment.node.epoch_ref)
        (item,) = epoch.ranges
        self.assertEqual({runtime.load_record(ref).packet_ref for _, ref in item.records}, {item.packet_ref})
        self.assertIn(item.packet_ref, runtime.store.references(segment.node.epoch_ref))
        restorer = LineageRestorer(runtime, SoftwarePool(), arena_bytes=1 << 26)
        self.assertEqual(restorer.history(segment.node_ref), restorer.history(segment.node_ref, audit=True))

    def test_serialized_forms_are_unchanged_without_a_pack(self):
        runtime, heads = _commit_chain(self, "pack-format", count=1, legacy=True)
        node = runtime.load_node(heads[0])
        self.assertEqual(ChunkNode.from_dict(node.to_dict()), node)
        with self.assertRaises(ConfigurationError):
            ChunkNode.from_dict({**node.to_dict(), "packet": None})
        with self.assertRaises(ConfigurationError):
            ChunkNode.from_dict({**node.to_dict(), "packet": "not-a-ref"})
        item = TokenRangeRecords("c", 0, s6.TOKENS, s6.FRAMES, s6.HEIGHT, s6.WIDTH, node.matrix_records)
        self.assertNotIn("packet", item.to_dict())
        self.assertEqual(TokenRangeRecords.from_dict(item.to_dict()), item)
        recorded = TokenRangeRecords("c", 0, s6.TOKENS, s6.FRAMES, s6.HEIGHT, s6.WIDTH, node.matrix_records,
                                     packet_ref="manifest:" + "a" * 64)
        self.assertEqual(TokenRangeRecords.from_dict(recorded.to_dict()), recorded)
        self.assertEqual(recorded.to_dict()["packet"], "manifest:" + "a" * 64)


if __name__ == "__main__":
    unittest.main()


class StoreBatchTests(unittest.TestCase):
    """3./4. Batched store access for staging: one index query for sizes, parallel reads, batched manifests."""

    def setUp(self):
        self.runtime, self.heads = _commit_chain(self, "store-batch", count=2)
        self.store = self.runtime.store

    def test_get_many_matches_get_in_order_and_detects_corruption(self):
        node = self.runtime.load_node(self.heads[-1])
        packet = self.runtime.load_packet(node.packet_ref)
        blobs = list(packet["payloads"].values()) + [node.packet_ref, node.packet_ref]
        expected = [self.store.get(blob) for blob in blobs]
        self.assertEqual(self.store.get_many(blobs), expected)
        self.assertEqual(self.store.get_many(blobs, workers=1), expected)
        self.assertEqual(self.store.get_many([]), [])
        sizes = self.store.object_sizes(blobs)
        self.assertEqual(sizes, {blob: len(data) for blob, data in zip(blobs, expected)})
        missing = "blob:" + "0" * 64
        with self.assertRaises(IntegrityError):
            self.store.object_sizes([blobs[0], missing])
        with self.assertRaises(IntegrityError):
            self.store.get_many([blobs[0], missing])
        with self.assertRaises(ConfigurationError):
            self.store.get_many(blobs, workers=0)
        path = self.store.path_for(blobs[0])
        data = bytearray(path.read_bytes())
        data[len(data) // 2] ^= 1
        path.write_bytes(bytes(data))
        self.store.get_many(blobs[:1])  # reads follow verify_reads (off)
        with self.assertRaises(IntegrityError):
            self.store.get_many(blobs[:1], verify=True)

    def test_batched_manifests_and_versions_equal_the_single_loaders(self):
        node = self.runtime.load_node(self.heads[-1])
        refs = [node.packet_ref, self.heads[0], self.heads[-1]]
        self.assertEqual(self.store.get_manifests_with_references(refs),
                         [self.store.get_manifest_with_references(ref) for ref in refs])
        with self.assertRaises(IntegrityError):
            self.store.get_manifests_with_references(["blob:" + "0" * 64])
        versions = [binding.version_ref for _, binding in node.active_basis]
        fresh = type(self.runtime.basis_versions)(self.store, self.runtime.model)
        with self.store.transaction():
            batched = fresh.load_many(versions)
            singles = [type(self.runtime.basis_versions)(self.store, self.runtime.model).load(ref) for ref in versions]
        self.assertEqual(batched, singles)
        # prefetch_blocks reads (and checks) only blocks a prefix needs and the cache lacks.
        blocks: dict = {}
        with self.store.transaction():
            read = fresh.prefetch_blocks([(ref, fresh.load(ref).total_rank) for ref in versions], blocks)
            self.assertEqual(read, len(blocks))
            self.assertEqual(fresh.prefetch_blocks([(versions[0], 1)], blocks), 0)
            for ref in versions:
                self.assertTrue(torch.equal(fresh.assemble(ref, block_cache=blocks), fresh.assemble(ref)))
        block = next(iter(blocks))
        path = self.store.path_for(block)
        data = bytearray(path.read_bytes())
        data[0] ^= 1
        path.write_bytes(bytes(data))
        with self.assertRaises(IntegrityError), self.store.transaction():
            fresh.prefetch_blocks([(ref, fresh.load(ref).total_rank) for ref in versions], {})

    def test_packet_payload_sizes_take_one_index_query(self):
        node = self.runtime.load_node(self.heads[-1])
        calls = []
        original = self.store.object_sizes

        def counting(ids):
            ids = list(ids)
            calls.append(len(ids))
            return original(ids)
        self.runtime._lineage_packets.clear()
        with mock.patch.object(self.store, "object_sizes", side_effect=counting), \
                mock.patch.object(self.store, "object_size", side_effect=AssertionError("per-object size lookup")):
            with self.runtime._lock, self.store.transaction():
                self.runtime.load_packet(node.packet_ref)
        self.assertEqual(len(calls), 1)

    def test_drop_manifest_caches_forgets_every_parsed_manifest(self):
        restorer = LineageRestorer(self.runtime, SoftwarePool(), arena_bytes=1 << 30)
        history = restorer.history(self.heads[-1])
        packs = [self.runtime.node_pack(item["packet"]) for item in history]
        caches = (self.runtime._lineage_nodes, self.runtime._lineage_packets, self.runtime.basis_versions._cache)
        self.assertTrue(all(caches))
        self.runtime.load_record(self.runtime.load_node(self.heads[-1]).matrix_records[0][1])
        self.assertTrue(self.runtime._lineage_records)
        self.runtime.drop_manifest_caches()
        self.assertFalse(any((*caches, self.runtime._lineage_records, self.runtime._lineage_epochs)))
        self.assertEqual(restorer.history(self.heads[-1]), history)
        for item, pack in zip(history, packs):
            again = self.runtime.node_pack(item["packet"])
            self.assertEqual(again.metadata, pack.metadata)
            self.assertEqual(again.payloads, pack.payloads)


class _Ticket:
    def __init__(self, done=True, event=None):
        self._done, self._event, self.waited = done, event, 0

    def done(self):
        return self._done

    def event(self, timeout=None):
        self.waited += 1
        self._done = True
        return self._event


class DeviceAdoptionTests(unittest.TestCase):
    """2. Staged device versions are adopted by the device basis cache (no store reads, no copy)."""

    def setUp(self):
        self.runtime, self.heads = _commit_chain(self, "adopt-device", count=2)
        node = self.runtime.load_node(self.heads[-1])
        self.versions = [binding.version_ref for _, binding in node.active_basis]
        self.views = {ref: self.runtime.basis_versions.assemble(ref) for ref in self.versions}
        self.chains = {ref: [item for item, _ in self.runtime.basis_versions.chain(ref)] for ref in self.versions}

    def cache(self, budget=1 << 30):
        from reelcache.archive_restore import DeviceBasisCache
        return DeviceBasisCache(self.runtime, "cpu", budget)

    def test_adopted_versions_serve_requests_without_store_reads(self):
        cache = self.cache()
        for ref in self.versions:
            self.assertTrue(cache.adopt_device(ref, self.chains[ref], self.views[ref].clone()))
        self.assertFalse(cache.adopt_device(self.versions[0], self.chains[self.versions[0]],
                                            self.views[self.versions[0]].clone()))
        with mock.patch.object(self.runtime.store, "get", side_effect=AssertionError("store read")):
            views, tokens = cache.acquire_many([(ref, self.views[ref].shape[0]) for ref in self.versions])
        for ref, view in zip(self.versions, views):
            self.assertTrue(torch.equal(view, self.views[ref]))
        for token in tokens:
            cache.release(token)
        self.assertEqual(cache.stats()["misses"], 0)
        self.assertEqual(cache.stats()["adopted_device"], len(set(self.versions)))

    def test_rejects_views_that_are_not_the_stored_complete_version(self):
        cache = self.cache()
        ref = self.versions[0]
        with self.assertRaises(IntegrityError):
            cache.adopt_device(ref, self.chains[ref][:-1] + ["manifest:" + "b" * 64], self.views[ref].clone())
        with self.assertRaises(IntegrityError):
            cache.adopt_device(ref, self.chains[ref], self.views[ref][:-1].clone())
        with self.assertRaises(ConfigurationError):
            cache.adopt_device(ref, self.chains[ref], self.views[ref].float())
        small = self.cache(budget=1)
        self.assertFalse(small.adopt_device(ref, self.chains[ref], self.views[ref].clone()))
        self.assertEqual(small.stats()["adopt_skipped"], 1)

    def test_batched_adoption_validates_in_at_most_one_store_transaction(self):
        import fcntl
        items = [(ref, self.chains[ref], self.views[ref].clone(), None) for ref in dict.fromkeys(self.versions)]
        real, exclusive = fcntl.flock, []

        def counting(fd, operation):
            if operation == fcntl.LOCK_EX:  # one per outermost store transaction
                exclusive.append(fd)
            return real(fd, operation)
        self.runtime.drop_manifest_caches()
        cache = self.cache()
        with mock.patch("reelcache.store.fcntl.flock", side_effect=counting):
            self.assertEqual(cache.adopt_device_many(items), [True] * len(items))
        self.assertEqual(len(exclusive), 1)
        exclusive.clear()
        with mock.patch("reelcache.store.fcntl.flock", side_effect=counting):
            self.assertEqual(self.cache().adopt_device_many(items), [True] * len(items))
        self.assertEqual(exclusive, [])  # versions parsed already: no store transaction at all
        self.assertEqual(cache.adopt_device_many(items), [False] * len(items))  # already resident
        self.assertEqual(cache.adopt_device_many([]), [])
        fresh = self.cache()
        broken = list(items)
        broken[-1] = (broken[-1][0], broken[-1][1], broken[-1][2][:-1].clone(), None)
        with self.assertRaises(IntegrityError):  # one invalid item: nothing of the batch is adopted
            fresh.adopt_device_many(broken)
        self.assertEqual(fresh.stats().get("adopted_device", 0), 0)
        with self.assertRaises(ConfigurationError):
            fresh.adopt_device_many([items[0][:3]])

    def test_pending_readiness_is_handed_to_consumers_or_waited_for(self):
        cache = self.cache()
        ref = self.versions[0]
        ticket = _Ticket(done=False)
        cache.adopt_device(ref, self.chains[ref], self.views[ref].clone(), ticket)
        readiness: list = []
        _, tokens = cache.acquire_many([(ref, 1)], readiness=readiness)
        self.assertEqual(readiness, [ticket])
        self.assertEqual(ticket.waited, 0)
        cache.release(tokens[0])
        _, tokens = cache.acquire_many([(ref, 1)])  # no readiness list: waited for on the host
        self.assertEqual(ticket.waited, 1)
        cache.release(tokens[0])
        readiness = []
        _, tokens = cache.acquire_many([(ref, 1)], readiness=readiness)
        self.assertEqual(readiness, [None])  # a done ticket with no CUDA event is complete
        cache.release(tokens[0])


class StagingEngineHookTests(unittest.TestCase):
    """2. LineageRestorer(staging_engine=...) stages cold histories and adopts the result into its caches."""

    def test_staged_restore_equals_the_store_path_and_warms_the_caches(self):
        from reelcache import restore_kernels as rk
        from reelcache.restore_staging import StagingEngine
        from reelcache.types import MatrixKey
        runtime, heads = _commit_chain(self, "staging-hook", count=3)

        def destination_factory():
            caches = {}

            def destination(node_id, key, tile):
                key = key if isinstance(key, MatrixKey) else key[0]
                if key not in caches:
                    caches[key] = torch.zeros((1, 3 * s6.TOKENS, 1, 128), dtype=torch.float32)
                return rk.NativeView(caches[key], "BTHD", token_offset=tile.token_start)
            return caches, destination
        reference, destination = destination_factory()
        LineageRestorer(runtime, SoftwarePool(), arena_bytes=1 << 30).restore(heads[-1], destination)
        engine = StagingEngine("cpu", threads=2, prepare_threads=2, pinned_bytes=1 << 22)
        self.addCleanup(engine.close)
        restorer = LineageRestorer(runtime, SoftwarePool(), arena_bytes=1 << 30, staging_engine=engine)
        caches, destination = destination_factory()
        from reelcache.archive_restore import DeviceBasisCache
        batches = []
        original = DeviceBasisCache.adopt_device_many

        def batched(cache, items):
            batches.append(len(items))
            return original(cache, items)
        # The staged history's versions are adopted in one batch (not one validation per version).
        with mock.patch.object(DeviceBasisCache, "adopt_device", side_effect=AssertionError("per-version adoption")), \
                mock.patch.object(DeviceBasisCache, "adopt_device_many", autospec=True, side_effect=batched):
            restorer.restore(heads[-1], destination)
        self.assertEqual(len(batches), 1)
        self.assertGreater(batches[0], 0)
        self.assertEqual(engine.stats()["histories"], 1)
        self.assertIsNotNone(restorer.last_staging)
        for key, value in reference.items():
            self.assertTrue(torch.equal(caches[key], value), key)
        history = restorer.history(heads[-1])
        self.assertTrue(all(restorer.prepared.contains(item["packet"]) for item in history))
        self.assertGreater(restorer.bases.stats()["adopted_device"], 0)
        # Warm now: the next restore uses the caches, not the engine.
        caches, destination = destination_factory()
        restorer.restore(heads[-1], destination)
        self.assertEqual(engine.stats()["histories"], 1)
        for key, value in reference.items():
            self.assertTrue(torch.equal(caches[key], value), key)
        with self.assertRaises(ConfigurationError):
            LineageRestorer(runtime, SoftwarePool(), arena_bytes=1 << 30, staging_engine=object())


class RecordedAccessUnitTests(unittest.TestCase):
    """5. Node packs record each picture's access-unit length per tile (no byte-range scan on restore)."""

    def test_recorded_access_units_equal_the_stream_scan(self):
        from reelcache import _hevc
        from reelcache.node_packets import recorded_access_units
        runtime, heads = _commit_chain(self, "access-units", count=1)
        packet = runtime.load_packet(runtime.load_node(heads[0]).packet_ref)
        metadata, metrics = packet["metadata"], packet["metrics"]
        self.assertIn("tile_access_units", metrics)
        stream = runtime.store.get(packet["payloads"]["stream.hevc"])
        units = _hevc.all_intra_access_units(stream, metadata["canvas"]["pictures"])
        spans = recorded_access_units(metadata, metrics)
        self.assertEqual(len(spans), len(metadata["tiles"]))
        for tile, tile_spans in zip(metadata["tiles"], spans):
            first, count = tile["pictures"]
            self.assertEqual(tile_spans, [(unit["offset"], unit["length"]) for unit in units[first:first + count]])
        self.assertIsNone(recorded_access_units(metadata, {}))
        self.assertIsNone(recorded_access_units(metadata, None))
        tampered = [list(lengths) for lengths in metrics["tile_access_units"]]
        tampered[0][0] += 1
        with self.assertRaises(IntegrityError):
            recorded_access_units(metadata, {"tile_access_units": tampered})
        with self.assertRaises(IntegrityError):
            recorded_access_units(metadata, {"tile_access_units": metrics["tile_access_units"][:-1]})
