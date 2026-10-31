"""Step 6: packed latest-method lineage commits (node packs referencing basis versions), streams, restart."""

from __future__ import annotations

import unittest

import torch

from reelcache.codecs import PACK_SCHEMA, decode_packed
from reelcache.errors import ConfigurationError, PendingError, StateError, UnsupportedError
from reelcache.lineage_commit import ArchiveOptions, source_fingerprint
from reelcache.manager import KVRuntime
from reelcache.types import CodecSpec, EnergyTargets, MatrixKey
from reelcache.rank_method import RankMethodConfig
from tests.step6_support import KEYS, MODEL, TOKENS, RuntimeFactory, chunk, method, node_content, registry


class PackedLineageTests(unittest.TestCase):
    def setUp(self):
        self.runtimes = RuntimeFactory(self)

    def commit_chain(self, runtime, count=3, *, options=None, lag=0, branch="main", streams=()):
        if branch not in runtime.list_lineages():
            runtime.create_lineage(branch, streams=streams)
        reg = registry()
        commits = []
        for index in range(count):
            matrices, contexts = chunk(index, streams=streams)
            commits.append(runtime.commit_node(branch, f"chunk-{index:03d}", matrices, contexts, method=method(lag),
                                               registry=reg, options=options))
        return commits, reg

    def test_one_node_pack_per_chunk_with_external_versioned_bases(self):
        runtime = self.runtimes()
        commits, _ = self.commit_chain(runtime, options=ArchiveOptions(measure_post_codec=True, encode_threads=2))
        decisions = set()
        for commit in commits:
            node = commit.node
            packets = set()
            for index, (entry, record_ref) in enumerate(node.matrix_records):
                record = runtime.load_record(record_ref)
                packets.add(record.packet_ref)
                packet = runtime.load_packet(record.packet_ref)
                self.assertEqual(packet["basis_storage"], "external_version_prefix_tiles")
                self.assertEqual(packet["metadata"]["schema"], PACK_SCHEMA)
                self.assertFalse(any(name.startswith("basis.") for name in packet["payloads"]))
                self.assertEqual(packet["tile_bases"][record.packet_tile],
                                 {"basis_version": record.basis_version_ref, "rank_view": record.rank_view})
                self.assertGreaterEqual(record.energy.stored_factor_energy, record.energy.target)
                self.assertIsNotNone(record.energy.post_codec_relative_error)
                decisions.add(record.decision["decision"])
            self.assertEqual(len(packets), 1)
            canvas = runtime.load_packet(packets.pop())["metadata"]["canvas"]
            self.assertGreaterEqual(min(canvas["width"], canvas["height"]), 160)
        self.assertIn("fresh", decisions)
        self.assertTrue(decisions & {"share", "augment", "refit"})
        head = commits[-1].node_ref
        storage = runtime.node_storage(head)
        self.assertEqual(storage["packets"], 3)
        for key in KEYS:
            with runtime.acquire_node_history(head, key, dtype=torch.float32) as lease:
                self.assertEqual(tuple(lease.tensor.shape), (3 * TOKENS, 128))
                source = torch.cat([chunk(index)[0][key].float() for index in range(3)])
                error = float((lease.tensor - source).norm() / source.norm())
                self.assertLess(error, 0.4)
        pack = runtime.matrix_packet(commits[0].node.matrix_records[0][1])
        decoded = decode_packed(pack, keys=[KEYS[0]], dtype=torch.float32)[KEYS[0]]
        self.assertEqual(tuple(decoded.shape), (TOKENS, 128))

    def test_reuse_shares_inherited_rows_across_node_packs(self):
        runtime = self.runtimes()
        commits, _ = self.commit_chain(runtime)
        reused = 0
        for previous, commit in zip(commits, commits[1:]):
            for entry, record_ref in commit.node.matrix_records:
                record = runtime.load_record(record_ref, telemetry=True)
                parent = previous.node.binding(entry)
                if record.decision["decision"] == "share":
                    self.assertEqual(record.basis_version_ref, parent.version_ref)
                    self.assertEqual(record.decision["work"]["counters"]["candidate_fit_count"], 0)
                    reused += 1
                elif record.decision["decision"] == "augment":
                    version = runtime.load_basis_version(record.basis_version_ref)
                    self.assertEqual(version.parent_ref, parent.version_ref)
                    reused += 1
        self.assertGreater(reused, 0)

    def test_per_matrix_reference_path_and_option_validation(self):
        from reelcache.archive_restore import SoftwareLineageRestorer
        runtime = self.runtimes()
        commits, _ = self.commit_chain(runtime, count=2, options=ArchiveOptions(packing="per_matrix",
                                                                             orthogonalizer="householder"))
        record = runtime.load_record(commits[1].node.matrix_records[0][1])
        self.assertIsNone(record.packet_tile)
        self.assertEqual(runtime.load_packet(record.packet_ref)["basis_storage"], "external_version_prefix")
        # Per-matrix (Step 4 / v1-style) records keep the per-matrix decode path; the packed device path refuses.
        with runtime.acquire_node_history(commits[1].node_ref, KEYS[0], dtype=torch.float32) as lease:
            self.assertEqual(tuple(lease.tensor.shape), (2 * TOKENS, 128))
        with self.assertRaises(UnsupportedError):
            SoftwareLineageRestorer(runtime).history(commits[1].node_ref)
        svd = RankMethodConfig(EnergyTargets("t", 0.95, 0.9, True), CodecSpec(kind="svd", rank=1))
        runtime.create_lineage("svd")
        matrices, contexts = chunk(0)
        commit = runtime.commit_node("svd", "c0", matrices, contexts, method=svd, registry=registry())
        self.assertIsNone(runtime.load_record(commit.node.matrix_records[0][1]).packet_tile)
        with self.assertRaises(UnsupportedError):
            runtime.create_lineage("svd2")
            runtime.commit_node("svd2", "c0", matrices, contexts, method=svd, registry=registry(),
                                options=ArchiveOptions(packing="node"))
        with self.assertRaises(ConfigurationError):
            ArchiveOptions(measure_post_codec=True, validate_decode=False)
        with self.assertRaises(ConfigurationError):
            ArchiveOptions(canvas_ladder=(64, 32))

    def test_cfg_streams_commit_atomically_and_restore_per_stream(self):
        runtime = self.runtimes()
        streams = ("cond", "uncond")
        commits, _ = self.commit_chain(runtime, count=2, streams=streams)
        node = commits[-1].node
        self.assertEqual(node.streams, streams)
        self.assertEqual({entry for entry, _ in node.matrix_records},
                         {(key, stream) for key in KEYS for stream in streams})
        packets = {runtime.load_record(ref).packet_ref for _, ref in node.matrix_records}
        self.assertEqual(len(packets), 1)
        for stream in streams:
            with runtime.acquire_node_history(commits[-1].node_ref, KEYS[1], stream=stream,
                                              dtype=torch.float32) as lease:
                self.assertEqual(tuple(lease.tensor.shape), (2 * TOKENS, 128))
        with self.assertRaises(ConfigurationError):
            runtime.acquire_node_history(commits[-1].node_ref, KEYS[1])
        matrices, contexts = chunk(2, streams=streams)
        partial = {entry: value for entry, value in matrices.items() if entry[1] == "cond"}
        with self.assertRaises(ConfigurationError):
            runtime.commit_node("main", "chunk-002", partial, {e: contexts[e] for e in partial}, method=method(),
                                registry=registry())
        self.assertEqual(runtime.lineage_head("main"), commits[-1].node_ref)

    def test_recache_and_segment_epochs_use_node_packs(self):
        runtime = self.runtimes()
        commits, reg = self.commit_chain(runtime, count=2)
        matrices, contexts = chunk(5, token_start=TOKENS)
        recache = runtime.commit_recache("main", "switch-b", [("chunk-001", matrices, contexts)], method=method(),
                                         registry=reg, epoch_id="e1")
        pieces = runtime.resolve_history(recache.node_ref)
        self.assertEqual([piece.entry.source for piece in pieces], ["chunk", "epoch"])
        record = runtime.load_record(pieces[1].record(KEYS[0]))
        self.assertIsNotNone(record.packet_tile)
        runtime.create_lineage("segments")
        matrices, contexts = chunk(0)
        segment = runtime.commit_segment("segments", "seg-0", matrices, contexts, method=method(), registry=reg,
                                         epoch_id="s0")
        self.assertEqual(runtime.resolve_history(segment.node_ref)[0].entry.source, "epoch")

    def test_rate_lag_uses_the_grandparents_measured_rates(self):
        runtime = self.runtimes()
        commits, _ = self.commit_chain(runtime, count=3, lag=1)
        report = runtime.node_commit_report(commits[2].node_ref)
        self.assertEqual(report["rate_lag"], 1)
        estimates = [row["estimate"] for row in commits[2].report["records"] if row["estimate"]]
        rates0 = {str(entry): binding.coefficient_rate for entry, binding in commits[0].node.active_basis}
        for row in commits[2].report["records"]:
            if row["estimate"] and row["estimate"]["augmented"]["estimate_type"] == "measured_coefficient_rate":
                key = MatrixKey.from_dict(row["key"])
                self.assertEqual(row["estimate"]["augmented"]["coefficient_rate"], rates0[str(key)])
        self.assertTrue(estimates)
        with self.assertRaises(UnsupportedError):
            runtime.create_lineage("lagged-reference")
            matrices, contexts = chunk(0)
            runtime.commit_node("lagged-reference", "c0", matrices, contexts, method=method(1), registry=registry(),
                                options=ArchiveOptions(packing="per_matrix"))

    def test_garbage_collection_leases_and_restart_recovery(self):
        runtime = self.runtimes()
        commits, reg = self.commit_chain(runtime, count=2)
        runtime.fork_lineage(commits[0].node_ref, "fork")
        matrices, contexts = chunk(9, token_start=TOKENS)
        fork = runtime.commit_node("fork", "chunk-f1", matrices, contexts, method=method(), registry=reg)
        runtime.delete_lineage("main")
        runtime.store.collect_garbage()
        with runtime.acquire_node_history(fork.node_ref, KEYS[0], dtype=torch.float32) as lease:
            self.assertEqual(tuple(lease.tensor.shape), (2 * TOKENS, 128))
        root = runtime.store.root
        expected = node_content(runtime, fork.node_ref)
        runtime.close()
        reopened = KVRuntime(root, MODEL)
        self.addCleanup(reopened.close)
        self.assertEqual(reopened.lineage_head("fork"), fork.node_ref)
        self.assertEqual(node_content(reopened, fork.node_ref), expected)
        matrices, contexts = chunk(10, token_start=2 * TOKENS)
        resumed = reopened.commit_node("fork", "chunk-f2", matrices, contexts, method=method(), registry=reg)
        self.assertEqual(resumed.node.frontier, 3 * TOKENS)
        with self.assertRaises(StateError):
            reopened.lineage_head("main")

    def test_source_fingerprint_is_exact_and_device_independent(self):
        value = chunk(0)[0][KEYS[0]]
        self.assertEqual(source_fingerprint(value), source_fingerprint(value.clone()))
        changed = value.clone()
        changed[3, 5] = changed[3, 5] + 1
        self.assertNotEqual(source_fingerprint(value), source_fingerprint(changed))
        self.assertNotEqual(source_fingerprint(value.float()), source_fingerprint(value))
        if torch.cuda.is_available():
            self.assertEqual(source_fingerprint(value.cuda()), source_fingerprint(value))

    def test_device_basis_cache_batches_prefixes_and_never_leaks_pins(self):
        from reelcache.archive_restore import DeviceBasisCache
        from reelcache.errors import CapacityError
        runtime = self.runtimes()
        commits, _ = self.commit_chain(runtime, count=3)
        requests = []
        for commit in commits:
            for _, record_ref in commit.node.matrix_records:
                record = runtime.load_record(record_ref)
                requests.append((record.basis_version_ref, record.rank_view))
        distinct = {ref for ref, _ in requests}
        with runtime.store.transaction():
            expected = [runtime.basis_versions.assemble(ref, rank) for ref, rank in requests]
            chains = {ref: [item for item, _ in runtime.basis_versions.chain(ref)] for ref in distinct}
        # Some versions extend others (augment chains); a batch serves ancestors as prefixes.
        ancestors = {ref for ref in distinct if any(ref in chain[:-1] for chain in chains.values())}
        self.assertTrue(ancestors)
        cache = DeviceBasisCache(runtime, "cpu", 1 << 30)
        views, tokens = cache.acquire_many(requests)
        for view, want in zip(views, expected):
            self.assertTrue(torch.equal(view, want))
        stats = cache.stats()
        self.assertEqual(stats["uploads"], len(distinct - ancestors))
        self.assertEqual(stats["misses"], len(distinct))
        self.assertEqual(stats["pinned"], stats["versions"])
        with self.assertRaises(Exception):
            cache.clear()  # held by the batch
        for token in tokens:
            cache.release(token)
        self.assertEqual(cache.stats()["pinned"], 0)
        with self.assertRaises(StateError):
            cache.release(tokens[0])
        # A second batch is served entirely from residency (exact or descendant-prefix hits).
        views, tokens = cache.acquire_many(requests)
        stats = cache.stats()
        self.assertEqual(stats["hits"] + stats["prefix_hits"], len(requests))
        self.assertEqual(stats["uploads"], len(distinct - ancestors))
        for token in tokens:
            cache.release(token)
        # Over budget: CapacityError and no pin survives; resident unpinned versions are evicted LRU.
        largest = max(view.numel() * 2 for view in expected)
        small = DeviceBasisCache(runtime, "cpu", largest)
        with self.assertRaises(CapacityError):
            small.acquire_many(requests)
        self.assertEqual(small.stats()["pinned"], 0)
        single, token = small.acquire(*requests[0])
        self.assertTrue(torch.equal(single, expected[0]))
        small.release(token)
        other = next(index for index, (ref, _) in enumerate(requests)
                     if ref not in chains[requests[0][0]] and requests[0][0] not in chains[ref])
        view, token = small.acquire(*requests[other])
        self.assertTrue(torch.equal(view, expected[other]))
        self.assertGreaterEqual(small.stats()["evictions"], 1)
        small.release(token)
        small.clear()
        cache.clear()
        self.assertEqual(cache.stats()["resident_bytes"], 0)
        # Seal-time adoption in commit order: a descendant absorbs its resident ancestors (prefix views).
        adopted = DeviceBasisCache(runtime, "cpu", 1 << 30)
        with runtime.store.transaction():
            full = {ref: runtime.basis_versions.assemble(ref) for ref in distinct}
        for ref in sorted(distinct, key=lambda item: len(chains[item])):
            adopted.adopt([(ref, chains[ref], full[ref])])
        stats = adopted.stats()
        self.assertEqual(stats["adopted"], len(distinct))
        self.assertEqual(stats["absorbed"], len(ancestors))
        self.assertEqual(stats["versions"], len(distinct - ancestors))
        self.assertEqual(stats["resident_bytes"], sum(full[ref].numel() * 2 for ref in distinct - ancestors))
        views, tokens = adopted.acquire_many(requests)
        self.assertEqual(adopted.stats()["uploads"], 0)
        for view, want in zip(views, expected):
            self.assertTrue(torch.equal(view, want))
        for token in tokens:
            adopted.release(token)
        self.assertEqual(adopted.adopt([(ref, chains[ref], full[ref]) for ref in distinct]), 0)


if __name__ == "__main__":
    unittest.main()
