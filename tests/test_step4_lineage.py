"""Step 4: lineage nodes, immutable basis versions, GC/leases, forks, restore and cache epochs."""

import hashlib
import unittest
from unittest import mock

import torch

from reelcache.basis_versions import attach_basis, rows_to_bytes
from reelcache.errors import ConfigurationError, IntegrityError, StateError, UnsupportedError
from reelcache.manager import KVRuntime
from reelcache.predictor_registry import RankPredictorRegistry
from reelcache.rank_method import RankMethodConfig
from reelcache.types import CodecSpec, EnergyTargets, MatrixKey, ModelIdentity, TensorContext
from tests.core_support import CoreWorkspaceTest

# Basis rows (2*C bytes) cost more than FP32 coefficient columns (4*N bytes) here, so the
# paper's byte rule can prefer augmentation; with N >> C it refits (see test_step4_rank_method).
CHANNELS = 256
MODEL = ModelIdentity("fixture", "lineage-v1", 1, CHANNELS, "tokens_channels", "float32")
KEYS = (MatrixKey(0, "K"), MatrixKey(0, "V"))
FRAMES, HEIGHT, WIDTH = 3, 4, 8
TOKENS = FRAMES * HEIGHT * WIDTH
SPECTRUM = torch.logspace(0, -1.5, TOKENS, dtype=torch.float64)  # r98 = 54, r95 = 41 of 96 directions


def orthonormal(seed):
    generator = torch.Generator().manual_seed(seed)
    return torch.linalg.qr(torch.randn(CHANNELS, TOKENS, generator=generator, dtype=torch.float64))[0].T


def rotate(basis, rows, seed):
    mixed = basis.clone()
    mixed[rows] = orthonormal(seed)[rows]
    return torch.linalg.qr(mixed.T)[0].T


def chunk(seed, start, bases, *, spectrum=SPECTRUM, token_start=None):
    matrices, contexts = {}, {}
    for index, key in enumerate(KEYS):
        generator = torch.Generator().manual_seed(seed * 10 + index)
        left = torch.linalg.qr(torch.randn(TOKENS, TOKENS, generator=generator, dtype=torch.float64))[0]
        basis = bases[key] if isinstance(bases, dict) else bases
        matrices[key] = ((left * spectrum) @ basis).float()
        contexts[key] = TensorContext(MODEL, key, start if token_start is None else token_start, FRAMES, HEIGHT,
                                      WIDTH, "pre_rope" if key.target == "K" else "none")
    return matrices, contexts


def method(**kwargs):
    kwargs.setdefault("coefficient_bytes_proxy", 0.05)
    return RankMethodConfig(EnergyTargets("fixture", 0.98, 0.95, True), CodecSpec("svd", rank=1), **kwargs)


def energy(source, restored):
    return 1 - float((source.double() - restored.double()).square().sum() / source.double().square().sum())


class LineageTestCase(CoreWorkspaceTest):
    def setUp(self):
        super().setUp()
        self.registry = RankPredictorRegistry("lineage-tests")
        self.runtime = KVRuntime(self.workspace / "store", MODEL, max_resident_bytes=1 << 28)
        self.addCleanup(self._close)

    def _close(self):
        if not self.runtime._closed:
            self.runtime.close()

    def commit(self, branch, chunk_id, data, **kwargs):
        matrices, contexts = data
        kwargs.setdefault("method", method())
        return self.runtime.commit_node(branch, chunk_id, matrices, contexts, registry=self.registry, **kwargs)

    def record(self, result, key, index=0):
        return self.runtime.load_record(result.records[index][1][key], telemetry=True)

    def decisions(self, result):
        return {row["key"]["target"]: row["decision"] for row in result.report["records"]}


class BasisVersionLineageTests(LineageTestCase):
    def test_augmentation_chain_restore_and_c3_tests_c2_extension(self):
        base = orthonormal(1)
        mixed = rotate(base, slice(6, 12), 2)
        self.runtime.create_lineage("main")
        c1_data, c2_data, c3_data = chunk(1, 0, base), chunk(2, TOKENS, mixed), chunk(3, 2 * TOKENS, mixed)
        c1 = self.commit("main", "c1", c1_data)
        self.assertEqual(self.decisions(c1), {"K": "fresh", "V": "fresh"})
        v1 = {key: self.record(c1, key).basis_version_ref for key in KEYS}
        v1_rows = {key: rows_to_bytes(self.runtime.basis_versions.assemble(v1[key])) for key in KEYS}
        v1_manifest = {key: self.runtime.store.get(v1[key]) for key in KEYS}
        c2 = self.commit("main", "c2", c2_data)
        self.assertEqual(self.decisions(c2)["K"], "augment")
        key = KEYS[0]
        r2 = self.record(c2, key)
        v2 = self.runtime.load_basis_version(r2.basis_version_ref)
        self.assertEqual(v2.parent_ref, v1[key])
        self.assertEqual(v2.depth, 1)
        self.assertEqual(r2.rank_view, v2.total_rank)
        view = self.runtime.basis_versions.assemble(r2.basis_version_ref)
        self.assertEqual(rows_to_bytes(view[:v2.parent_rank]), v1_rows[key])
        self.assertEqual(self.runtime.store.get(v1[key]), v1_manifest[key])
        self.assertEqual(c2.node.binding(key).version_ref, r2.basis_version_ref)
        c3 = self.commit("main", "c3", c3_data)
        r3 = self.record(c3, key)
        self.assertEqual(self.decisions(c3)["K"], "share")
        self.assertEqual(r3.basis_version_ref, r2.basis_version_ref)
        self.assertGreater(r3.rank_view, v2.parent_rank)
        counters = r3.decision["work"]["counters"]
        self.assertEqual((counters["candidate_fit_count"], counters["eigh_count"]), (0, 0))
        for result, data in ((c1, c1_data), (c2, c2_data), (c3, c3_data)):
            for item in KEYS:
                record = self.record(result, item)
                target = 0.98 if item.target == "K" else 0.95
                self.assertGreaterEqual(record.energy.stored_factor_energy, target)
                self.assertGreaterEqual(record.energy.ideal_subspace_energy, record.energy.stored_factor_energy - 1e-9)
                self.assertGreaterEqual(energy(data[0][item], self.runtime.decoder(
                    self.runtime.matrix_packet(result.records[0][1][item]))), target)
        sequence = self.registry.sequence
        with self.runtime.acquire_node_history(c3.node_ref, key) as lease:
            restored = lease.tensor.clone()
        self.assertEqual(tuple(restored.shape), (3 * TOKENS, CHANNELS))
        for index, (result, data) in enumerate(((c1, c1_data), (c2, c2_data), (c3, c3_data))):
            decoded = self.runtime.decoder(self.runtime.matrix_packet(result.records[0][1][key]))
            self.assertTrue(torch.equal(restored[index * TOKENS:(index + 1) * TOKENS], decoded))
        self.assertEqual(self.registry.sequence, sequence)
        self.assertEqual(self.runtime.active_leases, 0)
        storage = self.runtime.node_storage(c3.node_ref)
        self.assertEqual(storage["records"], 6)
        blocks = {self.runtime.basis_versions.load(self.record(result, item).basis_version_ref).added_rows_ref
                  for result in (c1, c2, c3) for item in KEYS}
        self.assertEqual(storage["unique_basis_blocks"], len(blocks))
        packet = self.runtime.matrix_packet(c3.records[0][1][key])
        with self.assertRaises(IntegrityError):
            attach_basis(packet.metadata, packet.payloads, bytes(len(packet.basis)), rank_view=r3.rank_view,
                         channels=CHANNELS)

    def test_records_bind_predictor_revisions_and_estimates(self):
        self.runtime.create_lineage("main")
        c1 = self.commit("main", "c1", chunk(1, 0, orthonormal(1)))
        record = self.record(c1, KEYS[0])
        self.assertIn("reelcache-rank-predictor:lineage-tests:", record.predictor_revision_ref)
        prediction = record.decision["predictions"][0]
        self.assertEqual(prediction["predictor"]["state"]["revision"], 0)
        self.assertEqual(c1.report["registry_sequence_before"], 0)
        self.assertEqual(c1.report["registry_sequence_after"], sum(c1.observations_applied))
        c2 = self.commit("main", "c2", chunk(2, TOKENS, rotate(orthonormal(1), slice(6, 12), 2)))
        estimate = self.record(c2, KEYS[0]).decision["estimate"]
        self.assertEqual(estimate["augmented"]["estimate_type"], "measured_coefficient_rate")
        self.assertEqual(estimate["r_fresh"], self.record(c1, KEYS[0]).rank_view)
        comparison = self.record(c2, KEYS[0]).decision["estimate_vs_actual"]
        self.assertGreater(comparison["actual_incremental_bytes"], 0)
        report = self.runtime.node_commit_report(c2.node_ref)
        self.assertIn("augment", report["decisions"])
        self.assertGreater(report["plans"][0]["stages"]["certify"], 0)

    def test_hevc_records_measure_post_codec_error(self):
        from reelcache.codecs import decode_packed
        from reelcache.lineage_commit import ArchiveOptions
        hevc = RankMethodConfig(EnergyTargets("fixture", 0.98, 0.95, True), CodecSpec("svd_hevc", rank=1))
        # Step 4 per-matrix reference packets and Step 6 node packs (post-codec error measured on request).
        for branch, options in (("per-matrix", ArchiveOptions(packing="per_matrix", orthogonalizer="householder")),
                                ("node", ArchiveOptions(measure_post_codec=True))):
            self.runtime.create_lineage(branch)
            result = self.commit(branch, "c1", chunk(1, 0, orthonormal(1)), method=hevc, options=options)
            record = self.record(result, KEYS[0])
            self.assertEqual(record.codec.kind, "svd_hevc")
            self.assertIs(record.codec.inter_prediction, False)
            self.assertIsNotNone(record.energy.post_codec_relative_error)
            self.assertGreater(record.energy.post_codec_relative_error, 0)
            packet = self.runtime.matrix_packet(result.records[0][1][KEYS[0]])
            if record.packet_tile is None:
                decoded = self.runtime.decoder(packet)
            else:
                decoded = decode_packed(packet, keys=[record.entry])[record.entry]
            self.assertEqual(tuple(decoded.shape), (TOKENS, CHANNELS))


class ForkAndStorageTests(LineageTestCase):
    def test_forks_extend_independently_and_share_rows_once(self):
        base = orthonormal(1)
        self.runtime.create_lineage("main")
        n1 = self.commit("main", "c1", chunk(1, 0, base))
        n2 = self.commit("main", "c2", chunk(2, TOKENS, rotate(base, slice(6, 12), 2)))
        self.runtime.fork_lineage(n1.node_ref, "alt")
        alt2 = self.commit("alt", "c2", chunk(3, TOKENS, rotate(base, slice(20, 26), 3)))
        key = KEYS[0]
        v1 = n1.node.binding(key).version_ref
        main_v2, alt_v2 = n2.node.binding(key).version_ref, alt2.node.binding(key).version_ref
        self.assertNotEqual(main_v2, alt_v2)
        self.assertEqual(self.runtime.basis_versions.load(main_v2).parent_ref, v1)
        self.assertEqual(self.runtime.basis_versions.load(alt_v2).parent_ref, v1)
        self.assertEqual(self.runtime.lineage_head("main"), n2.node_ref)
        self.assertEqual(self.runtime.load_node(n1.node_ref).binding(key).version_ref, v1)
        main_blocks = self.runtime.basis_versions.blocks(main_v2)
        alt_blocks = self.runtime.basis_versions.blocks(alt_v2)
        self.assertEqual(main_blocks[0], alt_blocks[0])
        self.assertNotEqual(main_blocks[1], alt_blocks[1])
        shared = self.runtime.store.path_for(main_blocks[0])
        self.assertTrue(shared.is_file())
        n3 = self.commit("main", "c3", chunk(4, 2 * TOKENS, rotate(base, slice(6, 12), 2)))
        self.assertEqual(self.record(n3, key).basis_version_ref, main_v2)

    def test_gc_and_leases_cover_basis_parents_and_blocks(self):
        base = orthonormal(1)
        self.runtime.create_lineage("main")
        n1 = self.commit("main", "c1", chunk(1, 0, base))
        self.runtime.fork_lineage(n1.node_ref, "alt")
        alt2 = self.commit("alt", "c2", chunk(3, TOKENS, rotate(base, slice(20, 26), 3)))
        key = KEYS[0]
        alt_blocks = self.runtime.basis_versions.blocks(alt2.node.binding(key).version_ref)
        self.runtime.delete_lineage("alt")
        self.runtime.collect_garbage()
        self.assertFalse(self.runtime.store.path_for(alt_blocks[1]).exists())
        self.assertTrue(self.runtime.store.path_for(alt_blocks[0]).exists())
        lease = self.runtime.acquire_node_history(n1.node_ref, key)
        self.runtime.delete_lineage("main")
        self.runtime.collect_garbage()
        self.assertTrue(self.runtime.store.path_for(alt_blocks[0]).exists())
        self.assertTrue(lease.tensor.isfinite().all())
        lease.release()
        self.runtime.collect_garbage()
        self.assertFalse(self.runtime.store.path_for(alt_blocks[0]).exists())

    def test_reopen_revalidates_lineages(self):
        self.runtime.create_lineage("main")
        n1 = self.commit("main", "c1", chunk(1, 0, orthonormal(1)))
        pieces = [piece.to_dict() for piece in self.runtime.resolve_history(n1.node_ref)]
        record = self.record(n1, KEYS[1])
        self.runtime.close()
        self.runtime = KVRuntime(self.workspace / "store", MODEL, verify_reads=True)
        self.assertEqual(self.runtime.list_lineages(), {"main": n1.node_ref})
        self.assertEqual(self.runtime.load_node(n1.node_ref), n1.node)
        self.assertEqual(self.runtime.load_record(n1.records[0][1][KEYS[1]], telemetry=True), record)
        self.assertEqual([piece.to_dict() for piece in self.runtime.resolve_history(n1.node_ref)], pieces)

    def test_retry_of_the_same_source_is_idempotent(self):
        base = orthonormal(1)
        self.runtime.create_lineage("main")
        n1 = self.commit("main", "c1", chunk(1, 0, base))
        self.runtime.fork_lineage(n1.node_ref, "attempt-a")
        self.runtime.fork_lineage(n1.node_ref, "attempt-b")
        data = chunk(2, TOKENS, orthonormal(7))
        first = self.commit("attempt-a", "c2", data)
        sequence = self.registry.sequence
        self.assertGreater(len(first.observations_applied), 0)
        second = self.commit("attempt-b", "c2", data)
        self.assertEqual(self.registry.sequence, sequence)
        self.assertEqual(len(second.observations_applied), len(first.observations_applied))
        self.assertTrue(all(not applied for applied in second.observations_applied))
        for key in KEYS:
            # Same verified representation; the records differ only in the predictor revision they bind.
            one, two = self.record(first, key), self.record(second, key)
            self.assertEqual((one.basis_version_ref, one.rank_view), (two.basis_version_ref, two.rank_view))
            self.assertEqual(self.runtime.load_packet(one.packet_ref)["payloads"],
                             self.runtime.load_packet(two.packet_ref)["payloads"])

    def test_failed_publication_keeps_head_registry_and_collects_staging(self):
        self.runtime.create_lineage("main")
        head = self.runtime.lineage_head("main")
        before = self.runtime.store.stats()["object_count"]
        with mock.patch.object(KVRuntime, "_publish", side_effect=RuntimeError("injected publication failure")):
            with self.assertRaises(RuntimeError):
                self.commit("main", "c1", chunk(1, 0, orthonormal(1)))
        self.assertEqual(self.runtime.lineage_head("main"), head)
        self.assertEqual(self.registry.sequence, 0)
        self.assertEqual(self.runtime.store.stats()["object_count"], before)
        self.commit("main", "c1", chunk(1, 0, orthonormal(1)))

    def test_commit_validation_and_rewind(self):
        self.runtime.create_lineage("main")
        with self.assertRaises(StateError):
            self.runtime.create_lineage("main")
        matrices, contexts = chunk(1, 0, orthonormal(1))
        post = dict(contexts)
        post[KEYS[0]] = TensorContext(MODEL, KEYS[0], 0, FRAMES, HEIGHT, WIDTH, "post_rope")
        with self.assertRaises(ConfigurationError):
            self.runtime.commit_node("main", "c1", matrices, post, method=method(), registry=self.registry)
        with self.assertRaises(ConfigurationError):
            self.commit("main", "c1", chunk(1, TOKENS, orthonormal(1)))
        n1 = self.commit("main", "c1", (matrices, contexts))
        with self.assertRaises(StateError):
            self.commit("main", "c1", chunk(2, TOKENS, orthonormal(1)))
        n2 = self.commit("main", "c2", chunk(2, TOKENS, orthonormal(1)))
        self.runtime.rewind_lineage("main", n1.node_ref)
        self.assertEqual(self.runtime.lineage_head("main"), n1.node_ref)
        with self.assertRaises(StateError):
            self.runtime.rewind_lineage("main", n2.node_ref)
        custom = KVRuntime(self.workspace / "other", MODEL, encoder=lambda *a, **k: None, decoder=lambda *a, **k: None)
        self.addCleanup(custom.close)
        custom.create_lineage("main")
        with self.assertRaises(UnsupportedError):
            custom.commit_node("main", "c1", matrices, contexts, method=method(), registry=self.registry)


class CacheEpochTests(LineageTestCase):
    def test_longlive_prompt_switch_recache_epochs(self):
        prompt_a = {KEYS[0]: orthonormal(11), KEYS[1]: orthonormal(12)}
        prompt_b = {key: rotate(prompt_a[key], slice(4, 16), 13 + index) for index, key in enumerate(KEYS)}
        prompt_c = {key: rotate(prompt_a[key], slice(30, 44), 23 + index) for index, key in enumerate(KEYS)}
        self.runtime.create_lineage("A")
        nodes, sources = {}, {}
        for index in range(1, 11):
            data = chunk(100 + index, (index - 1) * TOKENS, prompt_a)
            nodes[index] = self.commit("A", f"c{index}", data)
            sources[("A", index)] = data[0]
        n10 = nodes[10].node_ref
        recache = []
        for index in (8, 9, 10):
            data = chunk(200 + index, (index - 1) * TOKENS, prompt_b)
            sources[("B", index)] = data[0]
            recache.append((f"c{index}", *data))
        self.runtime.fork_lineage(n10, "B")
        switch = self.runtime.commit_recache(
            "B", "switch-to-B", recache, method=method(), registry=self.registry, epoch_id="e1",
            details={"cause": "prompt switch before c11", "sink": "c1 keeps e0", "recomputed_discarded": "c7"},
        )
        self.assertEqual(switch.node.node_kind, "recache")
        self.assertEqual((switch.node.frames, switch.node.frontier), (0, 10 * TOKENS))
        epoch = self.runtime.load_epoch(switch.node.epoch_ref)
        self.assertEqual((epoch.epoch_id, epoch.cause), ("e1", "kv_recache"))
        self.assertEqual(epoch.parent_epoch_ref, nodes[10].node.epoch_ref)
        self.assertEqual([item.chunk_id for item in epoch.ranges], ["c8", "c9", "c10"])
        data = chunk(211, 10 * TOKENS, prompt_b)
        sources[("B", 11)] = data[0]
        n11 = self.commit("B", "c11", data)
        self.assertEqual(n11.node.cache_epoch_id, "e1")
        key = KEYS[0]
        before = self.runtime.resolve_history(n10, [(7 * TOKENS, 10 * TOKENS)])
        self.assertEqual([piece.entry.chunk_id for piece in before], ["c8", "c9", "c10"])
        self.assertEqual({piece.entry.epoch_id for piece in before}, {"e0"})
        self.assertEqual({piece.entry.source for piece in before}, {"chunk"})
        attended = [(0, TOKENS), (8 * TOKENS, 11 * TOKENS)]
        after = self.runtime.resolve_history(n11.node_ref, attended)
        self.assertEqual([(piece.entry.chunk_id, piece.entry.epoch_id, piece.entry.source) for piece in after],
                         [("c1", "e0", "chunk"), ("c9", "e1", "epoch"), ("c10", "e1", "epoch"),
                          ("c11", "e1", "chunk")])
        self.assertEqual(after[0].entry.node_ref, nodes[1].node_ref)
        self.assertEqual(after[1].entry.node_ref, switch.node_ref)
        with self.runtime.acquire_node_history(n10, key, ranges=[(7 * TOKENS, 10 * TOKENS)]) as lease:
            kv_a = lease.tensor.clone()
        with self.runtime.acquire_node_history(n11.node_ref, key, ranges=attended) as lease:
            kv_b = lease.tensor.clone()
        for offset, index in enumerate((8, 9, 10)):
            part = kv_a[offset * TOKENS:(offset + 1) * TOKENS]
            self.assertGreaterEqual(energy(sources[("A", index)][key], part), 0.98)
            self.assertLess(energy(sources[("B", index)][key], part), 0.9)
        self.assertGreaterEqual(energy(sources[("A", 1)][key], kv_b[:TOKENS]), 0.98)
        for offset, index in enumerate((9, 10, 11)):
            part = kv_b[(offset + 1) * TOKENS:(offset + 2) * TOKENS]
            self.assertGreaterEqual(energy(sources[("B", index)][key], part), 0.98)
        self.assertEqual(self.runtime.lineage_head("A"), n10)
        decisions = {row["decision"] for row in switch.report["records"]}
        self.assertTrue(decisions <= {"share", "augment", "refit"})
        self.runtime.fork_lineage(n10, "C")
        recache_c = [(f"c{index}", *chunk(300 + index, (index - 1) * TOKENS, prompt_c)) for index in (8, 9, 10)]
        switch_c = self.runtime.commit_recache("C", "switch-to-C", recache_c, method=method(), registry=self.registry,
                                               epoch_id="e1")
        self.assertNotEqual(switch_c.node.epoch_ref, switch.node.epoch_ref)
        self.assertEqual(self.runtime.load_epoch(switch_c.node.epoch_ref).parent_epoch_ref,
                         nodes[10].node.epoch_ref)
        resolved_c = self.runtime.resolve_history(switch_c.node_ref, [(8 * TOKENS, 9 * TOKENS)])[0]
        self.assertEqual(resolved_c.entry.node_ref, switch_c.node_ref)
        self.assertEqual(self.runtime.resolve_history(switch.node_ref, [(8 * TOKENS, 9 * TOKENS)])[0].entry.node_ref,
                         switch.node_ref)
        with self.assertRaises(ConfigurationError):
            self.runtime.commit_recache("C", "bad", [("c4", *chunk(9, 4 * TOKENS, prompt_c))], method=method(),
                                        registry=self.registry, epoch_id="e2")
        with self.assertRaises(ConfigurationError):
            self.runtime.commit_recache("C", "bad", [("c99", *chunk(9, 0, prompt_c))], method=method(),
                                        registry=self.registry, epoch_id="e2")

    def test_longcat_segments_replace_their_conditioning_cache(self):
        basis = {KEYS[0]: orthonormal(31), KEYS[1]: orthonormal(32)}
        self.runtime.create_lineage("longcat")
        s1_data = chunk(41, 0, basis)
        s1 = self.runtime.commit_segment("longcat", "segment-1", *s1_data, method=method(), registry=self.registry,
                                         epoch_id="s1", details={"conditioning": "last 13 RGB frames -> 4 latents"})
        s2_data = chunk(42, 0, {key: rotate(value, slice(2, 9), 50) for key, value in basis.items()})
        s2 = self.runtime.commit_segment("longcat", "segment-2", *s2_data, method=method(), registry=self.registry,
                                         epoch_id="s2")
        self.assertEqual((s2.node.node_kind, s2.node.frontier, s2.node.token_start), ("segment", TOKENS, 0))
        epoch = self.runtime.load_epoch(s2.node.epoch_ref)
        self.assertEqual((epoch.cause, epoch.position_space), ("longcat_segment", "segment_local"))
        self.assertEqual(epoch.parent_epoch_ref, s1.node.epoch_ref)
        key = KEYS[0]
        for result, data in ((s1, s1_data), (s2, s2_data)):
            (piece,) = self.runtime.resolve_history(result.node_ref)
            self.assertEqual(piece.entry.node_ref, result.node_ref)
            with self.runtime.acquire_node_history(result.node_ref, key) as lease:
                self.assertEqual(tuple(lease.tensor.shape), (TOKENS, CHANNELS))
                self.assertGreaterEqual(energy(data[0][key], lease.tensor), 0.98)
        with self.assertRaises(StateError):
            self.commit("longcat", "chunk", chunk(43, TOKENS, basis))
        with self.assertRaises(StateError):
            self.runtime.commit_segment("longcat", "segment-3", *chunk(44, TOKENS, basis), method=method(),
                                        registry=self.registry, epoch_id="s3")
        self.runtime.create_lineage("global")
        self.commit("global", "c1", chunk(45, 0, basis))
        with self.assertRaises(StateError):
            self.runtime.commit_segment("global", "segment-1", *chunk(46, 0, basis), method=method(),
                                        registry=self.registry, epoch_id="s1")


if __name__ == "__main__":
    unittest.main()
