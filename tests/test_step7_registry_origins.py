"""CR-RES-1 (SGLang Step 7): identical content archived by two sessions must not fail archival.

The predictor registry learns a committed source once (``FitObservation.observation_id``: learner scope + source
identity). A fit's label depends on the prediction it starts from (the top-k subspace size; a session's fit seed),
so a second session or lineage that commits byte-identical content can observe the same source with another payload.
Observations carry their commit context (``predictor_registry.observation_origin``: lineage, commit, pinned registry
revision): a repeat from another origin is a skipped duplicate (first writer wins, ``divergent_duplicates``); a
repeat from the same origin (a retried publish, a replay) must reproduce its payload or raises IntegrityError.
Origins are run context: records and content addresses do not change.
"""

from __future__ import annotations

import json
import unittest
from dataclasses import replace

from reelcache.archive_pipeline import ArchivePipeline
from reelcache.errors import IntegrityError
from reelcache.lineage_commit import ArchiveOptions
from reelcache.predictor_registry import RankPredictorRegistry, observation_origin
from reelcache.types import FitObservation, PredictorKey, canonical_digest
from tests import step6_support as s6

KEY = PredictorKey("fixture-v1", 0, "K", 480, 96, "fp32_gram_eigh.v1", "sketch_qr_subspace.v1:m32:q2", "fresh")


def _observation(source, x, label, origin=None):
    return FitObservation(KEY, canonical_digest({"source": source}), 0.5, 0.98, x, label, origin=origin)


def _origin(name):
    return canonical_digest({"origin": name})


class RegistryOriginTests(unittest.TestCase):
    def test_another_origin_is_a_skipped_duplicate_and_the_first_writer_wins(self):
        registry = RankPredictorRegistry("origins")
        first = _observation("s", 1.2, 80, _origin("session-a"))
        other = _observation("s", 1.25, 96, _origin("session-b"))  # same source, another fit path
        self.assertEqual(registry.observe([first]), (True,))
        view = registry.pin_with([other])  # folding it must not raise either
        self.assertEqual(view.speculative, ())
        self.assertEqual(registry.observe([other]), (False,))
        self.assertEqual(registry.stats, {"applied": 1, "duplicates": 1, "rejected": 0, "divergent_duplicates": 1})
        self.assertEqual(registry.to_dict()["applied"], [[first.observation_id, first.payload_digest,
                                                          first.origin]])
        self.assertEqual(registry.pin().state(KEY).observations, 1)

    def test_a_conflicting_repeat_from_the_same_origin_still_raises(self):
        registry = RankPredictorRegistry("origins")
        first = _observation("s", 1.2, 80, _origin("session-a"))
        registry.observe([first])
        self.assertEqual(registry.observe([first]), (False,))  # a retried publish: idempotent
        tampered = replace(first, observed_rank=81)
        with self.assertRaises(IntegrityError):
            registry.observe([tampered])
        with self.assertRaises(IntegrityError):
            registry.pin_with([tampered])
        legacy = RankPredictorRegistry("legacy")  # observations without origins keep the strict check
        legacy.observe([_observation("s", 1.2, 80)])
        with self.assertRaises(IntegrityError):
            legacy.observe([_observation("s", 1.2, 81, _origin("session-b"))])
        with self.assertRaises(IntegrityError):  # within one batch: same origin, two payloads
            RankPredictorRegistry("batch").observe([first, tampered])
        self.assertEqual(registry.stats["rejected"], 1)

    def test_origins_round_trip_and_legacy_snapshots_load(self):
        registry = RankPredictorRegistry("origins")
        registry.observe([_observation("a", 1.2, 80, _origin("one")), _observation("b", 0.7, 64)])
        data = json.loads(json.dumps(registry.to_dict()))
        self.assertEqual([len(entry) for entry in data["applied"]], [3, 2])
        restored = RankPredictorRegistry.from_dict(data)
        self.assertEqual(restored.to_dict(), registry.to_dict())
        self.assertEqual(restored.observe([_observation("a", 1.3, 96, _origin("two"))]), (False,))
        with self.assertRaises(IntegrityError):
            restored.observe([_observation("b", 0.7, 65, _origin("two"))])  # unknown origin of the first: strict
        tampered = json.loads(json.dumps(data))
        tampered["applied"][0][2] = "not-a-digest"
        with self.assertRaises(IntegrityError):
            RankPredictorRegistry.from_dict(tampered)
        with self.assertRaises(Exception):
            FitObservation(KEY, canonical_digest({"source": "x"}), 0.5, 0.98, 1.0, 64, origin="bad")
        self.assertNotIn("origin", _observation("a", 1.2, 80, _origin("one")).to_dict())  # never serialized


class TwoSessionTests(unittest.TestCase):
    """Commits through the runtime and the archive pipeline: two sessions with byte-identical chunks."""

    def setUp(self):
        self.runtimes = s6.RuntimeFactory(self)

    def sessions(self, registry, *, count=3, seeds=(0, 1), pipelined=False, rate_lag=0):
        runtime = self.runtimes("two-sessions")
        heads = {}
        for branch, seed in zip(("a", "b"), seeds):
            runtime.create_lineage(branch)
            method = replace(s6.method(rate_lag), seed=seed)  # a per-session fit seed: other labels, same sources
            if pipelined:
                pipeline = ArchivePipeline(runtime, method=method, registry=registry,
                                           options=ArchiveOptions(encode_threads=1), encode_workers=2,
                                           max_source_chunks=2, max_unpublished=3)
                try:
                    tickets = [pipeline.submit(branch, f"c{index:06d}", *s6.chunk(index)) for index in range(count)]
                    heads[branch] = pipeline.drain(branch, timeout=300)
                    self.assertEqual(heads[branch], tickets[-1].result())
                finally:
                    pipeline.close(drain=False, timeout=60)
            else:
                for index in range(count):
                    heads[branch] = runtime.commit_node(branch, f"c{index:06d}", *s6.chunk(index), method=method,
                                                        registry=registry,
                                                        options=ArchiveOptions(encode_threads=1)).node_ref
        return runtime, heads

    def test_two_sessions_with_identical_content_both_publish(self):
        for pipelined in (False, True):
            with self.subTest(pipelined=pipelined):
                registry = RankPredictorRegistry(f"two-sessions-{pipelined}")
                runtime, heads = self.sessions(registry, pipelined=pipelined, rate_lag=1 if pipelined else 0)
                for branch in ("a", "b"):
                    self.assertEqual(runtime.load_node(heads[branch]).frontier, 3 * s6.TOKENS)
                self.assertGreater(registry.stats["divergent_duplicates"], 0)
                self.assertEqual(registry.stats["rejected"], 0)

    def test_origins_leave_content_addresses_unchanged(self):
        # The same two sessions in two stores with equal fresh registries: equal node ids (origins are not content;
        # the registry id itself is: records name the predictor revisions they used).
        first, _ = self.sessions(RankPredictorRegistry("ids"), seeds=(0, 1))
        second, _ = self.sessions(RankPredictorRegistry("ids"), seeds=(0, 1))
        self.assertEqual(first.lineage_head("a"), second.lineage_head("a"))
        self.assertEqual(first.lineage_head("b"), second.lineage_head("b"))

    def test_replay_registry_reproduces_the_shared_run(self):
        snapshot = RankPredictorRegistry("replayed").to_dict()
        live = RankPredictorRegistry.from_dict(snapshot)
        self.sessions(live)
        replay = RankPredictorRegistry.from_dict(snapshot, mode="replay", expected_order=live.log)
        self.sessions(replay)
        self.assertEqual(replay.to_dict(), live.to_dict())
        self.assertEqual(replay.stats, live.stats)
        self.assertGreater(live.stats["divergent_duplicates"], 0)
        self.assertTrue(all(len(entry) == 3 for entry in live.to_dict()["applied"]))

    def test_a_retried_commit_is_idempotent_and_a_deviating_one_raises(self):
        registry = RankPredictorRegistry("retries")
        batches = []
        original = registry.observe

        def recording(observations):
            observations = list(observations)
            batches.append(observations)
            return original(observations)
        registry.observe = recording
        self.sessions(registry, count=2, seeds=(0,))
        committed = [item for batch in batches for item in batch]
        self.assertTrue(committed and all(item.origin is not None for item in committed))
        self.assertEqual(original(committed), tuple(False for _ in committed))  # the same publish, retried
        deviating = replace(committed[0], observed_rank=committed[0].observed_rank + 1)
        with self.assertRaises(IntegrityError):  # same lineage, commit and revision: a true conflict
            original([deviating])

    def test_observation_origin_binds_lineage_commit_and_revision(self):
        registry = RankPredictorRegistry("origin-parts")
        view = registry.pin()
        base = observation_origin("a", "chunk", "c000001", view)
        self.assertEqual(base, observation_origin("a", "chunk", "c000001", registry.pin()))
        self.assertNotEqual(base, observation_origin("b", "chunk", "c000001", view))
        self.assertNotEqual(base, observation_origin("a", "chunk", "c000002", view))
        registry.observe([_observation("s", 1.2, 80, base)])
        self.assertNotEqual(base, observation_origin("a", "chunk", "c000001", registry.pin()))


if __name__ == "__main__":
    unittest.main()
