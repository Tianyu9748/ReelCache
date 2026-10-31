"""Step 4 real-data checks on native CausVid normalized pre-RoPE K and direct V (480x832, C=1536).

Sources (test assets under ``$REELCACHE_TEST_ASSETS/step4``; skipped when that is not set):

* ``native_causvid/gpu-optimization-controls-v1/original-15-{K,V}.reelcache``: exact native
  clean-chunk captures (chunk 0, layer 15, BF16 [4680, 1536]) of the 2026-09-22 RTX PRO 6000 run.
* ``sglang_integration/runtime_gate/step4_gpu/capture/causvid_t00_4387197``: Step 4
  capture of all layers of three consecutive clean chunks (prompt t00, seed 0) with the same
  native harness; layer 15 of chunks 0-2 is used for the multi-chunk inherit/certify/augment path.
"""

import json
import unittest
from pathlib import Path

import torch

from reelcache.manager import KVRuntime
from reelcache.predictor_registry import RankPredictorRegistry
from reelcache.rank_method import RankMethodConfig, plan_chunk
from reelcache.serialization import deserialize_tensors
from reelcache.types import (
    ActiveBasisBinding, CodecSpec, MatrixKey, ModelIdentity, TensorContext, canonical_digest, energy_targets,
)
from tests.assets import asset
from tests.core_support import CoreWorkspaceTest

#: Test assets: the captures in this release's tensor format (bit-identical to the evaluation's files).
ORIGINALS = ("step4", "native_causvid", "gpu-optimization-controls-v1")
CAPTURE = ("step4", "sglang_integration", "runtime_gate", "step4_gpu", "capture", "causvid_t00_4387197")
TARGETS = energy_targets("causvid")


def load_originals(layer=15):
    originals = asset(*ORIGINALS)
    rows = json.loads((originals / "original-sources.json").read_text())["sources"]
    sources, contexts = {}, {}
    for row in rows:
        context = TensorContext.from_dict(row["context"])
        if context.key.layer != layer:
            continue
        sources[context.key] = deserialize_tensors((originals / Path(row["path"]).name).read_bytes())["source"]
        contexts[context.key] = context
    return sources, contexts


def exact_crossing(source, fraction):
    values = torch.linalg.eigvalsh(source.double().T @ source.double()).flip(0).clamp_min(0)
    cumulative = values.cumsum(0) / values.sum()
    return int((cumulative < fraction).sum()) + 1, float(cumulative[31])


def stored_energy(source, rows16):
    q = rows16.float()
    reconstructed = (source.float() @ q.T) @ q
    return 1 - float((source.double() - reconstructed.double()).square().sum() / source.double().square().sum())


class RealCausVidFreshFitTests(unittest.TestCase):
    def test_verify_grow_trim_on_native_layer15_matches_exact_crossings(self):
        sources, contexts = load_originals()
        self.assertEqual({key.target for key in sources}, {"K", "V"})
        config = RankMethodConfig(TARGETS, CodecSpec("svd_hevc", rank=1))
        registry = RankPredictorRegistry("real-causvid")
        state = torch.get_rng_state()
        plan = plan_chunk(config, registry.pin(), sources, contexts, {key: ActiveBasisBinding() for key in sources},
                          {key: None for key in sources}, chunk_id="native-c0",
                          source_ids={key: canonical_digest(["native", key.to_dict()]) for key in sources})
        self.assertTrue(torch.equal(torch.get_rng_state(), state))
        for key, item in plan.plans.items():
            target = TARGETS.for_target(key)
            crossing, optimal32 = exact_crossing(sources[key], target)
            self.assertEqual(item.decision, "fresh")
            self.assertLessEqual(abs(item.observed_rank - crossing), 1)
            self.assertEqual(item.rank_view, 32 * ((item.observed_rank + 31) // 32))
            self.assertGreaterEqual(stored_energy(sources[key], item.new_rows), target)
            self.assertLess(stored_energy(sources[key], item.new_rows[:item.rank_view - 32]), target)
            anchor = item.predictions[0]["anchor_energy"]
            self.assertLessEqual(anchor, optimal32 + 1e-6)
            self.assertGreater(anchor, optimal32 - 0.02)
            self.assertEqual(item.work["counters"]["eigh_count"], 1)
            self.assertEqual(len(item.observations), 1)


class RealCausVidLineageTests(CoreWorkspaceTest):
    """Three consecutive native clean chunks of one layer through the store (relabeled as a 1-layer fixture)."""

    def test_consecutive_native_chunks_inherit_certify_and_meet_targets(self):
        capture = asset(*CAPTURE)
        manifest = json.loads((capture / "sources.json").read_text())
        self.assertEqual(manifest["chunks"], 3)
        model = ModelIdentity("causvid-layer15", manifest["model"]["fingerprint"], 1, 1536, "tokens_channels",
                              "bfloat16")
        registry = RankPredictorRegistry("real-lineage")
        config = RankMethodConfig(TARGETS, CodecSpec("svd", rank=1))
        with KVRuntime(self.workspace / "store", model) as runtime:
            runtime.create_lineage("t00")
            results = []
            for chunk in range(3):
                matrices, contexts = {}, {}
                for target in ("K", "V"):
                    path = capture / f"original-t{chunk * 4680:06d}-15-{target}.reelcache"
                    key = MatrixKey(0, target)
                    matrices[key] = deserialize_tensors(path.read_bytes())["source"]
                    contexts[key] = TensorContext(model, key, chunk * 4680, 3, 30, 52,
                                                  "pre_rope" if target == "K" else "none")
                results.append((runtime.commit_node("t00", f"c{chunk}", matrices, contexts, method=config,
                                                    registry=registry), matrices))
            first = results[0][0].report["records"]
            self.assertEqual({row["decision"] for row in first}, {"fresh"})
            for result, matrices in results:
                for row in result.report["records"]:
                    key = MatrixKey(0, row["key"]["target"])
                    record = runtime.load_record(result.records[0][1][key])
                    self.assertGreaterEqual(record.energy.stored_factor_energy, TARGETS.for_target(key))
                    packet = runtime.matrix_packet(result.records[0][1][key])
                    if record.packet_tile is None:
                        decoded = runtime.decoder(packet)
                    else:
                        from reelcache.codecs import decode_packed
                        decoded = decode_packed(packet, keys=[record.entry])[record.entry]
                    self.assertGreaterEqual(1 - float(
                        (matrices[key].double() - decoded.double()).square().sum()
                        / matrices[key].double().square().sum()), TARGETS.for_target(key) - 1e-6)
                    if row["decision"] == "share":
                        self.assertEqual(row["counters"]["candidate_fit_count"], 0)
            later = [row["decision"] for result, _ in results[1:] for row in result.report["records"]]
            self.assertTrue(set(later) <= {"share", "augment", "refit"})


if __name__ == "__main__":
    unittest.main()
