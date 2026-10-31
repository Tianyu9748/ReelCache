"""Workload v2 (freeze W1a): the CausVid / RAVEN prompt switch, semantics ``cross_attention_only`` (user decision).

* Capability: CausVid (both visibility policies) and RAVEN change prompts through LongLive 1.0's
  ``prompt_switches``; LongLive 1.0 keeps its recache epoch, LongLive 2.0 its multi-shot form; the profiles'
  data (fingerprints, bridge identity: archive roots and node ids) is F6's.
* Per-block conditioning ids exactly as LongLive 1.0's when a CausVid / RAVEN request carries ``prompt_switches``
  (possibly empty); none without the field (F6 records).
* Through the bridge (toy stage emulating the profile stage's cross-attention switch: no recache, the history KV
  stays): a pause at the switch chunk and a resume with the switch is bitwise the uninterrupted run; a fork at the
  pre-switch node runs another future prompt; restored blocks generated under another prompt (or under the other
  form) are refused without moving a head; an unchanged prompt equals the F6 path; the switch is recorded on the
  nodes from chunk c on and in the workload telemetry (``prompt_transition.kind == "switch"``).

The SGLang stage itself (``sglang_stages.ProfileCausalDenoisingStage``) imports SGLang; its behaviour is checked on
the GPU (W1a GPU validation (a)/(b)).
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch

from reelcache.errors import UnsupportedError
from reelcache.integrations.sglang.controls import CROSS_ATTENTION_SWITCH, PROMPT_CHANGE, prompt_change_capability
from reelcache.integrations.sglang.profiles import LONGLIVE2
from reelcache.integrations.sglang.profiles import ModelProfile as BridgeProfile
from reelcache.types import canonical_digest
from reelcache_sglang_models.causal_wan import CAUSVID, LONGLIVE1, RAVEN
from reelcache_sglang_models.causal_wan import lifecycle as lc
from reelcache_sglang_models.causal_wan import profiles as pf
from tests.test_sglang_bridge import (FRAMES, GRID, LAYERS, PROFILE, TPF, BridgeTestCase, Capture, Chunk,
                                      FullHistoryCache, handle, request, rope_cos_sin, rotate_key, toy_kv)

#: F6 (tree 5be577ce83082926) identities, computed from runtime_gate/workload/trees/w0_f6: (profile fingerprint[:16],
#: bridge model fingerprint = canonical_digest({"profile": bridge profile, "model_path"})). The capability must not
#: change them: they name the archive roots and enter every node id. Values under this release's names and placeholder
#: weight paths (MODEL_ROOT): the digests of the same inputs with the renamed identifiers and the paths mapped to /models.
F6_IDENTITIES = {
    "causvid": ("842d7e29bc09c28a", "cfa3ff8a1f561aa5e360d91cc84c75eaaf35b40d5c945dc2a45dfbed5c825061"),
    "causvid:window_21_frames": ("842d7e29bc09c28a",
                                 "902b7edb4dcb708b1a1d2efe301008b76ac97ceedf437b591c533314cd8e5526"),
    "raven": ("058384b8168417af", "ee17994aea9ec9057c026588d41f29deaf6a617bb6a02657c2e574d80a34c9a9"),
    "longlive1": ("9a4f9d5c9eaf719c", "fd815ec558fcfa5b94311d1c4cd2502d11c73d991ce0119b38521ccba594d1fa"),
}
BLOCKS = 4


def bridge_profile(profile, visibility=None) -> BridgeProfile:
    return BridgeProfile.from_dict(pf.bridge_profile_fields(profile, visibility, extended=True))


def cross_request(session, *, prompt="A", switches=None, blocks=BLOCKS, profile=CAUSVID):
    """A CausVid-like request: ``prompt_switches`` is a timeline field (outside the conditioning digest, as the
    profile stage's ``_CAUSAL_KV_IDENTITY_EXCLUDE``); the per-block ids come from the profile rule."""
    req = request(session, prompt=prompt, blocks=blocks)
    req.prompt_switches = switches
    req.block_conditioning = pf.switch_block_conditioning(profile, prompt, switches, blocks)
    return req


def cross_drive(bridge, req, *, stage=None):
    """One request like ``ProfileCausalDenoisingStage.forward`` with a cross-attention prompt switch: a block uses its
    segment's prompt (the toy's K/V and latents depend on it); at a switch the stage resets its cross-attention caches
    (nothing for the toy) and records a ``prompt_transition`` span; no recache forward ever runs and the history KV
    stays. Returns (summary, latents, caches, run)."""
    caches = stage.caches_for(req.streams) if stage is not None else {
        stream: [FullHistoryCache(8 * FRAMES * TPF) for _ in range(LAYERS)] for stream in req.streams}
    latents = torch.zeros(1, 4, sum(req.block_sizes), *GRID)
    generator = torch.Generator().manual_seed(req.conditioning["seed"])
    torch.randn(3, generator=generator)  # latent preparation draws
    run = bridge.begin_request(req)
    target = SimpleNamespace(caches=caches, latents=latents, generator=generator, rope_cos_sin=rope_cos_sin,
                             rotate_key=rotate_key, device=torch.device("cpu"))
    switches = {item["chunk"]: item["prompt"] for item in (getattr(req, "prompt_switches", None) or ())}
    try:
        resume = run.restore(target)
        telemetry = getattr(run, "workload_telemetry", None)
        prompt = req.conditioning["prompt"]
        for at in sorted(switches):  # the segment of the first new block (switches before it are restored state)
            if at < resume:
                prompt = switches[at]
        start = 0
        for index, frames in enumerate(req.block_sizes):
            if index < resume:
                start += frames
                continue
            if index in switches:  # cross-attention only: the new prompt from this block on, no recache
                transition = None if telemetry is None else telemetry.begin("prompt_transition")
                prompt = switches[index]
                if telemetry is not None:
                    telemetry.end(transition)
                    telemetry.prompt_transition("switch")
            if telemetry is not None:
                telemetry.mark_generation()
            chunk = Chunk(index, start, frames, start * TPF, frames * TPF, float(start), False, 0)
            history = caches["cond"][0].k[:, :caches["cond"][0].end].float().mean() if caches["cond"][0].end else 0.0
            noise = torch.randn(1, 4, frames, *GRID, generator=generator)
            # the new block "cross-attends" to its prompt: its latents depend on it
            latents[:, :, start:start + frames] = noise + float(history) + 1e-3 * sum(map(ord, prompt))
            cos, sin = rope_cos_sin(chunk.rope_start_frame, frames)
            for stream in req.streams:
                capture = Capture(chunk, run.stream_of(caches[stream]))
                run.begin_clean_forward(capture)
                for layer in range(LAYERS):
                    key, value = toy_kv(prompt, chunk, layer, stream, caches[stream][layer])
                    run.capture_layer(capture, layer, key, value, (cos, sin))
                    caches[stream][layer].update(rotate_key(key, cos, sin), value, chunk.token_start)
                run.end_clean_forward(capture)
            start += frames
            if run.end_chunk(chunk, latents[:, :, start - frames:start].clone()):
                break
        return run.finish(error=None), latents, caches, run
    except BaseException as exc:
        run.finish(error=exc)
        raise


class CapabilityTests(unittest.TestCase):
    def test_capability_on_for_causvid_and_raven(self):
        for profile, visibility in ((CAUSVID, None), (CAUSVID, "window_21_frames"), (RAVEN, None)):
            capability = prompt_change_capability(bridge_profile(profile, visibility))
            self.assertEqual((capability["supported"], capability["mode"], capability["request_fields"]),
                             (True, "cross_attention_only", ["prompt_switches"]), (profile.profile_id, visibility))
            self.assertIn("never recomputed", capability["reason"])
        self.assertEqual(prompt_change_capability(bridge_profile(LONGLIVE1))["mode"], "recache_epoch")
        self.assertEqual(prompt_change_capability(LONGLIVE2)["mode"], "multi_shot")
        self.assertFalse(prompt_change_capability(PROFILE)["supported"])  # a model without native prompt change
        self.assertEqual(PROMPT_CHANGE["none"]["supported"], False)  # the table itself is F6's
        capability = prompt_change_capability(bridge_profile(RAVEN))
        capability["request_fields"].append("x")  # a copy: the module constant is never mutated
        self.assertEqual(CROSS_ATTENTION_SWITCH["request_fields"], ["prompt_switches"])

    def test_prompt_switch_modes(self):
        self.assertEqual(pf.prompt_switch_mode(CAUSVID), pf.PROMPT_SWITCH_CROSS_ATTENTION)
        self.assertEqual(pf.prompt_switch_mode(RAVEN), "cross_attention_only")
        self.assertEqual(pf.prompt_switch_mode(LONGLIVE1), pf.PROMPT_SWITCH_RECACHE)
        for profile in (CAUSVID, RAVEN):  # no recache policy: the switch never re-materializes KV
            self.assertEqual(profile.recache.trigger, "none")

    def test_profile_data_and_bridge_identity_are_f6s(self):
        for profile, visibility in ((CAUSVID, None), (CAUSVID, "window_21_frames"), (RAVEN, None),
                                    (LONGLIVE1, None)):
            bridge = bridge_profile(profile, visibility)
            identity = canonical_digest({"profile": bridge.to_dict(), "model_path": profile.sglang.model_path})
            self.assertEqual((profile.fingerprint()[:16], identity), F6_IDENTITIES[bridge.name], bridge.name)
            if profile is not LONGLIVE1:  # CausVid / RAVEN keep recache "none": the switch is not profile data
                self.assertNotIn("prompt_switch", str(profile.to_dict()))


class BlockConditioningTests(unittest.TestCase):
    def test_conditioning_ids_per_block_exactly_as_longlive1(self):
        switches = [{"chunk": 2, "prompt": "B"}]
        for profile in (CAUSVID, RAVEN):
            ids = pf.switch_block_conditioning(profile, "A", switches, BLOCKS)
            a, b = lc.prompt_digest("A"), lc.prompt_digest("B")
            self.assertEqual(ids, (f"s0:{a}", f"s0:{a}", f"s1:{b}", f"s1:{b}"))
            self.assertEqual(ids, pf.switch_block_conditioning(LONGLIVE1, "A", switches, BLOCKS))
            self.assertEqual(ids, lc.block_conditioning_ids("A", switches, BLOCKS))
            # a switch right after the first chunk is legal here (LongLive 1.0's chunk-5 rule is its sink rule)
            early = pf.switch_block_conditioning(profile, "A", [{"chunk": 1, "prompt": "B"}], BLOCKS)
            self.assertEqual([item.split(":")[0] for item in early], ["s0", "s1", "s1", "s1"])

    def test_without_the_field_the_f6_path_has_no_ids(self):
        for profile in (CAUSVID, RAVEN):
            self.assertIsNone(pf.switch_block_conditioning(profile, "A", None, BLOCKS))  # F6: no per-block ids
            self.assertEqual(pf.switch_block_conditioning(profile, "A", [], BLOCKS),
                             (f"s0:{lc.prompt_digest('A')}",) * BLOCKS)  # the switch path, no switch yet
        # LongLive 1.0 always had ids (F6 normalized None to []); a profile without switches never has any
        self.assertEqual(pf.switch_block_conditioning(LONGLIVE1, "A", None, BLOCKS),
                         lc.block_conditioning_ids("A", [], BLOCKS))
        no_switch = SimpleNamespace(recache=SimpleNamespace(trigger="none"), profile_id="other")
        self.assertIsNone(pf.prompt_switch_mode(no_switch))
        self.assertIsNone(pf.switch_block_conditioning(no_switch, "A", [], BLOCKS))

    def test_resume_at_the_switch_applies_it_anew(self):
        """The stage resumes a cross-attention switch with ``switch_restored=False``: switches before the resume
        point are restored state (their segment), the one at it is applied by the request (no recache to restore)."""
        at_switch = lc.resume_plan(num_chunks=BLOCKS, switch_chunks=[2], resume_blocks=2, switch_restored=False)
        self.assertEqual(at_switch.restored_switches, ())
        self.assertEqual(lc.segment_of(2, [s for s in [2] if s < 2]), 0)  # segment before the stage applies it
        after = lc.resume_plan(num_chunks=BLOCKS, switch_chunks=[2], resume_blocks=3)
        self.assertEqual((after.restored_switches, after.segment), ((2,), 1))


class CrossAttentionSwitchSessionTests(BridgeTestCase):
    representation = "raw"
    SWITCH = [{"chunk": 2, "prompt": "B"}]

    def test_pause_at_the_switch_then_resume_is_bitwise_the_uninterrupted_run(self):
        _, expected, expected_caches, _ = cross_drive(self.bridge("reference"),
                                                      cross_request(handle("ref"), switches=self.SWITCH))
        bridge = self.bridge()
        cross_drive(bridge, cross_request(handle("s1", pause_after_chunks=2), switches=self.SWITCH))
        bridge.drop_resident()  # restore from the archive, not the resident KV
        resumed, latents, caches, _ = cross_drive(bridge, cross_request(handle("s1", "resume"),
                                                                        switches=self.SWITCH))
        self.assertEqual((resumed["resumed_from_chunks"], resumed["generated_chunks"]), (2, BLOCKS))
        self.assertEqual(resumed["restore"]["mode"], "archive")
        self.assertTrue(torch.equal(latents, expected))
        self.assertCachesEqual(caches, expected_caches)
        # no recache: only chunk nodes on the path, the switch is recorded from chunk 2 on
        nodes = bridge.archive.path(bridge.checkpoint("s1")["head"])
        self.assertEqual([node.kind for node in nodes], ["root"] + ["chunk"] * BLOCKS)
        b = lc.prompt_digest("B")
        self.assertEqual([node.record.get("block_conditioning") for node in nodes[1:]],
                         list(pf.switch_block_conditioning(CAUSVID, "A", self.SWITCH, BLOCKS)))
        self.assertTrue(all(node.record["block_conditioning"] == f"s1:{b}" for node in nodes[3:]))

    def test_fork_at_the_pre_switch_node_and_rejected_restored_mismatches(self):
        bridge = self.bridge()
        cross_drive(bridge, cross_request(handle("s1"), switches=self.SWITCH))
        nodes = bridge.archive.path(bridge.checkpoint("s1")["head"])
        pre_switch, post_switch = nodes[2].ref, nodes[3].ref
        # rewind to the pre-switch node under the same switch: bitwise the uninterrupted run
        _, expected, _, _ = cross_drive(self.bridge("reference"), cross_request(handle("ref"), switches=self.SWITCH))
        rewound, latents, _, _ = cross_drive(bridge, cross_request(handle("s1", "rewind", checkpoint=pre_switch),
                                                                   switches=self.SWITCH))
        self.assertEqual(rewound["resumed_from_chunks"], 2)
        self.assertTrue(torch.equal(latents, expected))
        # a fork at the pre-switch node under another future prompt C: valid, equals the uninterrupted A -> C run
        to_c = [{"chunk": 2, "prompt": "C"}]
        _, expected_c, _, _ = cross_drive(self.bridge("reference_c"), cross_request(handle("ref"), switches=to_c))
        forked, latents, _, _ = cross_drive(bridge, cross_request(
            handle("f", "fork", from_session="s1", checkpoint=pre_switch), switches=to_c))
        self.assertEqual((forked["resumed_from_chunks"], forked["generated_chunks"]), (2, BLOCKS))
        self.assertTrue(torch.equal(latents, expected_c))
        # restored blocks generated under B are refused under C; a switch moved earlier contradicts block 1
        head = bridge.archive.head("s1")
        for checkpoint, switches in ((post_switch, to_c), (pre_switch, [{"chunk": 1, "prompt": "B"}])):
            with self.assertRaises(UnsupportedError):
                cross_drive(bridge, cross_request(handle("g", "fork", from_session="s1", checkpoint=checkpoint),
                                                  switches=switches))
            with self.assertRaises(UnsupportedError):
                cross_drive(bridge, cross_request(handle("s1", "rewind", checkpoint=checkpoint), switches=switches))
        self.assertIsNone(bridge.archive.head("g"))
        self.assertEqual(bridge.archive.head("s1"), head)  # nothing moved

    def test_a_session_keeps_one_form_of_the_field(self):
        """The F6 form (no ``prompt_switches``: no ids) and the switch path (ids) do not mix within a session:
        restored blocks generated under the other form are refused, explicitly, and nothing moves."""
        bridge = self.bridge()
        cross_drive(bridge, cross_request(handle("old", pause_after_chunks=2), switches=None))
        with self.assertRaises(UnsupportedError) as caught:
            cross_drive(bridge, cross_request(handle("old", "resume"), switches=self.SWITCH))
        self.assertIn("block conditioning", str(caught.exception))
        self.assertEqual(bridge.controller.get("old").generated_chunks, 2)
        cross_drive(bridge, cross_request(handle("new", pause_after_chunks=2), switches=[]))  # the switch path
        resumed, _, _, _ = cross_drive(bridge, cross_request(handle("new", "resume"), switches=self.SWITCH))
        self.assertEqual(resumed["generated_chunks"], BLOCKS)
        with self.assertRaises(UnsupportedError):
            cross_drive(bridge, cross_request(handle("new", "rewind", checkpoint=2), switches=None))

    def test_unchanged_prompt_equals_the_old_path(self):
        old, new = self.bridge("f6"), self.bridge("enabled")
        f6, f6_latents, f6_caches, _ = cross_drive(old, cross_request(handle("s"), switches=None))
        enabled, latents, caches, _ = cross_drive(new, cross_request(handle("s"), switches=[]))
        self.assertTrue(torch.equal(latents, f6_latents))
        self.assertCachesEqual(caches, f6_caches)
        self.assertEqual(enabled["chunk_digests"], f6["chunk_digests"])
        # the F6 path's records carry no per-block ids (its node ids are F6's); the switch path's carry s0 ids
        self.assertTrue(all("block_conditioning" not in node.record for node in old.archive.path(f6["head"])[1:]))
        self.assertEqual({node.record["block_conditioning"] for node in new.archive.path(enabled["head"])[1:]},
                         {f"s0:{lc.prompt_digest('A')}"})

    def test_the_switch_at_chunk_c_is_recorded(self):
        bridge = self.bridge(workload_telemetry=True)
        first, _, _, _ = cross_drive(bridge, cross_request(handle("s1", pause_after_chunks=2), switches=self.SWITCH))
        before = bridge.workload_record(first["request_key"])
        self.assertEqual((before["prompt_transition"]["kind"], before["prompt_transition"]["host_ms"]), ("none", 0))
        resumed, _, _, run = cross_drive(bridge, cross_request(handle("s1", "resume"), switches=self.SWITCH))
        record = bridge.workload_record(resumed["request_key"])
        self.assertEqual(record["prompt_transition"]["kind"], "switch")
        transitions = [span for span in record["spans"] if span["name"] == "prompt_transition"]
        self.assertEqual(len(transitions), 1)
        self.assertEqual(transitions[0]["phase"], "prep")  # before the first new block's denoise
        self.assertFalse(any(span["name"] == "recache" for span in record["spans"]))  # never a recache
        timeline = bridge.controller.get("s1").timeline
        self.assertEqual([(item["kind"], item["block"], item["conditioning"].split(":")[0]) for item in timeline],
                         [("chunk", 0, "s0"), ("chunk", 1, "s0"), ("chunk", 2, "s1"), ("chunk", 3, "s1")])


if __name__ == "__main__":
    unittest.main()
