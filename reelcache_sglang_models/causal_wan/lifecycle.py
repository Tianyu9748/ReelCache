"""Session lifecycle of the causal-Wan profiles on the ReelCache bridge (pure Python: no torch, no SGLang).

Step 12 phase 2 attaches the Step 11 bridge (``reelcache.integrations.sglang``, SGLang ``--reelcache-enable``) to the
profile stages. This module holds the plain-data decisions the SGLang stages make around it:

* which prompt segment a chunk belongs to, and whether a resumed request still has to run the LongLive 1.0
  prompt-switch recache before its first generated chunk (``resume_plan``): a switch before a restored chunk is
  already part of the restored state (its recache epoch is on the restored path), a switch exactly at the resume
  point is part of it only if the restored head is that switch's recache node;
* the recache cache epoch's identity and JSON details (``recache_epoch_id``, ``recache_epoch_details``) and the
  per-block conditioning identity of a prompt-switch timeline (``block_conditioning_ids``);
* the bridge runtime configuration of a profile (``bridge_runtime_config``: ``--reelcache-config`` JSON with the
  profile inline, so the three 1.3B profiles never depend on a process-wide profile registry);
* the self-attention cache geometry of a request (``request_cache_tokens``): the stage keeps its caches across
  requests and reuses them (cursors reset, K/V kept, so a session's KV can stay resident) only for the same geometry.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .profiles import ModelProfile, VisibilityPolicy, bridge_profile_fields

RECACHE_EPOCH_PREFIX = "switch"


def prompt_digest(prompt: str) -> str:
    """Stable short identity of a prompt text (sha256 of its UTF-8 bytes, 12 hex digits)."""
    if not isinstance(prompt, str) or not prompt:
        raise ValueError("a prompt digest needs a non-empty prompt")
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:12]


def segment_of(chunk_index: int, switch_chunks: Sequence[int]) -> int:
    """Prompt segment of a chunk: the number of prompt switches at or before it (segment 0 = initial prompt)."""
    if chunk_index < 0:
        raise ValueError("chunk_index must be non-negative")
    return sum(1 for s in switch_chunks if s <= chunk_index)


def block_conditioning_ids(prompt: str, switches: Sequence[Mapping[str, Any]], num_blocks: int) -> tuple[str, ...]:
    """Per-block conditioning identity of a prompt-switch timeline (SGLang ``CausalKVRequest.block_conditioning``).

    Block ``i`` is generated under its segment's prompt: id ``s<segment>:<prompt digest>``. The segment index is
    part of the id because a switch re-materializes KV (a recache epoch) even when the new prompt equals the old
    one. The bridge compares these ids over the restored blocks only (a recache epoch by the block it precedes), so
    a session can be forked or resumed under a different FUTURE schedule (report 3.7: a fork from n10 under prompt
    C runs its own recache epoch) while a restored block or epoch generated under another prompt is refused.
    """
    if num_blocks <= 0:
        raise ValueError("num_blocks must be positive")
    chunks = [int(s["chunk"]) for s in switches]
    prompts = [prompt] + [s["prompt"] for s in switches]
    digests = [prompt_digest(p) for p in prompts]
    return tuple(f"s{segment}:{digests[segment]}" for segment in (segment_of(i, chunks) for i in range(num_blocks)))


def recache_epoch_id(switch_before_chunk: int, prompt: str) -> str:
    """Epoch id of the prompt-switch recache before ``switch_before_chunk`` under ``prompt`` (no '/')."""
    if switch_before_chunk <= 0:
        raise ValueError("a prompt switch happens before a chunk > 0")
    return f"{RECACHE_EPOCH_PREFIX}-c{switch_before_chunk:04d}-{prompt_digest(prompt)}"


def recache_epoch_details(cache_epoch_fields: Mapping[str, Any], *, prompt: str, segment: int,
                          profile_id: str) -> dict[str, Any]:
    """JSON details of a recache epoch: ``RecacheRecord.cache_epoch_fields()`` plus the new prompt's identity."""
    out = {key: (list(value) if isinstance(value, tuple) else value) for key, value in cache_epoch_fields.items()}
    out.update(profile=profile_id, prompt_sha256_12=prompt_digest(prompt), segment=int(segment))
    return out


def switch_recache_restored(restore_report: Mapping[str, Any] | None, switch_chunk: int) -> bool | None:
    """Whether the recache of a prompt switch before ``switch_chunk`` is on the restored path.

    Called only when a request resumes exactly at a switch (``resume_blocks == switch_chunk``): the restored head is
    either the last chunk before the switch (the recache has not run yet) or that switch's recache node (it has).
    The recache re-materializes the chunks right before the switch (LongLive 1.0: the 9 non-sink buffer frames), so
    the restored KV of chunk ``switch_chunk - 1`` comes from a recache epoch exactly in the second case: a chunk's
    KV can only be overridden by a recache that ran after the chunk was generated. Returns None when the bridge's
    restore report carries no visibility plans (cannot be decided: the caller must refuse, never guess).
    """
    inferred = _switch_recache_from_plans(restore_report, switch_chunk)
    if restore_report and restore_report.get("head_kind") is not None:
        # The bridge reports the restored head (Step 11 CR-12p2-4): a recache head preceding this switch's block.
        head_says = restore_report["head_kind"] == "recache" and restore_report.get("head_block") == switch_chunk
        if inferred is None:
            return head_says
        if head_says and not inferred:
            # A recache head whose epoch is only partly on the path (the RAW reference chains one node per range:
            # an intermediate node re-materialized only the first chunk): the installed KV of the chunk before the
            # switch is still the pre-switch KV, so the recache must run again.
            return False
        if inferred and not head_says:
            raise ValueError(f"restored head {restore_report['head_kind']!r} (block {restore_report.get('head_block')}) "
                             f"contradicts the restored sources of chunk {switch_chunk - 1}")
    return inferred


def _switch_recache_from_plans(restore_report: Mapping[str, Any] | None, switch_chunk: int) -> bool | None:
    if not restore_report or not isinstance(restore_report.get("plans"), Mapping) or not restore_report["plans"]:
        return None
    target = switch_chunk - 1
    sources = set()
    for plan in restore_report["plans"].values():
        for span in plan.get("spans", ()):
            if int(span.get("chunk", -1)) == target:
                sources.add(span.get("source"))
    if not sources:
        return None
    if len(sources) != 1 or not sources <= {"chunk", "recache"}:
        raise ValueError(f"inconsistent restored sources {sorted(map(str, sources))} for chunk {target}")
    return sources == {"recache"}


@dataclass(frozen=True)
class ResumePlan:
    """What a (possibly resumed) request does before its first generated chunk."""

    resume_blocks: int
    segment: int  # prompt segment of the first generated chunk
    restored_switches: tuple[int, ...]  # switches whose recache is part of the restored state
    recache_before_first_chunk: bool  # run the switch recache before chunk ``resume_blocks``


def resume_plan(*, num_chunks: int, switch_chunks: Sequence[int], resume_blocks: int,
                switch_restored: bool | None = None) -> ResumePlan:
    """Segment and recache decisions for a request that resumes after ``resume_blocks`` restored chunks.

    ``switch_restored`` is required (``switch_recache_restored``) when ``resume_blocks`` is a switch chunk.
    """
    switches = sorted(int(s) for s in switch_chunks)
    if len(set(switches)) != len(switches) or any(s <= 0 or s >= num_chunks for s in switches):
        raise ValueError(f"prompt switches {switches} must be distinct chunks in (0, {num_chunks})")
    if not 0 <= resume_blocks < num_chunks:
        raise ValueError(f"resume_blocks {resume_blocks} outside [0, {num_chunks})")
    restored = [s for s in switches if s < resume_blocks]
    at_resume = resume_blocks in switches
    if at_resume and resume_blocks > 0:
        if switch_restored is None:
            raise ValueError(f"resuming exactly at the prompt switch before chunk {resume_blocks} needs to know "
                             "whether its recache epoch is on the restored path")
        if switch_restored:
            restored.append(resume_blocks)
    run_recache = at_resume and resume_blocks > 0 and not switch_restored
    return ResumePlan(resume_blocks=resume_blocks, segment=segment_of(resume_blocks, switches),
                      restored_switches=tuple(restored), recache_before_first_chunk=run_recache)


def block_prompt_texts(prompt: str, switches: Sequence[Mapping[str, Any]], num_blocks: int) -> list[str]:
    """W1b: the prompt text every block is generated under (its segment's prompt), for node sidecars."""
    chunks = [int(s["chunk"]) for s in switches]
    prompts = [prompt] + [s["prompt"] for s in switches]
    return [prompts[segment_of(i, chunks)] for i in range(num_blocks)]


def replay_schedule(steps: Sequence[Mapping[str, Any]], switch_chunks: Sequence[int]) -> list[dict[str, Any]]:
    """W1b exact recompute arm, profile stages: the conditioning of every replay step (the bridge's ``replay_plan``
    steps, native order) under the request's prompt-switch schedule. The bridge has already checked every restored
    block's (and epoch's) conditioning id against the request, so a block's segment here is its historical one.

    * ``chunk`` step of block ``i``: segment ``segment_of(i)``; ``reset_cross_attention`` when the previous forward used
      another prompt (a switch whose recache is not on the path, e.g. a cross-attention-only switch);
    * ``recache`` step before block ``s`` (a LongLive 1.0 prompt-switch epoch): ``(s - 1``'s segment, ``s``'s segment)``;
      ``s`` must be a switch of the request. The recache resets the cross-attention caches before and after.
    """
    switches = sorted(int(s) for s in switch_chunks)
    out: list[dict[str, Any]] = []
    current = None
    for step in steps:
        kind = step.get("kind")
        if kind == "chunk":
            block = int(step["block"])
            segment = segment_of(block, switches)
            out.append({"kind": "chunk", "block": block, "segment": segment, "reset_cross_attention": segment != current})
            current = segment
        elif kind == "recache":
            block = int(step["precedes_block"])
            if block not in switches:
                raise ValueError(f"the restored path holds a cache epoch before block {block}, which is not a prompt "
                                 f"switch of this request (switches {switches})")
            new = segment_of(block, switches)
            out.append({"kind": "recache", "precedes_block": block, "previous_segment": segment_of(block - 1, switches),
                        "new_segment": new, "reset_cross_attention": True})
            current = None
        else:
            raise ValueError(f"unknown replay step kind {kind!r}")
    return out


def recache_matches(step: Mapping[str, Any], plan: Mapping[str, Any]) -> bool:
    """Whether a replayed recache (``RecacheRecord.plan``) re-materialized exactly the archived epoch's window and
    written ranges (tokens)."""
    fs = int(plan["frame_seq"])
    start, end = (int(x) for x in plan["recompute_tokens"])
    first, last = (int(x) for x in plan["written_global_frames"])
    window = step.get("window") or {}
    return ((int(window.get("token_start", -1)), int(window.get("num_tokens", -1))) == (start, end - start)
            and [list(map(int, item)) for item in step.get("written") or []] == [[first * fs, (last - first) * fs]])


#: Marks an SGLang stage seam ``_begin_causal_kv_run`` that already runs the W1b replay hook (``sglang_stages``).
REPLAY_SEAM_MARKER = "_reelcache_replay_seam"


def pipeline_stages(pipeline: Any) -> list:
    """The stages of an SGLang pipeline (``ComposedPipelineBase.stages``), of a worker (its ``pipeline``), or of an
    iterable of stages."""
    for owner in (pipeline, getattr(pipeline, "pipeline", None)):
        stages = getattr(owner, "stages", None)
        if stages is not None and not callable(stages):
            return list(stages)
    return list(pipeline)


def needs_replay_seam(stage: Any) -> bool:
    """Whether a causal stage's SGLang seam ``_begin_causal_kv_run`` still lacks the W1b replay hook: SGLang's own
    causal stages (LongLive 2.0 and any model ``CausalDMDDenoisingStage`` serves) until wrapped; never ReelCache's
    profile stages (their class carries it) or stages without the seam."""
    begin = getattr(stage, "_begin_causal_kv_run", None)
    return callable(begin) and not getattr(begin, REPLAY_SEAM_MARKER, False)


def request_cache_tokens(profile: ModelProfile, policy: VisibilityPolicy, *, total_frames: int,
                         tokens_per_frame: int) -> dict[str, int]:
    """Self-attention cache geometry (tokens) the profile stage gives a request of ``total_frames`` latent frames.

    ``full_history`` holds every frame of the request, but at least two chunks so a one-chunk request still writes
    its KV; a sink + window policy holds ``window`` frames (sink included) whatever the request length. The stage keeps
    its caches across requests and reuses them only for a request with this same geometry (resident session KV): a
    full-history request of another length gets new caches, which are never resident.
    """
    if tokens_per_frame <= 0:
        raise ValueError("tokens_per_frame must be positive")
    frames = policy.cache_latent_frames(total_frames)
    if policy.window_latent_frames is None:
        frames = max(frames, 2 * profile.chunk_latent_frames)
    size = frames * tokens_per_frame
    return {"cache_tokens": size, "sink_tokens": policy.sink_latent_frames * tokens_per_frame,
            "attention_window_tokens": size}


def bridge_accepts_extended_fields() -> bool:
    """Whether the importable ReelCache bridge's ModelProfile has ``capture_dtypes``/``matmul_precision`` (CR-12p2-2/3).

    False when ReelCache's SGLang bridge is not importable (the config is then for an older or absent bridge)."""
    try:
        from reelcache.integrations.sglang.profiles import ModelProfile as BridgeProfile
    except ImportError:
        return False
    names = getattr(BridgeProfile, "__dataclass_fields__", {})
    return "capture_dtypes" in names and "matmul_precision" in names


def bridge_runtime_config(profile: ModelProfile, /, *, store_root: str, visibility: str | None = None,
                          representation: str = "svd_hevc", **overrides: Any) -> dict[str, Any]:
    """``--reelcache-config`` JSON for a profile: the bridge profile inline (``profile_fields``) plus options.

    The CausVid ``full_history`` and ``window_21_frames`` bridge profiles share the profile directory path, so
    they cannot both be registered process-wide; a worker serves the policy named here.

    A LongLive 1.0 prompt-switch recache is archived as ONE node whose all-intra pack has one canvas per
    re-materialized chunk (3 for the 9 non-sink frames). The Step 6 archive pipeline stages every canvas of a node
    before it can encode it, so its host staging pool (``max_staged_canvases``, bridge default 2) must hold at least
    that many canvases or the fitter waits forever (observed: job 4395902). The profile therefore asks for
    ``max(2, recache chunks)`` unless the caller overrides it.
    """
    if representation not in ("raw", "raw_offload", "svd_hevc", "svd", "hevc", "latents_only"):  # bridge validates too
        raise ValueError("representation must be 'raw', 'raw_offload', 'svd_hevc', 'svd', 'hevc' or 'latents_only'")
    fields = bridge_profile_fields(profile, visibility, extended=bridge_accepts_extended_fields())
    config = {"store_root": store_root, "representation": representation,
              "profile_fields": {key: (list(value) if isinstance(value, tuple) else value)
                                 for key, value in fields.items()}}
    for key in overrides:
        if key in ("store_root", "representation", "profile_fields", "profile"):
            raise ValueError(f"{key} is set by bridge_runtime_config itself")
    if representation in ("svd_hevc", "hevc") and profile.recache.trigger == "prompt_switch":  # both pack canvases
        config["max_staged_canvases"] = max(2, -(-profile.recache.recache_latent_frames // profile.chunk_latent_frames))
    config.update(overrides)
    return config


__all__ = ["RECACHE_EPOCH_PREFIX", "REPLAY_SEAM_MARKER", "ResumePlan", "block_conditioning_ids", "block_prompt_texts",
           "bridge_accepts_extended_fields", "bridge_runtime_config", "needs_replay_seam", "pipeline_stages",
           "prompt_digest", "recache_epoch_details",
           "recache_epoch_id", "recache_matches", "replay_schedule", "request_cache_tokens", "resume_plan",
           "segment_of", "switch_recache_restored"]
