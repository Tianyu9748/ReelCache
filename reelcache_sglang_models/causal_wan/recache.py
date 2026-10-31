"""LongLive 1.0 prompt-switch KV-recache semantics as pure index math (no torch, no SGLang).

Upstream reference: NVlabs/LongLive@c6a27e771e803b1c12ffa28b82ba69b42472020e,
``LongLive1.0/pipeline/interactive_causal_inference.py:34-96`` (``_recache_after_switch``) and the KV write and
attention logic of ``LongLive1.0/wan/modules/causal_model.py:206-347``.

At a switch before chunk n (``current_start_frame`` = frames generated so far) the pipeline

1. zeroes K/V when ``global_sink`` is false (indices are kept), and resets every cross-attention cache;
2. runs ONE clean forward (timestep ``context_noise``) over the last ``min(local_attn_size, current_start_frame)``
   frames under the NEW prompt, with ``current_start`` at the first of them. In the attention the span ends at
   ``global_end_index``, so ``is_recompute = current_end <= global_end and current_start > 0`` holds and the
   write starts at ``max(local_start, sink_tokens)`` (the sink is protected) unless ``sink_recache_after_switch``
   (= not global_sink). Keys/values seen by the recache = sink + rewritten window, no mask: the pass is not
   causal inside the window. Indices are not moved (``_apply_cache_updates`` skips recomputes);
3. resets the cross-attention caches again.

``plan_recache`` derives the buffer slots and global frames that are rewritten, kept and discarded;
``RecacheRecord`` is the plain-data account of one recache, shaped so Step 14 can publish it as a
``CacheEpoch`` (cause, parent, re-materialized token ranges).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

RECACHE_FORWARD_KIND = "kv_recache"
CLEAN_COMMIT_FORWARD_KIND = "clean_commit"


@dataclass(frozen=True)
class RecachePlan:
    current_start_frame: int
    num_recache_frames: int
    recache_start_frame: int
    frame_seq: int
    sink_tokens: int
    is_recompute: bool
    sink_recache: bool
    zero_kv_before: bool
    local_start: int
    local_end: int
    write_start: int
    global_end_before: int
    local_end_before: int

    @property
    def recompute_tokens(self) -> tuple[int, int]:
        """Global token range fed through the recache forward."""
        return (self.recache_start_frame * self.frame_seq, self.current_start_frame * self.frame_seq)

    @property
    def written_slots(self) -> tuple[int, int]:
        return (self.write_start, self.local_end)

    @property
    def written_global_frames(self) -> tuple[int, int]:
        offset = (self.write_start - self.local_start) // self.frame_seq
        return (self.recache_start_frame + offset, self.current_start_frame)

    @property
    def discarded_global_frames(self) -> tuple[int, int]:
        """Frames recomputed by the forward but not written back (c7 in the report example)."""
        return (self.recache_start_frame, self.written_global_frames[0])

    @property
    def kept_sink_slots(self) -> tuple[int, int]:
        return (0, self.write_start) if self.write_start > 0 else (0, 0)

    def to_dict(self) -> dict:
        out = asdict(self)
        out.update(recompute_tokens=list(self.recompute_tokens), written_slots=list(self.written_slots),
                   written_global_frames=list(self.written_global_frames),
                   discarded_global_frames=list(self.discarded_global_frames),
                   kept_sink_slots=list(self.kept_sink_slots))
        return out


def plan_recache(*, current_start_frame: int, local_attn_frames: int, sink_frames: int, frame_seq: int,
                 global_end_tokens: int, local_end_tokens: int, global_sink: bool) -> RecachePlan | None:
    """Upstream ``_recache_after_switch`` + the recompute branch of ``CausalWanSelfAttention.forward``.

    Returns ``None`` for a switch at frame 0 (upstream returns before any forward).
    """
    if current_start_frame < 0 or frame_seq <= 0 or sink_frames < 0:
        raise ValueError("bad recache arguments")
    if current_start_frame == 0:
        return None
    num = current_start_frame if local_attn_frames == -1 else min(local_attn_frames, current_start_frame)
    start_frame = current_start_frame - num
    current_start = start_frame * frame_seq
    num_new = num * frame_seq
    current_end = current_start + num_new
    if current_end > global_end_tokens:
        raise ValueError(f"recache span ends at token {current_end} beyond the generated {global_end_tokens}")
    is_recompute = current_end <= global_end_tokens and current_start > 0
    local_end = local_end_tokens + current_end - global_end_tokens
    local_start = local_end - num_new
    if local_start < 0:
        raise ValueError(f"recache span ({num} frames) exceeds the KV buffer content ({local_end_tokens} tokens)")
    sink_tokens = sink_frames * frame_seq
    sink_recache = not global_sink
    write_start = max(local_start, sink_tokens) if is_recompute else local_start
    if sink_recache:
        write_start = local_start
    return RecachePlan(current_start_frame=current_start_frame, num_recache_frames=num, recache_start_frame=start_frame,
                       frame_seq=frame_seq, sink_tokens=sink_tokens, is_recompute=is_recompute,
                       sink_recache=sink_recache, zero_kv_before=not global_sink, local_start=local_start,
                       local_end=local_end, write_start=write_start, global_end_before=global_end_tokens,
                       local_end_before=local_end_tokens)


@dataclass
class RecacheRecord:
    """One executed recache (plain data; no tensors). Step 14 publishes it as a ``CacheEpoch``."""
    profile_id: str
    switch_before_chunk: int
    previous_segment: int
    new_segment: int
    plan: dict
    context_timestep: int
    cross_attention_reset: bool
    indices_after: dict = field(default_factory=dict)
    seconds: float | None = None
    forward_kind: str = RECACHE_FORWARD_KIND
    cause: str = "prompt_switch"
    # ReelCache session runs: the archived cache epoch (epoch id, written token ranges, recache window); None otherwise.
    cache_epoch_archive: dict | None = None

    def cache_epoch_fields(self) -> dict:
        """CacheEpoch view (report 3.2): cause and re-materialized token ranges; IDs are assigned by the runtime."""
        written = self.plan["written_global_frames"]
        fs = self.plan["frame_seq"]
        return {"cause": f"{self.cause}:before_chunk_{self.switch_before_chunk}",
                "rematerialized_latent_frames": [written[0], written[1]],
                "rematerialized_tokens": [written[0] * fs, written[1] * fs],
                "recomputed_but_discarded_latent_frames": self.plan["discarded_global_frames"],
                "kept_sink_slots": self.plan["kept_sink_slots"],
                "segments": [self.previous_segment, self.new_segment]}

    def to_dict(self) -> dict:
        out = asdict(self)
        out["cache_epoch"] = self.cache_epoch_fields()
        return out
