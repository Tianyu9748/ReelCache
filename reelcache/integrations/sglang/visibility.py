"""Visibility plans: which archived token ranges the next chunk attends to, and where they live.

The slot placement comes from SGLang's own cache logic replayed on a position
shadow (``CausalSelfAttentionKVCache.replay_layout``); this module maps every
attended slot span to the archived range holding those tokens (the newest
covering node on the path, so a recache epoch overrides the chunk that
originally produced those frames, on this path only) and to their original
RoPE positions. Physical slots are not global positions: sinks, pinned shot
sinks and the rolling window are explicit spans here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from ...errors import IntegrityError
from .nodes import coverage


@dataclass(frozen=True)
class TokenSpan:
    """Tokens ``[node_offset, node_offset + length)`` of one archived range, installed at ``slot_start``."""

    node_ref: str
    chunk_index: int
    stream: str
    node_offset: int
    length: int
    slot_start: int
    token_start: int  # global position of the first token
    rope_start_frame: float  # temporal RoPE position of the first frame of the span
    frames: int
    role: str  # "sink" | "pinned" | "window"
    range_tokens: int  # length of the whole archived range the span reads from
    source: str  # "chunk" | "recache"

    def to_dict(self) -> dict[str, Any]:
        return {"node": self.node_ref, "chunk": self.chunk_index, "stream": self.stream,
                "node_offset": self.node_offset, "length": self.length, "slot_start": self.slot_start,
                "token_start": self.token_start, "rope_start_frame": self.rope_start_frame, "frames": self.frames,
                "role": self.role, "range_tokens": self.range_tokens, "source": self.source}


@dataclass(frozen=True)
class VisibilityPlan:
    """Attended history of one CFG stream for the next chunk, plus the cache cursors to install."""

    stream: str
    next_chunk: tuple[int, int]
    layout: Any  # sglang CausalKVCacheLayout (plain dataclass; kept opaque here)
    spans: tuple[TokenSpan, ...]
    dead_slots: tuple[tuple[int, int], ...]  # live slots the next chunk never reads (zero-filled)

    @property
    def attended_tokens(self) -> int:
        return sum(span.length for span in self.spans)

    def nodes(self) -> list[str]:
        seen = []
        for span in self.spans:
            if span.node_ref not in seen:
                seen.append(span.node_ref)
        return seen

    def placement(self) -> list[tuple[int, int, int]]:
        """``(token_start, length, slot_start)`` of every span (identical plans install identically)."""
        return [(span.token_start, span.length, span.slot_start) for span in self.spans]

    def to_dict(self) -> dict[str, Any]:
        layout = self.layout
        return {
            "stream": self.stream, "next_chunk": list(self.next_chunk),
            "cursors": {"global_end": layout.global_end_index, "local_end": layout.local_end_index,
                        "pinned_start": layout.pinned_start, "pinned_len": layout.pinned_len},
            "cache": {"cache_size": layout.cache_size, "sink_tokens": layout.sink_tokens,
                      "global_sink_tokens": layout.global_sink_tokens,
                      "attention_window_size": layout.attention_window_size},
            "spans": [span.to_dict() for span in self.spans],
            "dead_slots": [list(item) for item in self.dead_slots],
            "attended_tokens": self.attended_tokens,
        }


def _role(layout: Any, slot: int) -> str:
    if layout.pinned_len > 0 and layout.pinned_start <= slot < layout.pinned_start + layout.pinned_len:
        return "pinned"
    sink = max(layout.global_sink_tokens, layout.sink_tokens)
    return "sink" if slot < sink else "window"


def build_visibility_plan(stream: str, layout: Any, nodes: Sequence[Any], *, tokens_per_frame: int,
                          next_chunk: tuple[int, int]) -> VisibilityPlan:
    """Split the layout's attended slot spans at archived-range boundaries.

    ``nodes`` are the path's chunk and recache nodes in order; each attended
    token resolves to the newest archived range covering it.
    """
    if layout.attended is None:
        raise IntegrityError("visibility planning needs the attended spans of the next chunk")
    items = coverage(nodes)
    spans = []
    for span in layout.attended:
        cursor = 0
        while cursor < span.length:
            position = span.global_start + cursor
            index = next((i for i in range(len(items) - 1, -1, -1)
                          if items[i][1].token_start <= position < items[i][1].token_end), None)
            if index is None:
                raise IntegrityError(f"attended token {position} is not covered by any archived range")
            node, item = items[index]
            offset = position - item.token_start
            newer = [other.token_start for _, other in items[index + 1:]
                     if position < other.token_start < item.token_end]
            length = min(span.length - cursor, item.token_end - position, *(start - position for start in newer))
            if offset % tokens_per_frame or length % tokens_per_frame:
                raise IntegrityError("attended spans must cover whole latent frames")
            slot = span.local_start + cursor
            spans.append(TokenSpan(node.ref, item.chunk_index, stream, offset, length, slot, position,
                                   item.rope_start_frame + offset // tokens_per_frame, length // tokens_per_frame,
                                   _role(layout, slot), item.num_tokens, node.kind))
            cursor += length
    attended = {slot for span in layout.attended for slot in range(span.local_start, span.local_end)}
    dead, start = [], None
    for slot in range(layout.local_end_index + 1):
        is_dead = slot < layout.local_end_index and slot not in attended
        if is_dead and start is None:
            start = slot
        elif not is_dead and start is not None:
            dead.append((start, slot))
            start = None
    return VisibilityPlan(stream, tuple(next_chunk), layout, tuple(spans), tuple(dead))


def cache_appends(nodes: Sequence[Any]) -> list[tuple[int, int, bool]]:
    """Clean chunk appends in order (recache nodes rewrite existing slots and append nothing)."""
    return [(node.token_start, node.num_tokens, node.scene_cut) for node in nodes if node.kind == "chunk"]


def check_cache_state(layout: Any, recorded: Mapping[str, Any], stream: str) -> None:
    """The replayed cursors must equal the native cache cursors recorded at commit time."""
    expected = recorded.get(stream)
    if expected is None:
        raise IntegrityError(f"the archived chunk has no cache cursor record for stream {stream!r}")
    actual = {"global_end": layout.global_end_index, "local_end": layout.local_end_index,
              "pinned_start": layout.pinned_start, "pinned_len": layout.pinned_len}
    if {key: expected[key] for key in actual} != actual:
        raise IntegrityError(f"replayed cache cursors {actual} differ from the native cursors {dict(expected)}")


__all__ = ["TokenSpan", "VisibilityPlan", "build_visibility_plan", "cache_appends", "check_cache_state"]
