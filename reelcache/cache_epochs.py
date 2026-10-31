"""Cache epochs and newest-version-on-path history resolution.

A lineage's ancestor path (root first) determines which KV version backs each
attended token range. Chunk nodes publish the KV of their own new frames.
A ``recache`` node (LongLive 1.0 prompt switch) adds no frames and publishes a
``kv_recache`` :class:`~reelcache.types.CacheEpoch` with new KV for earlier frames;
a ``segment`` node (LongCat-Video) publishes a ``longcat_segment`` epoch whose
single segment-local range replaces, never extends, the previous segment's.
Restore resolves every attended sub-range to the newest node on the path that
covers it; nodes before an epoch keep resolving to the versions they had.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from .errors import ConfigurationError, StateError
from .types import CacheEpoch, ChunkNode, MatrixKey


@dataclass(frozen=True)
class CoverageEntry:
    token_start: int
    token_end: int
    chunk_id: str
    records: tuple[tuple[object, str], ...]
    source: str
    node_ref: str
    epoch_id: str
    packet_ref: str | None = None  # the range's node pack when its node/epoch records it (packed commits)


@dataclass(frozen=True)
class ResolvedPiece:
    """One attended sub-range and the record set (and entry span) that backs it."""

    token_start: int
    token_end: int
    entry: CoverageEntry

    @property
    def offset(self) -> int:
        return self.token_start - self.entry.token_start

    def record(self, key: MatrixKey, stream: str | None = None) -> str:
        entry = key if stream is None else (key, stream)
        for candidate, ref in self.entry.records:
            if candidate == entry:
                return ref
        raise StateError(f"resolved range lacks a record for {entry}")

    def to_dict(self) -> dict:
        return {"token_start": self.token_start, "token_end": self.token_end,
                "entry_start": self.entry.token_start, "entry_end": self.entry.token_end,
                "chunk_id": self.entry.chunk_id, "source": self.entry.source,
                "node": self.entry.node_ref, "epoch_id": self.entry.epoch_id}


def coverage(node_ref: str, node: ChunkNode, epoch: CacheEpoch) -> list[CoverageEntry]:
    """The KV ranges a node itself publishes (empty for roots)."""
    if node.node_kind == "chunk":
        return [CoverageEntry(node.token_start, node.token_end, node.chunk_id, node.matrix_records,
                              "chunk", node_ref, node.cache_epoch_id, node.packet_ref)]
    if node.node_kind in ("recache", "segment"):
        return [CoverageEntry(item.token_start, item.token_end, item.chunk_id, item.records, "epoch",
                              node_ref, epoch.epoch_id, item.packet_ref) for item in epoch.ranges]
    return []


def resolve(path: Sequence[tuple[str, ChunkNode, CacheEpoch]],
            ranges: Iterable[tuple[int, int]]) -> list[ResolvedPiece]:
    """Resolve each attended ``[start, end)`` to the newest covering entry on ``path``.

    ``path`` is ordered root first. Every requested token must be covered;
    uncovered history raises StateError rather than being zero-filled.
    """
    entries_by_node = [coverage(ref, node, epoch) for ref, node, epoch in path]
    pieces: list[ResolvedPiece] = []
    for start, end in ranges:
        if type(start) is not int or type(end) is not int or not 0 <= start < end:
            raise ConfigurationError("attended ranges must be nonempty nonnegative integer intervals")
        uncovered = [(start, end)]
        for entries in reversed(entries_by_node):
            if not uncovered:
                break
            remaining = []
            for low, high in uncovered:
                cursor_pieces = []
                for entry in entries:
                    left, right = max(low, entry.token_start), min(high, entry.token_end)
                    if left < right:
                        cursor_pieces.append((left, right, entry))
                cursor_pieces.sort(key=lambda item: item[0])
                cursor = low
                for left, right, entry in cursor_pieces:
                    if left > cursor:
                        remaining.append((cursor, left))
                    pieces.append(ResolvedPiece(left, right, entry))
                    cursor = max(cursor, right)
                if cursor < high:
                    remaining.append((cursor, high))
            uncovered = remaining
        if uncovered:
            raise StateError(f"attended history {uncovered} is not covered by the node's ancestor path")
    pieces.sort(key=lambda piece: piece.token_start)
    for left, right in zip(pieces, pieces[1:]):
        if left.token_end > right.token_start:
            raise ConfigurationError("attended ranges overlap; request each token once")
    return pieces
