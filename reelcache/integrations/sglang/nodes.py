"""Plain-data view of committed session nodes, shared by both archive backends.

A session is a branch of the archive state tree. Its nodes are a root
(conditioning identity), ``chunk`` nodes (one clean causal block, every CFG
stream) and ``recache`` nodes (a cache epoch re-materializing earlier chunks'
KV at their original positions, no new frames). Visibility planning and
restore only need this view; how a backend stores nodes (RAW manifests or
Step 6 lineage nodes) stays inside the backend.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping

import torch

from ...errors import IntegrityError
from ...run_telemetry import split_run_telemetry
from ...types import json_copy

RECORD_KEY = "bridge.record"
LATENTS_KEY = "latents"
GENERATOR_PREFIX = "generator."


@dataclass(frozen=True)
class CoverageRange:
    """Global tokens ``[token_start, token_start + num_tokens)`` held by a node (one archived chunk's range)."""

    token_start: int
    num_tokens: int
    chunk_index: int
    rope_start_frame: float

    @property
    def token_end(self) -> int:
        return self.token_start + self.num_tokens


@dataclass(frozen=True)
class BridgeNode:
    """A committed node as plain data (``kind``: ``root`` | ``chunk`` | ``recache``)."""

    ref: str
    kind: str
    parent_ref: str | None
    root_ref: str
    chunk_index: int
    start_frame: int
    num_frames: int
    token_start: int
    num_tokens: int
    rope_start_frame: float
    scene_cut: bool
    shot_index: int
    streams: tuple[str, ...]
    record: Mapping[str, Any]
    cache_state: Mapping[str, Any]
    ranges: tuple[CoverageRange, ...] = ()

    @property
    def token_end(self) -> int:
        return self.token_start + self.num_tokens


def chunk_node(ref: str, parent_ref: str | None, root_ref: str, record: Mapping[str, Any]) -> BridgeNode:
    try:
        node = BridgeNode(
            ref, "chunk", parent_ref, root_ref, int(record["chunk_index"]), int(record["start_frame"]),
            int(record["num_frames"]), int(record["token_start"]), int(record["num_tokens"]),
            float(record["rope_start_frame"]), bool(record["scene_cut"]), int(record["shot_index"]),
            tuple(record["streams"]), record, record.get("cache_state", {}),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise IntegrityError(f"{ref}: incomplete bridge chunk record") from exc
    return BridgeNode(**{**node.__dict__, "ranges": (CoverageRange(node.token_start, node.num_tokens,
                                                                   node.chunk_index, node.rope_start_frame),)})


def recache_node(ref: str, parent_ref: str | None, root_ref: str, record: Mapping[str, Any],
                 ranges: tuple[CoverageRange, ...], frontier: int) -> BridgeNode:
    return BridgeNode(ref, "recache", parent_ref, root_ref, -1, 0, 0, frontier, 0, 0.0, False, 0,
                      tuple(record.get("streams", ())), record, record.get("cache_state", {}), tuple(ranges))


def root_node(ref: str, record: Mapping[str, Any]) -> BridgeNode:
    return BridgeNode(ref, "root", None, ref, -1, 0, 0, 0, 0, 0.0, False, 0, (), record, {}, ())


def coverage(nodes) -> list[tuple[BridgeNode, CoverageRange]]:
    """Every archived range of a path in commit order (newest last; recache ranges after their chunks)."""
    return [(node, item) for node in nodes for item in node.ranges]


#: Bridge record fields that are run provenance, not content: kept in the unhashed telemetry sidecar so identical
#: sessions produce identical node ids (besides run telemetry such as ``*_seconds``, see ``reelcache.run_telemetry``).
RUN_FIELDS = ("request_id",)
TELEMETRY_KEY = "bridge"


def split_record(record: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """``(content, telemetry)`` of a bridge record: content is hashed into node ids, telemetry is not.

    The request's ``conditioning`` (user content, compared through its digest) is never split."""
    record = json_copy(dict(record), name="bridge record")
    conditioning = record.pop("conditioning", None)
    content, telemetry = split_run_telemetry(record)
    if conditioning is not None:
        content["conditioning"] = conditioning
    run = {key: content.pop(key) for key in RUN_FIELDS if key in content}
    if run:
        telemetry = {**(telemetry or {}), **run}
    return content, telemetry


def annotate(store: Any, ref: str, telemetry: Mapping[str, Any] | None) -> None:
    """Merge bridge run telemetry into an object's unhashed sidecar (keeps what the core recorded there)."""
    if not telemetry:
        return
    existing = dict(store.telemetry(ref) or {})
    existing[TELEMETRY_KEY] = {**dict(existing.get(TELEMETRY_KEY) or {}), **json_copy(dict(telemetry),
                                                                                      name="bridge telemetry")}
    store.put_telemetry(ref, existing)


def record_tensor(record: Mapping[str, Any]) -> torch.Tensor:
    data = json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
    return torch.tensor(list(data), dtype=torch.uint8)


def tensor_record(tensor: torch.Tensor) -> dict[str, Any]:
    if not isinstance(tensor, torch.Tensor) or tensor.dtype != torch.uint8 or tensor.dim() != 1:
        raise IntegrityError("bridge record must be a 1-D uint8 tensor")
    return json.loads(bytes(tensor.tolist()).decode())


# -- workload v2 (W1b): session trees ---------------------------------------------------------------------------------
#: Ownership refs of ``tree_retention="full"``: ``tree/<session>/<generation_index:08d>`` -> the generated chunk node.
TREE_REF_PREFIX = "tree/"
#: The raw-tier holder that keeps every owned node's raw entries of a session (session ids never contain "/").
TREE_HOLDER_SUFFIX = "/tree"
#: W1.3 (``tree_switch="detach"``): a detached branch is ``<session>~d<n>-<hex8>`` -- one name component (a lineage
#: name); "~" never occurs in a session id (``NAME_PATTERN``), so the mark also hides the branch from session listings.
#: Its temporary head ref: ``lineages/<branch>`` (lineage arms) or ``sglang/detached/<branch>`` (raw_offload).
DETACH_MARK = "~d"
DETACHED_REF_PREFIX = "sglang/detached/"
#: W1.3 (F2): a closed detached branch whose last published node no tree ref covers (e.g. a recache whose chunk failed)
#: keeps it rooted under ``sglang/detached-keep/<session>/<branch>``, removed with the whole tree.
DETACHED_KEEP_PREFIX = "sglang/detached-keep/"


def detached_keep_ref(session_id: str, branch: str) -> str:
    tree_prefix(session_id)  # validates the session id
    return f"{DETACHED_KEEP_PREFIX}{session_id}/{branch}"


def detached_branch(session_id: str, sequence: int, token: str) -> str:
    """The id of a session's ``sequence``-th detached branch (``token``: unique across worker restarts)."""
    tree_prefix(session_id)  # validates the session id
    return f"{session_id}{DETACH_MARK}{int(sequence)}-{token}"


def is_detached_branch(name: str) -> bool:
    return isinstance(name, str) and DETACH_MARK in name


def detached_session(branch: str) -> str:
    """The session a detached branch id belongs to."""
    return branch.split(DETACH_MARK, 1)[0]


def tree_prefix(session_id: str) -> str:
    if not isinstance(session_id, str) or not session_id or "/" in session_id or "\x00" in session_id:
        raise IntegrityError("session ids are nonempty names without '/'")
    return f"{TREE_REF_PREFIX}{session_id}/"


def tree_ref(session_id: str, generation_index: int) -> str:
    """The ownership ref of one generated node of a session tree."""
    if type(generation_index) is not int or not 1 <= generation_index < 10 ** 8:
        raise IntegrityError(f"generation indices are integers in [1, 1e8), got {generation_index!r}")
    return f"{tree_prefix(session_id)}{generation_index:08d}"


def tree_entries(refs: Mapping[str, str], session_id: str) -> dict[int, str]:
    """``{generation_index: node_ref}`` of a session from ``store.list_refs(tree_prefix(session_id))``."""
    prefix = tree_prefix(session_id)
    entries = {}
    for name, node in refs.items():
        suffix = name[len(prefix):] if name.startswith(prefix) else ""
        if len(suffix) != 8 or not suffix.isdigit():
            raise IntegrityError(f"{name!r} is not a session tree ref of {session_id!r}")
        entries[int(suffix)] = node
    return entries


def partial_recache_node(node: BridgeNode) -> bool:
    """Whether a recache node is a non-final range of a chained epoch (the RAW layout: one node per range): its
    epoch re-materialized more of the window after it, so it is no state generation had and never a checkpoint."""
    if node.kind != "recache":
        return False
    written = node.record.get("written") or []
    ends = [int(start) + int(length) for start, length in written]
    if not ends or not node.ranges:
        return False
    return max(item.token_end for item in node.ranges) < max(ends)


# -- workload v2 (W1b): the exact recompute arm ----------------------------------------------------------------------
def replay_steps(nodes) -> list[dict[str, Any]]:
    """The clean forwards that rebuild a path's native cache, in native order (root excluded): one ``chunk`` step
    per chunk node (its context forward under its block conditioning) and one ``recache`` step per cache epoch (the
    epoch's chained RAW range nodes merged), each with the record fields a stage needs to rerun it and to check it.
    Plain data; the stage supplies each block's conditioning from the request (validated against these ids)."""
    steps: list[dict[str, Any]] = []
    for node in nodes:
        record = node.record
        if node.kind == "chunk":
            steps.append({"kind": "chunk", "block": node.chunk_index, "node": node.ref,
                          "start_frame": node.start_frame, "num_frames": node.num_frames,
                          "token_start": node.token_start, "num_tokens": node.num_tokens,
                          "rope_start_frame": float(node.rope_start_frame), "scene_cut": bool(node.scene_cut),
                          "shot_index": int(node.shot_index), "conditioning": record.get("block_conditioning")})
        elif node.kind == "recache":
            epoch = record.get("epoch")
            last = steps[-1] if steps else None
            ranges = [[item.token_start, item.num_tokens, item.chunk_index] for item in node.ranges]
            if last is not None and last["kind"] == "recache" and last["epoch"] == epoch:
                last["ranges"].extend(ranges)
                last["nodes"].append(node.ref)
                continue
            steps.append({"kind": "recache", "epoch": epoch, "nodes": [node.ref],
                          "precedes_block": record.get("precedes_block"),
                          "window": json_copy(record.get("window"), name="recache window"),
                          "written": [list(item) for item in record.get("written") or []], "ranges": ranges,
                          "conditioning": record.get("block_conditioning"),
                          "details": json_copy(record.get("epoch_details") or record.get("details") or {},
                                               name="recache details")})
    return steps


__all__ = ["BridgeNode", "CoverageRange", "GENERATOR_PREFIX", "LATENTS_KEY", "RECORD_KEY", "RUN_FIELDS",
           "TELEMETRY_KEY", "TREE_HOLDER_SUFFIX", "TREE_REF_PREFIX", "annotate", "chunk_node", "coverage",
           "partial_recache_node", "recache_node", "record_tensor", "replay_steps", "root_node", "split_record",
           "tensor_record", "tree_entries", "tree_prefix", "tree_ref"]
