"""RAW session archive of the SGLang bridge (``representation="raw"``: the exact reference).

Nodes are immutable manifests in the runtime's object store: a root per
session tree (conditioning identity and geometry), one ``chunk`` node per
committed clean block (every CFG stream together, so a node never exists with
only one stream) and ``recache`` nodes (one per re-materialized chunk range of
a cache epoch). Every matrix is a RAW ``reelcache.tensor.v1`` packet of the
captured normalized pre-RoPE K or V, so a restore reproduces the native cache
bitwise. Commits are synchronous: RAW is a reference point (bitwise parity of
the capture/restore/install path), not a performance path. Session heads are
compare-and-swap refs ``sglang/sessions/<id>``; forks are new heads at an
existing node and share every ancestor object once. Manifests hash content
only (run telemetry such as timings and request ids lives in the store's
unhashed sidecars), so identical sessions produce identical ids.

The lossy representation (``svd_hevc``) uses Step 6 lineages instead
(:mod:`.lineage_archive`); both expose the same backend interface.

Workload v2 (W1b): ``kv_objects=False`` is the exact recompute arm's archive
(``representation="latents_only"``): nodes keep latents, generator and
execution state and their bridge records, no KV packets; a
:class:`~reelcache.raw_tier.RawChunkTier` (``raw_tier``; the lineage archive's
budget fields) keeps captured K/V as GPU / host raw copies in front, and a
restore that misses them replays the path in the stage. Session-tree refs
bound with :meth:`RawSessionArchive.bind_publication_refs` are set in the
transaction that moves the session head; :meth:`RawSessionArchive.set_tree_ownership`
lets a rewind reach any node the session owns.

Workload v2 (CONTRACT §10): ``write_behind=True`` is ``representation="raw_offload"``:
the same RAW nodes, written behind the raw tier by one ordered background writer.
A commit adopts the captured K/V as a raw-tier entry (no copy) and returns; it
waits only while the tier's budgets are full of unwritten entries (the tier's
admission, recorded with its cause). The writer writes each node's packets
(from the entry's host copy, else a D2H copy through a pinned staging buffer
on its own stream), publishes the session head (compare-and-swap, with bound
refs) and only then marks the entry published, i.e. evictable. Restores take
every attended range from its best copy: GPU raw, host raw, else the store
(per-span tier records). ``store_io="nocache"`` drops object file pages after
durable writes and reads (``ObjectStore.set_io_policy``).
"""

from __future__ import annotations

import os
import sqlite3
import threading
import time
from collections import deque
from typing import Any, Iterable, Mapping, Sequence

import torch

from ...archive_pipeline import ArchiveFailure
from ...codecs import decode_tensor, encode_tensor
from ...errors import CapacityError, ConfigurationError, IntegrityError, PendingError, QuarantinedError, StateError
from ...lineage_commit import frame_free, raisable
from ...manager import KVRuntime
from ...run_telemetry import split_run_telemetry
from ...serialization import deserialize_tensors, serialize_tensors
from ...types import CodecSpec, EncodedTensor, MatrixKey, ModelIdentity, json_copy
from .nodes import (GENERATOR_PREFIX, LATENTS_KEY, BridgeNode, CoverageRange, annotate, chunk_node, recache_node,
                    root_node, split_record)
from .restore import AttentionKVLease, kernel_warmup_shapes, restore_dense, warm_restore_kernels
from .workload_telemetry import offload_tier_label, raw_tier_label, tier_segment

NODE_SCHEMA = "reelcache.sglang.causal_chunk.v2"
PACKET_SCHEMA = "reelcache.sglang.packet.v1"
SESSION_PREFIX = "sglang/sessions/"


def key_label(key: Any) -> str:
    return f"{key.layer}:{key.target}"


class CompletedTicket:
    """A synchronous commit's result (the interface of ``ArchiveTicket``)."""

    def __init__(self, node_ref: str, node_id: str):
        self.node_ref, self.node_id = node_ref, node_id
        self.error = None

    def done(self) -> bool:
        return True

    def result(self, timeout: float | None = None) -> str:
        return self.node_ref


# -- workload v2 (CONTRACT §10): ``raw_offload`` write-behind ---------------------------------------------------------
class WriteBehindTicket:
    """A ``raw_offload`` commit (the interface of ``ArchiveTicket``): resolved once the ordered writer made its node(s)
    durable and moved the session head onto them, or with the session's ``ArchiveFailure``. Plain fields only (the
    bridge keeps them in its request history)."""

    def __init__(self, branch_id: str, node_id: str, token_end: int):
        self.branch_id, self.node_id, self.token_end = branch_id, node_id, token_end
        self.node_ref: str | None = None
        self.error: BaseException | None = None
        self.report: dict[str, Any] = {}
        self.admission: dict[str, Any] | None = None
        self._event = threading.Event()

    def done(self) -> bool:
        return self._event.is_set()

    def result(self, timeout: float | None = None) -> str:
        if not self._event.wait(timeout):
            raise PendingError(f"chunk {self.node_id} is not written yet")
        if self.error is not None:
            raise raisable(self.error)  # a copy: the caller's frames never attach to the kept error
        return self.node_ref

    def _resolve(self, node_ref: str | None = None, error: BaseException | None = None) -> None:
        if not self._event.is_set():
            self.node_ref, self.error = node_ref, frame_free(error)
            self._event.set()


class _WriteSession:
    """Write-behind state of one session: its unretired jobs in commit order and the durable head they chain on."""

    def __init__(self, session_id: str, head: str, frontier: int):
        self.session_id, self.published_ref = session_id, head
        self.jobs: list[_WriteJob] = []
        self.error: BaseException | None = None
        self.published_ids: set[str] = set()
        self.generated_frontier = self.published_frontier = frontier
        # W1.3 (``tree_switch="detach"``): a detached branch's own head ref (None: the session's head ref), the
        # session it was detached from, whether the writer is inside its publishing transaction now, and whether a
        # detach waits for it (the writer starts no further publication meanwhile, F5).
        self.head_name: str | None = None
        self.detached_from: str | None = None
        self.publishing = False
        self.detaching = False


class _WriteJob:
    """One queued commit: its nodes' pieces (a chunk, or one per recache range), each with its raw-tier entry (the
    only copy of the captured K/V until the node is durable); no tensor references of its own."""

    __slots__ = ("session", "kind", "node_id", "root_ref", "pieces", "streams", "execution_state", "ticket",
                 "submitted", "nbytes")

    def __init__(self, session, kind, node_id, root_ref, pieces, streams, execution_state, ticket, nbytes):
        self.session, self.kind, self.node_id, self.root_ref = session, kind, node_id, root_ref
        self.pieces, self.streams, self.execution_state = pieces, streams, execution_state
        self.ticket, self.nbytes = ticket, nbytes
        self.submitted = time.perf_counter()

def _tier_options(tier: Any) -> bool:
    """wl-w1b7: whether a raw tier runs a non-default ``retain`` / ``promote`` option (its records say so)."""
    return tier is not None and (getattr(tier, "retain", "drop") != "drop"
                                 or getattr(tier, "promotion_mode", "off") != "off")


def _lease_origin(lease: Any) -> dict[str, Any]:
    """wl-w1b7 per-chunk fields of a raw-tier hit: where the copy came from (``captured`` at commit, ``store`` a
    promoted store read, ``replay`` a promoted replay), its K form (``pre_rope`` / ``installed``) and whether the entry
    was kept unheld since its last restore (``retained``)."""
    return {"origin": lease.entry.origin, "form": lease.entry.form, "retained": bool(getattr(lease, "retained", False))}


class RawSessionArchive:
    """Exact RAW session archive in a :class:`~reelcache.manager.KVRuntime` store (synchronous)."""

    name = "raw_exact"
    asynchronous = False

    def __init__(self, root: str, model: ModelIdentity, *, device: Any, max_store_bytes: int | None = None,
                 max_resident_bytes: int = 16 << 30, verify_reads: bool = False, audit_on_open: bool | None = False,
                 on_open_error: str = "quarantine", read_ahead_bytes: int | None = None, kv_objects: bool = True,
                 raw_tier: Any = None, write_behind: bool = False, store_io: str = "buffered", config: Any = None):
        """``audit_on_open`` / ``on_open_error`` default to the bridge's serving defaults (``ReelCacheRuntimeConfig``):
        session heads are graph-checked at open (``audit_on_open``: hash-audited) and a damaged one is quarantined
        (:meth:`quarantined`), or fails the open with ``on_open_error="raise"``.

        ``read_ahead_bytes`` (None: ``KVRuntime.HISTORY_READ_AHEAD_BYTES``; 0 disables) bounds the dense payload
        bytes a restore reads ahead, in its consumption order, on the store's read-ahead threads (``store.ReadAhead``:
        the checks of ``get``, so verified reads of several objects hash in parallel); a failed read-ahead object is
        read again where it is used, which raises as before."""
        if read_ahead_bytes is not None and (type(read_ahead_bytes) is not int or read_ahead_bytes < 0):
            raise ConfigurationError("read_ahead_bytes must be a nonnegative integer or None (default)")
        if write_behind and (raw_tier is None or not kv_objects):
            raise ConfigurationError("raw_offload (write_behind) writes RAW objects behind a raw tier")
        os.makedirs(root, exist_ok=True)
        self.runtime = KVRuntime(root, model, encoder=encode_tensor, decoder=decode_tensor,
                                 max_store_bytes=max_store_bytes, max_resident_bytes=max_resident_bytes,
                                 verify_reads=verify_reads, verify_references=False,
                                 audit_on_open=audit_on_open, on_open_error=on_open_error)
        self.store = self.runtime.store
        if store_io != "buffered":  # workload v2: before any object of this archive is read
            try:
                self.store.set_io_policy(store_io, drop_existing=True)
            except BaseException:
                self.runtime.close()
                raise
        self.model = model
        self.root = root
        self.device = torch.device(device)
        self._raw: dict[str, Mapping[str, Any]] = {}
        self._nodes: dict[str, BridgeNode] = {}
        self.read_ahead_bytes = KVRuntime.HISTORY_READ_AHEAD_BYTES if read_ahead_bytes is None else read_ahead_bytes
        self.stats_ = {"commits": 0, "commit_seconds": 0.0, "restores": 0}
        # W1b: the recompute arm stores no KV (``kv_objects=False``); its raw tier (or None) sits in front.
        self.kv_objects = bool(kv_objects)
        if not self.kv_objects:
            self.name = "latents_only"
        self.raw_tier = raw_tier
        self._publication_refs: dict[tuple[str, str], tuple[str, ...]] = {}
        # W1.3 (R4): every access to the bindings (submit, the writer, detach, close) holds this lock, never across a
        # store transaction
        self._bindings_lock = threading.RLock()
        self._owned_prefix: Any = None
        self.tier_stats_ = {"not_admitted": 0, "restores": 0, "misses": 0}
        self._tier_cond = threading.Condition()
        if raw_tier is not None:
            raw_tier.attach(self._tier_cond)  # adoption, publication and waits share one condition (the pipeline's)
        # CONTRACT §10 (raw_offload): the ordered write-behind (all state under ``_tier_cond``)
        self.write_behind = bool(write_behind)
        self.asynchronous = self.write_behind
        if self.write_behind:
            self.name = "raw_offload"
        self._wb_sessions: dict[str, _WriteSession] = {}
        self._wb_jobs: deque = deque()
        self._wb_active: _WriteJob | None = None
        self._wb_thread: threading.Thread | None = None
        self._wb_stop = False
        self._wb_staging: torch.Tensor | None = None  # the writer's pinned D2H staging buffer (one matrix)
        self._wb_streams: dict[str, Any] = {}
        self.write_hook: Any = None  # test seam: ``write_hook(stage, job)`` in the writer thread ("start")
        self.close_timeout = 600.0
        self.wb_stats_ = {"submitted": 0, "written": 0, "nodes": 0, "failed": 0, "cancelled": 0, "bytes": 0,
                          "write_seconds": 0.0, "d2h_seconds": 0.0, "queue_seconds": 0.0, "max_queue_seconds": 0.0,
                          "peak_queued": 0, "sources": {"host_raw": 0, "gpu_raw": 0}, "admission_waits": 0,
                          "admission_wait_seconds": 0.0, "admission_causes": {}, "sync_fallbacks": 0}
        # W1.1: with a raw tier and the bridge configuration (raw_offload, latents_only), the lineage archive's
        # device-memory seams, shared (its functions bound to this archive, not copied): the decode and restore
        # memory-critical sections (pressure response on entry: least recently used GPU raw copies demoted, then the
        # registered trims; OOM relief inside), the request-start / pre-restore relief and the in-place restore query.
        # No fitter: never an archival deferral, pause or decode entry guard. F6 raw (no tier) has none of them.
        if raw_tier is not None and config is not None:
            import types

            from .lineage_archive import LineageSessionArchive

            self.config, self.fitter, self.raw_install_in_place = config, None, True
            for name in ("memory_critical", "relieve_pressure", "relieve_device_memory", "restore_allocates"):
                setattr(self, name, types.MethodType(getattr(LineageSessionArchive, name), self))
            self._oom_relief = LineageSessionArchive._oom_relief  # a staticmethod: the plain function
        self.read_ahead_stats_ = {"windows": 0, "taken": 0, "taken_bytes": 0, "failed": 0, "dropped": 0,
                                  "plan_failures": 0}
        # Session heads found damaged at open (``on_open_error="quarantine"``): session id -> (head, error).
        self._quarantine: dict[str, tuple[str, str]] = {}
        try:
            for name, ref in self.store.list_refs(SESSION_PREFIX).items():
                try:
                    (self.store.verify if self.runtime.audit_on_open else self.store.check)(ref)
                except IntegrityError as exc:
                    if on_open_error != "quarantine" or isinstance(exc.__cause__, sqlite3.Error):
                        raise  # "raise": a damaged head fails the open, like the lineage archive's
                    self._quarantine[name[len(SESSION_PREFIX):]] = (ref, f"{type(exc).__name__}: {exc}")
        except BaseException:
            self.runtime.close()
            raise

    # -- sessions ---------------------------------------------------------------------------
    @staticmethod
    def _ref_name(session_id: str) -> str:
        if not isinstance(session_id, str) or not session_id or "/" in session_id or "\x00" in session_id:
            raise ConfigurationError("session ids are nonempty names without '/'")
        return SESSION_PREFIX + session_id

    def head(self, session_id: str) -> str | None:
        return self.store.get_ref(self._ref_name(session_id))

    def sessions(self) -> dict[str, str]:
        """Session heads, without the quarantined ones (see :meth:`quarantined`)."""
        return {name[len(SESSION_PREFIX):]: ref for name, ref in self.store.list_refs(SESSION_PREFIX).items()
                if name[len(SESSION_PREFIX):] not in self._quarantine}

    def quarantined(self) -> dict[str, str]:
        """``{session_id: error}`` of sessions whose head was found damaged when the store was opened."""
        return {session_id: error for session_id, (_, error) in self._quarantine.items()}

    def describe(self) -> dict[str, Any]:
        described = {"backend": self.name, "quarantined": self.quarantined()}
        if not self.kv_objects or self.raw_tier is not None:  # W1b recompute arm: what it keeps
            described.update(kv_objects=self.kv_objects, raw_tier=None if self.raw_tier is None else {
                "gpu_raw_budget_bytes": self.raw_tier.gpu_budget(), "host_raw_budget_bytes": self.raw_tier.host_budget(),
                "host_derivation": dict(self.raw_tier.host_derivation),
                **({"gpu_only": True} if getattr(self.raw_tier, "gpu_only", False) else {})})  # W1.4.1
        if self.write_behind or self.store.io_policy != "buffered":  # workload v2 (CONTRACT §10)
            described.update(write_behind=self.write_behind, store_io=self.store.io_stats())
        return described

    def create_session(self, session_id: str, *, record: Mapping[str, Any], streams: Sequence[str],
                       replace: bool = False, telemetry: Mapping[str, Any] | None = None) -> str:
        name = self._ref_name(session_id)
        current = self.store.get_ref(name)
        if current is not None and not replace:
            raise StateError(f"causal KV session {session_id!r} already exists")
        if self.write_behind:  # raw_offload: a replaced session's unwritten commits never publish
            self.abandon(session_id)
        content, run = split_record({**dict(record), "streams": list(streams)})
        with self.store.transaction():
            root_ref = self.store.put_manifest({
                "type": "sglang_causal_root", "schema": NODE_SCHEMA, "model": self.model.to_dict(),
                "record": content,
            })
        self.annotate(root_ref, {**(run or {}), **dict(telemetry or {})})
        self.store.set_ref(name, root_ref, expected=current)
        self._quarantine.pop(session_id, None)
        self.raw_release(session_id)
        return root_ref

    def annotate(self, ref: str, telemetry: Mapping[str, Any] | None) -> None:
        """Attach bridge run telemetry (request ids, configuration) to a node's unhashed sidecar."""
        annotate(self.store, ref, telemetry)

    def fork(self, node_ref: str, session_id: str) -> None:
        self.load(node_ref)
        self.store.set_ref(self._ref_name(session_id), node_ref, expected=None)
        self.raw_hold(session_id, node_ref)

    def rewind(self, session_id: str, node_ref: str) -> None:
        """Move the head back to an ancestor (compare-and-swap); with tree ownership (``set_tree_ownership``) also to
        any node on the path of one of the session's ownership refs (e.g. an old branch)."""
        name = self._ref_name(session_id)
        head = self.store.get_ref(name)
        if head is None:
            raise StateError(f"unknown causal KV session {session_id!r}")
        if self.write_behind:  # raw_offload: the head moves outside the writer (its write state must be released)
            self._wb_forget(session_id, 0)
        if node_ref not in {node.ref for node in self.path(head)}:
            prefix = None if self._owned_prefix is None else self._owned_prefix(session_id)
            if prefix is None:
                raise StateError("rewind target is not an ancestor of the session head")
            owners = set(self.store.list_refs(prefix).values())
            if node_ref not in owners and not any(node_ref in {node.ref for node in self.path(owner)}
                                                  for owner in owners):
                raise StateError("rewind target is neither an ancestor of the session head nor a node the session "
                                 "owns")
        self.store.set_ref(name, node_ref, expected=head)
        self.raw_hold(session_id, node_ref)

    def delete(self, session_id: str) -> None:
        name = self._ref_name(session_id)
        if self.write_behind:
            self.abandon(session_id)
        current = self.store.get_ref(name)
        if current is not None:
            self.store.delete_ref(name, expected=current)
        self._quarantine.pop(session_id, None)
        self.raw_release(session_id, drop=True)
        self.raw_forget(session_id)

    # -- workload v2 (W1b): session trees ------------------------------------------------------------------------------
    def set_tree_ownership(self, prefix: Any) -> None:
        """``prefix(session_id) -> ref prefix | None``: rewinds also accept nodes on the paths of the refs under it."""
        if prefix is not None and not callable(prefix):
            raise ConfigurationError("the tree ownership prefix is a callable or None")
        self._owned_prefix = prefix

    def bind_publication_refs(self, session_id: str, node_id: str, names: Sequence[str]) -> None:
        """Refs set to the node that commit ``node_id`` publishes, in the transaction that moves the session head
        (compare-and-swap: each must be absent or already name that node, else the commit fails)."""
        names = tuple(names)
        for item in names:
            if not isinstance(item, str) or not item or "\x00" in item or item.startswith(SESSION_PREFIX):
                raise ConfigurationError(f"invalid publication ref name {item!r}")
        with self._bindings_lock:
            if names:
                self._publication_refs[(session_id, node_id)] = names
            else:
                self._publication_refs.pop((session_id, node_id), None)

    def unbind_publication_refs(self, session_id: str) -> int:
        with self._bindings_lock:
            keys = [key for key in self._publication_refs if key[0] == session_id]
            for key in keys:
                del self._publication_refs[key]
            return len(keys)

    def _publish_head(self, session_id: str, node_id: str, node_ref: str, expected: str, *,
                      name: str | None = None) -> None:
        """Move the session head (compare-and-swap) and set the commit's bound refs in the same transaction (the
        binding is consumed by this attempt, published or not). ``name`` (W1.3): a detached branch's own head ref."""
        head_name = self._ref_name(session_id) if name is None else name
        with self._bindings_lock:
            names = self._publication_refs.pop((session_id, node_id), ())
        if not names:
            self.store.set_ref(head_name, node_ref, expected=expected)
            return
        with self.store.transaction():
            self.store.set_ref(head_name, node_ref, expected=expected)
            for item in names:
                current = self.store.get_ref(item)
                if current is None:
                    self.store.set_ref(item, node_ref, expected=None)
                elif current != node_ref:
                    raise StateError(f"publication ref {item} already names {current}, not {node_ref} "
                                     "(compare-and-swap)")

    # -- raw tier holders (the lineage archive's interface; no-ops without a tier) ------------------------------------
    def raw_hold(self, session_id: str, head_ref: str) -> dict[str, int] | None:
        """``session_id`` holds exactly the raw entries of ``head_ref``'s path (entries no holder keeps drop)."""
        if self.raw_tier is None:
            return None
        return self.raw_tier.hold(session_id, [node.ref for node in self.path(head_ref)[1:]], exclusive=True)

    def raw_release(self, session_id: str, *, drop: bool = False) -> dict[str, int] | None:
        """Detach ``session_id`` from every raw entry; ``drop`` (whole-tree delete, wl-w1b7): the entries it leaves
        without holders drop whatever ``raw_tier_retain``."""
        if self.raw_tier is None:
            return None
        return self.raw_tier.release_holder(session_id, drop=True if drop else None)

    release_session = raw_release

    def raw_forget(self, session_id: str) -> int:
        """wl-w1b7 (whole-tree delete): drop the unheld raw entries the session's holders held last."""
        return 0 if self.raw_tier is None else self.raw_tier.drop_unheld(session_id)

    # Step 19 raw-tier controls (``raw_tier`` session control), the lineage archive's subset: stats, demote, drop.
    def raw_tier_description(self) -> dict[str, Any]:
        if self.raw_tier is None:
            return {"enabled": False, "reason": "archive without a raw tier"}
        stats = self.raw_tier.stats()
        return {"enabled": stats["enabled"], "disabled_reason": stats["disabled_reason"], "budgets": stats["budgets"],
                "gpu_raw_budget_bytes": stats["gpu_raw_budget_bytes"],
                "host_raw_budget_bytes": stats["host_raw_budget_bytes"], "raw_restore": "prefer",
                "pinned": self.raw_tier.pinned, "kv_objects": self.kv_objects}

    def _raw_entries(self, session_id: str | None, *, published: bool = True) -> list:
        if self.raw_tier is None:
            raise StateError("this archive has no raw tier")
        if session_id is None:
            return [entry for entry in self.raw_tier.entries() if entry.node_ref is not None or not published]
        head = self.head(session_id)
        refs = [] if head is None else [node.ref for node in self.path(head)[1:]]
        entries = [entry for entry in self.raw_tier.entries(node_refs=refs) if entry.node_ref is not None]
        if not published:  # raw_offload: the session's commits the writer has not made durable yet
            entries += [entry for entry in self.raw_tier.entries(holder=session_id) if entry.node_ref is None]
        return entries

    def demote_raw(self, session_id: str | None = None, *, wait: bool = True,
                   timeout: float | None = None) -> dict[str, Any]:
        return self.raw_tier.demote(self._raw_entries(session_id, published=not self.write_behind), wait=wait,
                                    timeout=timeout) if self.raw_tier is not None else self._raw_entries(session_id)

    def drop_raw(self, session_id: str | None = None, *, drain: bool = True,
                 timeout: float | None = None) -> dict[str, Any]:
        """Drop both raw tiers of the session's (or every) published entries: the next restore of them replays (the
        recompute arm's miss path) or reads the store (raw_offload; ``drain`` first writes the pending commits,
        whose raw copies never drop before they are durable)."""
        if self.write_behind and drain:
            for name in ([session_id] if session_id is not None else list(self._wb_sessions)):
                self.drain(name, timeout=timeout)
        return self.raw_tier.drop(self._raw_entries(session_id)) if self.raw_tier is not None \
            else self._raw_entries(session_id)

    def place_raw(self, session_id: str, placement: Mapping[Any, str], *, wait: bool = True,
                  timeout: float | None = None) -> dict[str, Any]:
        """Explicit placement of the session's published chunks (the ``raw_tier`` control's ``place``): ``{chunk
        index or node ref: "gpu_raw" | "host_raw" | "store"}`` (``"compressed"``, the lineage archive's name, is
        ``"store"`` here: the raw copies drop and restores read the backing store, or replay without KV objects).
        raw_offload writes the session's pending commits first."""
        if self.raw_tier is None:
            raise StateError("this archive has no raw tier")
        if self.write_behind:
            self.drain(session_id, timeout=timeout)
        head = self.head(session_id)
        if head is None:
            raise StateError(f"unknown session {session_id!r}")
        nodes = self.path(head)[1:]
        by_ref = {entry.node_ref: entry for entry in self.raw_tier.entries(node_refs=[node.ref for node in nodes])}
        groups: dict[str, list] = {"gpu_raw": [], "host_raw": [], "store": []}
        missing = []
        for key, tier in placement.items():
            tier = "store" if tier == "compressed" else tier
            if tier not in groups:
                raise ConfigurationError(f"placement tiers are {tuple(groups)} (or 'compressed'), got {tier!r}")
            refs = ([node.ref for node in nodes if node.kind == "chunk" and node.chunk_index == key]
                    if isinstance(key, int) else [key])
            entries = [by_ref[ref] for ref in refs if ref in by_ref]
            if not entries and tier != "store":
                missing.append(key)
            groups[tier].extend(entries)
        if missing:
            raise StateError(f"no raw copy left for {missing}: placement cannot recreate dropped raw data")
        return {"gpu_raw": self.raw_tier.promote(groups["gpu_raw"], timeout=timeout) if groups["gpu_raw"] else None,
                "host_raw": (self.raw_tier.demote(groups["host_raw"], wait=wait, timeout=timeout)
                             if groups["host_raw"] else None),
                "store": self.raw_tier.drop(groups["store"]) if groups["store"] else None}

    def reset_raw_peaks(self) -> None:
        if self.raw_tier is not None:
            self.raw_tier.reset_peaks()

    def shrink_raw_buffers(self, keep_bytes: int = 0) -> dict[str, Any]:
        if self.raw_tier is None:
            raise StateError("this archive has no raw tier")
        return self.raw_tier.shrink_buffers(keep_bytes)

    # -- nodes -------------------------------------------------------------------------------
    def _manifest(self, ref: str) -> Mapping[str, Any]:
        data = self._raw.get(ref)
        if data is None:
            data = self.store.get_manifest(ref)
            if data.get("schema") != NODE_SCHEMA or data.get("model") != self.model.to_dict():
                raise IntegrityError(f"{ref} is not a RAW bridge node of this model")
            self._raw[ref] = data
        return data

    def load(self, ref: str) -> BridgeNode:
        if self._quarantine:
            for session_id, (head, error) in self._quarantine.items():
                if head == ref:
                    raise QuarantinedError(f"session {session_id!r} is quarantined: its head was damaged when the "
                                           f"store was opened ({error}); delete or replace it")
        cached = self._nodes.get(ref)
        if cached is not None:
            return cached
        data = self._manifest(ref)
        kind = data.get("type")
        if kind == "sglang_causal_root":
            node = root_node(ref, data["record"])
        elif kind == "sglang_causal_chunk":
            node = chunk_node(ref, data["parent"], data["root"], data["record"])
        elif kind == "sglang_causal_recache":
            record = data["record"]
            item = CoverageRange(int(record["token_start"]), int(record["num_tokens"]), int(record["chunk_index"]),
                                 float(record["rope_start_frame"]))
            node = recache_node(ref, data["parent"], data["root"], record, (item,), int(record["frontier"]))
        else:
            raise IntegrityError(f"{ref} is not a bridge node")
        self._nodes[ref] = node
        return node

    def path(self, ref: str) -> list[BridgeNode]:
        """Nodes from the root to ``ref`` (inclusive)."""
        nodes, current = [], ref
        while current is not None:
            node = self.load(current)
            nodes.append(node)
            current = node.parent_ref
            if len(nodes) > 1_000_000:
                raise IntegrityError("node ancestry does not terminate")
        nodes.reverse()
        if nodes[0].kind != "root" or any(node.kind not in ("chunk", "recache") for node in nodes[1:]):
            raise IntegrityError("a bridge node path must be one root followed by chunk and recache nodes")
        return nodes

    def root_record(self, ref: str) -> Mapping[str, Any]:
        return self.path(ref)[0].record

    def execution_state(self, node: BridgeNode) -> dict[str, torch.Tensor]:
        state_ref = self._manifest(node.ref).get("state")
        return {} if state_ref is None else deserialize_tensors(self.store.get(state_ref))

    def chunk_latents(self, node: BridgeNode, device: Any = "cpu") -> torch.Tensor:
        return self.execution_state(node)[LATENTS_KEY].to(device)

    # -- commits -----------------------------------------------------------------------------
    def _put_packet(self, encoded: EncodedTensor) -> str:
        blobs = {name: self.store.put(bytes(value)) for name, value in encoded.payloads.items()}
        basis = None if encoded.basis is None else self.store.put(bytes(encoded.basis))
        references = list(blobs.values()) + ([] if basis is None else [basis])
        metrics, timings = split_run_telemetry(json_copy(encoded.metrics or {}, name="packet metrics"))
        ref = self.store.put_manifest({
            "type": "sglang_packet", "schema": PACKET_SCHEMA, "model": self.model.to_dict(),
            "metadata": json_copy(encoded.metadata, name="packet metadata"), "payloads": blobs, "basis": basis,
            "metrics": metrics,
        }, references=references)
        if timings is not None:
            self.store.put_telemetry(ref, {"metrics": timings})
        return ref

    def _put(self, kind: str, *, parent_ref: str, root_ref: str, record: Mapping[str, Any],
             matrices: Mapping[Any, torch.Tensor], contexts: Mapping[Any, Any], streams: Sequence[str],
             execution_state: Mapping[str, torch.Tensor] | None) -> tuple[str, dict[str, Any]]:
        started = time.perf_counter()
        # W1b recompute arm (kv_objects=False): the captured K/V are never copied or stored (the raw tier may keep them)
        sources = {entry: tensor.detach().to("cpu") for entry, tensor in matrices.items()} if self.kv_objects else {}
        transfer = time.perf_counter() - started
        created: list[str] = []
        try:
            with self.store.transaction():
                try:
                    packets, locations = {}, {}
                    for stream in streams:
                        keys = sorted((entry[0] for entry in sources if entry[1] == stream),
                                      key=lambda item: (item.layer, item.target))
                        packets[stream], locations[stream] = [], {}
                        for key in keys:
                            locations[stream][key_label(key)] = [len(packets[stream]), None]
                            packets[stream].append(self._put_packet(
                                encode_tensor(sources[(key, stream)], contexts[(key, stream)], CodecSpec())))
                    state_ref = None if not execution_state else self.store.put(
                        serialize_tensors({name: value.detach().to("cpu")
                                           for name, value in execution_state.items()}))
                    references = [parent_ref, root_ref, *(ref for refs in packets.values() for ref in refs)]
                    references += [state_ref] if state_ref is not None else []
                    node_ref = self.store.put_manifest({
                        "type": "sglang_causal_chunk" if kind == "chunk" else "sglang_causal_recache",
                        "schema": NODE_SCHEMA, "model": self.model.to_dict(), "parent": parent_ref,
                        "root": root_ref, "record": split_record(record)[0],
                        "packets": packets, "locations": locations, "state": state_ref,
                    }, references=references)
                finally:
                    created.extend(self.store.created_objects)
        except BaseException:
            self.store.collect_garbage(candidates=created)
            raise
        native = sum(value.numel() * value.element_size() for value in (sources or matrices).values())
        report = {"representation": "raw", "native_bytes": native, "packet_payload_bytes": native,
                  "transfer_seconds": transfer, "commit_seconds": time.perf_counter() - started,
                  "objects_created": len(created)}
        if not self.kv_objects:
            report.update(representation="latents_only", packet_payload_bytes=0)
        self.stats_["commits"] += 1
        self.stats_["commit_seconds"] += report["commit_seconds"]
        return node_ref, report

    # -- workload v2 (W1b): the recompute arm's raw tier (synchronous adoption; the pipeline's protocol) --------------
    def _tier_adopt(self, session_id: str, node_id: str, groups: Sequence[tuple[str, int, Mapping[Any, Any]]],
                    ready_event: Any, release: Any) -> Any:
        """The commit's captured K/V become a raw-tier entry (no copy; ``release`` runs when the tier gives them
        up), or None (no tier, tier disabled, or no admission: the node is then a replay miss)."""
        tier = self.raw_tier
        if tier is None:
            return None
        nbytes = sum(tensor.numel() * tensor.element_size() for _, _, matrices in groups for tensor in matrices.values())
        with self._tier_cond:
            if not tier.usable(nbytes):
                self.tier_stats_["not_admitted"] += 1
                return None
            cause = tier.admission(nbytes)
            while cause == "gpu_spill":  # device copies beyond the budget are being demoted: a fraction of a chunk
                self._tier_cond.wait(0.5)
                cause = tier.admission(nbytes)
            if cause is not None:
                self.tier_stats_["not_admitted"] += 1
                return None
            reservation = tier.reserve(nbytes)
            try:
                return tier.adopt(reservation, branch=session_id, node_id=node_id, groups=list(groups),
                                  ready_event=ready_event, release=release)
            except BaseException:
                tier.unreserve(reservation)
                raise

    def _tier_published(self, entry: Any, node_ref: str) -> None:
        if entry is not None:
            with self._tier_cond:
                self.raw_tier.published(entry, node_ref)

    def _tier_retire(self, entry: Any) -> None:
        if entry is not None:
            with self._tier_cond:
                self.raw_tier.retire(entry, "failed")

    def commit_chunk(self, session_id: str, *, node_id: str, parent_ref: str, root_ref: str,
                     matrices: Mapping[Any, torch.Tensor], contexts: Mapping[Any, Any], streams: Sequence[str],
                     record: Mapping[str, Any], execution_state: Mapping[str, torch.Tensor],
                     ready_event: Any = None, release=None, timeout: Any = None) -> CompletedTicket:
        if self.write_behind:  # CONTRACT §10: queued for the ordered writer behind the raw tier
            return self._wb_commit(session_id, "chunk", node_id=node_id, parent_ref=parent_ref, root_ref=root_ref,
                                   pieces=[(dict(record), node_id, matrices, contexts)], streams=streams,
                                   execution_state=execution_state, ready_event=ready_event, release=release,
                                   timeout=timeout, token_end=int(record["token_start"]) + int(record["num_tokens"]))
        entry = self._tier_adopt(session_id, node_id, [(node_id, int(record["token_start"]), matrices)],
                                 ready_event, release) if self.raw_tier is not None else None
        if entry is not None:
            release = None  # the tier releases the sources when it gives up their device copies
        try:
            node_ref, report = self._put("chunk", parent_ref=parent_ref, root_ref=root_ref, record=record,
                                         matrices=matrices, contexts=contexts, streams=streams,
                                         execution_state=execution_state)
            self._publish_head(session_id, node_id, node_ref, parent_ref)
        except BaseException:
            self._tier_retire(entry)
            raise
        finally:
            if release is not None:
                release(None)
        self._tier_published(entry, node_ref)
        ticket = CompletedTicket(node_ref, node_id)
        ticket.report = report
        return ticket

    def commit_recache(self, session_id: str, *, node_id: str, parent_ref: str, root_ref: str,
                       ranges: Sequence[tuple[Mapping[str, Any], Mapping[Any, torch.Tensor], Mapping[Any, Any]]],
                       streams: Sequence[str], record: Mapping[str, Any],
                       execution_state: Mapping[str, torch.Tensor] | None, epoch_id: str,
                       details: Mapping[str, Any] | None = None, ready_event: Any = None, release=None,
                       timeout: Any = None) -> CompletedTicket:
        """One recache node per re-materialized chunk range, chained on the session head."""
        if self.write_behind:  # CONTRACT §10: one job, one node (and raw-tier entry) per range, written in order
            pieces = [({**dict(record), **dict(range_record), "epoch": epoch_id, "epoch_details": dict(details or {})},
                       str(range_record.get("chunk_id", index)), matrices, contexts)
                      for index, (range_record, matrices, contexts) in enumerate(ranges)]
            return self._wb_commit(session_id, "recache", node_id=node_id, parent_ref=parent_ref, root_ref=root_ref,
                                   pieces=pieces, streams=streams, execution_state=execution_state,
                                   ready_event=ready_event, release=release, timeout=timeout, token_end=None)
        head = parent_ref
        reports = []
        try:
            for index, (range_record, matrices, contexts) in enumerate(ranges):
                entry = None
                if self.raw_tier is not None:  # W1b: one raw entry per range node (the last one releases)
                    last = index == len(ranges) - 1
                    entry = self._tier_adopt(session_id, f"{node_id}/{index}",
                                             [(str(range_record.get("chunk_id", index)),
                                               int(range_record["token_start"]), matrices)],
                                             ready_event, release if last else None)
                    if entry is not None and last:
                        release = None
                try:
                    node_ref, report = self._put("recache", parent_ref=head, root_ref=root_ref,
                                                 record={**dict(record), **dict(range_record), "epoch": epoch_id,
                                                         "epoch_details": dict(details or {})},
                                                 matrices=matrices, contexts=contexts, streams=streams,
                                                 execution_state=execution_state)
                    self._publish_head(session_id, node_id, node_ref, head)
                except BaseException:
                    self._tier_retire(entry)
                    raise
                self._tier_published(entry, node_ref)
                head = node_ref
                reports.append(report)
        finally:
            if release is not None:
                release(None)
        ticket = CompletedTicket(head, node_id)
        ticket.report = {"representation": "raw" if self.kv_objects else "latents_only", "range_reports": reports}
        return ticket

    # Synchronous commits: every ticket is published when ``commit_*`` returns, nothing ever waits. raw_offload
    # (CONTRACT §10): the write-behind's protocol, as the lineage pipeline's (``ArchivePipeline``).
    def drain(self, session_id: str, *, through: str | None = None, timeout: float | None = None) -> str | None:
        if not self.write_behind:
            return self.head(session_id)
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._tier_cond:
            session = self._wb_sessions.get(session_id)
            if session is None:
                return self.head(session_id)
            if through is not None and through not in ({job.node_id for job in session.jobs} | session.published_ids):
                raise ConfigurationError(f"{through} was not submitted to session {session_id!r}")
            while True:
                if session.error is not None:
                    raise raisable(session.error)
                pending = [job.node_id for job in session.jobs]
                if not pending or (through is not None and through not in pending):
                    return session.published_ref
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    raise PendingError("raw_offload drain timed out; the written frontier trails generation")
                self._wb_ensure_writer()
                self._tier_cond.wait(0.5 if remaining is None else min(remaining, 0.5))

    def release(self, session_id: str, timeout: float | None = None) -> str | None:
        """raw_offload: drop the session's write state once its jobs retired (written, or retired after a failure:
        acknowledges it); returns the durable head."""
        if self.write_behind:
            self._wb_forget(session_id, timeout)
        return self.head(session_id)

    def abandon(self, session_id: str) -> str | None:
        """raw_offload: retire every unwritten commit of the session (a discarded session); returns its head."""
        if self.write_behind:
            with self._tier_cond:
                session = self._wb_sessions.get(session_id)
                if session is not None:
                    if session.error is None:
                        session.error = frame_free(ArchiveFailure(
                            f"archival of session {session_id} was cancelled", branch_id=session_id, node_id="*"))
                    self._wb_cancel(session)
                    self._tier_cond.notify_all()
            self._wb_forget(session_id, None)
        return self.head(session_id)

    def frontiers(self, session_id: str) -> dict[str, Any]:
        with self._tier_cond:
            session = self._wb_sessions.get(session_id) if self.write_behind else None
            if session is not None:
                return {"generated_frontier": session.generated_frontier,
                        "published_frontier": session.published_frontier, "published_ref": session.published_ref,
                        "pending": len(session.jobs), "error": None if session.error is None else str(session.error)}
        head = self.head(session_id)
        frontier = 0 if head is None else max((node.token_end for node in self.path(head)[1:]), default=0)
        return {"generated_frontier": frontier, "published_frontier": frontier, "published_ref": head,
                "pending": 0, "error": None}

    def live(self, session_id: str) -> bool:
        """raw_offload: whether the writer keeps state for the session (submitted since its last release)."""
        return self.write_behind and session_id in self._wb_sessions

    # -- W1.3: detached branches (``tree_switch="detach"``) ------------------------------------------------------------
    def detach(self, session_id: str, branch: str) -> dict[str, Any]:
        """raw_offload: hand the session's unwritten commits to the detached branch ``branch``: they keep being
        written in order and publish to the branch's own head ref (``sglang/detached/<branch>``, created at their
        durable head: the cut), with their bound refs, never moving the session head. Waits only for a publishing
        transaction in progress (the writer keeps the flag until the head and the entry advanced, F1, and starts no
        further publication while a detach waits, F5).

        Strong exception guarantee: an exception before the commit point (the branch's ref created) is raised as
        ``DetachRolledBack`` once the write state, bindings, holders and refs are unchanged (the created ref rolled
        back), so the caller may drain instead -- the session's own failure too, which the drain then surfaces; after
        it the hand-off is in memory and cannot fail (a raw-tier bookkeeping error disables the tier, recorded).
        Returns ``{"branch", "moved" (node ids; []: nothing in flight, synchronous archives always), "cut"}``. Test
        seam: ``detach_hook(step)``."""
        from ...archive_pipeline import DetachRolledBack
        from .nodes import DETACHED_REF_PREFIX

        if not self.write_behind:
            return {"branch": branch, "moved": [], "cut": None}
        hook = getattr(self, "detach_hook", None)
        with self._tier_cond:
            session = self._wb_sessions.get(session_id)
            if session is None:
                return {"branch": branch, "moved": [], "cut": None}
            session.detaching = True
            try:
                created = None
                try:
                    while session.publishing:
                        self._tier_cond.wait(0.05)
                    if session.error is not None:
                        raise raisable(session.error)
                    cut = session.published_ref
                    if not session.jobs:
                        return {"branch": branch, "moved": [], "cut": cut}
                    if branch in self._wb_sessions:
                        raise StateError(f"detached branch {branch!r} exists")
                    moved = [job.node_id for job in session.jobs]
                    if hook is not None:
                        hook("validated")
                    name = DETACHED_REF_PREFIX + branch
                    self.store.set_ref(name, cut, expected=None)
                    created = name
                    if hook is not None:
                        hook("created")
                except BaseException as exc:
                    if created is not None:  # roll back; if that fails the guarantee is lost: not DetachRolledBack
                        self.store.delete_ref(created, expected=cut)
                    raise DetachRolledBack(f"{type(exc).__name__}: {exc}") from exc
                # -- commit point: the hand-off below is in memory and cannot fail ------------------------------------
                with self._bindings_lock:  # a later commit of the session with the same node id binds its own refs
                    for node_id in moved:
                        names = self._publication_refs.pop((session_id, node_id), None)
                        if names is not None:
                            self._publication_refs[(branch, node_id)] = names
                del self._wb_sessions[session_id]
                session.session_id, session.head_name, session.detached_from = branch, name, session_id
                self._wb_sessions[branch] = session
                self.wb_stats_["detached"] = self.wb_stats_.get("detached", 0) + len(moved)
                if self.raw_tier is not None:
                    try:
                        if hook is not None:
                            hook("transfer")
                        self.raw_tier.transfer([piece["entry"] for job in session.jobs for piece in job.pieces
                                                if piece["entry"] is not None], session_id, branch)
                    except BaseException as exc:  # noqa: BLE001 - after the commit point: recorded, the tier disabled
                        self.wb_stats_["detach_bookkeeping_errors"] = self.wb_stats_.get(
                            "detach_bookkeeping_errors", 0) + 1
                        try:
                            self.raw_tier.disable(f"bookkeeping error at detach: {type(exc).__name__}: {exc}")
                        except BaseException:  # noqa: BLE001 - the count above records it; never raised here
                            pass
            finally:
                session.detaching = False
                self._tier_cond.notify_all()
        return {"branch": branch, "moved": moved, "cut": cut}

    def detached_head(self, branch: str) -> str | None:
        """A detached branch's head ref value (its last published node, or its cut), None once closed."""
        from .nodes import DETACHED_REF_PREFIX

        return self.store.get_ref(DETACHED_REF_PREFIX + branch)

    def abandon_detached(self, branch: str) -> dict[str, Any]:
        """Retire every unwritten commit of a detached branch (idempotent; waits for a write in progress, which never
        publishes). ``quiesced``: nothing of the branch is in flight."""
        self.abandon(branch)
        with self._tier_cond:
            return {"branch": branch, "quiesced": branch not in self._wb_sessions}

    def close_detached(self, branch: str, timeout: float | None = 0, keep: Any = None) -> dict[str, Any]:
        """Once every commit of the detached branch retired (published or failed): drop its write state and its
        publication bindings, root its last published node under ``keep(head)`` when that names a ref (F2: a node no
        tree ref covers), delete its head ref and release its raw-tier holder. Idempotent (``closed``);
        ``PendingError`` while commits are in flight."""
        from .nodes import DETACHED_REF_PREFIX

        if self.write_behind:
            self._wb_forget(branch, timeout)
        name = DETACHED_REF_PREFIX + branch
        head = self.store.get_ref(name)
        kept = None
        if head is not None:
            kept = keep(head) if callable(keep) else keep
            with self.store.transaction():
                if kept is not None:
                    self.store.set_ref(kept, head)
                self.store.delete_ref(name, expected=head)
        self.unbind_publication_refs(branch)  # (R4: under the bindings lock)
        raw = self.raw_tier.release_holder(branch) if self.raw_tier is not None else None
        return {"branch": branch, "head": head, "kept": kept, "raw_tier": raw, "closed": True}

    def stale_detached(self) -> dict[str, str]:
        """At open: ``{branch: head}`` of the detached head refs a previous process left (a detached branch never
        outlives its process); the controller closes each, keeping what no tree ref covers (F2)."""
        from .nodes import DETACHED_REF_PREFIX

        return {name[len(DETACHED_REF_PREFIX):]: head
                for name, head in self.store.list_refs(DETACHED_REF_PREFIX).items()}

    def backpressure(self) -> tuple[int, float]:
        """raw_offload: (commits that waited for tier admission, seconds waited) so far."""
        return int(self.wb_stats_["admission_waits"]), float(self.wb_stats_["admission_wait_seconds"])

    def commit_report(self, ticket: CompletedTicket) -> dict[str, Any]:
        return dict(getattr(ticket, "report", {}))

    # -- restore -------------------------------------------------------------------------------
    def _packet_manifest(self, ref: str, manifests: dict[str, Any] | None = None) -> Mapping[str, Any]:
        data = None if manifests is None else manifests.get(ref)
        if data is None:
            data = self.store.get_manifest(ref)
            if data.get("schema") != PACKET_SCHEMA or data.get("model") != self.model.to_dict():
                raise IntegrityError(f"{ref} is not a bridge packet of this model")
            if manifests is not None:
                manifests[ref] = data
        return data

    def packet(self, ref: str, *, read_ahead: Any = None, manifests: dict[str, Any] | None = None) -> EncodedTensor:
        """A stored packet; ``read_ahead`` (a ``store.ReadAhead``) supplies objects it read and checked ahead,
        ``manifests`` (a restore's cache) packet manifests the restore read already (each checked when read)."""
        data = self._packet_manifest(ref, manifests)

        def read(blob: str) -> bytes:
            value = None if read_ahead is None else read_ahead.take(blob)
            return self.store.get(blob) if value is None else value
        payloads = {name: read(blob) for name, blob in data["payloads"].items()}
        basis = None if data["basis"] is None else read(data["basis"])
        return EncodedTensor(data["metadata"], payloads, basis, data["metrics"])

    def raw_matrix(self, node_ref: str, stream: str, key: MatrixKey, *, read_ahead: Any = None,
                   manifests: dict[str, Any] | None = None) -> torch.Tensor:
        """The exact archived pre-RoPE K or V ``[N, C]`` of one stream of a node."""
        if not self.kv_objects:
            raise StateError("a latents-only archive (the recompute arm) stores no KV objects")
        data = self._manifest(node_ref)
        index, _ = data["locations"][stream][key_label(key)]
        return decode_tensor(self.packet(data["packets"][stream][index], read_ahead=read_ahead, manifests=manifests))

    def _read_ahead(self, plans: Mapping[str, Any], num_layers: int, refs: Any = None):
        """``(window, manifests)`` reading a restore's dense payloads ahead in ``restore_dense`` order (layer, span,
        K/V), or ``(None, None)``. Never raises: each object is still read, and fails, where it is used. ``refs``
        (raw_offload): only the spans of these nodes (the others come from the raw tier)."""
        if not self.read_ahead_bytes:
            return None, None
        manifests: dict[str, Any] = {}
        try:
            with self.store._lock:
                if self.store._depth or self.store.closed:
                    return None, None  # a failed plan must not invalidate a caller's store transaction
                with self.store.transaction():
                    spans = [(stream, span) for stream, plan in plans.items() for span in plan.spans
                             if refs is None or span.node_ref in refs]
                    blobs = []
                    for layer in range(num_layers):
                        for stream, span in spans:
                            data = self._manifest(span.node_ref)
                            for target in ("K", "V"):
                                index, _ = data["locations"][stream][key_label(MatrixKey(layer, target))]
                                packet = self._packet_manifest(data["packets"][stream][index], manifests)
                                blobs.extend(packet["payloads"].values())
                                if packet["basis"] is not None:
                                    blobs.append(packet["basis"])
                    window = self.store.read_ahead(blobs, max_bytes=self.read_ahead_bytes)
        except Exception:
            self.read_ahead_stats_["plan_failures"] += 1
            return None, None
        self.read_ahead_stats_["windows"] += 1
        return window, manifests

    def _end_read_ahead(self, window) -> None:
        if window is not None:
            window.close()
            stats = window.stats()
            for name in ("taken", "taken_bytes", "failed", "dropped"):
                self.read_ahead_stats_[name] += stats[name]

    def prewarm(self, head_ref: str) -> float:
        return 0.0

    def warmup(self, profile: Any, **_options: Any) -> dict[str, Any]:
        """Restore-kernel warm-up of the RAW path (dense rows, engine RoPE; CR-12p2-11). The engine's own RoPE
        kernel is not warmed (``restore_kernels.warmup`` calls an identity in its place)."""
        report: dict[str, Any] = {"components": {}}
        started = time.perf_counter()
        warm_restore_kernels(report, self.device, lambda: kernel_warmup_shapes(profile, self.model, path="dense"))
        report["seconds"] = time.perf_counter() - started
        return report

    def restore(self, *, head_ref: str, plans: Mapping[str, Any], caches: Mapping[str, Any], rope_cos_sin,
                rotate_key, rope_mode: str, num_layers: int, grid: tuple[int, int],
                tier_leases: Mapping[str, Any] | None = None, promote_holder: str | None = None,
                promote_span: Any = None) -> AttentionKVLease:
        """``promote_holder`` (wl-w1b7, ``raw_tier_promote="host"``): raw_offload store reads are promoted into host
        raw, held by that session (``promote_span(start_ns, end_ns)``: called after each node's copy)."""
        if tier_leases is not None or not self.kv_objects:  # W1b: the recompute arm's raw-tier install
            return self._restore_tier(plans=plans, caches=caches, rope_cos_sin=rope_cos_sin, rotate_key=rotate_key,
                                      num_layers=num_layers, grid=grid, leases=tier_leases)
        if self.write_behind:  # CONTRACT §10: GPU raw, then host raw, then the store, per span
            return self._restore_offload(plans=plans, caches=caches, rope_cos_sin=rope_cos_sin, rotate_key=rotate_key,
                                         num_layers=num_layers, grid=grid,
                                         **({"promote_holder": promote_holder, "promote_span": promote_span}
                                            if promote_holder is not None else {}))
        refs = sorted({span.node_ref for plan in plans.values() for span in plan.spans})
        lease = AttentionKVLease(store_lease=self.store.lease(refs, purpose="read", verify=False) if refs else None,
                                 plans=plans, description={"nodes": refs, "representation": "raw",
                                                           "decoder": "dense_raw", "rope_mode": "native"})
        window, manifests = self._read_ahead(plans, num_layers)
        try:
            restore_dense(lease, plans=plans, caches=caches, rope_cos_sin=rope_cos_sin, rotate_key=rotate_key,
                          num_layers=num_layers, device=self.device, residency=self.runtime.residency, grid=grid,
                          source=lambda span, stream, key: self.raw_matrix(span.node_ref, stream, key,
                                                                           read_ahead=window, manifests=manifests),
                          key_dtype=getattr(torch, self.model.dtype_for("K")))
        except BaseException:
            lease.close()
            raise
        finally:
            self._end_read_ahead(window)
        self.stats_["restores"] += 1
        return lease

    # -- workload v2 (W1b): raw-tier restores of the recompute arm --------------------------------------------------------
    def acquire_tier(self, plans: Mapping[str, Any]) -> dict[str, Any] | None:
        """Raw-tier leases (pinned until released) covering every attended range of ``plans``, or None when any
        range has no complete GPU / host raw copy (the recompute arm then replays). Never partial."""
        tier = self.raw_tier
        if tier is None:
            return None
        refs = self._tier_cover(plans)
        if refs is None:
            self.tier_stats_["misses"] += 1
            return None
        leases: dict[str, Any] = {}
        for ref in sorted(refs):
            lease = tier.acquire(ref)
            if lease is None:  # evicted meanwhile (the tier's worker): a miss, nothing kept pinned
                for item in leases.values():
                    item.release()
                self.tier_stats_["misses"] += 1
                return None
            leases[ref] = lease
        return leases

    def _usable_entries(self, refs: Iterable[str]) -> dict[str, Any]:
        """The published raw-tier entries among ``refs`` with a complete GPU or host copy (none pinned)."""
        return {entry.node_ref: entry for entry in self.raw_tier.entries(node_refs=set(refs))
                if entry.state == "published" and not entry.drop_requested
                and (entry.on_device or entry.host_state == "complete")}

    def _tier_cover(self, plans: Mapping[str, Any]) -> set[str] | None:
        """The node refs of ``plans`` when the raw tier has a usable copy of every attended range, else None."""
        spans = [span for plan in plans.values() for span in plan.spans]
        refs = {span.node_ref for span in spans}
        usable = self._usable_entries(refs)
        if any(span.node_ref not in usable or usable[span.node_ref].group_for(span.token_start - span.node_offset)
               is None for span in spans):
            return None
        return refs

    def tier_cached_refs(self, refs: Iterable[str]) -> set[str]:
        """W1.4.1 (N2): which of ``refs`` the raw tier holds now (published, a complete GPU / host copy); records
        nothing and pins nothing (the bridge's cached-ancestor search)."""
        return set() if self.raw_tier is None else set(self._usable_entries(refs))

    def tier_covers(self, plans: Mapping[str, Any]) -> bool:
        """W1.4.1 (N2): whether :meth:`acquire_tier` would find every attended range of ``plans`` now (no records, no
        pins; a later acquisition may still miss when an eviction runs in between)."""
        return self.raw_tier is not None and self._tier_cover(plans) is not None

    @staticmethod
    def _tier_matrix(leases: Mapping[str, Any], span: Any, stream: str, key: MatrixKey) -> torch.Tensor:
        lease = leases[span.node_ref]
        tensor = lease.matrix(span.token_start - span.node_offset, (key, stream))
        if tensor is None:
            raise IntegrityError(f"raw tier entry of {span.node_ref} lacks {key} of stream {stream!r}")
        return tensor

    def _restore_tier(self, *, plans: Mapping[str, Any], caches: Mapping[str, Any], rope_cos_sin, rotate_key,
                      num_layers: int, grid: tuple[int, int], leases: Mapping[str, Any] | None) -> AttentionKVLease:
        """The RAW install (engine RoPE, bitwise the native cache) with every attended range from a raw copy."""
        if leases is None:
            leases = self.acquire_tier(plans)
            if leases is None:
                raise StateError("a latents-only archive restores KV from its raw tier only; this history misses it "
                                 "(the recompute arm replays it in the stage)")
        refs = sorted({span.node_ref for plan in plans.values() for span in plan.spans})
        lease = AttentionKVLease(store_lease=None, plans=plans, description={
            "nodes": refs, "representation": "raw_tier", "decoder": "dense_raw", "rope_mode": "native",
            "kv_objects": self.kv_objects})
        for item in leases.values():
            lease.track(finalizer=item.release)
        started = time.perf_counter()
        installed = {ref for ref, item in leases.items() if item.entry.form == "installed"}  # wl-w1b7 promotions
        with tier_segment(raw_tier_label(item.tier for item in leases.values())):  # workload telemetry (W1.1)
            try:
                restore_dense(lease, plans=plans, caches=caches, rope_cos_sin=rope_cos_sin, rotate_key=rotate_key,
                              num_layers=num_layers, device=self.device, residency=self.runtime.residency, grid=grid,
                              source=lambda span, stream, key: self._tier_matrix(leases, span, stream, key),
                              key_dtype=getattr(torch, self.model.dtype_for("K")),
                              **({"rotated": lambda stream, span: span.node_ref in installed} if installed else {}))
            except BaseException:
                lease.close()
                raise
        per_chunk, chunks, nbytes = [], {}, {}
        for stream, plan in plans.items():
            for span in plan.spans:
                tier = leases[span.node_ref].tier
                group = leases[span.node_ref].entry.group_for(span.token_start - span.node_offset)
                stream_bytes = sum(meta[5] for key, meta in group.meta.items() if key[1] == stream)
                size = stream_bytes * span.length // max(1, group.rows or 1)
                per_chunk.append({"stream": stream, "chunk": span.chunk_index, "node": span.node_ref,
                                  "token_start": span.token_start, "tokens": span.length, "tier": tier,
                                  "bytes": size})
                if _tier_options(self.raw_tier):  # wl-w1b7: where the copy came from, its form, retained or not
                    per_chunk[-1].update(_lease_origin(leases[span.node_ref]))
                chunks[tier] = chunks.get(tier, 0) + 1
                nbytes[tier] = nbytes.get(tier, 0) + size
        lease.stats["tiers"] = {"per_chunk": per_chunk, "chunks": chunks, "bytes": nbytes,
                                "raw_seconds": time.perf_counter() - started, "policy": "prefer"}
        if _tier_options(self.raw_tier):  # wl-w1b7: restored nodes whose raw copy was kept unheld (retain="lru")
            lease.stats["tiers"]["retained_hits"] = sum(1 for item in leases.values() if item.retained)
        self.stats_["restores"] += 1
        self.tier_stats_["restores"] += 1
        return lease

    # -- workload v2 (CONTRACT §10): raw_offload, the RAW archive's ordered write-behind -------------------------------
    def _wb_commit(self, session_id: str, kind: str, *, node_id: str, parent_ref: str, root_ref: str,
                   pieces: Sequence[tuple[Mapping[str, Any], str, Mapping[Any, torch.Tensor], Mapping[Any, Any]]],
                   streams: Sequence[str], execution_state: Mapping[str, torch.Tensor] | None, ready_event: Any,
                   release: Any, timeout: Any, token_end: int | None) -> Any:
        """Adopt the commit's captured K/V into the raw tier (one entry per node, no copy) once the tier admits the
        whole commit -- waiting, with the cause recorded, only while its budgets are full of unwritten entries -- and
        queue it for the writer; returns its ticket at once. A degraded tier (host budget below one chunk) writes the
        commit synchronously after the session's queue, as ``raw``."""
        tier = self.raw_tier
        timeout = None if timeout is ... else timeout
        # The execution state (latents, generator states) is small: host copies now; the producer reuses its tensors.
        state = None if not execution_state else {name: value.detach().to("cpu")
                                                  for name, value in execution_state.items()}
        sizes = [sum(tensor.numel() * tensor.element_size() for tensor in matrices.values())
                 for _, _, matrices, _ in pieces]
        total = sum(sizes)
        with self._tier_cond:
            known = session_id in self._wb_sessions
        frontier = None if known else max((node.token_end for node in self.path(parent_ref)[1:]), default=0)
        deadline = None if timeout is None else time.monotonic() + timeout
        started = time.perf_counter()
        waited_for, usable = None, True
        with self._tier_cond:
            while True:
                if self._wb_stop:
                    raise StateError("the raw_offload archive is closed")
                session = self._wb_sessions.get(session_id)
                if session is not None and session.error is not None:
                    raise raisable(session.error)
                if not tier.usable(max(sizes, default=0)):
                    usable = False
                    break
                cause = tier.admission(total)
                if cause is None:
                    break
                waited_for = cause
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    raise CapacityError(f"raw_offload: the raw tier stayed full of unwritten chunks ({cause})")
                self._wb_ensure_writer()
                self._tier_cond.wait(0.5 if remaining is None else min(0.5, remaining))
            seconds = time.perf_counter() - started
            admission = {"raw_tier": usable, "waited": waited_for is not None,
                         "seconds": seconds if waited_for is not None else 0.0, "cause": waited_for, "bytes": total}
            if waited_for is not None:
                self.wb_stats_["admission_waits"] += 1
                self.wb_stats_["admission_wait_seconds"] += seconds
                causes = self.wb_stats_["admission_causes"]
                causes[waited_for] = causes.get(waited_for, 0) + 1
                tier.record_wait(waited_for, seconds, branch=session_id, node_id=node_id)
            if session is None:
                session = self._wb_sessions[session_id] = _WriteSession(
                    session_id, parent_ref, frontier if frontier is not None else 0)
            # An existing session keeps its own durable head: the caller's ``parent_ref`` is the bridge's published
            # head, which trails the writer until it reconciles; head moves outside the writer (rewind, fork, replace,
            # delete) release or abandon the session's write state first.
            if usable:
                entries, handed = [], False
                try:
                    for index, ((record, piece_id, matrices, _), nbytes) in enumerate(zip(pieces, sizes)):
                        last = index == len(pieces) - 1
                        reservation = tier.reserve(nbytes)
                        try:
                            entries.append(tier.adopt(
                                reservation, branch=session_id,
                                node_id=node_id if len(pieces) == 1 else f"{node_id}/{index}",
                                groups=[(piece_id, int(record["token_start"]), matrices)], ready_event=ready_event,
                                release=release if last else None))
                        except BaseException:
                            tier.unreserve(reservation)
                            raise
                        handed = handed or last
                except BaseException:
                    for entry in entries:
                        tier.retire(entry, "failed")
                    if not handed and release is not None:
                        release(None)
                    raise
                ticket = WriteBehindTicket(session_id, node_id,
                                           session.generated_frontier if token_end is None else token_end)
                ticket.admission = admission
                job = _WriteJob(session, kind, node_id, root_ref,
                                [{"record": record, "keys": list(matrices), "contexts": contexts, "entry": entry}
                                 for (record, _, matrices, contexts), entry in zip(pieces, entries)],
                                tuple(streams), state, ticket, total)
                session.jobs.append(job)
                session.generated_frontier = ticket.token_end
                self._wb_jobs.append(job)
                self.wb_stats_["submitted"] += 1
                self.wb_stats_["peak_queued"] = max(self.wb_stats_["peak_queued"], len(self._wb_jobs))
                self._wb_ensure_writer()
                self._tier_cond.notify_all()
                return ticket
            # Degraded tier: write as ``raw`` in this thread once the session's queue is written (commit order).
            self.wb_stats_["sync_fallbacks"] += 1
            while session.jobs and session.error is None:
                self._wb_ensure_writer()
                self._tier_cond.wait(0.5)
            if session.error is not None:
                raise raisable(session.error)
            head = session.published_ref
        try:
            reports = []
            for record, _, matrices, contexts in pieces:
                node_ref, report = self._put(kind, parent_ref=head, root_ref=root_ref, record=record,
                                             matrices=matrices, contexts=contexts, streams=streams,
                                             execution_state=state)
                self._publish_head(session_id, node_id, node_ref, head)
                head = node_ref
                reports.append(report)
        finally:
            if release is not None:
                release(None)
        with self._tier_cond:
            session.published_ref = head
            session.published_ids.add(node_id)
            if token_end is not None:
                session.generated_frontier = session.published_frontier = token_end
        ticket = CompletedTicket(head, node_id)
        ticket.report = {"representation": "raw_offload", "synchronous": True, "admission": admission,
                         "range_reports": reports}
        return ticket

    def _wb_ensure_writer(self) -> None:
        """Start the writer thread when none runs (caller holds ``_tier_cond``)."""
        if not self._wb_stop and (self._wb_thread is None or not self._wb_thread.is_alive()):
            self._wb_thread = threading.Thread(target=self._wb_loop, name="reelcache-raw-offload-writer", daemon=True)
            self._wb_thread.start()

    def _wb_loop(self) -> None:
        """The writer: jobs in submission order (one at a time, so every session's nodes publish in commit order)."""
        while True:
            with self._tier_cond:
                while not self._wb_jobs and not self._wb_stop:
                    self._tier_cond.wait()
                if not self._wb_jobs:
                    return
                job = self._wb_jobs.popleft()
                session = job.session
                if session.error is not None:  # failed or abandoned meanwhile
                    self._wb_retire(job, session.error)
                    self._tier_cond.notify_all()
                    job = session = None
                    continue
                self._wb_active = job
                for piece in job.pieces:
                    self.raw_tier._pin(piece["entry"], "write")  # its copies stay while they are written
                parent = session.published_ref
            error, reports, started = None, [], time.perf_counter()
            try:
                hook = self.write_hook
                if hook is not None:
                    hook("start", job)
                for piece in job.pieces:
                    node_ref, report = self._wb_write_piece(job, piece, parent)
                    with self._tier_cond:
                        while session.detaching and session.error is None:  # W1.3 (F5): a waiting detach goes first
                            self._tier_cond.wait(0.05)
                        failure = session.error
                        if failure is None:  # W1.3: the head ref is read with the flag set (a detach waits for it)
                            binding, head_name, session.publishing = session.session_id, session.head_name, True
                    if failure is not None:  # abandoned while written: never published
                        raise raisable(failure)
                    try:  # W1.3 (F1): the flag clears with the advanced head and entry, or with the installed error
                        self._publish_head(binding, job.node_id, node_ref, parent, name=head_name)
                        publish_hook = getattr(self, "publish_hook", None)  # test seam: after the CAS, before the state
                        if publish_hook is not None:
                            publish_hook(job, node_ref)
                        with self._tier_cond:
                            entry, piece["entry"] = piece["entry"], None
                            self.raw_tier.published(entry, node_ref)  # durable: its copies are evictable now
                            self.raw_tier._unpin(entry, "write")
                            session.published_ref = node_ref
                            self.wb_stats_["nodes"] += 1
                            session.publishing = False
                            self._tier_cond.notify_all()
                    except BaseException as exc:
                        with self._tier_cond:
                            if session.error is None:
                                session.error = frame_free(exc if isinstance(exc, ArchiveFailure) else ArchiveFailure(
                                    f"raw_offload: writing {job.kind} {job.node_id} of session {session.session_id} "
                                    f"failed: {exc!r}", branch_id=session.session_id, node_id=job.node_id, cause=exc))
                                self.wb_stats_["failed"] += 1
                            session.publishing = False
                            self._tier_cond.notify_all()
                        raise
                    parent = node_ref
                    reports.append(report)
            except BaseException as exc:  # noqa: BLE001 - the session's failure (its later commits retire)
                error = exc
            seconds = time.perf_counter() - started
            with self._tier_cond:
                self._wb_active = None
                if error is None:
                    session.jobs.remove(job)
                    session.published_ids.add(job.node_id)
                    session.published_frontier = max(session.published_frontier, job.ticket.token_end)
                    queued = started - job.submitted
                    stats = self.wb_stats_
                    stats["written"] += 1
                    stats["bytes"] += job.nbytes
                    stats["write_seconds"] += seconds
                    stats["queue_seconds"] += queued
                    stats["max_queue_seconds"] = max(stats["max_queue_seconds"], queued)
                    for report in reports:
                        stats["d2h_seconds"] += report["d2h_seconds"]
                        stats["sources"][report["source"]] += 1
                    job.ticket.report = {
                        "representation": "raw_offload", "kind": job.kind, "native_bytes": job.nbytes,
                        "packet_payload_bytes": job.nbytes, "queue_seconds": queued, "write_seconds": seconds,
                        "commit_seconds": seconds, "admission": job.ticket.admission, "range_reports": reports}
                    self.stats_["commits"] += 1
                    self.stats_["commit_seconds"] += seconds
                    job.ticket._resolve(session.published_ref)
                else:
                    if session.error is None:
                        session.error = frame_free(error if isinstance(error, ArchiveFailure) else ArchiveFailure(
                            f"raw_offload: writing {job.kind} {job.node_id} of session {session.session_id} failed: "
                            f"{error!r}", branch_id=session.session_id, node_id=job.node_id, cause=error))
                        self.wb_stats_["failed"] += 1
                    self._wb_retire(job, session.error)
                    self._wb_cancel(session)
                self._tier_cond.notify_all()
            job = session = error = reports = None  # no tensors or frames outlive the job

    def _wb_retire(self, job: _WriteJob, error: BaseException) -> None:
        """A job that will not publish (failed, cancelled): its unwritten entries drop (caller holds the condition)."""
        for piece in job.pieces:
            entry, piece["entry"] = piece["entry"], None
            if entry is None:
                continue
            if entry.pins.get("write"):
                self.raw_tier._unpin(entry, "write")
            self.raw_tier.retire(entry, "failed")
        if job in job.session.jobs:
            job.session.jobs.remove(job)
        job.ticket._resolve(error=error)

    def _wb_cancel(self, session: _WriteSession) -> None:
        """Retire the session's queued jobs (not the one being written); caller holds the condition."""
        for job in [job for job in session.jobs if job is not self._wb_active]:
            try:
                self._wb_jobs.remove(job)
            except ValueError:
                pass
            self._wb_retire(job, session.error)
            self.wb_stats_["cancelled"] += 1

    def _wb_forget(self, session_id: str, timeout: float | None) -> None:
        """Drop the session's write state once its jobs retired (``PendingError`` after ``timeout``)."""
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._tier_cond:
            session = self._wb_sessions.get(session_id)
            if session is None:
                return
            while session.jobs:
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    raise PendingError(f"raw_offload writes of session {session_id!r} are still pending")
                self._wb_ensure_writer()
                self._tier_cond.wait(0.5 if remaining is None else min(0.5, remaining))
            if self._wb_sessions.get(session_id) is session:
                del self._wb_sessions[session_id]

    def _wb_wait_idle(self, timeout: float | None) -> None:
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._tier_cond:
            while self._wb_jobs or self._wb_active is not None:
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    raise PendingError("raw_offload writes are still pending")
                self._wb_ensure_writer()
                self._tier_cond.wait(0.5 if remaining is None else min(0.5, remaining))

    def _wb_close(self) -> None:
        """Write what is queued (bounded by ``close_timeout``), retire anything left, stop the writer."""
        try:
            self._wb_wait_idle(self.close_timeout)
        except PendingError:
            pass
        with self._tier_cond:
            self._wb_stop = True
            closed = StateError("the raw_offload archive closed before this commit was written")
            for job in list(self._wb_jobs):
                self._wb_retire(job, closed)
            self._wb_jobs.clear()
            self._tier_cond.notify_all()
            thread = self._wb_thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(30)
        self._wb_staging = None

    def _wb_write_piece(self, job: _WriteJob, piece: Mapping[str, Any], parent_ref: str) -> tuple[str, dict[str, Any]]:
        """Write one node durably (its packets, execution state and manifest; one short store transaction per
        object, so the store lock is never held across a whole node) from the entry's host copy, else a D2H copy of
        its device copy. Objects of a failed node are collected; nothing is published here."""
        entry = piece["entry"]
        with self._tier_cond:
            group = entry.groups[0]
            host = dict(group.host) if entry.host_state == "complete" and group.host is not None else None
            device = dict(group.device) if host is None and entry.on_device and group.device is not None else None
        if host is None and device is None:
            raise IntegrityError(f"raw entry {entry.label} holds no complete copy (raw tier invariant I1)")
        started, d2h, created = time.perf_counter(), 0.0, []
        packets: dict[str, list[str]] = {}
        locations: dict[str, dict[str, list]] = {}
        try:
            for stream in job.streams:
                keys = sorted((key for key, item in piece["keys"] if item == stream),
                              key=lambda item: (item.layer, item.target))
                packets[stream], locations[stream] = [], {}
                for key in keys:
                    if host is not None:
                        tensor = host[(key, stream)]
                    else:
                        clock = time.perf_counter()
                        tensor = self._wb_to_host(device[(key, stream)], entry.ready_event)
                        d2h += time.perf_counter() - clock
                    encoded = encode_tensor(tensor, piece["contexts"][(key, stream)], CodecSpec())
                    tensor = None
                    with self.store.transaction():
                        try:
                            ref = self._put_packet(encoded)
                        finally:
                            created.extend(self.store.created_objects)
                    encoded = None
                    locations[stream][key_label(key)] = [len(packets[stream]), None]
                    packets[stream].append(ref)
            with self.store.transaction():
                try:
                    state_ref = None if not job.execution_state else self.store.put(
                        serialize_tensors(job.execution_state))
                    references = [parent_ref, job.root_ref, *(ref for refs in packets.values() for ref in refs)]
                    references += [state_ref] if state_ref is not None else []
                    node_ref = self.store.put_manifest({
                        "type": "sglang_causal_chunk" if job.kind == "chunk" else "sglang_causal_recache",
                        "schema": NODE_SCHEMA, "model": self.model.to_dict(), "parent": parent_ref,
                        "root": job.root_ref, "record": split_record(piece["record"])[0],
                        "packets": packets, "locations": locations, "state": state_ref,
                    }, references=references)
                finally:
                    created.extend(self.store.created_objects)
        except BaseException:
            self.store.collect_garbage(candidates=created)
            raise
        return node_ref, {"node": node_ref, "native_bytes": entry.nbytes, "objects_created": len(created),
                          "seconds": time.perf_counter() - started, "d2h_seconds": d2h,
                          "source": "host_raw" if host is not None else "gpu_raw"}

    def _wb_to_host(self, tensor: torch.Tensor, ready_event: Any) -> torch.Tensor:
        """A host view of a device matrix: copied on the writer's own stream (after the producer's ``ready_event``)
        into its pinned staging buffer (one matrix; reused, so valid until the next call)."""
        if tensor.device.type != "cuda":
            return tensor
        nbytes = tensor.numel() * tensor.element_size()
        if self._wb_staging is None or self._wb_staging.numel() < nbytes:
            self._wb_staging = None
            self._wb_staging = torch.empty(nbytes, dtype=torch.uint8, pin_memory=True)
        view = self._wb_staging[:nbytes].view(tensor.dtype).view(tensor.shape)
        stream = self._wb_streams.get(str(tensor.device))
        if stream is None:
            stream = self._wb_streams[str(tensor.device)] = torch.cuda.Stream(tensor.device)
        with torch.cuda.device(tensor.device), torch.cuda.stream(stream):
            if ready_event is not None:
                stream.wait_event(ready_event)
            view.copy_(tensor, non_blocking=True)
            event = torch.cuda.Event()
            event.record(stream)
        event.synchronize()
        return view

    def write_behind_stats(self) -> dict[str, Any]:
        with self._tier_cond:
            return {**{key: (dict(value) if isinstance(value, dict) else value) for key, value in self.wb_stats_.items()},
                    "queued": len(self._wb_jobs), "queued_bytes": sum(job.nbytes for job in self._wb_jobs),
                    "writing": None if self._wb_active is None else self._wb_active.node_id,
                    "sessions": {name: {"pending": len(item.jobs), "published": item.published_ref,
                                        "error": None if item.error is None else str(item.error)}
                                 for name, item in self._wb_sessions.items()}}

    def _restore_offload(self, *, plans: Mapping[str, Any], caches: Mapping[str, Any], rope_cos_sin, rotate_key,
                         num_layers: int, grid: tuple[int, int], promote_holder: str | None = None,
                         promote_span: Any = None) -> AttentionKVLease:
        """raw_offload restore: every attended node from its best copy -- a GPU raw hit, else a host raw hit (raw-tier
        leases, pinned until the lease closes; LRU touched), else the store (read ahead) -- then the RAW install
        (engine RoPE, bitwise the native cache). ``tiers`` records each span's tier and attended bytes; store reads
        are ``nvme`` under ``store_io="nocache"`` (cold by construction), else ``store`` (page cache unmeasured)."""
        spans = [(stream, span) for stream, plan in plans.items() for span in plan.spans]
        refs = sorted({span.node_ref for _, span in spans})
        leases: dict[str, Any] = {}
        try:
            for ref in refs:
                item = self.raw_tier.acquire(ref)
                if item is None:
                    continue
                if any(item.entry.group_for(span.token_start - span.node_offset) is None
                       for _, span in spans if span.node_ref == ref):
                    item.release()  # not the node's ranges: the store serves it
                    continue
                leases[ref] = item
            store_refs = [ref for ref in refs if ref not in leases]
            store_lease = self.store.lease(store_refs, purpose="read", verify=False) if store_refs else None
        except BaseException:
            for item in leases.values():
                item.release()
            raise
        policy = self.store.io_policy
        store_tier = "nvme" if policy == "nocache" else "store"
        lease = AttentionKVLease(store_lease=store_lease, plans=plans, description={
            "nodes": refs, "representation": "raw_offload", "decoder": "dense_raw", "rope_mode": "native",
            "store_io": policy})
        for item in leases.values():
            lease.track(finalizer=item.release)
        window, manifests = self._read_ahead(plans, num_layers, refs=set(store_refs)) if store_refs else (None, None)
        # wl-w1b7 (raw_tier_promote="host"): the store-read nodes are promoted into host raw. W1.4.1 (C2): their jobs
        # open before the install and take every row right after its read; the restore keeps none for promotion
        promote = promote_holder is not None and self.raw_tier.promotion_mode == "host" and bool(store_refs)
        jobs, promotion = (self._open_store_promotions(spans, plans, num_layers, store_refs, promote_holder,
                                                       promote_span) if promote else ({}, None))

        def source(span: Any, stream: str, key: MatrixKey) -> torch.Tensor:
            if span.node_ref in leases:
                return self._tier_matrix(leases, span, stream, key)
            rows = self.raw_matrix(span.node_ref, stream, key, read_ahead=window, manifests=manifests)
            job = jobs.get(span.node_ref)
            if job is not None:
                self.raw_tier.promotion_offer(job, (key, stream), rows)
            return rows

        started = time.perf_counter()
        label = offload_tier_label(leases[span.node_ref].tier if span.node_ref in leases else store_tier
                                   for _, span in spans)  # workload telemetry (W1.1): restore_sync.<tier>
        with tier_segment(label):
            try:
                restore_dense(lease, plans=plans, caches=caches, rope_cos_sin=rope_cos_sin, rotate_key=rotate_key,
                              num_layers=num_layers, device=self.device, residency=self.runtime.residency, grid=grid,
                              source=source, key_dtype=getattr(torch, self.model.dtype_for("K")))
            except BaseException as exc:
                for job in jobs.values():
                    self.raw_tier.promotion_close(job, error=f"restore failed: {type(exc).__name__}: {exc}")
                lease.close()
                raise
            finally:
                self._end_read_ahead(window)
        for job in jobs.values():
            self.raw_tier.promotion_close(job)
        seconds = time.perf_counter() - started
        token_bytes = num_layers * sum(
            self.model.channels_for(target) * torch.empty((), dtype=getattr(torch, self.model.dtype_for(target)))
            .element_size() for target in ("K", "V"))
        per_chunk, chunks, nbytes = [], {}, {}
        for stream, span in spans:
            tier = leases[span.node_ref].tier if span.node_ref in leases else store_tier
            size = span.length * token_bytes
            per_chunk.append({"stream": stream, "chunk": span.chunk_index, "node": span.node_ref,
                              "token_start": span.token_start, "tokens": span.length, "tier": tier, "bytes": size})
            if span.node_ref in leases and _tier_options(self.raw_tier):  # wl-w1b7: the raw copy's origin, ...
                per_chunk[-1].update(_lease_origin(leases[span.node_ref]))
            chunks[tier] = chunks.get(tier, 0) + 1
            nbytes[tier] = nbytes.get(tier, 0) + size
        lease.stats["tiers"] = {"per_chunk": per_chunk, "chunks": chunks, "bytes": nbytes, "raw_seconds": seconds,
                                "policy": "prefer", "store_io": policy, "store_nodes": len(store_refs)}
        if _tier_options(self.raw_tier):  # wl-w1b7: restored nodes whose raw copy was kept unheld (retain="lru")
            lease.stats["tiers"]["retained_hits"] = sum(1 for item in leases.values() if item.retained)
        if promotion is not None:
            lease.stats["promotion"] = self._store_promotion_record(promotion, jobs)
        self.stats_["restores"] += 1
        return lease

    def _open_store_promotions(self, spans: Sequence[tuple[str, Any]], plans: Mapping[str, Any], num_layers: int,
                               store_refs: Sequence[str], holder: str,
                               on_copied: Any = None) -> tuple[dict[str, Any], dict[str, Any]]:
        """wl-w1b7 / W1.4.1 (C2): open the promotion job of every node the store serves -- its full pre-RoPE rows,
        every plan stream and layer -- held by ``holder``, before the install reads them (``on_copied`` after each
        node's copy). A node attended partially (several bases, or not by every stream) is skipped
        (``partial_range``). Returns ``(jobs by node, the record so far)``."""
        started = time.perf_counter()
        streams = tuple(plans)
        report: dict[str, Any] = {"scope": "store_read", "nodes": 0, "bytes": 0, "host_bytes": 0, "skips": {},
                                  "skip_bytes": {}, "started": started}
        dtypes = {target: getattr(torch, self.model.dtype_for(target)) for target in ("K", "V")}
        channels = {target: self.model.channels_for(target) for target in ("K", "V")}
        jobs: dict[str, Any] = {}
        for ref in sorted(set(store_refs)):
            mine = [(stream, span) for stream, span in spans if span.node_ref == ref]
            bases = {span.token_start - span.node_offset for _, span in mine}
            rows = {span.range_tokens for _, span in mine}
            chunk = next((span.chunk_index for _, span in mine), None)
            metas = {(MatrixKey(layer, target), stream): (dtypes[target], (next(iter(rows)), channels[target]))
                     for layer in range(num_layers) for target in ("K", "V") for stream in streams} if rows else {}
            nbytes = sum(int(shape[0]) * int(shape[1]) * torch.empty((), dtype=dtype).element_size()
                         for dtype, shape in metas.values())
            if len(bases) != 1 or len(rows) != 1 or {stream for stream, _ in mine} != set(streams):
                skip = "partial_range"
                self.raw_tier.count_promotion_skip(skip, nbytes)
            else:
                job, skip = self.raw_tier.promotion_open(
                    node_ref=ref, groups=[(f"c{chunk:06d}" if chunk is not None else ref, next(iter(bases)), metas)],
                    holder=holder, origin="store", device=self.device, on_copied=on_copied)
                if job is not None:
                    jobs[ref] = job
                    continue
            report["skips"][skip] = report["skips"].get(skip, 0) + 1
            report["skip_bytes"][skip] = report["skip_bytes"].get(skip, 0) + nbytes
        return jobs, report

    def _store_promotion_record(self, report: dict[str, Any], jobs: Mapping[str, Any]) -> dict[str, Any]:
        """The restore record's ``promotion`` once the install closed every job: the nodes still promoting (their
        copies complete on the worker), their raw and host bytes, and the skips (at admission, or a job that ended
        during the install, e.g. ``transient_cap``)."""
        out = {key: value for key, value in report.items() if key != "started"}
        for job in jobs.values():
            if job.state == "abandoned":
                out["skips"][job.reason] = out["skips"].get(job.reason, 0) + 1
                out["skip_bytes"][job.reason] = out["skip_bytes"].get(job.reason, 0) + job.entry.nbytes
            else:
                out["nodes"] += 1
                out["bytes"] += job.entry.nbytes
                out["host_bytes"] += job.entry.host_nbytes
        out["seconds"] = time.perf_counter() - report["started"]
        return out

    def storage(self, ref: str) -> dict[str, Any]:
        """Unique stored bytes of the path to ``ref`` (each object charged once)."""
        seen, total, packets = set(), 0, 0
        nodes = self.path(ref)[1:]
        for node in nodes:
            for refs in self._manifest(node.ref)["packets"].values():
                for packet_ref in refs:
                    for blob in self.store.references(packet_ref):
                        if blob not in seen:
                            seen.add(blob)
                            total += self.store.object_size(blob)
                    packets += 1
        return {"nodes": len(nodes), "packets": packets, "payload_bytes": total}

    def stats(self) -> dict[str, Any]:
        stats = {"backend": self.name, **self.stats_, "residency": self.runtime.residency.stats(),
                 "quarantined": self.quarantined(), "damaged_objects": len(self.store.damaged),
                 "read_ahead": {**self.read_ahead_stats_, "max_bytes": self.read_ahead_bytes}}
        if self.raw_tier is not None:  # W1b recompute arm
            stats["raw_tier"] = {**self.raw_tier.stats(), "archive": dict(self.tier_stats_)}
        if self.write_behind:  # CONTRACT §10 raw_offload
            stats["write_behind"] = self.write_behind_stats()
        if self.write_behind or self.store.io_policy != "buffered":
            stats["store_io"] = self.store.io_stats()
        return stats

    def collect_garbage(self) -> dict[str, Any]:
        """Reclaim store objects no ref (session head, named checkpoint) or lease reaches; drops node caches.
        raw_offload: after the queued writes (objects written but not yet referenced by their node are no
        garbage)."""
        if self.write_behind:
            self._wb_wait_idle(None)
        collected = dict(self.runtime.collect_garbage())
        self._raw.clear()
        self._nodes.clear()
        return collected

    def close(self) -> None:
        if self.write_behind:
            self._wb_close()
        if self.raw_tier is not None:
            self.raw_tier.close()
        self.runtime.close()


def generator_state(generators: Sequence[Any]) -> dict[str, torch.Tensor]:
    return {f"{GENERATOR_PREFIX}{index}": generator.get_state() for index, generator in enumerate(generators)}


def raw_tier_for(config: Any) -> Any:
    """W1b: the recompute arm's raw tier from the bridge configuration: the lineage archive's raw-tier fields (GPU /
    host budgets, reserve, pinning, workers, lag bound), so every arm runs the same LRU budgets. Host pools the
    lineage archive also reserves (staging ring, prepared packs, canvases) do not exist here: an automatic host
    budget is derived without them (configure ``host_raw_budget_bytes`` for budget-matched arms)."""
    from ...raw_tier import RawChunkTier

    return RawChunkTier(gpu_budget_bytes=config.gpu_raw_budget_bytes, gpu_budget_chunks=config.max_source_chunks,
                        gpu_reserve_bytes=getattr(config, "gpu_raw_reserve_bytes", None),
                        host_budget_bytes=config.host_raw_budget_bytes, host_reserved_bytes=0,
                        workers=config.raw_tier_workers, pinned=config.host_raw_pinned,
                        max_lag_seconds=config.max_archival_lag_seconds,
                        **({"retain": config.raw_tier_retain} if getattr(config, "raw_tier_retain", "drop") != "drop"
                           else {}),
                        **({"promote": config.raw_tier_promote}
                           if getattr(config, "raw_tier_promote", "off") != "off" else {}),
                        # W1.4.1 ("nossd"): the recompute arm with a zero host budget is GPU raw only
                        **({"gpu_only": True} if config.representation == "latents_only"
                           and config.host_raw_budget_bytes == 0 else {}))


__all__ = ["CompletedTicket", "NODE_SCHEMA", "PACKET_SCHEMA", "RawSessionArchive", "SESSION_PREFIX",
           "generator_state", "key_label", "raw_tier_for"]
