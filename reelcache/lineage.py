"""Latest-method lineages: immutable state-tree nodes in the runtime's existing store.

``KVRuntime`` mixes in :class:`LineageRuntimeMixin`. A lineage is a named head
reference (``lineages/<branch>``) to an immutable :class:`~reelcache.types.ChunkNode`.
Every node references its parent node, its cache epoch, its matrix records and
the active basis versions the next commit tests, so store leases, recovery and
garbage collection see complete graphs (including every basis parent version
and row block) through ordinary manifest references. Nodes are never mutated;
forks are new heads at an existing node and extend their inherited bases
independently, sharing every common object once.

Commits run the synchronous latest-method reference (:mod:`reelcache.rank_method`):
plan on CPU or CUDA from a pinned predictor revision, re-measure the final
stored representation on the CPU source with the exact FP32 projection the
packet stores, encode an independent coefficient packet through the prepared
factor contract, store new basis blocks/versions, and publish the node with a
compare-and-swap of the head. Verified observations train the shared registry
only after publication; restoring or reading nodes never trains it.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import torch

from .basis_versions import BasisVersionStore, attach_basis, detach_basis, rows_to_bytes
from .cache_epochs import ResolvedPiece, resolve
from .errors import (
    CapacityError, ConfigurationError, IntegrityError, ReelCacheError, PendingError, StateError, UnsupportedError,
)
from .factorization import Factors, project_onto_basis, tensor_sha256
from .predictor_registry import RankPredictorRegistry, observation_origin
from .lineage_commit import ArchiveOptions, CommitJob, ParentState
from .rank_method import LatestMethodPlanner, MatrixPlan, RankMethodConfig, rate_identity, span_energy
from .residency import HistoryLease
from .run_telemetry import merge_run_telemetry, split_run_telemetry
from .serialization import resolve_dtype, serialize_tensors, tensor_nbytes
from .types import (
    ActiveBasisBinding, BasisVersion, CacheEpoch, ChunkNode, EncodedTensor, EnergyReport, MatrixKey,
    MatrixRecord, TensorContext, TokenRangeRecords, canonical_digest, entry_key, entry_order, entry_parts,
    json_copy, object_ref, thaw_json,
)


LINEAGE_PREFIX = "lineages/"
_MANIFEST_TYPES = ("node", "cache_epoch", "matrix_record", "packet")


@dataclass(frozen=True)
class LineageCommit:
    """A published node, its matrix records and the commit's evidence."""

    branch_id: str
    node_ref: str
    node: ChunkNode
    records: tuple[tuple[str, Mapping[MatrixKey, str]], ...]
    report: Mapping[str, Any]
    observations_applied: tuple[bool, ...]


class LineageRuntimeMixin:
    """State-tree lineage operations; requires the KVRuntime store/lock/model attributes."""

    def _lineage_init(self) -> None:
        self._lineage_pending: set[str] = set()
        self._lineage_nodes: dict[str, tuple[ChunkNode, CacheEpoch, dict[str, Any]]] = {}
        self._lineage_records: dict[str, MatrixRecord] = {}
        self._lineage_packets: dict[str, dict[str, Any]] = {}
        self._lineage_epochs: dict[str, CacheEpoch] = {}
        self.basis_versions = BasisVersionStore(self.store, self.model)
        self.max_basis_view_bytes = 2 * max(self.model.channels_for("K"), self.model.channels_for("V")) ** 2
        # Workload v2 (W1b) session trees: refs set with a node's publication (``bind_publication_refs``) and the
        # ownership refs a rewind accepts besides the head's ancestors (``set_lineage_ownership``).
        self._publication_refs: dict[tuple[str, str], tuple[str, ...]] = {}
        self._lineage_owned_prefix: Any = None

    def drop_manifest_caches(self) -> None:
        """Forget every parsed lineage manifest (nodes, matrix records, node packs, epochs, basis versions): the next
        reads parse them from the store again (cold-session measurements; stored objects are unchanged)."""
        with self._lock:
            self._lineage_nodes.clear()
            self._lineage_records.clear()
            self._lineage_packets.clear()
            self._lineage_epochs.clear()
            self.basis_versions.drop_caches()

    def forget_unstored_manifests(self) -> dict[str, int]:
        """Drop the parsed manifests (lineage nodes, matrix records, node packs, epochs, basis versions) and basis
        memos (finite blocks, view digests) of objects the store no longer holds (CR-17-3).

        Garbage collection reclaims deleted sessions' objects; their parsed manifests would otherwise stay cached
        in host memory for the life of the process (one index query over the cached ids; live entries stay)."""
        with self._lock:
            caches = {"nodes": self._lineage_nodes, "records": self._lineage_records,
                      "packets": self._lineage_packets, "epochs": self._lineage_epochs}
            ids = [ref for cache in caches.values() for ref in cache] + self.basis_versions.cached_ids()
            report = dict.fromkeys(caches, 0)
            if not ids:
                return {**report, "basis_versions": 0, "finite_blocks": 0, "view_digests": 0}
            present = self.store.indexed(ids)
            for name, cache in caches.items():
                gone = [ref for ref in cache if ref not in present]
                for ref in gone:
                    del cache[ref]
                report[name] = len(gone)
            return {**report, **self.basis_versions.forget_unstored(present)}

    # -- names and validation -------------------------------------------------
    def _lineage_name(self, branch_id: str) -> str:
        if (not isinstance(branch_id, str) or not branch_id or "\x00" in branch_id
                or "/" in branch_id or branch_id in {".", ".."}):
            raise ConfigurationError("lineage ids must be nonempty single-component names")
        return LINEAGE_PREFIX + branch_id

    def _entries(self, streams: Sequence[str] = ()) -> tuple:
        """Node entry keys: every layer's K and V, per CFG stream when the lineage has streams."""
        if not streams:
            return self._keys
        return tuple(sorted(((key, stream) for key in self._keys for stream in streams), key=entry_order))

    def _lineage_manifest(self, ref: str, kind: str) -> tuple[dict[str, Any], tuple[str, ...]]:
        object_ref(ref, f"{kind} manifest")
        data, references = self.store.get_manifest_with_references(ref)
        if data.get("type") != kind or data.get("version") != 1:
            raise IntegrityError(f"expected a version-1 {kind} manifest")
        if data.get("model") != self.model.to_dict():
            raise IntegrityError(f"{kind} manifest belongs to another model identity")
        return data, references

    def _expect_references(self, references: Sequence[str], expected: Sequence[str | None], kind: str) -> None:
        if list(references) != sorted({item for item in expected if item is not None}):
            raise IntegrityError(f"{kind} references are incomplete or extraneous")

    def load_packet(self, ref: str) -> dict[str, Any]:
        cached = self._lineage_packets.get(ref)
        if cached is not None:
            return cached
        data, references = self._lineage_manifest(ref, "packet")
        try:
            packed = data.get("basis_storage") == "external_version_prefix_tiles"
            fields = {"type", "version", "model", "metadata", "payloads", "metrics", "basis_storage"}
            if set(data) != (fields | {"tile_bases"} if packed else fields):
                raise IntegrityError("packet manifest has missing or extraneous fields")
            payloads = data["payloads"]
            if (not isinstance(payloads, dict) or not payloads
                    or any(not isinstance(ref_, str) or not ref_.startswith("blob:") for ref_ in payloads.values())):
                raise IntegrityError("packet payloads must reference immutable blobs")
            if data["basis_storage"] not in ("external_version_prefix", "external_version_prefix_tiles", "none"):
                raise IntegrityError("unsupported packet basis storage")
            descriptors = data["metadata"].get("payloads")
            expected_names = set(payloads)
            versions = []
            if packed:
                from .codecs import FACTOR_PACK_SCHEMA, PACK_SCHEMA
                metadata = data["metadata"]
                tiles, bases = metadata.get("tiles"), data["tile_bases"]
                # wl-base (W1c): FP16 factor packs (SVD-only baseline) bind external tile bases the same way
                if (metadata.get("schema") not in (PACK_SCHEMA, FACTOR_PACK_SCHEMA) or not isinstance(tiles, list)
                        or not isinstance(bases, list) or len(tiles) != len(bases) or not tiles):
                    raise IntegrityError("a packed node packet needs one external basis binding per tile")
                for tile, binding in zip(tiles, bases):
                    if (set(binding) != {"basis_version", "rank_view"} or binding["rank_view"] != tile["basis"]["view_rank"]
                            or tile["basis"]["payload"] in payloads):
                        raise IntegrityError("packed tile basis binding is inconsistent or embedded")
                    object_ref(binding["basis_version"], "packed tile basis version")
                    expected_names.add(tile["basis"]["payload"])
                    versions.append(binding["basis_version"])
            if not isinstance(descriptors, dict) or set(descriptors) != expected_names:
                raise IntegrityError("packet payload descriptors do not cover its blobs")
            sizes = self.store.object_sizes(payloads.values())  # one index query for every payload
            for name, blob in payloads.items():
                if (descriptors[name].get("sha256") != blob.split(":", 1)[1]
                        or descriptors[name].get("bytes") != sizes[blob]):
                    raise IntegrityError("packet payload descriptor differs from the stored blob")
        except (AttributeError, KeyError, TypeError) as exc:
            raise IntegrityError("invalid packet manifest") from exc
        self._expect_references(references, list(payloads.values()) + versions, "packet")
        self._lineage_packets[ref] = data
        return data

    def load_record(self, ref: str, *, telemetry: bool = False) -> MatrixRecord:
        """Validated matrix record (packet, basis version and prefix binding). ``telemetry=True``: the decision
        merged with the record's run telemetry (predictions, planner work) and the predictor revision, from the
        unhashed sidecar (records written before F4 carry them inline)."""
        if telemetry:
            record = self.load_record(ref)
            sidecar = self.store.telemetry(ref) or {}
            if not sidecar:
                return record
            decision = merge_run_telemetry(thaw_json(record.decision), sidecar.get("decision"))
            return MatrixRecord(record.context, record.codec, record.packet_ref, record.basis_version_ref,
                                record.rank_view, record.energy,
                                sidecar.get("predictor_revision", record.predictor_revision_ref), decision,
                                packet_tile=record.packet_tile, stream=record.stream)
        cached = self._lineage_records.get(ref)
        if cached is not None:
            return cached
        data, references = self._lineage_manifest(ref, "matrix_record")
        try:
            if set(data) != {"type", "version", "model", "record"}:
                raise IntegrityError("matrix record manifest has missing or extraneous fields")
            record = MatrixRecord.from_dict(data["record"])
        except ConfigurationError as exc:
            raise IntegrityError("invalid matrix record") from exc
        self._expect_references(references, [record.packet_ref, record.basis_version_ref], "matrix record")
        if record.context.model != self.model:
            raise IntegrityError("matrix record belongs to another model")
        packet = self.load_packet(record.packet_ref)
        metadata = packet["metadata"]
        if record.packet_tile is None and (
                metadata.get("context") != record.context.to_dict() or metadata.get("kind") != record.codec.kind
                or metadata.get("shape") != list(record.context.shape)):
            raise IntegrityError("packet header differs from its matrix record")
        if record.codec.kind == "raw":
            if packet["basis_storage"] != "none":
                raise IntegrityError("RAW packets carry no basis")
        elif record.packet_tile is not None:
            version = self.basis_versions.load(record.basis_version_ref)
            if packet["basis_storage"] != "external_version_prefix_tiles" or record.packet_tile >= len(metadata["tiles"]):
                raise IntegrityError("packed matrix record names no tile of its node pack")
            tile = metadata["tiles"][record.packet_tile]
            binding = packet["tile_bases"][record.packet_tile]
            descriptor = metadata["payloads"].get(tile["basis"]["payload"]) or {}
            if (tile.get("context") != record.context.to_dict() or tile.get("stream") != record.stream
                    or tile.get("codec") != {k: v for k, v in record.codec.to_dict().items() if k != "ffmpeg_path"}
                    or binding != {"basis_version": record.basis_version_ref, "rank_view": record.rank_view}
                    or version.key != record.context.key or record.rank_view > version.total_rank
                    or descriptor.get("shape") != [record.rank_view, version.channels]
                    or tile.get("coefficient_shape") != [record.context.token_count, record.rank_view]):
                raise IntegrityError("packed matrix record is not bound to its tile and basis prefix")
        else:
            version = self.basis_versions.load(record.basis_version_ref)
            descriptor = metadata.get("basis") or {}
            if (packet["basis_storage"] != "external_version_prefix" or version.key != record.context.key
                    or record.rank_view > version.total_rank or record.stream is not None
                    or descriptor.get("shape") != [record.rank_view, version.channels]
                    or descriptor.get("view_rank") != record.rank_view
                    or metadata.get("coefficient_shape") != [record.context.token_count, record.rank_view]):
                raise IntegrityError("matrix record rank view is not a legal prefix bound by its packet")
        self._lineage_records[ref] = record
        return record

    def load_epoch(self, ref: str) -> CacheEpoch:
        cached = self._lineage_epochs.get(ref)
        if cached is not None:
            return cached
        data, references = self._lineage_manifest(ref, "cache_epoch")
        try:
            if set(data) != {"type", "version", "model", "epoch"}:
                raise IntegrityError("cache epoch manifest has missing or extraneous fields")
            epoch = CacheEpoch.from_dict(data["epoch"])
        except ConfigurationError as exc:
            raise IntegrityError("invalid cache epoch") from exc
        self._expect_references(
            references, [epoch.parent_epoch_ref, *(ref_ for item in epoch.ranges for _, ref_ in item.records),
                         *(item.packet_ref for item in epoch.ranges)],
            "cache epoch",
        )
        if epoch.parent_epoch_ref is not None:
            self.load_epoch(epoch.parent_epoch_ref)
        for item in epoch.ranges:
            if {key for key, _ in item.records} != set(self._entries(item.streams)):
                raise IntegrityError("an epoch range must carry every layer's K and V of every stream")
        self._lineage_epochs[ref] = epoch
        return epoch

    def load_node(self, ref: str) -> ChunkNode:
        with self._lock, self.store.transaction():
            return self._node_entry(ref)[0]

    def load_basis_version(self, ref: str) -> BasisVersion:
        with self._lock, self.store.transaction():
            return self.basis_versions.load(ref)

    def _node_entry(self, ref: str) -> tuple[ChunkNode, CacheEpoch, dict[str, Any]]:
        if self._quarantined_objects:
            self._check_quarantined(ref)
        cached = self._lineage_nodes.get(ref)
        if cached is not None:
            return cached
        data, references = self._lineage_manifest(ref, "node")
        try:
            if set(data) != {"type", "version", "model", "node", "commit"}:
                raise IntegrityError("node manifest has missing or extraneous fields")
            node = ChunkNode.from_dict(data["node"])
        except ConfigurationError as exc:
            raise IntegrityError("invalid chunk node") from exc
        versions = [binding.version_ref for _, binding in node.active_basis]
        self.basis_versions.load_many(ref for ref in versions if ref is not None)  # batched; validated below
        self._expect_references(references, [
            node.parent_ref, node.epoch_ref, node.execution_state_ref, node.packet_ref,
            *(record for _, record in node.matrix_records), *versions,
        ], "node")
        epoch = self.load_epoch(node.epoch_ref)
        if epoch.epoch_id != node.cache_epoch_id:
            raise IntegrityError("node epoch id disagrees with its epoch record")
        if {key for key, _ in node.active_basis} != set(self._entries(node.streams)):
            raise IntegrityError("a node must bind every layer's K and V active basis of every stream")
        for entry, binding in node.active_basis:
            if binding.version_ref is not None:
                version = self.basis_versions.load(binding.version_ref)
                if (version.key != entry_parts(entry)[0] or version.total_rank != binding.total_rank
                        or version.group_id != binding.group_id):
                    raise IntegrityError("active basis binding disagrees with its version")
        if any(item.streams != node.streams for item in epoch.ranges):
            raise IntegrityError("epoch ranges and their node declare different CFG streams")
        if node.parent_ref is None:
            if epoch.cause != "initial" or node.frontier != 0:
                raise IntegrityError("a lineage root starts an initial epoch at token zero")
        else:
            parent, parent_epoch, _ = self._node_entry(node.parent_ref)
            if parent.streams != node.streams:
                raise IntegrityError("a lineage keeps its CFG streams on every node")
            if node.node_kind == "chunk":
                if (parent.node_kind == "segment" or node.token_start != parent.frontier
                        or node.epoch_ref != parent.epoch_ref):
                    raise IntegrityError("a chunk appends at its parent's frontier within its epoch")
                if {key for key, _ in node.matrix_records} != set(self._entries(node.streams)):
                    raise IntegrityError("a chunk node carries every layer's K and V of every stream")
            elif node.node_kind == "recache":
                if (parent.node_kind not in ("chunk", "recache") or node.frontier != parent.frontier
                        or epoch.cause != "kv_recache" or epoch.parent_epoch_ref != parent.epoch_ref
                        or any(item.token_end > node.frontier for item in epoch.ranges)):
                    raise IntegrityError("a recache node re-materializes earlier frames in a child epoch")
            elif node.node_kind == "segment":
                if (parent.node_kind not in ("root", "segment") or epoch.cause != "longcat_segment"
                        or epoch.parent_epoch_ref != parent.epoch_ref
                        or (epoch.ranges[0].token_start, epoch.ranges[0].token_end)
                        != (node.token_start, node.token_end)):
                    raise IntegrityError("a segment node replaces, never appends to, segment-local history")
            else:
                raise IntegrityError("only lineage roots have no parent")
        entry = (node, epoch, data["commit"])
        self._lineage_nodes[ref] = entry
        return entry

    def node_path(self, ref: str) -> list[tuple[str, ChunkNode, CacheEpoch]]:
        """Validated ancestor path, root first."""
        path, current = [], ref
        while current is not None:
            node, epoch, _ = self._node_entry(current)
            path.append((current, node, epoch))
            current = node.parent_ref
        return path[::-1]

    def node_commit_report(self, ref: str) -> Mapping[str, Any]:
        """The node's commit report, including its run telemetry (timings live in an unhashed sidecar)."""
        with self._lock, self.store.transaction():
            telemetry = self.store.telemetry(ref) or {}
            return MappingProxyType(json_copy(merge_run_telemetry(self._node_entry(ref)[2],
                                                                  telemetry.get("commit"))))

    def run_telemetry(self, ref: str) -> dict[str, Any] | None:
        """Unhashed run telemetry of a stored node/record/packet (``{"commit"|"decision"|"metrics": ...}``).

        Content-addressed manifests hash content only; timings and scheduling
        counts of the run that wrote them are kept here (None for objects
        written before the split, whose manifests carry them inline).
        """
        with self._lock:
            return self.store.telemetry(ref)

    def _put_content(self, data: Mapping[str, Any], *, field: str, references) -> str:
        """Store a manifest whose ``field`` is split into hashed content and an unhashed telemetry sidecar."""
        content, telemetry = split_run_telemetry(data[field])
        ref = self.store.put_manifest({**data, field: content}, references=references)
        if telemetry is not None:
            self.store.put_telemetry(ref, {field: telemetry})
        return ref

    # -- lineage heads ----------------------------------------------------------
    def create_lineage(self, branch_id: str, *, epoch_id: str = "e0", details: Mapping[str, Any] | None = None,
                       streams: Sequence[str] = ()) -> str:
        """Create a root node (initial epoch, empty active bases) and its head.

        ``streams`` declares CFG stream labels (sorted, e.g. ``("cond",
        "uncond")``); every node of the lineage then carries every stream's
        matrices together, so no checkpoint ever has only one stream.
        """
        with self._lock:
            self._open()
            name = self._lineage_name(branch_id)
            streams = tuple(streams)
            with self.store.transaction():
                if self.store.get_ref(name) is not None:
                    raise StateError(f"lineage already exists: {branch_id}")
                epoch = CacheEpoch(epoch_id, None, "initial", "global", (), details or {})
                epoch_ref = self._put_epoch(epoch)
                node = ChunkNode("root", "root", None, epoch_id, epoch_ref, 0, 0, 0, 0, 0, 0, (),
                                 tuple((entry, ActiveBasisBinding()) for entry in self._entries(streams)),
                                 streams=streams)
                ref = self._put_node(node, {"kind": "root"})
                self.store.set_ref(name, ref, expected=None)
                return ref

    def lineage_head(self, branch_id: str) -> str:
        with self._lock, self.store.transaction():
            self._open()
            head = self.store.get_ref(self._lineage_name(branch_id))
            if head is None:
                raise StateError(f"lineage does not exist: {branch_id}")
            return head

    def list_lineages(self) -> dict[str, str]:
        with self._lock, self.store.transaction():
            return {name[len(LINEAGE_PREFIX):]: ref for name, ref in self.store.list_refs(LINEAGE_PREFIX).items()}

    def fork_lineage(self, node_ref: str, branch_id: str) -> None:
        """New head at any validated node; the source lineage is untouched."""
        with self._lock:
            self._open()
            name = self._lineage_name(branch_id)
            with self.store.transaction():
                self._node_entry(node_ref)
                if self.store.get_ref(name) is not None:
                    raise StateError(f"fork target already exists: {branch_id}")
                self.store.set_ref(name, node_ref, expected=None)

    def rewind_lineage(self, branch_id: str, node_ref: str) -> None:
        """Move a head to one of its own ancestors (compare-and-swap).

        With an ownership prefix (``set_lineage_ownership``; workload v2 session trees) the target may also be any
        node on the path of a ref under ``prefix(branch_id)`` (a node the branch owns, e.g. on an old branch)."""
        with self._lock:
            self._open()
            name = self._lineage_name(branch_id)
            if branch_id in self._lineage_pending:
                raise PendingError("cannot rewind a lineage with a commit in flight")
            with self.store.transaction():
                head = self.store.get_ref(name)
                if head is None:
                    raise StateError(f"lineage does not exist: {branch_id}")
                if node_ref not in {ref for ref, _, _ in self.node_path(head)}:
                    prefix = None if self._lineage_owned_prefix is None else self._lineage_owned_prefix(branch_id)
                    if prefix is None:
                        raise StateError("rewind target is not an ancestor of the lineage head")
                    if not self._owned_by_refs(node_ref, prefix):
                        raise StateError("rewind target is neither an ancestor of the lineage head nor a node the "
                                         "lineage owns")
                self.store.set_ref(name, node_ref, expected=head)

    # -- workload v2 (W1b): session-tree ownership and publication refs ------------------------------------------------
    def set_lineage_ownership(self, prefix: Any) -> None:
        """``prefix(branch_id) -> ref prefix | None``: a rewind also accepts nodes on the paths of the refs under the
        prefix (the branch's ownership refs). None (default) restores the ancestors-only rule."""
        if prefix is not None and not callable(prefix):
            raise ConfigurationError("the lineage ownership prefix is a callable or None")
        with self._lock:
            self._lineage_owned_prefix = prefix

    def _owned_by_refs(self, node_ref: str, prefix: str) -> bool:
        """Whether ``node_ref`` is on the path of a ref under ``prefix`` (inside the caller's store transaction)."""
        owners = self.store.list_refs(prefix)
        if node_ref in owners.values():
            return True
        return any(node_ref in {ref for ref, _, _ in self.node_path(owner)} for owner in set(owners.values()))

    def bind_publication_refs(self, branch_id: str, node_id: str, names: Sequence[str]) -> None:
        """Refs to set to the node that publishes commit ``node_id`` of lineage ``branch_id``, inside the store
        transaction that moves the head (compare-and-swap: each must be absent or already name that node, else the
        publication fails and the head does not move). A later binding of the same key replaces this one."""
        names = tuple(names)
        for item in names:
            if not isinstance(item, str) or not item or "\x00" in item or item.startswith(LINEAGE_PREFIX):
                raise ConfigurationError(f"invalid publication ref name {item!r}")
        with self._lock:
            if names:
                self._publication_refs[(branch_id, node_id)] = names
            else:
                self._publication_refs.pop((branch_id, node_id), None)

    def unbind_publication_refs(self, branch_id: str) -> int:
        """Drop every binding of ``branch_id`` (its unpublished commits were retired); returns how many."""
        with self._lock:
            keys = [key for key in self._publication_refs if key[0] == branch_id]
            for key in keys:
                del self._publication_refs[key]
            return len(keys)

    def rebind_publication_refs(self, branch_id: str, new_branch_id: str, node_ids: Sequence[str]) -> int:
        """W1.3: move the bindings of commits ``node_ids`` from ``branch_id`` to ``new_branch_id`` (their branch was
        detached: a later commit of ``branch_id`` with the same node id binds its own refs); returns how many."""
        with self._lock:
            moved = 0
            for node_id in node_ids:
                names = self._publication_refs.pop((branch_id, node_id), None)
                if names is not None:
                    self._publication_refs[(new_branch_id, node_id)] = names
                    moved += 1
            return moved

    def _set_publication_refs(self, branch_id: str, node_id: str, node_ref: str) -> tuple[str, ...]:
        """Inside the publishing transaction: set the refs bound to (branch, commit) (compare-and-swap)."""
        names = self._publication_refs.get((branch_id, node_id), ())
        for name in names:
            current = self.store.get_ref(name)
            if current is None:
                self.store.set_ref(name, node_ref, expected=None)
            elif current != node_ref:
                raise StateError(f"publication ref {name} already names {current}, not {node_ref} "
                                 "(compare-and-swap)")
        return names

    def _published_refs(self, branch_id: str, node_id: str, names: tuple[str, ...]) -> None:
        """After the publishing transaction committed: the binding is consumed."""
        if names:
            with self._lock:
                if self._publication_refs.get((branch_id, node_id)) == names:
                    del self._publication_refs[(branch_id, node_id)]

    def delete_lineage(self, branch_id: str) -> None:
        with self._lock:
            self._open()
            if branch_id in self._lineage_pending:
                raise PendingError("cannot delete a lineage with a commit in flight")
            with self.store.transaction():
                name = self._lineage_name(branch_id)
                head = self.store.get_ref(name)
                if head is None:
                    raise StateError(f"lineage does not exist: {branch_id}")
                self.store.delete_ref(name, expected=head)
            self._lift_quarantine(name)

    # -- manifests ---------------------------------------------------------------
    def _put_epoch(self, epoch: CacheEpoch) -> str:
        references = [epoch.parent_epoch_ref, *(ref for item in epoch.ranges for _, ref in item.records),
                      *(item.packet_ref for item in epoch.ranges)]
        return self.store.put_manifest(
            {"type": "cache_epoch", "version": 1, "model": self.model.to_dict(), "epoch": epoch.to_dict()},
            references=[item for item in references if item is not None],
        )

    def _put_node(self, node: ChunkNode, commit: Mapping[str, Any]) -> str:
        references = [node.parent_ref, node.epoch_ref, node.execution_state_ref, node.packet_ref,
                      *(ref for _, ref in node.matrix_records),
                      *(binding.version_ref for _, binding in node.active_basis)]
        return self._put_content(
            {"type": "node", "version": 1, "model": self.model.to_dict(), "node": node.to_dict(),
             "commit": json_copy(commit, name="node commit report")},
            field="commit", references=sorted({item for item in references if item is not None}),
        )

    # -- commits ------------------------------------------------------------------
    def commit_node(self, branch_id: str, chunk_id: str, matrices: Mapping[Any, torch.Tensor],
                    contexts: Mapping[Any, TensorContext], *, method: RankMethodConfig,
                    registry: RankPredictorRegistry, execution_state: Mapping[str, torch.Tensor] | None = None,
                    device: Any = "cpu", eigh=None, eigh_label: str | None = None,
                    expected_head: str | None = None, options: ArchiveOptions | None = None) -> LineageCommit:
        """Commit one complete clean chunk (all layers' pre-RoPE K and V, every stream) as a child node.

        ``options`` (default: node packing for all-intra svd_hevc, the Step 4
        per-matrix reference otherwise) selects the archival path; see
        :class:`reelcache.lineage_commit.ArchiveOptions`. Matrices of a lineage
        with CFG streams are keyed by ``(MatrixKey, stream)``.
        """
        return self._commit("chunk", branch_id, chunk_id, [(chunk_id, matrices, contexts)], method=method,
                            registry=registry, epoch_id=None, details=None, execution_state=execution_state,
                            device=device, eigh=eigh, eigh_label=eigh_label, expected_head=expected_head,
                            options=options)

    def commit_recache(self, branch_id: str, recache_id: str,
                       ranges: Sequence[tuple[str, Mapping[MatrixKey, torch.Tensor], Mapping[MatrixKey, TensorContext]]],
                       *, method: RankMethodConfig, registry: RankPredictorRegistry, epoch_id: str,
                       details: Mapping[str, Any] | None = None,
                       execution_state: Mapping[str, torch.Tensor] | None = None, device: Any = "cpu",
                       eigh=None, eigh_label: str | None = None, expected_head: str | None = None,
                       options: ArchiveOptions | None = None) -> LineageCommit:
        """Publish a zero-frame node whose new ``kv_recache`` epoch re-materializes earlier chunks.

        Each range names an existing chunk on the head's path and supplies its
        new clean K/V; ranges are committed in order through the ordinary
        certify/augment/refit rule against the lineage's active bases.
        """
        return self._commit("recache", branch_id, recache_id, list(ranges), method=method, registry=registry,
                            epoch_id=epoch_id, details=details or {}, execution_state=execution_state,
                            device=device, eigh=eigh, eigh_label=eigh_label, expected_head=expected_head,
                            options=options)

    def commit_segment(self, branch_id: str, segment_id: str, matrices: Mapping[MatrixKey, torch.Tensor],
                       contexts: Mapping[MatrixKey, TensorContext], *, method: RankMethodConfig,
                       registry: RankPredictorRegistry, epoch_id: str, details: Mapping[str, Any] | None = None,
                       execution_state: Mapping[str, torch.Tensor] | None = None, device: Any = "cpu",
                       eigh=None, eigh_label: str | None = None, expected_head: str | None = None,
                       options: ArchiveOptions | None = None) -> LineageCommit:
        """Publish a LongCat segment: a new segment-local conditioning-cache epoch, never appended."""
        return self._commit("segment", branch_id, segment_id, [(segment_id, matrices, contexts)], method=method,
                            registry=registry, epoch_id=epoch_id, details=details or {},
                            execution_state=execution_state, device=device, eigh=eigh, eigh_label=eigh_label,
                            expected_head=expected_head, options=options)

    def commit_raw_node(self, branch_id: str, chunk_id: str, matrices: Mapping[Any, torch.Tensor],
                        contexts: Mapping[Any, TensorContext], *,
                        execution_state: Mapping[str, torch.Tensor] | None = None,
                        expected_head: str | None = None) -> LineageCommit:
        """Commit one complete clean chunk as exact RAW packets (the lifecycle's identity reference).

        Every layer's captured pre-RoPE K and V (every stream) is stored bit
        for bit as one ``reelcache.tensor.v1`` RAW packet with a RAW
        :class:`~reelcache.types.MatrixRecord` (no basis, rank or energy), plus
        the execution state. RAW nodes extend RAW lineages only (the head has
        no active basis; the bindings pass through unchanged), are restored
        exactly by :class:`reelcache.archive_restore.RawLineageRestorer` or
        :meth:`acquire_node_history`, and are not compressed: a correctness
        reference, not an archival or restore performance path.
        """
        from .codecs import encode_tensor
        from .types import CodecSpec
        started = time.perf_counter()
        head_ref, path, parent, checked, pin = self._begin_commit(
            "chunk", branch_id, chunk_id, [(chunk_id, matrices, contexts)], method=None, expected_head=expected_head)
        head = path[-1][1]
        staged_pins, staged_objects = [], []
        published = False
        try:
            if any(binding.version_ref is not None for binding in parent.bindings.values()):
                raise UnsupportedError("RAW nodes extend RAW lineages only (this head has active lossy bases)")
            _, sources, contexts = checked[0]
            raw = CodecSpec(kind="raw")
            records = {}
            payload_bytes = 0
            for entry in sorted(contexts, key=entry_order):
                context = contexts[entry]
                encoded = encode_tensor(sources[entry].detach().to("cpu"), context, raw)
                with self.store.transaction():
                    try:
                        blobs = {name: self.store.put(value) for name, value in encoded.payloads.items()}
                        packet_ref = self._put_content({
                            "type": "packet", "version": 1, "model": self.model.to_dict(),
                            "metadata": encoded.metadata, "payloads": blobs, "metrics": json_copy(encoded.metrics),
                            "basis_storage": "none",
                        }, field="metrics", references=sorted(blobs.values()))
                        record = MatrixRecord(context, raw, packet_ref, None, None, None, None,
                                              {"schema": "reelcache.raw_record.v1", "representation": "raw"},
                                              stream=entry_parts(entry)[1])
                        record_ref = self.store.put_manifest(
                            {"type": "matrix_record", "version": 1, "model": self.model.to_dict(),
                             "record": record.to_dict()}, references=[packet_ref])
                        staged_pins.append(self.store.lease([packet_ref, record_ref], purpose="staging"))
                    finally:
                        staged_objects.extend(self.store.touched_objects)
                payload_bytes += sum(len(value) for value in encoded.payloads.values())
                records[entry] = record_ref
            name = self._lineage_name(branch_id)
            with self._lock:
                self._open()
                with self.store.transaction():
                    try:
                        if self.store.get_ref(name) != head_ref:
                            raise StateError("lineage head changed before publication")
                        state_ref = None
                        if execution_state is not None:
                            state_ref = self.store.put(serialize_tensors(execution_state))
                        context = next(iter(contexts.values()))
                        active = tuple(sorted(((entry, head.binding(*entry_parts(entry)))
                                               for entry in self._entries(head.streams)),
                                              key=lambda item: entry_order(item[0])))
                        node = ChunkNode("chunk", chunk_id, head_ref, head.cache_epoch_id, head.epoch_ref,
                                         context.token_start, context.token_end, context.frames, context.height,
                                         context.width, context.token_end,
                                         tuple(sorted(records.items(), key=lambda item: entry_order(item[0]))),
                                         active, state_ref, head.streams)
                        commit = {"kind": "chunk", "representation": "raw", "records": len(records),
                                  "payload_bytes": payload_bytes,
                                  "seconds_before_publication": time.perf_counter() - started}
                        node_ref = self._put_node(node, commit)
                        self._node_entry(node_ref)
                        self.store.set_ref(name, node_ref, expected=head_ref)
                        bound = self._set_publication_refs(branch_id, chunk_id, node_ref)
                    finally:
                        staged_objects.extend(self.store.touched_objects)
                self._published_refs(branch_id, chunk_id, bound)
            published = True
        finally:
            try:
                with self.store.transaction():
                    for lease in staged_pins:
                        lease.release()
                    pin.release()
                if not published:
                    self.store.collect_garbage(candidates=staged_objects)
            finally:
                with self._lock:
                    self._lineage_pending.discard(branch_id)
        return LineageCommit(
            branch_id, node_ref, node, ((chunk_id, MappingProxyType(dict(records))),),
            MappingProxyType({"kind": "chunk", "representation": "raw", "node": node_ref, "parent": head_ref,
                              "records": len(records), "payload_bytes": payload_bytes,
                              "seconds": time.perf_counter() - started}),
            (),
        )

    def _check_group(self, chunk_id: Any, matrices: Any, contexts: Any,
                     streams: Sequence[str] = ()) -> dict[Any, TensorContext]:
        if not isinstance(chunk_id, str) or not chunk_id:
            raise ConfigurationError("lineage chunk ids are nonempty strings")
        entries = self._entries(streams)
        if (not isinstance(matrices, Mapping) or not isinstance(contexts, Mapping)
                or set(matrices) != set(entries) or set(contexts) != set(entries)):
            raise ConfigurationError("a clean commit requires exactly every layer's K and V of every stream")
        reference = None
        for key in entries:
            context = contexts[key]
            if (not isinstance(context, TensorContext) or context.model != self.model
                    or context.key != entry_parts(key)[0]):
                raise ConfigurationError("matrix context model identity or key is incompatible")
            if context.key_space != ("pre_rope" if context.key.target == "K" else "none"):
                raise ConfigurationError("lineage commits factorize only normalized pre-RoPE K and direct V")
            geometry = (context.token_start, context.frames, context.height, context.width)
            if reference is None:
                reference = geometry
            if geometry != reference:
                raise ConfigurationError("all matrices of one range share positions and geometry")
            self._validate_matrix(matrices[key], context)
        return dict(contexts)

    def _commit(self, kind, branch_id, node_id, groups, *, method, registry, epoch_id, details, execution_state,
                device, eigh, eigh_label, expected_head, options=None) -> LineageCommit:
        if not isinstance(method, RankMethodConfig) or not isinstance(registry, RankPredictorRegistry):
            raise ConfigurationError("lineage commits require a RankMethodConfig and RankPredictorRegistry")
        packable = method.codec.kind == "svd_hevc" and method.codec.inter_prediction is False
        if options is None:
            options = ArchiveOptions(packing="node" if packable else "per_matrix")
        if not isinstance(options, ArchiveOptions):
            raise ConfigurationError("options must be ArchiveOptions")
        if options.packing == "node":
            if not packable:
                raise UnsupportedError("node packing carries only all-intra svd_hevc; use packing='per_matrix'")
            return self._commit_staged(kind, branch_id, node_id, groups, method=method, registry=registry,
                                       epoch_id=epoch_id, details=details, execution_state=execution_state,
                                       device=device, eigh=eigh, eigh_label=eigh_label, expected_head=expected_head,
                                       options=options)
        return self._commit_per_matrix(kind, branch_id, node_id, groups, method=method, registry=registry,
                                       epoch_id=epoch_id, details=details, execution_state=execution_state,
                                       device=device, eigh=eigh, eigh_label=eigh_label, expected_head=expected_head,
                                       orthogonalizer=options.orthogonalizer)

    def _begin_commit(self, kind, branch_id, node_id, groups, *, method, expected_head):
        """Reserve the lineage, validate the commit shape against its head and lease the head."""
        if not self._builtin_prepared_encoder(self.encoder):
            raise UnsupportedError("lineage commits require the built-in encoder or an exact HEVCProvider")
        if not isinstance(node_id, str) or not node_id:
            raise ConfigurationError("node ids are nonempty strings")
        if not groups:
            raise ConfigurationError("a commit needs at least one clean range")
        with self._lock:
            self._open()
            name = self._lineage_name(branch_id)
            if branch_id in self._lineage_pending:
                raise PendingError(f"lineage has a commit in flight: {branch_id}")
            with self.store.transaction():
                head_ref = self.store.get_ref(name)
                if head_ref is None:
                    raise StateError(f"lineage does not exist: {branch_id}")
                if expected_head is not None and head_ref != expected_head:
                    raise StateError("lineage head changed before commit")
                path = self.node_path(head_ref)
                parent = self._parent_state(branch_id, head_ref, path, 0 if method is None else method.rate_lag,
                                            rows=method is not None)
                checked = [(chunk_id, matrices, self._check_group(chunk_id, matrices, contexts, parent.streams))
                           for chunk_id, matrices, contexts in groups]
                if kind != "recache" and len(checked) != 1:
                    raise ConfigurationError("chunk and segment commits carry exactly one range")
                self._validate_commit_shape(kind, node_id, checked, parent)
                pin = self.store.lease(head_ref)
                self._lineage_pending.add(branch_id)
        return head_ref, path, parent, checked, pin

    def _parent_state(self, branch_id: str, head_ref: str, path, lag: int, *, rows: bool = True) -> ParentState:
        """Planning view of a published head; ``rows=False`` defers assembling its active basis rows.

        Deferred rows (``ParentState.rows is None``) are assembled by
        :meth:`assemble_parent_rows` on the planning thread, never on a
        generation thread that only submits work.
        """
        head = path[-1][1]
        entries = self._entries(head.streams)
        bindings = {entry: head.binding(*entry_parts(entry)) for entry in entries}
        if rows:
            rows = self._assemble_rows(bindings)
        else:
            rows = None
        rate_sources = []
        for _, node, _ in reversed(path[-(lag + 1):]):
            rate_sources.append({entry: (node.binding(*entry_parts(entry)).coefficient_rate,
                                         node.binding(*entry_parts(entry)).rate_identity) for entry in entries})
        first = next((node for _, node, _ in path if node.node_kind in ("chunk", "segment")), None)
        return ParentState(
            branch_id, head.streams, entries, head_ref, head.node_kind, head.frontier, head.cache_epoch_id,
            frozenset(node.chunk_id for _, node, _ in path),
            {node.chunk_id: (node.token_start, node.frames, node.height, node.width)
             for _, node, _ in path if node.node_kind == "chunk"},
            None if first is None else (first.height, first.width),
            any(node.node_kind == "segment" for _, node, _ in path), bindings, rows, tuple(rate_sources), (),
            None, len(path) - 1,
        )

    def _assemble_rows(self, bindings) -> dict:
        return {entry: (None if binding.version_ref is None else
                        self.basis_versions.assemble(binding.version_ref, max_bytes=self.max_basis_view_bytes))
                for entry, binding in bindings.items()}

    def assemble_parent_rows(self, parent: ParentState) -> ParentState:
        """``parent`` with its active basis rows assembled (no-op when present)."""
        if parent.rows is not None:
            return parent
        from dataclasses import replace
        with self._lock, self.store.transaction():
            return replace(parent, rows=self._assemble_rows(parent.bindings))

    def _commit_staged(self, kind, branch_id, node_id, groups, *, method, registry, epoch_id, details,
                       execution_state, device, eigh, eigh_label, expected_head, options) -> LineageCommit:
        started = time.perf_counter()
        head_ref, _, parent, checked, pin = self._begin_commit(kind, branch_id, node_id, groups, method=method,
                                                               expected_head=expected_head)
        job = CommitJob(self, kind, node_id, parent, checked, method=method, options=options, epoch_id=epoch_id,
                        details=details, execution_state=execution_state)
        published = False
        try:
            job.plan(registry, device=device, eigh=eigh, eigh_label=eigh_label)
            job.finalize()
            job.encode()
            job.store()
            self.publish_job(job, head_ref, started=started)
            published = True
        except BaseException as exc:
            job.fail(exc)
            raise
        finally:
            try:
                job.release_staging(collect=not published)
                with self.store.transaction():
                    pin.release()
            finally:
                with self._lock:
                    self._lineage_pending.discard(branch_id)
        applied = registry.observe(job.observations)
        return self._lineage_result(job, head_ref, registry, applied, started)

    def _lineage_result(self, job, head_ref, registry, applied, started) -> LineageCommit:
        return LineageCommit(
            job.parent.branch_id, job.node_ref, job.node,
            tuple((item.chunk_id, MappingProxyType(dict(item.records))) for item in job.ranges),
            MappingProxyType(json_copy({
                "kind": job.kind, "node": job.node_ref, "parent": head_ref,
                "registry_sequence_before": job.registry_view.sequence - len(job.registry_view.speculative),
                "registry_sequence_after": registry.sequence, "observations": len(job.observations),
                "observations_applied": sum(applied),
                "plans": [row for item in job.ranges for row in item.plan_reports],
                "records": job.record_reports(), "stage_seconds": job.stage_seconds,
                "seconds": time.perf_counter() - started,
            })),
            applied,
        )

    def publish_job(self, job: CommitJob, expected_head: str, *, started: float | None = None,
                    branch_id: str | None = None) -> str:
        """Publish a stored commit job as the child of ``expected_head`` (compare-and-swap). ``branch_id`` (W1.3): the
        lineage whose head moves and whose publication refs are set, when the job's branch was detached to it (the
        job's own ``parent.branch_id`` keeps the branch it was planned on); None: that branch."""
        branch = job.parent.branch_id if branch_id is None else branch_id
        name = self._lineage_name(branch)
        publish_started = time.perf_counter()
        with self._lock:
            self._open()
            with self.store.transaction():
                try:
                    head_ref = self.store.get_ref(name)
                    if head_ref != expected_head:
                        raise StateError("lineage head changed before publication")
                    head = self._node_entry(head_ref)[0]
                    node, _ = job.build_node(head_ref, head)
                    commit = job.commit_report()
                    commit["seconds_before_publication"] = time.perf_counter() - (started or job.started)
                    ref = self._put_node(node, commit)
                    self._node_entry(ref)
                    self.store.set_ref(name, ref, expected=head_ref)
                    bound = self._set_publication_refs(branch, job.node_id, ref)
                finally:
                    job.staged_objects.extend(self.store.touched_objects)
            self._published_refs(branch, job.node_id, bound)
        job.node_ref, job.node = ref, node
        job.stage_seconds["publish"] += time.perf_counter() - publish_started
        return ref

    def _commit_per_matrix(self, kind, branch_id, node_id, groups, *, method, registry, epoch_id, details,
                           execution_state, device, eigh, eigh_label, expected_head, orthogonalizer) -> LineageCommit:
        """The Step 4 reference: independent prepared-factor packets with a CPU re-measurement."""
        started = time.perf_counter()
        head_ref, path, parent, checked, pin = self._begin_commit(kind, branch_id, node_id, groups, method=method,
                                                                  expected_head=expected_head)
        head = path[-1][1]
        staged_pins, staged_objects = [], []
        published = False
        try:
            if parent.streams:
                raise UnsupportedError("per-matrix packets do not carry CFG stream labels; use node packing")
            if method.rate_lag:
                raise UnsupportedError("the per-matrix reference measures rates with rate_lag=0 only")
            view = registry.pin()
            origin = observation_origin(branch_id, kind, node_id, view)
            bindings = {key: head.binding(key) for key in self._keys}
            groups_out, observations, plan_reports, record_reports = [], [], [], []
            for chunk_id, matrices, contexts in checked:
                records, plan_report, rows_report, observed = self._commit_range(
                    kind, node_id, head_ref, chunk_id, matrices, contexts, bindings, method, view,
                    device, eigh, eigh_label, staged_pins, staged_objects, orthogonalizer,
                )
                groups_out.append((chunk_id, MappingProxyType(records), contexts))
                plan_reports.append(plan_report)
                record_reports.extend(rows_report)
                observations.extend(replace(observation, origin=origin) for observation in observed)
            name = self._lineage_name(branch_id)
            with self._lock:
                self._open()
                with self.store.transaction():
                    try:
                        node_ref, node = self._publish(
                            kind, node_id, head_ref, head, groups_out, bindings, epoch_id, details,
                            execution_state, method, view, plan_reports, record_reports, started,
                        )
                        self.store.set_ref(name, node_ref, expected=head_ref)
                        bound = self._set_publication_refs(branch_id, node_id, node_ref)
                    finally:
                        staged_objects.extend(self.store.touched_objects)
                self._published_refs(branch_id, node_id, bound)
                published = True
        finally:
            try:
                with self.store.transaction():
                    for lease in staged_pins:
                        lease.release()
                    pin.release()
                if not published:
                    self.store.collect_garbage(candidates=staged_objects)
            finally:
                with self._lock:
                    self._lineage_pending.discard(branch_id)
        applied = registry.observe(observations)
        return LineageCommit(
            branch_id, node_ref, node,
            tuple((chunk_id, records) for chunk_id, records, _ in groups_out),
            MappingProxyType(json_copy({
                "kind": kind, "node": node_ref, "parent": head_ref, "registry_sequence_before": view.sequence,
                "registry_sequence_after": registry.sequence, "observations": len(observations),
                "observations_applied": sum(applied), "plans": plan_reports, "records": record_reports,
                "seconds": time.perf_counter() - started,
            })),
            applied,
        )

    def _validate_commit_shape(self, kind, node_id, checked, parent: ParentState) -> None:
        if node_id in parent.used_ids:
            raise StateError("node id has already been committed on this lineage path")
        for chunk_id, _, contexts in checked:
            context = next(iter(contexts.values()))
            if parent.first_hw is not None and (context.height, context.width) != parent.first_hw:
                raise ConfigurationError("spatial geometry changes within a lineage")
            if kind == "chunk":
                if parent.segment_lineage:
                    raise StateError("LongCat segment lineages never append ordinary chunks")
                if context.token_start != parent.frontier:
                    raise ConfigurationError("a chunk must start at the lineage frontier with no gap or overlap")
            elif kind == "recache":
                original = parent.chunk_geometry.get(chunk_id)
                if (parent.segment_lineage or parent.head_kind not in ("chunk", "recache") or original is None
                        or original != (context.token_start, context.frames, context.height, context.width)):
                    raise ConfigurationError("a recache range must re-materialize an existing chunk on the path")
            elif parent.head_kind not in ("root", "segment") or context.token_start != 0:
                raise StateError("a segment starts its own segment-local cache; no cross-segment append")
        if kind == "recache":
            starts = [next(iter(contexts.values())).token_start for _, _, contexts in checked]
            if starts != sorted(starts) or len(set(starts)) != len(starts):
                raise ConfigurationError("recache ranges are ordered and distinct")

    def _commit_range(self, kind, node_id, head_ref, chunk_id, matrices, contexts, bindings, method, view,
                      device, eigh, eigh_label, staged_pins, staged_objects, orthogonalizer="householder"):
        device = torch.device(device)
        if device.type == "cuda" and eigh is None:
            from .gpu_fitting import select_eigh_solver
            eigh_label, eigh = select_eigh_solver("auto")
        parent_rows, sources, source_ids, sources_cpu = {}, {}, {}, {}
        with self.store.transaction():
            for key in self._keys:
                binding = bindings[key]
                parent_rows[key] = (None if binding.version_ref is None else
                                    self.basis_versions.assemble(binding.version_ref, max_bytes=self.max_basis_view_bytes))
        for key in self._keys:
            tensor = matrices[key]
            source = tensor.detach().to("cpu") if tensor.device.type != "cpu" else tensor.detach()
            sources_cpu[key] = source
            sources[key] = tensor
            context = contexts[key]
            source_ids[key] = canonical_digest({
                "model": self.model.fingerprint, "parent_node": head_ref, "commit": kind, "node_id": node_id,
                "chunk_id": chunk_id, "token_start": context.token_start, "token_end": context.token_end,
                "key": key.to_dict(), "source_sha256": tensor_sha256(source, tile_rows=method.tile_rows),
            })
        planner = LatestMethodPlanner(method, view, device=device, eigh=eigh, eigh_label=eigh_label,
                                      orthogonalizer=orthogonalizer)
        plan = planner.plan(sources, contexts, dict(bindings), parent_rows, chunk_id=f"{kind}:{node_id}:{chunk_id}",
                            source_ids=source_ids)
        records, record_reports, observations = {}, [], []
        for key in self._keys:
            item = plan.plans[key]
            record_ref, binding, report = self._materialize(
                item, sources_cpu[key], contexts[key], bindings[key], method, chunk_id, staged_pins, staged_objects,
            )
            records[key] = record_ref
            if item.sets_active:
                bindings[key] = binding
            record_reports.append({"chunk_id": chunk_id, **report})
            observations.extend(item.observations)
        plan_report = {
            "chunk_id": chunk_id, "device": plan.device, "eigh_solver": plan.eigh_solver,
            "pinned_sequence": plan.pinned_sequence, "seconds": plan.seconds, "stages": plan.stages,
            "counters": plan.counters, **({"sub_batches": list(plan.batches)} if plan.batches else {}),
        }
        return records, plan_report, record_reports, observations

    def _materialize(self, plan: MatrixPlan, source: torch.Tensor, context: TensorContext,
                     binding: ActiveBasisBinding, method: RankMethodConfig, chunk_id: str,
                     staged_pins: list, staged_objects: list) -> tuple[str, ActiveBasisBinding, dict[str, Any]]:
        from .codecs import factor_binding, validate_prepared_packet
        from .rank_method import chosen_estimate_bytes
        started = time.perf_counter()
        key, rank = plan.key, plan.rank_view
        new_block_bytes = 0
        with self.store.transaction():
            try:
                if plan.decision == "augment":
                    version_ref, version = self.basis_versions.put_extension(plan.parent_version_ref, plan.new_rows)
                    new_block_bytes = version.added_bytes
                elif plan.new_rows is not None:
                    version_ref, version = self.basis_versions.put_root(key, plan.new_rows)
                    new_block_bytes = version.added_bytes
                else:
                    version_ref = plan.parent_version_ref
                    version = self.basis_versions.load(version_ref)
                view_rows = self.basis_versions.assemble(version_ref, rank, max_bytes=self.max_basis_view_bytes)
                # Pinned until the record that references it is pinned (CR-14-7: another session's abort may
                # collect a version both wrote).
                staged_pins.append(self.store.lease(version_ref, purpose="staging"))
            finally:
                staged_objects.extend(self.store.touched_objects)
        # Final measurement: the exact FP32 projection the packet stores, on the CPU source.
        projection = project_onto_basis(source, view_rows, rank=rank, tile_rows=method.tile_rows)
        stored = 1.0 if projection.zero_input else 1.0 - projection.relative_residual ** 2
        if not plan.diagnostic and not projection.zero_input and stored < plan.target:
            raise IntegrityError(
                f"{key}: stored representation measures {stored:.9f} on the CPU source, below the "
                f"target {plan.target}; the decision device verified {plan.decision_energy:.9f}",
            )
        source_energy = projection.source_norm ** 2
        ideal = 1.0 if projection.zero_input else min(1.0, span_energy(projection.coefficients, view_rows,
                                                                       source_energy))
        spec = method.spec_for(rank)
        factors = Factors(view_rows, projection.coefficients, rank, metrics={
            "factorization_method": f"latest_method_{plan.decision}", "projection_convention": "P=X.float()@B.float().T",
            "projection_relative_residual": projection.relative_residual,
            "basis_condition_bound": projection.condition_bound, "zero_input": projection.zero_input,
            "tile_rows": method.tile_rows, "basis_dtype": "float16", "coefficient_dtype": "float32",
            "source_dtype": context.dtype, "guard_is_pre_codec": True,
        })
        binding_record = factor_binding(source, context, spec, factors, method.fingerprint)
        encoded = self.encoder(source, context, spec, prepared_factors=factors, prepared_binding=binding_record)
        provider = getattr(self.encoder, "__self__", None)
        packet_check = validate_prepared_packet(encoded, context, spec, factors, binding_record, source=source,
                                                ffmpeg_path=getattr(provider, "ffmpeg_path", None))
        metadata, payloads, metrics = detach_basis(encoded)
        view_bytes = rows_to_bytes(view_rows)
        attach_basis(metadata, payloads, view_bytes, rank_view=rank, channels=context.channels)
        post_codec = (projection.relative_residual if spec.kind == "svd"
                      else packet_check.get("post_hevc_relative_residual"))
        payload_bytes = sum(len(value) for value in payloads.values())
        from .codecs import _json_bytes
        metadata_bytes = len(_json_bytes(metadata))
        regime = "residual" if plan.decision == "augment" else "fresh"
        prediction = next((item for item in reversed(plan.predictions) if item["regime"] == regime), None)
        if prediction is not None and prediction.get("predictor") is None:
            prediction = None
        energy = EnergyReport(
            plan.target, ideal, stored, post_codec, plan.observed_rank, plan.bucket_rank, rank,
            projection.zero_input, plan.diagnostic,
        )
        added_rows = 0 if plan.new_rows is None else int(plan.new_rows.shape[0])
        actual = {
            "payload_bytes": payload_bytes, "packet_metadata_bytes": metadata_bytes,
            "new_basis_block_bytes": new_block_bytes,
            "incremental_bytes": payload_bytes + metadata_bytes + new_block_bytes,
            "inherited_view_rows": rank - min(rank, added_rows),
            "inherited_view_bytes": 2 * context.channels * (rank - min(rank, added_rows)),
        }
        decision = {
            **plan.summary(), "final_cpu_measurement": {
                "stored_factor_energy": stored, "ideal_subspace_energy": ideal,
                "relative_residual": projection.relative_residual, "condition_bound": projection.condition_bound,
                "decision_device_energy": plan.decision_energy, "post_codec_relative_error": post_codec,
            },
            "packet_validation": json_copy(packet_check), "actual_bytes": actual,
            "estimate_vs_actual": None if plan.estimate is None else {
                "estimated_chosen_bytes": chosen_estimate_bytes(plan.estimate),
                "actual_incremental_bytes": actual["incremental_bytes"],
            },
        }
        decision, decision_telemetry = split_run_telemetry(json_copy(decision))
        # F4: the predictor revision lives in the record's sidecar with the decision telemetry (see lineage_commit).
        revision = None if prediction is None else prediction["predictor"]["revision_ref"]
        record = MatrixRecord(context, spec, "manifest:" + "0" * 64, version_ref, rank, energy, None, decision)
        with self.store.transaction():
            try:
                blobs = {name: self.store.put(value) for name, value in payloads.items()}
                packet_ref = self._put_content({
                    "type": "packet", "version": 1, "model": self.model.to_dict(), "metadata": metadata,
                    "payloads": blobs, "metrics": metrics, "basis_storage": "external_version_prefix",
                }, field="metrics", references=list(blobs.values()))
                record = MatrixRecord(context, spec, packet_ref, version_ref, rank, energy,
                                      record.predictor_revision_ref, record.decision)
                record_ref = self.store.put_manifest(
                    {"type": "matrix_record", "version": 1, "model": self.model.to_dict(),
                     "record": record.to_dict()}, references=[packet_ref, version_ref],
                )
                sidecar = {} if decision_telemetry is None else {"decision": decision_telemetry}
                if revision is not None:
                    sidecar["predictor_revision"] = revision
                if sidecar:
                    self.store.put_telemetry(record_ref, sidecar)
                staged_pins.append(self.store.lease(record_ref, purpose="staging"))
            finally:
                staged_objects.extend(self.store.touched_objects)
        rate = payload_bytes / (context.token_count * rank)
        identity = rate_identity(context, spec)
        if plan.decision in ("fresh", "refit", "diagnostic_full_width"):
            updated = ActiveBasisBinding(version_ref, version.total_rank, version.group_id, rank, chunk_id,
                                         rate, identity)
        elif plan.decision in ("augment", "share"):
            updated = ActiveBasisBinding(version_ref, version.total_rank, version.group_id, binding.fresh_rank,
                                         binding.fresh_chunk_id, rate, identity, plan.refit_debt or 0.0)
        else:
            updated = binding
        report = {
            "key": key.to_dict(), "decision": plan.decision, "reason": plan.reason, "record": record_ref,
            "basis_version": version_ref, "group_id": version.group_id, "rank_view": rank,
            "version_total_rank": version.total_rank, "energy": energy.to_dict(), "actual_bytes": actual,
            "estimate": plan.estimate, "counters": plan.work["counters"],
            "materialize_seconds": time.perf_counter() - started,
        }
        return record_ref, updated, report

    def _publish(self, kind, node_id, head_ref, head, groups_out, bindings, epoch_id, details, execution_state,
                 method, view, plan_reports, record_reports, started) -> tuple[str, ChunkNode]:
        state_ref = None
        if execution_state is not None:
            state_ref = self.store.put(serialize_tensors(execution_state))
        active = tuple((key, bindings[key]) for key in self._keys)
        commit = {
            "kind": kind, "method": method.to_dict(), "registry": {"registry_id": view.registry_id,
                                                                    "pinned_sequence": view.sequence},
            "plans": plan_reports,
            "decisions": {decision: sum(row["decision"] == decision for row in record_reports)
                          for decision in sorted({row["decision"] for row in record_reports})},
            "incremental_bytes": sum(row["actual_bytes"]["incremental_bytes"] for row in record_reports),
            "new_basis_block_bytes": sum(row["actual_bytes"]["new_basis_block_bytes"] for row in record_reports),
        }
        if kind == "chunk":
            chunk_id, records, contexts = groups_out[0]
            context = contexts[self._keys[0]]
            node = ChunkNode("chunk", chunk_id, head_ref, head.cache_epoch_id, head.epoch_ref,
                             context.token_start, context.token_end, context.frames, context.height,
                             context.width, context.token_end, tuple(records.items()), active, state_ref)
        else:
            ranges = []
            for chunk_id, records, contexts in groups_out:
                context = contexts[self._keys[0]]
                ranges.append(TokenRangeRecords(chunk_id, context.token_start, context.token_end, context.frames,
                                                context.height, context.width, tuple(records.items())))
            if kind == "recache":
                epoch = CacheEpoch(epoch_id, head.epoch_ref, "kv_recache", "global", tuple(ranges), details)
                epoch_ref = self._put_epoch(epoch)
                node = ChunkNode("recache", node_id, head_ref, epoch_id, epoch_ref, head.frontier, head.frontier,
                                 0, 0, 0, head.frontier, (), active, state_ref)
            else:
                epoch = CacheEpoch(epoch_id, head.epoch_ref, "longcat_segment", "segment_local", tuple(ranges),
                                   details)
                epoch_ref = self._put_epoch(epoch)
                item = ranges[0]
                node = ChunkNode("segment", node_id, head_ref, epoch_id, epoch_ref, 0, item.token_end,
                                 item.frames, item.height, item.width, item.token_end, (), active, state_ref)
        commit["seconds_before_publication"] = time.perf_counter() - started
        ref = self._put_node(node, commit)
        self._node_entry(ref)
        return ref, node

    # -- restore -------------------------------------------------------------------
    def resolve_history(self, node_ref: str, ranges: Sequence[tuple[int, int]] | None = None) -> list[ResolvedPiece]:
        """Resolve attended ranges (default: the whole frontier) to the newest versions on the path."""
        with self._lock, self.store.transaction():
            self._open()
            path = self.node_path(node_ref)
            node = path[-1][1]
            if ranges is None:
                if node.frontier == 0:
                    return []
                ranges = [(0, node.frontier)]
            return resolve(path, ranges)

    def matrix_packet(self, record_ref: str) -> EncodedTensor:
        """The record's packet with its assembled, verified basis prefix (or every tile's) attached.

        A record that is a tile of a node pack returns the whole pack; decode
        its entry with ``codecs.decode_packed(pack, keys=[record.entry])``.
        """
        with self._lock, self.store.transaction():
            record = self.load_record(record_ref)
            if record.packet_tile is not None:
                return self.node_pack(record.packet_ref)
            packet = self.load_packet(record.packet_ref)
            payloads = {name: self.store.get(blob) for name, blob in packet["payloads"].items()}
            if record.codec.kind == "raw":
                return EncodedTensor(packet["metadata"], payloads)
            view = self.basis_versions.view_bytes(record.basis_version_ref, record.rank_view,
                                                  max_bytes=self.max_basis_view_bytes)
            # Assembled from hash-checked blocks; with verify_reads the descriptor's manifest was hash-checked too.
            return attach_basis(packet["metadata"], payloads, view, rank_view=record.rank_view,
                                channels=record.context.channels, check_digest=not self.store.verify_reads)

    def node_pack(self, packet_ref: str, *, bases: Sequence[bytes] | None = None,
                  block_cache: dict[str, bytes] | None = None) -> EncodedTensor:
        """A stored node pack with every tile's basis version prefix assembled, checked and attached.

        ``bases`` may supply already assembled prefix bytes (e.g. from a host
        basis cache); they are checked against the tile descriptors all the same.
        ``block_cache`` (caller-owned) shares checked row blocks across packs of
        one history, whose version chains overlap. Prefixes assembled here are
        hashed once per read, as their row blocks (``verify_reads`` also
        authenticates the descriptors, so they are not re-hashed against them).
        """
        from .node_packets import attach_pack_bases
        with self._lock, self.store.transaction():
            packet = self.load_packet(packet_ref)
            if packet["basis_storage"] != "external_version_prefix_tiles":
                raise ConfigurationError("not a node pack with external tile bases")
            names = list(packet["payloads"])
            payloads = dict(zip(names, self.store.get_many(packet["payloads"][name] for name in names)))
            check_digests = True
            if bases is None:
                block_cache = {} if block_cache is None else block_cache
                self.basis_versions.prefetch_blocks(
                    [(item["basis_version"], item["rank_view"]) for item in packet["tile_bases"]], block_cache)
                bases = [self.basis_versions.view_bytes(item["basis_version"], item["rank_view"],
                                                        max_bytes=self.max_basis_view_bytes, block_cache=block_cache)
                         for item in packet["tile_bases"]]
                check_digests = not self.store.verify_reads
            return attach_pack_bases(packet["metadata"], payloads, bases, check_digests=check_digests)

    def acquire_node_history(self, node_ref: str, key: MatrixKey, *, ranges: Sequence[tuple[int, int]] | None = None,
                             device: Any = "cpu", dtype: Any = None, decoder=None,
                             stream: str | None = None) -> HistoryLease:
        """Restore attended ranges of one K/V into a bounded lease (newest version per range).

        Admission charges the destination plus the largest per-packet decode
        workspace including its assembled basis view; record, packet, basis
        version and row-block objects stay pinned until the lease closes.
        Node-pack tiles decode through ``codecs.decode_packed`` (only the
        tile's access units); the device restore path is
        :class:`reelcache.archive_restore.LineageRestorer`.
        """
        if not isinstance(key, MatrixKey) or key not in self._keys:
            raise ConfigurationError("history key is outside the model")
        decoder = self.decoder if decoder is None else decoder
        if not callable(decoder):
            raise ConfigurationError("history decoder must be callable")
        with self._lock, self.store.transaction():
            self._open()
            path = self.node_path(node_ref)
            node = path[-1][1]
            if (stream is None) != (not node.streams) or (stream is not None and stream not in node.streams):
                raise ConfigurationError("history stream must name one of the lineage's CFG streams")
            if ranges is None:
                ranges = [(0, node.frontier)] if node.frontier else []
            pieces = resolve(path, ranges)
            if not pieces:
                raise StateError("no attended history to restore")
            records = [(piece, piece.record(key, stream), self.load_record(piece.record(key, stream)))
                       for piece in pieces]
            source_dtype = records[0][2].context.dtype
            if any(record.context.dtype != source_dtype for _, _, record in records):
                raise IntegrityError("restored ranges disagree on their source dtype")
            output_dtype = resolve_dtype(source_dtype) if dtype is None else resolve_dtype(dtype)
            if not output_dtype.is_floating_point:
                raise ConfigurationError("KV history must be restored to a floating-point dtype")
            rows = sum(piece.token_end - piece.token_start for piece in pieces)
            channels = self.model.channels_for(key)
            destination = tensor_nbytes((rows, channels), output_dtype)
            scratch = 0
            for _, _, record in records:
                element = max(4, torch.empty((), dtype=output_dtype).element_size())
                workspace = 2 * record.context.token_count * channels * element
                if record.codec.kind != "raw":
                    workspace += 4 * (record.rank_view * channels + record.context.token_count * record.rank_view)
                    workspace += 2 * record.rank_view * channels
                if record.packet_tile is not None:
                    canvas = self.load_packet(record.packet_ref)["metadata"]["canvas"]
                    tile = self.load_packet(record.packet_ref)["metadata"]["tiles"][record.packet_tile]
                    workspace += tile["pictures"][1] * canvas["planes"] * canvas["width"] * canvas["height"]
                scratch = max(scratch, workspace)
            pin = self.store.lease(sorted({ref for _, ref, _ in records}), verify=False)

        def load() -> torch.Tensor:
            result = torch.empty((rows, channels), device=device, dtype=output_dtype)
            cursor = 0
            for piece, ref, record in records:
                encoded = self.matrix_packet(ref)
                if record.packet_tile is not None:
                    from .codecs import decode_packed
                    decoded = decode_packed(encoded, keys=[record.entry], device=device,
                                            dtype=output_dtype)[record.entry]
                else:
                    decoded = decoder(encoded, device=device, dtype=output_dtype)
                if (not isinstance(decoded, torch.Tensor) or tuple(decoded.shape) != record.context.shape
                        or decoded.dtype != output_dtype):
                    raise IntegrityError("decoder shape/dtype does not match the committed record")
                length = piece.token_end - piece.token_start
                result[cursor:cursor + length].copy_(decoded[piece.offset:piece.offset + length])
                cursor += length
                del decoded, encoded
            return result

        return self.residency.acquire(destination, load, scratch_bytes=scratch, pin=pin)

    def node_storage(self, node_ref: str) -> dict[str, Any]:
        """Unique stored footprint of a node's attended records (each object charged once)."""
        with self._lock, self.store.transaction():
            pieces = self.resolve_history(node_ref) if self.load_node(node_ref).frontier else []
            refs = sorted({ref for piece in pieces for _, ref in piece.entry.records})
            packets, versions = set(), set()
            for ref in refs:
                record = self.load_record(ref)
                packets.add(record.packet_ref)
                if record.basis_version_ref is not None:
                    versions.add(record.basis_version_ref)
            payload = sum(self.store.object_size(blob) for packet in sorted(packets)
                          for blob in self.load_packet(packet)["payloads"].values())
            blocks = self.basis_versions.unique_block_bytes(sorted(versions))
            return {"records": len(refs), "packets": len(packets), "basis_versions": len(versions),
                    "unique_basis_blocks": len(blocks), "basis_block_bytes": sum(blocks.values()),
                    "payload_bytes": payload}
