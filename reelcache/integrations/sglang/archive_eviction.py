"""W1.4.3 (E1): LRU eviction of a compressed archive's node payloads under ``archive_budget_bytes``.

Only the ablation arms may evict: the configuration admits a budget with ``reuse=False`` alone (ReelCache's basis
sharing makes a node's payload part of its descendants'; it never evicts). The archive (a tmpfs store in the
evaluation) keeps every node's manifest, its ownership ref, clean latents, generator and execution state; eviction
removes WHOLE nodes' KV payloads -- every packet payload blob and basis row block a node's restore reads, all
matrices and CFG streams (:meth:`ObjectStore.evict`) -- least recently used first (access: publication, or a restore
that read the node from the archive), and drops their prepared packs and resident basis versions.

``live_bytes`` counts the payload objects of the nodes that are not evicted (an object several nodes share counts
once and is evicted with its last holder) and the orphans below. Every payload write is admitted before it reaches the
store (:meth:`writing`, around the store transaction that writes a commit's basis row blocks, and the one that writes
its packet payload blobs): the content-addressed objects it stores are reserved when physically absent -- never
written, or evicted -- each content once, so a deduplicated write costs nothing (review round 2); ``live_bytes +
reserved_bytes`` never exceeds the budget, in-flight publications included, and never understates the payload bytes in
the store. :meth:`published` turns a node's reservation into its live bytes under this object's lock (the payload check
and the holder registration are atomic with every eviction); a durable publication whose bookkeeping fails stays
counted (its reservation) and is registered before the next write is admitted. Manifests, records and execution state
are the unevictable sidecar and are not counted. A restore pins the nodes it reads
(:meth:`pin_available`), so nothing it reads is evicted under it; a reservation that finds only pinned nodes waits. A
reservation nothing can make room for -- no node to evict, no pin, every other in-flight commit waiting for room or
publishing only after one that does (publication is ordered per lineage) -- fails at once: the budget must exceed the
bytes the commits in flight write.

A node is evicted ALL OR NOTHING: when one of its own objects is protected -- leased, or reachable from a pending
commit's staging leases (:meth:`ObjectStore.evict`) -- nothing of it goes and the next least recently used node is
tried. An object in the store that no live node reads -- one a commit wrote and then did not publish (failed or
cancelled: counted in the same step its reservation goes, review round 2), the present part of a payload registered
evicted, one whose last live holder a whole-tree delete removed while another manifest still references it -- is an
orphan: counted live, evicted first. A commit cancelled while its reservation waits stops waiting at once.

The LRU order lives in memory: a reopened archive (:meth:`adopt`) starts from its published nodes, parents before
children (oldest first), not from the order of the run that wrote them (runs do not reopen mid-run).
"""

from __future__ import annotations

import contextlib
import threading
import time
from collections import Counter, OrderedDict
from typing import Any, Callable, Iterable, Mapping

from ...errors import CapacityError, ConfigurationError, StateError

#: How long a reservation waits for pinned nodes to be released before failing the commit.
RESERVE_WAIT_SECONDS = 300.0


class _Node:
    __slots__ = ("objects", "packets", "versions")

    def __init__(self, objects: frozenset, packets: tuple, versions: tuple):
        self.objects, self.packets, self.versions = objects, packets, versions


class ArchiveEviction:
    """The budget, the LRU of a store's archived nodes and their eviction (thread-safe; see the module docstring).

    ``on_evict(node_ref, packets, versions)`` runs after a node's payload left the store (drop prepared packs and
    resident basis versions). Lock order: this object's lock, then the store's."""

    def __init__(self, store: Any, budget_bytes: int, *, on_evict: Callable[[str, tuple, tuple], Any] | None = None,
                 wait_seconds: float = RESERVE_WAIT_SECONDS):
        if type(budget_bytes) is not int or budget_bytes <= 0:
            raise ConfigurationError("archive_budget_bytes must be a positive integer")
        self.store, self.budget, self.on_evict, self.wait_seconds = store, budget_bytes, on_evict, wait_seconds
        self._cond = threading.Condition()
        self._nodes: OrderedDict[str, _Node] = OrderedDict()  # live nodes, least recently used first
        self._evicted: dict[str, _Node] = {}
        self._holders: dict[str, set] = {}  # payload object -> live nodes that read it
        self._sizes: dict[str, int] = {}
        self._reserved: dict[int, int] = {}  # reservation key (a commit job's id) -> bytes
        self._pins: Counter = Counter()
        self._waiting: set = set()  # reservation keys blocked in reserve (deadlock detection)
        # a reservation key -> its commit's publication order ``(lineage token, sequence)`` (``key.budget_order``, set
        # by the archival pipeline; None: unknown): a commit publishes only after the earlier ones of its lineage
        self._order: dict[int, Any] = {}
        # objects in the store that no live node reads -- a shared object a commit (re)wrote and then did not publish,
        # or the present part of a payload registered evicted: counted live, evicted first
        self._orphans: dict[str, int] = {}
        self._written: dict[int, set] = {}  # a commit (key id) -> the absent objects it reserved and writes (writing)
        # durable publications whose bookkeeping failed (key id -> (key, node ref)): counted by their reservation and
        # registered before the next write is admitted (review round 2)
        self._unreconciled: dict[int, tuple[Any, str]] = {}
        self.live_bytes = self.reserved_bytes = self.peak_bytes = 0
        self.stats_ = {"evict": {"nodes": 0, "bytes": 0, "skipped_leased": 0, "orphans": 0}, "misses": 0,
                       "evicted_hits": 0,
                       "reservations": 0, "reserved_total_bytes": 0, "reserve_waits": 0, "reserve_wait_seconds": 0.0,
                       "republished": 0, "errors": 0,
                       # admitted writes: objects reserved (physically absent) and the ones found present (deduplicated)
                       "writes": {"objects": 0, "bytes": 0, "present_objects": 0, "present_bytes": 0},
                       "reconciled": 0}

    # -- the payload of a node ---------------------------------------------------------------------------------------
    def node_payload(self, node_ref: str) -> _Node:
        """The payload objects a restore of ``node_ref`` reads: its packets (a chunk node's own pack; a recache node's
        epoch range packs) -- their payload blobs and the row blocks of their basis versions (with parents)."""
        store = self.store
        node = store.get_manifest(node_ref)["node"]
        packets = [node["packet"]] if node.get("packet") else []
        if node.get("node_kind") in ("recache", "segment") and node.get("epoch"):
            epoch = store.get_manifest(node["epoch"])["epoch"]
            packets += [item["packet"] for item in epoch.get("ranges") or () if item.get("packet")]
        objects, versions = set(), []
        for packet_ref in dict.fromkeys(packets):
            packet = store.get_manifest(packet_ref)
            objects.update((packet.get("payloads") or {}).values())
            pending = [item["basis_version"] for item in packet.get("tile_bases") or ()]
            while pending:
                ref = pending.pop()
                if ref in versions:
                    continue
                versions.append(ref)
                record = store.get_manifest(ref)["record"]
                objects.add(record["added_rows"])
                if record.get("parent"):
                    pending.append(record["parent"])
        return _Node(frozenset(objects), tuple(dict.fromkeys(packets)), tuple(versions))

    # -- reservations (archival writers) ----------------------------------------------------------------------------
    def reserve(self, key: Any, nbytes: int) -> None:
        """Reserve ``nbytes`` for a write of commit ``key`` (before the store sees it): evict least recently used
        unpinned nodes until ``live + reserved + nbytes`` fits the budget; wait (bounded) when every candidate is
        pinned. Raises :class:`CapacityError` when it cannot fit -- at once when nothing can make room: no node to
        evict, no restore pin, and every other in-flight commit waits for room too or publishes only after one that
        does (``key.budget_order``: a later commit of the same lineage) -- and :class:`StateError` (at once) when the
        commit ended -- failed or cancelled, e.g. by a whole-tree delete or the warm-up purge -- before or while it
        waits."""
        nbytes = int(nbytes)
        if nbytes <= 0:
            return
        if nbytes > self.budget:
            raise CapacityError(f"an archive write of {nbytes} B exceeds archive_budget_bytes ({self.budget} B)")
        deadline, waited = None, 0.0
        with self._cond:
            self._reconcile()
            self._order.setdefault(id(key), getattr(key, "budget_order", None))
            try:
                while self.live_bytes + self.reserved_bytes + nbytes > self.budget:
                    self._raise_if_ended(key)
                    if self._evict_one():
                        continue
                    waiting = self._waiting | {id(key)}
                    if not self._pins and not self._orphans and all(self._blocked(other, waiting)
                                                                    for other in self._reserved if other != id(key)):
                        # nothing to evict, no restore to wait for, no orphan whose protection may lift (a failed
                        # commit's staging leases go right after it), and every other in-flight commit waits too or
                        # publishes only after one that does: no publication can make room (the budget is below the
                        # in-flight commits' bytes)
                        raise CapacityError(
                            f"archive budget {self.budget} B: {self.live_bytes} B live, {self.reserved_bytes} B "
                            f"reserved by in-flight commits; a write of {nbytes} B cannot fit and nothing can make "
                            "room (archive_budget_bytes is below the commits in flight)")
                    if deadline is None:
                        deadline = time.monotonic() + self.wait_seconds
                        self.stats_["reserve_waits"] += 1
                    left = deadline - time.monotonic()
                    if left <= 0:
                        raise CapacityError(
                            f"archive budget {self.budget} B: {self.live_bytes} B live, {self.reserved_bytes} B "
                            f"reserved; a write of {nbytes} B finds no unpinned node to evict")
                    started = time.monotonic()
                    self._waiting.add(id(key))
                    self._cond.wait(min(left, 1.0))
                    waited += time.monotonic() - started
                self._raise_if_ended(key)
            finally:
                self._waiting.discard(id(key))
                self.stats_["reserve_wait_seconds"] += waited
                if id(key) not in self._reserved:
                    self._order.pop(id(key), None)
            self._reserved[id(key)] = self._reserved.get(id(key), 0) + nbytes
            self._order[id(key)] = getattr(key, "budget_order", None)
            self.reserved_bytes += nbytes
            self.stats_["reservations"] += 1
            self.stats_["reserved_total_bytes"] += nbytes
            self.peak_bytes = max(self.peak_bytes, self.live_bytes + self.reserved_bytes)

    def _blocked(self, other: int, waiting: set) -> bool:
        """(caller holds the lock) Whether in-flight commit ``other`` (a reservation key id) cannot make room before
        the ``waiting`` ones: it waits for room itself, or it publishes only after one of them (a later commit of the
        same lineage: publication is ordered)."""
        if other in waiting:
            return True
        order = self._order.get(other)
        if order is None:
            return False
        return any(mine is not None and mine[0] == order[0] and mine[1] < order[1]
                   for mine in (self._order.get(key) for key in waiting))

    @staticmethod
    def _raise_if_ended(key: Any) -> None:
        """A commit that failed or was cancelled (``key.error``, set by the pipeline's ``_fail_from``) stops."""
        error = getattr(key, "error", None)
        if error is not None:
            raise StateError("the commit ended while it waited for archive budget room") from error

    def wake(self) -> None:
        """Wake the reservations that wait (a commit ended: it stops waiting at once)."""
        with self._cond:
            self._cond.notify_all()

    @contextlib.contextmanager
    def writing(self, key: Any, objects: Mapping[str, int]):
        """A commit's store transaction under the budget -- every payload write is admitted here (review round 2).
        ``objects`` (``{object id: bytes}``) are the content-addressed payload objects the transaction stores: basis
        row blocks, packet payload blobs, the HEVC-only arm's shared identity basis. Only those physically absent --
        never written, or evicted (writing the content again restores it) -- and not admitted for this commit before
        are reserved, each content once: a deduplicated write costs nothing. The lock is held until the block ends: no
        eviction runs between the presence check and the transaction's staging leases on what it writes and finds
        (:meth:`ObjectStore.evict`), and no other writer's check sees an object half admitted. A reservation that waits
        releases the lock, so the presence check runs again after it (an excess goes back). The commit counts as an
        in-flight writer also when it reserves nothing (the deadlock rule of :meth:`reserve`), and what it writes is
        remembered: if it ends unpublished those objects stay counted, as orphans (:meth:`release`)."""
        with self._cond:
            self._reconcile()
            self._raise_if_ended(key)
            ident = id(key)
            self._order.setdefault(ident, getattr(key, "budget_order", None))
            self._reserved.setdefault(ident, 0)
            written = self._written.setdefault(ident, set())
            counted, absent = 0, {}
            while True:
                ids = [object_id for object_id in objects if object_id not in written]
                present = (self.store.indexed(ids) - self.store.evicted_ids(ids)) if ids else set()
                absent = {object_id: int(objects[object_id]) for object_id in ids if object_id not in present}
                need = sum(absent.values())
                if need <= counted:
                    break
                self.reserve(key, need - counted)
                counted = need
            if counted > need:  # written meanwhile by another commit: the excess goes back
                self._reserved[ident] -= counted - need
                self.reserved_bytes -= counted - need
                self._cond.notify_all()
            written.update(absent)
            writes = self.stats_["writes"]
            writes["objects"] += len(absent)
            writes["bytes"] += need
            writes["present_objects"] += len(objects) - len(absent)
            writes["present_bytes"] += sum(int(size) for size in objects.values()) - need
            yield

    def release(self, key: Any) -> None:
        """A commit ended without publishing (failed or cancelled; the pipeline calls this before its staging leases
        go): its reservation goes and, in the same step, every object it wrote that is still in the store and read by
        no live node is counted as an orphan -- evicted first, or collected with the commit's staged objects -- so the
        bytes in the store are never understated (review round 2)."""
        with self._cond:
            ident = id(key)
            written = sorted(self._written.pop(ident, ()))
            self.reserved_bytes -= self._reserved.pop(ident, 0)
            self._order.pop(ident, None)
            self._unreconciled.pop(ident, None)
            self._orphan_many(written)
            self._cond.notify_all()

    def unreconciled(self, key: Any, node_ref: str, error: BaseException | None = None) -> None:
        """A durable publication whose bookkeeping failed outside :meth:`published` (the pipeline's last resort): its
        reservation and what it wrote stay counted; it is registered before the next write is admitted."""
        with self._cond:
            self._defer_registration(key, node_ref, error)

    def _defer_registration(self, key: Any, node_ref: str, error: BaseException | None) -> None:
        self._unreconciled[id(key)] = (key, node_ref)
        self.stats_["errors"] += 1
        if error is not None:
            self.stats_["last_error"] = f"{type(error).__name__}: {error}"

    def _reconcile(self) -> None:
        """(caller holds the lock) Durable publications whose bookkeeping failed are registered before any further
        write is admitted (review round 2); one that fails again stays counted by its reservation."""
        for ident, (key, node_ref) in list(self._unreconciled.items()):
            try:
                self._register(key, node_ref)
            except Exception as exc:  # noqa: BLE001 - conservative: it stays counted, retried before the next write
                self.stats_["errors"] += 1
                self.stats_["last_error"] = f"{type(exc).__name__}: {exc}"
            else:
                self.stats_["reconciled"] += 1

    def _orphan(self, object_id: str, size: int | None = None) -> None:
        """(caller holds the lock) ``object_id``, in the store but read by no live node, is counted live until it goes
        (the least recently used evictions take orphans first)."""
        if object_id in self._holders or object_id in self._orphans:
            return
        if size is None:
            if not self.store.indexed([object_id]) or self.store.evicted_ids([object_id]):
                return  # not physically present
            size = self.store.object_sizes([object_id])[object_id]
        self._orphans[object_id] = int(size)
        self.live_bytes += int(size)
        self.peak_bytes = max(self.peak_bytes, self.live_bytes + self.reserved_bytes)

    def _orphan_many(self, object_ids: list) -> None:
        """(caller holds the lock) :meth:`_orphan` for every one of ``object_ids`` still in the store (one query)."""
        ids = [object_id for object_id in object_ids
               if object_id not in self._holders and object_id not in self._orphans]
        if not ids:
            return
        present = sorted(self.store.indexed(ids) - self.store.evicted_ids(ids))
        sizes = self.store.object_sizes(present) if present else {}
        for object_id in present:
            self._orphan(object_id, sizes[object_id])

    def published(self, key: Any, node_ref: str) -> None:
        """Commit ``key`` published ``node_ref``: its payload becomes live (most recently used) and its reservation
        goes. A node published again (identical content, e.g. regenerated) is touched; one that was evicted is live
        again (the store restored its payload when the content was written again). The payload check and the holder
        registration run under this object's lock (then the store's): no eviction runs between them, so a node is
        never registered live with an object gone. Never raises: the publication is durable -- when its bookkeeping
        fails the reservation and what the commit wrote stay counted, and the node is registered before the next
        write is admitted (review round 2)."""
        with self._cond:
            try:
                self._register(key, node_ref)
            except Exception as exc:  # noqa: BLE001 - conservative accounting, reconciled before the next write
                self._defer_registration(key, node_ref, exc)
            self._cond.notify_all()

    def _register(self, key: Any, node_ref: str) -> None:
        """(caller holds the lock) :meth:`published`'s bookkeeping. Every store read runs before any state changes, so
        a failure leaves nothing half registered."""
        ident = id(key)
        payload = self.node_payload(node_ref)
        missing = self.store.evicted_ids(payload.objects)
        sizes = self.store.object_sizes(payload.objects)
        stray = [object_id for object_id in self._written.get(ident, ()) if object_id not in payload.objects]
        stray_present = sorted(self.store.indexed(stray) - self.store.evicted_ids(stray)) if stray else []
        stray_sizes = self.store.object_sizes(stray_present) if stray_present else {}
        # -- from here on no store call: the registration cannot fail half done
        self._written.pop(ident, None)
        self.reserved_bytes -= self._reserved.pop(ident, 0)
        self._order.pop(ident, None)
        self._unreconciled.pop(ident, None)
        if node_ref in self._nodes:
            self._nodes.move_to_end(node_ref)
        elif missing:  # content written again only in part (never for a deterministic commit): stays evicted;
            self._evicted[node_ref] = payload  # its present part is counted (orphans) until it goes
            for object_id in sorted(payload.objects - missing):
                self._orphan(object_id, sizes.get(object_id))
        else:
            if self._evicted.pop(node_ref, None) is not None:
                self.stats_["republished"] += 1
            self._add(node_ref, payload, sizes)
        for object_id in stray_present:  # written for this commit yet not part of its node: never for a real commit
            self._orphan(object_id, stray_sizes[object_id])
        try:
            self._enforce()
        except Exception as exc:  # noqa: BLE001 - the registration stands; an eviction error is recorded
            self.stats_["errors"] += 1
            self.stats_["last_error"] = f"{type(exc).__name__}: {exc}"

    def _add(self, node_ref: str, payload: _Node, sizes: dict) -> None:
        if not payload.objects:  # a root (or empty) node: nothing to evict, always restorable
            return
        for object_id in payload.objects:
            holders = self._holders.setdefault(object_id, set())
            if not holders:
                if self._orphans.pop(object_id, None) is None:  # an orphan is counted already
                    self.live_bytes += int(sizes.get(object_id, 0))
                self._sizes[object_id] = int(sizes.get(object_id, 0))
            holders.add(node_ref)
        self._nodes[node_ref] = payload
        self.peak_bytes = max(self.peak_bytes, self.live_bytes + self.reserved_bytes)

    def _enforce(self) -> None:
        """(caller holds the lock) Evict until the live and reserved bytes fit (a publication whose payload was
        larger than its reservation, or a restart)."""
        while self.live_bytes + self.reserved_bytes > self.budget and self._evict_one():
            pass

    def _evict_one(self) -> bool:
        """(caller holds the lock) Evict an orphan, else the least recently used node that can go whole (unpinned, none
        of its own objects protected in the store); False when there is none."""
        for object_id in list(self._orphans):
            result = self.store.evict([object_id])
            if not result["leased"]:
                self.live_bytes -= self._orphans.pop(object_id)
                self.stats_["evict"]["orphans"] += 1
                self.stats_["evict"]["bytes"] += result["bytes"]
                return True
        for node_ref in list(self._nodes):
            if self._pins.get(node_ref):
                continue
            if self._evict(node_ref):
                return True
        return False

    def _evict(self, node_ref: str) -> bool:
        """(caller holds the lock) Evict ``node_ref`` all or nothing: its own objects (no other live node reads them)
        leave the store together, or -- one of them protected (leased, or reachable from a pending commit's staging
        lease) -- nothing does and False is returned."""
        payload = self._nodes[node_ref]
        solo = [object_id for object_id in payload.objects if self._holders.get(object_id) == {node_ref}]
        result = self.store.evict(solo) if solo else {"objects": 0, "bytes": 0, "evicted": [], "leased": []}
        if result["leased"]:  # protected: the store evicted nothing (all or nothing)
            self.stats_["evict"]["skipped_leased"] += 1
            return False
        del self._nodes[node_ref]
        for object_id in payload.objects:
            holders = self._holders.get(object_id)
            if holders is not None:
                holders.discard(node_ref)
                if not holders:
                    del self._holders[object_id]
                    self.live_bytes -= self._sizes.pop(object_id, 0)
        self._evicted[node_ref] = payload
        self.stats_["evict"]["nodes"] += 1
        self.stats_["evict"]["bytes"] += result["bytes"]
        if self.on_evict is not None:
            try:
                self.on_evict(node_ref, payload.packets, payload.versions)
            except Exception:  # noqa: BLE001 - caches only; the payload is gone either way
                self.stats_["errors"] += 1
        return True

    # -- readers -----------------------------------------------------------------------------------------------------
    def is_evicted(self, node_ref: str) -> bool:
        with self._cond:
            return node_ref in self._evicted

    def evicted(self, node_refs: Iterable[str]) -> set:
        with self._cond:
            return {ref for ref in node_refs if ref in self._evicted}

    def pin_available(self, node_refs: Iterable[str], *, touch: bool = True) -> Any:
        """Pin ``node_refs`` (none evicted) for a restore that reads them and mark them used (``touch``: a restore;
        not a preparation); None (nothing pinned) when one of them is evicted. Release with :meth:`unpin`."""
        refs = tuple(dict.fromkeys(node_refs))
        with self._cond:
            if any(ref in self._evicted for ref in refs):
                return None
            for ref in refs:
                self._pins[ref] += 1
                if touch and ref in self._nodes:
                    self._nodes.move_to_end(ref)
            return refs

    def pin_live(self, node_refs: Iterable[str]) -> tuple:
        """Pin the refs of ``node_refs`` that are not evicted (no LRU touch): they stay readable until :meth:`unpin`
        of the returned token (the pinned refs) -- a restore's decision between its archive and a replay holds."""
        refs = tuple(dict.fromkeys(node_refs))
        with self._cond:
            live = tuple(ref for ref in refs if ref not in self._evicted)
            for ref in live:
                self._pins[ref] += 1
            return live

    def unpin(self, token: Any) -> None:
        if not token:
            return
        with self._cond:
            for ref in token:
                self._pins[ref] -= 1
                if self._pins[ref] <= 0:
                    del self._pins[ref]
            self._cond.notify_all()

    def count_miss(self, evicted_nodes: int) -> None:
        """A restore found ``evicted_nodes`` attended nodes evicted (records)."""
        with self._cond:
            self.stats_["misses"] += 1
            self.stats_["evicted_hits"] += int(evicted_nodes)

    # -- whole-tree delete, restart ----------------------------------------------------------------------------------
    def prune(self) -> dict[str, int]:
        """After garbage collection (a whole-tree delete): forget the nodes whose manifests the store no longer holds.
        One step under this object's lock (then the store's), the presence checked before anything is subtracted
        (review round 2): an object whose last live holder went stays counted -- as an orphan -- while another
        manifest keeps it in the store, and leaves ``live_bytes`` only once it is gone; collected orphans leave too;
        an unreconciled publication of a deleted node gives its reservation back (what it wrote and remains: orphans).
        Then the budget is enforced."""
        with self._cond:
            known = list(self._nodes) + list(self._evicted) + [ref for _, ref in self._unreconciled.values()]
            present = self.store.indexed(known) if known else set()
            dropped, freed = 0, {}
            for ref in [ref for ref in self._evicted if ref not in present]:
                del self._evicted[ref]
                dropped += 1
            for ref in [ref for ref in self._nodes if ref not in present]:
                payload = self._nodes.pop(ref)
                dropped += 1
                for object_id in payload.objects:
                    holders = self._holders.get(object_id)
                    if holders is not None:
                        holders.discard(ref)
                        if not holders:
                            del self._holders[object_id]
                            freed[object_id] = self._sizes.pop(object_id, 0)
            ids = list(freed) + list(self._orphans)
            remaining = (self.store.indexed(ids) - self.store.evicted_ids(ids)) if ids else set()
            for object_id, size in freed.items():
                if object_id in remaining:
                    self._orphans[object_id] = size  # still in the store: counted on, as an orphan
                else:
                    self.live_bytes -= size
            for object_id in [object_id for object_id in self._orphans if object_id not in remaining]:
                self.live_bytes -= self._orphans.pop(object_id)
            for ident, (_, ref) in list(self._unreconciled.items()):
                if ref not in present:  # deleted before it was reconciled
                    del self._unreconciled[ident]
                    written = sorted(self._written.pop(ident, ()))
                    self.reserved_bytes -= self._reserved.pop(ident, 0)
                    self._order.pop(ident, None)
                    self._orphan_many(written)
            try:
                self._enforce()
            except Exception as exc:  # noqa: BLE001 - recorded; the accounting above stands
                self.stats_["errors"] += 1
                self.stats_["last_error"] = f"{type(exc).__name__}: {exc}"
            self._cond.notify_all()
        return {"nodes": dropped}

    def adopt(self, node_refs: Iterable[str]) -> None:
        """Register published nodes found in the store (a reopened archive) in the given order -- parents before
        children, oldest first (the LRU order of the run that wrote them is not persisted; see the module docstring);
        evicted ones as such."""
        for ref in node_refs:
            with self._cond:
                if ref in self._nodes or ref in self._evicted:
                    continue
                payload = self.node_payload(ref)
                missing = self.store.evicted_ids(payload.objects)
                sizes = self.store.object_sizes(payload.objects)
                if missing:
                    self._evicted[ref] = payload
                    for object_id in sorted(payload.objects - missing):  # counted (an orphan) until a live node
                        self._orphan(object_id, sizes.get(object_id))  # adopted later reads it, or it goes
                else:
                    self._add(ref, payload, sizes)
        with self._cond:
            self._enforce()

    def stats(self) -> dict[str, Any]:
        with self._cond:
            return {"budget_bytes": self.budget, "live_bytes": self.live_bytes, "reserved_bytes": self.reserved_bytes,
                    "peak_bytes": self.peak_bytes, "live_nodes": len(self._nodes), "evicted_nodes": len(self._evicted),
                    "pinned_nodes": len(self._pins), "orphan_objects": len(self._orphans),
                    "orphan_bytes": sum(self._orphans.values()), "unreconciled": len(self._unreconciled),
                    **{key: (dict(value) if isinstance(value, dict) else value) for key, value in self.stats_.items()}}


__all__ = ["ArchiveEviction", "RESERVE_WAIT_SECONDS"]
