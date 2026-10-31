"""Device restore of packed lineage history through Step 5a's packed NVDEC path.

:class:`LineageRestorer` resolves a node's attended history (newest version
per range on the ancestor path, cache epochs included) to its node packs and
restores them with :class:`reelcache.packed_decode.PackedRestorer` (attended,
non-resident entries only) feeding :func:`reelcache.restore_kernels.reconstruct`,
directly into native cache slots with the model's RoPE.

* **Prepared packets at seal/load time.** Pack validation and tile-map
  preparation (30-45 ms per CausVid node) run once per packet: at publication
  (:meth:`LineageRestorer.adopt`, from the in-memory pack) or at first load,
  and are kept in a byte-bounded host cache keyed by the immutable packet
  reference. Cached tiles carry no basis bytes.
* **Device basis cache.** Tiles reference immutable basis versions. Their
  rank-view prefixes are served from :class:`DeviceBasisCache`: one device
  copy per version chain tip (a parent version is a prefix view of a cached
  descendant, and a resident unpinned ancestor is absorbed when its descendant
  arrives), charged against an explicit byte budget, LRU evicted only when no
  restore holds it. Versions are made resident at seal time from the
  in-memory pack (``adopt_bases``, best effort under the budget) or uploaded
  in one batch per restore.
* Legacy single-matrix packets (Step 4 per-matrix records, version-1 archives)
  are not decoded here; they keep the per-matrix path
  (:meth:`~reelcache.lineage.LineageRuntimeMixin.acquire_node_history`), which
  does not meet the restore budgets. Nothing falls back silently.
"""

from __future__ import annotations

import contextlib
import hashlib
import threading
import time
from collections import OrderedDict
from dataclasses import replace
from typing import Any, Callable, Iterable, Mapping, Sequence

import torch

from .errors import CapacityError, ConfigurationError, IntegrityError, StateError, UnsupportedError
from . import accel_c as _accel_c
from .types import entry_parts


def _normal_tensors():
    """Context for restorer-internal state: always normal tensors, whatever the caller's ``torch.inference_mode``.

    Canvases, resident bases and decoder-session pictures outlive the call that creates them and are written
    by decoder threads or later calls that may run outside inference mode (SGLang serves requests under
    ``torch.inference_mode``), where an in-place update of an inference tensor is an error. Reconstruction
    into the caller's destination still runs in the caller's mode.
    """
    return torch.inference_mode(False)


def _same_device(left: torch.device, right: torch.device) -> bool:
    if left.type != right.type:
        return False
    if left.type != "cuda":
        return True
    current = torch.cuda.current_device()
    return (current if left.index is None else left.index) == (current if right.index is None else right.index)


def _resident(rows: torch.Tensor, device: torch.device) -> torch.Tensor:
    """``rows`` on ``device`` as a normal tensor (an inference-tensor input is copied)."""
    with _normal_tensors():
        tensor = rows.to(device)
        return tensor.clone() if tensor.is_inference() else tensor


class DeviceBasisCache:
    """Immutable basis versions resident on one device under a charged byte budget.

    Resident versions are normal tensors even when acquired inside ``torch.inference_mode``.
    """

    def __init__(self, runtime, device: Any, budget_bytes: int):
        if type(budget_bytes) is not int or budget_bytes <= 0:
            raise ConfigurationError("the device basis cache needs a positive byte budget")
        self.runtime, self.device, self.budget_bytes = runtime, torch.device(device), budget_bytes
        self._entries: OrderedDict[str, dict] = OrderedDict()
        self._chain_index: dict[str, str] = {}
        self._lock = threading.Lock()
        self.stats_ = {"hits": 0, "prefix_hits": 0, "misses": 0, "uploads": 0, "upload_bytes": 0,
                       "evictions": 0, "absorbed": 0, "adopted": 0, "adopt_skipped": 0, "adopt_seconds": 0.0,
                       "resident_bytes": 0, "peak_resident_bytes": 0, "upload_seconds": 0.0}
        self._upload_stream = torch.cuda.Stream(self.device) if self.device.type == "cuda" else None

    def _owner(self, version_ref: str):
        """The resident entry serving ``version_ref`` (itself or a descendant chain tip); caller holds the lock."""
        owner = self._entries.get(version_ref)
        if owner is not None:
            return owner, "hits"
        tip = self._chain_index.get(version_ref)
        owner = None if tip is None else self._entries.get(tip)
        return (owner, "prefix_hits") if owner is not None else (None, None)

    def _pin(self, owner: dict, rank: int) -> torch.Tensor:
        if rank > owner["tensor"].shape[0]:
            raise IntegrityError("cached basis version is shorter than the requested rank view")
        owner["pins"] += 1
        self._entries.move_to_end(owner["ref"])
        return owner["tensor"][:rank]

    def acquire(self, version_ref: str, rank: int) -> tuple[torch.Tensor, str]:
        """A pinned contiguous ``[rank, C]`` FP16 device prefix and its pin token (see :meth:`acquire_many`)."""
        views, tokens = self.acquire_many([(version_ref, rank)])
        return views[0], tokens[0]

    def acquire_many(self, requests: Sequence[tuple[str, int]], *,
                     readiness: list | None = None) -> tuple[list[torch.Tensor], list[str]]:
        """Pinned device prefixes for ``(version_ref, rank)`` requests and one pin token per request.

        A token names the resident entry that serves the view (the version or a
        descendant chain tip); pass each to :meth:`release`. Misses are
        resolved in one store transaction, longest version first so an ancestor
        in the same batch becomes a prefix of its descendant's upload; each row
        block is read and checked once per batch and all uploads complete
        before return (one synchronization). On failure no pin taken by this
        call survives. A version adopted with a pending readiness
        (:meth:`adopt_device`) is waited for on the host before return, unless
        ``readiness`` is a list: then it receives, per request, that pending
        readiness (or None) for the caller's consumers to wait on (e.g.
        ``PreparedTile.ready``).
        """
        views: list = [None] * len(requests)
        tokens: list = [None] * len(requests)
        pinned: list[dict] = []
        batch: list[dict] = []
        missing: dict[str, list[int]] = {}
        pending: list = [None] * len(requests)

        def pin(index: int, owner: dict) -> None:
            views[index] = self._pin(owner, requests[index][1])
            tokens[index] = owner["ref"]
            pinned.append(owner)
            pending[index] = self._pending(owner)

        try:
            with self._lock:
                for index, (version_ref, _) in enumerate(requests):
                    owner, kind = self._owner(version_ref)
                    if owner is None:
                        missing.setdefault(version_ref, []).append(index)
                        continue
                    pin(index, owner)
                    self.stats_[kind] += 1
            if not missing:
                return views, self._resolve_readiness(tokens, pending, readiness)
            started = time.perf_counter()
            runtime = self.runtime
            versions = runtime.basis_versions
            with runtime._lock, runtime.store.transaction():
                versions.load_many(missing)  # batched manifest reads / index queries
                plans = sorted(((versions.load(ref).total_rank, ref, [item for item, _ in versions.chain(ref)])
                                for ref in missing), key=lambda plan: (-plan[0], plan[1]))
                covered: set[str] = set()
                blocks: dict[str, bytes] = {}
                # Every block of the uploads, read and checked in one parallel batch.
                versions.prefetch_blocks([(ref, total) for total, ref, _ in plans], blocks)
                assembled = []
                for _, ref, chain in plans:
                    if ref in covered:
                        continue
                    covered.update(chain)
                    assembled.append((ref, chain, versions.assemble(ref, max_bytes=runtime.max_basis_view_bytes,
                                                                    block_cache=blocks)))
            with self._lock:
                for ref, chain, rows in assembled:
                    if self._owner(ref)[0] is not None:
                        continue
                    self._make_room(rows.numel() * rows.element_size())
                    # Held by the batch until its requests are pinned, so it cannot evict itself.
                    batch.append(self._insert(ref, _resident(rows, self.device), chain, pins=1))
                    self.stats_["uploads"] += 1
                    self.stats_["upload_bytes"] += batch[-1]["bytes"]
                if batch and self.device.type == "cuda":
                    torch.cuda.current_stream(self.device).synchronize()
                for ref, indices in missing.items():
                    owner, _ = self._owner(ref)
                    if owner is None:
                        raise IntegrityError("an assembled basis version is not resident after upload")
                    self.stats_["misses"] += 1
                    for index in indices:
                        pin(index, owner)
                for entry in batch:
                    entry["pins"] -= 1
                batch.clear()
                self.stats_["upload_seconds"] += time.perf_counter() - started
            return views, self._resolve_readiness(tokens, pending, readiness)
        except BaseException:
            with self._lock:
                for entry in pinned + batch:
                    entry["pins"] -= 1
            raise

    def _insert(self, ref: str, tensor: torch.Tensor, chain: list[str], *, pins: int) -> dict:
        """Index a resident version; unpinned resident ancestors become prefix views of it. Caller holds the lock."""
        entry = {"ref": ref, "tensor": tensor, "bytes": tensor.numel() * tensor.element_size(), "pins": pins,
                 "chain": chain}
        self._entries[ref] = entry
        for ancestor in chain:
            current = self._chain_index.get(ancestor)
            if current is None or self._entries[current]["tensor"].shape[0] < tensor.shape[0]:
                self._chain_index[ancestor] = ref
        for ancestor in chain[:-1]:
            old = self._entries.get(ancestor)
            if old is not None and old["pins"] == 0:
                del self._entries[ancestor]
                self.stats_["resident_bytes"] -= old["bytes"]
                self.stats_["absorbed"] += 1
                for key, tip in list(self._chain_index.items()):
                    if tip == ancestor:
                        self._chain_index[key] = ref
        self.stats_["resident_bytes"] += entry["bytes"]
        self.stats_["peak_resident_bytes"] = max(self.stats_["peak_resident_bytes"], self.stats_["resident_bytes"])
        return entry

    def adopt(self, items: Sequence[tuple[str, list[str], torch.Tensor]]) -> int:
        """Make just-sealed versions resident from their host rows (seal time; no store reads).

        ``items`` are ``(version_ref, chain, rows)`` with ``rows`` the version's
        complete FP16 view (the caller checks it against the version digest).
        Uploads run on a private stream and complete before the versions become
        visible. Best effort under the budget: versions that do not fit next to
        pinned ones are skipped and counted (a later restore uploads them).
        """
        started = time.perf_counter()
        with self._lock:
            todo = [(ref, chain, rows) for ref, chain, rows in items
                    if (self._owner(ref)[0] is None or self._owner(ref)[0]["tensor"].shape[0] < rows.shape[0])]
        if not todo:
            return 0
        uploaded = []
        if self._upload_stream is not None:
            with torch.cuda.stream(self._upload_stream):
                uploaded = [(ref, chain, _resident(rows, self.device)) for ref, chain, rows in todo]
            self._upload_stream.synchronize()
        else:
            uploaded = [(ref, chain, _resident(rows, self.device)) for ref, chain, rows in todo]
        adopted = 0
        with self._lock:
            for ref, chain, tensor in uploaded:
                owner = self._owner(ref)[0]
                if owner is not None and owner["tensor"].shape[0] >= tensor.shape[0]:
                    continue
                try:
                    self._make_room(tensor.numel() * tensor.element_size())
                except CapacityError:
                    self.stats_["adopt_skipped"] += 1
                    continue
                self._insert(ref, tensor, chain, pins=0)
                adopted += 1
            self.stats_["adopted"] += adopted
            self.stats_["adopt_seconds"] += time.perf_counter() - started
        return adopted

    @staticmethod
    def _pending(owner: dict):
        """The entry's readiness while its rows may still be in flight (cleared once observed complete).

        A staging ticket not yet done is returned as is; once done its CUDA event (or the staging failure) is
        taken; a completed event clears the entry's readiness.
        """
        ready = owner.get("ready")
        if ready is None:
            return None
        if callable(getattr(ready, "event", None)):
            if not ready.done():
                return ready
            ready = owner["ready"] = ready.event(0)
        if ready is None or ready.query():
            owner["ready"] = None
            return None
        return ready

    @staticmethod
    def _resolve_readiness(tokens: list, pending: list, readiness: list | None) -> list:
        """Hand the pending readiness to the caller (``readiness`` list) or wait for it on the host."""
        if readiness is not None:
            readiness[:] = pending
            return tokens
        for ready in dict.fromkeys(item for item in pending if item is not None):
            event = ready.event(None) if callable(getattr(ready, "event", None)) else ready
            if event is not None:
                event.synchronize()
        return tokens

    def adopt_device(self, version_ref: str, chain: Sequence[str], tensor: torch.Tensor, ready: Any = None) -> bool:
        """Make a complete version resident from rows already on this cache's device (e.g. staged by
        :mod:`reelcache.restore_staging`): no store reads, no copy.

        ``tensor`` is the version's complete FP16 ``[total_rank, channels]`` view on this device, already checked
        against the version digest by the caller; ``chain`` its version chain, root first, ending at
        ``version_ref`` (checked against the stored chain). ``ready`` is None (rows complete), a CUDA event after
        which they are, or an object whose ``event(timeout)`` returns that event (a staging ticket): views served
        before it completes carry it (:meth:`acquire_many` ``readiness``) or are waited for. Best effort under the
        budget like :meth:`adopt`: returns False when the version does not fit next to pinned versions or a
        resident entry already serves it; the tensor is kept, not copied. Many versions: :meth:`adopt_device_many`.
        """
        return self.adopt_device_many([(version_ref, chain, tensor, ready)])[0]

    def adopt_device_many(self, items: Sequence[tuple[str, Sequence[str], torch.Tensor, Any]]) -> list[bool]:
        """:meth:`adopt_device` of many ``(version_ref, chain, tensor, ready)`` items (a staged history's complete
        versions) with batched validation: the versions and their chains are parsed by
        :meth:`BasisVersionStore.load_many` in one store transaction (none when all are parsed already), every item
        is checked before any is inserted (an invalid item raises and adopts nothing), then the items are inserted
        in order under the cache lock. Returns one flag per item, as :meth:`adopt_device`.
        """
        items = [tuple(item) for item in items]
        for item in items:
            if len(item) != 4:
                raise ConfigurationError("adopted device versions are (version_ref, chain, tensor, ready) items")
            tensor = item[2]
            if not isinstance(tensor, torch.Tensor) or tensor.dtype != torch.float16 or tensor.ndim != 2:
                raise ConfigurationError("an adopted device version is an FP16 [rows, channels] tensor")
            if not _same_device(tensor.device, self.device):
                raise ConfigurationError("an adopted device version must already live on the cache's device")
        if not items:
            return []
        runtime, versions = self.runtime, self.runtime.basis_versions
        refs = list(dict.fromkeys(item[0] for item in items))
        with runtime._lock:
            if versions.missing(refs):
                with runtime.store.transaction():
                    versions.load_many(refs)
            stored = {ref: [item for item, _ in versions.chain(ref)] for ref in refs}
            loaded = {ref: versions.load(ref) for ref in refs}
        for version_ref, chain, tensor, _ in items:
            if list(chain) != stored[version_ref]:
                raise IntegrityError("an adopted device version's chain differs from the stored chain")
            version = loaded[version_ref]
            if tuple(tensor.shape) != (version.total_rank, version.channels):
                raise IntegrityError("an adopted device version must be the complete version view")
        results = []
        with self._lock:
            for version_ref, _, tensor, ready in items:
                owner = self._owner(version_ref)[0]
                if owner is not None and owner["tensor"].shape[0] >= tensor.shape[0]:
                    results.append(False)
                    continue
                try:
                    self._make_room(tensor.numel() * tensor.element_size())
                except CapacityError:
                    self.stats_["adopt_skipped"] += 1
                    results.append(False)
                    continue
                entry = self._insert(version_ref, tensor, stored[version_ref], pins=0)
                entry["ready"] = ready
                self.stats_["adopted_device"] = self.stats_.get("adopted_device", 0) + 1
                results.append(True)
        return results

    def _make_room(self, nbytes: int) -> None:
        if nbytes > self.budget_bytes:
            raise CapacityError("one basis version exceeds the device basis cache budget")
        while self.stats_["resident_bytes"] + nbytes > self.budget_bytes:
            victim = next((entry for entry in self._entries.values() if entry["pins"] == 0), None)
            if victim is None:
                raise CapacityError("device basis cache is full of versions held by in-flight restores")
            self._drop(victim)
            self.stats_["evictions"] += 1

    def _drop(self, victim: dict) -> None:
        """Remove an unpinned resident entry; the ancestors it served move to another resident descendant (the
        longest) or leave the index. Caller holds the lock."""
        del self._entries[victim["ref"]]
        for ancestor, tip in list(self._chain_index.items()):
            if tip == victim["ref"]:
                replacement = max((entry for entry in self._entries.values() if ancestor in entry["chain"]),
                                  key=lambda entry: entry["tensor"].shape[0], default=None)
                if replacement is None:
                    del self._chain_index[ancestor]
                else:
                    self._chain_index[ancestor] = replacement["ref"]
        self.stats_["resident_bytes"] -= victim["bytes"]

    def forget(self, refs: Iterable[str]) -> dict[str, int]:
        """Drop the resident versions named by ``refs`` (e.g. versions the store's garbage collection reclaimed)
        that no restore holds; a pinned one stays (counted in ``held``). Returns the versions and bytes dropped."""
        dropped = released = held = 0
        with self._lock:
            for ref in dict.fromkeys(refs):
                entry = self._entries.get(ref)
                if entry is None:
                    continue
                if entry["pins"]:
                    held += 1
                    continue
                self._drop(entry)
                dropped += 1
                released += entry["bytes"]
            self.stats_["forgotten"] = self.stats_.get("forgotten", 0) + dropped
            self.stats_["forgotten_bytes"] = self.stats_.get("forgotten_bytes", 0) + released
        return {"versions": dropped, "bytes": released, "held": held}

    def refs(self) -> list[str]:
        """The resident version references (a snapshot)."""
        with self._lock:
            return list(self._entries)

    def release(self, token: str) -> None:
        """Release one pin token returned by :meth:`acquire` / :meth:`acquire_many`."""
        with self._lock:
            owner = self._entries.get(token)
            if owner is None or owner["pins"] <= 0:
                raise StateError("releasing a basis version that is not pinned")
            owner["pins"] -= 1

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {**self.stats_, "versions": len(self._entries), "budget_bytes": self.budget_bytes,
                    "pinned": sum(entry["pins"] > 0 for entry in self._entries.values())}

    def clear(self) -> None:
        with self._lock:
            if any(entry["pins"] for entry in self._entries.values()):
                raise StateError("cannot clear basis versions held by in-flight restores")
            self._entries.clear()
            self._chain_index.clear()
            self.stats_["resident_bytes"] = 0


def _retire_behind(handles: list, release_behind: int | None) -> int:
    """accel L (2026-10-30): after the newest reconstruction handle was appended, wait for and release the handle
    ``release_behind + 1`` positions behind it (its inputs and temporaries are dropped once its event completed).
    None: nothing (the default: every handle is held until the restore ends). Returns the handles released."""
    if release_behind is None:
        return 0
    index = len(handles) - release_behind - 2
    if index < 0:
        return 0
    handle = handles[index]
    release = getattr(handle, "release", None)
    if release is None or getattr(handle, "released", False):
        return 0
    release()
    return 1


def _verified_payloads(store) -> int | None:
    """The store's per-thread count of hash-checked payloads (Step 5c ``ObjectStore.verified_payloads``), or None."""
    counter = getattr(store, "verified_payloads", None)
    return None if counter is None else counter()


class RestoreActivity:
    """Restores in flight on a restorer, for archival workers that yield to them (SGLang Step 7; Step 5b's stall
    diagnosis: Python-heavy background archival slowed RAVEN restores by 9-16 % through GIL contention).

    :meth:`running` brackets one restore (restorers wrap ``restore`` in it). ``epoch`` counts busy periods (it
    increments when the first of overlapping restores starts). :meth:`wait_idle` blocks without holding the GIL
    until no restore runs or ``max_seconds`` passed and returns whether none runs.
    """

    def __init__(self):
        self._cond = threading.Condition()
        self._active = 0
        self.epoch = 0
        self.restores = 0

    @contextlib.contextmanager
    def running(self):
        with self._cond:
            if not self._active:
                self.epoch += 1
            self._active += 1
            self.restores += 1
        try:
            yield
        finally:
            with self._cond:
                self._active -= 1
                if not self._active:
                    self._cond.notify_all()

    def active(self) -> bool:
        return self._active > 0

    def wait_idle(self, max_seconds: float | None) -> bool:
        with self._cond:
            return self._cond.wait_for(lambda: self._active == 0, timeout=max_seconds)

    def stats(self) -> dict[str, Any]:
        with self._cond:
            return {"restores": self.restores, "epoch": self.epoch, "active": self._active}


class _PreparedCache:
    """Byte-bounded host cache of prepared (validated, tile-mapped) node packs without bases."""

    def __init__(self, budget_bytes: int):
        self.budget_bytes = budget_bytes
        self._entries: OrderedDict[str, tuple] = OrderedDict()
        self._bytes = 0
        self._lock = threading.Lock()
        self.stats_ = {"hits": 0, "misses": 0, "adopted": 0, "evictions": 0, "prepare_seconds": 0.0,
                       "load_seconds": 0.0}

    def contains(self, ref: str) -> bool:
        """Whether ``ref`` is prepared (no statistics, no LRU update)."""
        with self._lock:
            return ref in self._entries

    def touch(self, ref: str) -> bool:
        """Mark ``ref`` most recently used when it is prepared (no hit statistics); returns whether it was."""
        with self._lock:
            if ref not in self._entries:
                return False
            self._entries.move_to_end(ref)
            return True

    def get(self, ref: str):
        with self._lock:
            item = self._entries.get(ref)
            if item is not None:
                self._entries.move_to_end(ref)
                self.stats_["hits"] += 1
                return item[0]
            self.stats_["misses"] += 1
            return None

    def put(self, ref: str, prepared, bases: Sequence[tuple[str, int]]) -> None:
        nbytes = prepared.stream_bytes
        with self._lock:
            if ref in self._entries:
                return
            while self._entries and self._bytes + nbytes > self.budget_bytes:
                _, (_, old_bytes) = self._entries.popitem(last=False)
                self._bytes -= old_bytes
                self.stats_["evictions"] += 1
            if nbytes <= self.budget_bytes:
                self._entries[ref] = ((prepared, tuple(bases)), nbytes)
                self._bytes += nbytes

    def forget(self, refs: Iterable[str]) -> int:
        """Drop the prepared packs named by ``refs``; returns how many were cached."""
        dropped = 0
        with self._lock:
            for ref in dict.fromkeys(refs):
                item = self._entries.pop(ref, None)
                if item is not None:
                    self._bytes -= item[1]
                    dropped += 1
        return dropped

    def refs(self) -> list[str]:
        with self._lock:
            return list(self._entries)

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {**self.stats_, "packets": len(self._entries), "bytes": self._bytes,
                    "budget_bytes": self.budget_bytes}

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._bytes = 0


class _BlockCache(dict):
    """Checked basis row blocks shared by the packs of one history load, bounded in bytes."""

    def __init__(self, limit_bytes: int):
        super().__init__()
        self.limit_bytes, self.nbytes = limit_bytes, 0

    def __setitem__(self, key, value) -> None:
        if key not in self and self.nbytes + len(value) <= self.limit_bytes:
            super().__setitem__(key, value)
            self.nbytes += len(value)


def _strip_bases(prepared):
    return replace(prepared, tiles=tuple(replace(tile, basis=None, **({"ready": None} if hasattr(tile, "ready") else {}))
                                         for tile in prepared.tiles))


class LineageRestorer:
    """Packed device restore of lineage history (see module docstring).

    ``pool`` is a :class:`~reelcache.decoder_pool.DecoderPool` on ``device``;
    ``arena_bytes`` bounds decoded canvases; ``basis_budget_bytes`` bounds
    resident bases; ``prepared_budget_bytes`` bounds cached prepared packs.
    ``staging_engine`` (optional; e.g. :class:`reelcache.restore_staging.StagingEngine`)
    stages a history whose packs are not all prepared (cold session, Step 5c):
    ``stage(runtime, history)`` returns an object with ``refs`` (``NodeRef``
    values whose tiles carry device bases and ``ready`` readiness), ``wait()``,
    ``release()``, ``versions`` (``(version_ref, chain, tensor, ready)`` of
    complete versions), ``prepared`` / ``bindings`` (per packet) and ``stats``.
    When the restore finishes, its staged packs and complete versions are
    adopted into the prepared and device basis caches.
    """

    #: decomp_opt: the class default for restorers that set up their own state without this __init__ (W1.2's
    #: FactorLineageRestorer: no NVDEC packed restorer, no canvas arena): never pipelined
    pipelined_staging = False
    #: accel A1: likewise never a decoded tier for such restorers
    decoded_tier = None

    def __init__(self, runtime, pool, *, arena_bytes: int = 2 * 1024**3, basis_budget_bytes: int = 1024**3,
                 prepared_budget_bytes: int = 2 * 1024**3, granularity: Any = "layer", adopt_bases: bool = True,
                 staging_engine: Any = None, pipelined_staging: bool = False, decoded_tier: Any = None):
        from .packed_decode import PackedRestorer
        if type(adopt_bases) is not bool:
            raise ConfigurationError("adopt_bases must be boolean")
        if type(pipelined_staging) is not bool:
            raise ConfigurationError("pipelined_staging must be boolean")
        if staging_engine is not None and not callable(getattr(staging_engine, "stage", None)):
            raise ConfigurationError("a staging engine provides stage(runtime, history)")
        self.staging_engine = staging_engine
        # decomp_opt (2026-10-25): a cold (staged) restore submits each pack's decode jobs as soon as staging
        # prepared it (:meth:`_restore_pipelined`); False (default) stages the whole history first.
        self.pipelined_staging = pipelined_staging
        # accel A1 (2026-10-30): a reelcache.decoded_tier.DecodedTier (owned: closed by close()) routes cached nodes of a
        # restore past NVDEC; None (default) decodes every node.
        if decoded_tier is not None and not callable(getattr(decoded_tier, "route", None)):
            raise ConfigurationError("a decoded tier provides route(history, batch)")
        self.decoded_tier = decoded_tier
        self._staged: dict[str, Any] = {}
        self.last_staging: dict[str, Any] | None = None
        # The payload verification record of the calling thread's latest node_refs (freeze D, D-E): the restore
        # report's ``verification`` (``restore.lease.verification``); ``last_verification`` mirrors it.
        self._local = threading.local()
        self.last_verification: dict[str, Any] | None = None
        # Restores in flight (archive pipelines built with this restorer yield to them).
        self.activity = RestoreActivity()
        self.runtime, self.pool = runtime, pool
        # Freeze D (D-G, Step 5c): a pool that can hold queued warm-ups while restores run defers them to this
        # restorer's activity (older pools: no such hook, nothing changes).
        defer = getattr(pool, "defer_warm_ups_to", None)
        if defer is not None:
            defer(self.activity)
        self.adopt_bases = adopt_bases
        self.block_cache_bytes = 256 * 1024**2
        self.device = pool.device
        self.restorer = PackedRestorer(pool, arena_bytes=arena_bytes, granularity=granularity)
        self.bases = DeviceBasisCache(runtime, self.device, basis_budget_bytes)
        self.prepared = _PreparedCache(prepared_budget_bytes)
        self._warm_geometries: set = set()
        self.stats_ = {"restores": 0, "entries": 0, "nodes": 0, "seconds": 0.0, "plan_seconds": 0.0,
                       "device_trims": 0, "device_trimmed_bytes": 0}
        # Freeze F: under device-memory pressure the raw tier's response trims the unpinned resident bases (order
        # "restorer", after the decoder pool's canvases); held weakly, unregistered at close.
        self._trim_handle = None
        if self.device.type == "cuda":
            from .device_pressure import register_device_trim
            self._trim_handle = register_device_trim(f"restorer:{self.device.index}", self.trim_device_cache,
                                                     device=self.device)

    def trim_device_cache(self, reason: str) -> int:
        """Device-pressure trim (``reelcache.device_pressure``): drop the resident basis versions no restore pins; they
        are uploaded again on demand. Returns the bytes released."""
        released = int(self.bases.forget(self.bases.refs())["bytes"])
        self.stats_["device_trims"] += 1
        self.stats_["device_trimmed_bytes"] += released
        return released

    # -- preparation ----------------------------------------------------------------
    def _full_views(self, packet) -> dict[int, tuple[str, str, str, list[str]]]:
        """Tiles whose rank view is their whole version and whose descriptor carries the version digest.

        Caller holds the runtime lock inside a store transaction.
        """
        if not self.adopt_bases:
            return {}
        versions = self.runtime.basis_versions
        metadata = packet["metadata"]
        versions.load_many(binding["basis_version"] for binding in packet["tile_bases"])
        full = {}
        for index, (tile, binding) in enumerate(zip(metadata["tiles"], packet["tile_bases"])):
            version = versions.load(binding["basis_version"])
            descriptor = metadata["payloads"].get(tile["basis"]["payload"])
            if (binding["rank_view"] == version.total_rank and isinstance(descriptor, Mapping)
                    and descriptor.get("sha256") == version.view_sha256):
                full[index] = (binding["basis_version"], tile["basis"]["payload"], descriptor["sha256"],
                               [ref for ref, _ in versions.chain(binding["basis_version"])])
        return full

    def _adopt_views(self, prepared, encoded, full, *, check_bytes: bool) -> None:
        if not full:
            return
        by_index = {tile.index: tile for tile in prepared.tiles}
        items = []
        for index, (version_ref, payload, digest, chain) in full.items():
            if check_bytes and (  # accel C (hash_once): the encode-time digest of this very payload object, if recorded
                    _accel_c.known_digest(encoded, payload, encoded.payloads[payload])
                    or hashlib.sha256(encoded.payloads[payload]).hexdigest()) != digest:
                raise IntegrityError("sealed pack basis bytes differ from their version digest")
            items.append((version_ref, chain, by_index[index].basis))
        self.bases.adopt(items)

    def adopt(self, packet_ref: str, encoded) -> None:
        """Prepare a just-published in-memory pack (bases embedded) at seal time.

        With ``adopt_bases`` the pack's complete version views (tiles whose
        rank view is their whole version) also become resident in the device
        basis cache, after their bytes are checked against the tile descriptor
        and the version digest; shared-prefix tiles reuse their ancestors.

        Seal-time preparation only speeds up later restores. A failure other than an integrity error is recorded
        (``seal_stats()``) and never fails the commit (freeze D, R2):
        * a failed decoder warm-up (e.g. out of memory creating a session for a new canvas shape) leaves the
          geometry cold;
        * a failed preparation leaves the pack unprepared.
        A later restore prepares and warms the normal way. Integrity errors (bytes differing from their digests)
        still raise.
        """
        try:
            with _normal_tensors():
                self._adopt(packet_ref, encoded)
        except IntegrityError:
            raise
        except Exception as exc:  # noqa: BLE001 - recorded; the node stays restorable from the store
            self.prepared.forget([packet_ref])
            self._seal_failure("prepare", exc)

    def _adopt(self, packet_ref: str, encoded) -> None:
        started = time.perf_counter()
        with self.runtime._lock, self.runtime.store.transaction():
            locked = time.perf_counter()
            packet = self.runtime.load_packet(packet_ref)
            full = self._full_views(packet)
        loaded = time.perf_counter()
        self._bind_decoded(packet_ref, packet)
        prepared = self._prepare_packet(encoded)
        self._check_tiles(prepared, packet)
        validated = time.perf_counter()
        self._adopt_views(prepared, encoded, full, check_bytes=True)
        adopted = time.perf_counter()
        self.prepared.put(packet_ref, _strip_bases(prepared),
                          [(item["basis_version"], item["rank_view"]) for item in packet["tile_bases"]])
        self.prepared.stats_["adopted"] += 1
        self.prepared.stats_["prepare_seconds"] += time.perf_counter() - started
        for name, value in (("seal_lock_wait_seconds", locked - started), ("seal_load_seconds", loaded - locked),
                            ("seal_prepare_packet_seconds", validated - loaded),
                            ("seal_adopt_bases_seconds", adopted - validated)):
            self.prepared.stats_[name] = self.prepared.stats_.get(name, 0.0) + value
        # Decoder sessions cannot change picture size: create sessions for a new canonical canvas now.
        geometry = (prepared.width, prepared.height, prepared.pix_fmt)
        if geometry not in self._warm_geometries:
            warm_started = time.perf_counter()
            try:
                self.restorer.prewarm([prepared])
            except IntegrityError:
                raise
            except Exception as exc:  # noqa: BLE001 - the geometry stays cold: a later restore creates its sessions
                self._seal_failure("prewarm", exc)
            else:
                self._warm_geometries.add(geometry)
            self.prepared.stats_["prewarm_seconds"] = self.prepared.stats_.get("prewarm_seconds", 0.0) + (
                time.perf_counter() - warm_started)

    def _seal_failure(self, stage: str, exc: BaseException) -> None:
        with self.prepared._lock:
            seal = self.prepared.stats_.setdefault("seal", {"prepare_failures": 0, "prewarm_failures": 0,
                                                            "last_seal_failure": None})
            seal[f"{stage}_failures"] += 1
            seal["last_seal_failure"] = f"{stage}: {type(exc).__name__}: {str(exc).strip()[:300]}"

    def seal_stats(self) -> dict[str, Any]:
        """Seal-time preparation records: packs adopted, preparation and warm-up failures, the last failure."""
        with self.prepared._lock:
            seal = dict(self.prepared.stats_.get("seal") or {"prepare_failures": 0, "prewarm_failures": 0,
                                                             "last_seal_failure": None})
            seal["adopted"] = self.prepared.stats_.get("adopted", 0)
        return seal

    def _bind_decoded(self, packet_ref: str, packet) -> None:
        """accel A1: tell the decoded tier which HEVC stream ``packet_ref`` holds (its payload digest)."""
        if self.decoded_tier is not None:
            from .decoded_tier import stream_digest

            self.decoded_tier.bind(packet_ref, stream_digest(packet.get("metadata")))

    def _check_tiles(self, prepared, packet) -> None:
        metadata = packet["metadata"]
        if len(prepared.tiles) != len(metadata["tiles"]):
            raise IntegrityError("prepared pack tiles differ from the stored packet")

    def _prepare_packet(self, encoded):
        """A validated pack prepared for this restorer (``packed_decode.prepare_packet``; wl-base: the factor-pack
        restorer of the SVD-only baseline prepares its own packs)."""
        from .packed_decode import prepare_packet
        return prepare_packet(encoded)

    def _load(self, packet_ref: str, block_cache: dict[str, bytes] | None = None):
        """A prepared pack without bases; a store load also makes its complete versions device-resident."""
        cached = self.prepared.get(packet_ref)
        if cached is not None:
            return cached
        started = time.perf_counter()
        # node_pack checks every assembled prefix against its tile descriptor.
        encoded = self.runtime.node_pack(packet_ref, block_cache=block_cache)
        loaded = time.perf_counter()
        prepared = self._prepare_packet(encoded)
        with self.runtime._lock, self.runtime.store.transaction():
            packet = self.runtime.load_packet(packet_ref)
            full = self._full_views(packet)
        self._check_tiles(prepared, packet)
        self._bind_decoded(packet_ref, packet)
        self._adopt_views(prepared, encoded, full, check_bytes=False)
        bases = tuple((item["basis_version"], item["rank_view"]) for item in packet["tile_bases"])
        self.prepared.stats_["load_seconds"] += loaded - started
        self.prepared.stats_["prepare_seconds"] += time.perf_counter() - loaded
        stripped = _strip_bases(prepared)
        self.prepared.put(packet_ref, stripped, bases)
        return stripped, bases

    def ensure_prepared(self, packet_refs: Sequence[str]) -> dict[str, int]:
        """Keep node packs prepared in host memory (freeze D: called when the raw tier drops a published entry's
        last raw copy, so its next restore is a prepared one). Prepared packs are touched in the LRU; the others
        are read, validated and prepared host-side (no device work: bases stay where they are). Best effort: never
        raises (a later restore prepares or reads cold)."""
        result = {"touched": 0, "prepared": 0, "failed": 0}
        for ref in dict.fromkeys(packet_refs):
            try:
                if self.prepared.touch(ref):
                    result["touched"] += 1
                    continue
                with _normal_tensors():
                    self._prepare_host(ref)
                result["prepared"] += 1
            except Exception:  # noqa: BLE001 - best effort
                result["failed"] += 1
        with self.prepared._lock:
            for name, value in result.items():
                key = f"ensure_{name}"
                self.prepared.stats_[key] = self.prepared.stats_.get(key, 0) + value
        return result

    def _prepare_host(self, packet_ref: str) -> None:
        started = time.perf_counter()
        encoded = self.runtime.node_pack(packet_ref)
        loaded = time.perf_counter()
        prepared = self._prepare_packet(encoded)
        with self.runtime._lock, self.runtime.store.transaction():
            packet = self.runtime.load_packet(packet_ref)
        self._check_tiles(prepared, packet)
        self._bind_decoded(packet_ref, packet)
        bases = tuple((item["basis_version"], item["rank_view"]) for item in packet["tile_bases"])
        self.prepared.stats_["load_seconds"] += loaded - started
        self.prepared.stats_["prepare_seconds"] += time.perf_counter() - loaded
        self.prepared.put(packet_ref, _strip_bases(prepared), bases)

    # -- planning ----------------------------------------------------------------------
    def history(self, node_ref: str, *, ranges: Sequence[tuple[int, int]] | None = None,
                audit: bool = False) -> list[dict]:
        """Whole committed ranges backing the attended history (newest version on the path).

        A range whose node (or epoch) records its node pack -- packed commits since SGLang Step 7 -- resolves to that
        pack without loading its matrix records (the restore reads the pack's own tile bindings). ``audit=True``, and
        every older range, loads and validates each matrix record and requires them to name one node pack (the
        recorded one, when present).
        """
        if type(audit) is not bool:
            raise ConfigurationError("audit must be boolean")
        runtime = self.runtime
        result = []
        with runtime._lock, runtime.store.transaction():
            pieces = runtime.resolve_history(node_ref, ranges)
            records = {piece.entry.node_ref + piece.entry.chunk_id: [runtime.load_record(ref)
                                                                      for _, ref in piece.entry.records]
                       for piece in pieces if audit or piece.entry.packet_ref is None}
        for piece in pieces:
            entry = piece.entry
            if (piece.token_start, piece.token_end) != (entry.token_start, entry.token_end):
                raise UnsupportedError("device restore restores whole committed ranges; request range-aligned history")
            packet = entry.packet_ref
            checked = records.get(entry.node_ref + entry.chunk_id)
            if checked is not None:
                packets = set()
                for record in checked:
                    if record.packet_tile is None:
                        raise UnsupportedError(
                            "per-matrix (Step 4 / version-1) records use acquire_node_history, not the packed device "
                            "path")
                    packets.add(record.packet_ref)
                if len(packets) != 1:
                    raise IntegrityError("one committed range must be one node pack")
                named = packets.pop()
                if packet is not None and named != packet:
                    raise IntegrityError("a range's matrix records name another node pack than the range")
                packet = named
            result.append({"node_id": f"{entry.node_ref}:{entry.chunk_id}", "packet": packet,
                           "token_start": entry.token_start, "token_end": entry.token_end,
                           "chunk_id": entry.chunk_id, "source": entry.source, "epoch_id": entry.epoch_id})
        return result

    def node_refs(self, history: Sequence[Mapping[str, Any]], on_prepared: Callable | None = None):
        """``NodeRef`` values with device bases attached (pins released by :meth:`finish`).

        With a ``staging_engine``, a history whose packs are not all prepared is
        staged by the engine (cold session; one pin token for the staged
        history, adopted into the caches by :meth:`finish`). Otherwise the
        whole history's basis views are acquired in one batch (see
        :meth:`DeviceBasisCache.acquire_many`); a view whose version is still
        being copied carries its readiness (``PreparedTile.ready``).
        ``on_prepared`` (staged histories only) is passed to the engine's ``stage``: it receives each pack's
        ``NodeRef`` as soon as the pack is prepared.
        """
        from .packed_decode import NodeRef
        history = list(history)
        if (self.staging_engine is not None and history
                and not all(self.prepared.contains(item["packet"]) for item in history)):
            if on_prepared is None:
                staged = self.staging_engine.stage(self.runtime, history)
            else:
                staged = self.staging_engine.stage(self.runtime, history, on_prepared=on_prepared)
            token = f"staged:{id(staged)}"
            self._staged[token] = staged
            self.last_staging = dict(getattr(staged, "stats", None) or {})
            self._set_verification(staged)
            return list(staged.refs), [token]
        since = _verified_payloads(self.runtime.store)
        loaded, requests, blocks = [], [], _BlockCache(self.block_cache_bytes)
        for item in history:
            stripped, bases = self._load(item["packet"], blocks)
            for tile, (_, rank) in zip(stripped.tiles, bases):
                if tile.rank != rank:
                    raise IntegrityError("prepared tile rank differs from its basis binding")
            loaded.append((item, stripped))
            requests.extend(bases)
        readiness: list = []
        views, pins = self.bases.acquire_many(requests, readiness=readiness)
        refs, offset = [], 0
        for item, stripped in loaded:
            count = len(stripped.tiles)
            tiles = tuple(replace(tile, basis=view) if ready is None else replace(tile, basis=view, ready=ready)
                          for tile, view, ready in zip(stripped.tiles, views[offset:offset + count],
                                                       readiness[offset:offset + count]))
            refs.append(NodeRef(item["node_id"], [replace(stripped, tiles=tiles)]))
            offset += count
        self._set_verification(None, since=since)  # after the store-path loads: their verified payloads count
        return refs, pins

    def _set_verification(self, staged, since=None) -> None:
        """The verification record of this restore: the staging engine's (``verification_report(store, staged)``,
        or ``verification_report(store, since=...)`` when nothing was staged: the payloads the store hash-checked
        for this thread since ``since``), the store path's without an engine (``store_verification_report``)."""
        store = self.runtime.store
        report = getattr(self.staging_engine, "verification_report", None)
        if callable(report):
            record = dict(report(store, staged) if staged is not None else report(store, since=since))
        else:
            from .restore_staging import store_verification_report

            record = store_verification_report(store, since)
        self._thread_state().verification = self.last_verification = record

    def _thread_state(self):
        local = self.__dict__.get("_local")
        if local is None:
            local = self.__dict__["_local"] = threading.local()
        return local

    def finish(self, pins: Sequence[str]) -> None:
        """Release the pins of :meth:`node_refs`; staged histories are adopted into the caches, then released.

        A staged history is adopted only after its :meth:`~reelcache.restore_staging.StagedHistory.wait` (the
        integrity gate) returned, and ``restore`` returns only after ``finish``: a failed verification fails the
        restore."""
        staged = [self._staged.pop(token) for token in pins if token in self._staged]
        try:
            for token in pins:
                if not token.startswith("staged:"):
                    self.bases.release(token)
        finally:
            error = None
            for item in staged:
                try:
                    self._adopt_staged(item)
                except BaseException as exc:  # noqa: BLE001 - every staged history is released first
                    error = error or exc
            if error is not None:
                raise error

    def _adopt_staged(self, staged) -> None:
        """Keep a finished staged history's prepared packs and complete versions; release the staging."""
        try:
            staged.wait()  # every basis copy enqueued; raises the staging failure (nothing is adopted then)
            for packet_ref, prepared in staged.prepared.items():
                self.adopt_prepared(packet_ref, prepared, staged.bindings[packet_ref])
            self.bases.adopt_device_many(staged.versions)  # one validation batch (at most one store transaction)
        finally:
            staged.release()

    def adopt_prepared(self, packet_ref: str, prepared, bindings: Sequence[tuple[str, int]]) -> None:
        """Keep a pack prepared elsewhere (e.g. staged) in the prepared cache (bases and readiness stripped)."""
        bindings = tuple((str(ref), int(rank)) for ref, rank in bindings)
        if len(bindings) != len(prepared.tiles) or any(tile.rank != rank
                                                        for tile, (_, rank) in zip(prepared.tiles, bindings)):
            raise IntegrityError("prepared tile ranks differ from their basis bindings")
        self.prepared.put(packet_ref, _strip_bases(prepared), bindings)

    def drop_caches(self) -> None:
        """Forget prepared packs and resident bases (no restore may hold them), and decoded-tier pictures."""
        self.bases.clear()
        self.prepared.clear()
        if self.decoded_tier is not None:
            self.decoded_tier.clear()

    def forget_unstored(self) -> dict[str, Any]:
        """Forget the prepared packs and resident basis versions whose objects the store no longer holds.

        Call after the store's garbage collection: seal-time adoption made every sealed node's versions resident,
        and the device basis cache is LRU-evicted only over its budget, so a deleted (or aborted) session's
        versions would otherwise stay on the device up to ``basis_budget_bytes`` (CR-17-3). Versions a restore
        holds stay. Returns what was dropped."""
        versions, packets = self.bases.refs(), self.prepared.refs()
        decoded = [] if self.decoded_tier is None else self.decoded_tier.packet_refs()  # accel A1
        if not versions and not packets and not decoded:
            return {"versions": 0, "bytes": 0, "held": 0, "packets": 0}
        present = self.runtime.store.indexed(versions + packets + decoded)
        report = self.bases.forget([ref for ref in versions if ref not in present])
        report["packets"] = self.prepared.forget([ref for ref in packets if ref not in present])
        if self.decoded_tier is not None:
            report["decoded_tier"] = self.decoded_tier.forget_packets([ref for ref in decoded if ref not in present])
        return report

    def prewarm(self, history: Sequence[Mapping[str, Any]]) -> float:
        """Create warm NVDEC sessions for the canvas sizes of ``history`` (off the critical path)."""
        with _normal_tensors():
            refs, pins = self.node_refs(history)
            try:
                return self.restorer.prewarm(refs)
            finally:
                self.finish(pins)

    def prewarm_lineages(self, branch_ids: Sequence[str] | None = None, *,
                         node_refs: Sequence[str] | None = None) -> dict[str, Any]:
        """Session-open preparation of stored history (call when a store is reopened, before any resume).

        Resolves the attended history of every lineage head (or of
        ``branch_ids`` / ``node_refs``), loads and prepares each node pack once
        (validation, tile maps), makes its complete basis versions
        device-resident and creates decoder sessions for its canvas sizes, so
        the first restore after a process restart runs like a restore of a
        just-sealed node. Bounded by the prepared-pack and basis budgets (LRU);
        returns the work done and its seconds (spent off the resume path).
        """
        started = time.perf_counter()
        skipped: list[str] = []
        if node_refs is None:
            lineages = self.runtime.list_lineages()
            if branch_ids is None:  # every lineage except those quarantined when the store was opened
                from .lineage import LINEAGE_PREFIX

                quarantined = getattr(self.runtime, "quarantined", {})
                skipped = sorted(name for name in lineages if LINEAGE_PREFIX + name in quarantined)
            names = sorted(set(lineages) - set(skipped)) if branch_ids is None else list(branch_ids)
            missing = [name for name in names if name not in lineages]
            if missing:
                raise StateError(f"lineages do not exist: {missing}")
            node_refs = [lineages[name] for name in names]
        items, seen = [], set()
        for ref in node_refs:
            for item in self.history(ref):
                if item["packet"] not in seen:
                    seen.add(item["packet"])
                    items.append(item)
        resolved = time.perf_counter()
        before = dict(self.prepared.stats_)
        decoder_seconds = self.prewarm(items) if items else 0.0
        report = {"heads": len(node_refs), "packets": len(items), "seconds": time.perf_counter() - started,
                  "resolve_seconds": resolved - started, "decoder_session_seconds": decoder_seconds,
                  "load_seconds": self.prepared.stats_.get("load_seconds", 0.0) - before.get("load_seconds", 0.0),
                  "prepare_seconds": (self.prepared.stats_.get("prepare_seconds", 0.0)
                                      - before.get("prepare_seconds", 0.0)),
                  "bases": self.bases.stats()}
        if skipped:
            report["quarantined"] = skipped
        return report

    # -- restore ------------------------------------------------------------------------
    def restore(self, node_ref: str, destination: Callable, rope: Callable | None = None, *,
                ranges: Sequence[tuple[int, int]] | None = None, keys: Sequence[Any] | None = None,
                resident: Sequence[Any] = (), stream: Any = None, synchronize: bool = True,
                layer_ahead: int | None = None, release_behind: int | None = None, **options) -> dict[str, Any]:
        """Restore every attended, non-resident entry into its native slot (all layers).

        ``destination(node_id, key, tile)`` returns a ``restore_kernels.NativeView``;
        ``rope(node_id, key, tile)`` returns ``RopePositions`` or None. Decoding of
        later layers overlaps reconstruction of earlier ones (layer order).
        ``layer_ahead=None`` submits every decode job before reconstructing (the
        canvas arena holds the whole history) when the whole history's canvases fit
        the arena budget, else it selects the bounded mode itself (the deepest
        ``d`` whose ``d + 2`` consecutive layers of canvases fit, at least 0;
        reported ``layer_ahead_auto``; before Step 7 Part C such restores raised
        CapacityError); ``layer_ahead=d`` (bounded mode)
        submits only the jobs of the next ``d + 1`` layers, so the arena holds
        about ``d + 1`` layers of canvases (consumed canvases are recycled after
        their reconstruction events): GPU-full histories restore at the cost of
        less decode parallelism (Step 5c: CausVid 105 chunks, d = 2). Results do
        not depend on the mode.

        ``release_behind`` (accel L, 2026-10-30; default None = every
        reconstruction handle, with its inputs and temporaries, is held until
        the restore ends): ``d`` waits for and releases each layer's handle once
        the layer ``d + 1`` positions later was issued, so at most ``d + 2``
        layers of reconstruction temporaries are alive (LingBot's native-RoPE
        restore holds ~3.45 GiB per 480P chunk otherwise). The host waits only
        for work issued ``d + 1`` layers earlier; the restored values are
        unchanged.
        """
        if layer_ahead is not None and (type(layer_ahead) is not int or layer_ahead < 0):
            raise ConfigurationError("layer_ahead must be None or a nonnegative integer")
        if release_behind is not None and (type(release_behind) is not int or release_behind < 0):
            raise ConfigurationError("release_behind must be None or a nonnegative integer")
        if release_behind is not None:
            options["release_behind"] = release_behind
        with self.activity.running():
            return self._restore(node_ref, destination, rope, ranges=ranges, keys=keys, resident=resident,
                                 stream=stream, synchronize=synchronize, layer_ahead=layer_ahead, **options)

    def _restore(self, node_ref, destination, rope, *, ranges, keys, resident, stream, synchronize, layer_ahead,
                 release_behind=None, **options) -> dict[str, Any]:
        started = time.perf_counter()
        history = self.history(node_ref, ranges=ranges)
        resolved = time.perf_counter()
        self._thread_state().verification = None
        if (self.pipelined_staging and layer_ahead is None and self.staging_engine is not None and history
                and not all(self.prepared.contains(item["packet"]) for item in history)):
            return self._restore_pipelined(history, destination, rope, keys=keys, resident=resident, stream=stream,
                                           synchronize=synchronize, started=started, resolved=resolved,
                                           release_behind=release_behind, **options)
        with _normal_tensors():
            refs, pins = self.node_refs(history)
        verification = getattr(self._thread_state(), "verification", None)
        planned = time.perf_counter()
        handles = []
        routes = None
        retired = 0
        try:
            # Canvases and stacked bases are internal (decoder threads write them): normal tensors. Every job of a
            # layer is submitted before its reconstruction, which runs in the caller's mode (destination writes).
            with _normal_tensors():
                batch = self.restorer.plan(refs, resident=resident, keys=keys)
            if self.decoded_tier is not None:  # accel A1: cached nodes are copied, not decoded
                routes = self.decoded_tier.route(history, batch)
            try:
                layers = batch.layers
                automatic = layer_ahead is None and bool(layers)
                if automatic:
                    layer_ahead = self._fitting_layer_ahead(batch, layers)
                    automatic = layer_ahead is not None
                with _normal_tensors():
                    if layer_ahead is None:
                        batch.submit_through(None)
                    elif layers:
                        batch.submit_through(layers[min(layer_ahead, len(layers) - 1)])
                submitted = time.perf_counter()
                for position, layer in enumerate(layers):
                    if layer_ahead is not None:
                        with _normal_tensors():
                            batch.submit_through(layers[min(position + layer_ahead, len(layers) - 1)])
                    handles.append(batch.reconstruct(batch.entries(layer), destination, rope, stream=stream,
                                                     **options))
                    retired += _retire_behind(handles, release_behind)
                issued = time.perf_counter()
                entries = len(batch)
            finally:
                batch.close()
            # Basis pins last until every reconstruction that reads the bases has completed.
            for handle in handles:
                if getattr(handle, "event", None) is not None:
                    if synchronize:
                        handle.event.synchronize()
                    else:
                        torch.cuda.current_stream(self.device).wait_event(handle.event)
        finally:
            if self.device.type == "cuda" and not synchronize:
                torch.cuda.current_stream(self.device).synchronize()
            if routes is not None:
                routes = self.decoded_tier.release(routes)
            self.finish(pins)
        seconds = time.perf_counter() - started
        self.stats_["restores"] += 1
        self.stats_["entries"] += entries
        self.stats_["nodes"] += len(history)
        self.stats_["seconds"] += seconds
        self.stats_["plan_seconds"] += planned - started
        report = {"nodes": len(history), "entries": entries, "seconds": seconds, "plan_seconds": planned - started,
                  "layer_ahead": layer_ahead, "layer_ahead_auto": automatic,
                  "breakdown": {"resolve_seconds": resolved - started, "prepare_bases_seconds": planned - resolved,
                                "decode_submit_seconds": submitted - planned,
                                "reconstruct_issue_seconds": issued - submitted,
                                "complete_seconds": time.perf_counter() - issued},
                  "handles": handles, "history": history, "verification": verification}
        if self.decoded_tier is not None:
            report["decoded_tier"] = routes
        if release_behind is not None:
            report["release_behind"] = {"depth": release_behind, "released_early": retired}
        return report

    def _restore_pipelined(self, history, destination, rope, *, keys, resident, stream, synchronize, started,
                           resolved, release_behind=None, **options) -> dict[str, Any]:
        """A cold (staged) restore whose decoding overlaps its staging (``pipelined_staging``; decomp_opt 2026-10-25).

        Each pack's decode jobs are planned and submitted as soon as staging prepared it (``stage(on_prepared=...)``),
        while the following packs are still read and prepared; reconstruction then runs pack by pack, layer by
        layer. Every tile is decoded and reconstructed exactly as by the batched path (the same KV, bitwise); only
        the order of the work differs. Packs whose canvases would exceed the arena budget are left unsubmitted: their
        reconstruction submits them layer by layer (the bounded mode's lazy submission) as earlier canvases retire.
        """
        batches, handles = [], []
        budget = self.restorer.arena.budget_bytes
        state = {"bytes": 0, "early": 0, "lazy": 0, "first_submit": None}

        def on_prepared(ref):
            with _normal_tensors():
                batch = self.restorer.plan([ref], resident=resident, keys=keys)
            batches.append(batch)
            need = sum(job.pictures * job.packet.planes * job.packet.height * job.packet.width for job in batch.jobs)
            if state["lazy"] == 0 and state["bytes"] + need <= budget:
                state["bytes"] += need
                with _normal_tensors():
                    batch.submit_through(None)
                state["early"] += 1
                if state["first_submit"] is None:
                    state["first_submit"] = time.perf_counter()
            else:
                state["lazy"] += 1

        pins = None
        entries = 0
        retired = 0
        try:
            failed = False
            try:
                with _normal_tensors():
                    refs, pins = self.node_refs(history, on_prepared=on_prepared)
                verification = getattr(self._thread_state(), "verification", None)
                planned = submitted = time.perf_counter()
                for batch in batches:
                    for layer in batch.layers:
                        handles.append(batch.reconstruct(batch.entries(layer), destination, rope, stream=stream,
                                                         **options))
                        retired += _retire_behind(handles, release_behind)
                issued = time.perf_counter()
                entries = sum(len(batch) for batch in batches)
            except BaseException:
                failed = True
                raise
            finally:
                error = None
                for batch in batches:
                    try:
                        batch.close()
                    except BaseException as exc:  # noqa: BLE001 - every batch is closed; the first error surfaces
                        error = error or exc
                if error is not None and not failed:
                    raise error
            # Basis pins last until every reconstruction that reads the bases has completed.
            for handle in handles:
                if getattr(handle, "event", None) is not None:
                    if synchronize:
                        handle.event.synchronize()
                    else:
                        torch.cuda.current_stream(self.device).wait_event(handle.event)
        finally:
            if self.device.type == "cuda" and not synchronize:
                torch.cuda.current_stream(self.device).synchronize()
            if pins is not None:
                self.finish(pins)
        seconds = time.perf_counter() - started
        self.stats_["restores"] += 1
        self.stats_["entries"] += entries
        self.stats_["nodes"] += len(history)
        self.stats_["seconds"] += seconds
        self.stats_["plan_seconds"] += planned - started
        report = {"nodes": len(history), "entries": entries, "seconds": seconds, "plan_seconds": planned - started,
                  "layer_ahead": None, "layer_ahead_auto": False,
                  "pipelined": {"batches": len(batches), "early_batches": state["early"],
                                "lazy_batches": state["lazy"], "early_canvas_bytes": state["bytes"],
                                "first_submit_seconds": (None if state["first_submit"] is None
                                                         else state["first_submit"] - resolved)},
                  "breakdown": {"resolve_seconds": resolved - started, "prepare_bases_seconds": planned - resolved,
                                "decode_submit_seconds": submitted - planned,
                                "reconstruct_issue_seconds": issued - submitted,
                                "complete_seconds": time.perf_counter() - issued},
                  "handles": handles, "history": history, "verification": verification}
        if release_behind is not None:
            report["release_behind"] = {"depth": release_behind, "released_early": retired}
        return report

    def _fitting_layer_ahead(self, batch, layers) -> int | None:
        """None when every canvas of ``batch`` fits the arena budget at once, else the deepest bounded depth whose
        windows of ``d + 2`` consecutive layers (submitted ahead, being reconstructed, retiring) fit (at least 0)."""
        per_layer: dict = {}
        for job in batch.jobs:
            packet = job.packet
            per_layer[job.first_layer] = (per_layer.get(job.first_layer, 0)
                                          + job.pictures * packet.planes * packet.height * packet.width)
        budget = self.restorer.arena.budget_bytes
        if sum(per_layer.values()) <= budget:
            return None
        sizes = [per_layer.get(layer, 0) for layer in layers]
        depth = 0
        while depth + 1 < len(sizes):
            window = depth + 3
            if max(sum(sizes[start:start + window]) for start in range(max(1, len(sizes) - window + 1))) > budget:
                break
            depth += 1
        return depth

    def stats(self) -> dict[str, Any]:
        record = {**self.stats_, "bases": self.bases.stats(), "prepared": self.prepared.stats(),
                  "restorer": self.restorer.stats()}
        if self.decoded_tier is not None:
            record["decoded_tier"] = self.decoded_tier.stats()
        return record

    def close(self) -> None:
        if self._trim_handle is not None:
            self._trim_handle.unregister()
            self._trim_handle = None
        self.bases.clear()
        if self.decoded_tier is not None:
            self.decoded_tier.close()


class SoftwareLineageRestorer:
    """The :class:`LineageRestorer` interface on the software decoder (CPU, tests, no NVDEC).

    Each attended node pack tile is decoded with ``codecs.decode_packed`` and
    written into its native slot by ``restore_kernels.reconstruct_reference``
    (same destination/RoPE callbacks). Not a performance path.
    """

    def __init__(self, runtime):
        self.runtime = runtime
        self.stats_ = {"restores": 0, "entries": 0, "nodes": 0, "seconds": 0.0}
        self.activity = RestoreActivity()

    def history(self, node_ref, *, ranges=None):
        return LineageRestorer.history(self, node_ref, ranges=ranges)

    def restore(self, node_ref, destination, rope=None, *, ranges=None, keys=None, resident=(), **_options):
        with self.activity.running():
            return self._restore(node_ref, destination, rope, ranges=ranges, keys=keys, resident=resident)

    def _restore(self, node_ref, destination, rope, *, ranges, keys, resident):
        from . import restore_kernels as rk
        from .codecs import decode_packed, packed_keys
        started = time.perf_counter()
        history = self.history(node_ref, ranges=ranges)
        since = _verified_payloads(self.runtime.store)
        resident_nodes = {item for item in resident if isinstance(item, str)}
        resident_entries = {item for item in resident if isinstance(item, tuple)}
        entries = 0
        for item in history:
            if item["node_id"] in resident_nodes:
                continue
            encoded = self.runtime.node_pack(item["packet"])
            selected = [entry for entry in packed_keys(encoded)
                        if (keys is None or entry in keys) and (item["node_id"], entry) not in resident_entries]
            if not selected:
                continue
            decoded = decode_packed(encoded, keys=selected, dtype=torch.float32)
            jobs = []
            for entry in selected:
                key, _ = entry_parts(entry)
                context = None
                for tile in encoded.metadata["tiles"]:
                    if tile["key"] == key.to_dict() and tile["stream"] == entry_parts(entry)[1]:
                        context = tile["context"]
                grid = (context["frames"], context["height"], context["width"])

                class _Tile:
                    token_start = context["token_start"]
                tile = _Tile()
                tile.grid = grid
                jobs.append(rk.ReconstructionJob(
                    rk.DenseSource(decoded[entry]), destination(item["node_id"], entry, tile), grid,
                    rope=None if rope is None else rope(item["node_id"], entry, tile), label=str(entry)))
                entries += 1
            rk.reconstruct_reference(jobs)
        seconds = time.perf_counter() - started
        self.stats_["restores"] += 1
        self.stats_["entries"] += entries
        self.stats_["nodes"] += len(history)
        self.stats_["seconds"] += seconds
        from .restore_staging import store_verification_report

        return {"nodes": len(history), "entries": entries, "seconds": seconds, "plan_seconds": 0.0,
                "handles": [], "history": history,
                "verification": store_verification_report(self.runtime.store, since)}

    def stats(self):
        return dict(self.stats_)


class RawLineageRestorer:
    """Exact restore of RAW lineage history (:class:`reelcache.archive_pipeline.RawArchive`) into native slots.

    Each attended RAW record's exact pre-RoPE K / V is decoded on the host,
    moved to ``device`` and written by ``restore_kernels.reconstruct``
    (reference backend) with the engine's own RoPE callable (``native_rope``),
    so a restored cache equals the native cache bit for bit. Same
    ``restore(node_ref, destination, rope)`` interface as
    :class:`LineageRestorer`; not a performance path.
    """

    exact_rope = True

    def __init__(self, runtime, device: Any = "cpu"):
        self.runtime, self.device = runtime, torch.device(device)
        self.stats_ = {"restores": 0, "entries": 0, "nodes": 0, "seconds": 0.0}
        self.activity = RestoreActivity()

    def history(self, node_ref: str, *, ranges=None) -> list[dict]:
        pieces = self.runtime.resolve_history(node_ref, ranges)
        return [{"node_id": f"{piece.entry.node_ref}:{piece.entry.chunk_id}", "token_start": piece.token_start,
                 "token_end": piece.token_end, "chunk_id": piece.entry.chunk_id, "source": piece.entry.source,
                 "epoch_id": piece.entry.epoch_id} for piece in pieces]

    def restore(self, node_ref: str, destination: Callable, rope: Callable | None = None, *,
                native_rope: Callable | None = None, ranges=None, keys=None, resident=(), **_options):
        with self.activity.running():
            return self._restore(node_ref, destination, rope, native_rope=native_rope, ranges=ranges, keys=keys,
                                 resident=resident)

    def _restore(self, node_ref, destination, rope, *, native_rope, ranges, keys, resident):
        from . import restore_kernels as rk
        from .codecs import decode_tensor
        if rope is not None and native_rope is None:
            raise ConfigurationError("RAW identity restores rotate K with the engine's native RoPE callable")
        started = time.perf_counter()
        runtime = self.runtime
        pieces = runtime.resolve_history(node_ref, ranges)
        jobs, history = [], []
        for piece in pieces:
            entry = piece.entry
            if (piece.token_start, piece.token_end) != (entry.token_start, entry.token_end):
                raise UnsupportedError("RAW restore restores whole committed ranges")
            node_id = f"{entry.node_ref}:{entry.chunk_id}"
            history.append({"node_id": node_id, "token_start": entry.token_start, "token_end": entry.token_end,
                            "chunk_id": entry.chunk_id})
            if node_id in resident:
                continue
            for item, record_ref in entry.records:
                if keys is not None and item not in keys:
                    continue
                with runtime._lock, runtime.store.transaction():
                    record = runtime.load_record(record_ref)
                if record.codec.kind != "raw":
                    raise UnsupportedError("RawLineageRestorer restores RAW records only; use LineageRestorer")
                values = decode_tensor(runtime.matrix_packet(record_ref)).to(self.device)
                context = record.context

                class _Tile:
                    token_start = context.token_start
                    grid = (context.frames, context.height, context.width)

                tile = _Tile()
                jobs.append(rk.ReconstructionJob(
                    rk.DenseSource(values), destination(node_id, item, tile), tile.grid,
                    rope=None if rope is None else rope(node_id, item, tile), label=f"{node_id}:{item}"))
        handle = rk.reconstruct(jobs, backend="reference", native_rope=native_rope) if jobs else None
        if handle is not None and getattr(handle, "event", None) is not None:
            handle.event.synchronize()
        elif self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        seconds = time.perf_counter() - started
        self.stats_["restores"] += 1
        self.stats_["entries"] += len(jobs)
        self.stats_["nodes"] += len(pieces)
        self.stats_["seconds"] += seconds
        return {"nodes": len(pieces), "entries": len(jobs), "seconds": seconds, "plan_seconds": 0.0,
                "handles": [] if handle is None else [handle], "history": history}

    def stats(self) -> dict[str, Any]:
        return dict(self.stats_)

    def close(self) -> None:
        pass
