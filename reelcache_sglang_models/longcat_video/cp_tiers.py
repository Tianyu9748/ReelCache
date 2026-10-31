"""Per-rank raw tiers of a context-parallel LongCat conditioning cache (freeze D; review amendment 1, Step 7 R4).

Under context parallelism every rank holds only its token rows of each epoch (Step 16:
``reelcache.integrations.longcat_factor_shard.RowShardedCacheArchive``; ~5.3 GiB per rank per 720p epoch at CP=2).
:class:`TieredRowArchive` wraps that archive and keeps each rank's host copy of its rows as a ``host_raw`` entry:

* **Capture** (seam callable, local): the archive copies the rank's rows to host memory for its deferred flush; the
  tier references that copy (no second copy). It serves restores before and after publication.
* **Publication**: :meth:`TieredRowArchive.flush` (collective) runs the archive's two-phase publication. An entry
  becomes evictable only when its epoch left the archive's pending set as archived, i.e. after the commit agreed on
  every owner. Published entries stay within ``host_raw_budget_bytes`` per rank (least recently used first; a
  restore hit touches its entry); unpublished ones are the archive's flush sources and always stay.
* **Install** (seam callable, collective), one epoch-wide tier: every rank all-gathers which tier it holds
  (``coordinator.checkpoint``), and all take the best tier available on *every* rank. ``host_raw``: each rank
  stages its rows into fresh tensors and verifies them (complete copy, its native rows, shapes and dtype)
  (preflight), then all agree (commit). A failure on any rank rolls every rank back (staged rows dropped, nothing
  installed) and all fall back together. Otherwise, or after a rollback, the archive's own collective install
  serves the armed representation (``None`` when unarmed: every rank builds natively). A rank can therefore
  neither block in a collective its peers skip nor install a tier its peers do not.
* **Records**: ``last_restore_tier`` ``{tier, per_rank, rollback, bytes, seconds}`` with Step 7's tier labels; a
  compressed install is ``compressed_cold`` (Step 16 reads every owner pack per restore: no prepared state).

First cut: no GPU raw (``gpu_raw`` budget 0) and no prepare-on-evict for owner packs (Step 7 R6 gap).
"""

from __future__ import annotations

import collections
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Mapping

import torch

from .raw_tiers import RECORD_TIERS, host_raw_budget, new_token


@dataclass(eq=False)
class RowEntry:
    """One epoch's host rows on this rank."""

    token: str
    key: str
    session_id: str | None
    host: dict  # MatrixKey -> [rows, C] BF16 host copy (the archive's capture buffer)
    rows_sha256: str
    nbytes: int
    state: str = "unpublished"  # unpublished | published
    pins: int = 0
    used: int = 0
    captured_at: float = field(default_factory=time.perf_counter)
    published_at: float | None = None

    def describe(self) -> dict:
        return {"token": self.token, "epoch_key": self.key[:16], "session_id": self.session_id, "state": self.state,
                "pins": self.pins, "bytes": self.nbytes}


class TieredRowArchive:
    """Raw tiers of one rank over a :class:`~reelcache.integrations.longcat_factor_shard.RowShardedCacheArchive`.

    Construct identically on every rank and register ``CacheRowSeam(heads=H, capture=tiered.capture,
    install=tiered.install)``. ``host_raw_budget_bytes``: published host rows kept on this rank (None: auto, half of
    this rank's share of the cgroup headroom, ``workers`` ranks sharing it, default the CP world size, after
    ``fixed_bytes``; derived at the first capture from its size). ``raw_restore``: ``prefer`` | ``off``."""

    def __init__(self, archive, *, host_raw_budget_bytes: int | None = None, raw_restore: str = "prefer",
                 workers: int | None = None, fixed_bytes: int = 0, memory: Mapping[str, Any] | None = None):
        if raw_restore not in ("prefer", "off"):
            raise ValueError("raw_restore must be 'prefer' or 'off'")
        if host_raw_budget_bytes is not None and (type(host_raw_budget_bytes) is not int or host_raw_budget_bytes < 0):
            raise ValueError("host_raw_budget_bytes must be None or a nonnegative integer")
        self.archive = archive
        self.raw_restore = raw_restore
        self.workers = archive.world_size if workers is None else int(workers)
        self.fixed_bytes = int(fixed_bytes)
        self._memory = memory
        self.budget_bytes = host_raw_budget_bytes
        self.budget: dict | None = None if host_raw_budget_bytes is None else {
            "source": "host_raw_budget_bytes", "budget_bytes": host_raw_budget_bytes}
        self._entries: collections.OrderedDict[str, RowEntry] = collections.OrderedDict()
        self._lock = threading.Lock()
        self._clock = 0
        self.last_restore_tier: dict | None = None
        self.events: list[dict] = []
        self._stats = {"host_peak_bytes": 0, "evictions": 0, "evicted_bytes": 0, "rollbacks": 0,
                       "restores": {}, "installs": 0}

    # -- identities ------------------------------------------------------------------------------------------
    @property
    def rank(self) -> int:
        return self.archive.rank

    @property
    def world_size(self) -> int:
        return self.archive.world_size

    def arm(self, representation: str | None) -> None:
        self.archive.arm(representation)

    @staticmethod
    def epoch_key(epoch) -> str:
        from reelcache.integrations.longcat_factor_shard import epoch_key

        return epoch_key(epoch)

    def _tick(self) -> int:
        self._clock += 1
        return self._clock

    # -- capture (seam callable; local) --------------------------------------------------------------------
    def capture(self, epoch, keys, values) -> None:
        """The archive's capture, then this rank's ``host_raw`` entry referencing the same host rows."""
        self.archive.capture(epoch, keys, values)
        key = self.epoch_key(epoch)
        item = self.archive.pending[key]
        nbytes = sum(t.numel() * t.element_size() for t in item["host"].values())
        if self.budget is None:
            self.budget = host_raw_budget(epoch_bytes=max(1, nbytes), workers=max(1, self.workers),
                                          fixed_bytes=self.fixed_bytes, memory=self._memory)
            self.budget_bytes = int(self.budget["budget_bytes"])
        with self._lock:
            self._entries[key] = RowEntry(new_token(), key, getattr(epoch, "session_id", None), item["host"],
                                          item["rows_sha256"], nbytes, used=self._tick())
        self._account()
        self.events.append({"event": "captured", "epoch_key": key[:16], "rank": self.rank, "bytes": nbytes})

    # -- publication (collective) ----------------------------------------------------------------------------
    def flush(self) -> list:
        """The archive's collective flush; epochs it published (on every owner) become evictable."""
        try:
            return self.archive.flush()
        finally:
            self._mark_published()

    def _mark_published(self) -> None:
        pending, archived = set(self.archive.pending), set(self.archive.archived)
        with self._lock:
            for key, entry in self._entries.items():
                if entry.state == "unpublished" and key not in pending and key in archived:
                    entry.state, entry.published_at = "published", time.perf_counter()
        self.enforce_budget()

    def enforce_budget(self) -> list:
        """Drop the least recently used published entries beyond the budget (pinned ones stay)."""
        budget = self.budget_bytes
        if budget is None:
            return []
        dropped = []
        with self._lock:
            published = [e for e in self._entries.values() if e.state == "published"]
            total = sum(e.nbytes for e in published)
            for entry in sorted(published, key=lambda e: e.used):
                if total <= budget:
                    break
                if entry.pins:
                    continue
                self._entries.pop(entry.key, None)
                total -= entry.nbytes
                self._stats["evictions"] += 1
                self._stats["evicted_bytes"] += entry.nbytes
                dropped.append(entry)
        for entry in dropped:
            entry.host = {}
            self.events.append({"event": "evicted", "epoch_key": entry.key[:16], "rank": self.rank})
        self._account()
        return dropped

    # -- install (seam callable; collective) ---------------------------------------------------------------
    def _lookup(self, key: str) -> RowEntry | None:
        if self.raw_restore != "prefer":
            return None
        with self._lock:
            entry = self._entries.get(key)
            if entry is None or not entry.host:
                return None
            entry.pins += 1
            entry.used = self._tick()
            return entry

    def _stage(self, entry: RowEntry, epoch) -> tuple[list, list]:
        """Preflight: this rank's rows of ``epoch`` from its host copy into fresh tensors, verified."""
        from reelcache.integrations.longcat_factor_shard import rows_digest
        from reelcache.types import MatrixKey

        archive = self.archive
        native = archive.native_owners(epoch)
        rows = native.rows(self.rank)
        if rows_digest(rows) != entry.rows_sha256:
            raise ValueError("the host rows were captured for another row split")
        device = archive.domain.device
        keys, values = [], []
        for layer in range(archive.layers):
            for target, out in (("K", keys), ("V", values)):
                source = entry.host.get(MatrixKey(layer, target))
                if source is None or tuple(source.shape) != (rows.numel(), archive.channels) \
                        or source.dtype != torch.bfloat16:
                    raise ValueError(f"host rows {target}{layer} are incomplete or malformed")
                if device.type == "cuda":
                    tensor = torch.empty(source.shape, dtype=source.dtype, device=device)
                    tensor.copy_(source, non_blocking=source.is_pinned())
                else:
                    tensor = source.clone()
                out.append(tensor)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        return keys, values

    def install(self, epoch):
        """Collective: this rank's ``(keys, values)`` rows of ``epoch`` from the one tier every rank holds, or the
        archive's install (``None`` when unarmed). See the module docstring."""
        from reelcache.distributed import CollectiveCancelled, PeerFailure

        archive = self.archive
        coordinator = archive.coordinator
        started = time.perf_counter()
        key = self.epoch_key(epoch)
        entry = self._lookup(key)
        local = "host_raw" if entry is not None else None
        record = {"rank": self.rank, "epoch_key": key[:16], "rollback": None}
        staged = None
        try:
            rows = coordinator.checkpoint(f"longcat_cc_tier:{key[:32]}:available", details={"tier": local})
            record["per_rank"] = [row["tier"] for row in rows]
            if all(row["tier"] == "host_raw" for row in rows):
                error = None
                try:
                    staged = self._stage(entry, epoch)
                except Exception as exc:  # noqa: BLE001 - agreed below: every rank rolls back
                    error = exc
                cancel = getattr(archive, "cancel", None)
                try:
                    coordinator.checkpoint(f"longcat_cc_tier:{key[:32]}:preflight", error=error,
                                           cancelled=bool(cancel and cancel()))
                except CollectiveCancelled:
                    staged = error = None
                    raise
                except PeerFailure as failure:
                    staged = error = None  # rollback on every rank: nothing installed
                    record["rollback"] = str(failure)[:300]
                    with self._lock:
                        self._stats["rollbacks"] += 1
                if staged is not None:
                    keys, values = staged
                    staged = None
                    self._note(record, "host_raw", entry.nbytes, started)
                    return keys, values
            result = archive.install(epoch)
            tier = None if result is None else "compressed_cold"
            nbytes = 0 if result is None else sum(t.numel() * t.element_size() for t in result[0] + result[1])
            self._note(record, tier, nbytes, started)
            return result
        finally:
            if entry is not None:
                with self._lock:
                    entry.pins = max(0, entry.pins - 1)
                self.enforce_budget()

    def _note(self, record: dict, tier: str | None, nbytes: int, started: float) -> None:
        if tier is not None and tier not in RECORD_TIERS:
            raise ValueError(f"unknown tier {tier!r}")
        record.update(tier=tier or "build", bytes=int(nbytes), seconds=time.perf_counter() - started)
        self.last_restore_tier = record
        with self._lock:
            self._stats["installs"] += 1
            row = self._stats["restores"].setdefault(record["tier"], {"count": 0, "bytes": 0, "seconds": 0.0})
            row["count"] += 1
            row["bytes"] += int(nbytes)
            row["seconds"] += record["seconds"]
        self.events.append({"event": "installed", **record})

    # -- controls and records (Step 7 §9 names) ----------------------------------------------------------------
    def drop_raw(self, session_id: str | None = None) -> dict:
        """Drop this rank's published host rows of the session (all with None); unpublished ones are flush sources
        and stay. Local: the next install agrees on the tier collectively."""
        dropped, kept = [], 0
        with self._lock:
            for key, entry in list(self._entries.items()):
                if session_id is not None and entry.session_id != session_id:
                    continue
                if entry.state != "published" or entry.pins:
                    kept += 1
                    continue
                self._entries.pop(key)
                dropped.append(entry)
        for entry in dropped:
            entry.host = {}
        self._account()
        return {"entries": len(dropped), "bytes": sum(e.nbytes for e in dropped), "kept_unpublished_or_in_use": kept}

    def _bytes(self) -> dict:  # caller holds the lock
        host = sum(e.nbytes for e in self._entries.values() if e.host)
        unpublished = sum(e.nbytes for e in self._entries.values() if e.host and e.state != "published")
        return {"host": host, "unpublished": unpublished}

    def _account(self) -> None:
        with self._lock:
            self._stats["host_peak_bytes"] = max(self._stats["host_peak_bytes"], self._bytes()["host"])

    def reset_raw_peaks(self) -> None:
        with self._lock:
            self._stats["host_peak_bytes"] = self._bytes()["host"]

    def stats(self) -> dict:
        """``pipeline.raw_tier`` of this rank (Step 7 §9 names; no GPU raw in the first cut)."""
        with self._lock:
            sizes = self._bytes()
            entries = list(self._entries.values())
            return {"rank": self.rank, "gpu_bytes": 0, "gpu_peak_bytes": 0, "host_bytes": sizes["host"],
                    "host_peak_bytes": self._stats["host_peak_bytes"], "unpublished_bytes": sizes["unpublished"],
                    "entries": {"total": len(entries),
                                "published": sum(1 for e in entries if e.state == "published"),
                                "unpublished": sum(1 for e in entries if e.state == "unpublished")},
                    "budgets": {"gpu_raw_budget_bytes": 0, "host_raw_budget_bytes": self.budget_bytes,
                                "derivation": self.budget},
                    "evictions": self._stats["evictions"], "evicted_bytes": self._stats["evicted_bytes"],
                    "rollbacks": self._stats["rollbacks"], "installs": self._stats["installs"],
                    "restores": {k: dict(v) for k, v in self._stats["restores"].items()},
                    "raw_restore": self.raw_restore}

    def entries(self) -> list[dict]:
        with self._lock:
            return [entry.describe() for entry in self._entries.values()]

    def close(self) -> None:
        with self._lock:
            for entry in self._entries.values():
                entry.host = {}
            self._entries.clear()
        self.archive.close()


__all__ = ["RowEntry", "TieredRowArchive"]
