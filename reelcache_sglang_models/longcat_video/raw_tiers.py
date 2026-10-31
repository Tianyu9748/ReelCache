"""Per-epoch raw tiers of LongCat-Video conditioning caches (freeze D; Step 13's part of Step 7's DESIGN_D_tiers §6).

Unit = one segment epoch: the 96 pre-RoPE K/V matrices ``[T, C]`` of a segment's conditioning cache (10.57 GiB at
720p). RoPE is applied inside attention, so every raw copy is bitwise the native build and a restore needs no re-rope
(``rope=None``). Tiers, best first:

* ``resident``: the cache a stage holds for its own segment. LongCat never resumes a resident cache across
  segments (continuation builds a new cache from new frames); the label exists for Step 7's tier names.
* ``gpu_raw``: the built cache kept on the GPU after its segment (zero-copy restore), within
  ``gpu_raw_budget_bytes``. :func:`gpu_raw_budget` derives the auto value: whole epochs above the measured generation
  need, so A100-80GB gets 0 and RTX PRO 6000 gets 1. Entries beyond the budget are dropped at request admission and at
  every memory-critical section entry.
* ``host_raw``: the pinned host capture slot. It is the planner's source until the epoch's compressed node is
  published; after that it stays as an LRU entry (by access) instead of being freed. A restore from it is one
  direct H2D copy (no store read, no hash verification).
* ``compressed``: the published node. Records say ``compressed_prepared`` (its pack is in the restorer's prepared
  cache) or ``compressed_cold``. When a host entry is evicted, the backend touches or re-prepares its node.

Identity (review amendment 4): an entry is keyed by a unique submission token until its node is published, then also
by its ``node_ref``. A restore looks the entry up by the epoch's content key ``(model fingerprint, conditioning-latents
digest, token grid)``; captures are de-duplicated by that key, so a regenerated or rewound epoch with other content
never shares an entry.

Invariant I1 (Step 7): until an entry is published, at least one complete copy exists. Its GPU copy is dropped only
after the host copy is complete, and an unpublished host copy is never evicted. When every slot holds an unpublished
epoch, :class:`HostFull` asks for the last resort: wait for the oldest one to publish.

Budgets (review amendment 2):

* :func:`cgroup_memory` reads the tightest cgroup v2 ``memory.max`` and its ``memory.current`` on the process's path
  (v1: ``memory.limit_in_bytes`` and ``memory.usage_in_bytes``; no limit: ``MemAvailable``), plus the cgroup's clean
  page cache (Step 7 R5).
* :func:`host_raw_budget` divides that headroom by the workers sharing the cgroup (``raw_tier_workers``, default
  ``LOCAL_WORLD_SIZE``, else 1: :func:`local_workers`), subtracts this worker's fixed host consumers (restore staging
  ring, archive canvas pool, prepared cache, NVENC probe) and takes ``HOST_RAW_FRACTION`` of the rest, capped
  (Step 7 R5; per worker: ``fraction * (headroom / workers - fixed)``).
* No minimum above the safe headroom: zero slots is a result. The provider then disables the tier (Step 7 R5) and
  captures with C3 back-pressure in at most the slots the physical share holds, or not at all (recorded).
"""

from __future__ import annotations

import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import torch

from .conditioning_cache import ConditioningCacheError

#: Step 7's tier names (``reelcache.raw_tier.TIER_NAMES``), best first.
TIER_NAMES = ("resident", "gpu_raw", "host_raw", "compressed")
#: Per-epoch record labels (review amendment 3): the compressed tier split by prepared state.
RECORD_TIERS = ("resident", "gpu_raw", "host_raw", "compressed_prepared", "compressed_cold")
#: Measured device memory above the resident model while LongCat generates. A100-80GB 4414175: 74.25 GiB device use
#: outside sections and 72.41 inside, against 37.29 GiB resident. RTX PRO 6000 4412739: 74.05 GiB in sections (eager
#: planner overlap). Both include about 9 GiB outside PyTorch.
SECTION_NEED_BYTES = 37 << 30
#: Safety margin kept free on the device beyond the measured need.
GPU_RAW_MARGIN_BYTES = 4 << 30
#: d5 (freeze F5): device memory an overlapped svd_hevc restore adds inside a restored segment's first denoising step
#: (the restore's side stream and the restorer's working buffers next to the forward). A100-80GB 4424386, plain vs
#: overlapped arms: +0.3-0.5 GiB allocated and about +0.6 GiB device use inside step 0. Added to the GPU raw budget's
#: generation need while the session overlap (``overlap_restore``) is effective.
OVERLAP_RESTORE_NEED_BYTES = 640 << 20
#: d5: host bytes of the archive pipeline's tile workspaces for fingerprints and finalization (Step 7 dev22i, freeze
#: E onward): about one range's finalize workspace. A100-80GB 4421998 ``host_workspace`` peaks: 708,575,232 and
#: 741,826,560 B. Reserved among the worker's fixed host consumers, so the raw tier's auto budget leaves it free.
FINALIZE_WORKSPACE_BYTES = 768 << 20
#: Host raw default: half of this worker's share of the cgroup headroom, capped (Step 7 §3).
HOST_RAW_FRACTION = 0.5
HOST_RAW_CAP_BYTES = 64 << 30
#: Host bytes of one archive canvas staging buffer of a LongCat 720p node, measured (``pipeline.stats()["staging"]
#: ["peak_bytes"]``: 1,847,328,768 on RTX 4403958 and A100-80GB 4403960); the pipeline holds up to two.
CANVAS_STAGING_BYTES = 1_847_328_768


class HostFull(ConditioningCacheError):
    """Every host slot holds an unpublished epoch: the last resort applies (wait for the oldest to publish)."""


# ------------------------------------------------------------------------------------------------------- budgets
def _read_int(path: Path) -> int | None:
    try:
        text = path.read_text().strip()
    except OSError:
        return None
    if not text or text == "max":
        return None
    try:
        return int(text)
    except ValueError:
        return None


def _reclaimable_cache(path: Path, version: str) -> int:
    """Clean, unshared page cache of a cgroup (``memory.stat``; Step 7's ``raw_tier._reclaimable_cache``): v2
    ``file - shmem - file_dirty - file_writeback``, v1 ``total_cache - total_shmem - total_dirty - total_writeback``;
    0 when unreadable."""
    names = (("file", "shmem", "file_dirty", "file_writeback") if version == "v2"
             else ("total_cache", "total_shmem", "total_dirty", "total_writeback"))
    values: dict[str, int] = {}
    try:
        for line in path.read_text().splitlines():
            parts = line.split()
            if len(parts) == 2 and parts[0] in names:
                values[parts[0]] = int(parts[1])
    except (OSError, ValueError):
        return 0
    cache, shared, dirty, writeback = (values.get(name, 0) for name in names)
    return max(0, cache - shared - dirty - writeback)


def cgroup_memory(*, proc_root: str | os.PathLike = "/proc", sys_root: str | os.PathLike = "/sys/fs/cgroup") -> dict:
    """This process's memory headroom: the tightest cgroup limit on its path minus that cgroup's usage, plus its clean
    page cache (reclaimed before the cgroup's OOM killer runs; Step 7 R5: this cluster's job cgroups hold ~100 GB of
    page cache, e.g. the model weights, so ``memory.current`` sits near the limit).

    cgroup v2 (``0::<path>``) walks ``<path>`` up to the root and takes the ancestor with the least
    ``memory.max - memory.current + clean cache``. Slurm sets ``--mem`` on the job's cgroup, and steps inherit it.
    cgroup v1 walks the ``memory`` controller the same way (``memory.limit_in_bytes`` / ``memory.usage_in_bytes``; a
    limit above 2^60 means none). Without any limit the record falls back to ``MemAvailable`` of ``/proc/meminfo``."""
    proc_root, sys_root = Path(proc_root), Path(sys_root)
    try:
        lines = (proc_root / "self" / "cgroup").read_text().splitlines()
    except OSError:
        lines = []
    candidates = []
    for line in lines:
        parts = line.split(":", 2)
        if len(parts) != 3:
            continue
        hierarchy, controllers, path = parts
        if hierarchy == "0" and controllers == "":
            candidates.append(("v2", sys_root, path, "memory.max", "memory.current"))
        elif "memory" in controllers.split(","):
            candidates.append(("v1", sys_root / "memory", path, "memory.limit_in_bytes", "memory.usage_in_bytes"))
    best = None
    for version, root, path, limit_name, usage_name in candidates:
        relative = Path(path.lstrip("/"))
        levels = [relative] + list(relative.parents)
        for level in levels:
            directory = root / level
            limit = _read_int(directory / limit_name)
            if limit is None or (version == "v1" and limit >= 1 << 60):
                continue
            usage = _read_int(directory / usage_name)
            if usage is None:
                continue
            reclaimable = _reclaimable_cache(directory / "memory.stat", version)
            headroom = max(0, limit - usage + reclaimable)
            if best is None or headroom < best["headroom_bytes"]:
                best = {"version": version, "path": "/" + str(level) if str(level) != "." else "/",
                        "limit_bytes": limit, "current_bytes": usage, "reclaimable_cache_bytes": reclaimable,
                        "headroom_bytes": headroom, "source": f"cgroup {version} {limit_name} - current + clean cache"}
    if best is not None:
        return best
    available = None
    try:
        for line in (proc_root / "meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                available = int(line.split()[1]) * 1024
                break
    except (OSError, ValueError, IndexError):
        available = None
    return {"version": None, "path": None, "limit_bytes": None, "current_bytes": None,
            "headroom_bytes": available or 0, "source": "MemAvailable (no cgroup memory limit)"}


def local_workers(explicit: int | None = None) -> int:
    """Workers sharing this job's memory cgroup (Step 7 ``raw_tier_workers``): explicit, else ``LOCAL_WORLD_SIZE``,
    else 1."""
    if explicit is not None:
        return int(explicit)
    try:
        return max(1, int(os.environ.get("LOCAL_WORLD_SIZE", "1")))
    except ValueError:
        return 1


def host_raw_budget(*, epoch_bytes: int, workers: int = 1, fixed_bytes: int = 0, memory: Mapping | None = None,
                    explicit_bytes: int | None = None, explicit_slots: int | None = None,
                    fraction: float = HOST_RAW_FRACTION, cap_bytes: int = HOST_RAW_CAP_BYTES) -> dict:
    """Host raw budget and slot count of one worker (review amendment 2; Step 7 R5).

    ``physical_bytes = headroom / workers - fixed_bytes``: this worker's share of the cgroup headroom after its fixed
    host consumers (restore staging ring, canvas pool, prepared cache, NVENC probe). The auto budget is ``fraction``
    of it, capped at ``cap_bytes``; no minimum is imposed, so the slot count may be 0. Explicit budgets
    (``explicit_bytes`` or ``explicit_slots``) are taken as given; ``record["fits"]`` says whether they stay within
    ``physical_bytes``."""
    if type(epoch_bytes) is not int or epoch_bytes < 1:
        raise ValueError("epoch_bytes must be a positive integer")
    if type(workers) is not int or workers < 1:
        raise ValueError("workers must be a positive integer")
    memory = dict(cgroup_memory() if memory is None else memory)
    per_worker = int(memory.get("headroom_bytes") or 0) // workers
    physical = max(0, per_worker - int(fixed_bytes))
    if explicit_slots is not None:
        budget, source = int(explicit_slots) * epoch_bytes, "capture_slots"
    elif explicit_bytes is not None:
        budget, source = int(explicit_bytes), "host_raw_budget_bytes"
    else:
        budget, source = int(min(cap_bytes, fraction * physical)), "auto"
    slots = budget // epoch_bytes
    return {"source": source, "budget_bytes": budget, "slots": int(slots), "epoch_bytes": epoch_bytes,
            "workers": workers, "fixed_bytes": int(fixed_bytes), "worker_headroom_bytes": per_worker,
            "physical_bytes": physical, "physical_slots": physical // epoch_bytes,
            "fits": slots * epoch_bytes <= physical, "fraction": fraction, "cap_bytes": cap_bytes, "memory": memory}


def gpu_raw_budget(*, total_bytes: int, resident_bytes: int, epoch_bytes: int, reserved_bytes: int = 0,
                   need_bytes: int = SECTION_NEED_BYTES, margin_bytes: int = GPU_RAW_MARGIN_BYTES,
                   explicit_bytes: int | None = None) -> dict:
    """GPU raw budget of one worker: whole epochs that fit above the resident model, the measured generation need,
    the margin and ``reserved_bytes`` (e.g. the NVENC probe's device memory).

    A100-80GB: 79.25 − 37.3 − 37 − 4 < 1 epoch, so 0. RTX PRO 6000: 94.97 − 37.3 − 37 − 4 ≈ 16.7 GiB, so 1 epoch. An
    explicit budget is taken as given; ``fits`` records whether it stays within the free memory."""
    free = int(total_bytes) - int(resident_bytes) - int(need_bytes) - int(margin_bytes) - int(reserved_bytes)
    if explicit_bytes is not None:
        budget, source = int(explicit_bytes), "gpu_raw_budget_bytes"
    else:
        budget, source = max(0, free) // epoch_bytes * epoch_bytes, "auto"
    return {"source": source, "budget_bytes": budget, "epochs": budget // epoch_bytes, "epoch_bytes": epoch_bytes,
            "total_bytes": int(total_bytes), "resident_bytes": int(resident_bytes), "need_bytes": int(need_bytes),
            "margin_bytes": int(margin_bytes), "reserved_bytes": int(reserved_bytes), "free_bytes": max(0, free),
            "fits": budget <= max(0, free)}


# --------------------------------------------------------------------------------------------------------- entries
@dataclass(eq=False)
class RawEntry:
    """One captured epoch's raw copies (see the module docstring)."""

    token: str
    key: tuple
    epoch: Any
    nbytes: int
    sessions: set = field(default_factory=set)
    state: str = "unpublished"  # unpublished | published | failed
    node_ref: str | None = None
    slot: int | None = None
    host: dict | None = None  # MatrixKey -> pinned / CPU [T, C] copy
    host_event: Any = None  # D2H completion event of the host copy
    gpu: tuple | None = None  # (keys_pre_rope, values): the built cache's per-layer tensors
    resident: bool = False  # in use by the segment that built or restored it
    pins: int = 0
    used_host: int = 0
    used_gpu: int = 0
    captured_at: float = field(default_factory=time.perf_counter)
    published_at: float | None = None

    def host_ready(self) -> bool:
        """The host copy is complete (its D2H event, if any, has finished)."""
        if self.host is None:
            return False
        event = self.host_event
        return event is None or bool(event.query())

    def describe(self) -> dict:
        return {"token": self.token, "segment_index": getattr(self.epoch, "segment_index", None),
                "session_id": getattr(self.epoch, "session_id", None), "state": self.state,
                "node_ref": self.node_ref, "gpu": self.gpu is not None, "host": self.host is not None,
                "resident": self.resident, "pins": self.pins, "bytes": self.nbytes}


def new_token() -> str:
    return f"capture-{uuid.uuid4().hex}"


class EpochTierIndex:
    """The host slots and per-epoch raw entries of one backend (see the module docstring).

    ``slots``: host slots (pinned when ``pin``), allocated lazily and reused. ``retain``: published host copies stay as
    ``host_raw`` LRU entries (raw tier on); otherwise a slot is freed at publication (C3). ``gpu_budget_bytes``:
    the retained GPU copies allowed beyond the resident ones. ``on_evict(entry)`` is called, outside the lock, when a
    published entry loses its last raw copy (the backend touches or re-prepares its compressed node).
    """

    def __init__(self, slots: int, *, pin: bool, retain: bool = False, gpu_budget_bytes: int = 0,
                 on_evict: Callable[[RawEntry], None] | None = None):
        if type(slots) is not int or slots < 1:
            raise ValueError("a capture needs at least one host slot")
        if type(gpu_budget_bytes) is not int or gpu_budget_bytes < 0:
            raise ValueError("gpu_budget_bytes must be a nonnegative integer")
        self.count, self.pin, self.retain = slots, bool(pin), bool(retain)
        self.gpu_budget_bytes = gpu_budget_bytes
        self.on_evict = on_evict
        self._buffers: list[dict | None] = [None] * slots
        self._owner: list[str | None] = [None] * slots
        self._entries: dict[str, RawEntry] = {}
        self._by_key: dict[tuple, str] = {}
        self._by_ref: dict[str, str] = {}
        self._clock = 0
        self._lock = threading.Lock()
        self.allocate_seconds = 0.0
        self._stats = {"gpu_peak_bytes": 0, "host_peak_bytes": 0, "host_allocated_peak_bytes": 0,
                       "evictions": {"gpu": 0, "host": 0}, "evicted_bytes": {"gpu": 0, "host": 0},
                       "last_resort_backpressure_seconds": 0.0, "last_resort_backpressure_waits": 0,
                       "last_resort_backpressure_causes": {}, "retained_after_publication": 0}

    # -- slots ---------------------------------------------------------------------------------------------------
    def _tick(self) -> int:
        self._clock += 1
        return self._clock

    def acquire(self, token: str, shapes: Mapping, dtype: torch.dtype) -> tuple[int, dict]:
        """A host slot for capture ``token``: a free slot, else the least recently used published entry's slot
        (evicted), else :class:`HostFull`."""
        evicted = None
        with self._lock:
            index = next((i for i, owner in enumerate(self._owner) if owner is None), None)
            if index is None and self.retain:
                victims = [e for e in self._entries.values() if e.slot is not None and e.state == "published"
                           and e.pins == 0]
                if victims:
                    evicted = min(victims, key=lambda e: e.used_host)
                    index = evicted.slot
                    self._drop_host(evicted)
            if index is None:
                raise HostFull(f"all {self.count} capture slots hold unpublished epochs (flush or drain first; the "
                               "session provider waits for the oldest to publish)")
            self._owner[index] = token
        if evicted is not None:
            self._after_evict(evicted)
        slot = self._buffers[index]
        if slot is None or set(slot) != set(shapes) or any(
                tuple(slot[k].shape) != tuple(shapes[k]) or slot[k].dtype != dtype for k in shapes):
            started = time.perf_counter()
            slot = {k: torch.empty(tuple(s), dtype=dtype, pin_memory=self.pin) for k, s in shapes.items()}
            self.allocate_seconds += time.perf_counter() - started
            self._buffers[index] = slot
        self._account()
        return index, slot

    def release_slot(self, index: int) -> None:
        """Free a slot acquired for a capture that did not happen (e.g. a failed copy)."""
        with self._lock:
            self._owner[index] = None
        self._account()

    def preallocate(self, shapes: Mapping, dtype: torch.dtype, slots: int | None = None) -> float:
        """Allocate free slots' buffers ahead of time (session open). Returns seconds."""
        started = time.perf_counter()
        for index in range(self.count if slots is None else min(slots, self.count)):
            with self._lock:
                if self._owner[index] is not None:
                    continue
            slot = self._buffers[index]
            if slot is None or set(slot) != set(shapes):
                self._buffers[index] = {k: torch.empty(tuple(s), dtype=dtype, pin_memory=self.pin)
                                        for k, s in shapes.items()}
        self._account()
        return time.perf_counter() - started

    def free_slots(self) -> int:
        """Slots a capture can take now without waiting: free ones plus evictable published entries."""
        with self._lock:
            free = sum(1 for owner in self._owner if owner is None)
            if self.retain:
                free += sum(1 for e in self._entries.values() if e.slot is not None and e.state == "published"
                            and e.pins == 0)
            return free

    def busy_slots(self) -> int:
        with self._lock:
            return sum(1 for owner in self._owner if owner is not None)

    # -- entries -------------------------------------------------------------------------------------------------
    def add(self, entry: RawEntry) -> RawEntry:
        with self._lock:
            entry.used_host = entry.used_gpu = self._tick()
            if getattr(entry.epoch, "session_id", None) is not None:
                entry.sessions.add(entry.epoch.session_id)
            self._entries[entry.token] = entry
            self._by_key[entry.key] = entry.token
        self._account()
        return entry

    def get(self, token: str) -> RawEntry | None:
        with self._lock:
            return self._entries.get(token)

    def by_ref(self, node_ref: str) -> RawEntry | None:
        with self._lock:
            token = self._by_ref.get(node_ref)
            return None if token is None else self._entries.get(token)

    def by_key(self, key: tuple) -> RawEntry | None:
        with self._lock:
            token = self._by_key.get(key)
            return None if token is None else self._entries.get(token)

    def published(self, token: str, node_ref: str | None) -> None:
        """The entry's compressed node (or sealed RAW) is published: with ``retain`` it stays as an LRU entry, else
        its slot is freed (C3)."""
        dropped = None
        with self._lock:
            entry = self._entries.get(token)
            if entry is None or entry.state == "published":
                return
            entry.state, entry.node_ref, entry.published_at = "published", node_ref, time.perf_counter()
            if node_ref is not None:
                self._by_ref[node_ref] = token
            if self.retain:
                self._stats["retained_after_publication"] += 1
            else:
                self._drop_host(entry, evicted=False)
                entry.gpu = None
                dropped = self._forget(entry)
        self._account()
        return dropped

    def failed(self, token: str) -> None:
        with self._lock:
            entry = self._entries.get(token)
            if entry is not None:
                entry.state = "failed"

    def retire(self, token: str) -> None:
        """Drop an entry's raw copies and forget it (a capture retired before archival, reset, close)."""
        with self._lock:
            entry = self._entries.get(token)
            if entry is None:
                return
            self._drop_host(entry, evicted=False)
            entry.gpu = None
            self._forget(entry)
        self._account()

    def _drop_host(self, entry: RawEntry, *, evicted: bool = True) -> None:  # caller holds the lock
        if entry.slot is not None:
            self._owner[entry.slot] = None
        if evicted and entry.host is not None and entry.state == "published":
            self._stats["evictions"]["host"] += 1
            self._stats["evicted_bytes"]["host"] += entry.nbytes
        entry.slot, entry.host, entry.host_event = None, None, None

    def _forget(self, entry: RawEntry) -> RawEntry:  # caller holds the lock
        self._entries.pop(entry.token, None)
        if self._by_key.get(entry.key) == entry.token:
            self._by_key.pop(entry.key, None)
        if entry.node_ref is not None and self._by_ref.get(entry.node_ref) == entry.token:
            self._by_ref.pop(entry.node_ref, None)
        return entry

    def _after_evict(self, entry: RawEntry) -> None:
        with self._lock:
            gone = entry.host is None and entry.gpu is None
            if gone:
                self._forget(entry)
        if gone and self.on_evict is not None:  # the last raw copy is gone: prepare-on-evict (R6)
            self.on_evict(entry)

    # -- GPU copies ----------------------------------------------------------------------------------------------
    def retain_gpu(self, token: str, keys: tuple, values: tuple) -> bool:
        """Keep a built cache's tensors as the entry's GPU copy (resident while its segment runs). False when the
        budget allows no GPU raw at all."""
        if self.gpu_budget_bytes <= 0:
            return False
        with self._lock:
            entry = self._entries.get(token)
            if entry is None:
                return False
            entry.gpu, entry.resident, entry.used_gpu = (tuple(keys), tuple(values)), True, self._tick()
        self._account()
        return True

    def released(self, key: tuple) -> list:
        """The segment that held ``key``'s cache ended: its GPU copy (if retained) becomes ``gpu_raw``, then the
        budget applies."""
        with self._lock:
            token = self._by_key.get(key)
            entry = None if token is None else self._entries.get(token)
            if entry is not None and entry.resident:
                entry.resident = False
                entry.used_gpu = self._tick()
        return self.enforce_gpu_budget()

    def release_all(self) -> list:
        """No segment is running: every resident copy becomes retained, then the budget applies."""
        with self._lock:
            for entry in self._entries.values():
                if entry.resident and entry.pins == 0:
                    entry.resident = False
                    entry.used_gpu = self._tick()
        return self.enforce_gpu_budget()

    def enforce_gpu_budget(self, budget: int | None = None) -> list:
        """Drop the least recently used retained GPU copies beyond ``budget`` (default the index's). A GPU copy is
        dropped only when the host copy is complete or the entry is published (I1). Returns the dropped entries."""
        budget = self.gpu_budget_bytes if budget is None else budget
        dropped, waits = [], []
        with self._lock:
            retained = [e for e in self._entries.values() if e.gpu is not None and not e.resident]
            total = sum(e.nbytes for e in retained)
            for entry in sorted(retained, key=lambda e: e.used_gpu):
                if total <= budget:
                    break
                if entry.pins:
                    continue
                if entry.state != "published" and entry.host is None:
                    continue  # its only complete copy (never happens: a capture copies to the host first)
                if entry.state != "published" and entry.host_event is not None:
                    waits.append(entry.host_event)
                entry.gpu = None
                total -= entry.nbytes
                self._stats["evictions"]["gpu"] += 1
                self._stats["evicted_bytes"]["gpu"] += entry.nbytes
                dropped.append(entry)
        for event in waits:  # the host copy completes before the GPU tensors are released to the allocator
            event.synchronize()
        for entry in dropped:
            with self._lock:
                gone = entry.host is None and entry.gpu is None
                if gone:
                    self._forget(entry)
            if gone and self.on_evict is not None:
                self.on_evict(entry)
        self._account()
        return dropped

    # -- restore lookups -----------------------------------------------------------------------------------------
    def lookup(self, key: tuple, *, session_id: str | None = None, allow_gpu: bool = True) -> tuple:
        """The best raw tier of the epoch ``key``: ``(entry, "gpu_raw" | "host_raw")`` with the entry pinned, or
        ``(None, None)``. A host copy whose D2H is still running is waited for by the caller (``host_event``)."""
        with self._lock:
            token = self._by_key.get(key)
            entry = None if token is None else self._entries.get(token)
            if entry is None:
                return None, None
            if allow_gpu and entry.gpu is not None and not entry.resident:
                tier = "gpu_raw"
                entry.used_gpu = self._tick()
            elif entry.host is not None:
                tier = "host_raw"
                entry.used_host = self._tick()
            else:
                return None, None
            entry.pins += 1
            if session_id is not None:
                entry.sessions.add(session_id)
        return entry, tier

    def unpin(self, entry: RawEntry, *, resident: bool = False) -> None:
        with self._lock:
            entry.pins = max(0, entry.pins - 1)
            if resident and entry.gpu is not None:
                entry.resident = True
        self._account()

    # -- controls (Step 7 §9 names) ------------------------------------------------------------------------------
    def _session_entries(self, session_id: str | None) -> list:
        return [e for e in self._entries.values() if session_id is None or session_id in e.sessions]

    def demote(self, session_id: str | None = None) -> dict:
        """GPU raw → host raw for the session's entries (all with None): drop their retained GPU copies."""
        started = time.perf_counter()
        with self._lock:
            victims = [e for e in self._session_entries(session_id) if e.gpu is not None and not e.resident
                       and e.pins == 0 and (e.host is not None or e.state == "published")]
            events = [e.host_event for e in victims if e.host_event is not None and e.state != "published"]
        for event in events:
            event.synchronize()
        dropped = 0
        with self._lock:
            for entry in victims:
                if entry.gpu is not None and entry.pins == 0:
                    entry.gpu = None
                    dropped += entry.nbytes
                    if entry.host is None:
                        self._forget(entry)
        self._account()
        return {"entries": len(victims), "bytes": dropped, "seconds": time.perf_counter() - started}

    def drop(self, session_id: str | None = None) -> dict:
        """Drop both raw tiers of the session's published entries (compressed only afterwards). Unpublished entries
        keep their host copy (I1): drain first (the provider's ``drop_raw`` does)."""
        evicted, kept, nbytes = [], 0, 0
        with self._lock:
            for entry in self._session_entries(session_id):
                if entry.pins or entry.resident:
                    kept += 1
                    continue
                if entry.state != "published":
                    kept += 1
                    continue
                nbytes += entry.nbytes * ((entry.host is not None) + (entry.gpu is not None))
                self._drop_host(entry)
                entry.gpu = None
                self._forget(entry)
                evicted.append(entry)
        for entry in evicted:
            if self.on_evict is not None:
                self.on_evict(entry)
        self._account()
        return {"entries": len(evicted), "kept_unpublished_or_in_use": kept, "bytes": nbytes}

    def note_backpressure(self, seconds: float, cause: str) -> None:
        with self._lock:
            self._stats["last_resort_backpressure_seconds"] += float(seconds)
            self._stats["last_resort_backpressure_waits"] += 1
            causes = self._stats["last_resort_backpressure_causes"]
            causes[cause] = causes.get(cause, 0) + 1

    def oldest_unpublished(self) -> RawEntry | None:
        with self._lock:
            pending = [e for e in self._entries.values() if e.state == "unpublished" and e.slot is not None]
            return min(pending, key=lambda e: e.captured_at) if pending else None

    # -- accounting ----------------------------------------------------------------------------------------------
    def _bytes(self) -> dict:  # caller holds the lock
        gpu = sum(e.nbytes for e in self._entries.values() if e.gpu is not None and not e.resident)
        host = sum(e.nbytes for e in self._entries.values() if e.host is not None)
        unpublished = sum(e.nbytes for e in self._entries.values() if e.host is not None and e.state != "published")
        allocated = sum(sum(t.numel() * t.element_size() for t in slot.values())
                        for slot in self._buffers if slot is not None)
        return {"gpu": gpu, "host": host, "unpublished": unpublished, "allocated": allocated}

    def _account(self) -> None:
        with self._lock:
            sizes = self._bytes()
            self._stats["gpu_peak_bytes"] = max(self._stats["gpu_peak_bytes"], sizes["gpu"])
            self._stats["host_peak_bytes"] = max(self._stats["host_peak_bytes"], sizes["host"])
            self._stats["host_allocated_peak_bytes"] = max(self._stats["host_allocated_peak_bytes"],
                                                           sizes["allocated"])

    def reset_peaks(self) -> None:
        with self._lock:
            sizes = self._bytes()
            self._stats["gpu_peak_bytes"] = sizes["gpu"]
            self._stats["host_peak_bytes"] = sizes["host"]
            self._stats["host_allocated_peak_bytes"] = sizes["allocated"]

    def stats(self) -> dict:
        with self._lock:
            sizes = self._bytes()
            entries = list(self._entries.values())
            busy = sum(1 for owner in self._owner if owner is not None)
            retained = sum(1 for e in entries if e.slot is not None and e.state == "published")
            return {"gpu_bytes": sizes["gpu"], "gpu_peak_bytes": self._stats["gpu_peak_bytes"],
                    "host_bytes": sizes["host"], "host_peak_bytes": self._stats["host_peak_bytes"],
                    "host_allocated_bytes": sizes["allocated"],
                    "host_allocated_peak_bytes": self._stats["host_allocated_peak_bytes"],
                    "unpublished_bytes": sizes["unpublished"],
                    "entries": {"total": len(entries),
                                "gpu_raw": sum(1 for e in entries if e.gpu is not None and not e.resident),
                                "resident": sum(1 for e in entries if e.resident),
                                "host_raw": sum(1 for e in entries if e.host is not None),
                                "unpublished": sum(1 for e in entries if e.state == "unpublished"),
                                "published": sum(1 for e in entries if e.state == "published"),
                                "failed": sum(1 for e in entries if e.state == "failed")},
                    "slots": {"count": self.count, "busy": busy, "retained": retained,
                              "free": self.count - busy, "pinned": self.pin},
                    "budgets": {"gpu_raw_budget_bytes": self.gpu_budget_bytes, "host_slots": self.count,
                                "retain": self.retain},
                    "evictions": dict(self._stats["evictions"]), "evicted_bytes": dict(self._stats["evicted_bytes"]),
                    "retained_after_publication": self._stats["retained_after_publication"],
                    "last_resort_backpressure_seconds": self._stats["last_resort_backpressure_seconds"],
                    "last_resort_backpressure_waits": self._stats["last_resort_backpressure_waits"],
                    "last_resort_backpressure_causes": dict(self._stats["last_resort_backpressure_causes"]),
                    "allocate_seconds": self.allocate_seconds}

    def entries(self, session_id: str | None = None) -> list[dict]:
        with self._lock:
            return [e.describe() for e in self._session_entries(session_id)]

    def clear_free_buffers(self) -> None:
        """Release the pinned buffers of free slots (session release / idle: no ratchet)."""
        with self._lock:
            self._buffers = [slot if owner is not None else None for slot, owner in zip(self._buffers, self._owner)]
        self._account()

    def close(self) -> None:
        with self._lock:
            for entry in list(self._entries.values()):
                entry.host, entry.gpu, entry.host_event, entry.slot = None, None, None, None
            self._entries.clear()
            self._by_key.clear()
            self._by_ref.clear()
            self._owner = [None] * self.count
            self._buffers = [None] * self.count
        self._account()


__all__ = ["CANVAS_STAGING_BYTES", "EpochTierIndex", "GPU_RAW_MARGIN_BYTES", "HOST_RAW_CAP_BYTES", "HOST_RAW_FRACTION",
           "HostFull", "RECORD_TIERS", "RawEntry", "SECTION_NEED_BYTES", "TIER_NAMES", "cgroup_memory",
           "gpu_raw_budget", "host_raw_budget", "local_workers", "new_token"]
