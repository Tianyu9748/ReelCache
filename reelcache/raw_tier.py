"""Raw-chunk tier of asynchronous archival (freeze D, Step 7; ``DESIGN_D_tiers.md``).

A submitted commit's clean source -- the pre-RoPE K and V of every layer and
CFG stream, exactly as captured -- serves both its compression (the planner)
and lossless restores until its compressed node is published. Copies live on
the capture device ("GPU raw") or in host memory ("host raw"):

* **I1** a raw entry keeps at least one complete copy until its node is
  published (or its commit failed or was cancelled); a device copy is dropped
  only after a complete host copy exists. Moving device -> host is always
  allowed.
* **I2** ``submit`` never waits for archival while the budgets admit the new
  entry; when device + host budgets are full of unpublished entries (or the
  optional lag bound is exceeded) the producer waits (last-resort
  backpressure, recorded with its cause).
* **I3** a published entry is evictable: device copies are demoted to host,
  host copies dropped, least recently *used* first (restore hits touch).
* The planner always reads its sources on their capture device (the tier
  re-stages spilled copies there), so payload, packet and manifest-content
  bytes, decisions and ranks do not depend on where the raw copy was.

Entries are identified by a unique tier token until publication and by
their immutable node reference afterwards; tensors are stored per committed
range (``chunk_id``) and entry key ``(MatrixKey, stream)``. Holders (sessions)
attach and detach atomically; a published entry without holders is dropped.

wl-w1b7 (symmetric LRU tiers, both opt-in; the defaults are the behaviour above):

* ``retain="lru"``: a published entry that loses its last holder (session close,
  replace, an exclusive hold of another path, a detached branch's close) stays as
  an UNHELD cache entry; budgets evict unheld entries first (LRU among them),
  then held ones by the rules above; a later hold re-attaches it, and the next
  restore of an entry kept unheld since its last restore is a retained hit
  (``retain.retained_hits``, :attr:`RawLease.retained`). Whole-tree delete drops
  (``release_holder(drop=True)``, :meth:`drop_unheld`), as does :meth:`close`.
* ``promote="host"``: :meth:`promote_host` / :meth:`promote_installed` insert a
  restored node into host raw (pooled, pinned blocks), held by the restoring
  session: pre-RoPE rows a store read produced (copied host to host on the
  promotion worker), or the post-RoPE rows a replay installed (``form=
  "installed"``: copied device to host on the tier's copy stream). A promotion
  is an ordinary LRU insert (coordinator 21:25): it evicts unheld entries first,
  then held ones by the rule above, never pinned or in-flight copies; it is
  skipped only when it cannot fit at all or the pinned pool is full.
* ``gpu_only=True`` (W1.4.1 "nossd", the recompute arm with ``host_raw_budget_bytes=0``):
  GPU raw only -- a zero host budget no longer degrades the tier; nothing ever
  enters host raw, and a published entry the budgets or the pressure response
  evict from the device is dropped (deleted; its node is then a replay miss).

All state is guarded by one condition, shared with the owning
:class:`~reelcache.archive_pipeline.ArchivePipeline` (:meth:`RawChunkTier.attach`),
so a producer waiting for room releases the lock publication needs. Copies
(D2H spills, prepare-on-drop callbacks) run on one worker thread outside the
lock; uploads for the planner run on the planner's stream.
"""

from __future__ import annotations

import contextlib
import os
import threading
import time
from collections import deque
from typing import Any, Callable, Iterable, Mapping, Sequence

import torch

from .errors import CapacityError, ConfigurationError, StateError

#: Tier labels of a restore's per-chunk records (``resident`` is decided by the bridge before an archive restore).
TIER_NAMES = ("resident", "gpu_raw", "host_raw", "compressed_prepared", "compressed_cold")
#: Archival states a raw entry observes (the pipeline's stages).
ARCHIVAL_STATES = ("submitted", "planning", "fitted", "published", "failed", "cancelled")
#: wl-w1b7: what a published entry that loses its last holder becomes (``drop``: dropped, the default; ``lru``: kept
#: as an unheld LRU entry) and whether restores insert their miss-path rows into host raw (``host``) or not (``off``).
RETAIN_MODES = ("drop", "lru")
PROMOTE_MODES = ("off", "host")
#: Why a promotion was skipped (``promotion.skips``): ``cannot_fit`` -- the host budget cannot take it even after
#: evicting every evictable entry (pinned, in-flight, unpublished and placed copies stay); ``pinned_pool_full`` -- the
#: pinned block pool cannot take its blocks next to the live ones.
PROMOTION_SKIPS = ("cannot_fit", "pinned_pool_full", "partial_range", "exists", "disabled", "error",
                   "transient_cap", "cancelled", "superseded")
#: W1.4.1 (C2): the floor of the transient promotion cap (``max(2 x the largest promoted node, this)``).
PROMOTION_TRANSIENT_FLOOR = 2 << 30
AUTO_HOST_FRACTION = 0.5
AUTO_HOST_CAP_BYTES = 64 << 30
HOST_ALIGN = 512
HOST_CLASS_MIN = 1 << 20  # pooled host blocks up to 1 MiB keep the 512 B alignment
HOST_CLASS_BITS = 5  # larger blocks round up to 1/32 of their power of two (<= 3.1 %)
RECENT_EVENTS = 64


# -- memory headroom -------------------------------------------------------------------------------------------------
def _read_int(path: str) -> int | None:
    try:
        with open(path) as handle:
            text = handle.read().strip()
    except OSError:
        return None
    if not text or text == "max":
        return None
    try:
        return int(text.split()[0])
    except ValueError:
        return None


def _reclaimable_cache(path: str, version: int) -> int:
    """Clean, unshared page cache of a cgroup (``memory.stat``): v2 ``file - shmem - file_dirty - file_writeback``,
    v1 ``total_cache - total_shmem - total_dirty - total_writeback``; 0 when unreadable."""
    names = (("file", "shmem", "file_dirty", "file_writeback") if version == 2
             else ("total_cache", "total_shmem", "total_dirty", "total_writeback"))
    values: dict[str, int] = {}
    try:
        with open(path) as handle:
            for line in handle:
                parts = line.split()
                if len(parts) == 2 and parts[0] in names:
                    values[parts[0]] = int(parts[1])
    except (OSError, ValueError):
        return 0
    cache, shared, dirty, writeback = (values.get(name, 0) for name in names)
    return max(0, cache - shared - dirty - writeback)


def cgroup_memory(proc_cgroup: str = "/proc/self/cgroup", root: str = "/sys/fs/cgroup") -> dict[str, Any] | None:
    """The tightest memory limit on this process's cgroup path (v2 ``memory.max``/``memory.current``, v1
    ``memory.limit_in_bytes``/``memory.usage_in_bytes``): ``{limit, current, reclaimable_cache, headroom, path,
    version}`` or None. Slurm sets ``--mem`` on the job's cgroup; the leaf often reads ``max``, so every ancestor is
    checked. Headroom = limit - current + clean page cache (reclaimed before an OOM kill)."""
    try:
        with open(proc_cgroup) as handle:
            lines = handle.read().splitlines()
    except OSError:
        return None
    best = None
    for line in lines:
        parts = line.split(":", 2)
        if len(parts) != 3:
            continue
        hierarchy, controllers, path = parts
        if hierarchy == "0" and controllers == "":
            base, files, version = root, ("memory.max", "memory.current"), 2
        elif "memory" in controllers.split(","):
            base, files, version = os.path.join(root, "memory"), ("memory.limit_in_bytes",
                                                                  "memory.usage_in_bytes"), 1
        else:
            continue
        current = path or "/"
        while True:
            directory = os.path.join(base, current.lstrip("/"))
            limit, usage = _read_int(os.path.join(directory, files[0])), _read_int(os.path.join(directory, files[1]))
            if limit is not None and usage is not None and limit < (1 << 60):
                # Clean page cache is reclaimed before the cgroup's OOM killer runs: it is headroom (model weights
                # read through the page cache otherwise fill memory.current to the limit).
                reclaimable = _reclaimable_cache(os.path.join(directory, "memory.stat"), version)
                headroom = limit - usage + reclaimable
                if best is None or headroom < best["headroom"]:
                    best = {"limit": limit, "current": usage, "reclaimable_cache": reclaimable,
                            "headroom": headroom, "path": directory, "version": version}
            if current in ("/", ""):
                break
            current = os.path.dirname(current.rstrip("/")) or "/"
    return best


def mem_available(meminfo: str = "/proc/meminfo") -> int | None:
    try:
        with open(meminfo) as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        return None
    return None


def local_workers(explicit: int | None = None) -> int:
    """Workers sharing this job's memory cgroup: explicit, else ``LOCAL_WORLD_SIZE`` (torchrun), else 1."""
    if explicit is not None:
        return explicit
    try:
        return max(1, int(os.environ.get("LOCAL_WORLD_SIZE", "1")))
    except ValueError:
        return 1


def auto_host_budget(*, reserved_bytes: int = 0, workers: int = 1, fraction: float = AUTO_HOST_FRACTION,
                     cap: int = AUTO_HOST_CAP_BYTES, memory: Mapping[str, Any] | None = ...,
                     available: int | None = ...) -> tuple[int, dict[str, Any]]:
    """``(budget, derivation)``: ``fraction`` of this worker's share of the memory headroom (cgroup ``memory.max``
    minus ``memory.current``, else ``MemAvailable``) after ``reserved_bytes`` of other host pools, capped at
    ``cap``. No minimum is applied: a budget below one chunk degrades the tier (recorded by the tier)."""
    memory = cgroup_memory() if memory is ... else memory
    if memory is not None:
        headroom, source = int(memory["headroom"]), "cgroup"
    else:
        available = mem_available() if available is ... else available
        headroom, source = (int(available), "meminfo") if available is not None else (0, "unknown")
    workers = max(1, int(workers))
    usable = max(0, headroom - int(reserved_bytes))
    budget = int(min(cap, fraction * usable / workers))
    derivation = {"source": source, "headroom_bytes": headroom, "reserved_bytes": int(reserved_bytes),
                  "workers": workers, "fraction": fraction, "cap_bytes": cap, "budget_bytes": budget}
    if memory is not None:
        derivation.update(limit_bytes=int(memory["limit"]), current_bytes=int(memory["current"]),
                          reclaimable_cache_bytes=int(memory.get("reclaimable_cache", 0)), cgroup=memory.get("path"))
    return budget, derivation


# -- device memory pressure (freeze F) --------------------------------------------------------------------------------
#: Pressure response defaults: the reserve is at least this many of the largest entry, this fraction of the device,
#: and the largest memory-critical working set observed (with this margin).
PRESSURE_RESERVE_CHUNKS = 2
PRESSURE_RESERVE_FRACTION = 0.05
PRESSURE_WORKING_SET_MARGIN = 1.1


def cuda_device_memory(device: torch.device) -> dict[str, int]:
    """Device memory of ``device`` for the pressure response: the driver's free/total bytes and the caching
    allocator's reserved, allocated, inactive-split and peak-allocated bytes (the peak is read, never reset)."""
    free, total = torch.cuda.mem_get_info(device)
    stats = torch.cuda.memory_stats(device)
    return {"free": int(free), "total": int(total), "reserved": int(stats.get("reserved_bytes.all.current", 0)),
            "allocated": int(stats.get("allocated_bytes.all.current", 0)),
            "inactive_split": int(stats.get("inactive_split_bytes.all.current", 0)),
            "peak": int(stats.get("allocated_bytes.all.peak", 0))}


#: dev22j8v4 (F5.2's RAVEN decode OOM): the caching allocator returns an expandable segment's memory to the driver page
#: by page -- 2 MiB pages in the small pool, 20 MiB (kLargeBuffer) in the large pool (its mapping granularity: the
#: allocator maps 20 MiB at a time; probes 4445094 / 4445098: a freed 3.3 MiB block between live ones is never given
#: back). A larger page here only undercounts what is releasable (the safe side).
EXPANDABLE_PAGE_BYTES = {"small": 2 << 20, "large": 20 << 20}


def cuda_device_segments(device: torch.device) -> list:
    """The caching allocator's segments on ``device`` (``torch.cuda.memory_snapshot``, without trace entries where the
    running torch can leave them out; the production allocator records no history)."""
    try:
        segments = torch.cuda.memory_snapshot(include_traces=False)
    except TypeError:  # torch < 2.10: traces are only built while history is recorded
        segments = torch.cuda.memory_snapshot()
    index = device.index if device.index is not None else torch.cuda.current_device()
    return [segment for segment in segments if segment.get("device", index) == index]


def releasable_device_bytes(segments) -> dict[str, int]:
    """What the caching allocator can actually give back to the driver from ``segments`` (dev22j8v4): ``releasable``
    = wholly free segments plus, in expandable segments, the whole pages inside free blocks (see
    :data:`EXPANDABLE_PAGE_BYTES`), default pool only. Not releasable: ``fragments`` (free blocks inside used segments
    and sub-page parts), ``private`` (free blocks of private pools: CUDA graphs, ``torch.cuda.MemPool``) and ``pending``
    (freed blocks still awaiting their streams' events). Under ``expandable_segments`` the allocator's
    ``inactive_split_bytes`` counts none of these fragments (probes 4445094 / 4445098), so :func:`device_headroom`
    overstates what an allocation can get."""
    result = {"releasable": 0, "fragments": 0, "private": 0, "pending": 0, "segments": 0}
    for segment in segments:
        result["segments"] += 1
        blocks = segment.get("blocks") or ()
        inactive = [block for block in blocks if block.get("state") == "inactive"]
        free = sum(int(block.get("size", 0)) for block in inactive)
        pending = sum(int(block.get("size", 0)) for block in blocks if block.get("state") == "active_pending_free")
        if tuple(segment.get("segment_pool_id") or (0, 0)) != (0, 0):
            result["private"] += free + pending
            continue
        result["pending"] += pending
        if segment.get("is_expandable"):
            page = EXPANDABLE_PAGE_BYTES["small" if segment.get("segment_type") == "small" else "large"]
            base = int(segment.get("address", 0))
            for block in inactive:
                start = int(block.get("address", base)) - base
                end = start + int(block.get("size", 0))
                whole = max(0, (end // page) * page - (-(-start // page)) * page)
                result["releasable"] += whole
                result["fragments"] += int(block.get("size", 0)) - whole
        elif int(segment.get("allocated_size", 0)) == 0 and pending == 0:
            result["releasable"] += int(segment.get("total_size", 0))
        else:
            result["fragments"] += free
    return result


def strict_device_headroom(memory: Mapping[str, int], segments) -> dict[str, int]:
    """dev22j8v4: the bytes a new allocation can actually get -- the driver's free memory plus what the caching
    allocator can release (:func:`releasable_device_bytes`); nothing fragmented, private or pending counts."""
    released = releasable_device_bytes(segments)
    free = int(memory.get("free", 0))
    return {"headroom": free + released["releasable"], "free": free, "releasable": released["releasable"],
            "unreleasable": released["fragments"] + released["private"] + released["pending"],
            "fragments": released["fragments"], "private": released["private"], "pending": released["pending"],
            "segments": released["segments"]}


def device_headroom(memory: Mapping[str, int], *, conservative: bool = True) -> int:
    """Bytes new allocations can count on: the driver's free memory plus the allocator's cached free memory.
    ``conservative`` (the synchronous checks right before a decode or a request's KV allocation) counts only wholly
    free cached segments: free fragments inside used segments (inactive split) may not fit a large request. The
    background admission check uses the optimistic count, so it never demotes in a process whose cache is merely
    fragmented while it has room (restore tiers stay put in cells that fit)."""
    cached = int(memory.get("reserved", 0)) - int(memory.get("allocated", 0))
    if conservative:
        cached -= int(memory.get("inactive_split", 0))
    return int(memory.get("free", 0)) + max(0, cached)


# -- host buffers -----------------------------------------------------------------------------------------------------
def _size_class(nbytes: int) -> int:
    """Bytes of the pooled host block holding ``nbytes``: up to ``HOST_CLASS_MIN`` the 512 B alignment, above it
    1/2**HOST_CLASS_BITS of the power of two below (at most 3.1 % rounding). Matrices of one model share a class."""
    if nbytes <= HOST_CLASS_MIN:
        return _align(nbytes)
    granule = 1 << (nbytes.bit_length() - 1 - HOST_CLASS_BITS)
    return -(-nbytes // granule) * granule


class _HostBlock:
    """One pooled host block (one raw matrix); pinned by ``cudaHostRegister`` (exact class size, no power-of-two
    rounding). Only :meth:`_HostPool.shrink` unregisters and frees it."""

    __slots__ = ("tensor", "nbytes", "pinned", "_registered")

    def __init__(self, nbytes: int, pin: bool):
        self.tensor = torch.empty(nbytes, dtype=torch.uint8)
        self.nbytes, self.pinned, self._registered = nbytes, False, None
        if pin and nbytes:
            pointer = self.tensor.data_ptr()
            if int(torch.cuda.cudart().cudaHostRegister(pointer, nbytes, 1)) == 0:
                self.pinned, self._registered = True, pointer

    def close(self) -> None:
        if self._registered is not None:
            torch.cuda.cudart().cudaHostUnregister(self._registered)
            self._registered = None
        self.tensor = None



class _HostPool:
    """Size-class pool of host blocks (freeze D2, coordinator 2026-10-17 03:50).

    Unregistering (or freeing) pinned memory holds the CUDA driver lock -- about 1.5 s per 0.86 GB buffer on a
    loaded node -- and stalls every concurrent CUDA call of the process, restores and generation alike, even from a
    background thread. So a dropped raw copy never unpins or frees on the hot path: its blocks return to the pool and
    later copies of the same class reuse them (no registration either). Blocks are freed only by :meth:`shrink`: at
    tier close, at a session release that leaves the tier idle with the pool above its keep bytes, or by an explicit
    idle shrink; every shrink is recorded (reason, blocks, bytes, seconds).

    R5: ``limit_bytes`` (the tier's host budget) caps the pool's bytes, live and cached together. A new block that
    would exceed it first frees cached blocks of other classes (reason ``budget``). Entries of one shape reuse their
    classes, so this happens only when the shapes change (it never does for one model's chunks)."""

    def __init__(self, limit_bytes: int | None = None):
        self._lock = threading.Lock()
        self._free: dict[tuple[int, bool], list[_HostBlock]] = {}
        self.cached_bytes = 0
        self.live_bytes = 0
        # W1.4.1 (I2): bytes being allocated right now and bytes reserved for blocks a promotion will acquire; every
        # capacity check counts both (an allocation used to be invisible between its check and its live increment)
        self.pending_bytes = 0
        self.reserved_bytes = 0
        self.limit_bytes = limit_bytes
        self.stats = {"allocations": 0, "allocated_bytes": 0, "reuses": 0, "returned": 0, "pin_seconds": 0.0,
                      "pinned": 0, "pageable": 0, "pageable_fallbacks": 0, "freed": 0, "freed_bytes": 0,
                      "peak_bytes": 0, "shrinks": 0, "shrink_seconds": 0.0, "over_limit": 0}
        self.shrink_events: deque = deque(maxlen=RECENT_EVENTS)

    def reserve(self, nbytes: int, *, freeing: int = 0) -> bool:
        """W1.4.1 (I2): hold physical capacity for ``nbytes`` of blocks a promotion acquires later
        (``acquire(reserved=)``; or :meth:`unreserve`). False when the live, pending and reserved blocks plus
        ``nbytes`` exceed the limit; ``freeing``: live bytes the caller's evictions return to the pool (cached blocks
        are reusable or freed by the acquisition's own shrink)."""
        with self._lock:
            if (self.limit_bytes is not None and self.live_bytes - int(freeing) + self.pending_bytes
                    + self.reserved_bytes + int(nbytes) > self.limit_bytes):
                return False
            self.reserved_bytes += int(nbytes)
            return True

    def unreserve(self, nbytes: int) -> None:
        with self._lock:
            self.reserved_bytes = max(0, self.reserved_bytes - int(nbytes))

    def acquire(self, sizes: Sequence[int], pin: bool, *, reserved: int = 0) -> list[_HostBlock]:
        """One block per size (its class), reused from the pool when one is cached, else allocated (and pinned).
        ``reserved``: bytes of a :meth:`reserve` these blocks consume."""
        blocks: list[_HostBlock | None] = []
        missing = []
        with self._lock:
            self.reserved_bytes = max(0, self.reserved_bytes - int(reserved))
            for index, nbytes in enumerate(sizes):
                size = _size_class(int(nbytes))
                free = self._free.get((size, pin))
                if free:
                    blocks.append(free.pop())
                    self.cached_bytes -= size
                    self.live_bytes += size
                    self.stats["reuses"] += 1
                else:
                    blocks.append(None)
                    missing.append((index, size))
            self.pending_bytes += sum(size for _, size in missing)  # I2: seen by concurrent checks from now on
        if not missing:
            return blocks
        allocated: set[int] = set()
        try:
            if self.limit_bytes is not None:
                with self._lock:
                    excess = (self.live_bytes + self.cached_bytes + self.pending_bytes + self.reserved_bytes
                              - self.limit_bytes)
                    keep = self.cached_bytes - excess
                if excess > 0:
                    self.shrink(max(0, keep), reason="budget")
                    with self._lock:
                        if (self.live_bytes + self.cached_bytes + self.pending_bytes + self.reserved_bytes
                                > self.limit_bytes):
                            self.stats["over_limit"] += 1  # live blocks alone exceed it (pending block returns)
            for index, size in missing:
                started = time.perf_counter()
                block = _HostBlock(size, pin)
                seconds = time.perf_counter() - started
                blocks[index] = block
                allocated.add(index)
                with self._lock:
                    self.pending_bytes -= size
                    self.live_bytes += size
                    self.stats["allocations"] += 1
                    self.stats["allocated_bytes"] += size
                    self.stats["pin_seconds"] += seconds if pin else 0.0
                    self.stats["pinned" if block.pinned else "pageable"] += 1
                    self.stats["pageable_fallbacks"] += 1 if pin and not block.pinned else 0
                    self.stats["peak_bytes"] = max(self.stats["peak_bytes"], self.live_bytes + self.cached_bytes)
        except BaseException:
            with self._lock:  # what was not allocated is no longer pending
                self.pending_bytes -= sum(size for index, size in missing if index not in allocated)
            self.release([block for block in blocks if block is not None])
            raise
        return blocks

    def release(self, blocks: Iterable[_HostBlock]) -> None:
        """Return blocks to the pool (no driver call)."""
        with self._lock:
            for block in blocks:
                self._free.setdefault((block.nbytes, block.pinned), []).append(block)
                self.live_bytes -= block.nbytes
                self.cached_bytes += block.nbytes
                self.stats["returned"] += 1

    def shrink(self, keep_bytes: int = 0, *, reason: str) -> dict[str, Any]:
        """Free cached blocks beyond ``keep_bytes`` of cached bytes (largest classes first); recorded."""
        freed = []
        with self._lock:
            for key in sorted(self._free, reverse=True):
                items = self._free[key]
                while items and self.cached_bytes > keep_bytes:
                    block = items.pop()
                    self.cached_bytes -= block.nbytes
                    freed.append(block)
                if not items:
                    del self._free[key]
        started = time.perf_counter()
        for block in freed:
            block.close()
        seconds = time.perf_counter() - started
        event = {"reason": reason, "blocks": len(freed), "bytes": sum(block.nbytes for block in freed),
                 "pinned_blocks": sum(1 for block in freed if block.pinned), "seconds": seconds, "time": time.time(),
                 "keep_bytes": int(keep_bytes)}
        with self._lock:
            self.stats["freed"] += len(freed)
            self.stats["freed_bytes"] += event["bytes"]
            if freed:
                self.stats["shrinks"] += 1
                self.stats["shrink_seconds"] += seconds
                self.shrink_events.append(event)
            event["cached_bytes"] = self.cached_bytes
        return event

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            classes: dict[str, int] = {}
            for (size, _), items in self._free.items():
                classes[str(size)] = classes.get(str(size), 0) + len(items)
            return {**self.stats, "live_bytes": self.live_bytes, "cached_bytes": self.cached_bytes,
                    "capacity_bytes": self.live_bytes + self.cached_bytes, "limit_bytes": self.limit_bytes,
                    "cached_blocks_by_class": classes,
                    "shrink_events": [dict(item) for item in self.shrink_events]}



# -- entries ---------------------------------------------------------------------------------------------------------
def _dense(shape: Sequence[int], stride: Sequence[int]) -> bool:
    dims = sorted((step, size) for step, size in zip(stride, shape) if size != 1)
    expected = 1
    for step, size in dims:
        if step != expected:
            return False
        expected *= size
    return True


def _align(value: int) -> int:
    return -(-value // HOST_ALIGN) * HOST_ALIGN


def _numel(shape: Sequence[int]) -> int:
    total = 1
    for size in shape:
        total *= int(size)
    return total


def _meta_rows(matrices: Mapping[Any, Any]) -> dict[Any, torch.Tensor]:
    """Rows as tensors: a ``(dtype, shape)`` description becomes a ``meta`` tensor (no storage)."""
    return {key: value if isinstance(value, torch.Tensor) else torch.empty(tuple(value[1]), dtype=value[0],
                                                                            device="meta")
            for key, value in matrices.items()}


def _rows_bytes(groups: Sequence[tuple[Any, Any, Mapping[Any, Any]]]) -> int:
    """Raw bytes of a promotion's rows (``(chunk_id, token_start, {key: tensor or (dtype, shape)})`` groups)."""
    return sum(int(tensor.numel()) * tensor.element_size() for _, _, matrices in groups
               for tensor in _meta_rows(matrices).values())


class _PromotionJob:
    """W1.4.1: one node's promotion. ``host``: rows a restore read from the store, offered one by one
    (:meth:`RawChunkTier.promotion_offer`) and copied into the entry's pinned blocks by the promotion worker, which
    releases each row at once (``open`` -> ``closed`` by the restore -> ``done``, or ``abandoned``); ``device``:
    device-to-host copies already issued, completed by the worker once their aggregate event completed."""

    __slots__ = ("kind", "entry", "holder", "on_copied", "state", "reason", "pending", "received", "expected",
                 "copied", "max_row", "blocks", "views", "events", "active", "cancelled", "unsafe", "start_ns",
                 "seconds")

    def __init__(self, kind: str, entry: Any, holder: str, on_copied: Any):
        self.kind, self.entry, self.holder, self.on_copied = kind, entry, holder, on_copied
        self.state, self.reason = ("open" if kind == "host" else "issued"), None
        self.pending: deque = deque()
        self.received: set = set()
        self.expected: set = set()
        self.copied: set = set()
        self.max_row = 0
        self.blocks: list | None = None
        self.views: dict = {}
        self.events: tuple = (None, None)
        self.active = self.cancelled = self.unsafe = False
        self.start_ns: int | None = None
        self.seconds = 0.0


def _in_tree(owner: str, name: str) -> bool:
    """Whether holder ``name`` belongs to session ``owner``'s tree: the session itself, its tree holder ``owner/...``
    or a detached branch ``owner~...`` ("/" and "~" never occur in a session id)."""
    return name == owner or name.startswith(owner + "/") or name.startswith(owner + "~")


def _tree_owner(holder: str) -> str:
    """The session whose tree holder ``holder`` belongs to (inverse of :func:`_in_tree`)."""
    return holder.split("/", 1)[0].split("~", 1)[0]


def _layer_of(key: Any) -> int:
    """The layer of an entry key ``(MatrixKey, stream)`` or ``MatrixKey`` (wl-w1b7 promotion order)."""
    matrix = key[0] if isinstance(key, tuple) else key
    return int(getattr(matrix, "layer", 0))


class _RawGroup:
    """One committed range of an entry: tensors by entry key ``(MatrixKey, stream)`` (or ``MatrixKey``)."""

    __slots__ = ("chunk_id", "token_start", "rows", "meta", "device", "host")

    def __init__(self, chunk_id: str, token_start: int | None, matrices: Mapping[Any, torch.Tensor], offset: int):
        self.chunk_id, self.token_start = chunk_id, token_start
        self.meta: dict[Any, tuple] = {}
        self.rows = None
        for key, tensor in matrices.items():
            nbytes = tensor.numel() * tensor.element_size()
            self.meta[key] = (tensor.device, tensor.dtype, tuple(tensor.shape), tuple(tensor.stride()), offset, nbytes)
            offset += _align(nbytes)
            if self.rows is None and tensor.dim() >= 1:
                self.rows = int(tensor.shape[0])
        self.device: dict[Any, torch.Tensor] | None = dict(matrices)
        self.host: dict[Any, torch.Tensor] | None = None

    @property
    def end_offset(self) -> int:
        return max((offset + _align(nbytes) for *_, offset, nbytes in self.meta.values()), default=0)


class RawEntry:
    """The raw copies of one submitted commit (see the module docstring)."""

    def __init__(self, token: int, label: tuple[str, str], groups: list[_RawGroup], nbytes: int, *,
                 holders: Iterable[str], release: Callable[[Any], None] | None, ready_event: Any, cuda: bool):
        self.token, self.label = token, label
        self.node_ref: str | None = None
        self.groups = groups
        self.nbytes = nbytes
        # Real host bytes (R5): one pooled block per matrix, rounded to its size class.
        self.host_nbytes = sum(_size_class(meta[5]) for group in groups for meta in group.meta.values())
        self.state = "submitted"
        self.on_device = True
        self.device_owned = False  # True once the device copies are tier uploads (no producer release pending)
        self.host_state = "none"  # none | queued | copying | complete
        self.host_blocks: list[_HostBlock] | None = None
        self.host_read_event = None
        self.pins: dict[str, int] = {}
        self.last_used = 0
        self.release, self.ready_event, self.read_event = release, ready_event, None
        self.holders = set(holders)
        self.placed: str | None = None
        self.force_host = False
        self.drop_requested = False
        self.dropped = False
        self.spill_failed = False
        self.cuda = cuda
        self.submitted = time.perf_counter()
        # wl-w1b7: where the copy came from (``captured`` at commit; ``store`` / ``replay`` promoted by a restore), its
        # K form (``pre_rope``; ``installed``: post-RoPE rows as a restore installed them) and, once no holder holds it,
        # the holders that held it last (the whole-tree delete of their session drops it)
        self.origin, self.form = "captured", "pre_rope"
        self.released_by: frozenset | None = None
        self.retained = False  # kept unheld (retain="lru") since its last restore: the next one is a retained hit
        self.pool_reserved = 0  # W1.4.1 (I2): pinned-pool bytes reserved for a promotion's blocks, not yet acquired
        self.drop_on_publish = False  # W1.4.1: its session's tree was deleted while it was unpublished
        # W1.4.3 (E1, review C1): False for KV computed on a history installed (in part) from the lossy archive -- a
        # replay promoted from such a history, or a capture made on one (:meth:`RawChunkTier.inexact_capture`)
        self.exact = True

    @property
    def pinned(self) -> bool:
        return any(self.pins.values())

    @property
    def host_pinned(self) -> bool:
        """Whether every host block of the complete host copy is page-locked."""
        return bool(self.host_blocks) and all(block.pinned for block in self.host_blocks)

    @property
    def host_buffer(self) -> _HostBlock | None:
        """dev22a-f compatibility: the first host block (``.pinned``) of the host copy, or None."""
        return self.host_blocks[0] if self.host_blocks else None

    @property
    def unpublished(self) -> bool:
        return self.state in ("submitted", "planning", "fitted")

    def group_for(self, token_start: int) -> _RawGroup | None:
        for group in self.groups:
            if group.token_start == token_start:
                return group
        return None

    def describe(self) -> dict[str, Any]:
        described = {"token": self.token, "branch": self.label[0], "node_id": self.label[1], "node_ref": self.node_ref,
                     "state": self.state, "device": self.on_device, "host": self.host_state, "bytes": self.nbytes,
                     "host_bytes": self.host_nbytes, "holders": sorted(self.holders), "pins": dict(self.pins),
                     "placed": self.placed}
        if self.origin != "captured":  # wl-w1b7 promoted entries only
            described.update(origin=self.origin, form=self.form)
        return described


class _Reservation:
    __slots__ = ("token", "nbytes", "open")

    def __init__(self, token: int, nbytes: int):
        self.token, self.nbytes, self.open = token, nbytes, True


class RawLease:
    """A restore's read access to one entry's raw copy (pinned until :meth:`release`)."""

    def __init__(self, tier: RawChunkTier, entry: RawEntry, tier_name: str):
        self._tier, self.entry, self.tier = tier, entry, tier_name
        self.released = False
        self.retained = False  # wl-w1b7: the entry was kept unheld (retain="lru") since its last restore

    def matrix(self, token_start: int, key: Any) -> torch.Tensor | None:
        group = self.entry.group_for(token_start)
        if group is None:
            return None
        tensors = group.device if self.tier == "gpu_raw" else group.host
        return None if tensors is None else tensors.get(key)

    def release(self) -> None:
        if not self.released:
            self.released = True
            self._tier._unpin(self.entry, "restore")


# -- the tier ---------------------------------------------------------------------------------------------------------
class RawChunkTier:
    """Device and host raw copies of submitted commits under byte budgets (see the module docstring).

    ``gpu_budget_bytes``: device-copy budget; None = ``gpu_budget_chunks`` x the largest entry seen (C3's
    held-source footprint). ``host_budget_bytes``: host-copy budget (buffer bytes); None = :func:`auto_host_budget`
    (``host_reserved_bytes`` of other host pools, ``workers`` sharing the cgroup). A host budget below one entry
    degrades the tier (the pipeline then applies C3 backpressure; recorded). ``max_lag_seconds``: optional bound on
    the estimated drain time of unpublished entries (last-resort cause ``lag_bound``). ``on_host_drop(node_ref)``
    runs on the worker thread when a published entry's last raw copy is dropped (e.g. prepare its compressed node).
    Host copies live in pooled blocks that are never unpinned on the hot path (:class:`_HostPool`);
    ``keep_free_bytes`` is the pool's cached bytes a session release keeps when it leaves the tier idle (None: the
    host budget), :meth:`shrink_buffers` an explicit idle shrink. ``retain`` / ``promote`` (wl-w1b7, default
    ``"drop"`` / ``"off"``): see the module docstring.
    """

    def __init__(self, *, gpu_budget_bytes: int | None = None, gpu_budget_chunks: int = 2,
                 host_budget_bytes: int | None = None, host_reserved_bytes: int = 0, workers: int | None = None,
                 pinned: bool = True, max_lag_seconds: float | None = None,
                 on_host_drop: Callable[[str], Any] | None = None, keep_free_bytes: int | None = None,
                 host_derivation: Mapping[str, Any] | None = None, gpu_reserve_bytes: int | None = None,
                 device_memory: Callable[[torch.device], Mapping[str, int]] | None = None,
                 device_segments: Callable[[torch.device], list] | None = None, retain: str = "drop",
                 promote: str = "off", gpu_only: bool = False):
        if type(gpu_only) is not bool or (gpu_only and host_budget_bytes != 0):
            raise ConfigurationError("gpu_only (GPU raw only) is a boolean and needs host_budget_bytes=0")
        self.gpu_only = gpu_only
        if retain not in RETAIN_MODES:
            raise ConfigurationError(f"retain must be one of {RETAIN_MODES}, got {retain!r}")
        if promote not in PROMOTE_MODES:
            raise ConfigurationError(f"promote must be one of {PROMOTE_MODES}, got {promote!r}")
        self.retain, self.promotion_mode = retain, promote
        for name, value in (("gpu_budget_bytes", gpu_budget_bytes), ("host_budget_bytes", host_budget_bytes),
                            ("gpu_reserve_bytes", gpu_reserve_bytes)):
            if value is not None and (type(value) is not int or value < 0):
                raise ConfigurationError(f"{name} must be None or a nonnegative integer")
        if type(gpu_budget_chunks) is not int or gpu_budget_chunks < 1:
            raise ConfigurationError("gpu_budget_chunks must be a positive integer")
        if type(pinned) is not bool:
            raise ConfigurationError("pinned must be boolean")
        if max_lag_seconds is not None and (isinstance(max_lag_seconds, bool) or not max_lag_seconds > 0):
            raise ConfigurationError("max_lag_seconds must be None or a positive number of seconds")
        if on_host_drop is not None and not callable(on_host_drop):
            raise ConfigurationError("on_host_drop must be callable")
        if device_memory is not None and not callable(device_memory):
            raise ConfigurationError("device_memory must be callable")
        if device_segments is not None and not callable(device_segments):
            raise ConfigurationError("device_segments must be callable")
        # dev22j8v4: the allocator's segments for the decode guard's strict headroom (default: cuda_device_segments
        # where the device memory is CUDA's own; tests inject a provider, a fake device memory alone gives none)
        self._device_segments = device_segments
        self._cond = threading.Condition()
        # Pressure response (freeze F): None = auto reserve, 0 = off; device_memory(device) -> memory dict (default:
        # cuda_device_memory for CUDA copies; tests inject a fake provider).
        self._reserve_config, self._device_memory = gpu_reserve_bytes, device_memory
        self._working_set = 0
        # dev22j8v5 (F5.4 finding 1): the largest working set of a CLEAN section (no node ran unpaused through it:
        # none in flight, paused, or waited for before the mark) -- the decode floor's learned value -- and the GPU
        # raw bytes this tier dropped (monotonic): a section adds back what the tier released while it ran before its
        # peak (``_section_drops``: the cumulative count after each drop inside an open section, its bytes and the
        # allocator's peak then).
        self._clean_working_set = 0
        self._device_released_bytes = 0
        self._open_sections = 0
        self._section_drops: deque = deque(maxlen=256)
        self._device_total: int | None = None  # the device's total bytes, from the last query
        self._device_hint: torch.device | None = None
        self._pressure_queued = False
        self.pressure_ = {"checks": 0, "pressured": 0, "demotions": 0, "demoted_entries": 0, "demoted_bytes": 0,
                          "wait_seconds": 0.0, "max_wait_seconds": 0.0, "timeouts": 0, "skipped_placed_bytes": 0,
                          "oom_reliefs": 0, "errors": 0, "sections": 0, "by_where": {}, "trim_calls": 0,
                          "trimmed_bytes": 0, "trimmed_by_component": {}, "deferred_sections": 0,
                          "archival_deferred_seconds": 0.0, "rechecks": 0, "fitter_recheck_deferrals": 0}
        self._pressure_events: deque = deque(maxlen=RECENT_EVENTS)
        self._gpu_budget, self.gpu_budget_chunks = gpu_budget_bytes, gpu_budget_chunks
        if host_budget_bytes is None:
            budget, derivation = auto_host_budget(reserved_bytes=host_reserved_bytes, workers=local_workers(workers))
            self._host_budget, self.host_derivation = budget, {"mode": "auto", **derivation}
        else:
            self._host_budget = host_budget_bytes
            self.host_derivation = {"mode": "config", "budget_bytes": host_budget_bytes, **dict(host_derivation or {})}
        self.pinned, self.max_lag_seconds, self.on_host_drop = pinned, max_lag_seconds, on_host_drop
        self.keep_free_bytes = keep_free_bytes
        self.disabled_reason: str | None = None
        self._buffers = _HostPool(limit_bytes=self._host_budget)
        self._entries: dict[int, RawEntry] = {}
        self._by_ref: dict[str, RawEntry] = {}
        self._token = 0
        self._clock = 0
        self._max_entry = 0
        self._device_bytes = self._host_bytes = self._unpublished_bytes = self._reserved = 0
        self._host_reserved_spill = 0
        self._unpublished_count = 0
        self._publish_times: deque = deque(maxlen=9)
        self._jobs: deque = deque()
        self._active = 0  # worker jobs popped and still running (flush waits for them)
        self._worker: threading.Thread | None = None
        self._closed = False
        self._streams: dict[str, Any] = {}
        self.stats_ = {"adopted": 0, "adopted_bytes": 0, "published": 0, "spills": 0, "spill_bytes": 0,
                       "spill_seconds": 0.0, "spill_failures": 0, "uploads": 0, "upload_bytes": 0,
                       "upload_seconds": 0.0, "evictions_gpu": 0, "evictions_host": 0, "drops_published": 0,
                       "drops_failed": 0, "drops_cancelled": 0, "drops_released": 0, "drops_duplicate": 0,
                       "demotions": 0, "promotions": 0, "last_resort_backpressure_waits": 0,
                       "last_resort_backpressure_seconds": 0.0, "last_resort_causes": {"host_full": 0, "lag_bound": 0},
                       "spill_waits": 0, "spill_wait_seconds": 0.0, "restores": {"gpu_raw": 0, "host_raw": 0},
                       "restore_misses": 0, "prepare_on_drop": {"queued": 0, "done": 0, "errors": 0},
                       "gpu_peak_bytes": 0, "host_peak_bytes": 0, "unpublished_peak_bytes": 0,
                       "degraded": 0, "bookkeeping_errors": 0, "worker_errors": 0}
        self._events: deque = deque(maxlen=RECENT_EVENTS)
        # wl-w1b7 records (reported only when ``retain`` or ``promote`` is not the default)
        self.retain_ = {"retained": 0, "reattached": 0, "retained_hits": 0, "dropped_on_delete": 0,
                        "evictions": {"gpu_held": 0, "gpu_unheld": 0, "host_held": 0, "host_unheld": 0}}
        self.promotion_ = {"requested": 0, "entries": 0, "bytes": 0, "host_copied_entries": 0,
                           "host_copied_bytes": 0, "host_copy_seconds": 0.0, "d2h_entries": 0, "d2h_bytes": 0,
                           "d2h_seconds": 0.0, "issue_seconds": 0.0, "pinned_entries": 0, "pageable_entries": 0,
                           "hits": 0, "dropped_before_complete": 0, "transferred": 0, "transient_peak_bytes": 0,
                           "quarantined_blocks": 0, "quarantined_bytes": 0,
                           "skips": {reason: 0 for reason in PROMOTION_SKIPS},
                           "skip_bytes": {reason: 0 for reason in PROMOTION_SKIPS}}
        self._promote_jobs: deque = deque()
        self._promote_active = 0
        self._promote_worker: threading.Thread | None = None
        # W1.4.1 (C2): transient promotion bytes (rows offered, not yet copied, plus the row being copied), the
        # copies in flight, the largest row / node promoted so far (the cap), and quarantined blocks (I1)
        self._transient = 0
        self._copying = 0
        self._copy_row = 0
        self._largest_promotion = 0
        self._quarantined: list = []
        self._inexact_labels: set = set()  # W1.4.3 (review C1): captures on a lossy history (inexact_capture)

    # -- wiring ------------------------------------------------------------------------------------------------------
    def attach(self, cond: threading.Condition) -> None:
        """Share the owner's condition (before any use): every wait releases the lock publication needs."""
        if self._entries or self._worker is not None or self._reserved:
            raise StateError("a raw tier is attached before use")
        self._cond = cond

    @property
    def enabled(self) -> bool:
        return self.disabled_reason is None and not self._closed

    def gpu_budget(self) -> int:
        if self._gpu_budget is not None:
            return self._gpu_budget
        return self.gpu_budget_chunks * self._max_entry

    def host_budget(self) -> int:
        return self._host_budget

    def _disable(self, reason: str) -> None:
        """Stop serving restores and admitting entries (caller holds the condition); live entries keep their copies
        until their commits publish or retire, then drop."""
        if self.disabled_reason is None:
            self.disabled_reason = reason
            self._event("disabled", reason=reason)

    # -- admission (caller holds the condition) -------------------------------------------------------------------
    def usable(self, nbytes: int) -> bool:
        """Whether the tier takes a new entry of ``nbytes``: False when disabled, or (degrading the tier, recorded)
        when the host budget cannot hold one entry -- the pipeline then applies C3 backpressure."""
        if not self.enabled:
            return False
        self._max_entry = max(self._max_entry, nbytes)
        if self._host_budget < _align(nbytes) * 1 and not self.gpu_only:
            self.stats_["degraded"] += 1
            self._disable(f"host raw budget {self._host_budget} B is below one chunk ({nbytes} B): C3 backpressure")
            return False
        return True

    def admission(self, nbytes: int) -> str | None:
        """None when a new entry of ``nbytes`` is admitted now, else the cause of the wait."""
        room = self.gpu_budget() + self._host_budget
        if self._unpublished_bytes + self._reserved and self._unpublished_bytes + self._reserved + nbytes > room:
            return "host_full"
        if self.max_lag_seconds is not None and self.lag_estimate() > self.max_lag_seconds:
            return "lag_bound"
        # Device copies beyond the budget are being demoted: wait for those spills (a fraction of a chunk interval) so
        # that, with the new chunk, the device holds at most the budget plus that chunk -- C3's held sources plus the
        # producer's newest capture.
        if self._spilling() and self._device_bytes > self.gpu_budget():
            return "gpu_spill"
        return None

    def lag_estimate(self) -> float:
        times = list(self._publish_times)
        if len(times) < 2 or not self._unpublished_count:
            return 0.0
        intervals = sorted(b - a for a, b in zip(times, times[1:]))
        return self._unpublished_count * intervals[len(intervals) // 2]

    def _spilling(self) -> bool:
        return any(entry.host_state in ("queued", "copying") for entry in self._entries.values())

    def reserve(self, nbytes: int) -> _Reservation:
        self._token += 1
        self._reserved += nbytes
        return _Reservation(self._token, nbytes)

    def unreserve(self, reservation: _Reservation) -> None:
        if reservation.open:
            reservation.open = False
            self._reserved -= reservation.nbytes
            self._cond.notify_all()

    def record_wait(self, cause: str, seconds: float, *, branch: str, node_id: str) -> None:
        if cause == "gpu_spill":
            self.stats_["spill_waits"] += 1
            self.stats_["spill_wait_seconds"] += seconds
        else:
            self.stats_["last_resort_backpressure_waits"] += 1
            self.stats_["last_resort_backpressure_seconds"] += seconds
            self.stats_["last_resort_causes"][cause] = self.stats_["last_resort_causes"].get(cause, 0) + 1
        self._event("backpressure", cause=cause, seconds=seconds, branch=branch, node_id=node_id,
                    unpublished_bytes=self._unpublished_bytes)

    @contextlib.contextmanager
    def inexact_capture(self, branch: str, node_id: str):
        """W1.4.3 (review C1): the entry a commit of ``(branch, node_id)`` submitted inside this block adopts holds KV
        computed on a history installed (in part) from the lossy archive -- no clean replay: ``exact`` False (an anchor
        built on it is reported ``compressed``)."""
        label = (branch, node_id)
        with self._cond:
            self._inexact_labels.add(label)
        try:
            yield
        finally:
            with self._cond:
                self._inexact_labels.discard(label)

    def adopt(self, reservation: _Reservation, *, branch: str, node_id: str,
              groups: Sequence[tuple[str, int | None, Mapping[Any, torch.Tensor]]], ready_event: Any = None,
              release: Callable[[Any], None] | None = None) -> RawEntry:
        """Take the producer's tensors (no copy) as the entry's device copy; ``release(event)`` is called when the
        tier gives them up. Caller holds the condition and passes the open reservation of the entry's bytes."""
        if not reservation.open:
            raise StateError("raw tier reservation is closed")
        built, offset, nbytes = [], 0, 0
        cuda = False
        for chunk_id, token_start, matrices in groups:
            group = _RawGroup(chunk_id, token_start, matrices, offset)
            offset = group.end_offset
            nbytes += sum(meta[5] for meta in group.meta.values())
            cuda = cuda or any(meta[0].type == "cuda" for meta in group.meta.values())
            built.append(group)
        reservation.open = False
        self._reserved -= reservation.nbytes
        entry = RawEntry(reservation.token, (branch, node_id), built, nbytes, holders=(branch,), release=release,
                         ready_event=ready_event, cuda=cuda)
        if (branch, node_id) in self._inexact_labels:  # W1.4.3 (review C1): captured on a lossy history
            entry.exact = False
        self._clock += 1
        entry.last_used = self._clock
        self._entries[entry.token] = entry
        self._device_bytes += nbytes
        self._unpublished_bytes += nbytes
        self._unpublished_count += 1
        self._max_entry = max(self._max_entry, nbytes)
        self.stats_["adopted"] += 1
        self.stats_["adopted_bytes"] += nbytes
        if self._device_hint is None and (cuda or self._device_memory is not None):
            devices = [meta[0] for group in built for meta in group.meta.values()]
            self._device_hint = next((device for device in devices if device.type == "cuda"),
                                     devices[0] if devices else None)
        self._peaks()
        self._balance()
        if cuda or self._device_memory is not None:
            self._request_pressure_check()
        return entry

    # -- planner -------------------------------------------------------------------------------------------------------
    def stage_for_plan(self, entry: RawEntry, stream: Any = None) -> list[dict[Any, torch.Tensor]]:
        """The entry's matrices per range on their capture devices (uploaded from the host copy when the device copy
        was demoted), pinned for the planner until :meth:`fitted`. Call on the planner's stream."""
        with self._cond:
            self._pin(entry, "plan")
            entry.state = "planning" if entry.state == "submitted" else entry.state
            if entry.on_device:
                return [dict(group.device) for group in entry.groups]
            if entry.host_state != "complete":
                self._unpin(entry, "plan")
                raise StateError(f"raw entry {entry.label} has no complete copy (invariant I1)")
        started = time.perf_counter()
        staged = []
        try:
            for group in entry.groups:
                tensors = {}
                for key, (device, dtype, shape, stride, _, _) in group.meta.items():
                    target = (torch.empty_strided(shape, stride, dtype=dtype, device=device) if _dense(shape, stride)
                              else torch.empty(shape, dtype=dtype, device=device))
                    target.copy_(group.host[key], non_blocking=entry.host_pinned)
                    tensors[key] = target
                staged.append(tensors)
            event = None
            if entry.cuda:
                # After the uploads' reads of the host blocks: a dropped copy's blocks return to the pool only then.
                device = next(meta[0] for group in entry.groups for meta in group.meta.values()
                              if meta[0].type == "cuda")
                event = torch.cuda.Event()
                event.record(stream if stream is not None else torch.cuda.current_stream(device))
        except BaseException:
            with self._cond:
                self._unpin(entry, "plan")
            raise
        with self._cond:
            if entry.host_blocks is not None:
                entry.host_read_event = event
            if entry.dropped or entry.drop_requested:
                return staged  # retired while uploading: the planner's own tensors, not a tier copy
            for group, tensors in zip(entry.groups, staged):
                group.device = tensors
            entry.on_device, entry.device_owned = True, True
            self._device_bytes += entry.nbytes
            self.stats_["uploads"] += 1
            self.stats_["upload_bytes"] += entry.nbytes
            self.stats_["upload_seconds"] += time.perf_counter() - started
            self._peaks()
        return staged

    def fitted(self, entry: RawEntry, event: Any = None) -> None:
        """The planner finished reading the sources (``event``: after its last device read). Caller holds the
        condition."""
        if entry.state in ("submitted", "planning"):
            entry.state = "fitted"
        entry.read_event = event
        self._unpin(entry, "plan")

    # -- publication and retirement (caller holds the condition) ---------------------------------------------------
    def published(self, entry: RawEntry, node_ref: str) -> None:
        """Idempotent: the entry's node is durable; its copies become evictable."""
        if entry.state == "published":
            return
        was_unpublished = entry.unpublished
        entry.state, entry.node_ref = "published", node_ref
        if was_unpublished:
            self._unpublished_bytes -= entry.nbytes
            self._unpublished_count -= 1
        self._publish_times.append(time.perf_counter())
        self.stats_["published"] += 1
        if entry.dropped:
            return
        if not self.enabled:
            self._request_drop(entry, "drops_published")
            return
        existing = self._by_ref.get(node_ref)
        if existing is not None and existing is not entry and not existing.dropped:
            if existing.host_state != "promoting":
                existing.holders |= entry.holders
                self._request_drop(entry, "drops_duplicate")
                return
            # W1.4.1: a promotion still copying (no copy to serve yet) yields to the commit's complete copies of the
            # same node: it ends, its holders pass to the commit's entry
            entry.holders |= existing.holders
            self._supersede_promotion(existing)
        self._by_ref[node_ref] = entry
        if not entry.holders:
            if entry.drop_on_publish:  # W1.4.1: deleted with its tree while unpublished: never retained
                self._request_drop(entry, "drops_released")
            else:
                self._lost_holders(entry, entry.released_by or ())
        self._balance()

    def retire(self, entry: RawEntry, state: str) -> None:
        """A failed or cancelled commit: its copies are dropped once no worker reads them (R7)."""
        if entry.state == "published" or entry.dropped:
            return
        if entry.unpublished:
            self._unpublished_bytes -= entry.nbytes
            self._unpublished_count -= 1
        entry.state = "cancelled" if state == "cancelled" else "failed"
        self._request_drop(entry, "drops_cancelled" if entry.state == "cancelled" else "drops_failed")
        self._cond.notify_all()

    # -- holders ---------------------------------------------------------------------------------------------------
    def hold(self, holder: str, node_refs: Iterable[str], *, exclusive: bool = True) -> dict[str, int]:
        """Make ``holder`` hold exactly the published entries of ``node_refs`` (``exclusive``; else add them) plus its
        unpublished ones. Entries left without holders are dropped once published (no memory ratchet)."""
        refs = set(node_refs)
        attached = detached = 0
        with self._cond:
            for entry in list(self._entries.values()):
                if entry.node_ref is not None and entry.node_ref in refs:
                    if holder not in entry.holders:
                        if self.retain == "lru" and self._is_unheld(entry):  # wl-w1b7: an unheld entry is a hit
                            self.retain_["reattached"] += 1
                            entry.released_by = None
                        entry.holders.add(holder)
                        attached += 1
                elif exclusive and holder in entry.holders and not entry.unpublished:
                    entry.holders.discard(holder)
                    self._reown_promotions(entry, holder)
                    detached += 1
                    if not entry.holders:
                        self._lost_holders(entry, (holder,))
            self._balance()
            self._cond.notify_all()
        return {"attached": attached, "detached": detached}

    def release_holder(self, holder: str, *, drop: bool | None = None) -> dict[str, int]:
        """Detach ``holder`` from every entry (session close, delete, replace). Dropped copies return their blocks to
        the pool; when the release leaves the tier idle (no entry, job or reservation) the pool is shrunk to its keep
        bytes (``keep_free_bytes``, default the host budget), recorded with reason ``session_release``.
        wl-w1b7: ``drop=True`` (whole-tree delete) drops the entries it leaves without holders whatever ``retain``;
        otherwise ``retain="lru"`` keeps them as unheld entries."""
        detached = 0
        with self._cond:
            if drop and self._promote_jobs:  # W1.4.1 (C2) / W1.4.3: the tree's promotions end first, unless held on
                owner = _tree_owner(holder)
                self._settle_promotions(lambda name: _in_tree(owner, name), "cancelled")
            for entry in list(self._entries.values()):
                if holder in entry.holders:
                    entry.holders.discard(holder)
                    self._reown_promotions(entry, holder)
                    detached += 1
                    if not entry.holders and not entry.unpublished:
                        if drop:
                            self._request_drop(entry, "drops_released")
                        else:
                            self._lost_holders(entry, (holder,))
                    elif not entry.holders and self.retain == "lru":  # attributed when it publishes unheld
                        entry.released_by = frozenset((holder,))
                        entry.drop_on_publish = bool(drop)  # W1.4.1: a whole-tree delete drops it when it publishes
            self._balance()
            self._cond.notify_all()
            idle = self._idle()
        freed = 0
        if idle and self._buffers.cached_bytes > self._keep_free():
            freed = self._buffers.shrink(self._keep_free(), reason="session_release")["bytes"]
        return {"detached": detached, "freed_host_bytes": freed}

    def drop_unheld(self, owner: str) -> int:
        """wl-w1b7 (whole-tree delete of session ``owner``): drop every unheld entry whose last holders all belonged to
        that session -- ``owner`` itself, its tree holder ``owner/...`` and its detached branches ``owner~...`` -- so
        a deleted tree leaves no cached raw copy behind (the warm-up purge's isolation). Returns how many dropped."""
        dropped = 0
        with self._cond:
            if self._promote_jobs:  # W1.4.1 (C2) / W1.4.3: the tree's promotions end first, unless held on
                self._settle_promotions(lambda name: _in_tree(owner, name), "cancelled")
            for entry in list(self._entries.values()):
                if not entry.released_by or entry.holders or not all(
                        _in_tree(owner, name) for name in entry.released_by):
                    continue
                if entry.unpublished:  # W1.4.1: still unpublished: dropped when it publishes, never retained
                    entry.drop_on_publish = True
                elif self._is_unheld(entry):
                    self._request_drop(entry, "drops_released")
                    dropped += 1
            if dropped:
                self.retain_["dropped_on_delete"] += dropped
                self._balance()
                self._cond.notify_all()
        return dropped

    def _is_unheld(self, entry: RawEntry) -> bool:
        """A published entry nobody holds that is still cached (``retain="lru"``)."""
        return (entry.state == "published" and not entry.holders and not entry.dropped
                and not entry.drop_requested)

    def _lost_holders(self, entry: RawEntry, last: Iterable[str]) -> None:
        """A published entry lost its last holder (caller holds the condition): dropped (``retain="drop"``, the
        F6 rule) or kept as an unheld LRU entry attributed to the holders that held it last (``"lru"``)."""
        if self.retain != "lru" or entry.state != "published" or entry.dropped or entry.drop_requested:
            self._request_drop(entry, "drops_released")
            return
        entry.released_by = frozenset(last)
        entry.retained = True
        self.retain_["retained"] += 1

    def _evicted(self, where: str, entry: RawEntry) -> None:
        """Split an eviction count by holder state (wl-w1b7 records; the totals stay where they were)."""
        self.retain_["evictions"][f"{where}_{'unheld' if not entry.holders else 'held'}"] += 1

    def _lru_key(self, entry: RawEntry):
        """Eviction order: least recently used first; with ``retain="lru"`` unheld entries before held ones."""
        if self.retain == "lru":
            return (0 if self._is_unheld(entry) else 1, entry.last_used)
        return entry.last_used

    def transfer(self, entries: Iterable[RawEntry], holder: str, new_holder: str) -> int:
        """W1.3 (``tree_switch="detach"``): ``entries`` (a detached branch's commits) pass from ``holder`` to
        ``new_holder`` -- their label's branch and their holder -- so the live session's exclusive holds and session
        controls no longer reach them. Nothing else changes (state, copies, pins, only-copy protection). Returns how
        many entries moved."""
        moved = 0
        with self._cond:
            for entry in entries:
                if entry.dropped:
                    continue
                if entry.label[0] == holder:
                    entry.label = (new_holder, entry.label[1])
                if holder in entry.holders:
                    entry.holders.discard(holder)
                    entry.holders.add(new_holder)
                    self._reown_promotions(entry, holder)
                moved += 1
            self._cond.notify_all()
        return moved

    # -- restore promotion (wl-w1b7, ``promote="host"``; W1.4.1: review C2, I1, I2) ---------------------------------
    def promotion_cap(self) -> int:
        """W1.4.1 (C2): the cap on transient promotion bytes -- rows a restore offered and the worker has not copied
        yet, plus the destination of the row being copied: max(2 x the largest promoted node, 2 GiB)."""
        return max(2 * self._largest_promotion, PROMOTION_TRANSIENT_FLOOR)

    def promotion_open(self, *, node_ref: str, groups: Sequence[tuple[str, int | None, Mapping[Any, Any]]],
                       holder: str, origin: str, device: Any,
                       on_copied: Callable[[int, int], Any] | None = None) -> tuple[Any, str | None]:
        """W1.4.1 (C2): admit a node a restore is about to read from the store into host raw, held by ``holder``,
        BEFORE its rows arrive. ``groups``: ``(chunk_id, token_start, {entry key: tensor or (dtype, shape)})``.
        Admission is the ordinary LRU insert behind a physical pinned-pool reservation (I2: nothing is evicted for a
        promotion the pool cannot take). The restore then offers every row right after reading it
        (:meth:`promotion_offer`; it keeps none for promotion) and closes the job after its install
        (:meth:`promotion_close`). The promotion worker copies each offered row into the entry's pinned blocks and
        releases it at once; ``on_copied(start_ns, end_ns)`` runs on the worker once the node is complete. Returns
        ``(job, None)`` or ``(None, skip reason)``."""
        device = torch.device(device)
        with self._cond:
            entry, skip = self._promotion_entry(node_ref, groups, holder, origin, "pre_rope", device)
            if entry is None:
                return None, skip
            job = _PromotionJob("host", entry, holder, on_copied)
            job.expected = {(index, key) for index, group in enumerate(entry.groups) for key in group.meta}
            job.max_row = max((meta[5] for group in entry.groups for meta in group.meta.values()), default=0)
            self._largest_promotion = max(self._largest_promotion, entry.nbytes)
            self._copy_row = max(self._copy_row, job.max_row)
            self._promote_jobs.append(job)
            self._ensure_promote_worker()
            return job, None

    def promotion_offer(self, job: Any, key: Any, tensor: torch.Tensor, *, group: int = 0) -> bool:
        """W1.4.1 (C2): one row the restore just read for ``job``'s node (``key`` in group ``group``); non-blocking.
        Taken while the transient bytes stay within :meth:`promotion_cap` (room is kept for the row being copied);
        otherwise the job ends at once (skip ``transient_cap``) and every row it holds is released. A row offered
        again (a node attended by several spans) is taken once. Returns whether the job holds the row."""
        if job is None:
            return False
        with self._cond:
            if job.state != "open":
                return False
            groups = job.entry.groups
            vkey = (group, key)
            if vkey in job.received:
                return True
            if not 0 <= group < len(groups) or key not in groups[group].meta:
                self._abandon(job, "error", f"unexpected row {key!r}")
                return False
            _, dtype, shape, _, _, nbytes = groups[group].meta[key]
            if tensor.dtype != dtype or tensor.numel() * tensor.element_size() != nbytes:
                self._abandon(job, "error", f"row {key!r} is {tensor.dtype} x {tuple(tensor.shape)}, expected "
                                            f"{dtype} x {shape}")
                return False
            if self._transient + nbytes + max(job.max_row, self._copy_row) > self.promotion_cap():
                self._abandon(job, "transient_cap", None)
                return False
            job.received.add(vkey)
            job.pending.append((vkey, tensor))
            self._add_transient(nbytes)
            self._cond.notify_all()
            return True

    def promotion_close(self, job: Any, *, error: str | None = None) -> None:
        """W1.4.1 (C2): the restore's install ended -- no more rows come for ``job`` (``error``: it failed; the job
        ends and its rows are released)."""
        if job is None:
            return
        with self._cond:
            if job.state != "open":
                return
            if error is not None:
                self._abandon(job, "error", error)
                return
            job.state = "closed"
            self._cond.notify_all()

    def promote_host(self, *, node_ref: str, groups: Sequence[tuple[str, int | None, Mapping[Any, torch.Tensor]]],
                     holder: str, origin: str, device: Any,
                     on_copied: Callable[[int, int], Any] | None = None) -> dict[str, Any]:
        """A whole node's rows at once (:meth:`promotion_open`, an offer per row, :meth:`promotion_close`): copied
        into pooled, pinned host blocks on the promotion worker; the entry serves restores once complete. Returns
        ``{"promoted": bool, "skip": reason | None, "bytes": raw bytes, "skip_bytes": raw bytes skipped,
        "host_bytes": the host blocks' bytes}``."""
        job, skip = self.promotion_open(node_ref=node_ref, groups=groups, holder=holder, origin=origin,
                                        device=device, on_copied=on_copied)
        if job is None:
            return {"promoted": False, "skip": skip, "bytes": 0, "skip_bytes": _rows_bytes(groups), "host_bytes": 0}
        for index, (_, _, matrices) in enumerate(groups):
            for key, tensor in matrices.items():
                if not self.promotion_offer(job, key, tensor, group=index):
                    break
        self.promotion_close(job)
        with self._cond:
            if job.state == "abandoned":
                return {"promoted": False, "skip": job.reason, "bytes": 0, "skip_bytes": job.entry.nbytes,
                        "host_bytes": 0}
            return {"promoted": True, "skip": None, "bytes": job.entry.nbytes, "skip_bytes": 0,
                    "host_bytes": job.entry.host_nbytes}

    def promote_installed(self, nodes: Sequence[Mapping[str, Any]], *, holder: str, origin: str, device: Any,
                          after_event: Any = None) -> dict[str, Any]:
        """Insert restored nodes into host raw from the rows a restore installed (``form="installed"``: post-RoPE K,
        V), held by ``holder``: :meth:`promotion_prepare_installed` then :meth:`promotion_issue_installed`."""
        prepared = self.promotion_prepare_installed(nodes, holder=holder, origin=origin, device=device)
        return self.promotion_issue_installed(prepared, after_event=after_event)

    def promotion_prepare_installed(self, nodes: Sequence[Mapping[str, Any]], *, holder: str, origin: str,
                                    device: Any, exact: bool = True) -> dict[str, Any]:
        """W1.4.1 (N1): the first half of an installed promotion, run when the replay ends (prep): admission (the
        ordinary LRU insert behind a pinned-pool reservation) and the destination blocks. ``nodes``: ``{"node_ref",
        "groups": [(chunk_id, token_start, {(MatrixKey, stream): device rows})]}``. The copies are issued later by
        :meth:`promotion_issue_installed` (the first new forward: they overlap generation), or the preparation is
        given back by :meth:`promotion_cancel_installed`."""
        device = torch.device(device)
        prepared: dict[str, Any] = {"device": device, "holder": holder, "planned": [], "blocks": {}, "views": {},
                                    "skips": {}, "skip_bytes": {}, "state": "prepared"}
        with self._cond:
            for node in nodes:
                entry, skip = self._promotion_entry(node["node_ref"], node["groups"], holder, origin, "installed",
                                                    device)
                if entry is None:
                    prepared["skips"][skip] = prepared["skips"].get(skip, 0) + 1
                    prepared["skip_bytes"][skip] = prepared["skip_bytes"].get(skip, 0) + _rows_bytes(node["groups"])
                    continue
                entry.exact = bool(exact)  # W1.4.3 (E1): rows replayed on a lossy history are no clean replay
                sources = {(id(group), key): tensor for group, (_, _, matrices) in zip(entry.groups, node["groups"])
                           for key, tensor in matrices.items()}
                prepared["planned"].append((entry, sources))
                self._largest_promotion = max(self._largest_promotion, entry.nbytes)
        try:
            for entry, _ in prepared["planned"]:
                sizes = [meta[5] for group in entry.groups for meta in group.meta.values()]
                reserved, entry.pool_reserved = entry.pool_reserved, 0
                prepared["blocks"][entry.token] = self._buffers.acquire(sizes, entry.cuda and self.pinned,
                                                                        reserved=reserved)
            for entry, _ in prepared["planned"]:
                blocks = iter(prepared["blocks"][entry.token])
                for group in entry.groups:
                    for key, (_, dtype, shape, _, _, nbytes) in group.meta.items():
                        prepared["views"][(entry.token, id(group), key)] = next(blocks).tensor[:nbytes].view(
                            dtype).view(shape)
        except BaseException as exc:  # noqa: BLE001 - nothing was issued: every block can go back
            self._give_back(prepared, f"{type(exc).__name__}: {exc}", safe=True)
        return prepared

    def promotion_issue_installed(self, prepared: dict[str, Any], *, after_event: Any = None) -> dict[str, Any]:
        """W1.4.1 (N1): issue a prepared installed promotion's device-to-host copies into its pinned blocks, layer by
        layer on the tier's copy stream after ``after_event``. ``layer_events`` maps each layer to the event after its
        copies (the fence of a cache that may overwrite restored slots); ``copy_events`` holds the timing events
        around all the copies -- the second is the AGGREGATE completion event every later writer of those cache slots
        must wait for (C1). The entries serve restores once the worker saw their copies complete. Returns the promoted
        refs, raw bytes (``bytes``; skipped: ``skips`` / ``skip_bytes`` by reason), the host blocks' bytes, the
        events and ``issue_ns`` (the host interval that issued the copies). I1: a failure after a copy was issued
        synchronizes the copy stream before any block returns to the pool; if that fails the blocks are quarantined."""
        started, issue_start = time.perf_counter(), time.monotonic_ns()
        device = prepared["device"]
        report: dict[str, Any] = {"promoted": [], "skips": dict(prepared["skips"]),
                                  "skip_bytes": dict(prepared["skip_bytes"]), "bytes": 0, "host_bytes": 0,
                                  "layer_events": {}, "copy_events": None, "issue_ns": None}
        if "error" in prepared:
            report["error"] = prepared["error"]
        planned, views = prepared["planned"], prepared["views"]
        if prepared["state"] != "prepared" or not planned:
            prepared["state"] = "issued"
            return report
        prepared["state"] = "issued"
        stream, issued, start_event, done = None, False, None, None
        try:
            layers = sorted({_layer_of(key) for entry, _ in planned for group in entry.groups for key in group.meta})
            stream = self._stream(device) if device.type == "cuda" else None
            context = (torch.cuda.device(device), torch.cuda.stream(stream)) if stream is not None else ()
            with contextlib.ExitStack() as stack:
                for item in context:
                    stack.enter_context(item)
                if stream is not None and after_event is not None:
                    stream.wait_event(after_event)
                start_event = torch.cuda.Event(enable_timing=True) if stream is not None else None
                if start_event is not None:
                    start_event.record(stream)
                for layer in layers:  # layer-major: layer L's copies finish first, its fence releases first
                    for entry, sources in planned:
                        for group in entry.groups:
                            for key in group.meta:
                                if _layer_of(key) != layer:
                                    continue
                                source = sources[(id(group), key)]
                                target = views[(entry.token, id(group), key)]
                                issued = True  # I1: from here on a failure must make the copies complete first
                                target.view(source.shape).copy_(source, non_blocking=stream is not None)
                                if stream is not None and source.device.type == "cuda":
                                    source.record_stream(stream)
                    if stream is not None:
                        event = torch.cuda.Event()
                        event.record(stream)
                        report["layer_events"][layer] = event
                if stream is not None:
                    done = torch.cuda.Event(enable_timing=True)
                    done.record(stream)
        except BaseException as exc:  # noqa: BLE001 - promotion is an optimization: nothing it planned stays
            safe = True
            if issued and stream is not None:  # I1: the issued copies end before their blocks can be reused
                try:
                    stream.synchronize()
                except BaseException:  # noqa: BLE001 - the copies' state is unknown: never reuse their blocks
                    safe = False
            self._give_back(prepared, f"{type(exc).__name__}: {exc}", safe=safe, issued=issued)
            report.update(promoted=[], layer_events={}, error=f"{type(exc).__name__}: {exc}", copies_safe=safe)
            return report
        report["issue_ns"] = (issue_start, time.monotonic_ns())
        if start_event is not None and done is not None:
            report["copy_events"] = (start_event, done)
        with self._cond:
            self.promotion_["issue_seconds"] += time.perf_counter() - started
            for entry, _ in planned:
                job = _PromotionJob("device", entry, prepared["holder"], None)
                job.blocks = prepared["blocks"][entry.token]
                job.views = {(id(group), key): views[(entry.token, id(group), key)] for group in entry.groups
                             for key in group.meta}
                job.events = (start_event, done)
                self._promote_jobs.append(job)
                report["promoted"].append(entry.node_ref)
                report["bytes"] += entry.nbytes
                report["host_bytes"] += entry.host_nbytes
            self._ensure_promote_worker()
        return report

    def promotion_cancel_installed(self, prepared: dict[str, Any], reason: str = "cancelled") -> None:
        """W1.4.1 (N1): a prepared installed promotion that is never issued (its request failed first): no copy was
        issued, so its blocks go straight back; its entries drop."""
        if prepared.get("state") == "prepared" and prepared["planned"]:
            self._give_back(prepared, None, safe=True, reason=reason)
        prepared["state"] = "cancelled"

    def _give_back(self, prepared: dict[str, Any], error: str | None, *, safe: bool, issued: bool = False,
                   reason: str = "error") -> None:
        """Release (``safe``) or quarantine a prepared promotion's blocks, return its reservations, drop its
        entries."""
        with self._cond:
            for entry, _ in prepared["planned"]:
                blocks = prepared["blocks"].pop(entry.token, None)
                if blocks is not None:
                    if safe:
                        self._buffers.release(blocks)
                    else:
                        self._quarantine(blocks)
                elif entry.pool_reserved:
                    self._buffers.unreserve(entry.pool_reserved)
                    entry.pool_reserved = 0
                self._host_reserved_spill -= entry.host_nbytes
                self.promotion_["skips"][reason] = self.promotion_["skips"].get(reason, 0) + 1
                self.promotion_["skip_bytes"][reason] = self.promotion_["skip_bytes"].get(reason, 0) + entry.nbytes
                entry.host_state = "none"
                self._drop_now(entry)
                if error is not None:
                    self._event("promotion_error", node_ref=entry.node_ref, error=error, issued=issued,
                                quarantined=not safe)
            prepared["planned"] = []
            if error is not None:
                prepared["error"] = error
            self._cond.notify_all()

    def count_promotion_skip(self, reason: str, nbytes: int = 0) -> None:
        """A promotion the caller skipped before reaching the tier (e.g. ``partial_range``; ``nbytes``: its rows)."""
        with self._cond:
            self.promotion_["requested"] += 1
            self.promotion_["skips"][reason] = self.promotion_["skips"].get(reason, 0) + 1
            self.promotion_["skip_bytes"][reason] = self.promotion_["skip_bytes"].get(reason, 0) + int(nbytes)

    def _promotion_entry(self, node_ref: str, groups, holder: str, origin: str, form: str, device: torch.device):
        """A published, host-only entry for ``node_ref`` whose host copy is being made (``host_state="promoting"``;
        caller holds the condition), or ``(None, skip reason)``. Room is an ordinary LRU insert (unheld entries
        first, then held ones; never pinned or in-flight copies). W1.4.1 (I2): the victims are planned, the pinned
        pool's physical capacity is reserved (counting allocations in flight, other reservations and what the
        victims return), and only then are the victims evicted: ``cannot_fit`` / ``pinned_pool_full`` evict
        nothing. The reservation (``entry.pool_reserved``) is consumed when the blocks are acquired."""
        self.promotion_["requested"] += 1
        skip = None
        if not self.enabled or self.promotion_mode != "host":
            skip = "disabled"
        else:
            existing = self._by_ref.get(node_ref)
            if existing is not None and not existing.dropped and not existing.drop_requested:
                skip = "exists"
        built, offset, nbytes = [], 0, 0
        host_nbytes = 0
        if skip is None:
            for chunk_id, token_start, matrices in groups:
                group = _RawGroup(chunk_id, token_start, _meta_rows(matrices), offset)
                # the entry's copies belong on the restore device (placements, uploads), whatever the rows' source
                group.meta = {key: (device, dtype, shape if len(shape) == 2 else (shape[0], _numel(shape[1:])),
                                    (_numel(shape[1:]), 1) if len(shape) != 2 else stride, start, size)
                              for key, (_, dtype, shape, stride, start, size) in group.meta.items()}
                group.device = None
                offset = group.end_offset
                nbytes += sum(meta[5] for meta in group.meta.values())
                built.append(group)
            host_nbytes = sum(_size_class(meta[5]) for group in built for meta in group.meta.values())
            fits, victims, freed = self._plan_host_room(host_nbytes)
            if not fits:
                skip = "cannot_fit"
            elif not self._buffers.reserve(host_nbytes, freeing=sum(entry.host_nbytes for entry in victims
                                                                    if not entry.host_read_event)):
                skip = "pinned_pool_full"  # e.g. blocks still waiting for their uploads before they return
            else:
                self._evict_host(victims)
        if skip is not None:
            self.promotion_["skips"][skip] += 1
            self.promotion_["skip_bytes"][skip] += _rows_bytes(groups)
            return None, skip
        self._token += 1
        entry = RawEntry(self._token, (holder, f"promoted:{origin}"), built, nbytes, holders=(holder,), release=None,
                         ready_event=None, cuda=device.type == "cuda")
        entry.state, entry.node_ref = "published", node_ref
        entry.on_device, entry.device_owned, entry.host_state = False, True, "promoting"
        entry.origin, entry.form = origin, form
        entry.pool_reserved = host_nbytes
        self._clock += 1
        entry.last_used = self._clock
        self._entries[entry.token] = entry
        self._by_ref[node_ref] = entry
        self._host_reserved_spill += entry.host_nbytes
        self._peaks()
        return entry, None

    def _plan_host_room(self, nbytes: int) -> tuple[bool, list, int]:
        """(caller holds the condition) The LRU victims :meth:`_host_room` would evict for ``nbytes``, without
        evicting: ``(fits, victims, freed host bytes)``."""
        budget = self._host_budget
        used = self._host_bytes + self._host_reserved_spill
        if used + nbytes <= budget:
            return True, [], 0
        victims, freed = [], 0
        for entry in sorted((entry for entry in self._entries.values()
                             if entry.state == "published" and entry.host_state == "complete" and not entry.pinned
                             and not entry.dropped and entry.placed != "host_raw"), key=self._lru_key):
            if used - freed + nbytes <= budget:
                break
            victims.append(entry)
            freed += entry.host_nbytes
        return used - freed + nbytes <= budget, victims, freed

    def _evict_host(self, victims: Sequence[RawEntry]) -> None:
        """(caller holds the condition) Evict planned host victims exactly as :meth:`_host_room` does."""
        for entry in victims:
            self.stats_["evictions_host"] += 1
            self._evicted("host", entry)
            if entry.on_device:
                self._drop_host(entry, prepare=False)
            else:
                self._drop_now(entry, prepare=True)

    def _add_transient(self, nbytes: int) -> None:
        self._transient += nbytes
        if self._transient > self.promotion_["transient_peak_bytes"]:
            self.promotion_["transient_peak_bytes"] = self._transient

    def _quarantine(self, blocks: Sequence[_HostBlock]) -> None:
        """W1.4.1 (I1): blocks a copy may still write (its completion could not be established): never reused."""
        self._quarantined.extend(blocks)
        self.promotion_["quarantined_blocks"] += len(blocks)
        self.promotion_["quarantined_bytes"] += sum(block.nbytes for block in blocks)

    def _abandon(self, job: Any, reason: str, error: str | None) -> None:
        """(caller holds the condition) A host job ends without its entry: rows it holds are released at once; its
        entry and blocks go now, or when the worker finishes the row it is copying."""
        if job.state in ("done", "abandoned"):
            return
        self._transient -= sum(tensor.numel() * tensor.element_size() for _, tensor in job.pending)
        job.pending.clear()
        job.state, job.reason = "abandoned", reason
        self.promotion_["skips"][reason] = self.promotion_["skips"].get(reason, 0) + 1
        self.promotion_["skip_bytes"][reason] = self.promotion_["skip_bytes"].get(reason, 0) + job.entry.nbytes
        if error is not None:
            self._event("promotion_error", node_ref=job.entry.node_ref, error=error)
        if not job.active:
            self._retire(job)
        self._cond.notify_all()

    def _retire(self, job: Any) -> None:
        """(caller holds the condition) Remove an ended job: its blocks back to the pool (or quarantined), its pool
        reservation and host reservation returned, its entry dropped unless it completed."""
        if job in self._promote_jobs:
            self._promote_jobs.remove(job)
        entry = job.entry
        if job.state == "done":
            return
        if job.blocks is not None:
            if job.unsafe:
                self._quarantine(job.blocks)
            else:
                self._buffers.release(job.blocks)
            job.blocks = None
        if entry.pool_reserved:
            self._buffers.unreserve(entry.pool_reserved)
            entry.pool_reserved = 0
        if entry.host_state == "promoting":
            self._host_reserved_spill -= entry.host_nbytes
            entry.host_state = "none"
            if not entry.dropped:
                self._drop_now(entry)
        self._cond.notify_all()

    def _supersede_promotion(self, entry: RawEntry) -> None:
        """(caller holds the condition) The promotion making ``entry`` ends: a host job is abandoned (its rows
        released); a device job is cancelled (the worker retires it once its copies completed)."""
        for job in list(self._promote_jobs):
            if job.entry is entry:
                if job.kind == "host":
                    self._abandon(job, "superseded", None)
                else:
                    job.cancelled = True
                    self.promotion_["skips"]["superseded"] += 1
                    self.promotion_["skip_bytes"]["superseded"] += entry.nbytes
        if not entry.dropped and entry.host_state == "promoting" and not any(job.entry is entry
                                                                            for job in self._promote_jobs):
            self._drop_now(entry)
        self._cond.notify_all()

    def _reown_promotions(self, entry: RawEntry, leaving: str) -> None:
        """(caller holds the condition) W1.4.3 (review I7): ``leaving`` stopped holding ``entry``: a promotion job of
        the entry it owned is owned by a current holder from now on (the first, sorted; counted
        ``promotion.transferred``), so a later whole-tree delete settles it by who holds the entry NOW. No holder
        left (a close keeps the entry unheld, retain="lru"): the leaving holder stays the owner -- its tree's delete
        ends it."""
        for job in self._promote_jobs:
            if job.entry is entry and job.holder == leaving and entry.holders:
                job.holder = sorted(entry.holders)[0]
                self.promotion_["transferred"] += 1

    def _settle_promotions(self, member: Callable[[str], bool], reason: str) -> int:
        """(caller holds the condition) W1.4.3 (delta review, C2): a whole-tree delete settles the promotion jobs of
        its holders (``member``). A job whose entry another holder still holds -- one outside the deleted tree, e.g.
        a session that forked from it -- is NOT cancelled: it completes for that holder, which takes the job over
        (``job.holder``; counted ``promotion.transferred``). The rest end now (:meth:`_cancel_promotions`). Returns
        how many were cancelled."""
        cancel = []
        for job in [job for job in self._promote_jobs if member(job.holder)]:
            survivors = sorted(name for name in job.entry.holders if not member(name))
            if survivors and job.state not in ("done", "abandoned") and not job.cancelled:
                job.holder = survivors[0]
                self.promotion_["transferred"] += 1
            else:
                cancel.append(job)
        if not cancel:
            return 0
        chosen = set(map(id, cancel))
        return self._cancel_promotions(lambda name: True, reason, jobs=[job for job in self._promote_jobs
                                                                        if id(job) in chosen])

    def _cancel_promotions(self, match: Callable[[str], bool], reason: str, *, jobs: list | None = None) -> int:
        """(caller holds the condition) W1.4.1 (C2): the promotion jobs of the matching holders (or ``jobs``) end now
        -- queued rows released, entries dropped -- and this returns only after the worker finished any row it was
        copying for them and every device copy they issued completed (whole-tree delete, close: nothing of theirs
        runs on). Each ended job counts once in ``promotion.skips.<reason>`` (W1.4.3: issued device jobs too)."""
        if jobs is None:
            jobs = [job for job in self._promote_jobs if match(job.holder)]
        for job in jobs:
            if job.kind == "host":
                self._abandon(job, reason, None)
            elif not job.cancelled:
                job.cancelled = True
                self.promotion_["skips"][reason] = self.promotion_["skips"].get(reason, 0) + 1
                self.promotion_["skip_bytes"][reason] = (self.promotion_["skip_bytes"].get(reason, 0)
                                                         + job.entry.nbytes)
        self._cond.notify_all()
        self._wait(lambda: not any(job in self._promote_jobs or job.active for job in jobs), None,
                   "promotion cancel")
        return len(jobs)

    def _ensure_promote_worker(self) -> None:
        if self._promote_worker is None or not self._promote_worker.is_alive():
            self._promote_worker = threading.Thread(target=self._promote_work, name="reelcache-raw-promote",
                                                    daemon=True)
            self._promote_worker.start()
        self._cond.notify_all()

    def _next_promotion(self) -> Any:
        """(caller holds the condition) The next promotion job with work: host jobs first (a row to copy, a closed
        job to complete), then device jobs (wait for their copies)."""
        for job in self._promote_jobs:
            if job.kind == "host" and not job.active and (job.pending or job.state == "closed"):
                return job
        for job in self._promote_jobs:
            if job.kind == "device" and not job.active:
                return job
        return None

    def _promote_work(self) -> None:
        """The promotion worker: copies offered rows into pinned blocks, completes closed jobs, and waits for device
        copies to complete. Separate from the spill worker, so a promotion never delays a demotion."""
        while True:
            with self._cond:
                job = self._next_promotion()
                while job is None and not self._closed:
                    self._cond.wait()
                    job = self._next_promotion()
                if job is None:
                    return
                job.active = True
                self._promote_active += 1
            try:
                if job.kind == "host":
                    self._promote_host_step(job)
                else:
                    self._promote_device(job)
            except Exception as exc:  # noqa: BLE001 - recorded; the worker keeps serving
                with self._cond:
                    self.stats_["worker_errors"] += 1
                    self._event("worker_error", job=job.kind, error=f"{type(exc).__name__}: {exc}")
                    if job.kind == "host" and job.state != "done":
                        job.active = False
                        self._abandon(job, "error", f"{type(exc).__name__}: {exc}")
            finally:
                with self._cond:
                    job.active = False
                    self._promote_active -= 1
                    if job.state == "abandoned":
                        self._retire(job)
                    self._cond.notify_all()

    def _promote_host_step(self, job: Any) -> None:
        """One unit of a host job: acquire its destination blocks (once), then copy one offered row into them and
        release the row at once; or, when the restore closed the job and every row was copied, complete the entry
        (``on_copied`` after)."""
        entry = job.entry
        if job.blocks is None:
            with self._cond:
                if job.state == "abandoned":
                    return
                reserved, entry.pool_reserved = entry.pool_reserved, 0
            sizes = [meta[5] for group in entry.groups for meta in group.meta.values()]
            blocks = self._buffers.acquire(sizes, entry.cuda and self.pinned, reserved=reserved)
            views, pool = {}, iter(blocks)
            for index, group in enumerate(entry.groups):
                for key, (_, dtype, shape, _, _, nbytes) in group.meta.items():
                    views[(index, key)] = next(pool).tensor[:nbytes].view(dtype).view(shape)
            with self._cond:
                job.blocks, job.views = blocks, views  # an abandoned job's blocks are released by its retirement
        row = callback = None
        with self._cond:
            if job.state == "abandoned":
                return
            if job.pending:
                vkey, source = job.pending.popleft()
                destination = job.views[vkey]
                written = destination.numel() * destination.element_size()
                cap = self.promotion_cap()
                # the destination being written counts too; room for it was kept when the row was taken, so this
                # waits only while another copy is in flight (never with the single promotion worker)
                self._wait(lambda: self._transient + written <= cap or not self._copying, None, "promotion copy")
                self._copying += 1
                self._add_transient(written)
                row, source = (vkey, source, destination, written), None
            elif job.state == "closed":
                if job.copied != job.expected:  # the restore did not read every row: an incomplete node
                    self._abandon(job, "partial_range", None)
                    return
                ready = {(id(group), key): job.views[(index, key)] for index, group in enumerate(entry.groups)
                         for key in group.meta}
                completed = self._promote_finish(entry, job.blocks, ready, seconds=job.seconds, kind="host",
                                                 error=None)
                job.blocks, job.state = None, "done"
                if job in self._promote_jobs:
                    self._promote_jobs.remove(job)
                if completed and job.on_copied is not None:
                    now = time.monotonic_ns()
                    callback = (job.on_copied, job.start_ns if job.start_ns is not None else now, now)
            else:
                return
        if callback is not None:
            try:  # e.g. the restoring request's promotion_copy span (telemetry only)
                callback[0](callback[1], callback[2])
            except Exception as exc:  # noqa: BLE001 - recorded
                with self._cond:
                    self._event("promotion_callback_error", node_ref=entry.node_ref,
                                error=f"{type(exc).__name__}: {exc}")
            return
        if row is None:
            return
        vkey, source, destination, written = row
        row = None
        started = time.perf_counter()
        if job.start_ns is None:
            job.start_ns = time.monotonic_ns()
        nbytes = source.numel() * source.element_size()
        try:
            destination.view(source.shape).copy_(source)
        finally:
            source = None  # released: the only reference the promotion held
            with self._cond:
                self._copying -= 1
                self._transient -= nbytes + written
                job.seconds += time.perf_counter() - started
                job.copied.add(vkey)
                self._cond.notify_all()

    def _promote_device(self, job: Any) -> None:
        """A device job: wait for its device-to-host copies (the aggregate event), then complete the entry -- or, if
        that wait fails, retire it with its blocks quarantined (I1: the copies may still write them)."""
        error, seconds = None, 0.0
        start, done = job.events
        wall = time.perf_counter()
        try:
            if done is not None:
                done.synchronize()
                seconds = start.elapsed_time(done) / 1e3 if start is not None else time.perf_counter() - wall
        except BaseException as exc:  # noqa: BLE001 - recorded
            error = f"{type(exc).__name__}: {exc}"
        with self._cond:
            if error is not None:
                job.unsafe = True
            if job.cancelled and error is None:  # counted once when cancelled (skips.cancelled / superseded)
                self._buffers.release(job.blocks)
                job.blocks = None
                job.entry.host_state, job.state = "none", "abandoned"
                self._host_reserved_spill -= job.entry.host_nbytes
                if not job.entry.dropped:
                    self._drop_now(job.entry)
                if job in self._promote_jobs:
                    self._promote_jobs.remove(job)
                return
            if error is not None:
                self._host_reserved_spill -= job.entry.host_nbytes
                if not job.cancelled:  # W1.4.3 (review M10): a cancelled job was counted once, as cancelled
                    self.promotion_["skips"]["error"] += 1
                    self.promotion_["skip_bytes"]["error"] += job.entry.nbytes
                self._event("promotion_error", node_ref=job.entry.node_ref, error=error, quarantined=True)
                self._quarantine(job.blocks)
                job.blocks = None
                job.entry.host_state, job.state = "none", "abandoned"
                if not job.entry.dropped:
                    self._drop_now(job.entry)
                if job in self._promote_jobs:
                    self._promote_jobs.remove(job)
                return
            self._promote_finish(job.entry, job.blocks, job.views, seconds=seconds, kind="d2h", error=None)
            job.blocks = None
            job.state = "done"
            if job in self._promote_jobs:
                self._promote_jobs.remove(job)

    def _promote_finish(self, entry: RawEntry, blocks: list | None, ready: Mapping[tuple, torch.Tensor] | None, *,
                        seconds: float, kind: str, error: str | None) -> bool:
        """(caller holds the condition) Complete a promotion whose copies are DONE (``ready``: host views by
        ``(id(group), key)``), or retire it (an error, a drop). Returns whether the entry completed."""
        self._host_reserved_spill -= entry.host_nbytes
        if error is not None or ready is None or entry.dropped or entry.drop_requested:
            if blocks is not None:
                self._buffers.release(blocks)
            if error is not None:
                self.promotion_["skips"]["error"] += 1
                self.promotion_["skip_bytes"]["error"] += entry.nbytes
                self._event("promotion_error", node_ref=entry.node_ref, error=error)
            elif entry.dropped or entry.drop_requested:
                self.promotion_["dropped_before_complete"] += 1
            entry.host_state = "none"
            if not entry.dropped:
                self._drop_now(entry)
            completed = False
        else:
            completed = True
            for group in entry.groups:
                group.host = {key: ready[(id(group), key)] for key in group.meta}
            entry.host_blocks, entry.host_state = blocks, "complete"
            self._host_bytes += entry.host_nbytes
            stats = self.promotion_
            stats["entries"] += 1
            stats["bytes"] += entry.nbytes
            stats["pinned_entries" if entry.host_pinned else "pageable_entries"] += 1
            if kind == "host":
                stats["host_copied_entries"] += 1
                stats["host_copied_bytes"] += entry.nbytes
                stats["host_copy_seconds"] += seconds
            else:
                stats["d2h_entries"] += 1
                stats["d2h_bytes"] += entry.nbytes
                stats["d2h_seconds"] += seconds
            self._peaks()
        self._balance()
        self._cond.notify_all()
        return completed

    def shrink_buffers(self, keep_bytes: int = 0, *, reason: str = "explicit") -> dict[str, Any]:
        """Explicit idle shrink: unpin and free the pool's cached host blocks beyond ``keep_bytes``. Holds the CUDA
        driver lock about 1.5 s per GB freed: call it when no request runs. Recorded in ``host_buffers``."""
        if type(keep_bytes) is not int or keep_bytes < 0:
            raise ConfigurationError("keep_bytes must be a nonnegative integer")
        with self._cond:
            self._event("shrink_requested", reason=reason, keep_bytes=keep_bytes, idle=self._idle())
        return self._buffers.shrink(keep_bytes, reason=reason)

    # -- restores --------------------------------------------------------------------------------------------------
    def acquire(self, node_ref: str) -> RawLease | None:
        """Read access to a published entry's best raw copy (device, else host), or None; touches its LRU clock."""
        with self._cond:
            entry = self._by_ref.get(node_ref)
            if (not self.enabled or entry is None or entry.dropped or entry.drop_requested
                    or entry.state != "published"):
                self.stats_["restore_misses"] += 1
                return None
            if entry.on_device:
                tier = "gpu_raw"
            elif entry.host_state == "complete":
                tier = "host_raw"
            else:
                self.stats_["restore_misses"] += 1
                return None
            self._pin(entry, "restore")
            self._touch(entry)
            self.stats_["restores"][tier] += 1
            lease = RawLease(self, entry, tier)
            if entry.retained:  # wl-w1b7: kept unheld since its last restore (e.g. its session closed and reopened)
                entry.retained, lease.retained = False, True
                self.retain_["retained_hits"] += 1
            if entry.origin != "captured":
                self.promotion_["hits"] += 1
            return lease

    def pin_copies(self, node_refs: Iterable[str]) -> list[RawEntry]:
        """W1.4.3 (E1): pin the published entries of ``node_refs`` that hold a usable copy (on the device, or a
        complete host copy) -- no eviction, demotion or drop -- until :meth:`unpin_copies`. An availability pin, not a
        read: no LRU touch, no restore counts (the restore that follows :meth:`acquire` s them)."""
        held = []
        with self._cond:
            for ref in dict.fromkeys(node_refs):
                entry = self._by_ref.get(ref)
                if (not self.enabled or entry is None or entry.dropped or entry.drop_requested
                        or entry.state != "published" or not (entry.on_device or entry.host_state == "complete")):
                    continue
                self._pin(entry, "availability")
                held.append(entry)
        return held

    def unpin_copies(self, entries: Iterable[RawEntry]) -> None:
        """Release :meth:`pin_copies`."""
        for entry in entries:
            self._unpin(entry, "availability")

    # -- controls --------------------------------------------------------------------------------------------------
    def entries(self, *, holder: str | None = None, node_refs: Iterable[str] | None = None) -> list[RawEntry]:
        refs = None if node_refs is None else set(node_refs)
        with self._cond:
            return [entry for entry in self._entries.values() if not entry.dropped
                    and (holder is None or holder in entry.holders)
                    and (refs is None or entry.node_ref in refs)]

    def demote(self, entries: Sequence[RawEntry], *, wait: bool = True, timeout: float | None = None,
               place: bool = True) -> dict:
        """Move device copies to host (spill if needed, then drop the device copy); waits for the copies.
        ``place`` (explicit placement, Step 19 controls): the host copies stay until the session releases them,
        exempt from LRU eviction. ``place=False`` (a memory-critical section's evacuation): they stay evictable."""
        started = time.perf_counter()
        with self._cond:
            targets = [entry for entry in entries if not entry.dropped and entry.on_device]
            for entry in targets:
                entry.force_host = True
                if place:
                    entry.placed = "host_raw"
                elif entry.placed == "gpu_raw":
                    entry.placed = None
                self.stats_["demotions"] += 1
            self._balance()
            if wait:
                self._wait(lambda: all(entry.dropped or not entry.on_device or entry.spill_failed
                                       for entry in targets), timeout, "demotion")
        return {"entries": len(targets), "bytes": sum(entry.nbytes for entry in targets),
                "seconds": time.perf_counter() - started,
                "failed": sum(1 for entry in targets if entry.spill_failed)}

    def drop(self, entries: Sequence[RawEntry]) -> dict:
        """Drop the raw copies of published entries (compressed only afterwards)."""
        with self._cond:
            pending = [entry.label for entry in entries if entry.unpublished]
            if pending:
                raise StateError(f"raw entries {pending} are not published: drain before dropping (invariant I1)")
            targets = [entry for entry in entries if not entry.dropped]
            for entry in targets:
                self._request_drop(entry, "drops_released")
            self._cond.notify_all()
        return {"entries": len(targets), "bytes": sum(entry.nbytes for entry in targets)}

    def promote(self, entries: Sequence[RawEntry], *, timeout: float | None = None) -> dict:
        """Explicit host -> device placement (no automatic promotion exists); refuses beyond the device budget. An
        entry already on the device is placed in place (dev22j6): pinned like a new placement, so neither the GPU
        budget's eviction nor the pressure response demotes it (``placed_in_place`` counts them)."""
        started = time.perf_counter()
        with self._cond:
            in_place = [entry for entry in entries if not entry.dropped and entry.on_device]
            for entry in in_place:
                entry.placed, entry.force_host = "gpu_raw", False
            todo = [entry for entry in entries if not entry.dropped and not entry.on_device]
            need = sum(entry.nbytes for entry in todo)
            if self._device_bytes + need > self.gpu_budget():
                raise CapacityError(f"promoting {need} B exceeds the GPU raw budget ({self.gpu_budget()} B, "
                                    f"{self._device_bytes} B in use)")
            for entry in todo:
                if entry.host_state != "complete":
                    raise StateError(f"raw entry {entry.label} has no host copy to promote")
                self._pin(entry, "promote")
        try:
            for entry in todo:
                tensors_by_group = []
                for group in entry.groups:
                    tensors = {}
                    for key, (device, dtype, shape, stride, _, _) in group.meta.items():
                        target = (torch.empty_strided(shape, stride, dtype=dtype, device=device)
                                  if _dense(shape, stride) else torch.empty(shape, dtype=dtype, device=device))
                        target.copy_(group.host[key])
                        tensors[key] = target
                    tensors_by_group.append(tensors)
                with self._cond:
                    for group, tensors in zip(entry.groups, tensors_by_group):
                        group.device = tensors
                    entry.on_device, entry.device_owned = True, True
                    entry.placed, entry.force_host = "gpu_raw", False
                    self._device_bytes += entry.nbytes
                    self.stats_["promotions"] += 1
                    self._peaks()
        finally:
            with self._cond:
                for entry in todo:
                    self._unpin(entry, "promote")
        for entry in todo:
            if entry.cuda:
                torch.cuda.synchronize(entry.groups[0].meta[next(iter(entry.groups[0].meta))][0])
        return {"entries": len(todo), "bytes": need, "seconds": time.perf_counter() - started,
                "placed_in_place": len(in_place)}

    # -- device memory pressure (freeze F; DESIGN_D_tiers.md §13) -------------------------------------------------
    @property
    def pressure_enabled(self) -> bool:
        return self._reserve_config != 0

    def reserve_bytes(self, total: int | None = None) -> int:
        """Device headroom GPU raw copies must leave: ``gpu_reserve_bytes`` when configured, else the largest of
        ``PRESSURE_RESERVE_CHUNKS`` x the largest entry, ``PRESSURE_RESERVE_FRACTION`` of the device and the largest
        memory-critical working set observed (x ``PRESSURE_WORKING_SET_MARGIN``)."""
        if self._reserve_config is not None:
            return self._reserve_config
        total = self._device_total if total is None else total
        return max(PRESSURE_RESERVE_CHUNKS * _align(self._max_entry),
                   int(PRESSURE_RESERVE_FRACTION * total) if total else 0,
                   int(PRESSURE_WORKING_SET_MARGIN * self._working_set))

    def learned_working_set_bytes(self) -> int:
        """The largest memory-critical section working set learned so far (0: none yet; dev22j6)."""
        with self._cond:
            return int(self._working_set)

    def reserve_configured(self) -> bool:
        """Whether ``gpu_reserve_bytes`` fixes the reserve (then nothing needs to be learned; dev22j6)."""
        return self._reserve_config is not None

    def _memory(self, device: torch.device | None) -> dict[str, int] | None:
        if device is None:
            return None
        provider = self._device_memory
        if provider is None:
            if device.type != "cuda":
                return None
            provider = cuda_device_memory
        return dict(provider(device))

    def _segments(self, device: torch.device | None) -> list | None:
        """The allocator's segments on ``device`` for the strict headroom (dev22j8v4), None when unavailable (no
        CUDA device, or an injected device memory without an injected segment provider)."""
        if device is None:
            return None
        provider = self._device_segments
        if provider is None:
            if self._device_memory is not None or device.type != "cuda":
                return None
            provider = cuda_device_segments
        return list(provider(device))

    def _gpu_device(self) -> torch.device | None:
        """The device of the GPU raw copies (caller holds the condition), or None without device copies."""
        for entry in self._entries.values():
            if entry.on_device and not entry.dropped:
                for group in entry.groups:
                    for meta in group.meta.values():
                        return meta[0]
        return None

    def headroom_check(self, need_bytes: int, *, device: Any = None) -> dict | None:
        """The fitter re-check of a watched memory-critical section (dev22j4): whether ``need_bytes`` of new device
        work fit above the reserve now (conservative headroom, as at the decode seam). Returns ``{headroom,
        reserve, need_bytes, target, short}``. Returns None with the response off or no device to query."""
        if not self.pressure_enabled or not self.enabled:
            return None
        with self._cond:
            device = device if device is not None else (self._gpu_device() or self._device_hint)
        memory = self._memory(device) if device is not None else None
        if memory is None:
            return None
        headroom = device_headroom(memory)
        with self._cond:
            if memory.get("total"):
                self._device_total = int(memory["total"])
            reserve = self.reserve_bytes()
            self.pressure_["rechecks"] += 1
        need = max(0, int(need_bytes))
        return {"headroom": headroom, "reserve": reserve, "need_bytes": need, "target": reserve + need,
                "short": headroom < reserve + need}

    def pressure_check(self, where: str = "decode", *, wait: bool = True, timeout: float | None = 30.0,
                       needed_bytes: int | None = None) -> dict | None:
        """Layer 1/2 of the pressure response: when the device headroom is below the reserve, demote least recently
        used GPU raw copies of fitted and published entries (never unpublished ones, never explicit ``gpu_raw``
        placements) until the deficit is covered; ``wait`` blocks until their device copies are gone (spilled).
        Not a placement: the host copies stay evictable. ``needed_bytes`` raises the target above the reserve (e.g.
        the KV cache a request is about to allocate). A synchronous check (``wait``) that is still short after the
        demotion trims the registered device caches in order (:mod:`reelcache.device_pressure`) and empties the CUDA
        cache. Returns the recorded event, or None without device copies (no device query) or with the response off
        (``gpu_reserve_bytes=0``)."""
        if not self.pressure_enabled or not self.enabled:
            return None
        with self._cond:
            device = self._gpu_device()
        memory = self._memory(device)
        if memory is None:
            return None
        event = self._relieve(where, memory, wait=wait, timeout=timeout, everything=False, needed_bytes=needed_bytes)
        if wait and event.get("pressure"):
            self._trim_after(event, device, everything=False)
        return event

    def _trim_after(self, event: dict, device, *, everything: bool) -> None:
        """Synchronous events still short after the demotion: the registered trims, in order, then empty_cache
        (outside the tier's lock: trims run component code). Recorded in the event and the stats."""
        from .device_pressure import trim_device_caches

        target = event.get("target")

        def headroom() -> int:
            memory = self._memory(device)
            return 0 if memory is None else device_headroom(memory)

        try:
            after = headroom()
            if not everything and target is not None and after >= target:
                with self._cond:
                    event["headroom_after"] = after
                return
            trims = trim_device_caches(event["where"], device=device,
                                       needed_bytes=None if everything else target, headroom=headroom)
        except Exception as exc:  # noqa: BLE001 - best effort
            trims = {"error": f"{type(exc).__name__}: {exc}", "trims": [], "bytes": 0}
        try:
            after = headroom()
        except Exception:  # noqa: BLE001 - recording only
            after = None
        with self._cond:  # the event object is the one in the stats' recent events
            event["trims"], event["headroom_after"] = trims, after
            stats = self.pressure_
            stats["trim_calls"] += 1
            stats["trimmed_bytes"] += int(trims.get("bytes", 0))
            for record in trims.get("trims", []):
                stats["trimmed_by_component"][record["name"]] = (
                    stats["trimmed_by_component"].get(record["name"], 0) + int(record.get("bytes", 0)))

    def relieve_all(self, where: str = "oom", *, timeout: float | None = 30.0) -> dict | None:
        """Layer 3: demote every GPU raw copy that is not being planned (explicit placements and unpublished copies
        included; their host copies keep I1) and wait. The caller empties the CUDA cache and re-raises."""
        if not self.enabled:
            return None
        with self._cond:
            device = self._gpu_device() or self._device_hint
        try:
            memory = self._memory(device) if device is not None else None
        except Exception:  # noqa: BLE001 - a failed query after an OOM must not hide it
            memory = None
        event = self._relieve(where, memory, wait=True, timeout=timeout, everything=True)
        self._trim_after(event, device, everything=True)
        return event

    def _relieve(self, where: str, memory: Mapping[str, int] | None, *, wait: bool, timeout: float | None,
                 everything: bool, needed_bytes: int | None = None) -> dict:
        started = time.perf_counter()
        headroom = None if memory is None else device_headroom(memory, conservative=where != "admission")
        with self._cond:
            if memory is not None and memory.get("total"):
                self._device_total = int(memory["total"])
            reserve = self.reserve_bytes()
            target = max(reserve, int(needed_bytes or 0))
            stats = self.pressure_
            stats["checks"] += 1
            stats["by_where"][where] = stats["by_where"].get(where, 0) + 1
            deficit = None if everything else target - headroom
            event = {"event": "pressure", "where": where, "time": time.time(), "headroom": headroom,
                     "inactive_split": None if memory is None else int(memory.get("inactive_split", 0)),
                     "reserve": reserve, "needed_bytes": needed_bytes, "target": target,
                     "gpu_bytes_before": self._device_bytes, "deficit": deficit}
            if not everything and deficit <= 0:
                return {**event, "pressure": False}
            stats["pressured"] += 1
            device = [entry for entry in self._entries.values() if entry.on_device and not entry.dropped
                      and entry.pins.get("plan", 0) == 0 and (everything or entry.state in ("fitted", "published"))]
            placed = [entry for entry in device if entry.placed == "gpu_raw"]
            if not everything:
                stats["skipped_placed_bytes"] += sum(entry.nbytes for entry in placed)
                device = [entry for entry in device if entry.placed != "gpu_raw"]
            selected, covered = [], 0
            for entry in sorted(device, key=self._lru_key):
                if not everything and covered >= deficit:
                    break
                entry.force_host = True
                if entry.placed == "gpu_raw":
                    entry.placed = None
                selected.append(entry)
                covered += entry.nbytes
            self._balance()
            timed_out = False
            if wait and selected:
                try:
                    self._wait(lambda: all(entry.dropped or not entry.on_device or entry.spill_failed
                                           for entry in selected), timeout, f"pressure demotion ({where})")
                except CapacityError:
                    timed_out = True
                    stats["timeouts"] += 1
            seconds = time.perf_counter() - started
            freed = sum(entry.nbytes for entry in selected if entry.dropped or not entry.on_device)
            stats["demotions"] += 1 if selected else 0
            stats["demoted_entries"] += len(selected)
            stats["demoted_bytes"] += freed
            if wait:
                stats["wait_seconds"] += seconds
                stats["max_wait_seconds"] = max(stats["max_wait_seconds"], seconds)
            if where == "oom":
                stats["oom_reliefs"] += 1
            event.update(pressure=True, selected_entries=len(selected), selected_bytes=covered, freed_bytes=freed,
                         placed_skipped_bytes=0 if everything else sum(entry.nbytes for entry in placed),
                         failed=sum(1 for entry in selected if entry.spill_failed), waited=bool(wait and selected),
                         timed_out=timed_out, seconds=seconds, gpu_bytes_after=self._device_bytes)
            self._pressure_events.append(event)
            self._event("pressure", where=where, freed_bytes=freed, seconds=seconds, deficit=deficit)
            return event

    def annotate_pressure_event(self, event: dict, **fields) -> None:
        """Add fields to a recorded pressure event (e.g. the section's archival deferral), under the tier's lock."""
        with self._cond:
            event.update(fields)
            if fields.get("deferred_archival"):
                self.pressure_["deferred_sections"] = self.pressure_.get("deferred_sections", 0) + 1
                self.pressure_["archival_deferred_seconds"] = (self.pressure_.get("archival_deferred_seconds", 0.0)
                                                               + float(fields.get("archival_deferred_seconds") or 0))

    def record_recheck_deferrals(self, where: str, events: Sequence[Mapping[str, Any]]) -> None:
        """A watched section's fitter re-check deferrals (dev22j4), kept with the pressure events. Each event has
        ``deferred_archival``, ``reason: "fitter_recheck"``, the node, its bytes and the wait. The section counts as
        one deferred section."""
        if not events:
            return
        with self._cond:
            stats = self.pressure_
            stats["fitter_recheck_deferrals"] += len(events)
            stats["deferred_sections"] += 1
            for event in events:
                seconds = float(event.get("waited_seconds") or 0.0)
                stats["archival_deferred_seconds"] += seconds
                self._pressure_events.append({**event, "event": "pressure", "where": where, "time": time.time(),
                                              "pressure": True, "deferred_archival": True,
                                              "reason": "fitter_recheck", "archival_deferred_seconds": seconds})

    def record_pressure_event(self, event: dict) -> None:
        """Record a pressure event handled outside the tier (trims without GPU raw copies, e.g. at request start)."""
        with self._cond:
            stats = self.pressure_
            stats["checks"] += 1
            stats["by_where"][event["where"]] = stats["by_where"].get(event["where"], 0) + 1
            if event.get("pressure"):
                stats["pressured"] += 1
            trims = event.get("trims") or {}
            if trims:
                stats["trim_calls"] += 1
                stats["trimmed_bytes"] += int(trims.get("bytes", 0))
                for record in trims.get("trims", []):
                    stats["trimmed_by_component"][record["name"]] = (
                        stats["trimmed_by_component"].get(record["name"], 0) + int(record.get("bytes", 0)))
            self._pressure_events.append(event)

    def section_begin(self) -> dict | None:
        """Start of a memory-critical section (the VAE decode): the allocator's allocated and peak bytes, so
        :meth:`section_end` can learn the section's working set (no peak reset: other tools read the peak), and the
        GPU raw bytes this tier has dropped so far (dev22j8v5). The caller marks after any wait for a node (whose
        freed memory would otherwise hide the decode's own need)."""
        with self._cond:
            device = self._gpu_device() or self._device_hint
            released = self._device_released_bytes
        try:
            memory = self._memory(device)
        except Exception:  # noqa: BLE001 - recording only
            return None
        if memory is None:
            return None
        with self._cond:
            self._open_sections += 1
        return {"device": device, "allocated": memory["allocated"], "peak": memory.get("peak", 0),
                "released": released}

    def section_end(self, token: dict | None, *, clean: bool = False) -> int | None:
        """End of the section: when it set a new allocator peak, its working set is that peak minus the allocation
        at entry, plus the GPU raw bytes this tier released meanwhile before that peak (dev22j8v5: they lowered the
        peak, the decode still needed them); the largest observed working set raises the auto reserve. ``clean`` (no node ran unpaused
        through the section) also raises the clean working set, the decode floor's learned value
        (:meth:`learned_clean_working_set_bytes`). Returns the measured working set."""
        if token is None:
            return None
        try:
            memory = self._memory(token["device"])
        except Exception:  # noqa: BLE001 - recording only
            memory = None
        with self._cond:
            self._open_sections = max(0, self._open_sections - 1)
            peak = 0 if memory is None else memory.get("peak", 0)
            baseline = int(token.get("released", self._device_released_bytes))
            # a release lowered the section's peak only when the peak came after it (the allocator's peak then was
            # below the final one); a release after the peak did not (unknown: added back)
            released = sum(nbytes for cumulative, nbytes, at in self._section_drops
                           if cumulative > baseline and (at is None or at < peak))
            if not self._open_sections:
                self._section_drops.clear()
        if memory is None:
            return None
        working = peak - token["allocated"] + released if peak > token["peak"] else None
        with self._cond:
            self.pressure_["sections"] += 1
            if working is not None and working > self._working_set:
                self._working_set = int(working)
            if clean and working is not None and working > self._clean_working_set:
                self._clean_working_set = int(working)
        return working

    def _peak_now(self) -> int | None:
        """The allocator's peak allocation now (a drop inside an open section; dev22j8v5), None when unknown."""
        device = self._gpu_device() or self._device_hint
        try:
            memory = self._memory(device) if device is not None else None
        except Exception:  # noqa: BLE001 - unknown
            return None
        return None if memory is None else int(memory.get("peak", 0))

    def learned_clean_working_set_bytes(self) -> int:
        """The largest working set of a clean memory-critical section (0: none yet; dev22j8v5)."""
        with self._cond:
            return int(self._clean_working_set)

    def _request_pressure_check(self) -> None:
        """Layer 2 (caller holds the condition): a background admission check on the worker (no device query and no
        wait on the producer's path)."""
        if self.pressure_enabled and not self._pressure_queued and self._device_bytes:
            self._pressure_queued = True
            self._jobs.append(("pressure", "admission"))
            self._ensure_worker()

    def flush(self, timeout: float | None = None) -> None:
        """Wait until no spill, buffer return or callback is queued or running (tests, harnesses). Every worker job
        notifies the condition when it ends, so this returns as soon as the tier is idle."""
        with self._cond:
            self._wait(lambda: not self._jobs and not self._active and not self._promote_jobs
                       and not self._promote_active
                       and not any(entry.host_state in ("queued", "copying", "promoting")
                                   for entry in self._entries.values()),
                       timeout, "raw tier flush")

    def reset_peaks(self) -> None:
        with self._cond:
            self.stats_["gpu_peak_bytes"] = self._device_bytes
            self.stats_["host_peak_bytes"] = self._host_bytes
            self.stats_["unpublished_peak_bytes"] = self._unpublished_bytes

    def disable(self, reason: str) -> None:
        with self._cond:
            self._disable(reason)
            self.stats_["bookkeeping_errors"] += 1 if reason.startswith("bookkeeping") else 0
            for entry in list(self._entries.values()):
                if not entry.unpublished:
                    self._request_drop(entry, "drops_released")
            self._cond.notify_all()

    def close(self) -> None:
        with self._cond:
            if self._promote_jobs:  # W1.4.1 (C2): no promotion runs on past close (rows released, copies done)
                self._cancel_promotions(lambda name: True, "cancelled")
            self._closed = True
            for entry in list(self._entries.values()):
                entry.pins.clear()
                self._request_drop(entry, "drops_released")
            self._cond.notify_all()
        for worker in (self._worker, self._promote_worker):
            if worker is not None and worker is not threading.current_thread():
                worker.join(60)
        self._buffers.shrink(0, reason="close")

    # -- records ---------------------------------------------------------------------------------------------------
    def stats(self) -> dict[str, Any]:
        with self._cond:
            entries = [entry for entry in self._entries.values() if not entry.dropped]
            gpu_budget, gpu_source = self.gpu_budget(), (
                "config" if self._gpu_budget is not None else f"auto:{self.gpu_budget_chunks}x largest entry")
            return {
                "enabled": self.enabled, "disabled_reason": self.disabled_reason,
                # Resolved budgets (D2): numbers, also when configured "auto" (None); the GPU auto budget resolves
                # with the first entry (0 before it).
                "gpu_raw_budget_bytes": gpu_budget, "host_raw_budget_bytes": self._host_budget,
                "budgets": {"gpu_bytes": gpu_budget, "host_bytes": self._host_budget,
                            "gpu_raw_budget_bytes": gpu_budget, "host_raw_budget_bytes": self._host_budget,
                            "gpu_budget_chunks": self.gpu_budget_chunks,
                            "gpu_resolved": self._gpu_budget is not None or self._max_entry > 0,
                            "keep_free_bytes": self._keep_free(),
                            "gpu_source": gpu_source, "host_source": dict(self.host_derivation)},
                "gpu_bytes": self._device_bytes, "host_bytes": self._host_bytes,
                # R5: the host-block pool (live copies + cached idle blocks), capped at the host budget
                "host_pool_bytes": self._buffers.live_bytes + self._buffers.cached_bytes,
                "host_pool_peak_bytes": self._buffers.stats["peak_bytes"],
                "unpublished_bytes": self._unpublished_bytes, "reserved_bytes": self._reserved,
                "unpublished_entries": self._unpublished_count, "largest_entry_bytes": self._max_entry,
                "lag_estimate_seconds": self.lag_estimate(),
                "entries": {"total": len(entries),
                            "device_only": sum(1 for e in entries if e.on_device and e.host_state != "complete"),
                            "host_only": sum(1 for e in entries if not e.on_device and e.host_state == "complete"),
                            "both": sum(1 for e in entries if e.on_device and e.host_state == "complete")},
                **{key: (dict(value) if isinstance(value, dict) else value) for key, value in self.stats_.items()},
                "host_buffers": self._buffers.snapshot(),
                "pressure": {**{key: (dict(value) if isinstance(value, dict) else value)
                                for key, value in self.pressure_.items()},
                             "enabled": self.pressure_enabled,
                             "reserve_bytes": self.reserve_bytes(),
                             "reserve_source": "config" if self._reserve_config is not None else "auto",
                             "working_set_bytes": self._working_set,
                             "clean_working_set_bytes": self._clean_working_set,
                             "device_released_bytes": self._device_released_bytes,
                             "events": [dict(item) for item in self._pressure_events]},
                "last_resort_events": [dict(item) for item in self._events if item["event"] == "backpressure"
                                       and item["cause"] != "gpu_spill"],
                "events": [dict(item) for item in self._events],
                **(self._retain_stats(entries) if self.retain != "drop" or self.promotion_mode != "off" else {}),
                **({"gpu_only": True} if self.gpu_only else {}),
            }

    def _retain_stats(self, entries: Sequence[RawEntry]) -> dict[str, Any]:
        """wl-w1b7 records (caller holds the condition): unheld entries and bytes per tier, re-attachments (hits),
        restore hits on unheld entries, evictions by holder state, and the restore promotions."""
        unheld = [entry for entry in entries if self._is_unheld(entry)]
        promoted = [entry for entry in entries if entry.origin != "captured"]
        return {
            "retain": {"mode": self.retain,
                       "unheld": {"entries": {"gpu": sum(1 for e in unheld if e.on_device),
                                              "host": sum(1 for e in unheld if e.host_state == "complete")},
                                  "bytes": {"gpu": sum(e.nbytes for e in unheld if e.on_device),
                                            "host": sum(e.host_nbytes for e in unheld if e.host_state == "complete")}},
                       **{key: (dict(value) if isinstance(value, dict) else value)
                          for key, value in self.retain_.items()}},
            "promotion": {"mode": self.promotion_mode,
                          "live": {"entries": len(promoted), "bytes": sum(e.host_nbytes for e in promoted
                                                                          if e.host_state == "complete")},
                          "jobs": len(self._promote_jobs), "active": self._promote_active,
                          "transient_bytes": self._transient, "transient_cap_bytes": self.promotion_cap(),
                          **{key: (dict(value) if isinstance(value, dict) else value)
                             for key, value in self.promotion_.items()}},
        }

    def _event(self, name: str, **fields) -> None:
        self._events.append({"event": name, "time": time.time(), **fields})

    def _peaks(self) -> None:
        stats = self.stats_
        stats["gpu_peak_bytes"] = max(stats["gpu_peak_bytes"], self._device_bytes)
        stats["host_peak_bytes"] = max(stats["host_peak_bytes"], self._host_bytes + self._host_reserved_spill)
        stats["unpublished_peak_bytes"] = max(stats["unpublished_peak_bytes"], self._unpublished_bytes)

    # -- internals (caller holds the condition) -------------------------------------------------------------------
    def _wait(self, done: Callable[[], bool], timeout: float | None, what: str) -> None:
        deadline = None if timeout is None else time.monotonic() + timeout
        while not done():
            remaining = None if deadline is None else deadline - time.monotonic()
            if remaining is not None and remaining <= 0:
                raise CapacityError(f"{what} did not finish within {timeout} s")
            self._cond.wait(remaining if remaining is not None else 0.5)

    def _touch(self, entry: RawEntry) -> None:
        self._clock += 1
        entry.last_used = self._clock

    def _pin(self, entry: RawEntry, reason: str) -> None:
        entry.pins[reason] = entry.pins.get(reason, 0) + 1

    def _unpin(self, entry: RawEntry, reason: str) -> None:
        with self._cond:
            count = entry.pins.get(reason, 0) - 1
            if count > 0:
                entry.pins[reason] = count
            else:
                entry.pins.pop(reason, None)
            if entry.drop_requested and not entry.pinned:
                self._drop_now(entry)
            self._balance()
            self._cond.notify_all()

    def _keep_free(self) -> int:
        """Cached pool bytes an idle session release keeps (``keep_free_bytes``, default the host budget)."""
        if self.keep_free_bytes is not None:
            return self.keep_free_bytes
        return self._host_budget

    def _idle(self) -> bool:
        return (not self._entries and not self._jobs and not self._active and not self._reserved
                and not self._promote_jobs and not self._promote_active)  # W1.4.1 (C2): promotion work too

    def _request_drop(self, entry: RawEntry, counter: str) -> None:
        if entry.dropped:
            return
        if not entry.drop_requested:
            entry.drop_requested = True
            self.stats_[counter] += 1
        if not entry.pinned and entry.host_state != "copying":
            self._drop_now(entry)

    def _drop_now(self, entry: RawEntry, *, prepare: bool = False) -> None:
        if entry.dropped:
            return
        self._drop_device(entry)
        self._drop_host(entry, prepare=prepare)
        entry.dropped = True
        self._entries.pop(entry.token, None)
        if entry.node_ref is not None and self._by_ref.get(entry.node_ref) is entry:
            del self._by_ref[entry.node_ref]
        if entry.unpublished:
            self._unpublished_bytes -= entry.nbytes
            self._unpublished_count -= 1
            entry.state = "cancelled"
        self._cond.notify_all()

    def _drop_device(self, entry: RawEntry) -> None:
        if not entry.on_device:
            return
        for group in entry.groups:
            group.device = None
        entry.on_device = False
        self._device_bytes -= entry.nbytes
        self._device_released_bytes += entry.nbytes  # dev22j8v5: added back by a section running meanwhile
        if self._open_sections:
            self._section_drops.append((self._device_released_bytes, entry.nbytes, self._peak_now()))
        if not entry.device_owned and entry.release is not None:
            release, entry.release = entry.release, None
            event = entry.read_event
            try:
                if event is not None and event.query():
                    event = None
            except Exception:  # noqa: BLE001 - a CPU stand-in event
                pass
            release(event)
        entry.device_owned = True

    def _drop_host(self, entry: RawEntry, *, prepare: bool) -> None:
        if entry.host_state != "complete":
            return
        blocks, event = entry.host_blocks, entry.host_read_event
        for group in entry.groups:
            group.host = None
        entry.host_blocks, entry.host_state = None, "none"
        self._host_bytes -= entry.host_nbytes
        if blocks:
            # The blocks go back to the pool (never unpinned here: that holds the CUDA driver lock); after pending
            # uploads from them, waited for on the worker.
            if event is None:
                self._buffers.release(blocks)
            else:
                self._jobs.append(("release", (blocks, event)))
                self._ensure_worker()
        entry.host_read_event = None
        if prepare and entry.node_ref is not None and self.on_host_drop is not None:
            self._jobs.append(("prepare", entry.node_ref))
            self.stats_["prepare_on_drop"]["queued"] += 1
            self._ensure_worker()

    def _host_room(self, nbytes: int) -> bool:
        """Make room for ``nbytes`` of host copies by dropping least recently used published host copies (wl-w1b7:
        with ``retain="lru"`` unheld ones first)."""
        budget = self._host_budget
        if self._host_bytes + self._host_reserved_spill + nbytes <= budget:
            return True
        victims = sorted((entry for entry in self._entries.values()
                          if entry.state == "published" and entry.host_state == "complete" and not entry.pinned
                          and not entry.dropped and entry.placed != "host_raw"),
                         key=self._lru_key)
        for entry in victims:
            if self._host_bytes + self._host_reserved_spill + nbytes <= budget:
                break
            self.stats_["evictions_host"] += 1
            self._evicted("host", entry)
            if entry.on_device:
                self._drop_host(entry, prepare=False)
            else:
                self._drop_now(entry, prepare=True)
        return self._host_bytes + self._host_reserved_spill + nbytes <= budget

    def _queue_spill(self, entry: RawEntry) -> bool:
        if entry.host_state != "none" or entry.spill_failed or entry.dropped:
            return entry.host_state in ("queued", "copying")
        if not self._host_room(entry.host_nbytes):
            return False
        entry.host_state = "queued"
        self._host_reserved_spill += entry.host_nbytes
        self._jobs.append(("spill", entry))
        self._ensure_worker()
        return True

    def _balance(self) -> None:
        """Enforce the budgets: demote device copies beyond the GPU budget (fitted and published entries least
        recently used first, then queued ones newest first), drop host copies of published entries beyond the host
        budget. Never drops the only copy of an unpublished entry (I1)."""
        if self._closed:
            return
        for entry in list(self._entries.values()):
            if entry.force_host and entry.on_device and not entry.dropped:
                if entry.host_state == "complete":
                    if not entry.pinned:
                        self._drop_device(entry)
                elif (self.gpu_only and entry.state == "published" and not entry.pinned
                      and entry.host_state == "none" and entry.placed != "host_raw"):
                    self.stats_["evictions_gpu"] += 1  # W1.4.1 GPU raw only: the demotion deletes the raw copy
                    self._evicted("gpu", entry)
                    self._drop_now(entry, prepare=True)
                elif not self._queue_spill(entry) and entry.host_state == "none" and not entry.spill_failed:
                    entry.spill_failed = True  # the host budget cannot take it: the demotion reports it
                    self._event("demotion_blocked", token=entry.token, reason="host raw budget full")
        excess = self._device_bytes - self.gpu_budget()
        if excess > 0:
            device = [entry for entry in self._entries.values() if entry.on_device and not entry.dropped
                      and entry.placed != "gpu_raw" and entry.pins.get("plan", 0) == 0]
            first = sorted((entry for entry in device if entry.state in ("fitted", "published")),
                           key=self._lru_key)
            later = sorted((entry for entry in device if entry.state in ("submitted", "planning")),
                           key=lambda entry: -entry.token)
            for entry in first + later:
                if excess <= 0:
                    break
                if entry.host_state == "complete":
                    if not entry.pinned:
                        self._drop_device(entry)
                        self.stats_["evictions_gpu"] += 1
                        self._evicted("gpu", entry)
                        excess -= entry.nbytes
                elif entry.host_state in ("queued", "copying"):
                    excess -= entry.nbytes
                elif entry.state == "published" and not entry.pinned and not self._host_room(entry.host_nbytes):
                    self.stats_["evictions_gpu"] += 1  # no host room: a published copy may go (I3)
                    self._evicted("gpu", entry)
                    self._drop_now(entry, prepare=True)
                    excess -= entry.nbytes
                elif self._queue_spill(entry):
                    excess -= entry.nbytes
        if self._host_bytes > self._host_budget:
            self._host_room(0)
        self._peaks()

    # -- worker ------------------------------------------------------------------------------------------------------
    def _ensure_worker(self) -> None:
        if self._worker is None or not self._worker.is_alive():
            self._worker = threading.Thread(target=self._work, name="reelcache-raw-tier", daemon=True)
            self._worker.start()
        self._cond.notify_all()

    def _stream(self, device: torch.device):
        key = str(device)
        stream = self._streams.get(key)
        if stream is None:
            stream = self._streams[key] = torch.cuda.Stream(device)
        return stream

    def _work(self) -> None:
        while True:
            with self._cond:
                while not self._jobs and not self._closed:
                    self._cond.wait()
                if not self._jobs:
                    return
                kind, item = self._jobs.popleft()
                if kind == "spill":
                    entry = item
                    if entry.dropped or entry.drop_requested or entry.host_state != "queued" or not entry.on_device:
                        if entry.host_state == "queued":
                            entry.host_state = "none"
                            self._host_reserved_spill -= entry.host_nbytes
                        self._balance()
                        self._cond.notify_all()
                        entry = item = None
                        continue
                    entry.host_state = "copying"
                    sources = [(group, key, group.device[key], meta) for group in entry.groups
                               for key, meta in group.meta.items()]
                    self._pin(entry, "spill")
                self._active += 1
            try:
                if kind == "release":
                    blocks, event = item
                    try:
                        if event is not None:
                            event.synchronize()
                    finally:
                        self._buffers.release(blocks)
                elif kind == "pressure":
                    with self._cond:
                        self._pressure_queued = False
                        device = self._gpu_device()
                    try:
                        memory = self._memory(device)
                        if memory is not None:
                            self._relieve(item, memory, wait=False, timeout=None, everything=False)
                    except Exception as exc:  # noqa: BLE001 - recorded: the check is an optimization
                        with self._cond:
                            self.pressure_["errors"] += 1
                            self._event("pressure_error", where=item, error=f"{type(exc).__name__}: {exc}")
                elif kind == "prepare":
                    try:
                        self.on_host_drop(item)
                        outcome = "done"
                    except Exception:  # noqa: BLE001 - best effort: a later restore prepares or reads cold
                        outcome = "errors"
                    with self._cond:
                        self.stats_["prepare_on_drop"][outcome] += 1
                else:
                    self._spill(entry, sources)
            except Exception as exc:  # noqa: BLE001 - e.g. a failed event wait: recorded, the worker keeps serving
                with self._cond:
                    self.stats_["worker_errors"] += 1
                    self._event("worker_error", job=kind, error=f"{type(exc).__name__}: {exc}")
            finally:
                # The job's locals hold its tensors (a spill's ``sources`` are the device copies it read): drop them
                # now. Before F they lived on in this idle frame until the next spill, so the last spilled chunk's
                # device copies (CausVid: 60 x [4680, 1536] BF16, 0.8 GiB) outlived their drop (5b's carry-over OOM).
                entry = sources = item = None
                # Every job ends with a notify: flush() and other waiters re-check at once (D2: a release job used to
                # end silently, so flush(timeout) slept its whole timeout).
                with self._cond:
                    self._active -= 1
                    self._cond.notify_all()

    def _spill(self, entry: RawEntry, sources) -> None:
        started = time.perf_counter()
        blocks, error, stream = None, None, None
        try:
            # One pooled block per matrix (reused when cached: no registration, no driver lock).
            blocks = self._buffers.acquire([meta[5] for _, _, _, meta in sources], entry.cuda and self.pinned)
            views = {}
            device = next((meta[0] for _, _, _, meta in sources if meta[0].type == "cuda"), None)
            if device is not None:
                stream = self._stream(device)
                with torch.cuda.device(device), torch.cuda.stream(stream):
                    if entry.ready_event is not None:
                        stream.wait_event(entry.ready_event)
                    for (group, key, source, (_, dtype, shape, _, _, nbytes)), block in zip(sources, blocks):
                        view = block.tensor[:nbytes].view(dtype).view(shape)
                        view.copy_(source, non_blocking=block.pinned)
                        if source.device.type == "cuda":
                            source.record_stream(stream)
                        views[(id(group), key)] = view
                    event = torch.cuda.Event()
                    event.record(stream)
                event.synchronize()
            else:
                for (group, key, source, (_, dtype, shape, _, _, nbytes)), block in zip(sources, blocks):
                    view = block.tensor[:nbytes].view(dtype).view(shape)
                    view.copy_(source)
                    views[(id(group), key)] = view
        except BaseException as exc:  # noqa: BLE001 - recorded; the device copy stays (I1)
            error = f"{type(exc).__name__}: {exc}"  # text only: the exception's traceback would pin this frame
            if stream is not None and blocks is not None:
                try:  # no copy into the blocks may still run when they return to the pool
                    stream.synchronize()
                except Exception:  # noqa: BLE001 - a broken context: the blocks are never reused safely anyway
                    pass
        # This job's references to the device copies go before the completion below. The entry's own references go
        # in _drop_device under the lock, so a waiter woken by its notify finds the memory freed. A synchronous
        # demotion measures the headroom and empties the CUDA cache next. F gpucheck 4422935: woken first, it
        # counted 46-48 of a chunk's 60 copies still allocated; the decode's first kernel load then hit a driver OOM.
        source = None
        sources.clear()
        with self._cond:
            self._host_reserved_spill -= entry.host_nbytes
            entry.pins["spill"] = entry.pins.get("spill", 1) - 1
            if not entry.pins["spill"]:
                del entry.pins["spill"]
            if error is not None or entry.dropped:
                entry.host_state = "none"
                if error is not None:
                    entry.spill_failed = True
                    self.stats_["spill_failures"] += 1
                    self._event("spill_failure", token=entry.token, error=error)
                if blocks is not None:
                    self._buffers.release(blocks)
            else:
                for group in entry.groups:
                    group.host = {key: views[(id(group), key)] for key in group.meta}
                entry.host_blocks, entry.host_state = blocks, "complete"
                self._host_bytes += entry.host_nbytes
                self.stats_["spills"] += 1
                self.stats_["spill_bytes"] += entry.nbytes
                self.stats_["spill_seconds"] += time.perf_counter() - started
            if entry.drop_requested and not entry.pinned:
                self._drop_now(entry)
            self._balance()
            self._cond.notify_all()

__all__ = ["ARCHIVAL_STATES", "RawChunkTier", "RawEntry", "RawLease", "TIER_NAMES", "auto_host_budget",
           "cgroup_memory", "local_workers", "mem_available"]
