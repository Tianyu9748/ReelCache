"""Decoded-coefficient tier (accel A1, 2026-10-30; plan.md §12.6): normative decoded pictures of hot node packs.

A restore of a long history is bounded by NVDEC (~0.9 GB/s of bitstream on an RTX PRO 6000): beyond a few chunks the
decoder falls behind the per-layer compute of the forward that consumes the restored KV, and the forward waits. This
tier keeps, for node packs whose bytes were decoded anyway, the HEVC decoder's output -- the uint8 pictures of every
tile, cropped to the tile's rectangle (the only pixels the restore kernels read) -- in pinned host memory, and lets a
restore route some nodes past NVDEC: their jobs' canvases are filled by host-to-device copies (copy engine, ~25 GB/s)
and then reconstructed by the unchanged restore kernels. HEVC decoding is normative (every conforming decoder outputs
the same pictures), so a routed node restores bitwise the same KV as its NVDEC decode; the stored bytes, the storage
format and the codec are unchanged.

* **Fill.** :func:`offer_decoded` is called by ``node_packets.encode_node_pack`` with the archival validation decode
  (``validate_decode``: FFmpeg, already computed; no extra decode work), keyed by the SHA-256 of the pack's HEVC stream
  (its payload descriptor digest, the content that determines the pictures). Every open tier whose model matches takes
  a copy of the tiles' rectangles (budget-bounded LRU; a node larger than the budget is skipped).
* **Binding.** A restore names packs by their packet reference: :meth:`DecodedTier.bind` maps a packet reference to
  its stream digest when the restorer adopts (seal time) or loads the pack.
* **Routing.** :meth:`DecodedTier.route` runs per restore batch. ``"deficit"`` routes the OLDEST history nodes whose
  pictures are cached until the predicted NVDEC time per layer (bitstream bytes / ``nvdec_gbps``) is at most
  ``target`` x the per-layer compute ``c(h) = a + b * min(h, window)`` of the consuming forward (h = history nodes of
  the call); ``"all"`` routes every cached node. Routed entries are leased until :meth:`release`.
* **Copies.** One worker thread issues each routed job's copies in submission (layer) order on the tier's own CUDA
  stream into the job's arena canvas, records an event and finishes the job's ticket (the decoder pool's contract:
  reconstruction waits for the event). Host buffers are pinned with ``cudaHostRegister`` (exact size); an evicted
  entry is freed only after its leases are released and its last copy completed.
* **Invalidation.** :meth:`forget_packets` drops the entries of packs the store evicted or collected.

Defaults: no tier exists unless ``decoded_tier_bytes > 0`` (``ReelCacheRuntimeConfig``); with no open tier
:func:`offer_decoded` returns at once and every restore takes the base path.
"""

from __future__ import annotations

import bisect
import mmap
import queue
import threading
import time
import weakref
from collections import OrderedDict
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

from .errors import ConfigurationError, IntegrityError, StateError

ROUTES = ("deficit", "all")
#: Per-layer compute of the first denoise forward, c(h) = a + b * min(h, window) [ms], by bridge model name (RTX PRO
#: 6000, evaluation recompute-arm medians; benchmarks/accel_20261030/analysis/a1_sim_inputs.json). Unknown models: the
#: CausVid entry.
LAYER_COMPUTE_MS = {"causvid": (2.58, 0.486, None), "raven": (3.23, 0.524, None), "longlive1": (4.04, 0.0, 3),
                    "longlive2": (10.91, 0.0, 3), "lingbot": (18.43, 1.188, None)}
#: Effective aggregate NVDEC rate of a prepared restore (8 workers; prepared_per_chunk.csv: 0.89-1.0 GB/s).
NVDEC_GBPS = 0.9
#: Route until the predicted NVDEC time per layer is at most this fraction of c(h) (headroom for NVDEC slowing down
#: next to the forward).
TARGET = 0.85
#: Bytes allocated (and freed) once on the copy stream when the tier is created: the stream's caching-allocator pool
#: then holds a block for the copies' staging temporaries, so no restore's first copy waits in cudaMalloc (which can
#: wait for the device: 0.25 s behind a busy GPU, measured on an L40S).
WARM_BYTES = 32 << 20

_PAGE = mmap.PAGESIZE
#: Sub-allocation unit of the tier's slab (bytes).
_ALIGN = 4096
_TIERS: "weakref.WeakSet[DecodedTier]" = weakref.WeakSet()
_TIERS_LOCK = threading.Lock()


def offer_decoded(key: str, pictures: Any, tiles: Sequence[Mapping[str, Any]], *, model: Any = None) -> int:
    """Hand one pack's normative decode to every open tier (``node_packets.encode_node_pack``'s hook).

    ``key``: the SHA-256 hex digest of the pack's HEVC stream; ``pictures``: the decoded canvas ``[P, planes, H, W]``
    uint8 (numpy or torch); ``tiles``: the pack's tile records (``pictures`` = [first, count], ``rect`` = [x, y, w, h]);
    ``model``: the node's ``ModelIdentity`` (a tier bound to another model ignores the offer). Never raises (a failed
    fill is counted by the tier); returns the number of tiers that kept it.
    """
    if not _TIERS:
        return 0
    with _TIERS_LOCK:
        tiers = list(_TIERS)
    kept = 0
    for tier in tiers:
        kept += bool(tier.offer(key, pictures, tiles, model=model))
    return kept


def open_tiers() -> list["DecodedTier"]:
    with _TIERS_LOCK:
        return list(_TIERS)


def layer_compute_model(model_name: str | None, layer_ms: Any = None) -> tuple[float, float, int | None, str]:
    """``(a, b, window, source)`` of c(h) for a model (``layer_ms``: an explicit ``[a, b]`` override)."""
    if layer_ms is not None:
        a, b = (float(value) for value in layer_ms)
        return a, b, None, "configured"
    name = (model_name or "").lower()
    for known in sorted(LAYER_COMPUTE_MS, key=len, reverse=True):
        if known in name:
            a, b, window = LAYER_COMPUTE_MS[known]
            return a, b, window, f"default:{known}"
    a, b, window = LAYER_COMPUTE_MS["causvid"]
    return a, b, window, "default:causvid(unknown model)"


class _Slab:
    """The tier's host memory: ONE buffer, registered (pinned) once when the tier is created, sub-allocated first fit
    in ``_ALIGN`` units with coalescing frees (caller holds the tier lock). A fill is then a plain memcpy: A_v3
    registered every entry, and each fill's cudaHostRegister / cudaHostUnregister slowed the concurrent archive
    fitter by ~10 % (benchmarks/accel_20261030 PROGRESS_A.md)."""

    def __init__(self, nbytes: int, pinned: bool):
        self.size = max(_ALIGN, nbytes // _ALIGN * _ALIGN)  # never more than the budget (one unit at least)
        raw = torch.empty(self.size + 2 * _PAGE, dtype=torch.uint8)
        start = (-raw.data_ptr()) % _PAGE
        self.buffer = raw[start:start + self.size]
        self.registered = False
        if pinned:  # cudaHostRegisterPortable (1): pinned for every context
            self.registered = int(torch.cuda.cudart().cudaHostRegister(self.buffer.data_ptr(), self.size, 1)) == 0
        self._free = [(0, self.size)]  # sorted (offset, size) holes

    @staticmethod
    def _units(nbytes: int) -> int:
        return -(-max(nbytes, 1) // _ALIGN) * _ALIGN

    def allocate(self, nbytes: int) -> int | None:
        need = self._units(nbytes)
        for position, (offset, size) in enumerate(self._free):
            if size >= need:
                if size == need:
                    del self._free[position]
                else:
                    self._free[position] = (offset + need, size - need)
                return offset
        return None

    def free(self, offset: int, nbytes: int) -> None:
        need = self._units(nbytes)
        position = bisect.bisect_left(self._free, (offset, 0))
        self._free.insert(position, (offset, need))
        if position + 1 < len(self._free) and offset + need == self._free[position + 1][0]:
            self._free[position] = (offset, need + self._free.pop(position + 1)[1])
        if position > 0 and sum(self._free[position - 1]) == offset:
            previous = self._free[position - 1]
            self._free[position - 1] = (previous[0], previous[1] + self._free.pop(position)[1])

    def free_bytes(self) -> int:
        return sum(size for _, size in self._free)

    def close(self) -> None:
        if self.registered:
            torch.cuda.cudart().cudaHostUnregister(self.buffer.data_ptr())
            self.registered = False


class _Entry:
    __slots__ = ("key", "buffer", "offset", "tiles", "geometry", "nbytes", "pins", "event", "packets")

    def __init__(self, key, buffer, offset, tiles, geometry, nbytes):
        self.key, self.buffer, self.offset = key, buffer, offset
        self.tiles, self.geometry, self.nbytes = tiles, geometry, nbytes
        self.pins = 0
        self.event = None
        self.packets: set[str] = set()

    def view(self, index: int) -> torch.Tensor:
        offset, count, planes, height, width = self.tiles[index][:5]
        return self.buffer[offset:offset + count * planes * height * width].view(count, planes, height, width)

    def covers(self, packet, tiles) -> bool:
        """Whether this entry holds every tile of ``tiles`` of ``packet`` (a PreparedPacket) with its placement."""
        if self.geometry != (packet.pictures, packet.planes, packet.height, packet.width):
            return False
        for tile in tiles:
            found = self.tiles.get(tile.index)
            x, y, w, h = tile.rect
            if found is None or found[1:] != (tile.pictures[1], packet.planes, h, w, tile.pictures[0], x, y):
                return False
        return True


class _Route:
    """A routed job's source: :meth:`submit` replaces ``DecoderPool.submit`` for that job (``packed_decode``)."""

    __slots__ = ("tier", "entry", "record")

    def __init__(self, tier, entry, record):
        self.tier, self.entry, self.record = tier, entry, record

    def submit(self, job):
        return self.tier._submit(job, self.entry, self.record)


class DecodedTier:
    """Pinned-host LRU of normative decoded tile pictures and the restore routing over it (see the module docstring).

    ``budget_bytes``: host bytes of cached pictures (one slab, registered once). ``device``: the restore device
    (CUDA: pinned slab, a private copy stream; CPU: plain slab, synchronous copies -- tests).
    ``route``: ``"deficit"`` or ``"all"``. ``model``: the ``ModelIdentity`` whose packs are kept (None: any) and whose
    name selects the default c(h) (``layer_ms`` overrides it with ``[a, b]`` ms). ``pinned``: default CUDA devices.
    """

    def __init__(self, budget_bytes: int, *, device: Any = "cpu", route: str = "deficit", model: Any = None,
                 layer_ms: Any = None, pinned: bool | None = None, nvdec_gbps: float = NVDEC_GBPS,
                 target: float = TARGET):
        if type(budget_bytes) is not int or budget_bytes <= 0:
            raise ConfigurationError("a decoded tier needs a positive byte budget")
        if route not in ROUTES:
            raise ConfigurationError(f"decoded tier route must be one of {ROUTES}")
        if not (isinstance(nvdec_gbps, (int, float)) and nvdec_gbps > 0 and 0 < target <= 4):
            raise ConfigurationError("nvdec_gbps must be positive and target in (0, 4]")
        self.device = torch.device(device)
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.budget_bytes, self.route_policy = budget_bytes, route
        self.model = model
        self.nvdec_gbps, self.target = float(nvdec_gbps), float(target)
        self.layer_model = layer_compute_model(getattr(model, "name", None), layer_ms)
        self.pinned = self.device.type == "cuda" if pinned is None else bool(pinned)
        self._entries: OrderedDict[str, _Entry] = OrderedDict()
        self._bound: dict[str, str] = {}
        self._retiring: list[_Entry] = []
        self._lock = threading.Lock()
        self._resident = 0
        self._slab = _Slab(budget_bytes, self.pinned)  # pinned once, here (bridge attach)
        self.stream = torch.cuda.Stream(device=self.device) if self.device.type == "cuda" else None
        self._jobs: queue.Queue = queue.Queue()
        self._worker: threading.Thread | None = None
        self._closed = False
        if self.stream is not None:  # off any restore's path: the copy stream's allocator block and the copy worker
            with torch.inference_mode(False), torch.cuda.stream(self.stream):
                torch.empty(WARM_BYTES, dtype=torch.uint8, device=self.device)
            self.stream.synchronize()
            self._worker = threading.Thread(target=self._run, name="reelcache-decoded-tier", daemon=True)
            self._worker.start()
        self.stats_ = {"offers": 0, "fills": 0, "fill_bytes": 0, "fill_seconds": 0.0, "fill_errors": 0,
                       "fill_skipped_large": 0, "fill_skipped_full": 0, "fill_skipped_model": 0, "fill_duplicates": 0,
                       "evictions": 0,
                       "evicted_bytes": 0, "forgotten": 0, "freed": 0, "bound": 0, "restores": 0, "routed_restores": 0,
                       "hits": 0, "misses": 0, "routed_nodes": 0, "routed_jobs": 0, "routed_bytes": 0,
                       "routed_bitstream_bytes": 0, "copy_issue_seconds": 0.0, "copy_errors": 0,
                       "peak_resident_bytes": 0, "last_fill_error": None, "slab_registered": self._slab.registered}
        with _TIERS_LOCK:
            _TIERS.add(self)

    # -- fill ---------------------------------------------------------------------------------------------------------
    def offer(self, key: str, pictures: Any, tiles: Sequence[Mapping[str, Any]], *, model: Any = None) -> bool:
        """Keep the tiles' rectangles of one decoded pack (see :func:`offer_decoded`); False if not kept."""
        started = time.perf_counter()
        try:
            with self._lock:
                if self._closed:
                    return False
                self.stats_["offers"] += 1
                if self.model is not None and model is not None and model != self.model:
                    self.stats_["fill_skipped_model"] += 1
                    return False
                if key in self._entries:
                    self._entries.move_to_end(key)
                    self.stats_["fill_duplicates"] += 1
                    return True
            if isinstance(pictures, torch.Tensor):
                pictures = pictures.detach().to("cpu").numpy()
            pictures = np.asarray(pictures)
            if pictures.dtype != np.uint8 or pictures.ndim != 4:
                raise IntegrityError("decoded pictures must be uint8 [pictures, planes, height, width]")
            count_all, planes, height, width = pictures.shape
            layout, total = {}, 0
            for index, tile in enumerate(tiles):
                first, count = (int(value) for value in tile["pictures"])
                x, y, w, h = (int(value) for value in tile["rect"])
                if first < 0 or first + count > count_all or x < 0 or y < 0 or x + w > width or y + h > height:
                    raise IntegrityError("a tile lies outside its decoded canvas")
                layout[index] = (total, count, planes, h, w, first, x, y)
                total += count * planes * h * w
            if total > self.budget_bytes:
                with self._lock:
                    self.stats_["fill_skipped_large"] += 1
                return False
            with self._lock:
                if self._closed:
                    return False
                start = self._reserve(total)
                if start is None:  # the older entries are leased by restores in flight (or still being copied)
                    self.stats_["fill_skipped_full"] += 1
                    return False
                buffer = self._slab.buffer[start:start + total]
            try:
                host = buffer.numpy()
                for offset, count, planes_, h, w, first, x, y in layout.values():
                    np.copyto(host[offset:offset + count * planes_ * h * w].reshape(count, planes_, h, w),
                              pictures[first:first + count, :, y:y + h, x:x + w])
            except BaseException:
                with self._lock:
                    self._slab.free(start, total)
                raise
            entry = _Entry(key, buffer, start, layout, (count_all, planes, height, width), total)
            with self._lock:
                if self._closed or key in self._entries:  # closed meanwhile, or a concurrent offer of the same pack
                    if not self._closed:
                        self._slab.free(start, total)
                    return not self._closed
                self._entries[key] = entry
                self._resident += total
                for ref, bound in self._bound.items():  # bound before the offer (never in practice): keep the link
                    if bound == key:
                        entry.packets.add(ref)
                self.stats_["fills"] += 1
                self.stats_["fill_bytes"] += total
                self.stats_["peak_resident_bytes"] = max(self.stats_["peak_resident_bytes"], self._resident)
                self.stats_["fill_seconds"] += time.perf_counter() - started
            return True
        except Exception as exc:  # noqa: BLE001 - a fill never fails archival; counted and recorded
            with self._lock:
                self.stats_["fill_errors"] += 1
                self.stats_["last_fill_error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
            return False

    def _reserve(self, nbytes: int) -> int | None:
        """A slab range for ``nbytes``, evicting least recently used, unleased entries until one fits (caller holds
        the lock); None if the leased (or still copying) entries leave no room."""
        self._sweep()
        start = self._slab.allocate(nbytes)
        for key in list(self._entries):
            if start is not None:
                break
            entry = self._entries[key]
            if entry.pins:
                continue
            self._evict(key)
            self.stats_["evictions"] += 1
            self.stats_["evicted_bytes"] += entry.nbytes
            self._sweep()
            start = self._slab.allocate(nbytes)
        return start

    def _evict(self, key: str) -> _Entry:
        entry = self._entries.pop(key)
        self._resident -= entry.nbytes
        for ref in entry.packets:
            if self._bound.get(ref) == key:
                del self._bound[ref]
        self._retiring.append(entry)
        return entry

    def _sweep(self) -> None:
        """Free evicted entries no restore leases and whose last copy completed (caller holds the lock)."""
        still = []
        for entry in self._retiring:
            if entry.pins or (entry.event is not None and not entry.event.query()):
                still.append(entry)
                continue
            if entry.buffer is not None and self._slab is not None:
                self._slab.free(entry.offset, entry.nbytes)
            entry.buffer = None
            self.stats_["freed"] += 1
        self._retiring = still

    # -- binding and invalidation -----------------------------------------------------------------------------------
    def bind(self, packet_ref: str, key: str | None) -> None:
        """``packet_ref``'s pack has the HEVC stream digest ``key`` (seal-time adoption or a store load)."""
        if not isinstance(key, str) or not key:
            return
        with self._lock:
            if self._bound.get(packet_ref) == key:
                return
            self._bound[packet_ref] = key
            entry = self._entries.get(key)
            if entry is not None:
                entry.packets.add(packet_ref)
            self.stats_["bound"] += 1

    def packet_refs(self) -> list[str]:
        with self._lock:
            return list(self._bound)

    def forget_packets(self, packet_refs: Iterable[str]) -> dict[str, int]:
        """Drop the entries of packs the store evicted or collected (and their bindings)."""
        dropped = released = 0
        with self._lock:
            for ref in dict.fromkeys(packet_refs):
                key = self._bound.pop(ref, None)
                if key is None:
                    continue
                entry = self._entries.get(key)
                if entry is None:
                    continue
                entry.packets.discard(ref)
                if entry.packets and any(self._bound.get(other) == key for other in entry.packets):
                    continue  # another live pack has the same stream
                self._evict(key)
                dropped += 1
                released += entry.nbytes
            self.stats_["forgotten"] += dropped
            self._sweep()
        return {"entries": dropped, "bytes": released}

    def clear(self) -> None:
        with self._lock:
            for key in list(self._entries):
                self._evict(key)
            self._bound.clear()
            self._sweep()

    # -- routing ------------------------------------------------------------------------------------------------------
    def layer_compute_ms(self, history: int) -> float:
        a, b, window, _ = self.layer_model
        return a + b * (history if window is None else min(history, window))

    def route(self, history: Sequence[Mapping[str, Any]], batch: Any) -> dict[str, Any] | None:
        """Route some nodes of one planned :class:`~reelcache.packed_decode.RestoreBatch` past NVDEC (before any job is
        submitted); returns the routing record (pass it to :meth:`release` once the batch closed) or None when the
        batch has no job."""
        jobs = list(batch.jobs)
        if not jobs:
            return None
        by_node: dict[str, list] = {}
        for job in jobs:
            if job.submitted:
                raise StateError("decoded-tier routing runs before any job of the batch is submitted")
            by_node.setdefault(job.node_id, []).append(job)
        items = {item["node_id"]: item for item in history}
        order = sorted(by_node, key=lambda node: (int(items.get(node, {}).get("token_start", 0)), node))
        layers = max(1, len(batch.layers))
        ms_per_byte = 1.0 / (self.nvdec_gbps * 1e6)
        node_bytes = {node: sum(job.nbytes for job in node_jobs) for node, node_jobs in by_node.items()}
        c_ms = self.layer_compute_ms(len(history))
        before = sum(node_bytes.values()) * ms_per_byte / layers
        per_layer, routed, hits, misses = before, [], 0, 0
        record = {"route": self.route_policy, "history": len(history), "nodes": len(by_node), "layers": layers,
                  "c_ms": c_ms, "c_model": self.layer_model[3], "target": self.target, "nvdec_gbps": self.nvdec_gbps,
                  "predicted_nvdec_ms_per_layer": before}
        with self._lock:
            if self._closed:
                return None
            for node in order:
                if self.route_policy == "deficit" and per_layer <= self.target * c_ms:
                    break
                packet_ref = items.get(node, {}).get("packet")
                key = self._bound.get(packet_ref) if packet_ref is not None else None
                entry = None if key is None else self._entries.get(key)
                node_jobs = by_node[node]
                if entry is None or not all(entry.covers(job.packet, job.tiles) for job in node_jobs):
                    misses += 1
                    continue
                entry.pins += 1
                self._entries.move_to_end(key)
                routed.append((node, entry))
                hits += 1
                per_layer -= node_bytes[node] * ms_per_byte / layers
            self.stats_["restores"] += 1
            self.stats_["routed_restores"] += bool(routed)
            self.stats_["hits"] += hits
            self.stats_["misses"] += misses
            self.stats_["routed_nodes"] += len(routed)
        copied = sum(entry.view(tile.index).numel() for node, entry in routed for job in by_node[node]
                     for tile in job.tiles)
        record.update({"routed_nodes": len(routed), "hits": hits, "misses": misses,
                       "routed_jobs": sum(len(by_node[node]) for node, _ in routed),
                       "routed_bitstream_bytes": sum(node_bytes[node] for node, _ in routed),
                       "routed_bytes": copied, "predicted_nvdec_ms_per_layer_after": per_layer,
                       "routed_order": [order.index(node) for node, _ in routed], "copy_events": [],
                       "_entries": [entry for _, entry in routed]})
        for node, entry in routed:
            route = _Route(self, entry, record)
            for job in by_node[node]:
                job.routed = route
        with self._lock:
            self.stats_["routed_jobs"] += record["routed_jobs"]
            self.stats_["routed_bytes"] += copied
            self.stats_["routed_bitstream_bytes"] += record["routed_bitstream_bytes"]
        return record

    def release(self, record: Mapping[str, Any] | None) -> dict[str, Any] | None:
        """Release a routing's leases; returns the public record (H2D device milliseconds once the copies completed,
        else None)."""
        if record is None:
            return None
        entries = record.get("_entries") or []
        with self._lock:
            for entry in entries:
                entry.pins -= 1
            self._sweep()
        device_ms = 0.0
        for start, end in record.get("copy_events") or []:
            if not end.query():
                device_ms = None
                break
            device_ms += start.elapsed_time(end)
        public = {key: value for key, value in record.items() if not key.startswith("_") and key != "copy_events"}
        public["h2d_device_ms"] = device_ms if record.get("copy_events") else (0.0 if not entries else None)
        return public

    # -- copies -------------------------------------------------------------------------------------------------------
    def _submit(self, job, entry: _Entry, record: dict):
        from .decoder_pool import DecodeTicket

        ticket = DecodeTicket(0, job.pictures)
        ticket.slot = -1
        if self.device.type != "cuda":  # tests: synchronous copies on the caller's thread
            self._copy(job, entry, record, ticket)
            return ticket
        with self._lock:
            if self._closed:
                raise StateError("the decoded tier is closed")
        self._jobs.put((job, entry, record, ticket))
        return ticket

    def _run(self) -> None:
        try:
            torch.cuda.set_device(self.device)
        except BaseException:  # noqa: BLE001 - every ticket then fails below
            pass
        while True:
            item = self._jobs.get()
            if item is None:
                return
            self._copy(*item)

    def _copy(self, job, entry: _Entry, record: dict, ticket) -> None:
        started = time.perf_counter()
        try:
            stream = self.stream
            if stream is None:
                for tile in job.tiles:
                    offset, (x, y, w, h) = job.offsets[tile.index], tile.rect
                    source = entry.view(tile.index)
                    job.canvas[offset:offset + source.shape[0], :, y:y + h, x:x + w].copy_(source)
                event = None
            else:
                with torch.inference_mode(False), torch.cuda.stream(stream):
                    begin = torch.cuda.Event(enable_timing=True)
                    begin.record(stream)
                    for tile in job.tiles:
                        offset, (x, y, w, h) = job.offsets[tile.index], tile.rect
                        source = entry.view(tile.index)
                        job.canvas[offset:offset + source.shape[0], :, y:y + h, x:x + w].copy_(source,
                                                                                               non_blocking=True)
                    event = torch.cuda.Event(enable_timing=True)
                    event.record(stream)
                job.canvas.record_stream(stream)
                record["copy_events"].append((begin, event))
                with self._lock:
                    entry.event = event
            with self._lock:
                self.stats_["copy_issue_seconds"] += time.perf_counter() - started
            ticket._finish(event=event)
        except BaseException as exc:  # noqa: BLE001 - delivered through the ticket (the restore fails, typed)
            with self._lock:
                self.stats_["copy_errors"] += 1
            ticket._finish(error=exc)

    # -- records / lifecycle ------------------------------------------------------------------------------------------
    def stats(self) -> dict[str, Any]:
        with self._lock:
            self._sweep()
            return {**self.stats_, "entries": len(self._entries), "resident_bytes": self._resident,
                    "budget_bytes": self.budget_bytes, "bindings": len(self._bound), "pinned": self.pinned,
                    "slab_free_bytes": self._slab.free_bytes(), "slab_holes": len(self._slab._free),
                    "leased": sum(entry.pins > 0 for entry in self._entries.values()),
                    "retiring": len(self._retiring), "route": self.route_policy,
                    "layer_compute_ms": {"a": self.layer_model[0], "b": self.layer_model[1],
                                         "window": self.layer_model[2], "source": self.layer_model[3]},
                    "nvdec_gbps": self.nvdec_gbps, "target": self.target}

    def close(self) -> None:
        """Stop the copy worker (queued copies finish), unregister the fill hook, drop every entry and release the
        slab (unpinned once no restore leases an entry; a lease still held keeps it pinned until process exit)."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            worker = self._worker
        with _TIERS_LOCK:
            _TIERS.discard(self)
        if worker is not None:
            self._jobs.put(None)
            worker.join()
        if self.stream is not None:
            self.stream.synchronize()
        with self._lock:
            for key in list(self._entries):
                self._evict(key)
            self._bound.clear()
            self._sweep()
            if not self._retiring:
                self._slab.close()


def stream_digest(metadata: Mapping[str, Any] | None) -> str | None:
    """The HEVC stream payload digest of a node pack's metadata (None for packs without one, e.g. factor packs)."""
    try:
        digest = metadata["payloads"]["stream.hevc"]["sha256"]
    except (KeyError, TypeError):
        return None
    return digest if isinstance(digest, str) else None


__all__ = ["DecodedTier", "LAYER_COMPUTE_MS", "NVDEC_GBPS", "ROUTES", "TARGET", "layer_compute_model",
           "offer_decoded", "open_tiers", "stream_digest"]
