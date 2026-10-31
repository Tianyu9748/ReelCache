"""Device restore of schema-v2 all-intra packets: NVDEC canvases feeding ``restore_kernels``.

Pipeline (no host copies of decoded pixels, no dense coefficient intermediates):

1. :func:`prepare_packet` validates a ``reelcache.tensor_pack.v2`` (temporal or
   spatial) or a single-matrix ``reelcache.tensor.v2`` all-intra packet once, off
   the critical path, and keeps its tile map, expanded affines and FP16 bases.
2. :class:`PackedRestorer` plans decode jobs for the requested node entries,
   skipping resident ones (attended-context-only restore). A job is whole tiles
   of ONE packet decoded into ONE uint8 device canvas ``[P, planes, H, W]``;
   temporal packets decode only the selected tiles' byte ranges.
3. Jobs are submitted asynchronously to a :class:`~reelcache.decoder_pool.DecoderPool`
   in layer order; canvases come from a bounded
   :class:`~reelcache.decoder_pool.CanvasArena`.
4. :meth:`RestoreBatch.reconstruct` builds ``restore_kernels`` picture sources
   through the tile map and calls :func:`reelcache.restore_kernels.reconstruct`
   waiting on the decode (and basis upload) events. A canvas is recycled only
   after every reconstruction reading it has completed.

Inter-coded and legacy version-1 packets are rejected with ``UnsupportedError``
(use :class:`~reelcache.nvdec.NVDECDecoder`); nothing falls back to software.

:meth:`PackedRestorer.warmup` is the one per-process decode-side warm-up (e.g. at
backend attach): a decoder session for every canonical node canvas
(:func:`canonical_canvas_shapes`) on every worker.

Callers may run inside ``torch.inference_mode`` (SGLang serves requests under
it): canvases and prepared, uploaded and cached bases are always normal tensors
(reused across calls in either mode; canvases are written by decoder threads),
while reconstruction writes the caller's destination in the caller's mode.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import time

import numpy as np
import torch

from . import _hevc
from . import restore_kernels as rk
from .codecs import HEVC_SCHEMA, PACK_SCHEMA, _from_bytes, _is_pack, _public_key, _validate_pack, _validate_packet
from .decoder_pool import CanvasArena, DecoderPool
from .errors import ConfigurationError, IntegrityError, StateError, UnsupportedError
from .nvdec import normal_tensors
from .types import EncodedTensor

GRANULARITIES = ("tile", "layer", "packet")


@dataclass(frozen=True, eq=False)
class PreparedTile:
    """One matrix of a prepared packet: tile-map placement, affine and FP16 basis.

    ``ready`` is None or the readiness of a ``basis`` still being staged asynchronously (e.g. by
    :mod:`reelcache.restore_staging`): a CUDA event after which the device basis holds its rows, or an object whose
    ``event(timeout)`` host-waits until the rows' copy was enqueued and returns that event (raising the staging
    failure, e.g. an integrity error, instead). Reconstructions wait for it; decoding does not.
    """

    key: object
    layer: int
    target: str
    index: int
    pictures: tuple
    rect: tuple
    layout: Mapping
    scale: np.ndarray
    offset: np.ndarray
    basis: torch.Tensor
    rank: int
    grid: tuple
    token_start: int
    units: tuple
    nbytes: int
    ready: object = None


@dataclass(frozen=True, eq=False)
class PreparedPacket:
    """A validated all-intra version-2 packet ready for device restore."""

    schema: str
    node: Mapping
    width: int
    height: int
    pix_fmt: str
    pictures: int
    tiles: tuple
    stream_units: tuple
    random_access: bool
    stream_bytes: int
    padding: Mapping | None

    @property
    def planes(self):
        return 3 if self.pix_fmt == "yuv444p" else 1

    def tile(self, key):
        for tile in self.tiles:
            if tile.key == key:
                return tile
        raise ConfigurationError(f"prepared packet does not contain {key!r}")


def _affine(raw, normalization, rank):
    affine = np.frombuffer(raw, dtype="<f4").reshape(normalization["affine_shape"])
    _hevc.validate_affine(affine, normalization, rank)
    scale, offset = rk.expand_affine(affine, normalization, rank)
    return np.ascontiguousarray(scale, dtype=np.float32), np.ascontiguousarray(offset, dtype=np.float32)


def canonical_canvas_shapes(frames, height, width, pix_fmt, *, channels, max_rank=None, ladder=None):
    """``(width, height, pix_fmt)`` of every canonical node canvas of one tile geometry, smallest first.

    Lineage node packs use only the canonical canvases of ``node_packets.canvas_ladder`` (one per rank step of
    the ladder, default ``CANVAS_RANK_LADDER``), so warming these covers every pack of the geometry. ``max_rank``
    keeps the canvases needed for nodes up to that rank (default: every rank up to ``channels``); canvases larger
    than the HEVC maximum picture dimension (which no pack can use) are left out.
    """
    from .node_packets import CANVAS_RANK_LADDER, canvas_ladder

    if type(channels) is not int or channels < 1:
        raise ConfigurationError("channels must be a positive integer")
    max_rank = channels if max_rank is None else max_rank
    if type(max_rank) is not int or not 1 <= max_rank <= channels:
        raise ConfigurationError("max_rank must be an integer in [1, channels]")
    _hevc.validate_parameters(pix_fmt)
    shapes = []
    for step, canvas_width, canvas_height in canvas_ladder(frames, height, width, pix_fmt, max_rank=channels,
                                                           ladder=CANVAS_RANK_LADDER if ladder is None else ladder):
        shape = (canvas_width, canvas_height, pix_fmt)
        if max(canvas_width, canvas_height) <= _hevc.MAX_DIMENSION and shape not in shapes:
            shapes.append(shape)
        if step >= max_rank:
            break
    return shapes


def prepare_packet(encoded, *, key=None, verify=False, pin_memory=False, bases=None, executor=None, validated=None):
    """Validate one version-2 all-intra packet for device restore.

    ``key`` overrides the public key of a single-matrix packet (e.g. a
    ``(MatrixKey, stream)`` CFG label recorded by ``NodeEncoding.locations``).
    ``verify`` re-hashes payloads and access units (stores already verify on
    read). ``pin_memory`` pins the host bases so uploads are asynchronous.
    ``bases`` (node packs only): the tile bases staged outside the pack, e.g. by
    :mod:`reelcache.restore_staging`: one ``(tensor, ready)`` per tile, an FP16
    tensor of the tile's basis descriptor shape on any device whose bytes the
    caller checked against the descriptor (length and SHA256), and its
    readiness (see :class:`PreparedTile`). The pack then carries no basis
    payloads (``codecs._validate_pack(external_bases=True)``).
    ``executor`` (a ``concurrent.futures.Executor``) runs the tiles'
    access-unit scans in parallel (the scan releases the GIL).
    ``validated`` (node packs, ``verify=False``): the ``codecs._validate_pack``
    result for this very metadata object, e.g. computed while the payloads
    were still being read from same-length placeholders; it is used only when
    every payload has the validated length (otherwise ``IntegrityError``).
    """
    if not isinstance(encoded, EncodedTensor):
        raise ConfigurationError("encoded must be an EncodedTensor")
    if type(verify) is not bool or type(pin_memory) is not bool:
        raise ConfigurationError("verify and pin_memory must be boolean")

    def host_basis(raw, shape):
        with normal_tensors():
            basis = _from_bytes(raw, "float16", shape)
            return basis.pin_memory() if pin_memory else basis

    if _is_pack(encoded):
        if key is not None:
            raise ConfigurationError("key overrides apply only to single-matrix packets")
        if validated is None:
            pack = _validate_pack(encoded, verify=verify, external_bases=bases is not None)
        else:
            if verify or validated.get("metadata") is not encoded.metadata or validated.get("external_bases") is not (
                    bases is not None) or {name: len(raw) for name, raw in encoded.payloads.items()} != validated.get(
                    "payload_lengths") or not isinstance(encoded.payloads.get("stream.hevc"), bytes):
                raise IntegrityError("payloads differ from the pack validated ahead of them")
            pack = dict(validated, stream=encoded.payloads["stream.hevc"])
        if bases is not None:
            bases = list(bases)
            if len(bases) != len(pack["tiles"]):
                raise ConfigurationError("external bases need one (tensor, ready) per pack tile")
            for item, shape in zip(bases, pack["basis_shapes"]):
                if (not isinstance(item, tuple) or len(item) != 2 or not isinstance(item[0], torch.Tensor)
                        or item[0].dtype != torch.float16 or tuple(item[0].shape) != tuple(shape)):
                    raise IntegrityError("an external tile basis differs from its descriptor shape or dtype")
        canvas, stream, metadata = pack["canvas"], pack["stream"], pack["metadata"]
        ranged = metadata["packing"]["random_access"] == "tile_byte_ranges"
        scans = None
        if ranged:
            ranges = [(*tile["byte_range"], tile["pictures"][1]) for _, tile, _, _ in pack["tiles"]]
            if executor is not None:
                scans = [executor.submit(_hevc.picture_units, stream, *item) for item in ranges]
        tiles = []
        for position, ((index, tile, context, spec), shape, canonical) in enumerate(
                zip(pack["tiles"], pack["basis_shapes"], pack["keys"])):
            scale, offset = _affine(encoded.payloads[tile["affine"]], tile["normalization"], spec.rank)
            if ranged:
                found = scans[position].result() if scans is not None else _hevc.picture_units(stream, *ranges[position])
                units = tuple((stream, start, size, 1) for start, size in found)
            else:
                units = ()
            if bases is None:
                basis, ready = host_basis(encoded.payloads[tile["basis"]["payload"]], shape), None
            else:
                basis, ready = bases[position]
            first, count = tile["pictures"]
            tiles.append(PreparedTile(
                _public_key(*canonical), context.key.layer, context.key.target, index, (first, count),
                tuple(tile["rect"]), tile["video"], scale, offset, basis, spec.rank,
                (context.frames, context.height, context.width), context.token_start, units,
                sum(unit[2] for unit in units), ready,
            ))
        if ranged:
            stream_units = tuple(unit for tile in tiles for unit in tile.units)
        else:
            stream_units = tuple((stream, unit["offset"], unit["length"], 1)
                                 for unit in _hevc.all_intra_access_units(stream, canvas["pictures"]))
        return PreparedPacket(
            PACK_SCHEMA, dict(metadata["node"]), canvas["width"], canvas["height"], canvas["pix_fmt"],
            canvas["pictures"], tuple(tiles), stream_units, ranged, len(stream), metadata["padding"],
        )
    if not isinstance(encoded.metadata, Mapping) or encoded.metadata.get("schema") != HEVC_SCHEMA:
        raise UnsupportedError(
            "device packed restore decodes only version-2 all-intra packets; legacy version-1 packets "
            "use NVDECDecoder (display-order demuxer path)",
        )
    if bases is not None:
        raise ConfigurationError("external bases apply only to node packs")
    metadata, context, spec = _validate_packet(encoded, verify=verify)
    if (
        spec.kind != "svd_hevc" or spec.inter_prediction is not False
        or metadata.get("hevc") != _hevc.encoder_record(False)
    ):
        raise UnsupportedError(
            "device packed restore requires all-intra svd_hevc packets; inter-coded packets use NVDECDecoder",
        )
    video = metadata["video"]
    streams = [encoded.payloads[f"coefficients.{index:03d}.hevc"] for index in range(video["segments"])]
    units = tuple((stream, start, size, 1) for stream in streams
                  for start, size in _hevc.picture_units(stream, 0, len(stream), video["frames"]))
    scale, offset = _affine(encoded.payloads["affine.f32"], metadata["normalization"], spec.rank)
    public = context.key if key is None else key
    if key is not None and (
        not isinstance(key, tuple) or len(key) != 2 or key[0] != context.key or not isinstance(key[1], str)
    ):
        raise ConfigurationError("a key override must be (the packet's MatrixKey, stream label)")
    pictures = video["frames"] * video["segments"]
    tile = PreparedTile(
        public, context.key.layer, context.key.target, 0, (0, pictures),
        (0, 0, video["image_width"], video["image_height"]), video, scale, offset,
        host_basis(encoded.basis, metadata["basis"]["shape"]), spec.rank,
        (context.frames, context.height, context.width), context.token_start, units,
        sum(map(len, streams)),
    )
    node = {
        "model": context.model.to_dict(), "token_start": context.token_start, "frames": context.frames,
        "height": context.height, "width": context.width, "node_id": None,
    }
    return PreparedPacket(
        HEVC_SCHEMA, node, video["image_width"], video["image_height"], spec.pix_fmt, pictures, (tile,), units,
        True, sum(map(len, streams)), None,
    )


@dataclass(frozen=True)
class NodeRef:
    """A history node to restore: its id and packets (or a zero-argument loader).

    The loader runs only when at least one of the node's requested entries is
    not resident, so fully resident nodes cost no storage read.
    """

    node_id: str
    packets: object

    def __post_init__(self):
        if not isinstance(self.node_id, str) or not self.node_id.strip():
            raise ConfigurationError("node_id must be a nonempty string")

    def load(self):
        packets = self.packets() if callable(self.packets) else self.packets
        prepared = []
        for packet in packets:
            prepared.append(packet if isinstance(packet, PreparedPacket) else prepare_packet(packet))
        return prepared


class _Job:
    __slots__ = ("node_id", "node_order", "packet", "tiles", "units", "pictures", "offsets", "nbytes",
                 "first_layer", "ticket", "canvas", "pending", "events", "released", "basis_event", "submitted",
                 "routed")

    def __init__(self, node_id, node_order, packet, tiles, units, pictures, offsets):
        self.node_id, self.node_order, self.packet = node_id, node_order, packet
        self.tiles, self.units, self.pictures, self.offsets = tiles, units, pictures, offsets
        self.nbytes = sum(unit[2] for unit in units)
        self.first_layer = min(tile.layer for tile in tiles)
        self.ticket = self.canvas = self.basis_event = None
        self.pending = {(node_id, tile.key) for tile in tiles}
        self.events = []
        self.released = self.submitted = False
        self.routed = None  # accel A1: a decoded-tier source (reelcache.decoded_tier) filling the canvas instead of NVDEC


class PackedRestorer:
    """Plans and submits device restores of schema-v2 packets on a decoder pool.

    ``granularity``: decode-job size: ``"tile"`` (one matrix), ``"layer"``
    (default: the K/V tiles of one layer of one packet), ``"packet"`` (all
    selected tiles of a packet) or a positive int (that many layers). Every
    NVDEC call decodes one access unit (one picture).
    ``cache_bases`` keeps uploaded FP16 bases resident on the device across
    batches (an explicit device-memory cost reported by :meth:`stats`).
    """

    def __init__(self, pool, *, arena=None, arena_bytes=None, granularity="layer", cache_bases=False):
        if not isinstance(pool, DecoderPool):
            raise ConfigurationError("pool must be a DecoderPool")
        if granularity not in GRANULARITIES and not (type(granularity) is int and granularity >= 1):
            raise ConfigurationError(f"granularity must be one of {GRANULARITIES} or a positive layer count")
        if type(cache_bases) is not bool:
            raise ConfigurationError("cache_bases must be boolean")
        if arena is None:
            if type(arena_bytes) is not int or arena_bytes <= 0:
                raise ConfigurationError("a restorer needs a CanvasArena or a positive arena_bytes budget")
            arena = CanvasArena(pool.device, arena_bytes)
        elif not isinstance(arena, CanvasArena) or arena.device != pool.device:
            raise ConfigurationError("arena must be a CanvasArena on the pool's device")
        self.pool, self.arena, self.device = pool, arena, pool.device
        self.granularity, self.cache_bases = granularity, cache_bases
        self.copy_stream = None
        if self.device.type == "cuda":
            with torch.cuda.device(self.device):
                self.copy_stream = torch.cuda.Stream(device=self.device)
        self._bases = {}
        self.stats_ = {"batches": 0, "jobs": 0, "entries": 0, "skipped_resident": 0, "nodes_loaded": 0,
                       "nodes_skipped": 0, "basis_upload_bytes": 0, "cached_basis_bytes": 0}

    def _groups(self, tiles):
        if self.granularity == "tile":
            return [[tile] for tile in tiles]
        if self.granularity == "packet":
            return [tiles]
        size = 1 if self.granularity == "layer" else self.granularity
        groups, current, bucket = [], [], None
        for tile in tiles:
            index = tile.layer // size
            if current and index != bucket:
                groups.append(current)
                current = []
            current.append(tile)
            bucket = index
        if current:
            groups.append(current)
        return groups

    def plan(self, nodes, *, resident=(), keys=None):
        """Decode jobs for every requested, non-resident entry of ``nodes`` (layer order)."""
        nodes = list(nodes)
        if not nodes or any(not isinstance(node, NodeRef) for node in nodes):
            raise ConfigurationError("nodes must be a nonempty sequence of NodeRef")
        if len({node.node_id for node in nodes}) != len(nodes):
            raise ConfigurationError("node ids must be distinct")
        resident_nodes, resident_entries = set(), set()
        for item in resident:
            if isinstance(item, str):
                resident_nodes.add(item)
            elif isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str):
                resident_entries.add(item)
            else:
                raise ConfigurationError("resident items are node ids or (node_id, key) entries")
        wanted = None if keys is None else set(keys)
        jobs, entries, skipped = [], {}, 0
        for order, node in enumerate(nodes):
            if node.node_id in resident_nodes:
                self.stats_["nodes_skipped"] += 1
                continue
            loaded = node.load()
            self.stats_["nodes_loaded"] += 1
            for packet in loaded:
                selected = []
                for tile in packet.tiles:
                    entry = (node.node_id, tile.key)
                    if wanted is not None and tile.key not in wanted:
                        continue
                    if entry in resident_entries:
                        skipped += 1
                        continue
                    if entry in entries:
                        raise ConfigurationError(f"entry {entry!r} appears in two packets")
                    selected.append(tile)
                if not selected:
                    continue
                if not packet.random_access:
                    offsets = {tile.index: tile.pictures[0] for tile in packet.tiles}
                    job = _Job(node.node_id, order, packet, selected, list(packet.stream_units), packet.pictures,
                               offsets)
                    jobs.append(job)
                    for tile in selected:
                        entries[(node.node_id, tile.key)] = (job, tile)
                    continue
                for group in self._groups(selected):
                    units, offsets, cursor = [], {}, 0
                    for tile in group:
                        offsets[tile.index] = cursor
                        cursor += tile.pictures[1]
                        units.extend(tile.units)
                    job = _Job(node.node_id, order, packet, group, units, cursor, offsets)
                    jobs.append(job)
                    for tile in group:
                        entries[(node.node_id, tile.key)] = (job, tile)
        jobs.sort(key=lambda job: (job.first_layer, job.node_order))
        self.stats_["skipped_resident"] += skipped
        return RestoreBatch(self, jobs, entries, skipped)

    def submit(self, nodes, *, resident=(), keys=None):
        """Plan and submit every decode job now (see :meth:`RestoreBatch.submit_through` for lazy use)."""
        batch = self.plan(nodes, resident=resident, keys=keys)
        batch.submit_through(None)
        return batch

    def _upload(self, tiles):
        """Device FP16 bases for ``tiles`` on the copy stream; returns (bases, event or None)."""
        result, uploaded = {}, False
        stream = self.copy_stream
        with normal_tensors(), (torch.cuda.stream(stream) if stream is not None else _null()):
            for tile in tiles:
                if stream is not None and tile.basis.device == self.device:
                    # Already resident (device basis cache, staging): ordered by its owner or ``tile.ready``.
                    result[id(tile)] = tile.basis
                    continue
                cached = self._bases.get(id(tile))
                if cached is not None and cached[0] is tile:
                    result[id(tile)] = cached[1]
                    continue
                value = tile.basis.to(self.device, non_blocking=True)
                uploaded = True
                self.stats_["basis_upload_bytes"] += value.numel() * value.element_size()
                if self.cache_bases:
                    self._bases[id(tile)] = (tile, value)
                    self.stats_["cached_basis_bytes"] += value.numel() * value.element_size()
                result[id(tile)] = value
        event = None
        if uploaded and stream is not None:
            event = torch.cuda.Event()
            event.record(stream)
        return result, event

    def prewarm(self, packets, timeout=None):
        """Create warm NVDEC sessions for every canvas geometry of ``packets`` on every worker.

        Accepts PreparedPacket values or NodeRef values (their packets). Session
        creation (the SDK builds the hardware decoder at the first picture) is
        thereby moved off the restore critical path. Returns wall seconds.
        """
        samples = {}
        for item in packets:
            for packet in (item.load() if isinstance(item, NodeRef) else [item]):
                if not isinstance(packet, PreparedPacket):
                    raise ConfigurationError("prewarm takes PreparedPacket or NodeRef values")
                key = (packet.width, packet.height, packet.pix_fmt)
                if key not in samples and packet.stream_units:
                    samples[key] = (*key, packet.stream_units[0])
        return self.pool.prewarm(list(samples.values()), timeout) if samples else 0.0

    def warmup(self, geometries, *, max_rank=None, timeout=None):
        """Warm the decode side of the restore path once per process, e.g. at backend attach (one call).

        ``geometries`` are the ``(frames, height, width, pix_fmt, channels)`` tile geometries of the model (K and V
        usually share one). Every canonical node canvas of each geometry (:func:`canonical_canvas_shapes`, up to
        ``max_rank``) gets a decoder session on every worker, so no restore of a lineage node pack creates one;
        repeated calls cost only the check. Returns the pool's report (``DecoderPool.prewarm_shapes``) with the
        shapes.
        """
        shapes = []
        for geometry in geometries:
            if not isinstance(geometry, tuple) or len(geometry) != 5:
                raise ConfigurationError("warmup geometries are (frames, height, width, pix_fmt, channels) tuples")
            frames, height, width, pix_fmt, channels = geometry
            for shape in canonical_canvas_shapes(frames, height, width, pix_fmt, channels=channels, max_rank=max_rank):
                if shape not in shapes:
                    shapes.append(shape)
        return {**self.pool.prewarm_shapes(shapes, timeout), "canvas_shapes": shapes}

    def clear_bases(self):
        self._bases.clear()
        self.stats_["cached_basis_bytes"] = 0

    def stats(self):
        return {**self.stats_, "pool": self.pool.stats(), "arena": self.arena.report()}

    def trim_device_cache(self, reason=None):
        """Free this restorer's idle decoded canvases (:meth:`CanvasArena.trim`); returns the bytes freed.

        Canvases leased by a restore in progress, or still read by its
        reconstructions, are untouched. Under device-memory pressure the
        decoder pool's registered trim (``decoder_pool:<gpu_id>`` in
        ``reelcache.device_pressure``) already covers this arena: owners need
        not register it. The caller hands the freed bytes back to the device
        with ``torch.cuda.empty_cache()``. Decoder sessions are kept.
        """
        return self.arena.trim(reason)


class RestoreBatch:
    """Submitted (or lazily submittable) decode jobs of one restore and their consumers.

    Entries are ``(node_id, key)``. Use from one consumer thread.
    """

    def __init__(self, restorer, jobs, entries, skipped):
        self.restorer, self.jobs, self._entries, self.skipped = restorer, jobs, entries, skipped
        self._cursor = 0
        self._bases = {}
        self._closed = self._counted = False
        self.created = time.perf_counter()

    def __len__(self):
        return len(self._entries)

    def entries(self, layer=None):
        """Entries in job order (optionally of one layer)."""
        result = []
        for job in self.jobs:
            for tile in job.tiles:
                if layer is None or tile.layer == layer:
                    result.append((job.node_id, tile.key))
        return result

    @property
    def layers(self):
        return sorted({tile.layer for job in self.jobs for tile in job.tiles})

    def submit_through(self, layer):
        """Submit every not-yet-submitted job whose first layer is <= ``layer`` (None: all)."""
        if self._closed:
            raise StateError("the restore batch is closed")
        restorer = self.restorer
        while self._cursor < len(self.jobs):
            job = self.jobs[self._cursor]
            if layer is not None and job.first_layer > layer:
                break
            packet = job.packet
            job.canvas = restorer.arena.acquire((job.pictures, packet.planes, packet.height, packet.width))
            try:
                # Arena canvases have no pending users, so decode never waits for the compute stream.
                if job.routed is not None:  # accel A1: copied from the decoded tier, not decoded
                    job.ticket = job.routed.submit(job)
                else:
                    job.ticket = restorer.pool.submit(job.units, job.canvas, pix_fmt=packet.pix_fmt, after=None)
            except BaseException:
                restorer.arena.release(job.canvas)
                job.canvas = None
                raise
            bases, job.basis_event = restorer._upload(job.tiles)
            self._bases.update(bases)
            job.submitted = True
            restorer.stats_["jobs"] += 1
            restorer.stats_["entries"] += len(job.tiles)
            self._cursor += 1
        if self._cursor == len(self.jobs) and self.jobs and not self._counted:
            restorer.stats_["batches"] += 1
            self._counted = True

    def _job(self, entry):
        try:
            job, tile = self._entries[entry]
        except KeyError:
            raise ConfigurationError(f"{entry!r} is not part of this restore batch") from None
        if not job.submitted:
            self.submit_through(job.first_layer)
        return job, tile

    def source(self, entry, timeout=None):
        """``restore_kernels.PictureSource`` of one entry (waits until its decode was issued)."""
        job, tile = self._job(entry)
        job.ticket.event(timeout)
        layout = tile.layout
        first = job.offsets[tile.index]
        frames = layout["frames"]
        indices = [[first + segment * frames + frame for frame in range(frames)]
                   for segment in range(layout["segments"])]
        return rk.PictureSource.from_canvases(job.canvas, layout, indices, (tile.rect[1], tile.rect[0]),
                                              tile.scale, tile.offset)

    def pixels(self, entry, timeout=None):
        """Synchronized ``[tokens, rank]`` uint8 pixels of one entry (identity tests/diagnostics)."""
        from .nvdec import unpack_device_tile

        job, tile = self._job(entry)
        job.ticket.synchronize(timeout)
        return unpack_device_tile(job.canvas, job.offsets[tile.index], tile.rect, tile.layout)

    def wait_events(self, entries, timeout=None):
        """Decode, basis-upload and basis-readiness events of ``entries`` (host-waits only for decode issue and
        for staged bases' copies to be enqueued)."""
        events, seen, ready = [], set(), set()
        for entry in entries:
            job, tile = self._job(entry)
            if tile.ready is not None and id(tile.ready) not in ready:
                ready.add(id(tile.ready))
                staged = tile.ready.event(timeout) if callable(getattr(tile.ready, "event", None)) else tile.ready
                if staged is not None:
                    events.append(staged)
            if id(job) in seen:
                continue
            seen.add(id(job))
            decoded = job.ticket.event(timeout)
            if decoded is not None:
                events.append(decoded)
            if job.basis_event is not None:
                events.append(job.basis_event)
        return events

    def reconstruction_jobs(self, entries, destination, rope=None, timeout=None):
        """``restore_kernels.ReconstructionJob`` per entry.

        ``destination(node_id, key, tile)`` returns the native ``NativeView``;
        ``rope(node_id, key, tile)`` returns ``RopePositions`` or None (V, or K of
        a cache that stores pre-RoPE keys).
        """
        result = []
        for entry in entries:
            job, tile = self._job(entry)
            node_id, key = entry
            result.append(rk.ReconstructionJob(
                self.source(entry, timeout), destination(node_id, key, tile), tile.grid,
                basis=self._bases[id(tile)], rank=tile.rank,
                rope=None if rope is None else rope(node_id, key, tile), label=f"{node_id}:{key}",
            ))
        return result

    def reconstruct(self, entries, destination, rope=None, *, stream=None, timeout=None, **options):
        """Reconstruct ``entries`` into their native slots; returns the ``RestoreHandle``.

        Canvases whose entries have all been reconstructed are recycled after
        the handles' events. ``options`` are passed to ``restore_kernels.reconstruct``.
        """
        if self._closed:
            raise StateError("the restore batch is closed")
        entries = list(entries)
        jobs = self.reconstruction_jobs(entries, destination, rope, timeout)
        handle = rk.reconstruct(jobs, stream=stream, wait_events=self.wait_events(entries, timeout), **options)
        for entry in entries:
            job, _ = self._entries[entry]
            if entry in job.pending:
                job.pending.discard(entry)
                if handle.event is not None:
                    job.events.append(handle.event)
                if not job.pending:
                    self._release(job)
        return handle

    def _release(self, job, extra=()):
        if job.released or job.canvas is None:
            return
        events = list(job.events) + list(extra)
        self.restorer.arena.release(job.canvas, after=events)
        job.released = True

    def wait(self, timeout=None):
        """Host-wait until every submitted decode was issued; returns the job decode events."""
        return [job.ticket.event(timeout) for job in self.jobs if job.submitted]

    def synchronize(self, timeout=None):
        for job in self.jobs:
            if job.submitted:
                job.ticket.synchronize(timeout)

    def close(self):
        """Recycle every canvas once its decode and consumers complete; idempotent."""
        if self._closed:
            return
        error = None
        for job in self.jobs:
            if not job.submitted or job.released:
                continue
            try:
                decode = job.ticket.event()
            except BaseException as exc:  # noqa: BLE001 - surface the first decode failure after cleanup
                decode = None
                error = error or exc
                job.ticket._done.wait()
            self._release(job, () if decode is None else (decode,))
        self._closed = True
        self._bases.clear()
        if error is not None:
            raise error

    def stats(self):
        submitted = [job for job in self.jobs if job.submitted]
        return {
            "jobs": len(self.jobs), "submitted": len(submitted), "entries": len(self._entries),
            "skipped_resident": self.skipped,
            "pictures": sum(job.pictures for job in self.jobs),
            "bitstream_bytes": sum(job.nbytes for job in self.jobs),
            "decode_calls": sum(len(job.units) for job in self.jobs),
            "workers": sorted({job.ticket.slot for job in submitted}),
            "cold_jobs": sum(job.ticket.cold for job in submitted if job.ticket.done()),
        }

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


def device_tile_pixels(canvas, first, tile):
    """Tile-map extraction of one prepared tile from a decoded canvas (diagnostics)."""
    from .nvdec import unpack_device_tile

    if not isinstance(tile, PreparedTile):
        raise ConfigurationError("tile must be a PreparedTile")
    if canvas.shape[2] < tile.rect[1] + tile.rect[3] or canvas.shape[3] < tile.rect[0] + tile.rect[2]:
        raise IntegrityError("tile rectangle lies outside the canvas")
    return unpack_device_tile(canvas, first, tile.rect, tile.layout)


__all__ = [
    "GRANULARITIES", "NodeRef", "PackedRestorer", "PreparedPacket", "PreparedTile", "RestoreBatch",
    "canonical_canvas_shapes", "device_tile_pixels", "prepare_packet",
]
