"""Cold-session staging of lineage node packs for device restore (TASK cold-session-restore).

A cold restore (the session's packs not prepared, its bases not on the GPU) spends most of its time on CPU work
that a prepared restore never repeats: object reads, basis assembly with its integrity checks, host
tensor<->bytes conversions, pack validation, access-unit scans and synchronous basis uploads (Step 5c cold
breakdown, ``benchmarks/restore_e2e``). :class:`StagingEngine` does that work once per object, on worker
threads (file reads, SHA256, ``memmem`` scans and host-to-device copies release the GIL), overlapped with
decoding:

* manifests: the runtime's own validated loaders (``load_packet``, basis-version chains), unchanged;
* payloads: each pack's HEVC stream and affines are read once and checked against their descriptors (length;
  SHA256 too when the store verifies reads); the pack is prepared from them with external bases
  (``packed_decode.prepare_packet(bases=...)``), its access-unit scans in parallel;
* bases: ONE device tensor per basis version-chain tip the history needs (an ancestor version is a row prefix of
  its descendant, so every tile of a chain is a prefix view of its tip). The tip's row blocks are read once,
  straight into a reusable pinned host ring; every block is checked against its content address, and one running
  SHA256 over the tip's rows checks every tile prefix against its pack descriptor and every complete version of
  the chain against its version digest (the checks of ``BasisVersionStore.assemble`` and ``attach_pack_bases``,
  each byte hashed twice instead of up to five times). Only verified rows are copied to the device, on a staging
  stream; each tile's ``ready`` ticket yields that copy's CUDA event, so decoding starts before the bases land
  and every reconstruction waits for exactly its own bases. A staging failure (e.g. an integrity error) is raised
  by the ticket before any reconstruction reads the rows, and by :meth:`StagedHistory.release`.

Integrity mode: that of the store path (``KVRuntime(verify_reads=...)``): stored lengths are checked against their
descriptors, basis row blocks and basis views are always hashed, payload bytes are hashed when the store verifies
reads. Reads run while the caller's thread holds the runtime lock and a store transaction (garbage collection
cannot remove an object being read).

Asynchronous verification (freeze D, D-E; ``verify_threads`` > 0, the default): metadata (packet manifests, basis
version chains, node maps) is verified before use, as before, and basis rows as above (on the staging threads,
overlapped with decoding; only reconstruction waits for them). The payload bytes -- the bitstreams and affines
decoding starts from -- are no longer hashed before the pack is prepared: packs are prepared from the bytes read
and decoded at once, and the payloads are hashed on the engine's verifier threads (``reelcache-verify``) once staging
has read them all (the largest first; reads never compete with this hashing), in parallel with decode and
reconstruction. The integrity gate is :meth:`StagedHistory.wait`, which the restorer calls before it adopts
anything and before its restore returns (``LineageRestorer.finish``): it waits for every verification of the
history and raises the first failure. So a restore returns, and its staged packs and bases are adopted, only when
every object it used verified; a mismatch quarantines the object in the store (``ObjectStore.quarantine``: every
later read verifies it before use -- staging re-checks quarantined objects before reading anything -- and raises
``QuarantinedError`` while its bytes still fail) and fails the restore with the IntegrityError, nothing adopted.
Payload bytes are decoded before they are verified: corrupt bytes fail typed (structural checks, NVDEC's own) or
decode to values that the gate discards with the restore. ``verify_threads=0`` hashes payloads on the reading
threads before use (the behavior before D-E).

FP16 factor packs (W1.2, wl-base: the SVD-only baseline, ``reelcache.factor_pack.v1``) are staged the same way --
the same plan, reads, staged bases and integrity gate -- and prepared by ``factor_packs.prepare_factor_packet``
(:func:`_pack_functions` selects by schema). Since no decode follows that their hashing could overlap, a factor
pack's large payload is hashed while it is read, range by range in order (:meth:`StagingEngine._verify_stream`;
asynchronous mode only); HEVC packs are unchanged.
"""

from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import hashlib
import os
import threading
import time
from types import MappingProxyType
from typing import Any, Mapping

import torch

from .errors import ConfigurationError, IntegrityError, QuarantinedError, StateError
from .nvdec import normal_tensors
from .types import EncodedTensor, MatrixKey

MIB = 1024**2
_ALIGN = 256


def _now():
    return time.perf_counter()


class StagingTicket:
    """Readiness of one staged basis tip.

    ``event(timeout)`` host-waits until the tip's verified rows were enqueued for the device and returns the
    copy's CUDA event (None for a CPU engine), or raises the staging failure.
    """

    __slots__ = ("_done", "_event", "_error", "label")

    def __init__(self, label=""):
        self._done = threading.Event()
        self._event = self._error = None
        self.label = label

    def _set(self, event):
        self._event = event
        self._done.set()

    def _fail(self, error):
        self._error = error
        self._done.set()

    def done(self):
        return self._done.is_set()

    def event(self, timeout=None):
        if not self._done.wait(timeout):
            raise StateError(f"basis staging of {self.label} did not finish within {timeout} s")
        if self._error is not None:
            if isinstance(self._error, IntegrityError):
                raise IntegrityError(f"basis staging of {self.label} failed: {self._error}") from self._error
            raise StateError(f"basis staging of {self.label} failed: {self._error!r}") from self._error
        return self._event


class _Segment:
    __slots__ = ("start", "nbytes", "tensor", "view", "released", "event")

    def __init__(self, start, nbytes, tensor, view):
        self.start, self.nbytes, self.tensor, self.view = start, nbytes, tensor, view
        self.released = threading.Event()
        self.event = None


class PinnedRing:
    """A reusable pinned host buffer handed out in FIFO order (session independent; allocated once).

    A segment's bytes are reused only after the device copy that read them completed (its CUDA event). A request
    larger than the ring gets a one-off pinned allocation (``stats['oversize']``).
    """

    def __init__(self, nbytes):
        if type(nbytes) is not int or nbytes < _ALIGN:
            raise ConfigurationError(f"the pinned staging ring needs at least {_ALIGN} bytes")
        self.capacity = nbytes
        self.buffer = self._view = None
        self._head = 0
        self._pending: deque[_Segment] = deque()
        self._lock = threading.Lock()
        self.stats = {"segments": 0, "bytes": 0, "waits": 0, "wait_seconds": 0.0, "oversize": 0}

    def allocate(self):
        if self.buffer is None:
            with normal_tensors():
                self.buffer = torch.empty(self.capacity, dtype=torch.uint8, pin_memory=torch.cuda.is_available())
            self._view = memoryview(self.buffer.numpy())
        return self.buffer

    def reserve(self, nbytes):
        """A segment of ``nbytes`` (waits for older segments' copies when the ring wraps onto them)."""
        if type(nbytes) is not int or nbytes < 1:
            raise ConfigurationError("a staging segment needs a positive byte count")
        if nbytes > self.capacity:
            with normal_tensors():
                tensor = torch.empty(nbytes, dtype=torch.uint8, pin_memory=torch.cuda.is_available())
            with self._lock:
                self.stats["oversize"] += 1
            return _Segment(-1, nbytes, tensor, memoryview(tensor.numpy()))
        self.allocate()
        size = -(-nbytes // _ALIGN) * _ALIGN
        with self._lock:
            start = self._head if self._head + size <= self.capacity else 0
            for segment in list(self._pending):
                if segment.start < start + size and start < segment.start + segment.nbytes:
                    waited = _now()
                    segment.released.wait()
                    if segment.event is not None:
                        segment.event.synchronize()
                    self.stats["waits"] += 1
                    self.stats["wait_seconds"] += _now() - waited
                    self._pending.remove(segment)
            while self._pending and self._pending[0].released.is_set() and (
                    self._pending[0].event is None or self._pending[0].event.query()):
                self._pending.popleft()
            segment = _Segment(start, size, self.buffer[start:start + nbytes], self._view[start:start + nbytes])
            self._pending.append(segment)
            self._head = start + size
            self.stats["segments"] += 1
            self.stats["bytes"] += nbytes
            return segment

    @staticmethod
    def release(segment, event):
        """Return ``segment`` once ``event`` (the copy that read it; None: nothing did) completes."""
        segment.event = event
        segment.released.set()


@dataclass
class _Tip:
    ref: str
    chain: list
    channels: int
    rows: int = 0
    blocks: list = field(default_factory=list)
    checks: dict = field(default_factory=dict)
    order: tuple = ()
    tensor: Any = None
    ticket: Any = None
    read_done: Any = None
    alias: Any = None  # W1.2: the tip whose staging this byte-identical one shares (``StagingEngine._plan``)

    @property
    def nbytes(self):
        return self.rows * self.channels * 2


@dataclass
class StagedHistory:
    """Prepared node packs of one history with staged device bases (see :class:`StagingEngine`).

    ``refs``: ``NodeRef`` values for ``PackedRestorer.plan``; ``prepared``: packet ref -> ``PreparedPacket``
    (its tiles hold device prefix views and ``ready`` tickets); ``bindings``: packet ref -> its tiles'
    ``(basis_version, rank_view)``; ``versions``: ``(version_ref, chain_refs, tensor, ticket)`` of tips staged
    as complete versions (adoptable by a device basis cache once :meth:`wait` returned); ``stats``.
    ``futures``: the tip staging tasks (bases verified, then copied); ``verifications``: the payload verification
    futures (asynchronous verification).
    """

    refs: list = field(default_factory=list)
    prepared: dict = field(default_factory=dict)
    bindings: dict = field(default_factory=dict)
    versions: list = field(default_factory=list)
    stats: dict = field(default_factory=dict)
    tickets: list = field(default_factory=list)
    futures: list = field(default_factory=list)
    verifications: list = field(default_factory=list)
    released: bool = False
    engine: Any = None

    def wait(self, timeout=None):
        """The integrity gate: host-wait until every basis copy was enqueued and every object of the history was
        verified; raises the first staging or verification failure (nothing may be adopted or returned before)."""
        for ticket in self.tickets:
            ticket.event(timeout)
        started, pending = _now(), not self._verified()
        error = self._collect(timeout)
        waited = _now() - started if pending else 0.0
        self.stats["verify_gate_wait_seconds"] = self.stats.get("verify_gate_wait_seconds", 0.0) + waited
        if self.engine is not None:
            self.engine._add(verify_gate_waits=int(pending), verify_gate_wait_seconds=waited)
        if error is not None:
            self._raise(error)

    def _verified(self):
        """Whether every staging task and verification has finished (failed ones included)."""
        for future in self.futures:
            if not future.done():
                return False
            verification = None if future.cancelled() or future.exception() is not None else future.result()
            if verification is not None and not verification.done():
                return False
        return all(future.done() for future in self.verifications)

    def _collect(self, timeout):
        """Wait for every staging task and verification; returns the first failure in history order (tips first)."""
        error = None
        for future in self.futures:
            try:
                verification = future.result(timeout)
                if verification is not None:
                    verification.result(timeout)
            except BaseException as exc:  # noqa: BLE001 - reported once every task finished
                error = error or exc
        for future in self.verifications:
            try:
                future.result(timeout)
            except BaseException as exc:  # noqa: BLE001
                error = error or exc
        return error

    @staticmethod
    def _raise(error):
        if isinstance(error, IntegrityError):
            raise error
        raise StateError(f"basis staging failed: {error!r}") from error

    def release(self, timeout=None):
        """Wait for every staging task and verification (pinned segments return to the ring), then drop the device
        tensors.

        Raises the first staging or verification failure, so a restore that never read a failed tip still fails.
        Idempotent.
        """
        if self.released:
            return
        self.released = True
        error = self._collect(timeout)
        self.refs, self.prepared, self.versions = [], {}, []
        if error is not None:
            self._raise(error)


def _pack_view(metadata, payloads):
    """An EncodedTensor over a runtime-validated, parsed packet manifest without copying its JSON.

    ``prepare_packet`` runs every structural check on it; the payloads are the bytes read here. Read-only.
    """
    encoded = object.__new__(EncodedTensor)
    object.__setattr__(encoded, "metadata", metadata)
    object.__setattr__(encoded, "payloads", MappingProxyType(payloads))
    object.__setattr__(encoded, "basis", None)
    object.__setattr__(encoded, "metrics", None)
    return encoded


def _pack_functions(metadata):
    """``(validate, prepare, factor)`` of a staged node pack by its schema: the HEVC node pack's
    (``packed_decode._validate_pack`` / ``prepare_packet``) or, W1.2 (wl-base), the FP16 factor pack's
    (``factor_packs.validate_factor_pack`` / ``prepare_factor_packet``; same external tile bases)."""
    from .codecs import FACTOR_PACK_SCHEMA

    if isinstance(metadata, Mapping) and metadata.get("schema") == FACTOR_PACK_SCHEMA:
        from .factor_packs import prepare_factor_packet, validate_factor_pack

        return validate_factor_pack, prepare_factor_packet, True
    from .packed_decode import _validate_pack, prepare_packet

    return _validate_pack, prepare_packet, False


def _read_object(path, nbytes, store=None):
    with open(path, "rb", buffering=0) as handle:
        data = handle.readall()
        _drop_read(store, handle.fileno(), len(data))
    if len(data) != nbytes:
        raise IntegrityError(f"stored object {os.path.basename(path)[:16]}... holds {len(data)} bytes; "
                             f"its descriptor declares {nbytes}")
    return data


def _drop_read(store, fd, nbytes):
    """Workload v2 ``store_io="nocache"`` (CONTRACT §10): the store drops the pages of an object file read here."""
    drop = getattr(store, "drop_read_cache", None)
    if drop is not None:
        drop(fd, nbytes)


def _libc_pread():
    import ctypes
    import ctypes.util

    try:
        name = ctypes.util.find_library("c")
        function = ctypes.CDLL(name, use_errno=True).pread if name else None
        allocate = ctypes.pythonapi.PyBytes_FromStringAndSize
    except (OSError, AttributeError):
        return None
    if function is None:
        return None
    function.restype = ctypes.c_ssize_t
    function.argtypes = (ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_long)
    allocate.restype = ctypes.py_object
    allocate.argtypes = (ctypes.c_void_p, ctypes.c_ssize_t)

    def new_bytes(nbytes):
        data = allocate(None, nbytes)
        return data, ctypes.cast(ctypes.c_char_p(data), ctypes.c_void_p).value

    def read_range(descriptor, address, offset, length):
        done = 0
        while done < length:
            count = function(descriptor, address + offset + done, length - done, offset + done)
            if count <= 0:
                raise IntegrityError(f"short read of a stored object ({ctypes.get_errno()})")
            done += count

    return new_bytes, read_range


# Large payloads (e.g. LongCat's ~385 MB node stream) are read in parallel ranges straight into a fresh bytes object
# (allocated uninitialized, filled by pread with the GIL released, never visible before it is complete).
_PREAD = _libc_pread()
_PARALLEL_READ_BYTES = 8 * MIB
_READ_CHUNK_BYTES = 4 * MIB
# Payloads at least this large are verified by a task of their own (smaller ones of a pack share one task).
_VERIFY_ALONE_BYTES = 1 * MIB


class _ParallelRead:
    """A large object read as parallel ranges on ``pool``; ``result()`` returns its bytes."""

    def __init__(self, pool, path, nbytes, store=None):
        self.descriptor = os.open(path, os.O_RDONLY)
        self.store, self.nbytes = store, nbytes
        try:
            size = os.fstat(self.descriptor).st_size
            if size != nbytes:
                raise IntegrityError(f"stored object {os.path.basename(path)[:16]}... holds {size} bytes; "
                                     f"its descriptor declares {nbytes}")
            new_bytes, read_range = _PREAD
            self.data, address = new_bytes(nbytes)
            self.futures = [pool.submit(read_range, self.descriptor, address, offset,
                                        min(_READ_CHUNK_BYTES, nbytes - offset))
                            for offset in range(0, nbytes, _READ_CHUNK_BYTES)]
        except BaseException:
            os.close(self.descriptor)
            raise

    def result(self):
        try:
            for future in self.futures:
                future.result()
            _drop_read(self.store, self.descriptor, self.nbytes)
        finally:
            self.close()
        data, self.data = self.data, None
        return data

    def close(self):
        """Stop pending ranges, wait for running ones and close the file (idempotent)."""
        if self.descriptor is None:
            return
        for future in self.futures:
            future.cancel()
        for future in self.futures:
            if not future.cancelled():
                future.exception()
        os.close(self.descriptor)
        self.descriptor = None


class StagingEngine:
    """Parallel, overlapped staging of cold lineage node packs on ``device`` (see the module docstring).

    ``threads`` workers stage basis tips and as many read large payload ranges; ``prepare_threads`` workers
    read the other payloads and as many scan access units (separate pools, so pack preparation, which gates
    decoding, never queues behind basis work); ``verify_threads`` workers hash payloads while they are decoded and
    reconstructed (``0``: on the reading threads, before use; module docstring);
    ``pinned_bytes`` sizes the reusable pinned ring. :meth:`warmup` allocates the ring and starts every thread
    (session independent, e.g. at backend attach).
    """

    def __init__(self, device, *, threads=8, prepare_threads=4, pinned_bytes=256 * MIB, verify_threads=4,
                 verify_bases=True):
        for value, name in ((threads, "threads"), (prepare_threads, "prepare_threads")):
            if type(value) is not int or value < 1:
                raise ConfigurationError(f"{name} must be a positive integer")
        if type(verify_threads) is not int or verify_threads < 0:
            raise ConfigurationError("verify_threads must be a nonnegative integer (0: verify before use)")
        if type(verify_bases) is not bool:
            raise ConfigurationError("verify_bases must be a bool")
        self.device = torch.device(device)
        self.threads, self.prepare_threads, self.verify_threads = threads, prepare_threads, verify_threads
        # decomp_opt (2026-10-25): False skips the basis row hashing of a store that does not verify reads (the
        # store's verify_reads off: no object of a restore is hashed); True (default) hashes basis rows always.
        self.verify_bases = verify_bases
        self.ring = PinnedRing(pinned_bytes)
        self.stream = None
        if self.device.type == "cuda":
            with torch.cuda.device(self.device):
                self.stream = torch.cuda.Stream(device=self.device)
        self._bases = ThreadPoolExecutor(threads, thread_name_prefix="reelcache-stage-bases")
        self._payloads = ThreadPoolExecutor(prepare_threads, thread_name_prefix="reelcache-stage-payloads")
        self._chunks = ThreadPoolExecutor(threads, thread_name_prefix="reelcache-stage-chunks")
        self._scans = ThreadPoolExecutor(prepare_threads, thread_name_prefix="reelcache-stage-scans")
        self._verify = (ThreadPoolExecutor(verify_threads, thread_name_prefix="reelcache-verify")
                        if verify_threads else None)
        self._lock = threading.Lock()
        self._closed = False
        self.stats_ = {"histories": 0, "packs": 0, "tips": 0, "tip_bytes": 0, "blocks": 0, "block_bytes": 0,
                       "payload_bytes": 0, "seconds": 0.0, "thread_read_seconds": 0.0,
                       "thread_hash_seconds": 0.0, "thread_copy_seconds": 0.0,
                       # D-E records: payloads and bytes hashed on the verifier threads (``verify_threads`` > 0) and
                       # their thread seconds, failed checks (payloads or bases), and the integrity gate's waits for
                       # verifications that had not finished when the restore reached it.
                       "verify_objects": 0, "verify_bytes": 0, "verify_seconds": 0.0, "verify_failures": 0,
                       "verify_gate_waits": 0, "verify_gate_wait_seconds": 0.0, "verify_streamed": 0,
                       # decomp_opt records: basis tips staged without hashing (verify_bases=False, reads unverified)
                       "unhashed_tips": 0}

    def _pools(self):
        pools = [(self._bases, self.threads), (self._payloads, self.prepare_threads), (self._chunks, self.threads),
                 (self._scans, self.prepare_threads)]
        return pools + ([(self._verify, self.verify_threads)] if self._verify is not None else [])

    def warmup(self):
        """Allocate the pinned ring and start every worker thread (once per process); returns its cost."""
        started = _now()
        self.ring.allocate()
        for pool, count in self._pools():
            list(pool.map(lambda _: None, range(count)))
        return {"seconds": _now() - started, "pinned_bytes": self.ring.capacity, "threads": self.threads,
                "prepare_threads": self.prepare_threads, "verify_threads": self.verify_threads}

    def _add(self, **values):
        with self._lock:
            for name, value in values.items():
                self.stats_[name] += value

    def verification_mode(self, store):
        """The configured verification of the payloads this engine stages from ``store``: ``"async"`` (hashed on the
        verifier threads, gated before the restore returns), ``"sync"`` (``verify_threads=0``: hashed before use)
        or ``"off"`` (the store does not verify reads). Manifests and basis rows are verified in every mode."""
        if not getattr(store, "verify_reads", False):
            return "off"
        return "async" if self._verify is not None else "sync"

    def verification_report(self, store, staged=None, *, since=None):
        """The restore's ``verification`` record (``restore.lease.verification``): ``mode`` (configured),
        ``verified_payloads`` and ``payload_wait_seconds``. A staged history reports the payload objects it verified
        and its staging wait for payload reads. A restore that staged nothing reports the payload objects the store
        hash-checked for the calling thread since ``since`` (a :meth:`ObjectStore.verified_payloads` value taken
        when the restore began; omitted: 0) -- e.g. a store-path load of packs not prepared; 0 when it read nothing
        (a prepared restore)."""
        recorded = (getattr(staged, "stats", None) or {}).get("verification") if staged is not None else None
        if recorded is not None:
            return dict(recorded)
        return {"mode": self.verification_mode(store), "verified_payloads": _verified_since(store, since),
                "payload_wait_seconds": 0.0}

    def _mismatch(self, store, object_id, message):
        """A failed check of ``object_id``'s bytes: quarantine it in ``store`` and raise the IntegrityError."""
        quarantine = getattr(store, "quarantine", None)
        if quarantine is not None:
            quarantine(object_id, message)
        self._add(verify_failures=1)
        raise IntegrityError(message)

    # -- planning (the caller holds the runtime lock and a store transaction) ---------------------------------
    def _plan(self, runtime, history):
        store, versions = runtime.store, runtime.basis_versions
        packs, needed = [], {}
        for item in history:
            packet = runtime.load_packet(item["packet"])
            if packet.get("basis_storage") != "external_version_prefix_tiles":
                raise ConfigurationError("staging restores node packs with external tile bases only")
            metadata = packet["metadata"]
            descriptors = metadata["payloads"]
            tiles = []
            for tile, binding in zip(metadata["tiles"], packet["tile_bases"]):
                version_ref, rank = binding["basis_version"], binding["rank_view"]
                chain = versions.chain(version_ref)
                version = chain[-1][1]
                descriptor = descriptors.get(tile["basis"]["payload"])
                if (version.key != MatrixKey.from_dict(tile["key"]) or type(rank) is not int
                        or not 1 <= rank <= version.total_rank or not isinstance(descriptor, Mapping)
                        or descriptor.get("shape") != [rank, version.channels] or descriptor.get("dtype") != "float16"
                        or descriptor.get("bytes") != rank * version.channels * 2):
                    raise IntegrityError("packed tile is not bound to a legal prefix of its basis version")
                tiles.append((version_ref, rank, descriptor["sha256"], (tile["key"]["layer"], tile["key"]["target"])))
                needed.setdefault(version_ref, chain)
            paths = {blob: str(store.path_for(blob)) for blob in packet["payloads"].values()}
            packs.append({"item": item, "packet": packet, "tiles": tiles, "paths": paths})
        ancestors = {ancestor for chain in needed.values() for ancestor, _ in chain[:-1]}
        tips, served = {}, {}
        for ref in sorted(needed):
            if ref in ancestors:
                continue
            chain = needed[ref]
            tips[ref] = _Tip(ref, chain, chain[-1][1].channels)
            for member, _ in chain:
                served.setdefault(member, ref)
        for pack in packs:
            pack["assign"] = []
            for version_ref, rank, digest, order in pack["tiles"]:
                tip = tips[served[version_ref]]
                tip.rows = max(tip.rows, rank)
                tip.checks.setdefault(rank * tip.channels * 2, []).append(
                    (digest, "assembled basis version prefix differs from the packed tile descriptor",
                     pack["item"]["packet"]))
                tip.order = min(tip.order, order) if tip.order else order
                pack["assign"].append((tip, rank))
        for tip in tips.values():
            remaining = tip.rows
            for member, version in tip.chain:
                if remaining <= 0:
                    break
                take = min(version.added_rank, remaining)
                tip.blocks.append((version.added_rows_ref, str(store.path_for(version.added_rows_ref)),
                                   version.added_bytes, take * tip.channels * 2))
                remaining -= take
                if version.total_rank <= tip.rows:
                    tip.checks.setdefault(version.total_rank * tip.channels * 2, []).append(
                        (version.view_sha256, "assembled basis view differs from its version digest", member))
        # W1.2 (wl-base): chains whose staged rows are byte-identical -- the same content-addressed row blocks, rows
        # and width, e.g. the HEVC-only arm's identity basis (one version per key over ONE block) -- share one tip:
        # read, hashed and copied once; every check of every sharer runs on its rows
        canonical = {}
        for tip in sorted(tips.values(), key=lambda tip: (tip.order, tip.ref)):
            first = canonical.setdefault((tuple((block, take) for block, _, _, take in tip.blocks), tip.rows,
                                          tip.channels), tip)
            if first is not tip:
                tip.alias = first
                for boundary, items in tip.checks.items():
                    first.checks.setdefault(boundary, []).extend(items)
        check = getattr(store, "check_quarantined", None)
        if check is not None:  # an object that failed a check earlier is verified now, before any use
            check([blob for pack in packs for blob in pack["paths"]]
                  + [block for tip in tips.values() for block, *_ in tip.blocks])
        return packs, sorted(tips.values(), key=lambda tip: (tip.order, tip.ref))

    # -- worker tasks --------------------------------------------------------------------------------------
    def _check_object(self, store, object_id, data):
        if hashlib.sha256(data).hexdigest() != object_id.split(":", 1)[1]:
            self._mismatch(store, object_id, f"immutable object hash or length mismatch: {object_id}")

    def _verify_objects(self, store, items):
        """Verifier task: check ``[(object_id, bytes)]`` against their content addresses."""
        started, nbytes = _now(), 0
        try:
            for object_id, data in items:
                self._check_object(store, object_id, data)
                nbytes += len(data)
        finally:
            self._add(verify_objects=len(items), verify_bytes=nbytes, verify_seconds=_now() - started)

    def _verify_stream(self, store, object_id, data, ranges):
        """Verifier task (W1.2, factor packs): hash an object read by :class:`_ParallelRead` range by range, in
        order, each once its read finished (``ranges``: the reader's range futures, offset order), then check it
        against its content address like :meth:`_check_object`. ``verify_seconds`` counts the hashing only."""
        digest, hashing, view = hashlib.sha256(), 0.0, memoryview(data)
        try:
            for index, future in enumerate(ranges):
                future.result()
                mark = _now()
                digest.update(view[index * _READ_CHUNK_BYTES:(index + 1) * _READ_CHUNK_BYTES])
                hashing += _now() - mark
        finally:
            view.release()
            self._add(verify_objects=1, verify_bytes=len(data), verify_seconds=hashing, verify_streamed=1)
        if digest.hexdigest() != object_id.split(":", 1)[1]:
            self._mismatch(store, object_id, f"immutable object hash or length mismatch: {object_id}")

    def _read_payloads(self, pack, names, verify, store):
        started = _now()
        packet, payloads, nbytes = pack["packet"], {}, 0
        descriptors = packet["metadata"]["payloads"]
        for name in names:
            blob = packet["payloads"][name]
            data = _read_object(pack["paths"][blob], descriptors[name]["bytes"], store)
            if verify and self._verify is None:
                self._check_object(store, blob, data)
            payloads[name] = data
            nbytes += len(data)
        self._add(thread_read_seconds=_now() - started, payload_bytes=nbytes)
        return payloads

    @staticmethod
    def _validate_ahead(validate, pack):
        """``_validate_pack`` of the pack's metadata over same-length placeholders, or None (the pack is then
        validated with its payloads, which reports any malformed descriptor precisely)."""
        packet = pack["packet"]
        try:
            lengths = {name: packet["metadata"]["payloads"][name]["bytes"] for name in packet["payloads"]}
        except (KeyError, TypeError):
            return None
        if any(type(length) is not int or length < 0 for length in lengths.values()):
            return None
        placeholders = {name: bytes(length) for name, length in lengths.items()}
        try:
            validated = validate(_pack_view(packet["metadata"], placeholders), verify=False, external_bases=True)
        finally:
            placeholders.clear()
        if isinstance(validated, dict):  # an HEVC pack's (a factor pack's validation is its parsed tile list)
            validated["stream"] = None  # the real stream replaces the placeholder in prepare_packet
        return validated

    def _finish_read(self, pack, name, reader, verify, store):
        started = _now()
        data = reader.result()
        if verify and self._verify is None:
            self._check_object(store, pack["packet"]["payloads"][name], data)
        self._add(thread_read_seconds=_now() - started, payload_bytes=len(data))
        return {name: data}

    def _submit_payloads(self, pack, verify, store):
        """Futures of the pack's payload dicts: large objects as parallel ranges, the rest in one read task. W1.2:
        a factor pack's large payload (``pack["stream_verify"]``, asynchronous verification) is also hashed while it
        is read (:meth:`_verify_stream`), since no decode follows that its hashing could overlap."""
        packet, futures, small = pack["packet"], [], []
        descriptors = packet["metadata"]["payloads"]
        for name in sorted(packet["payloads"]):
            nbytes = descriptors[name]["bytes"]
            if _PREAD is not None and type(nbytes) is int and nbytes >= _PARALLEL_READ_BYTES:
                reader = _ParallelRead(self._chunks, pack["paths"][packet["payloads"][name]], nbytes, store)
                pack.setdefault("readers", []).append(reader)
                if verify and self._verify is not None and pack.get("stream_verify"):
                    pack.setdefault("streamed", {})[name] = self._verify.submit(
                        self._verify_stream, store, packet["payloads"][name], reader.data, list(reader.futures))
                futures.append(self._payloads.submit(self._finish_read, pack, name, reader, verify, store))
            elif name == "stream.hevc":
                futures.append(self._payloads.submit(self._read_payloads, pack, [name], verify, store))
            else:
                small.append(name)
        if small:
            futures.append(self._payloads.submit(self._read_payloads, pack, small, verify, store))
        return futures

    def _check_tip(self, store, tip, view, blocks):
        """Every block against its content address, then one running SHA256 over the tip's rows against every
        tile-prefix descriptor and complete-version digest (a failed check quarantines the object it names)."""
        for block_ref, start, take, tail in blocks:
            digest = hashlib.sha256(view[start:start + take])
            if tail:
                digest.update(tail)
            if digest.hexdigest() != block_ref.split(":", 1)[1]:
                self._mismatch(store, block_ref, f"basis row block bytes differ from their content address: "
                                                 f"{block_ref}")
        running, position = hashlib.sha256(), 0
        for boundary in sorted(tip.checks):
            running.update(view[position:boundary])
            position = boundary
            digest = running.copy().hexdigest()
            for expected, message, owner in tip.checks[boundary]:
                if digest != expected:
                    self._mismatch(store, owner, f"{message}: {owner}")

    def _stage_tip(self, tip, store):
        """Read the tip's blocks into the ring, verify them (``_check_tip``) and copy them to the device; the
        ticket fires after the copy was enqueued (reconstruction never reads unverified bases)."""
        segment = None
        try:
            started = _now()
            try:
                segment = self.ring.reserve(tip.nbytes)
                view, cursor, blocks = segment.view, 0, []
                for block_ref, path, block_bytes, take in tip.blocks:
                    with open(path, "rb", buffering=0) as handle:
                        got = 0
                        while got < take:
                            count = handle.readinto(view[cursor + got:cursor + take])
                            if not count:
                                break
                            got += count
                        tail = handle.readall() if got == take else b""
                        _drop_read(store, handle.fileno(), got + len(tail))
                    if got != take or len(tail) != block_bytes - take:
                        raise IntegrityError("basis row block length disagrees with its version")
                    blocks.append((block_ref, cursor, take, tail))
                    cursor += take
            finally:
                tip.read_done.set()
            read = _now()
            if self.verify_bases or bool(getattr(store, "verify_reads", True)):
                self._check_tip(store, tip, view, blocks)
            else:
                self._add(unhashed_tips=1)
            hashed = _now()
            source = segment.tensor.view(tip.rows, 2 * tip.channels)
            event = None
            if self.stream is not None:
                with torch.cuda.stream(self.stream):
                    tip.tensor.view(torch.uint8).copy_(source, non_blocking=True)
                    event = torch.cuda.Event()
                    event.record(self.stream)
            else:
                tip.tensor.view(torch.uint8).copy_(source)
            self.ring.release(segment, event)
            segment = None
            self._add(thread_read_seconds=read - started, thread_hash_seconds=hashed - read,
                      thread_copy_seconds=_now() - hashed, blocks=len(blocks),
                      block_bytes=sum(take + len(tail) for _, _, take, tail in blocks))
            tip.ticket._set(event)
        except BaseException as exc:  # noqa: BLE001 - delivered by the ticket and StagedHistory.release
            if segment is not None:
                self.ring.release(segment, None)
            tip.ticket._fail(exc)
            raise

    def _verification_failure(self, store, packs, verify):
        """After a failed staging: the first hash mismatch among the payloads read so far (an IntegrityError: the
        error to report, its object quarantined), or None. They are hashed here: their verification starts only once
        staging succeeded."""
        if verify:
            for pack in packs:
                for name, data in (pack.get("payloads") or {}).items():
                    try:
                        self._check_object(store, pack["packet"]["payloads"][name], data)
                    except IntegrityError as exc:
                        return exc
        return None

    def _submit_verifications(self, store, packs, staged):
        """Hash every payload on the verifier threads once staging read them all, the largest first (so reads never
        compete with hashing; the history's verification ends as early as its largest object allows); small
        objects of a pack in one task."""
        tasks = []
        for pack in packs:
            small = []
            streamed = pack.get("streamed") or {}
            staged.verifications.extend(streamed.values())  # W1.2: hashed while read (_verify_stream)
            for name, data in pack["payloads"].items():
                if name in streamed:
                    continue
                item = (pack["packet"]["payloads"][name], data)
                if len(data) >= _VERIFY_ALONE_BYTES:
                    tasks.append((len(data), [item]))
                else:
                    small.append(item)
            if small:
                tasks.append((sum(len(data) for _, data in small), small))
        tasks.sort(key=lambda task: -task[0])
        staged.verifications.extend(self._verify.submit(self._verify_objects, store, items) for _, items in tasks)

    # -- staging -------------------------------------------------------------------------------------------
    def stage(self, runtime, history, on_prepared=None):
        """Stage the node packs of ``history`` (``LineageRestorer.history`` items); returns a :class:`StagedHistory`.

        Returns once every pack is prepared and every object was read; device copies (tile ``ready`` tickets) and
        verification continue on the workers until :meth:`StagedHistory.wait` (the integrity gate). Holds the
        runtime lock and a store transaction while objects are read.

        ``on_prepared`` (optional; decomp_opt 2026-10-25): called on this thread with each pack's ``NodeRef`` as soon
        as the pack is prepared, while the following packs are still being read and prepared (e.g. to submit its
        decode jobs at once: ``LineageRestorer(pipelined_staging=True)``). An exception it raises fails the staging.
        """
        from .packed_decode import NodeRef

        if self._closed:
            raise StateError("the staging engine is closed")
        if on_prepared is not None and not callable(on_prepared):
            raise ConfigurationError("on_prepared must be callable")
        history = list(history)
        if not history:
            raise ConfigurationError("staging needs a nonempty history")
        started = _now()
        staged = StagedHistory(engine=self)
        store = runtime.store
        verify = bool(store.verify_reads)
        packs, tips, started_tips, shared = [], [], [], []
        callback_seconds = 0.0
        with normal_tensors(), runtime._lock, runtime.store.transaction():
            try:
                packs, tips = self._plan(runtime, history)
                shared = [tip for tip in tips if tip.alias is not None]  # W1.2: staged by the tip they alias
                tips = [tip for tip in tips if tip.alias is None]
                planned = _now()
                for tip in tips:
                    tip.tensor = torch.empty((tip.rows, tip.channels), dtype=torch.float16, device=self.device)
                    tip.ticket = StagingTicket(tip.ref[:24])
                    tip.read_done = threading.Event()
                for tip in shared:
                    tip.tensor, tip.ticket, tip.read_done = tip.alias.tensor, tip.alias.ticket, tip.alias.read_done
                if self.stream is not None:
                    for tip in tips:
                        tip.tensor.record_stream(self.stream)
                    # Tip tensors may reuse memory the current stream freed: the staging copies wait for it.
                    self.stream.wait_stream(torch.cuda.current_stream(self.device))
                allocated = _now()
                for pack in packs:
                    # by schema: the HEVC node pack's or (W1.2) the factor pack's validation / preparation
                    pack["validate"], pack["prepare"], pack["stream_verify"] = _pack_functions(
                        pack["packet"]["metadata"])
                    pack["futures"] = self._submit_payloads(pack, verify, store)
                for tip in tips:
                    future = self._bases.submit(self._stage_tip, tip, store)
                    started_tips.append((tip, future))
                    staged.futures.append(future)
                    staged.tickets.append(tip.ticket)
                submitted = _now()
                # Validate every pack's metadata while its payloads are still being read: same-length zero
                # placeholders (calloc'd, never touched) stand in for the bytes (verify=False checks lengths only).
                for pack in packs:
                    pack["validated"] = self._validate_ahead(pack["validate"], pack)
                validated = _now()
                waited = prepared = 0.0
                for pack in packs:
                    mark = _now()
                    payloads = {}
                    for future in pack["futures"]:
                        payloads.update(future.result())
                    pack["payloads"] = payloads
                    waited += _now() - mark
                    mark = _now()
                    bases = [(tip.tensor[:rank], tip.ticket) for tip, rank in pack["assign"]]
                    packet = pack["prepare"](_pack_view(pack["packet"]["metadata"], payloads), bases=bases,
                                             executor=self._scans, validated=pack["validated"])
                    prepared += _now() - mark
                    ref = pack["item"]["packet"]
                    staged.prepared[ref] = packet
                    staged.bindings[ref] = tuple((version_ref, rank) for version_ref, rank, *_ in pack["tiles"])
                    staged.refs.append(NodeRef(pack["item"]["node_id"], [packet]))
                    if on_prepared is not None:
                        mark = _now()
                        on_prepared(staged.refs[-1])
                        callback_seconds += _now() - mark
                mark = _now()
                for tip in tips:
                    tip.read_done.wait()
                reads_tail = _now() - mark
                if verify and self._verify is not None:
                    self._submit_verifications(store, packs, staged)
            except BaseException as exc:
                # No task may read a store object once the transaction ends: cancel what has not started and wait
                # for the reads of what has.
                for tip, future in started_tips:
                    if not future.cancel():
                        tip.read_done.wait()
                for pack in packs:
                    for future in pack.get("futures", []):
                        if not future.cancel():
                            future.exception()
                    for reader in pack.get("readers", []):
                        reader.close()
                mismatch = self._verification_failure(store, packs, verify)
                if mismatch is not None and mismatch is not exc:
                    # Preparing unverified bytes failed (e.g. a structural check): the hash mismatch behind it is the
                    # error to report (its object is quarantined).
                    raise mismatch from exc
                raise
        for tip in tips + shared:  # a shared tip's complete version is served by its alias's rows
            if tip.rows == tip.chain[-1][1].total_rank:
                staged.versions.append((tip.ref, [ref for ref, _ in tip.chain], tip.tensor, tip.ticket))
        finished = _now()
        staged.stats = {
            "seconds": finished - started, "plan_seconds": planned - started, "allocate_seconds": allocated - planned,
            "submit_seconds": submitted - allocated, "validate_ahead_seconds": validated - submitted,
            "payload_wait_seconds": waited, "prepare_seconds": prepared, "on_prepared_seconds": callback_seconds,
            "reads_tail_seconds": reads_tail, "packs": len(packs), "tips": len(tips), "shared_tips": len(shared),
            "tip_bytes": sum(tip.nbytes for tip in tips), "complete_versions": len(staged.versions),
            "verify_payloads": verify, "verify_async": self._verify is not None,
            "verification": {"mode": self.verification_mode(store),
                             "verified_payloads": sum(len(pack["payloads"]) for pack in packs) if verify else 0,
                             "payload_wait_seconds": waited},
        }
        self._add(histories=1, packs=len(packs), tips=len(tips), tip_bytes=staged.stats["tip_bytes"],
                  seconds=finished - started)
        return staged

    def stats(self):
        with self._lock:
            return {**self.stats_, "ring": dict(self.ring.stats), "ring_bytes": self.ring.capacity}

    def close(self):
        if self._closed:
            return
        self._closed = True
        for pool, _ in self._pools():
            pool.shutdown(wait=True)


def _verified_since(store, since):
    counter = getattr(store, "verified_payloads", None)
    if since is None or counter is None:
        return 0
    return max(0, int(counter()) - int(since))


def store_verification_report(store, since=None):
    """The ``verification`` record of a restore without a staging engine (store path, e.g. the CPU
    ``SoftwareLineageRestorer``): ``mode`` ``"sync"`` (the store hash-checks every payload before it is used) or
    ``"off"``, ``verified_payloads`` hash-checked for the calling thread since ``since`` (an
    :meth:`ObjectStore.verified_payloads` value taken when the restore began), no staging wait."""
    return {"mode": "sync" if getattr(store, "verify_reads", False) else "off",
            "verified_payloads": _verified_since(store, since), "payload_wait_seconds": 0.0}


__all__ = ["PinnedRing", "StagedHistory", "StagingEngine", "StagingTicket", "store_verification_report"]
