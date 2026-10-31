"""Immutable content-addressed objects with transactional roots and GC pins."""

from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import CancelledError, Future, ThreadPoolExecutor
import contextlib
import fcntl
import hashlib
import os
from pathlib import Path
import re
import sqlite3
import threading
import time
from typing import Any, Iterable, Iterator, Mapping
import uuid
import weakref

from . import accel_c as _accel_c
from .errors import CapacityError, ConfigurationError, IntegrityError, QuarantinedError, StateError
from .serialization import canonical_json, parse_json
from .telemetry import span


_OBJECT_ID = re.compile(r"^(blob|basis|manifest):([0-9a-f]{64})$")
_LOCKS_GUARD = threading.Lock()
_ROOT_LOCKS: dict[str, threading.RLock] = {}
_ANY = object()
#: Workload v2 (CONTRACT §10, ``store_io``): object file I/O policies. ``buffered`` (default, F6) leaves the OS page
#: cache alone. ``nocache`` drops an object file's pages (``posix_fadvise(POSIX_FADV_DONTNEED)``) after its durable
#: write (the data is fsynced, i.e. written back, first) and after every read of it, here and by readers that open
#: object files themselves (:meth:`ObjectStore.drop_read_cache`), so the page cache never serves as an unbudgeted
#: extra tier. The SQLite index is not an object store file and keeps its pages.
IO_POLICIES = ("buffered", "nocache")


class EvictedObjectError(StateError):
    """W1.4.3 (E1): a read of an object an archive budget evicted (:meth:`ObjectStore.evict`): its payload is gone,
    its index row and the edges to it stay, so the graph above it remains whole. Never a corruption (no quarantine)."""


def _process_stamp(pid: int) -> str | None:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
        return stat[stat.rfind(")") + 2:].split()[19]
    except (OSError, IndexError):
        return None


def _alive(pid: int, stamp: str | None) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    current = _process_stamp(pid)
    return stamp is None or current is None or stamp == current


class ObjectLease:
    """A durable GC pin; tensor lifetime is handled separately by residency."""

    def __init__(self, store: ObjectStore, lease_id: str, object_ids: tuple[str, ...]) -> None:
        self._store: ObjectStore | None = store
        self.lease_id = lease_id
        self.object_ids = object_ids

    @property
    def active(self) -> bool:
        return self._store is not None and not self._store.closed

    def release(self) -> None:
        store, self._store = self._store, None
        if store is not None and not store.closed:
            store._release_lease(self.lease_id)

    close = release

    def __enter__(self) -> ObjectLease:
        if not self.active:
            raise StateError("object lease is no longer active")
        return self

    def __exit__(self, *args: Any) -> None:
        self.release()

    def __del__(self) -> None:
        try:
            self.release()
        except Exception:
            pass


class ReadAhead:
    """A bounded, ordered background read-ahead of immutable objects for one reader that :meth:`take` s them.

    :meth:`schedule` queues objects (sizes from one index query; ids the index lacks are skipped, so the reader's own
    read reports them). Reads start in queue order on the store's read-ahead threads while the bytes read ahead and
    not yet taken stay within ``max_bytes`` (a larger object starts only when nothing else is held). Each read applies
    exactly the checks of :meth:`ObjectStore.get` -- with ``verify_reads`` kind, length and SHA-256 against the
    content address -- in its thread, so verified reads of different objects hash in parallel (hashlib releases the
    GIL). :meth:`take` returns an object's checked bytes, waiting while its read runs, or None when it was not read
    ahead or its read failed: the reader then reads the object itself, so every error surfaces unchanged at the read
    that uses the object and unchecked bytes are never returned.
    """

    def __init__(self, store: ObjectStore, max_bytes: int) -> None:
        if type(max_bytes) is not int or max_bytes < 1:
            raise ConfigurationError("read-ahead max_bytes must be a positive integer")
        self.store, self.max_bytes = store, max_bytes
        self._lock = threading.Lock()
        self._queue: OrderedDict[str, tuple[str, int]] = OrderedDict()  # not started: id -> (kind, size)
        self._started: dict[str, tuple[Future, int]] = {}
        self._held = 0  # bytes of started reads not taken yet
        self._closed = False
        self.last_used = time.monotonic()
        self.stats_ = {"scheduled": 0, "started": 0, "taken": 0, "taken_bytes": 0, "failed": 0, "missed": 0,
                       "dropped": 0, "peak_held_bytes": 0}

    def schedule(self, object_ids: Iterable[str]) -> int:
        """Queue objects (in order; repeated or already queued ids once) and start reads within the bound."""
        ids = [object_id for object_id in dict.fromkeys(object_ids)]
        for object_id in ids:
            self.store.path_for(object_id)
        with self.store.transaction():
            rows = self.store._index_rows(ids, "kind, size")
        with self._lock:
            if self._closed:
                return 0
            added = 0
            for object_id in ids:
                if object_id in rows and object_id not in self._queue and object_id not in self._started:
                    self._queue[object_id] = rows[object_id]
                    added += 1
            self.stats_["scheduled"] += added
            self.last_used = time.monotonic()
            self._fill()
            return added

    def _fill(self) -> None:
        """Start queued reads within the bound. Caller holds ``_lock``."""
        while self._queue and not self._closed:
            object_id, (kind, size) = next(iter(self._queue.items()))
            if self._started and self._held + size > self.max_bytes:
                return
            try:
                future = self.store._read_ahead_pool().submit(
                    self.store._read_checked, object_id, kind, size, self.store.verify_reads)
            except (RuntimeError, StateError):  # the store closed: the reader reads (and reports) itself
                return
            del self._queue[object_id]
            self._started[object_id] = (future, size)
            self._held += size
            self.stats_["started"] += 1
            self.stats_["peak_held_bytes"] = max(self.stats_["peak_held_bytes"], self._held)

    def take(self, object_id: str) -> bytes | None:
        """The checked bytes of a read-ahead object (once), or None: the caller reads it itself."""
        with self._lock:
            self.last_used = time.monotonic()
            entry = self._started.pop(object_id, None)
            if entry is None:
                if self._queue.pop(object_id, None) is not None:
                    self.stats_["missed"] += 1
                return None
            self._held -= entry[1]
            self._fill()
        try:
            data = entry[0].result()
        except (Exception, CancelledError):  # e.g. IntegrityError: the caller's own read raises it where it is used
            with self._lock:
                self.stats_["failed"] += 1
            return None
        with self._lock:
            self.stats_["taken"] += 1
            self.stats_["taken_bytes"] += len(data)
        if self.store.verify_reads:
            self.store._count_verified(object_id)  # hash-checked by the read; credited to the thread using the bytes
        return data

    @property
    def held_bytes(self) -> int:
        return self._held

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {**self.stats_, "queued": len(self._queue), "held": len(self._started), "held_bytes": self._held}

    def close(self) -> None:
        """Drop queued reads and read-ahead bytes not taken (running reads finish and are discarded)."""
        with self._lock:
            self._closed = True
            self.stats_["dropped"] += len(self._queue) + len(self._started)
            self._queue.clear()
            for future, _ in self._started.values():
                future.cancel()
            self._started.clear()
            self._held = 0


class ObjectStore:
    """Owns immutable blob/basis/manifest namespaces and atomic named roots.

    All publication and GC operations serialize through a per-root process lock
    and an advisory filesystem lock. A transaction may leave unreachable bytes
    after interruption, but can never expose a partially published root.

    Integrity modes: ``verify_reads`` hash-checks every object read for use
    (``get``, ``get_many``, manifests). ``verify_references`` makes graph
    operations (reference updates, leases, manifest writes, deduplication and
    GC liveness walks) also read and hash every object they reach instead of
    checking the index; ``None`` follows ``verify_reads`` (the original strict
    mode, whose publication cost grows with the reachable bytes).
    ``verify_reads=True, verify_references=False`` verifies what is read and
    keeps graph operations at their index-check cost. ``verify()`` is always a
    full forced audit.

    ``on_damage`` is the garbage-collection policy for damaged graphs found by
    its liveness walk (a corrupt or missing manifest, a missing index row):
    ``"raise"`` fails the collection (the original behavior); ``"keep"`` keeps
    everything the index records as reachable from the damaged object (its
    edges, without reading it), collects the rest, and lists the damaged
    objects in :attr:`damaged`. Healthy stores collect identically.

    Object quarantine (freeze D, D-E; process-local, per store instance): an
    object whose bytes failed a hash check -- a verified read here, or a
    reader's own check such as the restore staging verifier
    (:meth:`quarantine`) -- is listed in :attr:`quarantined_objects`, and every
    later read of it is verified before its bytes are returned, whatever
    ``verify_reads`` says (``get``, ``get_many``, manifests, read-ahead; a
    reader that verifies asynchronously calls :meth:`check_quarantined` before
    it uses bytes unverified): while its bytes still fail, the read raises
    :class:`~reelcache.errors.QuarantinedError` (an IntegrityError) and nothing is
    decoded from them; once they verify (a repaired file, an audit with
    :meth:`verify`, or a put that writes the object again) the quarantine is
    lifted.
    """

    def __init__(self, root: str | os.PathLike[str], *, max_bytes: int | None = None,
                 verify_reads: bool = False, verify_references: bool | None = None,
                 on_damage: str = "raise") -> None:
        if max_bytes is not None and (type(max_bytes) is not int or max_bytes < 0):
            raise ConfigurationError("max_bytes must be a nonnegative integer")
        if type(verify_reads) is not bool:
            raise ConfigurationError("verify_reads must be boolean")
        if verify_references is not None and type(verify_references) is not bool:
            raise ConfigurationError("verify_references must be boolean or None (follow verify_reads)")
        if on_damage not in ("raise", "keep"):
            raise ConfigurationError("on_damage must be 'raise' or 'keep'")
        self.verify_reads = verify_reads
        self.verify_references = verify_reads if verify_references is None else verify_references
        self.on_damage = on_damage
        self._damaged: dict[str, str] = {}
        self._quarantined: dict[str, str] = {}  # object id -> the failed check (see the class docstring)
        self._quarantine_lock = threading.Lock()  # not the store lock: verifier threads report while a reader holds it
        self._verified_local = threading.local()  # verified payload reads per calling thread (verified_payloads)
        self.root = Path(root).resolve()
        self.max_bytes = max_bytes
        self.root.mkdir(parents=True, exist_ok=True)
        for kind in ("blob", "basis", "manifest"):
            (self.root / "objects" / kind).mkdir(parents=True, exist_ok=True)
        (self.root / "staging").mkdir(exist_ok=True)
        with _LOCKS_GUARD:
            self._lock = _ROOT_LOCKS.setdefault(str(self.root), threading.RLock())
        had_lock_file = (self.root / "store.lock").exists()
        self._lock_file = (self.root / "store.lock").open("a+b")
        self._closed = False
        self._depth = 0
        self._rollback_only = False
        self._created: list[str] = []
        # Objects written or found already stored in the current outer transaction (in order, unique).
        self._touched: dict[str, None] = {}
        # (event, object_id) of the current outer transaction, delivered to ``hooks`` after it commits.
        self._events: list[tuple[str, str]] = []
        # Test seam: ``hooks(event, object_id)`` runs after each committed outer transaction, outside every store
        # lock, once per event: "created" (new object), "existing" (a put found the object stored), "deleted" (GC).
        self.hooks: Any = None
        self._dirty_directories: set[Path] = set()
        # Manifests whose complete subgraph was walked (process-local; committed with the transaction).
        self._complete_manifests: set[str] = set()
        self._pending_complete: set[str] = set()
        self._recovered_staging: set[str] = set()
        self._owner = uuid.uuid4().hex
        self._leases: weakref.WeakSet[ObjectLease] = weakref.WeakSet()
        self._readers: ThreadPoolExecutor | None = None  # get_many's file readers (created on first use)
        self._read_ahead_readers: ThreadPoolExecutor | None = None  # ReadAhead's readers (created on first use)
        self._read_ahead_guard = threading.Lock()
        self.io_policy = "buffered"  # workload v2 ``store_io`` (set_io_policy)
        self._io_lock = threading.Lock()
        self.io_stats_ = {"read_calls": 0, "read_bytes": 0, "write_calls": 0, "write_bytes": 0, "existing_calls": 0,
                          "existing_bytes": 0, "errors": 0, "last_error": None}
        self._evicted: set[str] = set()  # W1.4.3 (E1): objects whose payload an archive budget evicted (evict)
        self._unevicted: list[str] = []  # W1.4.3 (E1): evicted objects written again in the current transaction
        try:
            with self._lock:
                fcntl.flock(self._lock_file, fcntl.LOCK_EX)
                try:
                    index_path = self.root / "index.sqlite3"
                    existing_index = index_path.exists()
                    if not existing_index and (
                        had_lock_file
                        or any((self.root / "objects").glob("*/*"))
                        or any((self.root / "staging").iterdir())
                    ):
                        raise IntegrityError("object store index is missing; refusing to reset existing state")
                    if existing_index and index_path.stat().st_size < 100:
                        raise IntegrityError("object store index is truncated")
                    self._db = sqlite3.connect(
                        index_path, timeout=60, isolation_level=None,
                        check_same_thread=False,
                    )
                    if existing_index:
                        tables = {row[0] for row in self._db.execute(
                            "SELECT name FROM sqlite_master WHERE type='table'"
                        )}
                        if not {"settings", "objects", "edges", "refs", "owners", "leases"} <= tables:
                            raise IntegrityError("object store index schema is incomplete")
                        version = self._db.execute(
                            "SELECT value FROM settings WHERE key='format'"
                        ).fetchone()
                        if version != ("reelcache.store.v1",):
                            raise IntegrityError("unsupported or missing object store format")
                    self._db.execute("PRAGMA foreign_keys=ON")
                    self._db.execute("PRAGMA journal_mode=WAL")
                    self._db.execute("PRAGMA synchronous=FULL")
                    self._db.executescript("""
                        CREATE TABLE IF NOT EXISTS settings (
                            key TEXT PRIMARY KEY, value TEXT NOT NULL
                        );
                        CREATE TABLE IF NOT EXISTS objects (
                            id TEXT PRIMARY KEY, kind TEXT NOT NULL, size INTEGER NOT NULL
                        );
                        CREATE TABLE IF NOT EXISTS edges (
                            parent TEXT NOT NULL REFERENCES objects(id),
                            child TEXT NOT NULL REFERENCES objects(id),
                            PRIMARY KEY(parent, child)
                        );
                        CREATE TABLE IF NOT EXISTS refs (
                            name TEXT PRIMARY KEY, object_id TEXT NOT NULL REFERENCES objects(id)
                        );
                        CREATE TABLE IF NOT EXISTS owners (
                            id TEXT PRIMARY KEY, pid INTEGER NOT NULL, stamp TEXT
                        );
                        CREATE TABLE IF NOT EXISTS leases (
                            id TEXT NOT NULL, owner TEXT NOT NULL REFERENCES owners(id),
                            object_id TEXT NOT NULL REFERENCES objects(id),
                            PRIMARY KEY(id, object_id)
                        );
                        CREATE TABLE IF NOT EXISTS telemetry (
                            object_id TEXT PRIMARY KEY REFERENCES objects(id) ON DELETE CASCADE,
                            data TEXT NOT NULL
                        );
                    """)
                    version = self._db.execute(
                        "SELECT value FROM settings WHERE key='format'"
                    ).fetchone()
                    if version is not None and version[0] != "reelcache.store.v1":
                        raise IntegrityError("unsupported object store format")
                    self._db.execute(
                        "INSERT OR IGNORE INTO settings VALUES ('format', 'reelcache.store.v1')"
                    )
                    if self._db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='evicted'"
                                        ).fetchone() is not None:  # W1.4.3 (E1): only stores that evicted have it
                        self._evicted = {row[0] for row in self._db.execute("SELECT id FROM evicted")}
                finally:
                    fcntl.flock(self._lock_file, fcntl.LOCK_UN)
            with self.transaction():
                self._reap_owners()
                self._db.execute("INSERT INTO owners VALUES (?, ?, ?)",
                                 (self._owner, os.getpid(), _process_stamp(os.getpid())))
                self._recover_files()
            if self._recovered_staging:
                self.collect_garbage(candidates=self._recovered_staging)
        except sqlite3.Error as exc:
            self._closed = True
            if hasattr(self, "_db"):
                self._db.close()
            self._lock_file.close()
            raise IntegrityError("cannot open the object store index") from exc
        except BaseException:
            self._closed = True
            if hasattr(self, "_db"):
                self._db.close()
            self._lock_file.close()
            raise

    @property
    def closed(self) -> bool:
        return self._closed

    @contextlib.contextmanager
    def transaction(self) -> Iterator[ObjectStore]:
        events: list[tuple[str, str]] = []
        with self._lock:
            if self._closed:
                raise StateError("object store is closed")
            outermost = self._depth == 0
            if outermost:
                fcntl.flock(self._lock_file, fcntl.LOCK_EX)
                try:
                    self._db.execute("BEGIN IMMEDIATE")
                except sqlite3.Error as exc:
                    fcntl.flock(self._lock_file, fcntl.LOCK_UN)
                    raise IntegrityError("cannot start an object store transaction") from exc
                self._rollback_only = False
                self._created = []
                self._touched = {}
                self._events = []
                self._unevicted = []
            self._depth += 1
            try:
                yield self
                if outermost:
                    if self._rollback_only:
                        raise StateError("a failed nested store operation invalidated the transaction")
                    for directory in sorted(self._dirty_directories):
                        self._sync_directory(directory)
                    with span("store.commit"):
                        self._db.execute("COMMIT")
                    self._complete_manifests.update(self._pending_complete)
                    if self._unevicted:  # W1.4.3 (E1): payloads written again are readable from now on
                        self._evicted.difference_update(self._unevicted)
                    if self.hooks is not None:
                        events = self._events
            except BaseException as exc:
                self._rollback_only = True
                if outermost and self._db.in_transaction:
                    self._db.execute("ROLLBACK")
                if isinstance(exc, sqlite3.Error):
                    raise IntegrityError("object store index operation failed") from exc
                raise
            finally:
                self._depth -= 1
                if outermost:
                    self._dirty_directories.clear()
                    self._pending_complete.clear()
                    self._events = []
                    fcntl.flock(self._lock_file, fcntl.LOCK_UN)
        hooks = self.hooks
        for event, object_id in events:
            hooks(event, object_id)

    def path_for(self, object_id: str) -> Path:
        match = _OBJECT_ID.fullmatch(object_id) if isinstance(object_id, str) else None
        if match is None:
            raise IntegrityError(f"invalid immutable object id: {object_id!r}")
        return self.root / "objects" / match[1] / match[2]

    # -- workload v2 (CONTRACT §10): the object file I/O policy (``IO_POLICIES``) -------------------------------------
    def set_io_policy(self, policy: str, *, drop_existing: bool = False) -> dict[str, Any]:
        """Set the object file I/O policy. ``drop_existing`` (with ``nocache``) also drops the pages of every stored
        object file now, e.g. pages an earlier process left cached. Returns :meth:`io_stats`."""
        if policy not in IO_POLICIES:
            raise ConfigurationError(f"store io policy must be one of {IO_POLICIES}, got {policy!r}")
        self.io_policy = policy
        if drop_existing and policy == "nocache":
            for kind in ("blob", "basis", "manifest"):
                for path in (self.root / "objects" / kind).iterdir():
                    try:
                        fd = os.open(path, os.O_RDONLY)
                    except OSError:
                        continue  # collected meanwhile
                    try:
                        self._dontneed(fd, os.fstat(fd).st_size, "existing")
                    finally:
                        os.close(fd)
        return self.io_stats()

    def io_stats(self) -> dict[str, Any]:
        """The policy and the pages it dropped: calls and file bytes after reads, durable writes and at
        ``set_io_policy(drop_existing=True)``; failed advice calls (the data stays correct, only cached)."""
        with self._io_lock:
            return {"policy": self.io_policy, **self.io_stats_}

    def drop_read_cache(self, fd: int, nbytes: int) -> None:
        """For readers that open object files themselves (restore staging): under ``nocache`` drop the pages of the
        object file just read through ``fd``; a no-op under ``buffered``."""
        if self.io_policy == "nocache":
            self._dontneed(fd, nbytes, "read")

    def _dontneed(self, fd: int, nbytes: int, what: str) -> None:
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        except (OSError, AttributeError) as exc:
            with self._io_lock:
                self.io_stats_["errors"] += 1
                self.io_stats_["last_error"] = f"{type(exc).__name__}: {exc}"
            return
        with self._io_lock:
            self.io_stats_[f"{what}_calls"] += 1
            self.io_stats_[f"{what}_bytes"] += int(nbytes)

    def _read_file(self, path: Path) -> bytes:
        """An object file's bytes; under ``nocache`` its pages are dropped after the read."""
        if self.io_policy != "nocache":
            return path.read_bytes()
        with open(path, "rb", buffering=0) as handle:
            data = handle.readall()
            self._dontneed(handle.fileno(), len(data), "read")
        return data

    def _get(self, object_id: str, *, verify: bool | None = None) -> bytes:
        """Read an indexed object; hash/length checks run only when verifying.

        Default reads follow ``verify_reads``. ``verify()`` always forces the check. A quarantined object is
        always verified (see the class docstring).
        """
        verify = self.verify_reads if verify is None else verify
        path = self.path_for(object_id)
        if self._evicted and object_id in self._evicted:
            raise EvictedObjectError(f"object {object_id} was evicted by the archive budget")
        quarantined = bool(self._quarantined) and object_id in self._quarantined
        row = self._db.execute("SELECT kind, size FROM objects WHERE id=?", (object_id,)).fetchone()
        if row is None:
            raise IntegrityError(f"missing immutable object: {object_id}")
        try:
            with span("store.read", nbytes=row[1]):
                data = self._read_file(path)
        except OSError as exc:
            raise IntegrityError(f"cannot read immutable object: {object_id}") from exc
        if verify or quarantined:
            kind, digest = object_id.split(":")
            with span("store.hash", nbytes=len(data)):
                self._checked(object_id, row[0] == kind and row[1] == len(data)
                              and hashlib.sha256(data).hexdigest() == digest)
            self._count_verified(object_id)
        return data

    def _require(self, object_id: str) -> None:
        """Existence check; reads and hashes the object only with ``verify_references``."""
        if self.verify_references:
            self._get(object_id, verify=True)
        elif self._db.execute("SELECT 1 FROM objects WHERE id=?", (object_id,)).fetchone() is None:
            raise IntegrityError(f"missing immutable object: {object_id}")

    #: SQLite's default bound on host parameters per statement is 999; index queries use chunks below it.
    _QUERY_CHUNK = 900
    #: Parallel file readers of :meth:`get_many` (reads and hash checks release the GIL).
    READ_WORKERS = 8
    #: Parallel background readers shared by this store's :class:`ReadAhead` windows.
    READ_AHEAD_WORKERS = 4

    def _read_checked(self, object_id: str, kind: str, size: int, verify: bool) -> bytes:
        """The file read and checks of :meth:`get` for an indexed ``(kind, size)`` row (no index access; any thread)."""
        if self._evicted and object_id in self._evicted:
            raise EvictedObjectError(f"object {object_id} was evicted by the archive budget")
        try:
            data = self._read_file(self.path_for(object_id))
        except OSError as exc:
            raise IntegrityError(f"cannot read immutable object: {object_id}") from exc
        if verify or (self._quarantined and object_id in self._quarantined):
            expected_kind, digest = object_id.split(":")
            self._checked(object_id, kind == expected_kind and size == len(data)
                          and hashlib.sha256(data).hexdigest() == digest)
        return data

    # -- verified payload reads (records only; freeze D convention ``verified_payloads``) ----------------------
    def _count_verified(self, object_id: str) -> None:
        if object_id.startswith("blob:"):
            local = self._verified_local
            local.payloads = getattr(local, "payloads", 0) + 1

    def verified_payloads(self) -> int:
        """Payload (``blob``) objects this store hash-checked for the calling thread so far -- reads it made itself
        (verified reads, and re-checks of quarantined objects), :meth:`get_many` batches with verification (checked on
        reader threads, counted for the caller) and read-ahead objects it took from a verifying store. Monotonic per
        thread: a restore's count is the difference across it (records only)."""
        return getattr(self._verified_local, "payloads", 0)

    # -- object quarantine (process-local; see the class docstring) ------------------------------------------
    def _checked(self, object_id: str, ok: bool) -> None:
        """The outcome of a hash/length check of ``object_id``'s bytes: a failure quarantines it and raises (a
        QuarantinedError when it was quarantined already), a success lifts its quarantine."""
        if ok:
            if self._quarantined:
                self.lift_quarantine(object_id)
            return
        message = f"immutable object hash or length mismatch: {object_id}"
        with self._quarantine_lock:
            first = self._quarantined.setdefault(object_id, message)
        if first is not message:
            raise QuarantinedError(f"immutable object {object_id} is quarantined: its bytes failed a check in this "
                                   f"process (IntegrityError: {first}) and fail it again")
        raise IntegrityError(message)

    def quarantine(self, object_id: str, reason: str) -> None:
        """Quarantine an object whose bytes failed a check (idempotent; the first reason is kept). Any thread."""
        self.path_for(object_id)
        with self._quarantine_lock:
            self._quarantined.setdefault(object_id, str(reason))

    def lift_quarantine(self, object_id: str) -> bool:
        """Forget an object's quarantine; returns whether it was quarantined."""
        with self._quarantine_lock:
            return self._quarantined.pop(object_id, None) is not None

    @property
    def quarantined_objects(self) -> dict[str, str]:
        """``{object_id: failed check}`` of the objects quarantined by this store instance."""
        with self._quarantine_lock:
            return dict(self._quarantined)

    def check_quarantined(self, object_ids: Iterable[str]) -> None:
        """Re-verify the quarantined objects among ``object_ids`` now (a reader's check before it uses their bytes
        unverified): QuarantinedError for the first whose bytes still fail; those that verify are lifted."""
        if not self._quarantined:
            return
        with self.transaction():
            for object_id in dict.fromkeys(object_ids):
                if object_id in self._quarantined:
                    self._get(object_id, verify=True)

    def _read_ahead_pool(self) -> ThreadPoolExecutor:
        with self._read_ahead_guard:  # not the store lock: windows start reads while a transaction reads elsewhere
            if self._closed:
                raise StateError("object store is closed")
            if self._read_ahead_readers is None:
                self._read_ahead_readers = ThreadPoolExecutor(max_workers=self.READ_AHEAD_WORKERS,
                                                              thread_name_prefix="reelcache-store-read-ahead")
            return self._read_ahead_readers

    def read_ahead(self, object_ids: Iterable[str] = (), *, max_bytes: int) -> ReadAhead:
        """A :class:`ReadAhead` window of at most ``max_bytes`` over ``object_ids`` (more may be scheduled later)."""
        window = ReadAhead(self, max_bytes)
        window.schedule(object_ids)
        return window

    def _index_rows(self, object_ids: list[str], columns: str) -> dict[str, tuple]:
        """``{id: row}`` of ``columns`` for indexed objects among ``object_ids`` (chunked ``IN`` queries)."""
        rows: dict[str, tuple] = {}
        for start in range(0, len(object_ids), self._QUERY_CHUNK):
            part = object_ids[start:start + self._QUERY_CHUNK]
            query = f"SELECT id, {columns} FROM objects WHERE id IN ({','.join('?' * len(part))})"
            for row in self._db.execute(query, part):
                rows[row[0]] = row[1:]
        return rows

    def object_sizes(self, object_ids: Iterable[str]) -> dict[str, int]:
        """Indexed byte sizes of many objects in one index query; IntegrityError names the first missing one."""
        ids = list(dict.fromkeys(object_ids))
        for object_id in ids:
            self.path_for(object_id)
        with self.transaction():
            rows = self._index_rows(ids, "size")
        missing = next((object_id for object_id in ids if object_id not in rows), None)
        if missing is not None:
            raise IntegrityError(f"missing immutable object: {missing}")
        return {object_id: rows[object_id][0] for object_id in ids}

    def indexed(self, object_ids: Iterable[str]) -> set[str]:
        """The objects among ``object_ids`` that the index holds (chunked index queries; absent ids are no error)."""
        ids = list(dict.fromkeys(object_ids))
        for object_id in ids:
            self.path_for(object_id)
        with self.transaction():
            return set(self._index_rows(ids, "size"))

    def get_many(self, object_ids: Iterable[str], *, verify: bool | None = None,
                 workers: int | None = None) -> list[bytes]:
        """Read many indexed objects: one index query, then parallel file reads (and hash/length checks).

        The batched form of :meth:`get` for restore staging: results follow ``object_ids`` (a repeated id is read
        once); ``verify`` defaults to ``verify_reads`` (``verify()`` semantics per object when true); ``workers``
        bounds the reader threads (default :attr:`READ_WORKERS`; 1 reads in the calling thread). Reads run inside
        one store transaction, so garbage collection cannot remove an object mid-batch.
        """
        verify = self.verify_reads if verify is None else verify
        if workers is not None and (type(workers) is not int or workers < 1):
            raise ConfigurationError("get_many workers must be a positive integer")
        ids = list(object_ids)
        unique = list(dict.fromkeys(ids))
        for object_id in unique:
            self.path_for(object_id)

        def read(object_id: str) -> bytes:
            kind, size = rows[object_id]
            return self._read_checked(object_id, kind, size, verify)
        with self.transaction():
            rows = self._index_rows(unique, "kind, size")
            missing = next((object_id for object_id in unique if object_id not in rows), None)
            if missing is not None:
                raise IntegrityError(f"missing immutable object: {missing}")
            count = min(len(unique), workers or self.READ_WORKERS)
            with span("store.read", nbytes=sum(rows[object_id][1] for object_id in unique)):
                if count <= 1:
                    data = [read(object_id) for object_id in unique]
                else:
                    if self._readers is None:
                        self._readers = ThreadPoolExecutor(max_workers=self.READ_WORKERS,
                                                           thread_name_prefix="reelcache-store-read")
                    futures = [self._readers.submit(read, object_id) for object_id in unique]
                    data = [future.result() for future in futures]
        if verify:
            for object_id in unique:
                self._count_verified(object_id)  # hash-checked on a reader thread, for this caller
        by_id = dict(zip(unique, data))
        return [by_id[object_id] for object_id in ids]

    def get(self, object_id: str) -> bytes:
        with self.transaction():
            return self._get(object_id)

    get_bytes = get

    def object_size(self, object_id: str) -> int:
        """Return indexed bytes; get/verify additionally checks file contents."""
        self.path_for(object_id)
        with self.transaction():
            row = self._db.execute("SELECT size FROM objects WHERE id=?", (object_id,)).fetchone()
            if row is None:
                raise IntegrityError(f"missing immutable object: {object_id}")
            return row[0]

    def _sync_directory(self, directory: Path) -> None:
        fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    def _put(self, data: bytes, kind: str) -> str:
        if kind not in {"blob", "basis", "manifest"}:
            raise ConfigurationError("object kind must be blob, basis, or manifest")
        if not isinstance(data, bytes):
            raise ConfigurationError("immutable objects require bytes")
        # accel C (hash_once): the digest computed for this very payload object at encode, when known
        object_id = f"{kind}:{_accel_c.store_digest(data) or hashlib.sha256(data).hexdigest()}"
        path = self.path_for(object_id)
        exists = self._db.execute("SELECT 1 FROM objects WHERE id=?", (object_id,)).fetchone()
        if exists is not None and self._evicted and object_id in self._evicted:
            # W1.4.3 (E1): identical content written again (e.g. a regenerated node) restores the evicted payload
            self._write_file(path, data)
            self._db.execute("DELETE FROM evicted WHERE id=?", (object_id,))
            self._unevicted.append(object_id)
            self._touched[object_id] = None
            self._events.append(("created", object_id))
            return object_id
        if exists is not None:
            self._require(object_id)
            self._touched[object_id] = None
            self._events.append(("existing", object_id))
            return object_id
        if self.max_bytes is not None:
            used = self._db.execute("SELECT COALESCE(SUM(size), 0) FROM objects").fetchone()[0]
            if used + len(data) > self.max_bytes:
                raise CapacityError(
                    f"object store requires {used + len(data)} bytes, budget is {self.max_bytes}"
                )
        if path.exists():
            try:
                existing = self._read_file(path)
            except OSError as exc:
                raise IntegrityError(f"cannot read orphaned immutable object: {object_id}") from exc
            if existing != data:
                raise IntegrityError(f"corrupt orphaned immutable object: {object_id}")
            self.lift_quarantine(object_id)  # its bytes were just compared equal to the content
        else:
            staged = self.root / "staging" / f"pending-{uuid.uuid4().hex}"
            try:
                with staged.open("xb") as handle:
                    handle.write(data)
                    handle.flush()
                    os.fsync(handle.fileno())
                    if self.io_policy == "nocache":  # written back (fsync): its pages go now
                        self._dontneed(handle.fileno(), len(data), "write")
                os.link(staged, path)
                # Directory entries are fsynced once per directory before the index commits.
                self._dirty_directories.add(path.parent)
            finally:
                staged.unlink(missing_ok=True)
            self.lift_quarantine(object_id)  # a fresh file (a collected object written again)
        self._db.execute("INSERT INTO objects VALUES (?, ?, ?)", (object_id, kind, len(data)))
        self._created.append(object_id)
        self._touched[object_id] = None
        self._events.append(("created", object_id))
        return object_id

    @property
    def created_objects(self) -> tuple[str, ...]:
        """New object IDs in the caller's currently active outer transaction."""
        with self._lock:
            if not self._depth:
                raise StateError("created_objects requires an active store transaction")
            return tuple(self._created)

    @property
    def touched_objects(self) -> tuple[str, ...]:
        """Object IDs written in the caller's active outer transaction, new or already stored (content-addressed
        deduplication: another session may have written identical content first)."""
        with self._lock:
            if not self._depth:
                raise StateError("touched_objects requires an active store transaction")
            return tuple(self._touched)

    def put(self, data: bytes, *, kind: str = "blob") -> str:
        if kind == "manifest":
            raise ConfigurationError("use put_manifest to register manifest references")
        with self.transaction():
            return self._put(data, kind)

    put_bytes = put

    def put_manifest(self, data: Mapping[str, Any], *, references: Iterable[str] = ()) -> str:
        if not isinstance(data, Mapping):
            raise ConfigurationError("manifest data must be a JSON mapping")
        references = tuple(sorted(set(references)))
        with self.transaction():
            for reference in references:
                self._require(reference)
            encoded = canonical_json({
                "format": "reelcache.manifest.v1", "references": list(references), "data": data,
            })
            object_id = self._put(encoded, "manifest")
            for reference in references:
                self._db.execute("INSERT OR IGNORE INTO edges VALUES (?, ?)",
                                 (object_id, reference))
            return object_id

    def _manifest(self, object_id: str) -> tuple[dict[str, Any], tuple[str, ...]]:
        if not isinstance(object_id, str) or not object_id.startswith("manifest:"):
            raise IntegrityError("expected a manifest object")
        value = parse_json(self._get(object_id))
        if (not isinstance(value, dict)
                or set(value) != {"format", "references", "data"}
                or value["format"] != "reelcache.manifest.v1"
                or not isinstance(value["data"], dict)
                or not isinstance(value["references"], list)
                or any(not isinstance(item, str) for item in value["references"])
                or value["references"] != sorted(set(value["references"]))):
            raise IntegrityError("invalid immutable manifest")
        references = tuple(value["references"])
        for reference in references:
            self.path_for(reference)
        edges = tuple(row[0] for row in self._db.execute(
            "SELECT child FROM edges WHERE parent=? ORDER BY child", (object_id,)
        ))
        if edges != references:
            raise IntegrityError("manifest references disagree with the store index")
        return value["data"], references

    def put_telemetry(self, object_id: str, data: Mapping[str, Any]) -> None:
        """Attach unhashed run telemetry (timings, scheduling) to a stored object; the latest write wins.

        Telemetry never enters an object's bytes or content address; it is
        removed together with the object by garbage collection.
        """
        self.path_for(object_id)
        if not isinstance(data, Mapping):
            raise ConfigurationError("object telemetry must be a JSON mapping")
        encoded = canonical_json(data).decode("utf-8")
        with self.transaction():
            if self._db.execute("SELECT 1 FROM objects WHERE id=?", (object_id,)).fetchone() is None:
                raise IntegrityError(f"missing immutable object: {object_id}")
            self._db.execute(
                "INSERT INTO telemetry VALUES (?, ?) ON CONFLICT(object_id) DO UPDATE SET data=excluded.data",
                (object_id, encoded),
            )

    def telemetry(self, object_id: str) -> dict[str, Any] | None:
        """The object's run telemetry sidecar, or None (none recorded, or written before sidecars existed)."""
        self.path_for(object_id)
        with self.transaction():
            row = self._db.execute("SELECT data FROM telemetry WHERE object_id=?", (object_id,)).fetchone()
        return None if row is None else parse_json(row[0].encode("utf-8"))

    def get_manifest(self, object_id: str) -> dict[str, Any]:
        with self.transaction():
            return self._manifest(object_id)[0]

    def get_manifest_with_references(self, object_id: str) -> tuple[dict[str, Any], tuple[str, ...]]:
        with self.transaction():
            return self._manifest(object_id)

    def get_manifests_with_references(self, object_ids: Iterable[str]) -> list[tuple[dict[str, Any], tuple[str, ...]]]:
        """:meth:`get_manifest_with_references` of many manifests: parallel reads (:meth:`get_many`) and one edges
        query instead of one read and one query per manifest; the same checks per manifest."""
        ids = list(object_ids)
        for object_id in ids:
            if not isinstance(object_id, str) or not object_id.startswith("manifest:"):
                raise IntegrityError("expected a manifest object")
        unique = list(dict.fromkeys(ids))
        with self.transaction():
            raw = self.get_many(unique)
            edges: dict[str, list[str]] = {object_id: [] for object_id in unique}
            for start in range(0, len(unique), self._QUERY_CHUNK):
                part = unique[start:start + self._QUERY_CHUNK]
                query = (f"SELECT parent, child FROM edges WHERE parent IN ({','.join('?' * len(part))}) "
                         "ORDER BY parent, child")
                for parent, child in self._db.execute(query, part):
                    edges[parent].append(child)
        results = {}
        for object_id, data in zip(unique, raw):
            value = parse_json(data)
            if (not isinstance(value, dict)
                    or set(value) != {"format", "references", "data"}
                    or value["format"] != "reelcache.manifest.v1"
                    or not isinstance(value["data"], dict)
                    or not isinstance(value["references"], list)
                    or any(not isinstance(item, str) for item in value["references"])
                    or value["references"] != sorted(set(value["references"]))):
                raise IntegrityError("invalid immutable manifest")
            references = tuple(value["references"])
            for reference in references:
                self.path_for(reference)
            if tuple(edges[object_id]) != references:
                raise IntegrityError("manifest references disagree with the store index")
            results[object_id] = (value["data"], references)
        return [results[object_id] for object_id in ids]

    def references(self, object_id: str) -> tuple[str, ...]:
        with self.transaction():
            return self._manifest(object_id)[1]

    def _walk(self, object_ids: Iterable[str], *, verify: bool, force: bool = False,
              memo: bool = False, damaged: dict[str, str] | None = None) -> set[str]:
        """Visit a graph. ``memo`` skips manifests whose complete subgraph was already walked.

        Every stored manifest's direct references existed when it was written and
        garbage collection never deletes a child of a surviving parent, so a
        subgraph found complete stays complete; memoized walks (ref updates and
        leases, without ``verify_references``) therefore check only new objects.
        Liveness (GC), audits and ``verify_references`` stores always walk everything.
        ``damaged`` (GC with ``on_damage="keep"``) records damaged objects instead of raising and follows a damaged
        manifest's children through the index's edges; damaged objects stay in the visited (live) set.
        """
        use_memo = memo and not force and not self.verify_references
        pending, visited = list(object_ids), set()
        while pending:
            object_id = pending.pop()
            if object_id in visited:
                continue
            visited.add(object_id)
            if self._evicted and object_id in self._evicted:
                continue  # W1.4.3 (E1): an evicted payload is live (its row stays) but has nothing to check
            if object_id.startswith("manifest:"):
                if use_memo and object_id in self._complete_manifests:
                    # Another process may have collected it: an existing manifest has a complete subgraph.
                    if self._db.execute("SELECT 1 FROM objects WHERE id=?", (object_id,)).fetchone() is not None:
                        continue
                    self._complete_manifests.discard(object_id)
                try:
                    if force:
                        self._get(object_id, verify=True)
                    _, refs = self._manifest(object_id)
                except IntegrityError as exc:
                    if damaged is None:
                        raise
                    damaged[object_id] = str(exc)
                    refs = tuple(row[0] for row in self._db.execute(
                        "SELECT child FROM edges WHERE parent=?", (object_id,)))
                pending.extend(refs)
            elif verify:
                try:
                    if force:
                        self._get(object_id, verify=True)
                    else:
                        self._require(object_id)
                except IntegrityError as exc:
                    if damaged is None:
                        raise
                    damaged[object_id] = str(exc)
            elif self._db.execute("SELECT 1 FROM objects WHERE id=?", (object_id,)).fetchone() is None:
                if damaged is None:
                    raise IntegrityError(f"missing referenced object: {object_id}")
                damaged[object_id] = f"missing referenced object: {object_id}"
        if use_memo:
            self._pending_complete.update(item for item in visited if item.startswith("manifest:"))
        return visited

    def verify(self, object_id: str, *, recursive: bool = True) -> None:
        """Explicit audit: always hash-verifies, regardless of ``verify_reads`` (lifting the quarantine of every
        object that verifies)."""
        with self.transaction():
            if recursive:
                self._walk([object_id], verify=True, force=True)
            else:
                self._get(object_id, verify=True)

    def check(self, object_id: str) -> None:
        """Cheap graph check: manifests parse and every reference exists."""
        with self.transaction():
            self._walk([object_id], verify=False)

    @property
    def damaged(self) -> dict[str, str]:
        """``{object_id: error}`` of the damaged objects the latest garbage collection kept (``on_damage="keep"``)."""
        with self._lock:
            return dict(self._damaged)

    def _ref_name(self, name: str) -> None:
        if not isinstance(name, str) or not name or "\x00" in name:
            raise ConfigurationError("reference names must be nonempty strings without NUL")

    def get_ref(self, name: str) -> str | None:
        self._ref_name(name)
        with self.transaction():
            row = self._db.execute("SELECT object_id FROM refs WHERE name=?", (name,)).fetchone()
            return None if row is None else row[0]

    def set_ref(self, name: str, object_id: str, *, expected: Any = _ANY) -> None:
        self._ref_name(name)
        with self.transaction():
            current = self.get_ref(name)
            if expected is not _ANY and current != expected:
                raise StateError(f"reference changed concurrently: {name}")
            self._walk([object_id], verify=True, memo=True)
            self._db.execute(
                "INSERT INTO refs VALUES (?, ?) ON CONFLICT(name) DO UPDATE SET object_id=excluded.object_id",
                (name, object_id),
            )

    def delete_ref(self, name: str, *, expected: Any = _ANY) -> None:
        self._ref_name(name)
        with self.transaction():
            current = self.get_ref(name)
            if expected is not _ANY and current != expected:
                raise StateError(f"reference changed concurrently: {name}")
            if current is None:
                raise StateError(f"reference does not exist: {name}")
            self._db.execute("DELETE FROM refs WHERE name=?", (name,))

    def list_refs(self, prefix: str = "") -> dict[str, str]:
        with self.transaction():
            return {
                name: object_id for name, object_id in
                self._db.execute("SELECT name, object_id FROM refs ORDER BY name")
                if name.startswith(prefix)
            }

    def lease(
        self, object_ids: str | Iterable[str], *, purpose: str = "read", verify: bool = True
    ) -> ObjectLease:
        """Pin a graph; deferred data verification still checks its manifests.

        With verify=False, get()/verify() must validate data before use. History
        restoration uses this to admit memory before reading payloads and avoid
        hashing and reading each payload twice.
        """
        if purpose not in {"read", "staging"}:
            raise ConfigurationError("object lease purpose must be read or staging")
        if type(verify) is not bool:
            raise ConfigurationError("lease verification must be boolean")
        if isinstance(object_ids, str):
            object_ids = [object_ids]
        ids = tuple(sorted(set(object_ids)))
        if not ids:
            raise ConfigurationError("an object lease requires at least one object")
        with self.transaction():
            self._walk(ids, verify=verify, memo=True)
            lease_id = ("staging-" if purpose == "staging" else "") + uuid.uuid4().hex
            for object_id in ids:
                self._db.execute("INSERT INTO leases VALUES (?, ?, ?)",
                                 (lease_id, self._owner, object_id))
            lease = ObjectLease(self, lease_id, ids)
            self._leases.add(lease)
            return lease

    def _release_lease(self, lease_id: str) -> None:
        with self.transaction():
            self._db.execute("DELETE FROM leases WHERE id=? AND owner=?", (lease_id, self._owner))

    def _reap_owners(self) -> None:
        for owner, pid, stamp in list(self._db.execute("SELECT id, pid, stamp FROM owners")):
            if not _alive(pid, stamp):
                pending = [
                    object_id for lease_id, object_id in self._db.execute(
                        "SELECT id, object_id FROM leases WHERE owner=?", (owner,)
                    ) if lease_id.startswith("staging-")
                ]
                while pending:
                    object_id = pending.pop()
                    if object_id in self._recovered_staging:
                        continue
                    self._recovered_staging.add(object_id)
                    pending.extend(row[0] for row in self._db.execute(
                        "SELECT child FROM edges WHERE parent=?", (object_id,)
                    ))
                self._db.execute("DELETE FROM leases WHERE owner=?", (owner,))
                self._db.execute("DELETE FROM owners WHERE id=?", (owner,))

    def _recover_files(self) -> tuple[int, int]:
        removed, reclaimed = 0, 0
        for path in (self.root / "staging").glob("pending-*"):
            if path.is_file() and re.fullmatch(r"pending-[0-9a-f]{32}", path.name):
                reclaimed += path.stat().st_size
                path.unlink()
                removed += 1
        known = {row[0] for row in self._db.execute("SELECT id FROM objects")}
        known -= self._evicted - set(self._unevicted)  # W1.4.3 (E1): an evicted object's file goes too (a crash after
        # its eviction commit); one written again in this transaction stays
        for kind in ("blob", "basis", "manifest"):
            for path in (self.root / "objects" / kind).iterdir():
                if (path.is_file() and re.fullmatch(r"[0-9a-f]{64}", path.name)
                        and f"{kind}:{path.name}" not in known):
                    reclaimed += path.stat().st_size
                    path.unlink()
                    removed += 1
        return removed, reclaimed

    def collect_garbage(self, *, candidates: Iterable[str] | None = None) -> dict[str, int]:
        candidate_ids = None if candidates is None else set(candidates)
        if candidate_ids is not None:
            for object_id in candidate_ids:
                self.path_for(object_id)
        with self.transaction():
            self._reap_owners()
            recovered_staging = set(self._recovered_staging)
            if candidate_ids is not None:
                candidate_ids |= recovered_staging
            roots = [row[0] for row in self._db.execute("SELECT object_id FROM refs")]
            roots += [row[0] for row in self._db.execute("SELECT object_id FROM leases")]
            damaged: dict[str, str] | None = {} if self.on_damage == "keep" else None
            live = self._walk(roots, verify=True, damaged=damaged)
            if damaged is not None:
                self._damaged = damaged  # what this complete liveness walk found (earlier findings may be gone)
            objects = dict(self._db.execute("SELECT id, size FROM objects"))
            dead_ids = set(objects) - live
            if candidate_ids is not None:
                dead_ids &= candidate_ids
                edges = tuple(self._db.execute("SELECT parent, child FROM edges"))
                while True:
                    referenced = {child for parent, child in edges
                                  if parent not in dead_ids and child in dead_ids}
                    if not referenced:
                        break
                    dead_ids -= referenced
            dead = [(object_id, objects[object_id]) for object_id in dead_ids]
            for object_id, _ in dead:
                self._db.execute("DELETE FROM edges WHERE parent=?", (object_id,))
                self._db.execute("DELETE FROM telemetry WHERE object_id=?", (object_id,))
            for object_id, _ in dead:
                self._db.execute("DELETE FROM objects WHERE id=?", (object_id,))
                self._events.append(("deleted", object_id))
            self._complete_manifests.difference_update(object_id for object_id, _ in dead)
            if self._evicted:  # W1.4.3 (E1): collected evicted objects (their rows cascade out of ``evicted``)
                self._evicted.difference_update(object_id for object_id, _ in dead)
            # Index deletion is committed before files are removed. An interrupted
            # sweep therefore leaves only unindexed garbage, never missing live data.
        with self.transaction():
            removed, reclaimed = self._recover_files()
            self._recovered_staging.difference_update(recovered_staging)
        result = {"objects_removed": removed, "bytes_reclaimed": reclaimed, "live_objects": len(live)}
        if damaged:
            result["damaged_objects"] = len(damaged)
        return result

    gc = collect_garbage

    # -- W1.4.3 (E1): archive-budget eviction ------------------------------------------------------------------------
    def evict(self, object_ids: Iterable[str]) -> dict[str, Any]:
        """Evict payload objects (``blob`` / ``basis``) ALL OR NOTHING: their files go, their index rows and the
        edges to them stay (the manifests above them remain whole and collectable), and reads of them raise
        :class:`EvictedObjectError`. When one of them is protected (:meth:`_protected`: leased directly, or reachable
        from a pending commit's staging lease) nothing is evicted and the protected ones are returned in ``leased``.
        Runs outside store transactions: the files are unlinked after the eviction commits (a crash in between leaves
        only files the next open removes). Writing identical content again restores an object."""
        ids = sorted({object_id for object_id in object_ids if not object_id.startswith("manifest:")})
        for object_id in ids:
            self.path_for(object_id)
        evicted, leased, nbytes = [], [], 0
        with self._lock:
            if self._depth:
                raise StateError("evict runs outside store transactions")
            with self.transaction():
                self._db.execute("CREATE TABLE IF NOT EXISTS evicted ("
                                 "id TEXT PRIMARY KEY REFERENCES objects(id) ON DELETE CASCADE)")
                rows = self._index_rows(ids, "kind, size")
                leased = sorted(self._protected(ids)) if ids else []
                if not leased:
                    for object_id in ids:
                        if object_id not in rows or object_id in self._evicted:
                            continue
                        self._db.execute("INSERT INTO evicted VALUES (?)", (object_id,))
                        evicted.append(object_id)
                        nbytes += rows[object_id][1]
            self._evicted.update(evicted)
            for object_id in evicted:
                self.path_for(object_id).unlink(missing_ok=True)
        return {"objects": len(evicted), "bytes": nbytes, "evicted": evicted, "leased": leased}

    def _protected(self, object_ids: Iterable[str]) -> set[str]:
        """(in a transaction) The objects of ``object_ids`` no eviction may take: leased directly (any lease), or
        reachable from a staging lease -- a pending commit's packets, basis versions and records, leased in the
        transactions that wrote or found them, so everything its publication will reference (its payload blobs, the
        row blocks of every version it uses, e.g. the HEVC-only arm's shared identity basis) stays. Read leases protect
        their own objects only (a commit's lease on its parent head would otherwise pin a whole history)."""
        wanted = set(object_ids)
        protected = {row[0] for row in self._db.execute("SELECT DISTINCT object_id FROM leases")} & wanted
        if self._db.execute("SELECT 1 FROM leases WHERE id LIKE 'staging-%' LIMIT 1").fetchone() is not None:
            reached = self._db.execute(
                "WITH RECURSIVE reach(id) AS (SELECT object_id FROM leases WHERE id LIKE 'staging-%' UNION "
                "SELECT edges.child FROM edges JOIN reach ON edges.parent = reach.id) SELECT id FROM reach")
            protected |= {row[0] for row in reached} & wanted
        return protected

    def evicted_ids(self, object_ids: Iterable[str] | None = None) -> set[str]:
        """The evicted objects (of ``object_ids``)."""
        with self._lock:
            return set(self._evicted) if object_ids is None else {item for item in object_ids
                                                                  if item in self._evicted}

    def _write_file(self, path: Path, data: bytes) -> None:
        """Durably (re)write an object file (the staged write of :meth:`_put`)."""
        path.unlink(missing_ok=True)
        staged = self.root / "staging" / f"pending-{uuid.uuid4().hex}"
        try:
            with staged.open("xb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
                if self.io_policy == "nocache":
                    self._dontneed(handle.fileno(), len(data), "write")
            os.link(staged, path)
            self._dirty_directories.add(path.parent)
        finally:
            staged.unlink(missing_ok=True)

    def stats(self) -> dict[str, int]:
        with self.transaction():
            result = {"object_count": 0, "object_bytes": 0, "blob_bytes": 0,
                      "basis_bytes": 0, "manifest_bytes": 0}
            for kind, count, size in self._db.execute(
                "SELECT kind, COUNT(*), SUM(size) FROM objects GROUP BY kind"
            ):
                result["object_count"] += count
                result["object_bytes"] += size
                result[f"{kind}_bytes"] = size
            result["total_bytes"] = result["object_bytes"]
            result["references"] = self._db.execute("SELECT COUNT(*) FROM refs").fetchone()[0]
            result["leases"] = self._db.execute("SELECT COUNT(DISTINCT id) FROM leases").fetchone()[0]
            result["filesystem_bytes"] = sum(
                path.stat().st_size for path in self.root.rglob("*") if path.is_file()
            )
            if self._evicted:  # W1.4.3 (E1): index rows whose payload an archive budget evicted
                rows = self._index_rows(sorted(self._evicted), "size")
                result["evicted_objects"] = len(rows)
                result["evicted_bytes"] = sum(row[0] for row in rows.values())
            return result

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            if self._depth:
                raise StateError("cannot close an object store inside a transaction")
            with self.transaction():
                self._db.execute("DELETE FROM leases WHERE owner=?", (self._owner,))
                self._db.execute("DELETE FROM owners WHERE id=?", (self._owner,))
            self._closed = True
            for lease in tuple(self._leases):
                lease._store = None
            if self._readers is not None:
                self._readers.shutdown(wait=True)
                self._readers = None
            with self._read_ahead_guard:
                readers, self._read_ahead_readers = self._read_ahead_readers, None
            if readers is not None:
                readers.shutdown(wait=True, cancel_futures=True)
            self._db.close()
            self._lock_file.close()

    def __enter__(self) -> ObjectStore:
        if self.closed:
            raise StateError("object store is closed")
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


ContentAddressedStore = ObjectStore
