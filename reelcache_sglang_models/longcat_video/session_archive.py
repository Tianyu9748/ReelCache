"""Store-backed LongCat-Video session records (Step 14 CR-14-3; SGLang-free).

A LongCat session is a chain of segment *boundaries*. Boundary ``k`` is the state before segment ``k``: the 13
conditioning frames (uint8 ``[13, H, W, 3]``; ``k = 0``: the conditioning video of the ``generate`` request,
``k > 0``: the last 13 decoded frames of segment ``k - 1``), the generator state segment ``k`` starts with, the
request geometry and, for ``k > 0``, the record of segment ``k - 1`` (prompt, conditioning-cache epoch, cache source,
digests, times). From a boundary the port regenerates the following segments exactly (its own frames + generator
state: the phase-3 resume protocol), and the segment's conditioning cache is restored from its archived
``longcat_segment`` epoch when one exists (the cache depends only on the conditioning latents, not on the prompt).

Everything lives in the ReelCache object store of the cache archive (public store API only):

* boundary frames: one blob; a boundary: one manifest referencing its frames blob and its parent boundary;
* a session: the ref ``longcat/sessions/<session_id>`` to a session manifest that references its head boundary;
* a named checkpoint: the ref ``longcat/checkpoints/<session_id>/<name>`` to a boundary.

Session refs change by compare-and-swap after the new boundary is stored, so a session head is always a complete
boundary: a crash leaves the previous head, and the orphaned objects are garbage. Forks share boundaries
(copy-on-write); garbage collection keeps whatever a session, a named checkpoint or a cache lineage reaches.

Integrity (CR-17-1): a record that does not read back as written -- a missing or corrupt object (the store verifies
reads), a foreign schema, a frames blob that does not match its digest, a broken boundary chain -- raises
:class:`SessionArchiveIntegrityError` (a :class:`reelcache.errors.IntegrityError`: typed ``integrity``). :meth:`
SessionArchive.scan` reads every session and named checkpoint independently and reports the damaged ones instead of
failing on the first.
"""

from __future__ import annotations

import hashlib
import re
import sqlite3
import time
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
from reelcache.errors import IntegrityError

SESSION_PREFIX = "longcat/sessions/"
CHECKPOINT_PREFIX = "longcat/checkpoints/"
BOUNDARY_SCHEMA = "reelcache_sglang_models.longcat_video.session_boundary.v1"
SESSION_SCHEMA = "reelcache_sglang_models.longcat_video.session.v1"
#: Checkpoint and fork-target names (the Step 14 control surface's pattern): names, never paths.
NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_ANY = object()


class SessionArchiveError(RuntimeError):
    """A malformed, missing or concurrently changed session record."""


class SessionArchiveIntegrityError(SessionArchiveError, IntegrityError):
    """A stored session record that is damaged: foreign schema, malformed fields, frames that do not match their
    digest, a broken boundary chain (typed ``integrity``)."""


def check_session_id(session_id: Any) -> str:
    """Session ids: nonempty, no '/', no NUL, at most 256 characters (the Step 14 handle rule)."""
    if (not isinstance(session_id, str) or not session_id.strip() or "/" in session_id or "\x00" in session_id
            or len(session_id) > 256):
        raise SessionArchiveError(f"session ids are nonempty strings without '/' (<= 256 chars); got {session_id!r}")
    return session_id


@dataclass(frozen=True)
class Boundary:
    """One stored boundary (see the module docstring)."""

    ref: str
    segments: int
    frames_ref: str
    frames_sha256: str
    frames_shape: tuple
    generator_state: str
    geometry: dict
    parent: str | None
    segment: dict | None
    session_origin: str
    created: float

    def summary(self) -> dict:
        return {"node": self.ref, "segments": self.segments, "frames_sha256": self.frames_sha256,
                "generator_state_sha256": hashlib.sha256(self.generator_state.encode()).hexdigest(),
                "session_origin": self.session_origin, "created": self.created,
                "segment": None if self.segment is None else {
                    key: self.segment.get(key) for key in ("index", "prompt_sha256", "cache_source", "epoch_digest",
                                                           "final_latents_sha256", "frames_uint8_sha256",
                                                           "request_id")}}


class SessionArchive:
    """Boundaries, sessions and named checkpoints in one ReelCache ``ObjectStore`` (``runtime.store``)."""

    def __init__(self, store):
        self.store = store

    # -- boundaries ------------------------------------------------------------------------------------------
    def put_boundary(self, *, segments: int, frames: np.ndarray, generator_state: str, geometry: Mapping[str, Any],
                     parent: str | None, segment: Mapping[str, Any] | None, session_origin: str) -> Boundary:
        if type(segments) is not int or segments < 0:
            raise SessionArchiveError("segments must be a nonnegative int")
        if (segments == 0) != (parent is None) or (segments == 0) != (segment is None):
            raise SessionArchiveError("boundary 0 has no parent and no segment record; later boundaries have both")
        frames = np.ascontiguousarray(frames)
        if frames.dtype != np.uint8 or frames.ndim != 4 or frames.shape[-1] != 3:
            raise SessionArchiveError(f"boundary frames are uint8 [F, H, W, 3]; got {frames.dtype} {frames.shape}")
        if not isinstance(generator_state, str) or not generator_state:
            raise SessionArchiveError("a boundary needs the generator state its next segment starts with")
        if parent is not None and self.boundary(parent).segments != segments - 1:
            raise SessionArchiveError("a boundary's parent is the boundary one segment before it")
        data = frames.tobytes()
        frames_ref = self.store.put(data)
        manifest = {"type": "longcat_boundary", "schema": BOUNDARY_SCHEMA, "segments": segments,
                    "frames": frames_ref, "frames_sha256": hashlib.sha256(data).hexdigest(),
                    "frames_shape": list(frames.shape), "generator_state": generator_state,
                    "geometry": dict(geometry), "parent": parent,
                    "segment": None if segment is None else dict(segment), "session_origin": session_origin,
                    "created": time.time()}
        references = [frames_ref] + ([parent] if parent is not None else [])
        ref = self.store.put_manifest(manifest, references=references)
        return self._boundary(ref, manifest)

    @staticmethod
    def _boundary(ref: str, data: Mapping[str, Any]) -> Boundary:
        if data.get("type") != "longcat_boundary" or data.get("schema") != BOUNDARY_SCHEMA:
            raise SessionArchiveIntegrityError(f"{ref} is not a LongCat session boundary")
        try:
            return Boundary(ref=ref, segments=int(data["segments"]), frames_ref=data["frames"],
                            frames_sha256=data["frames_sha256"], frames_shape=tuple(data["frames_shape"]),
                            generator_state=data["generator_state"], geometry=dict(data["geometry"]),
                            parent=data.get("parent"), segment=data.get("segment"),
                            session_origin=data.get("session_origin", ""), created=float(data.get("created", 0.0)))
        except (KeyError, TypeError, ValueError) as exc:
            raise SessionArchiveIntegrityError(f"{ref}: malformed LongCat session boundary ({exc!r})") from exc

    def boundary(self, ref: str) -> Boundary:
        if not isinstance(ref, str) or not ref.startswith("manifest:"):  # callers validate user input first
            raise SessionArchiveIntegrityError(f"not a boundary reference: {ref!r}")
        return self._boundary(ref, self.store.get_manifest(ref))

    def frames(self, boundary: Boundary) -> np.ndarray:
        data = self.store.get(boundary.frames_ref)
        if hashlib.sha256(data).hexdigest() != boundary.frames_sha256:
            raise SessionArchiveIntegrityError(f"boundary {boundary.ref}: frame bytes do not match their digest")
        return np.frombuffer(data, dtype=np.uint8).reshape(boundary.frames_shape).copy()

    def path(self, head: str) -> list[Boundary]:
        """Boundaries from boundary 0 to ``head``."""
        out = []
        ref = head
        while ref is not None:
            boundary = self.boundary(ref)
            if out and boundary.segments != out[-1].segments - 1:
                raise SessionArchiveIntegrityError(f"boundary chain of {head} is not contiguous at {ref}")
            out.append(boundary)
            ref = boundary.parent
        if out[-1].segments != 0:
            raise SessionArchiveIntegrityError(f"boundary chain of {head} does not reach boundary 0")
        return out[::-1]

    # -- sessions --------------------------------------------------------------------------------------------
    def session_ids(self) -> list[str]:
        return sorted(name[len(SESSION_PREFIX):] for name in self.store.list_refs(SESSION_PREFIX))

    def session(self, session_id: str) -> dict | None:
        ref = self.store.get_ref(SESSION_PREFIX + check_session_id(session_id))
        if ref is None:
            return None
        data = self.store.get_manifest(ref)
        if (data.get("type") != "longcat_session" or data.get("schema") != SESSION_SCHEMA
                or not isinstance(data.get("head"), str)):
            raise SessionArchiveIntegrityError(f"{ref} is not a LongCat session record")
        return {**data, "ref": ref}

    def put_session(self, session_id: str, *, head: str, expected: Any = _ANY, created: float | None = None,
                    source: Mapping[str, Any] | None = None, prompt_sha256: str | None = None) -> dict:
        """Store a session record with head boundary ``head`` and move the session ref to it (compare-and-swap
        against ``expected``: the previous record ref, or None for a new session)."""
        check_session_id(session_id)
        boundary = self.boundary(head)
        now = time.time()
        data = {"type": "longcat_session", "schema": SESSION_SCHEMA, "session_id": session_id, "head": head,
                "segments": boundary.segments, "created": now if created is None else created, "updated": now,
                "source": None if source is None else dict(source), "head_prompt_sha256": prompt_sha256}
        ref = self.store.put_manifest(data, references=[head])
        try:
            if expected is _ANY:
                self.store.set_ref(SESSION_PREFIX + session_id, ref)
            else:
                self.store.set_ref(SESSION_PREFIX + session_id, ref, expected=expected)
        except Exception as exc:  # noqa: BLE001 - StateError of the store: a concurrent change
            raise SessionArchiveError(f"session {session_id!r} changed concurrently: {exc}") from exc
        return {**data, "ref": ref}

    def delete_session(self, session_id: str) -> dict:
        """Remove the session ref and its named checkpoints (objects become garbage unless reached otherwise)."""
        record = self.session(session_id)
        if record is None:
            raise SessionArchiveError(f"unknown session {session_id!r}")
        names = self.named(session_id)
        for name in names:
            self.store.delete_ref(self._checkpoint_ref(session_id, name))
        self.store.delete_ref(SESSION_PREFIX + session_id, expected=record["ref"])
        return {"session_id": session_id, "head": record["head"], "deleted_checkpoints": sorted(names)}

    # -- named checkpoints -----------------------------------------------------------------------------------
    @staticmethod
    def _checkpoint_ref(session_id: str, name: str) -> str:
        if not isinstance(name, str) or NAME_PATTERN.fullmatch(name) is None:
            raise SessionArchiveError(f"checkpoint names match {NAME_PATTERN.pattern}; got {name!r}")
        return f"{CHECKPOINT_PREFIX}{check_session_id(session_id)}/{name}"

    def named(self, session_id: str) -> dict[str, str]:
        prefix = f"{CHECKPOINT_PREFIX}{check_session_id(session_id)}/"
        return {name[len(prefix):]: ref for name, ref in self.store.list_refs(prefix).items()}

    def pin(self, session_id: str, name: str, boundary: str, *, replace: bool = False) -> str | None:
        """Name ``boundary``; returns the boundary the name held before (None if new)."""
        ref_name = self._checkpoint_ref(session_id, name)
        self.boundary(boundary)
        previous = self.store.get_ref(ref_name)
        if previous is not None and not replace and previous != boundary:
            raise SessionArchiveError(f"session {session_id!r} already has a checkpoint named {name!r}")
        self.store.set_ref(ref_name, boundary, expected=previous)
        return previous

    def unpin(self, session_id: str, name: str) -> str:
        ref_name = self._checkpoint_ref(session_id, name)
        previous = self.store.get_ref(ref_name)
        if previous is None:
            raise KeyError(name)
        self.store.delete_ref(ref_name, expected=previous)
        return previous

    # -- reachability ----------------------------------------------------------------------------------------
    def live_boundaries(self) -> dict[str, Boundary]:
        """Every boundary a session head or a named checkpoint reaches (with their ancestors); the first damaged
        record raises (:meth:`scan` reports damage instead)."""
        return self.scan(strict=True).boundaries

    def scan(self, *, strict: bool = False) -> "ArchiveScan":
        """Read every session record and named checkpoint with its boundary chain, each on its own: a damaged one
        (an IntegrityError: missing or corrupt object, foreign or malformed record, broken chain) is reported in
        ``damaged`` by its ref name -- with the boundaries read before the damage in ``paths`` -- and the others are
        read normally (``strict``: raise instead). Boundaries shared by several heads are read once."""
        result = ArchiveScan()
        failed: dict[str, Exception] = {}
        heads = [(SESSION_PREFIX + sid, sid, None) for sid in self.session_ids()]
        heads += [(name, name[len(CHECKPOINT_PREFIX):].split("/", 1)[0], ref)
                  for name, ref in sorted(self.store.list_refs(CHECKPOINT_PREFIX).items())]
        for name, sid, head in heads:
            refs: list[str] = []
            try:
                if head is None:
                    record = self.session(sid)
                    if record is None:  # deleted meanwhile
                        continue
                    head = record["head"]
                ref = head
                while ref is not None:
                    if ref in failed:
                        raise failed[ref]
                    boundary = result.boundaries.get(ref)
                    if boundary is None:
                        try:
                            boundary = self.boundary(ref)
                        except (IntegrityError, SessionArchiveError) as exc:
                            failed[ref] = exc
                            raise
                    if refs and boundary.segments != result.boundaries[refs[-1]].segments - 1:
                        raise SessionArchiveIntegrityError(f"boundary chain of {head} is not contiguous at {ref}")
                    result.boundaries[ref] = boundary
                    refs.append(ref)
                    ref = boundary.parent
                if result.boundaries[refs[-1]].segments != 0:
                    raise SessionArchiveIntegrityError(f"boundary chain of {head} does not reach boundary 0")
            except (IntegrityError, SessionArchiveError) as exc:
                if strict or isinstance(exc.__cause__, sqlite3.Error):  # the store index failed: not damage
                    raise
                result.damaged[name] = f"{type(exc).__name__}: {exc}"
            result.paths[name] = refs[::-1]
        return result


@dataclass
class ArchiveScan:
    """:meth:`SessionArchive.scan`: the readable boundaries; the boundary refs of every session head and named
    checkpoint by ref name (boundary 0 first; for a damaged one the boundaries read before the damage, head side);
    the damaged ones (ref name -> error)."""

    boundaries: dict = field(default_factory=dict)
    paths: dict = field(default_factory=dict)
    damaged: dict = field(default_factory=dict)

    @staticmethod
    def session_of(name: str) -> str:
        """The session id of a session or checkpoint ref name."""
        if name.startswith(SESSION_PREFIX):
            return name[len(SESSION_PREFIX):]
        return name[len(CHECKPOINT_PREFIX):].split("/", 1)[0]

    def damaged_sessions(self) -> dict[str, str]:
        """Session id -> error, for damaged session records and boundary chains."""
        return {name[len(SESSION_PREFIX):]: error for name, error in self.damaged.items()
                if name.startswith(SESSION_PREFIX)}

    def damaged_checkpoints(self) -> dict[str, dict[str, str]]:
        """Session id -> {checkpoint name: error}, for damaged named checkpoints."""
        out: dict[str, dict[str, str]] = {}
        for name, error in self.damaged.items():
            if name.startswith(CHECKPOINT_PREFIX):
                sid, _, checkpoint = name[len(CHECKPOINT_PREFIX):].partition("/")
                out.setdefault(sid, {})[checkpoint] = error
        return out


__all__ = ["ArchiveScan", "BOUNDARY_SCHEMA", "Boundary", "CHECKPOINT_PREFIX", "NAME_PATTERN", "SESSION_PREFIX",
           "SESSION_SCHEMA", "SessionArchive", "SessionArchiveError", "SessionArchiveIntegrityError",
           "check_session_id"]
