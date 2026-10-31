"""Immutable basis row blocks and ordered version manifests in the existing store.

A basis version is a content-addressed manifest (``type: basis_version``) that
references its parent version (if any) and exactly one newly added FP16 row
block stored as a ``basis:`` object. The canonical contiguous view of a version
is its ancestors' rows followed by its own, in order. Inherited rows are never
copied, reordered or rewritten; sharing a basis means sharing these objects,
so store leases and garbage collection see every parent version and block
through ordinary manifest references, and each block is stored once.

External-basis packets: a lossy matrix packet produced from prepared factors
carries a standard ``metadata["basis"]`` descriptor (shape, view rank, SHA256)
of the exact prefix view it was encoded with, but its basis bytes are not
persisted with the packet. Decoding assembles that prefix from the version's
blocks inside a bounded, charged workspace, checks it against the descriptor,
and only then attaches it (:func:`attach_basis`); a mismatch is an integrity
error, never a silent substitution.
"""

from __future__ import annotations

import hashlib
import sys
from collections import OrderedDict
from typing import Any, Iterable, Mapping

import numpy as np
import torch

from .errors import CapacityError, ConfigurationError, IntegrityError
from .store import ObjectStore
from .types import (
    BasisVersion, EncodedTensor, MatrixKey, ModelIdentity, json_copy, object_ref,
)


MANIFEST_TYPE = "basis_version"
#: FP16 exponent bits: all ones encode Inf / NaN.
_FP16_EXPONENT = 0x7C00
#: Bound of a store's memo of row blocks found finite (immutable content addresses; cleared beyond it).
FINITE_BLOCK_MEMO = 1 << 20


def _finite_fp16(data) -> bool:
    """Whether native-order FP16 bytes hold only finite values (no exponent of all ones).

    One NumPy pass on the calling thread: the restore path assembles thousands of small blocks per session open, and
    PyTorch CPU ops (``torch.isfinite``, copies, ``cat``) on them run on the intra-op OpenMP pool, whose per-op
    synchronization cost tens of ms on a 16-thread allocation straddling NUMA nodes (SGLang Step 17 CR-17-2).
    """
    bits = np.frombuffer(data, dtype=np.uint16)
    return not bool(np.any(np.bitwise_and(bits, _FP16_EXPONENT) == _FP16_EXPONENT))


def rows_to_bytes(rows: torch.Tensor) -> bytes:
    if not isinstance(rows, torch.Tensor) or rows.dtype != torch.float16 or rows.ndim != 2 or min(rows.shape) < 1:
        raise ConfigurationError("basis row blocks are nonempty FP16 [rows, channels] tensors")
    # NumPy copy and check on the calling thread (no intra-op pool); the bytes are the contiguous row-major FP16 view.
    data = rows.detach().to("cpu").numpy().tobytes()
    if not _finite_fp16(data):
        raise ConfigurationError("basis row blocks must be finite")
    if sys.byteorder != "little":
        raise ConfigurationError("basis row serialization requires a little-endian host")
    return data


def rows_from_bytes(data: bytes, rows: int, channels: int) -> torch.Tensor:
    if not isinstance(data, (bytes, bytearray)) or len(data) != rows * channels * 2:
        raise IntegrityError("basis row block length disagrees with its declared geometry")
    return torch.from_numpy(np.frombuffer(data, dtype=np.float16).reshape(rows, channels).copy())


class BasisVersionStore:
    """Validated basis-version operations over one runtime's :class:`ObjectStore`.

    Callers own transactions; every method may be used inside an outer
    ``store.transaction()``. Validated versions are cached by reference
    (immutable content), so long chains are not revalidated repeatedly.
    """

    def __init__(self, store: ObjectStore, model: ModelIdentity):
        if not isinstance(store, ObjectStore) or not isinstance(model, ModelIdentity):
            raise ConfigurationError("basis versions require an ObjectStore and ModelIdentity")
        self.store, self.model = store, model
        self._cache: dict[str, BasisVersion] = {}
        # Row blocks found entirely finite (content-addressed, so a checked block never needs the check again).
        self._finite_blocks: set[str] = set()
        # SHA-256 states of the views of versions this process wrote: an extension continues its parent's digest
        # from the bytes it hashed, instead of re-reading (and with verify_reads re-hashing) the parent's blocks.
        self._view_digests: OrderedDict[str, Any] = OrderedDict()

    #: Bound of the view-digest states kept for extensions (about 200 bytes each).
    VIEW_DIGESTS = 16384

    def drop_caches(self) -> None:
        """Forget parsed version manifests, checked blocks and view-digest states (the store is unchanged)."""
        self._cache.clear()
        self._finite_blocks.clear()
        self._view_digests.clear()

    def cached_ids(self) -> list[str]:
        """Object ids the caches hold: parsed versions, blocks found finite, versions with a view-digest state."""
        return [*self._cache, *self._finite_blocks, *self._view_digests]

    def forget_unstored(self, present: set[str]) -> dict[str, int]:
        """Drop cached versions, finite-block memos and view-digest states of objects not in ``present`` (the ids the
        store still holds, e.g. after garbage collection: CR-17-3). A kept version's chain stays cached (a live
        version's ancestors are live)."""
        versions = [ref for ref in self._cache if ref not in present]
        for ref in versions:
            del self._cache[ref]
        blocks = [block for block in self._finite_blocks if block not in present]
        self._finite_blocks.difference_update(blocks)
        digests = [ref for ref in self._view_digests if ref not in present]
        for ref in digests:
            del self._view_digests[ref]
        return {"basis_versions": len(versions), "finite_blocks": len(blocks), "view_digests": len(digests)}

    def missing(self, refs: Iterable[str]) -> list[str]:
        """The versions of ``refs`` not parsed yet (a parsed version's whole chain is parsed: :meth:`load`)."""
        return [ref for ref in dict.fromkeys(refs) if ref not in self._cache]

    # -- creation --------------------------------------------------------------
    def _check_rows(self, key: MatrixKey, rows: torch.Tensor) -> None:
        if not isinstance(key, MatrixKey) or key.layer >= self.model.layers:
            raise ConfigurationError("basis key is outside the model")
        if rows.ndim != 2 or rows.shape[1] != self.model.channels_for(key):
            raise ConfigurationError("basis rows do not match the model's channel width")

    def put_root(self, key: MatrixKey, rows: torch.Tensor) -> tuple[str, BasisVersion]:
        """Store a new group's first block and its depth-zero version."""
        self._check_rows(key, rows)
        data = rows_to_bytes(rows)
        block = self.store.put(data, kind="basis")
        digest = hashlib.sha256(data)
        version = BasisVersion(
            parent_ref=None, added_rows_ref=block, added_rank=int(rows.shape[0]),
            total_rank=int(rows.shape[0]), channels=int(rows.shape[1]),
            group_id=BasisVersion.root_group_id(self.model.fingerprint, key, block), key=key,
            model_fingerprint=self.model.fingerprint, depth=0, view_sha256=digest.hexdigest(),
        )
        ref = self._put_version(version)
        self._keep_view_digest(ref, digest)
        return ref, version

    def put_extension(self, parent_ref: str, rows: torch.Tensor) -> tuple[str, BasisVersion]:
        """Append an immutable row block to ``parent_ref``; the parent is not modified."""
        parent = self.load(parent_ref)
        self._check_rows(parent.key, rows)
        if parent.total_rank + rows.shape[0] > parent.channels:
            raise ConfigurationError("an extension cannot exceed the channel count")
        data = rows_to_bytes(rows)
        known = self._view_digests.get(parent_ref)
        if known is not None:
            digest = known.copy()
        else:
            digest = hashlib.sha256()
            for block in self._blocks(parent_ref):
                digest.update(self.store.get(block[0]))
        digest.update(data)
        block = self.store.put(data, kind="basis")
        version = BasisVersion(
            parent_ref=parent_ref, added_rows_ref=block, added_rank=int(rows.shape[0]),
            total_rank=parent.total_rank + int(rows.shape[0]), channels=parent.channels,
            group_id=parent.group_id, key=parent.key, model_fingerprint=parent.model_fingerprint,
            depth=parent.depth + 1, view_sha256=digest.hexdigest(),
        )
        ref = self._put_version(version)
        self._keep_view_digest(ref, digest)
        return ref, version

    def _keep_view_digest(self, ref: str, digest) -> None:
        self._view_digests[ref] = digest
        self._view_digests.move_to_end(ref)
        while len(self._view_digests) > self.VIEW_DIGESTS:
            self._view_digests.popitem(last=False)

    def _put_version(self, version: BasisVersion) -> str:
        references = [version.added_rows_ref] + ([] if version.parent_ref is None else [version.parent_ref])
        ref = self.store.put_manifest({"type": MANIFEST_TYPE, "version": 1, "record": version.to_dict()},
                                      references=references)
        self._cache[ref] = version
        return ref

    # -- validation and reads ----------------------------------------------------
    def load(self, ref: str) -> BasisVersion:
        """Validate one version and (recursively, cached) its whole parent chain."""
        object_ref(ref, "basis version")
        cached = self._cache.get(ref)
        if cached is not None:
            return cached
        version = self._parse(*self.store.get_manifest_with_references(ref))
        if self.store.object_size(version.added_rows_ref) != version.added_bytes:
            raise IntegrityError("basis row block size disagrees with its version")
        if version.parent_ref is not None:
            self.load(version.parent_ref)
        self._link(version)
        self._cache[ref] = version
        return version

    def load_many(self, refs: Iterable[str]) -> list[BasisVersion]:
        """:meth:`load` of many versions and their chains with batched store access (same checks).

        Per chain level: one parallel manifest read with one edges query
        (:meth:`ObjectStore.get_manifests_with_references`) and one block-size
        query (:meth:`ObjectStore.object_sizes`), instead of a read and two
        queries per version; parents are validated before their extensions.
        """
        refs = list(refs)
        for ref in refs:
            object_ref(ref, "basis version")
        parsed: dict[str, BasisVersion] = {}
        pending = [ref for ref in dict.fromkeys(refs) if ref not in self._cache]
        while pending:
            versions = {ref: self._parse(data, references)
                        for ref, (data, references) in zip(pending, self.store.get_manifests_with_references(pending))}
            sizes = self.store.object_sizes(version.added_rows_ref for version in versions.values())
            for version in versions.values():
                if sizes[version.added_rows_ref] != version.added_bytes:
                    raise IntegrityError("basis row block size disagrees with its version")
            parsed.update(versions)
            pending = list(dict.fromkeys(version.parent_ref for version in versions.values()
                                         if version.parent_ref is not None and version.parent_ref not in self._cache
                                         and version.parent_ref not in parsed))
        for ref, version in sorted(parsed.items(), key=lambda item: item[1].depth):
            if ref not in self._cache:
                self._link(version)
                self._cache[ref] = version
        return [self._cache[ref] for ref in refs]

    def _parse(self, data, references) -> BasisVersion:
        """A version manifest's record after its structural, reference and model checks (no store access)."""
        try:
            if data.get("type") != MANIFEST_TYPE or data.get("version") != 1 or set(data) != {
                "type", "version", "record",
            }:
                raise IntegrityError("unsupported basis version manifest")
            version = BasisVersion.from_dict(data["record"])
        except (ConfigurationError, TypeError, AttributeError) as exc:
            raise IntegrityError("invalid basis version manifest") from exc
        expected = sorted({version.added_rows_ref} | ({version.parent_ref} if version.parent_ref else set()))
        if list(references) != expected:
            raise IntegrityError("basis version references disagree with its parent/block")
        if version.model_fingerprint != self.model.fingerprint or version.key.layer >= self.model.layers:
            raise IntegrityError("basis version belongs to another model")
        if version.channels != self.model.channels_for(version.key):
            raise IntegrityError("basis version channel width disagrees with the model")
        return version

    def _link(self, version: BasisVersion) -> None:
        """Root group identity, or consistency with the (cached, validated) parent version."""
        if version.parent_ref is None:
            if version.group_id != BasisVersion.root_group_id(version.model_fingerprint, version.key,
                                                              version.added_rows_ref):
                raise IntegrityError("root basis group identity is not bound to its first block")
            return
        parent = self._cache[version.parent_ref]
        if (parent.total_rank + version.added_rank != version.total_rank
                or parent.channels != version.channels or parent.key != version.key
                or parent.group_id != version.group_id or parent.depth + 1 != version.depth
                or parent.model_fingerprint != version.model_fingerprint):
            raise IntegrityError("basis extension is inconsistent with its parent version")

    def chain(self, ref: str) -> list[tuple[str, BasisVersion]]:
        """Versions from the group root to ``ref`` (inclusive)."""
        result = []
        current: str | None = ref
        while current is not None:
            version = self.load(current)
            result.append((current, version))
            current = version.parent_ref
        return result[::-1]

    def _blocks(self, ref: str) -> list[tuple[str, int]]:
        return [(version.added_rows_ref, version.added_rank) for _, version in self.chain(ref)]

    def blocks(self, ref: str) -> tuple[str, ...]:
        return tuple(block for block, _ in self._blocks(ref))

    def prefetch_blocks(self, requests: Iterable[tuple[str, int]], block_cache: dict[str, bytes]) -> int:
        """Read and check in one parallel batch every row block the ``(version_ref, rows)`` prefixes need.

        Versions load through :meth:`load_many`; blocks missing from
        ``block_cache`` are read with :meth:`ObjectStore.get_many` with
        verification (content address, kind and length checked in the reader
        threads) and stored there -- the checked blocks :meth:`assemble` trusts.
        Returns the number of blocks read.
        """
        requests = list(requests)
        self.load_many(ref for ref, _ in requests)
        needed = []
        for ref, rows in requests:
            remaining = rows
            for block, count in self._blocks(ref):
                if remaining <= 0:
                    break
                if block not in block_cache:
                    needed.append(block)
                remaining -= count
        needed = list(dict.fromkeys(needed))
        for block, data in zip(needed, self.store.get_many(needed, verify=True)):
            block_cache[block] = data
        return len(needed)

    def _prefix(self, ref: str, rows: int | None, max_bytes: int | None,
                block_cache: dict[str, bytes] | None) -> tuple[BasisVersion, int, list]:
        """``(version, rows, [(block, data, size)])``: the checked blocks of a legal prefix and the bytes it takes
        from each, the full view checked against ``view_sha256`` (shared by :meth:`assemble` / :meth:`view_bytes`).

        Each block is hashed once per read, against its content address (by the store with ``verify_reads``, here
        otherwise). With ``verify_reads`` the version manifests were hash-checked when read as well, so the recorded
        view digest would only re-hash authenticated bytes and is not recomputed (L1, SGLang freeze C); without it the
        digest stays the check of the (unverified) version chain."""
        version = self.load(ref)
        rows = version.total_rank if rows is None else rows
        if type(rows) is not int or not 1 <= rows <= version.total_rank:
            raise ConfigurationError("a basis view must be a nonempty legal prefix of its version")
        nbytes = rows * version.channels * 2
        if max_bytes is not None and nbytes > max_bytes:
            raise CapacityError(f"basis view needs {nbytes} bytes; workspace bound is {max_bytes}")
        parts, remaining = [], rows
        verified = self.store.verify_reads
        # Only the complete view has a recorded digest (prefix bytes are checked by their callers).
        digest = hashlib.sha256() if rows == version.total_rank and not verified else None
        for block, count in self._blocks(ref):
            if remaining <= 0:
                break
            data = None if block_cache is None else block_cache.get(block)
            if data is None:
                data = self.store.get(block)  # with verify_reads: kind, length and SHA-256 checked by the store
                if not verified and hashlib.sha256(data).hexdigest() != block.split(":", 1)[1]:
                    raise IntegrityError("basis row block bytes differ from their content address")
                if block_cache is not None:
                    block_cache[block] = data
            take = min(count, remaining)
            size = take * version.channels * 2
            if len(data) < size:
                raise IntegrityError("basis row block length disagrees with its declared geometry")
            if digest is not None:
                digest.update(memoryview(data)[:size])
            parts.append((block, data, size))
            remaining -= take
        if digest is not None and digest.hexdigest() != version.view_sha256:
            raise IntegrityError("assembled basis view differs from its version digest")
        return version, rows, parts

    def assemble(self, ref: str, rows: int | None = None, *, max_bytes: int | None = None,
                 block_cache: dict[str, bytes] | None = None) -> torch.Tensor:
        """Contiguous FP16 prefix view ``[rows, channels]`` from the version's ordered blocks.

        Only the blocks the prefix needs are read. ``max_bytes`` bounds the
        assembled workspace (CapacityError when exceeded). The full view is
        checked against ``view_sha256``; prefix bytes are content-addressed
        block slices and callers compare them with the packet descriptor.
        ``block_cache`` (caller-owned, one batch) keeps blocks already read and
        checked so versions sharing ancestors read each block once. The view is
        copied block by block into one NumPy buffer on the calling thread (no
        intra-op pool; CR-17-2) and returned as a tensor sharing it.
        """
        version, rows, parts = self._prefix(ref, rows, max_bytes, block_cache)
        view = np.empty(rows * version.channels * 2, dtype=np.uint8)
        position = 0
        for _, data, size in parts:
            view[position:position + size] = np.frombuffer(data, dtype=np.uint8, count=size)
            position += size
        return torch.from_numpy(view.view(np.float16).reshape(rows, version.channels))

    def view_bytes(self, ref: str, rows: int, *, max_bytes: int | None = None,
                   block_cache: dict[str, bytes] | None = None) -> bytes:
        """``rows_to_bytes(assemble(...))`` without tensors: the checked prefix bytes (bitwise the same).

        Finiteness is checked on the bytes with NumPy, once per row block while
        the block is entirely finite (memo; content-addressed blocks never
        change), otherwise on the prefix taken from it (ConfigurationError, as
        :func:`rows_to_bytes`).
        """
        if sys.byteorder != "little":
            raise ConfigurationError("basis row serialization requires a little-endian host")
        _, _, parts = self._prefix(ref, rows, max_bytes, block_cache)
        chunks = []
        for block, data, size in parts:
            chunk = memoryview(data)[:size]
            if block not in self._finite_blocks:
                if _finite_fp16(data):
                    if len(self._finite_blocks) >= FINITE_BLOCK_MEMO:
                        self._finite_blocks.clear()
                    self._finite_blocks.add(block)
                elif not _finite_fp16(chunk):
                    raise ConfigurationError("basis row blocks must be finite")
            chunks.append(chunk)
        return b"".join(chunks)

    def unique_block_bytes(self, refs: Iterable[str]) -> dict[str, int]:
        """Each referenced row block once, with its stored size (charged once)."""
        result: dict[str, int] = {}
        for ref in refs:
            if ref is None:
                continue
            for _, version in self.chain(ref):
                result.setdefault(version.added_rows_ref, version.added_bytes)
        return result


def detach_basis(encoded: EncodedTensor) -> tuple[dict[str, Any], dict[str, bytes], dict[str, Any] | None]:
    """Split a prepared lossy packet into header, payloads and metrics, dropping basis bytes.

    The header's basis descriptor is retained verbatim; it binds the exact
    prefix view the decoder must be given back via :func:`attach_basis`.
    """
    if not isinstance(encoded, EncodedTensor) or encoded.basis is None:
        raise ConfigurationError("only lossy single-matrix packets carry a detachable basis")
    descriptor = encoded.metadata.get("basis") if isinstance(encoded.metadata, Mapping) else None
    if not isinstance(descriptor, Mapping) or hashlib.sha256(encoded.basis).hexdigest() != descriptor.get("sha256"):
        raise IntegrityError("packet basis bytes disagree with their descriptor")
    return (json_copy(encoded.metadata), dict(encoded.payloads),
            None if encoded.metrics is None else json_copy(encoded.metrics))


def attach_basis(metadata: Mapping[str, Any], payloads: Mapping[str, bytes], basis: bytes, *,
                 rank_view: int, channels: int, check_digest: bool = True) -> EncodedTensor:
    """Re-attach an assembled prefix view after checking it against the packet descriptor.

    ``check_digest=False`` skips only the descriptor SHA-256 (as ``attach_pack_bases(check_digests=False)``): for
    views assembled from hash-checked row blocks of a ``verify_reads`` store, whose manifests are hash-checked too.
    """
    descriptor = metadata.get("basis") if isinstance(metadata, Mapping) else None
    if (not isinstance(descriptor, Mapping) or descriptor.get("shape") != [rank_view, channels]
            or descriptor.get("view_rank") != rank_view or descriptor.get("dtype") != "float16"
            or descriptor.get("bytes") != len(basis)
            or (check_digest and hashlib.sha256(basis).hexdigest() != descriptor.get("sha256"))):
        raise IntegrityError("assembled basis view differs from the packet's bound prefix")
    return EncodedTensor(metadata, payloads, basis=basis)
