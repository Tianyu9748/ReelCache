"""Factor-sharded archive and restore of a context-parallel (row-sharded) conditioning cache (Step 16 integration).

Under LongCat-Video context parallelism every rank builds and reads only its token rows of each layer's pre-RoPE
K and V: ``reelcache.factor_sharding.NativeOwners.spatial_rows`` (2-D spatial split of every latent frame). This
module archives such a cache without replicating it and restores each rank's rows from storage:

* ``svd_hevc`` (the lossy product path): per matrix the owner-fit reference of
  :class:`~reelcache.factor_sharding.FactorShardDomain` gathers the matrix transiently onto its fit owner, fits it at
  the model's energy target with :func:`energy_fit` (FP64 Gram eigendecomposition, rank prefix certified on the
  stored FP16-basis / FP32-coefficient factors: the K98/V95 rule of the energy sweeps and the Step 15 harvests) and
  scatters the factors to their stable owners (default: whole-matrix owners). Every owner stores only its own
  basis rows and one or more independently encoded all-intra HEVC fragment packs of its coefficient columns in its
  own :class:`~reelcache.factor_sharding.OwnerFactorStore`; the node is published in two phases. A restore decodes
  every owner's packs on that owner's GPU and delivers each rank exactly its native rows
  (``restore_node(..., NativeOwners.spatial_rows(...), rope=None)``; ``mode="factors"``: each native owner
  reconstructs its rows locally with one GEMM per token tile).
* ``raw`` (the exact reference of the identity gate): every rank stores its own rows as exact BF16 bytes in its own
  store, published in two phases; a restore reads them back on the same rank. Never a silent substitute for the
  lossy path.

Archival is deferred: :meth:`RowShardedCacheArchive.capture` (the cache seam's capture callable) copies the rank's
rows to host memory; :meth:`~RowShardedCacheArchive.flush` (collective; call it when generation is idle) fits,
encodes and publishes. :meth:`~RowShardedCacheArchive.install` (the seam's install callable) restores the armed
representation for the requested epoch on every rank (collective); an armed install that cannot be served raises
on every rank (no silent rebuild), an unarmed archive returns ``None`` so the stage builds natively.

Decoder sessions (CR-17-5): a fragment pack's canvas geometry depends on its data (owned rows per tile), so it is
not a canonical shape a process warm-up could know in advance. :func:`pack_canvas_shapes` derives every owner's pack
canvases from the published node map alone (no pack is read); :meth:`RowShardedCacheArchive.prewarm_canvases` creates
those sessions on every worker of the archive's decoder pool (``DecoderPool.prewarm_shapes``) at attach and right
after each flush, so a restore never creates or warms a session itself. Decoded values are unchanged (a session only
decodes; the restore path is the same).

The module is duck-typed against the SGLang port's ``ConditioningCacheEpoch`` (``full_epoch()``, ``as_dict()``,
``token_grid``, ``local_tokens``, ``row_shard.native_owners_kwargs(...)``), so core ``reelcache`` never imports
``reelcache_sglang_models``. Register it with ``CacheRowSeam(heads=H, capture=archive.capture,
install=archive.install)``.
"""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import math
import time
from typing import Any, Callable, Mapping, Sequence

import torch

from ..distributed import KVLayout, PeerFailure
from ..errors import ConfigurationError, IntegrityError, StateError
from ..factor_sharding import REF_PREFIX as STORE_REF_PREFIX
from ..factor_sharding import FactorShardDomain, FitResult, NativeOwners, entry_label
from ..types import CodecSpec, MatrixKey, ModelIdentity, TensorContext

REPRESENTATIONS = ("svd_hevc", "raw")
FIT_SCHEMA = "reelcache.longcat_factor_shard.fp64_gram_eigh_prefix.v1"
RAW_MANIFEST_TYPE = "longcat_cc_raw_rows"
RAW_REF_PREFIX = "longcat_cc_raw/"
NODE_PREFIX = "lc-cc-"
RAW_NODE_PREFIX = "lc-raw-"
TARGETS = ("K", "V")
# Energy-sweep constants of the reference fit (energy_sweeps/scripts/sweep/factors.py).
CROSSING_TOL = 1e-12
TRACE_GUARD = 1e-6


def epoch_key(epoch) -> str:
    """Digest of the whole epoch (row shard removed): identical on every rank of the CP group."""
    whole = epoch.full_epoch() if hasattr(epoch, "full_epoch") else epoch
    data = whole.as_dict() if hasattr(whole, "as_dict") else dict(whole)
    data = {k: v for k, v in data.items() if k not in ("row_shard", "local_tokens")}
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def pack_canvas_shapes(node_map, owner: int) -> list[tuple[int, int, str]]:
    """``(width, height, pix_fmt)`` of each of ``owner``'s fragment packs of an ``svd_hevc`` node, in pack order.

    Derived from the agreed node map alone (each tile's context and owned rows -> its channel-tile video layout ->
    the encoder's temporal canvas, ``reelcache.factor_sharding._placements``), so no pack is read: what a decoder pool
    needs to create the sessions before a restore. ``[]`` for an owner without packs or a non-HEVC node."""
    from .. import _hevc
    from ..factor_sharding import _placements, entry_from_dict, ranges_rows

    codec = node_map.codec or {}
    pix_fmt = codec.get("pix_fmt")
    if pix_fmt is None:
        return []
    shapes = []
    for pack in node_map.owner_record(owner)["packs"]:
        videos = []
        for tile in pack["tiles"]:
            context = node_map.context(entry_from_dict(tile["entry"]))
            rows = ranges_rows([tuple(item) for item in tile["ranges"]])
            videos.append(_hevc.video_layout(context.frames, context.height, context.width, rows, pix_fmt))
        width, height, _, _ = _placements(videos)
        shapes.append((width, height, pix_fmt))
    return shapes


def native_owners_for(epoch, *, kv_heads: int, head_dim: int, world_size: int) -> NativeOwners:
    """Native row owners of ``epoch`` (its CP spatial split, or all rows on a one-rank group)."""
    shard = getattr(epoch, "row_shard", None)
    if shard is None:
        if world_size != 1:
            raise ConfigurationError("a whole (unsharded) cache epoch needs a one-rank group")
        frames, height, width = epoch.token_grid
        return NativeOwners(KVLayout(frames * height * width, kv_heads, head_dim, 1), "rows")
    kwargs = shard.native_owners_kwargs(epoch.token_grid, kv_heads=kv_heads, head_dim=head_dim)
    native = NativeOwners.spatial_rows(**kwargs)
    if native.world_size != world_size:
        raise ConfigurationError(f"epoch row shard spans {native.world_size} ranks, the group has {world_size}")
    return native


def rows_digest(rows: torch.Tensor) -> str:
    return hashlib.sha256(rows.to(torch.int64).cpu().contiguous().numpy().astype("<i8").tobytes()).hexdigest()


@contextmanager
def exact_fp32():
    """TF32 off for the fit and its certification (true FP32 projections, reproducible)."""
    previous = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
                torch.get_float32_matmul_precision())
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = previous[0], previous[1]
        torch.set_float32_matmul_precision(previous[2])


def energy_fit(energy: float, *, extra_rows: int = 64, tile_rows: int = 1024) -> Callable:
    """Owner-fit callable at an energy target (the K98/V95 rule of the energy sweeps / Step 15 harvests).

    FP64 Gram ``X^T X`` and its eigendecomposition on the fit owner; the spectral rank is the smallest prefix
    whose eigen-energy reaches ``energy``; the FP16 basis rows are the sign-normalized leading eigenvectors; the
    rank prefix is then certified on the STORED factors (FP16 basis, FP32 coefficients ``P = X Q^T``) and grown
    one row at a time (at most ``extra_rows``) until the measured stored-factor energy meets the target.
    """
    if isinstance(energy, bool) or not isinstance(energy, (int, float)) or not 0.0 < float(energy) < 1.0:
        raise ConfigurationError("energy target must be in (0, 1)")
    if type(extra_rows) is not int or extra_rows < 0:
        raise ConfigurationError("extra_rows must be a nonnegative integer")
    energy = float(energy)

    def fit(matrix: torch.Tensor, context: TensorContext, parent_basis) -> FitResult:
        from ..factorization import project_onto_basis

        if parent_basis is not None:
            raise ConfigurationError("energy_fit is a fresh fit (no parent basis version)")
        if matrix.ndim != 2 or not bool(torch.isfinite(matrix).all()):
            raise ConfigurationError("energy_fit needs a finite [tokens, channels] matrix")
        started = time.perf_counter()
        tokens, channels = matrix.shape
        with torch.no_grad(), exact_fp32():
            x64 = matrix.to(torch.float64)
            norm_sq = float((x64 * x64).sum())
            if norm_sq <= 0.0:
                raise ConfigurationError(f"{context.key}: zero matrix")
            gram = x64.T @ x64
            del x64
            gram = (gram + gram.T) * 0.5
            evals, evecs = torch.linalg.eigh(gram)
            del gram
            evals, evecs = evals.flip(0), evecs.flip(1)
            raw_min = float(evals.min())
            trace_error = abs(float(evals.sum()) - norm_sq) / norm_sq
            if trace_error > TRACE_GUARD or -min(raw_min, 0.0) > TRACE_GUARD * norm_sq:
                raise IntegrityError(f"{context.key}: spectrum guard failed (trace error {trace_error:.3e})")
            cdf = torch.cumsum(evals.clamp_min(0), 0) / norm_sq
            domain = min(tokens, channels)
            spectral = int(torch.searchsorted(cdf, torch.tensor([energy - CROSSING_TOL], dtype=cdf.dtype,
                                                                device=cdf.device)).item()) + 1
            spectral = max(1, min(spectral, domain))
            limit = min(domain, spectral + extra_rows)
            fitted = evecs[:, :limit].T.contiguous()
            del evecs
            pivots = fitted.abs().argmax(dim=1, keepdim=True)
            signs = torch.where(fitted.gather(1, pivots) < 0, -1.0, 1.0).to(fitted.dtype)
            basis16 = (fitted * signs).to(torch.float16).contiguous()
            del fitted
            if not bool(torch.isfinite(basis16).all()):
                raise IntegrityError(f"{context.key}: non-finite FP16 basis")
            rank, projections = spectral, 0
            while True:
                projection = project_onto_basis(matrix, basis16[:rank].contiguous(), rank=rank, tile_rows=tile_rows)
                projections += 1
                stored = 1.0 - projection.relative_residual ** 2
                if stored >= energy or rank >= limit:
                    break
                rank += 1
            if stored < energy:
                raise IntegrityError(f"{context.key}: stored-factor energy {stored:.6f} < {energy} at rank {rank}")
            cdf_at = float(cdf[rank - 1])
        return FitResult(coefficients=projection.coefficients.float().contiguous(), rank_view=rank,
                         fresh_basis=basis16[:rank].contiguous(), metrics={
                             "factorization_method": "fp64_gram_eigh_prefix_certified", "fit_schema": FIT_SCHEMA,
                             "energy_target": energy, "spectral_rank": spectral, "rank": rank,
                             "spectral_energy_at_rank": cdf_at, "stored_energy": stored,
                             "projection_relative_residual": projection.relative_residual,
                             "projections": projections, "trace_relative_error": trace_error,
                             "norm_sq": norm_sq, "fit_seconds": time.perf_counter() - started})
    return fit


class RowShardedCacheArchive:
    """Per-rank capture / archive / install of a row-sharded K/V cache over a :class:`FactorShardDomain`.

    Construct identically on every rank of the context-parallel group (the domain's group). ``layers``, ``heads``
    and ``head_dim`` describe the cache (matrices ``[tokens, heads * head_dim]``, keys pre-RoPE). ``energy`` maps
    ``"K"`` / ``"V"`` to the fit's energy targets (e.g. ``reelcache.types.energy_targets("longcat_720p")``).
    ``pool`` / ``arena`` (CUDA): the rank's :class:`reelcache.decoder_pool.DecoderPool` and ``CanvasArena`` for
    NVDEC decode of its own packs. ``threads`` / ``pack_groups``: libx265 threads and independent streams per
    owner at archival.
    """

    def __init__(self, domain: FactorShardDomain, *, identity: ModelIdentity, layers: int, heads: int,
                 head_dim: int, energy: Mapping[str, float] | Any, codec: CodecSpec | None = None,
                 representations: Sequence[str] = REPRESENTATIONS, pool=None, arena=None, threads: int = 8,
                 pack_groups: int = 4, pin_memory: bool | None = None, fit_extra_rows: int = 64,
                 ffmpeg_path: str | None = None, warm_canvases: bool = True) -> None:
        if not isinstance(domain, FactorShardDomain):
            raise ConfigurationError("the archive needs a FactorShardDomain")
        if not isinstance(identity, ModelIdentity):
            raise ConfigurationError("identity must be a ModelIdentity")
        for name, value in (("layers", layers), ("heads", heads), ("head_dim", head_dim), ("threads", threads),
                            ("pack_groups", pack_groups)):
            if type(value) is not int or value < 1:
                raise ConfigurationError(f"{name} must be a positive integer")
        if identity.kv_channels != heads * head_dim or identity.layers != layers:
            raise ConfigurationError("identity geometry differs from the cache geometry")
        representations = tuple(representations)
        if not representations or any(item not in REPRESENTATIONS for item in representations):
            raise ConfigurationError(f"representations must be drawn from {REPRESENTATIONS}")
        if hasattr(energy, "for_target"):
            energy = {target: energy.for_target(target) for target in TARGETS}
        self.energy = {target: float(energy[target]) for target in TARGETS}
        self.fits = {target: energy_fit(self.energy[target], extra_rows=fit_extra_rows) for target in TARGETS}
        self.domain, self.identity = domain, identity
        self.layers, self.heads, self.head_dim = layers, heads, head_dim
        self.channels = heads * head_dim
        self.codec = codec if codec is not None else CodecSpec("svd_hevc", rank=self.channels)
        if self.codec.kind != "svd_hevc":
            raise ConfigurationError("the lossy representation is svd_hevc")
        self.representations = representations
        self.pool, self.arena, self.threads, self.pack_groups = pool, arena, threads, pack_groups
        if type(warm_canvases) is not bool:
            raise ConfigurationError("warm_canvases must be boolean")
        self.warm_canvases = warm_canvases  # create new pack canvases' decoder sessions right after each flush
        self.ffmpeg_path = ffmpeg_path
        self.pin_memory = (domain.device.type == "cuda") if pin_memory is None else bool(pin_memory)
        self.pending: dict[str, dict[str, Any]] = {}
        self.archived: dict[str, dict[str, Any]] = {}
        self.armed: str | None = None
        self.cancel: Callable | None = None  # optional cancellation predicate of installs (agreed in the domain)
        # optional observer of this rank's fitted shards before staging: factor_hook(epoch_key, shards)
        self.factor_hook: Callable | None = None
        self.events: list[dict[str, Any]] = []

    # -- identities ----------------------------------------------------------------------------------------
    @property
    def rank(self) -> int:
        return self.domain.rank

    @property
    def world_size(self) -> int:
        return self.domain.world_size

    @property
    def coordinator(self):
        return self.domain.coordinator

    def native_owners(self, epoch) -> NativeOwners:
        """The epoch's native row owners; its row shard must be this rank's (group rank == CP rank)."""
        shard = getattr(epoch, "row_shard", None)
        if shard is not None and int(shard.rank) != self.rank:
            raise ConfigurationError(f"epoch rows belong to CP rank {shard.rank}, this owner is rank {self.rank}")
        return native_owners_for(epoch, kv_heads=self.heads, head_dim=self.head_dim, world_size=self.world_size)

    def keys(self) -> list[MatrixKey]:
        return [MatrixKey(layer, target) for layer in range(self.layers) for target in TARGETS]

    def context(self, key: MatrixKey, token_grid) -> TensorContext:
        frames, height, width = (int(x) for x in token_grid)
        return TensorContext(self.identity, key, 0, frames, height, width, "pre_rope" if key.target == "K" else "none",
                             source_dtype="bfloat16")

    @staticmethod
    def node_id(key: str, representation: str) -> str:
        return (NODE_PREFIX if representation == "svd_hevc" else RAW_NODE_PREFIX) + key[:32]

    # -- capture (seam callable; local) ------------------------------------------------------------------
    def capture(self, epoch, keys: Sequence[torch.Tensor], values: Sequence[torch.Tensor]) -> None:
        """Copy this rank's row matrices of ``epoch`` to host memory (archived at :meth:`flush`)."""
        started = time.perf_counter()
        native = self.native_owners(epoch)
        rows = native.rows(self.rank)
        if hasattr(epoch, "row_index") and not torch.equal(epoch.row_index().to(torch.long), rows):
            raise ConfigurationError("the cache's rows differ from the native owners' rows of this rank")
        if len(keys) != self.layers or len(values) != self.layers:
            raise ConfigurationError(f"capture needs {self.layers} key and value matrices")
        host = {}
        for target, mats in (("K", keys), ("V", values)):
            for layer, value in enumerate(mats):
                if tuple(value.shape) != (rows.numel(), self.channels) or value.dtype != torch.bfloat16:
                    raise ConfigurationError(f"{target}{layer}: expected BF16 [{rows.numel()}, {self.channels}] rows")
                buffer = torch.empty(value.shape, dtype=value.dtype, pin_memory=self.pin_memory)
                buffer.copy_(value, non_blocking=self.pin_memory)
                host[MatrixKey(layer, target)] = buffer
        if self.domain.device.type == "cuda":
            torch.cuda.synchronize(self.domain.device)
        key = epoch_key(epoch)
        self.pending[key] = {"epoch": epoch, "epoch_dict": epoch.full_epoch().as_dict(), "rows": rows.clone(),
                             "rows_sha256": rows_digest(rows), "host": host, "token_grid": tuple(epoch.token_grid)}
        self.events.append({"event": "captured", "epoch_key": key, "rank": self.rank, "rows": int(rows.numel()),
                            "seconds": time.perf_counter() - started,
                            "bytes": sum(t.numel() * t.element_size() for t in host.values())})

    # -- archival (collective) --------------------------------------------------------------------------
    def flush(self) -> list[dict[str, Any]]:
        """Archive every captured epoch in every configured representation (collective, in key order)."""
        keys = sorted(self.pending)
        rows = self.coordinator.checkpoint("longcat_cc_flush:begin", details={"epochs": keys})
        if any(row["epochs"] != keys for row in rows):
            raise PeerFailure("ranks captured different conditioning-cache epochs")
        reports = []
        for key in keys:
            item = self.pending[key]
            report = {"epoch_key": key, "rank": self.rank, "epoch": item["epoch_dict"]}
            if "svd_hevc" in self.representations:
                report["svd_hevc"] = self._archive_factors(key, item)
            if "raw" in self.representations:
                report["raw"] = self._archive_raw(key, item)
            self.archived[key] = {"epoch_dict": item["epoch_dict"], "rows_sha256": item["rows_sha256"],
                                  "token_grid": item["token_grid"], "report": report}
            del self.pending[key]
            reports.append(report)
        if self.warm_canvases and keys and "svd_hevc" in self.representations and self.pool is not None:
            try:  # an optimization of later restores: the nodes are published whatever happens here
                warmed = self.prewarm_canvases(node_ids=[self.node_id(key, "svd_hevc") for key in reversed(keys)])
            except Exception as exc:  # noqa: BLE001 - recorded; a restore then creates its sessions itself
                warmed = {"error": f"{type(exc).__name__}: {exc}", "rank": self.rank}
                self.events.append({"event": "canvases_prewarm_failed", **warmed})
            for report in reports:
                report["canvas_prewarm"] = warmed
        return reports

    # -- decoder sessions (local; CR-17-5) ------------------------------------------------------------------
    def canvas_shapes(self, node_ids: Sequence[str]) -> list[tuple[int, int, str]]:
        """This owner's distinct pack canvases of the given ``svd_hevc`` nodes, in first-seen order (reads only the
        node maps)."""
        shapes = []
        for node_id in node_ids:
            for shape in pack_canvas_shapes(self.domain.store.node_map(node_id), self.rank):
                if shape not in shapes:
                    shapes.append(shape)
        return shapes

    def prewarm_canvases(self, epochs: Sequence[Any] | None = None, *, node_ids: Sequence[str] | None = None
                         ) -> dict[str, Any]:
        """Create this owner's decoder sessions for the pack canvases of its ``svd_hevc`` nodes on every worker of the
        archive's pool, ahead of (never inside) a restore. Call at attach / process warm-up (``epochs``: the
        session's epochs; default: every node published in this owner's store) — :meth:`flush` calls it for the nodes
        it just published. Local; reads only node maps (no pack is read or decoded); the pool's synthetic pictures
        warm each geometry. Shapes beyond the pool's per-worker session LRU (most recent nodes first) are reported in
        ``not_warmed``, not warmed (they would evict each other)."""
        started = time.perf_counter()
        if node_ids is None:
            if epochs is not None:
                node_ids = [self.node_id(epoch_key(epoch), "svd_hevc") for epoch in epochs]
            else:
                names = self.domain.store.store.list_refs(STORE_REF_PREFIX + NODE_PREFIX)
                node_ids = [name[len(STORE_REF_PREFIX):] for name in sorted(names)]
        node_ids = [node_id for node_id in node_ids if self.domain.store.published(node_id) is not None]
        shapes = self.canvas_shapes(node_ids)
        report = {"node_ids": list(node_ids), "shapes": [list(shape) for shape in shapes], "not_warmed": [],
                  "rank": self.rank}
        prewarm = getattr(self.pool, "prewarm_shapes", None)
        if self.pool is None or prewarm is None or not shapes:
            report["skipped"] = "no decoder pool with prewarm_shapes" if shapes else "no pack canvases"
        else:
            limit = int(getattr(self.pool, "max_sessions_per_worker", len(shapes)))
            report["not_warmed"] = [list(shape) for shape in shapes[limit:]]
            report["pool"] = prewarm(tuple(shapes[:limit]))
        report["seconds"] = time.perf_counter() - started
        self.events.append({"event": "canvases_prewarmed", **report})
        return report

    def _archive_factors(self, key: str, item: Mapping[str, Any]) -> dict[str, Any]:
        domain = self.domain
        device = domain.device
        native = self.native_owners(item["epoch"])
        node_id = self.node_id(key, "svd_hevc")
        started = time.perf_counter()
        fits_before = len(domain.counters["owner_fit"])
        shards = {}
        for entry in self.keys():
            local = item["host"][entry].to(device, non_blocking=True)
            shards[entry] = domain.fit_and_scatter(entry, local, native, self.context(entry, item["token_grid"]),
                                                   fit=self.fits[entry.target])
            del local
        fit_seconds = time.perf_counter() - started
        if self.factor_hook is not None:
            self.factor_hook(key, shards)
        hook_seconds = time.perf_counter() - started - fit_seconds
        stats_before = domain.store.stats()
        staged = domain.stage_node(node_id, shards, kind="svd_hevc", codec=self.codec, ffmpeg_path=self.ffmpeg_path,
                                   threads=self.threads, pack_groups=self.pack_groups)
        try:
            node_map = domain.publish_node(staged)
        except BaseException:
            try:
                domain.discard_staged(staged)
            except Exception:  # noqa: BLE001 - the publication failure is raised
                pass
            raise
        stats_after = domain.store.stats()
        fits = [dict(row) for row in domain.counters["owner_fit"][fits_before:] if row is not None]
        mine = node_map.owner_record(self.rank)
        del shards
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return {
            "node_id": node_id, "node_map_sha256": node_map.digest, "fit_and_scatter_seconds": fit_seconds,
            "factor_hook_seconds": hook_seconds,
            "stage_publish_seconds": time.perf_counter() - started - fit_seconds - hook_seconds,
            "encode_seconds": staged["encode_seconds"], "wall_seconds": time.perf_counter() - started,
            "owner_pack_bytes": sum(pack["bytes"] for pack in mine["packs"]),
            "owner_basis_bytes": sum(segment["bytes"] for segment in mine["segments"]),
            "owner_packs": len(mine["packs"]),
            "store_bytes_added": stats_after["object_bytes"] - stats_before["object_bytes"],
            "ranks": {entry_label(entry): node_map.spec(entry).rank_view for entry in node_map.entries},
            "rank_sum": sum(node_map.spec(entry).rank_view for entry in node_map.entries),
            "owner_fits": fits,
        }

    def _archive_raw(self, key: str, item: Mapping[str, Any]) -> dict[str, Any]:
        store = self.domain.store.store
        node_id = self.node_id(key, "raw")
        started = time.perf_counter()
        entries, refs, error, manifest = [], [], None, None
        try:
            for entry in self.keys():
                value = item["host"][entry].contiguous()
                data = value.view(torch.int16).numpy().tobytes()
                ref = store.put(data, kind="blob")
                refs.append(ref)
                entries.append({"layer": entry.layer, "target": entry.target, "ref": ref,
                                "shape": list(value.shape), "dtype": "bfloat16", "bytes": len(data)})
            manifest = store.put_manifest({
                "type": RAW_MANIFEST_TYPE, "node_id": node_id, "owner": self.rank, "world_size": self.world_size,
                "epoch_key": key, "epoch": item["epoch_dict"], "rows_sha256": item["rows_sha256"],
                "entries": entries}, references=refs)
        except Exception as exc:  # noqa: BLE001 - agreed below; nothing published
            error = exc
        self.coordinator.checkpoint(f"longcat_cc_raw:{node_id}:prepare", error=error,
                                    details=None if error else {"manifest": manifest})
        name, error = RAW_REF_PREFIX + node_id, None
        try:
            current = store.get_ref(name)
            if current is not None and current != manifest:
                raise StateError(f"{node_id} is already published with other rows")
            store.set_ref(name, manifest, expected=current)
        except Exception as exc:  # noqa: BLE001
            error = exc
        try:
            self.coordinator.checkpoint(f"longcat_cc_raw:{node_id}:commit", error=error)
        except PeerFailure:
            if error is None:
                try:
                    store.delete_ref(name, expected=manifest)
                except Exception:  # noqa: BLE001 - the agreed failure is raised
                    pass
            raise
        return {"node_id": node_id, "manifest": manifest, "owner_bytes": sum(e["bytes"] for e in entries),
                "wall_seconds": time.perf_counter() - started}

    # -- restore / install (collective) ----------------------------------------------------------------
    def arm(self, representation: str | None) -> None:
        """Select what :meth:`install` restores (``None``: let the stage build natively)."""
        if representation is not None and representation not in self.representations:
            raise ConfigurationError(f"representation {representation!r} is not archived here")
        self.armed = representation

    def install(self, epoch):
        """Seam install callable: this rank's restored ``(keys, values)`` row matrices, or ``None`` if unarmed."""
        if self.armed is None:
            return None
        keys, values, timing = self.restore_rows(epoch, self.armed, cancel=self.cancel)
        self.events.append({"event": "installed", "representation": self.armed, "rank": self.rank, **timing})
        return keys, values

    def restore_rows(self, epoch, representation: str, *, cancel: Callable | None = None,
                     dtype=torch.bfloat16) -> tuple[list[torch.Tensor], list[torch.Tensor], dict[str, Any]]:
        """Collective restore of this rank's rows of ``epoch`` (keys pre-RoPE, never rotated)."""
        if representation not in REPRESENTATIONS:
            raise ConfigurationError(f"unknown representation {representation!r}")
        device = self.domain.device
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        key = epoch_key(epoch)
        node_id = self.node_id(key, representation)
        native, node_map, error = None, None, None
        try:
            native = self.native_owners(epoch)
            if representation == "svd_hevc":
                node_map = self.domain.store.node_map(node_id)
        except Exception as exc:  # noqa: BLE001 - agreed before any payload moves
            error = exc
        try:
            self.coordinator.checkpoint(f"longcat_cc_install:{node_id}:lookup", error=error,
                                        details=None if node_map is None else {"map": node_map.digest})
        except BaseException:
            error = None  # drop the local failure (and its traceback's frames) before the agreed one propagates
            raise
        error = None
        looked_up = time.perf_counter()
        if representation == "svd_hevc":
            outputs = self.domain.restore_node(node_map, native, dtype=dtype, rope=None, pool=self.pool,
                                               arena=self.arena, cancel=cancel)
            keys = [outputs[MatrixKey(layer, "K")] for layer in range(self.layers)]
            values = [outputs[MatrixKey(layer, "V")] for layer in range(self.layers)]
        else:
            keys, values = self._restore_raw(node_id, native, cancel=cancel, dtype=dtype)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        finished = time.perf_counter()
        return keys, values, {"node_id": node_id, "lookup_seconds": looked_up - started,
                              "restore_seconds": finished - looked_up, "total_seconds": finished - started,
                              "rows": int(native.rows(self.rank).numel())}

    def _restore_raw(self, node_id: str, native: NativeOwners, *, cancel=None, dtype=torch.bfloat16):
        store = self.domain.store.store
        device = self.domain.device
        mats, error = {}, None
        try:
            if dtype != torch.bfloat16:
                raise ConfigurationError("raw rows restore as BF16 only")
            manifest = store.get_ref(RAW_REF_PREFIX + node_id)
            if manifest is None:
                raise StateError(f"raw rows of {node_id} are not published on owner {self.rank}")
            data = store.get_manifest(manifest)
            rows = native.rows(self.rank)
            if (data.get("type") != RAW_MANIFEST_TYPE or data.get("owner") != self.rank
                    or data.get("world_size") != self.world_size or data.get("rows_sha256") != rows_digest(rows)):
                raise IntegrityError("raw rows were archived for another owner, group or row split")
            for item in data["entries"]:
                raw = store.get(item["ref"])
                if hashlib.sha256(raw).hexdigest() != item["ref"].split(":", 1)[1] or len(raw) != item["bytes"]:
                    raise IntegrityError("raw rows differ from their content address")
                value = torch.frombuffer(bytearray(raw), dtype=torch.int16).view(torch.bfloat16)
                mats[MatrixKey(int(item["layer"]), item["target"])] = value.reshape(item["shape"]).to(device)
            if set(mats) != set(self.keys()):
                raise IntegrityError("raw rows manifest does not cover every layer")
        except Exception as exc:  # noqa: BLE001 - agreed: every rank fails the same install
            error = exc
        try:
            self.coordinator.checkpoint(f"longcat_cc_install:{node_id}:raw_loaded", error=error,
                                        cancelled=bool(cancel and cancel()))
        except BaseException:
            mats = error = value = rows = None  # a failed install keeps no loaded rows alive in its traceback
            raise
        return ([mats[MatrixKey(layer, "K")] for layer in range(self.layers)],
                [mats[MatrixKey(layer, "V")] for layer in range(self.layers)])

    # -- accounting ---------------------------------------------------------------------------------------
    def storage(self) -> dict[str, Any]:
        """Collective: every owner's store statistics and their sum."""
        return self.domain.aggregate_storage()

    def close(self) -> None:
        self.pending.clear()


def rel_fro(a: torch.Tensor, b: torch.Tensor) -> float:
    a64, b64 = a.double(), b.double()
    norm = float(b64.norm())
    return float((a64 - b64).norm()) / norm if norm > 0 else (0.0 if float(a64.norm()) == 0 else math.inf)
