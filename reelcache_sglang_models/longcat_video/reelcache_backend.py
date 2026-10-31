"""ReelCache archival and compressed restore of LongCat-Video conditioning-cache epochs (Step 13 phase 3).

Each continuation segment's conditioning cache (every layer's pre-RoPE K and V, BF16 ``[1, T, H, D]``; 48 x 2
matrices of ``[14432, 4096]`` at 704x1312) reaches :class:`ReelCacheConditioningCacheBackend` through the port's typed
seam (:class:`~.conditioning_cache.ConditioningCacheBackend`) and is archived in a ReelCache
:class:`reelcache.manager.KVRuntime` store in one or both representations:

* ``svd_hevc`` (the compressed product path): one ReelCache lineage per session whose nodes are ``segment`` commits
  of the latest method (Step 4/6: rank-32 anchor sketch, online ridge predictor, verify/grow/trim, certify/share/
  augment/refit against the lineage's active bases) at the model's energy targets (``energy_targets(
  "longcat_720p")`` = K98/V95), all-intra libx265 node packs on canonical canvases, published in order by
  :class:`reelcache.archive_pipeline.ArchivePipeline`. Every node publishes a ``longcat_segment``
  :class:`reelcache.types.CacheEpoch` with one segment-local range ``[0, T)`` whose parent is the previous segment's
  epoch; the runtime rejects a cross-segment append. A rewind that regenerates segment ``k`` forks the session
  lineage at segment ``k - 1``.
* ``raw`` (the exact reference of the identity gate): the same matrices as RAW packets on one runtime branch per
  epoch (``seal_chunk`` with ``CodecSpec()``); never a silent substitute for the compressed path.

On resume or rewind to a segment boundary the stage calls :meth:`ReelCacheConditioningCacheBackend.restore`. When a
restore representation is armed (:meth:`~ReelCacheConditioningCacheBackend.arm_restore`), the epoch is resolved by
its model fingerprint, conditioning-latent digest and token grid (the cache is a function of the 4 conditioning
latents only, not of the prompt or session) and restored into fresh BTHD tensors: ``svd_hevc`` through
:class:`reelcache.archive_restore.LineageRestorer` (packed NVDEC decode + fused ``restore_kernels`` reconstruction,
``rope=None``: LongCat caches pre-RoPE keys and rotates at read time) or, on the CPU,
:class:`reelcache.archive_restore.SoftwareLineageRestorer`; ``raw`` through the runtime's RAW history and
``restore_kernels`` dense sources. An armed restore that cannot be served raises (no silent rebuild); an unarmed
backend returns ``None`` and the stage builds the cache natively.

Archival schedules (``schedule``):

* ``deferred`` (default): at capture the matrices are copied asynchronously (side stream) to pinned host slots;
  :meth:`ReelCacheConditioningCacheBackend.flush` (call it at a pause, when generation is idle) submits them: to a
  CUDA planner either re-uploaded at flush (``deferred_sources="device"``) or as the host copies, which the planner
  uploads one node at a time when it plans that node (``"host"``: one node's 11.35 GB on the device at a time; the
  setting for a planner next to the served model); a CPU pipeline always receives the host copies.
* ``eager``: the host copy is submitted as soon as it lands, so archival runs behind the following segments'
  generation and a durable pause waits only for what is still unpublished. Allowed for a CPU pipeline and for a GPU
  pipeline with bounded planning (``ArchiveOptions(plan_memory_bytes=B)``, Step 7: a LongCat node at B = 8 GiB peaks
  ~1.9 GiB above its sources, unbounded ~35 GiB, which does not fit next to a running LongCat generation).

GPU planning next to the served model (serving UX): a bounded GPU planner plans a node in ~10 s instead of minutes on
the host. :meth:`~ReelCacheConditioningCacheBackend.set_defer_planning` keeps GPU planning out of the stage's
memory-critical sections (denoising, VAE decode: :meth:`~ReelCacheConditioningCacheBackend.generation_section`, Step 7
``ArchivePipeline.defer_gpu_work``); planning then runs between them and after the request.

Integrity (CR-17-1): :func:`open_runtime` / :func:`gpu_archive_stack` open the store with the serving defaults
(``INTEGRITY_DEFAULTS``): every object a restore reads is hash-verified, opening checks heads at index cost (no hash
audit of every stored byte) and quarantines damaged heads. A restore that meets damage raises a typed
``IntegrityError`` and quarantines the epoch; re-indexing skips damaged archives and quarantines the epochs they may
hold (:meth:`ReelCacheConditioningCacheBackend.reload_index`).

Cold-session staging (CR-17-4): :func:`gpu_archive_stack` builds the packed restorer with one Step 5c
``StagingEngine`` on the worker GPU by default (:func:`lineage_restorer`, ``STAGING_DEFAULTS``): cold histories are
staged (parallel reads and hash checks into a bounded pinned ring, asynchronous basis uploads, no torch CPU ops),
bitwise the restore of the store path, which ``staging_engine=False`` selects.

Raw tiers (freeze D, ``raw_tier=True``; :mod:`.raw_tiers`, DESIGN §17): a capture's host slot stays after its node is
published, as a ``host_raw`` LRU entry, and the built cache itself stays on its device as ``gpu_raw`` within
``gpu_raw_budget_bytes`` (:meth:`~ReelCacheConditioningCacheBackend.cache_released`).
:meth:`~ReelCacheConditioningCacheBackend.restore_raw_tier` serves either bitwise, published or not; the planner still
receives only host copies. ``raw_tier=False`` (default here) is C3: slots are freed at publication.
:func:`gpu_archive_stack` ``encoder``: ``x265`` | ``nvenc`` | ``auto`` (:func:`select_encoder`).

The module imports neither SGLang nor CUDA-only code; core ``reelcache`` never imports it.
"""

from __future__ import annotations

import functools
import hashlib
import os
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence

import torch

from .conditioning_cache import (
    LAYOUT_BTHD,
    SOURCE_NATIVE_BUILD,
    SOURCE_RESTORED,
    ConditioningCacheEpoch,
    ConditioningCacheError,
    LayerReadiness,
    LongCatConditioningCache,
)
from .raw_tiers import RECORD_TIERS, EpochTierIndex, HostFull, RawEntry, new_token
from .spec import LONGCAT_VIDEO_ARCH, LongCatVideoArch

REPRESENTATIONS = ("svd_hevc", "raw")
SCHEDULES = ("deferred", "eager")
DEFERRED_SOURCES = ("device", "host")
#: Freeze D (Step 7 ``encoder``): ``x265`` (C3), ``nvenc`` (must open, else an error), ``auto`` (NVENC if the GPU has it).
ENCODERS = ("auto", "x265", "nvenc")
#: Freeze D (Step 7 ``raw_restore``): ``prefer`` restores from a raw tier when one holds the epoch; ``off`` = C3.
RAW_RESTORE_MODES = ("prefer", "off")
MODEL_NAME = "longcat_video_13.6b_direct_720p"
ENERGY_PROFILE = "longcat_720p"
BRANCH_PREFIX = "longcat-"
HOLDER_PREFIX = BRANCH_PREFIX + "held~"
RAW_BRANCH_PREFIX = "longcat_raw-"
DETAILS_SCHEMA = "reelcache_sglang_models.longcat_video.segment_epoch.v1"
NEW_BASIS_DECISIONS = ("fresh", "refit", "augment", "diagnostic_full_width")
# decoder sessions per pool worker: every canonical node canvas of the 4x44x82 conditioning-cache tile geometry
# (18 up to rank 4096; prewarm_process warms them) fits
LONGCAT_DECODER_SESSIONS = 20


# ---------------------------------------------------------------------------------------------------------------
# identities
# ---------------------------------------------------------------------------------------------------------------
def model_identity(fingerprint: str | None = None, arch: LongCatVideoArch = LONGCAT_VIDEO_ARCH, *,
                   name: str = MODEL_NAME, dtype: str = "bfloat16"):
    """The ReelCache :class:`~reelcache.types.ModelIdentity` of the port: layers = depth, C = heads * head_dim."""
    from reelcache.types import ModelIdentity

    if fingerprint is None:
        from .weights import MODEL_FINGERPRINT
        fingerprint = MODEL_FINGERPRINT
    return ModelIdentity(name, fingerprint, arch.depth, arch.num_heads * arch.head_dim, "tokens_channels", dtype)


def default_method(*, rate_lag: int = 0):
    """The latest method at the LongCat energy profile (K98/V95, all-intra svd_hevc, default parameters)."""
    from reelcache.rank_method import RankMethodConfig
    from reelcache.types import CodecSpec, energy_targets

    return RankMethodConfig(energy_targets(ENERGY_PROFILE), CodecSpec(kind="svd_hevc", rank=1, inter_prediction=False),
                            rate_lag=rate_lag)


def diagnostic_method(key_energy: float, value_energy: float, *, rate_lag: int = 0):
    """The latest method at explicit K/V energy fractions (sensitivity studies; labeled provisional, not a profile)."""
    from reelcache.rank_method import RankMethodConfig
    from reelcache.types import CodecSpec, EnergyTargets

    label = f"{ENERGY_PROFILE}_diag_k{key_energy:g}_v{value_energy:g}"
    targets = EnergyTargets(label, key_energy, value_energy, True,
                            "diagnostic energy targets of a sensitivity study; not a selected LongCat profile")
    return RankMethodConfig(targets, CodecSpec(kind="svd_hevc", rank=1, inter_prediction=False), rate_lag=rate_lag)


def epoch_key(epoch: ConditioningCacheEpoch) -> tuple:
    """What determines a conditioning cache: model, conditioning latents and token grid (not prompt/session)."""
    return (epoch.model_fingerprint, epoch.cond_latents_sha256, tuple(epoch.token_grid))


def _safe_name(text: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9._-]", "_", text)[:80]
    return f"{clean}.{hashlib.sha256(text.encode()).hexdigest()[:10]}"


def lineage_branch(session_id: str | None) -> str:
    """The ReelCache lineage of one session (single-component name)."""
    return BRANCH_PREFIX + _safe_name(session_id or "default")


def holder_branch(key: tuple) -> str:
    """The holder lineage of one epoch key (:meth:`ReelCacheConditioningCacheBackend.hold_epochs`); session lineage names
    never start with it (``_safe_name`` maps ``~`` to ``_``)."""
    return HOLDER_PREFIX + hashlib.sha256(repr((key[0], key[1], tuple(key[2]))).encode()).hexdigest()[:32]


def raw_branch(epoch: ConditioningCacheEpoch) -> str:
    """The RAW reference branch of one epoch (named by its model and conditioning-latent digests)."""
    return f"{RAW_BRANCH_PREFIX}{epoch.cond_latents_sha256[:32]}-{epoch.model_fingerprint[:16]}"


def epoch_id(epoch: ConditioningCacheEpoch) -> str:
    return f"seg{epoch.segment_index:04d}-{epoch.cond_latents_sha256[:16]}"


def node_id(epoch: ConditioningCacheEpoch) -> str:
    return f"segment-{epoch.segment_index:04d}-{epoch.cond_latents_sha256[:16]}"


def matrix_keys(layers: int) -> tuple:
    from reelcache.types import MatrixKey

    return tuple(MatrixKey(layer, target) for layer in range(layers) for target in ("K", "V"))


#: The conditioning-cache token grid of the served 720p geometry (13 conditioning frames -> 4 latent frames).
SERVED_TOKEN_GRID = (4, 44, 82)


def epoch_nbytes(model, token_grid: Sequence[int] = SERVED_TOKEN_GRID) -> int:
    """Bytes of one epoch's raw copy (all K/V matrices ``[T, C]``): 11,349,786,624 for LongCat at 720p."""
    from reelcache.serialization import resolve_dtype

    tokens = int(token_grid[0]) * int(token_grid[1]) * int(token_grid[2])
    total = 0
    for key in matrix_keys(model.layers):
        itemsize = torch.empty((), dtype=resolve_dtype(model.dtype_for(key.target))).element_size()
        total += tokens * model.channels_for(key.target) * itemsize
    return total


def matrix_contexts(model, epoch: ConditioningCacheEpoch) -> dict:
    """Canonical contexts of one epoch: segment-local tokens ``[0, T)`` in (frame, row, column) order."""
    from reelcache.types import TensorContext

    frames, height, width = epoch.token_grid
    return {key: TensorContext(model, key, 0, frames, height, width, "pre_rope" if key.target == "K" else "none")
            for key in matrix_keys(model.layers)}


def _refuse_row_shard(epoch: ConditioningCacheEpoch) -> None:
    if epoch.row_shard is not None:
        raise ConditioningCacheError(
            "ReelCacheConditioningCacheBackend archives and restores whole caches on one GPU; a row-sharded "
            "(context-parallel) epoch needs a factor-sharded backend (reelcache.factor_sharding owners restoring each "
            "rank's NativeOwners.spatial_rows), refusing instead of mixing shards")


def _now() -> float:
    return time.perf_counter()


def _outside_inference_mode(function):
    """Run a restore outside InferenceMode (with grad disabled).

    SGLang runs pipeline stages under ``torch.inference_mode()``; tensors created there are inference tensors, which
    may not be updated in place outside InferenceMode -- and the decoder pool's worker threads (thread-local mode off)
    write the decoded canvases in place. Restore buffers (destination caches, canvases, device bases) are therefore
    created and written with InferenceMode disabled; the stage reads them normally afterwards.
    """
    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        with torch.inference_mode(False), torch.no_grad():
            return function(*args, **kwargs)
    return wrapper


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


# ---------------------------------------------------------------------------------------------------------------
# integrity (CR-17-1)
# ---------------------------------------------------------------------------------------------------------------
#: Serving integrity defaults of a LongCat archive (CR-17-1). ``verify_reads``: every archived object a restore reads
#: is hash-verified, so a corrupt packet fails the restore with a typed ``IntegrityError`` instead of restoring wrong
#: values (hardware HEVC decoding does not detect bitstream damage; RAW packets have no decoder at all).
#: ``audit_on_open=False``: opening the store checks every head's graph at index cost (manifests parse and are
#: verified, every reference is indexed) instead of hashing every stored byte. ``on_open_error="quarantine"``: a
#: damaged head is quarantined (listed; a typed error when used) and every other head works.
INTEGRITY_DEFAULTS = {"verify_reads": True, "audit_on_open": False, "on_open_error": "quarantine"}
OPEN_ERROR_POLICIES = ("raise", "quarantine")


def open_runtime(store_root, model, *, verify_reads: bool = True, audit_on_open: bool | None = False,
                 on_open_error: str = "quarantine", **kwargs):
    """Open the archive's :class:`reelcache.manager.KVRuntime` with the serving integrity settings (CR-17-1).

    Graph operations (publication, leases, garbage-collection liveness) stay at index cost
    (``verify_references=False``); ``verify_reads`` verifies what is read for use. The integrity flags need a core with
    ``KVRuntime(verify_references, audit_on_open, on_open_error)`` (Step 7 dev20); on an older core only the old
    behavior (``verify_reads=False, audit_on_open=False, on_open_error="raise"``) is available -- anything else
    raises instead of silently running with weaker integrity (there, ``verify_reads`` also hashes every object a
    graph operation reaches: publication cost growing with the stored bytes).
    """
    import inspect
    from pathlib import Path

    from reelcache.manager import KVRuntime

    if type(verify_reads) is not bool:
        raise ConditioningCacheError("verify_reads must be a bool")
    if audit_on_open is not None and type(audit_on_open) is not bool:
        raise ConditioningCacheError("audit_on_open must be a bool or None (follow verify_reads)")
    if on_open_error not in OPEN_ERROR_POLICIES:
        raise ConditioningCacheError(f"on_open_error must be one of {OPEN_ERROR_POLICIES}")
    parameters = inspect.signature(KVRuntime.__init__).parameters
    if all(name in parameters for name in ("verify_references", "audit_on_open", "on_open_error")):
        return KVRuntime(Path(store_root), model, verify_reads=verify_reads, verify_references=False,
                         audit_on_open=audit_on_open, on_open_error=on_open_error, **kwargs)
    if verify_reads or audit_on_open or on_open_error != "raise":
        raise ConditioningCacheError(
            "this reelcache core has no KVRuntime(verify_references, audit_on_open, on_open_error) (CR-17-1, Step 7 "
            "dev20): verify-on-read without an open-time audit and quarantine of damaged heads need it; pass "
            "verify_reads=False, audit_on_open=False, on_open_error='raise' for the old behavior")
    return KVRuntime(Path(store_root), model, verify_reads=False, **kwargs)


def integrity_settings(runtime) -> dict:
    """The integrity settings a runtime was opened with (reported at attach)."""
    store = runtime.store
    return {"verify_reads": bool(store.verify_reads),
            "verify_references": bool(getattr(store, "verify_references", store.verify_reads)),
            "audit_on_open": getattr(runtime, "audit_on_open", bool(store.verify_reads)),
            "on_open_error": getattr(runtime, "on_open_error", "raise")}


def _damage(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def is_damage(exc: BaseException) -> bool:
    """An IntegrityError reporting damaged stored data -- not a failure of the store's index database (the core wraps
    sqlite errors in IntegrityError too; those are re-raised, never quarantined)."""
    import sqlite3

    from reelcache.errors import IntegrityError

    return isinstance(exc, IntegrityError) and not isinstance(exc.__cause__, sqlite3.Error)


def _lineage_ref(branch: str) -> str:
    """The ref name of a lineage (the key of ``KVRuntime.quarantined``)."""
    from reelcache.lineage import LINEAGE_PREFIX

    return LINEAGE_PREFIX + branch


def _release_frames(exc: BaseException) -> None:
    """Drop the locals of the finished frames an exception's traceback holds (a failed restore's 11 GB of
    destination tensors, staging buffers) so device memory returns to its baseline while the error propagates."""
    import traceback

    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        if exc.__traceback__ is not None:
            traceback.clear_frames(exc.__traceback__)
        exc = exc.__cause__ or exc.__context__


# ---------------------------------------------------------------------------------------------------------------
# bookkeeping
# ---------------------------------------------------------------------------------------------------------------
@dataclass
class _Capture:
    """One captured epoch waiting for (or under) archival."""

    epoch: ConditioningCacheEpoch
    key: tuple
    representations: tuple
    generator_state: str | None = None
    cond_latents: torch.Tensor | None = None
    host: dict | None = None  # MatrixKey -> pinned/CPU canonical copy (archival source and RAW)
    host_event: Any = None  # D2H completion event
    slot: int | None = None
    branch: str | None = None
    ticket: Any = None
    raw_done: bool = False
    times: dict = field(default_factory=dict)
    parent_key: tuple | None = None  # the session timeline's segment k-1 when this epoch was captured
    token: str | None = None  # the raw tier's submission token (freeze D; unique per capture)


@dataclass
class ArchivedEpoch:
    """Where one epoch's archive lives (per representation); ``quarantined``: representation -> why its archive is
    known damaged (a typed error when restored, never a silent rebuild)."""

    key: tuple
    epoch: ConditioningCacheEpoch
    lossy: dict | None = None  # {"branch", "node_id", "epoch_id", "ticket", "node_ref"}
    raw: dict | None = None  # {"branch", "chunk_id"}
    quarantined: dict = field(default_factory=dict)

    def summary(self) -> dict:
        lossy = None if self.lossy is None else {k: v for k, v in self.lossy.items() if k != "ticket"}
        return {"epoch": self.epoch.as_dict(), "lossy": lossy, "raw": self.raw,
                **({"quarantined": dict(self.quarantined)} if self.quarantined else {})}


# ---------------------------------------------------------------------------------------------------------------
# the backend
# ---------------------------------------------------------------------------------------------------------------
class ReelCacheConditioningCacheBackend:
    """Archive LongCat conditioning-cache epochs in ReelCache and restore them on resume/rewind (module docstring).

    ``runtime``: a :class:`reelcache.manager.KVRuntime` whose model is :func:`model_identity` of the served model.
    ``pipeline``: an :class:`reelcache.archive_pipeline.ArchivePipeline` on that runtime (node packs; required for
    ``svd_hevc`` archival). ``restorer``: :class:`~reelcache.archive_restore.LineageRestorer` (GPU) or
    :class:`~reelcache.archive_restore.SoftwareLineageRestorer` (CPU) for ``svd_hevc`` restores. ``archive``: the
    representations archived for every newly built epoch (empty: capture nothing). ``plan_device``: where deferred
    sources are re-uploaded for planning (default: the pipeline's device). ``deferred_sources``: ``"device"``
    (default: a CUDA planner receives the deferred epochs re-uploaded at flush) or ``"host"`` (it receives the pinned
    host copies and uploads each node's sources when it plans that node). ``heads``: attention heads of the cache
    (default: the LongCat DiT's 32 when the model has its 4096 channels). ``restore_device``: where restored caches
    are allocated (default: the restorer's device, else the planning device, else the CPU).

    Integrity (CR-17-1): the runtime's store decides whether reads are hash-verified (:func:`open_runtime`: on by
    default for serving). A restore that hits a damaged archive (a corrupt or missing packet, a quarantined head)
    raises a typed :class:`reelcache.errors.IntegrityError`, quarantines that epoch's archive in the index
    (:meth:`archive_state` ``"quarantined"``; later restores fail the same way without reading) and releases the
    restore's buffers; it never returns a cache built from damaged bytes and never rebuilds silently. Re-indexing
    (:meth:`reload_index`) under the runtime's ``on_open_error="quarantine"`` skips damaged lineages and RAW branches
    and quarantines the epochs they may hold; ``"raise"`` fails as before.
    """

    def __init__(self, runtime, *, pipeline=None, restorer=None, archive: Sequence[str] = ("svd_hevc",),
                 schedule: str = "deferred", plan_device: Any = None, restore_device: Any = None,
                 heads: int | None = None, capture_slots: int = 2, pin_memory: bool | None = None,
                 restore_options: Mapping[str, Any] | None = None, raw_install_backend: str = "auto",
                 timeout: float | None = 3600.0, deferred_sources: str = "device", raw_tier: bool = False,
                 gpu_raw_budget_bytes: int = 0, raw_restore: str = "prefer"):
        from reelcache.types import ModelIdentity

        model = getattr(runtime, "model", None)
        if not isinstance(model, ModelIdentity):
            raise ConditioningCacheError("runtime must be a reelcache KVRuntime with a ModelIdentity")
        if schedule not in SCHEDULES:
            raise ConditioningCacheError(f"schedule must be one of {SCHEDULES}")
        if deferred_sources not in DEFERRED_SOURCES:
            raise ConditioningCacheError(f"deferred_sources must be one of {DEFERRED_SOURCES}")
        if schedule == "eager" and (pipeline is None or (torch.device(pipeline.device).type != "cpu" and getattr(
                getattr(pipeline, "options", None), "plan_memory_bytes", None) is None)):
            raise ConditioningCacheError("eager archival needs a CPU archive pipeline or a GPU pipeline with bounded "
                                         "planning (ArchiveOptions(plan_memory_bytes=...)): the unbounded GPU "
                                         "planner's working set does not fit next to a running generation")
        if pipeline is not None and getattr(pipeline, "runtime", None) is not runtime:
            raise ConditioningCacheError("the archive pipeline must belong to the same runtime")
        if type(capture_slots) is not int or capture_slots < 1:
            raise ConditioningCacheError("capture_slots must be a positive integer")
        if type(raw_tier) is not bool:
            raise ConditioningCacheError("raw_tier must be boolean")
        if type(gpu_raw_budget_bytes) is not int or gpu_raw_budget_bytes < 0:
            raise ConditioningCacheError("gpu_raw_budget_bytes must be a nonnegative integer")
        if raw_restore not in RAW_RESTORE_MODES:
            raise ConditioningCacheError(f"raw_restore must be one of {RAW_RESTORE_MODES}")
        channels = model.channels_for("K")
        if model.channels_for("V") != channels:
            raise ConditioningCacheError("LongCat K and V have the same channel count")
        if heads is None:
            if channels != LONGCAT_VIDEO_ARCH.num_heads * LONGCAT_VIDEO_ARCH.head_dim:
                raise ConditioningCacheError("pass heads= for a model other than the 4096-channel LongCat DiT")
            heads = LONGCAT_VIDEO_ARCH.num_heads
        if type(heads) is not int or heads < 1 or channels % heads:
            raise ConditioningCacheError("heads must be a positive divisor of the model's channels")
        self.runtime, self.model = runtime, model
        self.pipeline, self.restorer = pipeline, restorer
        self.heads, self.head_dim = heads, channels // heads
        self.schedule = schedule
        self.deferred_sources = deferred_sources
        self.plan_device = None if plan_device is None else torch.device(plan_device)
        self.restore_device = None if restore_device is None else torch.device(restore_device)
        self._pin_slots = torch.cuda.is_available() if pin_memory is None else bool(pin_memory)
        self._capture_slots = capture_slots
        self.raw_tier, self.raw_restore = raw_tier, raw_restore
        self.restore_options = dict(restore_options or {})
        self.last_reload: dict | None = None
        self.overlap: bool | None = None  # an explicit set_overlap (None: unset; d5)
        self.process_warm: dict | None = None
        method = getattr(pipeline, "method", None) if pipeline is not None else None
        self.pix_fmt = (method.codec.pix_fmt if method is not None else default_method().codec.pix_fmt)
        self.raw_install_backend = raw_install_backend
        self.timeout = timeout
        self.defer_planning = False
        self.release_planner_memory = False
        self.archive: tuple = ()
        self.set_archive(archive)
        self.restore_representation: str | None = None
        self._lock = threading.RLock()
        # freeze D: host capture slots and per-epoch raw tiers (raw_tiers.py); without raw_tier a slot is freed at
        # publication and nothing stays on the GPU (C3)
        self._tiers = EpochTierIndex(capture_slots, pin=self._pin_slots, retain=raw_tier,
                                     gpu_budget_bytes=gpu_raw_budget_bytes if raw_tier else 0, on_evict=self._on_evict)
        self.last_restore_tier: dict | None = None
        self._restore_tiers: dict[str, dict] = {}
        self._evicted: list = []
        self._prepare_on_evict = {"queued": 0, "prepared": 0, "already_prepared": 0, "errors": 0, "seconds": 0.0}
        self._commit_to_publish: list[dict] = []
        self.encoder_info: dict | None = None  # set by gpu_archive_stack (requested / selected / reason / driver)
        self._encoder_commits: dict[str, int] = {}
        self._pending: list[_Capture] = []
        self._index: dict[tuple, ArchivedEpoch] = {}
        self._sessions: dict[str, dict] = {}
        self._copy_streams: dict[str, Any] = {}
        self.events: list[dict] = []
        self.integrity = integrity_settings(runtime)
        # damaged archives found by the last (re)index: LongCat lineage -> error, RAW reference branch -> error
        self.damaged_lineages: dict[str, str] = {}
        self.damaged_raw: dict[str, str] = {}

    # -- configuration -----------------------------------------------------------------------------------------
    def arm_restore(self, representation: str | None) -> None:
        """``"svd_hevc"`` / ``"raw"``: restore every requested epoch from that archive (raise if absent);
        ``None``: never restore (the stage builds natively)."""
        if representation is not None and representation not in REPRESENTATIONS:
            raise ConditioningCacheError(f"restore representation must be None or one of {REPRESENTATIONS}")
        self.restore_representation = representation

    def set_archive(self, representations: Sequence[str]) -> None:
        representations = tuple(representations)
        if any(rep not in REPRESENTATIONS for rep in representations) or len(set(representations)) != len(
                representations):
            raise ConditioningCacheError(f"archive representations must be distinct values of {REPRESENTATIONS}")
        if "svd_hevc" in representations and self.pipeline is None:
            raise ConditioningCacheError("svd_hevc archival needs an ArchivePipeline")
        self.archive = representations

    def _dtype(self) -> torch.dtype:
        from reelcache.serialization import resolve_dtype

        return resolve_dtype(self.model.dtype_for("K"))

    def prepare(self, token_grid: Sequence[int] = (4, 44, 82), *, slots: int | None = None) -> float:
        """Allocate pinned capture slots ahead of time (session open; off the generation path). Returns seconds."""
        tokens = int(token_grid[0]) * int(token_grid[1]) * int(token_grid[2])
        shapes = {key: (tokens, self.model.channels_for(key)) for key in matrix_keys(self.model.layers)}
        return self._tiers.preallocate(shapes, self._dtype(), slots)

    # -- ConditioningCacheBackend: capture -------------------------------------------------------------------------
    def on_cache_built(self, cache: LongCatConditioningCache) -> None:
        if not self.archive:
            return
        started = _now()
        self._check_cache(cache)
        epoch = cache.epoch
        key = epoch_key(epoch)
        with self._lock:
            wanted = self._missing_representations(key, self.archive)
        if not wanted:
            self._note_timeline(epoch, key)
            self._event("capture_deduplicated", epoch, representations=list(self.archive))
            return
        capture = _Capture(epoch, key, wanted, generator_state=cache.metadata.get("segment_generator_state"),
                           token=new_token())
        if epoch.segment_index > 0:
            capture.parent_key = self._session(epoch.session_id)["timeline"].get(epoch.segment_index - 1)
        if cache.cond_latents is not None:
            capture.cond_latents = cache.cond_latents.detach().to("cpu", copy=True).contiguous()
        matrices = self._canonical(cache)
        device = next(iter(matrices.values())).device
        self._copy_to_host(capture, matrices, device)
        entry = self._tiers.add(RawEntry(capture.token, key, epoch, cache.nbytes, slot=capture.slot,
                                         host=capture.host, host_event=capture.host_event))
        if self.raw_tier:  # the built cache stays on its device as gpu_raw within the budget (none: 0)
            capture.times["gpu_raw_retained"] = self._tiers.retain_gpu(entry.token, cache.keys_pre_rope, cache.values)
        with self._lock:
            self._pending.append(capture)
        self._note_timeline(epoch, key)
        eager = self.schedule == "eager" and "svd_hevc" in wanted
        if eager:
            # The planner reads the host copy (a GPU planner uploads it when it plans the node): wait for the D2H
            # (the stage synchronizes the device next anyway).
            if capture.host_event is not None:
                capture.host_event.synchronize()
                capture.host_event = None
            self._submit_lossy(capture, dict(capture.host))
        capture.times["capture_seconds"] = _now() - started
        self._event("captured", epoch, representations=list(wanted), schedule=self.schedule,
                    submitted_now=eager, seconds=capture.times["capture_seconds"],
                    slot_allocate_seconds_total=self._tiers.allocate_seconds, token=capture.token)

    def _check_cache(self, cache: LongCatConditioningCache) -> None:
        if not isinstance(cache, LongCatConditioningCache):
            raise ConditioningCacheError("on_cache_built expects a LongCatConditioningCache")
        if cache.source != SOURCE_NATIVE_BUILD:
            raise ConditioningCacheError("only natively built caches are archived")
        if cache.layout != LAYOUT_BTHD:
            raise ConditioningCacheError("the ReelCache backend archives BTHD caches")
        _refuse_row_shard(cache.epoch)
        if cache.epoch.model_fingerprint != self.model.fingerprint:
            raise ConditioningCacheError(
                f"cache model fingerprint {cache.epoch.model_fingerprint[:16]} differs from the archive's model "
                f"{self.model.fingerprint[:16]}")
        cache.validate(layers=self.model.layers, heads=self.heads, head_dim=self.head_dim, dtype=self._dtype())

    @staticmethod
    def _canonical(cache: LongCatConditioningCache) -> dict:
        """``MatrixKey -> [T, C]`` views of the cache storage (no copy)."""
        from reelcache.types import MatrixKey

        matrices = {}
        for layer in range(cache.num_layers):
            k, v = cache.layer_matrices(layer)
            matrices[MatrixKey(layer, "K")] = k
            matrices[MatrixKey(layer, "V")] = v
        return matrices

    def _missing_representations(self, key: tuple, wanted: Sequence[str]) -> tuple:
        archived = self._index.get(key)
        pending = [capture for capture in self._pending if capture.key == key]
        missing = []
        for rep in wanted:
            done = archived is not None and (archived.lossy if rep == "svd_hevc" else archived.raw) is not None
            queued = any(rep in capture.representations for capture in pending)
            if not done and not queued:
                missing.append(rep)
        return tuple(missing)

    def _copy_stream(self, device: torch.device):
        name = str(device)
        stream = self._copy_streams.get(name)
        if stream is None:
            stream = torch.cuda.Stream(device)
            self._copy_streams[name] = stream
        return stream

    def _copy_to_host(self, capture: _Capture, matrices: Mapping, device: torch.device) -> None:
        shapes = {key: tuple(value.shape) for key, value in matrices.items()}
        started = _now()
        try:
            capture.slot, host = self._tiers.acquire(capture.token, shapes, self._dtype())
        except HostFull:
            self.refresh_published()  # epochs whose archival finished free (or retain) their slots, then retry
            capture.slot, host = self._tiers.acquire(capture.token, shapes, self._dtype())
        capture.times["slot_acquire_seconds"] = _now() - started
        if device.type == "cuda":
            stream = self._copy_stream(device)
            stream.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.stream(stream):
                for key, value in matrices.items():
                    host[key].copy_(value, non_blocking=True)
                    value.record_stream(stream)
            event = torch.cuda.Event()
            event.record(stream)
            capture.host_event = event
        else:
            for key, value in matrices.items():
                host[key].copy_(value)
        capture.host = host

    # -- archival ---------------------------------------------------------------------------------------------------
    def _session(self, session_id: str | None) -> dict:
        """Per-session lineage state (created on first use, or reloaded from the store).

        ``branch``: the lineage new segment nodes extend; ``segments``: the segment nodes on that branch's path
        (``{index: {"node_ref", "node_id", "key", ...}}``); ``timeline``: the epoch key of every segment of the
        session's current timeline (captured, deduplicated, restored, or seeded by a session provider), which decides
        the parent node of the next archived segment; ``forks``: the highest ``~rN`` branch of the session.
        """
        name = session_id or "default"
        state = self._sessions.get(name)
        if state is not None:
            return state
        branch = lineage_branch(session_id)
        runtime = self.runtime
        lineages = runtime.list_lineages()
        if branch not in lineages:
            runtime.create_lineage(branch, details={"schema": DETAILS_SCHEMA, "session_id": session_id,
                                                    "model": self.model.to_dict()})
        forks = [int(match.group(1)) for other in lineages
                 if (match := re.fullmatch(re.escape(branch) + r"~r(\d+)", other)) is not None]
        state = {"branch": branch, "segments": self._branch_segments(branch), "forks": max(forks, default=0),
                 "timeline": {}}
        self._sessions[name] = state
        return state

    def _branch_segments(self, branch: str) -> dict:
        runtime = self.runtime
        segments = {}
        for ref, node, epoch in runtime.node_path(runtime.lineage_head(branch)):
            if node.node_kind == "segment":
                port = epoch.details.get("port_epoch", {})
                key = (port["model_fingerprint"], port["cond_latents_sha256"], tuple(port["token_grid"]))
                segments[int(port["segment_index"])] = {"node_ref": ref, "node_id": node.chunk_id, "key": key}
        return segments

    def _note_timeline(self, epoch: ConditioningCacheEpoch, key: tuple) -> None:
        """Segment ``epoch.segment_index`` of the session's current timeline is ``key`` (later ones are dropped)."""
        timeline = self._session(epoch.session_id)["timeline"]
        for index in [index for index in timeline if index >= epoch.segment_index]:
            del timeline[index]
        timeline[epoch.segment_index] = key

    def seed_timeline(self, session_id: str | None, epochs: Mapping[int, ConditioningCacheEpoch]) -> None:
        """Declare a session's timeline before generating from one of its boundaries (session providers: resume,
        fork, rewind, possibly in a new process), so a new segment's node gets its true parent."""
        state = self._session(session_id)
        state["timeline"] = {int(index): epoch_key(epoch) for index, epoch in epochs.items()}

    def _published_node(self, key: tuple) -> str | None:
        with self._lock:
            record = self._index.get(key)
        if record is None or record.lossy is None or "svd_hevc" in record.quarantined:
            return None  # a quarantined node is no parent for new archives either
        ref = record.lossy.get("node_ref")
        if ref is None and record.lossy.get("ticket") is not None:
            ref = record.lossy["ticket"].result(self.timeout)
            record.lossy["node_ref"] = ref
        return ref

    def _target_branch(self, capture: _Capture) -> str:
        """The lineage the segment's node extends; never appends a segment after a node that is not its parent.

        In order (the branch's last segment is the timeline's segment ``k-1``; queued commits allowed): the current
        branch. Otherwise -- a rewind regenerating a segment, a resume/fork from a restored or deduplicated parent, a
        parent on another session's branch -- a new branch ``~rN`` forked at the parent's published node; when the
        parent was never archived, a fresh branch from a root (no basis sharing; recorded)."""
        state = self._session(capture.epoch.session_id)
        index = capture.epoch.segment_index
        segments = state["segments"]
        # the parent as of capture time: archival is deferred, the timeline may have moved on since (rewinds)
        parent_key = capture.parent_key if index > 0 else None
        last = max(segments) if segments else -1
        if last == index - 1 and (index == 0 or parent_key is None or segments[last].get("key") in (None, parent_key)):
            return state["branch"]
        if self.pipeline is not None:  # the branch's queued commits publish before the session moves off it
            self.pipeline.drain(state["branch"], timeout=self.timeout)
            self._resolve_tickets(state)
        parent_ref = None
        if index == 0:
            parent_ref = self.runtime.node_path(self.runtime.lineage_head(state["branch"]))[0][0]
        else:
            if parent_key is None and segments.get(index - 1) is not None:
                parent_key = segments[index - 1].get("key")
            if parent_key is not None:
                parent_ref = self._published_node(parent_key)
        if index > 0 and parent_key is None:
            raise ConditioningCacheError(
                f"cannot archive segment {index} of session {capture.epoch.session_id!r}: its lineage ends at "
                f"segment {last} and the session's timeline has no segment {index - 1} (segments are published in "
                "order; a session provider declares the timeline with seed_timeline)")
        if parent_ref is None and not segments and index > 0:
            self._event("lineage_parent_unarchived", capture.epoch, branch=state["branch"])
            return state["branch"]  # an empty lineage: its root is the parent
        state["forks"] += 1
        fork = f"{lineage_branch(capture.epoch.session_id)}~r{state['forks']}"
        if parent_ref is not None:
            self.runtime.fork_lineage(parent_ref, fork)
        else:
            self.runtime.create_lineage(fork, details={"schema": DETAILS_SCHEMA, "session_id": capture.epoch.session_id,
                                                       "model": self.model.to_dict()})
        if self.pipeline is not None:
            self.pipeline.reset(state["branch"], timeout=self.timeout)
        state["branch"] = fork
        state["segments"] = self._branch_segments(fork)
        self._event("lineage_forked", capture.epoch, branch=fork, parent_ref=parent_ref)
        return fork

    @staticmethod
    def _resolve_tickets(state: dict) -> None:
        for entry in state["segments"].values():
            if entry.get("node_ref") is None and entry.get("ticket") is not None and entry["ticket"].done():
                entry["node_ref"] = entry["ticket"].result(0)

    @staticmethod
    def _details(capture: _Capture) -> dict:
        return {"schema": DETAILS_SCHEMA, "port_epoch": capture.epoch.as_dict(),
                "generator_state": capture.generator_state,
                "cond_latents_shape": None if capture.cond_latents is None else list(capture.cond_latents.shape),
                "cond_latents_dtype": None if capture.cond_latents is None else str(capture.cond_latents.dtype)}

    @staticmethod
    def _execution_state(capture: _Capture) -> dict:
        state = {}
        if capture.cond_latents is not None:
            state["cond_latents"] = capture.cond_latents
        if capture.generator_state is not None:
            state["generator_state"] = torch.tensor(list(bytes.fromhex(capture.generator_state)), dtype=torch.uint8)
        return state

    def _submit_lossy(self, capture: _Capture, matrices: Mapping, *, ready_stream=None) -> None:
        epoch = capture.epoch
        branch = self._target_branch(capture)
        ready = None
        if ready_stream is not None:
            ready = torch.cuda.Event()
            ready.record(ready_stream)
        held = {"matrices": dict(matrices)}

        def release(_event, held=held, capture=capture):
            capture.times["sources_released_at"] = _now()
            held.clear()

        started = _now()
        ticket = self.pipeline.submit(branch, node_id(epoch), held["matrices"], matrix_contexts(self.model, epoch),
                                      kind="segment", epoch_id=epoch_id(epoch), details=self._details(capture),
                                      execution_state=self._execution_state(capture), ready_event=ready,
                                      release=release, timeout=self.timeout)
        capture.times["submit_seconds"] = _now() - started
        capture.times["submitted_at"] = started
        capture.branch, capture.ticket = branch, ticket
        state = self._session(epoch.session_id)
        state["segments"][epoch.segment_index] = {"node_id": node_id(epoch), "ticket": ticket, "node_ref": None,
                                                  "key": capture.key}
        with self._lock:
            record = self._index.setdefault(capture.key, ArchivedEpoch(capture.key, epoch))
            record.lossy = {"branch": branch, "node_id": node_id(epoch), "epoch_id": epoch_id(epoch),
                            "ticket": ticket, "node_ref": None}

    def _seal_raw(self, capture: _Capture) -> None:
        from reelcache.errors import StateError
        from reelcache.types import CodecSpec

        epoch = capture.epoch
        branch = raw_branch(epoch)
        started = _now()
        try:
            self.runtime.create_branch(branch)
            fresh = True
        except StateError:
            fresh = False  # sealed earlier in this store: the name binds the model and the conditioning latents
        if fresh or not self.runtime.branch_info(branch)["chunk_ids"]:
            self.runtime.seal_chunk(branch, "segment", capture.host, matrix_contexts(self.model, epoch), CodecSpec())
        with self._lock:
            record = self._index.setdefault(capture.key, ArchivedEpoch(capture.key, epoch))
            record.raw = {"branch": branch, "chunk_id": "segment"}
        capture.raw_done = True
        capture.times["raw_seal_seconds"] = _now() - started
        self._event("raw_sealed", epoch, branch=branch, reused=not fresh, seconds=capture.times["raw_seal_seconds"])

    def flush(self, *, wait: bool = True) -> dict:
        """Archive every captured epoch (capture order); with ``wait`` block until all are published.

        Deferred lossy sources are re-uploaded to ``plan_device`` (default: the pipeline's device) so the pipeline
        plans them there; RAW epochs are sealed synchronously from the host copy.
        """
        started = _now()
        with self._lock:
            pending = list(self._pending)
        for capture in pending:
            if capture.host_event is not None:
                capture.host_event.synchronize()
                capture.host_event = None
            if "raw" in capture.representations and not capture.raw_done:
                self._seal_raw(capture)
            if "svd_hevc" in capture.representations and capture.ticket is None:
                self._submit_deferred(capture)
        result = {"captures": len(pending), "flush_submit_seconds": _now() - started}
        if wait:
            result.update(self.drain())
        return result

    def _submit_deferred(self, capture: _Capture) -> None:
        device = self.plan_device if self.plan_device is not None else torch.device(self.pipeline.device)
        if device.type == "cuda" and self.deferred_sources == "device":
            stream = self._copy_stream(device)
            started = _now()
            with torch.cuda.stream(stream):
                matrices = {key: value.to(device, non_blocking=True) for key, value in capture.host.items()}
            capture.times["reupload_issue_seconds"] = _now() - started
            self._submit_lossy(capture, matrices, ready_stream=stream)
        else:
            self._submit_lossy(capture, dict(capture.host))

    def drain(self) -> dict:
        """Wait until every submitted epoch is published (or raise its archival failure); free capture slots."""
        started = _now()
        with self._lock:
            pending = list(self._pending)
        published = []
        for capture in pending:
            if capture.ticket is None and "svd_hevc" in capture.representations:
                raise ConditioningCacheError("drain() before flush(): a deferred epoch was never submitted")
            if capture.ticket is not None:
                try:
                    ref = capture.ticket.result(self.timeout)
                except BaseException:
                    self._tiers.failed(capture.token)
                    raise
                published.append(ref)
            else:
                ref = None
            if "raw" in capture.representations and not capture.raw_done:
                raise ConditioningCacheError("drain() before flush(): a RAW epoch was never sealed")
            self._publish_capture(capture, ref)
        return {"published": published, "drain_seconds": _now() - started}

    def _publish_capture(self, capture: _Capture, ref: str | None) -> None:
        """One captured epoch is published (its svd_hevc node ``ref`` and/or its sealed RAW): record it, hand its host
        slot to the raw tier (an LRU entry with ``raw_tier``, else freed) and retire the capture."""
        with self._lock:
            if capture not in self._pending:
                return
            self._pending.remove(capture)
        if ref is not None:
            with self._lock:
                self._index[capture.key].lossy["node_ref"] = ref
            entry = self._session(capture.epoch.session_id)["segments"].get(capture.epoch.segment_index)
            if entry is not None and entry.get("ticket") is capture.ticket:
                entry["node_ref"] = ref
            capture.times["published_after_submit_seconds"] = _now() - capture.times["submitted_at"]
            self._note_commit_to_publish(capture, ref)
            try:
                stored = node_bytes(self.runtime, ref)
            except Exception as exc:  # noqa: BLE001 - a record only: the node is published either way
                stored = {"error": f"{type(exc).__name__}: {exc}"[:200]}
            self._event("published", capture.epoch, node_ref=ref, branch=capture.branch, bytes=stored,
                        times=dict(capture.times))
        if capture.host_event is not None:
            capture.host_event.synchronize()
            capture.host_event = None
        self._tiers.published(capture.token, ref)
        capture.slot, capture.host = None, None

    def refresh_published(self) -> int:
        """Publish, without waiting, every captured epoch whose archival has finished (``ticket.done()``; RAW-only
        captures once sealed): their host slots become raw-tier entries (or free) before a drain. A failed archival
        is left for :meth:`drain` to raise. Returns the number published now."""
        with self._lock:
            pending = list(self._pending)
        count = 0
        for capture in pending:
            lossy = "svd_hevc" in capture.representations
            if lossy:
                if capture.ticket is None or not capture.ticket.done():
                    continue
                try:
                    ref = capture.ticket.result(0)
                except Exception:  # noqa: BLE001 - raised again by drain()
                    self._tiers.failed(capture.token)
                    continue
            else:
                ref = None
            if "raw" in capture.representations and not capture.raw_done:
                continue
            self._publish_capture(capture, ref)
            count += 1
        return count

    def reload_index(self, live_epochs: Iterable[ConditioningCacheEpoch] | None = None, *,
                     owners: Mapping[tuple, Iterable[str]] | None = None) -> int:
        """Index the published segment epochs of this store (after a process restart or garbage collection).

        svd_hevc: the segment nodes on every LongCat lineage's path (session lineages, their ``~rN`` forks and the
        holder lineages of :meth:`hold_epochs`). raw: the RAW reference branch (named by the epoch's model and
        conditioning latents) of every indexed epoch and of every epoch of ``live_epochs`` whose branch holds the
        sealed segment -- a RAW branch carries no epoch identity of its own, so a store archived with ``raw`` only is
        re-indexed from the epochs its caller still references (a session provider: the epochs its live boundaries
        record). With ``live_epochs`` the index is limited to those epochs (what the sessions can still reach; the
        rest is collected by garbage collection or no longer referenced); without it every published svd_hevc epoch
        (and its RAW branch) is indexed (standalone use).

        Damaged archives (CR-17-1), with the runtime's ``on_open_error="quarantine"``: a lineage the runtime
        quarantined or whose path does not load is skipped (``damaged_lineages``); a live epoch it may hold -- not
        found on any healthy lineage, and the lineage belongs to a session that records the epoch (``owners``:
        epoch key -> session ids; the capturing session always counts) or is its holder -- is quarantined; a RAW
        branch that does not load quarantines its epoch's ``raw`` archive (``damaged_raw``). ``"raise"``: the first
        damaged archive raises, as before.

        Returns the number of newly indexed (epoch, representation) entries; ``last_reload`` splits them.
        """
        from reelcache.errors import IntegrityError, StateError

        tolerate = self.integrity["on_open_error"] == "quarantine"
        live = None if live_epochs is None else {epoch_key(epoch): epoch for epoch in live_epochs}
        count = 0
        runtime = self.runtime
        quarantined = self._core_quarantined()
        damaged: dict[str, str] = {}
        for branch, head in runtime.list_lineages().items():
            if not branch.startswith(BRANCH_PREFIX):
                continue
            reason = quarantined.get(_lineage_ref(branch))
            if reason is None:
                try:
                    path = runtime.node_path(head)
                except IntegrityError as exc:
                    if not tolerate or not is_damage(exc):
                        raise
                    reason = _damage(exc)
            if reason is not None:
                damaged[branch] = reason
                continue
            for ref, node, stored in path:
                if node.node_kind != "segment":
                    continue
                epoch = self._stored_epoch(stored)
                key = epoch_key(epoch)
                if live is not None and key not in live:
                    continue
                with self._lock:
                    record = self._index.setdefault(key, ArchivedEpoch(key, epoch))
                    if record.lossy is None:
                        record.lossy = {"branch": branch, "node_id": node.chunk_id, "epoch_id": stored.epoch_id,
                                        "ticket": None, "node_ref": ref}
                        count += 1
        self.damaged_lineages = damaged
        lost = 0
        if damaged:
            for key, epoch in (live or {}).items():
                with self._lock:
                    record = self._index.get(key)
                if record is not None and record.lossy is not None:
                    continue
                reason = self._family_damage(epoch, (owners or {}).get(key, ()))
                if reason is not None:
                    self.quarantine_epoch(epoch, "svd_hevc", reason)
                    lost += 1
        with self._lock:
            candidates = {key: record.epoch for key, record in self._index.items() if record.raw is None}
        for key, epoch in (live or {}).items():
            candidates.setdefault(key, epoch)
        raw = 0
        damaged_raw: dict[str, str] = {}
        for key, epoch in candidates.items():
            branch = raw_branch(epoch)
            reason = quarantined.get("branches/" + branch)
            if reason is None:
                try:
                    info = runtime.branch_info(branch)
                except StateError:
                    continue
                except IntegrityError as exc:
                    if not tolerate or not is_damage(exc):
                        raise
                    reason = _damage(exc)
            if reason is not None:
                damaged_raw[branch] = reason
                self.quarantine_epoch(epoch, "raw", f"its RAW branch {branch} is damaged: {reason}")
                continue
            if "segment" not in info.get("chunk_ids", ()):
                continue
            with self._lock:
                record = self._index.setdefault(key, ArchivedEpoch(key, epoch))
                if record.raw is None:
                    record.raw = {"branch": branch, "chunk_id": "segment"}
                    raw += 1
        self.damaged_raw = damaged_raw
        self.last_reload = {"svd_hevc": count, "raw": raw, "live_filter": None if live is None else len(live),
                            "damaged_lineages": len(damaged), "damaged_raw_branches": len(damaged_raw),
                            "quarantined_epochs": {"svd_hevc": lost, "raw": len(damaged_raw)}}
        return count + raw

    def _core_quarantined(self) -> dict[str, str]:
        """``{ref name: error}`` of the heads the runtime quarantined (``on_open_error="quarantine"``)."""
        return dict(getattr(self.runtime, "quarantined", None) or {})

    def _family_damage(self, epoch: ConditioningCacheEpoch, owners: Iterable[str]) -> str | None:
        """Why a live epoch absent from every healthy lineage may sit on a damaged one: a damaged lineage of a session
        that records it (its base lineage or a ``~rN`` fork; the capturing session always counts) or its holder."""
        holder = holder_branch(epoch_key(epoch))
        bases = {lineage_branch(session) for session in {epoch.session_id, *owners}}
        for branch, reason in sorted(self.damaged_lineages.items()):
            base, _, fork = branch.partition("~r")
            if branch == holder or branch in bases or (fork.isdigit() and base in bases):
                return f"its cache lineage {branch} is quarantined: {reason}"
        return None

    def quarantine_epoch(self, epoch: ConditioningCacheEpoch, representation: str, reason: str) -> None:
        """Mark one epoch's archive of ``representation`` damaged: :meth:`archive_state` reports ``"quarantined"``
        and a restore raises a typed IntegrityError (a process-local mark; the next re-index re-derives it)."""
        if representation not in REPRESENTATIONS:
            raise ConditioningCacheError(f"representation must be one of {REPRESENTATIONS}")
        key = epoch_key(epoch)
        with self._lock:
            record = self._index.setdefault(key, ArchivedEpoch(key, epoch))
            first = representation not in record.quarantined
            record.quarantined[representation] = str(reason)
        if first:
            self._event("quarantined", epoch, representation=representation, reason=str(reason))

    def quarantined_epochs(self, keys: Iterable[tuple] | None = None) -> list[dict]:
        """Every quarantined epoch archive (or those of the epoch ``keys``): segment index, capturing session,
        digest, representation, reason."""
        wanted = None if keys is None else set(keys)
        with self._lock:
            return [{"segment_index": record.epoch.segment_index, "session_id": record.epoch.session_id,
                     "epoch_digest": record.epoch.cond_latents_sha256[:16], "representation": representation,
                     "reason": reason}
                    for record in self._index.values() if wanted is None or record.key in wanted
                    for representation, reason in record.quarantined.items()]

    def damage_report(self) -> dict:
        """Damage known to this backend: the runtime's quarantined heads, damaged LongCat lineages and RAW branches
        of the last (re)index, quarantined epoch archives."""
        return {"runtime_quarantined": self._core_quarantined(), "damaged_lineages": dict(self.damaged_lineages),
                "damaged_raw_branches": dict(self.damaged_raw), "quarantined_epochs": self.quarantined_epochs()}

    @staticmethod
    def _stored_epoch(stored) -> ConditioningCacheEpoch:
        """The port epoch a segment node's cache epoch records (``details["port_epoch"]``)."""
        port = dict(stored.details.get("port_epoch", {}))
        port["token_grid"] = tuple(port["token_grid"])
        port.pop("tokens", None)
        return ConditioningCacheEpoch(**port)

    def _lineage_nodes(self, *, exclude: Iterable[str] = (), holders: bool = True) -> set:
        """Every node ref on the path of a LongCat lineage head (optionally without some lineages / the holders);
        damaged lineages (quarantined by the runtime, or whose path does not load) are skipped and recorded."""
        from reelcache.errors import IntegrityError

        exclude = set(exclude)
        quarantined = self._core_quarantined()
        nodes = set()
        for branch, head in self.runtime.list_lineages().items():
            if not branch.startswith(BRANCH_PREFIX) or branch in exclude:
                continue
            if not holders and branch.startswith(HOLDER_PREFIX):
                continue
            if branch in self.damaged_lineages or _lineage_ref(branch) in quarantined:
                continue
            try:
                nodes.update(ref for ref, _, _ in self.runtime.node_path(head))
            except IntegrityError as exc:
                if self.integrity["on_open_error"] != "quarantine" or not is_damage(exc):
                    raise
                self.damaged_lineages[branch] = _damage(exc)
        return nodes

    def hold_epochs(self, epochs: Iterable[ConditioningCacheEpoch], *, dropping: Iterable[str] = ()) -> list[str]:
        """Keep the published svd_hevc nodes of ``epochs`` restorable when the lineages ``dropping`` are deleted.

        A session's lineage also carries epochs other sessions only restored (the cache is a function of the
        conditioning latents: a fork or rewind restores the source's epoch instead of archiving it again). Before a
        session provider deletes a session's lineages it holds every epoch its live boundaries still record whose
        node no surviving lineage reaches: a holder lineage ``longcat-held~<digest>`` forked at that node (deepest
        first, so one holder keeps the epochs on its path). Returns the holders created."""
        runtime = self.runtime
        lineages = runtime.list_lineages()
        reached = self._lineage_nodes(exclude=dropping)
        wanted = []
        for epoch in epochs:
            key = epoch_key(epoch)
            if self.archive_state(epoch, "svd_hevc") != "published":
                continue
            ref = self._published_node(key)
            if ref is not None and ref not in reached:
                wanted.append((len(runtime.node_path(ref)), ref, key))
        created = []
        for _, ref, key in sorted(wanted, key=lambda item: (-item[0], item[1])):
            if ref in reached:
                continue
            branch = holder_branch(key)
            if branch in lineages:  # an older holder of the same epoch at another node
                runtime.delete_lineage(branch)
            runtime.fork_lineage(ref, branch)
            reached.update(item for item, _, _ in runtime.node_path(ref))
            created.append(branch)
        return created

    def release_holders(self, live_epochs: Iterable[ConditioningCacheEpoch]) -> list[str]:
        """Delete the holder lineages no live epoch needs (its epoch is not live, or a session lineage reaches its
        node); call :meth:`hold_epochs` afterwards for live epochs only a released holder reached. A damaged holder
        (its head does not load) is kept."""
        from reelcache.errors import IntegrityError

        runtime = self.runtime
        live = {epoch_key(epoch) for epoch in live_epochs}
        sessions = self._lineage_nodes(holders=False)
        quarantined = self._core_quarantined()
        released = []
        for branch, head in runtime.list_lineages().items():
            if not branch.startswith(HOLDER_PREFIX):
                continue
            if branch in self.damaged_lineages or _lineage_ref(branch) in quarantined:
                continue
            try:
                _, node, stored = runtime.node_path(head)[-1]
            except IntegrityError as exc:
                if self.integrity["on_open_error"] != "quarantine" or not is_damage(exc):
                    raise
                self.damaged_lineages[branch] = _damage(exc)
                continue
            key = epoch_key(self._stored_epoch(stored)) if node.node_kind == "segment" else None
            if key not in live or head in sessions:
                runtime.delete_lineage(branch)
                released.append(branch)
        return released

    # -- ConditioningCacheBackend: restore ----------------------------------------------------------------------
    def restore(self, epoch: ConditioningCacheEpoch, *,
                overlapped: bool | None = None) -> LongCatConditioningCache | None:
        """Restore an archived epoch with the armed representation (None when unarmed).

        ``overlapped`` (svd_hevc; None: this backend's explicit :meth:`set_overlap`, plain when unset): the cache is
        returned at once
        and a background thread restores it layer by layer (:meth:`_restore_lossy_overlapped`); the caller waits per
        layer and joins it (the stage: after the first denoising step). Its tier is labelled when the restore
        completes (d5): ``last_restore_tier`` reads ``pending`` until then, and the join record carries ``tier``,
        ``bytes`` and ``seconds``. RAW restores are always plain.

        A damaged archive raises a typed :class:`reelcache.errors.IntegrityError` (CR-17-1): the epoch is quarantined,
        the restore's buffers are released and nothing is returned -- never a cache from damaged bytes."""
        from reelcache.errors import IntegrityError

        if overlapped is not None and type(overlapped) is not bool:
            raise ConditioningCacheError("overlapped must be a bool or None")
        representation = self.restore_representation
        self.last_restore_tier = None
        if representation is None:
            return None
        _refuse_row_shard(epoch)
        record = self.lookup(epoch, representation)
        self._note_timeline(epoch, record.key)
        try:
            if representation == "svd_hevc":
                label = self._compressed_state()
                if bool(self.overlap) if overlapped is None else overlapped:
                    # labelled on the restore thread when it completes: a cold restore reads and prepares the node
                    # pack there, after this call returns (d4 labelled it here, before the read)
                    return self._restore_lossy_overlapped(epoch, record, label=label)
                started = _now()
                cache = self._restore_lossy(epoch, record)
                if label is not None:
                    label = label()
                self._note_restore(label or "compressed_cold", cache.nbytes, _now() - started,
                                   node_ref=(record.lossy or {}).get("node_ref"))
                return cache
            return self._restore_raw(epoch, record)
        except IntegrityError as exc:
            _release_frames(exc)
            if not is_damage(exc):
                raise
            raise self._damaged(epoch, representation, exc) from exc

    def _damaged(self, epoch: ConditioningCacheEpoch, representation: str, exc: BaseException):
        """Quarantine an epoch whose restore hit a damaged archive; the typed error the request fails with."""
        from reelcache.errors import IntegrityError

        reason = _damage(exc)
        self.quarantine_epoch(epoch, representation, reason)
        return IntegrityError(
            f"the archived {representation} conditioning cache of segment {epoch.segment_index} (conditioning "
            f"latents {epoch.cond_latents_sha256[:16]}) is damaged and was not restored: {reason}; the epoch is "
            "quarantined (later restores fail the same way; the session's archive is unchanged)")

    # -- freeze D raw tiers -----------------------------------------------------------------------------------------
    def restore_raw_tier(self, epoch: ConditioningCacheEpoch) -> LongCatConditioningCache | None:
        """Restore ``epoch`` from a raw tier when one holds it (``raw_tier`` and ``raw_restore="prefer"``), else None.

        ``gpu_raw``: the retained tensors of the built cache, zero-copy (the stage only reads them; the entry is
        resident until :meth:`cache_released`). ``host_raw``: one direct H2D copy of the pinned slot into fresh
        tensors (no store read, no hash verification; a running D2H is waited for). Both are bitwise the native build
        (pre-RoPE). Published or not: a complete raw copy is the exact cache. Records ``last_restore_tier``."""
        if not self.raw_tier or self.raw_restore == "off":
            return None
        _refuse_row_shard(epoch)
        self.refresh_published()
        key = epoch_key(epoch)
        entry, tier = self._tiers.lookup(key, session_id=epoch.session_id)
        if entry is None:
            return None
        resident = False
        try:
            started = _now()
            if tier == "gpu_raw":
                keys, values = entry.gpu
                metadata = {"restored_from": "gpu_raw", "token": entry.token, "node_ref": entry.node_ref}
                cache = LongCatConditioningCache(epoch=epoch, keys_pre_rope=keys, values=values, source=SOURCE_RESTORED,
                                                 metadata=metadata)
                resident = True
            else:
                if entry.host_event is not None:
                    entry.host_event.synchronize()
                device = self._device()
                keys, values = self._allocate(epoch, device)
                tokens, channels = epoch.tokens, self.heads * self.head_dim
                from reelcache.types import MatrixKey

                for layer in range(self.model.layers):
                    keys[layer].view(tokens, channels).copy_(entry.host[MatrixKey(layer, "K")], non_blocking=True)
                    values[layer].view(tokens, channels).copy_(entry.host[MatrixKey(layer, "V")], non_blocking=True)
                _sync(device)
                metadata = {"restored_from": "host_raw", "token": entry.token, "node_ref": entry.node_ref}
                cache = LongCatConditioningCache(epoch=epoch, keys_pre_rope=keys, values=values, source=SOURCE_RESTORED,
                                                 metadata=metadata)
            seconds = _now() - started
            cache.metadata["restore_seconds"] = seconds
            self._note_restore(tier, entry.nbytes, seconds, node_ref=entry.node_ref, token=entry.token,
                               state=entry.state)
            self._note_timeline(epoch, key)
            self._event("restored", epoch, representation="raw_tier", tier=tier, token=entry.token,
                        node_ref=entry.node_ref, state=entry.state, seconds=seconds, bytes=entry.nbytes)
            return cache
        finally:
            self._tiers.unpin(entry, resident=resident)

    def raw_tier_state(self, epoch: ConditioningCacheEpoch) -> str | None:
        """The raw tier that would serve ``epoch`` now (``gpu_raw`` / ``host_raw``), or None."""
        if not self.raw_tier or self.raw_restore == "off":
            return None
        entry = self._tiers.by_key(epoch_key(epoch))
        if entry is None:
            return None
        if entry.gpu is not None and not entry.resident:
            return "gpu_raw"
        return "host_raw" if entry.host is not None else None

    def cache_released(self, epoch: ConditioningCacheEpoch) -> None:
        """The stage dropped the cache of ``epoch`` (after its denoising): a retained GPU copy becomes ``gpu_raw`` and
        the GPU raw budget applies (A100-80GB: 0, so it is dropped at once)."""
        if self.raw_tier:
            self._tiers.released(epoch_key(epoch))

    def release_resident(self) -> dict:
        """No segment runs (request admission or finish, also after a failed segment): every resident GPU copy
        becomes ``gpu_raw`` and the budget applies, so a cache is never held past its request."""
        if not self.raw_tier:
            return {"dropped": 0, "bytes": 0}
        dropped = self._tiers.release_all()
        return {"dropped": len(dropped), "bytes": sum(entry.nbytes for entry in dropped)}

    def enforce_gpu_raw_budget(self, budget: int | None = None) -> dict:
        """Drop retained GPU raw copies beyond the budget (request admission, memory-critical section entry)."""
        dropped = self._tiers.enforce_gpu_budget(budget) if self.raw_tier else []
        return {"dropped": len(dropped), "bytes": sum(entry.nbytes for entry in dropped)}

    def make_room(self, *, timeout: float | None = None) -> dict:
        """Last resort when no host slot is free (every slot holds an unpublished epoch): wait for the oldest
        submitted epoch to publish (eager), or drain everything (a deferred schedule has nothing submitted yet).
        Called outside memory-critical sections (at capture); recorded with its cause."""
        started = _now()
        cause, waited, blocked = "host_full", 0, False
        timeout = self.timeout if timeout is None else timeout
        while self.free_capture_slots() < 1:
            blocked = True
            with self._lock:
                submitted = [c for c in self._pending if c.ticket is not None]
            if not submitted:
                self.flush(wait=True)
                break
            oldest = min(submitted, key=lambda c: c.times.get("submitted_at", 0.0))
            try:
                ref = oldest.ticket.result(timeout)
            except BaseException:
                self._tiers.failed(oldest.token)
                raise
            self._publish_capture(oldest, ref)
            waited += 1
        seconds = _now() - started
        if blocked:
            self._tiers.note_backpressure(seconds, cause)
        return {"seconds": seconds, "cause": cause if blocked else None, "published": waited}

    def _compressed_state(self):
        """A callable giving ``compressed_prepared`` / ``compressed_cold`` for the restore about to run: cold when
        the restorer staged the history (cold session) or loaded a pack from the store (prepared-cache miss)."""
        restorer = self.restorer
        if restorer is None or not hasattr(restorer, "stats"):
            return None
        try:
            prepared = restorer.stats().get("prepared")
        except Exception:  # noqa: BLE001 - records only
            return None
        if not isinstance(prepared, Mapping) or prepared.get("misses") is None:
            return lambda: "compressed_cold"  # no prepared cache (software restorer): every restore reads the store
        before = prepared["misses"]
        if hasattr(restorer, "last_staging"):
            restorer.last_staging = None

        def label() -> str:
            try:
                after = (restorer.stats().get("prepared") or {}).get("misses")
            except Exception:  # noqa: BLE001
                after = before
            staged = getattr(restorer, "last_staging", None) is not None
            return "compressed_cold" if staged or (before is not None and after is not None and after > before) \
                else "compressed_prepared"
        return label

    def _note_restore(self, tier: str, nbytes: int, seconds: float, **detail) -> None:
        if tier not in RECORD_TIERS:
            raise ConditioningCacheError(f"unknown tier {tier!r}")
        self.last_restore_tier = {"tier": tier, "bytes": int(nbytes), "seconds": float(seconds), **detail}
        with self._lock:
            row = self._restore_tiers.setdefault(tier, {"count": 0, "bytes": 0, "seconds": 0.0})
            row["count"] += 1
            row["bytes"] += int(nbytes)
            row["seconds"] += float(seconds)

    def _note_commit_to_publish(self, capture: _Capture, node_ref: str) -> None:
        seconds = capture.times.get("published_after_submit_seconds")
        if self.pipeline is not None and capture.branch is not None:
            node = node_id(capture.epoch)
            try:
                for item in reversed(self.pipeline.stats().get("timeline") or []):
                    if item.get("node_id") == node and item.get("branch") == capture.branch:
                        seconds = item.get("store_end", seconds)
                        break
            except Exception:  # noqa: BLE001 - records only
                pass
        selected = (self.encoder_info or {}).get("selected")
        with self._lock:
            self._commit_to_publish.append({"branch": capture.branch, "node_id": node_id(capture.epoch),
                                            "seconds": seconds, "node_ref": node_ref})
            del self._commit_to_publish[:-256]
            if selected is not None:
                self._encoder_commits[selected] = self._encoder_commits.get(selected, 0) + 1

    def _on_evict(self, entry: RawEntry) -> None:
        """A published epoch lost its last raw copy: queue its compressed node for the prepared state at the next
        idle point (:meth:`prepare_evicted`), so its next restore is ``compressed_prepared``, not cold."""
        if entry.node_ref is None or entry.state != "published":
            return
        with self._lock:
            self._evicted.append(entry.epoch)
            self._prepare_on_evict["queued"] += 1

    def prepare_evicted(self) -> dict:
        """Idle point (request end, pause, ``drop_raw``): keep the compressed nodes of epochs whose last raw copy was
        evicted since the last call prepared (review amendment 3; Step 7 R6), so their next restore is
        ``compressed_prepared``, not cold. With Step 7's ``LineageRestorer.ensure_prepared`` (dev22a): their node packs
        are touched in the prepared LRU or prepared host-side (no device work). Without it: ``prepare_epochs``
        (prepared packs + device bases, bounded by the restorer's budgets). Never on the generation path."""
        with self._lock:
            epochs, self._evicted = self._evicted, []
        if not epochs or self.restorer is None:
            return {"epochs": 0}
        started = _now()
        report = {"epochs": len(epochs)}
        try:
            with self._lock:
                records = [self._index.get(epoch_key(e)) for e in epochs]
            known = [record for record in records if record is not None and record.lossy is not None
                     and record.lossy.get("node_ref") is not None and "svd_hevc" not in record.quarantined]
            ensure = getattr(self.restorer, "ensure_prepared", None)
            if known and ensure is not None:
                packets = []
                for record in known:
                    for item in self.restorer.history(record.lossy["node_ref"]):
                        packet = item.get("packet")
                        if packet is not None and packet not in packets:
                            packets.append(packet)
                result = dict(ensure(packets))
                report.update(mode="ensure_prepared", packets=len(packets), result=result)
                with self._lock:
                    self._prepare_on_evict["prepared"] += int(result.get("prepared", 0))
                    self._prepare_on_evict["already_prepared"] += int(result.get("touched", 0))
                    self._prepare_on_evict["errors"] += int(result.get("failed", 0))
            elif known:
                result = self.prepare_epochs([record.epoch for record in known])
                report.update(mode="prepare_epochs",
                              result={k: v for k, v in result.items() if k in ("seconds", "quarantined", "skipped")})
                with self._lock:
                    self._prepare_on_evict["prepared"] += len(known)
        except Exception as exc:  # noqa: BLE001 - a later restore of the epoch is cold instead (recorded)
            report["error"] = f"{type(exc).__name__}: {exc}"[:300]
            with self._lock:
                self._prepare_on_evict["errors"] += 1
        seconds = _now() - started
        with self._lock:
            self._prepare_on_evict["seconds"] += seconds
        report["seconds"] = seconds
        return report

    def demote_raw(self, session_id: str | None = None) -> dict:
        """Step 7 ``demote_raw``: GPU raw → host raw for the session's epochs (all with None)."""
        return self._tiers.demote(session_id)

    def drop_raw(self, session_id: str | None = None, *, drain: bool = True) -> dict:
        """Step 7 ``drop_raw``: drain (default), then drop both raw tiers of the session's published epochs, leaving
        compressed only (their nodes are queued for the prepared state)."""
        if drain:
            self.flush(wait=True)
        result = self._tiers.drop(session_id)
        result["prepared"] = self.prepare_evicted()
        return result

    def reset_raw_peaks(self) -> None:
        self._tiers.reset_peaks()

    def raw_tier_stats(self) -> dict:
        """Step 7 §9 ``pipeline.raw_tier`` plus restores per tier, prepare-on-evict, commit→publish and the encoder."""
        self.refresh_published()
        stats = self._tiers.stats()
        with self._lock:
            recent = list(self._commit_to_publish)
            restores = {tier: dict(row) for tier, row in self._restore_tiers.items()}
            prepare = dict(self._prepare_on_evict)
            commits = dict(self._encoder_commits)
        values = sorted(row["seconds"] for row in recent if row.get("seconds") is not None)

        def quantile(q: float):
            return None if not values else values[min(len(values) - 1, int(round(q * (len(values) - 1))))]

        return {"raw_tier": {**stats, "enabled": self.raw_tier, "raw_restore": self.raw_restore,
                             "restores": restores, "prepare_on_evict": prepare},
                "commit_to_publish": {"count": len(values), "mean": sum(values) / len(values) if values else None,
                                      "p50": quantile(0.5), "p95": quantile(0.95), "max": values[-1] if values else None,
                                      "last": recent[-1]["seconds"] if recent else None, "recent": recent[-16:]},
                "encoder": {**(self.encoder_info or {}), "commits": commits}}

    def raw_entries(self, session_id: str | None = None) -> list[dict]:
        return self._tiers.entries(session_id)

    def warmup(self, epoch: ConditioningCacheEpoch | None = None, representation: str = "svd_hevc") -> dict:
        """Restore one archived epoch and discard it, off the request path (session open / after archival).

        The first restore of a process pays one-time costs (Triton kernel compilation or its disk-cache load,
        decoder sessions for a new canvas size, prepared packs and device bases of a store-cold node); this moves
        them out of the first resume. ``epoch=None`` picks the most recently archived epoch of ``representation``.
        """
        if epoch is None:
            with self._lock:
                records = [r for r in self._index.values()
                           if (r.lossy if representation == "svd_hevc" else r.raw) is not None]
            if not records:
                raise ConditioningCacheError(f"warmup needs an archived {representation} epoch")
            epoch = records[-1].epoch
        record = self.lookup(epoch, representation)
        started = _now()
        cache = (self._restore_lossy(epoch, record) if representation == "svd_hevc"
                 else self._restore_raw(epoch, record))
        seconds = _now() - started
        self.events[-1]["event"] = "warmup_restore"
        del cache
        return {"representation": representation, "segment_index": epoch.segment_index, "seconds": seconds}

    # -- warm-up: process level and session level (CPU packing -> the prepared state) -------------------------------
    @_outside_inference_mode
    def prewarm_process(self, token_grids: Sequence[Sequence[int]] = ((4, 44, 82),), *, max_rank: int | None = None,
                        kernels: bool = True, decoders: bool = True) -> dict:
        """Process-level warm-up at backend attach, independent of any session (no archived data is read).

        Restore accounting (user rule of 2026-10-15 22:51): excluded from restore time. ``kernels``:
        ``restore_kernels.warmup`` compiles/loads every Triton variant the packed restore launches for LongCat's
        launch groups (BTHD BF16 cache, picture sources, no rotation -- pre-RoPE K and V -- ranks 1 .. ``max_rank``,
        default the model's channels). ``decoders``: a decoder session for every canonical node canvas of the
        conditioning-cache tile geometry of each ``token_grids`` entry, up to ``max_rank``, on every pool worker
        (``PackedRestorer.warmup``). A restorer with a cold-session staging engine (CR-17-4) allocates its pinned
        ring and starts its worker threads (``StagingEngine.warmup``). A core without these APIs raises (no silent
        partial warm-up).
        """
        from reelcache import restore_kernels as rk

        started = _now()
        channels = self.heads * self.head_dim
        top = channels if max_rank is None else int(max_rank)
        if not 1 <= top <= channels:
            raise ConditioningCacheError(f"max_rank must be in [1, {channels}]")
        device = self._device()
        report = {"level": "process", "max_rank": top, "token_grids": [list(g) for g in token_grids]}
        if kernels:
            if not hasattr(rk, "warmup") or not hasattr(rk, "WarmupShape"):
                raise ConditioningCacheError("this reelcache core has no restore_kernels.warmup (process-level kernel "
                                             "warm-up); use a core with it or skip kernels explicitly")
            shape = rk.WarmupShape(heads=self.heads, head_dim=self.head_dim, dtype=self._dtype(), layout="BTHD",
                                   source="pictures", rope="none", ranks=(1, top))
            t0 = _now()
            warm = rk.warmup(device, [shape])
            report["kernels"] = {k: v for k, v in warm.items() if k != "shapes"}
            report["kernels_seconds"] = _now() - t0
        if decoders:
            packed = getattr(self.restorer, "restorer", None)
            if packed is None or not hasattr(packed, "warmup"):
                if device.type == "cuda":
                    raise ConditioningCacheError("the restorer has no PackedRestorer.warmup (canonical canvas decoder "
                                                 "sessions); use a core with it or skip decoders explicitly")
                report["decoders"] = f"{type(self.restorer).__name__}: no hardware decoder sessions to warm"
            else:
                geometries = sorted({(int(g[0]), int(g[1]), int(g[2]), self.pix_fmt, channels) for g in token_grids})
                t0 = _now()
                report["decoders"] = packed.warmup(geometries, max_rank=top)
                report["decoders_seconds"] = _now() - t0
        engine = getattr(self.restorer, "staging_engine", None)
        if engine is not None:  # cold-session staging (CR-17-4): pinned ring allocated, worker threads started
            report["staging_engine"] = engine.warmup()
        report["seconds"] = _now() - started
        self.process_warm = report
        return report

    @_outside_inference_mode
    def prepare_epochs(self, epochs: Sequence[ConditioningCacheEpoch] | None = None) -> dict:
        """Session-level CPU packing of archived svd_hevc epochs: brings them to the prepared state.

        Reads and prepares the epochs' node packs, makes their basis versions device-resident and creates decoder
        sessions for their canvas sizes (``LineageRestorer.prewarm(history)``; bounded by the restorer's budgets).
        A restore after this runs in the prepared state, like a restore of a node sealed in the same process: the
        restore requirement row (user rule of 2026-10-15 22:51). Its own cost (store reads, pack preparation, basis
        assembly and upload) is the CPU packing time, measured and reported separately (an optimization goal, not
        part of the requirement). ``epochs=None`` prepares every indexed svd_hevc epoch.

        Damaged archives (CR-17-1; runtime ``on_open_error="quarantine"``): quarantined epochs are skipped; when
        preparing hits a damaged archive, each epoch is prepared on its own and the damaged ones are quarantined
        (``report["quarantined"]``: their segment indices) -- the others reach the prepared state.
        """
        from reelcache.errors import IntegrityError

        started = _now()
        with self._lock:
            if epochs is None:
                records = [r for r in self._index.values() if r.lossy is not None]
            else:
                records = [self._index.get(epoch_key(epoch)) for epoch in epochs]
        if not records or any(r is None or r.lossy is None for r in records):
            raise ConditioningCacheError("prepare_epochs needs archived svd_hevc epochs (archive and flush, or "
                                         "reload_index)")
        if self.restorer is None:
            raise ConditioningCacheError("prepare_epochs needs a restorer")
        skipped = [r.epoch.segment_index for r in records if "svd_hevc" in r.quarantined]
        records = [r for r in records if "svd_hevc" not in r.quarantined]
        refs = []
        for record in records:
            if record.lossy.get("node_ref") is None:
                record.lossy["node_ref"] = record.lossy["ticket"].result(self.timeout)
            refs.append(record.lossy["node_ref"])
        report = {"level": "session (CPU packing -> prepared state)", "epochs": [r.epoch.segment_index for r in records],
                  "node_refs": refs, "skipped_quarantined": skipped}
        warm = hasattr(self.restorer, "prewarm")

        def stage(subset, seen):
            items = []
            for record in subset:
                for item in self.restorer.history(record.lossy["node_ref"]):
                    ident = item.get("packet") or item.get("node_id") or repr(item)
                    if ident not in seen:
                        seen.add(ident)
                        items.append(item)
            if not warm:
                return items, None, 0.0
            t0 = _now()
            decoder = self.restorer.prewarm(items)
            return items, decoder, _now() - t0

        quarantined = []
        try:
            items, decoder_seconds, prepare_seconds = stage(records, set())
        except IntegrityError as exc:
            if self.integrity["on_open_error"] != "quarantine" or not is_damage(exc):
                raise
            _release_frames(exc)
            items, decoder_seconds, prepare_seconds, seen = [], 0.0, 0.0, set()
            for record in records:
                try:
                    part, decoder, seconds = stage([record], seen)
                except IntegrityError as inner:
                    _release_frames(inner)
                    if not is_damage(inner):
                        raise
                    self.quarantine_epoch(record.epoch, "svd_hevc", _damage(inner))
                    quarantined.append(record.epoch.segment_index)
                    continue
                items += part
                decoder_seconds += decoder or 0.0
                prepare_seconds += seconds
        report["quarantined"] = quarantined
        report["packets"] = len(items)
        if warm:
            report["decoder_session_seconds"] = decoder_seconds
            report["packs_bases_decoders_seconds"] = prepare_seconds
            bases = getattr(self.restorer, "bases", None)
            report["bases"] = bases.stats() if bases is not None and hasattr(bases, "stats") else None
        else:
            report["packs_bases_decoders_seconds"] = None
            report["staging"] = f"{type(self.restorer).__name__} keeps no prepared packs or device bases"
        report["seconds"] = _now() - started
        if records:
            self._event("prepared_epochs", records[0].epoch, **{k: v for k, v in report.items() if k != "bases"})
        return report

    def prewarm(self, epochs: Sequence[ConditioningCacheEpoch] | None = None, *, kernels: bool = True,
                token_grids: Sequence[Sequence[int]] = ((4, 44, 82),), max_rank: int | None = None) -> dict:
        """``prewarm_process`` (process level) + ``prepare_epochs`` (session level: CPU packing to the prepared state).

        Restores after this run in the prepared state (the requirement row); cold-session restores after
        ``prewarm_process`` only include the CPU packing (informational).
        """
        report = {}
        if kernels:
            report["process"] = self.prewarm_process(token_grids, max_rank=max_rank)
        report["session"] = self.prepare_epochs(epochs)
        report["seconds"] = sum(part["seconds"] for part in report.values() if isinstance(part, dict))
        return report

    def set_defer_planning(self, enabled: bool, *, release_cached_memory: bool = False) -> None:
        """While generation runs a memory-critical section (``generation_section``: the stage's denoising and VAE
        decode), no GPU planning runs: archival of earlier segments then plans only between those sections. Needs a
        CUDA archive pipeline. With the pipeline's group-level gate (:attr:`planning_gate`, g7) a section waits for
        the planner's current matrix group; otherwise (``ArchivePipeline.defer_gpu_work``, g6) for its current node.
        ``release_cached_memory``: when a section opens, the CUDA caching allocator's unoccupied blocks are returned
        to the device (the planner's freed sources and workspace live in its own stream's pool; CR-17-6)."""
        if type(enabled) is not bool or type(release_cached_memory) is not bool:
            raise ConditioningCacheError("defer_planning and release_cached_memory must be boolean")
        if enabled and (self.pipeline is None or torch.device(self.pipeline.device).type != "cuda"):
            raise ConditioningCacheError("defer_planning applies to a CUDA archive pipeline (GPU planning)")
        self.defer_planning = enabled
        self.release_planner_memory = enabled and release_cached_memory
        gate = self.planning_gate
        if gate is not None:
            gate.configure(enabled=enabled, release_cached_memory=self.release_planner_memory)

    @property
    def planning_gate(self):
        """The archive fitter's group-level :class:`~.planning_gate.PlanningGate` (g7) when the pipeline plans host
        sources (a CPU pipeline, or ``deferred_sources="host"``: a node's sources stay on the host except the group
        being planned, and its finalization runs on the host), else None."""
        if self.pipeline is None:
            return None
        if torch.device(self.pipeline.device).type == "cuda" and self.deferred_sources != "host":
            return None
        return getattr(getattr(self.pipeline, "fitter", None), "gate", None)

    def planning_gate_stats(self) -> dict | None:
        gate = self.planning_gate
        if gate is None:
            return None
        fitter = getattr(self.pipeline, "fitter", None)
        return {**gate.stats(), "last_residency": getattr(fitter, "last_residency", None)}

    def generation_section(self):
        """Context manager the stage opens around denoising and decoding when ``defer_planning`` is on: the
        planning gate's section (entry waits for the planner's current matrix group, g7), else ``defer_gpu_work``
        of the GPU archive pipeline (entry waits until the fitter is outside GPU work, at most one node); a no-op
        without ``defer_planning``."""
        import contextlib

        if self.raw_tier:
            self.enforce_gpu_raw_budget()  # freeze D: retained GPU raw beyond the budget leaves before the section
        if not self.defer_planning:
            return contextlib.nullcontext()
        gate = self.planning_gate
        if gate is not None:
            return gate.section(timeout=self.timeout)
        return self.pipeline.defer_gpu_work(wait=True, timeout=self.timeout,
                                            release_cached_memory=self.release_planner_memory)

    def set_overlap(self, enabled: bool | None) -> None:
        """An explicit layer-ahead setting for svd_hevc restores (the cache returns at once and completes layer by
        layer in a background thread). Set, it applies to direct ``restore()`` calls and -- as in d4 -- to session
        requests without a handle override (tools, harnesses, tests). Unset (``None``, the default), direct calls are
        plain and session requests follow the provider's ``overlap_restore`` (on since d5). ``restore(...,
        overlapped=)`` overrides it per call."""
        if enabled is not None and type(enabled) is not bool:
            raise ConditioningCacheError("overlap must be a bool or None")
        self.overlap = enabled

    def lookup(self, epoch: ConditioningCacheEpoch, representation: str) -> ArchivedEpoch:
        from reelcache.errors import IntegrityError

        key = epoch_key(epoch)
        with self._lock:
            record = self._index.get(key)
            queued = any(capture.key == key and representation in capture.representations
                         for capture in self._pending)
        if record is not None and representation in record.quarantined:
            raise IntegrityError(
                f"the archived {representation} conditioning cache of segment {epoch.segment_index} (conditioning "
                f"latents {epoch.cond_latents_sha256[:16]}) is quarantined: {record.quarantined[representation]}; "
                "refusing to restore it or to rebuild silently")
        if record is None or (record.lossy if representation == "svd_hevc" else record.raw) is None:
            hint = " (captured but not archived yet: call flush())" if queued else ""
            raise ConditioningCacheError(
                f"{representation} restore of segment {epoch.segment_index} requested, but no archived epoch matches "
                f"conditioning latents {epoch.cond_latents_sha256[:16]}{hint}; refusing to rebuild silently")
        return record

    def _device(self) -> torch.device:
        if self.restore_device is not None:
            return self.restore_device
        device = getattr(self.restorer, "device", None)
        if device is not None:
            return torch.device(device)
        if self.plan_device is not None:
            return self.plan_device
        return torch.device("cpu")

    def _allocate(self, epoch: ConditioningCacheEpoch, device: torch.device):
        shape = (1, epoch.tokens, self.heads, self.head_dim)
        dtype = self._dtype()
        keys = tuple(torch.empty(shape, dtype=dtype, device=device) for _ in range(self.model.layers))
        values = tuple(torch.empty(shape, dtype=dtype, device=device) for _ in range(self.model.layers))
        return keys, values

    def _check_node(self, node_ref: str, epoch: ConditioningCacheEpoch) -> dict:
        runtime = self.runtime
        node = runtime.load_node(node_ref)
        stored = runtime.load_epoch(node.epoch_ref)
        port = stored.details.get("port_epoch", {})
        if (node.node_kind != "segment" or stored.cause != "longcat_segment" or stored.position_space != "segment_local"
                or port.get("cond_latents_sha256") != epoch.cond_latents_sha256
                or port.get("model_fingerprint") != epoch.model_fingerprint
                or list(port.get("token_grid", [])) != list(epoch.token_grid)
                or node.token_start != 0 or node.token_end != epoch.tokens
                or (node.frames, node.height, node.width) != tuple(epoch.token_grid)):
            raise ConditioningCacheError(f"archived node {node_ref} does not hold the requested epoch {epoch}")
        return {"node_kind": node.node_kind, "epoch_id": stored.epoch_id, "cause": stored.cause,
                "position_space": stored.position_space, "archived_port_epoch": dict(port),
                "parent_epoch_ref": stored.parent_epoch_ref}

    @_outside_inference_mode
    def _restore_lossy(self, epoch: ConditioningCacheEpoch, record: ArchivedEpoch) -> LongCatConditioningCache:
        from reelcache import restore_kernels as rk

        if self.restorer is None:
            raise ConditioningCacheError("svd_hevc restore needs a LineageRestorer (GPU) or SoftwareLineageRestorer")
        started = _now()
        lossy = record.lossy
        node_ref = lossy.get("node_ref")
        if node_ref is None:
            node_ref = lossy["ticket"].result(self.timeout)
            lossy["node_ref"] = node_ref
        published = _now()
        identity = self._check_node(node_ref, epoch)
        checked = _now()
        device = self._device()
        keys, values = self._allocate(epoch, device)
        allocated = _now()
        written = set()

        def destination(_node_id, key, tile):
            if int(tile.token_start) != 0:
                raise ConditioningCacheError("a LongCat segment epoch is segment-local (token_start 0)")
            written.add((key.layer, key.target))
            target = keys[key.layer] if key.target == "K" else values[key.layer]
            return rk.NativeView(target, LAYOUT_BTHD, token_offset=0)

        result = self.restorer.restore(node_ref, destination, None, **self.restore_options)
        _sync(device)
        finished = _now()
        expected = {(layer, target) for layer in range(self.model.layers) for target in ("K", "V")}
        if written != expected or result.get("entries") != len(expected):
            raise ConditioningCacheError(
                f"restore wrote {len(written)} of {len(expected)} matrices ({result.get('entries')} entries)")
        stats = {key: value for key, value in result.items() if key not in ("handles", "history")}
        info = {"representation": "svd_hevc", "node_ref": node_ref, "identity": identity,
                "wait_published_seconds": published - started, "check_seconds": checked - published,
                "allocate_seconds": allocated - checked, "restorer_seconds": result.get("seconds"),
                "restorer_breakdown": result.get("breakdown"), "restorer": stats,
                "backend_seconds": finished - started}
        self._event("restored", epoch, **info)
        return LongCatConditioningCache(epoch=epoch, keys_pre_rope=keys, values=values, source=SOURCE_RESTORED,
                                        metadata={"restored_from": "reelcache_svd_hevc", "node_ref": node_ref,
                                                  "restore_seconds": finished - started})

    @_outside_inference_mode
    def _restore_lossy_overlapped(self, epoch: ConditioningCacheEpoch, record: ArchivedEpoch, *,
                                  label: Callable[[], str] | None = None) -> LongCatConditioningCache:
        """Layer-ahead svd_hevc restore: returns the allocated cache at once; a background thread restores it.

        The restorer issues layers in increasing order (decodes of all layers submitted up front, then one
        reconstruction per layer on this restore's side stream). When it starts issuing layer ``L``, every earlier
        layer has been issued: its readiness is marked then (a CUDA event on the side stream), the last layers after
        the restore returns. The DiT's first step waits per layer (``LayerReadiness.wait``), so decode and
        reconstruction of later layers overlap the forward of earlier ones. Reading and preparing the node pack and
        uploading its bases precede the first layer (one node pack object; per-layer reads are a core pack-format
        matter). A restorer that issues layers out of order makes the restore fail explicitly.

        The tier (``label``: :meth:`_compressed_state`, evaluated after the restore) is noted when the thread
        completes (``_note_restore``; ``last_restore_tier`` is ``pending`` until then) and added to the readiness
        record (``tier``, ``bytes``, ``seconds`` from this call).
        """
        import threading

        started = _now()
        lossy = record.lossy
        node_ref = lossy.get("node_ref")
        if node_ref is None:
            node_ref = lossy["ticket"].result(self.timeout)
            lossy["node_ref"] = node_ref
        identity = self._check_node(node_ref, epoch)
        device = self._device()
        keys, values = self._allocate(epoch, device)
        layers = self.model.layers
        nbytes = sum(t.numel() * t.element_size() for t in (*keys, *values))
        ready = LayerReadiness(layers, device)
        stream = torch.cuda.Stream(device) if device.type == "cuda" else None
        progressive = hasattr(self.restorer, "restorer")  # LineageRestorer: layer-ordered reconstruction loop
        state = {"current": -1, "seen": [], "written": set()}
        lock = threading.Lock()

        def destination(_node_id, key, tile):
            if int(tile.token_start) != 0:
                raise ConditioningCacheError("a LongCat segment epoch is segment-local (token_start 0)")
            layer = int(key.layer)
            with lock:
                if layer != state["current"]:
                    if layer < state["current"] or layer in state["seen"]:
                        raise ConditioningCacheError("the restorer issued layers out of order; the overlapped restore "
                                                     "needs layer-ordered reconstruction")
                    if progressive:  # every layer before ``layer`` has been issued on ``stream``
                        for done in range(layer):
                            if not ready.ready(done):
                                ready.mark(done, stream)
                    state["current"] = layer
                    state["seen"].append(layer)
                state["written"].add((layer, key.target))
            from reelcache import restore_kernels as rk
            target = keys[layer] if key.target == "K" else values[layer]
            return rk.NativeView(target, LAYOUT_BTHD, token_offset=0)

        def run():
            try:
                if device.type == "cuda":
                    torch.cuda.set_device(device)
                t0 = _now()
                context = torch.cuda.stream(stream) if stream is not None else None
                if context is not None:
                    with context:
                        result = self.restorer.restore(node_ref, destination, None, stream=stream,
                                                       **self.restore_options)
                else:
                    result = self.restorer.restore(node_ref, destination, None, **self.restore_options)
                for layer in range(layers):
                    if not ready.ready(layer):
                        ready.mark(layer, stream)
                if stream is not None:
                    stream.synchronize()
                finished = _now()
                expected = {(layer, target) for layer in range(layers) for target in ("K", "V")}
                if state["written"] != expected or result.get("entries") != len(expected):
                    raise ConditioningCacheError(f"restore wrote {len(state['written'])} of {len(expected)} matrices "
                                                 f"({result.get('entries')} entries)")
                first_mark = min(t for t in ready.marked_at if t is not None)
                ready.info.update({
                    "representation": "svd_hevc", "node_ref": node_ref, "identity": identity, "overlapped": True,
                    "progressive_marks": progressive, "restorer_seconds": result.get("seconds"),
                    "restorer_breakdown": result.get("breakdown"),
                    "restore_complete_seconds_from_call": finished - started,
                    "first_layer_ready_seconds_from_call": first_mark - started,
                    "last_layer_ready_seconds_from_call": max(t for t in ready.marked_at if t is not None) - started,
                    "thread_start_delay_seconds": t0 - started})
                tier = None
                if label is not None:
                    try:
                        tier = label()
                    except Exception:  # noqa: BLE001 - records only
                        tier = None
                tier = tier or "compressed_cold"
                ready.info.update(tier=tier, bytes=nbytes, seconds=finished - started)
                self._note_restore(tier, nbytes, finished - started, node_ref=node_ref, overlapped=True)
                self._event("restored", epoch, **ready.info)
            except BaseException as exc:  # noqa: BLE001 - every waiter re-raises it
                from reelcache.errors import IntegrityError

                if isinstance(exc, IntegrityError):
                    _release_frames(exc)
                    if is_damage(exc):
                        exc = self._damaged(epoch, "svd_hevc", exc)
                ready.fail(exc)

        cache = LongCatConditioningCache(epoch=epoch, keys_pre_rope=keys, values=values, source=SOURCE_RESTORED,
                                         metadata={"restored_from": "reelcache_svd_hevc", "node_ref": node_ref,
                                                   "overlapped": True}, ready=ready)
        # until the thread completes; set before it starts, so its completed record is never overwritten
        self.last_restore_tier = {"tier": None, "pending": True, "overlapped": True, "bytes": nbytes, "seconds": None,
                                  "node_ref": node_ref}
        thread = threading.Thread(target=run, name="longcat-overlapped-restore", daemon=True)
        ready.attach(thread)
        thread.start()
        return cache

    @_outside_inference_mode
    def _restore_raw(self, epoch: ConditioningCacheEpoch, record: ArchivedEpoch) -> LongCatConditioningCache:
        from reelcache import restore_kernels as rk
        from reelcache.types import MatrixKey

        started = _now()
        branch = record.raw["branch"]
        device = self._device()
        keys, values = self._allocate(epoch, device)
        grid = tuple(epoch.token_grid)
        read_seconds = install_seconds = 0.0
        backend = self.raw_install_backend if device.type == "cuda" else "reference"
        for layer in range(self.model.layers):
            t0 = _now()
            leases = [self.runtime.acquire_history(branch, MatrixKey(layer, target), device=device,
                                                   dtype=self._dtype()) for target in ("K", "V")]
            t1 = _now()
            try:
                jobs = [rk.ReconstructionJob(rk.DenseSource(lease.tensor),
                                             rk.NativeView(dest[layer], LAYOUT_BTHD, token_offset=0), grid,
                                             label=f"raw:{layer}:{target}")
                        for lease, dest, target in zip(leases, (keys, values), ("K", "V"))]
                handle = rk.reconstruct(jobs, backend=backend)
                handle.synchronize()
                handle.release()
            finally:
                for lease in leases:
                    lease.close()
            read_seconds += t1 - t0
            install_seconds += _now() - t1
        _sync(device)
        info = {"representation": "raw", "branch": branch, "read_seconds": read_seconds,
                "install_seconds": install_seconds, "backend_seconds": _now() - started, "install_backend": backend}
        self._event("restored", epoch, **info)
        return LongCatConditioningCache(epoch=epoch, keys_pre_rope=keys, values=values, source=SOURCE_RESTORED,
                                        metadata={"restored_from": "reelcache_raw", "branch": branch})

    # -- reporting ------------------------------------------------------------------------------------------------
    def _event(self, name: str, epoch: ConditioningCacheEpoch, **fields) -> None:
        self.events.append({"event": name, "time": time.time(), "segment_index": epoch.segment_index,
                            "session_id": epoch.session_id, "cond_latents_sha256": epoch.cond_latents_sha256,
                            **fields})

    def archived(self) -> list[dict]:
        with self._lock:
            return [record.summary() for record in self._index.values()]

    def node_ref(self, epoch: ConditioningCacheEpoch) -> str | None:
        """The published svd_hevc node of an epoch (None when not archived/published or quarantined)."""
        with self._lock:
            record = self._index.get(epoch_key(epoch))
        if record is None or record.lossy is None or "svd_hevc" in record.quarantined:
            return None
        return record.lossy.get("node_ref")

    def pending(self) -> int:
        with self._lock:
            return len(self._pending)

    # -- session-provider support (Step 14 CR-14-3) -------------------------------------------------------------
    def pending_epochs(self, session_id: str | None = None) -> list[dict]:
        """Captured epochs not yet drained (all, or one session's): segment index, digest, submitted or not."""
        with self._lock:
            return [{"segment_index": capture.epoch.segment_index, "session_id": capture.epoch.session_id,
                     "epoch_digest": capture.epoch.cond_latents_sha256[:16], "submitted": capture.ticket is not None}
                    for capture in self._pending if session_id is None or capture.epoch.session_id == session_id]

    def archive_state(self, epoch: ConditioningCacheEpoch, representation: str) -> str | None:
        """``"published"`` (restorable now), ``"pending"`` (captured or in archival), ``"failed"`` (its archival
        raised), ``"quarantined"`` (its archive is damaged: a restore raises a typed IntegrityError; CR-17-1) or None
        (never archived) -- for callers that restore only what is published and build otherwise."""
        if representation not in REPRESENTATIONS:
            raise ConditioningCacheError(f"representation must be one of {REPRESENTATIONS}")
        key = epoch_key(epoch)
        with self._lock:
            record = self._index.get(key)
            queued = any(capture.key == key and representation in capture.representations
                         for capture in self._pending)
        if record is not None and representation in record.quarantined:
            return "quarantined"
        entry = None if record is None else (record.lossy if representation == "svd_hevc" else record.raw)
        if entry is None:
            return "pending" if queued else None
        if representation == "raw" or entry.get("node_ref") is not None:
            return "published"
        ticket = entry.get("ticket")
        if ticket is not None and ticket.done():
            try:
                entry["node_ref"] = ticket.result(0)
            except Exception:  # noqa: BLE001 - reported as a failed archival, raised again by drain()
                return "failed"
            return "published"
        return "pending"

    def free_capture_slots(self) -> int:
        """Slots a capture can take without waiting: free ones plus published raw-tier entries it may evict."""
        self.refresh_published()
        return self._tiers.free_slots()

    def discard_pending(self, session_id: str) -> int:
        """Retire one session's captured epochs that were never submitted for archival (``close``/``delete`` with
        abort); submitted ones publish (drain them). Returns the number retired."""
        with self._lock:
            retired = [capture for capture in self._pending
                       if capture.epoch.session_id == session_id and capture.ticket is None and not capture.raw_done]
            for capture in retired:
                self._pending.remove(capture)
        for capture in retired:
            if capture.host_event is not None:
                capture.host_event.synchronize()
                capture.host_event = None
            self._tiers.retire(capture.token)
            capture.slot, capture.host = None, None
            self._event("capture_retired", capture.epoch, representations=list(capture.representations))
        return len(retired)

    def session_branches(self, session_id: str | None) -> list[str]:
        """The session's lineages in the store: its base branch and its ``~rN`` forks."""
        base = lineage_branch(session_id)
        return sorted(branch for branch in self.runtime.list_lineages()
                      if branch == base or re.fullmatch(re.escape(base) + r"~r\d+", branch))

    def forget_session(self, session_id: str | None) -> None:
        """Drop the in-memory lineage state of a session (after its lineages were deleted or replaced)."""
        self._sessions.pop(session_id or "default", None)

    def refresh_index(self, live_epochs: Iterable[ConditioningCacheEpoch] | None = None, *,
                      owners: Mapping[tuple, Iterable[str]] | None = None) -> int:
        """Re-index from the store after garbage collection (published records only; nothing may be pending);
        ``live_epochs``, ``owners``: as :meth:`reload_index` (quarantine marks are re-derived)."""
        with self._lock:
            if self._pending:
                raise ConditioningCacheError("refresh_index with captured epochs pending: drain first")
            self._index.clear()
        return self.reload_index(live_epochs, owners=owners)

    def close(self) -> None:
        """Drop the pinned slots; captured epochs must be flushed first (never discarded silently)."""
        with self._lock:
            pending = len(self._pending)
        if pending:
            raise ConditioningCacheError(f"{pending} captured epochs were never archived; call flush() first")
        self._tiers.close()


def node_bytes(runtime, node_ref: str) -> dict:
    """Stored bytes and decisions of one segment node (its records live in its epoch's single range).

    ``payload_bytes``: packet payloads (HEVC stream, affines). ``basis_block_bytes``: every FP16 basis row block its
    records reference (shared ancestors included). ``new_basis_bytes``: rows added by this node's own versions
    (fresh/refit roots and augment extensions; shared prefixes of earlier segments excluded).
    """
    storage = dict(runtime.node_storage(node_ref))
    decisions: dict[str, int] = {}
    ranks: dict[str, list] = {"K": [], "V": []}
    energies: dict[str, list] = {"K": [], "V": []}
    new_basis = 0
    for piece in runtime.resolve_history(node_ref):
        for entry, record_ref in piece.entry.records:
            record = runtime.load_record(record_ref)
            decision = record.decision.get("decision")
            decisions[decision] = decisions.get(decision, 0) + 1
            key = entry if not isinstance(entry, tuple) else entry[0]
            ranks[key.target].append(int(record.rank_view or 0))
            if record.energy is not None:
                energies[key.target].append(float(record.energy.stored_factor_energy))
            if decision in NEW_BASIS_DECISIONS and record.basis_version_ref is not None:
                new_basis += runtime.load_basis_version(record.basis_version_ref).added_bytes
    storage.update(decisions=decisions, rank_sum={t: sum(v) for t, v in ranks.items()},
                   rank_max={t: max(v, default=0) for t, v in ranks.items()},
                   min_stored_energy={t: min(v, default=None) for t, v in energies.items()},
                   ranks=ranks, new_basis_bytes=new_basis,
                   total_bytes=storage["payload_bytes"] + storage["basis_block_bytes"])
    return storage


def node_matrix_rows(runtime, node_ref: str) -> dict[str, dict]:
    """Every matrix of one segment node (``"layer:target"`` -> decision, rank view, stored factor energy, energy
    target), from the records of its epoch's single range."""
    node = runtime.load_node(node_ref)
    records = list(node.matrix_records) or [pair for item in runtime.load_epoch(node.epoch_ref).ranges
                                            for pair in item.records]
    if not records:
        raise ConditioningCacheError(f"node {node_ref} publishes no matrix records")
    rows = {}
    for entry, record_ref in records:
        key = entry if not isinstance(entry, tuple) else entry[0]
        record = runtime.load_record(record_ref)
        rows[f"{key.layer}:{key.target}"] = {
            "decision": record.decision.get("decision"), "rank": None if record.rank_view is None
            else int(record.rank_view), "stored_energy": None if record.energy is None
            else float(record.energy.stored_factor_energy),
            "target": record.decision.get("target") if record.energy is None else float(record.energy.target)}
    return rows


def compare_node_rows(reference: Mapping[str, Mapping], other: Mapping[str, Mapping]) -> dict:
    """Per-matrix comparison of two plans of the same epoch (e.g. CPU- vs GPU-planned): the documented planner
    tolerance is identical decisions and rank views (Step 4: the CPU planner matches the GPU plan rank view for rank
    view); stored energies may differ in the last bits. Also: matrices below their energy target in ``other``."""
    if set(reference) != set(other):
        raise ConditioningCacheError("the two nodes cover different matrices")
    decisions = [key for key in reference if reference[key]["decision"] != other[key]["decision"]]
    ranks = [key for key in reference if reference[key]["rank"] != other[key]["rank"]]
    deltas = [abs(float(other[key]["stored_energy"]) - float(reference[key]["stored_energy"])) for key in reference
              if reference[key]["stored_energy"] is not None and other[key]["stored_energy"] is not None]
    below = [key for key, row in other.items()
             if row["stored_energy"] is None or row["target"] is None or row["stored_energy"] < row["target"]]
    return {"matrices": len(reference), "decision_differences": sorted(decisions), "rank_differences": sorted(ranks),
            "max_abs_stored_energy_delta": max(deltas, default=0.0), "below_target": sorted(below),
            "rank_sum": {"reference": sum(int(row["rank"] or 0) for row in reference.values()),
                         "other": sum(int(row["rank"] or 0) for row in other.values())},
            "within_planner_tolerance": not decisions and not ranks and not below}


# ---------------------------------------------------------------------------------------------------------------
# GPU stack (archive pipeline with the GPU fitter, packed NVDEC restorer)
# ---------------------------------------------------------------------------------------------------------------
def gpu_uuid(index: int = 0) -> str:
    return "GPU-" + str(torch.cuda.get_device_properties(index).uuid).removeprefix("GPU-")


def default_decode_workers(index: int = 0) -> int:
    """Decoder-pool workers per GPU (Step 5a: 8 on RTX PRO 6000 with 4 NVDEC engines, 10 on A100 with 5)."""
    name = torch.cuda.get_device_properties(index).name
    return 10 if "A100" in name else 8


#: Cold-session staging of the packed restore (CR-17-4): one Step 5c :class:`reelcache.restore_staging.StagingEngine`
#: per worker on its GPU stages every history whose packs are not prepared (cold restores and session-level
#: preparation): parallel reads and hash checks into a bounded pinned ring, asynchronous basis uploads; restored caches
#: are bitwise those of the store path. The SGLang bridge's configuration (Step 5c's recommendation): threads None =
#: min(8, max(2, cpus // 2)) of the worker's CPU affinity, prepare threads None = max(1, threads // 2), a 512 MiB ring.
STAGING_DEFAULTS = {"staging_engine": True, "staging_threads": None, "staging_prepare_threads": None,
                    "staging_pinned_bytes": 512 << 20}


def staging_thread_counts(threads: int | None = None, prepare_threads: int | None = None) -> tuple[int, int]:
    """The staging engine's (threads, prepare threads): explicit values, else the defaults of ``STAGING_DEFAULTS``
    from this process's CPU affinity (bounded: at most 8 + 4 per pool pair)."""
    import os

    for value, name in ((threads, "staging_threads"), (prepare_threads, "staging_prepare_threads")):
        if value is not None and (type(value) is not int or not 1 <= value <= 64):
            raise ConditioningCacheError(f"{name} must be None or an integer in [1, 64]")
    cpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    threads = min(8, max(2, cpus // 2)) if threads is None else threads
    return threads, (max(1, threads // 2) if prepare_threads is None else prepare_threads)


def lineage_restorer(runtime, pool, device, *, arena_bytes: int = 4 * 1024**3, basis_budget_bytes: int = 2 * 1024**3,
                     prepared_budget_bytes: int = 4 * 1024**3, staging_engine: bool = True,
                     staging_threads: int | None = None, staging_prepare_threads: int | None = None,
                     staging_pinned_bytes: int = STAGING_DEFAULTS["staging_pinned_bytes"]):
    """The packed :class:`reelcache.archive_restore.LineageRestorer` of a LongCat worker and its staging engine.

    ``staging_engine`` (default on, CR-17-4): a :class:`reelcache.restore_staging.StagingEngine` on ``device`` stages
    cold histories (:data:`STAGING_DEFAULTS`); ``False`` restores cold histories through the store path. Returns
    ``(restorer, engine)``; the caller owns the engine (close it after the restorer).
    """
    from reelcache.archive_restore import LineageRestorer

    if type(staging_engine) is not bool:
        raise ConditioningCacheError("staging_engine must be a bool")
    if type(staging_pinned_bytes) is not int or staging_pinned_bytes < 1 << 20:
        raise ConditioningCacheError("staging_pinned_bytes must be an integer of at least 1 MiB")
    engine = None
    if staging_engine:
        from reelcache.restore_staging import StagingEngine

        threads, prepare_threads = staging_thread_counts(staging_threads, staging_prepare_threads)
        engine = StagingEngine(torch.device(device), threads=threads, prepare_threads=prepare_threads,
                               pinned_bytes=staging_pinned_bytes)
    try:
        restorer = LineageRestorer(runtime, pool, arena_bytes=arena_bytes, basis_budget_bytes=basis_budget_bytes,
                                   prepared_budget_bytes=prepared_budget_bytes, staging_engine=engine)
    except BaseException:
        if engine is not None:
            engine.close()
        raise
    return restorer, engine


def staging_description(engine, *, reason: str | None = None) -> dict:
    """What a worker reports about cold-session staging (startup and ``capabilities``)."""
    if engine is None:
        return {"enabled": False, "reason": reason or "staging_engine=False: cold restores use the store path"}
    return {"enabled": True, "device": str(engine.device), "threads": engine.threads,
            "prepare_threads": engine.prepare_threads, "pinned_bytes": engine.ring.capacity}


@dataclass
class GpuArchiveStack:
    """A LongCat archive on one GPU: runtime, archive pipeline (GPU fitter), decoder pool, lineage restorer and its
    cold-session staging engine (None when disabled)."""

    runtime: Any
    pipeline: Any
    restorer: Any
    pool: Any
    fitter: Any
    config: dict
    staging_engine: Any = None
    encoder: Any = None  # the NVENC encoder (freeze D), kept for the process lifetime; None with libx265

    def backend(self, **kwargs) -> ReelCacheConditioningCacheBackend:
        backend = ReelCacheConditioningCacheBackend(self.runtime, pipeline=self.pipeline, restorer=self.restorer,
                                                 **kwargs)
        backend.encoder_info = dict(self.config.get("encoder") or {})
        return backend

    def close(self) -> None:
        errors = []
        engine, encoder = self.staging_engine, self.encoder
        for closer in (lambda: self.pipeline.close(), lambda: encoder.close() if encoder is not None else None,
                       lambda: self.restorer.close(), lambda: engine.close() if engine is not None else None,
                       lambda: self.pool.close(), lambda: self.runtime.close()):
            try:
                closer()
            except BaseException as exc:  # noqa: BLE001 - close everything, report the first failure
                errors.append(exc)
        if errors:
            raise errors[0]


def gpu_archive_stack(store_root, model, *, device_index: int = 0, plan_device: str = "cuda",
                      decode_workers: int | None = None, encode_workers: int = 1, encode_threads: int = 16,
                      registry_name: str = "longcat_video", method=None, options=None,
                      arena_bytes: int = 4 * 1024**3, basis_budget_bytes: int = 2 * 1024**3,
                      prepared_budget_bytes: int = 4 * 1024**3, eigh_solver: str = "auto",
                      decoder_sessions_per_worker: int = LONGCAT_DECODER_SESSIONS,
                      plan_memory_bytes: int | None = None, verify_reads: bool = True,
                      audit_on_open: bool | None = False, on_open_error: str = "quarantine",
                      staging_engine: bool = True, staging_threads: int | None = None,
                      staging_prepare_threads: int | None = None,
                      staging_pinned_bytes: int = STAGING_DEFAULTS["staging_pinned_bytes"],
                      gated_planner: bool = True, encoder: str = "x265") -> GpuArchiveStack:
    """Build the archive/restore stack for one GPU.

    ``plan_device="cuda"`` plans on the GPU with :class:`reelcache.gpu_fitting.GPUFitter` (``gram_batched``, batched
    eigensolver; the Step 6 configuration); ``"cpu"`` plans on the host (an explicit choice where the planner's
    working set does not fit next to the served model). Restores always use the GPU (packed NVDEC +
    restore_kernels); published packs are prepared at seal time (``LineageRestorer.adopt``).
    ``decoder_sessions_per_worker`` (default 20) holds a session for every canonical node canvas of the LongCat
    conditioning-cache geometry (18 up to rank 4096), as ``prewarm_process`` creates them. ``plan_memory_bytes``
    bounds planning and finalization (``ArchiveOptions(plan_memory_bytes=...)``, Step 7: a LongCat node at an 8 GiB
    budget peaks at 1.9 GiB above the sources, with the node id of unbounded planning); a GPU planner next to the
    served model needs it (unbounded: ~35 GiB above the sources). ``verify_reads`` / ``audit_on_open`` /
    ``on_open_error``: the store's integrity settings (:func:`open_runtime`; default the CR-17-1 serving defaults:
    verify every object a restore reads, no hash audit of the whole store at open, quarantine damaged heads).
    ``staging_engine`` / ``staging_threads`` / ``staging_prepare_threads`` / ``staging_pinned_bytes``: the cold-session
    staging engine of the restorer on this GPU (:func:`lineage_restorer`; default on, CR-17-4). ``gated_planner``
    (default true; g7): the GPU planner is :class:`~.planning_gate.GatedGPUFitter`. Its matrix groups are units of
    the worker's :class:`~.planning_gate.PlanningGate`, and a node's host sources go to the device one group at a
    time (identical plans). False: :class:`reelcache.gpu_fitting.GPUFitter`, so sections wait for the whole node (g6).
    ``encoder`` (freeze D): ``x265`` (default, C3), ``nvenc`` or ``auto`` (:func:`select_encoder`: NVENC when it
    opens and qualifies on this GPU, else libx265 with the reason recorded; an explicit ``nvenc`` that cannot open
    raises). With NVENC the method's codec is the encoder's specification (qp 8, p1/hq, yuv444p), so node bytes and
    ids differ from libx265's; the encoder lives as long as the stack.
    """
    from reelcache.archive_pipeline import ArchivePipeline
    from reelcache.decoder_pool import DecoderPool
    from reelcache.lineage_commit import ArchiveOptions
    from reelcache.predictor_registry import RankPredictorRegistry

    if plan_device not in ("cuda", "cpu"):
        raise ConditioningCacheError("plan_device must be 'cuda' or 'cpu'")
    if encoder not in ENCODERS:
        raise ConditioningCacheError(f"encoder must be one of {ENCODERS}")
    if type(gated_planner) is not bool:
        raise ConditioningCacheError("gated_planner must be boolean")
    if type(decoder_sessions_per_worker) is not int or decoder_sessions_per_worker < 1:
        raise ConditioningCacheError("decoder_sessions_per_worker must be a positive integer")
    workers = default_decode_workers(device_index) if decode_workers is None else decode_workers
    uuid = gpu_uuid(device_index)
    runtime = open_runtime(store_root, model, verify_reads=verify_reads, audit_on_open=audit_on_open,
                           on_open_error=on_open_error)
    pool = DecoderPool(gpu_id=device_index, gpu_uuid=uuid, workers=workers,
                       max_sessions_per_worker=decoder_sessions_per_worker)
    restorer, engine = lineage_restorer(runtime, pool, torch.device("cuda", device_index), arena_bytes=arena_bytes,
                                        basis_budget_bytes=basis_budget_bytes,
                                        prepared_budget_bytes=prepared_budget_bytes, staging_engine=staging_engine,
                                        staging_threads=staging_threads,
                                        staging_prepare_threads=staging_prepare_threads,
                                        staging_pinned_bytes=staging_pinned_bytes)
    method = default_method() if method is None else method
    nvenc, encoder_info = select_encoder(encoder, device_index)
    if nvenc is not None:
        import dataclasses

        codec = method.codec
        method = dataclasses.replace(method, codec=nvenc.spec(codec.rank, pix_fmt=codec.pix_fmt,
                                                              normalization=codec.normalization,
                                                              protected_fraction=codec.protected_fraction))
        encoder_info["spec"] = method.codec.to_dict() if hasattr(method.codec, "to_dict") else str(method.codec)
    encoder_info["method_fingerprint"] = method.fingerprint
    if options is not None and plan_memory_bytes is not None:
        raise ConditioningCacheError("pass plan_memory_bytes inside options, not both")
    options = (ArchiveOptions(encode_threads=encode_threads, plan_memory_bytes=plan_memory_bytes)
               if options is None else options)
    fitter = None
    device = "cpu"
    if plan_device == "cuda":
        if gated_planner:
            from .planning_gate import GatedGPUFitter as fitter_class
        else:
            from reelcache.gpu_fitting import GPUFitter as fitter_class
        fitter = fitter_class(gpu_id=device_index, gpu_uuid=uuid, method="gram_batched", eigh_solver=eigh_solver)
        device = f"cuda:{device_index}"
    pipeline = ArchivePipeline(runtime, method=method, registry=RankPredictorRegistry(registry_name), options=options,
                               device=device, fitter=fitter, encode_workers=encode_workers, max_source_chunks=2,
                               max_staged_canvases=2, max_unpublished=4, restorer=restorer,
                               **({"encoder": nvenc} if nvenc is not None else {}))
    config = {"store_root": str(store_root), "device_index": device_index, "gpu_uuid": uuid,
              "gpu_name": torch.cuda.get_device_properties(device_index).name, "plan_device": device,
              "decode_workers": workers, "encode_workers": encode_workers, "encode_threads": encode_threads,
              "method": method.to_dict(), "method_fingerprint": method.fingerprint, "options": options.to_dict(),
              "arena_bytes": arena_bytes, "basis_budget_bytes": basis_budget_bytes,
              "prepared_budget_bytes": prepared_budget_bytes, "eigh_solver": eigh_solver,
              "decoder_sessions_per_worker": decoder_sessions_per_worker,
              "plan_memory_bytes": options.plan_memory_bytes, "integrity": integrity_settings(runtime),
              "staging_engine": staging_description(engine), "encoder": encoder_info,
              "planning_gate": ({"granularity": "group", "lazy_sources": True}
                                if getattr(fitter, "gate", None) is not None else
                                {"granularity": "node" if fitter is not None else None, "lazy_sources": False})}
    return GpuArchiveStack(runtime, pipeline, restorer, pool, fitter, config, staging_engine=engine, encoder=nvenc)


def _rss_bytes() -> int | None:
    """This process's resident set (``/proc/self/statm``), or None."""
    try:
        with open("/proc/self/statm") as handle:
            return int(handle.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError, IndexError):
        return None


def select_encoder(requested: str, device_index: int | None, *, factory=None) -> tuple:
    """Freeze D ``encoder``: ``(NVENCEncoder or None, record)``, probed once at attach (review amendment 5).

    ``x265``: libx265 (no probe). ``nvenc``: an NVENC encoder that opens and passes its qualification, else
    :class:`ConditioningCacheError` (no silent fallback). ``auto``: NVENC when it opens and qualifies, else libx265
    with the reason recorded. The record holds ``requested``, ``selected``, ``reason``, ``device``, ``driver``,
    ``version`` and the device memory the probe took (``probe_device_bytes``: sessions and surfaces outside PyTorch,
    counted against the GPU raw budget). ``factory(gpu_id)`` replaces :class:`reelcache.nvenc.NVENCEncoder` (tests)."""
    from reelcache.errors import UnsupportedError

    if requested not in ENCODERS:
        raise ConditioningCacheError(f"encoder must be one of {ENCODERS}")
    record = {"requested": requested, "selected": "x265", "reason": None, "device": None, "driver": None,
              "version": None, "probe_device_bytes": 0, "probe_host_bytes": 0}
    if requested == "x265":
        record["reason"] = "x265 requested"
        return None, record
    if device_index is None or (factory is None and not torch.cuda.is_available()):
        if requested == "nvenc":
            raise ConditioningCacheError("encoder 'nvenc' needs a CUDA device")
        record["reason"] = "auto: no CUDA device"
        return None, record
    if factory is None:
        from reelcache.nvenc import NVENCEncoder

        def factory(gpu_id):
            return NVENCEncoder(gpu_id=gpu_id, pix_fmts=("yuv444p",))
    before = None
    if torch.cuda.is_available():
        torch.cuda.synchronize(device_index)
        before = torch.cuda.mem_get_info(device_index)[0]
    rss_before = _rss_bytes()
    try:
        encoder = factory(device_index)
    except UnsupportedError as exc:
        if requested == "nvenc":
            raise ConditioningCacheError(f"encoder 'nvenc' requested but NVENC is unavailable: {exc}") from exc
        record["reason"] = f"auto: NVENC unavailable ({str(exc)[:300]}); libx265"
        return None, record
    if before is not None:
        torch.cuda.synchronize(device_index)
        record["probe_device_bytes"] = max(0, before - torch.cuda.mem_get_info(device_index)[0])
    rss_after = _rss_bytes()
    if rss_before is not None and rss_after is not None:
        record["probe_host_bytes"] = max(0, rss_after - rss_before)
    implementation = dict(getattr(encoder, "implementation", {}) or {})
    record.update(selected="nvenc", reason=f"{requested}: NVENC opened and qualified on GPU {device_index}",
                  device=implementation.get("gpu"), driver=implementation.get("driver_version"),
                  version=getattr(encoder, "version", None),
                  qualification=getattr(encoder, "qualification", None),
                  encoder_record=encoder.encoder_record("yuv444p") if hasattr(encoder, "encoder_record") else None)
    return encoder, record
