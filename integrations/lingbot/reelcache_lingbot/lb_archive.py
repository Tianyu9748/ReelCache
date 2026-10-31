"""ReelCache archive for LingBot chunks: the runtime's ``LineageSessionArchive`` driven without SGLang.

The Wan reelcache arm's resolved bridge config is the template (``bridge_config_template.json`` beside this module: the
LongLive 1.0 cal7 F8 reelcache_gf run 4464330; tree 60c5e24b), with the ablation's representation fields per arm. Two
overrides:
- the archive's own raw tier is off: the evaluation runner's arms (not part of this repository) keep their host raw copies themselves, inside B;
- the budgets are lbq's.

* Commit: one node per chunk on a lineage branch. A head continuation commits on the session's current branch (the
  pipeline publishes in order). A rewind first waits until the target is published (``publish_wait``), then forks a
  new branch from it. Sources: the engine's GPU-resident copies with a ready event and a release callback (harness
  default since the archival profile 4507952: finalize and fingerprints then run on the decision device), or host
  tensors (the pipeline then finalizes on the CPU).
* Restore: ``restorer.restore(head_ref, destination, rope, ranges, native_rope=...)`` writes the attended chunks of
  the target's path straight into LingBot's cache rows (BTHD ``NativeView``). K is re-roped by LingBot's own
  ``causal_rope_apply`` at the chunk's absolute frame (``native_rope``); V is written as is.
* ``commit_report`` gives one commits.jsonl row per node (the format the frozen compression tool reads).

:class:`FakeArchive` is the CPU stand-in (tests): it stores the matrices with a deterministic perturbation (lossy)
and reports sizes from a fixed ratio.
"""

from __future__ import annotations

import collections
import contextlib
import json
import math
import os
from pathlib import Path
import threading
import time
from typing import Any, Callable, Mapping, Sequence

TEMPLATE = Path(os.environ.get("REELCACHE_LINGBOT_TEMPLATE", str(Path(__file__).resolve().parent / "bridge_config_template.json")))  # the resolved Wan reelcache-arm bridge config (eval run 4464330); archive_config() overrides store_root, budgets and tiers
#: the ablation ladder (w14n_wave_v8 arms.NOSSD_ARMS / v11): representation fields per arm
REPRESENTATIONS: dict[str, dict[str, Any]] = {
    "reelcache": {"representation": "svd_hevc", "rank_policy": "adaptive", "reuse": True},
    "abl_a": {"representation": "svd", "reuse": False},
    "abl_b": {"representation": "hevc", "reuse": False},
    "abl_c": {"representation": "svd_hevc", "rank_policy": "fixed_fraction", "reuse": False},
    "abl_d": {"representation": "svd_hevc", "rank_policy": "adaptive", "reuse": False},
    "abl_e": {"representation": "svd_hevc", "rank_policy": "adaptive", "reuse": True},
}


LBQ_PLAN_MEMORY_BYTES = 4 << 30
#: accel L (2026-10-30): harness options carried in the arm's ``extra`` (popped before the ReelCache config is built):
#: ``lb_overlap_restore`` (bool, default False: the synchronous restore), ``lb_overlap_release_behind`` (int >= 0 or
#: None, default 2: the overlapped restore releases each layer's reconstruction temporaries once the layer d + 1
#: later was issued; ``LineageRestorer.restore(release_behind=...)`` of tree L), ``lb_restore_group_chunks`` (int >= 0,
#: default 0: lb_wave_v1g's grouped synchronous restore, g chunks per call)
#: ``lb_native_rope`` ("model", default: the model's ``rope`` = LingBot's causal_rope_apply; "lean": the model's
#: ``rope_lean``, the same kernels without causal_rope_apply's two float64 copies, bitwise equal)
#: ``lb_overlap_stream_priority`` (None, default: a default-priority restore stream; an int, e.g. -1: the overlapped
#: restore's CUDA stream gets that priority, above the generation's default stream: its kernels are dispatched first
#: when SMs free up; numerics unchanged)
HARNESS_EXTRA = {"lb_overlap_restore": False, "lb_overlap_release_behind": 2, "lb_restore_group_chunks": 0,
                 "lb_native_rope": "model", "lb_overlap_stream_priority": None}


def split_extra(extra: Mapping[str, Any] | None) -> tuple[dict[str, Any], dict[str, Any]]:
    """``(harness options with defaults, ReelCache config overrides)`` of an arm's ``extra``."""
    extra = dict(extra or {})
    options = {key: extra.pop(key, default) for key, default in HARNESS_EXTRA.items()}
    if type(options["lb_overlap_restore"]) is not bool:
        raise ArchiveError("lb_overlap_restore must be a bool")
    behind = options["lb_overlap_release_behind"]
    if behind is not None and (type(behind) is not int or behind < 0):
        raise ArchiveError("lb_overlap_release_behind must be None or a nonnegative int")
    if type(options["lb_restore_group_chunks"]) is not int or options["lb_restore_group_chunks"] < 0:
        raise ArchiveError("lb_restore_group_chunks must be a nonnegative int")
    if options["lb_native_rope"] not in ("model", "lean"):
        raise ArchiveError("lb_native_rope must be 'model' or 'lean'")
    priority = options["lb_overlap_stream_priority"]
    if priority is not None and type(priority) is not int:
        raise ArchiveError("lb_overlap_stream_priority must be None or an int")
    return options, extra
#: the decode guard's wait for the fitter's acknowledgement of a pause (the frozen PAUSE_ACK_TIMEOUT is 0.1 s; an
#: unacknowledged pause still holds at the fitter's next step boundary; recorded)
GUARD_ACK_TIMEOUT_S = 0.5


class ArchiveError(RuntimeError):
    pass


@contextlib.contextmanager
def decode_guard(pipeline: Any, need_bytes: int, room: Callable[[], int], *,
                 trim: Callable[[int], Mapping[str, Any]] | None = None,
                 ack_timeout: float = GUARD_ACK_TIMEOUT_S):
    """A VAE decode next to background archival on one device (720P SP=2: rank 0's decode is the device's memory
    peak). The frozen decode entry guard's ladder (``LineageSessionArchive.memory_critical``, dev22j6-8) for an
    archive without a raw tier, through the pipeline's public sections; codec and fitting unchanged:

    1. the fitter pauses at its next step boundary (``pause_gpu_work``): it allocates nothing more and no queued node
       starts while the decode runs;
    2. ``room() >= need_bytes``: the decode runs inside the pause (``action`` "pause");
    3. else the registered device caches are trimmed (decoder pool, restorer bases: rebuilt on demand; then the
       allocator's unoccupied blocks are released) and the room re-checked ("trim");
    4. else the pause is left and the decode waits until the node on the GPU finishes
       (``defer_gpu_work(wait=True, release_cached_memory=True)``; "wait": an exposed wait, recorded).

    ``room()``: device bytes the decode may still allocate; ``need_bytes``: its working set plus a margin. Yields the
    record (actions, rooms, seconds)."""
    rec: dict[str, Any] = {"need_bytes": int(need_bytes), "action": None}
    started = time.perf_counter()
    with contextlib.ExitStack() as paused:
        info = paused.enter_context(pipeline.pause_gpu_work(ack_timeout=ack_timeout))
        rec["pause"] = {k: (info or {}).get(k) for k in ("acked", "ack_seconds", "inflight", "landed")}
        rec["room_paused_bytes"] = int(room())
        if rec["room_paused_bytes"] >= need_bytes:
            rec["action"] = "pause"
        elif trim is not None:
            rec["trim"] = dict(trim(int(need_bytes)) or {})
            rec["room_trimmed_bytes"] = int(room())
            if rec["room_trimmed_bytes"] >= need_bytes:
                rec["action"] = "trim"
        if rec["action"] is not None:
            rec["guard_seconds"] = time.perf_counter() - started
            yield rec
            return
    # a paused fitter would never finish its node: the pause is left before waiting for it
    waited = time.perf_counter()
    with pipeline.defer_gpu_work(wait=True, release_cached_memory=True):
        rec.update(action="wait", wait_seconds=time.perf_counter() - waited, room_waited_bytes=int(room()),
                   guard_seconds=time.perf_counter() - started)
        yield rec


def archive_config(arm: str, *, store_root: str, prepared_budget_bytes: int, device_type: str,
                   extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    cfg = json.loads(TEMPLATE.read_text())
    cfg.update(REPRESENTATIONS[arm])
    cfg.update(store_root=store_root, raw_tier=False, raw_tier_retain="drop", host_raw_budget_bytes=0,
               gpu_raw_budget_bytes=0, prepared_budget_bytes=int(prepared_budget_bytes), store_io="buffered",
               defer_archival_during_generation=False, defer_archival_during_decode=False)
    if device_type != "cuda":  # CPU tests: software decode, CPU fitting, libx265
        cfg.update(decoder="software", fit_device="cpu", staging_engine=False, seal_prepare=False, encoder="x265")
    else:
        # bounded archival GPU working set (the LongLive 2.0 A100-80GB precedent, cr17_3): planned and finalized in
        # groups of <= 4 GiB (identical plans and bytes), one captured chunk's sources at a time. Unbounded, a LingBot
        # chunk's fit took ~18 GiB (G4 peak 88.2 GiB allocated) and NVENC session creation segfaulted once the
        # full-history cache left too little free memory (smoke 4506367)
        cfg.update(plan_memory_bytes=LBQ_PLAN_MEMORY_BYTES, max_source_chunks=1)
    cfg.update(dict(extra or {}))
    return cfg


class ReelCacheArchive:
    """One ``LineageSessionArchive`` per process (one arm per process: lc-ablation's rule after 4496099)."""

    kind = "reelcache"

    def __init__(self, root: str, *, arm: str, targets: tuple[float, float], provisional_targets: bool, device: Any,
                 layers: int, channels: int, token_grid: tuple[int, int, int], prepared_budget_bytes: int,
                 rope: Callable[[Any, int], Any], extra: Mapping[str, Any] | None = None, channel_tiles: int = 1,
                 head_dim: int = 128):
        import torch
        from reelcache.integrations.sglang.config import ReelCacheRuntimeConfig
        from reelcache.integrations.sglang.lineage_archive import LineageSessionArchive
        from reelcache.restore_kernels import NativeView, RopePositions, RopeTable
        from reelcache.types import EnergyTargets, MatrixKey, ModelIdentity, TensorContext

        self.torch = torch
        self.NativeView, self.RopePositions, self.MatrixKey, self.TensorContext = (NativeView, RopePositions, MatrixKey,
                                                                                  TensorContext)
        self.device = torch.device(device)
        self.options, extra = split_extra(extra)
        self.config_dict = archive_config(arm, store_root=root, prepared_budget_bytes=prepared_budget_bytes,
                                          device_type=self.device.type, extra=extra)
        self.config = ReelCacheRuntimeConfig.from_dict(self.config_dict)
        if self.options["lb_overlap_restore"] and getattr(self.config, "pipelined_restore", False):
            raise ArchiveError("lb_overlap_restore needs layer-ordered restores: pipelined_restore must be off")
        self.layers, self.channels, self.grid = int(layers), int(channels), tuple(token_grid)
        self.tokens = self.grid[0] * self.grid[1] * self.grid[2]
        # coordinator 2026-10-22 17:03, ablation (b) at 480P (stated deviation): the channel dimension tiled into
        # ``channel_tiles`` blocks of whole heads, each committed as its own matrix (MatrixKey layer = layer * tiles +
        # tile) of channels / tiles columns, so the identity basis has <= tokens columns; restores write each block
        # into its head slice of the native cache
        self.tiles = int(channel_tiles)
        if self.tiles < 1 or self.channels % (self.tiles * head_dim):
            raise ArchiveError(f"channel_tiles {self.tiles} must split {self.channels} channels into whole heads")
        self.tile_channels = self.channels // self.tiles
        self.tile_heads = self.tile_channels // head_dim
        self.model = ModelIdentity("lingbot-world-fast" if self.tiles == 1 else f"lingbot-world-fast-tiles{self.tiles}",
                                   "robbyant/lingbot-world-fast@b7a1db2d+a43bec7", self.layers * self.tiles,
                                   self.tile_channels, "tokens_channels", "bfloat16")
        self.targets = EnergyTargets(f"lbq-K{round(targets[0] * 100)}V{round(targets[1] * 100)}", float(targets[0]),
                                     float(targets[1]), bool(provisional_targets))
        self.archive = LineageSessionArchive(root, self.model, config=self.config, targets=self.targets,
                                             device=self.device)
        self.rope_fn = rope
        pairs = head_dim // 2
        # a separable placeholder table (rows = start frame); the native callable does the actual rotation
        sections = (pairs - 2 * (pairs // 3), pairs // 3, pairs // 3)
        self.table = RopeTable(torch.ones(4096, pairs, dtype=torch.float64), torch.zeros(4096, pairs,
                                                                                       dtype=torch.float64),
                               sections=sections)
        self.branch_of: dict[str, str] = {}       # session -> current lineage branch
        self.branch_tip: dict[str, Any] = {}      # branch -> node key of its last submitted commit
        self.ref: dict[Any, str] = {}             # node key -> published node ref
        self.tickets: dict[Any, Any] = {}         # node key -> ticket (until reported)
        self.reports: dict[Any, dict] = {}
        self.root_ref: dict[str, str] = {}
        self.branches = 0
        self.lock = threading.Lock()

    # -- lineage bookkeeping -------------------------------------------------------------------------------------
    def describe(self) -> dict[str, Any]:
        out = self.archive.describe()
        out["config"] = {k: self.config_dict[k] for k in ("representation", "rank_policy", "reuse", "decoder",
                                                          "fit_device", "hevc_qp", "hevc_preset", "prepared_budget_bytes",
                                                          "encoder", "nvdec_workers", "max_unpublished") if k in
                         self.config_dict}
        out["targets"] = {"profile": self.targets.profile, "K": self.targets.key, "V": self.targets.value,
                          "provisional": self.targets.provisional}
        out["harness_restore"] = dict(self.options)
        return out

    def _new_branch(self, session: str) -> str:
        self.branches += 1
        return f"{session}.b{self.branches:03d}"

    def ensure_session(self, session: str) -> None:
        if session in self.branch_of:
            return
        branch = self._new_branch(session)
        self.root_ref[session] = self.archive.create_session(branch, record={"session": session}, streams=())
        self.branch_of[session] = branch

    def commit(self, session: str, key: Any, parent_key: Any, chunk_index: int, k_pre: Sequence[Any],
               v: Sequence[Any], latents: Any, *, ready_event: Any = None,
               release: Callable[[Any], None] | None = None) -> dict[str, Any]:
        """Submit node ``key`` (child of ``parent_key``; None = the session root). Returns publish-wait details.

        ``k_pre`` / ``v``: host tensors (the pipeline finalizes them on the CPU) or GPU-resident sources owned by the
        caller until the pipeline calls ``release(event)`` (after finalize; the caller's next writer waits on
        ``event``); ``ready_event``: recorded after the sources were written (the fitter's stream waits on it)."""
        self.ensure_session(session)
        branch = self.branch_of[session]
        waited = 0.0
        if parent_key is not None and self.branch_tip.get(branch) != parent_key:
            t0 = time.perf_counter()
            parent_ref = self.wait_published(parent_key)
            waited = time.perf_counter() - t0
            branch = self._new_branch(session)
            self.archive.fork(parent_ref, branch)
            self.branch_of[session] = branch
        elif parent_key is None and self.branch_tip.get(branch) is not None:
            branch = self._new_branch(session)
            self.archive.create_session(branch, record={"session": session}, streams=())
            self.branch_of[session] = branch
        matrices, contexts = {}, {}
        start = chunk_index * self.tokens
        width = self.tile_channels
        for layer in range(self.layers):
            for target, tensor in (("K", k_pre[layer]), ("V", v[layer])):
                for t in range(self.tiles):
                    mk = self.MatrixKey(layer * self.tiles + t, target)
                    matrices[mk] = tensor if self.tiles == 1 else tensor[:, t * width:(t + 1) * width]
                    contexts[mk] = self.TensorContext(self.model, mk, start, self.grid[0], self.grid[1], self.grid[2],
                                                      "pre_rope" if target == "K" else "none")
        ticket = self.archive.commit_chunk(branch, node_id=f"{key[0]}:g{key[1]:03d}", parent_ref="", root_ref="",
                                           matrices=matrices, contexts=contexts, streams=(),
                                           record={"session": session, "generation": key[1], "chunk": chunk_index},
                                           execution_state={"latents": latents.detach().to("cpu").float()},
                                           ready_event=ready_event, release=release)
        with self.lock:
            self.tickets[key] = ticket
            self.branch_tip[branch] = key
        return {"branch": branch, "publish_wait_s": waited, "forked": waited > 0 or parent_key is None}

    def published(self, key: Any) -> bool:
        if key in self.ref:
            return True
        ticket = self.tickets.get(key)
        if ticket is not None and ticket.done():
            self.ref[key] = ticket.result(0)
            return True
        return False

    def wait_published(self, key: Any, timeout: float | None = None) -> str:
        if key in self.ref:
            return self.ref[key]
        ticket = self.tickets.get(key)
        if ticket is None:
            raise ArchiveError(f"node {key} was never committed")
        ref = ticket.result(timeout)
        self.ref[key] = ref
        return ref

    def drain(self) -> dict[str, Any]:
        started = time.perf_counter()
        for branch in list(self.branch_tip):
            self.archive.drain(branch)
        for key in list(self.tickets):
            self.wait_published(key)
        return {"seconds": time.perf_counter() - started, "published": len(self.ref)}

    def inflight(self) -> int:
        return sum(1 for key in self.tickets if not self.published(key))

    def report(self, key: Any) -> dict[str, Any]:
        """The commits.jsonl entry of a published node (Wan format; per-matrix rows summarized)."""
        if key in self.reports:
            return self.reports[key]
        ticket = self.tickets[key]
        self.wait_published(key)
        rep = self.archive.commit_report(ticket)
        rows = rep.pop("matrices", [])
        energy = {t: {"count": 0, "min": None, "sum": 0.0, "below_target": 0} for t in ("K", "V")}
        for row in rows:
            t = row["key"].split(":")[1]
            e = row.get("stored_factor_energy")
            if e is None:
                continue
            item = energy[t]
            item["count"] += 1
            item["sum"] += e
            item["min"] = e if item["min"] is None else min(item["min"], e)
            item["below_target"] += int(e < (row.get("target") or 0))
        rep.update(chunk=None if key is None else key[1], published=True, stored_energy=energy,
                   ranks={t: sorted({row["rank_view"] for row in rows if row["key"].endswith(t)}) for t in ("K", "V")},
                   matrices=len(rows))
        for k in ("method", "options", "registry"):
            rep.pop(k, None)
        self.reports[key] = rep
        return rep

    # -- restore ---------------------------------------------------------------------------------------------------
    def restore(self, head_key: Any, chunks: Sequence[tuple[int, int, Any]], destination_tensors: Callable[[int], tuple],
                native_dtype: Any, *, overlap: bool | None = None, release_behind: int | None = None,
                group_chunks: int | None = None, anchor: Any = None, native_rope_impl: str | None = None,
                stream_priority: int | None = None) -> dict[str, Any]:
        """Restore the attended chunks ``[(chunk_index, slot_start, node key)]`` of ``head_key``'s path (the keys are
        the path's nodes at those chunk indices) into the cache rows.

        accel L (2026-10-30; None = this archive's ``extra`` options, see :data:`HARNESS_EXTRA`):
        - ``overlap``: start the restore layer-overlapped with the next forward and return at once (``reelcache_lingbot.lb_overlap``;
          ``destination_tensors`` must be the bound method of a model with ``attach_overlapped_restore``, which
          joins it); the returned record is filled when the model joins the restore. ``anchor``: a CUDA timing event
          on the generation stream (restore window and per-layer stalls are reported relative to it).
        - ``release_behind``: ``LineageRestorer.restore(release_behind=...)`` (tree L): release each layer's
          reconstruction temporaries once the layer ``d + 1`` later was issued (default: the overlapped restore uses
          ``lb_overlap_release_behind``; the synchronous one holds them to the end, as lb_wave_v1).
        - ``group_chunks``: lb_wave_v1g's grouped synchronous restore (calls of at most g chunks, each synchronized).
        - ``native_rope_impl``: "model" (the model's ``rope``) or "lean" (its ``rope_lean``); None: ``lb_native_rope``.
        - ``stream_priority``: the overlapped restore's CUDA stream priority (None: ``lb_overlap_stream_priority``).
        Every mode writes the same rows (bitwise)."""
        torch = self.torch
        head_ref = self.wait_published(head_key)
        slots = {ci * self.tokens: (ci, slot) for ci, slot, _ in chunks}
        impl = self.options["lb_native_rope"] if native_rope_impl is None else native_rope_impl
        if impl == "model":
            rope_fn = self.rope_fn
        elif impl == "lean" and callable(getattr(getattr(self.rope_fn, "__self__", None), "rope_lean", None)):
            rope_fn = self.rope_fn.__self__.rope_lean
        else:
            raise ArchiveError(f"native RoPE {impl!r} is not available (the model's rope / rope_lean)")

        def destination(node_id, entry, tile):
            key = entry if isinstance(entry, self.MatrixKey) else entry[0]
            ci, slot = slots[int(tile.token_start)]
            layer, t = divmod(key.layer, self.tiles)
            k_cache, v_cache = destination_tensors(layer)
            cache = k_cache if key.target == "K" else v_cache
            if self.tiles > 1:  # the tile's head slice (a strided BTHD view)
                cache = cache[:, :, t * self.tile_heads:(t + 1) * self.tile_heads]
            return self.NativeView(cache, "BTHD", token_offset=slot)

        def rope(node_id, entry, tile):
            key = entry if isinstance(entry, self.MatrixKey) else entry[0]
            if key.target != "K":
                return None
            ci, _ = slots[int(tile.token_start)]
            return self.RopePositions(self.table, (ci * self.grid[0], 0, 0))

        def native_rope(values, job):
            ci = int(job.rope.rows[0]) // self.grid[0]
            n = values.shape[0]
            return rope_fn(values.reshape(n, -1), ci)

        ranges = [(ci * self.tokens, (ci + 1) * self.tokens) for ci, _, _ in chunks]
        if self.options["lb_overlap_restore"] if overlap is None else bool(overlap):
            behind = self.options["lb_overlap_release_behind"] if release_behind is None else release_behind
            priority = self.options["lb_overlap_stream_priority"] if stream_priority is None else stream_priority
            return self._restore_overlapped(head_ref, destination, rope, native_rope, native_dtype,
                                            destination_tensors, ranges, release_behind=behind, anchor=anchor,
                                            priority=priority)
        group = self.options["lb_restore_group_chunks"] if group_chunks is None else int(group_chunks)
        extra = {} if release_behind is None else {"release_behind": int(release_behind)}
        started = time.perf_counter()
        if group and len(chunks) > group:
            return self._restore_grouped(head_ref, chunks, group, destination, rope, native_rope, native_dtype,
                                         started, extra)
        report = self.archive.restorer.restore(head_ref, destination, rope, ranges=ranges, native_rope=native_rope,
                                               native_rope_dtype=native_dtype, **extra)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        report.pop("handles", None)
        history = report.pop("history", [])
        out = {"seconds": time.perf_counter() - started, "nodes": report.get("nodes"),
               "entries": report.get("entries"), "history_entries": len(history),
               "breakdown": report.get("breakdown"), "layer_ahead": report.get("layer_ahead")}
        if extra:
            out["release_behind"] = report.get("release_behind")
        return out

    def _restore_grouped(self, head_ref, chunks, size, destination, rope, native_rope, native_dtype, started, extra):
        """lb_wave_v1g (Addendum 2): the window in calls of at most ``size`` chunks (chunk order), each synchronized
        before the next, so one call's reconstruction temporaries are released before the next call allocates its
        own. Same entries, destinations and values as one call."""
        torch = self.torch
        out = {"nodes": 0, "entries": 0, "history_entries": 0, "breakdown": {}, "layer_ahead": [],
               "groups": [], "group_chunks": size}
        for start in range(0, len(chunks), size):
            group = list(chunks[start:start + size])
            ranges = [(ci * self.tokens, (ci + 1) * self.tokens) for ci, _, _ in group]
            t0 = time.perf_counter()
            report = self.archive.restorer.restore(head_ref, destination, rope, ranges=ranges,
                                                   native_rope=native_rope, native_rope_dtype=native_dtype, **extra)
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            report.pop("handles", None)
            history = report.pop("history", [])
            out["nodes"] += int(report.get("nodes") or 0)
            out["entries"] += int(report.get("entries") or 0)
            out["history_entries"] += len(history)
            for k, v in (report.get("breakdown") or {}).items():
                if isinstance(v, (int, float)):
                    out["breakdown"][k] = out["breakdown"].get(k, 0.0) + v
            out["layer_ahead"].append(report.get("layer_ahead"))
            out["groups"].append({"chunks": [ci for ci, _, _ in group], "seconds": time.perf_counter() - t0})
            del report, history
        out["seconds"] = time.perf_counter() - started
        return out

    def _restore_overlapped(self, head_ref, destination, rope, native_rope, native_dtype, destination_tensors, ranges,
                            *, release_behind, anchor, priority=None):
        """accel L: the restore on its own thread and CUDA stream (``reelcache.overlapped_restore``), ordered after the
        caller's pending work on the generation stream (placements, raw installs, earlier forwards reading the rows);
        the model's first forward waits per layer. Unprepared (cold) histories are staged by the restore thread."""
        from reelcache.overlapped_restore import start_overlapped_restore

        from .lb_overlap import OverlapSession, StallEvents

        torch = self.torch
        model = getattr(destination_tensors, "__self__", None)
        attach = getattr(model, "attach_overlapped_restore", None)
        if attach is None:
            raise ArchiveError("lb_overlap_restore: destination_tensors must be a bound method of a model with "
                               "attach_overlapped_restore")
        started = time.perf_counter()
        device, prepared, stream = self.device, None, None
        if device.type == "cuda":
            if device.index is None:  # the restore thread sets its device: an explicit index
                device = torch.device("cuda", torch.cuda.current_device())
            prepared = torch.cuda.Event()
            prepared.record(torch.cuda.current_stream(device))
            if priority is not None:
                stream = torch.cuda.Stream(device=device, priority=int(priority))
        recorder = StallEvents(torch, device, anchor=anchor)
        options = {"native_rope": native_rope, "native_rope_dtype": native_dtype}
        if release_behind is not None:
            options["release_behind"] = int(release_behind)
        readiness = start_overlapped_restore(self.archive.restorer, head_ref, destination, rope,
                                             layers=self.layers * self.tiles, ranges=ranges, require_prepared=False,
                                             wait_event=prepared, device=device, telemetry=recorder, stream=stream,
                                             **options)
        record = {"overlapped": True, "seconds": None, "start_call_seconds": time.perf_counter() - started,
                  "release_behind_requested": release_behind, "chunks": len(ranges),
                  "stream_priority": None if stream is None else stream.priority}
        session = OverlapSession(readiness, layers=self.layers, tiles=self.tiles, record=record, recorder=recorder,
                                 started=started)
        try:
            attach(session)
        except BaseException:
            session.abort()
            raise
        return record

    def forget_session(self, session: str) -> dict[str, Any]:
        """Delete every lineage branch of ``session`` (the warm-up's throwaway tree) and reclaim its objects."""
        self.drain()
        branches = [b for b in self.branch_tip if b.split(".b")[0] == session]
        if session in self.branch_of:
            branches.append(self.branch_of[session])
        for branch in sorted(set(branches)):
            # the pipeline keeps a drained branch's planning state (and the runtime's pending mark) until release;
            # delete refuses it before (smoke 4506173: "cannot delete a lineage with a commit in flight")
            self.archive.release(branch)
            if self.archive.head(branch) is not None:
                self.archive.delete(branch)
            self.branch_tip.pop(branch, None)
        self.branch_of.pop(session, None)
        self.root_ref.pop(session, None)
        for key in [k for k in self.tickets if k[0] == session]:
            self.tickets.pop(key, None)
            self.ref.pop(key, None)
            self.reports.pop(key, None)
        collected = self.archive.collect_garbage()
        return {"branches_deleted": sorted(set(branches)), "collected": {k: v for k, v in collected.items()
                                                                         if isinstance(v, (int, float))},
                "archive_bytes_after": self.archive_bytes()}

    # -- the VAE decode next to background archival --------------------------------------------------------------
    def decode_section(self, need_bytes: int, room: Callable[[], int]):
        """:func:`decode_guard` on this archive's pipeline, trimming the device caches registered for its device."""
        from reelcache.device_pressure import trim_device_caches

        def trim(need: int) -> dict[str, Any]:
            out = trim_device_caches("decode", device=self.device, needed_bytes=need, headroom=room)
            return {"bytes": out.get("bytes"), "seconds": out.get("seconds"),
                    "trims": [{k: t.get(k) for k in ("name", "bytes", "error")} for t in out.get("trims") or []]}

        return decode_guard(self.archive.pipeline, need_bytes, room, trim=trim)

    # -- accounting -----------------------------------------------------------------------------------------------
    def archive_bytes(self) -> int:
        return int(self.archive.store.stats()["object_bytes"])

    def prepared_bytes(self) -> int:
        prepared = getattr(self.archive.restorer, "prepared", None)
        return 0 if prepared is None else int(prepared.stats()["bytes"])

    def stats(self) -> dict[str, Any]:
        try:
            return self.archive.stats()
        except Exception as exc:  # noqa: BLE001 - reporting only
            return {"error": repr(exc)}

    def close(self) -> None:
        try:
            self.drain()
        finally:
            self.archive.close() if hasattr(self.archive, "close") else None


class FakeArchive:
    """CPU stand-in with the archive's interface: asynchronous 'archival' (a worker thread, a fixed per-node delay),
    lossy storage (a deterministic relative perturbation), sizes = raw / ratio."""

    kind = "fake"

    def __init__(self, root: str, *, arm: str, ratio: float = 16.0, delay_s: float = 0.0, error: float = 1e-3,
                 layers: int, channels: int, token_grid: tuple, rope: Callable[[Any, int], Any], elem_bytes: int = 2,
                 **_):
        import torch

        self.torch = torch
        self.arm, self.ratio, self.delay_s, self.error = arm, float(ratio), float(delay_s), float(error)
        self.layers, self.channels, self.grid = layers, channels, tuple(token_grid)
        self.tokens = self.grid[0] * self.grid[1] * self.grid[2]
        self.elem_bytes = elem_bytes
        self.rope_fn = rope
        self.store: dict[Any, tuple] = {}
        self.pending: collections.deque = collections.deque()
        self.done: set = set()
        self.reports: dict[Any, dict] = {}
        self.cv = threading.Condition()
        self.closed = False
        self.releases = 0            # device-source contract: sources released after 'finalize'
        self.source_violations = 0   # held sources rewritten by the caller before their release
        self.thread = threading.Thread(target=self._loop, name="fake-archival", daemon=True)
        self.thread.start()

    def describe(self) -> dict[str, Any]:
        return {"backend": "fake", "ratio": self.ratio, "arm": self.arm}

    def ensure_session(self, session: str) -> None:
        pass

    def _loop(self) -> None:
        while True:
            with self.cv:
                while not self.pending and not self.closed:
                    self.cv.wait()
                if self.closed and not self.pending:
                    return
                key, k_pre, v, chunk_index, release, digest = self.pending[0]
            time.sleep(self.delay_s)
            if release is not None and self._digest(k_pre, v) != digest:  # the caller rewrote held sources
                self.source_violations += 1
            pert = lambda t, s: t.double() * (1.0 + self.error * math.sin(s))  # noqa: E731
            stored = ([pert(k, i) for i, k in enumerate(k_pre)], [pert(x, i + 7) for i, x in enumerate(v)], chunk_index)
            if release is not None:  # 'finalize' done: the caller's sources are released (no device: no event)
                del k_pre, v
                release(None)
                self.releases += 1
            native = 2 * self.layers * self.tokens * self.channels * self.elem_bytes
            with self.cv:
                self.store[key] = stored
                self.reports[key] = {"representation": self.arm, "kind": "chunk", "chunk": key[1],
                                     "native_bytes": native, "incremental_bytes": int(native / self.ratio),
                                     "compression_ratio": self.ratio, "published": True,
                                     "seconds_before_publication": self.delay_s, "decisions": None}
                self.pending.popleft()
                self.done.add(key)
                self.cv.notify_all()

    @staticmethod
    def _digest(k_pre, v) -> float:
        return float(sum(float(t.double().abs().sum()) + float(t.double().sum()) for t in list(k_pre) + list(v)))

    def commit(self, session, key, parent_key, chunk_index, k_pre, v, latents, *, ready_event=None,
               release=None) -> dict[str, Any]:
        """Without ``release`` the matrices are copied at once (host sources); with it they stay the caller's until
        the worker has read them and calls ``release(None)`` (the device-source contract)."""
        t0 = time.perf_counter()
        waited = 0.0
        if parent_key is not None and parent_key not in self.done:
            in_queue = any(item[0] == parent_key for item in self.pending)
            if in_queue and self.pending and self.pending[-1][0] != parent_key:
                self.wait_published(parent_key)
                waited = time.perf_counter() - t0
        if release is None:
            item = (key, [k.clone() for k in k_pre], [x.clone() for x in v], chunk_index, None, None)
        else:
            item = (key, list(k_pre), list(v), chunk_index, release, self._digest(k_pre, v))
        with self.cv:
            self.pending.append(item)
            self.cv.notify_all()
        return {"branch": session, "publish_wait_s": waited, "forked": False}

    def published(self, key) -> bool:
        return key in self.done

    def wait_published(self, key, timeout=None) -> str:
        with self.cv:
            while key not in self.done:
                if not any(item[0] == key for item in self.pending):
                    raise ArchiveError(f"node {key} was never committed")
                self.cv.wait(timeout=1.0)
        return f"fake:{key}"

    def drain(self) -> dict[str, Any]:
        started = time.perf_counter()
        with self.cv:
            while self.pending:
                self.cv.wait(timeout=1.0)
        return {"seconds": time.perf_counter() - started, "published": len(self.done)}

    def inflight(self) -> int:
        return len(self.pending)

    def report(self, key) -> dict[str, Any]:
        self.wait_published(key)
        return dict(self.reports[key])

    def restore(self, head_key, chunks, destination_tensors, native_dtype) -> dict[str, Any]:
        """``chunks`` = [(chunk_index, slot, node_key)] for the fake (it stores per node, not per lineage)."""
        started = time.perf_counter()
        for ci, slot, key in chunks:
            self.wait_published(key)
            k_pre, v, _ = self.store[key]
            for layer in range(self.layers):
                k_cache, v_cache = destination_tensors(layer)
                n = k_pre[layer].shape[0]
                heads = k_cache.shape[2]
                k_cache[0, slot:slot + n] = self.rope_fn(k_pre[layer], ci).reshape(n, heads, -1).to(k_cache.dtype)
                v_cache[0, slot:slot + n] = v[layer].reshape(n, heads, -1).to(v_cache.dtype)
        return {"seconds": time.perf_counter() - started, "nodes": len(chunks)}

    def forget_session(self, session: str) -> dict[str, Any]:
        self.drain()
        for key in [k for k in self.store if k[0] == session]:
            self.store.pop(key, None)
            self.reports.pop(key, None)
            self.done.discard(key)
        return {"archive_bytes_after": self.archive_bytes()}

    @contextlib.contextmanager
    def decode_section(self, need_bytes: int, room: Callable[[], int]):
        """No device and no GPU fitter: nothing to guard (recorded)."""
        yield {"need_bytes": int(need_bytes), "action": "none"}

    def archive_bytes(self) -> int:
        return sum(int(r["incremental_bytes"]) for r in self.reports.values())

    def prepared_bytes(self) -> int:
        return 0

    def stats(self) -> dict[str, Any]:
        return {"published": len(self.done), "pending": len(self.pending), "source_releases": self.releases,
                "source_violations": self.source_violations}

    def close(self) -> None:
        with self.cv:
            self.closed = True
            self.cv.notify_all()
        self.thread.join(timeout=10)
