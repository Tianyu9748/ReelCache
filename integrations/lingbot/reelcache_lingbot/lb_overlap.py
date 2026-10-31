"""LingBot history restore layer-overlapped with the next chunk's first DiT forward (accel track L, 2026-10-30;
plan.md section 12 A4). Harness flag ``lb_overlap_restore`` (default off: the synchronous restore of lb_wave_v1).

* Synchronous (base): ``ReelCacheArchive.restore`` -> ``LineageRestorer.restore`` writes every layer of every history
  chunk into the shared self-attention cache and synchronizes; then ``LingBotModel.generate`` runs the 4 denoising
  forwards and the clean pass.
* Overlapped: ``ReelCacheArchive.restore`` starts ``reelcache.overlapped_restore.start_overlapped_restore`` (the same
  restorer call on its own thread and CUDA stream, after the caller's pending work; layer order; one CUDA event per
  layer) and returns at once, handing an :class:`OverlapSession` to the model. During the FIRST forward, the
  self-attention of block ``l`` waits for layer ``l`` (a forward pre-hook on ``blocks[l].self_attn``: a host wait
  until the restore thread issued the layer, then a wait of the generation stream on the layer's event; no device
  synchronize). The hooks are removed after the last layer's wait: the other forwards read the installed KV in
  stream order. ``generate`` joins the restore (its integrity gate included) before x0 leaves it; on a failure it
  quiesces the restore (no restore work in flight) and re-raises.

The restored KV is the restorer's own output in both modes (class (i): bitwise; the tool checks it).
:class:`StallEvents` is the readiness's ``stall_recorder`` / telemetry: a CUDA event pair on the generation stream
around every layer wait (GPU idle time caused by the wait) and the restore stream's active window.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Mapping, Sequence


class StallEvents:
    """``LayerReadiness`` telemetry: per-layer event pairs on the consuming stream and the restore stream's window.

    After the restore joined and the device synchronized, :meth:`summary` gives the GPU stall per layer (ms, the
    consuming stream's idle time between the two events) and the restore window relative to ``anchor`` (an event
    recorded on the consuming stream before the restore started)."""

    def __init__(self, torch_mod: Any, device: Any, anchor: Any = None):
        self.torch = torch_mod
        self.device = device
        self.anchor = anchor
        self.cuda = getattr(device, "type", str(device)) == "cuda"
        self.pairs: list[tuple[int, Any, Any, bool]] = []
        self.windows: dict[str, list[Any]] = {}

    def _event(self, stream: Any = None):
        event = self.torch.cuda.Event(enable_timing=True)
        event.record(stream if stream is not None else self.torch.cuda.current_stream(self.device))
        return event

    def stall_begin(self, stream: Any, ready: bool = False):
        return (self._event(stream) if self.cuda else None, bool(ready), stream)

    def stall_end(self, stall: Any, layer: int) -> None:
        begin, ready, stream = stall
        self.pairs.append((int(layer), begin, self._event(stream) if self.cuda else None, ready))

    def window_begin(self, source: str, stream: Any):
        return (source, self._event(stream) if self.cuda else None, stream)

    def window_end(self, window: Any) -> None:
        source, begin, stream = window
        self.windows[source] = [begin, self._event(stream) if self.cuda else None]

    def summary(self) -> dict[str, Any]:
        """Call after the device synchronized (every event completed)."""
        out: dict[str, Any] = {"layers_waited": len(self.pairs),
                               "ready_at_wait": sum(1 for *_, ready in self.pairs if ready)}
        if not self.cuda:
            return out
        stalls = [0.0] * (max((layer for layer, *_ in self.pairs), default=-1) + 1)
        for layer, begin, end, _ in self.pairs:
            stalls[layer] += float(begin.elapsed_time(end))
        out["stall_ms"] = stalls
        out["stall_ms_total"] = sum(stalls)
        if self.anchor is not None:
            for source, (begin, end) in self.windows.items():
                if begin is not None and end is not None:
                    out[f"{source}_window_ms"] = [float(self.anchor.elapsed_time(begin)),
                                                  float(self.anchor.elapsed_time(end))]
        return out


class OverlapSession:
    """One pending overlapped restore of the shared cache: its readiness, the per-layer waits of the first forward
    and the live restore record (filled by :meth:`join`).

    ``layers``: the model's blocks; ``tiles``: readiness layers per block (channel tiles of ablation (b): readiness
    layer ``l * tiles + t``). ``record``: the dict ``ReelCacheArchive.restore`` returned (updated in place)."""

    def __init__(self, readiness: Any, *, layers: int, tiles: int = 1, record: dict | None = None,
                 recorder: StallEvents | None = None, started: float | None = None,
                 finish: Callable[[dict], Mapping[str, Any]] | None = None):
        self.readiness = readiness
        self.layers, self.tiles = int(layers), int(tiles)
        self.record = record if record is not None else {}
        self.recorder = recorder
        self.started = time.perf_counter() if started is None else started
        self.finish = finish
        self.hooks: list = []
        self.waited_layers = 0
        self.host_wait_s = 0.0
        self.first_wait_at: float | None = None
        self.last_wait_at: float | None = None
        self.state = "pending"  # pending -> joined | aborted

    # -- consumer side --------------------------------------------------------------------------------------------
    def attach(self, modules: Sequence[Any]) -> None:
        """Forward pre-hooks on ``modules[l]`` (the self-attention of block ``l``): wait for layer ``l``."""
        if len(modules) != self.layers:
            raise ValueError(f"{len(modules)} attention modules for a {self.layers}-layer restore")
        if self.hooks:
            raise RuntimeError("the overlapped restore's waits are already attached")
        self.hooks = [module.register_forward_pre_hook(self._hook(layer)) for layer, module in enumerate(modules)]

    def _hook(self, layer: int):
        def hook(_module, _args):
            self.wait(layer)
        return hook

    def wait(self, layer: int) -> None:
        """Before block ``layer`` reads the cache: host wait until issued, then the stream wait (typed failures)."""
        clock = time.perf_counter()
        if self.first_wait_at is None:
            self.first_wait_at = clock
        for tile in range(self.tiles):
            self.readiness.wait(layer * self.tiles + tile)
        now = time.perf_counter()
        self.host_wait_s += now - clock
        self.waited_layers += 1
        self.last_wait_at = now
        if layer == self.layers - 1:
            self.detach()  # every layer waited for: later forwards read the installed KV in stream order

    def detach(self) -> None:
        hooks, self.hooks = self.hooks, []
        for handle in hooks:
            handle.remove()

    # -- completion -----------------------------------------------------------------------------------------------
    def join(self) -> dict:
        """Join the restore (its failure re-raised, typed) and fill the record. Commit or emit nothing derived from
        the restored KV before this returned."""
        self.detach()
        if self.state != "pending":
            return self.record
        try:
            info = self.readiness.join()
        except BaseException:
            self.state = "aborted"
            self.readiness.quiesce()
            raise
        self.state = "joined"
        joined = time.perf_counter()
        timeline = self.readiness.timeline()
        self.record.update({
            "seconds": joined - self.started, "joined_seconds_from_call": joined - self.started,
            "nodes": info.get("nodes"), "entries": info.get("entries"),
            "history_entries": len(info.get("history") or []) if isinstance(info.get("history"), list) else None,
            "breakdown": info.get("breakdown"), "layer_ahead": info.get("layer_ahead"),
            "release_behind": info.get("release_behind"),
            "overlap": {"progressive": info.get("progressive"), "bound_entries": info.get("bound_entries"),
                        "thread_start_delay_seconds": info.get("thread_start_delay_seconds"),
                        "restore_source": info.get("restore_source"),
                        "first_layer_ready_seconds_from_call": info.get("first_layer_ready_seconds_from_call"),
                        "last_layer_ready_seconds_from_call": info.get("last_layer_ready_seconds_from_call"),
                        "restore_complete_seconds_from_call": info.get("restore_complete_seconds_from_call"),
                        "layers_waited": self.waited_layers, "host_wait_seconds": self.host_wait_s,
                        "first_wait_seconds_from_call": (None if self.first_wait_at is None
                                                         else self.first_wait_at - self.started),
                        "last_wait_seconds_from_call": (None if self.last_wait_at is None
                                                        else self.last_wait_at - self.started),
                        "readiness_waited_seconds": timeline.get("waited_seconds")}})
        if self.finish is not None:
            self.record.update(dict(self.finish(info) or {}))
        return self.record

    def abort(self) -> bool:
        """A failed or abandoned request: no hooks, no restore work in flight. True if the restore had completed."""
        self.detach()
        if self.state != "pending":
            return self.state == "joined"
        self.state = "aborted"
        ok = self.readiness.quiesce()
        self.record.update({"aborted": True, "restore_completed": ok})
        return ok

    def gpu_summary(self) -> dict[str, Any] | None:
        """The recorder's per-layer GPU stalls and restore window (after a device synchronize)."""
        return None if self.recorder is None else self.recorder.summary()
