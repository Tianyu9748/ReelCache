"""Device-memory pressure: a registry of trimmable device caches (freeze F, Step 7).

The raw tier's pressure response (:meth:`reelcache.raw_tier.RawChunkTier.pressure_check`) first demotes GPU raw copies.
When the device headroom is still below what the caller needs, it trims the caches other components registered here,
in a fixed order, then returns the freed segments to the driver (``torch.cuda.empty_cache``).

Contract of a trim callable ``trim(reason: str) -> int``:

* free only device memory the component can rebuild on demand (idle decoder canvases and sessions' surfaces, cached
  bases, unclaimed resident state, scratch workspaces): never memory an in-flight operation still uses;
* be thread-safe with the component's own users and return promptly (no wait on unrelated GPU work);
* return the device bytes released to the allocator (a best estimate; 0 when nothing was freed);
* ``reason`` is ``"decode"`` (the VAE-decode seam), ``"request_start"`` (before a request allocates its KV cache) or
  ``"oom"`` (after an out-of-memory error). Trims never run from background admission checks.

Exceptions are recorded and never fail the pressure response. Components register at construction and unregister
when they close (:meth:`TrimHandle.unregister`); the registry keeps weak references to bound methods, so a component
collected without unregistering drops out on its own.
"""

from __future__ import annotations

import itertools
import threading
import time
import weakref
from typing import Any, Callable

#: The fixed order (lower first) after the raw tier's demotion. Owners: the decoder pool (Step 5c), the restorer's
#: device-resident bases (Step 7), the bridge's unclaimed resident state (Step 11).
TRIM_ORDER = {"decoder_pool": 10, "restorer": 20, "bridge_resident": 30}
DEFAULT_ORDER = 50
REASONS = ("decode", "request_start", "oom")


class TrimHandle:
    """A registration; :meth:`unregister` removes it (idempotent)."""

    def __init__(self, registry: DeviceTrimRegistry, key: int):
        self._registry, self.key = registry, key

    def unregister(self) -> None:
        self._registry._remove(self.key)


class DeviceTrimRegistry:
    """Process-wide registry of device-cache trims (see the module docstring)."""

    def __init__(self):
        self._lock = threading.Lock()
        self._entries: dict[int, dict[str, Any]] = {}
        self._keys = itertools.count()

    def register(self, name: str, trim: Callable[[str], int], *, order: int | None = None,
                 device: Any = None) -> TrimHandle:
        """Register ``trim`` under ``name`` (order: :data:`TRIM_ORDER` by name, else ``order`` or 50). ``device``:
        the CUDA device the cache lives on (None: any)."""
        if not isinstance(name, str) or not name:
            raise ValueError("a trim needs a nonempty name")
        if not callable(trim):
            raise ValueError("trim must be callable")
        rank = TRIM_ORDER.get(name.split(":", 1)[0], DEFAULT_ORDER) if order is None else int(order)
        try:  # bound methods: weak (a collected component drops out); plain callables: strong
            reference = weakref.WeakMethod(trim)
        except TypeError:
            reference = (lambda trim=trim: trim)
        key = next(self._keys)
        with self._lock:
            self._entries[key] = {"name": name, "order": rank, "device": _index(device), "trim": reference}
        return TrimHandle(self, key)

    def _remove(self, key: int) -> None:
        with self._lock:
            self._entries.pop(key, None)

    def registered(self) -> list[dict[str, Any]]:
        """The live registrations in call order: ``[{name, order, device}]``."""
        with self._lock:
            items = sorted(self._entries.items(), key=lambda item: (item[1]["order"], item[0]))
            return [{"name": entry["name"], "order": entry["order"], "device": entry["device"]}
                    for _, entry in items if entry["trim"]() is not None]

    def trim(self, reason: str, *, device: Any = None, needed_bytes: int | None = None,
             headroom: Callable[[], int] | None = None,
             empty_cache: Callable[[], Any] | None = None) -> dict[str, Any]:
        """Call the registered trims for ``device`` in order; with ``needed_bytes`` and ``headroom`` stop once
        ``headroom()`` covers it. Then ``empty_cache`` (default ``torch.cuda.empty_cache`` when CUDA is available).
        Returns ``{reason, trims: [{name, bytes, seconds, error?}], bytes, seconds, empty_cache_seconds}``."""
        started = time.perf_counter()
        wanted = _index(device)
        with self._lock:
            items = sorted(self._entries.items(), key=lambda item: (item[1]["order"], item[0]))
        records, total = [], 0
        for key, entry in items:
            if wanted is not None and entry["device"] is not None and entry["device"] != wanted:
                continue
            if needed_bytes is not None and headroom is not None:
                try:
                    if headroom() >= needed_bytes:
                        break
                except Exception:  # noqa: BLE001 - keep trimming without a measurement
                    pass
            trim = entry["trim"]()
            if trim is None:  # the component was collected without unregistering
                self._remove(key)
                continue
            clock = time.perf_counter()
            record: dict[str, Any] = {"name": entry["name"], "bytes": 0}
            try:
                freed = int(trim(reason) or 0)
                record["bytes"] = max(0, freed)
                total += record["bytes"]
            except Exception as exc:  # noqa: BLE001 - recorded: a trim never fails the response
                record["error"] = f"{type(exc).__name__}: {exc}"
            record["seconds"] = time.perf_counter() - clock
            records.append(record)
        clock = time.perf_counter()
        if empty_cache is None:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            empty_cache()
        return {"reason": reason, "trims": records, "bytes": total, "empty_cache_seconds": time.perf_counter() - clock,
                "seconds": time.perf_counter() - started}


def _index(device: Any) -> int | None:
    if device is None:
        return None
    index = getattr(device, "index", device)
    return None if index is None else int(index)


REGISTRY = DeviceTrimRegistry()


def register_device_trim(name: str, trim: Callable[[str], int], *, order: int | None = None,
                         device: Any = None) -> TrimHandle:
    """Register a component's ``trim_device_cache(reason) -> bytes_freed`` with the process-wide registry."""
    return REGISTRY.register(name, trim, order=order, device=device)


def trim_device_caches(reason: str, **kwargs) -> dict[str, Any]:
    """Run the process-wide registry (see :meth:`DeviceTrimRegistry.trim`)."""
    return REGISTRY.trim(reason, **kwargs)


__all__ = ["DEFAULT_ORDER", "DeviceTrimRegistry", "REASONS", "REGISTRY", "TRIM_ORDER", "TrimHandle",
           "register_device_trim", "trim_device_caches"]
