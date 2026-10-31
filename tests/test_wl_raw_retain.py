"""wl-w1b7: symmetric raw-tier LRU -- ``raw_tier_retain`` and ``raw_tier_promote`` (both opt-in) -- on the CPU harness.

* ``raw_tier_retain="lru"``: a published raw entry that loses its last holder (close, replace, an exclusive hold of
  another path, a detached branch's close) stays as an unheld LRU entry; budgets evict unheld entries first; a hold
  re-attaches it (a hit); a whole-tree delete drops what it releases and the unheld entries the session held last.
* ``raw_tier_promote="host"`` (raw_offload and the recompute arm only): a restore inserts its miss path's rows into host
  raw, held by the restoring session -- raw_offload store reads (pre-RoPE, copied into pinned pool blocks on the
  promotion worker), recompute replays (post-RoPE "installed" rows, copied device to host after the replay; fenced on
  rolling caches only). A promotion evicts unheld entries, never held ones.
* The defaults are W1.3.2's tier, event for event (``DEFAULT_SCENARIO_GOLDEN`` was recorded with freeze_w132's code).
"""

import contextlib
import pathlib
import random
import threading
import types
import unittest
import weakref
from types import SimpleNamespace
from unittest import mock

import torch

from reelcache.errors import ConfigurationError
from reelcache.integrations.sglang import ReelCacheRuntimeConfig
from reelcache.integrations.sglang import bridge as bridge_module
from reelcache.integrations.sglang import workload_telemetry as wt
from reelcache import raw_tier as raw_tier_module
from reelcache.raw_tier import RawChunkTier
from reelcache.types import MatrixKey
from tests.test_sglang_bridge import PROFILE, handle
from tests.test_wl_tree import TreeStage, TreeTestCase, assert_caches_bitwise, run_tree

ROWS, COLS = 8, 16
TIMING_KEYS = ("time", "seconds", "spill_seconds", "upload_seconds")


def matrices(value: float, *, layers: int = 1, streams=("cond",)) -> dict:
    return {(MatrixKey(layer, target), stream): torch.full((ROWS, COLS), value + layer + (0.5 if target == "V" else 0))
            for layer in range(layers) for target in ("K", "V") for stream in streams}


def adopt(tier, holder: str, node_id: str, value: float):
    with tier._cond:
        reservation = tier.reserve(2 * ROWS * COLS * 4)
        return tier.adopt(reservation, branch=holder, node_id=node_id, groups=[(node_id, 0, matrices(value))])


def publish(tier, entry, ref: str) -> None:
    with tier._cond:
        tier.published(entry, ref)


def retire(tier, entry, state: str) -> None:
    with tier._cond:
        tier.retire(entry, state)


def settle(tier) -> None:
    tier.flush(30)


def default_scenario(tier_class) -> dict:
    """A scripted holder / budget / restore scenario on a default tier (CPU copies). The record is the tier's state
    after every step -- each entry's state, copies and holders, and the counters -- plus its events (no timing)."""
    entry_bytes = 2 * ROWS * COLS * 4
    tier = tier_class(gpu_budget_bytes=3 * entry_bytes, host_budget_bytes=3 * 1024, pinned=False)
    steps = []

    def snapshot(name, on=None, into=None):
        on = tier if on is None else on
        settle(on)
        stats = on.stats()
        with on._cond:
            entries = sorted((entry.label[1], entry.state, entry.node_ref, entry.on_device, entry.host_state,
                              sorted(entry.holders)) for entry in on._entries.values() if not entry.dropped)
        (steps if into is None else into).append([name, entries, {
            key: (dict(value) if isinstance(value, dict) else value) for key, value in stats.items()
            if key in ("spills", "evictions_gpu", "evictions_host", "drops_published", "drops_failed",
                       "drops_cancelled", "drops_released", "drops_duplicate", "restores", "restore_misses",
                       "gpu_bytes", "host_bytes", "unpublished_bytes")}])

    entries = []
    for index in range(6):
        entries.append(adopt(tier, "s", f"c{index}", float(index)))
        snapshot(f"adopt c{index}")
    for index in range(4):
        publish(tier, entries[index], f"ref{index}")
        snapshot(f"publish ref{index}")
    tier.hold("s", ["ref0", "ref1"])  # a rewind: the session's path is ref0, ref1
    snapshot("rewind")
    tier.hold("t", ["ref0"], exclusive=False)  # a fork shares ref0
    lease = tier.acquire("ref0")
    missed = tier.acquire("ref3")
    if lease is not None:
        lease.release()
    snapshot("fork and restores")
    tier.release_holder("s")  # session close
    snapshot("close s")
    publish(tier, entries[4], "ref4")  # published after its session closed: no holder
    retire(tier, entries[5], "failed")
    snapshot("late publish and failure")
    tier.release_holder("t")
    snapshot("close t")
    # the eviction ORDER among published copies (several candidates): restores make dref1, dref0 the most recently
    # used, then one more publication needs GPU and host room (LRU victims; another order records other states)
    for index in range(5):
        publish(tier, adopt(tier, "u", f"d{index}", 10.0 + index), f"dref{index}")
        snapshot(f"publish dref{index}")
    for ref in ("dref1", "dref0"):
        touched = tier.acquire(ref)
        if touched is not None:
            touched.release()
    snapshot("restores touch dref1, dref0")
    publish(tier, adopt(tier, "u", "d5", 15.0), "dref5")
    snapshot("publish dref5 under pressure")
    publish(tier, adopt(tier, "u", "d6", 16.0), "dref6")  # its spill needs host room: an LRU host copy goes
    snapshot("publish dref6 under host pressure")
    tier.release_holder("u")
    snapshot("close u")
    stats = tier.stats()
    record = {"lease": None if lease is None else lease.tier, "missed": missed is None, "steps": steps,
              "events": [{key: value for key, value in item.items() if key not in TIMING_KEYS}
                         for item in stats["events"]],
              "has_retain_records": "retain" in stats or "promotion" in stats}
    tier.close()
    # D4 (W1.4.1 verifier): LRU -- not FIFO -- among device-resident published copies under GPU pressure, and the
    # pressure response (_relieve) demoting least recently used device copies, with its exact ordered event list. A
    # separate tier: its device memory reads roomy until the explicit check (no admission-time pressure job)
    memory = {"free": 1 << 40, "reserved": 0, "allocated": 0, "inactive_split": 0, "total": 1 << 40}
    second = tier_class(gpu_budget_bytes=3 * entry_bytes, host_budget_bytes=4 * 1024, pinned=False,
                        gpu_reserve_bytes=2 * entry_bytes, device_memory=lambda device: dict(memory))
    pressure_steps: list = []
    for index in range(3):
        publish(second, adopt(second, "v", f"e{index}", 20.0 + index), f"eref{index}")
    snapshot("publish eref0-2 (the GPU budget full)", second, pressure_steps)
    touched = second.acquire("eref0")  # the OLDEST device copy becomes the most recently used
    if touched is not None:
        touched.release()
    snapshot("a restore touches eref0", second, pressure_steps)
    publish(second, adopt(second, "v", "e3", 23.0), "eref3")  # over the GPU budget: LRU demotes eref1 (FIFO: eref0)
    snapshot("publish eref3 under GPU pressure", second, pressure_steps)
    touched = second.acquire("eref2")
    if touched is not None:
        touched.release()
    snapshot("a restore touches eref2", second, pressure_steps)
    memory["free"] = 0  # below the reserve: the response demotes the least recently used device copies
    checked = second.pressure_check("decode", wait=False)
    snapshot("pressure response", second, pressure_steps)
    second.release_holder("v")
    snapshot("close v", second, pressure_steps)
    pressure_stats = second.stats()
    record.update(pressure_steps=pressure_steps,
                  pressure_events=[{key: value for key, value in item.items() if key not in TIMING_KEYS}
                                   for item in pressure_stats["events"]],
                  pressure_check=None if checked is None else {
                      key: checked.get(key) for key in ("pressure", "selected_entries", "selected_bytes", "deficit",
                                                        "target", "headroom", "gpu_bytes_before")})
    second.close()
    return record


#: ``default_scenario(RawChunkTier)`` with freeze_w132's ``reelcache/raw_tier.py`` (2f42b152841bc75e), recorded by
#: ``dev_w1b/scratch_w13/w1b7_golden.py``.
DEFAULT_SCENARIO_GOLDEN = {'events': [],
                           'has_retain_records': False,
                           'lease': 'gpu_raw',
                           'missed': True,
                           'pressure_check': {'deficit': 2048,
                                              'gpu_bytes_before': 3072,
                                              'headroom': 0,
                                              'pressure': True,
                                              'selected_bytes': 2048,
                                              'selected_entries': 2,
                                              'target': 2048},
                           'pressure_events': [{'deficit': 2048,
                                                'event': 'pressure',
                                                'freed_bytes': 0,
                                                'where': 'decode'}],
                           'pressure_steps': [['publish eref0-2 (the GPU budget full)',
                                               [('e0', 'published', 'eref0', True, 'none', ['v']),
                                                ('e1', 'published', 'eref1', True, 'none', ['v']),
                                                ('e2', 'published', 'eref2', True, 'none', ['v'])],
                                               {'drops_cancelled': 0,
                                                'drops_duplicate': 0,
                                                'drops_failed': 0,
                                                'drops_published': 0,
                                                'drops_released': 0,
                                                'evictions_gpu': 0,
                                                'evictions_host': 0,
                                                'gpu_bytes': 3072,
                                                'host_bytes': 0,
                                                'restore_misses': 0,
                                                'restores': {'gpu_raw': 0, 'host_raw': 0},
                                                'spills': 0,
                                                'unpublished_bytes': 0}],
                                              ['a restore touches eref0',
                                               [('e0', 'published', 'eref0', True, 'none', ['v']),
                                                ('e1', 'published', 'eref1', True, 'none', ['v']),
                                                ('e2', 'published', 'eref2', True, 'none', ['v'])],
                                               {'drops_cancelled': 0,
                                                'drops_duplicate': 0,
                                                'drops_failed': 0,
                                                'drops_published': 0,
                                                'drops_released': 0,
                                                'evictions_gpu': 0,
                                                'evictions_host': 0,
                                                'gpu_bytes': 3072,
                                                'host_bytes': 0,
                                                'restore_misses': 0,
                                                'restores': {'gpu_raw': 1, 'host_raw': 0},
                                                'spills': 0,
                                                'unpublished_bytes': 0}],
                                              ['publish eref3 under GPU pressure',
                                               [('e0', 'published', 'eref0', True, 'none', ['v']),
                                                ('e1', 'published', 'eref1', False, 'complete', ['v']),
                                                ('e2', 'published', 'eref2', True, 'none', ['v']),
                                                ('e3', 'published', 'eref3', True, 'none', ['v'])],
                                               {'drops_cancelled': 0,
                                                'drops_duplicate': 0,
                                                'drops_failed': 0,
                                                'drops_published': 0,
                                                'drops_released': 0,
                                                'evictions_gpu': 1,
                                                'evictions_host': 0,
                                                'gpu_bytes': 3072,
                                                'host_bytes': 1024,
                                                'restore_misses': 0,
                                                'restores': {'gpu_raw': 1, 'host_raw': 0},
                                                'spills': 1,
                                                'unpublished_bytes': 0}],
                                              ['a restore touches eref2',
                                               [('e0', 'published', 'eref0', True, 'none', ['v']),
                                                ('e1', 'published', 'eref1', False, 'complete', ['v']),
                                                ('e2', 'published', 'eref2', True, 'none', ['v']),
                                                ('e3', 'published', 'eref3', True, 'none', ['v'])],
                                               {'drops_cancelled': 0,
                                                'drops_duplicate': 0,
                                                'drops_failed': 0,
                                                'drops_published': 0,
                                                'drops_released': 0,
                                                'evictions_gpu': 1,
                                                'evictions_host': 0,
                                                'gpu_bytes': 3072,
                                                'host_bytes': 1024,
                                                'restore_misses': 0,
                                                'restores': {'gpu_raw': 2, 'host_raw': 0},
                                                'spills': 1,
                                                'unpublished_bytes': 0}],
                                              ['pressure response',
                                               [('e0', 'published', 'eref0', False, 'complete', ['v']),
                                                ('e1', 'published', 'eref1', False, 'complete', ['v']),
                                                ('e2', 'published', 'eref2', True, 'none', ['v']),
                                                ('e3', 'published', 'eref3', False, 'complete', ['v'])],
                                               {'drops_cancelled': 0,
                                                'drops_duplicate': 0,
                                                'drops_failed': 0,
                                                'drops_published': 0,
                                                'drops_released': 0,
                                                'evictions_gpu': 1,
                                                'evictions_host': 0,
                                                'gpu_bytes': 1024,
                                                'host_bytes': 3072,
                                                'restore_misses': 0,
                                                'restores': {'gpu_raw': 2, 'host_raw': 0},
                                                'spills': 3,
                                                'unpublished_bytes': 0}],
                                              ['close v',
                                               [],
                                               {'drops_cancelled': 0,
                                                'drops_duplicate': 0,
                                                'drops_failed': 0,
                                                'drops_published': 0,
                                                'drops_released': 4,
                                                'evictions_gpu': 1,
                                                'evictions_host': 0,
                                                'gpu_bytes': 0,
                                                'host_bytes': 0,
                                                'restore_misses': 0,
                                                'restores': {'gpu_raw': 2, 'host_raw': 0},
                                                'spills': 3,
                                                'unpublished_bytes': 0}]],
                           'steps': [['adopt c0',
                                      [('c0', 'submitted', None, True, 'none', ['s'])],
                                      {'drops_cancelled': 0,
                                       'drops_duplicate': 0,
                                       'drops_failed': 0,
                                       'drops_published': 0,
                                       'drops_released': 0,
                                       'evictions_gpu': 0,
                                       'evictions_host': 0,
                                       'gpu_bytes': 1024,
                                       'host_bytes': 0,
                                       'restore_misses': 0,
                                       'restores': {'gpu_raw': 0, 'host_raw': 0},
                                       'spills': 0,
                                       'unpublished_bytes': 1024}],
                                     ['adopt c1',
                                      [('c0', 'submitted', None, True, 'none', ['s']),
                                       ('c1', 'submitted', None, True, 'none', ['s'])],
                                      {'drops_cancelled': 0,
                                       'drops_duplicate': 0,
                                       'drops_failed': 0,
                                       'drops_published': 0,
                                       'drops_released': 0,
                                       'evictions_gpu': 0,
                                       'evictions_host': 0,
                                       'gpu_bytes': 2048,
                                       'host_bytes': 0,
                                       'restore_misses': 0,
                                       'restores': {'gpu_raw': 0, 'host_raw': 0},
                                       'spills': 0,
                                       'unpublished_bytes': 2048}],
                                     ['adopt c2',
                                      [('c0', 'submitted', None, True, 'none', ['s']),
                                       ('c1', 'submitted', None, True, 'none', ['s']),
                                       ('c2', 'submitted', None, True, 'none', ['s'])],
                                      {'drops_cancelled': 0,
                                       'drops_duplicate': 0,
                                       'drops_failed': 0,
                                       'drops_published': 0,
                                       'drops_released': 0,
                                       'evictions_gpu': 0,
                                       'evictions_host': 0,
                                       'gpu_bytes': 3072,
                                       'host_bytes': 0,
                                       'restore_misses': 0,
                                       'restores': {'gpu_raw': 0, 'host_raw': 0},
                                       'spills': 0,
                                       'unpublished_bytes': 3072}],
                                     ['adopt c3',
                                      [('c0', 'submitted', None, True, 'none', ['s']),
                                       ('c1', 'submitted', None, True, 'none', ['s']),
                                       ('c2', 'submitted', None, True, 'none', ['s']),
                                       ('c3', 'submitted', None, False, 'complete', ['s'])],
                                      {'drops_cancelled': 0,
                                       'drops_duplicate': 0,
                                       'drops_failed': 0,
                                       'drops_published': 0,
                                       'drops_released': 0,
                                       'evictions_gpu': 1,
                                       'evictions_host': 0,
                                       'gpu_bytes': 3072,
                                       'host_bytes': 1024,
                                       'restore_misses': 0,
                                       'restores': {'gpu_raw': 0, 'host_raw': 0},
                                       'spills': 1,
                                       'unpublished_bytes': 4096}],
                                     ['adopt c4',
                                      [('c0', 'submitted', None, True, 'none', ['s']),
                                       ('c1', 'submitted', None, True, 'none', ['s']),
                                       ('c2', 'submitted', None, True, 'none', ['s']),
                                       ('c3', 'submitted', None, False, 'complete', ['s']),
                                       ('c4', 'submitted', None, False, 'complete', ['s'])],
                                      {'drops_cancelled': 0,
                                       'drops_duplicate': 0,
                                       'drops_failed': 0,
                                       'drops_published': 0,
                                       'drops_released': 0,
                                       'evictions_gpu': 2,
                                       'evictions_host': 0,
                                       'gpu_bytes': 3072,
                                       'host_bytes': 2048,
                                       'restore_misses': 0,
                                       'restores': {'gpu_raw': 0, 'host_raw': 0},
                                       'spills': 2,
                                       'unpublished_bytes': 5120}],
                                     ['adopt c5',
                                      [('c0', 'submitted', None, True, 'none', ['s']),
                                       ('c1', 'submitted', None, True, 'none', ['s']),
                                       ('c2', 'submitted', None, True, 'none', ['s']),
                                       ('c3', 'submitted', None, False, 'complete', ['s']),
                                       ('c4', 'submitted', None, False, 'complete', ['s']),
                                       ('c5', 'submitted', None, False, 'complete', ['s'])],
                                      {'drops_cancelled': 0,
                                       'drops_duplicate': 0,
                                       'drops_failed': 0,
                                       'drops_published': 0,
                                       'drops_released': 0,
                                       'evictions_gpu': 3,
                                       'evictions_host': 0,
                                       'gpu_bytes': 3072,
                                       'host_bytes': 3072,
                                       'restore_misses': 0,
                                       'restores': {'gpu_raw': 0, 'host_raw': 0},
                                       'spills': 3,
                                       'unpublished_bytes': 6144}],
                                     ['publish ref0',
                                      [('c0', 'published', 'ref0', True, 'none', ['s']),
                                       ('c1', 'submitted', None, True, 'none', ['s']),
                                       ('c2', 'submitted', None, True, 'none', ['s']),
                                       ('c3', 'submitted', None, False, 'complete', ['s']),
                                       ('c4', 'submitted', None, False, 'complete', ['s']),
                                       ('c5', 'submitted', None, False, 'complete', ['s'])],
                                      {'drops_cancelled': 0,
                                       'drops_duplicate': 0,
                                       'drops_failed': 0,
                                       'drops_published': 0,
                                       'drops_released': 0,
                                       'evictions_gpu': 3,
                                       'evictions_host': 0,
                                       'gpu_bytes': 3072,
                                       'host_bytes': 3072,
                                       'restore_misses': 0,
                                       'restores': {'gpu_raw': 0, 'host_raw': 0},
                                       'spills': 3,
                                       'unpublished_bytes': 5120}],
                                     ['publish ref1',
                                      [('c0', 'published', 'ref0', True, 'none', ['s']),
                                       ('c1', 'published', 'ref1', True, 'none', ['s']),
                                       ('c2', 'submitted', None, True, 'none', ['s']),
                                       ('c3', 'submitted', None, False, 'complete', ['s']),
                                       ('c4', 'submitted', None, False, 'complete', ['s']),
                                       ('c5', 'submitted', None, False, 'complete', ['s'])],
                                      {'drops_cancelled': 0,
                                       'drops_duplicate': 0,
                                       'drops_failed': 0,
                                       'drops_published': 0,
                                       'drops_released': 0,
                                       'evictions_gpu': 3,
                                       'evictions_host': 0,
                                       'gpu_bytes': 3072,
                                       'host_bytes': 3072,
                                       'restore_misses': 0,
                                       'restores': {'gpu_raw': 0, 'host_raw': 0},
                                       'spills': 3,
                                       'unpublished_bytes': 4096}],
                                     ['publish ref2',
                                      [('c0', 'published', 'ref0', True, 'none', ['s']),
                                       ('c1', 'published', 'ref1', True, 'none', ['s']),
                                       ('c2', 'published', 'ref2', True, 'none', ['s']),
                                       ('c3', 'submitted', None, False, 'complete', ['s']),
                                       ('c4', 'submitted', None, False, 'complete', ['s']),
                                       ('c5', 'submitted', None, False, 'complete', ['s'])],
                                      {'drops_cancelled': 0,
                                       'drops_duplicate': 0,
                                       'drops_failed': 0,
                                       'drops_published': 0,
                                       'drops_released': 0,
                                       'evictions_gpu': 3,
                                       'evictions_host': 0,
                                       'gpu_bytes': 3072,
                                       'host_bytes': 3072,
                                       'restore_misses': 0,
                                       'restores': {'gpu_raw': 0, 'host_raw': 0},
                                       'spills': 3,
                                       'unpublished_bytes': 3072}],
                                     ['publish ref3',
                                      [('c0', 'published', 'ref0', True, 'none', ['s']),
                                       ('c1', 'published', 'ref1', True, 'none', ['s']),
                                       ('c2', 'published', 'ref2', True, 'none', ['s']),
                                       ('c3', 'published', 'ref3', False, 'complete', ['s']),
                                       ('c4', 'submitted', None, False, 'complete', ['s']),
                                       ('c5', 'submitted', None, False, 'complete', ['s'])],
                                      {'drops_cancelled': 0,
                                       'drops_duplicate': 0,
                                       'drops_failed': 0,
                                       'drops_published': 0,
                                       'drops_released': 0,
                                       'evictions_gpu': 3,
                                       'evictions_host': 0,
                                       'gpu_bytes': 3072,
                                       'host_bytes': 3072,
                                       'restore_misses': 0,
                                       'restores': {'gpu_raw': 0, 'host_raw': 0},
                                       'spills': 3,
                                       'unpublished_bytes': 2048}],
                                     ['rewind',
                                      [('c0', 'published', 'ref0', True, 'none', ['s']),
                                       ('c1', 'published', 'ref1', True, 'none', ['s']),
                                       ('c4', 'submitted', None, False, 'complete', ['s']),
                                       ('c5', 'submitted', None, False, 'complete', ['s'])],
                                      {'drops_cancelled': 0,
                                       'drops_duplicate': 0,
                                       'drops_failed': 0,
                                       'drops_published': 0,
                                       'drops_released': 2,
                                       'evictions_gpu': 3,
                                       'evictions_host': 0,
                                       'gpu_bytes': 2048,
                                       'host_bytes': 2048,
                                       'restore_misses': 0,
                                       'restores': {'gpu_raw': 0, 'host_raw': 0},
                                       'spills': 3,
                                       'unpublished_bytes': 2048}],
                                     ['fork and restores',
                                      [('c0', 'published', 'ref0', True, 'none', ['s', 't']),
                                       ('c1', 'published', 'ref1', True, 'none', ['s']),
                                       ('c4', 'submitted', None, False, 'complete', ['s']),
                                       ('c5', 'submitted', None, False, 'complete', ['s'])],
                                      {'drops_cancelled': 0,
                                       'drops_duplicate': 0,
                                       'drops_failed': 0,
                                       'drops_published': 0,
                                       'drops_released': 2,
                                       'evictions_gpu': 3,
                                       'evictions_host': 0,
                                       'gpu_bytes': 2048,
                                       'host_bytes': 2048,
                                       'restore_misses': 1,
                                       'restores': {'gpu_raw': 1, 'host_raw': 0},
                                       'spills': 3,
                                       'unpublished_bytes': 2048}],
                                     ['close s',
                                      [('c0', 'published', 'ref0', True, 'none', ['t']),
                                       ('c4', 'submitted', None, False, 'complete', []),
                                       ('c5', 'submitted', None, False, 'complete', [])],
                                      {'drops_cancelled': 0,
                                       'drops_duplicate': 0,
                                       'drops_failed': 0,
                                       'drops_published': 0,
                                       'drops_released': 3,
                                       'evictions_gpu': 3,
                                       'evictions_host': 0,
                                       'gpu_bytes': 1024,
                                       'host_bytes': 2048,
                                       'restore_misses': 1,
                                       'restores': {'gpu_raw': 1, 'host_raw': 0},
                                       'spills': 3,
                                       'unpublished_bytes': 2048}],
                                     ['late publish and failure',
                                      [('c0', 'published', 'ref0', True, 'none', ['t'])],
                                      {'drops_cancelled': 0,
                                       'drops_duplicate': 0,
                                       'drops_failed': 1,
                                       'drops_published': 0,
                                       'drops_released': 4,
                                       'evictions_gpu': 3,
                                       'evictions_host': 0,
                                       'gpu_bytes': 1024,
                                       'host_bytes': 0,
                                       'restore_misses': 1,
                                       'restores': {'gpu_raw': 1, 'host_raw': 0},
                                       'spills': 3,
                                       'unpublished_bytes': 0}],
                                     ['close t',
                                      [],
                                      {'drops_cancelled': 0,
                                       'drops_duplicate': 0,
                                       'drops_failed': 1,
                                       'drops_published': 0,
                                       'drops_released': 5,
                                       'evictions_gpu': 3,
                                       'evictions_host': 0,
                                       'gpu_bytes': 0,
                                       'host_bytes': 0,
                                       'restore_misses': 1,
                                       'restores': {'gpu_raw': 1, 'host_raw': 0},
                                       'spills': 3,
                                       'unpublished_bytes': 0}],
                                     ['publish dref0',
                                      [('d0', 'published', 'dref0', True, 'none', ['u'])],
                                      {'drops_cancelled': 0,
                                       'drops_duplicate': 0,
                                       'drops_failed': 1,
                                       'drops_published': 0,
                                       'drops_released': 5,
                                       'evictions_gpu': 3,
                                       'evictions_host': 0,
                                       'gpu_bytes': 1024,
                                       'host_bytes': 0,
                                       'restore_misses': 1,
                                       'restores': {'gpu_raw': 1, 'host_raw': 0},
                                       'spills': 3,
                                       'unpublished_bytes': 0}],
                                     ['publish dref1',
                                      [('d0', 'published', 'dref0', True, 'none', ['u']),
                                       ('d1', 'published', 'dref1', True, 'none', ['u'])],
                                      {'drops_cancelled': 0,
                                       'drops_duplicate': 0,
                                       'drops_failed': 1,
                                       'drops_published': 0,
                                       'drops_released': 5,
                                       'evictions_gpu': 3,
                                       'evictions_host': 0,
                                       'gpu_bytes': 2048,
                                       'host_bytes': 0,
                                       'restore_misses': 1,
                                       'restores': {'gpu_raw': 1, 'host_raw': 0},
                                       'spills': 3,
                                       'unpublished_bytes': 0}],
                                     ['publish dref2',
                                      [('d0', 'published', 'dref0', True, 'none', ['u']),
                                       ('d1', 'published', 'dref1', True, 'none', ['u']),
                                       ('d2', 'published', 'dref2', True, 'none', ['u'])],
                                      {'drops_cancelled': 0,
                                       'drops_duplicate': 0,
                                       'drops_failed': 1,
                                       'drops_published': 0,
                                       'drops_released': 5,
                                       'evictions_gpu': 3,
                                       'evictions_host': 0,
                                       'gpu_bytes': 3072,
                                       'host_bytes': 0,
                                       'restore_misses': 1,
                                       'restores': {'gpu_raw': 1, 'host_raw': 0},
                                       'spills': 3,
                                       'unpublished_bytes': 0}],
                                     ['publish dref3',
                                      [('d0', 'published', 'dref0', False, 'complete', ['u']),
                                       ('d1', 'published', 'dref1', True, 'none', ['u']),
                                       ('d2', 'published', 'dref2', True, 'none', ['u']),
                                       ('d3', 'published', 'dref3', True, 'none', ['u'])],
                                      {'drops_cancelled': 0,
                                       'drops_duplicate': 0,
                                       'drops_failed': 1,
                                       'drops_published': 0,
                                       'drops_released': 5,
                                       'evictions_gpu': 4,
                                       'evictions_host': 0,
                                       'gpu_bytes': 3072,
                                       'host_bytes': 1024,
                                       'restore_misses': 1,
                                       'restores': {'gpu_raw': 1, 'host_raw': 0},
                                       'spills': 4,
                                       'unpublished_bytes': 0}],
                                     ['publish dref4',
                                      [('d0', 'published', 'dref0', False, 'complete', ['u']),
                                       ('d1', 'published', 'dref1', False, 'complete', ['u']),
                                       ('d2', 'published', 'dref2', True, 'none', ['u']),
                                       ('d3', 'published', 'dref3', True, 'none', ['u']),
                                       ('d4', 'published', 'dref4', True, 'none', ['u'])],
                                      {'drops_cancelled': 0,
                                       'drops_duplicate': 0,
                                       'drops_failed': 1,
                                       'drops_published': 0,
                                       'drops_released': 5,
                                       'evictions_gpu': 5,
                                       'evictions_host': 0,
                                       'gpu_bytes': 3072,
                                       'host_bytes': 2048,
                                       'restore_misses': 1,
                                       'restores': {'gpu_raw': 1, 'host_raw': 0},
                                       'spills': 5,
                                       'unpublished_bytes': 0}],
                                     ['restores touch dref1, dref0',
                                      [('d0', 'published', 'dref0', False, 'complete', ['u']),
                                       ('d1', 'published', 'dref1', False, 'complete', ['u']),
                                       ('d2', 'published', 'dref2', True, 'none', ['u']),
                                       ('d3', 'published', 'dref3', True, 'none', ['u']),
                                       ('d4', 'published', 'dref4', True, 'none', ['u'])],
                                      {'drops_cancelled': 0,
                                       'drops_duplicate': 0,
                                       'drops_failed': 1,
                                       'drops_published': 0,
                                       'drops_released': 5,
                                       'evictions_gpu': 5,
                                       'evictions_host': 0,
                                       'gpu_bytes': 3072,
                                       'host_bytes': 2048,
                                       'restore_misses': 1,
                                       'restores': {'gpu_raw': 1, 'host_raw': 2},
                                       'spills': 5,
                                       'unpublished_bytes': 0}],
                                     ['publish dref5 under pressure',
                                      [('d0', 'published', 'dref0', False, 'complete', ['u']),
                                       ('d1', 'published', 'dref1', False, 'complete', ['u']),
                                       ('d2', 'published', 'dref2', False, 'complete', ['u']),
                                       ('d3', 'published', 'dref3', True, 'none', ['u']),
                                       ('d4', 'published', 'dref4', True, 'none', ['u']),
                                       ('d5', 'published', 'dref5', True, 'none', ['u'])],
                                      {'drops_cancelled': 0,
                                       'drops_duplicate': 0,
                                       'drops_failed': 1,
                                       'drops_published': 0,
                                       'drops_released': 5,
                                       'evictions_gpu': 6,
                                       'evictions_host': 0,
                                       'gpu_bytes': 3072,
                                       'host_bytes': 3072,
                                       'restore_misses': 1,
                                       'restores': {'gpu_raw': 1, 'host_raw': 2},
                                       'spills': 6,
                                       'unpublished_bytes': 0}],
                                     ['publish dref6 under host pressure',
                                      [('d0', 'published', 'dref0', False, 'complete', ['u']),
                                       ('d1', 'published', 'dref1', False, 'complete', ['u']),
                                       ('d3', 'published', 'dref3', False, 'complete', ['u']),
                                       ('d4', 'published', 'dref4', True, 'none', ['u']),
                                       ('d5', 'published', 'dref5', True, 'none', ['u']),
                                       ('d6', 'published', 'dref6', True, 'none', ['u'])],
                                      {'drops_cancelled': 0,
                                       'drops_duplicate': 0,
                                       'drops_failed': 1,
                                       'drops_published': 0,
                                       'drops_released': 5,
                                       'evictions_gpu': 7,
                                       'evictions_host': 1,
                                       'gpu_bytes': 3072,
                                       'host_bytes': 3072,
                                       'restore_misses': 1,
                                       'restores': {'gpu_raw': 1, 'host_raw': 2},
                                       'spills': 7,
                                       'unpublished_bytes': 0}],
                                     ['close u',
                                      [],
                                      {'drops_cancelled': 0,
                                       'drops_duplicate': 0,
                                       'drops_failed': 1,
                                       'drops_published': 0,
                                       'drops_released': 11,
                                       'evictions_gpu': 7,
                                       'evictions_host': 1,
                                       'gpu_bytes': 0,
                                       'host_bytes': 0,
                                       'restore_misses': 1,
                                       'restores': {'gpu_raw': 1, 'host_raw': 2},
                                       'spills': 7,
                                       'unpublished_bytes': 0}]]}


class TierRetainTests(unittest.TestCase):
    def tier(self, *, retain="lru", promote="off", gpu=3, host=8, host_bytes=None):
        entry_bytes = 2 * ROWS * COLS * 4
        tier = RawChunkTier(gpu_budget_bytes=gpu * entry_bytes,
                            host_budget_bytes=host * 4096 if host_bytes is None else host_bytes, pinned=False,
                            retain=retain, promote=promote)
        self.addCleanup(tier.close)
        return tier

    def published(self, tier, holder, refs, value=0.0):
        made = []
        for offset, ref in enumerate(refs):
            entry = adopt(tier, holder, ref, value + offset)
            publish(tier, entry, ref)
            made.append(entry)
        settle(tier)
        return made

    def test_default_tier_event_sequence_equals_w132(self):
        record = default_scenario(RawChunkTier)
        self.assertFalse(record["has_retain_records"])  # the defaults add no record
        self.assertEqual(record, DEFAULT_SCENARIO_GOLDEN)

    def test_invalid_options(self):
        for bad in ({"retain": "keep"}, {"promote": "gpu"}):
            with self.assertRaises(ConfigurationError):
                RawChunkTier(gpu_budget_bytes=1, host_budget_bytes=1 << 20, pinned=False, **bad)

    def test_retain_on_close_keeps_an_unheld_entry_and_a_hold_reattaches_it(self):
        for retain in ("drop", "lru"):
            with self.subTest(retain=retain):
                tier = self.tier(retain=retain)
                self.published(tier, "s", ["a", "b"])
                tier.release_holder("s")  # session close
                settle(tier)
                kept = {entry.node_ref for entry in tier.entries()}
                if retain == "drop":
                    self.assertEqual(kept, set())
                    self.assertNotIn("retain", tier.stats())
                    continue
                self.assertEqual(kept, {"a", "b"})
                stats = tier.stats()["retain"]
                self.assertEqual((stats["retained"], stats["unheld"]["entries"]["gpu"]), (2, 2))
                self.assertEqual(tier.hold("s2", ["a"])["attached"], 1)  # a later hold: a hit, re-attached
                self.assertEqual(tier.stats()["retain"]["reattached"], 1)
                [entry] = tier.entries(node_refs=["a"])
                self.assertEqual((entry.holders, entry.released_by), ({"s2"}, None))
                hits = [tier.acquire(ref) for ref in ("a", "b")]  # "a" re-attached first (a reopen), "b" still unheld
                self.assertEqual([lease.retained for lease in hits], [True, True])  # both kept only by retain
                for lease in hits:
                    lease.release()
                self.assertEqual(tier.stats()["retain"]["retained_hits"], 2)
                again = tier.acquire("a")  # its next restore is an ordinary hit
                self.assertFalse(again.retained)
                again.release()
                self.assertEqual(tier.stats()["retain"]["retained_hits"], 2)

    def test_retain_on_an_exclusive_rehold_after_a_rewind(self):
        for retain in ("drop", "lru"):
            with self.subTest(retain=retain):
                tier = self.tier(retain=retain)
                self.published(tier, "s", ["a", "b", "c"])
                tier.hold("s", ["a"])  # the head moved back to a
                settle(tier)
                left = {entry.node_ref: sorted(entry.holders) for entry in tier.entries()}
                self.assertEqual(left, {"a": ["s"]} if retain == "drop" else {"a": ["s"], "b": [], "c": []})
                if retain == "lru":
                    self.assertEqual(tier.hold("s", ["a", "b"])["attached"], 1)  # forward again: b re-attached
                    self.assertEqual(tier.stats()["retain"]["reattached"], 1)

    def test_whole_tree_delete_drops_and_other_sessions_keep_theirs(self):
        tier = self.tier()
        self.published(tier, "s", ["a", "b"])
        self.published(tier, "x", ["y"])
        tier.hold("s/tree", ["a", "b"], exclusive=False)
        tier.release_holder("x")  # x closed: y unheld, attributed to x
        tier.hold("s", ["a"])  # s rewound: b keeps its tree holder
        tier.release_holder("s/tree")  # b unheld, attributed to s's tree holder
        settle(tier)
        self.assertEqual({entry.node_ref for entry in tier.entries()}, {"a", "b", "y"})
        tier.release_holder("s", drop=True)  # whole-tree delete of s: what it releases drops ...
        self.assertEqual(tier.drop_unheld("s"), 1)  # ... and the unheld entries it held last
        settle(tier)
        self.assertEqual({entry.node_ref for entry in tier.entries()}, {"y"})
        self.assertEqual(tier.stats()["retain"]["dropped_on_delete"], 1)
        self.assertEqual(tier.drop_unheld("s"), 0)  # idempotent; "sx" or "x" are other sessions
        self.assertEqual(tier.drop_unheld("x"), 1)

    def test_unheld_entries_are_evicted_before_held_ones(self):
        tier = self.tier(gpu=0, host_bytes=3 * 1024)  # host copies only, room for three toy entries
        held = self.published(tier, "s", ["held"])  # least recently used of all
        self.published(tier, "x", ["unheld"])
        tier.release_holder("x")
        settle(tier)
        self.published(tier, "t", ["new1", "new2"])  # needs room: the unheld entry goes first, not the older held one
        refs = {entry.node_ref for entry in tier.entries()}
        self.assertIn("held", refs)
        self.assertNotIn("unheld", refs)
        evictions = tier.stats()["retain"]["evictions"]
        self.assertGreaterEqual(evictions["host_unheld"], 1)
        self.assertEqual(evictions["host_held"], 0)
        self.assertEqual(held[0].holders, {"s"})

    def test_detach_transfer_then_release_leaves_unheld_entries_of_the_session(self):
        tier = self.tier()
        made = self.published(tier, "s", ["a", "b"])
        self.assertEqual(tier.transfer(made, "s", "s~d1-ab"), 2)  # a detached branch takes them
        tier.release_holder("s~d1-ab")  # the branch closed
        settle(tier)
        entries = {entry.node_ref: entry for entry in tier.entries()}
        self.assertEqual({ref: sorted(entry.released_by) for ref, entry in entries.items()},
                         {"a": ["s~d1-ab"], "b": ["s~d1-ab"]})
        self.assertEqual(tier.drop_unheld("s"), 2)  # the session's whole-tree delete reclaims them

    def test_promotion_is_an_ordinary_lru_insert_and_pinned_entries_survive(self):
        """Coordinator 21:25: a promotion evicts like a generation insert -- unheld entries first, then held ones by
        LRU -- never a pinned or in-flight copy; it is skipped only when it cannot fit at all."""
        entry_host = 1024  # two 512 B matrices per toy entry
        tier = self.tier(retain="lru", promote="host", gpu=0, host_bytes=3 * entry_host)  # host copies only
        self.published(tier, "s", ["old", "mid", "new"])  # the budget is full of held entries
        self.assertEqual({entry.host_state for entry in tier.entries()}, {"complete"})
        pinned = tier.acquire("old")  # a restore reads the least recently used one: pinned
        self.assertIsNotNone(pinned)
        source = matrices(7.0)
        out = tier.promote_host(node_ref="p", groups=[("p", 0, source)], holder="s", origin="store", device="cpu")
        self.assertTrue(out["promoted"])
        settle(tier)
        refs = {entry.node_ref for entry in tier.entries()}
        self.assertEqual(refs, {"old", "new", "p"})  # the LRU evictable held entry went; the pinned one survived
        self.assertEqual(tier.stats()["retain"]["evictions"]["host_held"], 1)
        lease = tier.acquire("p")
        self.assertEqual((lease.tier, lease.entry.origin, lease.entry.form), ("host_raw", "store", "pre_rope"))
        self.assertTrue(torch.equal(lease.matrix(0, (MatrixKey(0, "K"), "cond")), source[(MatrixKey(0, "K"), "cond")]))
        lease.release()  # "p": held by s, evictable again
        self.assertEqual(tier.promote_host(node_ref="p", groups=[("p", 0, source)], holder="s", origin="store",
                                           device="cpu")["skip"], "exists")
        # unheld entries go first, even when a held one is older
        self.published(tier, "x", ["unheld"])  # evicts the LRU evictable held entry ("new") to fit
        tier.release_holder("x")  # unheld now, and more recently used than the held "p"
        settle(tier)
        tier.promote_host(node_ref="q", groups=[("q", 0, matrices(8.0))], holder="s", origin="store", device="cpu")
        settle(tier)
        refs = {entry.node_ref for entry in tier.entries()}
        self.assertEqual(refs, {"old", "p", "q"})  # the unheld entry went, not the older held "p"; "old" pinned
        self.assertEqual(tier.stats()["retain"]["evictions"]["host_unheld"], 1)
        # every entry pinned or in flight: the promotion cannot fit at all
        others = [tier.acquire(ref) for ref in sorted(refs - {"old"})]
        out = tier.promote_host(node_ref="r", groups=[("r", 0, matrices(9.0))], holder="s", origin="store",
                                device="cpu")
        self.assertEqual(out, {"promoted": False, "skip": "cannot_fit", "bytes": 0, "skip_bytes": 1024,
                               "host_bytes": 0})
        promotion = tier.stats()["promotion"]
        self.assertEqual((promotion["skips"]["cannot_fit"], promotion["skip_bytes"]["cannot_fit"]), (1, 1024))
        for item in [pinned, *others]:
            if item is not None:
                item.release()

    def test_promotion_skips_when_the_pinned_pool_is_full(self):
        tier = self.tier(retain="lru", promote="host", gpu=1, host=4)
        with tier._buffers._lock:
            tier._buffers.live_bytes = tier._buffers.limit_bytes  # every pool byte is live elsewhere
        out = tier.promote_host(node_ref="p", groups=[("p", 0, matrices(1.0))], holder="s", origin="store",
                                device="cpu")
        with tier._buffers._lock:
            tier._buffers.live_bytes = 0
        self.assertEqual(out["skip"], "pinned_pool_full")

    def test_promoted_store_reads_are_copied_into_pinned_pool_blocks(self):
        tier = RawChunkTier(gpu_budget_bytes=2 * ROWS * COLS * 4, host_budget_bytes=8 * 4096, pinned=True,
                            retain="lru", promote="host")  # host_raw_pinned (the default of every arm)
        self.addCleanup(tier.close)
        requested = []
        real_acquire = tier._buffers.acquire

        def acquire(sizes, pin, **kwargs):
            requested.append(pin)
            return real_acquire(sizes, False, **kwargs)  # the CPU harness cannot page-lock: record it, copy unpinned
        tier._buffers.acquire = acquire
        source = matrices(3.0)
        tier.promote_host(node_ref="p", groups=[("p", 0, source)], holder="s", origin="store", device="cuda:0")
        settle(tier)
        self.assertEqual(requested, [True])  # pinned blocks requested for a CUDA restore device
        [entry] = tier.entries(node_refs=["p"])
        self.assertEqual((entry.host_state, len(entry.host_blocks), entry.cuda), ("complete", 2, True))
        for key, tensor in source.items():  # copies in pool blocks, not the read buffers themselves
            stored = entry.groups[0].host[key]
            self.assertTrue(torch.equal(stored, tensor))
            self.assertNotEqual(stored.data_ptr(), tensor.data_ptr())
        self.assertEqual(tier.stats()["promotion"]["host_copied_bytes"], entry.nbytes)
        with tier._cond:  # a later GPU placement targets the restore device recorded in the entry
            self.assertEqual({meta[0].type for meta in entry.groups[0].meta.values()}, {"cuda"})

    def test_installed_promotions_complete_per_layer(self):
        tier = self.tier(retain="lru", promote="host", gpu=1, host=8)
        rows = {(MatrixKey(layer, target), "cond"): torch.randn(ROWS, 2, COLS // 2)
                for layer in range(2) for target in ("K", "V")}
        out = tier.promote_installed([{"node_ref": "i", "groups": [("i", 0, rows)]}], holder="s", origin="replay",
                                     device="cpu")
        self.assertEqual((out["promoted"], out["layer_events"]), (["i"], {}))  # CPU: copies already done
        settle(tier)
        lease = tier.acquire("i")
        self.assertEqual((lease.entry.form, lease.entry.origin), ("installed", "replay"))
        for key, tensor in rows.items():
            self.assertTrue(torch.equal(lease.matrix(0, key), tensor.reshape(ROWS, COLS)))
        lease.release()

    def test_budgets_are_never_exceeded_under_random_operations(self):
        pressured = 0
        for seed in range(6):
            with self.subTest(seed=seed):
                pressured += self._random_operations(random.Random(seed))
        self.assertGreater(pressured, 0)  # coverage: promotions did meet a full host budget

    @staticmethod
    def _pressure(tier) -> int:
        with tier._cond:
            return tier.stats_["evictions_host"] + tier.promotion_["skips"]["cannot_fit"]

    def _random_operations(self, rng) -> int:
        """Seeded random holder / publication / restore / promotion operations; returns how many promotions met a
        full host budget (an eviction or ``cannot_fit``)."""
        tier = self.tier(retain=rng.choice(("drop", "lru")), promote="host", gpu=2, host_bytes=3 * 1024)
        live, published, refs, holders = [], [], 0, ["s", "t", "s/tree", "s~d1-x"]
        pressured = 0

        def check():
            with tier._cond:
                self.assertLessEqual(tier._host_bytes + tier._host_reserved_spill, tier._host_budget)
                excess = tier._device_bytes - tier.gpu_budget()
                if excess > 0:  # only unpublished or pinned device copies may stand above the GPU budget (I1)
                    keep = sum(entry.nbytes for entry in tier._entries.values()
                               if entry.on_device and (entry.unpublished or entry.pinned))
                    self.assertLessEqual(excess, keep)
                for entry in tier._entries.values():
                    if entry.unpublished:
                        self.assertFalse(entry.dropped)

        for _ in range(150):
            action = rng.choice(("adopt", "publish", "retire", "hold", "release", "forget", "acquire", "promote"))
            if action == "adopt" and len(live) < 4:
                with tier._cond:
                    if tier.admission(2 * ROWS * COLS * 4) is not None:
                        continue
                live.append(adopt(tier, rng.choice(holders), f"n{refs}", float(refs)))
                refs += 1
            elif action == "publish" and live:
                entry = live.pop(rng.randrange(len(live)))
                publish(tier, entry, f"ref{entry.token}")
                published.append(f"ref{entry.token}")
            elif action == "retire" and live:
                retire(tier, live.pop(rng.randrange(len(live))), rng.choice(("failed", "cancelled")))
            elif action == "hold" and published:
                tier.hold(rng.choice(holders), rng.sample(published, rng.randint(1, len(published))),
                          exclusive=rng.random() < 0.5)
            elif action == "release":
                tier.release_holder(rng.choice(holders), drop=rng.random() < 0.3)
            elif action == "forget":
                tier.drop_unheld(rng.choice(("s", "t")))
            elif action == "acquire" and published:
                lease = tier.acquire(rng.choice(published))
                if lease is not None:
                    lease.release()
            elif action == "promote":
                before = self._pressure(tier)
                ref = f"p{rng.randrange(10)}"
                tier.promote_host(node_ref=ref, groups=[(ref, 0, matrices(1.0))], holder=rng.choice(holders),
                                  origin="store", device="cpu")
                published.append(ref)
                settle(tier)
                pressured += self._pressure(tier) > before
            settle(tier)
            check()
        for entry in live:
            retire(tier, entry, "cancelled")
        settle(tier)
        check()
        return pressured


class ConfigTests(unittest.TestCase):
    def config(self, **values):
        return ReelCacheRuntimeConfig.from_dict({"store_root": "x", **values})

    def test_compressed_arms_reject_promote_host(self):
        for representation, extra in (("svd_hevc", {}), ("svd", {"reuse": False}), ("hevc", {"reuse": False}),
                                      ("svd_hevc", {"rank_policy": "fixed_uniform", "uniform_rank": 32,
                                                    "reuse": False})):
            with self.subTest(representation=representation, **extra):
                with self.assertRaisesRegex(ConfigurationError, "prepared tier"):
                    self.config(representation=representation, raw_tier_promote="host", **extra)
                self.assertEqual(self.config(representation=representation, raw_tier_retain="lru",
                                             **extra).raw_tier_retain, "lru")
        for representation, extra in (("raw_offload", {}), ("latents_only", {"restore_mode": "recompute"})):
            config = self.config(representation=representation, raw_tier_promote="host", raw_tier_retain="lru",
                                 **extra)
            self.assertEqual((config.raw_tier_promote, config.raw_tier_retain), ("host", "lru"))
        for bad in ({"representation": "raw", "raw_tier_retain": "lru"},
                    {"representation": "raw_offload", "raw_tier_promote": "gpu"},
                    {"representation": "raw_offload", "raw_tier_retain": "keep"},
                    {"representation": "svd_hevc", "raw_tier": False, "raw_tier_retain": "lru"}):
            with self.subTest(bad=bad), self.assertRaises(ConfigurationError):
                self.config(**bad)

    def test_defaults_stay_out_of_the_records(self):
        data = self.config(representation="raw_offload").to_dict()
        self.assertNotIn("raw_tier_retain", data)
        self.assertNotIn("raw_tier_promote", data)
        data = self.config(representation="raw_offload", raw_tier_retain="lru", raw_tier_promote="host").to_dict()
        self.assertEqual((data["raw_tier_retain"], data["raw_tier_promote"]), ("lru", "host"))
        self.assertEqual(ReelCacheRuntimeConfig.from_dict(data).raw_tier_promote, "host")


class BridgeRetainCase(TreeTestCase):
    """Toy bridges (CPU) with the new options; ``raw`` is the bitwise reference."""

    tree = "head"

    def build(self, bridge, letters, *, session="s"):
        stage = TreeStage()
        for depth in range(1, len(letters) + 1):
            run_tree(bridge, self.request_for(handle(session, "generate" if depth == 1 else "resume"),
                                              letters[:depth]), stage=stage)
            bridge.checkpoint(session)
            if bridge.archive.raw_tier is not None:
                bridge.archive.raw_tier.flush()

    def restore_child(self, bridge, letters, *, session="s"):
        bridge.drop_resident()
        return run_tree(bridge, self.request_for(handle(session, "resume"), letters))

    def rewind_child(self, bridge, node, letters, *, session="s"):
        """Rewind to ``node`` (the parent of the last letter's chunk) and generate that chunk again."""
        bridge.drop_resident()
        return run_tree(bridge, self.request_for(handle(session, "rewind", checkpoint=node), letters))

    def assert_same(self, left, right):
        self.assertTrue(torch.equal(left[1], right[1]))
        assert_caches_bitwise(self, left[2], right[2])


class RawOffloadPromotionTests(BridgeRetainCase):
    representation = "raw_offload"

    def test_store_reads_are_promoted_and_hit_bitwise(self):
        raw = self.bridge("raw", representation="raw")
        bridge = self.bridge("offload", raw_tier_retain="lru", raw_tier_promote="host")
        self.build(raw, "ABCD")
        self.build(bridge, "ABCD")
        node_d = bridge.archive.head("s")
        bridge.control("raw_tier", {"action": "drop"})  # every raw copy gone: the store serves the next restore
        expected = self.restore_child(raw, "ABCDE")
        first = self.restore_child(bridge, "ABCDE")
        tiers = first[0]["restore"]["lease"]["tiers"]["chunks"]
        self.assertEqual(tiers, {"store": 4})
        promotion = first[0]["restore"]["promotion"]
        self.assertEqual((promotion["scope"], promotion["nodes"], promotion["skips"], promotion["skip_bytes"]),
                         ("store_read", 4, {}, {}))
        self.assertEqual(promotion["bytes"], first[0]["restore"]["lease"]["tiers"]["bytes"]["store"])  # full nodes
        self.assertGreaterEqual(promotion["host_bytes"], promotion["bytes"])
        self.assertTrue(all("origin" not in item for item in first[0]["restore"]["lease"]["tiers"]["per_chunk"]))
        self.assert_same(first, expected)
        bridge.archive.raw_tier.flush()
        stats = bridge.archive.raw_tier.stats()["promotion"]
        self.assertEqual((stats["entries"], stats["skips"]["error"]), (4, 0))
        bridge.checkpoint("s")
        again = self.rewind_child(bridge, node_d, "ABCDE")  # the promoted copies serve it now
        per_chunk = again[0]["restore"]["lease"]["tiers"]["per_chunk"]
        self.assertEqual({item["tier"] for item in per_chunk}, {"host_raw"})
        self.assertEqual({(item["origin"], item["form"], item["retained"]) for item in per_chunk},
                         {("store", "pre_rope", False)})  # promoted store reads, never unheld
        self.assertEqual(again[0]["restore"]["lease"]["tiers"]["retained_hits"], 0)
        self.assertGreaterEqual(bridge.archive.raw_tier.stats()["promotion"]["hits"], 4)
        self.assert_same(again, expected)
        policy = bridge.describe()["raw_tier_policy"]
        self.assertEqual(policy, {"retain": "lru", "promote": "host", "promotion_scope": "store_read",
                                  "promotion_fences": False})
        with mock.patch.object(bridge_module, "promotion_fences_needed", return_value=True):  # a rolling profile
            self.assertFalse(bridge.describe()["raw_tier_policy"]["promotion_fences"])  # store reads never fence

    def test_close_retains_and_delete_drops_everything_of_the_session(self):
        bridge = self.bridge("offload", raw_tier_retain="lru")
        self.build(bridge, "ABC")
        bridge.controller.close_session("s")  # close: the entries stay as unheld cache entries
        tier = bridge.archive.raw_tier
        tier.flush()
        stats = tier.stats()
        self.assertEqual(stats["entries"]["total"], 3)
        self.assertEqual(stats["retain"]["unheld"]["entries"]["gpu"] + stats["retain"]["unheld"]["entries"]["host"]
                         >= 3, True)
        self.build(bridge, "GH", session="o")  # another session's entries
        bridge.controller.close_session("s", delete=True)  # the whole-tree delete reclaims s's unheld entries
        tier.flush()
        self.assertEqual({entry.label[0] for entry in tier.entries()}, {"o"})
        self.assertEqual(tier.stats()["retain"]["dropped_on_delete"], 3)

    def test_a_reopened_session_hits_its_retained_entries(self):
        raw = self.bridge("raw", representation="raw")
        bridge = self.bridge("offload", raw_tier_retain="lru")
        self.build(raw, "ABC")
        self.build(bridge, "ABC")
        bridge.checkpoint("s")
        bridge.controller.close_session("s")  # closed: its entries stay, unheld
        bridge.archive.raw_tier.flush()
        self.assertEqual(bridge.archive.raw_tier.stats()["retain"]["retained"], 3)
        expected = self.restore_child(raw, "ABCD")
        got = self.restore_child(bridge, "ABCD")  # the reopen restores from them
        tiers = got[0]["restore"]["lease"]["tiers"]
        self.assertEqual(set(tiers["chunks"]) - {"gpu_raw", "host_raw"}, set())
        self.assertEqual({(item["origin"], item["form"], item["retained"]) for item in tiers["per_chunk"]},
                         {("captured", "pre_rope", True)})
        self.assertEqual(tiers["retained_hits"], 3)
        self.assertEqual(bridge.archive.raw_tier.stats()["retain"]["retained_hits"], 3)
        self.assert_same(got, expected)

    def test_store_read_promotions_record_a_promotion_copy_span(self):
        bridge = self.bridge("offload", raw_tier_retain="lru", raw_tier_promote="host", workload_telemetry=True)
        self.build(bridge, "ABC")
        bridge.control("raw_tier", {"action": "drop"})
        bridge.drop_resident()
        req = self.request_for(handle("s", "resume"), "ABCD")
        run_tree(bridge, req)
        bridge.archive.raw_tier.flush()  # the copies ran on the promotion worker
        spans = [span for span in bridge.workload_record(req.request_id)["spans"] if span["name"] == "promotion_copy"]
        self.assertEqual(len(spans), 3)  # one per promoted node, host-timed on the worker
        self.assertEqual({span["stream"] for span in spans}, {"host"})
        self.assertTrue(all(span["host_end_ns"] >= span["host_start_ns"] for span in spans))
        stats = bridge.archive.raw_tier.stats()["promotion"]
        self.assertEqual((stats["host_copied_entries"], stats["d2h_entries"]), (3, 0))

    def test_defaults_are_f6_records(self):
        bridge = self.bridge("offload")
        self.build(bridge, "AB")
        self.assertNotIn("raw_tier_policy", bridge.describe())
        stats = bridge.archive.raw_tier.stats()
        self.assertNotIn("retain", stats)
        self.assertNotIn("promotion", stats)
        summary = self.restore_child(bridge, "ABC")[0]
        self.assertNotIn("promotion", summary["restore"])
        self.assertTrue(all("origin" not in item for item in summary["restore"]["lease"]["tiers"]["per_chunk"]))


class RecomputePromotionTests(BridgeRetainCase):
    representation = "latents_only"

    def recompute(self, store="latents", **overrides):
        return self.bridge(store, restore_mode="recompute", **overrides)

    def test_replays_promote_installed_rows_which_hit_bitwise_without_a_fence_on_full_history(self):
        raw = self.bridge("raw", representation="raw")
        bridge = self.recompute(raw_tier_retain="lru", raw_tier_promote="host")
        self.build(raw, "ABCD")
        self.build(bridge, "ABCD")
        node_d = bridge.archive.head("s")
        bridge.control("raw_tier", {"action": "drop"})  # every raw copy gone: the next restore replays
        expected = self.restore_child(raw, "ABCDE")
        first = self.restore_child(bridge, "ABCDE")
        restore = first[0]["restore"]
        self.assertEqual((restore["mode"], restore["source"]), ("recompute", "replay"))
        promotion = restore["promotion"]
        self.assertEqual((promotion["scope"], promotion["nodes"], promotion["rolling_cache"],
                          promotion["fenced_layers"]), ("replay", 4, False, 0))
        caches = first[2]
        self.assertTrue(all(getattr(cache, "restore_pending", None) is None
                            for items in caches.values() for cache in items))  # no fence on full history
        self.assert_same(first, expected)
        bridge.archive.raw_tier.flush()
        bridge.checkpoint("s")
        again = self.rewind_child(bridge, node_d, "ABCDE")  # every attended range is a promoted installed entry now
        restore = again[0]["restore"]
        self.assertEqual((restore["mode"], restore["source"]), ("archive", "raw_tier"))
        self.assertEqual(restore["lease"]["raw_install"]["installed_form_spans"], 4)
        self.assertEqual({(item["origin"], item["form"], item["retained"])
                          for item in restore["lease"]["tiers"]["per_chunk"]}, {("replay", "installed", False)})
        self.assert_same(again, expected)
        self.assertGreaterEqual(bridge.archive.raw_tier.stats()["promotion"]["hits"], 4)

    def test_rolling_caches_are_fenced_layer_by_layer(self):
        """The toy profile is full-history; the rule is patched to treat it as rolling (its own test is below)."""
        raw = self.bridge("raw", representation="raw")
        bridge = self.recompute(raw_tier_retain="lru", raw_tier_promote="host")
        self.build(raw, "ABC")
        self.build(bridge, "ABC")
        bridge.control("raw_tier", {"action": "drop"})
        with mock.patch.object(bridge_module, "promotion_fences_needed", return_value=True):
            got = self.restore_child(bridge, "ABCD")
        promotion = got[0]["restore"]["promotion"]
        layers = len(got[2][next(iter(got[2]))])
        self.assertEqual((promotion["rolling_cache"], promotion["fenced_layers"]), (True, layers))
        self.assertEqual(promotion["fence_waits"], layers)  # the next forward went through every fence, once each
        self.assert_same(got, self.restore_child(raw, "ABCD"))  # the next forward ran through the fences

    def test_replay_promotions_record_a_promotion_copy_span(self):
        bridge = self.recompute(raw_tier_retain="lru", raw_tier_promote="host", workload_telemetry=True)
        self.build(bridge, "ABC")
        bridge.control("raw_tier", {"action": "drop"})
        bridge.drop_resident()
        req = self.request_for(handle("s", "resume"), "ABCD")
        summary = run_tree(bridge, req)[0]
        promotion = summary["restore"]["promotion"]
        self.assertEqual((promotion["nodes"], promotion["skips"], promotion["skip_bytes"]), (3, {}, {}))
        self.assertGreaterEqual(promotion["host_bytes"], promotion["bytes"])
        spans = [span for span in bridge.workload_record(req.request_id)["spans"] if span["name"] == "promotion_copy"]
        self.assertEqual(len(spans), 1)  # one span for the replay's copies (CPU: host-timed, already done)
        bridge.archive.raw_tier.flush()
        self.assertEqual(bridge.archive.raw_tier.stats()["promotion"]["d2h_entries"], 3)

    def test_a_promotion_fence_records_its_span_and_the_same_stall(self):
        telemetry = wt.WorkloadTelemetry("cpu")
        run = SimpleNamespace(workload_telemetry=telemetry, restore_report={"promotion": {"fence_waits": 0}})
        waited = []
        stream = SimpleNamespace(wait_event=waited.append)
        fence = bridge_module._PromotionFence("copy-event", None, run, 3)
        with mock.patch.object(bridge_module.torch.cuda, "current_stream", return_value=stream):
            fence.wait()
            fence.wait()  # a later access has nothing left to wait for
        record = telemetry.resolve()
        self.assertEqual(waited, ["copy-event"])
        self.assertEqual([span["name"] for span in record["spans"]], ["promotion_fence"])
        self.assertEqual([(stall["source"], stall["layer"]) for stall in record["stalls"]], [("promotion_fence", 3)])
        self.assertEqual(run.restore_report["promotion"]["fence_waits"], 1)

    def test_fence_rule_follows_the_profile_visibility(self):
        from reelcache.integrations.sglang.profiles import LONGLIVE2

        self.assertFalse(bridge_module.promotion_fences_needed(PROFILE))  # full history (CausVid, RAVEN)
        self.assertTrue(bridge_module.promotion_fences_needed(LONGLIVE2))  # global sink + pinned window


# -- W1.4.1: the review's asynchronous risks (deterministic barriers and fake CUDA events; no sleeps) ----------------
class FakeEvent:
    """A CUDA event stand-in: ``done`` decides ``query()``; ``synchronize()`` may raise (a failed device)."""

    def __init__(self, *args, done=False, fail=False, log=None, **kwargs):
        self.done, self.fail, self.log = done, fail, log

    def record(self, stream=None):
        pass

    def query(self):
        return self.done

    def synchronize(self):
        if self.log is not None:
            self.log.append("event.synchronize")
        if self.fail:
            raise RuntimeError("device failure")
        self.done = True

    def elapsed_time(self, other):
        return 1.0


class FakeStream:
    def __init__(self, log, *, fail=False):
        self.log, self.fail = log, fail

    def wait_event(self, event):
        self.log.append(("wait_event", event))

    def synchronize(self):
        self.log.append("stream.synchronize")
        if self.fail:
            raise RuntimeError("device failure")


def blocking_acquire(tier, started: threading.Event, release: threading.Event):
    """Hold the promotion worker inside its destination-block acquisition until ``release``."""
    real = tier._buffers.acquire

    def acquire(sizes, pin, **kwargs):
        if threading.current_thread().name == "reelcache-raw-promote":
            started.set()
            release.wait(30)
        return real(sizes, pin, **kwargs)
    tier._buffers.acquire = acquire


class PromotionRaceTests(unittest.TestCase):
    """W1.4.1 review: every test fails on the W1.4 tier (cc59cacc), see its negative-control note."""

    def tier(self, **kwargs):
        options = {"gpu_budget_bytes": 0, "host_budget_bytes": 8 * 1024, "pinned": False, "retain": "lru",
                   "promote": "host", **kwargs}
        tier = RawChunkTier(**options)
        self.addCleanup(tier.close)
        return tier

    def test_concurrent_pool_acquisitions_count_each_others_pending_bytes(self):
        """I2 (the reviewer's 1,536 / 1,024 / 2 x 512 case): an allocation in flight is part of every capacity check.
        W1.4 checked, unlocked, allocated, then counted: the second acquisition saw room and nothing recorded."""
        pool = raw_tier_module._HostPool(limit_bytes=1536)
        started, release = threading.Event(), threading.Event()
        real_block = raw_tier_module._HostBlock

        def block(size, pin):
            if threading.current_thread().name == "first":
                started.set()
                release.wait(30)
            return real_block(size, False)
        with mock.patch.object(raw_tier_module, "_HostBlock", side_effect=block):
            first = threading.Thread(target=pool.acquire, args=([1024], False), name="first")
            first.start()
            self.assertTrue(started.wait(30))
            self.assertFalse(pool.reserve(1024))  # 1,024 B in flight: no room for 1,024 B more
            second = pool.acquire([512, 512], False)  # 2,048 B with the first: over the limit, and recorded
            release.set()
            first.join(30)
        self.assertEqual(pool.stats["over_limit"], 1)
        self.assertEqual((pool.live_bytes, pool.pending_bytes, len(second)), (2048, 0, 2))

    def test_promotion_reserves_the_pool_before_evicting(self):
        """I2: a promotion the pinned pool cannot take evicts nothing (W1.4 evicted the LRU entry first)."""
        tier = self.tier(host_budget_bytes=3 * 1024)
        made = TierRetainTests.published(self, tier, "s", ["a", "b", "c"])  # the budget is full of held entries
        self.assertEqual(len(made), 3)
        with tier._buffers._lock:
            tier._buffers.pending_bytes = 1024  # an allocation in flight elsewhere
        out = tier.promote_host(node_ref="p", groups=[("p", 0, matrices(1.0))], holder="s", origin="store",
                                device="cpu")
        with tier._buffers._lock:
            tier._buffers.pending_bytes = 0
        self.assertEqual(out["skip"], "pinned_pool_full")
        self.assertEqual({entry.node_ref for entry in tier.entries()}, {"a", "b", "c"})  # no victim
        self.assertEqual(tier.stats()["retain"]["evictions"]["host_held"], 0)

    def test_a_promotion_in_flight_is_not_served_and_yields_to_a_commit(self):
        """Acquire before completion: a promotion still copying serves nothing, and a commit of the same node is not
        dropped for it (W1.4 kept the incomplete promotion and dropped the commit's complete copy)."""
        tier = self.tier()
        started, release = threading.Event(), threading.Event()
        blocking_acquire(tier, started, release)
        tier.promote_host(node_ref="p", groups=[("p", 0, matrices(1.0))], holder="s", origin="store", device="cpu")
        self.assertTrue(started.wait(30))
        self.assertIsNone(tier.acquire("p"))  # in flight: a miss, never a partial copy
        entry = adopt(tier, "t", "p", 5.0)  # the session regenerates the same node: its commit publishes now
        publish(tier, entry, "p")
        lease = tier.acquire("p")
        self.assertIsNotNone(lease)  # the commit's complete copy serves at once
        self.assertEqual(lease.entry.origin, "captured")
        lease.release()
        release.set()
        settle(tier)
        self.assertEqual([item.origin for item in tier.entries(node_refs=["p"])], ["captured"])
        self.assertEqual(tier.stats()["promotion"]["skips"]["superseded"], 1)

    def test_delete_ends_queued_promotions_and_waits_for_the_active_one(self):
        """C2: a whole-tree delete releases the owner's queued rows and returns only after the copy in flight ended;
        nothing of the session runs on, and the tier is idle (W1.4: the delete returned at once, work ran on)."""
        tier = self.tier()
        started, release = threading.Event(), threading.Event()
        blocking_acquire(tier, started, release)
        tier.promote_host(node_ref="a", groups=[("a", 0, matrices(1.0))], holder="s", origin="store", device="cpu")
        self.assertTrue(started.wait(30))  # "a" is active on the worker
        queued = matrices(2.0)
        refs = [weakref.ref(tensor) for tensor in queued.values()]
        tier.promote_host(node_ref="b", groups=[("b", 0, queued)], holder="s/tree", origin="store", device="cpu")
        queued = None
        self.assertFalse(tier._idle())
        returned = threading.Event()
        deleter = threading.Thread(target=lambda: (tier.drop_unheld("s"), returned.set()))
        deleter.start()
        self.assertFalse(returned.wait(0.5))  # still waiting for the active copy
        self.assertTrue(all(ref() is None for ref in refs))  # the queued rows were released already
        release.set()
        deleter.join(30)
        self.assertTrue(returned.is_set())
        with tier._cond:
            self.assertEqual((len(tier._promote_jobs), tier._promote_active, tier._transient), (0, 0, 0))
        self.assertEqual(tier.entries(), [])
        self.assertTrue(tier._idle())

    def _delete_tree(self, tier, owner: str) -> None:
        """The whole-tree delete's tier calls (``archive.delete`` / ``close_session(delete=True)``)."""
        tier.release_holder(owner, drop=True)
        tier.release_holder(owner + "/tree", drop=True)
        tier.drop_unheld(owner)

    def test_delete_completes_a_promotion_another_session_holds(self):
        """W1.4.3 (delta review, C2): s promotes a node, t holds it too (a fork), s's tree is deleted while the copy
        waits on the worker: the promotion is not cancelled -- t takes it over and its restore is a host-raw hit
        (W1.4.1 cancelled it on the job's original holder: t paid a store read or a replay)."""
        tier = self.tier()
        started, release = threading.Event(), threading.Event()
        blocking_acquire(tier, started, release)
        tier.promote_host(node_ref="a", groups=[("a", 0, matrices(1.0))], holder="s", origin="store", device="cpu")
        self.assertTrue(started.wait(30))  # the worker is inside "a"'s promotion
        tier.hold("t", ["a"], exclusive=False)  # t forked from s: it holds the promoting entry too
        deleter = threading.Thread(target=self._delete_tree, args=(tier, "s"))
        deleter.start()
        deleter.join(5)
        returned = not deleter.is_alive()  # the delete neither cancels nor waits for t's promotion
        release.set()
        deleter.join(30)
        self.assertTrue(returned)
        settle(tier)
        lease = tier.acquire("a")
        self.assertIsNotNone(lease)
        self.assertEqual((lease.tier, sorted(lease.entry.holders)), ("host_raw", ["t"]))
        lease.release()
        stats = tier.stats()["promotion"]
        self.assertEqual((stats["transferred"], stats["entries"], stats["skips"]["cancelled"]), (1, 1, 0))
        self._delete_tree(tier, "t")  # the new owner's delete ends it as usual
        self.assertEqual(tier.entries(), [])

    def _gated_device_promotion(self, tier, holder: str):
        """An issued device promotion whose completion event the test opens (``gate``)."""
        gate, log = threading.Event(), []

        class GateEvent(FakeEvent):
            def query(self):
                return gate.is_set()

            def synchronize(self):
                gate.wait(30)
                self.done = True

        tier._stream = lambda device: FakeStream(log)
        rows = {(MatrixKey(0, target), "cond"): torch.randn(ROWS, COLS) for target in ("K", "V")}
        cuda = raw_tier_module.torch.cuda
        with mock.patch.object(cuda, "device", lambda device: contextlib.nullcontext()), \
                mock.patch.object(cuda, "stream", lambda stream: contextlib.nullcontext()), \
                mock.patch.object(cuda, "Event", GateEvent):
            out = tier.promote_installed([{"node_ref": "i", "groups": [("i", 0, rows)]}], holder=holder,
                                         origin="replay", device="cuda:0")
        self.assertEqual(out["promoted"], ["i"])
        return gate

    def test_a_deleted_issued_device_promotion_counts_one_cancelled_skip(self):
        """W1.4.3 (delta review, moderate a): an issued device promotion ended by its tree's delete counts once in
        ``promotion.skips.cancelled`` (W1.4.1 counted it only as ``dropped_before_complete``)."""
        tier = self.tier()
        gate = self._gated_device_promotion(tier, "s")
        deleter = threading.Thread(target=self._delete_tree, args=(tier, "s"))
        deleter.start()
        deleter.join(1)
        self.assertTrue(deleter.is_alive())  # it waits for the device copies it ended
        gate.set()
        deleter.join(30)
        settle(tier)
        stats = tier.stats()["promotion"]
        self.assertEqual((stats["skips"]["cancelled"], stats["skip_bytes"]["cancelled"] > 0,
                          stats["dropped_before_complete"], stats["entries"]), (1, True, 0, 0))
        self.assertEqual((tier.entries(), tier._buffers.live_bytes), ([], 0))

    def test_delete_completes_a_device_promotion_another_session_holds(self):
        """W1.4.3 (C2): the same for an issued device promotion -- taken over by the surviving holder, completed."""
        tier = self.tier()
        gate = self._gated_device_promotion(tier, "s")
        tier.hold("t", ["i"], exclusive=False)
        deleter = threading.Thread(target=self._delete_tree, args=(tier, "s"))
        deleter.start()
        deleter.join(5)
        returned = not deleter.is_alive()
        gate.set()
        deleter.join(30)
        self.assertTrue(returned)
        settle(tier)
        lease = tier.acquire("i")
        self.assertIsNotNone(lease)
        self.assertEqual((lease.tier, lease.entry.form, sorted(lease.entry.holders)), ("host_raw", "installed", ["t"]))
        lease.release()
        stats = tier.stats()["promotion"]
        self.assertEqual((stats["transferred"], stats["d2h_entries"], stats["skips"]["cancelled"]), (1, 1, 0))

    def _successive_survivors(self, tier, ref: str, release) -> None:
        """Review I7: s promotes ``ref``; t and u hold it too. s CLOSES (no delete): the job's owner becomes t, a
        current holder (W1.4.3 v1 kept s -- stale). t's tree is deleted: u takes the job over. u's tree is deleted:
        nothing holds the entry any more -- the job ends (cancelled once) and the delete leaves nothing running."""
        tier.hold("t", [ref], exclusive=False)
        tier.hold("u", [ref], exclusive=False)
        tier.release_holder("s")  # a close: s detaches, retain="lru"
        with tier._cond:
            self.assertEqual([job.holder for job in tier._promote_jobs], ["t"])
        deleter = threading.Thread(target=self._delete_tree, args=(tier, "t"))
        deleter.start()
        deleter.join(5)
        self.assertFalse(deleter.is_alive())  # u holds on: the job is not cancelled, the delete does not wait
        with tier._cond:
            self.assertEqual([job.holder for job in tier._promote_jobs], ["u"])
        deleter = threading.Thread(target=self._delete_tree, args=(tier, "u"))
        deleter.start()
        deleter.join(1)
        release()  # the job's copy can end: the last holder's delete cancels it and waits for it
        deleter.join(30)
        self.assertFalse(deleter.is_alive())
        settle(tier)
        stats = tier.stats()["promotion"]
        self.assertEqual((stats["transferred"], stats["skips"]["cancelled"], stats["entries"]), (2, 1, 0))
        with tier._cond:
            self.assertEqual(list(tier._promote_jobs), [])
        self.assertEqual(tier.entries(), [])

    def test_a_host_promotion_follows_its_holders_through_successive_survivors(self):
        tier = self.tier()
        started, release = threading.Event(), threading.Event()
        blocking_acquire(tier, started, release)
        tier.promote_host(node_ref="a", groups=[("a", 0, matrices(1.0))], holder="s", origin="store", device="cpu")
        self.assertTrue(started.wait(30))
        self._successive_survivors(tier, "a", release.set)

    def test_a_device_promotion_follows_its_holders_through_successive_survivors(self):
        tier = self.tier()
        gate = self._gated_device_promotion(tier, "s")
        self._successive_survivors(tier, "i", gate.set)

    def test_a_cancelled_device_promotion_whose_copy_fails_counts_one_terminal_skip(self):
        """Review M10: a device promotion cancelled by its tree's delete, whose copy then fails, ends with ONE
        terminal skip reason (``cancelled``, counted when cancelled), its blocks quarantined -- W1.4.3 v1 also counted
        it as ``error``."""
        tier = self.tier()
        gate, failing = threading.Event(), {"fail": True}

        class FailingEvent(FakeEvent):
            def query(self):
                return gate.is_set()

            def synchronize(self):
                gate.wait(30)
                if failing["fail"]:
                    raise RuntimeError("device copy failed")
                self.done = True

        log = []
        tier._stream = lambda device: FakeStream(log)
        rows = {(MatrixKey(0, target), "cond"): torch.randn(ROWS, COLS) for target in ("K", "V")}
        cuda = raw_tier_module.torch.cuda
        with mock.patch.object(cuda, "device", lambda device: contextlib.nullcontext()), \
                mock.patch.object(cuda, "stream", lambda stream: contextlib.nullcontext()), \
                mock.patch.object(cuda, "Event", FailingEvent):
            out = tier.promote_installed([{"node_ref": "i", "groups": [("i", 0, rows)]}], holder="s",
                                         origin="replay", device="cuda:0")
        self.assertEqual(out["promoted"], ["i"])
        deleter = threading.Thread(target=self._delete_tree, args=(tier, "s"))
        deleter.start()
        deleter.join(1)
        gate.set()
        deleter.join(30)
        self.assertFalse(deleter.is_alive())
        settle(tier)
        stats = tier.stats()["promotion"]
        self.assertEqual((stats["skips"]["cancelled"], stats["skips"]["error"], stats["skip_bytes"]["error"]),
                         (1, 0, 0))
        self.assertEqual(tier.entries(), [])

    def test_transient_rows_stay_within_the_cap_and_unadmitted_rows_are_released(self):
        """C2: offered rows count until copied; a node whose rows would pass the cap is not admitted and its rows are
        released at once (W1.4 kept every row until its copy, with no bound)."""
        tier = self.tier()
        started, release = threading.Event(), threading.Event()
        blocking_acquire(tier, started, release)
        with mock.patch.object(raw_tier_module, "PROMOTION_TRANSIENT_FLOOR", 0):  # cap = 2 x the largest node
            kept = {}
            for index, ref in enumerate(("a", "b", "c")):
                rows = matrices(float(index))
                kept[ref] = [weakref.ref(tensor) for tensor in rows.values()]
                tier.promote_host(node_ref=ref, groups=[(ref, 0, rows)], holder="s", origin="store", device="cpu")
                rows = None
                with tier._cond:
                    self.assertLessEqual(tier._transient, tier.promotion_cap())
            self.assertTrue(started.wait(30))
            stats = tier.stats()["promotion"]
            self.assertEqual((stats["skips"]["transient_cap"], stats["transient_cap_bytes"]), (2, 2 * 1024))
            self.assertTrue(all(ref() is None for ref in kept["b"] + kept["c"]))  # not admitted: released
            self.assertLessEqual(stats["transient_peak_bytes"], stats["transient_cap_bytes"])
            release.set()
            settle(tier)
        self.assertEqual({entry.node_ref for entry in tier.entries()}, {"a"})
        self.assertEqual(tier.stats()["promotion"]["transient_bytes"], 0)

    def _installed(self, tier, log, *, sync_fails):
        """``promote_installed`` on a (fake) CUDA copy stream whose layer-1 copy raises after layer 0's copies were
        issued (a ``meta`` source cannot be copied)."""
        stream = FakeStream(log, fail=sync_fails)
        tier._stream = lambda device: stream
        real_release = tier._buffers.release
        tier._buffers.release = lambda blocks: (log.append("pool.release"), real_release(blocks))
        rows = {(MatrixKey(0, target), "cond"): torch.randn(ROWS, COLS) for target in ("K", "V")}
        rows.update({(MatrixKey(1, target), "cond"): torch.empty(ROWS, COLS, device="meta") for target in ("K", "V")})
        cuda = raw_tier_module.torch.cuda
        with mock.patch.object(cuda, "device", lambda device: contextlib.nullcontext()), \
                mock.patch.object(cuda, "stream", lambda stream: contextlib.nullcontext()), \
                mock.patch.object(cuda, "Event", FakeEvent):
            return tier.promote_installed([{"node_ref": "i", "groups": [("i", 0, rows)]}], holder="s", origin="replay",
                                          device="cuda:0")

    def test_a_copy_failure_after_issue_synchronizes_before_releasing_blocks(self):
        """I1: copies were issued when the failure struck: the copy stream is synchronized before any block returns
        to the pool (W1.4 released the blocks with the copies possibly still running)."""
        tier, log = self.tier(), []
        out = self._installed(tier, log, sync_fails=False)
        self.assertIn("error", out)
        self.assertTrue(out["copies_safe"])
        self.assertEqual([item for item in log if isinstance(item, str)], ["stream.synchronize", "pool.release"])
        self.assertEqual(tier.stats()["promotion"]["quarantined_blocks"], 0)

    def test_a_copy_failure_whose_wait_fails_quarantines_the_blocks(self):
        """I1: if the copies cannot be shown complete, their blocks are never reused (quarantined)."""
        tier, log = self.tier(), []
        out = self._installed(tier, log, sync_fails=True)
        self.assertFalse(out["copies_safe"])
        self.assertEqual([item for item in log if isinstance(item, str)], ["stream.synchronize"])  # no release
        stats = tier.stats()["promotion"]
        self.assertEqual(stats["quarantined_blocks"], 4)
        self.assertEqual(tier._buffers.live_bytes, stats["quarantined_bytes"])


class CacheReuseBoundaryTests(BridgeRetainCase):
    """W1.4.1 (C1): a full-history replay promotion's device-to-host copies may still read the stage caches when the
    request ends; the next writer -- whatever request it is -- must wait for them."""

    representation = "latents_only"

    def test_finish_leaves_boundary_fences_on_every_layer_of_a_full_history_cache(self):
        """W1.4 left nothing: the next request could overwrite slot 0 while the copies read it."""
        bridge = self.bridge("latents", restore_mode="recompute", raw_tier_retain="lru", raw_tier_promote="host")
        self.build(bridge, "ABC")
        bridge.control("raw_tier", {"action": "drop"})
        done = FakeEvent(done=False)
        real = RawChunkTier.promotion_issue_installed

        def issue(tier, *args, **kwargs):  # the CPU copies, reported as still running on a device
            out = real(tier, *args, **kwargs)
            out["copy_events"] = (FakeEvent(done=True), done)
            return out
        with mock.patch.object(RawChunkTier, "promotion_issue_installed", issue):
            summary, _, caches, _ = self.restore_child(bridge, "ABCD")
        self.assertFalse(summary["restore"]["promotion"]["rolling_cache"])  # full history: no in-request fence
        fences = [cache.restore_pending for items in caches.values() for cache in items]
        self.assertTrue(fences and all(isinstance(fence, bridge_module._PromotionFence) and fence.boundary
                                       and fence.event is done for fence in fences))
        self.assertEqual(bridge._promotion_pending, [done])
        log = []
        with mock.patch.object(bridge_module.torch.cuda, "current_stream", lambda *a: FakeStream(log)):
            caches["cond"][0].await_restored_history()  # a plain request's first touch of layer 0
        self.assertEqual(log, [("wait_event", done)])

    def test_the_next_session_request_waits_on_its_stream_and_records_the_stall(self):
        bridge = self.bridge("latents", restore_mode="recompute", raw_tier_retain="lru", raw_tier_promote="host")
        pending, complete = FakeEvent(done=False), FakeEvent(done=True)
        bridge._promotion_pending = [pending, complete]
        boundary = bridge_module._PromotionFence(pending, None, None, 0, boundary=True)
        target = SimpleNamespace(device=torch.device("cuda", 0), caches={"cond": [SimpleNamespace(
            restore_pending=boundary)]})
        telemetry, log = wt.WorkloadTelemetry("cpu"), []
        with mock.patch.object(bridge_module.torch.cuda, "current_stream", lambda *a: FakeStream(log)):
            bridge._await_promotion_copies(target, telemetry)
        self.assertEqual(log, [("wait_event", pending)])  # device-side; the completed event is dropped
        self.assertEqual(bridge._promotion_pending, [pending])
        self.assertIsNone(target.caches["cond"][0].restore_pending)  # the stream already waits: the fence is lifted
        self.assertEqual([stall["source"] for stall in telemetry.resolve()["stalls"]], ["promotion_fence"])

    def test_a_failed_request_waits_for_its_copies_on_the_host(self):
        log = []
        run = SimpleNamespace(_promotion_done=[FakeEvent(done=False, log=log)], target=None, _deferred_promotion=None,
                              _lift_triggers=lambda: None)
        bridge_module._BridgeRun._promotion_boundary(run, RuntimeError("request failed"))
        self.assertEqual((log, run._promotion_done), (["event.synchronize"], []))  # abort: a host wait



class DeferredReplayPromotionTests(BridgeRetainCase):
    """W1.4.1 (N1): a recompute promotion is PREPARED at the replay's end and ISSUED by the first access after it (the
    first new forward, after the stage's pre-generation device sync), so its copies overlap generation; a copy
    still running at a device-wide sync is waited for explicitly and recorded (``promotion_wait``)."""

    representation = "latents_only"

    def test_the_copies_are_issued_by_the_first_access_after_the_replay(self):
        """W1.4 issued them inside end_replay, right before the stage's device-wide sync absorbed them as prep."""
        raw = self.bridge("raw", representation="raw")
        bridge = self.bridge("latents", restore_mode="recompute", raw_tier_retain="lru", raw_tier_promote="host")
        self.build(raw, "ABC")
        self.build(bridge, "ABC")
        bridge.control("raw_tier", {"action": "drop"})
        log, seen = [], {}
        real_end, real_issue = bridge_module._BridgeRun.end_replay, RawChunkTier.promotion_issue_installed

        def end_replay(run, *args, **kwargs):
            report = real_end(run, *args, **kwargs)
            log.append("end_replay")
            seen["pending"] = {type(cache.restore_pending).__name__ for items in run.target.caches.values()
                               for cache in items}
            return report

        def issue(tier, *args, **kwargs):
            log.append("issue")
            return real_issue(tier, *args, **kwargs)
        with mock.patch.object(bridge_module._BridgeRun, "end_replay", end_replay), \
                mock.patch.object(RawChunkTier, "promotion_issue_installed", issue):
            got = self.restore_child(bridge, "ABCD")
        self.assertEqual(log, ["end_replay", "issue"])
        self.assertEqual(seen["pending"], {"_PromotionTrigger"})  # every layer waits for its first access
        promotion = got[0]["restore"]["promotion"]
        self.assertEqual((promotion["issued"], promotion["nodes"]), (True, 3))
        self.assert_same(got, self.restore_child(raw, "ABCD"))
        self.assertTrue(all(not isinstance(cache.restore_pending, bridge_module._PromotionTrigger)
                            for items in got[2].values() for cache in items))  # none outlives the request

    def test_a_copy_still_running_at_a_device_sync_is_waited_for_and_recorded(self):
        log = []
        telemetry = wt.WorkloadTelemetry("cpu")
        run = SimpleNamespace(_promotion_done=[FakeEvent(done=False, log=log)], workload_telemetry=telemetry,
                              target=SimpleNamespace(device=torch.device("cpu")),
                              restore_report={"promotion": {"exposed_wait_seconds": 0.0}})
        bridge_module._BridgeRun.before_device_sync(run, "chunk_end")
        self.assertEqual(log, ["event.synchronize"])
        record = telemetry.resolve()
        self.assertEqual([(stall["source"], stall["kind"]) for stall in record["stalls"]],
                         [("promotion_wait", "host_wait")])
        self.assertEqual([span["name"] for span in record["spans"]], ["promotion_wait"])
        promotion = run.restore_report["promotion"]
        self.assertEqual([item["where"] for item in promotion["exposed_waits"]], ["chunk_end"])
        self.assertGreaterEqual(promotion["exposed_wait_seconds"], 0.0)
        bridge_module._BridgeRun.before_device_sync(run, "chunk_start")  # completed now: nothing more
        self.assertEqual((log, len(promotion["exposed_waits"])), (["event.synchronize"], 1))

    def test_a_prepared_promotion_never_issued_gives_its_blocks_back_on_failure(self):
        tier = RawChunkTier(gpu_budget_bytes=0, host_budget_bytes=8 * 1024, pinned=False, retain="lru",
                            promote="host")
        self.addCleanup(tier.close)
        rows = {(MatrixKey(0, target), "cond"): torch.randn(ROWS, COLS) for target in ("K", "V")}
        prepared = tier.promotion_prepare_installed([{"node_ref": "i", "groups": [("i", 0, rows)]}], holder="s",
                                                    origin="replay", device="cpu")
        self.assertGreater(tier._buffers.live_bytes, 0)
        cache = SimpleNamespace(restore_pending=None)
        run = SimpleNamespace(_deferred_promotion=prepared, _promotion_done=[], target=SimpleNamespace(
            caches={"cond": [cache]}), bridge=SimpleNamespace(archive=SimpleNamespace(raw_tier=tier)))
        for name in ("_drop_deferred_promotion", "_lift_triggers", "_issue_deferred_promotion"):
            setattr(run, name, types.MethodType(getattr(bridge_module._BridgeRun, name), run))
        cache.restore_pending = bridge_module._PromotionTrigger(run, None, "cond", 0)
        bridge_module._BridgeRun._promotion_boundary(run, RuntimeError("request failed"))
        self.assertIsNone(cache.restore_pending)
        self.assertEqual((tier._buffers.live_bytes, tier.entries()), (0, []))
        self.assertEqual(tier.stats()["promotion"]["skips"]["cancelled"], 1)

    def test_the_profile_stage_waits_for_promotion_copies_before_every_device_sync(self):
        """Static: in the ReelCache profile stage, every device-wide synchronize outside the replay is preceded by the
        promotion hook, and the recache's direct K/V zeroing by an await of restored history."""
        source = pathlib.Path(__file__).resolve().parent.parent / "reelcache_sglang_models/causal_wan/sglang_stages.py"
        lines = source.read_text().splitlines()  # read, not imported: the CPU harness has no SGLang
        syncs = [index for index, line in enumerate(lines)
                 if "torch.cuda.synchronize() if torch.cuda.is_available() else None" in line]
        self.assertEqual(len(syncs), 4)
        for index in syncs:
            self.assertIn("_before_device_sync(kv_run, ", lines[index - 1], lines[index])
        zero = next(index for index, line in enumerate(lines) if "c.k.zero_()" in line)
        self.assertIn("c.await_restored_history()", lines[zero - 1])


class UnpublishedAtDeleteTests(unittest.TestCase):
    """W1.4.1 (verifier minor): under ``retain="lru"`` an entry still unpublished at its tree's delete must not
    survive as an unheld entry once it publishes."""

    def test_a_whole_tree_delete_drops_an_unpublished_entry_when_it_publishes(self):
        for close_first in (False, True):
            with self.subTest(close_first=close_first):
                tier = RawChunkTier(gpu_budget_bytes=4 * 2 * ROWS * COLS * 4, host_budget_bytes=8 * 1024,
                                    pinned=False, retain="lru")
                self.addCleanup(tier.close)
                entry = adopt(tier, "s", "u", 1.0)  # its commit is still archiving
                if close_first:
                    tier.release_holder("s")  # a plain close first, then the delete
                    tier.drop_unheld("s")
                else:
                    tier.release_holder("s", drop=True)  # the whole-tree delete releases it
                    tier.drop_unheld("s")
                publish(tier, entry, "ref-u")
                settle(tier)
                self.assertEqual(tier.entries(), [])
                self.assertEqual(tier.stats()["retain"]["unheld"]["entries"], {"gpu": 0, "host": 0})


class GpuOnlyTierTests(unittest.TestCase):
    """W1.4.1 ("nossd"): ``gpu_only`` -- the recompute arm with ``host_raw_budget_bytes=0`` is GPU raw only."""

    def test_a_zero_host_budget_serves_gpu_raw_and_deletes_on_demotion(self):
        entry_bytes = 2 * ROWS * COLS * 4
        tier = RawChunkTier(gpu_budget_bytes=2 * entry_bytes, host_budget_bytes=0, pinned=False, retain="lru",
                            gpu_only=True)
        self.addCleanup(tier.close)
        with tier._cond:
            self.assertTrue(tier.usable(entry_bytes))  # no degradation: the host is not needed
        entries = []
        for index in range(3):  # one beyond the GPU budget: the least recently used published entry is deleted
            entries.append(adopt(tier, "s", f"n{index}", float(index)))
            publish(tier, entries[-1], f"ref-{index}")
        settle(tier)
        stats = tier.stats()
        self.assertEqual((stats["enabled"], stats["gpu_only"], stats["degraded"], stats["evictions_gpu"],
                          stats["host_bytes"], stats["spills"]), (True, True, 0, 1, 0, 0))
        self.assertEqual(sorted(entry.node_ref for entry in tier.entries()), ["ref-1", "ref-2"])
        lease = tier.acquire("ref-2")
        self.assertEqual(lease.tier, "gpu_raw")
        lease.release()
        # A memory-critical section's evacuation (the pressure response) deletes instead of demoting
        result = tier.demote(tier.entries(), place=False)
        self.assertEqual((result["entries"], result["failed"]), (2, 0))
        stats = tier.stats()
        self.assertEqual((stats["gpu_bytes"], stats["host_bytes"], stats["host_peak_bytes"], stats["evictions_gpu"]),
                         (0, 0, 0, 3))
        self.assertNotIn("demotion_blocked", [event["event"] for event in stats["events"]])

    def test_gpu_only_needs_a_zero_host_budget_and_the_default_still_degrades(self):
        with self.assertRaises(ConfigurationError):
            RawChunkTier(host_budget_bytes=1 << 20, gpu_only=True)
        tier = RawChunkTier(gpu_budget_bytes=1 << 20, host_budget_bytes=0, pinned=False)
        self.addCleanup(tier.close)
        with tier._cond:
            self.assertFalse(tier.usable(2 * ROWS * COLS * 4))  # W1.3.2 behaviour without the flag
        self.assertEqual((tier.stats()["degraded"], "gpu_only" in tier.stats()), (1, False))

    def test_the_recompute_arm_sets_it_only_for_a_zero_host_budget(self):
        from reelcache.integrations.sglang.archive import raw_tier_for

        base = {"store_root": "x", "representation": "latents_only", "restore_mode": "recompute"}
        for host, expected in ((0, True), (1 << 20, False), (None, False)):
            tier = raw_tier_for(ReelCacheRuntimeConfig.from_dict({**base, "host_raw_budget_bytes": host}))
            self.addCleanup(tier.close)
            self.assertIs(tier.gpu_only, expected, host)
        offload = raw_tier_for(ReelCacheRuntimeConfig.from_dict({"store_root": "x", "representation": "raw_offload",
                                                              "host_raw_budget_bytes": 0}))
        self.addCleanup(offload.close)
        self.assertFalse(offload.gpu_only)  # the store-backed raw_offload keeps W1.3.2's tier


if __name__ == "__main__":
    unittest.main()
