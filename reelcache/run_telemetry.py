"""Run telemetry kept out of content-addressed manifests.

A manifest's object id is the SHA-256 of its canonical JSON, so anything that
varies between runs of the same content (wall-clock timings, scheduling-dependent
counts) must not be hashed: identical commits must produce identical node,
record and packet ids. Writers split such fields off with
:func:`split_run_telemetry`, hash only the content, and attach the telemetry to
the stored object as an unhashed sidecar (:meth:`reelcache.store.ObjectStore.put_telemetry`).
Readers recombine both with :func:`merge_run_telemetry`; manifests written
before the split simply carry their telemetry inline and read unchanged.
"""

from __future__ import annotations

from typing import Any

#: Keys that are run telemetry wherever they occur in a split value (besides any ``seconds_*``/``*_seconds``
#: key). ``speculative_observations`` counts how many ancestor observations an archive pipeline had not yet
#: published when a commit planned; it depends on scheduling, while the predictions it feeds do not.
#: ``plan_memory_bytes`` / ``plan_memory`` / ``finalize_memory``: the working-set bound and the planning and
#: finalization groups it produced (execution configuration; stored content is identical with or without it).
#: ``sub_batches``: how a plan's eigensolver and sketch calls were split (follows the budget, device memory and the
#: chunk's fit set, not the content).
#: F4 (dev22j5): ``pinned_sequence``, ``predictions``, ``predicted_rank``, ``prediction_revision`` and
#: ``predictor_revision`` are predictor provenance. A bridge's sessions share one registry that learns in
#: publication order, so when two sessions archive concurrently these depend on scheduling (Step 19 sus1/sus2).
#: A record's content is its decision, rank, energies and bytes; a different prediction that changes any of those
#: still changes the record. ``work`` / ``counters``: planner work statistics (verification steps follow the
#: predicted start). ``encode_threads`` / ``hevc_encode_threads`` / ``x265_thread_params``: encoder threading,
#: which never changes the stream bytes (``x265_threads`` may be derived from the CPUs).
RUN_TELEMETRY_KEYS = frozenset({"seconds", "stages", "stage_seconds", "speculative_observations",
                                "pinned_sequence", "predictions", "predicted_rank", "prediction_revision",
                                "predictor_revision", "work", "counters", "encode_threads", "hevc_encode_threads",
                                "x265_thread_params",
                                "plan_memory_bytes", "plan_memory", "sub_batches", "finalize_memory"})


def is_run_telemetry_key(name: Any) -> bool:
    return isinstance(name, str) and (name in RUN_TELEMETRY_KEYS or name.startswith("seconds_")
                                      or name.endswith("_seconds"))


def split_run_telemetry(value: Any) -> tuple[Any, Any]:
    """``(content, telemetry)``: ``value`` without run telemetry, and the removed parts (or None).

    Dicts drop telemetry keys at any depth; the telemetry mirrors the nesting
    (lists keep their positions, with None for items without telemetry), so
    ``merge_run_telemetry(*split_run_telemetry(value)) == value``.
    """
    if isinstance(value, dict):
        content, telemetry = {}, {}
        for key, item in value.items():
            if is_run_telemetry_key(key):
                telemetry[key] = item
                continue
            content[key], nested = split_run_telemetry(item)
            if nested is not None:
                telemetry[key] = nested
        return content, (telemetry or None)
    if isinstance(value, (list, tuple)):
        parts = [split_run_telemetry(item) for item in value]
        nested = [telemetry for _, telemetry in parts]
        return [content for content, _ in parts], (nested if any(item is not None for item in nested) else None)
    return value, None


def merge_run_telemetry(content: Any, telemetry: Any) -> Any:
    """Recombine content with its split telemetry (inverse of :func:`split_run_telemetry`)."""
    if telemetry is None:
        return content
    if isinstance(content, dict) and isinstance(telemetry, dict):
        merged = dict(content)
        for key, item in telemetry.items():
            merged[key] = merge_run_telemetry(content[key], item) if key in content else item
        return merged
    if isinstance(content, list) and isinstance(telemetry, list) and len(content) == len(telemetry):
        return [merge_run_telemetry(item, nested) for item, nested in zip(content, telemetry)]
    return content
