"""Validation and clean-boundary schedules for explicitly synthetic video sessions."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import re
from typing import Any, Mapping, Sequence

from .errors import ConfigurationError
from .workloads import validate_workload


HISTORY_LENGTHS = (4, 8, 12, 16, 20)
CHANGE_TYPES = frozenset(("action", "camera", "object", "environment", "goal"))


def normalized_prompt(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def validate_session(session: Mapping[str, Any]) -> None:
    if not isinstance(session, Mapping):
        raise ConfigurationError("session must be an object")
    required = {"session_id", "title", "category", "prompts", "forks"}
    if not required <= session.keys():
        raise ConfigurationError(f"session lacks required fields: {sorted(required - session.keys())}")
    if not isinstance(session["session_id"], str) or re.fullmatch(r"s[0-9]{4}", session["session_id"]) is None:
        raise ConfigurationError("session_id must have the form s0000")
    for name in ("title", "category"):
        if not isinstance(session[name], str) or not session[name].strip():
            raise ConfigurationError(f"{name} must be a nonempty string")
    prompts = session["prompts"]
    if not isinstance(prompts, list) or len(prompts) < 20:
        raise ConfigurationError("each session requires at least 20 main-path prompts")
    source_session = int(session["session_id"][1:]) < 100
    for index, prompt in enumerate(prompts):
        inherited = source_session and index < 6
        if not isinstance(prompt, str):
            raise ConfigurationError("main-path prompts must be strings")
        words = len(prompt.split())
        if (inherited and words < 12) or (not inherited and not 18 <= words <= 35):
            raise ConfigurationError(
                "new main-path prompts require 18-35 words; inherited six-prompt prefixes require at least 12"
            )
    forks = session["forks"]
    if not isinstance(forks, list) or len(forks) != len(HISTORY_LENGTHS):
        raise ConfigurationError("each session requires one fork at each declared history length")
    lengths = []
    for fork in forks:
        if not isinstance(fork, Mapping) or set(fork) != {"history_prompt_count", "change_type", "prompts"}:
            raise ConfigurationError("fork fields must be history_prompt_count, change_type, prompts")
        history = fork["history_prompt_count"]
        if type(history) is not int or not 0 < history <= len(prompts):
            raise ConfigurationError("fork history must name an existing main-path prefix")
        lengths.append(history)
        if fork["change_type"] not in CHANGE_TYPES:
            raise ConfigurationError("unknown fork change_type")
        branch = fork["prompts"]
        if not isinstance(branch, list) or len(branch) < 2:
            raise ConfigurationError("each fork requires at least two additional continuation prompts")
        if any(not isinstance(p, str) or not 18 <= len(p.split()) <= 35 for p in branch):
            raise ConfigurationError("fork prompts require self-contained strings of 18-35 words")
        all_main = {normalized_prompt(p) for p in prompts}
        if any(normalized_prompt(p) in all_main for p in branch):
            raise ConfigurationError("fork continuations must be new prompts, not copies of the main path")
        if len({normalized_prompt(p) for p in branch}) != len(branch):
            raise ConfigurationError("fork continuation repeats a prompt")
    if tuple(lengths) != HISTORY_LENGTHS:
        raise ConfigurationError(f"fork histories must be ordered exactly as {HISTORY_LENGTHS}")


def validate_bundle(
    sessions: Sequence[Mapping[str, Any]], source_narratives: Sequence[Mapping[str, Any]], *,
    expected_sessions: int = 500,
) -> dict[str, Any]:
    if len(sessions) != expected_sessions:
        raise ConfigurationError(f"expected {expected_sessions} sessions, got {len(sessions)}")
    ids, titles, categories, changes = set(), set(), Counter(), Counter()
    new_seen, main_count, fork_count, fork_prompt_count = {}, 0, 0, 0
    source_seen = {
        normalized_prompt(p): (f"source-{i:03d}", f"main:{j + 1}")
        for i, source in enumerate(source_narratives) for j, p in enumerate(source["prompts"])
    }
    duplicates = []
    for index, session in enumerate(sessions):
        validate_session(session)
        expected_id = f"s{index:04d}"
        if session["session_id"] != expected_id or expected_id in ids:
            raise ConfigurationError(f"session order/identity mismatch at {index}: expected {expected_id}")
        ids.add(expected_id)
        title = normalized_prompt(session["title"])
        if title in titles:
            raise ConfigurationError(f"duplicate session title: {session['title']}")
        titles.add(title)
        categories[session["category"]] += 1
        prompts = session["prompts"]
        prefix = 0
        if index < len(source_narratives):
            original = source_narratives[index]["prompts"]
            if prompts[:len(original)] != original:
                raise ConfigurationError(f"{expected_id} changed its original narrative prefix")
            prefix = len(original)
        main_count += len(prompts)
        texts = [(f"main:{i + prefix + 1}", p) for i, p in enumerate(prompts[prefix:])]
        for fork in session["forks"]:
            history = fork["history_prompt_count"]
            changes[fork["change_type"]] += 1
            fork_count += 1
            fork_prompt_count += len(fork["prompts"])
            texts += [(f"fork:{history}:{i + 1}", p) for i, p in enumerate(fork["prompts"])]
        for location, text in texts:
            token = normalized_prompt(text)
            if token in source_seen:
                duplicates.append((expected_id, location, *source_seen[token]))
            elif token in new_seen:
                duplicates.append((expected_id, location, *new_seen[token]))
            else:
                new_seen[token] = (expected_id, location)
    if duplicates:
        raise ConfigurationError(f"duplicate generated prompts ({len(duplicates)}): {duplicates[:12]}")
    return {
        "sessions": len(sessions), "main_prompts": main_count, "forks": fork_count,
        "fork_prompts": fork_prompt_count, "total_prompts": main_count + fork_prompt_count,
        "min_main_prompts": min(len(s["prompts"]) for s in sessions),
        "history_prompt_counts": list(HISTORY_LENGTHS),
        "forks_per_history": {str(h): len(sessions) for h in HISTORY_LENGTHS},
        "preserved_source_sessions": min(len(source_narratives), len(sessions)),
        "generated_prompt_duplicates": 0, "categories": dict(categories), "change_types": dict(changes),
        "content_coherence": "author self-review; structural checks do not establish rendered video quality",
        "native_model_executed": False,
    }


def compile_session(
    session: Mapping[str, Any], *, chunks_per_prompt: int = 1, seed: int = 0,
) -> dict[str, Any]:
    """Build a valid fork DAG without mislabeling prompt history as native token history.

    The model-free default maps one prompt segment to one control chunk.
    Real model replay must supply an audited chunks_per_prompt mapping; no
    frames, tokens or seconds are inferred from these segment counts.
    """
    validate_session(session)
    if type(chunks_per_prompt) is not int or chunks_per_prompt < 1:
        raise ConfigurationError("chunks_per_prompt must be a positive integer")
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ConfigurationError("seed must be an integer in [0, 2**63)")
    prompts = list(session["prompts"])
    fork_indices = {}
    for fork in session["forks"]:
        first = len(prompts)
        prompts.extend(fork["prompts"])
        fork_indices[fork["history_prompt_count"]] = (fork, list(range(first, len(prompts))))
    events = []
    cursor, checkpoints = {"main": 0}, {"main": "main:0"}
    source = f"llm-authored:{session['session_id']}"

    def emit(operation, branch, prompt, **fields):
        event = {
            "event_id": len(events), "operation": operation, "branch_id": branch,
            "source_id": source, "source_event_index": len(events), "source_timestamp": None,
            "source_action": operation, "source_state_id": None, "mapping_rule": "declared_synthetic_fork_schedule",
            "provenance_class": "controlled_synthetic", "target_clean_chunk": cursor[branch],
            "checkpoint_id": checkpoints[branch], "prompt_index": prompt, "seed": seed, **fields,
        }
        events.append(event)
        return event

    def generate(branch, prompt, branch_seed):
        for _ in range(chunks_per_prompt):
            output = f"{branch}:generated-{len(events)}"
            emit("generate", branch, prompt, seed=branch_seed, output_checkpoint=output)
            cursor[branch] += 1
            checkpoints[branch] = output

    for index in range(len(session["prompts"])):
        generate("main", index, seed)
        history = index + 1
        if history not in fork_indices:
            continue
        fork, indices = fork_indices[history]
        branch = f"fork-h{history:02d}"
        parent = checkpoints["main"]
        cursor[branch] = cursor["main"]
        checkpoints[branch] = f"{branch}:fork"
        branch_seed = int.from_bytes(hashlib.sha256(
            f"{session['session_id']}:{history}:{seed}".encode()).digest()[:8], "little") % (2**63)
        emit("fork", branch, indices[0], parent_branch_id="main", target_checkpoint=parent,
             history_prompt_count=history, seed=branch_seed, seed_policy="new_seed",
             content_policy="authored_alternative", change_type=fork["change_type"])
        for prompt in indices:
            generate(branch, prompt, branch_seed)
        emit("resume", "main", min(index + 1, len(session["prompts"]) - 1), target_checkpoint=parent)
    workload = {
        "schema": "reelcache.trace_mapped_video_workload.v1", "episode_id": session["session_id"],
        "source_id": source, "source_provenance": "llm_authored_synthetic",
        "narrative_id": session["session_id"], "prompts": prompts, "policy": {"seed": seed},
        "history_mapping": {"unit": "completed_prompt_segments", "chunks_per_prompt": chunks_per_prompt,
                            "native_geometry_qualified": False},
        "events": events, "source_audit": [],
        "claims": {"recorded_video_user_trace": False, "recorded_agent_trace": False,
                   "native_model_executed": False, "video_quality_measured": False},
        "summary": {"operations": dict(Counter(e["operation"] for e in events)),
                    "branches": 1 + len(session["forks"]), "source_events": 0},
    }
    validate_workload(workload)
    return workload


def read_sessions(path):
    rows = []
    with open(path) as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                session = json.loads(line)
            except ValueError as exc:
                raise ConfigurationError(f"invalid session JSONL at {path}:{line_number}") from exc
            validate_session(session)
            rows.append(session)
    return rows
