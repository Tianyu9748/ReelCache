"""Compile recorded browser events into auditable clean-boundary video workloads.

This module does not load models or claim browser operations are video-user
operations. Chunk progress, prompt timing, and browser-to-video mappings are
explicit policies; original event order and source references are retained.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import math
from typing import Any, Mapping, Sequence

from .errors import ConfigurationError
from .types import json_copy


@dataclass(frozen=True)
class WorkloadPolicy:
    chunk_seconds: float
    prompt_seconds: float = 10.0
    chunks_per_action: int = 1
    pause_gap_seconds: float = 30.0
    seed: int = 0
    branch_content: str = "new_seed"
    generation_intents: tuple[str, ...] = ("load", "click", "scroll", "paste")
    infer_url_rewind: bool = False

    def __post_init__(self):
        for name in ("chunk_seconds", "prompt_seconds"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ConfigurationError(f"{name} must be positive finite seconds")
        if (isinstance(self.pause_gap_seconds, bool) or not isinstance(self.pause_gap_seconds, (int, float))
                or not math.isfinite(self.pause_gap_seconds) or self.pause_gap_seconds < 0):
            raise ConfigurationError("pause_gap_seconds must be nonnegative finite seconds")
        if type(self.chunks_per_action) is not int or self.chunks_per_action < 1:
            raise ConfigurationError("chunks_per_action must be a positive integer")
        if type(self.seed) is not int or not 0 <= self.seed < 2**63:
            raise ConfigurationError("seed must be an integer in [0, 2**63)")
        if self.branch_content not in ("new_seed", "next_prompt"):
            raise ConfigurationError("branch_content must be new_seed or next_prompt")
        if type(self.infer_url_rewind) is not bool:
            raise ConfigurationError("infer_url_rewind must be boolean")
        if not self.generation_intents or any(not isinstance(v, str) or not v for v in self.generation_intents):
            raise ConfigurationError("generation_intents must be nonempty strings")


def read_narratives(text: str) -> list[dict[str, Any]]:
    import json

    rows, seen = [], set()
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError as exc:
            raise ConfigurationError("invalid narrative JSONL") from exc
        prompts = row.get("prompts") if isinstance(row, dict) else None
        if not isinstance(prompts, list) or not prompts or any(not isinstance(p, str) or not p.strip() for p in prompts):
            raise ConfigurationError("each narrative must contain nonempty prompts")
        canonical = json_copy(prompts)
        digest = hashlib.sha256(json.dumps(canonical, ensure_ascii=True).encode()).hexdigest()
        if digest in seen:
            raise ConfigurationError("duplicate narrative; split at whole deduplicated narrative groups")
        seen.add(digest)
        rows.append({"narrative_id": digest, "prompts": canonical})
    if not rows:
        raise ConfigurationError("narrative input is empty")
    return rows


def _tab(value):
    return str(value) if isinstance(value, (str, int)) and not isinstance(value, bool) and str(value) not in ("", "-1") else None


def _state_id(value):
    return hashlib.sha256(str(value).encode()).hexdigest()[:24] if value else None


def browser_observations(replay: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Extract timing and navigation metadata, not user dialogue or page content."""
    if not isinstance(replay, Mapping) or not isinstance(replay.get("data"), list):
        raise ConfigurationError("WebLINX replay must contain a data list")
    observations = []
    for index, row in enumerate(replay["data"]):
        if not isinstance(row, Mapping):
            raise ConfigurationError(f"invalid replay row {index}")
        stamp = row.get("timestamp")
        if stamp is not None and (isinstance(stamp, bool) or not isinstance(stamp, (int, float)) or not math.isfinite(stamp)):
            raise ConfigurationError(f"invalid timestamp at replay row {index}")
        action = row.get("action", {})
        if not isinstance(action, Mapping):
            raise ConfigurationError(f"invalid action at replay row {index}")
        args = action.get("arguments", {})
        if not isinstance(args, Mapping):
            raise ConfigurationError(f"invalid action arguments at replay row {index}")
        properties, metadata = args.get("properties", {}), args.get("metadata", {})
        if not isinstance(properties, Mapping) or not isinstance(metadata, Mapping):
            raise ConfigurationError(f"invalid browser metadata at replay row {index}")
        qualifiers = properties.get("transitionQualifiers", [])
        if not isinstance(qualifiers, list):
            raise ConfigurationError(f"invalid navigation qualifiers at replay row {index}")
        observations.append({
            "source_event_index": index, "source_timestamp": stamp,
            "source_type": row.get("type"), "source_action": action.get("intent", row.get("type")),
            "tab_id": _tab(properties.get("tabId")) or _tab(metadata.get("tabId")),
            "origin_tab_id": _tab(properties.get("openerTabId")) or _tab(properties.get("tabIdOrigin")),
            "source_state_id": _state_id(properties.get("documentId")),
            "source_url_id": _state_id(metadata.get("url") or properties.get("url")),
            "back_forward": "forward_back" in qualifiers,
        })
    return observations


def compile_browser_workload(
    observations: Sequence[Mapping[str, Any]], narrative: Mapping[str, Any],
    policy: WorkloadPolicy, *, source_id: str, episode_id: str, source_provenance: str = "recorded_trace",
) -> dict[str, Any]:
    """Map tab lifecycle and navigation evidence to a video checkpoint DAG.

    Each content-producing browser action advances an explicitly configured
    number of clean video chunks. Idle gaps and tab events keep their source
    ordering. Missing identities produce audit-only rows, not invented forks.
    """
    if not source_id or not episode_id or not isinstance(policy, WorkloadPolicy):
        raise ConfigurationError("source, episode and typed policy are required")
    if source_provenance not in ("recorded_trace", "authored_fixture"):
        raise ConfigurationError("source_provenance must be recorded_trace or authored_fixture")
    prompts = narrative.get("prompts")
    if not isinstance(prompts, list) or not prompts or any(not isinstance(p, str) or not p.strip() for p in prompts):
        raise ConfigurationError("narrative needs nonempty prompts")
    branches = {"main": {"cursor": 0, "seed": policy.seed, "prompt_offset": 0, "checkpoint": "main:0"}}
    tabs, state_targets, url_targets = {}, {}, {}
    events, audit = [], []
    active, prior_timestamp, prior_index = "main", None, -1

    def prompt_index(branch):
        state = branches[branch]
        return min(len(prompts) - 1, int(state["cursor"] * policy.chunk_seconds / policy.prompt_seconds)
                   + state["prompt_offset"])

    def checkpoint(branch):
        return branches[branch]["checkpoint"]

    def emit(operation, obs, rule, *, branch=None, **fields):
        branch = active if branch is None else branch
        events.append({
            "event_id": len(events), "operation": operation, "branch_id": branch,
            "source_id": source_id, "source_event_index": obs["source_event_index"],
            "source_timestamp": obs.get("source_timestamp"), "source_action": obs.get("source_action"),
            "source_state_id": obs.get("source_state_id"), "mapping_rule": rule,
            "provenance_class": "trace_mapped", "target_clean_chunk": branches[branch]["cursor"],
            "checkpoint_id": checkpoint(branch), "prompt_index": prompt_index(branch),
            "seed": branches[branch]["seed"], **fields,
        })

    for raw in observations:
        obs = json_copy(raw, name="trace observation")
        index = obs.get("source_event_index")
        if type(index) is not int or index <= prior_index:
            raise ConfigurationError("source event indices must be strictly increasing; do not reorder the trace")
        prior_index = index
        stamp = obs.get("source_timestamp")
        if stamp is not None and (isinstance(stamp, bool) or not isinstance(stamp, (int, float)) or not math.isfinite(stamp)):
            raise ConfigurationError("source timestamp must be finite or absent")
        reason = None
        if stamp is not None and stamp < 0:
            audit.append({**obs, "mapping_status": "pre_recording", "event_ids": []})
            continue
        first = len(events)
        if prior_timestamp is not None and stamp is not None:
            gap = stamp - prior_timestamp
            if gap < 0:
                raise ConfigurationError("timestamps decrease; preserve the source order and audit clock errors first")
            if gap >= policy.pause_gap_seconds and gap > 0 and branches[active]["cursor"] > 0:
                emit("pause", obs, "interaction_gap_threshold", idle_seconds=gap,
                     provenance_class="trace_mapped", target_checkpoint=checkpoint(active))
                emit("resume", obs, "same_branch_after_measured_gap", target_checkpoint=checkpoint(active))
        if stamp is not None:
            prior_timestamp = stamp
        intent, tab = obs.get("source_action"), obs.get("tab_id")
        if obs.get("source_type") == "browser" and tab is not None and not tabs and intent != "tabcreate":
            tabs[tab] = active
        if intent == "tabcreate":
            if tab is None:
                reason = "missing_new_tab_id"
            elif tab in tabs:
                raise ConfigurationError("trace creates an already live tab")
            elif branches[active]["cursor"] == 0:
                reason = "no_clean_video_checkpoint_yet"
            else:
                parent = tabs.get(obs.get("origin_tab_id"), active)
                inferred = obs.get("origin_tab_id") not in tabs
                child = f"branch-{len(branches)}"
                seed = int.from_bytes(hashlib.sha256(f"{episode_id}:{index}:{policy.seed}".encode()).digest()[:8], "little")
                branches[child] = {
                    "cursor": branches[parent]["cursor"], "seed": seed % (2**63),
                    "prompt_offset": branches[parent]["prompt_offset"], "checkpoint": f"{child}:fork-{index}",
                }
                if policy.branch_content == "next_prompt":
                    branches[child]["prompt_offset"] += 1
                target = checkpoint(parent)
                emit("fork", obs, "tabcreate_from_opener" if not inferred else "tabcreate_from_active_tab",
                     branch=child, parent_branch_id=parent, target_checkpoint=target,
                     provenance_class="inferred_relationship" if inferred else "trace_mapped",
                     seed_policy="new_seed", content_policy=policy.branch_content)
                tabs[tab] = child
        elif intent == "tabswitch":
            if tab not in tabs:
                reason = "target_tab_has_no_mapped_branch"
            elif tabs[tab] != active:
                active = tabs[tab]
                if branches[active]["cursor"] > 0:
                    emit("resume", obs, "reactivate_recorded_tab", target_checkpoint=checkpoint(active))
        elif intent == "tabremove":
            if tab in tabs and tabs[tab] != active:
                emit("retire", obs, "close_recorded_inactive_tab", branch=tabs[tab])
                del tabs[tab]
            else:
                reason = "cannot_retire_active_or_unknown_tab_without_a_recorded_switch"
        if intent == "load":
            target = state_targets.get((active, obs.get("source_state_id"))) if obs.get("source_state_id") else None
            inferred = False
            if target is None and policy.infer_url_rewind and obs.get("back_forward"):
                target = url_targets.get((active, obs.get("source_url_id")))
                inferred = target is not None
            if target is not None and target[2] != checkpoint(active):
                operation = "rewind" if target[1] < branches[active]["cursor"] else "resume"
                emit(operation, obs, "revisit_recorded_document" if not inferred else "back_to_visited_url",
                     target_checkpoint=target[2], target_clean_chunk=target[1],
                     provenance_class="inferred_relationship" if inferred else "trace_mapped")
                branches[active]["cursor"] = target[1]
                branches[active]["checkpoint"] = target[2]
            elif obs.get("back_forward"):
                reason = "back_forward_has_no_identifiable_checkpoint_target"
        if obs.get("source_type") == "browser" and intent in policy.generation_intents and tab in tabs:
            if tabs[tab] != active:
                reason = "generation_event_belongs_to_inactive_tab"
            else:
                for _ in range(policy.chunks_per_action):
                    output_checkpoint = f"{active}:generated-{len(events)}"
                    emit("generate", obs, "native_clean_chunks_per_selected_action",
                         output_checkpoint=output_checkpoint)
                    branches[active]["cursor"] += 1
                    branches[active]["checkpoint"] = output_checkpoint
                target = (active, branches[active]["cursor"], checkpoint(active))
                if obs.get("source_state_id"):
                    state_targets.setdefault((active, obs["source_state_id"]), target)
                if obs.get("source_url_id"):
                    url_targets.setdefault((active, obs["source_url_id"]), target)
        audit.append({**obs, "mapping_status": reason or ("mapped" if len(events) > first else "audit_only"),
                      "event_ids": list(range(first, len(events)))})
    workload = {
        "schema": "reelcache.trace_mapped_video_workload.v1", "episode_id": episode_id,
        "source_id": source_id, "source_provenance": source_provenance,
        "narrative_id": narrative.get("narrative_id"),
        "prompts": list(prompts), "policy": asdict(policy), "events": events, "source_audit": audit,
        "claims": {"recorded_video_user_trace": False, "source_interaction_order_preserved": True,
                   "native_model_executed": False, "video_quality_measured": False},
        "summary": {"source_events": len(observations), "operations": dict(Counter(e["operation"] for e in events)),
                    "branches": len(branches),
                    "unmapped_reasons": dict(Counter(r["mapping_status"] for r in audit
                                                    if r["mapping_status"] not in ("mapped", "audit_only", "pre_recording")))},
    }
    validate_workload(workload)
    return workload


def validate_workload(workload: Mapping[str, Any]) -> None:
    """Replay the event DAG without a model and reject dangling or inconsistent checkpoints."""
    if workload.get("schema") != "reelcache.trace_mapped_video_workload.v1":
        raise ConfigurationError("unsupported workload schema")
    branches, saved, last_source = {"main": 0}, {"main:0": ("main", 0)}, -1
    retired = set()
    for index, event in enumerate(workload.get("events", [])):
        if event.get("event_id") != index or event["source_event_index"] < last_source:
            raise ConfigurationError("workload event order is inconsistent")
        last_source = event["source_event_index"]
        branch, operation = event["branch_id"], event["operation"]
        if operation == "fork":
            target = saved.get(event.get("target_checkpoint"))
            if branch in branches or target is None or target[0] != event.get("parent_branch_id"):
                raise ConfigurationError("fork must reference a saved parent checkpoint and a new child")
            branches[branch] = target[1]
            child_checkpoint = event["checkpoint_id"]
            if child_checkpoint in saved:
                raise ConfigurationError("child checkpoint must be new")
            saved[child_checkpoint] = (branch, target[1])
        elif branch not in branches or branch in retired:
            raise ConfigurationError("event references an absent or retired branch")
        elif operation in ("resume", "rewind"):
            target = saved.get(event.get("target_checkpoint"))
            if target is None or target[0] != branch:
                raise ConfigurationError("restore target is not a saved checkpoint on this branch")
            if operation == "rewind" and target[1] >= branches[branch]:
                raise ConfigurationError("rewind must target an earlier clean boundary")
            branches[branch] = target[1]
        elif operation == "generate":
            if event["target_clean_chunk"] != branches[branch]:
                raise ConfigurationError("generation cursor is inconsistent")
            if not 0 <= event["prompt_index"] < len(workload["prompts"]):
                raise ConfigurationError("generation prompt index is out of range")
            branches[branch] += 1
            output = event.get("output_checkpoint")
            if not isinstance(output, str) or not output or output in saved:
                raise ConfigurationError("generation must publish a new immutable checkpoint")
            saved[output] = (branch, branches[branch])
        elif operation == "retire":
            retired.add(branch)
        elif operation == "pause":
            target = saved.get(event.get("target_checkpoint"))
            if target is None or target != (branch, branches[branch]):
                raise ConfigurationError("pause must name the branch's saved clean boundary")
        else:
            raise ConfigurationError(f"unsupported workload operation: {operation}")


def replay_cpu_control(workload: Mapping[str, Any], store_root) -> dict[str, Any]:
    """Exercise a compiled workload with real KVRuntime RAW snapshots and tiny CPU matrices.

    This is a model-free correctness control, not a video or performance
    experiment. It checks restored histories, RNG continuation, parent
    isolation and durable store reopen at pause/resume.
    """
    import torch

    from .manager import KVRuntime
    from .types import CodecSpec, MatrixKey, ModelIdentity, TensorContext

    validate_workload(workload)
    if not any(e["operation"] == "generate" for e in workload["events"]):
        raise ConfigurationError("CPU control requires at least one generation event")
    model = ModelIdentity("workload-control", "synthetic-clean-kv-v1", 1, 4, "tokens_channels", "float32")
    keys = [MatrixKey(0, target) for target in ("K", "V")]
    runtime = KVRuntime(store_root, model, max_resident_bytes=1 << 20, verify_reads=True)
    saved, branches, results = {}, {}, []

    def snapshot(branch, token):
        generator, history = branches[branch]
        info = runtime.branch_info(branch)
        saved[token] = {
            "snapshot": runtime.snapshot(
                branch, metadata={
                    "clean_boundary": {"chunk_id": info["chunk_ids"][-1] if info["chunk_ids"] else None,
                                       "token_end": info["token_end"], "clean": True},
                    "conditioning": {"id": workload["narrative_id"]},
                    "execution": {"pending": False, "workload_control": True},
                    "rng": {"tensor": "rng"},
                    "latent": {"native_latents": "not applicable to synthetic matrices"},
                    "vae": {"native_decoder": "not applicable to synthetic matrices"},
                }, tensors={"rng": generator.get_state()},
            ),
            "history": {key: value.clone() for key, value in history.items()},
        }

    def restore(event, *, fork=False):
        branch = event["branch_id"]
        record = saved[event["target_checkpoint"]]
        state = runtime.fork(record["snapshot"], branch) if fork else runtime.restore(record["snapshot"], branch)
        generator = torch.Generator().set_state(state.tensors["rng"])
        if fork:
            generator.manual_seed(event["seed"])
        history = {key: value.clone() for key, value in record["history"].items()}
        branches[branch] = (generator, history)
        for key in keys:
            with runtime.acquire_history(branch, key) as lease:
                if not torch.equal(lease.tensor, history[key]):
                    raise ConfigurationError("RAW restored history differs from its checkpoint")
        if fork:
            snapshot(branch, event["checkpoint_id"])

    try:
        runtime.create_branch("main")
        generator = torch.Generator().manual_seed(workload["policy"]["seed"])
        branches["main"] = (generator, {key: torch.empty(0, 4) for key in keys})
        snapshot("main", "main:0")
        for event in workload["events"]:
            branch, operation = event["branch_id"], event["operation"]
            if operation == "generate":
                generator, history = branches[branch]
                before = runtime.branch_info(branch)
                prefix = {key: value.clone() for key, value in history.items()}
                prompt = workload["prompts"][event["prompt_index"]]
                offset = int(hashlib.sha256(prompt.encode()).hexdigest()[:4], 16) / 65536
                tensors = {key: torch.randn(6, 4, generator=generator) + offset for key in keys}
                contexts = {
                    key: TensorContext(model, key, before["token_end"], 1, 2, 3,
                                       "pre_rope" if key.target == "K" else "none")
                    for key in keys
                }
                runtime.seal_chunk(branch, f"event-{event['event_id']}", tensors, contexts, CodecSpec())
                for key in keys:
                    history[key] = torch.cat((history[key], tensors[key]))
                    with runtime.acquire_history(branch, key) as lease:
                        if not torch.equal(lease.tensor, history[key]) or not torch.equal(lease.tensor[:len(prefix[key])], prefix[key]):
                            raise ConfigurationError("RAW generation changed its history prefix")
                snapshot(branch, event["output_checkpoint"])
            elif operation == "fork":
                parent = event["parent_branch_id"]
                before = runtime.branch_info(parent)["head_id"]
                restore(event, fork=True)
                if runtime.branch_info(parent)["head_id"] != before:
                    raise ConfigurationError("fork changed its parent's immutable head")
            elif operation in ("resume", "rewind"):
                restore(event)
            elif operation == "pause":
                runtime.close()
                runtime = KVRuntime(store_root, model, max_resident_bytes=1 << 20, verify_reads=True)
            elif operation == "retire":
                runtime.delete_branch(branch)
                del branches[branch]
            results.append({"event_id": event["event_id"], "operation": operation, "status": "passed"})
        report = {
            "schema": "reelcache.workload_cpu_control.v1", "status": "passed",
            "scope": "synthetic CPU matrices and durable KVRuntime lifecycle only",
            "native_model_executed": False, "video_quality_measured": False,
            "source_provenance": workload["source_provenance"], "events": results,
            "operations": dict(Counter(e["operation"] for e in results)), "store": runtime.store.stats(),
        }
    finally:
        runtime.close()
    return report
