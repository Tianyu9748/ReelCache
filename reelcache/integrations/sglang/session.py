"""Worker-owned session state and the Python/worker-level controller of the SGLang bridge.

A session is a branch of the archive state tree. Archival runs behind
generation and across request boundaries, so a session has two heads:

* the **published head** (``head_ref``): the newest published node (clean
  chunk boundary) in the runtime store (a Step 6 lineage ``lineages/<id>``
  for ``svd_hevc``, ``sglang/sessions/<id>`` for the RAW reference); a durable
  checkpoint that survives a worker restart;
* the **generated head**: the published head plus the session's submitted,
  not yet published commits (``pending``, in order). Generation continues
  from it on the valid native KV.

Commits publish strictly in order; a failed commit fails its descendants
(Step 6 pipeline). A failure is *surfaced* once (the next request, pause,
checkpoint or close of the session raises it), *acknowledged* (the pipeline
retires the failed commits; earlier ones still publish) and the session
continues from its last published checkpoint. Explicit pause, checkpoint,
fork, rewind and close wait for the required published frontier (``drain``);
so does a restore from the archive. Requests reach a session through the
plain-data ``causal_kv_session`` handle; this controller exposes the same
operations to Python callers inside the worker, and :mod:`.controls` exposes
them as the worker's typed JSON control surface (Step 14). Prompt changes are
the model's native ones only (per-block conditioning of the request; see
:func:`.controls.prompt_change_capability`).

Named checkpoints pin a published node of a session (``checkpoint`` with a
name): a compare-and-swap store ref ``sglang/checkpoints/<session>/<name>``,
so the node and its ancestors stay reachable (garbage collection never
reclaims them) until the name is deleted. Fork and rewind accept a chunk
count, a node reference or such a name.

Workload v2 (W1b) tree retention (``tree_retention="full"``; default
``"head"`` is F6): every chunk node a session generates is owned through the
store ref ``tree/<session>/<generation_index:08d>``, bound before the commit
and set in the store transaction that publishes it (the generation index from
the request's ``generation_index`` metadata, else the session's next one);
its unhashed sidecar holds the generation index, the tree parent, request id,
depth, per-block conditioning ids and the prompt schedule. A rewind accepts
any owned node (CAS at the controller, the RAW archive and the core lineage),
never another session's node or a partial cache-epoch node; the branch it
leaves stays owned and its raw-tier entries stay held (``<session>/tree``).
A reopened session rebuilds its registry from the refs; garbage collection
keeps owned nodes; ``close_session(delete=True)`` removes the refs.
"""

from __future__ import annotations

import hashlib
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Mapping

from ...errors import ConfigurationError, PendingError, StateError
from ...types import json_copy
from .nodes import TREE_HOLDER_SUFFIX, partial_recache_node, tree_entries, tree_prefix, tree_ref
from .workload_telemetry import drain_span, session_open_span

STATUSES = ("idle", "running", "paused", "completed", "failed", "closed")
#: Store refs of named checkpoints: ``sglang/checkpoints/<session>/<name>`` -> node reference.
CHECKPOINT_REF_PREFIX = "sglang/checkpoints/"
#: Session ids and checkpoint names a client may choose (never paths; never node references, which hold ``:``).
NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
#: Events kept per session (older ones are dropped; the session's archive is the durable record).
MAX_HISTORY = 512


def checkpoint_ref(session_id: str, name: str) -> str:
    """The store ref of a named checkpoint (both names validated)."""
    for label, value in (("session id", session_id), ("checkpoint name", name)):
        if not isinstance(value, str) or NAME_PATTERN.fullmatch(value) is None:
            raise ConfigurationError(f"{label} must match {NAME_PATTERN.pattern}, got {value!r}")
    return f"{CHECKPOINT_REF_PREFIX}{session_id}/{name}"


@dataclass
class SGLangSessionState:
    """Mutable per-session state (process-local resource owner; never sent through request IPC)."""

    session_id: str
    root_ref: str
    head_ref: str  # published head (durable checkpoint)
    conditioning_digest: str
    identity: Mapping[str, Any]
    generated_chunks: int = 0  # chunks on the generated head (published + pending)
    status: str = "idle"
    pause_requested: bool = False
    # Submitted commits not yet seen published, in submission order:
    # {"node_id", "kind", "chunk", "ticket", "request_id", "submitted"}.
    pending: list = field(default_factory=list)
    # chunk index -> {"token_start", "num_tokens", "rope_start_frame", "num_frames"} on the generated head.
    chunk_table: dict = field(default_factory=dict)
    revision: int = 0  # renewed whenever the generated head changes (resident KV ancestry check)
    # Generated path in order: {"kind": "chunk"|"recache", "block", "conditioning", "epoch", "node_id"}; a recache
    # entry's block is the block it precedes. Compared with a request's per-block conditioning.
    timeline: list = field(default_factory=list)
    backpressure: dict = field(default_factory=lambda: {"waits": 0, "seconds": 0.0})
    archive_failures: int = 0
    history: list = field(default_factory=list)
    lock: threading.RLock = field(default_factory=threading.RLock, repr=False)
    # Workload v2 tree retention (``tree_retention="full"``): ``{generation_index: node_ref}`` of the session's owned
    # (published) nodes, the next index to assign, the sidecar of submitted, unpublished chunk commits by node id, and
    # the record of the last head switch (rewind: its drain, from/to).
    tree: dict = field(default_factory=dict)
    next_generation: int = 1
    tree_pending: dict = field(default_factory=dict)
    last_switch: dict | None = None
    # W1.3 (``tree_switch="detach"``): open detached branches, oldest first -- ``{"branch", "items" (their pending
    # commits, in order), "from_head", "to", "cut", "opened", "published", "failed", "events", "close_error"}`` (a
    # branch stays until its close succeeds) -- the branch counter and the failed detached commits (``"<branch>/
    # <node_id>"`` -> event, the newest ``DETACH_FAILURES_KEPT``; a rewind to one of them raises a clear error).
    detached: list = field(default_factory=list)
    detach_sequence: int = 0
    detach_failures: dict = field(default_factory=dict)
    # W1.4.3 (review C1; an evicting ablation arm): the session's attended history was installed (in part) from the
    # lossy archive by its last restore -- its captures from then on are no clean replay (inexact raw entries)
    lossy_history: bool = False

    def log(self, event: str, **details: Any) -> None:
        self.history.append({"event": event, "time": time.time(), **details})
        if len(self.history) > MAX_HISTORY:
            del self.history[:len(self.history) - MAX_HISTORY]

    @property
    def published(self) -> bool:
        return not self.pending

    def describe(self) -> dict[str, Any]:
        with self.lock:
            described = {"session_id": self.session_id, "root": self.root_ref, "head": self.head_ref,
                         "conditioning": self.conditioning_digest, "generated_chunks": self.generated_chunks,
                         "pending": [item["node_id"] for item in self.pending], "published": not self.pending,
                         "status": self.status, "pause_requested": self.pause_requested, "revision": self.revision,
                         "backpressure": dict(self.backpressure), "archive_failures": self.archive_failures,
                         "history": list(self.history)}
            if self.tree:  # tree retention: the owned nodes by generation index
                described["tree"] = {str(index): ref for index, ref in sorted(self.tree.items())}
            if self.detached:  # W1.3: detached branches still publishing (or whose close is retried: ``close_error``)
                described["detached"] = [{"branch": item["branch"], "pending": [entry["node_id"] for entry in
                                                                                 item["items"]],
                                          "published": item["published"], "failed": item["failed"],
                                          **({"close_error": item["close_error"]} if item.get("close_error") else {})}
                                         for item in self.detached]
            return described


def _chunks(nodes) -> int:
    return sum(node.kind == "chunk" for node in nodes)


def timeline_of(nodes) -> list[dict[str, Any]]:
    """The conditioning timeline of a node path (root excluded; see ``SGLangSessionState.timeline``)."""
    entries = []
    for node in nodes:
        if node.kind == "chunk":
            entries.append({"kind": "chunk", "block": node.chunk_index, "epoch": None, "node_id": node.ref,
                            "conditioning": node.record.get("block_conditioning")})
        elif node.kind == "recache":
            entries.append({"kind": "recache", "block": node.record.get("precedes_block"),
                            "epoch": node.record.get("epoch"), "node_id": node.ref,
                            "conditioning": node.record.get("block_conditioning")})
    return entries


def partial_epoch_nodes(nodes) -> set[str]:
    """Nodes holding a partially re-materialized cache epoch (the RAW reference chains one recache node per range;
    only the epoch's last node is a state generation had): never checkpoints."""
    return {node.ref for node, following in zip(nodes, nodes[1:])
            if node.kind == following.kind == "recache" and node.record.get("epoch") == following.record.get("epoch")}


def _chunk_table(nodes) -> dict[int, dict[str, Any]]:
    return {node.chunk_index: {"token_start": node.token_start, "num_tokens": node.num_tokens,
                               "rope_start_frame": float(node.rope_start_frame), "num_frames": node.num_frames}
            for node in nodes if node.kind == "chunk"}


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def prompt_schedule(request: Any, depth: int, block_prompts: Any = None) -> dict[str, Any]:
    """W1b: the prompt (shot) schedule of a node's path for its sidecar, one entry per block
    (``{"block", "sha256", "text"}`` plus the block's conditioning id when the request has per-block conditioning).
    First available source: the request's ``workload["prompt_schedule"]`` metadata (per block, as the runner bound
    it: ``{"text", "sha256"}`` objects or texts), the stage's per-block prompt texts (``block_prompts``), else derived
    from the request: its conditioning ``prompt`` for every block when it has no per-block conditioning, only the
    conditioning ids otherwise (``text`` None)."""
    ids = getattr(request, "block_conditioning", None)
    ids = None if ids is None else [str(item) for item in ids]
    workload = getattr(request, "workload", None)
    supplied = workload.get("prompt_schedule") if isinstance(workload, Mapping) else None
    blocks = []
    if supplied is not None:
        source = "request"
        for index, item in enumerate(list(supplied)[:depth]):
            text = item.get("text") if isinstance(item, Mapping) else item if isinstance(item, str) else None
            digest = item.get("sha256") if isinstance(item, Mapping) else None
            blocks.append({"block": index, "text": text,
                           "sha256": digest or (_sha256(text) if isinstance(text, str) else None)})
    elif block_prompts is not None:
        source = "stage"
        for index, text in enumerate(list(block_prompts)[:depth]):
            blocks.append({"block": index, "text": text, "sha256": _sha256(text) if isinstance(text, str) else None})
    else:
        source = "derived"
        prompt = (getattr(request, "conditioning", None) or {}).get("prompt")
        for index in range(depth):
            text = prompt if ids is None and isinstance(prompt, str) else None
            blocks.append({"block": index, "text": text, "sha256": None if text is None else _sha256(text)})
    if ids is not None:
        for entry in blocks:
            if entry["block"] < len(ids):
                entry["conditioning_id"] = ids[entry["block"]]
    return {"source": source, "blocks": blocks}


def planned_new_chunks(blocks: int, resume_blocks: int, pause_after: Any = None) -> int:
    """W1.2: the new chunks a request generates as planned, by the bridge run's pause rule (``_BridgeRun.end_chunk``):
    from ``resume_blocks`` up to ``pause_after`` (the handle's ``pause_after_chunks``: the session's chunk count at
    which the run pauses, checked after each new chunk, so at least one is generated) or the request's last block.
    A request sized for a fixed session total with ``pause_after_chunks = depth + 1`` plans one chunk, not
    ``blocks - resume_blocks``. A pause control in flight can still end a request earlier (its later indices are
    never used)."""
    end = int(blocks)
    if pause_after is not None:
        end = min(end, max(int(pause_after), int(resume_blocks) + 1))
    return max(1, end - int(resume_blocks))


class SGLangSessionController:
    """Python/worker-level session controls over a :class:`CausalWanKVBridge`'s archive backend."""

    def __init__(self, bridge: Any):
        self._bridge = bridge
        self._lock = threading.RLock()
        self._sessions: dict[str, SGLangSessionState] = {}
        # Workload v2 (W1b): ``tree_retention="full"``: every generated chunk node stays owned (``tree/<session>/...``
        # refs set in its publishing transaction) and any owned node is a rewind target (the archive accepts them).
        self.tree_mode = getattr(getattr(bridge, "config", None), "tree_retention", "head") == "full"
        if self.tree_mode:
            archive = self._archive
            owner = getattr(archive, "set_tree_ownership", None)
            if not callable(owner):
                owner = getattr(getattr(archive, "runtime", None), "set_lineage_ownership", None)
            if not callable(owner):
                raise ConfigurationError("tree_retention='full' needs an archive whose rewinds accept owned nodes")
            owner(tree_prefix)
        # W1.3: how a tree-mode rewind leaves a branch with commits in flight (``tree_switch``), its counters, and the
        # removal of detached head refs a previous process left (a detached branch never outlives its process)
        self.switch_mode = getattr(getattr(bridge, "config", None), "tree_switch", "drain") if self.tree_mode \
            else "drain"
        self._switch_lock = threading.Lock()
        self.switch_stats_ = {"branches_opened": 0, "branches_closed": 0, "commits_detached": 0,
                              "commits_published": 0, "commits_failed": 0, "backlog": 0, "backlog_peak": 0,
                              "fallbacks": 0, "fallback_reasons": [], "stale_refs_removed": []}
        if self.tree_mode:  # a stale branch's last published node stays rooted when no tree ref covers it (F2)
            stale = getattr(self._archive, "stale_detached", None)
            if callable(stale):
                from .nodes import detached_session

                for branch in sorted(stale()):
                    self._close_branch(detached_session(branch), branch)
                    self.switch_stats_["stale_refs_removed"].append(branch)

    @property
    def _archive(self):
        return self._bridge.archive

    # -- raw tier holders (freeze D: R1 attach/detach, I5 release at close) -----------------------------------------
    def _raw_tier_call(self, name: str, *args: Any, **kwargs: Any) -> dict[str, Any] | None:
        """Call the archive's raw-tier holder hook ``name`` (``raw_hold`` / ``release_session``), None when the
        archive has none (RAW archives, cores before freeze D). Holder bookkeeping never fails a control or a
        request: an error comes back as ``{"error": ...}`` (the tier's stats still show what it holds)."""
        hook = getattr(self._archive, name, None)
        if not callable(hook):
            return None
        try:
            return hook(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 - recorded; the lifecycle tests check tier bytes return to 0
            return {"error": f"{type(exc).__name__}: {exc}"}

    def _raw_hold(self, session_id: str, head_ref: str | None) -> dict[str, Any] | None:
        """After a head move the archive does not hold itself (design §11 R1: ``archive.fork`` / ``archive.rewind``
        do): the session holds exactly the raw-tier entries of ``head_ref``'s path plus its unpublished commits; an
        entry no session holds is dropped once published (forks share entries)."""
        return None if head_ref is None else self._raw_tier_call("raw_hold", session_id, head_ref)

    def _raw_release(self, session_id: str, *, drop: bool = False) -> dict[str, Any] | None:
        """At session close (R1: the bridge's call; ``archive.delete`` and ``create_session(replace=True)`` release
        themselves): the session detaches from every raw-tier entry, so no tier memory outlives it unless another
        session holds the entry (I5). Idempotent. wl-w1b7: with ``raw_tier_retain="lru"`` a close keeps the entries
        as unheld cache entries; ``drop`` (whole-tree delete) drops them."""
        if drop:
            return self._raw_tier_call("release_session", session_id, drop=True)
        return self._raw_tier_call("release_session", session_id)

    # -- workload v2 (W1b): session trees -------------------------------------------------------------------------------
    def tree(self, session_id: str) -> dict[int, str]:
        """``{generation_index: node_ref}`` of a session's ownership refs, from the store (empty in head mode)."""
        return tree_entries(self._archive.store.list_refs(tree_prefix(session_id)), session_id)

    def owned_nodes(self, state: SGLangSessionState) -> set[str]:
        """Every node the session owns: the paths of its ownership refs and of its head (tree mode)."""
        roots = set(self.tree(state.session_id).values()) | {state.head_ref}
        roots |= {branch["cut"] for branch in list(state.detached) if branch.get("cut")}  # W1.3 (R2): open cuts
        owned: set[str] = set()
        for ref in roots:
            if ref not in owned:
                owned.update(node.ref for node in self._archive.path(ref))
        return owned

    def _owns(self, state: SGLangSessionState, node_ref: str) -> bool:
        refs = set(self.tree(state.session_id).values())
        if node_ref in refs:
            return True
        return node_ref in self.owned_nodes(state)

    def _tree_hold(self, state: SGLangSessionState) -> dict[str, Any] | None:
        """Tree mode: before the head leaves a branch, a second holder (``<session>/tree``) takes the raw-tier entries
        of every owned node, so they stay in the LRU (budgets still evict them) instead of dropping when the session
        holder detaches. Never fails a control (an error is returned)."""
        tier = getattr(self._archive, "raw_tier", None)
        if not self.tree_mode or tier is None:
            return None
        try:
            refs = self.owned_nodes(state)
            if self.switch_mode == "detach":  # W1.3 (R2b): and what the keep refs of closed branches root
                refs |= self._kept_nodes(state)
            return tier.hold(state.session_id + TREE_HOLDER_SUFFIX, refs, exclusive=True)
        except Exception as exc:  # noqa: BLE001 - tier bookkeeping never fails a control
            return {"error": f"{type(exc).__name__}: {exc}"}

    def _kept_nodes(self, state: SGLangSessionState) -> set[str]:
        """W1.3 (R2b): the paths of the session's keep refs (closed detached branches whose last published node no tree
        ref covers, e.g. a recache epoch whose chunk failed). The tree holder keeps their raw entries like owned nodes'
        (they are not rewind targets: ``_owns`` is unchanged)."""
        from .nodes import DETACHED_KEEP_PREFIX

        kept: set[str] = set()
        for ref in self._archive.store.list_refs(f"{DETACHED_KEEP_PREFIX}{state.session_id}/").values():
            if ref not in kept:
                kept.update(node.ref for node in self._archive.path(ref))
        return kept

    def _hold_branch_path(self, state: SGLangSessionState, branch: str) -> None:
        """W1.3 (R2b): right before a detached branch's holder goes, the tree holder takes every node of the path of
        the branch's current head -- the non-final pieces of a multi-piece raw_offload recache included, which only
        the branch held and which no tree ref covers when the epoch's chunk failed. Never fails a reconcile."""
        tier = getattr(self._archive, "raw_tier", None)
        if tier is None:
            return
        try:
            head = self._archive.detached_head(branch)
            if head is not None:
                tier.hold(state.session_id + TREE_HOLDER_SUFFIX, [node.ref for node in self._archive.path(head)],
                          exclusive=False)
        except Exception as exc:  # noqa: BLE001 - the raw copies are a cache: the nodes restore from the store
            with state.lock:
                state.log("tree_hold_failed", branch=branch, error=repr(exc))

    def _bind(self, session_id: str, node_id: str, names: list[str]) -> None:
        """Refs the archive sets in the transaction that publishes commit ``node_id`` (the RAW archive's own binding,
        else the lineage runtime's)."""
        binder = getattr(self._archive, "bind_publication_refs", None)
        if not callable(binder):
            binder = getattr(getattr(self._archive, "runtime", None), "bind_publication_refs", None)
        if not callable(binder):
            raise ConfigurationError("tree_retention='full' needs an archive that sets refs with a publication")
        binder(session_id, node_id, names)

    def _unbind(self, session_id: str) -> None:
        for owner in (self._archive, getattr(self._archive, "runtime", None)):
            unbind = getattr(owner, "unbind_publication_refs", None)
            if callable(unbind):
                unbind(session_id)

    def _delete_tree(self, session_id: str) -> dict[str, str]:
        """Remove a session's ownership refs (whole-tree delete; garbage collection then reclaims what nothing else
        reaches). Returns ``{ref name: node}`` removed."""
        store = self._archive.store
        removed = {}
        for name, node in store.list_refs(tree_prefix(session_id)).items():
            store.delete_ref(name, expected=node)
            removed[name] = node
        from .nodes import DETACHED_KEEP_PREFIX  # W1.3 (F2): the kept heads of closed detached branches go too

        for name, node in store.list_refs(f"{DETACHED_KEEP_PREFIX}{session_id}/").items():
            store.delete_ref(name, expected=node)
            removed[name] = node
        self._unbind(session_id)
        return removed

    def prepare_commit(self, state: SGLangSessionState, *, kind: str, node_id: str, chunk: int | None,
                       request: Any, resume_blocks: int, block_prompts: Any = None,
                       pause_after: int | None = None) -> None:
        """Tree mode, before a chunk commit is handed to the archive: assign its generation index and bind its
        ownership ref ``tree/<session>/<index:08d>`` to the publication. The request's ``generation_index`` metadata
        (optional, read defensively) names its LAST planned new chunk (:func:`planned_new_chunks`: up to the run's pause
        point ``pause_after``, the handle's ``pause_after_chunks``, else its last block); earlier new chunks of a
        multi-chunk request take the indices before it; without it the next index of the session (monotonic). The
        node's unhashed sidecar (generation index, parent, request id, depth, per-block conditioning ids, prompt
        schedule) is written when it publishes (:meth:`reconcile`)."""
        if not self.tree_mode or kind != "chunk" or chunk is None:
            return
        workload = getattr(request, "workload", None)
        requested = getattr(request, "generation_index", None)
        if requested is None and isinstance(workload, Mapping):
            requested = workload.get("generation_index")
        new_chunks = planned_new_chunks(len(request.block_sizes), resume_blocks, pause_after)
        with state.lock:
            if requested is not None:
                if type(requested) is not int or requested < new_chunks:
                    raise ConfigurationError(f"generation_index must be an integer >= the request's {new_chunks} new "
                                             f"chunks, got {requested!r}")
                index = requested - (new_chunks - 1) + (chunk - int(resume_blocks))
            else:
                index = state.next_generation
            if index < 1:
                raise ConfigurationError(f"generation index {index} of chunk {chunk} is not positive")
            state.next_generation = max(state.next_generation, index + 1)
            ids = getattr(request, "block_conditioning", None)
            try:
                metadata = None if not isinstance(workload, Mapping) else json_copy(dict(workload),
                                                                                   name="workload metadata")
            except Exception:  # noqa: BLE001 - sidecar provenance never fails a commit
                metadata = {"unserializable": repr(workload)[:2000]}
            state.tree_pending[node_id] = {
                "session_id": state.session_id, "generation_index": index, "request_id": request.request_id,
                "depth": chunk + 1, "chunk_index": chunk,
                "conditioning_ids": None if ids is None else [str(item) for item in list(ids)[:chunk + 1]],
                "prompt_schedule": prompt_schedule(request, chunk + 1, block_prompts), "workload": metadata}
        self._bind(state.session_id, node_id, [tree_ref(state.session_id, index)])

    def _tree_sidecar(self, state: SGLangSessionState, node_ref: str, tree: Mapping[str, Any]) -> dict[str, Any]:
        """The ``tree`` sidecar of a published node: the generation's fields, its parent and every (session,
        generation index) that produced this node (a same-prompt regeneration deduplicates to an existing node)."""
        previous = ((self._archive.store.telemetry(node_ref) or {}).get("bridge") or {}).get("tree") or {}
        key = (tree["session_id"], tree["generation_index"])
        generations = [item for item in previous.get("generations") or []
                       if (item.get("session_id"), item.get("generation_index")) != key]
        generations.append({"session_id": tree["session_id"], "generation_index": tree["generation_index"],
                            "request_id": tree.get("request_id")})
        parent = self._archive.load(node_ref).parent_ref
        tree_parent = parent  # the generated node it continues (cache-epoch nodes in between are internal)
        while tree_parent is not None:
            node = self._archive.load(tree_parent)
            if node.kind in ("chunk", "root"):
                break
            tree_parent = node.parent_ref
        return {**tree, "parent": tree_parent, "parent_node": parent, "generations": generations[-64:],
                "deduplicated": len(generations) > 1}

    # -- registry -------------------------------------------------------------------------------------
    def get(self, session_id: str, *, required: bool = True) -> SGLangSessionState | None:
        with self._lock:
            state = self._sessions.get(session_id)
            if state is None and required:
                with session_open_span():  # workload telemetry (W1.2): a reopen inside a request's controller step
                    state = self._reopen(session_id)
            return state

    def _reopen(self, session_id: str) -> SGLangSessionState:
        """Rebuild a session from its persisted head (e.g. after a worker restart), with load-time preparation."""
        started = time.perf_counter()
        archive = self._archive
        head = archive.head(session_id)
        if head is None:
            raise StateError(f"unknown causal KV session {session_id!r}")
        path = archive.path(head)
        root = path[0]
        state = SGLangSessionState(session_id, root.ref, head, root.record["conditioning_digest"],
                                   root.record["identity"], _chunks(path[1:]), "paused",
                                   chunk_table=_chunk_table(path[1:]), timeline=timeline_of(path[1:]))
        if self.tree_mode:  # W1b: the registry is rebuilt from the session's ownership refs
            state.tree = self.tree(session_id)
            state.next_generation = max(state.tree, default=0) + 1
        prewarm = archive.prewarm(head) if self._bridge.config.prewarm else 0.0
        state.log("reopened", head=head, prewarm_seconds=prewarm, seconds=time.perf_counter() - started)
        self._sessions[session_id] = state
        self._raw_hold(session_id, head)  # raw entries of its path that other sessions kept (R1)
        return state

    def open(self, session_id: str) -> dict[str, Any]:
        """Open a persisted session before its next request: load its head and prepare its history."""
        with self._lock:
            existing = self._sessions.get(session_id)
            if existing is not None:
                return existing.describe()
            return self._reopen(session_id).describe()

    def create(self, session_id: str, *, record: Mapping[str, Any], streams, conditioning_digest: str,
               identity: Mapping[str, Any], replace: bool = False,
               telemetry: Mapping[str, Any] | None = None) -> SGLangSessionState:
        with self._lock:
            existing = self._sessions.get(session_id)
            if existing is not None and existing.status == "running":
                raise PendingError(f"causal KV session {session_id!r} is running")
            if (existing is not None or self._archive.head(session_id) is not None) and not replace:
                raise StateError(f"causal KV session {session_id!r} already exists (use resume, or replace=true)")
            if existing is not None or self._archive.live(session_id):
                # The replaced session is discarded: its unpublished commits are retired, never published.
                self._archive.abandon(session_id)
            if existing is not None and existing.detached:  # W1.3: and its detached branches' (refs removed)
                self._abandon_detached(existing)
            self._bridge.drop_resident(session_id)
            if self.tree_mode:  # W1b: a new session tree starts empty (a replaced tree is discarded)
                self._delete_tree(session_id)
                self._raw_tier_call("release_session", session_id + TREE_HOLDER_SUFFIX)
            root_ref = self._archive.create_session(session_id, record=record, streams=streams, replace=replace,
                                                    telemetry=telemetry)
            state = SGLangSessionState(session_id, root_ref, root_ref, conditioning_digest, dict(identity))
            if existing is not None:
                state.revision = existing.revision + 1
            state.log("created", root=root_ref, replaced=existing is not None)
            self._sessions[session_id] = state
            return state

    def sessions(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            return {session_id: state.describe() for session_id, state in self._sessions.items()}

    def state(self, session_id: str) -> SGLangSessionState | None:
        """The session's in-memory state, or None when it is not open (never reopens)."""
        with self._lock:
            return self._sessions.get(session_id)

    def persisted(self, session_id: str) -> dict[str, Any] | None:
        """Description of a session from its persisted head alone (never reopens or prepares); None if unknown."""
        head = self._archive.head(session_id)
        if head is None:
            return None
        path = self._archive.path(head)
        nodes = path[1:]
        return {"session_id": session_id, "root": path[0].ref, "head": head,
                "conditioning": path[0].record.get("conditioning_digest"), "generated_chunks": _chunks(nodes),
                "nodes": len(nodes), "pending": [], "published": True, "status": "closed"}

    # -- named checkpoints -------------------------------------------------------------------------------
    def named_checkpoints(self, session_id: str) -> dict[str, str]:
        """``{name: node reference}`` of the session's named checkpoints (none for ids that cannot hold names)."""
        if not isinstance(session_id, str) or NAME_PATTERN.fullmatch(session_id) is None:
            return {}
        prefix = f"{CHECKPOINT_REF_PREFIX}{session_id}/"
        refs = self._archive.store.list_refs(prefix)
        return {name[len(prefix):]: node for name, node in sorted(refs.items())}

    def pin(self, session_id: str, name: str, node_ref: str, *, replace: bool = False) -> str | None:
        """Name a published node of the session (compare-and-swap); returns the node the name held before."""
        ref = checkpoint_ref(session_id, name)
        store = self._archive.store
        previous = store.get_ref(ref)
        if previous is not None and previous != node_ref and not replace:
            raise StateError(f"session {session_id!r} already has a checkpoint named {name!r} (at {previous}); "
                             "delete it or pass replace")
        if previous != node_ref:
            store.set_ref(ref, node_ref, expected=previous)
        return previous

    def unpin(self, session_id: str, name: str) -> str:
        """Delete a named checkpoint; returns the node it named."""
        ref = checkpoint_ref(session_id, name)
        store = self._archive.store
        node = store.get_ref(ref)
        if node is None:
            raise KeyError(name)
        store.delete_ref(ref, expected=node)
        return node

    def unpin_all(self, session_id: str) -> dict[str, str]:
        removed = {}
        for name in self.named_checkpoints(session_id):
            try:
                removed[name] = self.unpin(session_id, name)
            except KeyError:
                continue
        return removed

    # -- publication -----------------------------------------------------------------------------------
    def track(self, state: SGLangSessionState, *, node_id: str, kind: str, ticket: Any, chunk: int | None,
              request_id: Any, telemetry: Mapping[str, Any] | None = None,
              timeline: Mapping[str, Any] | None = None) -> None:
        """Record a submitted commit on the generated head (then publish what already resolved)."""
        with state.lock:
            state.pending.append({"node_id": node_id, "kind": kind, "chunk": chunk, "ticket": ticket,
                                  "request_id": request_id, "submitted": time.time(),
                                  "telemetry": {**dict(telemetry or {}), "request_id": request_id}})
            tree = state.tree_pending.pop(node_id, None) if kind == "chunk" else None
            if tree is not None:  # W1b tree mode: the generation index and sidecar of this chunk commit
                state.pending[-1]["tree"] = tree
            if timeline is not None:
                state.timeline.append({**dict(timeline), "node_id": node_id})
            state.revision += 1
        self.reconcile(state, surface=False)

    def reconcile(self, state: SGLangSessionState, *, surface: bool = True,
                  stop_at: Any = None) -> list[dict[str, Any]]:
        """Advance the published head over resolved commits (never waits); returns the newly published ones.

        With ``surface`` an archival failure of the session is acknowledged and raised (``StateError``);
        otherwise it stays pending until the next request, pause, checkpoint or close of the session."""
        published, failed = [], None
        with state.lock:
            while state.pending:
                item = state.pending[0]
                ticket = item["ticket"]
                if not ticket.done() or getattr(ticket, "error", None) is not None:
                    break
                if stop_at is not None and item["node_id"] in stop_at:  # W1.3: handed to a detached branch
                    break
                state.pending.pop(0)
                state.head_ref = ticket.result(0)
                published.append(item)
                if item.get("tree") is not None:  # W1b: the ownership ref was set with the publication
                    state.tree[item["tree"]["generation_index"]] = state.head_ref
                state.log("published", node_id=item["node_id"], node=state.head_ref,
                          lag_seconds=time.time() - item["submitted"])
        for item in published:  # run provenance outside the hashed node content
            try:
                ref = item["ticket"].result(0)
                telemetry = item["telemetry"]
                if item.get("tree") is not None:
                    telemetry = {**telemetry, "tree": self._tree_sidecar(state, ref, item["tree"])}
                self._archive.annotate(ref, telemetry)
            except Exception as exc:  # noqa: BLE001 - provenance only, never fails a publication
                with state.lock:
                    state.log("telemetry_failed", node_id=item["node_id"], error=repr(exc))
        if surface and state.detached:  # W1.3: detached branches publish on their own (never raises)
            self._reconcile_detached(state)
        with state.lock:
            failed = next((item for item in state.pending if item["ticket"].done()
                           and getattr(item["ticket"], "error", None) is not None), None)
        if failed is not None and surface:
            raise self.acknowledge(state, failed["ticket"].error)
        return published

    def failure(self, state: SGLangSessionState) -> BaseException | None:
        """The session's archival failure that is not surfaced yet (never waits)."""
        with state.lock:
            for item in state.pending:
                error = getattr(item["ticket"], "error", None) if item["ticket"].done() else None
                if error is not None:
                    return error
        return None

    def acknowledge(self, state: SGLangSessionState, error: BaseException) -> StateError:
        """Retire a failed session's unpublished commits; the session continues from its published head.

        Returns the ``StateError`` to raise (the failure surfaces exactly once)."""
        head = self._archive.release(state.session_id)
        nodes = self._archive.path(head)[1:]
        with state.lock:
            lost = [item["node_id"] for item in state.pending if not item["ticket"].done()
                    or getattr(item["ticket"], "error", None) is not None]
            published = [item for item in state.pending if item["node_id"] not in lost]
            state.pending.clear()
            state.head_ref = head
            state.generated_chunks = _chunks(nodes)
            state.chunk_table = _chunk_table(nodes)
            state.timeline = timeline_of(nodes)
            state.revision += 1
            state.archive_failures += 1
            if state.status != "running":
                state.status = "failed"
            state.log("archive_failed", error=repr(error), head=head, lost=lost,
                      published=[item["node_id"] for item in published])
        if self.tree_mode:  # W1b: retired commits never publish their ownership refs; owned entries stay held
            self._unbind(state.session_id)
            registry = self.tree(state.session_id)  # commits published before the failure set their refs
            with state.lock:
                state.tree_pending.clear()
                state.tree = registry
            self._tree_hold(state)
        self._raw_hold(state.session_id, head)  # the head moved back: hold exactly its path
        self._bridge.drop_resident(state.session_id)
        message = (f"session {state.session_id!r}: archival failed ({error!r}); commits {lost} were not published "
                   f"and the session continues from its last published checkpoint {head}")
        failure = StateError(message)
        failure.__cause__ = error
        return failure

    def drain(self, state: SGLangSessionState, *, timeout: float | None = None) -> float:
        """Wait for the published frontier to reach the generated head as submitted now; returns seconds waited.

        Raises the session's archival failure (acknowledged) or ``PendingError`` on timeout."""
        self.reconcile(state)
        with state.lock:
            if not state.pending:
                return 0.0
            through = state.pending[-1]["node_id"]
        started = time.perf_counter()
        try:
            self._archive.drain(state.session_id, through=through, timeout=timeout)
        except PendingError:
            raise
        except BaseException as exc:  # noqa: BLE001 - the pipeline's failure of this session
            raise self.acknowledge(state, exc) from exc
        self.reconcile(state)
        seconds = time.perf_counter() - started
        with state.lock:
            state.log("drained", through=through, head=state.head_ref, seconds=seconds)
        return seconds

    def release_idle(self, *, keep: set[str] = frozenset()) -> list[str]:
        """Release the pipeline state of idle, fully published sessions (bounded device memory); never raises
        (a session that cannot be released now is released on a later request, its failures surface on its own
        next request, pause, checkpoint or close)."""
        released = []
        with self._lock:
            candidates = [state for session_id, state in self._sessions.items()
                          if session_id not in keep and state.status != "running"]
        for state in candidates:
            try:
                if state.detached:  # W1.3: idle sessions' detached branches are reconciled (and closed) here too
                    self._reconcile_detached(state)
                if not self._archive.live(state.session_id):
                    continue
                self.reconcile(state, surface=False)
                with state.lock:
                    if state.pending:
                        continue
                self._archive.release(state.session_id, timeout=0)
            except Exception as exc:  # noqa: BLE001 - an optimization, retried later
                with state.lock:
                    state.log("release_deferred", error=repr(exc))
                continue
            released.append(state.session_id)
        return released

    # -- controls ---------------------------------------------------------------------------------------
    def request_pause(self, session_id: str) -> None:
        """Pause the session's running request at its next clean chunk boundary (a durable pause by default)."""
        state = self.get(session_id)
        with state.lock:
            if state.status != "running":
                raise StateError(f"session {session_id!r} is {state.status}; nothing to pause")
            state.pause_requested = True
            state.log("pause_requested")

    def pause(self, session_id: str, *, timeout: float | None = None) -> dict[str, Any]:
        """Explicit pause: a running request stops at its next clean boundary and publishes before it returns;
        an idle session waits for its published frontier now. Returns the session description."""
        state = self.get(session_id)
        with state.lock:
            running = state.status == "running"
            if running:
                state.pause_requested = True
                state.log("pause_requested")
        if not running:
            self.checkpoint(session_id, timeout=timeout)
        return state.describe()

    def checkpoint(self, session_id: str, *, timeout: float | None = None) -> dict[str, Any]:
        """Wait until everything the session generated so far is published; returns the durable head. W1.3: its
        detached branches too (the whole tree is durable)."""
        state = self.get(session_id)
        seconds = self.drain(state, timeout=timeout)
        if state.detached:
            seconds += self._drain_detached(state, timeout=timeout)
        with state.lock:
            return {"session_id": session_id, "head": state.head_ref, "generated_chunks": state.generated_chunks,
                    "published": not state.pending, "wait_seconds": seconds}

    def checkpoints(self, session_id: str) -> list[dict[str, Any]]:
        """Clean boundaries of the session head path (drained first): ``{"chunks": n, "node": ref}``; nodes of a
        partially re-materialized epoch are listed with ``checkpoint: False``."""
        state = self.get(session_id)
        self.drain(state)
        path = self._archive.path(state.head_ref)
        partial = partial_epoch_nodes(path)
        result, chunks = [], 0
        for node in path:
            chunks += node.kind == "chunk"
            result.append({"chunks": chunks, "node": node.ref, "kind": node.kind,
                           "checkpoint": node.ref not in partial})
        return result

    def _resolve_checkpoint(self, state: SGLangSessionState, checkpoint: Any) -> str:
        """A chunk count, a node reference on the session's path, or a named checkpoint of the session (whose
        node may lie off the current head path, e.g. after a rewind: fork accepts it, rewind needs an ancestor)."""
        if checkpoint is None:
            return state.head_ref
        if type(checkpoint) is int:
            # The newest node (chunk or recache) at which the session had generated ``checkpoint`` chunks.
            boundaries = {item["chunks"]: item["node"] for item in self.checkpoints(state.session_id)
                          if item["checkpoint"]}
            if checkpoint not in boundaries:
                raise ConfigurationError(f"session {state.session_id!r} has {max(boundaries)} chunks; "
                                         f"no checkpoint after {checkpoint}")
            return boundaries[checkpoint]
        if isinstance(checkpoint, str) and NAME_PATTERN.fullmatch(checkpoint) is not None:
            named = self.named_checkpoints(state.session_id)
            if checkpoint not in named:
                raise StateError(f"session {state.session_id!r} has no checkpoint named {checkpoint!r}")
            return named[checkpoint]
        path = self._archive.path(state.head_ref)
        if checkpoint not in [node.ref for node in path]:
            if self.tree_mode and isinstance(checkpoint, str) and self._owns(state, checkpoint):
                # W1b tree mode: any node the session owns (e.g. on an old branch), but never a partial epoch node
                if partial_recache_node(self._archive.load(checkpoint)):
                    raise StateError(f"{checkpoint!r} holds a partially re-materialized cache epoch (one range of a "
                                     "multi-range recache); use the epoch's last node")
                return checkpoint
            raise StateError(f"checkpoint {checkpoint!r} is not on session {state.session_id!r}'s path"
                             + self._failed_detached_note(state))
        if checkpoint in partial_epoch_nodes(path):
            raise StateError(f"{checkpoint!r} holds a partially re-materialized cache epoch (one range of a "
                             "multi-range recache); use the epoch's last node")
        return checkpoint

    def fork(self, source_id: str, session_id: str, *, checkpoint: Any = None, check=None) -> SGLangSessionState:
        """New session at a clean boundary of ``source_id`` (published first); the source is untouched.
        ``check(timeline)`` may reject the checkpoint's path before anything is created."""
        source = self.get(source_id)
        with drain_span():  # workload telemetry (W1.1): the fork's source drain (no-op off a request's controller step)
            self.drain(source)
        with self._lock, source.lock:
            if session_id in self._sessions:
                raise StateError(f"fork target {session_id!r} already exists")
            node_ref = self._resolve_checkpoint(source, checkpoint)
            nodes = self._archive.path(node_ref)[1:]
            if check is not None:
                check(timeline_of(nodes))
            # Freeze F5 (cut 7): no separate store lookup of an archived target (``head``: a transaction listing every
            # lineage, ~3 ms per fork): the archive's fork refuses an existing target atomically, before writing
            try:
                self._archive.fork(node_ref, session_id)  # the target holds the source path's raw entries (R1)
            except StateError:
                if self._archive.head(session_id) is not None:
                    raise StateError(f"fork target {session_id!r} already exists") from None
                raise
            state = SGLangSessionState(session_id, source.root_ref, node_ref, source.conditioning_digest,
                                       dict(source.identity), _chunks(nodes), "paused",
                                       chunk_table=_chunk_table(nodes), timeline=timeline_of(nodes))
            state.log("forked", source=source_id, node=node_ref)
            source.log("fork_created", target=session_id, node=node_ref)
            self._sessions[session_id] = state
            return state

    def rewind(self, session_id: str, checkpoint: Any, *, check=None) -> SGLangSessionState:
        """Move the session head back to one of its own clean boundaries (published first; compare-and-swap).
        ``check(timeline)`` may reject the checkpoint's path before the head moves. Tree mode (W1b): to any node the
        session owns (its head moves off the current branch; that branch stays owned and its raw entries held)."""
        state = self.get(session_id)
        with state.lock:
            if state.status == "running":
                raise PendingError(f"cannot rewind running session {session_id!r}")
        if self.switch_mode == "detach":  # W1.3: wait only for the target; the old branch publishes on its own
            return self._rewind_detach(state, checkpoint, check=check)
        with drain_span():  # workload telemetry (W1.1): the rewind's (switch) drain
            drain_seconds = self.drain(state)
        with state.lock:
            node_ref = self._resolve_checkpoint(state, checkpoint)
            ancestor = node_ref == state.head_ref or node_ref in {node.ref for node in self._archive.path(state.head_ref)}
            if not ancestor and not (self.tree_mode and self._owns(state, node_ref)):
                raise StateError(f"checkpoint {checkpoint!r} of session {session_id!r} is not an ancestor of its "
                                 "head (a rewind moves the head back along its own path); fork from it instead")
            if check is not None:
                check(timeline_of(self._archive.path(node_ref)[1:]))
            if node_ref != state.head_ref:
                tree_hold = self._tree_hold(state)  # W1b: before the head leaves this branch (no-op in head mode)
                self._archive.release(session_id)  # the head moves outside the pipeline
                self._archive.rewind(session_id, node_ref)  # later raw entries detach (R1)
                state.revision += 1
                self._bridge.drop_resident(session_id)
            else:
                tree_hold = None
            state.last_switch = {"from_head": state.head_ref, "to": node_ref, "ancestor": ancestor,
                                 "drain_seconds": drain_seconds, "tree": self.tree_mode, "raw_tier_hold": tree_hold}
            state.log("rewound", from_head=state.head_ref, to=node_ref,
                      **({"ancestor": ancestor, "drain_seconds": drain_seconds} if self.tree_mode else {}))
            state.head_ref = node_ref
            nodes = self._archive.path(node_ref)[1:]
            state.generated_chunks = _chunks(nodes)
            state.chunk_table = _chunk_table(nodes)
            state.timeline = timeline_of(nodes)
            state.status = "paused"
            return state

    # -- W1.3: drain-free rewind (``tree_switch="detach"``) ----------------------------------------------------------
    def _rewind_detach(self, state: SGLangSessionState, checkpoint: Any, *, check=None) -> SGLangSessionState:
        """``tree_switch="detach"`` (tree mode): wait only for the target's own commit (a published target: nothing),
        hand the session's pending commits to a detached branch -- they keep publishing in order to its temporary ref
        with their tree refs and sidecars, never moving the session head -- and move the head at once
        (compare-and-swap). The request's ``drain`` span and ``drain_seconds`` are the target wait. When the archive
        cannot detach, the switch drains as ``drain`` does (``switch_fallback`` with its reason; counted)."""
        session_id = state.session_id
        self.reconcile(state)  # an attached archival failure surfaces first, as drain's first step does
        with drain_span():
            wait_seconds = self._await_target(state, checkpoint)
        with state.lock:
            node_ref = (self._published_checkpoint(state, checkpoint) if type(checkpoint) is int
                        else self._resolve_checkpoint(state, checkpoint))
            ancestor = node_ref == state.head_ref or node_ref in {node.ref for node in self._archive.path(state.head_ref)}
            if not ancestor and not self._owns(state, node_ref):
                raise StateError(f"checkpoint {checkpoint!r} of session {session_id!r} is not an ancestor of its "
                                 "head (a rewind moves the head back along its own path); fork from it instead")
            if check is not None:
                check(timeline_of(self._archive.path(node_ref)[1:]))
            switching = bool(state.pending) or node_ref != state.head_ref
        tree_hold, moved, fallback = None, [], None
        if switching:
            try:
                moved = self._detach(state, node_ref)
            except _DetachUnavailable as exc:  # nothing moved: the switch drains instead
                fallback = str(exc)
                with self._switch_lock:
                    self.switch_stats_["fallbacks"] += 1
                    self.switch_stats_["fallback_reasons"] = (self.switch_stats_["fallback_reasons"] + [fallback])[-16:]
                with drain_span():
                    wait_seconds += self.drain(state)
            # (R2) after the hand-off or the fallback drain -- the publications that completed meanwhile (the cut)
            # are owned now -- and before the head leaves this branch (as drain's switch)
            tree_hold = self._tree_hold(state)
        with state.lock:
            if switching:
                self._archive.release(session_id)  # nothing of the session is in flight in the archive any more
                self._archive.rewind(session_id, node_ref)  # compare-and-swap; the target path's raw entries held
                state.revision += 1
                self._bridge.drop_resident(session_id)
            state.last_switch = {"from_head": state.head_ref, "to": node_ref, "ancestor": ancestor,
                                 "drain_seconds": wait_seconds, "tree": True, "raw_tier_hold": tree_hold,
                                 "switch_mode": "detach", "detached_at_switch": len(moved),
                                 "detached_branch": state.detached[-1]["branch"] if moved else None,
                                 "switch_fallback": fallback is not None, "switch_fallback_reason": fallback}
            state.log("rewound", from_head=state.head_ref, to=node_ref, ancestor=ancestor, drain_seconds=wait_seconds,
                      switch_mode="detach", detached=len(moved), **({"fallback": fallback} if fallback else {}))
            state.head_ref = node_ref
            nodes = self._archive.path(node_ref)[1:]
            state.generated_chunks = _chunks(nodes)
            state.chunk_table = _chunk_table(nodes)
            state.timeline = timeline_of(nodes)
            state.status = "paused"
            return state

    def _await_target(self, state: SGLangSessionState, checkpoint: Any) -> float:
        """The detach-mode target wait. A node reference or a named checkpoint is published (a pending commit has no
        ref): no wait. A chunk count waits through the pending commit that completes it on the current branch (the
        commits after it are detached). No checkpoint (the generated head) drains. A target whose commit fails fails
        the rewind with a clear error: it is never retargeted."""
        if checkpoint is None:
            return self.drain(state)
        if type(checkpoint) is not int:
            return 0.0
        with state.lock:
            items = list(state.pending)
        through = None
        for item in items:
            if item["kind"] == "chunk" and item.get("chunk") is not None and item["chunk"] >= checkpoint:
                break
            through = item["node_id"]
        if through is None:
            return 0.0
        started = time.perf_counter()
        try:
            self._archive.drain(state.session_id, through=through)
        except PendingError:
            raise
        except BaseException as exc:  # noqa: BLE001 - the target's commit (or an ancestor's) failed
            failure = self.acknowledge(state, exc)
            raise StateError(f"rewind target {checkpoint!r} of session {state.session_id!r} failed to publish "
                             f"({failure}); a rewind is never retargeted") from exc
        self.reconcile(state)
        return time.perf_counter() - started

    def _published_checkpoint(self, state: SGLangSessionState, checkpoint: int) -> str:
        """A chunk count resolved on the published head path (``_resolve_checkpoint`` without its drain): the newest
        node at which the session had generated that many chunks, never a partial epoch node."""
        path = self._archive.path(state.head_ref)
        partial = partial_epoch_nodes(path)
        boundaries, chunks = {}, 0
        for node in path:
            chunks += node.kind == "chunk"
            if node.ref not in partial:
                boundaries[chunks] = node.ref
        if checkpoint not in boundaries:
            raise ConfigurationError(f"session {state.session_id!r} has {max(boundaries)} chunks; "
                                     f"no checkpoint after {checkpoint}")
        return boundaries[checkpoint]

    def _detach(self, state: SGLangSessionState, target: str) -> list[str]:
        """Hand the session's pending commits to a new detached branch (the archive's ``detach``): they keep publishing
        in order to its temporary ref. Returns their node ids ([]: none in flight). ``_DetachUnavailable`` when the
        archive cannot (nothing moved)."""
        import secrets

        from ...errors import IntegrityError
        from .nodes import detached_branch

        with state.lock:
            if not state.pending:
                return []
            state.detach_sequence += 1
            branch = detached_branch(state.session_id, state.detach_sequence, secrets.token_hex(4))
        detach = getattr(self._archive, "detach", None)
        if not callable(detach):
            raise _DetachUnavailable(f"archive {getattr(self._archive, 'name', '?')!r} cannot detach a branch")
        from ...archive_pipeline import DetachRolledBack

        try:
            result = detach(state.session_id, branch)
        except DetachRolledBack as exc:  # before the archive's commit point, rolled back: nothing moved (F4, R1)
            raise _DetachUnavailable(str(exc)) from exc
        moved, cut = list(result["moved"]), result.get("cut")
        # Commits that published to the session head before the detach are attached publications.
        self.reconcile(state, surface=False, stop_at=set(moved))
        with state.lock:
            items = list(state.pending)
            if [item["node_id"] for item in items] != moved:  # after the commit point: never a fallback (F4)
                raise IntegrityError(f"session {state.session_id!r}: the archive detached {moved} but the controller "
                                     f"tracks {[item['node_id'] for item in items]} as pending")
            if not items:
                return []
            state.pending.clear()
            state.detached.append({"branch": branch, "items": items, "from_head": state.head_ref, "to": target,
                                   "cut": cut, "opened": time.time(), "published": 0, "failed": 0, "events": []})
            state.log("detached", branch=branch, commits=moved, to=target, cut=cut)
        with self._switch_lock:
            stats = self.switch_stats_
            stats["branches_opened"] += 1
            stats["commits_detached"] += len(moved)
            stats["backlog"] += len(moved)
            stats["backlog_peak"] = max(stats["backlog_peak"], stats["backlog"])
        return moved

    def _reconcile_detached(self, state: SGLangSessionState, *, close_timeout: float | None = 0,
                            strict: bool = False) -> None:
        """Resolve detached commits in order (never waits on publication). A published one is owned as an attached
        publication is (registry, sidecar; its tree ref was set in its publishing transaction) and its raw entries stay
        in the LRU under the session's tree holder. A failed one -- its descendants on that branch fail with it -- is
        recorded (event, bounded ``detach_failures``), never raised to the live session. A branch whose commits all
        resolved is closed (``_close_branch``; ``close_timeout``: the wait for its retirement). Fail closed (F3): a
        branch stays tracked until its close succeeds (retried by later reconciles: ``release_idle``, ``checkpoint``,
        close); ``strict`` raises when one is left open."""
        published, closing, resolved = [], [], 0
        with state.lock:
            for branch in state.detached:
                while branch["items"]:
                    item = branch["items"][0]
                    ticket = item["ticket"]
                    if not ticket.done():
                        break
                    branch["items"].pop(0)
                    resolved += 1
                    error = getattr(ticket, "error", None)
                    if error is None:
                        ref = ticket.result(0)
                        branch["published"] += 1
                        if item.get("tree") is not None:
                            state.tree[item["tree"]["generation_index"]] = ref
                        published.append((item, ref))
                        state.log("published", node_id=item["node_id"], node=ref, branch=branch["branch"],
                                  lag_seconds=time.time() - item["submitted"])
                    else:
                        branch["failed"] += 1
                        event = {"branch": branch["branch"], "node_id": item["node_id"], "chunk": item.get("chunk"),
                                 "generation_index": (item.get("tree") or {}).get("generation_index"),
                                 "request_id": item.get("request_id"), "error": repr(error)}
                        branch["events"] = (branch["events"] + [event])[-DETACH_FAILURES_KEPT:]
                        state.detach_failures[f"{branch['branch']}/{item['node_id']}"] = event
                        while len(state.detach_failures) > DETACH_FAILURES_KEPT:  # bounded (F6)
                            state.detach_failures.pop(next(iter(state.detach_failures)))
                        state.log("detached_commit_failed", **event)
                if not branch["items"]:
                    closing.append(branch)
        if resolved:
            with self._switch_lock:
                stats = self.switch_stats_
                stats["commits_published"] += len(published)
                stats["commits_failed"] += resolved - len(published)
                stats["backlog"] -= resolved
        tier = getattr(self._archive, "raw_tier", None)
        for item, ref in published:  # provenance and raw-entry ownership, outside the hashed node content
            try:
                telemetry = item["telemetry"]
                if item.get("tree") is not None:
                    telemetry = {**telemetry, "tree": self._tree_sidecar(state, ref, item["tree"])}
                self._archive.annotate(ref, telemetry)
            except Exception as exc:  # noqa: BLE001 - provenance only, never fails a publication
                with state.lock:
                    state.log("telemetry_failed", node_id=item["node_id"], error=repr(exc))
            if tier is not None:  # its own error path (F9): the owned raw copy never depends on the annotation
                try:
                    tier.hold(state.session_id + TREE_HOLDER_SUFFIX, [ref], exclusive=False)
                except Exception as exc:  # noqa: BLE001 - the raw copy is a cache: the node restores from the store
                    with state.lock:
                        state.log("tree_hold_failed", node_id=item["node_id"], node=ref, error=repr(exc))
        if closing:  # (R2) the tree holder takes the nodes the branch's publications own (pieces, the cut) first
            self._tree_hold(state)
        failures = []
        for branch in closing:
            self._hold_branch_path(state, branch["branch"])  # (R2b) and its head's whole path, before its holder goes
            try:
                closed = self._close_branch(state.session_id, branch["branch"], timeout=close_timeout)
            except PendingError as exc:  # its retirement is still running: a later reconcile closes it
                failures.append((branch, exc))
                continue
            except Exception as exc:  # noqa: BLE001 - fail closed: tracked (its ref stays) and retried later
                failures.append((branch, exc))
                with state.lock:
                    branch["close_error"] = repr(exc)
                    state.log("detached_close_failed", branch=branch["branch"], error=repr(exc))
                with self._switch_lock:
                    self.switch_stats_["close_failures"] = self.switch_stats_.get("close_failures", 0) + 1
                continue
            with state.lock:
                removed = branch in state.detached  # a concurrent reconcile may have closed it (idempotent close)
                if removed:
                    state.detached.remove(branch)
                    state.log("detached_closed", branch=branch["branch"], published=branch["published"],
                              failed=branch["failed"], kept=closed.get("kept"))
            if removed:
                with self._switch_lock:
                    self.switch_stats_["branches_closed"] += 1
        if strict:
            with state.lock:
                left = [branch["branch"] for branch in state.detached]
            if left:
                cause = failures[0][1] if failures else None
                raise StateError(f"session {state.session_id!r}: detached branches {left} are not closed "
                                 f"({'commits in flight' if cause is None else repr(cause)})") from cause

    def _close_branch(self, session_id: str, branch: str, *, timeout: float | None = 0) -> dict[str, Any]:
        """Close a detached branch once quiescent (``PendingError`` before; idempotent): its temporary ref, bindings
        and raw-tier holder go; its last published node stays rooted under a hidden keep ref
        (``sglang/detached-keep/<session>/<branch>``, removed with the whole tree) unless a tree ref of the session
        names it (F2: e.g. a published recache whose chunk failed)."""
        from .nodes import detached_keep_ref

        store = self._archive.store

        def keep(head: str) -> str | None:
            return None if head in set(store.list_refs(tree_prefix(session_id)).values()) else \
                detached_keep_ref(session_id, branch)

        return self._archive.close_detached(branch, timeout=timeout, keep=keep)

    def drain_detached(self, state: SGLangSessionState, *, timeout: float | None = None) -> float:
        """Wait until every detached commit of the session resolved and close its branches (``checkpoint``,
        ``close_session``, the bridge's ``drain_all`` before garbage collection); a detached commit failure is
        recorded, never raised; a branch left open raises (fail closed, F3/F6). Returns seconds waited."""
        return self._drain_detached(state, timeout=timeout, strict=True)

    def detached_sessions(self) -> list[str]:
        """Open sessions with detached branches not closed yet."""
        with self._lock:
            return sorted(session_id for session_id, state in self._sessions.items() if state.detached)

    def _drain_detached(self, state: SGLangSessionState, *, timeout: float | None = None,
                        strict: bool = False) -> float:
        started = time.perf_counter()
        deadline = None if timeout is None else time.monotonic() + timeout
        with state.lock:
            tickets = [item["ticket"] for branch in state.detached for item in branch["items"]]
        for ticket in tickets:  # each resolves: published, or failed (with its branch tail)
            remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
            try:
                ticket.result(remaining)
            except PendingError:
                raise
            except BaseException:  # noqa: BLE001 - a failed detached commit: recorded by the reconcile below
                pass
        remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
        self._reconcile_detached(state, close_timeout=remaining, strict=strict)
        return time.perf_counter() - started

    def _abandon_detached(self, state: SGLangSessionState) -> list[str]:
        """Abort every detached branch of the session (delete, replace, aborted close): their unpublished commits are
        retired (recorded), the branches closed. Fail closed (F3, R3): every branch is tried, then any failure -- an
        abort that raised (even if its branch closed after all), a branch not quiescent or not closed -- raises, an open
        branch stays tracked, and the caller must not delete or recreate the session (a retry repeats this
        idempotently). Returns the retired node ids."""
        pending, errors = [], []
        for branch in list(state.detached):
            pending.extend(branch["items"])
            try:
                if not self._archive.abandon_detached(branch["branch"]).get("quiesced", True):
                    raise PendingError(f"detached branch {branch['branch']!r} is not quiescent")
            except Exception as exc:  # noqa: BLE001 - raised below, after the other branches
                errors.append(exc)
                with state.lock:
                    state.log("detached_abandon_failed", branch=branch["branch"], error=repr(exc))
        try:
            self._reconcile_detached(state, close_timeout=0 if errors else None, strict=True)
        except StateError:
            if not errors:
                raise
            raise StateError(f"session {state.session_id!r}: detached branches were not aborted "
                             f"({errors[0]!r})") from errors[0]
        if errors:  # (R3) even when every branch closed after all: the caller retries (then nothing is left to abort)
            raise StateError(f"session {state.session_id!r}: aborting its detached branches failed "
                             f"({errors[0]!r}); retry") from errors[0]
        return [item["node_id"] for item in pending
                if not (item["ticket"].done() and getattr(item["ticket"], "error", None) is None)]

    @staticmethod
    def _failed_detached_note(state: SGLangSessionState) -> str:
        """The failed detached commits of a session, for an error about a target that does not exist."""
        if not state.detach_failures:
            return ""
        failed = [f"{event['node_id']} (generation {event['generation_index']}, branch {event['branch']})"
                  for event in list(state.detach_failures.values())[-8:]]
        return f"; these detached commits failed and were never published: {', '.join(failed)}"

    def switch_fields(self, action: Any, state: SGLangSessionState) -> dict[str, Any]:
        """W1.3 per-request fields (``tree_switch="detach"`` only; drain mode adds none, F7): ``switch_mode``; a
        rewind adds what its switch did."""
        fields: dict[str, Any] = {"switch_mode": self.switch_mode}
        if action == "rewind":
            with state.lock:
                switch = dict(state.last_switch or {})
            fields.update(detached_at_switch=int(switch.get("detached_at_switch") or 0),
                          switch_fallback=bool(switch.get("switch_fallback", False)),
                          switch_fallback_reason=switch.get("switch_fallback_reason"))
        return fields

    def switch_stats(self) -> dict[str, Any]:
        """W1.3 bridge stats of the tree switch: detached branches and commits, the detached backlog, fallbacks."""
        with self._switch_lock:
            stats = dict(self.switch_stats_)
        return {"mode": self.switch_mode,
                "detached_branches": {"opened": stats["branches_opened"], "closed": stats["branches_closed"]},
                "detached_commits": {"detached": stats["commits_detached"], "published": stats["commits_published"],
                                     "failed": stats["commits_failed"]},
                "detached_backlog": {"current": stats["backlog"], "peak": stats["backlog_peak"]},
                "switch_fallbacks": stats["fallbacks"], "fallback_reasons": list(stats["fallback_reasons"]),
                "close_failures": stats.get("close_failures", 0),
                "stale_refs_removed": list(stats["stale_refs_removed"])}

    def _retire(self, state: SGLangSessionState) -> list[str]:
        """Abort: retire the session's unpublished commits; its published head stays authoritative."""
        head = self._archive.abandon(state.session_id) if self._archive.live(state.session_id) else None
        if head is None:
            head = self._archive.head(state.session_id) or state.head_ref
        nodes = self._archive.path(head)[1:]
        with state.lock:
            retired = [item["node_id"] for item in state.pending
                       if not (item["ticket"].done() and getattr(item["ticket"], "error", None) is None)]
            state.pending.clear()
            state.head_ref = head
            state.generated_chunks = _chunks(nodes)
            state.chunk_table = _chunk_table(nodes)
            state.timeline = timeline_of(nodes)
            state.revision += 1
            state.log("aborted", retired=retired, head=head)
        if self.tree_mode:  # W1b: as in ``acknowledge``
            self._unbind(state.session_id)
            registry = self.tree(state.session_id)
            with state.lock:
                state.tree_pending.clear()
                state.tree = registry
            self._tree_hold(state)
        self._raw_hold(state.session_id, head)  # retired commits' entries are dropped; the published path stays
        self._bridge.drop_resident(state.session_id)
        return retired

    def close_session(self, session_id: str, *, delete: bool = False, abort: bool = False) -> dict[str, Any]:
        """Close a session (it must not be running) and return a report.

        Its pending commits are published first, or retired with ``abort`` (the published head stays the
        authoritative checkpoint); its resident native KV and pipeline state are released. ``delete`` also
        removes its head ref and its named checkpoints: garbage collection then reclaims every object that no
        other session or checkpoint reaches. An archival failure is raised after the session is closed."""
        failure = None
        report: dict[str, Any] = {"session_id": session_id, "was_open": False, "retired": [], "deleted": False,
                                  "checkpoints_deleted": {}}
        with self._lock:
            state = self._sessions.get(session_id)
            if state is None and self._archive.head(session_id) is None:
                raise StateError(f"unknown causal KV session {session_id!r}")
            if state is not None:
                report["was_open"] = True
                with state.lock:
                    if state.status == "running":
                        raise PendingError(f"cannot close running session {session_id!r}")
                if abort or delete:
                    report["retired"] = self._retire(state)
                else:
                    try:
                        self.drain(state)
                    except BaseException as exc:  # noqa: BLE001 - raised once the session is closed
                        failure = exc
                if state.detached:  # W1.3: detached branches publish first (close) or are aborted (abort, delete); a
                    # branch left open raises before anything is closed or deleted (fail closed, F3)
                    if abort or delete:
                        report["detached_retired"] = self._abandon_detached(state)
                    else:
                        report["detached_wait_seconds"] = self._drain_detached(state, strict=True)
                with state.lock:
                    state.status = "closed"
                    state.log("closed", deleted=delete, aborted=abort)
                    report.update(head=state.head_ref, generated_chunks=state.generated_chunks)
                del self._sessions[session_id]
            self._bridge.drop_resident(session_id)
            if self._archive.live(session_id):
                if abort or delete:
                    self._archive.abandon(session_id)
                else:
                    self._archive.release(session_id)
            if "head" not in report:
                report["head"] = self._archive.head(session_id)
            if delete:
                report["checkpoints_deleted"] = self.unpin_all(session_id)
                tree = self._delete_tree(session_id)  # W1b: whole-tree delete (no ownership refs in head mode)
                if tree or self.tree_mode:
                    report["tree_deleted"] = tree
                self._archive.delete(session_id)
                report["deleted"] = True
            # After the drain or cancel (and delete): the session's raw-tier entries detach; those no other
            # session holds are dropped (no tier memory outlives a closed session, I5). wl-w1b7: with
            # raw_tier_retain="lru" a close keeps them as unheld cache entries; a whole-tree delete drops them and the
            # unheld entries the session's holders held last.
            report["raw_tier"] = self._raw_release(session_id, drop=delete)
            if self.tree_mode:  # W1b: the owned-node holder too
                report["tree_raw_tier"] = self._raw_release(session_id + TREE_HOLDER_SUFFIX, drop=delete)
            if delete:
                self._raw_tier_call("raw_forget", session_id)
        if failure is not None:
            raise failure
        return report

    def running(self) -> list[str]:
        with self._lock:
            return [session_id for session_id, state in self._sessions.items() if state.status == "running"]


class _DetachUnavailable(Exception):
    """W1.3: the archive could not detach the session's pending commits (nothing moved); the switch drains."""


#: W1.3 (F6): failed detached commits kept per session (and per branch) for errors and events.
DETACH_FAILURES_KEPT = 256


__all__ = ["SGLangSessionController", "SGLangSessionState", "STATUSES", "timeline_of"]
