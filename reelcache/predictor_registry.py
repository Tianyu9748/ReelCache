"""Model-scoped online rank predictors with immutable, pinnable revisions.

The paper's log-linear tail predictor ``ln r = a + b x`` with
``x = ln((1 - E_m) / (1 - E_t))`` is learned online with the 2x2 ridge update
of :class:`reelcache.types.OnlineRidgeState`. A :class:`RankPredictorRegistry` is a
mutable, process-local resource owner: it retains learners across compatible
prompts and branches (keyed by :class:`reelcache.types.PredictorKey`), serializes
updates, and publishes immutable revisions. Commits predict from a pinned
:class:`RegistryView` and report their verified observations afterwards, so a
chunk's predictions never depend on the order in which its own fits finish.

Observation identities bind the learner scope and the committed source, which
makes retried and replayed updates idempotent. A qualification replay registry
additionally enforces one recorded update order. Restoring archived nodes never
calls the registry: published labels are not re-learned.

A source is learned once. Observations carry their commit context
(:func:`observation_origin`: lineage, commit, pinned registry revision). A
repeated identity with the same payload is a no-op; with another payload it is
an integrity error when it comes from the same origin (a retried publish or a
replay must reproduce its payload) and a skipped duplicate when it comes from
another origin -- another session or lineage committing byte-identical content
(the label of a fit depends on the prediction it started from, e.g. the top-k
subspace size) or a re-commit after the learners changed: the first applied
payload wins (``stats["divergent_duplicates"]``).
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

from .errors import ConfigurationError, IntegrityError, StateError
from .types import (
    RANK_BUCKET, FitObservation, OnlineRidgeState, PredictorKey, _text, canonical_digest,
)


REGISTRY_SCHEMA = "reelcache.rank_predictor_registry.v1"


def observation_origin(lineage: str, kind: str, node_id: str, view: "RegistryView") -> str:
    """The commit context of a fit's observations: the lineage, the commit and the pinned registry revision whose
    predictions the fit started from. Identical for a retried or faithfully replayed commit; different for another
    session/lineage or after the learners changed."""
    return canonical_digest({"schema": "reelcache.observation_origin.v1", "lineage": lineage, "commit": kind,
                             "node_id": node_id, "registry": view.registry_id, "view": view.digest})


def _repeat(known: tuple[str, str | None], item: FitObservation) -> bool:
    """For an already applied identity: True when ``item`` is a duplicate to skip, else IntegrityError."""
    payload, origin = known
    if payload == item.payload_digest:
        return True
    if origin is not None and item.origin is not None and origin != item.origin:
        return True
    raise IntegrityError("conflicting observation payload for an applied source identity")


@dataclass(frozen=True)
class PredictorUse:
    """The exact learner revision, coefficients and prediction a fit used."""

    registry_id: str
    sequence: int
    key: PredictorKey
    state: OnlineRidgeState
    x: float
    log_rank: float
    predicted_rank: float

    @property
    def revision_ref(self) -> str:
        return (f"reelcache-rank-predictor:{self.registry_id}:{self.key.digest[:16]}"
                f":r{self.state.revision}:{self.state.digest[:16]}")

    def to_dict(self) -> dict[str, Any]:
        a, b = self.state.coefficients()
        return {
            "schema": "reelcache.predictor_use.v1", "registry_id": self.registry_id,
            "sequence": self.sequence, "key": self.key.to_dict(), "state": self.state.to_dict(),
            "a": a, "b": b, "x": self.x, "log_rank": self.log_rank,
            "predicted_rank": self.predicted_rank if math.isfinite(self.predicted_rank) else None,
            "predicted_rank_overflow": not math.isfinite(self.predicted_rank),
            "revision_ref": self.revision_ref,
        }


class RegistryView:
    """An immutable published registry revision used for a commit's predictions."""

    def __init__(self, registry_id: str, sequence: int, states: Mapping[PredictorKey, OnlineRidgeState],
                 anchor_rank: int, prior_strength: float, speculative: Sequence[str] = ()):
        self.registry_id = registry_id
        self.sequence = sequence
        self._states = MappingProxyType(dict(states))
        self.anchor_rank = anchor_rank
        self.prior_strength = prior_strength
        # Observation identities of unpublished ancestor commits folded into this view (pipelining).
        self.speculative = tuple(speculative)

    @property
    def states(self) -> Mapping[PredictorKey, OnlineRidgeState]:
        return self._states

    def state(self, key: PredictorKey) -> OnlineRidgeState:
        if not isinstance(key, PredictorKey):
            raise ConfigurationError("predictor lookups require a PredictorKey")
        state = self._states.get(key)
        return OnlineRidgeState.initial(self.anchor_rank, self.prior_strength) if state is None else state

    def predict(self, key: PredictorKey, x: float) -> PredictorUse:
        if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x):
            raise ConfigurationError("prediction requires a finite anchor feature")
        state = self.state(key)
        return PredictorUse(self.registry_id, self.sequence, key, state, float(x),
                            state.log_rank(x), state.predict(x))

    @property
    def digest(self) -> str:
        return canonical_digest({
            "registry_id": self.registry_id, "sequence": self.sequence,
            "states": sorted([key.digest, state.digest] for key, state in self._states.items()),
        })


class RankPredictorRegistry:
    """Mutable, thread-safe owner of shared model-scoped online learners.

    ``mode="shared"`` accepts observations in arrival order (other sessions may
    interleave). ``mode="replay"`` is the qualification mode: it is built from a
    recorded snapshot and the exact recorded observation order, and rejects any
    deviating or extra update; predictions then depend only on that order.
    """

    def __init__(self, registry_id: str, *, anchor_rank: int = RANK_BUCKET, prior_strength: float = 2.0,
                 mode: str = "shared", expected_order: Sequence[str] | None = None) -> None:
        _text(registry_id, "registry id")
        if type(anchor_rank) is not int or anchor_rank < 1:
            raise ConfigurationError("anchor rank must be a positive integer")
        OnlineRidgeState.initial(anchor_rank, prior_strength)
        if mode not in ("shared", "replay"):
            raise ConfigurationError("registry mode must be shared or replay")
        if (mode == "replay") != (expected_order is not None):
            raise ConfigurationError("only replay registries take a recorded update order")
        self.registry_id = registry_id
        self.anchor_rank = anchor_rank
        self.prior_strength = float(prior_strength)
        self.mode = mode
        self._lock = threading.Lock()
        self._states: dict[PredictorKey, OnlineRidgeState] = {}
        self._applied: dict[str, str] = {}
        # Commit context (observation_origin) of each applied identity; None when unknown (legacy snapshots).
        self._origins: dict[str, str | None] = {}
        self._log: list[str] = []
        self._sequence = 0
        self._expected = None if expected_order is None else tuple(expected_order)
        if self._expected is not None and (
            len(set(self._expected)) != len(self._expected)
            or any(not isinstance(item, str) or len(item) != 64 for item in self._expected)
        ):
            raise ConfigurationError("replay order must list unique observation identities")
        self.stats = {"applied": 0, "duplicates": 0, "rejected": 0, "divergent_duplicates": 0}

    @property
    def sequence(self) -> int:
        with self._lock:
            return self._sequence

    @property
    def log(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(self._log)

    def pin(self) -> RegistryView:
        """Publish the current immutable revision for a commit's predictions."""
        with self._lock:
            return RegistryView(self.registry_id, self._sequence, self._states,
                                self.anchor_rank, self.prior_strength)

    def state(self, key: PredictorKey) -> OnlineRidgeState:
        return self.pin().state(key)

    def pin_with(self, pending: Iterable[FitObservation]) -> RegistryView:
        """The pinned revision plus a lineage's own verified but unpublished observations.

        An ordered archive pipeline plans chunk ``n`` before chunks ``< n`` are
        published; folding their observations (in order, skipping those already
        applied) gives the predictions a synchronous commit sequence would make
        in the same session. Nothing is applied to the shared learners: the
        observations are applied by ``observe`` when their commits publish.
        Replay registries require the pending observations to follow the
        recorded order.
        """
        batch = list(pending)
        for item in batch:
            if not isinstance(item, FitObservation):
                raise ConfigurationError("registries learn only from FitObservation records")
            if item.observed_rank <= self.anchor_rank:
                raise ConfigurationError("only verified fits needing more than the anchor rank are observations")
        with self._lock:
            states = dict(self._states)
            applied = {identity: (payload, self._origins.get(identity)) for identity, payload in self._applied.items()}
            sequence, position = self._sequence, len(self._log)
            speculative = []
            for item in batch:
                identity, payload = item.observation_id, item.payload_digest
                known = applied.get(identity)
                if known is not None:
                    _repeat(known, item)  # skipped (same payload, or first writer from another origin) or raises
                    continue
                if self._expected is not None and (
                    position >= len(self._expected) or self._expected[position] != identity
                ):
                    raise StateError("replay registry received a pending observation out of its recorded order")
                state = states.get(item.key, OnlineRidgeState.initial(self.anchor_rank, self.prior_strength))
                states[item.key] = state.updated(item.x, item.observed_rank)
                applied[identity] = (payload, item.origin)
                speculative.append(identity)
                sequence += 1
                position += 1
            return RegistryView(self.registry_id, sequence, states, self.anchor_rank, self.prior_strength,
                                speculative)

    def observe(self, observations: Iterable[FitObservation]) -> tuple[bool, ...]:
        """Apply verified observations in the given order; returns per-item applied flags.

        A repeated identity with an identical payload is a no-op (retry/replay).
        With another payload it raises IntegrityError when both observations
        come from the same origin (or carry none), and is skipped -- the first
        applied payload wins -- when they come from different origins
        (``stats["divergent_duplicates"]``). The batch is validated completely
        before any learner changes.
        """
        batch = list(observations)
        for item in batch:
            if not isinstance(item, FitObservation):
                raise ConfigurationError("registries learn only from FitObservation records")
            if item.observed_rank <= self.anchor_rank:
                raise ConfigurationError("only verified fits needing more than the anchor rank are observations")
        with self._lock:
            applied, pending, divergent = [], {}, 0
            for item in batch:
                identity, payload = item.observation_id, item.payload_digest
                known = pending.get(identity)
                if known is None and identity in self._applied:
                    known = (self._applied[identity], self._origins.get(identity))
                if known is not None:
                    try:
                        _repeat(known, item)
                    except IntegrityError:
                        self.stats["rejected"] += 1
                        raise
                    divergent += known[0] != payload
                    applied.append(False)
                    continue
                pending[identity] = (payload, item.origin)
                applied.append(True)
            new = [item for item, flag in zip(batch, applied) if flag]
            if self._expected is not None:
                start = len(self._log)
                order = tuple(item.observation_id for item in new)
                if self._expected[start:start + len(order)] != order:
                    self.stats["rejected"] += 1
                    raise StateError("replay registry received an observation out of its recorded order")
            for item in new:
                state = self._states.get(item.key, OnlineRidgeState.initial(self.anchor_rank, self.prior_strength))
                self._states[item.key] = state.updated(item.x, item.observed_rank)
                self._applied[item.observation_id] = item.payload_digest
                self._origins[item.observation_id] = item.origin
                self._log.append(item.observation_id)
                self._sequence += 1
            self.stats["applied"] += len(new)
            self.stats["duplicates"] += len(batch) - len(new)
            self.stats["divergent_duplicates"] += divergent
            return tuple(applied)

    def to_dict(self) -> dict[str, Any]:
        with self._lock:
            return {
                "schema": REGISTRY_SCHEMA, "registry_id": self.registry_id,
                "anchor_rank": self.anchor_rank, "prior_strength": self.prior_strength,
                "sequence": self._sequence,
                "learners": [
                    {"key": key.to_dict(), "state": state.to_dict()}
                    for key, state in sorted(self._states.items(), key=lambda item: item[0])
                ],
                "applied": [[identity, self._applied[identity]]
                            + ([] if self._origins.get(identity) is None else [self._origins[identity]])
                            for identity in self._log],
            }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], *, mode: str = "shared",
                  expected_order: Sequence[str] | None = None) -> RankPredictorRegistry:
        """Restore a serialized registry; replay mode continues with a recorded order."""
        try:
            if data["schema"] != REGISTRY_SCHEMA:
                raise ConfigurationError("unsupported predictor registry schema")
            registry = cls(data["registry_id"], anchor_rank=data["anchor_rank"],
                           prior_strength=data["prior_strength"], mode=mode, expected_order=expected_order)
            states = {}
            for item in data["learners"]:
                key = PredictorKey.from_dict(item["key"])
                if key in states:
                    raise IntegrityError("duplicate learner in a serialized registry")
                states[key] = OnlineRidgeState.from_dict(item["state"])
            applied, origins = [], {}
            for entry in data["applied"]:
                if not isinstance(entry, (list, tuple)) or len(entry) not in (2, 3):
                    raise IntegrityError("serialized registry observations are [identity, payload(, origin)]")
                applied.append((entry[0], entry[1]))
                if len(entry) == 3:
                    if not isinstance(entry[2], str) or len(entry[2]) != 64:
                        raise IntegrityError("invalid serialized observation origin")
                    origins[entry[0]] = entry[2]
        except (KeyError, TypeError, ValueError) as exc:
            raise IntegrityError("invalid serialized predictor registry") from exc
        if (type(data["sequence"]) is not int or data["sequence"] != len(applied)
                or sum(state.observations for state in states.values()) != len(applied)
                or len({identity for identity, _ in applied}) != len(applied)):
            raise IntegrityError("serialized registry counts disagree with its applied observations")
        if expected_order is not None and tuple(expected_order[:len(applied)]) != tuple(
            identity for identity, _ in applied
        ):
            raise StateError("replay order does not start with the snapshot's applied observations")
        registry._states = states
        registry._applied = dict(applied)
        registry._origins = {identity: origins.get(identity) for identity, _ in applied}
        registry._log = [identity for identity, _ in applied]
        registry._sequence = len(applied)
        return registry
