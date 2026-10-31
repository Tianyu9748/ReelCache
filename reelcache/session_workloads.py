"""Session-tree rewind-and-generate workloads (workload v2): settings, sweeps, feasibility and frozen bundles.

One request operation (``ReelCache/docs/workload_generator/plan.md`` §3-§5): a request selects a session, picks a
source node, supplies a new prompt and generates exactly ONE new native chunk as a child of that node.

* **Node age.** Every node of a session gets a monotonically increasing *generation index* (setup nodes 1..d0, then
  one per request). The latest node is the newest by generation index, never the deepest. ``k = 0`` targets the
  latest node; ``k > 0`` targets the node generated ``k`` session generations ago, wherever it lies in the tree
  (old branches included): ``target = n - k`` with ``n`` the latest index before the request.
* **Two controls.** ``p_historical`` (probability of ``k > 0``) and a positive-gap profile (``uniform``, ``recent``
  = geometric with mean ``recent_mean`` truncated to legal ages, ``far`` = uniform over the upper half of the legal
  ages, ``fixed`` = ``gap_k``). The profile is a distribution over the **legal** positive ages of that request (the
  conditioned support is published per request). Ages are never renumbered, snapped or replaced: an empty legal
  support, an illegal fixed ``k`` or an illegal latest target raises :class:`InfeasibleError`.
* **Legal targets.** A node is a legal source when ``depth >= min_switch_depth`` (LongLive 1.0 rejects a prompt
  switch before chunk 5, so its target depth must be >= 5; 1 for the other models) and ``depth + 1 <= max_depth``.
  One tree set serves every model, so the strictest constraint over the declared models is used and recorded.
* **Service time S** (``service_time_s``) is an input per model: a number, or S(h) by target depth h, in which case
  S is the mean of S(h) over the bundle's actual target depths (trees do not depend on S, so they are planned first).
* **Sessions and arrivals.** Open loop: gamma inter-arrivals with mean ``S / rho`` and coefficient of variation
  ``arrival_cv`` (0 = constant spacing, 1 = Poisson); every arrival goes to a session chosen uniformly among the
  eligible ones (budget left and per-session cooldown met); no eligible session raises. Closed loop: each session
  issues its next request ``think`` seconds after its previous chunk returned (lognormal, mean
  ``think_time_mult * S``, CV ``think_cv``), so only the think times are frozen.
* **Independent RNG streams** (``numpy.random.SeedSequence(seed, spawn_key=(name, ...))``, PCG64, uniform doubles
  only; normal, gamma and lognormal variates are derived here so bundles do not depend on numpy's distribution
  code): ``cohort``, ``arrivals``, ``session_choice``, ``think`` (keyed by session and request ordinal),
  ``targeting`` (keyed by session and request ordinal) and ``content`` (reserved: v2 binding draws nothing). Trees,
  targets and prompts are therefore identical across rates, burstiness, session counts and arrival modes.
* **Setup prefix.** ``d0`` chunks along the main path: chunks 0-4 use main prompt 1 (the LongLive 1.0 rule), chunk
  ``j >= 5`` uses main prompt ``j - 3``. Setup nodes have generation indices 1..d0.
* **Content binding.** A new node continues its target's path. A target on the main storyline whose main-path child
  is still unused binds the next main prompt; every other child binds an authored continuation from a bank keyed by
  ``sha256(session_id, target path prompt ids per chunk, child ordinal)``. Beyond the session's main prompts the
  main storyline needs authored *main extensions* (same keying). This module never invents text: it emits the
  content slots (context, siblings) and :func:`validate_bank` checks authored entries. A bundle is *frozen* only
  when every prompt is bound and the used bank entries validate.

Python standard library and numpy only (no torch, no SGLang). The runner lives in
``artifacts/sglang_integration/benchmarks/interactive_native_v2/``.
"""

from __future__ import annotations

from dataclasses import MISSING, dataclass, field, fields, replace
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from .errors import ConfigurationError
from .synthetic_workloads import normalized_prompt, read_sessions

GENERATOR = "reelcache.session_workloads"
GENERATOR_VERSION = "2026-10-19.1"
SWEEP_SCHEMA = "reelcache.session_workload_sweep.v1"
BUNDLE_SCHEMA = "reelcache.session_workload_bundle.v1"
REPORT_SCHEMA = "reelcache.session_workload_feasibility.v1"
BANK_SCHEMA = "reelcache.session_workload_bank.v1"
BUNDLE_FILES = ("bindings.jsonl", "cohort.json", "config.json", "content_slots.jsonl", "coverage.json",
                "requests.jsonl", "setup.jsonl")
MANIFEST_NAME = "MANIFEST.sha256"
BUNDLE_ID_NAME = "BUNDLE_ID"

#: Chunks 0..PROMPT_HOLD_CHUNKS-1 of the main storyline share main prompt 1 (LongLive 1.0 rejects a switch earlier).
PROMPT_HOLD_CHUNKS = 5
#: Per-model target constraints (``min_switch_depth``: smallest legal source depth; ``max_depth``: largest node depth,
#: None = not bounded). The generator applies the strictest values over the declared models.
DEFAULT_MODELS: dict[str, dict[str, Any]] = {
    "causvid": {"min_switch_depth": 1, "max_depth": None},
    "raven": {"min_switch_depth": 1, "max_depth": None},
    "longlive1": {"min_switch_depth": 5, "max_depth": None},
    "longlive2": {"min_switch_depth": 1, "max_depth": None},
}
#: Service-time hints S(h) in seconds by target depth h (restored history chunks): Step 19 RTX PRO 6000 per-request
#: time to the first new chunk (new chunk + VAE context decode of up to 40 latents), resident (coordinator 2026-10-19
#: 03:30). For tests and examples only -- S is an input per model and the pilot measures it. Linear between points,
#: constant outside them (LongLive 1.0 saturates at h >= 16, LongLive 2.0 at h >= 6).
SERVICE_TIME_HINTS_S: dict[str, dict[int, float]] = {
    "causvid": {1: 0.90, 8: 3.3, 16: 5.3, 32: 6.3},
    "raven": {1: 1.06, 8: 3.6, 16: 5.8, 32: 6.9},
    "longlive1": {1: 0.99, 8: 3.2, 16: 4.7, 60: 4.7},
    "longlive2": {1: 5.5, 6: 14.5},
}
GAP_PROFILES = ("uniform", "recent", "far", "fixed")
ARRIVAL_MODES = ("open", "closed")
SPLIT_NAMES = {"evaluation_candidate": "evaluation_candidate", "evaluation": "evaluation_candidate",
               "eval": "evaluation_candidate", "development": "development", "dev": "development"}
STREAMS = ("cohort", "arrivals", "session_choice", "think", "targeting", "content")
WORD_RANGE = (18, 35)


class InfeasibleError(ConfigurationError):
    """A setting cannot be realized without changing arrivals, targets or content; never silently repaired.

    ``code`` names the rule (e.g. ``empty_support``, ``fixed_k_illegal``, ``latest_illegal``,
    ``no_eligible_session``); ``details`` carries the request context."""

    def __init__(self, code: str, message: str, **details: Any):
        super().__init__(f"{code}: {message}")
        self.code, self.details = code, details

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": str(self), **_plain(self.details)}


# ------------------------------------------------------------------------------------------------------ JSON helpers
def _plain(value: Any) -> Any:
    """JSON-safe plain data (numpy scalars to Python, tuples to lists); refuses NaN/inf."""
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        value = float(value)
        if not math.isfinite(value):
            raise ConfigurationError("non-finite value in workload data")
        return value
    if isinstance(value, Path):
        return str(value)
    return value


def canonical_json(value: Any) -> str:
    """Byte-stable JSON: sorted keys, compact separators, no NaN."""
    return json.dumps(_plain(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _pretty_json(value: Any) -> str:
    return json.dumps(_plain(value), sort_keys=True, indent=1, ensure_ascii=False, allow_nan=False) + "\n"


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# ------------------------------------------------------------------------------------------------------ RNG streams
def stream_word(name: str) -> int:
    """32-bit word of a stream name (the first element of its SeedSequence spawn key)."""
    if name not in STREAMS:
        raise ConfigurationError(f"unknown RNG stream {name!r}; streams are {STREAMS}")
    return int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], "little")


class Uniforms:
    """Uniform doubles in [0, 1) of one named stream: PCG64 seeded by ``SeedSequence(seed, spawn_key=(name, *key))``.

    Only ``Generator.random()`` is used (53-bit doubles from the bit generator, stable across numpy versions); the
    other variates below are derived from it with fixed algorithms."""

    def __init__(self, seed: int, name: str, *key: int):
        if type(seed) is not int or seed < 0:
            raise ConfigurationError("seed must be a nonnegative integer")
        if any(type(item) is not int or item < 0 for item in key):
            raise ConfigurationError("stream keys must be nonnegative integers")
        sequence = np.random.SeedSequence(entropy=seed, spawn_key=(stream_word(name), *key))
        self._generator = np.random.Generator(np.random.PCG64(sequence))
        self._spare: float | None = None
        self.draws = 0

    def next(self) -> float:
        self.draws += 1
        return float(self._generator.random())

    def normal(self) -> float:
        """Standard normal (Box-Muller; the second value of a pair is kept for the next call)."""
        if self._spare is not None:
            value, self._spare = self._spare, None
            return value
        radius = math.sqrt(-2.0 * math.log(1.0 - self.next()))
        angle = 2.0 * math.pi * self.next()
        self._spare = radius * math.sin(angle)
        return radius * math.cos(angle)

    def gamma(self, shape: float) -> float:
        """Gamma(shape, scale 1): Marsaglia-Tsang (shape >= 1); boosted with U**(1/shape) below 1."""
        if not shape > 0:
            raise ConfigurationError("gamma shape must be positive")
        if shape < 1.0:
            boost = (1.0 - self.next()) ** (1.0 / shape)
            return self.gamma(shape + 1.0) * boost
        d = shape - 1.0 / 3.0
        c = 1.0 / math.sqrt(9.0 * d)
        while True:
            x = self.normal()
            v = 1.0 + c * x
            if v <= 0.0:
                continue
            v = v * v * v
            u = 1.0 - self.next()  # (0, 1]
            if u < 1.0 - 0.0331 * x ** 4 or math.log(u) < 0.5 * x * x + d * (1.0 - v + math.log(v)):
                return d * v

    def interarrival(self, mean: float, cv: float) -> float:
        """Gamma inter-arrival with the given mean and coefficient of variation (cv 0 = constant)."""
        if cv == 0:
            return mean
        shape = 1.0 / (cv * cv)
        return mean * cv * cv * self.gamma(shape)

    def lognormal(self, mean: float, cv: float) -> float:
        """Lognormal with the given mean and coefficient of variation (mean 0 -> 0, cv 0 -> the mean)."""
        if mean == 0 or cv == 0:
            return float(mean)
        sigma2 = math.log(1.0 + cv * cv)
        return math.exp(math.log(mean) - 0.5 * sigma2 + math.sqrt(sigma2) * self.normal())


def session_number(session_id: str) -> int:
    if not isinstance(session_id, str) or len(session_id) < 2 or not session_id[1:].isdigit():
        raise ConfigurationError(f"session id {session_id!r} must look like s0123")
    return int(session_id[1:])


def think_seconds(seed: int, session_id: str, ordinal: int, mean: float, cv: float) -> float:
    """The frozen think time (closed loop) / cooldown (open loop) before request ``ordinal`` of a session."""
    if mean == 0:
        return 0.0
    return Uniforms(seed, "think", session_number(session_id), ordinal).lognormal(mean, cv)


# --------------------------------------------------------------------------------------------------------- settings
@dataclass(frozen=True)
class Rates:
    """S used for a bundle and the rates it implies (lambda = rho / S; think/cooldown mean = think_time_mult * S)."""

    service_s: float
    lambda_req: float
    mean_interarrival_s: float
    think_mean_s: float

    def to_dict(self) -> dict[str, float]:
        return {"service_time_s": self.service_s, "lambda_req_per_s": self.lambda_req,
                "mean_interarrival_s": self.mean_interarrival_s, "think_mean_s": self.think_mean_s}


def _check_service_time(value: Any) -> float | dict[str, Any]:
    """S: a positive number, or ``{"by_depth": {h: seconds}}`` (h >= 1, seconds > 0), normalized."""
    if isinstance(value, Mapping):
        if set(value) != {"by_depth"} or not isinstance(value["by_depth"], Mapping) or not value["by_depth"]:
            raise ConfigurationError('service_time_s must be a number or {"by_depth": {depth: seconds}}')
        points = {}
        for depth, seconds in value["by_depth"].items():
            try:
                h = int(depth)
            except (TypeError, ValueError):
                raise ConfigurationError(f"service_time_s depth {depth!r} is not an integer") from None
            if h < 1 or str(h) != str(depth).strip():
                raise ConfigurationError(f"service_time_s depth {depth!r} must be an integer >= 1")
            points[h] = _number(seconds, f"service_time_s[{h}]", minimum=0.0, strict=True)
        return {"by_depth": {str(h): points[h] for h in sorted(points)}}
    return _number(value, "service_time_s", minimum=0.0, strict=True)


def _check_models(models: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(models, Mapping) or not models:
        raise ConfigurationError("models must be a nonempty mapping name -> {min_switch_depth, max_depth}")
    out = {}
    for name, item in models.items():
        if not isinstance(name, str) or not name:
            raise ConfigurationError("model names must be nonempty strings")
        if not isinstance(item, Mapping) or set(item) - {"min_switch_depth", "max_depth"}:
            raise ConfigurationError(f"model {name!r}: fields are min_switch_depth and max_depth")
        msd = item.get("min_switch_depth", 1)
        maxd = item.get("max_depth")
        if type(msd) is not int or msd < 1:
            raise ConfigurationError(f"model {name!r}: min_switch_depth must be an integer >= 1")
        if maxd is not None and (type(maxd) is not int or maxd < 2):
            raise ConfigurationError(f"model {name!r}: max_depth must be null or an integer >= 2")
        out[name] = {"min_switch_depth": msd, "max_depth": maxd}
    return out


def _number(value: Any, name: str, *, minimum: float | None = None, strict: bool = False,
            maximum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ConfigurationError(f"{name} must be a finite number")
    value = float(value)
    if minimum is not None and (value <= minimum if strict else value < minimum):
        raise ConfigurationError(f"{name} must be {'>' if strict else '>='} {minimum}")
    if maximum is not None and value > maximum:
        raise ConfigurationError(f"{name} must be <= {maximum}")
    return value


def _integer(value: Any, name: str, minimum: int) -> int:
    if type(value) is not int or value < minimum:
        raise ConfigurationError(f"{name} must be an integer >= {minimum}")
    return value


@dataclass(frozen=True)
class WorkloadSetting:
    """One fully resolved workload setting (every sweep value expanded). Units: seconds, chunks, requests.

    ``service_time_s`` is S, the target model's mean service time, an input per model: a number, or
    ``{"by_depth": {h: seconds}}`` (S as a function of the target depth h, linear between points, constant outside),
    in which case S is the mean of S(h) over the bundle's actual target depths (coordinator 03:30; the pilot measures
    it). ``rho`` sets the aggregate rate ``lambda = rho / S``. ``think_time_mult`` is the mean think time in units of S:
    closed loop, the time between a chunk's return and the session's next request; open loop, a per-session
    cooldown (minimum spacing between one session's intended arrivals; 0 = none)."""

    dataset: str
    service_time_s: float | Mapping[str, Any]
    target_model: str = "unspecified"
    split: str = "evaluation_candidate"
    n_sessions: int = 8
    d0: int = 8
    budget: int = 24
    p_historical: float = 0.0
    gap_profile: str = "uniform"
    gap_k: int | None = None
    recent_mean: float = 2.0
    arrival_mode: str = "open"
    rho: float = 0.8
    arrival_cv: float = 1.0
    think_time_mult: float = 0.0
    think_cv: float = 1.0
    seed: int = 0
    video_seed: int = 0
    models: Mapping[str, Mapping[str, Any]] = field(default_factory=lambda: {k: dict(v) for k, v in
                                                                              DEFAULT_MODELS.items()})

    def __post_init__(self):
        if not isinstance(self.dataset, str) or not self.dataset:
            raise ConfigurationError("dataset must name the synthetic-500 dataset directory")
        if not isinstance(self.target_model, str) or not self.target_model:
            raise ConfigurationError("target_model must be a nonempty string (the model S belongs to)")
        if self.split not in SPLIT_NAMES:
            raise ConfigurationError(f"split must be one of {sorted(SPLIT_NAMES)}")
        object.__setattr__(self, "split", SPLIT_NAMES[self.split])
        object.__setattr__(self, "service_time_s", _check_service_time(self.service_time_s))
        _integer(self.n_sessions, "n_sessions", 1)
        _integer(self.d0, "d0", 1)
        _integer(self.budget, "budget", 1)
        _number(self.p_historical, "p_historical", minimum=0.0, maximum=1.0)
        if self.gap_profile not in GAP_PROFILES:
            raise ConfigurationError(f"gap_profile must be one of {GAP_PROFILES}")
        if self.gap_k is not None:
            _integer(self.gap_k, "gap_k", 1)
        if (self.gap_profile == "fixed") != (self.gap_k is not None):
            raise ConfigurationError("gap_k is required for (and only for) gap_profile 'fixed'")
        _number(self.recent_mean, "recent_mean", minimum=1.0)
        if self.arrival_mode not in ARRIVAL_MODES:
            raise ConfigurationError(f"arrival_mode must be one of {ARRIVAL_MODES}")
        _number(self.rho, "rho", minimum=0.0, strict=True)
        _number(self.arrival_cv, "arrival_cv", minimum=0.0)
        _number(self.think_time_mult, "think_time_mult", minimum=0.0)
        _number(self.think_cv, "think_cv", minimum=0.0)
        _integer(self.seed, "seed", 0)
        _integer(self.video_seed, "video_seed", 0)
        object.__setattr__(self, "models", _check_models(self.models))

    # -- derived
    @property
    def depth_dependent_service(self) -> bool:
        return isinstance(self.service_time_s, Mapping)

    def service_time_at(self, depth: int) -> float:
        """S(h) of a request whose target has depth ``h``."""
        if not self.depth_dependent_service:
            return float(self.service_time_s)
        points = sorted((int(h), float(v)) for h, v in self.service_time_s["by_depth"].items())
        if depth <= points[0][0]:
            return points[0][1]
        for (h0, v0), (h1, v1) in zip(points, points[1:]):
            if depth <= h1:
                return v0 + (v1 - v0) * (depth - h0) / (h1 - h0)
        return points[-1][1]

    def effective_service_time(self, target_depths: Sequence[int]) -> float:
        """S of a bundle: the number, or the mean of S(h) over the bundle's request target depths."""
        if not self.depth_dependent_service:
            return float(self.service_time_s)
        if not target_depths:
            raise ConfigurationError("a depth-dependent S needs at least one request")
        return math.fsum(self.service_time_at(int(d)) for d in target_depths) / len(target_depths)

    def rates(self, service_s: float | None = None) -> "Rates":
        """The rates implied by S (``service_s``; a numeric ``service_time_s`` when omitted)."""
        if service_s is None:
            if self.depth_dependent_service:
                raise ConfigurationError("S depends on the target depths; pass the bundle's effective S")
            service_s = float(self.service_time_s)
        return Rates(service_s, self.rho / service_s, service_s / self.rho, self.think_time_mult * service_s)

    @property
    def lambda_req(self) -> float:
        """Aggregate offered request rate (requests/s) = rho / S (numeric S only)."""
        return self.rates().lambda_req

    @property
    def mean_interarrival_s(self) -> float:
        return self.rates().mean_interarrival_s

    @property
    def think_mean_s(self) -> float:
        return self.rates().think_mean_s

    @property
    def min_switch_depth(self) -> int:
        return max(item["min_switch_depth"] for item in self.models.values())

    @property
    def max_depth(self) -> int | None:
        bounded = [item["max_depth"] for item in self.models.values() if item["max_depth"] is not None]
        return min(bounded) if bounded else None

    def constraint_record(self) -> dict[str, Any]:
        """The strictest target constraints and the models that set them (recorded in every bundle)."""
        msd, maxd = self.min_switch_depth, self.max_depth
        return {"min_switch_depth": msd, "max_depth": maxd,
                "min_switch_depth_from": sorted(n for n, m in self.models.items() if m["min_switch_depth"] == msd),
                "max_depth_from": sorted(n for n, m in self.models.items() if maxd is not None and
                                         m["max_depth"] == maxd),
                "models": {name: dict(item) for name, item in sorted(self.models.items())},
                "rule": "strictest over the declared models, so one tree set serves every model"}

    def to_dict(self) -> dict[str, Any]:
        out = {item.name: getattr(self, item.name) for item in fields(self)}
        out["models"] = {name: dict(item) for name, item in sorted(self.models.items())}
        return _plain(out)

    def tree_key(self) -> dict[str, Any]:
        """The parameters a session tree depends on (never rate, burstiness, N, think times or arrival mode)."""
        return {"dataset": self.dataset, "d0": self.d0, "budget": self.budget, "p_historical": self.p_historical,
                "gap_profile": self.gap_profile, "gap_k": self.gap_k, "recent_mean": self.recent_mean,
                "seed": self.seed, "min_switch_depth": self.min_switch_depth, "max_depth": self.max_depth}

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "WorkloadSetting":
        known = {item.name for item in fields(cls)}
        unknown = set(values) - known
        if unknown:
            raise ConfigurationError(f"unknown workload parameters: {sorted(unknown)}")
        missing = {"dataset", "service_time_s"} - set(values)
        if missing:
            raise ConfigurationError(f"missing workload parameters: {sorted(missing)}")
        return cls(**dict(values))


def setting_id_hash(setting: WorkloadSetting) -> str:
    return sha256_text(canonical_json(setting.to_dict()))[:16]


def structural_problems(setting: WorkloadSetting, *, candidates: int | None = None,
                        rates: "Rates | None" = None) -> list[dict[str, Any]]:
    """Seed-independent infeasibility of a setting (checked before any simulation or file write).

    The first request of every session has latest index ``n = d0`` and the setup chain as its tree, so its legal
    supports are known in advance; later supports only grow. Open loop: the mean per-session spacing is about
    ``N / lambda``; a cooldown (``think_time_mult * S``) at or above it is rejected."""
    problems: list[dict[str, Any]] = []
    msd, maxd, d0 = setting.min_switch_depth, setting.max_depth, setting.d0
    if d0 < msd:
        problems.append({"code": "latest_illegal", "message": f"d0={d0} < min_switch_depth={msd}: the first latest "
                                                              "target is not a legal source"})
    if maxd is not None and d0 + 1 > maxd:
        problems.append({"code": "max_depth", "message": f"d0 + 1 = {d0 + 1} > max_depth={maxd}"})
    if setting.p_historical > 0:
        first = [age for age in range(1, d0) if d0 - age >= msd and (maxd is None or d0 - age + 1 <= maxd)]
        if not first:
            problems.append({"code": "empty_support", "message": f"p_historical > 0 but the first request (latest "
                                                                f"index {d0}) has no legal positive age "
                                                                f"(min_switch_depth={msd})"})
        elif setting.gap_profile == "fixed" and setting.gap_k not in first:
            problems.append({"code": "fixed_k_illegal", "message": f"fixed k={setting.gap_k} is not a legal age at "
                                                                  f"the first request (legal ages {first})"})
    if setting.p_historical == 0 and maxd is not None and d0 + setting.budget > maxd:
        problems.append({"code": "max_depth", "message": f"latest-only sessions reach depth {d0 + setting.budget} > "
                                                        f"max_depth={maxd}"})
    if rates is None and not setting.depth_dependent_service:
        rates = setting.rates()
    if setting.arrival_mode == "open" and rates is not None and rates.think_mean_s > 0:
        spacing = setting.n_sessions / rates.lambda_req
        if rates.think_mean_s >= spacing:
            problems.append({"code": "think_ge_spacing",
                             "message": f"open loop: cooldown mean {rates.think_mean_s:g} s >= mean per-session "
                                        f"spacing N/lambda = {spacing:g} s"})
    if candidates is not None and setting.n_sessions > candidates:
        problems.append({"code": "not_enough_sessions", "message": f"n_sessions={setting.n_sessions} > "
                                                                  f"{candidates} candidates in split {setting.split}"})
    return problems


# ---------------------------------------------------------------------------------------------- dataset and cohort
@dataclass(frozen=True)
class Dataset:
    """The synthetic-500 content bank: validated sessions (``synthetic_workloads.read_sessions``) and splits."""

    root: str
    sessions: Mapping[str, Mapping[str, Any]]
    splits: Mapping[str, tuple[str, ...]]
    provenance: Mapping[str, Any]

    def session(self, session_id: str) -> Mapping[str, Any]:
        try:
            return self.sessions[session_id]
        except KeyError:
            raise ConfigurationError(f"session {session_id!r} is not in the dataset") from None

    def all_prompts(self) -> Iterable[str]:
        for session in self.sessions.values():
            yield from session["prompts"]
            for fork in session["forks"]:
                yield from fork["prompts"]


_DATASETS: dict[tuple, Dataset] = {}


def load_dataset(root: str | os.PathLike) -> Dataset:
    """Read ``sessions.jsonl`` (each session validated) and ``splits.json`` of a dataset directory (cached)."""
    root = Path(root)
    sessions_path, splits_path = root / "sessions.jsonl", root / "splits.json"
    for path in (sessions_path, splits_path):
        if not path.is_file():
            raise ConfigurationError(f"dataset file missing: {path}")
    stamp = tuple((str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in (sessions_path, splits_path))
    cached = _DATASETS.get(stamp)
    if cached is not None:
        return cached
    rows = read_sessions(sessions_path)
    sessions = {row["session_id"]: row for row in rows}
    if len(sessions) != len(rows):
        raise ConfigurationError("duplicate session ids in the dataset")
    splits_raw = json.loads(splits_path.read_text())
    splits = {}
    for name in ("development", "evaluation_candidate"):
        ids = splits_raw.get(name)
        if not isinstance(ids, list) or any(item not in sessions for item in ids):
            raise ConfigurationError(f"split {name!r} must list dataset session ids")
        splits[name] = tuple(ids)
    if set(splits["development"]) & set(splits["evaluation_candidate"]):
        raise ConfigurationError("development and evaluation splits overlap")
    provenance = {"root": str(root), "sessions_sha256": _sha256_file(sessions_path),
                  "splits_sha256": _sha256_file(splits_path), "sessions": len(sessions),
                  "split_sizes": {name: len(ids) for name, ids in splits.items()}}
    manifest = root / "manifest.json"
    if manifest.is_file():
        provenance["manifest_sha256"] = _sha256_file(manifest)
    dataset = Dataset(str(root), sessions, splits, provenance)
    _DATASETS[stamp] = dataset
    return dataset


#: Ordered keyword rules coarsening the 335 fine narrative categories (first match wins; substring match on the
#: lower-cased category with ``_`` read as a space). The last group takes everything else.
COARSE_CATEGORY_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("fantasy_story", ("fantasy", "magic", "mystery", "supernatural", "fiction", "whimsical", "cartoon",
                       "imaginative")),
    ("science_education", ("museum", "education", "science", "astronomy", "planetarium", "observatory", "robotic",
                           "mechanical", "optical", "laboratory", "geology", "cartography", "documentation",
                           "fieldwork", "chess", "classroom", "library", "learning")),
    ("garden_farm", ("garden", "orchard", "farm", "vineyard", "viticulture", "harvest", "seed", "greenhouse",
                     "botanical", "botany", "flower", "meadow", "olive", "crop", "beekeeping")),
    ("food_domestic", ("preparation", "bread", "pasta", "sushi", "dumpling", "tea", "coffee", "confection", "dessert",
                       "cafe", "kitchen", "home", "household", "interior", "wardrobe", "table", "organization",
                       "domestic", "baking", "bakery", "food", "culinary", "cozy")),
    ("craft_art", ("craft", "pottery", "clay", "ceramic", "paper", "textile", "fabric", "weav", "sewing", "knitting",
                   "embroidery", "basketry", "mosaic", "painting", "drawing", "printing", "sculpture", "macrame",
                   "felting", "quilling", "collage", "bookbinding", "sashiko", "mending", "braiding", "spinning",
                   "art", "photography", "stop motion", "miniature", "assemblage", "restoration", "repair",
                   "workshop", "boatbuilding", "tailoring", "writing", "visual", "design")),
    ("nature_wildlife", ("wildlife", "bird", "mammal", "insect", "amphibian", "forest", "grassland", "desert",
                         "savanna", "mountain", "alpine", "wetland", "river", "estuary", "fjord", "underwater",
                         "coast", "beach", "island", "lake", "pond", "nature", "macro", "animal", "pastoral",
                         "camping", "aquaculture", "canal", "harbor", "seaside")),
    ("people_places", ()),
)
TERTILES = ("short", "medium", "long")


def coarse_category(category: str) -> str:
    text = str(category).lower().replace("_", " ")
    for group, words in COARSE_CATEGORY_RULES:
        if not words or any(word in text for word in words):
            return group
    raise AssertionError("the last coarse group matches everything")


def prompt_length(session: Mapping[str, Any]) -> float:
    """Mean word count of the session's main prompts (the stratification length)."""
    prompts = session["prompts"]
    return sum(len(text.split()) for text in prompts) / len(prompts)


def select_cohort(dataset: Dataset, split: str, n_sessions: int, seed: int) -> dict[str, Any]:
    """Stratified, nested cohort: coarse category x prompt-length tertile over the split's candidates.

    A stratified permutation of every candidate is drawn from the ``cohort`` stream (members of each stratum shuffled,
    member ``i`` of a stratum of size ``m`` keyed ``(i + u_stratum) / m``, all candidates sorted by key); the cohort of
    size N is its first N entries, so smaller cohorts are prefixes of larger ones and every prefix is approximately
    proportionally stratified."""
    split = SPLIT_NAMES.get(split, split)
    candidates = list(dataset.splits[split])
    if n_sessions > len(candidates):
        raise InfeasibleError("not_enough_sessions", f"{n_sessions} sessions requested, split {split} has "
                                                     f"{len(candidates)}", split=split)
    lengths = {sid: prompt_length(dataset.session(sid)) for sid in candidates}
    ranked = sorted(candidates, key=lambda sid: (lengths[sid], sid))
    tertile = {sid: TERTILES[min(2, rank * 3 // len(ranked))] for rank, sid in enumerate(ranked)}
    group = {sid: coarse_category(dataset.session(sid)["category"]) for sid in candidates}
    strata: dict[tuple[str, str], list[str]] = {}
    for sid in sorted(candidates):
        strata.setdefault((group[sid], tertile[sid]), []).append(sid)
    order_names = [name for name, _ in COARSE_CATEGORY_RULES]
    stratum_keys = sorted(strata, key=lambda key: (order_names.index(key[0]), TERTILES.index(key[1])))
    draws = Uniforms(seed, "cohort")
    keyed = []
    for index, key in enumerate(stratum_keys):
        members = list(strata[key])
        for position in range(len(members) - 1, 0, -1):  # Fisher-Yates
            swap = min(int(draws.next() * (position + 1)), position)
            members[position], members[swap] = members[swap], members[position]
        offset = draws.next()
        keyed += [((rank + offset) / len(members), index, sid) for rank, sid in enumerate(members)]
    order = [sid for _, _, sid in sorted(keyed)]
    chosen = order[:n_sessions]
    # the first member (by rank) of the medium and long tertiles: rank r is in tertile r * 3 // n
    thresholds = [lengths[ranked[min(len(ranked) - 1, -(-len(ranked) * t // 3))]] for t in (1, 2)]
    return {
        "split": split, "candidates": len(candidates), "n_sessions": n_sessions, "sessions": chosen,
        "members": [{"session_id": sid, "category": dataset.session(sid)["category"], "coarse_category": group[sid],
                     "prompt_length_words": round(lengths[sid], 6), "length_tertile": tertile[sid],
                     "stratum": f"{group[sid]}/{tertile[sid]}", "rank": rank}
                    for rank, sid in enumerate(chosen)],
        "strata": [{"stratum": f"{key[0]}/{key[1]}", "candidates": len(strata[key]),
                    "selected": sum(1 for sid in chosen if (group[sid], tertile[sid]) == key)}
                   for key in stratum_keys],
        "length_tertile_thresholds_words": [round(value, 6) for value in thresholds],
        "coarse_category_rules": [{"group": name, "keywords": list(words)} for name, words in COARSE_CATEGORY_RULES],
        "stratified_order": order,
        "rule": "stratified permutation (cohort stream); cohort = first N of the order (nested across N)",
    }


# ------------------------------------------------------------------------------------------- session trees, targets
@dataclass
class TreeNode:
    """One clean chunk node of a planned session tree (generation index = node age order)."""

    generation_index: int
    parent: int | None
    depth: int  # chunks on the node's path, the node's own chunk included (chunk index = depth - 1)
    prompt_id: str
    prompt_source: str  # "main" | "bank"
    prompt_kind: str  # "main" | "main_extension" | "branch"
    prompt_key: str | None  # bank key (sha256 hex) for bank prompts
    main_index: int | None  # main storyline position (1-based) for main prompts and main extensions
    on_main: bool  # every node of the path is the first child of its parent (the main storyline)
    path_prompt_ids: tuple[str, ...]  # prompt id of every chunk of the path, this node's chunk last
    created_by: str  # "setup" or the request id
    children: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"generation_index": self.generation_index, "parent": self.parent, "depth": self.depth,
                "chunk_index": self.depth - 1, "prompt_id": self.prompt_id, "prompt_source": self.prompt_source,
                "prompt_kind": self.prompt_kind, "prompt_key": self.prompt_key, "main_index": self.main_index,
                "on_main": self.on_main, "created_by": self.created_by, "children": list(self.children)}


def main_prompt_index(chunk_index: int) -> int:
    """Main prompt (1-based) of main-storyline chunk ``chunk_index``: 1 for chunks 0-4, then ``chunk_index - 3``."""
    if type(chunk_index) is not int or chunk_index < 0:
        raise ConfigurationError("chunk_index must be a nonnegative integer")
    return 1 if chunk_index < PROMPT_HOLD_CHUNKS else chunk_index - PROMPT_HOLD_CHUNKS + 2


def slot_key(session_id: str, parent_path_prompt_ids: Sequence[str], child_ordinal: int) -> str:
    """Bank key of an authored continuation: sha256 of (session id, parent path prompt ids per chunk, child ordinal)."""
    return sha256_text(canonical_json({"session_id": session_id, "path": list(parent_path_prompt_ids),
                                       "child_ordinal": child_ordinal}))


def bank_prompt_id(session_id: str, key: str) -> str:
    return f"{session_id}:bank:{key[:24]}"


def prompt_segments(path_prompt_ids: Sequence[str]) -> list[dict[str, Any]]:
    """Runs of identical prompt ids along a path: ``[{start_chunk, chunks, prompt_id}]``."""
    segments: list[dict[str, Any]] = []
    for chunk, prompt_id in enumerate(path_prompt_ids):
        if segments and segments[-1]["prompt_id"] == prompt_id:
            segments[-1]["chunks"] += 1
        else:
            segments.append({"start_chunk": chunk, "chunks": 1, "prompt_id": prompt_id})
    return segments


class SessionTree:
    """Append-only planned tree of one session. Nodes are never removed or re-parented (full-tree retention)."""

    def __init__(self, session_id: str, main_prompts: int):
        session_number(session_id)
        if type(main_prompts) is not int or main_prompts < 1:
            raise ConfigurationError("a session needs at least one main prompt")
        self.session_id, self.main_prompts = session_id, main_prompts
        self.nodes: dict[int, TreeNode] = {}

    @property
    def latest(self) -> int:
        """Latest generation index (0 for an empty tree)."""
        return len(self.nodes)

    def append(self, parent: int | None, created_by: str) -> TreeNode:
        """New node ``latest + 1`` under ``parent`` (None: the root), with its prompt binding."""
        sid = self.session_id
        if parent is None:
            if self.nodes:
                raise ConfigurationError("the tree already has a root")
            depth, path, ordinal, parent_on_main = 1, (), 1, True
        else:
            if parent not in self.nodes:
                raise ConfigurationError(f"{sid}: parent {parent} does not exist")
            owner = self.nodes[parent]
            depth, path, ordinal = owner.depth + 1, owner.path_prompt_ids, len(owner.children) + 1
            parent_on_main = owner.on_main
        if parent_on_main and ordinal == 1:
            index = main_prompt_index(depth - 1)
            if index <= self.main_prompts:
                prompt_id, source, kind, key = f"{sid}:main:{index}", "main", "main", None
            else:
                key = slot_key(sid, path, ordinal)
                prompt_id, source, kind = bank_prompt_id(sid, key), "bank", "main_extension"
            on_main, main_index = True, index
        else:
            key = slot_key(sid, path, ordinal)
            prompt_id, source, kind, on_main, main_index = bank_prompt_id(sid, key), "bank", "branch", False, None
        node = TreeNode(self.latest + 1, parent, depth, prompt_id, source, kind, key, main_index, on_main,
                        path + (prompt_id,), created_by)
        self.nodes[node.generation_index] = node
        if parent is not None:
            self.nodes[parent].children.append(node.generation_index)
        return node

    def is_legal(self, generation_index: int, min_switch_depth: int, max_depth: int | None) -> bool:
        depth = self.nodes[generation_index].depth
        return depth >= min_switch_depth and (max_depth is None or depth + 1 <= max_depth)

    def legal_ages(self, min_switch_depth: int, max_depth: int | None) -> list[int]:
        """Legal positive ages relative to the latest node, ascending (age a = node latest - a)."""
        n = self.latest
        return [n - g for g in range(n - 1, 0, -1) if self.is_legal(g, min_switch_depth, max_depth)]

    def chain(self, generation_index: int) -> list[int]:
        """Generation indices from the root to the node."""
        out = []
        current: int | None = generation_index
        while current is not None:
            out.append(current)
            current = self.nodes[current].parent
        return out[::-1]

    def is_ancestor_or_self(self, ancestor: int, node: int) -> bool:
        return ancestor in self.chain(node)

    def stats(self) -> dict[str, Any]:
        by_depth: dict[int, int] = {}
        for node in self.nodes.values():
            by_depth[node.depth] = by_depth.get(node.depth, 0) + 1
        return {"nodes": len(self.nodes), "depth_max": max(by_depth) if by_depth else 0,
                "width_max": max(by_depth.values()) if by_depth else 0,
                "leaves": sum(1 for node in self.nodes.values() if not node.children),
                "branch_points": sum(1 for node in self.nodes.values() if len(node.children) > 1),
                "main_storyline_depth": max((n.depth for n in self.nodes.values() if n.on_main), default=0)}


def gap_distribution(profile: str, legal_ages: Iterable[int], *, gap_k: int | None = None,
                     recent_mean: float = 2.0) -> tuple[list[int], list[float]]:
    """Conditioned positive-gap distribution over the legal ages: ``(support, probabilities)``, ascending ages.

    ``uniform``: every legal age; ``recent``: P(a) proportional to (1 - 1/recent_mean)**(a - 1) on the legal ages
    (geometric with that mean, truncated); ``far``: uniform over the upper half of the legal ages (the largest
    ceil(m/2) of m); ``fixed``: exactly ``gap_k``. Raises :class:`InfeasibleError` when the conditioned support is
    empty; ages are never renumbered or snapped."""
    ages = sorted(set(int(age) for age in legal_ages))
    if any(age < 1 for age in ages):
        raise ConfigurationError("legal ages are positive")
    if not ages:
        raise InfeasibleError("empty_support", "no legal positive age")
    if profile == "uniform":
        support, weights = ages, [1.0] * len(ages)
    elif profile == "recent":
        q = 1.0 / float(recent_mean)
        weights_all = [(1.0 - q) ** (age - 1) for age in ages]
        pairs = [(age, weight) for age, weight in zip(ages, weights_all) if weight > 0]
        if not pairs:
            raise InfeasibleError("empty_support", f"recent profile (mean {recent_mean}) has no mass on legal "
                                                   f"ages {ages}")
        support, weights = [a for a, _ in pairs], [w for _, w in pairs]
    elif profile == "far":
        support = ages[len(ages) // 2:]
        weights = [1.0] * len(support)
    elif profile == "fixed":
        if gap_k not in ages:
            raise InfeasibleError("fixed_k_illegal", f"fixed k={gap_k} is not a legal age (legal ages {ages})")
        support, weights = [int(gap_k)], [1.0]
    else:
        raise ConfigurationError(f"gap_profile must be one of {GAP_PROFILES}")
    total = math.fsum(weights)
    return support, [weight / total for weight in weights]


def choose_age(support: Sequence[int], probabilities: Sequence[float], u: float) -> int:
    """Inverse-CDF draw from a conditioned support with one uniform ``u`` in [0, 1)."""
    if not support or len(support) != len(probabilities):
        raise ConfigurationError("support and probabilities must be nonempty and aligned")
    if len(set(probabilities)) == 1:
        return support[min(int(u * len(support)), len(support) - 1)]
    cumulative = 0.0
    for age, probability in zip(support, probabilities):
        cumulative += probability
        if u < cumulative:
            return age
    return support[-1]  # u beyond a cumulative sum that rounding left just below 1 (numerical, not a fallback)


def rewind_and_generate(tree: SessionTree, k: int, *, min_switch_depth: int, max_depth: int | None,
                        created_by: str) -> tuple[TreeNode, TreeNode]:
    """The single workload operation on a planned tree: target the node of age ``k`` (``latest - k``; 0 = latest by
    generation index) and append one new child under it. Returns ``(target, new node)``. An illegal or missing
    target raises :class:`InfeasibleError`; nothing is snapped to another node."""
    n = tree.latest
    if type(k) is not int or k < 0 or k >= n:
        raise InfeasibleError("illegal_age", f"age k={k} does not name a node (latest index {n})", k=k,
                              latest_generation_index=n)
    target = tree.nodes[n - k]
    if not tree.is_legal(target.generation_index, min_switch_depth, max_depth):
        code = "latest_illegal" if k == 0 else "illegal_age"
        raise InfeasibleError(code, f"node {target.generation_index} (age {k}, depth {target.depth}) is not a legal "
                                    f"source (min_switch_depth={min_switch_depth}, max_depth={max_depth})",
                              k=k, latest_generation_index=n)
    return target, tree.append(target.generation_index, created_by)


def plan_session(session: Mapping[str, Any], setting: WorkloadSetting) -> dict[str, Any]:
    """Setup chain plus ``budget`` requests of one session; depends only on :meth:`WorkloadSetting.tree_key` inputs.

    Request ``r`` draws two uniforms from the ``targeting`` stream keyed by (session, r): the first decides
    historical (``u < p_historical``), the second picks the age from the conditioned profile."""
    sid = session["session_id"]
    number = session_number(sid)
    tree = SessionTree(sid, len(session["prompts"]))
    for index in range(1, setting.d0 + 1):
        tree.append(index - 1 if index > 1 else None, "setup")
    msd, maxd, p = setting.min_switch_depth, setting.max_depth, setting.p_historical
    requests = []
    for ordinal in range(1, setting.budget + 1):
        n = tree.latest
        request_id = f"{sid}-r{ordinal:03d}"
        draws = Uniforms(setting.seed, "targeting", number, ordinal)
        u_historical, u_age = draws.next(), draws.next()
        legal = tree.legal_ages(msd, maxd)
        historical = u_historical < p
        context = {"session_id": sid, "ordinal": ordinal, "request_id": request_id, "latest_generation_index": n,
                   "legal_ages": legal}
        support = probabilities = None
        if historical:
            try:
                support, probabilities = gap_distribution(setting.gap_profile, legal, gap_k=setting.gap_k,
                                                          recent_mean=setting.recent_mean)
            except InfeasibleError as exc:
                raise InfeasibleError(exc.code, f"{request_id} (latest index {n}): {exc}", **context) from None
            k = choose_age(support, probabilities, u_age)
            chosen = (p * probabilities[support.index(k)])
        else:
            k, chosen = 0, 1.0 - p
        try:
            target, child = rewind_and_generate(tree, k, min_switch_depth=msd, max_depth=maxd, created_by=request_id)
        except InfeasibleError as exc:
            raise InfeasibleError(exc.code, f"{request_id}: {exc}", **context) from None
        on_latest_path = tree.is_ancestor_or_self(target.generation_index, n)
        requests.append({
            "request_id": request_id, "session_id": sid, "ordinal": ordinal,
            "latest_generation_index": n, "latest_depth": tree.nodes[n].depth,
            "source_generation_index": target.generation_index, "k": k, "historical": historical,
            "target_depth": target.depth, "target_on_latest_path": on_latest_path,
            "old_branch_target": historical and not on_latest_path,
            "planned_action": "resume" if k == 0 else "rewind",
            "legal_ages": legal, "legal_support_size": len(legal), "support": support,
            "support_probabilities": probabilities, "chosen_probability": chosen,
            "output_generation_index": child.generation_index, "output_depth": child.depth,
            "total_chunks": child.depth, "new_chunk_index": child.depth - 1, "new_chunks": 1,
            "child_ordinal": len(target.children),
            "depends_on": None if target.created_by == "setup" else target.created_by,
            "prompt_id": child.prompt_id, "prompt_source": child.prompt_source, "prompt_kind": child.prompt_kind,
            "prompt_key": child.prompt_key, "main_index": child.main_index,
            "path_prompt_ids": list(child.path_prompt_ids), "schedule": prompt_segments(child.path_prompt_ids),
        })
    return {"session_id": sid, "tree": tree, "requests": requests}


# ------------------------------------------------------------------------------------------------------ scheduling
def open_loop_arrivals(setting: WorkloadSetting, count: int | None = None, *,
                       rates: Rates | None = None) -> list[float]:
    """Intended arrival times (s from the window start) of the global open-loop process (``arrivals`` stream)."""
    rates = rates or setting.rates()
    total = setting.n_sessions * setting.budget if count is None else count
    draws = Uniforms(setting.seed, "arrivals")
    t, times = 0.0, []
    for _ in range(total):
        t += draws.interarrival(rates.mean_interarrival_s, setting.arrival_cv)
        times.append(t)
    return times


def assign_open_loop(arrival_times: Sequence[float], session_ids: Sequence[str], budget: int,
                     cooldown: Callable[[str, int], float], draw: Callable[[], float]) -> list[tuple[str, int]]:
    """Recipient of every arrival: uniform among the eligible sessions (budget left; for a session's request ``r >
    1``, ``t >= t(r - 1) + cooldown(session, r)``), in cohort order, one ``draw()`` per arrival. No eligible session
    raises :class:`InfeasibleError` (arrivals are never moved and cooldowns never shortened)."""
    remaining = {sid: budget for sid in session_ids}
    next_ordinal = {sid: 1 for sid in session_ids}
    last: dict[str, float | None] = {sid: None for sid in session_ids}
    out = []
    for index, t in enumerate(arrival_times):
        eligible = [sid for sid in session_ids if remaining[sid] > 0
                    and (last[sid] is None or t >= last[sid] + cooldown(sid, next_ordinal[sid]))]
        if not eligible:
            waiting = sum(1 for sid in session_ids if remaining[sid] > 0)
            raise InfeasibleError("no_eligible_session", f"arrival {index + 1} at t={t:.6g} s: {waiting} sessions "
                                                         "have budget left but none has met its cooldown",
                                  arrival_index=index + 1, t_intended_s=t, sessions_with_budget=waiting)
        sid = eligible[min(int(draw() * len(eligible)), len(eligible) - 1)]
        out.append((sid, next_ordinal[sid]))
        last[sid] = t
        remaining[sid] -= 1
        next_ordinal[sid] += 1
    return out


def schedule_requests(setting: WorkloadSetting, session_ids: Sequence[str], plans: Mapping[str, Mapping[str, Any]],
                      *, rates: Rates | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Attach the frozen timing to every planned request; returns (requests in issue order, schedule summary).

    Open loop: ``arrival_index``, ``t_intended_s`` and ``cooldown_s`` (request ``r > 1`` of a session may not arrive
    earlier than ``cooldown_s`` after request ``r - 1``; 0 without a cooldown). Closed loop: ``think_s`` (the session
    issues request ``r`` ``think_s`` after request ``r - 1`` returned; request 1 ``think_s`` after the window start);
    ``t_intended_s`` is decided at run time."""
    rates = rates or setting.rates()
    think_cache: dict[tuple[str, int], float] = {}

    def think(sid: str, ordinal: int) -> float:
        key = (sid, ordinal)
        if key not in think_cache:
            think_cache[key] = think_seconds(setting.seed, sid, ordinal, rates.think_mean_s, setting.think_cv)
        return think_cache[key]

    ordered: list[dict[str, Any]] = []
    summary: dict[str, Any] = {"mode": setting.arrival_mode}
    if setting.arrival_mode == "open":
        times = open_loop_arrivals(setting, rates=rates)
        assignment = assign_open_loop(times, session_ids, setting.budget, think,
                                      Uniforms(setting.seed, "session_choice").next)
        for index, ((sid, ordinal), t) in enumerate(zip(assignment, times), 1):
            request = dict(plans[sid]["requests"][ordinal - 1])
            request.update(arrival_index=index, t_intended_s=t, think_s=None,
                           cooldown_s=think(sid, ordinal) if ordinal > 1 else 0.0)
            ordered.append(request)
        gaps = [b - a for a, b in zip([0.0] + times[:-1], times)]
        spacing: list[float] = []
        for sid in session_ids:
            mine = [r["t_intended_s"] for r in ordered if r["session_id"] == sid]
            spacing += [b - a for a, b in zip(mine, mine[1:])]
        summary.update(arrivals=len(times), window_s=times[-1] if times else 0.0,
                       offered_rate_per_s=rates.lambda_req, mean_interarrival_s=rates.mean_interarrival_s,
                       realized_rate_per_s=(len(times) / times[-1]) if times and times[-1] > 0 else None,
                       realized_interarrival=_moments(gaps), realized_session_spacing=_moments(spacing),
                       expected_session_spacing_s=setting.n_sessions / rates.lambda_req)
    else:
        for sid in session_ids:
            for request in plans[sid]["requests"]:
                ordered.append({**request, "arrival_index": None, "t_intended_s": None, "cooldown_s": None,
                                "think_s": think(sid, request["ordinal"])})
        summary.update(arrivals=len(ordered), think_mean_s=rates.think_mean_s, think_cv=setting.think_cv,
                       realized_think=_moments([r["think_s"] for r in ordered]),
                       note="closed loop: issue times depend on completions; only think times are frozen")
    return ordered, summary


def _moments(values: Sequence[float]) -> dict[str, Any]:
    values = [float(v) for v in values]
    if not values:
        return {"count": 0, "mean": None, "cv": None, "min": None, "max": None}
    mean = math.fsum(values) / len(values)
    variance = math.fsum((v - mean) ** 2 for v in values) / len(values)
    return {"count": len(values), "mean": mean, "cv": (math.sqrt(variance) / mean) if mean > 0 else None,
            "min": min(values), "max": max(values)}


# --------------------------------------------------------------------------------------------- content and the bank
def load_bank(source: Any) -> dict[str, dict[str, Any]]:
    """An authored-continuation bank: a JSONL path (rows ``{"key", "text", ...}``), a JSON path
    (``{"schema": BANK_SCHEMA, "entries": [...]}``), or a mapping ``key -> text`` / ``key -> {"text", ...}``."""
    if source is None:
        return {}
    rows: Iterable[Any]
    if isinstance(source, (str, os.PathLike)):
        path = Path(source)
        text = path.read_text()
        if path.suffix == ".jsonl":
            rows = [json.loads(line) for line in text.splitlines() if line.strip()]
        else:
            document = json.loads(text)
            if not isinstance(document, Mapping) or document.get("schema") != BANK_SCHEMA:
                raise ConfigurationError(f"bank {path} must be JSONL or a {BANK_SCHEMA} document")
            rows = document.get("entries") or []
    elif isinstance(source, Mapping):
        rows = [{"key": key, **(value if isinstance(value, Mapping) else {"text": value})}
                for key, value in source.items()]
    else:
        raise ConfigurationError("bank must be a path or a mapping")
    bank: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping) or not isinstance(row.get("key"), str) or "text" not in row:
            raise ConfigurationError("bank entries need a string key and a text")
        key = row["key"]
        if len(key) != 64 or any(ch not in "0123456789abcdef" for ch in key):
            raise ConfigurationError(f"bank key {key!r} is not a sha256 hex digest")
        if key in bank:
            raise ConfigurationError(f"bank key {key} appears twice")
        bank[key] = dict(row)
    return bank


def _text_resolver(dataset: Dataset, bank: Mapping[str, Mapping[str, Any]], trees: Mapping[str, SessionTree]):
    keys = {node.prompt_id: node.prompt_key for tree in trees.values() for node in tree.nodes.values()
            if node.prompt_source == "bank"}

    def text_of(prompt_id: str) -> str | None:
        sid, source, rest = prompt_id.split(":", 2)
        if source == "main":
            return dataset.session(sid)["prompts"][int(rest) - 1]
        entry = bank.get(keys.get(prompt_id, ""))
        return None if entry is None else entry["text"]

    return text_of, keys


def bind_content(dataset: Dataset, trees: Mapping[str, SessionTree],
                 bank: Mapping[str, Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Prompt bindings of every node (main prompts from the dataset, the rest from the bank) and the content slots
    (every bank key of the trees, with its full parent-path context and its siblings)."""
    text_of, prompt_keys = _text_resolver(dataset, bank, trees)
    bindings: dict[str, dict[str, Any]] = {}
    slots: list[dict[str, Any]] = []
    for sid, tree in trees.items():
        session = dataset.session(sid)
        for gen, node in sorted(tree.nodes.items()):
            user = node.created_by if node.created_by != "setup" else f"setup:{gen}"
            entry = bindings.get(node.prompt_id)
            if entry is None:
                text = text_of(node.prompt_id)
                entry = bindings[node.prompt_id] = {
                    "prompt_id": node.prompt_id, "session_id": sid, "source": node.prompt_source,
                    "kind": node.prompt_kind, "main_index": node.main_index, "key": node.prompt_key,
                    "text": text, "sha256": None if text is None else sha256_text(text), "bound": text is not None,
                    "used_by": []}
                if node.prompt_source == "bank" and text is not None:
                    entry["bank_provenance"] = {k: v for k, v in bank[node.prompt_key].items()
                                                if k not in ("key", "text")}
            entry["used_by"].append(user)
            if node.prompt_source != "bank":
                continue
            parent = tree.nodes[node.parent] if node.parent is not None else None
            context = []
            if parent is not None:
                for segment in prompt_segments(parent.path_prompt_ids):
                    text = text_of(segment["prompt_id"])
                    context.append({**segment, "text": text, "key": prompt_keys.get(segment["prompt_id"])})
            siblings = []
            for other in (parent.children if parent is not None else []):
                if other == gen:
                    continue
                sibling = tree.nodes[other]
                siblings.append({"prompt_id": sibling.prompt_id, "prompt_source": sibling.prompt_source,
                                 "prompt_key": sibling.prompt_key, "generation_index": other,
                                 "text": text_of(sibling.prompt_id)})
            slots.append({
                "key": node.prompt_key, "prompt_id": node.prompt_id, "session_id": sid,
                "session_title": session["title"], "session_category": session["category"],
                "kind": node.prompt_kind, "main_index": node.main_index, "generation_index": gen,
                "chunk_index": node.depth - 1, "parent_generation_index": node.parent,
                "parent_depth": parent.depth if parent is not None else 0,
                "child_ordinal": (parent.children.index(gen) + 1) if parent is not None else 1,
                "parent_prompt_id": parent.prompt_id if parent is not None else None,
                "parent_prompt_key": parent.prompt_key if parent is not None else None,
                "parent_text": text_of(parent.prompt_id) if parent is not None else None,
                "context": context, "siblings": siblings,
                "depends_on_keys": sorted({item["key"] for item in context if item["key"] and item["text"] is None}),
                "used_by": [user], "bound": node.prompt_key in bank,
                "requirements": {"words": list(WORD_RANGE), "differs_from_siblings": True,
                                 "not_a_copy_of_parent": True, "continues_parent_path": True},
            })
    return [bindings[key] for key in sorted(bindings)], slots


def validate_bank(bank: Any, slots: Sequence[Mapping[str, Any]] | None = None, *,
                  dataset: Dataset | None = None) -> dict[str, Any]:
    """Check authored bank entries: 18-35 words; unique (normalized) within the bank and against the dataset's
    prompts; distinct from every sibling (exact and normalized); not a copy of the parent prompt (exact, normalized,
    or containing it). Returns a report (``ok`` = no errors; ``complete`` = every slot has an entry); never raises
    for content problems."""
    entries = load_bank(bank)
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    dataset_norm = {normalized_prompt(text) for text in dataset.all_prompts()} if dataset is not None else set()
    by_key = {slot["key"]: slot for slot in (slots or [])}
    seen: dict[str, str] = {}

    def text_for(prompt_text: str | None, key: str | None) -> str | None:
        if prompt_text is not None:
            return prompt_text
        entry = entries.get(key or "")
        return None if entry is None else entry.get("text")

    for key in sorted(entries):
        text = entries[key].get("text")
        if not isinstance(text, str) or not text.strip():
            errors.append({"key": key, "check": "empty", "detail": "text must be a nonempty string"})
            continue
        words = len(text.split())
        if not WORD_RANGE[0] <= words <= WORD_RANGE[1]:
            errors.append({"key": key, "check": "word_count", "detail": f"{words} words (need {WORD_RANGE[0]}-"
                                                                         f"{WORD_RANGE[1]})"})
        norm = normalized_prompt(text)
        if norm in seen:
            errors.append({"key": key, "check": "duplicate", "detail": f"same normalized text as {seen[norm]}"})
        else:
            seen[norm] = key
        if norm in dataset_norm:
            errors.append({"key": key, "check": "copies_dataset_prompt", "detail": "equals a dataset prompt"})
        slot = by_key.get(key)
        if slot is None:
            if slots is not None:
                warnings.append({"key": key, "check": "unknown_key", "detail": "no content slot has this key"})
            continue
        parent = text_for(slot.get("parent_text"), slot.get("parent_prompt_key"))
        if parent is not None:
            parent_norm = normalized_prompt(parent)
            if text == parent or norm == parent_norm:
                errors.append({"key": key, "check": "copies_parent", "detail": "equals the parent prompt"})
            elif parent_norm and parent_norm in norm:
                errors.append({"key": key, "check": "contains_parent", "detail": "contains the parent prompt"})
        for sibling in slot.get("siblings") or []:
            other = text_for(sibling.get("text"), sibling.get("prompt_key"))
            if other is None:
                continue
            if text == other:
                errors.append({"key": key, "check": "equals_sibling", "detail": sibling.get("prompt_id")})
            elif norm == normalized_prompt(other):
                errors.append({"key": key, "check": "equals_sibling_normalized", "detail": sibling.get("prompt_id")})
    missing = sorted(key for key in by_key if key not in entries)
    return {"schema": "reelcache.session_workload_bank_validation.v1", "entries": len(entries),
            "slots": len(by_key), "missing_keys": missing, "errors": errors, "warnings": warnings,
            "ok": not errors, "complete": not missing}


# ------------------------------------------------------------------------------------------------ coverage, bundles
def compute_coverage(setting: WorkloadSetting, trees: Mapping[str, SessionTree], requests: Sequence[Mapping],
                     schedule: Mapping[str, Any], bindings: Sequence[Mapping], slots: Sequence[Mapping],
                     bank_report: Mapping[str, Any], rates: Rates | None = None) -> dict[str, Any]:
    """Coverage of a workload: counts, realized vs target historical rate and gap histogram over the legal supports,
    support sizes, tree shapes, bindings and the open-loop feasibility margin."""
    total = len(requests)
    new_nodes = sum(len(tree.nodes) - setting.d0 for tree in trees.values())
    historical = [r for r in requests if r["historical"]]
    expected: dict[int, float] = {}
    realized: dict[int, int] = {}
    for request in historical:
        for age, probability in zip(request["support"], request["support_probabilities"]):
            expected[age] = expected.get(age, 0.0) + probability
        realized[request["k"]] = realized.get(request["k"], 0) + 1
    k_histogram: dict[int, int] = {}
    for request in requests:
        k_histogram[request["k"]] = k_histogram.get(request["k"], 0) + 1
    legal_sizes = [r["legal_support_size"] for r in requests]
    conditioned = [len(r["support"]) for r in historical]
    per_session = {}
    for sid, tree in trees.items():
        mine = [r for r in requests if r["session_id"] == sid]
        per_session[sid] = {"requests": len(mine), "historical": sum(r["historical"] for r in mine),
                            "old_branch_targets": sum(r["old_branch_target"] for r in mine),
                            "main_prompts": sum(r["prompt_source"] == "main" for r in mine),
                            "bank_prompts": sum(r["prompt_source"] == "bank" for r in mine), **tree.stats()}
    request_ids = {r["request_id"] for r in requests}
    by_id = {b["prompt_id"]: b for b in bindings}
    unbound_requests = sorted(r["request_id"] for r in requests if not by_id[r["prompt_id"]]["bound"])
    unbound_setup = sorted(user for b in bindings if not b["bound"] for user in b["used_by"]
                           if user not in request_ids)
    kinds: dict[str, int] = {}
    for request in requests:
        kinds[request["prompt_kind"]] = kinds.get(request["prompt_kind"], 0) + 1
    slot_kinds: dict[str, dict[str, int]] = {}
    for slot in slots:
        entry = slot_kinds.setdefault(slot["kind"], {"slots": 0, "unbound": 0})
        entry["slots"] += 1
        entry["unbound"] += not slot["bound"]
    frozen = not unbound_requests and not unbound_setup and bool(bank_report.get("ok"))
    rates = rates or setting.rates()
    cooldown = rates.think_mean_s if setting.arrival_mode == "open" else 0.0
    load = rates.lambda_req * cooldown
    return {
        "requests": total, "sessions": len(trees),
        "count_equality": {"requests": total, "new_nodes": new_nodes, "equal": total == new_nodes,
                           "expected": setting.n_sessions * setting.budget},
        "p_historical": {"target": setting.p_historical, "realized": (len(historical) / total) if total else None,
                         "historical_requests": len(historical),
                         "old_branch_targets": sum(r["old_branch_target"] for r in requests)},
        "gap": {"profile": setting.gap_profile, "gap_k": setting.gap_k, "recent_mean": setting.recent_mean,
                "target_expected_counts": {str(a): expected[a] for a in sorted(expected)},
                "realized_counts": {str(a): realized[a] for a in sorted(realized)},
                "basis": "expected = sum over historical requests of their conditioned legal-support probabilities"},
        "k_histogram": {str(k): k_histogram[k] for k in sorted(k_histogram)},
        "support_sizes": {"legal_positive": _moments(legal_sizes), "conditioned_historical": _moments(conditioned),
                          "empty_legal_supports": sum(1 for size in legal_sizes if size == 0)},
        "per_session": per_session,
        "trees": {"depth_max": max(s["depth_max"] for s in per_session.values()),
                  "width_max": max(s["width_max"] for s in per_session.values()),
                  "leaves": sum(s["leaves"] for s in per_session.values()),
                  "branch_points": sum(s["branch_points"] for s in per_session.values())},
        "bindings": {"requests_bound": total - len(unbound_requests), "requests_unbound": len(unbound_requests),
                     "unbound_request_ids": unbound_requests, "setup_unbound": unbound_setup,
                     "request_prompt_kinds": kinds, "slots": slot_kinds,
                     "slots_total": len(slots), "slots_unbound": sum(not s["bound"] for s in slots),
                     "bank_validation": {key: bank_report.get(key) for key in ("ok", "entries", "errors")},
                     "frozen": frozen},
        "feasibility": {"n_sessions": setting.n_sessions, "lambda_req_per_s": rates.lambda_req,
                        "mean_session_spacing_s": cooldown, "lambda_x_spacing": load,
                        "margin_sessions": setting.n_sessions - load,
                        "ratio": (setting.n_sessions / load) if load > 0 else None,
                        "expected_per_session_spacing_s": setting.n_sessions / rates.lambda_req,
                        "rule": "open loop needs N > lambda x mean cooldown (and cooldown < N / lambda)"},
        "schedule": dict(schedule),
    }


@dataclass
class Workload:
    """An in-memory workload (everything a bundle holds); :func:`write_bundle` freezes it."""

    setting: WorkloadSetting
    setting_id: str
    config: dict[str, Any]
    cohort: dict[str, Any]
    setup: list[dict[str, Any]]
    requests: list[dict[str, Any]]
    bindings: list[dict[str, Any]]
    slots: list[dict[str, Any]]
    coverage: dict[str, Any]
    trees: dict[str, SessionTree]

    @property
    def frozen(self) -> bool:
        return bool(self.coverage["bindings"]["frozen"])


def build_workload(setting: WorkloadSetting, *, dataset: Dataset | None = None, bank: Any = None,
                   setting_id: str = "setting", sweep: Mapping[str, Any] | None = None) -> Workload:
    """Plan, schedule and bind one setting in memory; raises :class:`InfeasibleError` (nothing is written)."""
    dataset = dataset or load_dataset(setting.dataset)
    problems = structural_problems(setting, candidates=len(dataset.splits[setting.split]))
    if problems:
        raise InfeasibleError(problems[0]["code"], "; ".join(p["message"] for p in problems), problems=problems)
    cohort = select_cohort(dataset, setting.split, setting.n_sessions, setting.seed)
    plans = {sid: plan_session(dataset.session(sid), setting) for sid in cohort["sessions"]}
    trees = {sid: plans[sid]["tree"] for sid in cohort["sessions"]}
    depths = [request["target_depth"] for sid in cohort["sessions"] for request in plans[sid]["requests"]]
    rates = setting.rates(setting.effective_service_time(depths))  # S: mean S(h) over the actual target depths
    problems = structural_problems(setting, rates=rates)
    if problems:
        raise InfeasibleError(problems[0]["code"], "; ".join(p["message"] for p in problems), problems=problems)
    requests, schedule = schedule_requests(setting, cohort["sessions"], plans, rates=rates)
    entries = load_bank(bank)
    bindings, slots = bind_content(dataset, trees, entries)
    used = {slot["key"] for slot in slots if slot["bound"]}
    bank_report = validate_bank({key: entries[key] for key in used}, slots, dataset=dataset)
    setup = []
    for rank, sid in enumerate(cohort["sessions"]):
        tree = trees[sid]
        head = tree.nodes[setting.d0]
        setup.append({"session_id": sid, "cohort_rank": rank, "video_seed": setting.video_seed, "d0": setting.d0,
                      "title": dataset.session(sid)["title"], "category": dataset.session(sid)["category"],
                      "nodes": [tree.nodes[g].to_dict() for g in range(1, setting.d0 + 1)],
                      "path_prompt_ids": list(head.path_prompt_ids), "schedule": prompt_segments(head.path_prompt_ids),
                      "final_tree": [node.to_dict() for _, node in sorted(tree.nodes.items())]})
    coverage = compute_coverage(setting, trees, requests, schedule, bindings, slots, bank_report, rates)
    config = {
        "schema": BUNDLE_SCHEMA, "generator": GENERATOR, "generator_version": GENERATOR_VERSION,
        "setting_id": setting_id, "setting_hash": setting_id_hash(setting), "sweep": dict(sweep or {}) or None,
        "setting": setting.to_dict(),
        "derived": {**rates.to_dict(), "requests": len(requests),
                    "service_time_basis": ("mean of S(h) over the requests' target depths" if
                                           setting.depth_dependent_service else "input number"),
                    "target_depth_histogram": {str(d): depths.count(d) for d in sorted(set(depths))},
                    "tree_key_sha256": sha256_text(canonical_json(setting.tree_key())),
                    "ignored_in_mode": (["rho", "arrival_cv"] if setting.arrival_mode == "closed" else [])},
        "constraints": setting.constraint_record(), "dataset": dict(dataset.provenance),
        "rng": {"algorithm": "numpy SeedSequence(entropy=seed, spawn_key=(stream word, *key)) -> PCG64; only "
                             "Generator.random() doubles are drawn; normal (Box-Muller), gamma (Marsaglia-Tsang) and "
                             "lognormal are derived in this module",
                "streams": {name: {"word": stream_word(name)} for name in STREAMS},
                "keys": {"cohort": [], "arrivals": [], "session_choice": [], "think": ["session number", "ordinal"],
                         "targeting": ["session number", "ordinal"], "content": "reserved (no draws in v2)"}},
        "content": {"prompt_hold_chunks": PROMPT_HOLD_CHUNKS,
                    "main_rule": "main storyline chunk j uses main prompt 1 for j < 5, else j - 3",
                    "bank_key": "sha256(canonical_json({session_id, path: parent prompt ids per chunk, "
                                "child_ordinal}))",
                    "v1_forks_used": False,
                    "v1_forks_note": "user 2026-10-19 00:4x: historical and extension prompts are authored new"},
        "request_mapping": {"k == 0": "resume (the source is the session's current head)",
                            "k > 0": "rewind to the source node ref", "total_chunks": "target_depth + 1",
                            "output": "new"},
    }
    return Workload(setting, setting_id, config, cohort, setup, requests, bindings, slots, coverage, trees)


def write_bundle(workload: Workload, out_dir: str | os.PathLike) -> dict[str, Any]:
    """Write the frozen bundle directory (files read-only, ``MANIFEST.sha256``, ``BUNDLE_ID``); refuses an existing
    directory. Identical workloads give byte-identical files."""
    out = Path(out_dir)
    if out.exists():
        raise ConfigurationError(f"bundle directory {out} exists; bundles are immutable")
    out.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{out.name}.staging-", dir=out.parent))
    try:
        documents = {"config.json": _pretty_json(workload.config), "cohort.json": _pretty_json(workload.cohort),
                     "coverage.json": _pretty_json(workload.coverage)}
        lines = {"setup.jsonl": workload.setup, "requests.jsonl": workload.requests,
                 "bindings.jsonl": workload.bindings, "content_slots.jsonl": workload.slots}
        for name, rows in lines.items():
            documents[name] = "".join(canonical_json(row) + "\n" for row in rows)
        digests = {}
        for name in BUNDLE_FILES:
            data = documents[name].encode("utf-8")
            (staging / name).write_bytes(data)
            digests[name] = hashlib.sha256(data).hexdigest()
        manifest = "".join(f"{digests[name]}  {name}\n" for name in BUNDLE_FILES).encode()
        (staging / MANIFEST_NAME).write_bytes(manifest)
        bundle_id = hashlib.sha256(manifest).hexdigest()[:16]
        (staging / BUNDLE_ID_NAME).write_text(bundle_id + "\n")
        for path in staging.iterdir():
            path.chmod(0o444)
        staging.chmod(0o555)
        os.rename(staging, out)
    except BaseException:
        if staging.exists():
            staging.chmod(0o755)
            shutil.rmtree(staging, ignore_errors=True)
        raise
    return {"path": str(out), "bundle_id": bundle_id, "files": digests, "frozen": workload.frozen,
            "requests": len(workload.requests)}


def load_bundle(path: str | os.PathLike, *, verify: bool = True) -> dict[str, Any]:
    """Read a bundle; with ``verify`` every file must match ``MANIFEST.sha256`` and ``BUNDLE_ID``."""
    root = Path(path)
    manifest = (root / MANIFEST_NAME).read_bytes()
    if verify:
        expected = {}
        for line in manifest.decode().splitlines():
            digest, name = line.split("  ", 1)
            expected[name] = digest
        if sorted(expected) != sorted(BUNDLE_FILES):
            raise ConfigurationError(f"bundle {root}: manifest lists {sorted(expected)}")
        for name, digest in expected.items():
            if _sha256_file(root / name) != digest:
                raise ConfigurationError(f"bundle {root}: {name} does not match its manifest")
        if (root / BUNDLE_ID_NAME).read_text().strip() != hashlib.sha256(manifest).hexdigest()[:16]:
            raise ConfigurationError(f"bundle {root}: BUNDLE_ID does not match the manifest")

    def jsonl(name: str) -> list[dict[str, Any]]:
        return [json.loads(line) for line in (root / name).read_text().splitlines() if line.strip()]

    config = json.loads((root / "config.json").read_text())
    coverage = json.loads((root / "coverage.json").read_text())
    if config.get("schema") != BUNDLE_SCHEMA:
        raise ConfigurationError(f"bundle {root}: schema {config.get('schema')!r} is not {BUNDLE_SCHEMA}")
    return {"path": str(root), "bundle_id": (root / BUNDLE_ID_NAME).read_text().strip(),
            "manifest_sha256": hashlib.sha256(manifest).hexdigest(), "config": config,
            "cohort": json.loads((root / "cohort.json").read_text()), "coverage": coverage,
            "setup": jsonl("setup.jsonl"), "requests": jsonl("requests.jsonl"),
            "bindings": {row["prompt_id"]: row for row in jsonl("bindings.jsonl")},
            "slots": jsonl("content_slots.jsonl"), "frozen": bool(coverage["bindings"]["frozen"])}


# ----------------------------------------------------------------------------------------------------------- sweeps
_RANGE_KEYS = {"start", "stop", "step"}


def expand_values(value: Any) -> list[Any]:
    """A sweep value: a list, ``{"values": [...]}``, an inclusive range ``{"start", "stop", "step"}`` or a scalar."""
    if isinstance(value, list):
        if not value:
            raise ConfigurationError("a sweep list must not be empty")
        return list(value)
    if isinstance(value, Mapping) and set(value) == {"values"}:
        return expand_values(list(value["values"]))
    if isinstance(value, Mapping) and set(value) == _RANGE_KEYS:
        start, stop, step = (value[k] for k in ("start", "stop", "step"))
        for name, item in (("start", start), ("stop", stop), ("step", step)):
            if isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(item):
                raise ConfigurationError(f"range {name} must be a finite number")
        if step <= 0 or stop < start:
            raise ConfigurationError("a range needs step > 0 and stop >= start (stop inclusive)")
        integral = all(type(item) is int for item in (start, stop, step))
        count = int(math.floor((stop - start) / step + 1e-9)) + 1
        if count > 10000:
            raise ConfigurationError("a range expands to more than 10000 values")
        return [start + i * step if integral else float(f"{start + i * step:.12g}") for i in range(count)]
    return [value]


def _is_swept(value: Any) -> bool:
    return isinstance(value, list) or (isinstance(value, Mapping) and (set(value) == {"values"}
                                                                       or set(value) == _RANGE_KEYS))


def _label(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:g}"
    if isinstance(value, (dict, list)):
        return sha256_text(canonical_json(value))[:8]
    return str(value)


def expand_sweep(document: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Settings of a sweep document, in order, deduplicated by content: ``[{setting_id, params, origin}]``.

    Document: ``{"schema": SWEEP_SCHEMA, "name", "base": {...}, "anchor": {...}, "sweep": {param: values},
    "settings": [{"id"?, param: value | values, ...}]}``. ``base`` holds shared inputs (dataset, S, models) and is
    merged under everything; parameters neither base nor anchor name take the :class:`WorkloadSetting` defaults.
    The anchor (all scalars) is one setting; ``sweep`` varies ONE parameter at a time around
    it (one-factor-at-a-time); every ``settings`` block is merged over the anchor and expands to the Cartesian product
    of its list/range values (explicit settings)."""
    if not isinstance(document, Mapping) or document.get("schema") != SWEEP_SCHEMA:
        raise ConfigurationError(f"a sweep document needs schema {SWEEP_SCHEMA!r}")
    unknown = set(document) - {"schema", "name", "description", "base", "anchor", "sweep", "settings", "notes"}
    if unknown:
        raise ConfigurationError(f"unknown sweep document fields: {sorted(unknown)}")
    base = dict(document.get("base") or {})
    defaults = {item.name: (item.default_factory() if callable(item.default_factory) else item.default)
                for item in fields(WorkloadSetting) if not (item.default is MISSING and item.default_factory is MISSING)}
    anchor = {**defaults, **base, **dict(document.get("anchor") or {})}
    for name, value in anchor.items():
        if _is_swept(value):
            raise ConfigurationError(f"anchor/base value of {name!r} must be a scalar, not a sweep")
    out: list[dict[str, Any]] = []
    seen: dict[str, int] = {}

    def add(setting_id: str, params: Mapping[str, Any], origin: Mapping[str, Any]):
        digest = sha256_text(canonical_json(params))
        if digest in seen:
            out[seen[digest]].setdefault("aliases", []).append(setting_id)
            return
        seen[digest] = len(out)
        out.append({"setting_id": setting_id, "params": dict(params), "origin": dict(origin)})

    if "anchor" in document:
        add("anchor", anchor, {"kind": "anchor"})
    for name, value in (document.get("sweep") or {}).items():
        if name not in anchor:
            raise ConfigurationError(f"swept parameter {name!r} has no anchor value")
        for item in expand_values(value):
            if item == anchor[name]:
                continue
            add(f"{name}={_label(item)}", {**anchor, name: item}, {"kind": "one_factor", "parameter": name,
                                                                  "value": item})
    for index, block in enumerate(document.get("settings") or []):
        if not isinstance(block, Mapping):
            raise ConfigurationError("settings entries must be objects")
        block = dict(block)
        block_id = str(block.pop("id", f"setting{index + 1}"))
        swept = sorted(name for name, value in block.items() if _is_swept(value))
        fixed = {name: value for name, value in block.items() if name not in swept}
        for combo in itertools.product(*(expand_values(block[name]) for name in swept)):
            chosen = dict(zip(swept, combo))
            suffix = ",".join(f"{name}={_label(value)}" for name, value in chosen.items())
            add(f"{block_id}:{suffix}" if suffix else block_id, {**anchor, **fixed, **chosen},
                {"kind": "explicit", "block": block_id, **({"values": chosen} if chosen else {})})
    if not out:
        raise ConfigurationError("the sweep document expands to no setting")
    return out


def check_settings(expanded: Sequence[Mapping[str, Any]], *, bank: Any = None,
                   sweep: Mapping[str, Any] | None = None, keep: bool = False) -> dict[str, Any]:
    """Feasibility of every expanded setting (validation, structural rules, full in-memory generation), the
    feasible region of each one-factor sweep and the content-slot counts; nothing is written. ``keep`` also
    returns the built workloads (``report["workloads"][setting_id]``)."""
    entries = load_bank(bank)
    rows, workloads, union, union_unbound = [], {}, {}, set()
    for item in expanded:
        row: dict[str, Any] = {"setting_id": item["setting_id"], "origin": item["origin"],
                               "aliases": item.get("aliases", [])}
        try:
            setting = WorkloadSetting.from_mapping(item["params"])
        except ConfigurationError as exc:
            row.update(valid=False, feasible=False, problems=[{"code": "invalid", "message": str(exc)}])
            rows.append(row)
            continue
        row.update(valid=True, setting_hash=setting_id_hash(setting))
        try:
            workload = build_workload(setting, bank=entries, setting_id=item["setting_id"], sweep=sweep)
        except InfeasibleError as exc:
            row.update(feasible=False, problems=exc.details.get("problems") or [exc.to_dict()])
            rows.append(row)
            continue
        coverage = workload.coverage
        kinds = coverage["bindings"]["slots"]
        row.update(feasible=True, problems=[], requests=coverage["requests"],
                   realized_p_historical=coverage["p_historical"]["realized"],
                   old_branch_targets=coverage["p_historical"]["old_branch_targets"],
                   slots=len(workload.slots), slots_unbound=coverage["bindings"]["slots_unbound"],
                   slots_by_kind=kinds, frozen=workload.frozen,
                   window_s=coverage["schedule"].get("window_s"))
        for slot in workload.slots:
            union.setdefault(slot["key"], slot["kind"])
            if not slot["bound"]:
                union_unbound.add(slot["key"])
        if keep:
            workloads[item["setting_id"]] = workload
        rows.append(row)
    region: dict[str, dict[str, bool]] = {}
    for row in rows:
        origin = row["origin"]
        if origin.get("kind") == "one_factor":
            region.setdefault(origin["parameter"], {})[_label(origin["value"])] = bool(row["feasible"])
    anchor = next((row for row in rows if row["origin"].get("kind") == "anchor"), None)
    report = {
        "schema": REPORT_SCHEMA, "generator_version": GENERATOR_VERSION, "sweep": dict(sweep or {}) or None,
        "settings": rows, "count": len(rows), "feasible": sum(bool(r["feasible"]) for r in rows),
        "infeasible": [r["setting_id"] for r in rows if not r["feasible"]],
        "anchor_feasible": None if anchor is None else bool(anchor["feasible"]), "region": region,
        "content_slots_union": {"keys": len(union), "unbound": len(union_unbound),
                                "by_kind": {kind: sum(1 for k in union.values() if k == kind)
                                            for kind in sorted(set(union.values()))}},
    }
    if keep:
        report["workloads"] = workloads
    return report


def union_slots(workloads: Iterable[Workload]) -> list[dict[str, Any]]:
    """Distinct content slots (by key) over several workloads, first occurrence kept, ordered by key."""
    slots: dict[str, dict[str, Any]] = {}
    for workload in workloads:
        for slot in workload.slots:
            if slot["key"] in slots:
                merged = slots[slot["key"]]
                merged["used_by"] = sorted(set(merged["used_by"]) | set(slot["used_by"]))
                merged.setdefault("settings", []).append(workload.setting_id)
            else:
                slots[slot["key"]] = {**slot, "settings": [workload.setting_id]}
    return [slots[key] for key in sorted(slots)]


__all__ = [
    "ARRIVAL_MODES", "BANK_SCHEMA", "BUNDLE_FILES", "BUNDLE_SCHEMA", "COARSE_CATEGORY_RULES", "DEFAULT_MODELS",
    "Dataset", "GAP_PROFILES", "GENERATOR_VERSION", "InfeasibleError", "PROMPT_HOLD_CHUNKS", "STREAMS",
    "SERVICE_TIME_HINTS_S", "SWEEP_SCHEMA", "Rates", "SessionTree", "TreeNode", "Uniforms", "Workload",
    "WorkloadSetting", "assign_open_loop",
    "bind_content", "build_workload", "canonical_json", "check_settings", "choose_age", "coarse_category",
    "expand_sweep", "expand_values", "gap_distribution", "load_bank", "load_bundle", "load_dataset",
    "main_prompt_index", "open_loop_arrivals", "plan_session", "prompt_segments", "rewind_and_generate",
    "schedule_requests",
    "select_cohort", "slot_key", "structural_problems", "think_seconds", "union_slots", "validate_bank",
    "write_bundle",
]
