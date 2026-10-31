"""CPU-importable contracts for isolated, source-bound MAGI study epochs.

The epoch owns one model/configuration, but no request state. A backend must
create and clear an entire request capsule. Any execution, collective or
cleanup failure poisons the epoch; restarting one rank is never supported.

The CLI consumes the parent's ``reelcache.rank_requests.v1`` batch. Its
hash-covered MAGI fields are ``cell_id``, ``asset_id``, ``native_source``,
``output``, ``asset_layout`` (config, dit_index, t5_directory, vae_config,
vae_weights, special_tokens; optional ffmpeg), ``resources`` (MagiResourceBudget fields),
``reconstruction_bytes``, ``store_bytes``, ``native_layer_bytes`` and
``spectrum_workspace_bytes`` and ``cpu_threads`` (per process, with the entire
worker group fitting its CPU allocation). Optional fields are
``spectrum_tile_rows``, ``retain_originals`` (request_id:chunk:layer:target),
``cache_mode`` (runtime/quality_hot), and ``quality_hot_bytes``. Every
allowance must fit each immutable request's budgets and history mode.
Every study batch must explicitly hash-cover ``sampler_mode="serial_clean_history"``
and its verified config must declare integer ``runtime_config.window_size=1``.
This is sixteen denoise steps followed by standalone clean recapture for EACH
chunk, not the historical window-four interleaved production transport.
``runtime_config.num_frames`` is the stock input builder's default horizon, not
a maximum or the study worker's loop bound. This worker does not call that
builder: the hash-covered request's ``chunks`` determines noise draws, cache
capacity, six-frame latent chunks and output count, without padding or resizing.
The optional hash-covered ``collective_timeout_seconds`` defaults to 600 and
must be an integer in [1,21600]. It binds worker NCCL/Gloo and all native engine
groups. The recorded deadline is a fault-detection setting, not evidence that a
whole owner seal or launch fits it; measured work and allocation wall time must
still be admitted separately before launch.

Individual requests use the parent's prompt/group/split, seed, chunks,
representation and output_namespace fields, plus an explicit paired ``case_id``
that is never inferred from an arm's request ID. An optional ``trajectory_id``
binds the current arm; otherwise the fresh execution namespace is its identity.
Omitted ``observe_sources`` enables RAW capture and exact-policy reuse only.
Approximate adaptive and static factor requests default to no offline oracle;
``observe_sources=true`` explicitly opts into separately charged instrumentation.
A representation's ``rank_policy``
is the complete typed policy document, never a replacement static rank map.
Static factor controls may instead supply ``representation.layer_ranks`` with
complete ``K``/``V`` integer lists, one rank per layer. They are mutually
exclusive with ``rank_policy`` and uniform ``rank_k``/``rank_v``. The request
freezes both lists into tuples; native encoding receives a fixed CodecSpec map.
Generation metadata explicitly labels ``rank_selection_mode`` as raw, uniform,
static_layerwise or adaptive. Native snapshots retain this source-bound request
under ``study_request`` as well as the complete committed fixed codec profile.
Use ``magi_model_identity`` to bind that policy to the same verified model.
Original source callbacks receive an owner-assembled [N,1024] matrix plus
portable dtype/hash/row-shard/position/history/request provenance. Hashing,
source staging and spectra remain charged offline study instrumentation.
Each ``layerNNN_K.json`` (or V) is a ``reelcache.magi_clean_observation.v2``
wrapper referencing an atomically written complete ``layerNNN_K.spectrum.json``
via ``spectrum_artifact`` path/SHA256/bytes, and the standard numeric
``layerNNN_K.rank.json`` via ``rank_observation_artifact``. The numeric identity
is bound into SpectrumSource.provenance before the oracle runs. The canonical
spectrum file is ``reelcache.spectrum.v1``; its file hash, framed source-tensor hash
and labelled headerless native value-byte hash are distinct.
Exact-policy observers return only prebound source provenance until core commits.
The worker then exports the SAME full oracle observation from each committed
RankDecision, without another eigensolve or retained tensor/direction queue.
RAW census and approximate-policy instrumentation use the standalone observer.
Each frozen disk grant covers staging, the capped runtime store, selected
originals, latents, complete spectra and conservative serialized metadata
allowances, as well as retained lossless RGB and decoder metadata. Batch
admission adds persistent generation output to the larger of transient staging
and ALL retained RGB/decoder metadata. Filesystem free-space and decode-phase
checks remain separate from each immutable request's disk grant.

Native preflight accepts the hash-covered batch selector ``preflight_backend``
(``reelcache`` by default, or RAW-only ``native_noop``). The latter uses only the
fresh native InferenceParams cache, never KVRuntime/MagiAdapter or snapshots.
``preflight_input_bundles`` maps request IDs to ``{"mode":"capture"}`` (the RAW
default) or ``{"mode":"replay","path":...,"sha256":...}``. Capture writes only
conditioning/masks and every fresh noise plus its immediate postdraw private RNG
state. Replay re-encodes conditioning and still draws/validates each chunk in
order. It never supplies generated latents/prefixes. These extras must precede
the parent's batch hash; they are rejected outside native_preflight.
Results reference ``reelcache.magi_preflight_inputs.v1`` bundles and per-rank JSONL
``reelcache.magi_attention_digest.v1`` records. Join digest records on binding_id,
rank, chunk_index, phase, step_index, layer, attention_call and visibility ranges;
hashes describe actual post-RoPE BF16 K/direct V, including archived prefixes.
All records are diagnostic inputs to the parent's proof collector, not passes.
RAW ``calibration_census`` additionally requires hash-covered ``detector_spec``
(the complete shared DetectorSpec), ``detector_workspace_bytes`` and
``detector_solver_workspace_bytes``. Its original-source callback runs the
shared fixed CPU QR/q2 acquisition before release, not on the saved spectrum.
Each ``layerNNN_K.detector.json`` contains the official
``reelcache_rank_detector_observations_v1`` document with BOTH threshold records,
the full recipe/fingerprint and actual acquisition/work telemetry. Only the
calibration split is eligible. The worker's explicit CPU thread count must
match the immutable detector recipe and fit the whole worker-group CPU grant.
Its native wrapper contains one charged detector work record shared by the two
threshold rows.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path, PurePosixPath
import random
import shutil
import struct
import threading
import time
from typing import Any, Mapping

from ..distributed import PeerFailure
from ..errors import ConfigurationError, IntegrityError, StateError
from ..study import (
    NATIVE_SOURCE_FPS, canonical_json, identity, read_json, study_cell, validate_protocol,
    validate_request_batch, verify_assets, verify_launch_receipt,
)
from .magi import MagiGeometry, MagiResourceBudget


REQUEST_SCHEMA = "reelcache.rank_requests.v1"
SAMPLER_MODE = "serial_clean_history"


def serial_sampler_contract():
    return {
        "sampler_mode": SAMPLER_MODE, "sampler_window_size": 1,
        "denoise_steps_per_chunk": 16,
        "timetable": {"points": 17, "grid": "uniform_then_square", "transform": "sd3",
                      "shift": 3.0, "dtype": "float32"},
        "standalone_clean_recapture": True, "clean_timestep": 0.9999,
        "fwd_extra_1st_chunk": False, "near_clean_steps": [12, 13, 14, 15],
        "shortcut_mode_changes_16_step_grid": False,
        "native_window4_schedule_parity_claimed": False,
        "horizon_source": "request.chunks",
        "runtime_num_frames_role": "native_input_builder_default_not_worker_limit",
    }


def worker_thread_allowance(batch, *, world_size=2):
    threads = batch.get("cpu_threads")
    if type(threads) is not int or threads < 1 or world_size not in (1, 2):
        raise ConfigurationError("MAGI requires an explicit positive per-worker cpu_threads allowance")
    capacity = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    granted = os.environ.get("SLURM_CPUS_PER_TASK")
    if granted is not None:
        try:
            granted = int(granted)
        except ValueError as exc:
            raise ConfigurationError("The scheduler CPU allocation is not a positive integer") from exc
        if granted < 1:
            raise ConfigurationError("The scheduler CPU allocation must be positive")
        capacity = min(capacity, granted)
    if threads * world_size > capacity:
        raise ConfigurationError("The simultaneous MAGI worker threads exceed the CPU allocation")
    return threads


def worker_collective_timeout(batch):
    seconds = batch.get("collective_timeout_seconds", 600)
    if type(seconds) is not int or not 1 <= seconds <= 21600:
        raise ConfigurationError("collective_timeout_seconds must be an integer in [1,21600]")
    return seconds


def magi_model_identity(*, cell_id, asset_id, config_id, source_closure_id):
    """The policy and native worker must bind this identical native identity."""
    from ..types import ModelIdentity

    cell = study_cell(cell_id)
    if cell.model != "magi_distill":
        raise ConfigurationError("A MAGI model identity requires a MAGI cell")
    return ModelIdentity(
        "magi_distill", identity({"assets": asset_id, "config": config_id, "source": source_closure_id}),
        cell.layers, cell.channels, "global_KV_channels", "bfloat16",
    )


class MagiEpochAborted(StateError):
    restart_both_ranks = True


@dataclass(frozen=True)
class MagiRankRequest:
    request_id: str
    cell_id: str
    prompt_id: str
    group_id: str
    split: str
    prompt: str
    seed: int
    chunks: int
    case_id: str
    arm: str = "raw"
    rank_k: int | None = None
    rank_v: int | None = None
    policy: Mapping[str, Any] | None = None
    observe_sources: bool | None = None
    factor_kind: str = "svd"
    output_namespace: str | None = None
    arm_id: str | None = None
    source_fps: float = 24
    trajectory_id: str | None = None
    layer_ranks: tuple[tuple[int, ...], tuple[int, ...]] | None = None

    def __post_init__(self):
        cell = study_cell(self.cell_id)
        if cell.model != "magi_distill":
            raise ConfigurationError("MAGI requests cannot switch to another model")
        for name in ("request_id", "prompt_id", "group_id", "prompt", "case_id"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ConfigurationError(f"MAGI {name} must be explicit")
        if self.split not in ("development", "calibration", "heldout"):
            raise ConfigurationError("A MAGI request needs an explicit protocol split")
        if type(self.seed) is not int or not 0 <= self.seed < 2**63:
            raise ConfigurationError("MAGI generation seed must be a nonnegative signed-64-bit integer")
        if type(self.chunks) is not int or self.chunks < 1:
            raise ConfigurationError("MAGI must generate at least one chunk from its own first noise")
        if self.arm not in ("raw", "svd", "svd_hevc", "adaptive"):
            raise ConfigurationError("Unknown MAGI rank-study arm")
        if self.observe_sources is None:
            object.__setattr__(self, "observe_sources", self.arm == "raw" or (
                self.arm == "adaptive" and isinstance(self.policy, Mapping)
                and self.policy.get("solver") == "exact_oracle"
            ))
        if type(self.observe_sources) is not bool:
            raise ConfigurationError("observe_sources must be boolean")
        if self.layer_ranks is not None:
            if (self.arm not in ("svd", "svd_hevc") or self.policy is not None
                    or self.rank_k is not None or self.rank_v is not None):
                raise ConfigurationError("Static layer ranks exclude RAW, rank_policy and uniform ranks")
            vectors = self.layer_ranks
            if isinstance(vectors, Mapping):
                if set(vectors) != {"K", "V"}:
                    raise ConfigurationError("Static layer ranks require exactly K and V vectors")
                vectors = (vectors["K"], vectors["V"])
            if (not isinstance(vectors, tuple) or len(vectors) != 2
                    or any(not isinstance(values, (tuple, list)) or len(values) != cell.layers for values in vectors)
                    or any(type(rank) is not int or not 1 <= rank <= min(cell.tokens, cell.channels)
                           for values in vectors for rank in values)):
                raise ConfigurationError("Static layer ranks must contain one valid integer per layer and target")
            object.__setattr__(self, "layer_ranks", tuple(tuple(values) for values in vectors))
        if self.arm in ("svd", "svd_hevc") and self.layer_ranks is None:
            for rank in (self.rank_k, self.rank_v):
                if type(rank) is not int or not 1 <= rank <= min(cell.tokens, cell.channels):
                    raise ConfigurationError("Uniform MAGI K/V ranks must be explicitly valid")
        elif self.arm not in ("svd", "svd_hevc") and (self.rank_k is not None or self.rank_v is not None):
            raise ConfigurationError("RAW/adaptive requests cannot supply ignored uniform ranks")
        if (self.arm == "adaptive") != (self.policy is not None):
            raise ConfigurationError("Only an adaptive MAGI request carries a typed policy document")
        if self.factor_kind not in ("svd", "svd_hevc"):
            raise ConfigurationError("A MAGI adaptive policy must name its factor representation")
        if self.output_namespace is not None:
            _asset_name(self.output_namespace)
        if self.arm_id is not None and (not isinstance(self.arm_id, str) or not self.arm_id):
            raise ConfigurationError("arm_id must be an explicit nonempty label")
        if self.trajectory_id is not None and (
            not isinstance(self.trajectory_id, str) or not self.trajectory_id.strip()
        ):
            raise ConfigurationError("trajectory_id must be an explicit nonempty label")
        if isinstance(self.source_fps, bool) or self.source_fps != NATIVE_SOURCE_FPS[self.cell_id]:
            raise ConfigurationError("The qualified MAGI stateless chunk contract is 24 native FPS")
        canonical_json(asdict(self))

    @classmethod
    def from_protocol(cls, row, protocol, cell_id):
        allowed = {"request_id", "prompt_id", "seed", "chunks", "arm", "rank_k", "rank_v",
                   "policy", "observe_sources", "case_id", "trajectory_id"}
        if not isinstance(row, dict) or set(row) - allowed:
            raise ConfigurationError("Unknown MAGI request fields (prefix/noise skipping is unsupported)")
        if "observe_sources" in row and type(row["observe_sources"]) is not bool:
            raise ConfigurationError("An explicit observe_sources field must be boolean")
        prompts = {item["prompt_id"]: item for item in protocol["prompts"]}
        if row.get("prompt_id") not in prompts or row.get("seed") not in protocol["seeds"]:
            raise ConfigurationError("MAGI prompt/seed is not in the bound official protocol")
        prompt = prompts[row["prompt_id"]]
        return cls(**row, cell_id=cell_id, group_id=prompt["group_id"],
                   split=prompt["split"], prompt=prompt["text"])

    @classmethod
    def from_study_request(cls, row):
        representation = row["representation"]
        policy = representation.get("rank_policy")
        adaptive = policy is not None
        if "observe_sources" in row and type(row["observe_sources"]) is not bool:
            raise ConfigurationError("An explicit observe_sources field must be boolean")
        if "rank_policy" in representation and not isinstance(policy, Mapping):
            raise ConfigurationError("representation.rank_policy must be a complete policy mapping")
        if "layer_ranks" in representation and not isinstance(representation["layer_ranks"], Mapping):
            raise ConfigurationError("representation.layer_ranks must be the complete K/V mapping")
        if "layer_ranks" in representation and set(representation) & {"rank_policy", "rank_k", "rank_v"}:
            raise ConfigurationError("Static layer ranks exclude all policy and uniform rank fields")
        if adaptive and (("rank_k" in representation) != ("rank_v" in representation)):
            raise ConfigurationError("Legacy adaptive rank hints must contain both K and V fields")
        return cls(
            request_id=row["request_id"], cell_id=row["cell_id"],
            case_id=row.get("case_id"), trajectory_id=row.get("trajectory_id"),
            prompt_id=row.get("prompt_id") or "development-" + identity(row["prompt"])[:16],
            group_id=row.get("group_id") or "development",
            split=row["split"], prompt=row["prompt"], seed=row["seed"], chunks=row["chunks"],
            arm="adaptive" if adaptive else representation["kind"],
            rank_k=None if adaptive else representation.get("rank_k"),
            rank_v=None if adaptive else representation.get("rank_v"), policy=policy,
            observe_sources=row.get("observe_sources"),
            factor_kind=representation["kind"] if representation["kind"] != "raw" else "svd",
            output_namespace=row["output_namespace"],
            arm_id=row.get("arm_id"), source_fps=row["source_fps"],
            layer_ranks=representation.get("layer_ranks"),
        )

    @property
    def rank_selection_mode(self):
        if self.arm == "raw":
            return "raw"
        if self.arm == "adaptive":
            return "adaptive"
        return "static_layerwise" if self.layer_ranks is not None else "uniform"

    @property
    def layer_rank_map(self):
        if self.layer_ranks is None:
            return None
        return {target: list(self.layer_ranks[index]) for index, target in enumerate(("K", "V"))}

    def static_specs(self):
        from ..types import CodecSpec, MatrixKey

        if self.arm not in ("svd", "svd_hevc") or self.policy is not None:
            raise ConfigurationError("Only a static factor request has a fixed CodecSpec map")
        return {
            MatrixKey(layer, target): CodecSpec(
                kind=self.arm,
                rank=(self.layer_ranks[index][layer] if self.layer_ranks is not None
                      else self.rank_k if target == "K" else self.rank_v),
                normalization="per_channel", pix_fmt="yuv444p", qp=8, preset="ultrafast",
            )
            for layer in range(study_cell(self.cell_id).layers)
            for index, target in enumerate(("K", "V"))
        }

    @property
    def request_hash(self):
        return identity(asdict(self))

    @property
    def latent_shape(self):
        cell = study_cell(self.cell_id)
        return (1, 16, cell.frames, cell.token_height * 2, cell.token_width * 2)

    def native_kwargs(self, index, *, clean, step, intervals, near_steps=(12, 13, 14, 15)):
        if type(index) is not int or not 0 <= index < self.chunks:
            raise ConfigurationError("Native chunk position is outside this request")
        if type(step) is not int or not 0 <= step < 16:
            raise ConfigurationError("The qualified distilled sampler has sixteen steps")
        near = not clean and step in near_steps
        return {
            "chunk_width": 6, "num_steps": 16, "slice_point": index, "range_num": index + 1,
            "denoising_range_num": 1, "fwd_extra_1st_chunk": False, "start_chunk_id": index,
            "end_chunk_id": index + 1 + int(near), "chunk_num": self.chunks,
            "extract_prefix_video_feature": clean, "distill_nearly_clean_chunk": near,
            "distill_interval": intervals[step % len(intervals)],
        }


def _asset_name(value):
    if not isinstance(value, str) or not value or "\\" in value:
        raise ConfigurationError("MAGI asset paths must be normalized relative names")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or str(path) != value or value == ".":
        raise ConfigurationError("MAGI assets must remain in their verified local root")
    return value


def validate_magi_checkpoint(root, files, index_name):
    """Validate exact index-listed mixed native dtypes without loading tensors."""
    root = Path(root)
    index_name = _asset_name(index_name)
    if index_name not in files:
        raise IntegrityError("The staged MAGI checkpoint index is missing")
    index = read_json(root / index_name)
    weights = index.get("weight_map")
    if not isinstance(weights, dict) or not weights or any(
        not isinstance(name, str) or not name or not isinstance(shard, str)
        for name, shard in weights.items()
    ):
        raise IntegrityError("Invalid MAGI safetensors weight index")
    directory = PurePosixPath(index_name).parent
    shards = {}
    for tensor, shard in weights.items():
        name = str(directory / _asset_name(shard))
        if name not in files or not name.endswith(".safetensors"):
            raise IntegrityError("Every index-listed MAGI shard must be explicitly staged")
        shards.setdefault(name, set()).add(tensor)
    encountered, dtypes = set(), set()
    descriptors = {}

    def unique_object(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise IntegrityError("Duplicate native safetensors header key")
            value[key] = item
        return value

    for name, expected in sorted(shards.items()):
        with (root / name).open("rb") as stream:
            prefix = stream.read(8)
            if len(prefix) != 8:
                raise IntegrityError("Truncated MAGI safetensors shard")
            size = struct.unpack("<Q", prefix)[0]
            if not 2 <= size <= min(128 * 1024**2, files[name]["bytes"] - 8):
                raise IntegrityError("Invalid MAGI safetensors header size")
            raw = stream.read(size)
        try:
            header = json.loads(raw, object_pairs_hook=unique_object)
        except (UnicodeError, ValueError) as exc:
            raise IntegrityError("Invalid MAGI safetensors header") from exc
        if not isinstance(header, dict):
            raise IntegrityError("Native safetensors header must be a tensor table")
        names = set(header) - {"__metadata__"}
        if names != expected or names & encountered:
            raise IntegrityError("MAGI shard keys differ from the exact checkpoint index")
        extents = []
        for key in names:
            description = header[key]
            dtype = description.get("dtype")
            shape, offsets = description.get("shape"), description.get("data_offsets")
            if dtype not in ("F32", "BF16") or not isinstance(shape, list) or any(
                type(value) is not int or value < 0 for value in shape
            ):
                raise IntegrityError("MAGI checkpoint changed its native mixed-F32/BF16 schema")
            count = 1
            for dimension in shape:
                count *= dimension
            if (not isinstance(offsets, list) or len(offsets) != 2
                    or any(type(value) is not int for value in offsets)
                    or not 0 <= offsets[0] <= offsets[1] <= files[name]["bytes"] - size - 8
                    or offsets[1] - offsets[0] != count * (4 if dtype == "F32" else 2)):
                raise IntegrityError("MAGI tensor byte extent differs from its declared native dtype")
            dtypes.add(dtype)
            extents.append(offsets)
            descriptors[key] = {"dtype": dtype, "shape": shape, "shard": name}
        offset = 0
        for start, end in sorted(extents):
            if start != offset:
                raise IntegrityError("MAGI tensor extents overlap or leave unindexed byte ranges")
            offset = end
        if offset != files[name]["bytes"] - size - 8:
            raise IntegrityError("MAGI shard has unindexed trailing tensor bytes")
        encountered.update(names)
    if dtypes != {"F32", "BF16"}:
        raise IntegrityError("MAGI requires the original mixed-F32/BF16 checkpoint, not blanket conversion")
    return {"index": index_name, "shards": sorted(shards), "tensors": descriptors,
            "native_promotion": "_high_precision_promoter", "blanket_dtype_conversion": False}


def validate_asset_layout(root, manifest, layout):
    required = {"dit_index", "t5_directory", "vae_config", "vae_weights", "special_tokens", "config"}
    if (not isinstance(layout, dict) or not required.issubset(layout)
            or set(layout) - required - {"ffmpeg"}):
        raise ConfigurationError(f"Explicit MAGI asset_layout keys required: {sorted(required)}")
    files = {row["name"]: row for row in manifest["files"]}
    for role, name in layout.items():
        _asset_name(name)
        if role != "t5_directory" and name not in files:
            raise IntegrityError(f"Missing verified local MAGI asset: {role}")
    t5 = layout["t5_directory"].rstrip("/") + "/"
    for name in ("config.json", "spiece.model", "tokenizer_config.json", "special_tokens_map.json"):
        if t5 + name not in files:
            raise IntegrityError(f"Missing verified local T5 asset: {name}")
    index_names = [t5 + name for name in ("model.safetensors.index.json", "pytorch_model.bin.index.json")
                   if t5 + name in files]
    if not index_names and t5 + "model.safetensors" not in files:
        raise IntegrityError("A complete local T5 weight index or single safetensors file is required")
    for name in index_names:
        index = read_json(Path(root) / name)
        if not isinstance(index.get("weight_map"), dict) or not index["weight_map"]:
            raise IntegrityError("Invalid staged T5 weight index")
        for shard in set(index["weight_map"].values()):
            if t5 + _asset_name(shard) not in files:
                raise IntegrityError("A T5 shard is absent from the verified local asset set")
    # A loader must not discover an unverified alternative checkpoint/tokenizer.
    actual = {
        str(path.relative_to(root)) for path in (Path(root) / layout["t5_directory"]).rglob("*")
        if path.is_file()
    }
    if actual != {name for name in files if name.startswith(t5)}:
        raise IntegrityError("The local T5 directory contains missing or unlisted assets")
    return validate_magi_checkpoint(root, files, layout["dit_index"])


def _decode_disk_admission(cell, chunks):
    rgb = chunks * 24 * 3 * cell.width * cell.height
    metadata = chunks * (1 << 20)
    return {"retained_rgb_bytes": rgb, "decoder_metadata_bytes": metadata,
            "required_bytes": rgb + metadata}


def _batch_disk_admission(requests, *, staging_bytes):
    persistent = sum(row["persistent_bytes"] for row in requests)
    rgb = sum(row["retained_rgb_bytes"] for row in requests)
    metadata = sum(row["decoder_metadata_bytes"] for row in requests)
    return {
        "persistent_generation_bytes": persistent, "transient_staging_bytes": staging_bytes,
        "retained_rgb_bytes": rgb, "decoder_metadata_bytes": metadata,
        "generation_peak_bytes": persistent + staging_bytes,
        "decode_peak_bytes": persistent + rgb + metadata,
        "required_bytes": persistent + max(staging_bytes, rgb + metadata),
    }


def _request_disk_admission(request, cell, *, staging_bytes, store_bytes, retained,
                            preflight=None, observer=None, detector=False):
    tensor_overhead = 16 * 1024
    retained_count = sum(retained.values())
    retained_payload = cell.tokens * cell.channels * (4 * retained["K"] + 2 * retained["V"])
    latent_payload = math.prod(request.latent_shape) * 4 * request.chunks
    observes = request.observe_sources if observer is None else observer
    observations = 2 * cell.layers * request.chunks if observes else 0
    request_bytes = len(canonical_json(asdict(request)))
    description_bound = request_bytes + (16 << 10) + cell.layers * request.chunks * 512
    # Spectra contain O(C) numerical arrays, not directions. Reserve separately
    # for their repeated capture/identity wrappers, audit records and batch index.
    spectrum_bytes = observations * ((cell.channels + 1) * 512 + (64 << 10))
    metadata_bytes = (4 * observations + 16) * description_bound
    if preflight:
        metadata_bytes += preflight["final_noise_rng_metadata_bytes"]
    components = {
        "staging_bytes": staging_bytes, "store_bytes": store_bytes,
        "retained_original_bytes": retained_payload + retained_count * tensor_overhead,
        "latent_bytes": latent_payload + request.chunks * tensor_overhead,
        "spectrum_bytes": spectrum_bytes, "metadata_bytes": metadata_bytes,
        "input_bundle_bytes": preflight["input_bundle_bytes"] if preflight else 0,
        "attention_digest_bytes": preflight["attention_digest_bytes"] if preflight else 0,
        "detector_record_bytes": observations * 2 * ((128 << 10) + 4 * request_bytes) if detector else 0,
    }
    persistent = sum(value for name, value in components.items() if name != "staging_bytes")
    decoded = _decode_disk_admission(cell, request.chunks)
    return {
        **components, "request_id": request.request_id,
        "retained_rgb_bytes": decoded["retained_rgb_bytes"],
        "decoder_metadata_bytes": decoded["decoder_metadata_bytes"],
        "retained_original_payload_bytes": retained_payload,
        "latent_payload_bytes": latent_payload,
        "observation_count": observations,
        "required_bytes": persistent + max(staging_bytes, decoded["required_bytes"]),
        "persistent_bytes": persistent,
        "accounting": "persistent generation plus max(staging, retained RGB and decoder metadata)",
    }


def verify_worker_inputs(protocol_path, assets_path, receipt_path, requests_path, *, check_storage=True,
                         execution_world_size=2):
    """Called by the CLI before importing any native module or constructing CUDA."""
    protocol = validate_protocol(read_json(protocol_path))
    receipt = read_json(receipt_path)
    verify_launch_receipt(receipt, protocol, phase=receipt.get("phase"))
    batch = read_json(requests_path)
    if not isinstance(batch, dict) or batch.get("schema") != REQUEST_SCHEMA:
        raise ConfigurationError("Unsupported MAGI request batch")
    if batch.get("protocol_id") != protocol["protocol_id"]:
        raise IntegrityError("MAGI request batch belongs to another protocol")
    cell = study_cell(batch.get("cell_id"))
    if cell.model != "magi_distill":
        raise ConfigurationError("A MAGI epoch must bind exactly one MAGI resolution")
    approved = validate_request_batch(
        batch, protocol, receipt, cell_id=cell.cell_id, phase=receipt["phase"],
    )
    if batch.get("sampler_mode") != SAMPLER_MODE:
        raise ConfigurationError("MAGI study batches must bind sampler_mode='serial_clean_history'")
    collective_timeout_seconds = worker_collective_timeout(batch)
    backend = batch.get("preflight_backend", "reelcache")
    if backend not in ("reelcache", "native_noop"):
        raise ConfigurationError("preflight_backend must be reelcache or native_noop")
    preflight = receipt["phase"] == "native_preflight"
    if backend == "native_noop" and (
        not preflight or any(row["representation"] != {"kind": "raw"} for row in approved)
    ):
        raise ConfigurationError("native_noop is restricted to native_preflight RAW requests")
    if not preflight and "preflight_input_bundles" in batch:
        raise ConfigurationError("Input-bundle instrumentation is restricted to native_preflight")
    worker_thread_allowance(batch, world_size=execution_world_size)
    assets_path = Path(assets_path).resolve(strict=True)
    if assets_path.name != "assets.json":
        raise ConfigurationError("--assets must name the verified local assets.json manifest")
    manifest = verify_assets(assets_path.parent, rehash=True)
    if batch.get("asset_id") != manifest["asset_id"]:
        raise IntegrityError("MAGI batch asset identity differs from the verified local copy")
    checkpoint = validate_asset_layout(assets_path.parent, manifest, batch.get("asset_layout"))
    config = read_json(assets_path.parent / batch["asset_layout"]["config"])
    from .magi_native import _validate_config

    _validate_config(config, cell_id=cell.cell_id, study_worker=True)
    records = receipt["source_closure"]["files"].values()
    bound = {str(Path(row["path"]).resolve()) for row in records}
    source = Path(batch["native_source"]).resolve(strict=True)
    native_files = list((source / "inference").rglob("*.py"))
    if not native_files or any(str(path.resolve()) not in bound for path in native_files):
        raise IntegrityError("The complete native MAGI source tree must belong to the reviewed launch closure")
    requests = [MagiRankRequest.from_study_request(row) for row in approved]
    if not requests:
        raise ConfigurationError("An admitted MAGI epoch cannot be empty")
    needs_hevc = any(request.arm == "svd_hevc" or (
        request.arm == "adaptive" and request.factor_kind == "svd_hevc"
    ) for request in requests)
    ffmpeg_name = batch["asset_layout"].get("ffmpeg")
    ffmpeg = None if ffmpeg_name is None else str((assets_path.parent / ffmpeg_name).resolve(strict=True))
    if needs_hevc and (ffmpeg is None or not os.access(ffmpeg, os.X_OK)):
        raise ConfigurationError("HEVC study arms require an executable verified local asset_layout.ffmpeg")
    expected_model = magi_model_identity(
        cell_id=cell.cell_id, asset_id=manifest["asset_id"], config_id=identity(config),
        source_closure_id=receipt["source_closure"]["closure_id"],
    )
    input_bundles = {}
    preflight_resources = {}
    detector_spec = None
    detector_admission_bytes = 0
    detector_fields = {"detector_spec", "detector_workspace_bytes", "detector_solver_workspace_bytes"}
    census = receipt["phase"] == "calibration_census"
    if census:
        from ..factorization import DetectorSpec, rank_fit_workspace_bytes

        if any(request.split != "calibration" for request in requests):
            raise ConfigurationError("RAW detector-bank acquisition is restricted to the calibration split")
        if any(request.arm != "raw" or not request.observe_sources for request in requests):
            raise ConfigurationError("RAW census must observe every original matrix and its fixed detector")
        if not detector_fields.issubset(batch):
            raise ConfigurationError("RAW census requires the complete frozen detector recipe and workspace grants")
        detector_spec = DetectorSpec.from_dict(batch["detector_spec"])
        if detector_spec.cpu_threads != batch["cpu_threads"]:
            raise ConfigurationError("RAW detector CPU threads differ from the admitted worker count")
        if detector_spec.fingerprint != DetectorSpec(cpu_threads=detector_spec.cpu_threads).fingerprint:
            raise ConfigurationError("RAW detector recipe/code/backend/threads differ from the frozen native epoch")
        for name in detector_fields - {"detector_spec"}:
            if type(batch[name]) is not int or batch[name] <= 0:
                raise ConfigurationError("Detector workspace and solver reserve must be explicit positive bytes")
        detector_admission_bytes = cell.tokens * cell.channels * 4 + rank_fit_workspace_bytes(
            (cell.tokens, cell.channels), min(cell.tokens, cell.channels),
            tile_rows=detector_spec.tile_rows, solver_workspace_bytes=batch["detector_solver_workspace_bytes"],
        )
        if batch["detector_workspace_bytes"] < detector_admission_bytes:
            raise ConfigurationError("RAW detector grant cannot admit its owned original copy and full-width fallback")
    elif detector_fields.intersection(batch):
        raise ConfigurationError("Standalone detector acquisition is restricted to RAW calibration census")
    if preflight:
        from .magi_preflight import input_binding, preflight_allowances, verify_input_bundle

        plans = batch.get("preflight_input_bundles", {})
        if not isinstance(plans, dict) or set(plans) - {request.request_id for request in requests}:
            raise ConfigurationError("preflight_input_bundles must be keyed by this batch's request IDs")
        for request in requests:
            plan = plans.get(request.request_id, {"mode": "capture"})
            if not isinstance(plan, dict) or plan.get("mode") not in ("capture", "replay"):
                raise ConfigurationError("A preflight input bundle must explicitly capture or replay")
            binding = input_binding(
                request, protocol_id=protocol["protocol_id"], asset_id=manifest["asset_id"],
                config_id=identity(config), launch_closure_id=receipt["source_closure"]["closure_id"],
            )
            preflight_resources[request.request_id] = preflight_allowances(request, cell)
            if plan["mode"] == "capture":
                if set(plan) != {"mode"} or request.arm != "raw":
                    raise ConfigurationError("Capture input bundles on a RAW trial; other preflights must replay")
            else:
                if set(plan) != {"mode", "path", "sha256"}:
                    raise ConfigurationError("Replay needs an explicit input-bundle path and file SHA256")
                verify_input_bundle(plan["path"], plan["sha256"], binding,
                                    preflight_resources[request.request_id]["input_bundle_bytes"])
            input_bundles[request.request_id] = {**plan, "binding": binding}
    for request in requests:
        if request.policy is not None:
            from ..online_rank import RankPolicy
            from ..types import MatrixKey
            from .magi import _tensor_context

            policy = RankPolicy.from_dict(request.policy)
            if policy.model != expected_model or policy.codec.kind != request.factor_kind:
                raise ConfigurationError("MAGI policy model/source/representation differs before model loading")
            if policy.solver == "qr_q2":
                from ..factorization import DetectorSpec

                if policy.detector_spec.cpu_threads != batch["cpu_threads"]:
                    raise ConfigurationError("Adaptive detector CPU threads differ from the admitted worker count")
                if policy.detector_spec.fingerprint != DetectorSpec(
                    cpu_threads=policy.detector_spec.cpu_threads,
                ).fingerprint:
                    raise ConfigurationError("Adaptive detector recipe/code/backend differs before model loading")
            if policy.codec.ffmpeg_path is not None:
                raise ConfigurationError("Portable MAGI policies must use the manifest-bound codec context, not a path fallback")
            for layer in range(cell.layers):
                for target, dtype in (("K", "float32"), ("V", "bfloat16")):
                    policy.validate_context(_tensor_context(
                        expected_model, MatrixKey(layer, target), 0, MagiGeometry.for_cell(cell.cell_id),
                        "pre_rope" if target == "K" else "none", dtype,
                    ))
            if request.chunks > policy.history_max or policy.history_min != 1:
                raise ConfigurationError("The from-scratch trajectory exceeds its frozen policy history support")
    budget = MagiResourceBudget(**batch["resources"])
    mode = batch.get("cache_mode", "runtime")
    if mode not in ("runtime", "quality_hot"):
        raise ConfigurationError("Unknown MAGI cache mode")
    hot_bytes = batch.get("quality_hot_bytes", 0)
    if type(hot_bytes) is not int or hot_bytes < 0 or (mode == "quality_hot") != (hot_bytes > 0):
        raise ConfigurationError("Only explicit quality_hot admits a positive dense-history budget")
    for name in ("reconstruction_bytes", "store_bytes", "native_layer_bytes", "spectrum_workspace_bytes"):
        if type(batch.get(name)) is not int or batch[name] <= 0:
            raise ConfigurationError(f"MAGI requires a positive explicit {name} allowance")
    detector_workspace = batch["detector_workspace_bytes"] if census else 0
    if (batch["spectrum_workspace_bytes"] + batch["reconstruction_bytes"] + detector_workspace
            > budget.workspace_host_bytes):
        raise ConfigurationError("The simultaneous host workspace must cover spectrum and reconstruction")
    promoted = ("x_embedder.", "y_embedder.", "t_embedder.", "final_linear.", "rope.")
    native_model_bytes = 0
    for name, descriptor in checkpoint["tensors"].items():
        promotion = name.startswith(promoted) or (
            "_xattn" not in name and any(part in name for part in (
                "q_layernorm", "k_layernorm", "self_attn_post_norm", "mlp_post_norm", "final_layernorm",
            ))
        )
        native_model_bytes += math.prod(descriptor["shape"]) * (
            4 if descriptor["dtype"] == "F32" or promotion else 2
        )
    if budget.workspace_device_bytes < native_model_bytes:
        raise ConfigurationError("Device workspace allowance does not even cover the native promoted DiT")
    files = {row["name"]: row for row in manifest["files"]}
    checkpoint_bytes = sum(files[name]["bytes"] for name in checkpoint["shards"])
    t5_prefix = batch["asset_layout"]["t5_directory"] + "/"
    t5_bytes = sum(row["bytes"] for name, row in files.items() if name.startswith(t5_prefix))
    if budget.workspace_host_bytes < max(
        cell.world_size * checkpoint_bytes,
        2 * t5_bytes + batch["spectrum_workspace_bytes"] + batch["reconstruction_bytes"] + detector_workspace,
    ):
        raise ConfigurationError("Global host workspace must cover BOTH native checkpoint copies and FP32 T5 plus study work")
    for request in requests:
        if request.policy is not None:
            policy = RankPolicy.from_dict(request.policy)
            if policy.workspace_limit_bytes > budget.workspace_host_bytes:
                raise ConfigurationError("Adaptive policy workspace exceeds this admitted native epoch")
    for row in approved:
        limits = row["budgets"]
        if (budget.host_bytes > limits["host_bytes"] or budget.device_bytes > limits["device_bytes_per_rank"]
                or budget.workspace_device_bytes > limits["workspace_bytes_per_rank"]
                or budget.staging_bytes > limits["source_staging_bytes_global"]
                or mode != ("runtime" if row["history_mode"] == "reload" else "quality_hot")
                or (mode == "quality_hot" and hot_bytes > limits["history_bytes_per_rank"])):
            raise ConfigurationError("MAGI epoch allowances exceed an immutable request resource contract")
    geometry = MagiGeometry.for_cell(cell.cell_id)
    admissions = []
    for request in requests:
        ledger = budget.admit(geometry, chunks=request.chunks, factor_keys=request.arm != "raw",
                              observer=request.observe_sources and backend != "native_noop",
                              quality_hot=mode == "quality_hot" or backend == "native_noop")
        if preflight:
            overhead = preflight_resources[request.request_id]
            ledger["preflight_host_bytes"] = overhead["host_bytes"]
            ledger["preflight_device_bytes"] = overhead["device_bytes_per_rank"]
            ledger["host_bytes"] += overhead["host_bytes"]
            ledger["device_bytes"] += overhead["device_bytes_per_rank"]
            if ledger["host_bytes"] > budget.host_bytes or ledger["device_bytes"] > budget.device_bytes:
                raise ConfigurationError("Input-bundle/hash instrumentation exceeds the simultaneous resource grant")
        if mode == "quality_hot" and ledger["native_history_current_bytes"] > hot_bytes:
            raise ConfigurationError("The entire MAGI request exceeds the admitted quality-hot cap")
        admissions.append(ledger)
    if backend == "native_noop" and any(
        ledger["native_history_current_bytes"] > row["budgets"]["history_bytes_per_rank"]
        for row, ledger in zip(approved, admissions)
    ):
        raise ConfigurationError("native_noop requires its entire native dense history to be admitted")
    from .magi_native import _output

    output = _output(batch["output"])
    parent = output.parent
    while not parent.exists():
        parent = parent.parent
    retained = batch.get("retain_originals", [])
    if backend == "native_noop" and retained:
        raise ConfigurationError("The independent native baseline does not capture adapter source originals")
    if (not isinstance(retained, list) or any(not isinstance(value, str) for value in retained)
            or len(set(retained)) != len(retained)):
        raise ConfigurationError("Retained original selectors must be an explicit unique list")
    by_id = {value.request_id: value for value in requests}
    retained_by_request = {name: {"K": 0, "V": 0} for name in by_id}
    for selector in retained:
        try:
            name, chunk, layer, target = selector.split(":")
            value = by_id[name]
            if not (
                value.observe_sources and 0 <= int(chunk) < value.chunks
                and 0 <= int(layer) < cell.layers and target in ("K", "V")
                and str(int(chunk)) == chunk and str(int(layer)) == layer
            ):
                raise ValueError
        except (AttributeError, KeyError, ValueError) as exc:
            raise ConfigurationError(
                "Retained original selector must be canonical request_id:chunk:layer:K_or_V with observation enabled",
            ) from exc
        retained_by_request[name][target] += 1
    disk_admissions = [
        _request_disk_admission(
            request, cell, staging_bytes=budget.staging_bytes,
            store_bytes=batch["store_bytes"] if backend == "reelcache" else 0,
            retained=retained_by_request[request.request_id],
            preflight=preflight_resources.get(request.request_id),
            observer=request.observe_sources and backend != "native_noop",
            detector=census,
        )
        for request in requests
    ]
    for row, disk in zip(approved, disk_admissions):
        if disk["required_bytes"] > row["budgets"]["disk_bytes"]:
            raise ConfigurationError(
                "The complete MAGI request disk bound exceeds its immutable request disk_bytes: "
                f"{disk['required_bytes']} > {row['budgets']['disk_bytes']}",
            )
    batch_disk = _batch_disk_admission(
        disk_admissions, staging_bytes=max(item["staging_bytes"] for item in admissions),
    )
    storage = batch_disk["required_bytes"]
    reserve = max(row["budgets"]["disk_reserve_bytes"] for row in approved)
    if check_storage and output.exists():
        raise StateError("MAGI generation output must be new; a failed epoch requires a fresh namespace")
    if check_storage and shutil.disk_usage(parent).free < storage + reserve:
        from ..errors import CapacityError

        raise CapacityError("Insufficient storage for the complete MAGI batch, staging, originals and reserve")
    return {
        "protocol": protocol, "receipt": receipt, "batch": batch, "config": config,
        "assets": manifest, "asset_root": assets_path.parent, "checkpoint": checkpoint,
        "requests": requests, "budget": budget, "admissions": admissions,
        "disk_admissions": disk_admissions,
        "batch_disk_admission": batch_disk,
        "preflight_backend": backend, "input_bundles": input_bundles,
        "preflight_resources": preflight_resources,
        "detector_spec": detector_spec,
        "detector_admission_bytes": detector_admission_bytes,
        "sampler_contract": serial_sampler_contract(),
        "collective_timeout_seconds": collective_timeout_seconds,
        "batch_id": batch["requests_id"],
        "storage_admission_bytes": storage, "native_model_bytes_per_rank": native_model_bytes,
        "ffmpeg": ffmpeg,
    }


@contextmanager
def request_rng(seed, device):
    """Restore ambient RNG; every request owns new generator state."""
    import torch

    python_state = random.getstate()
    try:
        import numpy as np
    except ImportError:
        np = None
    numpy_state = None if np is None else np.random.get_state()
    devices = [device.index] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        random.seed(seed)
        torch.random.default_generator.manual_seed(seed)
        if devices:
            torch.cuda.manual_seed(seed)
        if np is not None:
            np.random.seed(seed % 2**32)
        try:
            yield
        finally:
            random.setstate(python_state)
            if np is not None:
                np.random.set_state(numpy_state)


class MagiWorkerEpoch:
    """Persistent model, fresh requests, execution-thread-only rank agreement.

    Backend API: ``new_request(request, namespace)``, ``generate(state)``,
    ``clear_request(state)``, ``assert_clean()``, ``close()`` and ``device``.
    ``clear_request(None)`` must clear partially constructed state as well.
    """

    def __init__(self, *, cell_id, config_id, epoch_id, transport, backend_factory,
                 budget, cache_mode="runtime", quality_hot_bytes=0,
                 preflight_backend="reelcache", phase=None, preflight_resources=None):
        self.geometry = MagiGeometry.for_cell(cell_id)
        if transport.world_size != 2:
            raise ConfigurationError("MAGI rank-study epochs require the ordered SP2 rank group")
        if not isinstance(budget, MagiResourceBudget):
            raise ConfigurationError("The MAGI epoch requires explicit simultaneous memory allowances")
        if not config_id or not epoch_id or not callable(backend_factory):
            raise ConfigurationError("MAGI epoch/configuration identities and a backend are required")
        if cache_mode not in ("runtime", "quality_hot"):
            raise ConfigurationError("Unknown MAGI epoch cache mode")
        if (type(quality_hot_bytes) is not int or quality_hot_bytes < 0
                or (cache_mode == "quality_hot") != (quality_hot_bytes > 0)):
            raise ConfigurationError("The explicit quality-hot epoch needs its own positive byte allowance")
        self.cell_id, self.config_id, self.epoch_id = cell_id, config_id, epoch_id
        if preflight_backend not in ("reelcache", "native_noop") or (
            preflight_backend == "native_noop" and phase != "native_preflight"
        ):
            raise ConfigurationError("The independent native backend is preflight-only")
        self.preflight_backend = preflight_backend
        self.preflight_resources = preflight_resources or {}
        self.transport, self.backend_factory, self.budget = transport, backend_factory, budget
        self.cache_mode, self.quality_hot_bytes = cache_mode, quality_hot_bytes
        self.backend = None
        self.sequence = 0
        self.poisoned = False
        self.closed = False
        self._thread = threading.get_ident()
        self.completed = []
        self.startup_seconds = 0.0
        self.failure = None

    def _guard(self):
        if threading.get_ident() != self._thread:
            raise StateError("MAGI collectives must execute on the epoch execution thread")
        if self.poisoned or self.closed:
            raise MagiEpochAborted("MAGI epoch is closed/poisoned; restart BOTH ranks")

    def _agreement(self, operation, details=None, error=None):
        rows = self.transport.checkpoint(operation, details=details, error=error)
        if any(row != rows[0] for row in rows):
            raise StateError("MAGI ranks disagree on request order, identity or configuration")

    def _abort(self, reason):
        self.poisoned = True
        if self.backend is not None:
            try:
                self.backend.close()
            except BaseException:
                pass
            self.backend = None
        raise MagiEpochAborted(
            f"MAGI epoch aborted; restart BOTH ranks, never reuse this group: {reason}"
        ) from reason

    def run(self, request):
        self._guard()
        attempt_started = time.perf_counter()
        state, error, result = None, None, None
        namespace = None
        try:
            validation_error, ledger = None, None
            try:
                if not isinstance(request, MagiRankRequest) or request.cell_id != self.cell_id:
                    raise ConfigurationError("Resolution/configuration switching requires a new MAGI epoch")
                if self.preflight_backend == "native_noop" and request.arm != "raw":
                    raise ConfigurationError("The independent native backend accepts only RAW preflight")
                ledger = self.budget.admit(
                    self.geometry, chunks=request.chunks, factor_keys=request.arm != "raw",
                    observer=request.observe_sources and self.preflight_backend != "native_noop",
                    quality_hot=self.cache_mode == "quality_hot" or self.preflight_backend == "native_noop",
                )
                extra = self.preflight_resources.get(request.request_id)
                if extra is not None:
                    ledger["host_bytes"] += extra["host_bytes"]
                    ledger["device_bytes"] += extra["device_bytes_per_rank"]
                    if ledger["host_bytes"] > self.budget.host_bytes or ledger["device_bytes"] > self.budget.device_bytes:
                        raise ConfigurationError("Preflight input/hash resources exceed the request admission")
                if self.cache_mode == "quality_hot" and (
                    ledger["native_history_current_bytes"] > self.quality_hot_bytes
                ):
                    raise ConfigurationError("MAGI request exceeds its quality-hot budget")
            except Exception as exc:
                validation_error = exc
            details = {
                "epoch_id": self.epoch_id, "sequence": self.sequence, "cell_id": self.cell_id,
                "config_id": self.config_id,
                "request_hash": request.request_hash if isinstance(request, MagiRankRequest) else None,
                "request_id": request.request_id if isinstance(request, MagiRankRequest) else None,
                "resources": asdict(self.budget), "cache_mode": self.cache_mode,
                "quality_hot_bytes": self.quality_hot_bytes,
                "preflight_backend": self.preflight_backend,
            }
            self._agreement("magi_request_agreement", details, validation_error)
            namespace = "magi-" + identity(details)
            if self.backend is None:
                started = time.perf_counter()
                try:
                    self.backend = self.backend_factory()
                except BaseException as exc:
                    error = exc
                self._agreement("magi_epoch_loaded", error=error)
                self.startup_seconds = time.perf_counter() - started
            error = None
            try:
                self.backend.assert_clean()
            except BaseException as exc:
                error = exc
            self._agreement("magi_request_reset_checked", error=error)
            started = time.perf_counter()
            with request_rng(request.seed, self.backend.device):
                try:
                    state = self.backend.new_request(request, namespace)
                    result = self.backend.generate(state)
                except BaseException as exc:
                    error = exc
                finally:
                    try:
                        self.backend.clear_request(state)
                        self.backend.assert_clean()
                    except BaseException as exc:
                        error = error or exc
            if isinstance(error, PeerFailure) or getattr(self.transport, "_broken", False):
                raise error or PeerFailure("MAGI transport is poisoned; abort both ranks")
            self._agreement("magi_request_completed_and_clean", {
                "namespace": namespace, "request_hash": request.request_hash,
            }, error)
            elapsed = time.perf_counter() - started
            timings = self.transport.checkpoint("magi_request_rank_timing", details={
                "rank": getattr(self.transport, "rank", 0), "seconds": elapsed,
            })
            record = {
                "namespace": namespace, "sequence": self.sequence, "request_hash": request.request_hash,
                "seconds_generation_and_cleanup": max(row["seconds"] for row in timings),
                "per_rank_generation_and_cleanup": timings,
                "resource_admission": ledger, "result": result,
            }
            self.completed.append({"namespace": namespace, "request_hash": request.request_hash})
            self.sequence += 1
            return record
        except BaseException as exc:
            self.failure = {
                "sequence": self.sequence, "namespace": namespace,
                "request_id": request.request_id if isinstance(request, MagiRankRequest) else None,
                "attempt_seconds_including_failed_work_and_cleanup": time.perf_counter() - attempt_started,
                "error": str(exc), "restart_both_ranks": True,
                "committed_state": "only the durable runtime head is authoritative; no implicit RAW rerun",
            }
            self._abort(exc)

    def close(self):
        if self.closed:
            return
        self._guard()
        error = None
        try:
            if self.backend is not None:
                self.backend.assert_clean()
                self.backend.close()
                self.backend = None
        except BaseException as exc:
            error = exc
        try:
            self._agreement("magi_epoch_unloaded", error=error)
        except BaseException as exc:
            self._abort(exc)
        self.closed = True
