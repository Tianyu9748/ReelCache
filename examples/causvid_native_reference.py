"""Development-only original-native reference; no native/model imports here.

The admitted original pipeline is called unchanged with ``start_latents=None``.
An instance-local VAE boundary exports its completed latent tensor, not pixels.
The matched RAW result supplies checks and post-noise RNG state, never a prefix.
Both outputs require separate real VAE epochs before RGB parity can be measured.
"""

from __future__ import annotations

from contextlib import contextmanager
import gc
import inspect
from pathlib import Path
import time
from typing import Mapping

from examples import causvid_rank_worker as worker
from reelcache.errors import CapacityError, ConfigurationError, IntegrityError, StateError
from reelcache.types import MatrixKey, TensorContext


STATE_HASHES = {
    "noise": "initial_noise_sha256", "prompt_embeds": "conditioning_tensor_sha256",
    "initial_rng_state": "initial_rng_sha256", "denoising_rng_state": "denoising_rng_sha256",
    "rng_state": "final_rng_sha256", "latents": "latents_sha256",
}


def verify_reference_source(request, output_root, *, asset_id, model, provenance, native_execution):
    """Pure artifact/phase admission, called before any native import or load."""
    document = request.document
    worker._validate_reference_request(document)
    source = document["reference_source"]
    path = worker._inside(output_root, source["path"], exists=True)
    if worker.file_sha256(path) != source["sha256"]:
        raise IntegrityError("Matched RAW generation result changed")
    result = worker.read_json(path)
    if (
        result.get("schema") != "reelcache.causvid.rank_result.v1" or result.get("status") != "complete"
        or result.get("operation") != "generate" or result.get("asset_id") != asset_id
        or result.get("model") != model.to_dict() or result.get("generation_from_scratch") is not True
        or result.get("supplied_prefix") is not False or result.get("request_reset_verified") is not True
        or result.get("native_execution") is not native_execution
        or result.get("fixture") is not (not native_execution)
    ):
        raise IntegrityError("Reference input must be a completed from-scratch RAW result with matching evidence kind")
    original = result["request"]
    if (
        original.get("study_phase") != "native_preflight" or original.get("split") != "development"
        or original.get("rank_policy") is not None
        or any(spec.kind != "raw" for spec in worker.codec_specs(original.get("spec_config"), model.layers).values())
    ):
        raise ConfigurationError("Only development/native_preflight RAW generation may supply matched reference state")
    for name in (
        "case_id", "cell_id", "prompt_group_id", "text", "seed", "chunks", "conditioning_identity",
    ):
        if original.get(name) != document[name]:
            raise ConfigurationError(f"Matched RAW generation has a different {name}")
    for name in ("protocol_id", "launch_closure_id"):
        if native_execution:
            worker._sha(provenance.get(name), name)
        if result.get("provenance", {}).get(name) != provenance.get(name):
            raise IntegrityError("Matched RAW generation belongs to another protocol/source closure")
    record = result["latent_file"]
    latent = worker._inside(output_root, record["path"], exists=True)
    if latent.stat().st_size != record["bytes"] or worker.file_sha256(latent) != record["sha256"]:
        raise IntegrityError("Matched RAW generation state failed size/hash verification")
    budget = document["memory_budgets"]
    if 8 * record["bytes"] > budget["host_bytes"] - budget["model_host_bytes"] - budget["clean_host_bytes"]:
        raise CapacityError("Paired input, native output and serialization copies exceed the explicit host grant")
    worker._RequestOutputBudget(
        document, output_root, "native-reference", input_bytes=record["bytes"] + path.stat().st_size,
    )
    return result


def _load_state(result, output_root, max_bytes):
    from reelcache.adapters.causvid import tensor_digest
    from reelcache.serialization import deserialize_tensors

    record = result["latent_file"]
    path = worker._inside(output_root, record["path"], exists=True)
    if path.stat().st_size != record["bytes"] or record["bytes"] > max_bytes:
        raise IntegrityError("Saved generation state changed or exceeds host admission")
    raw = path.read_bytes()
    if worker.hashlib.sha256(raw).hexdigest() != record["sha256"]:
        raise IntegrityError("Saved generation state changed after admission")
    tensors = deserialize_tensors(raw, max_bytes=max_bytes)
    if set(tensors) != set(STATE_HASHES):
        raise IntegrityError("Saved generation must contain all six from-scratch state tensors")
    for name, field in STATE_HASHES.items():
        if tensor_digest(tensors[name]) != result.get(field):
            raise IntegrityError(f"Saved generation {name} differs from its declared tensor hash")
    return tensors


def compare_reference_outputs(
    output_root, *, raw_generation, native_reference, raw_decode, native_decode, max_host_bytes,
):
    """Measure tensor/RNG and separately decoded RGB equality, never JSON equality.

    Each input is a hash-bound ``{path,sha256}`` result reference below output_root.
    This model-free measurement is not a launch receipt or a G3 qualification:
    clean packet hashes still need the driver's ``verify_raw_chunk`` checks.
    """
    import numpy as np
    import torch

    worker._positive(max_host_bytes, "comparison host bytes")
    root = Path(output_root).resolve()
    references = (raw_generation, native_reference, raw_decode, native_decode)
    results = []
    metadata_bytes = 0
    for source in references:
        if not isinstance(source, Mapping) or set(source) != {"path", "sha256"}:
            raise ConfigurationError("Reference comparisons require four hash-bound result artifacts")
        worker._sha(source["sha256"], "comparison source SHA256")
        path = worker._inside(root, source["path"], exists=True)
        metadata_bytes += path.stat().st_size
        if metadata_bytes * 4 > max_host_bytes:
            raise CapacityError("Comparison metadata exceeds its host admission")
        if worker.file_sha256(path) != source["sha256"]:
            raise IntegrityError("A comparison input result changed")
        value = worker.read_json(path)
        if (
            value.get("schema") != "reelcache.causvid.rank_result.v1" or value.get("status") != "complete"
            or value.get("request", {}).get("study_phase") != "native_preflight"
            or value["request"].get("split") != "development"
        ):
            raise ConfigurationError("Reference comparisons require completed development/native_preflight artifacts")
        results.append(value)
    raw, native, rgb_raw, rgb_native = results
    if (
        raw.get("operation") != "generate" or native.get("operation") != "native-reference"
        or native.get("reference_source") != raw_generation
        or native.get("native_pipeline_inference_called") is not True
        or native.get("adapter_instantiated") is not False
        or raw.get("asset_id") != native.get("asset_id") or raw.get("model") != native.get("model")
        or raw.get("fixture") is not native.get("fixture")
        or raw.get("native_execution") is not native.get("native_execution")
    ):
        raise IntegrityError("Native reference and RAW comparison inputs are not a paired execution")
    for result in (raw, native):
        if (
            result.get("generation_from_scratch") is not True or result.get("supplied_prefix") is not False
            or result.get("request_reset_verified") is not True
            or type(result.get("fixture")) is not bool
            or result.get("native_execution") is not (result.get("fixture") is False)
            or result["request"].get("rank_policy") is not None
            or any(spec.kind != "raw" for spec in worker.codec_specs(
                result["request"].get("spec_config"), result["model"]["layers"],
            ).values())
        ):
            raise IntegrityError("Comparison generation must be an isolated, explicitly labelled from-scratch RAW run")
    for name in ("case_id", "cell_id", "prompt_group_id", "text", "seed", "chunks", "conditioning_identity"):
        if raw["request"].get(name) != native["request"].get(name):
            raise IntegrityError("Comparison generation case identities differ")
    for original, decoded, source in ((raw, rgb_raw, raw_generation), (native, rgb_native, native_reference)):
        if (
            decoded.get("operation") != "decode" or decoded.get("latent_source") != source
            or decoded.get("latents_sha256") != original.get("latents_sha256")
            or decoded.get("asset_id") != original.get("asset_id")
            or decoded.get("dit_loaded") is not False or decoded.get("vae_cache_reset_verified") is not True
        ):
            raise IntegrityError("RGB comparison requires each generation's separately decoded full-prefix output")
        for name in ("case_id", "trajectory_id", "arm_id", "seed", "chunks", "text"):
            if decoded["request"].get(name) != original["request"].get(name):
                raise IntegrityError("RGB output belongs to a different generation trajectory")
    input_bytes = 8 * sum(value["latent_file"]["bytes"] for value in (raw, native))
    input_bytes += 3 * sum(value["rgb_file"]["bytes"] for value in (rgb_raw, rgb_native)) + 4 * metadata_bytes
    if input_bytes > max_host_bytes:
        raise CapacityError("Tensor restoration and RGB comparison exceed their explicit host admission")
    states = [_load_state(value, root, max_host_bytes) for value in (raw, native)]
    comparisons = {name + "_bitwise_equal": torch.equal(states[0][name], states[1][name]) for name in STATE_HASHES}
    timeline = worker.rgb_windows(raw["request"]["chunks"], worker.NATIVE_FPS)
    pixels = []
    for value in (rgb_raw, rgb_native):
        record = value["rgb_file"]
        path = worker._inside(root, record["path"], exists=True)
        if path.stat().st_size != record["bytes"] or worker.file_sha256(path) != record["sha256"]:
            raise IntegrityError("A separately decoded RGB artifact changed")
        frames = np.load(path, mmap_mode="r", allow_pickle=False)
        if (
            frames.dtype != np.uint8 or frames.shape != (timeline["prefix_frame_counts"][-1], 480, 832, 3)
            or not frames.flags.c_contiguous or value["rgb"].get("decoded") is not True
            or value["rgb"].get("resize_or_crop") is not False
            or any(value["rgb"].get(name) != expected for name, expected in timeline.items())
            or worker.hashlib.sha256(memoryview(frames).cast("B")).hexdigest() != value.get("rgb_sha256")
        ):
            raise IntegrityError("RGB comparison input is not unchanged native 4F-3/FPS output")
        pixels.append(frames)
    comparisons["rgb_bitwise_equal"] = bool(np.array_equal(*pixels))
    return {
        "schema": "reelcache.causvid.native_comparison.v1",
        "case_id": raw["request"]["case_id"], "native_execution": raw["native_execution"], "fixture": raw["fixture"],
        "inputs": dict(zip(("raw_generation", "native_reference", "raw_decode", "native_decode"), references)),
        "comparisons": comparisons, "all_compared_outputs_equal": all(comparisons.values()),
        "comparison_scope": "actual_noise_conditioning_initial_postdraw_final_RNG_latents_and_separate_native_RGB",
        "compares_result_namespace_or_timing": False, "kv_packets_compared": False, "native_parity_claim": False,
    }


class _LatentExport(BaseException):
    def __init__(self, boundary, latents):
        self.boundary, self.latents = boundary, latents


@contextmanager
def _latent_boundary(pipeline):
    import torch

    if getattr(pipeline, "vae", None) is not None:
        raise StateError("A native-reference epoch must not load or inherit a VAE")

    class Boundary(torch.nn.Module):
        def decode_to_pixel(self, latents):
            raise _LatentExport(self, latents)

    boundary = Boundary()
    modules = getattr(pipeline, "_modules", None)
    registered_before = modules is not None and "vae" in modules
    attribute_before = "vae" in pipeline.__dict__
    previous = pipeline.__dict__.get("vae")
    pipeline.vae = boundary
    try:
        yield boundary
    finally:
        if getattr(pipeline, "vae", None) is not boundary:
            raise StateError("abort_epoch: the instance-local VAE export boundary was replaced")
        delattr(pipeline, "vae")
        if registered_before:
            pipeline.register_module("vae", None)
        elif attribute_before:
            setattr(pipeline, "vae", previous)


class CausVidNativeReferenceWorker(worker.CausVidRankWorker):
    """Reuse request isolation only; the adapter and runtime are never created."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        names = ("inference", "_initialize_kv_cache", "_initialize_crossattn_cache")
        self._entrypoints = {name: getattr(self.pipeline, name, None) for name in names}
        if not all(callable(value) for value in self._entrypoints.values()) or set(
            inspect.signature(self._entrypoints["inference"]).parameters
        ) != {"noise", "text_prompts", "start_latents", "return_latents", "rng"}:
            raise ConfigurationError("The original native inference/cache constructor seams changed")
        self._entry_signature = tuple(worker._generation_value(value) for value in self._entrypoints.values())

    def _check_entrypoints(self):
        self._check_generation()
        if (
            tuple(worker._generation_value(getattr(self.pipeline, name, None)) for name in self._entrypoints)
            != self._entry_signature
            or getattr(self.pipeline, "vae", None) is not None
            or worker._hook_signature(self.pipeline) != self._hooks
            or worker._parameter_signature(self.pipeline) != self._parameters
            or not worker._native_cache_empty(self.pipeline)
        ):
            self._poisoned = True
            raise StateError("abort_epoch: original native entrypoints/caches/hooks/parameters drifted")

    def run(self, request):
        if self._closed or self._poisoned:
            raise StateError("abort_epoch: this resident CausVid worker cannot accept another request")
        if not self._lock.acquire(blocking=False):
            raise StateError("CausVid model epochs accept only isolated serial requests")
        try:
            document = request.document if isinstance(request, worker.CausVidRankRequest) else request
            request = worker.CausVidRankRequest.from_dict(document, protocol=self.protocol, asset_id=self.asset_id)
            document = request.document
            worker._validate_reference_request(document)
            directory = worker._inside(self.output_root, document["output_namespace"])
            if (
                document["request_id"] in self._requests or document["output_namespace"] in self._namespaces
                or directory.exists()
            ):
                raise StateError("Native-reference request IDs and output namespaces must be new")
            if self.revalidate is not None:
                self.revalidate()
            self._check_entrypoints()
            if self.pipeline.denoising_step_list.cpu().tolist() != [1000, 757, 522]:
                raise ConfigurationError("Native reference requires the approved [1000,757,522] denoising schedule")
            if any(row["layer"] >= self.model.layers for row in document["capture"]["retain"]):
                raise ConfigurationError("Retained original selections must exist in the actual native model")
            source = verify_reference_source(
                request, self.output_root, asset_id=self.asset_id, model=self.model, provenance=self.provenance,
                native_execution=self.observation_evidence_kind == "native",
            )
            return self._run_reference(request, directory, source)
        finally:
            self._lock.release()

    def _run_reference(self, request, directory, source):
        import torch
        from examples.native_causvid import DenseNativeAudit
        from reelcache.adapters.causvid import tensor_digest, unique_tensor_bytes
        from reelcache.serialization import serialize_tensors

        document = request.document
        budget = document["memory_budgets"]
        source_path = worker._inside(self.output_root, document["reference_source"]["path"], exists=True)
        disk = worker._RequestOutputBudget(
            document, self.output_root, "native-reference",
            input_bytes=source["latent_file"]["bytes"] + source_path.stat().st_size,
        )
        directory.mkdir(parents=True, exist_ok=False, mode=0o700)
        self._requests.add(document["request_id"])
        self._namespaces.add(document["output_namespace"])
        started = time.perf_counter()
        state = paired = noise = latents = condition = None
        captures, cache_bytes = [], {}
        try:
            worker._json_new(directory / "request.json", document, disk=disk)
            paired = _load_state(source, self.output_root, budget["host_bytes"] - budget["model_host_bytes"])
            parameter = next(self.pipeline.generator.parameters())
            device, dtype = parameter.device, parameter.dtype
            shape = (1, document["chunks"] * self.config.frames_per_chunk, *self.config.latent_tail)
            if (
                tuple(paired["noise"].shape) != shape or paired["noise"].dtype != dtype
                or tuple(paired["latents"].shape) != shape or paired["latents"].dtype != dtype
                or any(not bool(torch.isfinite(paired[name]).all()) for name in ("noise", "latents", "prompt_embeds"))
            ):
                raise IntegrityError("Matched RAW generation state has invalid native geometry/dtype/values")
            global_rng = torch.get_rng_state().clone()
            device_rng = torch.cuda.get_rng_state(device).clone() if device.type == "cuda" else None
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            rng = torch.Generator(device=device).manual_seed(document["seed"])
            initial_rng = rng.get_state().clone()
            noise = torch.empty(shape, device=device, dtype=dtype).normal_(generator=rng)
            denoising_rng = rng.get_state().clone()
            if (
                not torch.equal(initial_rng, paired["initial_rng_state"])
                or not torch.equal(denoising_rng, paired["denoising_rng_state"])
                or not torch.equal(noise.cpu(), paired["noise"])
            ):
                raise IntegrityError("Matched RAW noise/RNG is not the explicit seed's complete upfront draw")
            rng.set_state(paired["denoising_rng_state"])

            def conditioning_observer(embeddings):
                nonlocal condition
                if not torch.equal(embeddings.detach().cpu(), paired["prompt_embeds"]):
                    raise IntegrityError("Original native conditioning differs from matched RAW prompt_embeds")
                condition = embeddings.detach().cpu().clone()
                for label, caches, length in (
                    ("native_kv_bytes", self.pipeline.kv_cache1, 21 * self.config.frame_tokens),
                    ("crossattn_bytes", self.pipeline.crossattn_cache, 512),
                ):
                    expected = (1, length, self.config.heads, self.config.head_dim)
                    if len(caches) != self.model.layers or any(
                        tuple(row[key].shape) != expected or row[key].dtype != dtype or row[key].device != device
                        for row in caches for key in ("k", "v")
                    ):
                        raise IntegrityError("Original native dense cache allocation contract changed")
                    cache_bytes[label] = unique_tensor_bytes(
                        [row[key] for row in caches for key in ("k", "v")],
                    )
                if sum(cache_bytes.values()) > budget["resident_bytes"]:
                    raise CapacityError("Actual original-native dense caches exceed their admitted resident grant")

            selected = {(row["chunk_id"], row["layer"], row["target"]) for row in document["capture"]["retain"]}

            def clean_observer(start, layer, target, matrix):
                context = TensorContext(
                    self.model, MatrixKey(layer, target), start, self.config.frames_per_chunk,
                    self.config.grid_height, self.config.grid_width, "pre_rope" if target == "K" else "none",
                )
                if tuple(matrix.shape) != context.shape or str(matrix.dtype).removeprefix("torch.") != context.dtype:
                    raise IntegrityError("Original native clean source differs from its pristine matrix context")
                row = {
                    "context": context.to_dict(), "original_bytes_sha256": tensor_digest(matrix),
                    "source_hash_format": "sha256-native-contiguous-bytes-no-header-v1",
                    "case_id": document["case_id"], "trajectory_id": document["trajectory_id"],
                    "history_mode": "original_native_dense", "capture": "separate_clean_t0",
                    "fixture": self.observation_evidence_kind != "native",
                    "native_execution": self.observation_evidence_kind == "native",
                    "source_provenance": {**self.provenance, "request_id": document["request_id"]},
                }
                chunk = start // self.config.chunk_tokens
                if (chunk, layer, target) in selected:
                    path = directory / "originals" / f"chunk-{chunk:04d}-layer-{layer:02d}-{target}.reelcache"
                    worker._bytes_new(
                        path, serialize_tensors({"source": matrix.detach().cpu()}), disk=disk, category="archive",
                    )
                    row["original"] = worker._file_record(path, self.output_root)
                captures.append(row)

            with torch.no_grad(), self.attention_scope(), _latent_boundary(self.pipeline) as boundary:
                with DenseNativeAudit(
                    self.pipeline, None, document["chunks"], conditioning_observer=conditioning_observer,
                    clean_observer=clean_observer,
                ) as audit:
                    try:
                        self.pipeline.inference(
                            noise=noise, text_prompts=[document["text"]], start_latents=None,
                            return_latents=True, rng=rng,
                        )
                    except _LatentExport as exported:
                        if exported.boundary is not boundary:
                            raise StateError("A foreign VAE boundary attempted to export native latents")
                        latents = exported.latents.detach().cpu().clone()
                    else:
                        raise IntegrityError("Original native inference did not reach the latents-only VAE boundary")
                    native_audit = audit.finish()
            self._check_generation()
            if (
                condition is None or latents is None or tuple(latents.shape) != shape or latents.dtype != dtype
                or not bool(torch.isfinite(latents).all())
                or len(captures) != document["chunks"] * 2 * self.model.layers
                or tensor_digest(noise) != source["initial_noise_sha256"]
                or not torch.equal(global_rng, torch.get_rng_state())
                or (device_rng is not None and not torch.equal(device_rng, torch.cuda.get_rng_state(device)))
            ):
                raise IntegrityError("Native reference output/capture/noise or private-RNG contract failed")
            memory = {
                **cache_bytes, "admitted_native_kv_bytes": worker.NATIVE_DENSE_KV_BYTES,
                "admitted_crossattn_bytes": worker.NATIVE_CROSSATTN_BYTES,
            }
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                memory["device_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
                memory["device_peak_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
                if memory["device_peak_allocated_bytes"] > budget["device_bytes"]:
                    raise CapacityError("Actual native-reference allocation peak exceeded its explicit device grant")
            state = {
                "latents": latents, "noise": noise.cpu(), "prompt_embeds": condition,
                "initial_rng_state": initial_rng, "denoising_rng_state": denoising_rng, "rng_state": rng.get_state(),
            }
            if 4 * unique_tensor_bytes([*state.values(), *paired.values()]) > budget["host_bytes"] - budget["model_host_bytes"]:
                raise CapacityError("Paired native reference state exceeds host serialization admission")
            path = directory / "latents.reelcache"
            worker._bytes_new(path, serialize_tensors(state), disk=disk)
            result = {
                "schema": "reelcache.causvid.rank_result.v1", "status": "complete", "operation": "native-reference",
                "request": document, "model": self.model.to_dict(), "asset_id": self.asset_id,
                **worker.rank_control_metadata(document),
                "provenance": self.provenance, "reference_source": document["reference_source"],
                "reference_latent_file": source["latent_file"], "latent_file": worker._file_record(path, self.output_root),
                **{field: tensor_digest(state[name]) for name, field in STATE_HASHES.items()},
                "generation_from_scratch": True, "supplied_prefix": False,
                "native_pipeline_inference_called": True, "adapter_instantiated": False,
                "vae_boundary": "instance_local_latents_only_export", "vae_loaded": False, "metrics_loaded": False,
                "native_audit": native_audit, "clean_captures": captures, "memory": memory,
                "execution_history_mode": "original_native_dense", "disk_admission": disk.plan,
                "raw_comparison": {
                    "latents_bitwise_equal": torch.equal(state["latents"], paired["latents"]),
                    "final_rng_bitwise_equal": torch.equal(state["rng_state"], paired["rng_state"]),
                    "noise_and_postdraw_rng_matched": True, "native_conditioning_matched": True,
                    "rgb_compared": False, "kv_packets_compared": False,
                },
                "rgb": {**worker.rgb_windows(document["chunks"], self.fps), "decoded": False},
                "native_parity_claim": False, "native_execution": self.observation_evidence_kind == "native",
                "fixture": self.observation_evidence_kind != "native",
                "elapsed_seconds": time.perf_counter() - started,
            }
            worker._reset_native_cache(self.pipeline)
            self._check_entrypoints()
            if self.revalidate is not None:
                self.revalidate()
            result["request_reset_verified"] = True
            worker._json_new(directory / "result.json", result, disk=disk)
            return result
        except BaseException as exc:
            self._poisoned = True
            worker._json_new(directory / "failure.json", {
                "schema": "reelcache.causvid.epoch_abort.v1", "status": "abort_epoch",
                "request_id": document["request_id"], "operation": "native-reference", "error": repr(exc),
                "elapsed_seconds": time.perf_counter() - started, "clean_captures": captures,
                "native_parity_claim": False,
            }, disk=disk, failure=True)
            raise
        finally:
            try:
                worker._reset_native_cache(self.pipeline)
                self._check_entrypoints()
            except BaseException:
                self._poisoned = True
                raise
            finally:
                state = paired = noise = latents = condition = None
                gc.collect()
