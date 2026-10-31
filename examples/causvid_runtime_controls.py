"""Bounded native CausVid controls for opt-in static publication and hot history."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import datetime as dt
import gc
import json
import os
import resource
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from examples import native_causvid as native


def verify_cpu_receipt(path):
    record = json.loads(Path(path).read_text())
    if (
        record.get("status") != "passed" or not record.get("source_unchanged_during_tests")
        or any(record.get(key) != 0 for key in ("failures", "errors", "skipped"))
        or not record.get("discovery", {}).get("full_suite")
        or not record.get("require_no_skips")
    ):
        raise ValueError("native controls require a complete no-skip CPU receipt")
    expected = {
        str(path.relative_to(ROOT))
        for directory in ("reelcache", "tests", "examples", "scripts")
        for path in (ROOT / directory).rglob("*.py")
    } | {"pyproject.toml"}
    if set(record.get("source_files", {})) != expected:
        raise ValueError("CPU receipt does not cover the complete current source inventory")
    for relative, digest in record["source_files"].items():
        if native.sha256(ROOT / relative) != digest:
            raise ValueError(f"source differs from CPU-qualified code: {relative}")
    return record["source_tree_sha256"]


def admit_device(uuid, *, minimum_free_gib):
    rows = subprocess.check_output([
        "nvidia-smi", "--query-gpu=uuid,name,memory.free", "--format=csv,noheader,nounits",
    ], text=True).splitlines()
    matching = [row.split(",") for row in rows if row.split(",")[0].strip() == uuid]
    if len(matching) != 1:
        raise ValueError("explicit GPU UUID is not visible on this host")
    row = [value.strip() for value in matching[0]]
    if "RTX PRO 6000 Blackwell" not in row[1] or int(row[2]) < minimum_free_gib * 1024:
        raise ValueError("expected an admitted RTX PRO 6000 Blackwell with sufficient free HBM")
    active = subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits",
    ], text=True).splitlines()
    if any(line.split(",")[0].strip() == uuid for line in active):
        raise ValueError("the granted GPU already has compute processes")
    memfree = int(next(line.split()[1] for line in Path("/proc/meminfo").read_text().splitlines()
                       if line.startswith("MemFree:"))) * 1024
    if memfree < 32 * 1024**3:
        raise ValueError("native pilot requires at least 32 GiB physical free host RAM")
    return {"uuid": uuid, "name": row[1], "free_mib": int(row[2]), "host_memfree_bytes": memfree}


def run(args, directory, provenance):
    import numpy as np
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from reelcache.adapters.causvid import CausVidAdapter, CausVidConfig, tensor_digest
    from reelcache.codecs import HEVCProvider
    from reelcache.execution import BoundedExecutor
    from reelcache.manager import KVRuntime
    from reelcache.nvdec import NVDECDecoder
    from reelcache.gpu_fitting import GPUFitter
    from reelcache.serialization import serialize_tensors
    from reelcache.telemetry import RuntimeTelemetry
    from reelcache.types import CodecSpec, MatrixKey, ModelIdentity

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if torch.cuda.device_count() != 1:
        raise RuntimeError("native controls require exactly one granted visible GPU")
    admitted_uuid = str(torch.cuda.get_device_properties(0).uuid)
    if admitted_uuid.removeprefix("GPU-") != args.gpu_uuid.removeprefix("GPU-"):
        raise RuntimeError("Torch device identity does not match the granted UUID")
    pipeline = native.build_pipeline(args)
    config = CausVidConfig()
    inputs = torch.load(provenance["prefix_inputs"]["path"], map_location="cpu", weights_only=True)
    rng = torch.Generator(device="cuda").manual_seed(0)
    noise = torch.randn((1, args.chunks * 3, 16, 60, 104), device="cuda",
                        dtype=torch.bfloat16, generator=rng).cpu()
    initial_rng = rng.get_state().clone()
    model = ModelIdentity("causvid", native.identity({
        "weights": provenance["weights"], "config": provenance["source_files"]["config"]["sha256"],
        "native": {key: row["sha256"] for key, row in provenance["source_files"].items()
                   if key.startswith("native/")},
    }), 30, 1536, "tokens_channels", "bfloat16")
    report = {
        "schema": "reelcache.causvid_runtime_controls.v1", "status": "running",
        "scope": "one reused development prompt and seed, not profile or held-out quality qualification",
        "gpu_uuid": args.gpu_uuid, "cpu_receipt_sha256": native.sha256(args.cpu_receipt),
        "source_tree_sha256": verify_cpu_receipt(args.cpu_receipt),
        "initial_noise_sha256": tensor_digest(noise), "initial_rng_sha256": tensor_digest(initial_rng),
        "chunks": args.chunks, "arms": [], "native_denoising_overlap_claim": False,
        "static_ranks": {"K": args.key_rank, "V": args.value_rank},
        "cuda": torch.version.cuda, "torch": torch.__version__,
        "source_provenance": provenance,
    }
    start = time.monotonic()
    conditioning = {}
    captured_sources = []

    def progress(label):
        native.write_json(directory / "progress.json", {
            "phase": label, "elapsed_seconds": time.monotonic() - start,
            "utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        }, mutable=True)
        print(label, flush=True)

    with native.instance_cross_attention_sdpa(pipeline), sdpa_kernel(
        [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION],
    ):
        progress("native dense RAW reference")
        pipeline.kv_cache1 = [
            {name: torch.zeros((1, args.chunks * config.chunk_tokens, 12, 128),
                               device="cuda", dtype=torch.bfloat16) for name in ("k", "v")}
            for _ in range(30)
        ]
        pipeline._initialize_crossattn_cache(batch_size=1, dtype=torch.bfloat16, device="cuda")
        def capture(_module, _inputs, output):
            conditioning["prompt_embeds"] = output["prompt_embeds"].detach().cpu().clone()
        hook = pipeline.text_encoder.register_forward_hook(capture)
        dense_rng = torch.Generator(device="cuda").set_state(initial_rng)
        try:
            def capture_original(token_start, layer, target, tensor):
                if not args.capture_sources:
                    return
                if not args.capture_all_sources and (token_start != 0 or layer not in (0, 15, 29)):
                    return
                from reelcache.types import TensorContext
                key = MatrixKey(layer, target)
                context = TensorContext(model, key, token_start, 3, 30, 52, "pre_rope" if target == "K" else "none")
                capture_dir = args.capture_dir or directory
                name = (f"original-{layer:02d}-{target}.reelcache" if not args.capture_all_sources
                        else f"original-t{token_start:06d}-{layer:02d}-{target}.reelcache")
                path = capture_dir / name
                data = serialize_tensors({"source": tensor.cpu()})
                with path.open("xb") as output:
                    output.write(data)
                captured_sources.append({
                    "context": context.to_dict(), "path": str(path),
                    "sha256": native.sha256(path), "tensor_sha256": tensor_digest(tensor),
                    "capture": "native normalized pre-RoPE K or direct V, supplied clean prefix",
                })
            with native.DenseNativeAudit(
                pipeline, inputs["latents"], args.chunks,
                clean_observer=capture_original if args.capture_sources else None,
            ) as audit:
                rgb, dense_latents = pipeline.inference(
                    noise=noise.to("cuda"), text_prompts=[provenance["prompt"]],
                    start_latents=inputs["latents"].to("cuda"), return_latents=True,
                    rng=dense_rng,
                )
            dense_audit = audit.finish()
        finally:
            hook.remove()
        dense_latents, dense_rgb = dense_latents.cpu(), native.video_rgb(rgb)
        del rgb
        pipeline.kv_cache1 = pipeline.crossattn_cache = None
        gc.collect()
        conditioning_id = native.identity({"prompt": provenance["prompt"], "tensor": tensor_digest(conditioning["prompt_embeds"])})
        report["dense_reference"] = {"audit": dense_audit, "latent_sha256": tensor_digest(dense_latents),
                                     "rgb_sha256": native.hashlib.sha256(dense_rgb.tobytes()).hexdigest(),
                                     "rng_sha256": tensor_digest(dense_rng.get_state())}
        if captured_sources:
            native.write_json(directory / "original-sources.json", {
                "sources": captured_sources, "source_tree_sha256": report["source_tree_sha256"],
                "provenance": provenance,
            })
        for kind in args.arms:
            reference = None
            specs = {MatrixKey(layer, target): CodecSpec() if kind == "raw" else CodecSpec(
                kind, rank=args.key_rank if target == "K" else args.value_rank,
                inter_prediction=args.hevc_inter_prediction and kind == "svd_hevc",
            ) for layer in range(30) for target in ("K", "V")}
            for optimized in (False, True):
                label = kind + ("-deferred-parallel-hot" if optimized else "-sync-reload")
                progress(label)
                state_root = args.store_root / label
                record = {"label": label, "codec": kind, "optimized": optimized, "steps": []}
                telemetry = RuntimeTelemetry()
                nvdec = NVDECDecoder(
                    gpu_id=0, gpu_uuid=args.gpu_uuid, device_output=args.nvdec_output == "device",
                ) if args.nvdec and (optimized or args.nvdec_both_arms) and kind == "svd_hevc" else None
                fitter = None if args.fitting_backend == "cpu" else GPUFitter(
                    gpu_id=0, gpu_uuid=args.gpu_uuid, method=args.fitting_backend,
                )
                provider = HEVCProvider(str(args.ffmpeg), nvdec=nvdec, fitter=fitter,
                                        verify_decoded_coefficients=args.verify_decoded_coefficients,
                                        verify_restored_packets=args.verify_restored_packets,
                                        validate_hevc_encode=args.validate_hevc_encode,
                                        x265_threads=args.x265_threads)
                quality = optimized and args.history_mode == "quality_hot"
                deferred = optimized and fitter is None and not args.parallel_kv_mib and not quality
                record["decode_backend"] = "nvdec" if nvdec is not None else "software"
                record["fitting_backend"] = args.fitting_backend
                record["deferred_publication"] = deferred
                record["verify_decoded_coefficients"] = args.verify_decoded_coefficients
                record["verify_restored_packets"] = args.verify_restored_packets
                record["verify_reads"] = args.verify_reads
                record["validate_hevc_encode"] = args.validate_hevc_encode
                record["x265_threads"] = args.x265_threads
                record["hevc_inter_prediction"] = args.hevc_inter_prediction if kind == "svd_hevc" else None
                record["codec_specs"] = {"K": specs[MatrixKey(0, "K")].to_dict(), "V": specs[MatrixKey(0, "V")].to_dict()}
                record["history_mode"] = "quality_hot" if quality else "reload"
                record["hot_history_budget_bytes"] = args.hot_mib * 1024**2 if optimized else 0
                paired_bytes = (args.parallel_kv_mib * 1024**2 if (optimized and not quality) or args.parallel_kv_both_arms
                                else 0)
                record["parallel_kv_budget_bytes"] = paired_bytes
                forward_events = []

                def forward_pre(_module, _args, kwargs):
                    event = torch.cuda.Event(enable_timing=True)
                    event.record()
                    forward_events.append([int(kwargs["current_start"]), event, None])

                def forward_post(_module, _args, _kwargs, _output):
                    event = torch.cuda.Event(enable_timing=True)
                    event.record()
                    forward_events[-1][2] = event

                forward_hooks = [
                    pipeline.generator.register_forward_pre_hook(forward_pre, with_kwargs=True),
                    pipeline.generator.register_forward_hook(forward_post, with_kwargs=True),
                ]
                with nvdec if nvdec is not None else nullcontext(), \
                        BoundedExecutor(max_workers=args.encoder_workers, max_pending=args.encoder_workers,
                                     max_staging_bytes=2 * 1024**3) as encoders, \
                        BoundedExecutor(max_workers=1, max_pending=1, max_staging_bytes=2 * 1024**3) as commits, \
                        KVRuntime(
                            state_root, model, max_resident_bytes=1024**3, max_pending_source_bytes=2 * 1024**3,
                            encoder=provider.encode_tensor, decoder=provider.decode_tensor,
                            executor=encoders if deferred else None,
                            encode_concurrency=args.encoder_workers if deferred else 1,
                            encode_workspace_bytes=256 * 1024**2 if deferred else 0,
                            verify_reads=args.verify_reads,
                            prefit=provider.prefit if fitter is not None else None,
                        ) as runtime:
                    with CausVidAdapter(
                        pipeline, runtime, "parent", config=config, specs=specs,
                        max_current_bytes=2 * 1024**3, max_clean_host_bytes=2 * 1024**3,
                        max_attention_bytes=1024**3, max_hot_history_bytes=args.hot_mib * 1024**2 if optimized else 0,
                        commit_executor=commits if deferred else None,
                        prefetch_history_bytes=args.prefetch_mib * 1024**2 if optimized and not quality else 0,
                        parallel_kv_bytes=paired_bytes,
                        history_mode="quality_hot" if quality else "reload",
                        max_quality_history_bytes=args.quality_mib * 1024**2 if quality else 0,
                    ) as adapter:
                        adapter.start(noise, conditioning, torch.Generator(device="cuda").set_state(initial_rng),
                                      conditioning_id=conditioning_id)
                        prefix_snapshot = None
                        try:
                            with telemetry.capture():
                                for index in range(args.chunks):
                                    torch.cuda.synchronize()
                                    torch.cuda.reset_peak_memory_stats()
                                    tick = time.monotonic()
                                    handle = adapter.step(
                                        clean_prefix=inputs["latents"] if index == 0 else None,
                                        defer_publication=deferred,
                                    )
                                    submitted = time.monotonic()
                                    if deferred:
                                        sealed = adapter.finish_step()
                                    else:
                                        sealed = handle
                                    torch.cuda.synchronize()
                                    if kind == "raw":
                                        native.verify_raw_chunk(runtime, sealed, dense_audit["clean_segment_hashes"][index * config.chunk_tokens])
                                    forwards = [
                                        {"token_start": start, "cuda_ms": begin.elapsed_time(end)}
                                        for start, begin, end in forward_events
                                    ]
                                    forward_events.clear()
                                    record["steps"].append({
                                        "chunk": index, "wall_seconds": time.monotonic() - tick,
                                        "generator_forwards": forwards,
                                        "step_return_seconds": submitted - tick,
                                        "publication_wait_seconds": time.monotonic() - submitted if deferred else 0,
                                        "memory": native.memory_sample(label + f"/{index}"),
                                        "owned_memory": adapter.memory_report(),
                                        "source_copies": runtime.async_stats(),
                                    })
                                    if index == 0:
                                        prefix_snapshot = adapter.snapshot()
                        except BaseException:
                            if adapter._pending_step is not None:
                                adapter._pending_step["handle"].cancel()
                                try:
                                    adapter.finish_step()
                                except BaseException as cleanup_error:
                                    print(f"Pending-step cleanup outcome: {cleanup_error!r}", file=sys.stderr, flush=True)
                            raise
                        latents, final_rng = adapter.latent_state, adapter.rng_state
                        if not torch.equal(dense_rng.get_state(), final_rng):
                            raise RuntimeError("runtime continuation changed native generation RNG")
                        video = adapter.decode()
                        frames = native.video_rgb(video)
                        del video
                        record["artifacts"] = native.save_output(directory / label, latents, frames, final_rng)
                        record["decoder_bytes"] = native.branch_bytes(runtime, "parent")
                        record["versus_dense"] = native.compare(dense_latents, dense_rgb, latents, frames, exact=kind == "raw")
                        if not optimized:
                            reference = (latents, frames, final_rng)
                        else:
                            if not torch.equal(reference[2], final_rng):
                                raise RuntimeError("deferred execution changed explicit RNG")
                            record["versus_same_profile"] = native.compare(
                                reference[0], reference[1], latents, frames, exact=True,
                            )
                            if kind == "raw":
                                adapter.restore(prefix_snapshot, "resumed", conditioning_id=conditioning_id)
                                while not adapter.finished:
                                    adapter.step(defer_publication=deferred)
                                    if deferred:
                                        adapter.finish_step()
                                replay = native.video_rgb(adapter.decode())
                                if (not torch.equal(adapter.latent_state, latents) or not np.array_equal(replay, frames)
                                        or not torch.equal(adapter.rng_state, final_rng)):
                                    raise RuntimeError("deferred native RAW resume parity failed")
                                record["resumed_latent_rgb_rng_exact"] = True
                        record["telemetry"] = telemetry.snapshot()
                        record["executor"] = encoders.stats()
                        record["residency"] = runtime.residency.stats()
                        record["nvdec"] = None if nvdec is None else nvdec.stats()
                        free, total = torch.cuda.mem_get_info()
                        record["post_arm_device_used_bytes"] = total - free
                        record["device_used_includes_driver_and_other_processes"] = True
                        record["process_maxrss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
                        record["process_maxrss_scope"] = "cumulative parent process HWM; excludes child FFmpeg RSS"
                    if runtime.active_leases or runtime.async_stats()["pending_commits"]:
                        raise RuntimeError("runtime allocations survived native adapter cleanup")
                for hook in forward_hooks:
                    hook.remove()
                record["prefit_stats"] = dict(fitter.prefit_stats) if fitter is not None else None
                verify_cpu_receipt(args.cpu_receipt)
                native.verify_sources(provenance)
                report["arms"].append(record)
                native.write_json(directory / (label + ".json"), record)
        report.update(status="passed", elapsed_seconds=time.monotonic() - start,
                      source_unchanged=True, completed_utc=dt.datetime.now(dt.timezone.utc).isoformat())
        native.write_json(directory / "result.json", report)


def parser():
    """The native CausVid CLI (including ``--hevc-inter-prediction``) plus the runtime controls."""
    parser = native.parser()
    parser.description = __doc__
    parser.add_argument("--cpu-receipt", type=Path, required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--store-root", type=Path, required=True)
    parser.add_argument("--minimum-free-gib", type=int, default=48)
    parser.add_argument("--max-seconds", type=int, default=14400)
    parser.add_argument("--encoder-workers", type=int, choices=(2, 4), default=2)
    parser.add_argument("--hot-mib", type=int, default=1024)
    parser.add_argument("--key-rank", type=int, default=320)
    parser.add_argument("--value-rank", type=int, default=576)
    parser.add_argument("--nvdec", action="store_true",
                        help="Use explicitly UUID-bound NVDEC for optimized HEVC restoration only")
    parser.add_argument("--nvdec-output", choices=("host", "device"), default="device")
    parser.add_argument("--verify-decoded-coefficients", action="store_true",
                        help="Opt into CPU SHA readback on the device-output path")
    parser.add_argument("--prefetch-mib", type=int, default=0)
    parser.add_argument("--fitting-backend", choices=("cpu", "gesvd", "gesvdj", "gesvda", "torch_lowrank",
                                                     "cupy", "randomized", "gram", "gram_batched"),
                        default="cpu")
    parser.add_argument("--capture-sources", action="store_true")
    parser.add_argument("--capture-all-sources", action="store_true",
                        help="With --capture-sources, capture every layer of every clean chunk")
    parser.add_argument("--capture-dir", type=Path, default=None,
                        help="Existing directory for captured sources (default: the run directory)")
    parser.add_argument("--parallel-kv-mib", type=int, default=0)
    parser.add_argument("--parallel-kv-both-arms", action="store_true",
                        help="Use paired K/V in both arms to isolate next-layer reconstruction lookahead")
    parser.add_argument("--nvdec-both-arms", action="store_true",
                        help="Use the same NVDEC backend in the reference and optimized arm")
    parser.add_argument("--verify-restored-packets", action="store_true",
                        help="Opt into restore-time payload re-hashing, basis conditioning and finite checks")
    parser.add_argument("--x265-threads", type=int, default=4,
                        help="x265 WPP pool (+frame threads at >=4) per encode; pixels identical to 1")
    parser.add_argument("--validate-hevc-encode", action="store_true",
                        help="Opt into the encode-time software HEVC validation decode (default off)")
    # --hevc-inter-prediction (BooleanOptionalAction, default False) is inherited from native.parser().
    parser.add_argument("--verify-reads", action="store_true",
                        help="Opt into per-read store SHA/length checks (default off; store.verify() still audits)")
    parser.add_argument("--history-mode", choices=("reload", "quality_hot"), default="reload",
                        help="quality_hot decodes each committed chunk once into GPU history (optimized arm)")
    parser.add_argument("--quality-mib", type=int, default=0)
    return parser


def main(argv=None):
    args = parser().parse_args(argv)
    if args.history_mode == "quality_hot" and (args.quality_mib <= 0 or args.hot_mib):
        raise ValueError("quality_hot needs --quality-mib > 0 and --hot-mib 0")
    if args.command == "run" and not args.allow_cuda:
        raise ValueError("native execution requires --allow-cuda and the explicit granted UUID")
    if args.arms[0] != "raw" or len(set(args.arms)) != len(args.arms):
        raise ValueError("RAW must lead the unique codec controls")
    if not 0 < args.max_seconds <= 14400 or args.hot_mib < 0 or args.minimum_free_gib < 48:
        raise ValueError("invalid bounded resource settings")
    if args.prefetch_mib < 0 or (args.prefetch_mib and args.hot_mib):
        raise ValueError("prefetch requires a nonnegative budget and disabled hot history")
    if args.parallel_kv_mib < 0 or (args.parallel_kv_mib and args.hot_mib):
        raise ValueError("paired K/V needs a nonnegative budget and disabled hot history")
    if args.parallel_kv_both_arms and not args.parallel_kv_mib:
        raise ValueError("paired controls require a positive K/V output budget")
    verify_cpu_receipt(args.cpu_receipt)
    if not native.re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,100}", args.run_id):
        raise ValueError("run-id must be a new simple identifier")
    if args.output_root is None or args.ffmpeg is None:
        raise ValueError("supply explicit --output-root and --ffmpeg")
    directory = args.output_root / args.run_id
    directory.mkdir(parents=True, exist_ok=False)
    native.environment(directory, gpu=None, threads=args.threads)
    try:
        provenance = native.preflight(args)
        provenance["source_files"]["runtime_controls_driver"] = {
            "path": str(Path(__file__).resolve()), "sha256": native.sha256(__file__),
        }
        native.write_json(directory / "preflight.json", provenance)
        if args.command == "preflight":
            return
        admission = admit_device(args.gpu_uuid, minimum_free_gib=args.minimum_free_gib)
        args.store_root.mkdir(parents=True, exist_ok=False)
        disk = os.statvfs(args.store_root)
        if disk.f_bavail * disk.f_frsize < 30 * 1024**3:
            raise ValueError("native control store requires at least 30 GiB free disk")
        native.write_json(directory / "execution.json", {
            "host": os.uname().nodename, "pid": os.getpid(), "argv": sys.argv, "admission": admission,
            "cpu_receipt_sha256": native.sha256(args.cpu_receipt), "max_seconds": args.max_seconds,
            "cpu_affinity": sorted(os.sched_getaffinity(0)),
        })
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_uuid
        def deadline(_signum, _frame):
            raise TimeoutError("bounded native runtime-control deadline exceeded")
        signal.signal(signal.SIGALRM, deadline)
        signal.alarm(args.max_seconds)
        try:
            run(args, directory, provenance)
        finally:
            signal.alarm(0)
    except BaseException as error:
        native.write_json(directory / "failure.json", {
            "error": repr(error), "traceback": native.traceback.format_exc(),
            "utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        })
        raise


if __name__ == "__main__":
    main()
