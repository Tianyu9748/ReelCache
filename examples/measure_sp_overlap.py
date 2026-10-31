"""Measure whether SP gather/scatter overlaps computation on two GPUs (NCCL/NVLink + NVDEC).

Each layer runs emulated native work on the forward stream: the emulated
Ulysses all-gather (its own NCCL group) plus a calibrated bf16 GEMM burn.
There are no per-layer host synchronizations, so the host runs ahead as in
the real model. For ``overlap`` off/on it records:

* the clean forward of chunk 0 (capture gathers only, no history),
* the noisy forward of the last chunk (cold history loads per layer),
* the same forward with only compute (no adapter), as the lower bound,
* a torch.profiler trace of rank 0: how much of our NCCL SendRecv kernel time
  (gather + scatter) runs concurrently with forward-stream kernels.

``hidden = (T_sync - T_overlap) / (T_sync - T_compute)`` is the fraction of the
serialized communication/decode cost that the overlap removed.
"""

import argparse
from datetime import timedelta
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _overlap_from_trace(path):
    """Fraction of NCCL SendRecv kernel time concurrent with forward-stream kernels."""
    events = json.loads(Path(path).read_text())["traceEvents"]
    kernels = [e for e in events if e.get("cat") == "kernel" and "dur" in e]
    copies = [e for e in events if e.get("cat") == "gpu_memcpy" and "dur" in e]
    if not kernels:
        return None
    by_stream = {}
    for e in kernels:
        if "nccl" not in e["name"].lower():
            by_stream[e["args"].get("stream")] = by_stream.get(e["args"].get("stream"), 0) + e["dur"]
    compute_stream = max(by_stream, key=by_stream.get)
    compute = sorted((e["ts"], e["ts"] + e["dur"]) for e in kernels
                     if e["args"].get("stream") == compute_stream and "nccl" not in e["name"].lower())
    merged = []
    for start, stop in compute:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], stop)
        else:
            merged.append([start, stop])

    def overlap(start, stop):
        total = 0.0
        for a, b in merged:
            if b <= start:
                continue
            if a >= stop:
                break
            total += min(b, stop) - max(a, start)
        return total

    def summarize(items):
        busy = sum(e["dur"] for e in items)
        hidden = sum(overlap(e["ts"], e["ts"] + e["dur"]) for e in items)
        return {"count": len(items), "busy_ms": busy / 1e3, "concurrent_with_compute_ms": hidden / 1e3,
                "concurrent_fraction": hidden / busy if busy else None}

    sendrecv = [e for e in kernels if "sendrecv" in e["name"].lower()]
    side_copies = [e for e in copies if e["args"].get("stream") != compute_stream]
    return {"compute_stream": compute_stream, "compute_busy_ms": sum(b - a for a, b in merged) / 1e3,
            "nccl_sendrecv": summarize(sendrecv), "side_stream_copies": summarize(side_copies)}


def worker(rank, cfg):
    from types import SimpleNamespace

    import torch
    import torch.distributed as dist

    from reelcache.adapters.magi import MagiAdapter, MagiGeometry
    from reelcache.codecs import HEVCProvider
    from reelcache.distributed import DistributedOwner
    from reelcache.execution import BoundedExecutor
    from reelcache.gpu_fitting import GPUFitter
    from reelcache.manager import KVRuntime
    from reelcache.nvdec import NVDECDecoder
    from reelcache.sp_kv import SPTieredHistory
    from reelcache.types import CodecSpec, MatrixKey, ModelIdentity
    from tests.test_magi_adapter import FakeModel, fake_rotary

    root = Path(cfg["root"])
    torch.cuda.set_device(rank)
    torch.set_num_threads(cfg["threads"])
    device = torch.device("cuda", rank)
    out = {"rank": rank, "runs": {}}
    try:
        dist.init_process_group("gloo", rank=rank, world_size=2, timeout=timedelta(seconds=900),
                                init_method=f"file://{root / 'rendezvous'}")
        data_group = dist.new_group(backend="nccl")
        native_group = dist.new_group(backend="nccl")
        L, chunks = cfg["layers"], cfg["chunks"]
        geometry = MagiGeometry(layers=L, kv_heads=8, head_dim=128, frames=cfg["frames"],
                                height=cfg["height"], width=cfg["width"], world_size=2)
        tokens = geometry.tokens
        identity = ModelIdentity("sp_overlap", "synthetic", L, geometry.channels, "global_KV_channels", "bfloat16")
        specs = {MatrixKey(i, t): CodecSpec("svd_hevc", rank=cfg["k_rank"] if t == "K" else cfg["v_rank"])
                 for i in range(L) for t in ("K", "V")}
        generator = torch.Generator().manual_seed(3)
        basis = torch.randn(160, geometry.channels, generator=generator)
        mixed = [((torch.randn(tokens, 160, generator=generator) @ basis
                   + 0.05 * torch.randn(tokens, geometry.channels, generator=generator)) / 6)
                 .reshape(tokens, 1, -1).to(device) for _ in range(chunks + 1)]
        angle = torch.arange(tokens, dtype=torch.float32)[:, None].expand(tokens, 64) * 1e-3
        cosine, sine = angle.cos().to(device), angle.sin().to(device)
        a = torch.randn(4096, 4096, device=device, dtype=torch.bfloat16)
        b = torch.randn(4096, 4096, device=device, dtype=torch.bfloat16)

        def burn(iterations):
            c = a
            for _ in range(iterations):
                c = torch.matmul(c, b) * 1e-3
            return c

        burn(3)
        torch.cuda.synchronize()
        started = time.perf_counter()
        burn(50)
        torch.cuda.synchronize()
        per = (time.perf_counter() - started) / 50
        iterations = max(1, round(cfg["compute_ms"] / 1e3 / per))
        dist.barrier()
        uuids = cfg["uuids"]
        fitter = GPUFitter(gpu_id=0, gpu_uuid=uuids[0], method="gram", shared=True) if rank == 0 else None

        def run(overlap, scenario, prefix=""):
            label = f"{prefix}{scenario}-{'overlap' if overlap else 'sync'}"
            pool = BoundedExecutor(max_workers=4, max_pending=4, max_staging_bytes=8 << 30) if rank == 0 else None
            runtime = None
            if rank == 0:
                encoder = HEVCProvider(cfg["ffmpeg"], fitter=fitter, x265_threads=4)
                runtime = KVRuntime(root / f"store-{label}", identity, max_resident_bytes=8 << 30,
                                    encoder=encoder.encode_tensor, encode_workspace_bytes=64 << 20)
            owner = DistributedOwner(expected_world_size=2, max_wire_bytes=64 << 20)
            per_chunk = 2 * L * tokens * geometry.layout.local_channels * 2
            budget = 0 if scenario == "all-cold" else cfg["hot_chunks"] * per_chunk
            factory = lambda: HEVCProvider(cfg["ffmpeg"], nvdec=NVDECDecoder(  # noqa: E731
                gpu_id=rank, gpu_uuid=uuids[rank], device_output=True))
            if overlap:
                sp = SPTieredHistory(coordinator=owner, data_group=data_group, layout=geometry.layout,
                                     device=device, native_dtype="bfloat16", decoder_factory=factory,
                                     rotary_apply=fake_rotary, hot_budget_bytes=budget, promote_decoded=False,
                                     overlap=True)
            else:
                sp = SPTieredHistory(coordinator=owner, data_group=data_group, layout=geometry.layout,
                                     device=device, native_dtype="bfloat16", decoder=factory(),
                                     rotary_apply=fake_rotary, hot_budget_bytes=budget, promote_decoded=False,
                                     overlap=False)
            model = FakeModel(geometry)
            cache = SimpleNamespace(max_batch_size=1, max_sequence_length=(chunks + 2) * tokens,
                                    key_value_memory_dict={}, update_kv_cache=True)
            adapter = MagiAdapter(model, cache, runtime, model_identity=identity, branch_id="main",
                                  staging_dir=root / f"staging-{label}-{rank}", geometry=geometry,
                                  coordinator=owner, specs=specs, rotary_apply=fake_rotary, device=device,
                                  max_layer_buffer_bytes=8 << 30, cache_mode="sp_tiered", sp_history=sp,
                                  stream_executor=pool, stream_queue_depth=cfg["queue_depth"])
            adapter.attach()

            def layers(chunk, clean, guarded=True):
                meta = SimpleNamespace(
                    slice_point=chunk, clip_token_nums=tokens, start_chunk_id=chunk, end_chunk_id=chunk + 1,
                    denoising_range_num=1, extract_prefix_video_feature=clean, distill_nearly_clean_chunk=False,
                    timestep=torch.full((1, 1), 0.9999 if clean else 0.0))
                rows = geometry.layout.rows(rank)
                for index, layer in enumerate(model.videodit_blocks.layers):
                    attention = layer.self_attention
                    source = mixed[chunk] + 0.01 * index
                    k = attention.get_k(source[rows], cosine[rows], sine[rows]) if guarded else None
                    v = attention.get_v(source[rows]) if guarded else None
                    if guarded:
                        packed = torch.cat((k, v), -1)
                    else:
                        packed = torch.empty(rows.stop - rows.start, 8, 256, device=device, dtype=torch.bfloat16)
                    pieces = [torch.empty_like(packed) for _ in range(2)]
                    dist.all_gather(pieces, packed, group=native_group)
                    heads = torch.cat(pieces)[:, rank * 4:(rank + 1) * 4].contiguous()
                    if guarded:
                        attention.adjust_key_and_value_for_inference(heads, cache, meta)
                    burn(iterations)

            def timed(chunk, clean, *, profile=None, guarded=True):
                dist.barrier()
                torch.cuda.synchronize()
                started = time.perf_counter()
                if profile is not None:
                    from torch.profiler import ProfilerActivity, profile as torch_profile
                    with torch_profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                        run_forward(chunk, clean, guarded)
                        torch.cuda.synchronize()
                    prof.export_chrome_trace(str(profile))
                else:
                    run_forward(chunk, clean, guarded)
                    torch.cuda.synchronize()
                return time.perf_counter() - started

            def run_forward(chunk, clean, guarded):
                if not guarded:
                    layers(chunk, clean, guarded=False)
                    return
                with adapter.execution(chunk_index=chunk, clean=clean, clean_timestep=0.9999 if clean else None):
                    layers(chunk, clean)

            row = {"iterations": iterations, "hot_budget_bytes": budget}
            trace_dir = root / "traces"
            trace_dir.mkdir(exist_ok=True)
            row["compute_only_s"] = min(timed(0, False, guarded=False) for _ in range(2))
            clean_trace = trace_dir / f"{label}-clean0-rank{rank}.json"
            row["clean0_forward_s"] = timed(0, True, profile=clean_trace if rank == 0 else None)
            sealed = time.perf_counter()
            adapter.seal_chunk()
            row["clean0_seal_after_forward_s"] = time.perf_counter() - sealed
            row["noisy_s_by_chunk"] = []
            for chunk in range(1, chunks):
                trace = None
                if cfg["profile_chunk"] == chunk and rank == 0:
                    trace = trace_dir / f"{label}-noisy{chunk}-rank{rank}.json"
                row["noisy_s_by_chunk"].append(timed(chunk, False, profile=trace))
                if trace is not None:
                    row[f"noisy{chunk}_trace"] = _overlap_from_trace(trace)
                timed(chunk, True)
                adapter.seal_chunk()
            # The adapter allows one noisy forward per boundary: the last unprofiled one is the measurement,
            # the next (one more history chunk) is profiled.
            row["final_noisy_s"] = row["noisy_s_by_chunk"][-1]
            noisy_trace = trace_dir / f"{label}-noisy-rank{rank}.json"
            row["final_noisy_profiled_s"] = timed(chunks, False, profile=noisy_trace if rank == 0 else None)
            if rank == 0:
                row["clean0_trace"] = _overlap_from_trace(clean_trace)
                row["noisy_trace"] = _overlap_from_trace(noisy_trace)
            row["sp"] = sp.report()
            adapter.detach()
            sp.close()
            if runtime is not None:
                runtime.close()
            if pool is not None:
                pool.close()
            out["runs"][label] = row
            dist.barrier()

        # Warm-up (NCCL communicators, NVDEC sessions, x265/cuSOLVER loading) is discarded.
        run(False, "all-cold", prefix="warmup-")
        run(True, "all-cold", prefix="warmup-")
        for scenario in cfg["scenarios"]:
            for overlap in (False, True):
                run(overlap, scenario)
        dist.destroy_process_group()
    except BaseException as exc:
        import traceback
        out["error"] = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
    finally:
        (root / f"rank{rank}.json").write_text(json.dumps(out))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ffmpeg", required=True)
    parser.add_argument("--uuids", nargs=2, required=True)
    parser.add_argument("--layers", type=int, default=12)
    parser.add_argument("--chunks", type=int, default=5,
                        help="Chunks generated; the measured noisy forward sees chunks-2 history chunks")
    parser.add_argument("--frames", type=int, default=6)
    parser.add_argument("--height", type=int, default=30)
    parser.add_argument("--width", type=int, default=52)
    parser.add_argument("--k-rank", type=int, default=256)
    parser.add_argument("--v-rank", type=int, default=384)
    parser.add_argument("--compute-ms", type=float, default=30.0, help="Emulated native compute per layer")
    parser.add_argument("--hot-chunks", type=int, default=2, help="Hot budget in chunks for the partial scenario")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--profile-chunk", type=int, default=None,
                        help="Also profile the noisy forward of this chunk index")
    parser.add_argument("--keep-traces", type=Path, default=None, help="Copy rank-0 traces here")
    parser.add_argument("--queue-depth", type=int, default=4,
                        help="Owner capture queue depth (gathered matrices waiting for encode)")
    parser.add_argument("--scenarios", nargs="+", default=["all-cold", "hot-newest"])
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    root = Path(os.environ.get("TMPDIR", "/tmp")) / f"sp-overlap-{uuid.uuid4().hex}"
    root.mkdir(parents=True)
    cfg = {**{k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items() if k != "output"},
           "root": str(root)}
    try:
        context = multiprocessing.get_context("spawn")
        processes = [context.Process(target=worker, args=(rank, cfg)) for rank in range(2)]
        for process in processes:
            process.start()
        for process in processes:
            process.join()
        rows = [json.loads((root / f"rank{rank}.json").read_text()) for rank in range(2)]
        for row in rows:
            if "error" in row:
                raise RuntimeError(row["error"])
        summary = {}
        for scenario in args.scenarios:
            sync, ovl = rows[0]["runs"][f"{scenario}-sync"], rows[0]["runs"][f"{scenario}-overlap"]
            compute = min(sync["compute_only_s"], ovl["compute_only_s"])
            t_sync, t_ovl = sync["final_noisy_s"], ovl["final_noisy_s"]
            c_sync, c_ovl = sync["clean0_forward_s"], ovl["clean0_forward_s"]
            seal_sync, seal_ovl = sync["clean0_seal_after_forward_s"], ovl["clean0_seal_after_forward_s"]
            summary[scenario] = {
                "compute_only_forward_s": round(compute, 4),
                "noisy_with_history_s": {"sync": round(t_sync, 4), "overlap": round(t_ovl, 4)},
                "noisy_hidden_fraction_of_serial_cost": round((t_sync - t_ovl) / (t_sync - compute), 3)
                if t_sync > compute else None,
                "clean0_capture_forward_s": {"sync": round(c_sync, 4), "overlap": round(c_ovl, 4)},
                "clean0_hidden_fraction_of_serial_cost": round((c_sync - c_ovl) / (c_sync - compute), 3)
                if c_sync > compute else None,
                "clean0_seal_after_forward_s": {"sync": round(seal_sync, 4), "overlap": round(seal_ovl, 4)},
                "trace_rank0": {mode: {"clean0": run["clean0_trace"], "noisy": run["noisy_trace"]}
                                for mode, run in (("sync", sync), ("overlap", ovl))},
                "sp_rank0": {mode: {k: run["sp"][k] for k in ("cold_loads", "hot_hits", "prefetch_hits",
                                                              "decode_seconds", "prefetch_wait_seconds",
                                                              "exchange_seconds", "packet_seconds")}
                             for mode, run in (("sync", sync), ("overlap", ovl))},
            }
        result = {"schema": "reelcache.sp_overlap_measurement.v1", "config": {**cfg, "root": None},
                  "summary": summary, "ranks": rows}
        args.output.write_text(json.dumps(result, indent=1))
        print(json.dumps(summary, indent=1))
        if args.keep_traces is not None:
            args.keep_traces.mkdir(parents=True, exist_ok=True)
            for trace in (root / "traces").glob("*.json"):
                shutil.copy(trace, args.keep_traces / trace.name)
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    main()
