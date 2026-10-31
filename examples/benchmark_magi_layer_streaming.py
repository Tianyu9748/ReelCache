"""SP=2 MAGI clean-forward + publication timing: gather-all-then-encode vs layer-streamed encode.

Two Gloo ranks run the fake MAGI attention seam (real adapter, real KVRuntime,
real SVD+HEVC encoder with an optional GPU fitter on the owner). Native layer
compute is emulated with a fixed per-layer sleep, so the result measures how
much publication work the streaming pipeline hides behind the forward.
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


def worker(rank, args):
    import torch
    import torch.distributed as dist
    from types import SimpleNamespace

    from reelcache.adapters.magi import MagiAdapter, MagiGeometry
    from reelcache.codecs import HEVCProvider
    from reelcache.distributed import DistributedOwner
    from reelcache.execution import BoundedExecutor
    from reelcache.manager import KVRuntime
    from reelcache.types import CodecSpec, MatrixKey, ModelIdentity
    from tests.test_magi_adapter import FakeModel, fake_rotary

    root = Path(args["root"])
    torch.set_num_threads(2)
    out = {"rank": rank, "modes": {}}
    try:
        dist.init_process_group("gloo", rank=rank, world_size=2, timeout=timedelta(seconds=600),
                                init_method=f"file://{root / 'rendezvous'}")
        stream_group = dist.new_group(backend="gloo")
        L = args["layers"]
        geometry = MagiGeometry(layers=L, kv_heads=8, head_dim=128, frames=3, height=30, width=52, world_size=2)
        identity = ModelIdentity("bench_magi", "synthetic", L, geometry.channels, "global_KV_channels", "bfloat16")
        specs = {MatrixKey(i, t): CodecSpec("svd_hevc", rank=args["k_rank"] if t == "K" else args["v_rank"])
                 for i in range(L) for t in ("K", "V")}
        generator = torch.Generator().manual_seed(7)
        tokens = geometry.tokens
        basis = torch.randn(96, geometry.channels, generator=generator)
        mixed_by_chunk = [
            (torch.randn(tokens, 96, generator=generator) @ basis
             + 0.05 * torch.randn(tokens, geometry.channels, generator=generator)).reshape(tokens, 1, -1) / 4
            for _ in range(args["chunks"])
        ]
        angle = torch.arange(tokens, dtype=torch.float32)[:, None].expand(tokens, 64) * 0.001
        cosine, sine = angle.cos(), angle.sin()
        fitter = None
        if rank == 0 and args["gpu_uuid"]:
            from reelcache.gpu_fitting import GPUFitter
            fitter = GPUFitter(gpu_id=0, gpu_uuid=args["gpu_uuid"], method="gram", shared=True)
        for mode in args["modes"]:
            streaming = mode != "serial"
            workers = int(mode.split("pool")[1]) if "pool" in mode else 0
            pool = (BoundedExecutor(max_workers=workers, max_pending=workers, max_staging_bytes=4 << 30)
                    if rank == 0 and workers else None)
            runtime = None
            if rank == 0:
                provider = HEVCProvider(args["ffmpeg"], fitter=fitter)
                runtime = KVRuntime(root / f"store-{mode}", identity, max_resident_bytes=1 << 31,
                                    encoder=provider.encode_tensor, decoder=provider.decode_tensor,
                                    encode_workspace_bytes=64 << 20)
            model = FakeModel(geometry)
            cache = SimpleNamespace(max_batch_size=1, max_sequence_length=(args["chunks"] + 1) * tokens,
                                    key_value_memory_dict={}, update_kv_cache=True)
            owner = DistributedOwner(expected_world_size=2, max_wire_bytes=16 << 20)
            adapter = MagiAdapter(
                model, cache, runtime, model_identity=identity, branch_id="main",
                staging_dir=root / f"staging-{mode}-{rank}", geometry=geometry, coordinator=owner,
                specs=specs, rotary_apply=fake_rotary, max_layer_buffer_bytes=1 << 31,
                layer_streaming=streaming, stream_group=stream_group if streaming else None,
                stream_executor=pool,
            )
            adapter.attach()
            rows = []
            for chunk in range(args["chunks"]):
                meta = SimpleNamespace(
                    slice_point=chunk, clip_token_nums=tokens, start_chunk_id=chunk, end_chunk_id=chunk + 1,
                    denoising_range_num=1, extract_prefix_video_feature=True, distill_nearly_clean_chunk=False,
                    timestep=torch.full((1, 1), 0.9999),
                )
                mixed = mixed_by_chunk[chunk]
                local_rows = geometry.layout.rows(rank)
                dist.barrier()
                started = time.perf_counter()
                with adapter.execution(chunk_index=chunk, clean=True, clean_timestep=0.9999):
                    for layer in model.videodit_blocks.layers:
                        attention = layer.self_attention
                        k = attention.get_k(mixed[local_rows], cosine[local_rows], sine[local_rows])
                        v = attention.get_v(mixed[local_rows])
                        packed = torch.cat((k, v), -1)
                        pieces = [torch.empty_like(packed) for _ in range(2)]
                        dist.all_gather(pieces, packed)
                        heads = torch.cat(pieces)[:, rank * 4:(rank + 1) * 4].contiguous()
                        attention.adjust_key_and_value_for_inference(heads, cache, meta)
                        time.sleep(args["layer_ms"] / 1e3)
                forward_done = time.perf_counter()
                adapter.seal_chunk()
                sealed = time.perf_counter()
                rows.append({"chunk": chunk, "forward_s": forward_done - started, "seal_s": sealed - forward_done,
                             "total_s": sealed - started, "stream": adapter.last_stream_stats})
            adapter.detach()
            if runtime is not None:
                info = runtime.branch_info("main")
                out["modes"][mode] = {"chunks": rows, "chunk_ids": info["chunk_ids"]}
                runtime.close()
            else:
                out["modes"][mode] = {"chunks": rows}
            if pool is not None:
                pool.close()
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
    parser.add_argument("--gpu-uuid", default=None, help="Owner GPU for gram fitting (shared fitter)")
    parser.add_argument("--layers", type=int, default=24)
    parser.add_argument("--chunks", type=int, default=3)
    parser.add_argument("--layer-ms", type=float, default=60.0)
    parser.add_argument("--k-rank", type=int, default=256)
    parser.add_argument("--v-rank", type=int, default=384)
    parser.add_argument("--modes", nargs="+", default=["serial", "stream", "stream-pool8"])
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    root = Path(os.environ.get("TMPDIR", "/tmp")) / f"magi-stream-bench-{uuid.uuid4().hex}"
    root.mkdir(parents=True)
    config = {"root": str(root), "ffmpeg": args.ffmpeg, "gpu_uuid": args.gpu_uuid, "layers": args.layers,
              "chunks": args.chunks, "layer_ms": args.layer_ms, "k_rank": args.k_rank, "v_rank": args.v_rank,
              "modes": args.modes}
    try:
        context = multiprocessing.get_context("spawn")
        if args.gpu_uuid:
            os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_uuid
        processes = [context.Process(target=worker, args=(rank, config)) for rank in range(2)]
        for process in processes:
            process.start()
        for process in processes:
            process.join()
        rows = [json.loads((root / f"rank{rank}.json").read_text()) for rank in range(2)]
        for row in rows:
            if "error" in row:
                raise RuntimeError(row["error"])
        summary = {}
        for mode in args.modes:
            chunks = rows[0]["modes"][mode]["chunks"]
            summary[mode] = {
                "forward_s": [round(c["forward_s"], 3) for c in chunks],
                "seal_s": [round(c["seal_s"], 3) for c in chunks],
                "total_s": [round(c["total_s"], 3) for c in chunks],
            }
        result = {"schema": "reelcache.magi_layer_streaming_benchmark.v1", "config": {**config, "root": None},
                  "summary": summary, "ranks": rows}
        args.output.write_text(json.dumps(result, indent=1))
        print(json.dumps(summary, indent=1))
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    main()
