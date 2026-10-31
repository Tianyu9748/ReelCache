"""Two-GPU (NCCL/NVLink + NVDEC) verification of tiered SP history for MAGI.

Runs the real MagiAdapter seam with an emulated Ulysses attention on two
GPUs (fake model, synthetic low-rank KV, realistic MAGI KV geometry) and a
real owner KVRuntime (GPU Gram fitting + multithreaded libx265). It compares:

* ``runtime``      existing path: owner decodes each rank's columns, Gloo CPU transfer
* ``sp-owner``     sp_tiered, all cold chunks decoded on the owner GPU, NCCL scatter
* ``sp-rr``        sp_tiered, cold chunks assigned round-robin to both GPUs (concurrent)
* ``sp-hot``       sp_tiered with native hot shards retained on each GPU

and checks bitwise parity (sp-rr == sp-owner; sp-hot == native clean KV).
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


def worker(rank, cfg):
    import hashlib
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
    torch.backends.cuda.matmul.allow_tf32 = False
    out = {"rank": rank, "modes": {}}
    try:
        dist.init_process_group("gloo", rank=rank, world_size=2, timeout=timedelta(seconds=900),
                                init_method=f"file://{root / 'rendezvous'}")
        data_group = dist.new_group(backend="nccl")
        native_group = dist.new_group(backend="nccl")
        L, chunks = cfg["layers"], cfg["chunks"]
        geometry = MagiGeometry(layers=L, kv_heads=8, head_dim=128, frames=cfg["frames"],
                                height=cfg["height"], width=cfg["width"], world_size=2)
        tokens = geometry.tokens
        identity = ModelIdentity("sp_gpu_verify", "synthetic", L, geometry.channels, "global_KV_channels", "bfloat16")
        specs = {MatrixKey(i, t): CodecSpec("svd_hevc", rank=cfg["k_rank"] if t == "K" else cfg["v_rank"])
                 for i in range(L) for t in ("K", "V")}
        generator = torch.Generator().manual_seed(3)
        basis = torch.randn(160, geometry.channels, generator=generator)
        mixed = [((torch.randn(tokens, 160, generator=generator) @ basis
                   + 0.05 * torch.randn(tokens, geometry.channels, generator=generator)) / 6)
                 .reshape(tokens, 1, -1).to(device) for _ in range(chunks + 1)]
        angle = torch.arange(tokens, dtype=torch.float32)[:, None].expand(tokens, 64) * 1e-3
        cosine, sine = angle.cos().to(device), angle.sin().to(device)
        uuids = cfg["uuids"]
        decoder_provider = HEVCProvider(cfg["ffmpeg"], nvdec=NVDECDecoder(
            gpu_id=rank, gpu_uuid=uuids[rank], device_output=True))
        fitter = GPUFitter(gpu_id=0, gpu_uuid=uuids[0], method="gram", shared=True) if rank == 0 else None
        with decoder_provider.nvdec:
            for mode in cfg["modes"]:
                cache_mode = "runtime" if mode == "runtime" else "sp_tiered"
                pool = BoundedExecutor(max_workers=4, max_pending=4, max_staging_bytes=8 << 30) if rank == 0 else None
                runtime = None
                if rank == 0:
                    encoder = HEVCProvider(cfg["ffmpeg"], fitter=fitter, x265_threads=4)
                    runtime = KVRuntime(root / f"store-{mode}", identity, max_resident_bytes=8 << 30,
                                        encoder=encoder.encode_tensor, decoder=decoder_provider.decode_tensor,
                                        encode_workspace_bytes=64 << 20)
                owner = DistributedOwner(expected_world_size=2, max_wire_bytes=64 << 20)
                sp = None
                if cache_mode == "sp_tiered":
                    sp = SPTieredHistory(
                        coordinator=owner, data_group=data_group, layout=geometry.layout, device=device,
                        native_dtype="bfloat16",
                        decoder_factory=lambda: HEVCProvider(cfg["ffmpeg"], nvdec=NVDECDecoder(
                            gpu_id=rank, gpu_uuid=uuids[rank], device_output=True)),
                        rotary_apply=fake_rotary,
                        hot_budget_bytes=(8 << 30) if mode == "sp-hot" else 0,
                        decode_placement="owner" if mode == "sp-owner" else "round_robin",
                    )
                model = FakeModel(geometry)
                cache = SimpleNamespace(max_batch_size=1, max_sequence_length=(chunks + 2) * tokens,
                                        key_value_memory_dict={}, update_kv_cache=True)
                adapter = MagiAdapter(
                    model, cache, runtime, model_identity=identity, branch_id="main",
                    staging_dir=root / f"staging-{mode}-{rank}", geometry=geometry, coordinator=owner,
                    specs=specs, rotary_apply=fake_rotary, device=device, max_layer_buffer_bytes=8 << 30,
                    cache_mode=cache_mode, sp_history=sp, stream_executor=pool if sp is not None else None,
                )
                adapter.attach()
                native = {}

                def forward(chunk, clean, record=None):
                    meta = SimpleNamespace(
                        slice_point=chunk, clip_token_nums=tokens, start_chunk_id=chunk, end_chunk_id=chunk + 1,
                        denoising_range_num=1, extract_prefix_video_feature=clean,
                        distill_nearly_clean_chunk=False, timestep=torch.full((1, 1), 0.9999 if clean else 0.0),
                    )
                    layer_ms = []
                    with adapter.execution(chunk_index=chunk, clean=clean, clean_timestep=0.9999 if clean else None):
                        rows = geometry.layout.rows(rank)
                        for index, layer in enumerate(model.videodit_blocks.layers):
                            torch.cuda.synchronize()
                            begun = time.perf_counter()
                            attention = layer.self_attention
                            source = mixed[chunk] + 0.01 * index
                            k = attention.get_k(source[rows], cosine[rows], sine[rows])
                            v = attention.get_v(source[rows])
                            packed = torch.cat((k, v), -1)
                            pieces = [torch.empty_like(packed) for _ in range(2)]
                            dist.all_gather(pieces, packed, group=native_group)
                            heads = torch.cat(pieces)[:, rank * 4:(rank + 1) * 4].contiguous()
                            if clean and mode == "sp-hot":
                                native[(chunk, index)] = hashlib.sha256(
                                    heads.cpu().view(torch.uint8).numpy().tobytes()).hexdigest()
                            key_hist, value_hist = attention.adjust_key_and_value_for_inference(heads, cache, meta)
                            if record is not None:
                                history = torch.cat((key_hist[:chunk * tokens], value_hist[:chunk * tokens]), -1)
                                record.append(history.cpu())
                            torch.cuda.synchronize()
                            layer_ms.append((time.perf_counter() - begun) * 1e3)
                    return layer_ms

                timings = []
                for chunk in range(chunks):
                    torch.cuda.synchronize()
                    started = time.perf_counter()
                    noisy_ms = forward(chunk, False)
                    clean_started = time.perf_counter()
                    forward(chunk, True)
                    sealed_started = time.perf_counter()
                    adapter.seal_chunk()
                    done = time.perf_counter()
                    timings.append({"chunk": chunk, "noisy_forward_s": clean_started - started,
                                    "noisy_layer_ms_mean": sum(noisy_ms) / len(noisy_ms),
                                    "clean_forward_s": sealed_started - clean_started,
                                    "seal_after_forward_s": done - sealed_started})
                history = []
                final_ms = forward(chunks, False, record=history)
                digests = [hashlib.sha256(h.view(torch.uint8).numpy().tobytes()).hexdigest() for h in history]
                row = {"timings": timings, "final_history_layer_ms": final_ms,
                       "final_history_layer_ms_mean": sum(final_ms) / len(final_ms),
                       "history_digests": digests, "sp": None if sp is None else sp.report()}
                if mode == "sp-hot":
                    # hot history must equal the native clean head slices exactly
                    ok = True
                    for layer, h in enumerate(history):
                        for c in range(chunks):
                            piece = h[c * tokens:(c + 1) * tokens].contiguous()
                            if hashlib.sha256(piece.view(torch.uint8).numpy().tobytes()).hexdigest() != native[(c, layer)]:
                                ok = False
                    row["hot_equals_native_clean_kv"] = ok
                if runtime is not None:
                    packets = []
                    for chunk_object in runtime.branch_info("main")["chunk_manifests"]:
                        for entry in runtime.store.get_manifest(chunk_object)["matrices"]:
                            packet = runtime.store.get_manifest(entry["tensor"])
                            packets.append(packet["encoded"]["payloads"])
                    row["packet_payload_ids_sha"] = hashlib.sha256(json.dumps(packets, sort_keys=True).encode()).hexdigest()
                torch.save(history, root / f"history-{mode}-rank{rank}.pt")
                adapter.detach()
                if sp is not None:
                    sp.close()
                if runtime is not None:
                    runtime.close()
                if pool is not None:
                    pool.close()
                out["modes"][mode] = row
                dist.barrier()
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
    parser.add_argument("--chunks", type=int, default=4)
    parser.add_argument("--frames", type=int, default=6)
    parser.add_argument("--height", type=int, default=30)
    parser.add_argument("--width", type=int, default=52)
    parser.add_argument("--k-rank", type=int, default=256)
    parser.add_argument("--v-rank", type=int, default=384)
    parser.add_argument("--threads", type=int, default=8, help="Torch CPU threads per rank process")
    parser.add_argument("--modes", nargs="+", default=["runtime", "sp-owner", "sp-rr", "sp-hot"])
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    import torch

    root = Path(os.environ.get("TMPDIR", "/tmp")) / f"sp-gpu-verify-{uuid.uuid4().hex}"
    root.mkdir(parents=True)
    cfg = {"root": str(root), "ffmpeg": args.ffmpeg, "uuids": args.uuids, "layers": args.layers,
           "chunks": args.chunks, "frames": args.frames, "height": args.height, "width": args.width,
           "k_rank": args.k_rank, "v_rank": args.v_rank, "modes": args.modes, "threads": args.threads}
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
        parity = {}
        for rank in range(2):
            loaded = {mode: torch.load(root / f"history-{mode}-rank{rank}.pt") for mode in args.modes}
            if "sp-rr" in loaded and "sp-owner" in loaded:
                parity[f"rank{rank}_sp_rr_equals_sp_owner_bitwise"] = all(
                    torch.equal(a, b) for a, b in zip(loaded["sp-rr"], loaded["sp-owner"]))
            if "runtime" in loaded and "sp-rr" in loaded:
                diffs = [((a.float() - b.float()).abs().max().item(), (a.float() - b.float()).norm().item()
                          / max(b.float().norm().item(), 1e-30)) for a, b in zip(loaded["sp-rr"], loaded["runtime"])]
                parity[f"rank{rank}_sp_rr_vs_runtime_max_abs"] = max(d[0] for d in diffs)
                parity[f"rank{rank}_sp_rr_vs_runtime_max_rel_fro"] = max(d[1] for d in diffs)
                parity[f"rank{rank}_sp_rr_equals_runtime_bitwise"] = all(
                    torch.equal(a, b) for a, b in zip(loaded["sp-rr"], loaded["runtime"]))
            if "sp-hot" in loaded and "sp-rr" in loaded:
                diffs = [(a.float() - b.float()).norm().item() / max(b.float().norm().item(), 1e-30)
                         for a, b in zip(loaded["sp-rr"], loaded["sp-hot"])]
                parity[f"rank{rank}_cold_vs_native_hot_max_rel_fro"] = max(diffs)
        summary = {}
        for mode in args.modes:
            r0, r1 = rows[0]["modes"][mode], rows[1]["modes"][mode]
            summary[mode] = {
                "final_history_layer_ms_mean": [round(r0["final_history_layer_ms_mean"], 2),
                                                round(r1["final_history_layer_ms_mean"], 2)],
                "noisy_layer_ms_mean_by_chunk": [round(t["noisy_layer_ms_mean"], 2) for t in r0["timings"]],
                "clean_forward_s": [round(t["clean_forward_s"], 3) for t in r0["timings"]],
                "seal_after_forward_s": [round(t["seal_after_forward_s"], 3) for t in r0["timings"]],
                "hot_equals_native_clean_kv": r0.get("hot_equals_native_clean_kv"),
                "packet_payload_ids_sha": r0.get("packet_payload_ids_sha"),
                "sp_rank0": r0["sp"], "sp_rank1": r1["sp"],
            }
        result = {"schema": "reelcache.sp_tiered_gpu_verification.v1", "config": {**cfg, "root": None},
                  "parity": parity, "summary": summary, "ranks": rows}
        args.output.write_text(json.dumps(result, indent=1))
        print(json.dumps({"parity": parity, "summary": {m: {k: v for k, v in s.items() if not k.startswith("sp_")}
                                                        for m, s in summary.items()}}, indent=1))
        for mode in args.modes:
            if summary[mode]["sp_rank0"]:
                print(mode, {f"{k}@rank{r}": summary[mode][f"sp_rank{r}"][k] for r in (0, 1)
                             for k in ("decoded_here", "cold_by_decoder_rank", "decode_seconds", "exchange_seconds",
                                       "packet_seconds", "gather_seconds", "hot_hits")})
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    main()
