"""Compare all fitting/compression backends on hash-bound original pre-RoPE K/V."""

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpu-receipt", type=Path, required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--ffmpeg", type=Path, required=True)
    parser.add_argument("--repeats", type=int, choices=(2, 3), default=3)
    parser.add_argument("--threads", type=int, choices=(1, 2, 4), default=4)
    args = parser.parse_args()
    from examples.causvid_runtime_controls import admit_device, verify_cpu_receipt
    identity = verify_cpu_receipt(args.cpu_receipt)
    admission = admit_device(args.gpu_uuid, minimum_free_gib=48)
    if args.output.exists():
        raise FileExistsError(args.output)
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_uuid
    import torch
    from reelcache.codecs import HEVCProvider, encoded_nbytes
    from reelcache.gpu_fitting import GPUFitter, GPU_FIT_METHODS
    from reelcache.serialization import deserialize_tensors
    from reelcache.types import CodecSpec, TensorContext

    torch.set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.init()
    cpu_rng, gpu_rng = torch.random.get_rng_state().clone(), torch.cuda.get_rng_state().clone()
    sources = json.loads(args.sources.read_text())
    records = []
    methods = ("cpu", *GPU_FIT_METHODS)
    for item in sources["sources"]:
        raw = Path(item["path"]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != item["sha256"]:
            raise ValueError("original-source file changed")
        source = deserialize_tensors(raw)["source"]
        context = TensorContext.from_dict(item["context"])
        if context.key.target == "K" and context.key_space != "pre_rope":
            raise ValueError("GPU fitting benchmark requires original pre-RoPE K")
        rank = 320 if context.key.target == "K" else 576
        exact = torch.linalg.svdvals(source.to("cuda", dtype=torch.float64), driver="gesvd")
        energy = exact.square().sum()
        optimal_residual = 0. if float(energy) == 0. else float((exact[rank:].square().sum() / energy).sqrt())
        del exact, energy
        for index, method in enumerate(methods):
            fitter = None if method == "cpu" else GPUFitter(gpu_id=0, gpu_uuid=args.gpu_uuid, method=method)
            provider = HEVCProvider(str(args.ffmpeg), fitter=fitter)
            spec = CodecSpec("svd_hevc", rank=rank)
            warmup = provider.encode_tensor(source, context, spec)
            del warmup
            samples = []
            for repeat in range(args.repeats):
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                start = time.perf_counter()
                packet = provider.encode_tensor(source, context, spec)
                torch.cuda.synchronize()
                duration = time.perf_counter() - start
                residual = packet.metrics["projection_relative_residual"]
                if not 0 <= residual <= 1.01 or residual + 1e-5 < optimal_residual:
                    raise RuntimeError("fitting residual violates the direct FP64 SVD control")
                if not torch.equal(cpu_rng, torch.random.get_rng_state()) or not torch.equal(gpu_rng, torch.cuda.get_rng_state()):
                    raise RuntimeError("fitting backend changed the generation RNG")
                samples.append({
                    "wall_seconds": duration, "metrics": packet.metrics,
                    "packet_bytes": encoded_nbytes(packet),
                    "optimal_fp64_residual": optimal_residual,
                    "residual_gap": residual - optimal_residual,
                    "peak_torch_allocated_bytes": torch.cuda.max_memory_allocated(),
                })
            records.append({
                "context": context.to_dict(), "source_file_sha256": item["sha256"], "backend": method,
                "samples": samples, "median_compression_seconds": statistics.median(s["wall_seconds"] for s in samples),
            })
            print(context.key, method, records[-1]["median_compression_seconds"], flush=True)
    verify_cpu_receipt(args.cpu_receipt)
    result = {
        "schema": "reelcache.gpu_fitting_benchmark.v1", "status": "passed",
        "source_tree_sha256": identity, "admission": admission,
        "source_manifest_sha256": hashlib.sha256(args.sources.read_bytes()).hexdigest(),
        "repeats": args.repeats, "cpu_threads": args.threads, "records": records,
        "methods": methods, "RNG_unchanged": True,
        "scope": "original native source matrices; full compression includes SVD, transfer, CPU normalization, HEVC and validation",
        "Torch_peak_excludes_CuPy_and_driver_allocations": True,
    }
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)


if __name__ == "__main__":
    main()
