"""Exercise durable clean chunks, restart and branching using small synthetic KV."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time


def parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New, empty output directory")
    parser.add_argument("--codec", choices=("raw", "svd", "svd_hevc"), default="raw")
    parser.add_argument(
        "--normalization", choices=("per_channel", "global", "leading_columns"),
        default="per_channel",
    )
    parser.add_argument(
        "--hevc-inter-prediction", action=argparse.BooleanOptionalAction, default=False,
        help="svd_hevc GOP: all-intra by default; enable the documented inter GOP explicitly",
    )
    parser.add_argument("--allow-lossy", action="store_true")
    return parser


def main(argv=None):
    cli = parser()
    args = cli.parse_args(argv)
    if args.codec != "raw" and not args.allow_lossy:
        cli.error("lossy demonstrations require --allow-lossy; they are not quality-qualified")
    if args.codec != "svd_hevc" and args.normalization != "per_channel":
        cli.error("normalization applies only to svd_hevc")
    if args.codec != "svd_hevc" and args.hevc_inter_prediction:
        cli.error("--hevc-inter-prediction applies only to svd_hevc")
    if args.output.exists() and any(args.output.iterdir()):
        cli.error("--output must be new or empty")
    args.output.mkdir(parents=True, exist_ok=True)
    for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(variable, "1")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import torch
    from reelcache.execution import BoundedExecutor
    from reelcache.manager import KVRuntime
    from reelcache.types import CodecSpec, MatrixKey, ModelIdentity, TensorContext

    torch.set_num_threads(1)
    generator = torch.Generator().manual_seed(7)
    model = ModelIdentity(
        "synthetic-demo",
        hashlib.sha256(b"reelcache-synthetic-v1-seed7-rank4").hexdigest(),
        2, 16, "canonical", "float32",
    )
    keys = [MatrixKey(layer, target) for layer in range(2) for target in ("K", "V")]
    bases = {key: torch.randn(4, 16, generator=generator) for key in keys}
    first = {
        key: torch.randn(64, 4, generator=generator) @ bases[key] for key in keys
    }
    second = {
        key: torch.randn(64, 4, generator=generator) @ bases[key] for key in keys
    }
    spec = CodecSpec() if args.codec == "raw" else CodecSpec(
        kind=args.codec, rank=4, normalization=args.normalization,
        protected_fraction=0.25 if args.normalization == "leading_columns" else 1.0,
        inter_prediction=args.hevc_inter_prediction,
    )

    def context(start):
        return {
            key: TensorContext(
                model, key, start, 2, 4, 8,
                ("post_rope" if args.codec == "raw" else "pre_rope")
                if key.target == "K" else "none",
            )
            for key in keys
        }

    runtime = KVRuntime(args.output / "state", model, max_resident_bytes=1 << 20)
    started = time.monotonic()
    try:
        runtime.create_branch("main")
        staged_bytes = sum(t.numel() * t.element_size() for t in first.values())
        with BoundedExecutor(max_staging_bytes=2 * staged_bytes) as executor:
            job = executor.submit(
                runtime.seal_chunk, "main", "chunk-0", first, context(0), spec,
                staging_bytes=staged_bytes,
            )
            job.result()
            scheduling = job.metrics()
        snapshot = runtime.snapshot(
            "main",
            metadata={
                "clean_boundary": {"chunk_id": "chunk-0", "token_end": 64, "clean": True},
                "conditioning": {"id": "synthetic"},
                "execution": {"next_chunk": 1, "pending": False},
                "rng": {"tensor": "rng"},
                "latent": {"native_latents": "not applicable to synthetic matrices"},
                "vae": {"native_decoder": "not applicable"},
            },
            tensors={"rng": generator.get_state()},
        )
        runtime.fork(snapshot, "child")
        runtime.seal_chunk(
            "child", "chunk-1", second, context(64), spec,
            reuse_threshold=0.01 if args.codec != "raw" else None,
        )
    finally:
        runtime.close()

    runtime = KVRuntime(args.output / "state", model, max_resident_bytes=1 << 20)
    errors = {}
    try:
        runtime.restore(snapshot, "resumed")
        for key in keys:
            with runtime.acquire_history("resumed", key) as lease:
                expected = first[key]
                errors[f"{key.layer}:{key.target}"] = float(
                    torch.linalg.vector_norm(lease.tensor - expected)
                    / torch.linalg.vector_norm(expected)
                )
                if args.codec == "raw" and not torch.equal(lease.tensor, expected):
                    raise AssertionError("RAW restart changed matrix values")
                restored_prefix = lease.tensor.clone()
            with runtime.acquire_history("main", key) as parent:
                if not torch.equal(parent.tensor, restored_prefix):
                    raise AssertionError("child append changed parent history")
            with runtime.acquire_history("child", key) as child:
                if (
                    child.tensor.shape != (128, 16)
                    or not torch.equal(child.tensor[:64], restored_prefix)
                ):
                    raise AssertionError("child history is incomplete")
                if args.codec == "raw" and not torch.equal(child.tensor[64:], second[key]):
                    raise AssertionError("RAW child continuation changed matrix values")
    finally:
        runtime.close()
    stored = sum(
        path.stat().st_size
        for path in (args.output / "state").rglob("*") if path.is_file()
    )
    report = {
        "schema": "reelcache_synthetic_demo_v1",
        "status": "passed",
        "codec": args.codec,
        "normalization": args.normalization if args.codec == "svd_hevc" else None,
        "codec_spec": spec.to_dict(),
        "hevc_inter_prediction": args.hevc_inter_prediction if args.codec == "svd_hevc" else None,
        "snapshot_id": snapshot,
        "restart_relative_KV_errors": errors,
        "parent_isolation": True,
        "state_store_bytes_including_snapshots": stored,
        "logical_unique_raw_KV_bytes": 2 * staged_bytes,
        "job_timing": scheduling,
        "wall_seconds": time.monotonic() - started,
        "native_video_quality_measured": False,
        "experimental_lossy": args.codec != "raw",
        "scope": "Synthetic CPU matrices, not a native model or GPU-memory benchmark",
    }
    with (args.output / "result.json").open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
