#!/usr/bin/env python3
"""Fresh official-prompt MAGI SP2 benchmark; --help/prepare never load a model.

``schema`` prints the immutable case/protocol JSON contract. Prepare in a NEW
source-bound checkout/output on parent-approved local SSD. ``condition`` needs
explicit CPU T5 permission; ``run`` needs both GPUs; ``decode`` needs a later,
separate GPU0 grant after SP2 exit. ``report`` measures paired SSIM and emits
lossless-RGB videos plus a handoff for the parent's official VBench evaluator.
The 24-frame prefix is excluded from each 96-frame, 24fps evaluated continuation.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    sub = cli.add_subparsers(dest="stage", required=True)
    sub.add_parser("schema", help="Print the exact immutable case/protocol contract (no selected prompts)")
    auxiliary = sub.add_parser("auxiliary-identity", help="Once-per-study CPU full T5/VAE/source hashing; no model imports")
    for option in ("output", "t5", "vae", "native-source", "qualified-plan"):
        auxiliary.add_argument(f"--{option}", type=Path, required=True)
    auxiliary.add_argument("--max-seconds", type=int, default=1800)
    prepare = sub.add_parser("prepare", help="Model-free binding of official prompt, sources and explicit budgets")
    for option in ("output", "case", "protocol", "qualified-plan", "source-project", "native-source", "weights", "t5", "vae"):
        prepare.add_argument(f"--{option}", type=Path, required=True)
    prepare.add_argument("--allow-lossy", action="store_true", help="Required fixed K256/V384 experimental opt-in")
    prepare.add_argument("--storage-placement", choices=("local_ssd", "nfs_prepare_only"), required=True)
    prepare.add_argument("--cpu-threads", type=int, default=4)
    prepare.add_argument("--max-seconds", type=int, default=10800, help="Hard per-case torchrun watchdog, default3h, maximum6h")
    prepare.add_argument("--stage-seconds", type=int, default=1800, help="Bound each native timed operation")
    prepare.add_argument("--condition-seconds", type=int, default=1800, help="Separate CPU FP32 T5 time budget")
    prepare.add_argument("--store-gib", type=int, default=100, help="Owner canonical store cap; require8GiB additional free disk")
    condition = sub.add_parser("condition", help="Fresh native FP32 T5 on CPU, never old-prompt conditioning")
    condition.add_argument("--output", type=Path, required=True)
    condition.add_argument("--cpu-model-grant", action="store_true")
    run = sub.add_parser("run", help="Fresh common prefix, native RAW gate and matched four-chunk closed-loop arms")
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--gpu-grant", action="store_true")
    decode = sub.add_parser("decode", help="Separate native stateless VAE and lossless-RGB96frame videos")
    decode.add_argument("--output", type=Path, required=True)
    decode.add_argument("--gpu-grant", action="store_true")
    decode.add_argument("--device", type=int, default=0)
    report = sub.add_parser("report", help="CPU paired SSIM, storage and official evaluator input contract")
    report.add_argument("--output", type=Path, required=True)
    return cli


def main(argv=None):
    args = parser().parse_args(argv)
    if args.stage in ("run", "decode") and not args.gpu_grant:
        raise SystemExit("GPU stage forbidden without the parent's explicit current grant.")
    if args.stage == "condition" and not args.cpu_model_grant:
        raise SystemExit("CPU T5 loading requires explicit --cpu-model-grant.")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from reelcache.adapters import magi_vbench

    try:
        getattr(magi_vbench, args.stage.replace("-", "_"))(args)
    except BaseException as exc:
        import os
        import traceback

        if hasattr(args, "output"):
            try:
                output = magi_vbench.native._output(args.output)
                if output.is_dir():
                    magi_vbench.native._write(output / f"error_{args.stage}_rank{os.environ.get('RANK', '0')}.json", {
                        "status": "failed", "stage": args.stage, "exception": str(exc),
                        "traceback": traceback.format_exc(),
                    })
            except (OSError, ValueError):
                pass
        raise


if __name__ == "__main__":
    main()
