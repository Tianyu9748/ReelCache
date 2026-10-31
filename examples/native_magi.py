#!/usr/bin/env python3
"""Source-bound MAGI SP2 verification, not a historical replay or throughput test.

Prepare and --help are CPU-only. ``run`` requires an explicit two-GPU grant;
``decode`` is a separate one-GPU stage after both SP2 workers exit. Outputs
must be new artifacts inside this ReelCache checkout. No model is loaded on
import. See ``prepare --help`` for the required external source/weight paths.

``worker`` uses source-authorized immutable study requests and verified local
assets, not legacy p00/p01 prefixes. ``decode-batch`` is a separate VAE stage
after both SP2 processes exit. Explicit quality-hot residency is a budgeted
study mode, never the default runtime policy or an HBM-compression claim.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    commands = cli.add_subparsers(dest="stage", required=True)
    prepare = commands.add_parser("prepare", help="CPU-only source/input/configuration binding")
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--binding", type=Path, required=True,
                         help="Existing p00/p01 pilot binding.json; inputs only, never an old replay")
    prepare.add_argument("--source-project", type=Path, required=True,
                         help="Existing video-state-managment checkout containing the proven builder")
    prepare.add_argument("--native-source", type=Path, required=True,
                         help="External FlowCache4MAGI-1 source checkout")
    prepare.add_argument("--weights", type=Path, required=True,
                         help="24B_distill/inference_weight.distill directory")
    prepare.add_argument("--vae", type=Path, required=True, help="External native VAE checkpoint directory")
    prepare.add_argument("--arms", nargs="+", choices=("raw", "svd", "svd_hevc"), default=["raw"])
    prepare.add_argument("--allow-lossy", action="store_true",
                         help="Explicit experimental opt-in; no video quality guarantee")
    prepare.add_argument("--continuations", type=int, choices=(1, 2), default=2)
    prepare.add_argument("--cpu-threads", type=int, default=4)
    prepare.add_argument("--resident-mib", type=int, default=1536,
                         help="Owner CPU reconstruction budget; native HBM is separately measured")
    run = commands.add_parser("run", help="Native SP2 RAW gates, fork/later chunk and selected lossy arms")
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--gpu-grant", action="store_true",
                     help="Required confirmation of the parent's explicit grant of BOTH GPUs")
    decode = commands.add_parser("decode", help="Native stateless-per-chunk VAE, after SP2 exits")
    decode.add_argument("--output", type=Path, required=True)
    decode.add_argument("--device", type=int, default=0)
    decode.add_argument("--gpu-grant", action="store_true",
                        help="Required separate one-GPU grant, after both SP2 workers exit")
    report = commands.add_parser("report", help="CPU-only loss-separated RGB/latent/storage/timing report")
    report.add_argument("--output", type=Path, required=True)
    for name, help_text in (
        ("worker", "Verified official-prompt SP2 latent batch; fresh state per request"),
        ("decode-batch", "Separate verified VAE batch, only after BOTH SP2 processes exit"),
    ):
        stage = commands.add_parser(name, help=help_text)
        stage.add_argument("--protocol", type=Path, required=True)
        stage.add_argument("--assets", type=Path, required=True, help="Verified local ROOT/assets.json")
        stage.add_argument("--launch-receipt", type=Path, required=True)
        stage.add_argument("--requests", type=Path, required=True,
                           help="Source-authorized reelcache.rank_requests.v1 homogeneous epoch")
        stage.add_argument("--gpu-grant", action="store_true",
                           help="Explicit parent grant; a passing code/asset receipt is NOT a GPU grant")
        if name == "decode-batch":
            stage.add_argument("--generation-manifest", type=Path, required=True)
            stage.add_argument("--device", type=int, default=0)
    return cli


def main(argv=None):
    args = parser().parse_args(argv)
    if args.stage in ("run", "decode") and not args.gpu_grant:
        raise SystemExit("GPU work is forbidden without an explicit parent grant; CPU readiness is not a grant.")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from reelcache.adapters import magi_native

    try:
        getattr(magi_native, args.stage.replace("-", "_"))(args)
    except BaseException as exc:
        import os
        import traceback

        try:
            output = magi_native._output(args.output) if getattr(args, "output", None) is not None else None
            if output is not None and output.is_dir():
                magi_native._write(
                    output / f"error_{args.stage}_rank{os.environ.get('RANK', '0')}.json",
                    {"status": "failed", "stage": args.stage, "exception": str(exc),
                     "traceback": traceback.format_exc()},
                )
        except (OSError, ValueError):
            pass
        raise


if __name__ == "__main__":
    main()
