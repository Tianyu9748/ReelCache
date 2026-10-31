"""Compare software, host-NVDEC and device-NVDEC restoration with stage timers."""

import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpu-receipt", type=Path, required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--ffmpeg", type=Path, required=True)
    parser.add_argument("--repeats", type=int, choices=(2, 3), default=2)
    args = parser.parse_args()
    from examples.causvid_runtime_controls import admit_device, verify_cpu_receipt
    source_id = verify_cpu_receipt(args.cpu_receipt)
    admit_device(args.gpu_uuid, minimum_free_gib=48)
    if args.output.exists():
        raise FileExistsError(args.output)
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_uuid
    import torch
    from reelcache.codecs import HEVCProvider
    from reelcache.nvdec import NVDECDecoder
    from reelcache.telemetry import RuntimeTelemetry
    from reelcache.types import EncodedTensor
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    root = args.store.resolve()

    def read(oid):
        kind, digest = oid.split(":")
        raw = (root / "objects" / kind / digest).read_bytes()
        if hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError("archive object hash mismatch")
        return raw
    def manifest(oid):
        return json.loads(read(oid))["data"]

    with sqlite3.connect((root / "index.sqlite3").as_uri() + "?mode=ro&immutable=1", uri=True) as db:
        head = db.execute("SELECT object_id FROM refs WHERE name='branches/parent'").fetchone()[0]
    packets = [matrix for chunk in manifest(head)["chunks"] for matrix in manifest(chunk)["matrices"]]
    rows = []
    with ExitStack() as stack:
        host = stack.enter_context(NVDECDecoder(gpu_id=0, gpu_uuid=args.gpu_uuid))
        device = stack.enter_context(NVDECDecoder(gpu_id=0, gpu_uuid=args.gpu_uuid, device_output=True))
        providers = {
            "software": HEVCProvider(str(args.ffmpeg)),
            "nvdec_host": HEVCProvider(str(args.ffmpeg), nvdec=host),
            "nvdec_device_verify": HEVCProvider(str(args.ffmpeg), nvdec=device, verify_decoded_coefficients=True),
            "nvdec_device_fast": HEVCProvider(str(args.ffmpeg), nvdec=device),
        }
        names = tuple(providers)
        for index, matrix in enumerate(packets):
            encoded = manifest(matrix["tensor"])["encoded"]
            packet = EncodedTensor(
                encoded["metadata"], {key: read(oid) for key, oid in encoded["payloads"].items()},
                read(encoded["basis"]),
            )
            dtype = torch.float32 if matrix["key"]["target"] == "K" else torch.bfloat16
            reference = providers["software"].decode_tensor(packet, device="cuda:0", dtype=dtype)
            order = names[index % len(names):] + names[:index % len(names)]
            for name in order:
                samples = []
                for _ in range(args.repeats):
                    torch.cuda.synchronize()
                    tick = time.perf_counter()
                    actual = providers[name].decode_tensor(packet, device="cuda:0", dtype=dtype)
                    torch.cuda.synchronize()
                    elapsed = time.perf_counter() - tick
                    if not torch.equal(actual, reference):
                        raise RuntimeError(f"device restoration differs: {name}, {matrix['key']}")
                    samples.append(elapsed)
                telemetry = RuntimeTelemetry(cuda_timing=True)
                with telemetry.capture():
                    actual = providers[name].decode_tensor(packet, device="cuda:0", dtype=dtype)
                torch.cuda.synchronize()
                if not torch.equal(actual, reference):
                    raise RuntimeError("profiled restoration differs from its control")
                rows.append({
                    "tensor": matrix["tensor"], "key": matrix["key"], "backend": name,
                    "unprofiled_seconds": samples, "profiled_stages": telemetry.snapshot(), "exact": True,
                })
            if (index + 1) % 30 == 0:
                print(f"Compared {index + 1}/{len(packets)} packets across four restoration backends", flush=True)
        stats = {"host": host.stats(), "device": device.stats()}
    verify_cpu_receipt(args.cpu_receipt)
    result = {
        "schema": "reelcache.gpu_restore_benchmark.v1", "status": "passed",
        "source_tree_sha256": source_id, "head": head, "packet_count": len(packets),
        "repeats": args.repeats, "results": rows, "decoder_stats": stats,
        "scope": "same archived packets; unprofiled synchronized total timing separate from synchronized CUDA-event phase profiling",
        "input_object_reads_timed": False,
        "CPU_checksum_default_on_device_path": False,
    }
    with args.output.open("x") as output:
        json.dump(result, output, indent=2)


if __name__ == "__main__":
    main()
