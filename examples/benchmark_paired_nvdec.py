"""Compare sequential/concurrent NVDEC pixel decoding on all archived K/V pairs."""

import argparse
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
    parser.add_argument("--repeats", type=int, choices=(3, 5), default=5)
    args = parser.parse_args()
    from examples.causvid_runtime_controls import admit_device, verify_cpu_receipt
    source = verify_cpu_receipt(args.cpu_receipt)
    admission = admit_device(args.gpu_uuid, minimum_free_gib=48)
    if args.output.exists():
        raise FileExistsError(args.output)
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_uuid
    import torch
    from reelcache.codecs import HEVCProvider
    from reelcache.nvdec import NVDECDecoder
    from reelcache.paired_decode import PairedDecoder
    torch.set_num_threads(1)
    store = args.store.resolve()
    def read(oid):
        kind, digest = oid.split(":")
        data = (store / "objects" / kind / digest).read_bytes()
        if hashlib.sha256(data).hexdigest() != digest:
            raise ValueError("archive content checksum failed")
        return data
    def manifest(oid):
        return json.loads(read(oid))["data"]
    with sqlite3.connect((store / "index.sqlite3").as_uri() + "?mode=ro&immutable=1", uri=True) as db:
        head = db.execute("SELECT object_id FROM refs WHERE name='branches/parent'").fetchone()[0]
    records = []
    with NVDECDecoder(gpu_id=0, gpu_uuid=args.gpu_uuid, device_output=True) as decoder, \
            PairedDecoder(HEVCProvider(nvdec=decoder), device="cuda:0", max_output_bytes=256 * 1024**2) as paired:
        for chunk_id in manifest(head)["chunks"]:
            chunk = manifest(chunk_id)
            for layer in range(30):
                inputs, ids = [], []
                for target in ("K", "V"):
                    row = next(x for x in chunk["matrices"] if x["key"] == {"layer": layer, "target": target})
                    encoded = manifest(row["tensor"])["encoded"]
                    layout = encoded["metadata"]["video"]
                    streams = [read(encoded["payloads"][f"coefficients.{i:03d}.hevc"])
                               for i in range(layout["segments"])]
                    inputs.append((streams, layout))
                    ids.append(row["tensor"])
                sequential = tuple(decoder.decode_pixels(*p)[0] for p in inputs)
                actual = paired.decode_pixels(inputs)
                if not all(torch.equal(a, b) for a, b in zip(sequential, actual)):
                    raise RuntimeError("warm paired NVDEC pixel parity failed")
                samples = []
                for repeat in range(args.repeats):
                    times, outputs = {}, {}
                    order = ("serial", "paired") if (layer + repeat) % 2 == 0 else ("paired", "serial")
                    for name in order:
                        torch.cuda.synchronize()
                        tick = time.perf_counter()
                        outputs[name] = (tuple(decoder.decode_pixels(*p)[0] for p in inputs)
                                         if name == "serial" else paired.decode_pixels(inputs))
                        torch.cuda.synchronize()
                        times[name] = time.perf_counter() - tick
                    if not all(torch.equal(a, b) for a, b in zip(outputs["serial"], outputs["paired"])):
                        raise RuntimeError("paired NVDEC changed coefficient pixels")
                    samples.append(times)
                records.append({"chunk": chunk_id, "layer": layer, "tensors": ids, "exact": True,
                                "timings": samples})
            print(f"Verified {len(records)} real K/V pairs", flush=True)
        stats = paired.stats()
    verify_cpu_receipt(args.cpu_receipt)
    result = {
        "schema": "reelcache.paired_nvdec_benchmark.v1", "status": "passed",
        "source_tree_sha256": source, "GPU_admission": admission,
        "store": str(store), "head": head, "pairs": records, "repeats": args.repeats,
        "paired_decoder": stats, "scope": "same preloaded streams; decode+GPU unpack, no packet reads or GEMM",
        "physical_decoder_engine_assignment_measured": False,
    }
    with args.output.open("x") as output:
        json.dump(result, output, indent=2)


if __name__ == "__main__":
    main()
