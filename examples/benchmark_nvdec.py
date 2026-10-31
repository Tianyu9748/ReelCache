"""Verify and time NVDEC against software on every archived HEVC tensor packet."""

from __future__ import annotations

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
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--store", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    cli.add_argument("--gpu-uuid", required=True)
    cli.add_argument("--ffmpeg", type=Path, required=True)
    cli.add_argument("--cpu-receipt", type=Path, required=True)
    cli.add_argument("--repeats", type=int, choices=(1, 2, 3), default=2)
    args = cli.parse_args()
    from examples.causvid_runtime_controls import admit_device, verify_cpu_receipt
    source = verify_cpu_receipt(args.cpu_receipt)
    admission = admit_device(args.gpu_uuid, minimum_free_gib=48)
    if args.output.exists():
        raise FileExistsError(args.output)
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_uuid
    import numpy as np
    import torch
    from reelcache import _hevc
    from reelcache.codecs import decode_tensor
    from reelcache.nvdec import NVDECDecoder
    from reelcache.types import EncodedTensor
    torch.set_num_threads(1)
    store = args.store.resolve()

    def read(oid):
        kind, digest = oid.split(":")
        raw = (store / "objects" / kind / digest).read_bytes()
        if hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError(f"archive object checksum failed: {oid}")
        return raw

    def manifest(oid):
        return json.loads(read(oid))["data"]

    with sqlite3.connect((store / "index.sqlite3").as_uri() + "?mode=ro&immutable=1", uri=True) as db:
        head = db.execute("SELECT object_id FROM refs WHERE name='branches/parent'").fetchone()[0]
    rows, packets = [], []
    for chunk_id in manifest(head)["chunks"]:
        for row in manifest(chunk_id)["matrices"]:
            packets.append((row, row["tensor"]))
    started = time.monotonic()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with NVDECDecoder(gpu_id=0, gpu_uuid=args.gpu_uuid) as gpu, _hevc.using_ffmpeg(args.ffmpeg):
        for index, (row, tensor_id) in enumerate(packets):
            encoded = manifest(tensor_id)["encoded"]
            packet = EncodedTensor(
                encoded["metadata"], {name: read(oid) for name, oid in encoded["payloads"].items()},
                None if encoded["basis"] is None else read(encoded["basis"]),
            )
            if packet.metadata["kind"] != "svd_hevc":
                raise ValueError("NVDEC benchmark expects an all-HEVC archive")
            layout = packet.metadata["video"]
            streams = [packet.payloads[f"coefficients.{i:03d}.hevc"] for i in range(layout["segments"])]
            samples = []
            for repeat in range(args.repeats):
                outputs, timings = {}, {}
                order = ("nvdec", "software") if (index + repeat) % 2 else ("software", "nvdec")
                for backend in order:
                    tick = time.perf_counter()
                    output, _ = (gpu.decode_pixels(streams, layout) if backend == "nvdec"
                                 else _hevc.decode_pixels(streams, layout))
                    timings[backend] = time.perf_counter() - tick
                    outputs[backend] = output
                if not np.array_equal(outputs["software"], outputs["nvdec"]):
                    raise RuntimeError(f"NVDEC pixels differ for {tensor_id}")
                samples.append(timings)
            reference = decode_tensor(packet)
            reconstructed = decode_tensor(packet, nvdec=gpu)
            if not torch.equal(reference, reconstructed):
                raise RuntimeError(f"NVDEC coefficient checksum/reconstruction parity failed: {tensor_id}")
            rows.append({
                "tensor": tensor_id, "key": row["key"], "pixels_exact": True, "reconstruction_exact": True,
                "video": layout, "timings": samples,
            })
            if (index + 1) % 30 == 0:
                print(f"Verified {index + 1}/{len(packets)} packet pixels and reconstructions", flush=True)
        stats = gpu.stats()
    verify_cpu_receipt(args.cpu_receipt)
    result = {
        "schema": "reelcache.nvdec_packet_benchmark.v1", "status": "passed",
        "gpu_admission": admission, "source_tree_sha256": source, "store": str(store), "head": head,
        "repeats": args.repeats, "packet_count": len(rows), "results": rows, "nvdec": stats,
        "wall_seconds": time.monotonic() - started,
        "timing_scope": "in-process native pixels plus host unpack; input object reads excluded; alternating backend order",
        "software_ffmpeg": str(args.ffmpeg), "software_ffmpeg_sha256": hashlib.sha256(args.ffmpeg.read_bytes()).hexdigest(),
        "cpu_receipt_sha256": hashlib.sha256(args.cpu_receipt.read_bytes()).hexdigest(),
        "cold_first_call_included": True, "native_video_quality_measured": False,
    }
    with args.output.open("x") as out:
        json.dump(result, out, indent=2)
    print(json.dumps({key: result[key] for key in ("status", "packet_count", "wall_seconds")}))


if __name__ == "__main__":
    main()
