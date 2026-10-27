"""Fixed byte-spaced sampling without label selection; preserve complete raw lines."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sample(source: Path, destination: Path, chunks: int, chunk_bytes: int) -> dict:
    if destination.exists():
        raise FileExistsError(destination)
    stat = source.stat()
    if stat.st_size < chunks * chunk_bytes:
        raise ValueError("Requested chunks exceed input size")
    segments, total, anomalies = [], 0, 0
    digest = hashlib.sha256()
    with source.open("rb") as reader, destination.open("wb") as writer:
        for i in range(chunks):
            requested = i * (stat.st_size - chunk_bytes) // max(1, chunks - 1)
            if requested:
                reader.seek(requested - 1)
                if reader.read(1) != b"\n":
                    reader.readline()
            else:
                reader.seek(0)
            actual = reader.tell()
            end = requested + chunk_bytes
            count, abnormal, first, last = 0, 0, None, None
            while reader.tell() < end:
                line = reader.readline()
                if not line:
                    break
                if not line.endswith(b"\n"):
                    # Skip an incomplete final record, never synthesize one.
                    break
                fields = line.split(maxsplit=2)
                stamp = float(fields[1])
                first = stamp if first is None else first
                last = stamp
                abnormal += int(fields[0] != b"-")
                count += 1
                writer.write(line)
                digest.update(line)
            segments.append({"requested_start_byte": requested, "start_byte": actual,
                             "end_byte": reader.tell(), "lines": count, "abnormal_lines": abnormal,
                             "first_epoch": first, "last_epoch": last})
            total += count
            anomalies += abnormal
    return {"source": str(source.resolve()), "source_bytes": stat.st_size,
            "source_mtime_ns": stat.st_mtime_ns, "sample": str(destination.resolve()),
            "sample_bytes": destination.stat().st_size, "sample_sha256": digest.hexdigest(),
            "lines": total, "abnormal_lines": anomalies, "segments": segments}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "inputs/pilot_20261026_64x1m")
    parser.add_argument("--chunks", type=int, default=64)
    parser.add_argument("--chunk-bytes", type=int, default=1024 * 1024)
    args = parser.parse_args()
    if args.chunks < 2 or args.chunk_bytes < 1:
        raise ValueError("Need at least two chunks and positive chunk size")
    folder = args.output.resolve()
    folder.mkdir(parents=True, exist_ok=True)
    base = Path("D:/PaperWork/count_granger/dataset")
    manifest = {"policy": "equally_spaced_byte_offsets_no_label_selection", "chunks": args.chunks,
                "chunk_bytes": args.chunk_bytes, "domains": {}}
    for name, filename in [("bgl", "BGL/BGL.log"), ("thunderbird", "Thunderbird/Thunderbird.log")]:
        result = sample(base / filename, folder / f"{name}.log", args.chunks, args.chunk_bytes)
        manifest["domains"][name] = result
        print(name, "lines", result["lines"], "abnormal lines", result["abnormal_lines"], flush=True)
    (folder / "sampling_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
