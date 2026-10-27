"""Download a fixed official model revision into this project, with LFS SHA-256 checks."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
REPO = "sentence-transformers/all-MiniLM-L6-v2"
REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
FILES = ["config.json", "config_sentence_transformers.json", "modules.json",
         "sentence_bert_config.json", "special_tokens_map.json", "tokenizer.json",
         "tokenizer_config.json", "vocab.txt", "1_Pooling/config.json", "model.safetensors"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "models/all-MiniLM-L6-v2")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        f"https://huggingface.co/api/models/{REPO}/revision/{REVISION}?blobs=true",
        headers={"User-Agent": "mtg-gan-reproduction/0.1"})
    with urllib.request.urlopen(request, timeout=30) as response:
        metadata = json.load(response)
    if metadata["sha"] != REVISION:
        raise RuntimeError("Unexpected upstream model revision")
    entries = {entry["rfilename"]: entry for entry in metadata["siblings"]}
    manifest = {"repository": REPO, "revision": REVISION, "files": {}}
    for name in FILES:
        entry = entries[name]
        destination = output / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        expected_hash = entry.get("lfs", {}).get("sha256")
        if destination.exists():
            digest = hashlib.sha256(destination.read_bytes()).hexdigest()
            if destination.stat().st_size != entry["size"] or (expected_hash and digest != expected_hash):
                raise RuntimeError(f"Existing model file failed verification: {destination}")
        else:
            print(f"Downloading {name} ({entry['size']:,} bytes)", flush=True)
            temporary = destination.with_name(destination.name + ".partial")
            digest_state = hashlib.sha256()
            size = 0
            with urllib.request.urlopen(f"https://huggingface.co/{REPO}/resolve/{REVISION}/{name}", timeout=60) as response:
                with temporary.open("wb") as stream:
                    while chunk := response.read(1024 * 1024):
                        stream.write(chunk)
                        digest_state.update(chunk)
                        size += len(chunk)
            digest = digest_state.hexdigest()
            if size != entry["size"] or (expected_hash and digest != expected_hash):
                raise RuntimeError(f"Downloaded model file failed verification: {name}")
            temporary.replace(destination)
        manifest["files"][name] = {"bytes": destination.stat().st_size, "sha256": digest,
                                    "upstream_lfs_sha256": expected_hash}
    (output / "download_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Verified model ready: {output}", flush=True)


if __name__ == "__main__":
    main()
