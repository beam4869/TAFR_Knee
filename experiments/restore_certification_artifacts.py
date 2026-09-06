"""Verify and restore the certification campaign's per-seed measurements."""

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "experiments/results/certification_gate/raw_manifest.json"


def safe_path(root, name):
    path = (root / name).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Path leaves destination: {name}")
    return path


def verify(data, entry):
    if len(data) != entry["bytes"] or hashlib.sha256(data).hexdigest() != entry["sha256"]:
        raise ValueError(f"Checksum mismatch: {entry['path']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--destination", type=Path, default=ROOT)
    args = parser.parse_args()
    target = args.destination.resolve()
    manifest = json.loads(MANIFEST.read_text())
    archive = safe_path(ROOT, manifest["archive"]["path"])
    verify(archive.read_bytes(), manifest["archive"])
    planned = []
    with zipfile.ZipFile(archive) as bundle:
        names = [entry["path"] for entry in manifest["files"]]
        if len(set(names)) != len(names) or sorted(bundle.namelist()) != sorted(names):
            raise ValueError("Archive membership mismatch")
        for entry in manifest["files"]:
            data = bundle.read(entry["path"])
            verify(data, entry)
            path = safe_path(target, entry["path"])
            if path.exists() and path.read_bytes() != data:
                raise ValueError(f"Different existing measurement; use --destination: {path}")
            planned.append((path, data))
    if not args.verify_only:
        for path, data in planned:
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
    print(f"Verified {len(planned)} records; restore requested={not args.verify_only}.")


if __name__ == "__main__":
    main()
