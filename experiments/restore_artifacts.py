"""Verify and restore the published, byte-preserved experiment artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "experiments/provenance/public_artifacts_2026-09-06.json"


def verify_bytes(data: bytes, entry: dict) -> None:
    if len(data) != entry["bytes"] or hashlib.sha256(data).hexdigest() != entry["sha256"]:
        raise ValueError(f"Artifact checksum mismatch: {entry['path']}")


def destination(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Artifact path leaves destination: {relative}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Check every archived and directly published file without extracting.",
    )
    parser.add_argument(
        "--destination",
        type=Path,
        default=ROOT,
        help="Repository root or a separate extraction directory.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing files whose contents differ from the release.",
    )
    args = parser.parse_args()
    target = args.destination.resolve()
    manifest = json.loads(MANIFEST.read_text())
    planned = []
    seen = set()
    total_bytes = 0

    # Verify the whole payload before writing any files.
    for archive in manifest["archives"]:
        archive_path = destination(ROOT, archive["path"])
        verify_bytes(archive_path.read_bytes(), archive)
        with zipfile.ZipFile(archive_path) as bundle:
            expected = {entry["path"] for entry in archive["files"]}
            if len(bundle.namelist()) != len(expected) or set(bundle.namelist()) != expected:
                raise ValueError(f"Archive member mismatch: {archive['path']}")
            for entry in archive["files"]:
                data = bundle.read(entry["path"])
                verify_bytes(data, entry)
                if entry["path"] in seen:
                    raise ValueError(f"Duplicate artifact: {entry['path']}")
                seen.add(entry["path"])
                total_bytes += len(data)
                planned.append((archive_path, entry))
    for entry in manifest["direct_files"]:
        data = destination(ROOT, entry["path"]).read_bytes()
        verify_bytes(data, entry)
        if entry["path"] in seen:
            raise ValueError(f"Duplicate artifact: {entry['path']}")
        seen.add(entry["path"])
        total_bytes += len(data)
        planned.append((None, entry))
    if len(seen) != manifest["total_files"] or total_bytes != manifest["total_uncompressed_bytes"]:
        raise ValueError("Manifest totals do not match the verified payload")
    print(f"Verified {len(seen)} files ({total_bytes:,} bytes).")
    if args.verify_only:
        return

    conflicts = []
    for _, entry in planned:
        path = destination(target, entry["path"])
        if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            conflicts.append(entry["path"])
    if conflicts and not args.overwrite:
        raise SystemExit(
            "Existing files differ; use a separate --destination or explicit --overwrite:\n"
            + "\n".join(conflicts)
        )

    restored = 0
    for archive_path, entry in planned:
        path = destination(target, entry["path"])
        if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"]:
            continue
        if archive_path is None:
            data = destination(ROOT, entry["path"]).read_bytes()
        else:
            with zipfile.ZipFile(archive_path) as bundle:
                data = bundle.read(entry["path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        restored += 1
    print(f"Restored {restored} files into {target}.")


if __name__ == "__main__":
    main()
