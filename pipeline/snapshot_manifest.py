#!/usr/bin/env python3
"""Create a portable as-of sidecar for an externally produced HftBacktest snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Iterable


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def latest_local_timestamp(paths: Iterable[str | Path]) -> int:
    import numpy as np

    latest: int | None = None
    for path in paths:
        with np.load(path, allow_pickle=False) as archive:
            if "data" not in archive:
                raise ValueError(f"NPZ file is missing the data array: {path}")
            events = archive["data"]
        if events.ndim != 1 or events.size == 0 or "local_ts" not in (events.dtype.names or ()):
            raise ValueError(f"NPZ file has no usable local timestamps: {path}")
        file_latest = int(events["local_ts"].max())
        latest = file_latest if latest is None else max(latest, file_latest)
    if latest is None:
        raise ValueError("at least one non-empty NPZ file is required")
    return latest


def write_snapshot_manifest(
    snapshot_path: str | Path,
    as_of_ns: int,
    source: str = "external",
) -> Path:
    snapshot = Path(snapshot_path)
    if isinstance(as_of_ns, bool) or not isinstance(as_of_ns, int) or as_of_ns < 0:
        raise ValueError("as_of_ns must be a non-negative integer")
    if not source.strip() or "/" in source or "\\" in source or ":" in source:
        raise ValueError("source must be a non-empty logical label, not a path")
    snapshot_latest = latest_local_timestamp([snapshot])
    if as_of_ns < snapshot_latest:
        raise ValueError("as_of_ns precedes a local timestamp stored in the snapshot")

    manifest = {
        "schema_version": 1,
        "as_of_ns": as_of_ns,
        "snapshot_sha256": sha256_file(snapshot),
        "source": source,
    }
    manifest_path = Path(f"{snapshot}.manifest.json")
    temporary = Path(f"{manifest_path}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, sort_keys=True, separators=(",", ":"))
        handle.write("\n")
    os.replace(temporary, manifest_path)
    return manifest_path


def write_snapshot_manifest_from_sources(
    snapshot_path: str | Path,
    source_files: Iterable[str | Path],
    source: str = "generated-eod",
) -> Path:
    return write_snapshot_manifest(
        snapshot_path,
        latest_local_timestamp(source_files),
        source=source,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Write the required as-of/hash sidecar for an externally produced snapshot."
    )
    parser.add_argument("--snapshot", required=True, help="Existing HftBacktest snapshot NPZ")
    parser.add_argument(
        "--as-of-ns",
        required=True,
        type=int,
        help="Latest information time represented by the snapshot, in nanoseconds",
    )
    parser.add_argument(
        "--source",
        default="external",
        help="Logical provenance label; do not include a machine-specific path",
    )
    args = parser.parse_args()
    manifest_path = write_snapshot_manifest(args.snapshot, args.as_of_ns, args.source)
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
