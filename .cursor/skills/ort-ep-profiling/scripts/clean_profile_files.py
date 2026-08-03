#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""Remove stale onnxruntime_perf_test profile_*.json files from a directory."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def clean_profile_files(directory: Path, dry_run: bool = False) -> list[Path]:
    matches = sorted(directory.glob("profile_*.json"))
    if not dry_run:
        for path in matches:
            path.unlink()
    return matches


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Delete profile_*.json files before a fresh perf_test -p profile run."
    )
    parser.add_argument(
        "directory",
        type=Path,
        nargs="?",
        default=Path("."),
        help="Directory to clean (default: current working directory)",
    )
    parser.add_argument(
        "-n",
        "--dry-run",
        action="store_true",
        help="List matching files without deleting them",
    )
    args = parser.parse_args()

    directory = args.directory.resolve()
    if not directory.is_dir():
        print(f"error: not a directory: {directory}", file=sys.stderr)
        return 1

    removed = clean_profile_files(directory, dry_run=args.dry_run)
    if not removed:
        print(f"no profile_*.json files in {directory}", file=sys.stderr)
        return 0

    verb = "would remove" if args.dry_run else "removed"
    for path in removed:
        print(f"{verb}: {path}", file=sys.stderr)
    print(f"{verb} {len(removed)} file(s) from {directory}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
