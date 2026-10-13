#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Fetch a gpu-test-package from CI when the caller does not have one.

The skill needs `hip-onnx-runner` and `hip-mlir-opt`, which CI already
builds and uploads for every commit. Rather than require a local build tree
or a manually placed package, take the newest one from a green build of
main.

The Windows and Linux builds publish separate artifacts, "gpu-test-package"
and "linux-gpu-test-package". Both the workflow and the name are pinned, so
a rename on either side fails loudly instead of fetching a package for the
wrong platform.

Downloads land in a per-artifact directory under LOCALAPPDATA and are
reused: the package is a few hundred megabytes, and a given artifact id
never changes.

Usage:
  python fetch_package.py [--print-only] [--refresh]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

REPO = "ROCm/hip-ep"
WORKFLOW = "windows-build.yml"  # "Windows Build"
ARTIFACT = "gpu-test-package"
BRANCH = "main"

# How many recent green runs to look through before giving up. Artifacts
# expire while their runs do not, so the newest green run is not always the
# newest available package.
RUNS_TO_SCAN = 15

REQUIRED_TOOLS = ("hip-onnx-runner.exe", "hip-mlir-opt.exe")


def cache_root() -> Path:
    base = os.environ.get("LOCALAPPDATA") or os.environ.get("TEMP") or "."
    return Path(base) / "hip-ep" / "gpu-test-package"


def gh() -> str:
    """The gh executable, or a message saying what to install."""
    found = shutil.which("gh")
    if not found:
        raise SystemExit(
            "the GitHub CLI (gh) is needed to fetch a package, and is not on "
            "PATH. Install it, or pass the package path instead."
        )
    return found


def gh_json(path: str) -> dict:
    """Call the GitHub API through gh, so it supplies the credentials."""
    proc = subprocess.run(
        [gh(), "api", path],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout).strip().splitlines()
        raise SystemExit(
            "gh api failed: " + (detail[-1] if detail else f"exit {proc.returncode}")
        )
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"gh api returned no JSON for {path}: {exc}") from exc


def find_artifact() -> tuple[int, int, str]:
    """Newest downloadable package from a green build of main.

    Returns (run id, artifact id, the commit it was built from).
    """
    runs = gh_json(
        f"repos/{REPO}/actions/workflows/{WORKFLOW}/runs"
        f"?branch={BRANCH}&status=success&per_page={RUNS_TO_SCAN}"
    ).get("workflow_runs", [])
    if not runs:
        raise SystemExit(
            f"no successful {WORKFLOW} run on {BRANCH}; pass the package path instead"
        )

    for run in runs:
        arts = gh_json(f"repos/{REPO}/actions/runs/{run['id']}/artifacts").get(
            "artifacts", []
        )
        for art in arts:
            if art["name"] == ARTIFACT and not art["expired"]:
                return run["id"], art["id"], run["head_sha"][:10]

    raise SystemExit(
        f"the last {len(runs)} green runs have no unexpired {ARTIFACT}; "
        "pass the package path instead"
    )


def is_usable(root: Path) -> bool:
    return all((root / "bin" / t).exists() for t in REQUIRED_TOOLS)


def package_root(extracted: Path) -> Path | None:
    """Where bin/ actually is, allowing for one wrapping directory.

    gh extracts a single named artifact straight into the target, but the
    zip has carried a `gpu-test-package/` level before.
    """
    if is_usable(extracted):
        return extracted
    for child in extracted.iterdir() if extracted.is_dir() else []:
        if child.is_dir() and is_usable(child):
            return child
    return None


def download(run_id: int, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [
            "gh",
            "run",
            "download",
            str(run_id),
            "--repo",
            REPO,
            "--name",
            ARTIFACT,
            "--dir",
            str(dest),
        ],
        text=True,
        encoding="utf-8",
        shell=True,
    )
    if proc.returncode != 0:
        shutil.rmtree(dest, ignore_errors=True)
        raise SystemExit(f"gh run download failed with exit code {proc.returncode}")


def ensure_package(refresh: bool = False) -> Path:
    """The path to a usable package, downloading one if needed."""
    run_id, artifact_id, sha = find_artifact()
    dest = cache_root() / str(artifact_id)

    if refresh:
        shutil.rmtree(dest, ignore_errors=True)
    cached = package_root(dest)
    if cached:
        print(f"Using cached gpu-test-package ({sha}): {cached}")
        return cached

    print(f"Downloading gpu-test-package from {BRANCH} ({sha}), a few hundred MB...")
    download(run_id, dest)
    root = package_root(dest)
    if root is None:
        # A layout change would otherwise show up much later, as a missing
        # tool in the middle of an analysis.
        raise SystemExit(
            f"downloaded artifact has no bin/{REQUIRED_TOOLS[0]} under {dest}"
        )
    print(f"OK: {root}")
    return root


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--print-only",
        action="store_true",
        help="Report which artifact would be used, without downloading.",
    )
    ap.add_argument("--refresh", action="store_true", help="Ignore any cached copy.")
    args = ap.parse_args()

    if args.print_only:
        run_id, artifact_id, sha = find_artifact()
        dest = cache_root() / str(artifact_id)
        state = "present" if package_root(dest) else "not downloaded"
        print(f"run {run_id}, artifact {artifact_id}, commit {sha}")
        print(f"cache: {dest}  ({state})")
        return

    # The path goes to stdout on its own line so the caller can read it back.
    path = ensure_package(refresh=args.refresh)
    print(f"PACKAGE_ROOT={path}")


if __name__ == "__main__":
    main()
