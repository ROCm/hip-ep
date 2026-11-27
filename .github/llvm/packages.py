#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Resolve immutable Linux LLVM artifacts from the default-branch publisher."""

import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

REPO = os.environ["GITHUB_REPOSITORY"]
DEFAULT = os.environ["DEFAULT_BRANCH"]
RECIPE = ".github/llvm/build.sh"
PUBLISHER = ".github/workflows/linux-llvm-publish.yml"
PACKAGE = re.compile(
    r"llvm-ubuntu24-x64-(x86(?:-amdgpu)?)-([0-9a-f]{40})-([0-9a-f]{40})"
)
UPSTREAM = "https://github.com/ROCm/llvm-project.git"


def api(path):
    return json.loads(subprocess.check_output(["gh", "api", path]))


def output(**values):
    with open(os.environ["GITHUB_OUTPUT"], "a") as stream:
        for key, value in values.items():
            print(f"{key}={value}", file=stream)


def recipe_hash():
    return subprocess.check_output(["git", "hash-object", RECIPE], text=True).strip()


def pinned_sha():
    for line in Path("cmake/deps.txt").read_text().splitlines():
        if line.startswith("llvm_project;"):
            _, url, sha = line.split(";")
            if url == UPSTREAM and re.fullmatch(r"[0-9a-f]{40}", sha):
                return sha
    raise ValueError("Expected a pinned ROCm/llvm-project commit in cmake/deps.txt")


def artifact(name):
    # A PR can upload an artifact with any name. Only accept packages from this
    # repository's default-branch workflow_run publisher.
    entries = api(f"repos/{REPO}/actions/artifacts?name={name}&per_page=100")[
        "artifacts"
    ]
    for item in entries:
        source = item["workflow_run"]
        if item["expired"] or not source or source["head_branch"] != DEFAULT:
            continue
        run = api(f"repos/{REPO}/actions/runs/{source['id']}")
        if run["event"] == "workflow_run" and run["path"] == PUBLISHER:
            return {"artifact_id": item["id"], "run_id": run["id"]}
    return None


def plan(variant):
    sha = pinned_sha()
    recipe = recipe_hash()
    name = f"llvm-ubuntu24-x64-{variant}-{sha}-{recipe}"
    if not PACKAGE.fullmatch(name):
        raise ValueError(f"Invalid package: {name}")
    tree = api(f"repos/{REPO}/git/trees/{DEFAULT}?recursive=1")
    if tree["truncated"]:
        raise RuntimeError(
            "Cannot verify the default-branch recipe from a truncated tree"
        )
    files = {entry["path"]: entry["sha"] for entry in tree["tree"]}
    # Bootstrap this workflow and test future recipe changes in the requesting
    # run. An unreviewed recipe must never publish a package for other PRs.
    preview = files.get(RECIPE) != recipe or PUBLISHER not in files
    found = None if preview else artifact(name)
    output(name=name, sha=sha, preview=str(preview).lower(), **(found or {}))


def request(run_id):
    # A reusable job exposes its package name only after build-required and
    # package lookup finish. Warm, docs-only and recipe-preview runs need no build.
    deadline = time.monotonic() + 5 * 3600
    while time.monotonic() < deadline:
        jobs = api(f"repos/{REPO}/actions/runs/{run_id}/jobs?per_page=100")["jobs"]
        for job in jobs:
            match = re.fullmatch(r".* / await (" + PACKAGE.pattern + ")", job["name"])
            if match and job["status"] in ("in_progress", "queued"):
                name, variant, sha, recipe = match.groups()
                if recipe != recipe_hash():
                    raise ValueError(
                        "Requested recipe differs from the publisher checkout"
                    )
                output(name=name, variant=variant, sha=sha)
                return
            if match and job["status"] == "completed":
                return
        # Older branches can still run the pre-publisher workflows.
        if any(
            job["name"] in ("build-linux", "build-linux-rmt")
            and job["status"] == "in_progress"
            for job in jobs
        ):
            return
        if api(f"repos/{REPO}/actions/runs/{run_id}")["status"] == "completed":
            return
        time.sleep(30)
    raise TimeoutError(f"No LLVM request resolved for run {run_id}")


def verify_source(sha):
    if not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise ValueError("Invalid LLVM SHA")
    if sha == pinned_sha():
        return
    # A fork-network commit can be fetched by SHA from GitHub. Before running
    # its build on the publisher, verify reachability from a canonical branch.
    refs = subprocess.check_output(
        ["git", "ls-remote", "--heads", UPSTREAM], text=True
    ).splitlines()
    heads = {}
    for line in refs:
        tip, ref = line.split()
        heads[ref] = tip
    if sha in heads.values():
        return
    branches = sorted(
        (r for r in heads if r.startswith("refs/heads/rmt-core")), reverse=True
    )
    for ref in ["refs/heads/amd-staging", *branches]:
        if ref not in heads:
            continue
        result = api(f"repos/ROCm/llvm-project/compare/{sha}...{heads[ref]}?per_page=1")
        if result["status"] in ("ahead", "identical"):
            return
    raise ValueError(f"LLVM commit {sha} is outside the approved upstream history")


def wait(name):
    deadline = time.monotonic() + 5 * 3600
    print(f"Waiting for the default-branch publisher: {name}", flush=True)
    while time.monotonic() < deadline:
        found = artifact(name)
        if found:
            output(**found)
            return
        # Fail promptly if the publisher for this request failed. A cancelled
        # pending publisher is normal: another producer still owns the key.
        runs = api(
            f"repos/{REPO}/actions/workflows/linux-llvm-publish.yml/runs"
            "?event=workflow_run&status=failure&per_page=50"
        )["workflow_runs"]
        title = f"Linux LLVM for run {os.environ['GITHUB_RUN_ID']} attempt {os.environ['GITHUB_RUN_ATTEMPT']}"
        for run in runs:
            if run["display_title"] == title:
                raise RuntimeError(f"LLVM publication failed: {run['html_url']}")
        time.sleep(60)
    raise TimeoutError(
        f"Package {name} was not published; inspect Linux LLVM publisher runs"
    )


if __name__ == "__main__":
    command, arg = sys.argv[1:]
    if command == "lookup":
        output(exists=str(bool(artifact(arg))).lower())
    else:
        {"plan": plan, "request": request, "verify": verify_source, "wait": wait}[
            command
        ](arg)
