#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Run one inference stage serially with immutable source/command receipts."""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import psutil
from prepare_baseline import ROOT, WORK, sha
from selector_assets import get_selector_asset


def selector_for_command(command):
    script = Path(command[0])
    script = (script if script.is_absolute() else ROOT / script).resolve()
    if script.is_relative_to((ROOT / "scripts/qwen_qsa").resolve()):
        variant_eight = any(
            arg == "--variant" and command[i + 1 : i + 2] == ["Eight"]
            for i, arg in enumerate(command)
        )
        key = "qwen-eight" if "eight" in script.stem or variant_eight else "qwen-four"
        return "qwen", get_selector_asset(key)
    return "olmo", get_selector_asset("olmo-four")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="No GPU work; do not acquire the GPU-stage lock",
    )
    parser.add_argument("--timeout-seconds", type=float, default=3600)
    parser.add_argument("label")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a Python script and its arguments are required")
    if args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be positive")
    model, asset = selector_for_command(command)
    selector = asset.weights
    selector_sha = asset.spec["compatibility_sha256"]
    before = sha(selector)
    assert before == asset.spec["files"]["selectors.safetensors"]
    folder = WORK / "stages" / args.label
    folder.mkdir(parents=True, exist_ok=True)
    status_path = folder / "status.json"
    if status_path.exists():
        parser.error(
            "stage label already exists; retain the old receipt and choose a new label"
        )
    lock = WORK / "gpu-stage.lock"
    if not args.cpu:
        try:
            with lock.open("x") as stream:
                stream.write(str(os.getpid()))
        except FileExistsError:
            parser.error("another recorded GPU stage owns the lock")
    receipt = {
        "state": "preparing",
        "command": [sys.executable, "-u", *command],
        "cwd": str(ROOT),
        "started_unix": time.time(),
        "training_updates": 0,
        "gpu_stage": not args.cpu,
        "timeout_seconds": args.timeout_seconds,
        "selector_model": model,
        "selector_path": str(selector),
        "selector_sha256_before": selector_sha,
        "selector_identity_kind": "historical_export_identity_sha256",
        "selector_file_sha256_before": before,
        "selector_asset": asset.receipt(),
        "source_sha256": {},
    }
    try:
        source = folder / "source"
        source.mkdir()
        for path in sorted((ROOT / "scripts/qsa").glob("*.py")):
            shutil.copyfile(path, source / path.name)
            receipt["source_sha256"][path.name] = sha(path)
        registry = ROOT / "scripts/qsa/selector_assets.json"
        shutil.copyfile(registry, source / registry.name)
        receipt["source_sha256"][registry.name] = sha(registry)
        if model == "qwen":
            qwen_source = source / "qwen_qsa"
            qwen_source.mkdir()
            for path in sorted((ROOT / "scripts/qwen_qsa").glob("*.py")):
                shutil.copyfile(path, qwen_source / path.name)
                receipt["source_sha256"]["qwen_qsa/" + path.name] = sha(path)
        receipt["state"] = "running"
        status_path.write_text(json.dumps(receipt, indent=2) + "\n")
        with (folder / "output.log").open("w") as log:
            process = subprocess.Popen(
                receipt["command"], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT
            )
            receipt["pid"] = process.pid
            status_path.write_text(json.dumps(receipt, indent=2) + "\n")
            try:
                low_memory_since = None
                while process.poll() is None:
                    if time.time() - receipt["started_unix"] > args.timeout_seconds:
                        raise TimeoutError(
                            f"{args.timeout_seconds:g}-second stage limit; partial evidence retained"
                        )
                    low = not args.cpu and psutil.virtual_memory().available < 2 * 2**30
                    low_memory_since = (
                        (low_memory_since or time.time()) if low else None
                    )
                    if low_memory_since and time.time() - low_memory_since > 5:
                        raise MemoryError(
                            "OS available RAM stayed below 2 GiB; stopped this stage before sustained paging"
                        )
                    time.sleep(1)
                receipt["exit_code"] = process.returncode
            except BaseException:
                # Windows venv redirectors can own a base-Python child. Stop
                # only this stage's descendants so no GPU child is orphaned.
                try:
                    children = psutil.Process(process.pid).children(recursive=True)
                except psutil.NoSuchProcess:
                    children = []
                for child in reversed(children):
                    try:
                        child.kill()
                    except psutil.NoSuchProcess:
                        pass
                if process.poll() is None:
                    process.kill()
                process.wait()
                raise
        assert get_selector_asset(asset.key).receipt() == asset.receipt()
        receipt["selector_sha256_after"] = selector_sha
        receipt["state"] = "complete" if receipt["exit_code"] == 0 else "failed"
    except BaseException as exc:
        receipt.update(state="failed", error=repr(exc))
        raise
    finally:
        receipt["selector_file_sha256_after"] = (
            sha(selector) if selector.exists() else None
        )
        receipt["selector_sha256_after"] = (
            selector_sha if receipt["selector_file_sha256_after"] == before else None
        )
        receipt["elapsed_seconds"] = time.time() - receipt["started_unix"]
        status_path.write_text(json.dumps(receipt, indent=2) + "\n")
        if not args.cpu and lock.exists() and lock.read_text() == str(os.getpid()):
            lock.unlink()
        print(json.dumps(receipt, indent=2), flush=True)
    if receipt["exit_code"]:
        raise SystemExit(receipt["exit_code"])


if __name__ == "__main__":
    main()
