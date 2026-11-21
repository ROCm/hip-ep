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

import psutil
from prepare_baseline import ROOT, SELECTOR, SELECTOR_SHA, WORK, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="No GPU work; do not acquire the GPU-stage lock",
    )
    parser.add_argument("label")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a Python script and its arguments are required")
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
        "selector_sha256_before": sha(SELECTOR),
        "source_sha256": {},
    }
    try:
        assert receipt["selector_sha256_before"] == SELECTOR_SHA
        source = folder / "source"
        source.mkdir()
        for path in sorted((ROOT / "scripts/qsa").glob("*.py")):
            shutil.copyfile(path, source / path.name)
            receipt["source_sha256"][path.name] = sha(path)
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
                    if time.time() - receipt["started_unix"] > 3600:
                        raise TimeoutError(
                            "One-hour stage limit; partial evidence retained"
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
            except (TimeoutError, MemoryError):
                # Windows venv redirectors can own a base-Python child. Stop
                # only this stage's descendants so no GPU child is orphaned.
                for child in reversed(
                    psutil.Process(process.pid).children(recursive=True)
                ):
                    try:
                        child.kill()
                    except psutil.NoSuchProcess:
                        pass
                process.kill()
                process.wait()
                raise
        receipt["selector_sha256_after"] = sha(SELECTOR)
        assert receipt["selector_sha256_after"] == SELECTOR_SHA
        receipt["state"] = "complete" if receipt["exit_code"] == 0 else "failed"
    except BaseException as exc:
        receipt.update(state="failed", error=repr(exc))
        raise
    finally:
        receipt["selector_sha256_after"] = sha(SELECTOR)
        receipt["elapsed_seconds"] = time.time() - receipt["started_unix"]
        status_path.write_text(json.dumps(receipt, indent=2) + "\n")
        if not args.cpu:
            lock.unlink()
        print(json.dumps(receipt, indent=2), flush=True)
    if receipt["exit_code"]:
        raise SystemExit(receipt["exit_code"])


if __name__ == "__main__":
    main()
