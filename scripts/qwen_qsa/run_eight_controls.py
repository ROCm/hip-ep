#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""After eight timings, check 4K default drift and expand only if it exceeds 10%."""

import json
import os
import shutil
import subprocess
import sys
import time

import psutil
from verify_handoff import ROOT, WORK, sha

TASK = WORK / "eight-qsa-v1"


def save(value):
    p = TASK / "control-status.json"
    temporary = p.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, p)


def main():
    state = dict(
        state="waiting_for_eight_campaign", stages=[], started_unix=time.time()
    )
    assert not (TASK / "control-status.json").exists()
    save(state)
    while True:
        campaign = json.loads((TASK / "status.json").read_text())
        assert campaign["state"] != "failed", "Debug the primary campaign first"
        if campaign["state"] == "complete":
            break
        assert time.time() - state["started_unix"] < 5400
        time.sleep(1)
    lock = ROOT / "_local/qsa/gpu-stage.lock"
    with lock.open("x") as stream:
        stream.write(str(os.getpid()))
    process = None
    state["state"] = "running"
    try:
        source = TASK / "control-source"
        source.mkdir()
        for p in (ROOT / "scripts/qwen_qsa").glob("*eight*.py"):
            shutil.copyfile(p, source / p.name)
        for context in [4096, 16384, 32768]:
            command = [
                sys.executable,
                "-u",
                "scripts/qwen_qsa/measure_eight_control.py",
                "--context",
                str(context),
            ]
            stage = dict(context=context, command=command, started_unix=time.time())
            state["stages"].append(stage)
            save(state)
            with (TASK / f"fresh-original-{context}.log").open("x") as log:
                process = subprocess.Popen(
                    command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT
                )
                stage["pid"] = process.pid
                save(state)
                low_since = None
                while process.poll() is None:
                    assert time.time() - stage["started_unix"] < 2400
                    low = psutil.virtual_memory().available < 4 * 2**30
                    low_since = (low_since or time.time()) if low else None
                    assert not low_since or time.time() - low_since < 5
                    time.sleep(1)
            stage.update(
                exit_code=process.returncode,
                elapsed_seconds=time.time() - stage["started_unix"],
            )
            assert process.returncode == 0
            save(state)
            if context == 4096:
                old = json.loads(
                    (
                        WORK / "native-gqa/paired-performance/4096-b1/summary.json"
                    ).read_text()
                )
                new = json.loads(
                    (TASK / "fresh-original/4096/summary.json").read_text()
                )
                drift = {
                    phase: 100
                    * (
                        new["phases"][phase]["mean_seconds"]
                        / old["phases"][phase]["Original"]["mean_seconds"]
                        - 1
                    )
                    for phase in ["prefill", "decode"]
                }
                state["default_4k_elapsed_drift_percent"] = drift
                state["expanded_controls"] = any(abs(x) > 10 for x in drift.values())
                if not state["expanded_controls"]:
                    break
        state.update(
            state="complete",
            worker_sha256=sha(ROOT / "scripts/qwen_qsa/measure_eight_control.py"),
        )
    except BaseException as exc:
        state.update(state="failed", error=repr(exc))
        if process and process.poll() is None:
            for child in reversed(psutil.Process(process.pid).children(recursive=True)):
                child.kill()
            process.kill()
            process.wait()
        raise
    finally:
        state["elapsed_seconds"] = time.time() - state["started_unix"]
        save(state)
        if lock.exists() and lock.read_text() == str(os.getpid()):
            lock.unlink()


if __name__ == "__main__":
    main()
