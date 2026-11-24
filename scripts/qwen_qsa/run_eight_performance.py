#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Supervise only the bounded eight-selector validation and performance stages."""

import json
import os
import shutil
import subprocess
import sys
import time

import psutil
from verify_handoff import ROOT, WORK, sha

TASK = WORK / "eight-qsa-v1"


def save(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def main():
    assert not os.environ.get("QWEN_QSA_INSTALL")
    assert not (TASK / "status.json").exists(), (
        "Preserve the earlier campaign and pause disposition; use a new versioned campaign"
    )
    export = TASK / "models/Eight/export.json"
    assert export.exists()
    lock = ROOT / "_local/qsa/gpu-stage.lock"
    with lock.open("x") as stream:
        stream.write(str(os.getpid()))
    state = dict(
        state="running",
        started_unix=time.time(),
        pid=os.getpid(),
        export_sha256=sha(export),
        stages=[],
        accuracy_evaluations_authorized=False,
    )
    source = TASK / "measurement-source"
    source.mkdir(exist_ok=False)
    paths = [
        *sorted((ROOT / "scripts/qwen_qsa").glob("*.py")),
        ROOT / "scripts/qsa/runtime.py",
    ]
    state["source_sha256"] = {}
    for path in paths:
        target = source / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        state["source_sha256"][str(path.relative_to(ROOT))] = sha(path)
    process = None
    try:
        for mode, context in [
            ("validate", None),
            ("benchmark", 4096),
            ("benchmark", 16384),
            ("benchmark", 32768),
        ]:
            assert all(
                sha(ROOT / name) == digest
                for name, digest in state["source_sha256"].items()
            )
            label = mode + ("-" + str(context) if context else "")
            command = [sys.executable, "-u", "scripts/qwen_qsa/measure_eight.py", mode]
            if context:
                command.extend(["--context", str(context)])
            stage = dict(
                label=label, command=command, started_unix=time.time(), state="running"
            )
            state["stages"].append(stage)
            save(TASK / "status.json", state)
            with (TASK / (label + ".log")).open("x", encoding="utf-8") as log:
                process = subprocess.Popen(
                    command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT
                )
                stage["pid"] = process.pid
                save(TASK / "status.json", state)
                low_since = None
                while process.poll() is None:
                    elapsed = time.time() - stage["started_unix"]
                    assert elapsed < 2400, "40 minute stage bound; retain evidence"
                    low = psutil.virtual_memory().available < 4 * 2**30
                    low_since = (low_since or time.time()) if low else None
                    assert not low_since or time.time() - low_since < 5, (
                        "Available RAM below 4GiB"
                    )
                    time.sleep(1)
            stage.update(
                exit_code=process.returncode,
                elapsed_seconds=time.time() - stage["started_unix"],
                state="complete" if process.returncode == 0 else "failed",
            )
            assert process.returncode == 0, f"Inspect {label}.log before any retry"
            save(TASK / "status.json", state)
            print(label, "complete", round(stage["elapsed_seconds"], 1), flush=True)
        state["state"] = "complete"
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
        save(TASK / "status.json", state)
        if lock.exists() and lock.read_text() == str(os.getpid()):
            lock.unlink()


if __name__ == "__main__":
    main()
