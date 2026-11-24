#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Run only the user-requested short Original B1 4K/16K/32K drift control."""

import json
import os
import shutil
import subprocess
import sys
import time

import psutil
from verify_handoff import ROOT, WORK, sha

TASK = WORK / "original-short-control-v1"


def save(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def main():
    evidence = json.loads((TASK / "preflight/eligibility.json").read_text())
    assert (
        evidence["no_competing_gpu_work_observed"]
        and evidence["windows_and_wsl_inspected"]
    )
    assert time.time() - evidence["checked_unix"] < 300
    assert not (TASK / "status.json").exists(), "Preserve prior attempts"
    source = TASK / "source"
    source.mkdir()
    names = [
        "measure_eight_control.py",
        "measure_eight.py",
        "native_inference.py",
        "benchmark_native.py",
        "export_eight.py",
        "verify_handoff.py",
        "run_original_short_control.py",
    ]
    files = [ROOT / "scripts/qwen_qsa" / name for name in names] + [
        ROOT / "scripts/qsa/runtime.py"
    ]
    frozen = {str(p.relative_to(ROOT)): sha(p) for p in files}
    for p in files:
        dest = source / p.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, dest)
    lock = ROOT / "_local/qsa/gpu-stage.lock"
    with lock.open("x") as stream:
        stream.write(str(os.getpid()))
    stop = TASK / "stop.json"
    state = dict(
        state="running",
        started_unix=time.time(),
        pid=os.getpid(),
        source_sha256=frozen,
        scope="Original only; B1 4096/16384/32768; two warmups and two measured requests per shape;64 forced decode inputs",
        eight_selector_runs=False,
        accuracy_evaluations_started=False,
        stages=[],
        preflight_sha256=sha(TASK / "preflight/eligibility.json"),
    )
    process = None
    try:
        for context in [4096, 16384, 32768]:
            if stop.exists():
                state["state"] = "stopped_at_request_boundary"
                break
            assert all(sha(ROOT / p) == digest for p, digest in frozen.items())
            command = [
                sys.executable,
                "-u",
                "scripts/qwen_qsa/measure_eight_control.py",
                "--context",
                str(context),
                "--output-root",
                str(TASK),
                "--stop-file",
                str(stop),
                "--measured-requests",
                "2",
            ]
            stage = dict(
                context=context,
                command=command,
                started_unix=time.time(),
                state="running",
            )
            state["stages"].append(stage)
            save(TASK / "status.json", state)
            with (TASK / f"{context}.log").open("x") as log:
                process = subprocess.Popen(
                    command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT
                )
                stage["pid"] = process.pid
                save(TASK / "status.json", state)
                low_since = None
                while process.poll() is None:
                    assert time.time() - stage["started_unix"] < 1500, (
                        "25-minute stage bound"
                    )
                    low = psutil.virtual_memory().available < 4 * 2**30
                    low_since = (low_since or time.time()) if low else None
                    assert not low_since or time.time() - low_since < 5
                    time.sleep(1)
            stage.update(
                exit_code=process.returncode,
                elapsed_seconds=time.time() - stage["started_unix"],
            )
            assert process.returncode == 0
            if stop.exists():
                state["state"] = "stopped_at_request_boundary"
                break
            assert (
                json.loads((TASK / str(context) / "summary.json").read_text())["status"]
                == "complete"
            )
            stage["state"] = "complete"
            print("Original", context, "complete", flush=True)
        else:
            state["state"] = "complete_awaiting_review"
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
