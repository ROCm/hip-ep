#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Run only the reviewed-scope Original/Eight B1 4K pilot after idle evidence."""

import json
import os
import shutil
import subprocess
import sys
import time

import psutil
from verify_handoff import ROOT, WORK, sha

TASK = WORK / "eight-qsa-clean-pilot-v1"


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
    assert time.time() - evidence["checked_unix"] < 300, (
        "Refresh the process/GPU idle check"
    )
    assert not (TASK / "status.json").exists(), "Preserve prior pilot attempts"
    assert not (ROOT / "_local/qsa/gpu-stage.lock").exists()
    state = dict(
        state="running",
        started_unix=time.time(),
        pid=os.getpid(),
        stages=[],
        scope="Original/Eight B1 4K only; two warmups,six measured requests,64 fixed decode inputs",
        accuracy_evaluations_started=False,
        longer_shapes_authorized=False,
        preflight_sha256=sha(TASK / "preflight/eligibility.json"),
    )
    source = TASK / "source"
    source.mkdir()
    paths = [
        *sorted((ROOT / "scripts/qwen_qsa").glob("*eight*.py")),
        ROOT / "scripts/qwen_qsa/native_inference.py",
        ROOT / "scripts/qwen_qsa/benchmark_native.py",
        ROOT / "scripts/qsa/runtime.py",
    ]
    frozen = {str(p.relative_to(ROOT)): sha(p) for p in paths}
    state["source_sha256"] = frozen
    for p in paths:
        dest = source / p.relative_to(ROOT)
        dest.parent.mkdir(exist_ok=True, parents=True)
        shutil.copyfile(p, dest)
    lock = ROOT / "_local/qsa/gpu-stage.lock"
    with lock.open("x") as stream:
        stream.write(str(os.getpid()))
    process = None
    stop = TASK / "stop.json"
    try:
        for variant, script in [
            ("Original", "measure_eight_control.py"),
            ("Eight", "measure_eight.py"),
        ]:
            if stop.exists():
                state["state"] = "stopped_at_request_boundary"
                break
            assert all(sha(ROOT / p) == digest for p, digest in frozen.items())
            command = [sys.executable, "-u", "scripts/qwen_qsa/" + script]
            if variant == "Eight":
                command.append("benchmark")
            command.extend(
                [
                    "--context",
                    "4096",
                    "--output-root",
                    str(TASK / variant),
                    "--stop-file",
                    str(stop),
                ]
            )
            stage = dict(
                variant=variant,
                command=command,
                started_unix=time.time(),
                state="running",
            )
            state["stages"].append(stage)
            save(TASK / "status.json", state)
            with (TASK / (variant + ".log")).open("x") as log:
                process = subprocess.Popen(
                    command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT
                )
                stage["pid"] = process.pid
                save(TASK / "status.json", state)
                low_since = None
                while process.poll() is None:
                    assert time.time() - stage["started_unix"] < 900, (
                        "15minute variant bound"
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
            summary = json.loads((TASK / variant / "4096/summary.json").read_text())
            assert summary["status"] == "complete"
            stage["state"] = "complete"
            print(variant, "complete", flush=True)
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
