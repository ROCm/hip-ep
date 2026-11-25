#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Run only Eight on the frozen BFCL50 cases, retaining the existing Original."""

import ctypes
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone

import psutil
from run_bfcl import kill_tree, linux_path, load, write
from verify_handoff import ROOT, WORK, sha

TASK = WORK / "bfcl-eight-v4-50"
BASELINE = WORK / "bfcl-v4-50"
os.environ["QWEN_BFCL_TASK"] = str(TASK)


class Power(ctypes.Structure):
    _fields_ = [
        ("ACLineStatus", ctypes.c_ubyte),
        ("BatteryFlag", ctypes.c_ubyte),
        ("BatteryLifePercent", ctypes.c_ubyte),
        ("SystemStatusFlag", ctypes.c_ubyte),
        ("BatteryLifeTime", ctypes.c_ulong),
        ("BatteryFullLifeTime", ctypes.c_ulong),
    ]


def main():
    TASK.mkdir(parents=True, exist_ok=True)
    home = subprocess.check_output(
        ["wsl", "-d", "Ubuntu", "--", "printenv", "HOME"], text=True
    ).strip()
    interpreter = os.environ.get(
        "QWEN_BFCL_CPU_PYTHON", home + "/.cache/hip-ep-qsa-bfcl/bin/python"
    )

    def cpu(*args):
        return [
            "wsl",
            "-d",
            "Ubuntu",
            "--",
            "env",
            "QWEN_BFCL_TASK=" + linux_path(TASK),
            "GIT_CONFIG_COUNT=0",
            interpreter,
            "-u",
            linux_path(ROOT / "scripts/qwen_qsa/bfcl_client.py"),
            *args,
        ]

    baseline_audit = load(BASELINE / "audit.json")
    assert baseline_audit["status"] == "passed" and baseline_audit[
        "rows_sha256"
    ] == sha(BASELINE / "rows.jsonl")
    for name in ["manifest.json", "initial-messages.json", "initial-profile.json"]:
        target = TASK / name
        if target.exists():
            assert target.read_bytes() == (BASELINE / name).read_bytes()
        else:
            shutil.copyfile(BASELINE / name, target)
    original = b"".join(
        line
        for line in (BASELINE / "rows.jsonl").read_bytes().splitlines(keepends=True)
        if json.loads(line)["variant"] == "Original"
    )
    assert len(original.splitlines()) == 50
    target = TASK / "original-reused.jsonl"
    if target.exists():
        assert target.read_bytes() == original
    else:
        target.write_bytes(original)
    subprocess.run(cpu("--prepare"), cwd=ROOT, check=True)
    assert sha(TASK / "manifest.json") == sha(BASELINE / "manifest.json")
    subprocess.run(
        [sys.executable, "-u", "scripts/qwen_qsa/bfcl_worker.py", "--profile"],
        cwd=ROOT,
        check=True,
    )
    sources = [
        "scripts/qwen_qsa/" + name
        for name in [
            "run_bfcl_eight.py",
            "run_bfcl.py",
            "bfcl_client.py",
            "bfcl_worker.py",
            "native_inference.py",
            "verify_handoff.py",
            "export_eight.py",
            "audit_bfcl_tokens.py",
        ]
    ] + ["scripts/qsa/bfcl_native_eval.py", "scripts/qsa/runtime.py"]
    initial = TASK / "source"
    for name in sources:
        target = initial / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copyfile(ROOT / name, target)
    checked = load(WORK / "eight-qsa-v1/validation.json")
    exported = load(WORK / "eight-qsa-v1/models/Eight/export.json")
    assert (
        checked["status"] == "passed"
        and exported["onnx_sha256"] == checked["graph_sha256"]
    )
    protocol = dict(
        benchmark="BFCL v4 frozen50-case component pilot",
        variants=["Eight"],
        model=exported["model"],
        model_revision=exported["revision"],
        bfcl_revision=load(TASK / "manifest.json")["bfcl_revision"],
        manifest_sha256=sha(TASK / "manifest.json"),
        selector_layers=exported["selector_layers"],
        selector_export_sha256=exported["selector_sha256"],
        graph_sha256=exported["onnx_sha256"],
        backbone_sha256=exported["data_sha256"],
        binary_sha256=checked["binary_sha256"],
        batch=1,
        capacity=32768,
        max_new_tokens=1024,
        thinking=False,
        decoding="greedy; both official EOS IDs; no constrained decoding",
        phase_policy="sparse prefill and sparse decode",
        initial_source_sha256={name: sha(initial / name) for name in sources},
        source_policy="Each attempt snapshots exact sources; never overwrite completed rows or raw generations",
        reused_original_task=BASELINE.relative_to(ROOT).as_posix(),
        reused_original_rows_sha256=sha(TASK / "original-reused.jsonl"),
        reused_original_audit_sha256=sha(BASELINE / "audit.json"),
        baseline_not_rerun=True,
        training_updates=0,
        official_overall_score=False,
        acceptance_gate_enabled=False,
        exclusions=["web_search", "memory", "format_sensitivity"],
        minimum_available_ram_bytes=4 * 2**30,
        maximum_attempt_seconds=18000,
        tokenizer_sha256=load(BASELINE / "protocol.json")["tokenizer_sha256"],
    )
    p = TASK / "protocol.json"
    if p.exists():
        assert load(p) == protocol
    else:
        write(p, protocol)
    lock = ROOT / "_local/qsa/gpu-stage.lock"
    with lock.open("x") as stream:
        stream.write(str(os.getpid()))
    status = dict(
        state="running",
        phase="setup",
        started_unix=time.time(),
        supervisor_pid=os.getpid(),
        variants=["Eight"],
        original_reused=True,
        protocol_sha256=sha(p),
        attempts=[],
    )
    worker = client = None
    logs = []
    try:
        for attempt in range(3):
            done = (
                [json.loads(s) for s in (TASK / "rows.jsonl").read_text().splitlines()]
                if (TASK / "rows.jsonl").exists()
                else []
            )
            if len(done) == 50:
                break
            assert len({r["id"] for r in done}) == len(done)
            if (TASK / "abort.json").exists():
                (TASK / "abort.json").rename(
                    TASK
                    / (
                        "abort-"
                        + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
                        + ".json"
                    )
                )
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            queue = TASK / "queues" / ("Eight-" + stamp)
            queue.mkdir(parents=True)
            frozen = {name: sha(ROOT / name) for name in sources}
            for name in sources:
                target = queue / "source" / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT / name, target)
            write(
                queue / "attempt.json",
                dict(
                    source_sha256=frozen,
                    protocol_sha256=sha(p),
                    variant="Eight",
                    started_unix=time.time(),
                ),
            )
            started = time.time()
            last_sample = 0
            record = dict(
                attempt=attempt + 1,
                queue=queue.relative_to(ROOT).as_posix(),
                started_unix=started,
                state="running",
            )
            status["attempts"].append(record)

            def heartbeat():
                nonlocal last_sample
                write(TASK / "lease.json", dict(unix=time.time(), pid=os.getpid()))
                assert time.time() - started < 18000, "Five-hour attempt limit"
                assert psutil.virtual_memory().available >= 4 * 2**30, "4GiB RAM guard"
                if time.time() - last_sample >= 30:
                    power = Power()
                    assert ctypes.windll.kernel32.GetSystemPowerStatus(
                        ctypes.byref(power)
                    )
                    path = TASK / "rows.jsonl"
                    count = (
                        sum(1 for _ in path.open(encoding="utf-8"))
                        if path.exists()
                        else 0
                    )
                    sample = dict(
                        unix=time.time(),
                        ACLineStatus=power.ACLineStatus,
                        available_ram_bytes=psutil.virtual_memory().available,
                        completed_cases=count,
                    )
                    with (TASK / "host-samples.jsonl").open("a") as f:
                        f.write(json.dumps(sample) + "\n")
                    status.update(
                        completed_cases=count,
                        last_heartbeat_unix=sample["unix"],
                        ACLineStatus=power.ACLineStatus,
                    )
                    write(TASK / "status.json", status)
                    last_sample = time.time()

            try:
                heartbeat()
                wlog = (queue / "worker.log").open("x")
                clog = (queue / "client.log").open("x")
                logs.extend([wlog, clog])
                worker = subprocess.Popen(
                    [
                        sys.executable,
                        "-u",
                        "scripts/qwen_qsa/bfcl_worker.py",
                        "--variant",
                        "Eight",
                        "--queue",
                        str(queue),
                    ],
                    cwd=ROOT,
                    stdout=wlog,
                    stderr=subprocess.STDOUT,
                )
                status.update(
                    phase="model_load", worker_pid=worker.pid, queue=str(queue)
                )
                write(TASK / "status.json", status)
                while not (queue / "ready.json").exists():
                    heartbeat()
                    assert worker.poll() is None, "Worker load failed"
                    assert time.time() - started < 600
                    time.sleep(0.5)
                ready = load(queue / "ready.json")
                assert (
                    ready["selector_layers"] == protocol["selector_layers"]
                    and ready["graph_sha256"] == protocol["graph_sha256"]
                )
                client = subprocess.Popen(
                    cpu("--variant", "Eight", "--queue", linux_path(queue)),
                    cwd=ROOT,
                    stdout=clog,
                    stderr=subprocess.STDOUT,
                )
                status.update(phase="generation_and_scoring", client_pid=client.pid)
                write(TASK / "status.json", status)
                while client.poll() is None:
                    heartbeat()
                    assert worker.poll() is None or (
                        worker.returncode == 0 and (queue / "stop.json").exists()
                    ), "Native worker failed"
                    time.sleep(0.5)
                assert client.returncode == 0, "Official BFCL CPU client failed"
                assert worker.wait(timeout=30) == 0
                assert load(queue / "stopped.json")["status"] == "complete"
                assert all(
                    sha(ROOT / name) == value for name, value in frozen.items()
                ), "Generation sources changed during attempt"
                record.update(state="complete", elapsed_seconds=time.time() - started)
                worker = client = None
                break
            except BaseException as exc:
                record.update(
                    state="failed",
                    error=repr(exc),
                    elapsed_seconds=time.time() - started,
                )
                write(TASK / "abort.json", record)
                write(TASK / "status.json", status)
                if worker is not None and worker.poll() is None:
                    kill_tree(worker)
                if client is not None and client.poll() is None:
                    try:
                        client.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        kill_tree(client)
                worker = client = None
                if attempt == 2:
                    raise
                print("Retrying only uncompleted cases after", repr(exc), flush=True)
        status.update(phase="independent_audit")
        write(TASK / "status.json", status)
        subprocess.run(cpu("--audit"), cwd=ROOT, check=True)
        subprocess.run(
            [sys.executable, "scripts/qwen_qsa/audit_bfcl_tokens.py"],
            cwd=ROOT,
            check=True,
        )
        status.update(
            state="generation_and_audits_complete",
            phase="awaiting_report",
            completed_cases=50,
        )
    except BaseException as exc:
        status.update(state="failed", error=repr(exc))
        write(TASK / "abort.json", status)
        raise
    finally:
        if worker is not None and worker.poll() is None:
            kill_tree(worker)
        if client is not None and client.poll() is None:
            try:
                client.wait(timeout=10)
            except subprocess.TimeoutExpired:
                kill_tree(client)
        for f in logs:
            f.close()
        status["elapsed_seconds"] = time.time() - status["started_unix"]
        write(TASK / "status.json", status)
        if lock.exists() and lock.read_text() == str(os.getpid()):
            lock.unlink()
    print(json.dumps(status, indent=2), flush=True)


if __name__ == "__main__":
    main()
