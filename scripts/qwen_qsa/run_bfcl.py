#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Supervise the paired 50-case BFCL pilot, with one resident Qwen variant."""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import psutil
from verify_handoff import ROOT, WORK, sha

TASK = WORK / "bfcl-v4-50"
CPU_PYTHON = os.environ.get("QWEN_BFCL_CPU_PYTHON")


def load(path):
    deadline = time.monotonic() + 10
    while True:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (PermissionError, FileNotFoundError):
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.05)


def write(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    deadline = time.monotonic() + 10
    while True:
        try:
            temporary.replace(path)
            return
        except PermissionError:
            assert time.monotonic() < deadline
            time.sleep(0.05)


def linux_path(path):
    path = path.resolve()
    return "/mnt/" + path.drive[0].lower() + path.as_posix()[2:]


def cpu_command(*args):
    global CPU_PYTHON
    if not CPU_PYTHON:
        cpu_home = subprocess.check_output(
            ["wsl", "-d", "Ubuntu", "--", "printenv", "HOME"], text=True
        ).strip()
        assert cpu_home.startswith("/"), (
            "Set QWEN_BFCL_CPU_PYTHON to the WSL CPU interpreter"
        )
        CPU_PYTHON = cpu_home.rstrip("/") + "/.cache/hip-ep-qsa-bfcl/bin/python"
    return [
        "wsl",
        "-d",
        "Ubuntu",
        "--",
        CPU_PYTHON,
        "-u",
        linux_path(ROOT / "scripts/qwen_qsa/bfcl_client.py"),
        *args,
    ]


def kill_tree(process):
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
    process.wait(timeout=30)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    TASK.mkdir(parents=True, exist_ok=True)
    subprocess.run(cpu_command("--prepare"), cwd=ROOT, check=True)
    subprocess.run(
        [sys.executable, "-u", "scripts/qwen_qsa/bfcl_worker.py", "--profile"],
        cwd=ROOT,
        check=True,
    )
    profile = load(TASK / "initial-profile.json")["cases"]
    print(
        "Initial prompt range:",
        min(c["initial_prompt_tokens"] for c in profile),
        max(c["initial_prompt_tokens"] for c in profile),
        flush=True,
    )
    if args.prepare_only:
        return
    if (TASK / "abort.json").exists():
        assert args.resume, "Investigate the prior aborted attempt, then use --resume"
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
        (TASK / "abort.json").rename(TASK / ("abort-" + stamp + ".json"))
    lock = ROOT / "_local/qsa/gpu-stage.lock"
    started = time.time()
    source_paths = [
        "scripts/qwen_qsa/bfcl_client.py",
        "scripts/qwen_qsa/bfcl_worker.py",
        "scripts/qwen_qsa/run_bfcl.py",
        "scripts/qwen_qsa/native_inference.py",
        "scripts/qwen_qsa/report_bfcl.py",
        "scripts/qwen_qsa/verify_handoff.py",
        "scripts/qsa/bfcl_native_eval.py",
        "scripts/qsa/runtime.py",
    ]
    sources = {name: sha(ROOT / name) for name in source_paths}
    snapshots = TASK / "source"
    snapshots.mkdir(exist_ok=True)
    for name in source_paths:
        target = snapshots / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.write_bytes((ROOT / name).read_bytes())
    verified = load(WORK / "handoff-verification.json")
    snapshot = Path(verified["backbone"]["snapshot"])
    validation = load(WORK / "native-gqa/full-model-validation.json")
    protocol = dict(
        benchmark="BFCL v4 50-case component pilot",
        model=verified["model"],
        model_revision=verified["revision"],
        bfcl_revision=load(TASK / "manifest.json")["bfcl_revision"],
        manifest_sha256=sha(TASK / "manifest.json"),
        batch=1,
        capacity=32768,
        max_new_tokens=1024,
        thinking=False,
        decoding="greedy; official EOS IDs; no constrained decoding",
        prompt_mode="Official BFCL prompting system instructions with actual Qwen chat template; Python-call parser",
        variant_schedule="All Original cases, then all Changed; one resident session at a time; accuracy only",
        variants=["Original", "Changed"],
        training_updates=0,
        initial_source_sha256={name: sha(snapshots / name) for name in source_paths},
        source_policy="Initial sources are preserved; each queue has an immutable attempt.json and source snapshot. Debugging resumes never overwrite completed model outputs.",
        binary_sha256=validation["binary_sha256"],
        frozen_selector_sha256=verified["selector_sha256"],
        tokenizer_sha256={
            name: sha(snapshot / name)
            for name in [
                "tokenizer.json",
                "tokenizer_config.json",
                "chat_template.jinja",
                "generation_config.json",
            ]
        },
        exports={
            v: load(WORK / "native-gqa/models" / v / "export.json")
            for v in ["Original", "Changed"]
        },
        official_overall_score=False,
        exclusions=["web_search", "memory", "format_sensitivity"],
        time_limit_seconds=14400,
        minimum_available_system_ram_bytes=4 * 2**30,
        sparse_first_query_position=2052,
        token_accounting="Prompt includes chat template and tool schema/history; completion includes generated EOS. Repeated histories count once per generation call.",
    )
    protocol_path = TASK / "protocol.json"
    if protocol_path.exists():
        assert load(protocol_path) == protocol
    else:
        write(protocol_path, protocol)
    status = dict(
        state="running",
        started_unix=started,
        supervisor_pid=os.getpid(),
        protocol_sha256=sha(protocol_path),
        variants_completed=[],
    )
    worker, client, logs = None, None, []
    with lock.open("x") as stream:
        stream.write(str(os.getpid()))
    try:

        def heartbeat():
            write(TASK / "lease.json", {"unix": time.time(), "pid": os.getpid()})
            assert time.time() - started < 14400, (
                "Four-hour pilot limit; retain partial evidence"
            )
            assert psutil.virtual_memory().available >= 4 * 2**30, (
                "4 GiB available RAM guard"
            )

        heartbeat()
        for variant in ["Original", "Changed"]:
            assert all(
                sha(ROOT / name) == expected for name, expected in sources.items()
            )
            rows = (
                [json.loads(s) for s in (TASK / "rows.jsonl").read_text().splitlines()]
                if (TASK / "rows.jsonl").exists()
                else []
            )
            if sum(row["variant"] == variant for row in rows) == 50:
                status["variants_completed"].append(variant)
                continue
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
            queue = TASK / "queues" / (variant + "-" + stamp)
            queue.mkdir(parents=True, exist_ok=False)
            for name in source_paths:
                target = queue / "source" / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes((ROOT / name).read_bytes())
            write(
                queue / "attempt.json",
                dict(
                    source_sha256=sources,
                    protocol_sha256=sha(protocol_path),
                    variant=variant,
                    started_unix=time.time(),
                ),
            )
            status.update(variant=variant, queue=str(queue), phase="model_load")
            write(TASK / "status.json", status)
            worker_log = (queue / "worker.log").open("w", encoding="utf-8")
            client_log = (queue / "client.log").open("w", encoding="utf-8")
            logs.extend([worker_log, client_log])
            worker = subprocess.Popen(
                [
                    sys.executable,
                    "-u",
                    "scripts/qwen_qsa/bfcl_worker.py",
                    "--variant",
                    variant,
                    "--queue",
                    str(queue),
                ],
                cwd=ROOT,
                stdout=worker_log,
                stderr=subprocess.STDOUT,
            )
            loaded = time.monotonic()
            while not (queue / "ready.json").exists():
                heartbeat()
                assert worker.poll() is None, (
                    "Worker failed to load; inspect worker.log"
                )
                assert time.monotonic() - loaded < 600, "Model load timeout"
                time.sleep(0.5)
            client = subprocess.Popen(
                cpu_command("--variant", variant, "--queue", linux_path(queue)),
                cwd=ROOT,
                stdout=client_log,
                stderr=subprocess.STDOUT,
            )
            status.update(
                phase="generation_and_scoring",
                worker_pid=worker.pid,
                client_pid=client.pid,
            )
            write(TASK / "status.json", status)
            while client.poll() is None:
                heartbeat()
                assert worker.poll() is None or (
                    worker.returncode == 0 and (queue / "stop.json").exists()
                ), "Native worker failed"
                time.sleep(0.5)
            assert client.returncode == 0, (
                "Official BFCL CPU client failed; inspect client.log"
            )
            assert worker.wait(timeout=30) == 0
            assert load(queue / "stopped.json")["status"] == "complete"
            worker, client = None, None
            status["variants_completed"].append(variant)
            write(TASK / "status.json", status)
        subprocess.run(cpu_command("--audit"), cwd=ROOT, check=True)
        subprocess.run(
            [sys.executable, "scripts/qwen_qsa/report_bfcl.py"], cwd=ROOT, check=True
        )
        status.update(state="complete", phase="audited_and_reported")
    except BaseException as exc:
        status.update(state="failed", error=repr(exc))
        write(TASK / "abort.json", status)
        raise
    finally:
        # Abort/lease files let the WSL client exit itself even if its Windows
        # launch process disappears; never terminate unrelated WSL workloads.
        if worker is not None and worker.poll() is None:
            kill_tree(worker)
        if client is not None and client.poll() is None:
            try:
                client.wait(timeout=10)
            except subprocess.TimeoutExpired:
                kill_tree(client)
        for log in logs:
            log.close()
        status["elapsed_seconds"] = time.time() - started
        write(TASK / "status.json", status)
        if lock.exists() and lock.read_text() == str(os.getpid()):
            lock.unlink()
        print(json.dumps(status, indent=2), flush=True)


if __name__ == "__main__":
    main()
