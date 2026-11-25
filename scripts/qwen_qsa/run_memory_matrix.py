#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Run memory-only Original/QSA probes, one fresh model process per configuration."""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import psutil
from sample_memory import WindowsGpuCounters, processes
from verify_handoff import ROOT, WORK, sha


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def terminate_worker(process):
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


def summarize(frames):
    def peak(field, rows):
        values = [row[field] for row in rows if row.get(field) is not None]
        return max(values) if values else None

    groups = {"overall": frames}
    for phase in sorted({row["phase"] for row in frames}):
        groups[phase] = [row for row in frames if row["phase"] == phase]
    return {
        phase: {
            "samples": len(rows),
            "peak_rss_bytes": peak("rss_bytes", rows),
            "peak_private_commit_bytes": peak("private_commit_bytes", rows),
            "worker_lifetime_peak_rss_bytes": peak(
                "worker_lifetime_peak_rss_bytes", rows
            ),
            "worker_lifetime_peak_private_commit_bytes": peak(
                "worker_lifetime_peak_private_commit_bytes", rows
            ),
            "peak_gpu_dedicated_bytes": peak("gpu_dedicated_bytes", rows),
            "peak_gpu_shared_bytes": peak("gpu_shared_bytes", rows),
            "peak_gpu_committed_bytes": peak("gpu_committed_bytes", rows),
            "minimum_system_ram_available_bytes": min(
                row["system_ram_available_bytes"] for row in rows
            ),
        }
        for phase, rows in groups.items()
        if rows
    }


def write_report(folder, cases):
    matrix = {
        "purpose": "Memory-only, one isolated session per worker",
        "cases": cases,
        "contexts": [4096, 16384, 32768],
        "batches": [1, 4, 8],
        "excluded": "64K excluded at user request after the co-resident timing run hit the RAM guard",
        "sample_interval_seconds": 0.1,
        "prefill_calls_per_probe": 1,
        "decode_calls_per_probe": 8,
        "capacity": "prefix + 64 slots, matching earlier timing cache capacity",
        "notes": [
            "No performance warmup or repeated timing trials; no throughput result is produced.",
            "Original and QSA run in fresh separate processes, one model at a time.",
            "WDDM gauges are driver-reported usage/commitment, not proven physical residency; do not add them to RSS/system RAM.",
            "GPU peaks are sampled at 100ms. OS worker lifetime high-water counters are retained separately.",
            "Incomplete/capacity-limited probes retain observed lower bounds but are not successful configuration peaks.",
        ],
    }
    (folder / "matrix.json").write_text(json.dumps(matrix, indent=2) + "\n")
    lines = [
        "<!--",
        "Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.",
        "Licensed under the MIT License.",
        "-->",
        "# Qwen native memory-only matrix",
        "",
        "One fresh hip-ep session per process, default and frozen QSA measured separately. Each probe runs one full prefill and eight fixed decode inputs, reserving prefix+64 KV slots. No repeated performance benchmark is run. 64K is excluded at the user's request.",
        "",
        "Numbers are GiB. GPU columns are peak Windows WDDM process **total commitment**, not physical residency. Process RSS is shown separately and must not be added to GPU commitment. Sampled GPU peaks can miss sub-100ms spikes; OS lifetime process high-water values and all phase traces are retained.",
        "",
        "| Batch | Prefix | Default status | QSA status | Default GPU commit | QSA GPU commit | Default process RSS | QSA process RSS |",
        "|---:|---:|---|---|---:|---:|---:|---:|",
    ]

    def cell(case, field):
        if not case:
            return "—"
        value = case["memory"]["overall"].get(field)
        suffix = "" if case["status"] == "complete" else "†"
        return f"{value / 2**30:.2f}{suffix}" if value is not None else "Unavailable"

    for context in [4096, 16384, 32768]:
        for batch in [1, 4, 8]:
            rows = {
                variant: next(
                    (
                        row
                        for row in cases
                        if row["variant"] == variant
                        and row["context"] == context
                        and row["batch"] == batch
                    ),
                    None,
                )
                for variant in ["Original", "Changed"]
            }
            a, b = rows["Original"], rows["Changed"]
            lines.append(
                f"| {batch} | {context // 1024}K | {a['status'] if a else 'Pending'} | {b['status'] if b else 'Pending'} | {cell(a, 'peak_gpu_committed_bytes')} | {cell(b, 'peak_gpu_committed_bytes')} | {cell(a, 'peak_rss_bytes')} | {cell(b, 'peak_rss_bytes')} |"
            )
    lines += [
        "",
        "A `capacity_limit` case was stopped promptly by the 4 GiB available-system-RAM guard. † marks values observed before an incomplete probe stopped; these are lower bounds, not completed configuration peaks. Raw samples, load/prefill/decode events, worker results, errors and identities are retained under `_local/qwen-qsa/memory-only-v1/`. Per-phase peaks and OS lifetime high-water counters are in `matrix.json`.",
        "",
    ]
    (ROOT / "docs/experiments/qwen-qsa-memory.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--contexts",
        nargs="+",
        type=int,
        choices=[4096, 16384, 32768],
        default=[4096, 16384, 32768],
    )
    parser.add_argument(
        "--batches", nargs="+", type=int, choices=[1, 4, 8], default=[1, 4, 8]
    )
    args = parser.parse_args()
    folder = WORK / "memory-only-v1"
    folder.mkdir(parents=True, exist_ok=True)
    matrix_path = folder / "matrix.json"
    cases = read_json(matrix_path)["cases"] if matrix_path.exists() else []
    lock = ROOT / "_local/qsa/gpu-stage.lock"
    with lock.open("x") as stream:
        stream.write(str(os.getpid()))
    try:
        counters = WindowsGpuCounters()
    except BaseException:
        lock.unlink()
        raise
    scripts = Path(__file__).parent
    sources = {
        name: sha(scripts / name)
        for name in [
            "memory_probe_worker.py",
            "run_memory_matrix.py",
            "sample_memory.py",
            "native_inference.py",
        ]
    }
    try:
        for context in args.contexts:
            for batch in args.batches:
                for variant in ["Original", "Changed"]:
                    assert all(
                        sha(scripts / name) == expected
                        for name, expected in sources.items()
                    )
                    existing = next(
                        (
                            row
                            for row in cases
                            if row["variant"] == variant
                            and row["context"] == context
                            and row["batch"] == batch
                        ),
                        None,
                    )
                    if existing:
                        assert existing["status"] in [
                            "complete",
                            "capacity_limit",
                            "allocation_failure",
                        ], "Investigate retained failed probe before resuming"
                        continue
                    case_folder = folder / f"b{batch}-s{context}-{variant}"
                    case_folder.mkdir(exist_ok=False)
                    (case_folder / "source").mkdir()
                    for name in sources:
                        (case_folder / "source" / name).write_bytes(
                            (scripts / name).read_bytes()
                        )
                    command = [
                        sys.executable,
                        "-u",
                        "scripts/qwen_qsa/memory_probe_worker.py",
                        "--variant",
                        variant,
                        "--context",
                        str(context),
                        "--batch",
                        str(batch),
                        "--output",
                        str(case_folder),
                    ]
                    started = time.time()
                    protocol = {
                        "command": command,
                        "source_sha256": sources,
                        "started_unix": started,
                        "interval_seconds": 0.1,
                        "minimum_ram_available_bytes": 4 * 2**30,
                        "time_limit_seconds": 1800,
                        "scope": "One isolated native session; no co-resident comparison model",
                    }
                    (case_folder / "protocol.json").write_text(
                        json.dumps(protocol, indent=2) + "\n"
                    )
                    print("Starting memory-only", batch, context, variant, flush=True)
                    frames, reason = [], None
                    with (
                        (case_folder / "worker.log").open("x", encoding="utf-8") as log,
                        (case_folder / "samples.jsonl").open(
                            "x", encoding="utf-8"
                        ) as trace,
                    ):
                        process = subprocess.Popen(
                            command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT
                        )
                        try:
                            while process.poll() is None:
                                workers = processes(process.pid)
                                phase = read_json(case_folder / "phase.json") or {
                                    "phase": "process_start"
                                }
                                ram = psutil.virtual_memory()
                                gpu = counters.sample(
                                    {worker["pid"] for worker in workers}
                                )

                                def gpu_sum(field):
                                    values = [
                                        row[field]
                                        for row in gpu.get("process", {}).values()
                                        if field in row
                                    ]
                                    return sum(values) if values else None

                                frame = {
                                    "unix": time.time(),
                                    "phase": phase["phase"],
                                    "processes": workers,
                                    "rss_bytes": sum(
                                        row["rss_bytes"] for row in workers
                                    ),
                                    "private_commit_bytes": sum(
                                        row["private_commit_bytes"] for row in workers
                                    ),
                                    "worker_lifetime_peak_rss_bytes": max(
                                        (
                                            row["lifetime_peak_rss_bytes"]
                                            for row in workers
                                        ),
                                        default=0,
                                    ),
                                    "worker_lifetime_peak_private_commit_bytes": max(
                                        (
                                            row["lifetime_peak_private_commit_bytes"]
                                            for row in workers
                                        ),
                                        default=0,
                                    ),
                                    "system_ram_available_bytes": ram.available,
                                    "gpu_dedicated_bytes": gpu_sum(
                                        "dedicated_usage_bytes"
                                    ),
                                    "gpu_shared_bytes": gpu_sum("shared_usage_bytes"),
                                    "gpu_committed_bytes": gpu_sum(
                                        "total_committed_bytes"
                                    ),
                                    "wddm": gpu,
                                }
                                frames.append(frame)
                                trace.write(json.dumps(frame, allow_nan=False) + "\n")
                                trace.flush()
                                (folder / "latest.json").write_text(
                                    json.dumps(
                                        {
                                            "variant": variant,
                                            "batch": batch,
                                            "context": context,
                                            **frame,
                                        }
                                    )
                                    + "\n"
                                )
                                if ram.available < 4 * 2**30:
                                    reason = "capacity_limit"
                                    terminate_worker(process)
                                    break
                                if time.time() - started > 1800:
                                    reason = "time_limit"
                                    terminate_worker(process)
                                    break
                                time.sleep(0.1)
                        except BaseException:
                            terminate_worker(process)
                            raise
                    worker_result = read_json(case_folder / "worker-result.json")
                    state = reason or (
                        "complete"
                        if process.returncode == 0
                        and worker_result
                        and worker_result["status"] == "complete"
                        else "failed"
                    )
                    error = worker_result.get("error", "") if worker_result else ""
                    if state == "failed" and any(
                        word in error.lower()
                        for word in [
                            "out of memory",
                            "outofmemory",
                            "allocation failed",
                        ]
                    ):
                        state = "allocation_failure"
                    result = {
                        "variant": variant,
                        "batch": batch,
                        "context": context,
                        "status": state,
                        "returncode": process.returncode,
                        "elapsed_seconds": time.time() - started,
                        "error": error,
                        "memory": summarize(frames),
                        "worker_result": worker_result,
                        "protocol_sha256": sha(case_folder / "protocol.json"),
                        "samples_sha256": sha(case_folder / "samples.jsonl"),
                        "source_sha256": sources,
                    }
                    (case_folder / "result.json").write_text(
                        json.dumps(result, indent=2) + "\n"
                    )
                    cases.append(result)
                    write_report(folder, cases)
                    print(
                        "Completed memory-only",
                        batch,
                        context,
                        variant,
                        state,
                        flush=True,
                    )
                    assert state in [
                        "complete",
                        "capacity_limit",
                        "allocation_failure",
                    ], f"Unexpected worker failure; inspect {case_folder}"
                    time.sleep(2 if state == "complete" else 5)
        write_report(folder, cases)
        print("Memory-only matrix complete", len(cases), "cases", flush=True)
    finally:
        counters.close()
        lock.unlink()


if __name__ == "__main__":
    main()
