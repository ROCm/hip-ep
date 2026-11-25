#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Sample OS/process/WDDM memory beside a running benchmark without a GPU context."""

import argparse
import ctypes
import json
import os
import re
import time
from ctypes import wintypes
from datetime import datetime, timezone
from pathlib import Path

import psutil
from verify_handoff import ROOT, WORK, sha


class CounterValueUnion(ctypes.Union):
    _fields_ = [
        ("large", ctypes.c_longlong),
        ("floating", ctypes.c_double),
        ("pointer", ctypes.c_void_p),
    ]


class CounterValue(ctypes.Structure):
    _fields_ = [("status", wintypes.DWORD), ("value", CounterValueUnion)]


class CounterItem(ctypes.Structure):
    _fields_ = [("name", wintypes.LPWSTR), ("formatted", CounterValue)]


class WindowsGpuCounters:
    def __init__(self):
        self.dll = ctypes.WinDLL("pdh.dll")
        self.dll.PdhOpenQueryW.argtypes = [
            wintypes.LPCWSTR,
            ctypes.c_size_t,
            ctypes.POINTER(wintypes.HANDLE),
        ]
        self.dll.PdhAddEnglishCounterW.argtypes = [
            wintypes.HANDLE,
            wintypes.LPCWSTR,
            ctypes.c_size_t,
            ctypes.POINTER(wintypes.HANDLE),
        ]
        self.dll.PdhCollectQueryData.argtypes = [wintypes.HANDLE]
        self.dll.PdhGetFormattedCounterArrayW.argtypes = [
            wintypes.HANDLE,
            wintypes.DWORD,
            ctypes.POINTER(wintypes.DWORD),
            ctypes.POINTER(wintypes.DWORD),
            ctypes.c_void_p,
        ]
        self.dll.PdhCloseQuery.argtypes = [wintypes.HANDLE]
        for name in [
            "PdhOpenQueryW",
            "PdhAddEnglishCounterW",
            "PdhCollectQueryData",
            "PdhGetFormattedCounterArrayW",
            "PdhCloseQuery",
        ]:
            getattr(self.dll, name).restype = wintypes.DWORD
        self.query = wintypes.HANDLE()
        status = self.dll.PdhOpenQueryW(None, 0, ctypes.byref(self.query))
        if status:
            raise RuntimeError(f"PdhOpenQueryW: 0x{status:08x}")
        self.counters = {}
        for scope in ["Process", "Adapter"]:
            for field in ["Shared Usage", "Dedicated Usage", "Total Committed"]:
                handle = wintypes.HANDLE()
                path = rf"\GPU {scope} Memory(*)\{field}"
                status = self.dll.PdhAddEnglishCounterW(
                    self.query, path, 0, ctypes.byref(handle)
                )
                if status:
                    self.close()
                    raise RuntimeError(f"{path}: 0x{status:08x}")
                self.counters[(scope.lower(), field.lower().replace(" ", "_"))] = handle

    def close(self):
        if self.query:
            self.dll.PdhCloseQuery(self.query)
            self.query = None

    def sample(self, pids):
        status = self.dll.PdhCollectQueryData(self.query)
        if status:
            return {"error": f"PdhCollectQueryData: 0x{status:08x}"}
        instances, errors = {"process": {}, "adapter": {}}, {}
        for (scope, field), handle in self.counters.items():
            size, count = wintypes.DWORD(), wintypes.DWORD()
            status = self.dll.PdhGetFormattedCounterArrayW(
                handle, 0x400, ctypes.byref(size), ctypes.byref(count), None
            )
            if status != 0x800007D2 or not size.value:
                errors[f"{scope}.{field}"] = f"0x{status:08x}"
                continue
            buffer = ctypes.create_string_buffer(size.value)
            status = self.dll.PdhGetFormattedCounterArrayW(
                handle, 0x400, ctypes.byref(size), ctypes.byref(count), buffer
            )
            if status:
                errors[f"{scope}.{field}"] = f"0x{status:08x}"
                continue
            items = ctypes.cast(buffer, ctypes.POINTER(CounterItem))
            for index in range(count.value):
                item = items[index]
                if item.formatted.status not in [0, 1]:
                    continue
                name = item.name
                if scope == "process":
                    match = re.match(r"pid_(\d+)_(.*)", name)
                    if not match or int(match[1]) not in pids:
                        continue
                    record = instances[scope].setdefault(
                        name, {"pid": int(match[1]), "adapter_instance": match[2]}
                    )
                else:
                    record = instances[scope].setdefault(name, {})
                record[field + "_bytes"] = item.formatted.value.large
        adapters = {row["adapter_instance"] for row in instances["process"].values()}
        instances["adapter"] = {
            name: row for name, row in instances["adapter"].items() if name in adapters
        }
        return {**instances, "errors": errors}


def current_stage(labels):
    receipts = []
    for label in labels:
        path = ROOT / "_local/qsa/stages" / label / "status.json"
        if path.exists():
            try:
                receipts.append((label, json.loads(path.read_text())))
            except (OSError, json.JSONDecodeError):
                continue
    active = [
        (label, row)
        for label, row in receipts
        if row["state"] == "running" and row.get("pid")
    ]
    return (
        max(active, key=lambda item: item[1]["started_unix"])
        if active
        else (None, None)
    )


def processes(pid):
    result = []
    if pid is None:
        return result
    try:
        parent = psutil.Process(pid)
        tree = [parent, *parent.children(recursive=True)]
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return result
    for process in tree:
        try:
            info = process.memory_info()
            result.append(
                {
                    "pid": process.pid,
                    "name": process.name(),
                    "rss_bytes": info.rss,
                    "private_commit_bytes": info.private,
                    "lifetime_peak_rss_bytes": info.peak_wset,
                    "lifetime_peak_private_commit_bytes": info.peak_pagefile,
                }
            )
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=WORK / "memory-v1")
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--contexts", nargs="+", type=int, default=[32768, 65536])
    parser.add_argument("--stop-context", type=int, default=65536)
    args = parser.parse_args()
    assert os.name == "nt" and args.interval >= 1
    args.output.mkdir(parents=True, exist_ok=True)
    assert not (args.output / "protocol.json").exists(), (
        "Preserve previous memory traces"
    )
    labels = [f"qwen-native-perf-{context}-b1" for context in args.contexts]
    started = time.time()
    existing = {}
    for label in labels:
        path = ROOT / "_local/qsa/stages" / label / "status.json"
        existing[label] = json.loads(path.read_text()) if path.exists() else None
    protocol = {
        "started_unix": started,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "interval_seconds": args.interval,
        "labels": labels,
        "sampler_pid": os.getpid(),
        "sampler_sha256": sha(Path(__file__)),
        "gpu_context_created": False,
        "source": "psutil process/system counters and native Windows PDH GPU memory gauges",
        "scope": "Joint benchmark worker process: Original and Changed sessions co-resident with serially shared KV",
        "notes": [
            "RSS and private commitment are distinct; WDDM gauges are driver-reported usage/commitment, not a proven physical residency total.",
            "Do not add process RSS, WDDM dedicated/shared usage, and system RAM: these can overlap.",
            "Sampled GPU peaks can miss sub-second spikes. OS process lifetime high-water counters are recorded separately.",
            "32K sampling begins mid-run; 64K sampler is attached before stage startup. Earlier contexts are not reconstructed.",
            "External CPU-only sampling adds no CUDA/HIP context or benchmark allocations. It starts partway through 32K and remains active throughout 64K.",
        ],
        "stage_state_at_start": {
            name: {key: row.get(key) for key in ["state", "started_unix", "pid"]}
            if row
            else None
            for name, row in existing.items()
        },
    }
    (args.output / "sampler.py").write_bytes(Path(__file__).read_bytes())
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    counters = WindowsGpuCounters()
    stop_label = f"qwen-native-perf-{args.stop_context}-b1"
    stop_path = ROOT / "_local/qsa/stages" / stop_label / "status.json"
    next_tick, samples, last_label = time.monotonic(), 0, None
    print("Memory sampler active", protocol["started_utc"], labels, flush=True)
    try:
        with (args.output / "samples.jsonl").open("x", encoding="utf-8") as stream:
            while True:
                label, status = current_stage(labels)
                workers = processes(status["pid"] if status else None)
                system = psutil.virtual_memory()
                frame = {
                    "unix": time.time(),
                    "stage": label,
                    "processes": workers,
                    "system_ram_total_bytes": system.total,
                    "system_ram_available_bytes": system.available,
                    "system_ram_used_bytes": system.used,
                    "rss_tree_sum_bytes": sum(row["rss_bytes"] for row in workers),
                    "private_commit_tree_sum_bytes": sum(
                        row["private_commit_bytes"] for row in workers
                    ),
                    "wddm": counters.sample({row["pid"] for row in workers}),
                }
                if label:
                    context = int(label.split("-")[-2])
                    rows = (
                        WORK
                        / "native-gqa/paired-performance"
                        / f"{context}-b1/rows.jsonl"
                    )
                    if rows.exists():
                        lines = rows.read_text().splitlines()
                        if lines:
                            frame["last_completed_measured_request"] = {
                                "row": json.loads(lines[-1]),
                                "rows_file_mtime_unix": rows.stat().st_mtime,
                                "completed_rows": len(lines),
                            }
                stream.write(json.dumps(frame, allow_nan=False) + "\n")
                stream.flush()
                (args.output / "latest.json").write_text(
                    json.dumps(frame, indent=2) + "\n"
                )
                samples += 1
                if label != last_label:
                    print(
                        "Tracking",
                        label,
                        "PIDs",
                        [row["pid"] for row in workers],
                        flush=True,
                    )
                    last_label = label
                if stop_path.exists():
                    try:
                        if json.loads(stop_path.read_text())["state"] in [
                            "complete",
                            "failed",
                        ]:
                            break
                    except (OSError, json.JSONDecodeError):
                        pass
                failed = []
                for name in labels:
                    path = ROOT / "_local/qsa/stages" / name / "status.json"
                    try:
                        if (
                            path.exists()
                            and json.loads(path.read_text())["state"] == "failed"
                        ):
                            failed.append(name)
                    except (OSError, json.JSONDecodeError):
                        pass
                if failed and not label:
                    break
                next_tick += args.interval
                time.sleep(max(0, next_tick - time.monotonic()))
    finally:
        counters.close()
        (args.output / "status.json").write_text(
            json.dumps(
                {
                    "state": "stopped",
                    "samples": samples,
                    "started_unix": started,
                    "ended_unix": time.time(),
                },
                indent=2,
            )
            + "\n"
        )


if __name__ == "__main__":
    main()
