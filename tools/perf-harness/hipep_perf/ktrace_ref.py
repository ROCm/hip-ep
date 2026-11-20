#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Read a ktrace capture as a ``trace.Trace``, to use ktrace as a reference.

ktrace (an independent, injected tracer on the same HIP runtime callback)
writes rocprofv3-shaped CSVs; loading them into the same structure lets one
attribution and summary path run over both, so differences are the tracers',
not the post-processing's. Timestamps in both are QPC nanoseconds.
"""

from __future__ import annotations

import csv
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from . import ring, trace

# ktrace cuts kernel names to this many characters (long Tensile names).
KTRACE_NAME_MAX = 255

_DIRECTION_KIND = {
    "MEMORY_COPY_DEVICE_TO_HOST": 0x11F3,
    "MEMORY_COPY_HOST_TO_DEVICE": 0x11F4,
    "MEMORY_COPY_DEVICE_TO_DEVICE": 0x11F5,
}


def _rows(path: Path):
    if not path.exists():
        return []
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def load(out_dir: Path, prefix: str) -> trace.Trace:
    out_dir = Path(out_dir)
    names: dict[int, str] = {}
    ids: dict[str, int] = {}

    def nid(s: str) -> int:
        if s not in ids:
            ids[s] = len(ids) + 1
            names[ids[s]] = s
        return ids[s]

    # seq keeps ktrace's dispatch order (Dispatch_Id), and the copies' file
    # order after all dispatches; the CSVs do not say how the two interleave.
    ops = []
    for r in _rows(out_dir / f"{prefix}_kernel_trace.csv"):
        ops.append(
            (
                int(r["Correlation_Id"]),
                int(r["Start_Timestamp"]),
                int(r["End_Timestamp"]),
                int(r["Queue_Id"]),
                0x11F0,
                trace.OP_DISPATCH,
                0,
                nid(r["Kernel_Name"]),
                int(r["Thread_Id"] or 0),
                int(r["Dispatch_Id"]),
            )
        )
    copy_base = 1 << 40
    for k, r in enumerate(_rows(out_dir / f"{prefix}_memory_copy_trace.csv")):
        ops.append(
            (
                int(r["Correlation_Id"]),
                int(r["Start_Timestamp"]),
                int(r["End_Timestamp"]),
                0,
                _DIRECTION_KIND.get(r["Direction"], 0),
                trace.OP_COPY,
                0,
                0,
                0,
                copy_base + k,
            )
        )
    o = np.array(ops, dtype=trace._OPS) if ops else np.empty(0, dtype=trace._OPS)

    api_ids: dict[str, int] = {}
    api = []
    pid = 0
    for r in _rows(out_dir / f"{prefix}_hip_api_trace.csv"):
        pid = pid or int(r["Process_Id"])
        fn = r["Function"]
        if fn not in api_ids:
            api_ids[fn] = len(api_ids) + 1
            names[ring.API_NAME_BASE | api_ids[fn]] = fn
        api.append(
            (
                int(r["Correlation_Id"]),
                int(r["Start_Timestamp"]),
                int(r["End_Timestamp"]),
                api_ids[fn],
                int(r["Thread_Id"]),
            )
        )
    a = np.array(api, dtype=trace._API) if api else np.empty(0, dtype=trace._API)

    roctx_rows = _rows(out_dir / f"{prefix}_roctx_trace.csv")
    if not pid and roctx_rows:
        pid = int(roctx_rows[0]["Process_Id"])
    tr = trace.Trace(
        meta={"source": "ktrace", "dropped": 0, "producer_pid": pid},
        names=names,
        ops=np.sort(o, order="begin"),
        api=a,
    )
    tr.device = _device(out_dir / f"{prefix}_agent_info.csv")
    per_tid: dict[int, list] = defaultdict(list)
    for r in roctx_rows:
        ev = (int(r["Timestamp"]), int(r["Thread_Id"]), r["Operation"], r["Message"])
        tr.roctx.append(ev)
        per_tid[ev[1]].append((ev[0], ev[2], ev[3]))
    last = int(max(o["end"].max() if len(o) else 0, a["exit"].max() if len(a) else 0))
    for tid, evs in per_tid.items():
        stack: list[int] = []
        for ts, op, msg in evs:  # file order is the thread's program order
            if op == "RANGE_PUSH":
                parent = stack[-1] if stack else -1
                tr.ranges.append(trace.Range(tid, ts, -1, msg, len(stack), parent))
                stack.append(len(tr.ranges) - 1)
            elif op == "RANGE_POP":
                if stack:
                    tr.ranges[stack.pop()].end = ts
                else:
                    tr.unbalanced_pops += 1
            elif op == "MARK":
                tr.marks.append((ts, tid, msg))
    for rg in tr.ranges:
        if rg.end < 0:
            rg.end = max(last, rg.start)
    return tr


# ktrace's command-kind names (its tracer's table) -> cl_command_type.
_KIND = {
    "KernelExecution": 0x11F0, "Task": 0x11F1, "NativeKernel": 0x11F2,
    "CopyDeviceToHost": 0x11F3, "CopyHostToDevice": 0x11F4,
    "CopyDeviceToDevice": 0x11F5, "ReadImage": 0x11F6, "WriteImage": 0x11F7,
    "CopyImage": 0x11F8, "CopyImageToBuffer": 0x11F9, "CopyBufferToImage": 0x11FA,
    "MapBuffer": 0x11FB, "MapImage": 0x11FC, "UnmapMemObject": 0x11FD,
    "Marker": 0x11FE, "AcquireGLObjects": 0x11FF, "ReleaseGLObjects": 0x1200,
    "CopyDeviceToHost2D": 0x1201, "CopyHostToDevice2D": 0x1202,
    "CopyDeviceToDevice2D": 0x1203, "User": 0x1204, "Barrier": 0x1205,
    "MigrateMemObjects": 0x1206, "FillBuffer": 0x1207, "FillImage": 0x1208,
    "SvmFree": 0x1209, "SvmMemcpy": 0x120A, "SvmMemFill": 0x120B,
    "SvmMap": 0x120C, "SvmUnmap": 0x120D,
}  # fmt: skip
_OP = {
    "dispatch": trace.OP_DISPATCH,
    "copy": trace.OP_COPY,
    "barrier": trace.OP_BARRIER,
}


def _ns(us: str) -> int:
    """ktrace's Nanoseconds(): std::round(double(us) * 1000), half away from 0."""
    x = float(us) * 1000.0
    f = math.floor(x)
    r = f + 1 if x - f >= 0.5 else f
    return max(0, int(r))


def load_spool(ops_csv: Path) -> trace.Trace:
    """A ktrace capture from its raw spool, the input of ktrace's C++ finalize.

    Unlike the finalized CSVs, the spool keeps every command (fills too) in
    the order the runtime reported them, so exports that depend on that order
    can be compared with ktrace's on the same input.
    """
    ops_csv = Path(ops_csv)
    side = {k: Path(f"{ops_csv}.{k}") for k in ("api.csv", "roctx.csv", "launch.csv")}
    names: dict[int, str] = {}
    ids: dict[str, int] = {}

    def nid(s: str) -> int:
        if s not in ids:
            ids[s] = len(ids) + 1
            names[ids[s]] = s
        return ids[s]

    need = ("Op", "Kind", "Bytes", "Device", "Queue", "Corr", "Begin_us", "End_us")
    ops = []
    for k, r in enumerate(_rows(ops_csv)):
        if not all(r.get(c) for c in need):
            continue
        dispatch = r["Op"] == "dispatch"
        ops.append(
            (
                int(r["Corr"]),
                _ns(r["Begin_us"]),
                _ns(r["End_us"]),
                int(r["Queue"]),
                _KIND.get(r["Kind"], 0),
                _OP.get(r["Op"], trace.OP_BARRIER),
                int(r["Device"]),
                nid(r["Kernel_Name"]) if dispatch else int(r["Bytes"]),
                0,
                k,
            )
        )
    o = np.array(ops, dtype=trace._OPS) if ops else np.empty(0, dtype=trace._OPS)
    api = []
    args: dict[int, str] = {}
    need = ("Corr", "Api", "Api_Id", "Thread", "Enter_us", "Exit_us")
    for r in _rows(side["api.csv"]):
        if not all(r.get(c) for c in need):
            continue
        names[ring.API_NAME_BASE | int(r["Api_Id"])] = r["Api"]
        if r.get("Args"):
            args[int(r["Corr"])] = r["Args"]
        api.append(
            (
                int(r["Corr"]),
                _ns(r["Enter_us"]),
                _ns(r["Exit_us"]),
                int(r["Api_Id"]),
                int(r["Thread"]),
            )
        )
    a = np.array(api, dtype=trace._API) if api else np.empty(0, dtype=trace._API)
    summary = Path(f"{ops_csv}.summary.txt")
    dev, pid = _summary(summary) if summary.exists() else ({}, 0)
    meta = {"source": "ktrace-spool", "dropped": 0}
    launch = _rows(side["launch.csv"])
    if launch:
        pid = pid or int(launch[0]["Process_Id"])
        meta["resume_ns"] = int(launch[0].get("Resume_Timestamp_ns") or 0)
    meta["producer_pid"] = pid
    tr = trace.Trace(meta=meta, names=names, ops=np.sort(o, order="begin"), api=a)
    tr.device = dev
    tr.api_args = args
    for r in _rows(side["roctx.csv"]):
        tr.roctx.append(
            (_ns(r["Timestamp_us"]), int(r["Thread"]), r["Op"], r["Message"])
        )
    return tr


def _summary(path: Path) -> tuple[dict, int]:
    """ktrace's ReadSummary(): the device lines of <spool>.summary.txt."""
    dev: dict = {}
    pid = 0
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        key, _, value = line.partition(":")
        key, value = key.strip(), value.strip()
        if key == "pid":
            pid = int(value or 0)
        elif key == "device" and value:
            dev["name"] = value
        elif key == "CUs":
            dev["cu_count"] = int(value)
        elif key == "warp size":
            dev["wave_front_size"] = int(value)
        elif key == "threads/CU":
            dev["threads_per_cu"] = int(value)
        elif key == "LDS/CU":
            dev["lds_per_cu"] = int(value)
        elif m := re.search(
            r"^clock / mem\s*:\s*(-?\d+) kHz / (-?\d+) kHz, bus (-?\d+) bit", line
        ):
            dev["clock_khz"] = int(m.group(1))
            dev["memory_clock_khz"] = int(m.group(2))
            dev["memory_bus_width"] = int(m.group(3))
    return dev, pid


def _device(path: Path) -> dict:
    """The device summary ktrace's agent table was written from."""
    gpu = next((r for r in _rows(path) if r["Agent_Type"] == "GPU"), None)
    if gpu is None:
        return {}
    wave = int(gpu["Wave_Front_Size"])
    return {
        "name": gpu["Name"],
        "cu_count": int(gpu["Cu_Count"]),
        "wave_front_size": wave,
        "threads_per_cu": int(gpu["Max_Waves_Per_Cu"]) * wave,
        "lds_per_cu": int(gpu["Local_Mem_Size"]),
        "clock_khz": int(gpu["Max_Engine_Clk_Ccompute"]) * 1000,
    }


def compare(ours: trace.Trace, ref: trace.Trace, top: int = 25) -> dict:
    """Side-by-side per-op attribution and kernel counts of two traces."""
    so, sr = trace.summarize(ours), trace.summarize(ref)
    fams = sorted(
        set(so["families"]) | set(sr["families"]),
        key=lambda k: -sr["families"].get(k, {}).get("gpu_ns", 0),
    )
    rows = []
    for f in fams:
        a, b = so["families"].get(f, {}), sr["families"].get(f, {})
        rows.append(
            {
                "family": f,
                "calls": (a.get("calls", 0), b.get("calls", 0)),
                "kernels": (a.get("kernels", 0), b.get("kernels", 0)),
                "gpu_ms": (a.get("gpu_ns", 0) / 1e6, b.get("gpu_ns", 0) / 1e6),
            }
        )
    ko, kr = Counter(), Counter()
    for k, v in so["kernels"].items():
        ko[k[:KTRACE_NAME_MAX]] += v["calls"]
    for k, v in sr["kernels"].items():
        kr[k[:KTRACE_NAME_MAX]] += v["calls"]
    diff = {
        k: (ko.get(k, 0), kr.get(k, 0))
        for k in set(ko) | set(kr)
        if ko.get(k, 0) != kr.get(k, 0)
    }
    print(
        f"{'family':<32}{'calls ours/ref':>18}{'kernels ours/ref':>20}{'gpu ms ours/ref':>22}"
    )
    for r in rows[:top]:
        c, k, g = r["calls"], r["kernels"], r["gpu_ms"]
        print(
            f"{r['family']:<32}{c[0]:>9}/{c[1]:<8}{k[0]:>10}/{k[1]:<9}{g[0]:>11.2f}/{g[1]:<10.2f}"
        )
    print(
        f"\nkernel names with different counts: {len(diff)} of {len(set(ko) | set(kr))}"
    )
    for k, (x, y) in sorted(diff.items(), key=lambda kv: -abs(kv[1][0] - kv[1][1]))[
        :10
    ]:
        print(f"  {x:>7} vs {y:<7} {k[:110]}")
    tot_o = sum(ko.values())
    tot_r = sum(kr.values())
    print(
        f"dispatches: ours {tot_o}, ref {tot_r}; unattributed GPU ms ours "
        f"{so['unattributed_gpu_ns'] / 1e6:.2f}, ref {sr['unattributed_gpu_ns'] / 1e6:.2f}"
    )
    return {
        "families": rows,
        "kernel_count_diffs": {k: list(v) for k, v in diff.items()},
        "dispatches": (tot_o, tot_r),
        "validation_ours": so["validation"],
        "validation_ref": sr["validation"],
    }
