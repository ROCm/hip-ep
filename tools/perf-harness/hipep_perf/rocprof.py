#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""rocprofv3 (ROCm 7.2.1, rocprofiler-sdk 1.1.0) export of a trace.

Writes the files ktrace's C++ finalize writes, with its conventions, so tools
that read either see the same thing: agent_info, hip_api_trace, kernel_trace,
memory_copy_trace, memory_allocation_trace, roctx_trace, results.json and the
rocprofv3 --stats tables.

The shim does not marshal call arguments, so fields that come from them are
written as ktrace writes them without KTRACE_API_ARGS: stream 0, launch
geometry and register counts 0, allocation size 0 and address 0. Dispatch ids
follow the order the runtime reported commands in, kernel ids the order kernels
first appear, and a command without a correlated call is charged to the
process id as its thread.
"""

from __future__ import annotations

import ctypes
import glob
import json
import math
import os
import platform
import re
import shutil
import socket
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np

from . import trace

KIND_MEMORY_COPY = 10
KIND_KERNEL_DISPATCH = 11
KIND_MEMORY_ALLOCATION = 16
KIND_HIP_RUNTIME = 21
KIND_HIP_COMPILER = 22

AGENT_HEADER = [
    "Node_Id", "Logical_Node_Id", "Agent_Type", "Cpu_Cores_Count", "Simd_Count",
    "Cpu_Core_Id_Base", "Simd_Id_Base", "Max_Waves_Per_Simd", "Lds_Size_In_Kb",
    "Gds_Size_In_Kb", "Num_Gws", "Wave_Front_Size", "Num_Xcc", "Cu_Count",
    "Array_Count", "Num_Shader_Banks", "Simd_Arrays_Per_Engine", "Cu_Per_Simd_Array",
    "Simd_Per_Cu", "Max_Slots_Scratch_Cu", "Gfx_Target_Version", "Vendor_Id",
    "Device_Id", "Location_Id", "Domain", "Drm_Render_Minor", "Num_Sdma_Engines",
    "Num_Sdma_Xgmi_Engines", "Num_Sdma_Queues_Per_Engine", "Num_Cp_Queues",
    "Max_Engine_Clk_Ccompute", "Max_Engine_Clk_Fcompute", "Sdma_Fw_Version",
    "Fw_Version", "Capability", "Cu_Per_Engine", "Max_Waves_Per_Cu", "Family_Id",
    "Workgroup_Max_Size", "Grid_Max_Size", "Local_Mem_Size", "Hive_Id", "Gpu_Id",
    "Workgroup_Max_Dim_X", "Workgroup_Max_Dim_Y", "Workgroup_Max_Dim_Z",
    "Grid_Max_Dim_X", "Grid_Max_Dim_Y", "Grid_Max_Dim_Z", "Name", "Vendor_Name",
    "Product_Name", "Model_Name",
]  # fmt: skip
HIP_HEADER = [
    "Domain", "Function", "Process_Id", "Thread_Id", "Correlation_Id",
    "Start_Timestamp", "End_Timestamp",
]  # fmt: skip
KERNEL_HEADER = [
    "Kind", "Agent_Id", "Queue_Id", "Stream_Id", "Thread_Id", "Dispatch_Id",
    "Kernel_Id", "Kernel_Name", "Correlation_Id", "Start_Timestamp", "End_Timestamp",
    "LDS_Block_Size", "Scratch_Size", "VGPR_Count", "Accum_VGPR_Count", "SGPR_Count",
    "Workgroup_Size_X", "Workgroup_Size_Y", "Workgroup_Size_Z", "Grid_Size_X",
    "Grid_Size_Y", "Grid_Size_Z",
]  # fmt: skip
COPY_HEADER = [
    "Kind", "Direction", "Stream_Id", "Source_Agent_Id", "Destination_Agent_Id",
    "Correlation_Id", "Start_Timestamp", "End_Timestamp",
]  # fmt: skip
ALLOC_HEADER = [
    "Kind", "Operation", "Agent_Id", "Allocation_Size", "Address", "Correlation_Id",
    "Start_Timestamp", "End_Timestamp",
]  # fmt: skip
ROCTX_HEADER = [
    "Kind", "Operation", "Process_Id", "Thread_Id", "Range_Id", "Timestamp", "Message",
]  # fmt: skip
STATS_HEADER = [
    "Name", "Calls", "TotalDurationNs", "AverageNs", "Percentage", "MinNs", "MaxNs",
    "StdDev",
]  # fmt: skip

# cl_command_type -> (direction, operation, source agent handle, destination)
COPY_DIRECTIONS = {
    0x11F3: ("MEMORY_COPY_DEVICE_TO_HOST", 3, 2, 1),
    0x11F4: ("MEMORY_COPY_HOST_TO_DEVICE", 2, 1, 2),
    0x11F5: ("MEMORY_COPY_DEVICE_TO_DEVICE", 4, 2, 2),
    0x11F6: ("MEMORY_COPY_DEVICE_TO_HOST", 3, 2, 1),
    0x11F7: ("MEMORY_COPY_HOST_TO_DEVICE", 2, 1, 2),
    0x11F8: ("MEMORY_COPY_DEVICE_TO_DEVICE", 4, 2, 2),
    0x11F9: ("MEMORY_COPY_DEVICE_TO_DEVICE", 4, 2, 2),
    0x11FA: ("MEMORY_COPY_DEVICE_TO_DEVICE", 4, 2, 2),
    0x1201: ("MEMORY_COPY_DEVICE_TO_HOST", 3, 2, 1),
    0x1202: ("MEMORY_COPY_HOST_TO_DEVICE", 2, 1, 2),
    0x1203: ("MEMORY_COPY_DEVICE_TO_DEVICE", 4, 2, 2),
    0x120A: ("MEMORY_COPY_DEVICE_TO_DEVICE", 4, 2, 2),
}
ALLOC_CALLS = {"hipMalloc", "hipHostMalloc", "hipFree", "hipHostFree"}
NULL_ADDRESS = "0x0000000000000000"
AGENT_CAPABILITIES = [
    "HotPluggable", "HSAMMUPresent", "SharedWithGraphics", "QueueSizePowerOfTwo",
    "QueueSize32bit", "QueueIdleEvent", "VALimit", "WatchPointsSupported",
    "WatchPointsTotalBits", "DoorbellType", "AQLQueueDoubleMap",
    "DebugTrapSupported", "WaveLaunchTrapOverrideSupported",
    "WaveLaunchModeSupported", "PreciseMemoryOperationsSupported",
    "DEPRECATED_SRAM_EDCSupport", "Mem_EDCSupport", "RASEventNotify",
    "ASICRevision", "SRAM_EDCSupport", "SVMAPISupported", "CoherentHostAccess",
    "DebugSupportedFirmware", "Reserved",
]  # fmt: skip


FILL_KINDS = {0x1207, 0x1208}
RGP_INDEX_COLUMN = 6


# ---- readable kernel names (ktrace's ResolveKernelNames) ------------------------


def _comgr_bind(path: str):
    """The demangle call of one amd_comgr DLL, or None if it lacks it."""
    try:
        lib = ctypes.CDLL(path)
        create, release, set_data, get_data, demangle = (
            lib.amd_comgr_create_data,
            lib.amd_comgr_release_data,
            lib.amd_comgr_set_data,
            lib.amd_comgr_get_data,
            lib.amd_comgr_demangle_symbol_name,
        )
    except (OSError, AttributeError):
        return None
    handle = ctypes.c_uint64
    create.argtypes = [ctypes.c_int, ctypes.POINTER(handle)]
    release.argtypes = [handle]
    set_data.argtypes = [handle, ctypes.c_size_t, ctypes.c_char_p]
    get_data.argtypes = [handle, ctypes.POINTER(ctypes.c_size_t), ctypes.c_char_p]
    demangle.argtypes = [handle, ctypes.POINTER(handle)]
    bytes_kind = 0x9  # AMD_COMGR_DATA_KIND_BYTES

    def one(name: str) -> str | None:
        src, dst = handle(0), handle(0)
        if create(bytes_kind, ctypes.byref(src)):
            return None
        try:
            raw = name.encode()
            if set_data(src, len(raw), raw) or demangle(src, ctypes.byref(dst)):
                return None
            size = ctypes.c_size_t(0)
            if get_data(dst, ctypes.byref(size), None) or not size.value:
                return None
            buf = ctypes.create_string_buffer(size.value)
            if get_data(dst, ctypes.byref(size), buf):
                return None
            return buf.raw[: size.value].rstrip(b"\x00").decode("utf-8", "replace")
        finally:
            if dst.value:
                release(dst)
            release(src)

    return one


def _comgr_demangler():
    """amd_comgr_demangle_symbol_name, or None. AMD_COMGR names the DLL."""
    cands = [os.environ.get("AMD_COMGR", "")]
    dirs = [os.environ.get("HIPEP_BIN", "")] + os.environ.get("PATH", "").split(";")
    for d in dirs:
        if d:
            cands += sorted(glob.glob(os.path.join(d, "amd_comgr*.dll")), reverse=True)
    for path in cands:
        if path and os.path.isfile(path) and (fn := _comgr_bind(path)):
            return fn
    return None


def kernel_label(value: str) -> str:
    value = value.replace("(anonymous namespace)::", "")
    depth = 0
    for i, ch in enumerate(value):
        if ch == "<":
            depth += 1
        elif ch == ">":
            depth -= 1
        elif ch == "(" and depth == 0:
            value = value[:i]
            break
    depth = 0
    for i, ch in enumerate(value):
        if ch == "<":
            depth += 1
        elif ch == ">":
            depth -= 1
        elif ch == " " and depth == 0:
            value = value[i + 1 :]
            break
    value = re.sub(r",\s+", ",", value).strip(" \t\r\n")
    m = re.search(r"^(Cijk_\w*?_MT\d+x\d+x\d+_MI\d+x\d+x\d+)_.*", value)
    return m.group(1) if m else value


def _kernel_fallback(mangled: str) -> str:
    m = re.fullmatch(r"_Z(\d+)(.*)", mangled, re.DOTALL)
    if m:
        return m.group(2)[: int(m.group(1))]
    m = re.fullmatch(r"_ZN(.*)", mangled, re.DOTALL)
    if not m:
        return mangled
    rest, last = m.group(1), mangled
    while m := re.fullmatch(r"(\d+)(.*)", rest, re.DOTALL):
        n = int(m.group(1))
        last, rest = m.group(2)[:n], m.group(2)[n:]
    return last


def readable_names(raw_names: list[str]) -> dict[str, str]:
    """Mangled name -> the label ktrace prints (comgr, else llvm-cxxfilt)."""
    names = list(dict.fromkeys(n for n in raw_names if n))
    out: dict[str, str] = {}
    if not names:
        return out
    demangle = _comgr_demangler()
    if demangle:
        for n in names:
            d = demangle(n)
            if not d:
                out = {}
                break
            out[n] = kernel_label(d)
    if not out:
        tool = shutil.which("llvm-cxxfilt") or shutil.which("c++filt")
        if tool:
            res = subprocess.run(
                [tool],
                input="\n".join(names) + "\n",
                capture_output=True,
                text=True,
                check=False,
            )
            lines = res.stdout.splitlines()
            if res.returncode == 0 and len(lines) >= len(names):
                out = {n: kernel_label(lines[i].strip()) for i, n in enumerate(names)}
    if not out:
        out = {n: _kernel_fallback(n) for n in names}
    owners: dict[str, list[str]] = defaultdict(list)
    for raw, label in out.items():
        owners[label].append(raw)
    for label, raws in owners.items():
        if len(raws) < 2:
            continue
        raws.sort()
        common = set(raws[0].split("_"))
        for r in raws[1:]:
            common &= set(r.split("_"))
        for i, r in enumerate(raws):
            varying = sorted(set(r.split("_")) - common)
            if varying and len(varying) <= 6:
                out[r] = f"{label}[{','.join(varying)}]"
            else:
                out[r] = f"{label}#{i + 1}"
    return out


class _Csv:
    """rocprofiler-sdk csv.hpp formatting: strings quoted, numbers bare."""

    def __init__(self, path: Path, header: list[str], quoted: set[int]):
        self.path, self.quoted, self.rows = path, quoted, 0
        self.fh = path.open("w", newline="", encoding="utf-8")
        self.fh.write(",".join(f'"{h}"' for h in header) + "\n")

    def row(self, values: list) -> None:
        self.fh.write(
            ",".join(
                f'"{v}"' if i in self.quoted else str(v) for i, v in enumerate(values)
            )
            + "\n"
        )
        self.rows += 1

    def close(self, keep_empty: bool = False) -> Path | None:
        self.fh.close()
        if self.rows or keep_empty:
            return self.path
        self.path.unlink()
        return None


class _Stats:
    """rocprofv3 --stats: Welford deviation, rows by total duration then name."""

    def __init__(self) -> None:
        self.by: dict[str, list] = {}
        self.total = [0, 0, None, 0, 0.0, 0.0]  # every call, for domain_stats

    @staticmethod
    def _add(s: list, ns: int) -> None:
        s[0] += 1
        s[1] += ns
        s[2] = ns if s[2] is None else min(s[2], ns)
        s[3] = max(s[3], ns)
        delta = ns - s[4]
        s[4] += delta / s[0]
        s[5] += delta * (ns - s[4])

    def add(self, name: str, ns: int) -> None:
        self._add(self.by.setdefault(name, [0, 0, None, 0, 0.0, 0.0]), ns)
        self._add(self.total, ns)

    @staticmethod
    def _num(v: float) -> str:
        return f"{v:.6f}" if v > 0.01 else f"{v:.8e}"

    @staticmethod
    def _pct(v: float) -> str:
        if v >= 1.0:
            return f"{v:.2f}"
        return f"{v:.4f}" if v > 0.01 else f"{v:.3e}"

    @classmethod
    def write_rows(cls, path: Path, rows: list[tuple[str, list]]) -> Path | None:
        if not rows:
            return None
        rows = sorted(rows, key=lambda r: (-r[1][1], r[0]))
        total = sum(r[1][1] for r in rows)
        out = _Csv(path, STATS_HEADER, {0})
        for name, s in rows:
            share = 100.0 * s[1] / total if total else 0.0
            sd = math.sqrt(s[5] / (s[0] - 1)) if s[0] > 1 else 0.0
            out.row(
                [
                    name,
                    s[0],
                    s[1],
                    cls._num(s[1] / s[0] if s[0] else 0.0),
                    cls._pct(share),
                    s[2] or 0,
                    s[3],
                    cls._num(sd),
                ]
            )
        return out.close()

    def write(self, path: Path, label: dict[str, str] | None = None) -> Path | None:
        label = label or {}
        return self.write_rows(path, [(label.get(k, k), s) for k, s in self.by.items()])


def _agent_row(dev: dict, gpu: bool) -> list:
    row = dict.fromkeys(AGENT_HEADER, 0)
    cu = max(int(dev.get("cu_count", 0)), 1) if gpu else 0
    wave = max(int(dev.get("wave_front_size", 0)), 1) if gpu else 0
    threads = max(wave, int(dev.get("threads_per_cu", 0))) if gpu else 0
    waves_cu = max(1, threads // wave) if gpu else 0
    lds = int(dev.get("lds_per_cu", 0)) if gpu else 0
    name = dev.get("name") or "AMD GPU" if gpu else _processor_name()
    row.update(
        {
            "Node_Id": 1 if gpu else 0,
            "Logical_Node_Id": 1 if gpu else 0,
            "Agent_Type": "GPU" if gpu else "CPU",
            "Cpu_Cores_Count": 0 if gpu else os.cpu_count() or 1,
            "Simd_Count": cu * 4,
            "Max_Waves_Per_Simd": max(1, waves_cu // 4) if gpu else 0,
            "Lds_Size_In_Kb": lds // 1024,
            "Wave_Front_Size": wave,
            "Num_Xcc": 1 if gpu else 0,
            "Cu_Count": cu,
            "Simd_Per_Cu": 4 if gpu else 0,
            "Vendor_Id": 4098 if gpu else 0,
            "Max_Engine_Clk_Ccompute": int(dev.get("clock_khz", 0)) // 1000
            if gpu
            else 0,
            "Max_Waves_Per_Cu": waves_cu,
            "Workgroup_Max_Size": 1024 if gpu else 0,
            "Local_Mem_Size": lds,
            "Workgroup_Max_Dim_X": 1024 if gpu else 0,
            "Workgroup_Max_Dim_Y": 1024 if gpu else 0,
            "Workgroup_Max_Dim_Z": 1024 if gpu else 0,
            "Name": name,
            "Vendor_Name": "AMD" if gpu else "",
            "Product_Name": name,
            "Model_Name": name,
        }
    )
    return [row[h] for h in AGENT_HEADER]


def _agent_json(handle: int, dev: dict) -> dict:
    gpu = handle == 2
    row = dict(zip(AGENT_HEADER, _agent_row(dev, gpu)))
    lower = {k.lower(): v for k, v in row.items()}
    out = {"size": 1, "id": {"handle": handle}, "type": 2 if gpu else 1}
    for key in (
        "cpu_cores_count", "simd_count", "mem_banks_count", "caches_count",
        "io_links_count", "cpu_core_id_base", "simd_id_base", "max_waves_per_simd",
        "lds_size_in_kb", "gds_size_in_kb", "num_gws", "wave_front_size", "num_xcc",
        "cu_count", "array_count", "num_shader_banks", "simd_arrays_per_engine",
        "cu_per_simd_array", "simd_per_cu", "max_slots_scratch_cu",
        "gfx_target_version", "vendor_id", "device_id", "location_id", "domain",
        "drm_render_minor", "num_sdma_engines", "num_sdma_xgmi_engines",
        "num_sdma_queues_per_engine", "num_cp_queues", "max_engine_clk_ccompute",
        "max_engine_clk_fcompute",
    ):  # fmt: skip
        out[key] = int(lower.get(key, 0))
    out["sdma_fw_version"] = {"uCodeSDMA": 0, "uCodeRes": 0}
    out["fw_version"] = {"uCode": 0, "Major": 0, "Minor": 0, "Stepping": 0}
    out["capability"] = dict.fromkeys(AGENT_CAPABILITIES, 0)
    out["cu_per_engine"] = 0
    out["max_waves_per_cu"] = int(row["Max_Waves_Per_Cu"])
    out["family_id"] = 0
    out["workgroup_max_size"] = int(row["Workgroup_Max_Size"])
    out["grid_max_size"] = 0
    out["local_mem_size"] = int(row["Local_Mem_Size"])
    out["hive_id"] = 0
    out["gpu_id"] = 0
    d = 1024 if gpu else 0
    out["workgroup_max_dim"] = {"x": d, "y": d, "z": d}
    out["grid_max_dim"] = {"x": 0, "y": 0, "z": 0}
    out["name"] = row["Name"]
    out["vendor_name"] = row["Vendor_Name"]
    out["product_name"] = row["Product_Name"]
    out["model_name"] = row["Model_Name"]
    out["node_id"] = handle - 1
    out["logical_node_id"] = handle - 1
    out["logical_node_type_id"] = 0
    out["runtime_visibility"] = dict.fromkeys(("hsa", "hip", "rccl", "rocdecode"), gpu)
    out["uuid"] = {"bytes": [0] * 16}
    out["mem_banks"], out["caches"], out["io_links"] = [], [], []
    out["gpu_index"] = 0 if gpu else -1
    return out


def _processor_name() -> str:
    return os.environ.get("PROCESSOR_IDENTIFIER") or platform.processor() or "CPU"


def _symbol(kernel_id: int, name: str) -> dict:
    return {
        "size": 1,
        "kernel_id": kernel_id,
        "code_object_id": 0,
        "kernel_name": name,
        "kernel_object": 0,
        "kernarg_segment_size": 0,
        "kernarg_segment_alignment": 0,
        "group_segment_size": 0,
        "private_segment_size": 0,
        "sgpr_count": 0,
        "arch_vgpr_count": 0,
        "accum_vgpr_count": 0,
        "kernel_code_entry_byte_offset": 0,
        "kernel_address": {"handle": 0},
        "formatted_kernel_name": name,
        "demangled_kernel_name": name,
        "truncated_kernel_name": name,
    }


def _parse_args(text: str) -> list[tuple[str, str]]:
    """ktrace's ParseArguments: whitespace-separated name=value items."""
    out = []
    for item in text.split():
        k, sep, v = item.partition("=")
        if sep:
            out.append((k, v))
    return out


_UINT = re.compile(r"\s*\+?(0[xX][0-9a-fA-F]+|0[0-7]*|[1-9][0-9]*)")


def _u64(text: str | None) -> int:
    """strtoull(text, base 0), 0 when nothing parses."""
    m = _UINT.match(text or "")
    if not m:
        return 0
    g = m.group(1)
    if g[:2] in ("0x", "0X"):
        return int(g, 16)
    return int(g, 8) if g.startswith("0") else int(g)


def _dim(text: str) -> tuple[int, int, int]:
    """ktrace's ParseDim: 'XxYxZ', all-or-nothing."""
    parts = text.split("x")
    if len(parts) != 3 or any(not p for p in parts):
        return (0, 0, 0)
    try:
        return tuple(int(p, 0) for p in parts)  # type: ignore[return-value]
    except ValueError:
        return (0, 0, 0)


def _corr(corr: int, asynchronous: bool) -> dict:
    out = {"internal": corr, "external": 0}
    if not asynchronous:
        out["ancestor"] = 0
    return out


class _Frag:
    """A JSON array written one record at a time, so large traces stay flat."""

    def __init__(self, path: Path):
        self.path = path
        self.fh = path.open("w", encoding="utf-8")
        self.n = 0

    def add(self, obj: dict) -> None:
        self.fh.write(("," if self.n else "") + json.dumps(obj, separators=(",", ":")))
        self.n += 1

    def close(self) -> None:
        self.fh.close()

    def copy_to(self, out) -> None:
        out.write("[")
        with self.path.open(encoding="utf-8") as fh:
            while chunk := fh.read(1 << 20):
                out.write(chunk)
        out.write("]")
        self.path.unlink()


def write(tr: trace.Trace, out_dir: Path, prefix: str = "trace") -> list[Path]:
    """Write the rocprofv3 file set; returns the paths written."""
    if tr.op_range is None:
        trace.attribute(tr)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pid = int(tr.meta.get("producer_pid", 0))
    dev = tr.device
    paths: list[Path] = []

    def p(name: str) -> Path:
        return out_dir / f"{prefix}_{name}"

    agents = _Csv(p("agent_info.csv"), AGENT_HEADER, {2, 49, 50, 51, 52})
    agents.row(_agent_row(dev, False))
    agents.row(_agent_row(dev, True))
    paths.append(agents.close())

    tmp = {k: _Frag(p(f"{k}.frag")) for k in ("hip", "kernel", "copy", "alloc")}
    hip_stats, kernel_stats, copy_stats, alloc_stats, marker_stats = (
        _Stats(),
        _Stats(),
        _Stats(),
        _Stats(),
        _Stats(),
    )
    api_names: dict[int, dict[int, str]] = defaultdict(dict)
    t_lo, t_hi = [], []

    hip_csv = _Csv(p("hip_api_trace.csv"), HIP_HEADER, {0, 1})
    alloc_csv = _Csv(p("memory_allocation_trace.csv"), ALLOC_HEADER, {0, 1, 2, 4})
    if len(tr.api):
        t_lo.append(int(tr.api["enter"].min()))
        t_hi.append(int(tr.api["exit"].max()))
    # Per correlation id: stream, grid, block, dynamic LDS, host copy.
    launches: dict[int, tuple] = {}
    stream_ids = {0: 0}
    for a in tr.api:
        corr, api_id, tid = int(a["corr"]), int(a["api_id"]), int(a["tid"])
        start, end = int(a["enter"]), int(a["exit"])
        fn = tr.api_name(api_id)
        kind = KIND_HIP_COMPILER if fn.startswith("__hip") else KIND_HIP_RUNTIME
        domain = (
            "HIP_COMPILER_API_EXT"
            if kind == KIND_HIP_COMPILER
            else "HIP_RUNTIME_API_EXT"
        )
        api_names[kind][api_id] = fn
        hip_csv.row([domain, fn, pid, tid, corr, start, end])
        hip_stats.add(fn, max(0, end - start))
        args = _parse_args(tr.api_args.get(corr, ""))
        values = dict(args)
        stream = stream_ids.setdefault(_u64(values.get("stream")), len(stream_ids))
        launches[corr] = (
            stream,
            _dim(values.get("grid", "")),
            _dim(values.get("block", "")),
            _u64(values.get("shmem")),
            fn.startswith("hipMemcpy")
            and (_u64(values.get("kind")) in (1, 2) or "HtoD" in fn or "DtoH" in fn),
        )
        tmp["hip"].add(
            {
                "size": 1,
                "kind": kind,
                "operation": api_id,
                "correlation_id": _corr(corr, False),
                "start_timestamp": start,
                "end_timestamp": end,
                "thread_id": tid,
                "args": [{"type": "uint64_t", "name": k, "value": v} for k, v in args],
                "retval": {"uint64_t_retval": 0},
                "stream_id": {"handle": stream},
            }
        )
        if fn in ALLOC_CALLS:
            allocate = "Malloc" in fn
            host = "Host" in fn
            op_name = (
                "MEMORY_ALLOCATION_ALLOCATE" if allocate else "MEMORY_ALLOCATION_FREE"
            )
            agent = ("Agent 0" if host else "Agent 1") if allocate else ""
            size = _u64(values.get("size")) if allocate else 0
            address = f"0x{_u64(values.get('ptr')):016x}"
            alloc_csv.row(
                ["MEMORY_ALLOCATION", op_name, agent, size, address, corr, start, end]
            )
            alloc_stats.add(op_name, max(0, end - start))
            tmp["alloc"].add(
                {
                    "size": 1,
                    "kind": KIND_MEMORY_ALLOCATION,
                    "operation": 1 if allocate else 3,
                    "thread_id": tid,
                    "correlation_id": _corr(corr, False),
                    "start_timestamp": start,
                    "end_timestamp": end,
                    "agent_id": {"handle": (1 if host else 2) if allocate else 0},
                    "address": address,
                    "allocation_size": size,
                    "stream_id": {"handle": stream},
                }
            )
    paths.append(hip_csv.close())
    paths.append(alloc_csv.close())

    roctx_csv = _Csv(p("roctx_trace.csv"), ROCTX_HEADER, {0, 1, 6})
    open_ranges: dict[int, list[tuple[int, str]]] = defaultdict(list)
    for ts, tid, op, msg in tr.roctx:
        roctx_csv.row(["ROCTX", op, pid, tid, 0, ts, msg])
        if op == "RANGE_PUSH":
            open_ranges[tid].append((ts, msg))
        elif op == "RANGE_POP" and open_ranges[tid]:
            begin, m = open_ranges[tid].pop()
            marker_stats.add(m, max(0, ts - begin))
        elif op == "MARK":
            marker_stats.add(msg, 0)
    paths.append(roctx_csv.close())

    order = np.argsort(tr.ops["seq"], kind="stable")
    if len(tr.ops):
        t_lo.append(int(tr.ops["begin"].min()))
        t_hi.append(int(tr.ops["end"].max()))
    disp = tr.ops["op"] == trace.OP_DISPATCH
    labels = readable_names([tr.name(x) for x in tr.ops["aux"][order][disp[order]]])
    kernel_csv = _Csv(p("kernel_trace.csv"), KERNEL_HEADER, {0, 1, 7})
    readable_header = list(KERNEL_HEADER)
    readable_header.insert(RGP_INDEX_COLUMN, "RGP_Dispatch_Index")
    readable_csv = _Csv(p("kernel_trace_readable.csv"), readable_header, {0, 1, 8})
    copy_csv = _Csv(p("memory_copy_trace.csv"), COPY_HEADER, {0, 1, 3, 4})
    kernel_ids: dict[str, int] = {}
    dispatch_id = rgp_index = 0
    next_corr = (int(tr.api["corr"].max()) if len(tr.api) else 0) + 1
    for i in order:
        o = tr.ops[i]
        corr = int(o["corr"])
        if not corr:
            corr, next_corr = next_corr, next_corr + 1
        start, end, queue = int(o["begin"]), int(o["end"]), int(o["queue"])
        tid = int(tr.op_tid[i]) or pid
        stream, grid, block, shmem, host = launches.get(
            int(o["corr"]), (0, (0, 0, 0), (0, 0, 0), 0, False)
        )
        if o["op"] == trace.OP_DISPATCH:
            dispatch_id += 1
            rgp_index += 1
            name = tr.name(o["aux"])
            kid = kernel_ids.setdefault(name, len(kernel_ids) + 1)
            row = ["KERNEL_DISPATCH", "Agent 1", queue, stream, tid, dispatch_id, kid,
                   name, corr, start, end, 0, 0, 0, 0, 0, *block, *grid]  # fmt: skip
            kernel_csv.row(row)
            row[7] = labels.get(name, name)
            row.insert(RGP_INDEX_COLUMN, rgp_index)
            readable_csv.row(row)
            kernel_stats.add(name, max(0, end - start))
            tmp["kernel"].add(
                {
                    "size": 1,
                    "kind": KIND_KERNEL_DISPATCH,
                    "operation": 2,
                    "thread_id": tid,
                    "correlation_id": _corr(corr, True),
                    "start_timestamp": start,
                    "end_timestamp": end,
                    "dispatch_info": {
                        "size": 1,
                        "agent_id": {"handle": 2},
                        "queue_id": {"handle": queue},
                        "kernel_id": kid,
                        "dispatch_id": dispatch_id,
                        "private_segment_size": 0,
                        "group_segment_size": shmem,
                        "workgroup_size": dict(zip("xyz", block)),
                        "grid_size": dict(zip("xyz", grid)),
                    },
                    "stream_id": {"handle": stream},
                }
            )
            continue
        kind = int(o["kind"])
        if o["op"] != trace.OP_COPY:
            continue
        d = COPY_DIRECTIONS.get(kind)
        # A host hipMemcpy is a DMA copy; any other device-to-device copy, and
        # every fill, runs a runtime blit kernel that RGP counts as a dispatch.
        if kind in FILL_KINDS or (d and d[1] == 4 and not host):
            rgp_index += 1
        if d:
            direction, operation, src, dst = d
            copy_csv.row(
                ["MEMORY_COPY", direction, stream, f"Agent {src - 1}", f"Agent {dst - 1}",
                 corr, start, end]
            )  # fmt: skip
            copy_stats.add(direction, max(0, end - start))
            tmp["copy"].add(
                {
                    "size": 1,
                    "kind": KIND_MEMORY_COPY,
                    "operation": operation,
                    "thread_id": tid,
                    "correlation_id": _corr(corr, True),
                    "start_timestamp": start,
                    "end_timestamp": end,
                    "dst_agent_id": {"handle": dst},
                    "src_agent_id": {"handle": src},
                    "bytes": int(o["aux"]),
                    "stream_id": {"handle": stream},
                }
            )
    paths.append(kernel_csv.close(keep_empty=bool(tr.roctx)))
    paths.append(readable_csv.close(keep_empty=bool(tr.roctx)))
    paths.append(copy_csv.close())

    tables = (
        ("kernel_stats.csv", kernel_stats, "KERNEL_DISPATCH"),
        ("hip_api_stats.csv", hip_stats, "HIP_API"),
        ("memory_copy_stats.csv", copy_stats, "MEMORY_COPY"),
        ("memory_allocation_stats.csv", alloc_stats, "MEMORY_ALLOCATION"),
        ("marker_api_stats.csv", marker_stats, "MARKER_API"),
    )
    domains = []
    for fname, st, domain in tables:
        paths.append(st.write(p(fname), labels if st is kernel_stats else None))
        if st.by:
            domains.append((domain, st.total))
    paths.append(_Stats.write_rows(p("domain_stats.csv"), domains))

    for f in tmp.values():
        f.close()
    with p("results.json").open("w", encoding="utf-8") as out:
        meta = {
            "node": {
                "id": 0,
                "hash": 0,
                "machine_id": "",
                "system_name": "Windows",
                "hostname": socket.gethostname(),
                "release": platform.release(),
                "version": platform.version(),
                "hardware_name": platform.machine(),
                "domain_name": "",
            },
            "pid": pid,
            "init_time": tr.meta.get("resume_ns") or (min(t_lo) if t_lo else 0),
            "fini_time": max(t_hi) if t_hi else 0,
            "command": [],
            "config": _config(out_dir, prefix),
        }
        strings = {
            "callback_records": [],
            "buffer_records": _string_tables(api_names),
            "marker_api": [],
            "correlation_id": {"external": []},
            "counters": {"dimension_ids": []},
            "pc_sample_instructions": [],
            "pc_sample_comments": [],
            "att_filenames": [],
            "code_object_snapshot_filenames": [],
        }
        symbols = [_symbol(0, "<unknown>")] + [
            _symbol(k, n) for n, k in sorted(kernel_ids.items(), key=lambda x: x[1])
        ]

        def dump(v) -> str:
            return json.dumps(v, separators=(",", ":"))

        out.write('{"rocprofiler-sdk-tool":[{"metadata":' + dump(meta))
        out.write(',"agents":' + dump([_agent_json(1, dev), _agent_json(2, dev)]))
        out.write(',"counters":[],"strings":' + dump(strings))
        out.write(',"code_objects":[],"kernel_symbols":' + dump(symbols))
        out.write(',"host_functions":[],"summary":[]')
        out.write(',"callback_records":{"counter_collection":[]}')
        out.write(',"buffer_records":{"kernel_dispatch":')
        tmp["kernel"].copy_to(out)
        out.write(',"hip_api":')
        tmp["hip"].copy_to(out)
        out.write(',"hsa_api":[],"marker_api":[],"rccl_api":[],"memory_copy":')
        tmp["copy"].copy_to(out)
        out.write(',"memory_allocation":')
        tmp["alloc"].copy_to(out)
        out.write(',"scratch_memory":[],"rocdecode_api":[],"rocjpeg_api":[]')
        out.write(',"pc_sample_host_trap":[],"pc_sample_stochastic":[]}}]}\n')
    paths.append(p("results.json"))
    return [x for x in paths if x is not None]


def _string_tables(api_names: dict[int, dict[int, str]]) -> list[dict]:
    kinds = [
        "NONE", "HSA_CORE_API", "HSA_AMD_EXT_API", "HSA_IMAGE_EXT_API",
        "HSA_FINALIZE_EXT_API", "HIP_RUNTIME_API", "HIP_COMPILER_API",
        "MARKER_CORE_API", "MARKER_CONTROL_API", "MARKER_NAME_API", "MEMORY_COPY",
        "KERNEL_DISPATCH", "SCRATCH_MEMORY", "CORRELATION_ID_RETIREMENT", "RCCL_API",
        "OMPT", "MEMORY_ALLOCATION", "RUNTIME_INITIALIZATION", "ROCDECODE_API",
        "ROCJPEG_API", "HIP_STREAM", "HIP_RUNTIME_API_EXT", "HIP_COMPILER_API_EXT",
    ]  # fmt: skip
    out = []
    for kind in range(KIND_HIP_COMPILER + 1):
        ops: list[str] = []
        name = ""
        if kind == KIND_MEMORY_COPY:
            name = "MEMORY_COPY"
            ops = ["", "MEMORY_COPY_HOST_TO_HOST", "MEMORY_COPY_HOST_TO_DEVICE",
                   "MEMORY_COPY_DEVICE_TO_HOST", "MEMORY_COPY_DEVICE_TO_DEVICE"]  # fmt: skip
        elif kind == KIND_KERNEL_DISPATCH:
            name = "KERNEL_DISPATCH"
            ops = ["", "", "KERNEL_DISPATCH_COMPLETE"]
        elif kind == KIND_MEMORY_ALLOCATION:
            name = "MEMORY_ALLOCATION"
            ops = ["", "MEMORY_ALLOCATION_ALLOCATE", "MEMORY_ALLOCATION_VMEM_ALLOCATE",
                   "MEMORY_ALLOCATION_FREE", "MEMORY_ALLOCATION_VMEM_FREE"]  # fmt: skip
        elif kind in (KIND_HIP_RUNTIME, KIND_HIP_COMPILER):
            name = kinds[kind]
            found = api_names.get(kind)
            if found:
                ops = [""] * (max(found) + 1)
                for i, fn in found.items():
                    ops[i] = fn
        out.append({"kind": name, "operations": ops})
    return out


def _config(out_dir: Path, prefix: str) -> dict:
    path = str(out_dir.resolve())
    return {
        "output_path": path,
        "output_file": prefix,
        "tmp_directory": path,
        "raw_output_path": path,
        "raw_output_file": "trace.records",
        "raw_tmp_directory": path,
        "perfetto_shmem_size_hint": 0,
        "perfetto_buffer_size": 0,
        "perfetto_buffer_fill_policy": "",
        "perfetto_backend": "",
        "summary": False,
        "summary_per_domain": False,
        "summary_groups": [],
        "summary_unit": "nsec",
        "summary_file": "",
        "csv_output": True,
        "json_output": True,
        "pftrace_output": False,
        "otf2_output": False,
        "summary_output": False,
        "rocpd_output": False,
        "kernel_rename": False,
        "group_by_queue": False,
    }
