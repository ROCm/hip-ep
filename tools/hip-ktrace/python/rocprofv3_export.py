#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""Finalize ktrace private spools as ROCm 7.2.1 rocprofv3 CSV and JSON.

The injected DLL deliberately writes a compact, append-only private format while
the application is running.  This program runs after the target exits and emits
the public contract.  Keeping the compatibility layer out of the callback path
preserves ktrace's bounded-memory, low-contention capture design.

Schema authority (in priority order):
  ROCprofiler-SDK rocm-7.2.1 source/lib/output/generate{CSV,JSON}.cpp
  ROCprofiler-SDK rocm-7.2.1 cxx/serialization/save.hpp
  ROCprofiler-SDK 1.1.0 documentation

The main files contain no custom fields.  Information rocprofv3 cannot express
is written to ``<prefix>_ktrace_extensions.json``.
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import json
import mmap
import os
import platform
import re
import shutil
import struct
import tempfile
from contextlib import ExitStack
from pathlib import Path

from kernel_names import resolve_names


HIP_RUNTIME_KIND = 21
HIP_COMPILER_KIND = 22
MEMORY_COPY_KIND = 10
KERNEL_DISPATCH_KIND = 11
MEMORY_ALLOCATION_KIND = 16

KERNEL_DISPATCH_COMPLETE = 2
ALLOCATE, VMEM_ALLOCATE, FREE, VMEM_FREE = 1, 2, 3, 4

AGENT_HEADER = [
    "Node_Id",
    "Logical_Node_Id",
    "Agent_Type",
    "Cpu_Cores_Count",
    "Simd_Count",
    "Cpu_Core_Id_Base",
    "Simd_Id_Base",
    "Max_Waves_Per_Simd",
    "Lds_Size_In_Kb",
    "Gds_Size_In_Kb",
    "Num_Gws",
    "Wave_Front_Size",
    "Num_Xcc",
    "Cu_Count",
    "Array_Count",
    "Num_Shader_Banks",
    "Simd_Arrays_Per_Engine",
    "Cu_Per_Simd_Array",
    "Simd_Per_Cu",
    "Max_Slots_Scratch_Cu",
    "Gfx_Target_Version",
    "Vendor_Id",
    "Device_Id",
    "Location_Id",
    "Domain",
    "Drm_Render_Minor",
    "Num_Sdma_Engines",
    "Num_Sdma_Xgmi_Engines",
    "Num_Sdma_Queues_Per_Engine",
    "Num_Cp_Queues",
    "Max_Engine_Clk_Ccompute",
    "Max_Engine_Clk_Fcompute",
    "Sdma_Fw_Version",
    "Fw_Version",
    "Capability",
    "Cu_Per_Engine",
    "Max_Waves_Per_Cu",
    "Family_Id",
    "Workgroup_Max_Size",
    "Grid_Max_Size",
    "Local_Mem_Size",
    "Hive_Id",
    "Gpu_Id",
    "Workgroup_Max_Dim_X",
    "Workgroup_Max_Dim_Y",
    "Workgroup_Max_Dim_Z",
    "Grid_Max_Dim_X",
    "Grid_Max_Dim_Y",
    "Grid_Max_Dim_Z",
    "Name",
    "Vendor_Name",
    "Product_Name",
    "Model_Name",
]
HIP_HEADER = [
    "Domain",
    "Function",
    "Process_Id",
    "Thread_Id",
    "Correlation_Id",
    "Start_Timestamp",
    "End_Timestamp",
]
KERNEL_HEADER = [
    "Kind",
    "Agent_Id",
    "Queue_Id",
    "Stream_Id",
    "Thread_Id",
    "Dispatch_Id",
    "Kernel_Id",
    "Kernel_Name",
    "Correlation_Id",
    "Start_Timestamp",
    "End_Timestamp",
    "LDS_Block_Size",
    "Scratch_Size",
    "VGPR_Count",
    "Accum_VGPR_Count",
    "SGPR_Count",
    "Workgroup_Size_X",
    "Workgroup_Size_Y",
    "Workgroup_Size_Z",
    "Grid_Size_X",
    "Grid_Size_Y",
    "Grid_Size_Z",
]
COPY_HEADER = [
    "Kind",
    "Direction",
    "Stream_Id",
    "Source_Agent_Id",
    "Destination_Agent_Id",
    "Correlation_Id",
    "Start_Timestamp",
    "End_Timestamp",
]
ALLOC_HEADER = [
    "Kind",
    "Operation",
    "Agent_Id",
    "Allocation_Size",
    "Address",
    "Correlation_Id",
    "Start_Timestamp",
    "End_Timestamp",
]
ROCTX_HEADER = [
    "Kind",
    "Operation",
    "Process_Id",
    "Thread_Id",
    "Range_Id",
    "Timestamp",
    "Message",
]

COPY_DIRECTIONS = {
    "CopyHostToDevice": ("MEMORY_COPY_HOST_TO_DEVICE", 1, 2),
    "CopyHostToDevice2D": ("MEMORY_COPY_HOST_TO_DEVICE", 1, 2),
    "WriteImage": ("MEMORY_COPY_HOST_TO_DEVICE", 1, 2),
    "CopyDeviceToHost": ("MEMORY_COPY_DEVICE_TO_HOST", 2, 1),
    "CopyDeviceToHost2D": ("MEMORY_COPY_DEVICE_TO_HOST", 2, 1),
    "ReadImage": ("MEMORY_COPY_DEVICE_TO_HOST", 2, 1),
    "CopyDeviceToDevice": ("MEMORY_COPY_DEVICE_TO_DEVICE", 2, 2),
    "CopyDeviceToDevice2D": ("MEMORY_COPY_DEVICE_TO_DEVICE", 2, 2),
    "CopyImage": ("MEMORY_COPY_DEVICE_TO_DEVICE", 2, 2),
    "CopyImageToBuffer": ("MEMORY_COPY_DEVICE_TO_DEVICE", 2, 2),
    "CopyBufferToImage": ("MEMORY_COPY_DEVICE_TO_DEVICE", 2, 2),
    "SvmMemcpy": ("MEMORY_COPY_DEVICE_TO_DEVICE", 2, 2),
}

# Fixed on-disk correlation lookup: API id, TID, stream id, grid xyz,
# workgroup xyz, dynamic LDS. Dense correlation ids make mmap lookup cheap.
META = struct.Struct("<IIQIIIIIIQ")
ZERO_META = (0,) * 10


def _peak_working_set() -> int:
    if os.name != "nt":
        return 0

    class ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_ulong),
            ("PageFaultCount", ctypes.c_ulong),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    counters = ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    psapi.GetProcessMemoryInfo.argtypes = (
        ctypes.c_void_p,
        ctypes.POINTER(ProcessMemoryCounters),
        ctypes.c_ulong,
    )
    psapi.GetProcessMemoryInfo.restype = ctypes.c_int
    process = kernel32.GetCurrentProcess()
    if psapi.GetProcessMemoryInfo(process, ctypes.byref(counters), counters.cb):
        return int(counters.PeakWorkingSetSize)
    return 0


def _ns(value: str) -> int:
    return int(round(float(value) * 1000.0))


def _q(value: object) -> str:
    # rocprofv3's CSV encoder quotes every string and does not RFC4180-escape
    # embedded quotes. HIP/kernel names do not normally contain one.
    return f'"{str(value)}"'


def _write_row(fh, values, string_columns=()):
    strings = set(string_columns)
    fh.write(
        ",".join(_q(v) if i in strings else str(v) for i, v in enumerate(values)) + "\n"
    )


def _write_header(fh, header):
    fh.write(",".join(_q(v) for v in header) + "\n")


class JsonFragment:
    """Comma-separated JSON object fragment, without surrounding brackets."""

    def __init__(self, path: Path):
        self.path = path
        self.fh = path.open("w", encoding="utf-8", newline="\n")
        self.count = 0

    def add(self, value):
        if self.count:
            self.fh.write(",")
        json.dump(value, self.fh, ensure_ascii=False, separators=(",", ":"))
        self.count += 1

    def close(self):
        self.fh.close()


class JsonObjectFragment(JsonFragment):
    """Comma-separated JSON object entries, without surrounding braces."""

    def add(self, key, value):
        if self.count:
            self.fh.write(",")
        self.fh.write(json.dumps(str(key)))
        self.fh.write(":")
        json.dump(value, self.fh, ensure_ascii=False, separators=(",", ":"))
        self.count += 1


def _copy_fragment(out, fragment: JsonFragment):
    fragment.close()
    object_fragment = isinstance(fragment, JsonObjectFragment)
    out.write("{" if object_fragment else "[")
    with fragment.path.open("r", encoding="utf-8") as src:
        while chunk := src.read(1024 * 1024):
            out.write(chunk)
    out.write("}" if object_fragment else "]")


def _parse_args(text: str) -> dict[str, str]:
    out = {}
    for item in (text or "").split():
        key, sep, value = item.partition("=")
        if sep:
            out[key] = value
    return out


def _dim(text: str | None) -> tuple[int, int, int]:
    try:
        x, y, z = (int(v, 0) for v in (text or "").split("x"))
        return x, y, z
    except (TypeError, ValueError):
        return 0, 0, 0


def _u64(text: str | None) -> int:
    try:
        return int(text or "0", 0)
    except ValueError:
        return 0


def _summary(path: Path) -> tuple[dict, str]:
    if not path.exists():
        return {}, ""
    text = path.read_text(encoding="utf-8", errors="replace")
    result = {}
    patterns = {
        "pid": r"^pid\s*:\s*(\d+)",
        "device_name": r"^device\s*:\s*(.+)$",
        "cu_count": r"^\s+CUs\s*:\s*(-?\d+)",
        "wave_front_size": r"^\s+warp size\s*:\s*(-?\d+)",
        "threads_per_cu": r"^\s+threads/CU\s*:\s*(-?\d+)",
        "local_mem_size": r"^\s+LDS/CU\s*:\s*(-?\d+)",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.MULTILINE)
        if match:
            result[key] = (
                int(match.group(1)) if key != "device_name" else match.group(1).strip()
            )
    clock = re.search(
        r"^\s+clock / mem\s*:\s*(-?\d+) kHz / (-?\d+) kHz, bus (-?\d+) bit",
        text,
        re.MULTILINE,
    )
    if clock:
        result["clock_khz"], result["memory_clock_khz"], result["memory_bus_width"] = (
            int(clock.group(1)),
            int(clock.group(2)),
            int(clock.group(3)),
        )
    return result, text


def _kernel_properties(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    with path.open(newline="") as fh:
        return {row["Kernel_Name"]: row for row in csv.DictReader(fh)}


def _agent_json(handle: int, kind: int, summary: dict) -> dict:
    gpu = kind == 2
    cu = max(1, int(summary.get("cu_count", 0))) if gpu else 0
    wave = max(1, int(summary.get("wave_front_size", 0))) if gpu else 0
    threads_per_cu = max(wave, int(summary.get("threads_per_cu", 0))) if gpu else 0
    waves_per_cu = max(1, threads_per_cu // wave) if gpu else 0
    waves_per_simd = max(1, waves_per_cu // 4) if gpu else 0
    name = (
        summary.get("device_name", "AMD GPU") if gpu else platform.processor() or "CPU"
    )
    capability_names = (
        "HotPluggable",
        "HSAMMUPresent",
        "SharedWithGraphics",
        "QueueSizePowerOfTwo",
        "QueueSize32bit",
        "QueueIdleEvent",
        "VALimit",
        "WatchPointsSupported",
        "WatchPointsTotalBits",
        "DoorbellType",
        "AQLQueueDoubleMap",
        "DebugTrapSupported",
        "WaveLaunchTrapOverrideSupported",
        "WaveLaunchModeSupported",
        "PreciseMemoryOperationsSupported",
        "DEPRECATED_SRAM_EDCSupport",
        "Mem_EDCSupport",
        "RASEventNotify",
        "ASICRevision",
        "SRAM_EDCSupport",
        "SVMAPISupported",
        "CoherentHostAccess",
        "DebugSupportedFirmware",
        "Reserved",
    )
    obj = {
        "size": 1,
        "id": {"handle": handle},
        "type": kind,
        "cpu_cores_count": (os.cpu_count() or 0) if not gpu else 0,
        "simd_count": cu * 4 if gpu else 0,
        "mem_banks_count": 0,
        "caches_count": 0,
        "io_links_count": 0,
        "cpu_core_id_base": 0,
        "simd_id_base": 0,
        "max_waves_per_simd": waves_per_simd,
        "lds_size_in_kb": int(summary.get("local_mem_size", 0)) // 1024 if gpu else 0,
        "gds_size_in_kb": 0,
        "num_gws": 0,
        "wave_front_size": wave,
        "num_xcc": 1 if gpu else 0,
        "cu_count": cu,
        "array_count": 0,
        "num_shader_banks": 0,
        "simd_arrays_per_engine": 0,
        "cu_per_simd_array": 0,
        "simd_per_cu": 4 if gpu else 0,
        "max_slots_scratch_cu": 0,
        "gfx_target_version": 0,
        "vendor_id": 0x1002 if gpu else 0,
        "device_id": 0,
        "location_id": 0,
        "domain": 0,
        "drm_render_minor": 0,
        "num_sdma_engines": 0,
        "num_sdma_xgmi_engines": 0,
        "num_sdma_queues_per_engine": 0,
        "num_cp_queues": 0,
        "max_engine_clk_ccompute": int(summary.get("clock_khz", 0)) // 1000
        if gpu
        else 0,
        "max_engine_clk_fcompute": 0,
        "sdma_fw_version": {"uCodeSDMA": 0, "uCodeRes": 0},
        "fw_version": {"uCode": 0, "Major": 0, "Minor": 0, "Stepping": 0},
        "capability": {key: 0 for key in capability_names},
        "cu_per_engine": 0,
        "max_waves_per_cu": waves_per_cu,
        "family_id": 0,
        "workgroup_max_size": 1024 if gpu else 0,
        "grid_max_size": 0,
        "local_mem_size": int(summary.get("local_mem_size", 0)) if gpu else 0,
        "hive_id": 0,
        "gpu_id": 0,
        "workgroup_max_dim": {
            "x": 1024 if gpu else 0,
            "y": 1024 if gpu else 0,
            "z": 1024 if gpu else 0,
        },
        "grid_max_dim": {"x": 0, "y": 0, "z": 0},
        "name": name,
        "vendor_name": "AMD" if gpu else "",
        "product_name": name,
        "model_name": name,
        "node_id": handle - 1,
        "logical_node_id": handle - 1,
        "logical_node_type_id": 0,
        "runtime_visibility": {"hsa": gpu, "hip": gpu, "rccl": gpu, "rocdecode": gpu},
        "uuid": {"bytes": [0] * 16},
        "mem_banks": [],
        "caches": [],
        "io_links": [],
        "gpu_index": 0 if gpu else -1,
    }
    return obj


def _agent_csv_rows(summary: dict):
    common = [0] * len(AGENT_HEADER)
    cpu = dict(zip(AGENT_HEADER, common))
    cpu.update(
        Node_Id=0,
        Logical_Node_Id=0,
        Agent_Type="CPU",
        Cpu_Cores_Count=os.cpu_count() or 0,
        Name=platform.processor() or "CPU",
        Vendor_Name="",
        Product_Name=platform.processor() or "CPU",
        Model_Name=platform.processor() or "CPU",
    )
    gpu = dict(zip(AGENT_HEADER, common))
    cu = max(1, int(summary.get("cu_count", 0)))
    wave = max(1, int(summary.get("wave_front_size", 0)))
    threads_per_cu = max(wave, int(summary.get("threads_per_cu", 0)))
    waves_per_cu = max(1, threads_per_cu // wave)
    waves_per_simd = max(1, waves_per_cu // 4)
    gpu.update(
        Node_Id=1,
        Logical_Node_Id=1,
        Agent_Type="GPU",
        Simd_Count=cu * 4,
        Max_Waves_Per_Simd=waves_per_simd,
        Lds_Size_In_Kb=int(summary.get("local_mem_size", 0)) // 1024,
        Wave_Front_Size=wave,
        Num_Xcc=1,
        Cu_Count=cu,
        Simd_Per_Cu=4,
        Vendor_Id=0x1002,
        Workgroup_Max_Size=1024,
        Workgroup_Max_Dim_X=1024,
        Workgroup_Max_Dim_Y=1024,
        Workgroup_Max_Dim_Z=1024,
        Name=summary.get("device_name", "AMD GPU"),
        Vendor_Name="AMD",
        Product_Name=summary.get("device_name", "AMD GPU"),
        Model_Name=summary.get("device_name", "AMD GPU"),
        Max_Engine_Clk_Ccompute=int(summary.get("clock_khz", 0)) // 1000,
        Local_Mem_Size=int(summary.get("local_mem_size", 0)),
        Max_Waves_Per_Cu=waves_per_cu,
    )
    return [[row[k] for k in AGENT_HEADER] for row in (cpu, gpu)]


def _kernel_symbol(kernel_id: int, name: str, props: dict) -> dict:
    vgpr = int(props.get("VGPR", 0) or 0)
    scratch = int(props.get("Scratch_Bytes", 0) or 0)
    lds = int(props.get("Static_LDS", 0) or 0)
    return {
        "size": 1,
        "kernel_id": kernel_id,
        "code_object_id": 0,
        "kernel_name": name,
        "kernel_object": 0,
        "kernarg_segment_size": 0,
        "kernarg_segment_alignment": 0,
        "group_segment_size": lds,
        "private_segment_size": scratch,
        "sgpr_count": 0,
        "arch_vgpr_count": vgpr,
        "accum_vgpr_count": 0,
        "kernel_code_entry_byte_offset": 0,
        "kernel_address": {"handle": 0},
        "formatted_kernel_name": name,
        "demangled_kernel_name": name,
        "truncated_kernel_name": name,
    }


def _name_tables(api_names: dict[int, dict[int, str]]) -> list[dict]:
    max_kind = max(HIP_COMPILER_KIND, MEMORY_ALLOCATION_KIND)
    tables = [{"kind": "", "operations": []} for _ in range(max_kind + 1)]

    def put(kind, name, operations):
        tables[kind] = {"kind": name, "operations": operations}

    put(
        MEMORY_COPY_KIND,
        "MEMORY_COPY",
        [
            "",
            "MEMORY_COPY_HOST_TO_HOST",
            "MEMORY_COPY_HOST_TO_DEVICE",
            "MEMORY_COPY_DEVICE_TO_HOST",
            "MEMORY_COPY_DEVICE_TO_DEVICE",
        ],
    )
    put(KERNEL_DISPATCH_KIND, "KERNEL_DISPATCH", ["", "", "KERNEL_DISPATCH_COMPLETE"])
    put(
        MEMORY_ALLOCATION_KIND,
        "MEMORY_ALLOCATION",
        [
            "",
            "MEMORY_ALLOCATION_ALLOCATE",
            "MEMORY_ALLOCATION_VMEM_ALLOCATE",
            "MEMORY_ALLOCATION_FREE",
            "MEMORY_ALLOCATION_VMEM_FREE",
        ],
    )
    for kind, kind_name in (
        (HIP_RUNTIME_KIND, "HIP_RUNTIME_API_EXT"),
        (HIP_COMPILER_KIND, "HIP_COMPILER_API_EXT"),
    ):
        names = api_names.get(kind, {})
        operations = [""] * ((max(names) + 1) if names else 0)
        for idx, name in names.items():
            operations[idx] = name
        put(kind, kind_name, operations)
    return tables


def _base_paths(ops_path: Path):
    return {
        "ops": ops_path,
        "api": Path(str(ops_path) + ".api.csv"),
        "kernels": Path(str(ops_path) + ".kernels.csv"),
        "vram": Path(str(ops_path) + ".vram.csv"),
        "roctx": Path(str(ops_path) + ".roctx.csv"),
        "launch": Path(str(ops_path) + ".launch.csv"),
        "summary": Path(str(ops_path) + ".summary.txt"),
    }


def _old_rows(path: Path, required: tuple[str, ...]):
    """Yield complete private-spool rows; a killed/detaching process can leave
    one final physical line incomplete, while every preceding batch is valid."""
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            if all(row.get(key) not in (None, "") for key in required):
                yield row


def export_capture(
    ops_path: Path, output_dir: Path, prefix: str | None, delete_spool=False
):
    paths = _base_paths(ops_path)
    if not paths["ops"].exists():
        raise FileNotFoundError(paths["ops"])
    output_dir.mkdir(parents=True, exist_ok=True)
    summary, summary_text = _summary(paths["summary"])
    pid = int(summary.get("pid", os.getpid()))
    launch_pid = pid
    resume_timestamp_ns = 0
    if paths["launch"].exists():
        with paths["launch"].open(newline="", encoding="utf-8-sig") as fh:
            launch = next(csv.DictReader(fh), None)
        if launch is None:
            raise ValueError(f"{paths['launch']}: empty launch metadata")
        launch_pid = int(launch.get("Process_Id") or launch_pid)
        resume_timestamp_ns = int(launch.get("Resume_Timestamp_ns") or 0)
    if not prefix:
        match = re.match(r"^\.(.+)_\d+_ops\.csv$", ops_path.name)
        prefix = match.group(1) if match else str(pid)
        # A second GPU-owning descendant must not overwrite the first process.
        if (output_dir / f"{prefix}_results.json").exists():
            prefix = f"{prefix}_{pid}"
    props = _kernel_properties(paths["kernels"])

    final_out = {
        name: output_dir / f"{prefix}_{name}"
        for name in (
            "agent_info.csv",
            "hip_api_trace.csv",
            "kernel_trace.csv",
            "kernel_trace_readable.csv",
            "memory_copy_trace.csv",
            "memory_allocation_trace.csv",
            "roctx_trace.csv",
            "ktrace_launch_info.csv",
            "results.json",
            "ktrace_extensions.json",
        )
    }
    temp_dir = Path(tempfile.mkdtemp(prefix=f".{prefix}_", dir=output_dir))
    out = {name: temp_dir / path.name for name, path in final_out.items()}

    with out["agent_info.csv"].open("w", encoding="utf-8", newline="\n") as fh:
        _write_header(fh, AGENT_HEADER)
        for row in _agent_csv_rows(summary):
            _write_row(fh, row, (2, 49, 50, 51, 52))
    if resume_timestamp_ns:
        with out["ktrace_launch_info.csv"].open(
            "w", encoding="utf-8", newline="\n"
        ) as fh:
            _write_header(fh, ("Process_Id", "Resume_Timestamp"))
            _write_row(fh, (str(launch_pid), str(resume_timestamp_ns)))

    fragments = {
        name: JsonFragment(temp_dir / f"{name}.frag")
        for name in ("hip", "kernel", "copy", "alloc", "unmapped", "copy_extras")
    }
    fragments["api_args"] = JsonObjectFragment(temp_dir / "api_args.frag")
    api_names: dict[int, dict[int, str]] = {HIP_RUNTIME_KIND: {}, HIP_COMPILER_KIND: {}}
    stream_ids = {0: 0}
    next_stream = 1
    meta_path = temp_dir / "corr.bin"
    max_corr = 0
    init_ns = 2**64 - 1
    fini_ns = 0

    api_count = 0
    alloc_count = 0
    with ExitStack() as stack:
        hip_csv = stack.enter_context(
            out["hip_api_trace.csv"].open("w", encoding="utf-8", newline="\n")
        )
        alloc_csv = stack.enter_context(
            out["memory_allocation_trace.csv"].open("w", encoding="utf-8", newline="\n")
        )
        meta_fh = stack.enter_context(meta_path.open("w+b"))
        _write_header(hip_csv, HIP_HEADER)
        _write_header(alloc_csv, ALLOC_HEADER)
        if paths["api"].exists():
            for row in _old_rows(
                paths["api"], ("Corr", "Api", "Api_Id", "Thread", "Enter_us", "Exit_us")
            ):
                corr = int(row["Corr"])
                api_id = int(row["Api_Id"])
                tid = int(row["Thread"])
                start, end = _ns(row["Enter_us"]), _ns(row["Exit_us"])
                init_ns, fini_ns = min(init_ns, start), max(fini_ns, end)
                function = row["Api"]
                kind = (
                    HIP_COMPILER_KIND
                    if function.startswith("__hip")
                    else HIP_RUNTIME_KIND
                )
                domain = (
                    "HIP_COMPILER_API_EXT"
                    if kind == HIP_COMPILER_KIND
                    else "HIP_RUNTIME_API_EXT"
                )
                api_names[kind][api_id] = function
                _write_row(
                    hip_csv, [domain, function, pid, tid, corr, start, end], (0, 1)
                )
                api_count += 1

                args = _parse_args(row.get("Args", ""))
                raw_stream = _u64(args.get("stream"))
                if raw_stream not in stream_ids:
                    stream_ids[raw_stream] = next_stream
                    next_stream += 1
                stream = stream_ids[raw_stream]
                grid, block = _dim(args.get("grid")), _dim(args.get("block"))
                shmem = _u64(args.get("shmem"))
                meta_fh.seek(corr * META.size)
                meta_fh.write(META.pack(api_id, tid, stream, *grid, *block, shmem))
                max_corr = max(max_corr, corr)

                arg_list = [
                    {"type": "uint64_t", "name": k, "value": v} for k, v in args.items()
                ]
                fragments["hip"].add(
                    {
                        "size": 1,
                        "kind": kind,
                        "operation": api_id,
                        "correlation_id": {
                            "internal": corr,
                            "external": 0,
                            "ancestor": 0,
                        },
                        "start_timestamp": start,
                        "end_timestamp": end,
                        "thread_id": tid,
                        "args": arg_list,
                        "retval": {"uint64_t_retval": 0},
                        "stream_id": {"handle": stream},
                    }
                )
                if args:
                    fragments["api_args"].add(corr, row.get("Args", ""))

                allocate = function in ("hipMalloc", "hipHostMalloc")
                free = function in ("hipFree", "hipHostFree")
                if allocate or free:
                    host = "Host" in function
                    op = ALLOCATE if allocate else FREE
                    op_name = (
                        "MEMORY_ALLOCATION_ALLOCATE"
                        if allocate
                        else "MEMORY_ALLOCATION_FREE"
                    )
                    agent_display = (
                        "Agent 0"
                        if (host and allocate)
                        else ("Agent 1" if allocate else "")
                    )
                    agent_handle = 1 if (host and allocate) else (2 if allocate else 0)
                    size = _u64(args.get("size")) if allocate else 0
                    address = _u64(args.get("ptr"))
                    address_s = f"0x{address:016x}"
                    _write_row(
                        alloc_csv,
                        [
                            "MEMORY_ALLOCATION",
                            op_name,
                            agent_display,
                            size,
                            address_s,
                            corr,
                            start,
                            end,
                        ],
                        (0, 1, 2, 4),
                    )
                    fragments["alloc"].add(
                        {
                            "size": 1,
                            "kind": MEMORY_ALLOCATION_KIND,
                            "operation": op,
                            "thread_id": tid,
                            "correlation_id": {
                                "internal": corr,
                                "external": 0,
                                "ancestor": 0,
                            },
                            "start_timestamp": start,
                            "end_timestamp": end,
                            "agent_id": {"handle": agent_handle},
                            "address": address_s,
                            "allocation_size": size,
                            "stream_id": {"handle": stream},
                        }
                    )
                    alloc_count += 1

    if api_count == 0:
        out["hip_api_trace.csv"].unlink(missing_ok=True)
    if alloc_count == 0:
        out["memory_allocation_trace.csv"].unlink(missing_ok=True)

    roctx_count = 0
    if paths["roctx"].exists():
        with out["roctx_trace.csv"].open("w", encoding="utf-8", newline="\n") as fh:
            _write_header(fh, ROCTX_HEADER)
            for row in _old_rows(
                paths["roctx"], ("Op", "Process", "Thread", "Range_Id", "Timestamp_us")
            ):
                _write_row(
                    fh,
                    [
                        "ROCTX",
                        row["Op"],
                        int(row["Process"]),
                        int(row["Thread"]),
                        int(row["Range_Id"]),
                        _ns(row["Timestamp_us"]),
                        row["Message"],
                    ],
                    (0, 1, 6),
                )
                roctx_count += 1
    if roctx_count == 0:
        out["roctx_trace.csv"].unlink(missing_ok=True)

    # Ensure an mmap can cover every correlation id referenced by an op.
    max_op_corr = 0
    raw_kernel_names = []
    seen_kernel_names = set()
    for row in _old_rows(paths["ops"], ("Corr", "Begin_us", "End_us")):
        max_op_corr = max(max_op_corr, int(row["Corr"]))
        name = row.get("Kernel_Name", "") if row.get("Op") == "dispatch" else ""
        if name and name not in seen_kernel_names:
            seen_kernel_names.add(name)
            raw_kernel_names.append(name)
    readable_names = resolve_names(raw_kernel_names)
    map_size = (max(max_corr, max_op_corr) + 1) * META.size
    with meta_path.open("r+b") as fh:
        fh.truncate(max(1, map_size))

    kernel_ids = {}
    dispatch_id = 0
    copy_count = 0
    next_synthetic_corr = max_corr + 1
    with ExitStack() as stack:
        kernel_csv = stack.enter_context(
            out["kernel_trace.csv"].open("w", encoding="utf-8", newline="\n")
        )
        readable_kernel_csv = stack.enter_context(
            out["kernel_trace_readable.csv"].open("w", encoding="utf-8", newline="\n")
        )
        copy_csv = stack.enter_context(
            out["memory_copy_trace.csv"].open("w", encoding="utf-8", newline="\n")
        )
        meta_fh = stack.enter_context(meta_path.open("r+b"))
        meta_map = stack.enter_context(
            mmap.mmap(meta_fh.fileno(), 0, access=mmap.ACCESS_READ)
        )
        _write_header(kernel_csv, KERNEL_HEADER)
        _write_header(readable_kernel_csv, KERNEL_HEADER)
        _write_header(copy_csv, COPY_HEADER)

        for row in _old_rows(
            paths["ops"],
            ("Op", "Kind", "Bytes", "Device", "Queue", "Corr", "Begin_us", "End_us"),
        ):
            raw_corr = int(row["Corr"])
            corr = raw_corr
            if corr == 0:
                corr = next_synthetic_corr
                next_synthetic_corr += 1
            start, end = _ns(row["Begin_us"]), _ns(row["End_us"])
            init_ns, fini_ns = min(init_ns, start), max(fini_ns, end)
            meta = ZERO_META
            off = raw_corr * META.size
            if raw_corr and off + META.size <= len(meta_map):
                meta = META.unpack_from(meta_map, off)
            _, tid, stream, gx, gy, gz, bx, by, bz, dyn_lds = meta
            if tid == 0:
                tid = pid

            if row["Op"] == "dispatch":
                dispatch_id += 1
                name = row["Kernel_Name"]
                kernel_id = kernel_ids.setdefault(name, len(kernel_ids) + 1)
                kp = props.get(name, {})
                static_lds = int(kp.get("Static_LDS", 0) or 0)
                lds = (
                    (static_lds + dyn_lds + 511) & ~511 if static_lds or dyn_lds else 0
                )
                scratch = int(kp.get("Scratch_Bytes", 0) or 0)
                vgpr = int(kp.get("VGPR", 0) or 0)
                queue = int(row["Queue"])
                values = [
                    "KERNEL_DISPATCH",
                    "Agent 1",
                    queue,
                    stream,
                    tid,
                    dispatch_id,
                    kernel_id,
                    name,
                    corr,
                    start,
                    end,
                    lds,
                    scratch,
                    vgpr,
                    0,
                    0,
                    bx,
                    by,
                    bz,
                    gx,
                    gy,
                    gz,
                ]
                _write_row(kernel_csv, values, (0, 1, 7))
                readable_values = list(values)
                readable_values[7] = readable_names.get(name, name)
                _write_row(readable_kernel_csv, readable_values, (0, 1, 7))
                fragments["kernel"].add(
                    {
                        "size": 1,
                        "kind": KERNEL_DISPATCH_KIND,
                        "operation": KERNEL_DISPATCH_COMPLETE,
                        "thread_id": tid,
                        "correlation_id": {"internal": corr, "external": 0},
                        "start_timestamp": start,
                        "end_timestamp": end,
                        "dispatch_info": {
                            "size": 1,
                            "agent_id": {"handle": 2},
                            "queue_id": {"handle": queue},
                            "kernel_id": kernel_id,
                            "dispatch_id": dispatch_id,
                            "private_segment_size": scratch,
                            "group_segment_size": static_lds + dyn_lds,
                            "workgroup_size": {"x": bx, "y": by, "z": bz},
                            "grid_size": {"x": gx, "y": gy, "z": gz},
                        },
                        "stream_id": {"handle": stream},
                    }
                )
            elif row["Op"] == "copy" and row["Kind"] in COPY_DIRECTIONS:
                direction, src_agent, dst_agent = COPY_DIRECTIONS[row["Kind"]]
                op = {
                    "MEMORY_COPY_HOST_TO_HOST": 1,
                    "MEMORY_COPY_HOST_TO_DEVICE": 2,
                    "MEMORY_COPY_DEVICE_TO_HOST": 3,
                    "MEMORY_COPY_DEVICE_TO_DEVICE": 4,
                }[direction]
                _write_row(
                    copy_csv,
                    [
                        "MEMORY_COPY",
                        direction,
                        stream,
                        f"Agent {src_agent - 1}",
                        f"Agent {dst_agent - 1}",
                        corr,
                        start,
                        end,
                    ],
                    (0, 1, 3, 4),
                )
                fragments["copy"].add(
                    {
                        "size": 1,
                        "kind": MEMORY_COPY_KIND,
                        "operation": op,
                        "thread_id": tid,
                        "correlation_id": {"internal": corr, "external": 0},
                        "start_timestamp": start,
                        "end_timestamp": end,
                        "dst_agent_id": {"handle": dst_agent},
                        "src_agent_id": {"handle": src_agent},
                        "bytes": int(row["Bytes"]),
                        "stream_id": {"handle": stream},
                    }
                )
                fragments["copy_extras"].add(
                    {
                        "correlation_id": corr,
                        "start_timestamp": start,
                        "bytes": int(row["Bytes"]),
                        "device": "Agent 1",
                        "queue": int(row["Queue"]),
                        "stream": stream,
                    }
                )
                copy_count += 1
            else:
                fragments["unmapped"].add(
                    {
                        "op": row["Op"],
                        "kind": row["Kind"],
                        "bytes": int(row["Bytes"]),
                        "device": "Agent 1",
                        "queue": int(row["Queue"]),
                        "correlation_id": corr,
                        "start_timestamp": start,
                        "end_timestamp": end,
                    }
                )

    if dispatch_id == 0:
        out["kernel_trace.csv"].unlink(missing_ok=True)
        out["kernel_trace_readable.csv"].unlink(missing_ok=True)
    if copy_count == 0:
        out["memory_copy_trace.csv"].unlink(missing_ok=True)

    kernel_symbols = [
        {
            "size": 1,
            "kernel_id": 0,
            "code_object_id": 0,
            "kernel_name": "<unknown>",
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
            "formatted_kernel_name": "<unknown>",
            "demangled_kernel_name": "<unknown>",
            "truncated_kernel_name": "<unknown>",
        }
    ]
    by_id = sorted(kernel_ids.items(), key=lambda item: item[1])
    kernel_symbols.extend(
        _kernel_symbol(kid, name, props.get(name, {})) for name, kid in by_id
    )

    name_tables = _name_tables(api_names)
    metadata = {
        "node": {
            "id": 0,
            "hash": 0,
            "machine_id": "",
            "system_name": platform.system(),
            "hostname": platform.node(),
            "release": platform.release(),
            "version": platform.version(),
            "hardware_name": platform.machine(),
            "domain_name": "",
        },
        "pid": pid,
        "init_time": (
            resume_timestamp_ns
            if resume_timestamp_ns
            else (0 if init_ns == 2**64 - 1 else init_ns)
        ),
        "fini_time": fini_ns,
        "command": [],
        "config": {
            "output_path": str(output_dir),
            "output_file": prefix,
            "tmp_directory": str(temp_dir),
            "raw_output_path": str(ops_path.parent),
            "raw_output_file": ops_path.name,
            "raw_tmp_directory": str(temp_dir),
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
        },
    }

    for fragment in fragments.values():
        if fragment.fh and not fragment.fh.closed:
            fragment.close()

    with out["results.json"].open("w", encoding="utf-8", newline="\n") as fh:
        fh.write('{"rocprofiler-sdk-tool":[{')
        fh.write('"metadata":')
        json.dump(metadata, fh, ensure_ascii=False, separators=(",", ":"))
        fh.write(',"agents":')
        json.dump(
            [_agent_json(1, 1, summary), _agent_json(2, 2, summary)],
            fh,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        fh.write(',"counters":[],"strings":')
        json.dump(
            {
                "callback_records": [],
                "buffer_records": name_tables,
                "marker_api": [],
                "correlation_id": {"external": []},
                "counters": {"dimension_ids": []},
                "pc_sample_instructions": [],
                "pc_sample_comments": [],
                "att_filenames": [],
                "code_object_snapshot_filenames": [],
            },
            fh,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        fh.write(',"code_objects":[],"kernel_symbols":')
        json.dump(kernel_symbols, fh, ensure_ascii=False, separators=(",", ":"))
        fh.write(',"host_functions":[],"summary":[]')
        fh.write(',"callback_records":{"counter_collection":[]}')
        fh.write(',"buffer_records":{')
        domains = [
            ("kernel_dispatch", "kernel"),
            ("hip_api", "hip"),
            ("hsa_api", None),
            ("marker_api", None),
            ("rccl_api", None),
            ("memory_copy", "copy"),
            ("memory_allocation", "alloc"),
            ("scratch_memory", None),
            ("rocdecode_api", None),
            ("rocjpeg_api", None),
            ("pc_sample_host_trap", None),
            ("pc_sample_stochastic", None),
        ]
        for i, (domain, fragment_name) in enumerate(domains):
            if i:
                fh.write(",")
            fh.write(json.dumps(domain) + ":")
            if fragment_name:
                _copy_fragment(fh, fragments[fragment_name])
            else:
                fh.write("[]")
        fh.write("}}]}\n")

    vram = []
    if paths["vram"].exists():
        with paths["vram"].open(newline="") as fh:
            vram = [
                {
                    "timestamp": _ns(r["Time_us"]),
                    "free_bytes": int(r["Free_Bytes"]),
                    "total_bytes": int(r["Total_Bytes"]),
                    "used_bytes": int(r["Used_Bytes"]),
                }
                for r in csv.DictReader(fh)
            ]
    extension = {
        "schema": "ktrace-extensions-1.1",
        "rocprofiler_schema": "ROCm 7.2.1 / ROCprofiler-SDK 1.1.0",
        "source_spool": str(ops_path),
        "process_resume_timestamp": resume_timestamp_ns,
        "field_quality": {
            "timestamps": "exact_qpc_nanoseconds",
            "queue_id": "exact_rocclr_hardware_queue",
            "correlation_id": "exact_hip_runtime_when_present_else_synthetic_gpu_record_id",
            "stream_id": "synthetic_session_id_from_hip_stream_pointer",
            "thread_id": "exact_from_correlated_hip_api_else_synthetic_process_id",
            "agent_id": "synthetic_cpu_gpu_topology",
            "dispatch_id": "synthetic_monotonic",
            "kernel_id": "synthetic_by_kernel_name",
            "sgpr_count": "unavailable_zero",
            "accum_vgpr_count": "unavailable_zero",
            "abi_size_fields": "synthetic_nonzero_windows_records_are_not_sdk_structs",
            "memory_allocation": "hip_api_level_not_hsa_allocation_trace",
        },
        "kernel_properties": list(props.values()),
        "vram_samples": vram,
        "capture_summary": summary_text,
        "api_arguments": None,
        "memory_copy_extras": None,
        "unmapped_gpu_commands": None,
    }
    # Stream the two potentially large extension arrays into an otherwise small object.
    with out["ktrace_extensions.json"].open("w", encoding="utf-8", newline="\n") as fh:
        head = dict(extension)
        head.pop("api_arguments")
        head.pop("memory_copy_extras")
        head.pop("unmapped_gpu_commands")
        text = json.dumps(head, ensure_ascii=False, separators=(",", ":"))
        fh.write(text[:-1])
        fh.write(',"memory_copy_extras":')
        _copy_fragment(fh, fragments["copy_extras"])
        fh.write(',"unmapped_gpu_commands":')
        _copy_fragment(fh, fragments["unmapped"])
        # Keep the multi-million-entry API array last. Readers that only need
        # GPU operations, VRAM or kernel properties can stop before scanning it.
        fh.write(',"api_arguments":')
        _copy_fragment(fh, fragments["api_args"])
        fh.write("}\n")

    for fragment in fragments.values():
        fragment.path.unlink(missing_ok=True)
    meta_path.unlink(missing_ok=True)

    # Publish the complete set atomically, one file at a time. A failed
    # conversion leaves only the hidden temp directory and the recoverable
    # private spools, never a public file that looks complete.
    for name, temp_path in out.items():
        final_path = final_out[name]
        if temp_path.exists():
            os.replace(temp_path, final_path)
        else:
            final_path.unlink(missing_ok=True)
    try:
        temp_dir.rmdir()
    except OSError:
        pass

    if delete_spool:
        for path in paths.values():
            path.unlink(missing_ok=True)

    return final_out


def finalize_directory(capture_dir: Path) -> list[dict[str, Path]]:
    """Finalize every private process spool in *capture_dir*.

    One application tree can contain multiple GPU-owning processes. The first
    process uses the requested output prefix; later processes receive a PID
    suffix via :func:`export_capture`'s collision handling.
    """
    capture_dir = capture_dir.resolve()
    spools = sorted(capture_dir.glob(".*_ops.csv"))
    if not spools:
        raise FileNotFoundError(f"no private ktrace spool found in {capture_dir}")

    outputs = []
    private_prefixes = set()
    for spool in spools:
        match = re.match(r"^\.(.+)_\d+_ops\.csv$", spool.name)
        if match:
            private_prefixes.add(match.group(1))
        print(f"[finalize] {spool.name}", flush=True)
        outputs.append(export_capture(spool, capture_dir, None, delete_spool=True))

    # Temp directories are named .<output-prefix>_<random>. Private spools are
    # files, so removing only hidden directories is safe after every export
    # above has succeeded.
    for path in capture_dir.iterdir():
        if path.is_dir() and any(
            path.name.startswith(f".{prefix}_") for prefix in private_prefixes
        ):
            shutil.rmtree(path, ignore_errors=True)
    return outputs


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="ktrace private spool -> ROCm 7.2.1 rocprofv3 output"
    )
    ap.add_argument("ops_csv", type=Path, help="private ktrace *_ops.csv spool")
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument(
        "--output-file", help="rocprofv3 output prefix (-o equivalent; default PID)"
    )
    ap.add_argument("--delete-spool", action="store_true")
    args = ap.parse_args(argv)
    outputs = export_capture(
        args.ops_csv, args.output_dir, args.output_file, args.delete_spool
    )
    for path in outputs.values():
        if path.exists():
            print(f"-> {path} ({path.stat().st_size / 1e6:.1f} MB)")
    peak = _peak_working_set()
    if peak:
        print(f"peak working set: {peak / 1048576:.1f} MiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
