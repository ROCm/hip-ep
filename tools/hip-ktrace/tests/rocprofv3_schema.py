#!/usr/bin/env python3
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""ROCm 7.2.1 rocprofv3 CSV/JSON compatibility validator.

The field lists and validation rules are derived from the MIT-licensed AMD
rocprofiler-sdk serializers and tests listed in THIRD_PARTY_NOTICES.md. This
file retains AMD's MIT attribution:

Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import sys
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


class ValidationError(ValueError):
    """Raised when output is not ROCm 7.2.1 rocprofv3-compatible."""


AGENT_INFO_HEADER = (
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
)

HIP_API_HEADER = (
    "Domain",
    "Function",
    "Process_Id",
    "Thread_Id",
    "Correlation_Id",
    "Start_Timestamp",
    "End_Timestamp",
)

KERNEL_HEADER = (
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
)

MEMORY_COPY_HEADER = (
    "Kind",
    "Direction",
    "Stream_Id",
    "Source_Agent_Id",
    "Destination_Agent_Id",
    "Correlation_Id",
    "Start_Timestamp",
    "End_Timestamp",
)

MEMORY_ALLOCATION_HEADER = (
    "Kind",
    "Operation",
    "Agent_Id",
    "Allocation_Size",
    "Address",
    "Correlation_Id",
    "Start_Timestamp",
    "End_Timestamp",
)

CSV_HEADERS = {
    "agent_info": AGENT_INFO_HEADER,
    "hip_api": HIP_API_HEADER,
    "kernel_dispatch": KERNEL_HEADER,
    "memory_copy": MEMORY_COPY_HEADER,
    "memory_allocation": MEMORY_ALLOCATION_HEADER,
}

CSV_STRING_COLUMNS = {
    "agent_info": {
        "Agent_Type",
        "Name",
        "Vendor_Name",
        "Product_Name",
        "Model_Name",
    },
    "hip_api": {"Domain", "Function"},
    "kernel_dispatch": {"Kind", "Agent_Id", "Kernel_Name"},
    "memory_copy": {
        "Kind",
        "Direction",
        "Source_Agent_Id",
        "Destination_Agent_Id",
    },
    "memory_allocation": {"Kind", "Operation", "Agent_Id", "Address"},
}

# rocprofiler_buffer_tracing_kind_t values from ROCm 7.2.1 fwd.h.
KIND_MEMORY_COPY = 10
KIND_KERNEL_DISPATCH = 11
KIND_MEMORY_ALLOCATION = 16
KIND_HIP_RUNTIME_EXT = 21
KIND_HIP_COMPILER_EXT = 22

MEMORY_COPY_OPERATIONS = {
    1: "MEMORY_COPY_HOST_TO_HOST",
    2: "MEMORY_COPY_HOST_TO_DEVICE",
    3: "MEMORY_COPY_DEVICE_TO_HOST",
    4: "MEMORY_COPY_DEVICE_TO_DEVICE",
}
MEMORY_ALLOCATION_OPERATIONS = {
    1: "MEMORY_ALLOCATION_ALLOCATE",
    2: "MEMORY_ALLOCATION_VMEM_ALLOCATE",
    3: "MEMORY_ALLOCATION_FREE",
    4: "MEMORY_ALLOCATION_VMEM_FREE",
}

PROCESS_KEYS = {
    "metadata",
    "agents",
    "counters",
    "strings",
    "code_objects",
    "kernel_symbols",
    "host_functions",
    "summary",
    "callback_records",
    "buffer_records",
}
STRINGS_KEYS = {
    "callback_records",
    "buffer_records",
    "marker_api",
    "correlation_id",
    "counters",
    "pc_sample_instructions",
    "pc_sample_comments",
    "att_filenames",
    "code_object_snapshot_filenames",
}
BUFFER_RECORD_KEYS = {
    "kernel_dispatch",
    "hip_api",
    "hsa_api",
    "marker_api",
    "rccl_api",
    "memory_copy",
    "memory_allocation",
    "scratch_memory",
    "rocdecode_api",
    "rocjpeg_api",
    "pc_sample_host_trap",
    "pc_sample_stochastic",
}

HIP_RECORD_KEYS = {
    "size",
    "kind",
    "operation",
    "correlation_id",
    "start_timestamp",
    "end_timestamp",
    "thread_id",
    "args",
    "retval",
    "stream_id",
}
KERNEL_RECORD_KEYS = {
    "size",
    "kind",
    "operation",
    "thread_id",
    "correlation_id",
    "start_timestamp",
    "end_timestamp",
    "dispatch_info",
    "stream_id",
}
MEMORY_COPY_RECORD_KEYS = {
    "size",
    "kind",
    "operation",
    "thread_id",
    "correlation_id",
    "start_timestamp",
    "end_timestamp",
    "dst_agent_id",
    "src_agent_id",
    "bytes",
    "stream_id",
}
MEMORY_ALLOCATION_RECORD_KEYS = {
    "size",
    "kind",
    "operation",
    "thread_id",
    "correlation_id",
    "start_timestamp",
    "end_timestamp",
    "agent_id",
    "address",
    "allocation_size",
    "stream_id",
}
DISPATCH_INFO_KEYS = {
    "size",
    "agent_id",
    "queue_id",
    "kernel_id",
    "dispatch_id",
    "private_segment_size",
    "group_segment_size",
    "workgroup_size",
    "grid_size",
}
CORRELATION_KEYS = {"internal", "external", "ancestor"}
ASYNC_CORRELATION_KEYS = {"internal", "external"}
KERNEL_SYMBOL_KEYS = {
    "size",
    "kernel_id",
    "code_object_id",
    "kernel_name",
    "kernel_object",
    "kernarg_segment_size",
    "kernarg_segment_alignment",
    "group_segment_size",
    "private_segment_size",
    "sgpr_count",
    "arch_vgpr_count",
    "accum_vgpr_count",
    "kernel_code_entry_byte_offset",
    "kernel_address",
    "formatted_kernel_name",
    "demangled_kernel_name",
    "truncated_kernel_name",
}
AGENT_KEYS = {
    "size",
    "id",
    "type",
    "cpu_cores_count",
    "simd_count",
    "mem_banks_count",
    "caches_count",
    "io_links_count",
    "cpu_core_id_base",
    "simd_id_base",
    "max_waves_per_simd",
    "lds_size_in_kb",
    "gds_size_in_kb",
    "num_gws",
    "wave_front_size",
    "num_xcc",
    "cu_count",
    "array_count",
    "num_shader_banks",
    "simd_arrays_per_engine",
    "cu_per_simd_array",
    "simd_per_cu",
    "max_slots_scratch_cu",
    "gfx_target_version",
    "vendor_id",
    "device_id",
    "location_id",
    "domain",
    "drm_render_minor",
    "num_sdma_engines",
    "num_sdma_xgmi_engines",
    "num_sdma_queues_per_engine",
    "num_cp_queues",
    "max_engine_clk_ccompute",
    "max_engine_clk_fcompute",
    "sdma_fw_version",
    "fw_version",
    "capability",
    "cu_per_engine",
    "max_waves_per_cu",
    "family_id",
    "workgroup_max_size",
    "grid_max_size",
    "local_mem_size",
    "hive_id",
    "gpu_id",
    "workgroup_max_dim",
    "grid_max_dim",
    "name",
    "vendor_name",
    "product_name",
    "model_name",
    "node_id",
    "logical_node_id",
    "logical_node_type_id",
    "runtime_visibility",
    "uuid",
    "mem_banks",
    "caches",
    "io_links",
    "gpu_index",
}

_UINT_RE = re.compile(r"^[0-9]+$")
_ADDRESS_RE = re.compile(r"^0x[0-9a-f]{16}$")
_AGENT_LABEL_RE = re.compile(r"^(Agent|CPU|GPU) ([0-9]+)$")


@dataclass(frozen=True)
class OutputPaths:
    agent_info: Path
    hip_api: Path
    kernel_dispatch: Path
    memory_copy: Path
    memory_allocation: Path
    json: Path

    @classmethod
    def from_directory(cls, directory: Path, prefix: str) -> "OutputPaths":
        return cls(
            directory / f"{prefix}_agent_info.csv",
            directory / f"{prefix}_hip_api_trace.csv",
            directory / f"{prefix}_kernel_trace.csv",
            directory / f"{prefix}_memory_copy_trace.csv",
            directory / f"{prefix}_memory_allocation_trace.csv",
            directory / f"{prefix}_results.json",
        )


@dataclass
class CsvTable:
    rows: list[dict[str, str]]


def _fail(message: str) -> None:
    raise ValidationError(message)


def _exact_keys(value: Any, expected: set[str], where: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        _fail(f"{where}: expected object, got {type(value).__name__}")
    actual = set(value)
    if actual != expected:
        _fail(
            f"{where}: field mismatch; missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )
    return value


def _list(value: Any, where: str) -> list[Any]:
    if not isinstance(value, list):
        _fail(f"{where}: expected array, got {type(value).__name__}")
    return value


def _string(value: Any, where: str, *, nonempty: bool = False) -> str:
    if not isinstance(value, str):
        _fail(f"{where}: expected string, got {type(value).__name__}")
    if nonempty and not value:
        _fail(f"{where}: expected non-empty string")
    return value


def _uint(value: Any, where: str, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        _fail(f"{where}: expected unsigned JSON integer, got {value!r}")
    if positive and value == 0:
        _fail(f"{where}: expected positive JSON integer")
    return value


def _uint_text(value: str, where: str, *, positive: bool = False) -> int:
    if not _UINT_RE.fullmatch(value):
        _fail(f"{where}: expected unsigned decimal integer, got {value!r}")
    result = int(value)
    if positive and result == 0:
        _fail(f"{where}: expected positive integer")
    return result


def _handle(value: Any, where: str, *, positive: bool = False) -> int:
    obj = _exact_keys(value, {"handle"}, where)
    return _uint(obj["handle"], f"{where}.handle", positive=positive)


def _dim3(value: Any, where: str, *, positive: bool = False) -> tuple[int, int, int]:
    obj = _exact_keys(value, {"x", "y", "z"}, where)
    return tuple(
        _uint(obj[key], f"{where}.{key}", positive=positive) for key in ("x", "y", "z")
    )  # type: ignore[return-value]


def _quote(value: str) -> str:
    if '"' in value or "\r" in value or "\n" in value:
        _fail("rocprofiler-sdk csv.hpp cannot encode quotes/newlines in string fields")
    return f'"{value}"'


def _read_csv(path: Path, table_name: str, *, required: bool = False) -> CsvTable:
    if not path.is_file():
        if required:
            _fail(f"{table_name}: file not found: {path}")
        return CsvTable([])
    text = path.read_text(encoding="utf-8")
    physical_lines = text.splitlines()
    if not physical_lines:
        _fail(f"{table_name}: empty CSV file")
    try:
        parsed = list(csv.reader(io.StringIO(text, newline=""), strict=True))
    except csv.Error as exc:
        _fail(f"{table_name}: malformed CSV: {exc}")
    if len(parsed) != len(physical_lines):
        _fail(
            f"{table_name}: embedded newlines are not emitted by rocprofiler-sdk csv.hpp"
        )

    header = CSV_HEADERS[table_name]
    if tuple(parsed[0]) != header:
        _fail(f"{table_name}: exact header mismatch: {parsed[0]!r}")
    expected_header = ",".join(_quote(column) for column in header)
    if physical_lines[0] != expected_header:
        _fail(f"{table_name}: header quoting mismatch")

    rows: list[dict[str, str]] = []
    string_columns = CSV_STRING_COLUMNS[table_name]
    for line_number, (raw, values) in enumerate(zip(physical_lines[1:], parsed[1:]), 2):
        if len(values) != len(header):
            _fail(
                f"{table_name}:{line_number}: expected {len(header)} columns, "
                f"got {len(values)}"
            )
        expected_raw = ",".join(
            _quote(value) if column in string_columns else value
            for column, value in zip(header, values)
        )
        if raw != expected_raw:
            _fail(
                f"{table_name}:{line_number}: exact string quoting/numeric formatting mismatch"
            )
        rows.append(dict(zip(header, values)))
    if not rows and required:
        _fail(f"{table_name}: expected at least one data row")
    return CsvTable(rows)


def _validate_csv_tables(paths: OutputPaths) -> dict[str, CsvTable]:
    tables = {
        "agent_info": _read_csv(paths.agent_info, "agent_info", required=True),
        "hip_api": _read_csv(paths.hip_api, "hip_api"),
        "kernel_dispatch": _read_csv(paths.kernel_dispatch, "kernel_dispatch"),
        "memory_copy": _read_csv(paths.memory_copy, "memory_copy"),
        "memory_allocation": _read_csv(paths.memory_allocation, "memory_allocation"),
    }

    for index, row in enumerate(tables["agent_info"].rows):
        where = f"agent_info[{index}]"
        for column in AGENT_INFO_HEADER:
            if column not in CSV_STRING_COLUMNS["agent_info"]:
                _uint_text(row[column], f"{where}.{column}")
        if row["Agent_Type"] not in {"CPU", "GPU"}:
            _fail(f"{where}.Agent_Type: expected CPU or GPU")
        cpu_cores = int(row["Cpu_Cores_Count"])
        simds = int(row["Simd_Count"])
        waves = int(row["Max_Waves_Per_Simd"])
        if row["Agent_Type"] == "CPU" and (cpu_cores <= 0 or simds != 0 or waves != 0):
            _fail(f"{where}: invalid CPU core/SIMD/wave counts")
        if row["Agent_Type"] == "GPU" and (cpu_cores != 0 or simds <= 0 or waves <= 0):
            _fail(f"{where}: invalid GPU core/SIMD/wave counts")

    for index, row in enumerate(tables["hip_api"].rows):
        where = f"hip_api[{index}]"
        if row["Domain"] not in {"HIP_RUNTIME_API_EXT", "HIP_COMPILER_API_EXT"}:
            _fail(f"{where}.Domain: rocprofv3 7.2.1 traces HIP EXT kinds")
        if not row["Function"]:
            _fail(f"{where}.Function: expected non-empty function name")
        for column in (
            "Process_Id",
            "Thread_Id",
            "Correlation_Id",
            "Start_Timestamp",
            "End_Timestamp",
        ):
            _uint_text(row[column], f"{where}.{column}", positive=True)
        if int(row["End_Timestamp"]) < int(row["Start_Timestamp"]):
            _fail(f"{where}: end timestamp precedes start timestamp")

    for index, row in enumerate(tables["kernel_dispatch"].rows):
        where = f"kernel_dispatch[{index}]"
        if row["Kind"] != "KERNEL_DISPATCH":
            _fail(f"{where}.Kind: expected KERNEL_DISPATCH")
        _agent_label(row["Agent_Id"], f"{where}.Agent_Id")
        if not row["Kernel_Name"]:
            _fail(f"{where}.Kernel_Name: expected non-empty name")
        for column in KERNEL_HEADER:
            if column not in CSV_STRING_COLUMNS["kernel_dispatch"]:
                _uint_text(row[column], f"{where}.{column}")
        for column in (
            "Queue_Id",
            "Dispatch_Id",
            "Kernel_Id",
            "Correlation_Id",
            "Start_Timestamp",
            "End_Timestamp",
        ):
            _uint_text(row[column], f"{where}.{column}", positive=True)
        if int(row["End_Timestamp"]) < int(row["Start_Timestamp"]):
            _fail(f"{where}: end timestamp precedes start timestamp")

    for index, row in enumerate(tables["memory_copy"].rows):
        where = f"memory_copy[{index}]"
        if row["Kind"] != "MEMORY_COPY":
            _fail(f"{where}.Kind: expected MEMORY_COPY")
        if row["Direction"] not in set(MEMORY_COPY_OPERATIONS.values()):
            _fail(f"{where}.Direction: invalid ROCm 7.2.1 memory-copy enum")
        _agent_label(row["Source_Agent_Id"], f"{where}.Source_Agent_Id")
        _agent_label(row["Destination_Agent_Id"], f"{where}.Destination_Agent_Id")
        for column in (
            "Stream_Id",
            "Correlation_Id",
            "Start_Timestamp",
            "End_Timestamp",
        ):
            _uint_text(row[column], f"{where}.{column}")
        _uint_text(row["Correlation_Id"], f"{where}.Correlation_Id", positive=True)
        _uint_text(row["Start_Timestamp"], f"{where}.Start_Timestamp", positive=True)
        _uint_text(row["End_Timestamp"], f"{where}.End_Timestamp", positive=True)
        if int(row["End_Timestamp"]) < int(row["Start_Timestamp"]):
            _fail(f"{where}: end timestamp precedes start timestamp")

    for index, row in enumerate(tables["memory_allocation"].rows):
        where = f"memory_allocation[{index}]"
        if row["Kind"] != "MEMORY_ALLOCATION":
            _fail(f"{where}.Kind: expected MEMORY_ALLOCATION")
        if row["Operation"] not in set(MEMORY_ALLOCATION_OPERATIONS.values()):
            _fail(f"{where}.Operation: invalid ROCm 7.2.1 memory-allocation enum")
        if not _ADDRESS_RE.fullmatch(row["Address"]):
            _fail(f"{where}.Address: expected 0x plus 16 lowercase hex digits")
        for column in (
            "Allocation_Size",
            "Correlation_Id",
            "Start_Timestamp",
            "End_Timestamp",
        ):
            _uint_text(row[column], f"{where}.{column}")
        if row["Operation"] in {
            "MEMORY_ALLOCATION_ALLOCATE",
            "MEMORY_ALLOCATION_VMEM_ALLOCATE",
        }:
            _agent_label(row["Agent_Id"], f"{where}.Agent_Id")
        elif row["Agent_Id"] != "":
            _fail(f"{where}.Agent_Id: free records must use the quoted empty string")
        if int(row["End_Timestamp"]) < int(row["Start_Timestamp"]):
            _fail(f"{where}: end timestamp precedes start timestamp")

    return tables


def _agent_label(value: str, where: str) -> tuple[str, int]:
    match = _AGENT_LABEL_RE.fullmatch(value)
    if not match:
        _fail(f"{where}: expected 'Agent N', 'CPU N', or 'GPU N', got {value!r}")
    return match.group(1), int(match.group(2))


def _name_ref(
    string_table: Sequence[Any], kind: int, operation: int, where: str
) -> tuple[str, str]:
    if kind >= len(string_table):
        _fail(f"{where}.kind: {kind} is outside strings.buffer_records")
    entry = _exact_keys(string_table[kind], {"kind", "operations"}, f"strings[{kind}]")
    kind_name = _string(entry["kind"], f"strings[{kind}].kind", nonempty=True)
    operations = _list(entry["operations"], f"strings[{kind}].operations")
    if operation >= len(operations):
        _fail(f"{where}.operation: {operation} is outside strings[{kind}].operations")
    operation_name = _string(
        operations[operation], f"strings[{kind}].operations[{operation}]", nonempty=True
    )
    return kind_name, operation_name


def _correlation(
    value: Any, where: str, *, asynchronous: bool = False
) -> tuple[int, int]:
    expected = ASYNC_CORRELATION_KEYS if asynchronous else CORRELATION_KEYS
    obj = _exact_keys(value, expected, where)
    internal = _uint(obj["internal"], f"{where}.internal", positive=True)
    external = _uint(obj["external"], f"{where}.external")
    if not asynchronous:
        _uint(obj["ancestor"], f"{where}.ancestor")
    return internal, external


def _validate_json(
    paths: OutputPaths,
) -> tuple[Mapping[str, Any], dict[str, list[Any]]]:
    if not paths.json.is_file():
        _fail(f"json: file not found: {paths.json}")
    try:
        root = json.loads(paths.json.read_text(encoding="utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        _fail(f"json: malformed UTF-8 JSON: {exc}")
    root_obj = _exact_keys(root, {"rocprofiler-sdk-tool"}, "json root")
    processes = _list(root_obj["rocprofiler-sdk-tool"], "rocprofiler-sdk-tool")
    if len(processes) != 1:
        _fail(
            "rocprofiler-sdk-tool: expected the single-process array emitted per output file"
        )
    process = _exact_keys(processes[0], PROCESS_KEYS, "rocprofiler-sdk-tool[0]")

    metadata = _exact_keys(
        process["metadata"],
        {"node", "pid", "init_time", "fini_time", "command", "config"},
        "metadata",
    )
    _uint(metadata["pid"], "metadata.pid", positive=True)
    init_time = _uint(metadata["init_time"], "metadata.init_time", positive=True)
    fini_time = _uint(metadata["fini_time"], "metadata.fini_time", positive=True)
    if fini_time < init_time:
        _fail("metadata: fini_time precedes init_time")
    _list(metadata["command"], "metadata.command")
    if not isinstance(metadata["node"], dict) or not isinstance(
        metadata["config"], dict
    ):
        _fail("metadata.node/config: expected objects")

    strings = _exact_keys(process["strings"], STRINGS_KEYS, "strings")
    string_table = _list(strings["buffer_records"], "strings.buffer_records")
    _list(strings["callback_records"], "strings.callback_records")
    _list(strings["marker_api"], "strings.marker_api")
    _exact_keys(strings["correlation_id"], {"external"}, "strings.correlation_id")
    _exact_keys(strings["counters"], {"dimension_ids"}, "strings.counters")
    for key in (
        "pc_sample_instructions",
        "pc_sample_comments",
        "att_filenames",
        "code_object_snapshot_filenames",
    ):
        _list(strings[key], f"strings.{key}")
    _list(strings["correlation_id"]["external"], "strings.correlation_id.external")
    _list(strings["counters"]["dimension_ids"], "strings.counters.dimension_ids")

    agents = _list(process["agents"], "agents")
    if not agents:
        _fail("agents: expected at least one agent")
    agent_handles: set[int] = set()
    for index, agent_value in enumerate(agents):
        where = f"agents[{index}]"
        agent = _exact_keys(agent_value, AGENT_KEYS, where)
        handle = _handle(agent["id"], f"{where}.id", positive=True)
        if handle in agent_handles:
            _fail(f"{where}.id: duplicate handle {handle}")
        agent_handles.add(handle)
        agent_type = _uint(agent["type"], f"{where}.type")
        if agent_type not in {1, 2}:
            _fail(f"{where}.type: expected ROCProfiler CPU(1) or GPU(2)")
        for key in (
            "size",
            "cpu_cores_count",
            "simd_count",
            "mem_banks_count",
            "caches_count",
            "io_links_count",
            "node_id",
            "logical_node_id",
            "logical_node_type_id",
        ):
            _uint(agent[key], f"{where}.{key}")
        for key in ("name", "vendor_name", "product_name", "model_name"):
            _string(agent[key], f"{where}.{key}")
        _dim3(agent["workgroup_max_dim"], f"{where}.workgroup_max_dim")
        _dim3(agent["grid_max_dim"], f"{where}.grid_max_dim")
        for key in ("mem_banks", "caches", "io_links"):
            values = _list(agent[key], f"{where}.{key}")
            count_key = f"{key}_count"
            if len(values) != agent[count_key]:
                _fail(f"{where}.{key}: length does not equal {count_key}")
        if not isinstance(agent["runtime_visibility"], dict):
            _fail(f"{where}.runtime_visibility: expected object")
        if not isinstance(agent["uuid"], dict):
            _fail(f"{where}.uuid: expected object")
        if agent_type == 1 and (
            agent["cpu_cores_count"] <= 0 or agent["simd_count"] != 0
        ):
            _fail(f"{where}: invalid CPU counts")
        if agent_type == 2 and (
            agent["cpu_cores_count"] != 0 or agent["simd_count"] <= 0
        ):
            _fail(f"{where}: invalid GPU counts")

    kernel_symbols = _list(process["kernel_symbols"], "kernel_symbols")
    if not kernel_symbols:
        _fail("kernel_symbols: missing mandatory kernel-id-zero entry")
    for index, symbol_value in enumerate(kernel_symbols):
        where = f"kernel_symbols[{index}]"
        symbol = _exact_keys(symbol_value, KERNEL_SYMBOL_KEYS, where)
        symbol_id = _uint(symbol["kernel_id"], f"{where}.kernel_id")
        if symbol_id not in ({0} if index == 0 else {0, index}):
            _fail(f"{where}.kernel_id: populated symbols must be indexed by kernel id")
        _string(symbol["kernel_name"], f"{where}.kernel_name", nonempty=symbol_id > 0)
        for key in (
            "formatted_kernel_name",
            "demangled_kernel_name",
            "truncated_kernel_name",
        ):
            _string(symbol[key], f"{where}.{key}", nonempty=index == 0 or symbol_id > 0)
        for key in KERNEL_SYMBOL_KEYS - {
            "kernel_name",
            "formatted_kernel_name",
            "demangled_kernel_name",
            "truncated_kernel_name",
            "kernel_code_entry_byte_offset",
            "kernel_address",
        }:
            _uint(symbol[key], f"{where}.{key}")
        offset = symbol["kernel_code_entry_byte_offset"]
        if isinstance(offset, bool) or not isinstance(offset, int):
            _fail(
                f"{where}.kernel_code_entry_byte_offset: expected signed JSON integer"
            )
        _handle(symbol["kernel_address"], f"{where}.kernel_address")

    _list(process["counters"], "counters")
    _list(process["code_objects"], "code_objects")
    _list(process["host_functions"], "host_functions")
    _list(process["summary"], "summary")
    _exact_keys(process["callback_records"], {"counter_collection"}, "callback_records")
    buffer_records_obj = _exact_keys(
        process["buffer_records"], BUFFER_RECORD_KEYS, "buffer_records"
    )
    records = {
        key: _list(buffer_records_obj[key], f"buffer_records.{key}")
        for key in BUFFER_RECORD_KEYS
    }

    for index, value in enumerate(records["hip_api"]):
        where = f"buffer_records.hip_api[{index}]"
        record = _exact_keys(value, HIP_RECORD_KEYS, where)
        kind = _uint(record["kind"], f"{where}.kind")
        operation = _uint(record["operation"], f"{where}.operation")
        if kind not in {KIND_HIP_RUNTIME_EXT, KIND_HIP_COMPILER_EXT}:
            _fail(
                f"{where}.kind: expected HIP_RUNTIME_API_EXT(21) or HIP_COMPILER_API_EXT(22)"
            )
        kind_name, _ = _name_ref(string_table, kind, operation, where)
        expected_name = (
            "HIP_RUNTIME_API_EXT"
            if kind == KIND_HIP_RUNTIME_EXT
            else "HIP_COMPILER_API_EXT"
        )
        if kind_name != expected_name:
            _fail(f"{where}.kind: strings table resolves to {kind_name!r}")
        _uint(record["size"], f"{where}.size", positive=True)
        _uint(record["thread_id"], f"{where}.thread_id", positive=True)
        _correlation(record["correlation_id"], f"{where}.correlation_id")
        start = _uint(
            record["start_timestamp"], f"{where}.start_timestamp", positive=True
        )
        end = _uint(record["end_timestamp"], f"{where}.end_timestamp", positive=True)
        if end < start:
            _fail(f"{where}: end timestamp precedes start timestamp")
        _handle(record["stream_id"], f"{where}.stream_id")
        args = _list(record["args"], f"{where}.args")
        for arg_index, arg in enumerate(args):
            arg_obj = _exact_keys(
                arg, {"type", "name", "value"}, f"{where}.args[{arg_index}]"
            )
            for key in ("type", "name", "value"):
                _string(arg_obj[key], f"{where}.args[{arg_index}].{key}")
        retval = _exact_keys(record["retval"], {"uint64_t_retval"}, f"{where}.retval")
        _uint(retval["uint64_t_retval"], f"{where}.retval.uint64_t_retval")

    for index, value in enumerate(records["kernel_dispatch"]):
        where = f"buffer_records.kernel_dispatch[{index}]"
        record = _exact_keys(value, KERNEL_RECORD_KEYS, where)
        kind = _uint(record["kind"], f"{where}.kind")
        operation = _uint(record["operation"], f"{where}.operation")
        if kind != KIND_KERNEL_DISPATCH or operation != 2:
            _fail(f"{where}: expected KERNEL_DISPATCH(11)/COMPLETE(2)")
        if _name_ref(string_table, kind, operation, where) != (
            "KERNEL_DISPATCH",
            "KERNEL_DISPATCH_COMPLETE",
        ):
            _fail(f"{where}: strings table has incorrect kernel-dispatch enum names")
        _uint(record["size"], f"{where}.size", positive=True)
        _uint(record["thread_id"], f"{where}.thread_id", positive=True)
        _correlation(
            record["correlation_id"], f"{where}.correlation_id", asynchronous=True
        )
        start = _uint(
            record["start_timestamp"], f"{where}.start_timestamp", positive=True
        )
        end = _uint(record["end_timestamp"], f"{where}.end_timestamp", positive=True)
        if end < start:
            _fail(f"{where}: end timestamp precedes start timestamp")
        _handle(record["stream_id"], f"{where}.stream_id")
        dispatch = _exact_keys(
            record["dispatch_info"], DISPATCH_INFO_KEYS, f"{where}.dispatch_info"
        )
        _uint(dispatch["size"], f"{where}.dispatch_info.size", positive=True)
        agent_handle = _handle(
            dispatch["agent_id"], f"{where}.dispatch_info.agent_id", positive=True
        )
        if agent_handle not in agent_handles:
            _fail(f"{where}.dispatch_info.agent_id: unknown agent handle")
        _handle(dispatch["queue_id"], f"{where}.dispatch_info.queue_id", positive=True)
        kernel_id = _uint(
            dispatch["kernel_id"], f"{where}.dispatch_info.kernel_id", positive=True
        )
        if kernel_id >= len(kernel_symbols):
            _fail(f"{where}.dispatch_info.kernel_id: missing kernel symbol")
        if kernel_symbols[kernel_id]["kernel_id"] != kernel_id:
            _fail(
                f"{where}.dispatch_info.kernel_id: references an unpopulated symbol slot"
            )
        _uint(
            dispatch["dispatch_id"], f"{where}.dispatch_info.dispatch_id", positive=True
        )
        _uint(
            dispatch["private_segment_size"],
            f"{where}.dispatch_info.private_segment_size",
        )
        _uint(
            dispatch["group_segment_size"], f"{where}.dispatch_info.group_segment_size"
        )
        # ROCm's ABI types are unsigned, not non-zero. Graph replay can produce
        # a valid record whose launch geometry is unavailable to this Windows
        # tracer; those fields are honestly emitted as zero.
        _dim3(dispatch["workgroup_size"], f"{where}.dispatch_info.workgroup_size")
        _dim3(dispatch["grid_size"], f"{where}.dispatch_info.grid_size")

    for index, value in enumerate(records["memory_copy"]):
        where = f"buffer_records.memory_copy[{index}]"
        record = _exact_keys(value, MEMORY_COPY_RECORD_KEYS, where)
        kind = _uint(record["kind"], f"{where}.kind")
        operation = _uint(record["operation"], f"{where}.operation")
        if kind != KIND_MEMORY_COPY or operation not in MEMORY_COPY_OPERATIONS:
            _fail(f"{where}: invalid ROCm 7.2.1 memory-copy enum")
        if _name_ref(string_table, kind, operation, where) != (
            "MEMORY_COPY",
            MEMORY_COPY_OPERATIONS[operation],
        ):
            _fail(f"{where}: strings table has incorrect memory-copy enum names")
        _uint(record["size"], f"{where}.size", positive=True)
        _uint(record["thread_id"], f"{where}.thread_id", positive=True)
        _correlation(
            record["correlation_id"], f"{where}.correlation_id", asynchronous=True
        )
        start = _uint(
            record["start_timestamp"], f"{where}.start_timestamp", positive=True
        )
        end = _uint(record["end_timestamp"], f"{where}.end_timestamp", positive=True)
        if end < start:
            _fail(f"{where}: end timestamp precedes start timestamp")
        copy_handles = {}
        for key in ("src_agent_id", "dst_agent_id"):
            copy_handles[key] = _handle(record[key], f"{where}.{key}", positive=True)
            if copy_handles[key] not in agent_handles:
                _fail(f"{where}.{key}: unknown agent handle")
        agent_types = {agent["id"]["handle"]: agent["type"] for agent in agents}
        actual_types = (
            agent_types[copy_handles["src_agent_id"]],
            agent_types[copy_handles["dst_agent_id"]],
        )
        expected_types = {1: (1, 1), 2: (1, 2), 3: (2, 1), 4: (2, 2)}
        if actual_types != expected_types[operation]:
            _fail(f"{where}: source/destination agent types contradict copy direction")
        _uint(record["bytes"], f"{where}.bytes")
        _handle(record["stream_id"], f"{where}.stream_id")

    for index, value in enumerate(records["memory_allocation"]):
        where = f"buffer_records.memory_allocation[{index}]"
        record = _exact_keys(value, MEMORY_ALLOCATION_RECORD_KEYS, where)
        kind = _uint(record["kind"], f"{where}.kind")
        operation = _uint(record["operation"], f"{where}.operation")
        if (
            kind != KIND_MEMORY_ALLOCATION
            or operation not in MEMORY_ALLOCATION_OPERATIONS
        ):
            _fail(f"{where}: invalid ROCm 7.2.1 memory-allocation enum")
        if _name_ref(string_table, kind, operation, where) != (
            "MEMORY_ALLOCATION",
            MEMORY_ALLOCATION_OPERATIONS[operation],
        ):
            _fail(f"{where}: strings table has incorrect memory-allocation enum names")
        _uint(record["size"], f"{where}.size", positive=True)
        _uint(record["thread_id"], f"{where}.thread_id", positive=True)
        _correlation(record["correlation_id"], f"{where}.correlation_id")
        start = _uint(
            record["start_timestamp"], f"{where}.start_timestamp", positive=True
        )
        end = _uint(record["end_timestamp"], f"{where}.end_timestamp", positive=True)
        if end < start:
            _fail(f"{where}: end timestamp precedes start timestamp")
        agent_handle = _handle(record["agent_id"], f"{where}.agent_id")
        if operation in {1, 2} and agent_handle not in agent_handles:
            _fail(f"{where}.agent_id: allocation must reference an agent")
        if operation in {3, 4} and agent_handle != 0:
            _fail(f"{where}.agent_id: free records must contain the null agent handle")
        address = _string(record["address"], f"{where}.address")
        if not _ADDRESS_RE.fullmatch(address):
            _fail(f"{where}.address: expected 0x plus 16 lowercase hex digits")
        _uint(record["allocation_size"], f"{where}.allocation_size")
        _handle(record["stream_id"], f"{where}.stream_id")

    return process, records


def _record_key(correlation: int, start: int, end: int) -> tuple[int, int, int]:
    return correlation, start, end


def _csv_key(row: Mapping[str, str]) -> tuple[int, int, int]:
    return _record_key(
        int(row["Correlation_Id"]),
        int(row["Start_Timestamp"]),
        int(row["End_Timestamp"]),
    )


def _json_key(record: Mapping[str, Any]) -> tuple[int, int, int]:
    return _record_key(
        record["correlation_id"]["internal"],
        record["start_timestamp"],
        record["end_timestamp"],
    )


def _resolve_csv_agent(
    label: str, agents: Sequence[Mapping[str, Any]], where: str
) -> Mapping[str, Any]:
    kind, index = _agent_label(label, where)
    if kind == "Agent":
        matches = [agent for agent in agents if agent["logical_node_id"] == index]
    else:
        expected_type = 1 if kind == "CPU" else 2
        matches = [
            agent
            for agent in agents
            if agent["type"] == expected_type and agent["logical_node_type_id"] == index
        ]
    if len(matches) != 1:
        _fail(f"{where}: label {label!r} does not resolve to exactly one JSON agent")
    return matches[0]


def _cross_validate(
    tables: Mapping[str, CsvTable],
    process: Mapping[str, Any],
    records: Mapping[str, list[Any]],
) -> None:
    agents = process["agents"]
    agent_rows = tables["agent_info"].rows
    if len(agent_rows) != len(agents):
        _fail("agent_info/JSON: agent count mismatch")
    json_by_logical_id = {agent["logical_node_id"]: agent for agent in agents}
    for index, row in enumerate(agent_rows):
        logical_id = int(row["Logical_Node_Id"])
        if logical_id not in json_by_logical_id:
            _fail(f"agent_info[{index}]: Logical_Node_Id missing from JSON agents")
        agent = json_by_logical_id[logical_id]
        comparisons = {
            "Node_Id": "node_id",
            "Cpu_Cores_Count": "cpu_cores_count",
            "Simd_Count": "simd_count",
            "Cpu_Core_Id_Base": "cpu_core_id_base",
            "Simd_Id_Base": "simd_id_base",
            "Max_Waves_Per_Simd": "max_waves_per_simd",
            "Lds_Size_In_Kb": "lds_size_in_kb",
            "Gds_Size_In_Kb": "gds_size_in_kb",
            "Num_Gws": "num_gws",
            "Wave_Front_Size": "wave_front_size",
            "Num_Xcc": "num_xcc",
            "Cu_Count": "cu_count",
            "Array_Count": "array_count",
            "Num_Shader_Banks": "num_shader_banks",
            "Simd_Arrays_Per_Engine": "simd_arrays_per_engine",
            "Cu_Per_Simd_Array": "cu_per_simd_array",
            "Simd_Per_Cu": "simd_per_cu",
            "Max_Slots_Scratch_Cu": "max_slots_scratch_cu",
            "Gfx_Target_Version": "gfx_target_version",
            "Vendor_Id": "vendor_id",
            "Device_Id": "device_id",
            "Location_Id": "location_id",
            "Domain": "domain",
            "Drm_Render_Minor": "drm_render_minor",
            "Num_Sdma_Engines": "num_sdma_engines",
            "Num_Sdma_Xgmi_Engines": "num_sdma_xgmi_engines",
            "Num_Sdma_Queues_Per_Engine": "num_sdma_queues_per_engine",
            "Num_Cp_Queues": "num_cp_queues",
            "Max_Engine_Clk_Ccompute": "max_engine_clk_ccompute",
            "Max_Engine_Clk_Fcompute": "max_engine_clk_fcompute",
            "Cu_Per_Engine": "cu_per_engine",
            "Max_Waves_Per_Cu": "max_waves_per_cu",
            "Family_Id": "family_id",
            "Workgroup_Max_Size": "workgroup_max_size",
            "Grid_Max_Size": "grid_max_size",
            "Local_Mem_Size": "local_mem_size",
            "Hive_Id": "hive_id",
            "Gpu_Id": "gpu_id",
        }
        for csv_name, json_name in comparisons.items():
            if int(row[csv_name]) != agent[json_name]:
                _fail(f"agent_info[{index}].{csv_name}: CSV/JSON mismatch")
        if row["Agent_Type"] != ("CPU" if agent["type"] == 1 else "GPU"):
            _fail(f"agent_info[{index}].Agent_Type: CSV/JSON mismatch")
        for csv_name, json_name in (
            ("Name", "name"),
            ("Vendor_Name", "vendor_name"),
            ("Product_Name", "product_name"),
            ("Model_Name", "model_name"),
        ):
            if row[csv_name] != agent[json_name]:
                _fail(f"agent_info[{index}].{csv_name}: CSV/JSON mismatch")
        for prefix, json_name in (
            ("Workgroup_Max_Dim", "workgroup_max_dim"),
            ("Grid_Max_Dim", "grid_max_dim"),
        ):
            for component in ("X", "Y", "Z"):
                if (
                    int(row[f"{prefix}_{component}"])
                    != agent[json_name][component.lower()]
                ):
                    _fail(
                        f"agent_info[{index}].{prefix}_{component}: CSV/JSON mismatch"
                    )

    domain_pairs = {
        "hip_api": records["hip_api"],
        "kernel_dispatch": records["kernel_dispatch"],
        "memory_copy": records["memory_copy"],
        "memory_allocation": records["memory_allocation"],
    }
    for domain, json_records in domain_pairs.items():
        csv_rows = tables[domain].rows
        if len(csv_rows) != len(json_records):
            _fail(f"{domain}: CSV/JSON record count mismatch")
        if Counter(map(_csv_key, csv_rows)) != Counter(map(_json_key, json_records)):
            _fail(f"{domain}: CSV/JSON correlation/timestamp consistency mismatch")

    strings = process["strings"]["buffer_records"]
    symbols = process["kernel_symbols"]
    pid = process["metadata"]["pid"]
    indexes = {}
    for domain, json_records in domain_pairs.items():
        index = defaultdict(deque)
        for record in json_records:
            index[_json_key(record)].append(record)
        indexes[domain] = index

    def take(domain: str, row: Mapping[str, str]):
        key = _csv_key(row)
        bucket = indexes[domain].get(key)
        if not bucket:
            _fail(f"{domain}: no JSON record for CSV key {key}")
        return bucket.popleft()

    for index, row in enumerate(tables["hip_api"].rows):
        record = take("hip_api", row)
        kind_name, operation_name = _name_ref(
            strings, record["kind"], record["operation"], f"hip_api[{index}]"
        )
        if row["Domain"] != kind_name or row["Function"] != operation_name:
            _fail(
                f"hip_api[{index}]: CSV names do not match JSON string-table references"
            )
        if (
            int(row["Process_Id"]) != pid
            or int(row["Thread_Id"]) != record["thread_id"]
        ):
            _fail(f"hip_api[{index}]: process/thread mismatch")

    for index, row in enumerate(tables["kernel_dispatch"].rows):
        record = take("kernel_dispatch", row)
        dispatch = record["dispatch_info"]
        symbol = symbols[dispatch["kernel_id"]]
        csv_agent = _resolve_csv_agent(
            row["Agent_Id"], agents, f"kernel_dispatch[{index}].Agent_Id"
        )
        checks = {
            "Queue_Id": dispatch["queue_id"]["handle"],
            "Stream_Id": record["stream_id"]["handle"],
            "Thread_Id": record["thread_id"],
            "Dispatch_Id": dispatch["dispatch_id"],
            "Kernel_Id": dispatch["kernel_id"],
            "Scratch_Size": dispatch["private_segment_size"],
            "VGPR_Count": symbol["arch_vgpr_count"],
            "Accum_VGPR_Count": symbol["accum_vgpr_count"],
            "SGPR_Count": symbol["sgpr_count"],
            "Workgroup_Size_X": dispatch["workgroup_size"]["x"],
            "Workgroup_Size_Y": dispatch["workgroup_size"]["y"],
            "Workgroup_Size_Z": dispatch["workgroup_size"]["z"],
            "Grid_Size_X": dispatch["grid_size"]["x"],
            "Grid_Size_Y": dispatch["grid_size"]["y"],
            "Grid_Size_Z": dispatch["grid_size"]["z"],
            "LDS_Block_Size": (symbol["group_segment_size"] + 511) & ~511,
        }
        for csv_name, json_value in checks.items():
            if int(row[csv_name]) != json_value:
                _fail(f"kernel_dispatch[{index}].{csv_name}: CSV/JSON mismatch")
        if csv_agent["id"]["handle"] != dispatch["agent_id"]["handle"]:
            _fail(f"kernel_dispatch[{index}].Agent_Id: CSV/JSON mismatch")
        if row["Kernel_Name"] != symbol["formatted_kernel_name"]:
            _fail(f"kernel_dispatch[{index}].Kernel_Name: CSV/JSON mismatch")

    for index, row in enumerate(tables["memory_copy"].rows):
        record = take("memory_copy", row)
        _, operation_name = _name_ref(
            strings, record["kind"], record["operation"], f"memory_copy[{index}]"
        )
        if row["Direction"] != operation_name:
            _fail(f"memory_copy[{index}].Direction: CSV/JSON mismatch")
        if int(row["Stream_Id"]) != record["stream_id"]["handle"]:
            _fail(f"memory_copy[{index}].Stream_Id: CSV/JSON mismatch")
        for csv_name, json_name in (
            ("Source_Agent_Id", "src_agent_id"),
            ("Destination_Agent_Id", "dst_agent_id"),
        ):
            csv_agent = _resolve_csv_agent(
                row[csv_name], agents, f"memory_copy[{index}].{csv_name}"
            )
            if csv_agent["id"]["handle"] != record[json_name]["handle"]:
                _fail(f"memory_copy[{index}].{csv_name}: CSV/JSON mismatch")

    for index, row in enumerate(tables["memory_allocation"].rows):
        record = take("memory_allocation", row)
        _, operation_name = _name_ref(
            strings, record["kind"], record["operation"], f"memory_allocation[{index}]"
        )
        if row["Operation"] != operation_name:
            _fail(f"memory_allocation[{index}].Operation: CSV/JSON mismatch")
        if int(row["Allocation_Size"]) != record["allocation_size"]:
            _fail(f"memory_allocation[{index}].Allocation_Size: CSV/JSON mismatch")
        if row["Address"] != record["address"]:
            _fail(f"memory_allocation[{index}].Address: CSV/JSON mismatch")
        if row["Agent_Id"]:
            csv_agent = _resolve_csv_agent(
                row["Agent_Id"], agents, f"memory_allocation[{index}].Agent_Id"
            )
            if csv_agent["id"]["handle"] != record["agent_id"]["handle"]:
                _fail(f"memory_allocation[{index}].Agent_Id: CSV/JSON mismatch")


def validate(paths: OutputPaths) -> None:
    """Validate one rocprofv3 output set, raising ValidationError on failure."""

    tables = _validate_csv_tables(paths)
    process, records = _validate_json(paths)
    _cross_validate(tables, process, records)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, help="Directory containing prefixed outputs"
    )
    parser.add_argument(
        "--prefix", help="rocprofv3 output-file prefix used with --output-dir"
    )
    parser.add_argument(
        "--fixtures",
        action="store_true",
        help="Validate the bundled synthetic ROCm 7.2.1 golden files",
    )
    parser.add_argument("--agent-info", type=Path)
    parser.add_argument("--hip-api", type=Path)
    parser.add_argument("--kernel", type=Path)
    parser.add_argument("--memory-copy", type=Path)
    parser.add_argument("--memory-allocation", type=Path)
    parser.add_argument("--json", type=Path)
    return parser


def _paths_from_args(
    args: argparse.Namespace, parser: argparse.ArgumentParser
) -> OutputPaths:
    if args.fixtures:
        if args.output_dir or args.prefix:
            parser.error("--fixtures cannot be combined with --output-dir/--prefix")
        fixture_dir = Path(__file__).resolve().parent / "fixtures" / "rocprofv3_7_2_1"
        return OutputPaths.from_directory(fixture_dir, "golden")
    if args.output_dir is not None or args.prefix is not None:
        if args.output_dir is None or args.prefix is None:
            parser.error("--output-dir and --prefix must be supplied together")
        return OutputPaths.from_directory(args.output_dir, args.prefix)
    explicit = (
        args.agent_info,
        args.hip_api,
        args.kernel,
        args.memory_copy,
        args.memory_allocation,
        args.json,
    )
    if any(value is None for value in explicit):
        parser.error(
            "use --fixtures, --output-dir/--prefix, or all six explicit output path options"
        )
    return OutputPaths(*explicit)


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    paths = _paths_from_args(args, parser)
    try:
        validate(paths)
    except (OSError, ValidationError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    print("PASS: ROCm 7.2.1 rocprofv3 CSV/JSON compatibility validated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
