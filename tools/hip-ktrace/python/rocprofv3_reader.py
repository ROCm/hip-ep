#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""Streaming reader for ktrace's ROCm 7.2.1 rocprofv3-shaped output.

The public CSV files deliberately use the official rocprofv3 schemas.  This
module translates those domain-specific rows into the small logical schema the
existing ktrace analysis tools use.  It also keeps the old ``*_ops.csv``
captures readable during the transition.

All potentially large inputs are streamed.  In particular,
``*_ktrace_extensions.json`` arrays are decoded one item at a time instead of
loading the sidecar with ``json.load``.
"""

from __future__ import annotations

import csv
import heapq
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator


DOMAIN_SUFFIXES = (
    "_kernel_trace_readable.csv",
    "_memory_allocation_trace.csv",
    "_memory_copy_trace.csv",
    "_hip_api_trace.csv",
    "_kernel_trace.csv",
    "_roctx_trace.csv",
    "_agent_info.csv",
    "_ktrace_launch_info.csv",
    "_hip_runtime_info.csv",
    "_cpu_sched.csv",
    "_cpu_samples.csv",
    "_cpu_images.csv",
    "_ktrace_extensions.json",
    "_results.json",
)

OLD_SIDECAR_SUFFIXES = (
    ".api.csv",
    ".kernels.csv",
    ".vram.csv",
    ".roctx.csv",
    ".launch.csv",
    ".summary.txt",
)


class CaptureNotFoundError(FileNotFoundError):
    """Raised when an input cannot be resolved to either capture layout."""


class _CharStream:
    """Small push-back character stream used by the incremental JSON reader."""

    def __init__(self, fh, chunk_size: int = 65536):
        self.fh = fh
        self.chunk_size = chunk_size
        self.chunk = ""
        self.at = 0
        self.pushed: list[str] = []

    def get(self) -> str:
        if self.pushed:
            return self.pushed.pop()
        if self.at >= len(self.chunk):
            self.chunk = self.fh.read(self.chunk_size)
            self.at = 0
            if not self.chunk:
                return ""
        ch = self.chunk[self.at]
        self.at += 1
        return ch

    def push(self, ch: str) -> None:
        if ch:
            self.pushed.append(ch)


def _nonspace(stream: _CharStream) -> str:
    ch = stream.get()
    while ch and ch.isspace():
        ch = stream.get()
    return ch


def _read_string_token(stream: _CharStream) -> str:
    """Read a JSON string after its opening quote and return decoded text."""
    token = ['"']
    escaped = False
    while True:
        ch = stream.get()
        if not ch:
            raise ValueError("unterminated JSON string")
        token.append(ch)
        if escaped:
            escaped = False
        elif ch == "\\":
            escaped = True
        elif ch == '"':
            return json.loads("".join(token))


def _read_json_value(stream: _CharStream, first: str | None = None):
    """Decode one JSON value while retaining only that value in memory."""
    first = first or _nonspace(stream)
    if not first:
        raise ValueError("unexpected end of JSON")

    if first == '"':
        stream.push(first)
        token = ['"']
        stream.get()
        escaped = False
        while True:
            ch = stream.get()
            if not ch:
                raise ValueError("unterminated JSON string")
            token.append(ch)
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                return json.loads("".join(token))

    if first in "[{":
        token = [first]
        stack = [first]
        in_string = False
        escaped = False
        pairs = {"]": "[", "}": "{"}
        while stack:
            ch = stream.get()
            if not ch:
                raise ValueError("unterminated JSON value")
            token.append(ch)
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
            elif ch == '"':
                in_string = True
            elif ch in "[{":
                stack.append(ch)
            elif ch in "]}":
                if not stack or stack.pop() != pairs[ch]:
                    raise ValueError("mismatched JSON delimiter")
        return json.loads("".join(token))

    token = [first]
    while True:
        ch = stream.get()
        if not ch or ch.isspace() or ch in ",]}":
            stream.push(ch)
            break
        token.append(ch)
    return json.loads("".join(token))


def _find_named_container(stream: _CharStream, names: set[str]) -> str | None:
    """Find the first array/object value whose key is in *names*."""
    while True:
        ch = stream.get()
        if not ch:
            return None
        if ch != '"':
            continue
        key = _read_string_token(stream)
        if key not in names:
            continue
        ch = _nonspace(stream)
        if ch != ":":
            stream.push(ch)
            continue
        ch = _nonspace(stream)
        if ch in "[{":
            return ch
        stream.push(ch)


def _iter_named_json(path: Path, names: Iterable[str]) -> Iterator:
    """Yield entries from a named JSON array or object with bounded memory."""
    with path.open(encoding="utf-8") as fh:
        stream = _CharStream(fh)
        opening = _find_named_container(stream, set(names))
        if opening is None:
            return

        if opening == "[":
            ch = _nonspace(stream)
            if ch == "]":
                return
            while ch:
                yield _read_json_value(stream, ch)
                ch = _nonspace(stream)
                if ch == "]":
                    return
                if ch != ",":
                    raise ValueError(f"{path}: expected ',' or ']' in extension array")
                ch = _nonspace(stream)
            raise ValueError(f"{path}: unterminated extension array")

        ch = _nonspace(stream)
        if ch == "}":
            return
        while ch:
            if ch != '"':
                raise ValueError(f"{path}: expected a string key in extension object")
            key = _read_string_token(stream)
            if _nonspace(stream) != ":":
                raise ValueError(f"{path}: expected ':' in extension object")
            value = _read_json_value(stream)
            if isinstance(value, dict):
                value.setdefault("_key", key)
                yield value
            else:
                yield {"_key": key, "value": value}
            ch = _nonspace(stream)
            if ch == "}":
                return
            if ch != ",":
                raise ValueError(f"{path}: expected ',' or '}}' in extension object")
            ch = _nonspace(stream)
        raise ValueError(f"{path}: unterminated extension object")


def _first(row: dict, *names: str, default=""):
    for name in names:
        if name in row and row[name] is not None:
            return row[name]
    return default


def _integer(value, default: int = 0) -> int:
    if value in (None, ""):
        return default
    if isinstance(value, str):
        try:
            return int(value, 0)
        except ValueError:
            return int(float(value))
    return int(value)


def _number(value, default: float = 0.0) -> float:
    if value in (None, ""):
        return default
    return float(value)


def _corr(row: dict) -> int:
    return _integer(
        _first(
            row, "Correlation_Id", "correlation_id", "Corr", "corr", "_key", default=0
        )
    )


def _format_args(value) -> str:
    """Normalize extension API arguments to the historical ``key=value`` text."""
    if not value:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return " ".join(f"{k}={v}" for k, v in value.items())
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, dict):
                name = _first(item, "name", "Name", "key", default="")
                val = _first(item, "value", "Value", "val", default="")
                if name:
                    parts.append(f"{name}={val}")
            elif item:
                parts.append(str(item))
        return " ".join(parts)
    return str(value)


def _args_value(row: dict) -> str:
    value = _first(row, "Args", "args", "decoded_args", "arguments", default="")
    if not value and "value" in row:
        value = row["value"]
    return _format_args(value)


def _arg_int(args: str, name: str) -> int:
    marker = name + "="
    for token in args.split():
        if token.startswith(marker):
            try:
                return int(token[len(marker) :], 0)
            except ValueError:
                return 0
    return 0


def _merge_args(left: str, right: str) -> str:
    if not left:
        return right
    if not right:
        return left
    existing = {part.partition("=")[0] for part in left.split()}
    extra = [part for part in right.split() if part.partition("=")[0] not in existing]
    return " ".join([left, *extra])


@dataclass(frozen=True)
class Capture:
    """A resolved rocprofv3 capture prefix or a legacy ktrace capture."""

    prefix: Path
    input_path: Path
    legacy_ops: Path | None = None

    @property
    def is_legacy(self) -> bool:
        return self.legacy_ops is not None

    @property
    def extension_path(self) -> Path:
        return Path(str(self.prefix) + "_ktrace_extensions.json")

    @property
    def source_path(self) -> Path:
        return self.legacy_ops or self.prefix

    def domain_path(self, domain: str) -> Path:
        if self.is_legacy:
            base = str(self.legacy_ops)
            legacy = {
                "ops": base,
                "api": base + ".api.csv",
                "kernels": base + ".kernels.csv",
                "vram": base + ".vram.csv",
                "roctx": base + ".roctx.csv",
                "launch": base + ".launch.csv",
                "summary": base + ".summary.txt",
            }
            return Path(legacy[domain])
        suffixes = {
            "api": "_hip_api_trace.csv",
            "roctx": "_roctx_trace.csv",
            "kernels": "_kernel_trace.csv",
            "copy": "_memory_copy_trace.csv",
            "alloc": "_memory_allocation_trace.csv",
            "agents": "_agent_info.csv",
            "launch": "_ktrace_launch_info.csv",
            "runtime": "_hip_runtime_info.csv",
            "cpu_sched": "_cpu_sched.csv",
            "cpu_samples": "_cpu_samples.csv",
            "cpu_images": "_cpu_images.csv",
            "results": "_results.json",
            "extensions": "_ktrace_extensions.json",
        }
        return Path(str(self.prefix) + suffixes[domain])

    def auto_output(self, suffix: str) -> str:
        return str(self.prefix) + suffix

    def _extension(self, *names: str) -> Iterator[dict]:
        if not self.extension_path.exists():
            return iter(())
        return _iter_named_json(self.extension_path, names)

    def _iter_extension_args(self) -> Iterator[dict]:
        for row in self._extension(
            "api_args", "decoded_api_args", "api_arguments", "hip_api_args", "apis"
        ):
            if not isinstance(row, dict):
                continue
            args = _args_value(row)
            corr = _corr(row)
            if corr and args:
                yield {"Corr": corr, "Args": args}

    @staticmethod
    def _merge_supplements(*streams: Iterator[dict]) -> Iterator[dict]:
        usable = [iter(stream) for stream in streams]
        if not usable:
            return iter(())
        return heapq.merge(*usable, key=lambda row: _integer(row["Corr"]))

    @staticmethod
    def _join_args(rows: Iterator[dict], supplements: Iterator[dict]) -> Iterator[dict]:
        extra = iter(supplements)
        pending = next(extra, None)
        for row in rows:
            corr = _integer(row["Corr"])
            args = row.get("Args", "")
            while pending is not None and _integer(pending["Corr"]) < corr:
                pending = next(extra, None)
            while pending is not None and _integer(pending["Corr"]) == corr:
                args = _merge_args(args, pending.get("Args", ""))
                pending = next(extra, None)
            row["Args"] = args
            yield row

    @staticmethod
    def _join_ordered_subset_args(
        rows: Iterator[dict], supplements: Iterator[dict]
    ) -> Iterator[dict]:
        """Join a side stream emitted as an order-preserving subset of rows."""
        extra = iter(supplements)
        pending = next(extra, None)
        for row in rows:
            if pending is not None and _integer(pending["Corr"]) == _integer(
                row["Corr"]
            ):
                row["Args"] = _merge_args(row.get("Args", ""), pending.get("Args", ""))
                pending = next(extra, None)
            yield row

    def iter_api(self, include_args: bool = True) -> Iterator[dict]:
        """Yield historical logical API rows, with timestamps in microseconds."""
        path = self.domain_path("api")
        if not path.exists():
            return iter(())

        def rows() -> Iterator[dict]:
            with path.open(newline="", encoding="utf-8-sig") as fh:
                for row in csv.DictReader(fh):
                    if self.is_legacy:
                        yield row
                        continue
                    start_ns = _integer(row["Start_Timestamp"])
                    end_ns = _integer(row["End_Timestamp"])
                    yield {
                        "Corr": _corr(row),
                        "Api": row["Function"],
                        "Api_Id": "",
                        "Thread": _integer(row["Thread_Id"]),
                        "Enter_us": start_ns / 1000.0,
                        "Exit_us": end_ns / 1000.0,
                        "Host_us": (end_ns - start_ns) / 1000.0,
                        "Args": "",
                    }

        base_rows = rows()
        if self.is_legacy or not include_args:
            return base_rows
        return self._join_ordered_subset_args(base_rows, self._iter_extension_args())

    def iter_roctx(self) -> Iterator[dict]:
        """Yield ROCTx point events with timestamps in microseconds."""
        path = self.domain_path("roctx")
        if not path.exists():
            return iter(())

        def rows() -> Iterator[dict]:
            with path.open(newline="", encoding="utf-8-sig") as fh:
                for row in csv.DictReader(fh):
                    if self.is_legacy:
                        yield {
                            "Operation": row["Op"],
                            "Process": _integer(row["Process"]),
                            "Thread": _integer(row["Thread"]),
                            "Range_Id": _integer(row["Range_Id"]),
                            "Timestamp_us": float(row["Timestamp_us"]),
                            "Message": row["Message"],
                        }
                    else:
                        yield {
                            "Operation": row["Operation"],
                            "Process": _integer(row["Process_Id"]),
                            "Thread": _integer(row["Thread_Id"]),
                            "Range_Id": _integer(row["Range_Id"]),
                            "Timestamp_us": _integer(row["Timestamp"]) / 1000.0,
                            "Message": row["Message"],
                        }

        return rows()

    def iter_allocations(self) -> Iterator[dict]:
        """Yield memory-allocation records with timestamps in microseconds."""
        if self.is_legacy:
            return iter(())
        path = self.domain_path("alloc")
        if not path.exists():
            return iter(())

        def rows():
            with path.open(newline="", encoding="utf-8-sig") as fh:
                for row in csv.DictReader(fh):
                    start_ns = _integer(row["Start_Timestamp"])
                    end_ns = _integer(row["End_Timestamp"])
                    yield {
                        "Operation": row["Operation"],
                        "Agent_Id": row.get("Agent_Id", ""),
                        "Allocation_Size": _integer(row.get("Allocation_Size", 0)),
                        "Address": row.get("Address", ""),
                        "Corr": _corr(row),
                        "Begin_us": start_ns / 1000.0,
                        "End_us": end_ns / 1000.0,
                    }

        return rows()

    def _iter_kernel_ops(self) -> Iterator[dict]:
        path = self.domain_path("kernels")
        if not path.exists():
            return
        with path.open(newline="", encoding="utf-8-sig") as fh:
            for row in csv.DictReader(fh):
                start_ns = _integer(row["Start_Timestamp"])
                end_ns = _integer(row["End_Timestamp"])
                yield {
                    "Id": row.get("Dispatch_Id", ""),
                    "Op": "dispatch",
                    "Kind": row.get("Kind", "KERNEL_DISPATCH"),
                    "Kernel_Name": row.get("Kernel_Name", ""),
                    "Bytes": 0,
                    "Device": row.get("Agent_Id", ""),
                    "Queue": _integer(row.get("Queue_Id", 0)),
                    "Stream": _integer(row.get("Stream_Id", 0)),
                    "Thread": _integer(row.get("Thread_Id", 0)),
                    "Corr": _corr(row),
                    "Begin_us": start_ns / 1000.0,
                    "End_us": end_ns / 1000.0,
                    "Duration_us": (end_ns - start_ns) / 1000.0,
                }

    def _copy_supplements(self) -> Iterator[dict]:
        for row in self._extension(
            "memory_copy_extras", "copy_extras", "memory_copies", "copy_records"
        ):
            if not isinstance(row, dict):
                continue
            corr = _corr(row)
            args = _args_value(row)
            size = _integer(_first(row, "Bytes", "bytes", "size", "Size", default=0))
            if not size:
                size = _arg_int(args, "size")
            if corr:
                yield {
                    "Corr": corr,
                    "Bytes": size,
                    "Device": str(_first(row, "Device", "device", default="")),
                    "Queue": _integer(_first(row, "Queue", "queue", default=0)),
                    "Stream": _integer(_first(row, "Stream", "stream", default=0)),
                }

    def _iter_copy_ops(self) -> Iterator[dict]:
        path = self.domain_path("copy")
        if not path.exists():
            return
        gpu_agents: set[str] = set()
        agents_path = self.domain_path("agents")
        if agents_path.exists():
            with agents_path.open(newline="", encoding="utf-8-sig") as fh:
                for agent in csv.DictReader(fh):
                    if agent.get("Agent_Type") == "GPU":
                        gpu_agents.add(f"Agent {_integer(agent.get('Node_Id', 0))}")
                        gpu_agents.add(
                            f"Agent {_integer(agent.get('Logical_Node_Id', 0))}"
                        )

        def rows() -> Iterator[dict]:
            with path.open(newline="", encoding="utf-8-sig") as fh:
                for row in csv.DictReader(fh):
                    start_ns = _integer(row["Start_Timestamp"])
                    end_ns = _integer(row["End_Timestamp"])
                    src = row.get("Source_Agent_Id", "")
                    dst = row.get("Destination_Agent_Id", "")
                    device = next(
                        (agent for agent in (dst, src) if agent in gpu_agents), ""
                    )
                    if not device:
                        device = src if dst == "Agent 0" and src else (dst or src)
                    yield {
                        "Id": "",
                        "Op": "copy",
                        "Kind": row.get("Direction", "MEMORY_COPY"),
                        "Kernel_Name": "",
                        "Bytes": 0,
                        "Device": device,
                        # rocprofv3 copy rows have a stream but no hardware queue.
                        "Queue": _integer(row.get("Stream_Id", 0)),
                        "Stream": _integer(row.get("Stream_Id", 0)),
                        "Thread": 0,
                        "Corr": _corr(row),
                        "Begin_us": start_ns / 1000.0,
                        "End_us": end_ns / 1000.0,
                        "Duration_us": (end_ns - start_ns) / 1000.0,
                        "Args": "",
                    }

        extras = iter(self._copy_supplements())
        pending = next(extras, None)
        for row in rows():
            corr = _integer(row["Corr"])
            if pending is not None and _integer(pending["Corr"]) == corr:
                row["Bytes"] = pending["Bytes"]
                row["Device"] = pending["Device"] or row["Device"]
                row["Queue"] = pending["Queue"]
                row["Stream"] = pending["Stream"]
                pending = next(extras, None)
            else:
                row["Bytes"] = 0
            row.pop("Args", None)
            yield row

    @staticmethod
    def _normalize_extension_op(row: dict) -> dict | None:
        op = str(_first(row, "Op", "op", "operation", default="")).lower()
        kind = str(_first(row, "Kind", "kind", "direction", default=""))
        if not op:
            low_kind = kind.lower()
            if "barrier" in low_kind:
                op = "barrier"
            elif "fill" in low_kind or "memset" in low_kind:
                op = "copy"
        # Standard dispatches/copies already live in the official domain CSVs.
        if op == "dispatch" or (
            op == "copy" and "fill" not in kind.lower() and "memset" not in kind.lower()
        ):
            return None
        if op not in ("barrier", "copy", "other"):
            return None

        begin_us = _first(row, "Begin_us", "begin_us", default=None)
        end_us = _first(row, "End_us", "end_us", default=None)
        if begin_us is None:
            begin_us = (
                _integer(
                    _first(
                        row, "Start_Timestamp", "start_timestamp", "begin_ns", default=0
                    )
                )
                / 1000.0
            )
        if end_us is None:
            end_us = (
                _integer(
                    _first(row, "End_Timestamp", "end_timestamp", "end_ns", default=0)
                )
                / 1000.0
            )
        begin_us, end_us = _number(begin_us), _number(end_us)
        return {
            "Id": str(_first(row, "Id", "id", default="")),
            "Op": op,
            "Kind": kind,
            "Kernel_Name": "",
            "Bytes": _integer(_first(row, "Bytes", "bytes", "size", default=0)),
            "Device": str(
                _first(row, "Device", "device", "Agent_Id", "agent_id", default="")
            ),
            "Queue": _integer(_first(row, "Queue", "queue", "Queue_Id", default=0)),
            "Stream": _integer(_first(row, "Stream", "stream", "Stream_Id", default=0)),
            "Thread": _integer(_first(row, "Thread", "thread", "Thread_Id", default=0)),
            "Corr": _corr(row),
            "Begin_us": begin_us,
            "End_us": end_us,
            "Duration_us": end_us - begin_us,
        }

    def _iter_extension_ops(self) -> Iterator[dict]:
        for row in self._extension(
            "unmapped_gpu_commands",
            "gpu_operations",
            "nonstandard_operations",
            "operations",
            "commands",
            "ops",
        ):
            if isinstance(row, dict):
                normalized = self._normalize_extension_op(row)
                if normalized is not None:
                    yield normalized

    def iter_ops(self) -> Iterator[dict]:
        """Yield dispatch/copy/extension command rows in timestamp order."""
        if self.is_legacy:
            with self.legacy_ops.open(newline="", encoding="utf-8-sig") as fh:
                yield from csv.DictReader(fh)
            return
        streams = (
            self._iter_kernel_ops(),
            self._iter_copy_ops(),
            self._iter_extension_ops(),
        )
        yield from heapq.merge(
            *streams, key=lambda row: _number(row.get("Begin_us", 0))
        )

    def _kernel_extras(self) -> tuple[dict[str, dict], dict[int, dict]]:
        by_name: dict[str, dict] = {}
        by_id: dict[int, dict] = {}
        for row in self._extension(
            "kernel_properties", "kernel_extras", "kernels", "kernel_info"
        ):
            if not isinstance(row, dict):
                continue
            name = str(_first(row, "Kernel_Name", "kernel_name", "name", default=""))
            kernel_id = _integer(
                _first(row, "Kernel_Id", "kernel_id", "id", "_key", default=0)
            )
            if name:
                by_name[name] = row
            if kernel_id:
                by_id[kernel_id] = row
        return by_name, by_id

    def iter_kernels(self) -> Iterator[dict]:
        """Yield one historical kernel-info row per distinct kernel name."""
        path = self.domain_path("kernels")
        if not path.exists():
            return iter(())
        if self.is_legacy:

            def old_rows():
                with path.open(newline="", encoding="utf-8-sig") as fh:
                    yield from csv.DictReader(fh)

            return old_rows()

        def rows():
            by_name, by_id = self._kernel_extras()
            seen: set[str] = set()
            with path.open(newline="", encoding="utf-8-sig") as fh:
                for row in csv.DictReader(fh):
                    name = row["Kernel_Name"]
                    if name in seen:
                        continue
                    seen.add(name)
                    extra = (
                        by_name.get(name)
                        or by_id.get(_integer(row.get("Kernel_Id", 0)))
                        or {}
                    )
                    yield {
                        "Kernel_Name": name,
                        "Func_Addr": str(
                            _first(extra, "Func_Addr", "func_addr", default="")
                        ),
                        "VGPR": str(
                            _first(
                                extra,
                                "VGPR",
                                "vgpr",
                                "VGPR_Count",
                                default=row.get("VGPR_Count", "0"),
                            )
                        ),
                        "Scratch_Bytes": str(
                            _first(
                                extra,
                                "Scratch_Bytes",
                                "scratch_bytes",
                                "Scratch_Size",
                                default=row.get("Scratch_Size", "0"),
                            )
                        ),
                        "Static_LDS": str(
                            _first(
                                extra,
                                "Static_LDS",
                                "static_lds",
                                "LDS_Block_Size",
                                default=row.get("LDS_Block_Size", "0"),
                            )
                        ),
                        "Max_Threads": str(
                            _first(extra, "Max_Threads", "max_threads", default="")
                        ),
                        "Binary_Version": str(
                            _first(
                                extra, "Binary_Version", "binary_version", default=""
                            )
                        ),
                        "Occ_Block_Size": str(
                            _first(
                                extra, "Occ_Block_Size", "occ_block_size", default=""
                            )
                        ),
                        "Occ_Dyn_LDS": str(
                            _first(extra, "Occ_Dyn_LDS", "occ_dyn_lds", default="")
                        ),
                        "Theoretical_Blocks_Per_CU": str(
                            _first(
                                extra,
                                "Theoretical_Blocks_Per_CU",
                                "theoretical_blocks_per_cu",
                                default="",
                            )
                        ),
                        "Theoretical_Occupancy_Pct": str(
                            _first(
                                extra,
                                "Theoretical_Occupancy_Pct",
                                "theoretical_occupancy_pct",
                                "occupancy_pct",
                                default="",
                            )
                        ),
                    }

        return rows()

    def iter_vram(self) -> Iterator[dict]:
        """Yield historical VRAM sample rows, streaming the extension sidecar."""
        if self.is_legacy:
            path = self.domain_path("vram")
            if not path.exists():
                return iter(())

            def old_rows():
                with path.open(newline="", encoding="utf-8-sig") as fh:
                    yield from csv.DictReader(fh)

            return old_rows()

        def rows():
            for row in self._extension(
                "vram_samples", "vram", "device_memory_samples", "memory_samples"
            ):
                if not isinstance(row, dict):
                    continue
                time_us = _first(row, "Time_us", "time_us", default=None)
                if time_us is None:
                    time_us = (
                        _integer(
                            _first(
                                row,
                                "Timestamp",
                                "timestamp",
                                "timestamp_ns",
                                "time_ns",
                                default=0,
                            )
                        )
                        / 1000.0
                    )
                total = _integer(
                    _first(row, "Total_Bytes", "total_bytes", "total", default=0)
                )
                free = _integer(
                    _first(row, "Free_Bytes", "free_bytes", "free", default=0)
                )
                used = _integer(
                    _first(
                        row,
                        "Used_Bytes",
                        "used_bytes",
                        "used",
                        default=max(0, total - free),
                    )
                )
                yield {
                    "Time_us": _number(time_us),
                    "Free_Bytes": free,
                    "Total_Bytes": total,
                    "Used_Bytes": used,
                }

        return rows()

    def device_name(self) -> str:
        if self.is_legacy:
            path = self.domain_path("summary")
            if not path.exists():
                return ""
            with path.open(encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    if line.startswith("device "):
                        return line.split(":", 1)[1].strip()
            return ""

        path = self.domain_path("agents")
        if not path.exists():
            return ""
        with path.open(newline="", encoding="utf-8-sig") as fh:
            rows = list(csv.DictReader(fh))
        gpu = next((row for row in rows if row.get("Agent_Type") == "GPU"), None)
        if not gpu:
            return ""
        return (
            gpu.get("Product_Name") or gpu.get("Name") or gpu.get("Model_Name") or ""
        ).strip()

    def resume_timestamp_us(self) -> float | None:
        """Return the launcher's QPC timestamp immediately before ResumeThread."""
        path = self.domain_path("launch")
        if not path.exists():
            return None
        with path.open(newline="", encoding="utf-8-sig") as fh:
            row = next(csv.DictReader(fh), None)
        if row is None:
            raise ValueError(f"{path}: empty launch metadata")
        column = "Resume_Timestamp_ns" if self.is_legacy else "Resume_Timestamp"
        timestamp_ns = _integer(row.get(column, 0))
        if timestamp_ns <= 0:
            raise ValueError(f"{path}: invalid resume timestamp")
        return timestamp_ns / 1000.0

    def runtime_info(self) -> dict[str, str]:
        path = self.domain_path("runtime")
        if not path.exists():
            return {}
        with path.open(newline="", encoding="utf-8-sig") as fh:
            row = next(csv.DictReader(fh), None)
        if not row:
            return {}
        return {
            "path": row.get("Hip_Runtime_Path", ""),
            "version": row.get("File_Version", ""),
        }

    def iter_cpu_sched(self) -> Iterator[dict]:
        path = self.domain_path("cpu_sched")
        if not path.exists():
            return iter(())

        def rows():
            with path.open(newline="", encoding="utf-8-sig") as fh:
                for row in csv.DictReader(fh):
                    yield {
                        "Timestamp_us": _integer(row["Timestamp_ns"]) / 1000.0,
                        "Cpu": _integer(row["Cpu"]),
                        "Old_Thread": _integer(row["Old_Thread"]),
                        "New_Thread": _integer(row["New_Thread"]),
                        "Old_State": _integer(row["Old_State"]),
                        "Wait_Reason": _integer(row["Wait_Reason"]),
                    }

        return rows()

    def iter_cpu_samples(self) -> Iterator[dict]:
        path = self.domain_path("cpu_samples")
        if not path.exists():
            return iter(())

        def rows():
            with path.open(newline="", encoding="utf-8-sig") as fh:
                for row in csv.DictReader(fh):
                    yield {
                        "Timestamp_us": _integer(row["Timestamp_ns"]) / 1000.0,
                        "Cpu": _integer(row["Cpu"]),
                        "Process": _integer(row["Process_Id"]),
                        "Thread": _integer(row["Thread_Id"]),
                        "Instruction_Pointer": row["Instruction_Pointer"],
                        "Stack": row["Stack"],
                    }

        return rows()


def _strip_suffix(path: str, suffix: str) -> str:
    return path[: -len(suffix)]


def discover_capture(source: str | os.PathLike | Capture) -> Capture:
    """Resolve a prefix, any domain file, results JSON, or legacy capture path."""
    if isinstance(source, Capture):
        return source

    raw = os.fspath(source)
    lower = raw.lower()
    for suffix in DOMAIN_SUFFIXES:
        if lower.endswith(suffix.lower()):
            prefix = Path(_strip_suffix(raw, suffix))
            anchor = Path(raw)
            if anchor.exists() or any(
                Path(str(prefix) + candidate).exists() for candidate in DOMAIN_SUFFIXES
            ):
                return Capture(prefix=prefix, input_path=anchor)
            break

    if lower.endswith("_ops.csv"):
        path = Path(raw)
        if not path.exists():
            raise CaptureNotFoundError(raw)
        prefix = Path(_strip_suffix(raw, "_ops.csv"))
        return Capture(prefix=prefix, input_path=path, legacy_ops=path)

    for suffix in OLD_SIDECAR_SUFFIXES:
        if lower.endswith(suffix):
            ops = Path(_strip_suffix(raw, suffix))
            if ops.exists() and str(ops).lower().endswith("_ops.csv"):
                prefix = Path(_strip_suffix(str(ops), "_ops.csv"))
                return Capture(prefix=prefix, input_path=Path(raw), legacy_ops=ops)

    prefix = Path(raw)
    if any(Path(raw + suffix).exists() for suffix in DOMAIN_SUFFIXES):
        return Capture(prefix=prefix, input_path=prefix)

    old_ops = Path(raw + "_ops.csv")
    if old_ops.exists():
        return Capture(prefix=prefix, input_path=prefix, legacy_ops=old_ops)

    # Also accept the old CSV stem itself, e.g. ``capture_ops`` for
    # ``capture_ops.csv``.
    old_csv = Path(raw + ".csv")
    if old_csv.exists() and old_csv.name.lower().endswith("_ops.csv"):
        legacy_prefix = Path(_strip_suffix(str(old_csv), "_ops.csv"))
        return Capture(prefix=legacy_prefix, input_path=prefix, legacy_ops=old_csv)

    expected = ", ".join(str(prefix) + suffix for suffix in DOMAIN_SUFFIXES[:5])
    raise CaptureNotFoundError(
        f"{raw}: not a rocprofv3 prefix/domain file or legacy _ops.csv capture; "
        f"looked for {expected}"
    )


def ensure_capture(source: str | os.PathLike | Capture) -> Capture:
    """Alias used by consumers that may already hold a resolved capture."""
    return discover_capture(source)
