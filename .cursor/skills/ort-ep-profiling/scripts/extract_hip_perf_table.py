#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""Extract the last HIPDNN_EP_PERF op table from onnxruntime_perf_test output."""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

SEP = re.compile(r"^\[PERF\] ={10,}$")
HEADER = re.compile(r"calls\s+gpu \(ms\)\s+cpu \(ms\)\s+gpu %")
DATA_ROW = re.compile(
    r"^\[PERF\](?P<prefix>    |  )"
    r"(?P<name>.+?)"
    r"\s+(?P<calls>\d+)"
    r"\s+(?P<gpu>[\d.]+|n/a)"
    r"\s+(?P<cpu>[\d.]+)"
    r"(?:\s+(?P<pct>[\d.]+%|n/a))?"
    r"\s*$"
)
TOTAL_ROW = re.compile(r"^\[PERF\]\s+TOTAL\s+(?P<gpu>[\d.]+)\s+(?P<cpu>[\d.]+)\s*$")


@dataclass
class ShapeRow:
    shape: str
    calls: int
    gpu_ms: float | None
    cpu_ms: float
    gpu_pct: float | None


@dataclass
class OpRow:
    name: str
    calls: int
    gpu_ms: float | None
    cpu_ms: float
    gpu_pct: float | None
    is_outer: bool = False
    shapes: list[ShapeRow] = field(default_factory=list)

    @property
    def has_gpu(self) -> bool:
        return self.gpu_ms is not None


def read_log_text(path: Path) -> str:
    raw = path.read_bytes()
    if raw.startswith(b"\xff\xfe") or raw.startswith(b"\xfe\xff"):
        return raw.decode("utf-16")
    if raw.startswith(b"\xef\xbb\xbf"):
        return raw.decode("utf-8-sig")
    return raw.decode("utf-8", errors="replace")


def _parse_float(value: str) -> float | None:
    if value == "n/a":
        return None
    return float(value)


def _parse_pct(value: str | None) -> float | None:
    if value is None or value == "n/a":
        return None
    return float(value.rstrip("%"))


def parse_perf_table(block: str) -> tuple[list[OpRow], float, float]:
    ops: list[OpRow] = []
    current: OpRow | None = None
    total_gpu = 0.0
    total_cpu = 0.0

    for line in block.splitlines():
        if SEP.match(line) or HEADER.search(line):
            continue

        total_match = TOTAL_ROW.match(line)
        if total_match:
            total_gpu = float(total_match.group("gpu"))
            total_cpu = float(total_match.group("cpu"))
            continue

        row_match = DATA_ROW.match(line)
        if not row_match:
            continue

        indent = row_match.group("prefix")
        name = row_match.group("name").strip()
        calls = int(row_match.group("calls"))
        gpu_ms = _parse_float(row_match.group("gpu"))
        cpu_ms = float(row_match.group("cpu"))
        gpu_pct = _parse_pct(row_match.group("pct"))

        if indent == "    ":
            if current is None:
                continue
            current.shapes.append(
                ShapeRow(
                    shape=name,
                    calls=calls,
                    gpu_ms=gpu_ms,
                    cpu_ms=cpu_ms,
                    gpu_pct=gpu_pct,
                )
            )
            continue

        current = OpRow(
            name=name, calls=calls, gpu_ms=gpu_ms, cpu_ms=cpu_ms, gpu_pct=gpu_pct
        )
        current.is_outer = name.startswith("[outer]")
        ops.append(current)

    if not ops:
        raise ValueError("No [PERF] op rows found in extracted table block")

    return ops, total_gpu, total_cpu


def sort_perf_rows(ops: list[OpRow]) -> None:
    for op in ops:
        op.shapes.sort(key=lambda shape: -(shape.gpu_ms or 0.0))

    ops.sort(
        key=lambda op: (
            -int(op.has_gpu),
            -(op.gpu_ms or 0.0) if op.has_gpu else 0.0,
            -op.cpu_ms if not op.has_gpu else 0.0,
        )
    )


def format_perf_table(ops: list[OpRow], total_gpu: float, total_cpu: float) -> str:
    max_name_len = 22
    for op in ops:
        max_name_len = max(max_name_len, len(op.name))
        for shape in op.shapes:
            if shape.shape:
                max_name_len = max(max_name_len, len(shape.shape) + 2)

    line_width = max_name_len + 2 + 5 + 1 + 9 + 1 + 9 + 1 + 6
    lines = [
        "[PERF] " + "=" * line_width,
        f"[PERF]  {'':<{max_name_len}} {'calls':>5} {'gpu (ms)':>9} {'cpu (ms)':>9} {'gpu %':>6}",
    ]

    grand_total_gpu = sum(op.gpu_ms or 0.0 for op in ops if op.has_gpu)
    if total_gpu <= 0:
        total_gpu = grand_total_gpu

    for op in ops:
        if op.has_gpu:
            pct = op.gpu_ms / total_gpu * 100 if total_gpu > 0 else 0.0
            lines.append(
                f"[PERF]  {op.name:<{max_name_len}} {op.calls:5d} {op.gpu_ms:9.1f} {op.cpu_ms:9.1f} {pct:5.1f}%"
            )
        else:
            lines.append(
                f"[PERF]  {op.name:<{max_name_len}} {op.calls:5d} {'n/a':>9} {op.cpu_ms:9.1f} {'n/a':>6}"
            )

        has_shapes = len(op.shapes) > 1 or (len(op.shapes) == 1 and op.shapes[0].shape)
        if has_shapes:
            for shape in op.shapes:
                if not shape.shape:
                    continue
                pct = (
                    shape.gpu_ms / total_gpu * 100
                    if total_gpu > 0 and shape.gpu_ms is not None
                    else 0.0
                )
                gpu_text = (
                    f"{shape.gpu_ms:9.1f}"
                    if shape.gpu_ms is not None
                    else f"{'n/a':>9}"
                )
                pct_text = f"{pct:5.1f}%" if shape.gpu_ms is not None else f"{'n/a':>6}"
                lines.append(
                    f"[PERF]    {shape.shape:<{max_name_len - 2}} {shape.calls:5d} {gpu_text} {shape.cpu_ms:9.1f} {pct_text}"
                )

    lines.append(
        f"[PERF]  {'TOTAL':<{max_name_len}} {'':>5} {total_gpu:9.1f} {total_cpu:9.1f}"
    )
    lines.append("[PERF] " + "=" * line_width)
    return "\n".join(lines)


def extract_last_perf_table(text: str) -> str:
    lines = text.splitlines()
    sep_indices = [i for i, line in enumerate(lines) if SEP.match(line)]

    tables: list[str] = []
    for idx, start in enumerate(sep_indices):
        if idx + 1 >= len(sep_indices):
            break
        end = sep_indices[idx + 1]
        block = lines[start : end + 1]
        if any(HEADER.search(line) for line in block):
            tables.append("\n".join(block))

    if not tables:
        raise ValueError(
            "No [PERF] op table found (expected separator + calls/gpu/cpu header block)"
        )

    ops, total_gpu, total_cpu = parse_perf_table(tables[-1])
    sort_perf_rows(ops)
    return format_perf_table(ops, total_gpu, total_cpu)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract the last HIPDNN_EP_PERF per-op table from a perf_test log."
    )
    parser.add_argument(
        "log_path", type=Path, help="Captured stdout+stderr from perf_test"
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Write extracted table here (default: stdout)",
    )
    args = parser.parse_args()

    if not args.log_path.is_file():
        print(f"error: log file not found: {args.log_path}", file=sys.stderr)
        return 1

    table = extract_last_perf_table(read_log_text(args.log_path))

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(table + "\n", encoding="utf-8")
        print(f"wrote {args.output}", file=sys.stderr)
    else:
        print(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
