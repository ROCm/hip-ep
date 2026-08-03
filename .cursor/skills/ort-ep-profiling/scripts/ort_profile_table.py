#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""Summarize onnxruntime_perf_test -p profile JSON into an op/shape table."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from ort_profile_shapes import shape_detail  # noqa: E402


def last_session_nodes(trace_path: Path) -> list[dict]:
    import json

    with trace_path.open(encoding="utf-8") as fp:
        trace = json.load(fp)

    sessions = [x for x in trace if x["cat"] == "Session" and x["name"] == "model_run"]
    if not sessions:
        raise ValueError(f"No Session/model_run events found in {trace_path}")

    last_session = max(sessions, key=lambda x: x["ts"])
    start = last_session["ts"]
    end = last_session["ts"] + last_session["dur"]

    return [
        x
        for x in trace
        if x["cat"] == "Node" and x["ts"] > start and (x["ts"] + x["dur"]) < end
    ]


def summarize_trace(trace_path: Path) -> list[tuple[str, str, int, float]]:
    stats: dict[str, dict[str, dict[str, float | int]]] = {}

    for node in last_session_nodes(trace_path):
        node_args = node["args"]
        op_type = node_args["op_name"]
        op_stats = stats.setdefault(op_type, {})

        shape = shape_detail(node_args)
        op_shape_stats = op_stats.setdefault(shape, {})

        op_shape_stats["num_calls"] = int(op_shape_stats.get("num_calls", 0)) + 1
        op_shape_stats["latency"] = float(op_shape_stats.get("latency", 0.0)) + (
            node["dur"] / 1000
        )

    rows: list[tuple[str, str, int, float]] = []
    for op_type, op_stats in stats.items():
        for shape, op_shape_stats in op_stats.items():
            rows.append(
                (
                    op_type,
                    shape,
                    int(op_shape_stats["num_calls"]),
                    float(op_shape_stats["latency"]),
                )
            )
    return rows


def summarize_by_op(
    rows: list[tuple[str, str, int, float]],
) -> list[tuple[str, int, float]]:
    totals: dict[str, dict[str, float | int]] = {}
    for op_type, _, num_calls, latency_ms in rows:
        op_total = totals.setdefault(op_type, {"calls": 0, "time_ms": 0.0})
        op_total["calls"] = int(op_total["calls"]) + num_calls
        op_total["time_ms"] = float(op_total["time_ms"]) + latency_ms

    return sorted(
        (
            (op_type, int(stats["calls"]), float(stats["time_ms"]))
            for op_type, stats in totals.items()
        ),
        key=lambda item: -item[2],
    )


def print_op_summary(rows: list[tuple[str, str, int, float]], out_fp) -> None:
    print("Op summary (sorted by total time):", file=out_fp)
    print("Op", "Calls", "Time (ms)", sep="\t", file=out_fp)
    for op_type, num_calls, latency_ms in summarize_by_op(rows):
        print(op_type, num_calls, f"{latency_ms:.3f}", sep="\t", file=out_fp)


def print_table(rows: list[tuple[str, str, int, float]], out_fp) -> None:
    print("Op", "Detail", "Calls", "Time (ms)", sep="\t", file=out_fp)
    for op_type, shape, num_calls, latency_ms in rows:
        print(op_type, shape, num_calls, f"{latency_ms:.3f}", sep="\t", file=out_fp)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert onnxruntime_perf_test profile JSON to an op/shape table."
    )
    parser.add_argument(
        "trace_path", type=Path, help="profile_<timestamp>.json from -p profile"
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Write TSV here (default: stdout)",
    )
    args = parser.parse_args()

    if not args.trace_path.is_file():
        print(f"error: trace file not found: {args.trace_path}", file=sys.stderr)
        return 1

    rows = summarize_trace(args.trace_path)
    print_op_summary(rows, sys.stderr)
    print(file=sys.stderr)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8", newline="\n") as fp:
            print_table(rows, fp)
        print(f"wrote {args.output}", file=sys.stderr)
    else:
        print_op_summary(rows, sys.stdout)
        print(file=sys.stdout)
        print_table(rows, sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
