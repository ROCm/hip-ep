#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Merge several Chrome-trace JSON files into ONE.

chrome://tracing and Perfetto only load a single file, so this stitches N
traces together. Each input is placed in its own process (``pid``) block so the
inputs render as separate, stacked track groups in the viewer.

Chrome Trace Event Format (both input and output look like this)::

    {
      "displayTimeUnit": "ns",
      "traceEvents": [
        {"name": "process_name", "ph": "M", "pid": 0, "tid": 0,
         "args": {"name": "hipdnn-ep session p1234"}},
        {"name": "thread_name",  "ph": "M", "pid": 0, "tid": 1,
         "args": {"name": "GPU (stream)"}},
        {"name": "matmul", "cat": "gpu", "ph": "X", "pid": 0, "tid": 1,
         "ts": 12345.6, "dur": 40.0, "args": {"shape": "m=3,n=3,k=4"}},
        {"name": "[outer] Compute", "cat": "total", "ph": "X", "pid": 0,
         "tid": 0, "ts": 12340.0, "dur": 120.0, "args": {}}
      ]
    }

  * ``"ph": "X"`` -- a complete duration span: ``ts`` = start (us), ``dur`` =
    length (us) on the shared time axis.
  * ``"ph": "M"`` -- metadata (``process_name`` / ``thread_name``), naming the
    ``pid``/``tid`` tracks.

An input may also be a bare JSON list of events (no ``traceEvents`` wrapper);
both shapes are accepted.

Clock handling:
  * Same-clock inputs (e.g. EP per-session files sharing the absolute trace
    axis produced by ``op_profile.cpp``) already align -- do NOT pass --zero.
  * Different-run / different-clock inputs (EP vs SQTT vs RGP golden) do not
    share a clock; pass --zero to shift each input so it starts at t=0, letting
    them stack from a common origin for shape comparison (they are NOT
    wall-clock aligned across runs).
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, List

# Each input's pids are offset by (index + 1) * PID_STRIDE so inputs never
# collide and render as separate process groups in the viewer.
PID_STRIDE = 1000


def load_trace_events(path: str) -> List[Dict[str, Any]]:
    """Read a trace file and return its list of trace events.

    Accepts both the standard ``{"traceEvents": [...]}`` object and a bare
    top-level list of events.
    """
    with open(path) as f:
        data = json.load(f)
    return data["traceEvents"] if isinstance(data, dict) else data


def _min_span_timestamp(events: List[Dict[str, Any]]) -> float:
    """Earliest start timestamp among duration (``ph=="X"``) events, or 0.0."""
    starts = [e["ts"] for e in events if e.get("ph") == "X" and "ts" in e]
    return min(starts) if starts else 0.0


def _reindex_input(
    events: List[Dict[str, Any]], index: int, label: str, zero: bool
) -> List[Dict[str, Any]]:
    """Return a copy of one input's events, offset into its own pid block.

    Every event's ``pid`` is shifted by ``(index + 1) * PID_STRIDE``; when
    ``zero`` is set, duration-event timestamps are rebased so the input starts
    at t=0. A ``process_name`` metadata event is appended per distinct pid so
    the input is clearly labeled in the merged view.
    """
    base_pid = (index + 1) * PID_STRIDE
    shift = _min_span_timestamp(events) if zero else 0.0

    out: List[Dict[str, Any]] = []
    seen_pids = set()
    for event in events:
        event = dict(event)  # don't mutate the caller's parsed data
        if "pid" in event:
            event["pid"] = base_pid + int(event["pid"])
            seen_pids.add(event["pid"])
        if zero and event.get("ph") == "X" and "ts" in event:
            event["ts"] = event["ts"] - shift
        out.append(event)

    for pid in seen_pids or {base_pid}:
        out.append(
            {
                "name": "process_name",
                "ph": "M",
                "pid": pid,
                "tid": 0,
                "args": {"name": f"[{index}] {label}"},
            }
        )
    return out


def merge_traces(inputs: List[str], zero: bool) -> List[Dict[str, Any]]:
    """Merge the given trace files into a single list of trace events."""
    merged: List[Dict[str, Any]] = []
    for index, path in enumerate(inputs):
        events = load_trace_events(path)
        merged.extend(_reindex_input(events, index, os.path.basename(path), zero))
    return merged


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="merge_chrome_traces.py",
        description="Merge several Chrome-trace JSON files into one.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("output", help="path to write the merged trace JSON")
    parser.add_argument("inputs", nargs="+", help="input trace JSON files to merge")
    parser.add_argument(
        "--zero",
        action="store_true",
        help="rebase each input's timestamps to start at t=0 (for inputs that "
        "do not share a clock); omit for same-clock EP session files",
    )
    return parser.parse_args(argv)


def main(argv: List[str]) -> int:
    args = parse_args(argv)

    merged = merge_traces(args.inputs, args.zero)

    with open(args.output, "w") as f:
        json.dump({"displayTimeUnit": "ns", "traceEvents": merged}, f)

    span_count = sum(1 for e in merged if e.get("ph") == "X")
    print(
        f"wrote {args.output}: {len(args.inputs)} inputs, "
        f"{span_count} span events, zero={args.zero}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
