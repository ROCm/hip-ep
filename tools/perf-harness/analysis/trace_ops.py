#!/usr/bin/env python3
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Operator inventory from an EP chrome trace (HIPDNN_EP_TRACE_FILE).

This answers "what ops does this model actually run, and in what order within
one step" before any RGP capture is taken. The capture positions itself on an op
by name, so the name has to come from the build under test rather than from the
op list of whatever model the harness was last pointed at.

The trace is not a throughput measurement. HIPDNN_EP_TRACE_FILE implies
hipdnn_ep_perf_enabled(), whose per-inference stream sync distorts exactly the
number this whole exercise is about. Shares here are structural; timings come
from SQTT.

Inferences are recovered from the "Compute" spans on the CPU track, which the
runtime emits once per ORT Run -- the same boundary the RGP fence counts, so a
Run index printed here can be passed straight to `rgp_capture.ps1
-AfterInferences`.
"""

from __future__ import annotations

import argparse
import collections
import json


def load(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        doc = json.load(fh)
    return doc["traceEvents"] if isinstance(doc, dict) else doc


def tracks(events: list[dict]) -> dict[str, int]:
    """Map thread_name -> tid. ChromeTrace emits every op on BOTH a CPU
    (wrapper) and a GPU (stream) track, so counting without splitting them
    doubles every call count."""
    out = {}
    for e in events:
        if e.get("ph") == "M" and e.get("name") == "thread_name":
            out[e["args"]["name"]] = e["tid"]
    return out


def split_runs(events: list[dict]) -> list[list[dict]]:
    """Group GPU op spans into Runs using the outer whole-Compute CPU spans.

    addComputeTotal emits one "[outer] Compute" span per Run on the CPU track
    (the bare "Compute" on the phase track is the pipeline breakdown, not a Run
    boundary). Op spans sit on the GPU track from a different epoch, so rather
    than trusting absolute alignment we cut the op stream at the Compute
    boundaries in time order, which tolerates a constant skew between the axes.
    """
    tid = tracks(events)
    gpu_tid = tid.get("GPU (stream)", 1)
    cpu_tid = tid.get("CPU (wrapper)", 0)
    outer = sorted(
        (
            e
            for e in events
            if e.get("ph") == "X"
            and e.get("tid") == cpu_tid
            and e.get("name") == "[outer] Compute"
        ),
        key=lambda e: e["ts"],
    )
    ops = sorted(
        (e for e in events if e.get("ph") == "X" and e.get("tid") == gpu_tid),
        key=lambda e: e["ts"],
    )
    if not outer:
        return [ops]
    bounds = [o["ts"] for o in outer]
    runs: list[list[dict]] = [[] for _ in bounds]
    i = 0
    for op in ops:
        while i + 1 < len(bounds) and op["ts"] >= bounds[i + 1]:
            i += 1
        runs[i].append(op)
    return runs


def tabulate(ops: list[dict], title: str, top: int = 40) -> None:
    by_op: dict[str, list[float]] = collections.defaultdict(list)
    shapes: dict[str, collections.Counter] = collections.defaultdict(
        collections.Counter
    )
    for e in ops:
        by_op[e["name"]].append(e.get("dur", 0.0))
        sh = (e.get("args") or {}).get("shape")
        if sh:
            shapes[e["name"]][sh] += 1
    total = sum(sum(v) for v in by_op.values()) or 1.0
    print(f"\n=== {title} ===")
    print(f"{'op':<26}{'calls':>7}{'gpu_ms':>10}{'share':>8}  shape (most common)")
    for name, durs in sorted(by_op.items(), key=lambda kv: -sum(kv[1]))[:top]:
        ms = sum(durs) / 1000.0
        common = shapes[name].most_common(1)
        sh = f"{common[0][0]} x{common[0][1]}" if common else "-"
        print(f"{name:<26}{len(durs):>7}{ms:>10.2f}{100*sum(durs)/total:>7.1f}%  {sh}")
    print(f"{'TOTAL':<26}{sum(len(v) for v in by_op.values()):>7}{total/1000:>10.2f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("trace", help="*.json written by HIPDNN_EP_TRACE_FILE")
    ap.add_argument(
        "--run",
        type=int,
        default=None,
        help="inspect a single Run index (default: summarise all)",
    )
    ap.add_argument(
        "--sequence",
        type=int,
        default=0,
        help="print the first N ops of the chosen Run in launch order",
    )
    args = ap.parse_args()

    events = load(args.trace)
    runs = split_runs(events)
    print(f"{args.trace}: {len(events)} events, {len(runs)} Runs")

    # Prefill and decode launch the SAME ops here -- one graph, different M --
    # so op count does not separate them and the token dimension has to. Take it
    # from the gqa shape, which carries sq (query tokens) and skv (KV length)
    # explicitly; sq>1 is prefill, sq==1 is a decode step.
    print(f"\n{'run':>4}{'ops':>6}{'gpu_ms':>9}  phase  gqa shape")
    for i, r in enumerate(runs):
        ms = sum(e.get("dur", 0.0) for e in r) / 1000.0
        gqa = next((e for e in r if e["name"] == "gqa"), None)
        sh = (gqa.get("args") or {}).get("shape", "-") if gqa else "-"
        phase = "prefill" if ("sq=1," not in sh and sh != "-") else "decode"
        if i < 4 or i >= len(runs) - 3:
            print(f"{i:>4}{len(r):>6}{ms:>9.1f}  {phase:<7} {sh}")
        elif i == 4:
            print(f"{'...':>4}")

    if args.run is not None:
        ops = runs[args.run]
        tabulate(ops, f"Run {args.run}")
        if args.sequence:
            print(f"\n--- first {args.sequence} ops of Run {args.run}, launch order ---")
            for e in ops[: args.sequence]:
                sh = (e.get("args") or {}).get("shape", "")
                print(f"  {e['name']:<26}{e.get('dur', 0):>9.1f}us  {sh}")
    else:
        tabulate([e for r in runs for e in r], "all Runs")


if __name__ == "__main__":
    main()
