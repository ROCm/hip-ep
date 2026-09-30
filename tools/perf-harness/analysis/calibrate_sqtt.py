#!/usr/bin/env python3
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Per-family SQTT inflation factors, from a capture and event-timed runs.

SQTT does not slow every kernel by the same amount, so one factor from the
capture's step span against the measured ms/token (decode_model --measured-ms)
misattributes time between families -- and folds the capture's inflated idle
gaps into kernel time on top. This measures each family against itself:

  SQTT      kernel time per decode step of the op's kernel families, from an
            rgp_parser *_dispatches.csv (complete steps only; see
            perfcommon.decode_step_windows).
  event     GPU time per decode step of the same op from a run with
            HIPDNN_EP_PERF_OPS=<op> (bench_tps.ps1 -PerfOps), which puts one
            start/end event pair around that op's calls and no others.

factor = SQTT / event. decode_model.py --calib divides each family by
max(1, factor): both numbers are upper bounds on the uninstrumented time, so
it keeps the smaller.

Two cross-checks, because a factor built on a wrong mapping looks just as
plausible as a right one:
  - dispatches per call (SQTT dispatches / event calls, per step) must be a
    near-integer >= 1: an op launches a whole number of kernels per call.
  - families the map assigns to no op are listed with their share, so a gap in
    coverage is visible rather than silently uncalibrated.

Neither side is the truth. SQTT stretches kernels; the event span adds the
marker packets themselves and the gaps between the op's own kernels, which
summed SQTT durations do not. On gemma-4-26B decode matmul_nbits read 11.57 ms
by events against 9.27 ms in SQTT (~10 us per call), and taking the event
number put the calibrated kernels 2.8 ms over the decoder's measured compute.
A factor below 1 therefore means "SQTT is the tighter bound here", not "SQTT
under-reports".
Both runs must use the same build and env (CI's), and the same prompt length:
kernel choice and KV length change per-family time.

The marker packets can be measured instead of argued about. The decoder is
GPU-bound (host_timeline.py: sync wait > 0), so whatever the markers add to the
GPU shows up one-for-one in the decoder's launch + sync wait per step. With
--base-timeline and --op-timeline OP=..., both host_timeline.py --json files:
  marker ms = op run's (dec_launch + dec_sync) - base run's
  event ms  = event ms - marker ms           (the factor uses this)
The base should be a run with HIPDNN_EP_PERF_OPS set to a name no op has
(bench_tps.ps1 -PerfOps none): PERF mode's own per-inference events and sync
then cancel, leaving only the op's start/end pairs. A plain run works too but
folds PERF mode's fixed cost into the marker estimate. A delta below zero or
at least the op's event time is noise or a broken run, so it is reported and
not applied.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys

from perfcommon import (
    LAYER_MARKERS,
    add_model_args,
    decode_op_stats,
    decode_step_windows,
    load_dispatches,
    op_family_map,
    op_for_family,
    perf_op_tables,
    specs_from_args,
)


def sqtt_per_step(
    path: str, layers: int, fmap: dict
) -> tuple[dict[str, dict], dict[str, float], dict]:
    """Per-op and unmapped-family SQTT us per decode step, decoder queue only."""
    rows = load_dispatches(path)
    wins, main_q, rejected = decode_step_windows(rows, layers)
    info = {"decoder_queue": main_q, "rejected_windows": rejected}
    if wins:
        sel = [r for s, e in wins for r in rows[s:e] if r.get("queue") == main_q]
        steps = float(len(wins))
        info["method"] = f"{len(wins)} complete step(s) between queue switches"
    else:
        # Single-queue (text) capture: count steps by the per-layer marker.
        counts = collections.Counter(r["family"] for r in rows)
        marker = next((f for f in LAYER_MARKERS if counts.get(f)), None)
        if not marker or not layers:
            raise SystemExit(
                f"{path}: no queue switches and no layer marker; cannot tell "
                "how many decode steps the window holds"
            )
        sel = rows
        steps = counts[marker] / layers
        info["method"] = f"{counts[marker]} {marker} / {layers} layers"
    info["steps"] = steps
    per_op: dict[str, dict] = collections.defaultdict(
        lambda: {"us": 0.0, "n": 0, "fams": collections.Counter()}
    )
    unmapped: dict[str, float] = collections.defaultdict(float)
    total = 0.0
    for r in sel:
        dur = float(r["dur_us"])
        total += dur
        op = op_for_family(r["family"], fmap)
        if op is None:
            unmapped[r["family"]] += dur
        else:
            per_op[op]["us"] += dur
            per_op[op]["n"] += 1
            per_op[op]["fams"][r["family"]] += 1
    info["kernel_us_per_step"] = total / steps
    return (
        {
            op: {
                "us": v["us"] / steps,
                "n": v["n"] / steps,
                "fams": {f: c / steps for f, c in v["fams"].items()},
            }
            for op, v in per_op.items()
        },
        {f: us / steps for f, us in unmapped.items()},
        info,
    )


def dec_gpu_ms(path: str) -> float:
    """Median decoder launch + sync wait per step, from a host_timeline.py
    --json file. dec_post is left out: PERF mode resolves its events there,
    after the sync, so it is host time the markers' GPU cost is not in."""
    with open(path) as f:
        s = json.load(f)["summary"]
    return (s["dec_launch"]["median_us"] + s["dec_sync"]["median_us"]) / 1000.0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        epilog=__doc__.split("\n\n", 1)[1],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "capture",
        nargs="?",
        help="*_dispatches.csv from tools/rgp_parser; omit to print the event-timed "
        "family table only",
    )
    ap.add_argument(
        "--perf-log",
        action="append",
        required=True,
        help="log of a HIPDNN_EP_PERF_OPS run (repeat, one per family)",
    )
    ap.add_argument(
        "--map",
        action="append",
        default=[],
        metavar="OP=FAM[,FAM*]",
        help="assign kernel families to an op (replaces its default; '*' = prefix)",
    )
    ap.add_argument(
        "--accept",
        action="append",
        default=[],
        metavar="OP",
        help="OP's fractional dispatches/call was checked by hand (e.g. hipBLASLt "
        "adds a PostGSU reduction kernel on some layers only); recorded in the JSON",
    )
    ap.add_argument(
        "--base-timeline",
        metavar="JSON",
        help="host_timeline.py --json of the marker-free run (-PerfOps none)",
    )
    ap.add_argument(
        "--op-timeline",
        action="append",
        default=[],
        metavar="OP=JSON",
        help="host_timeline.py --json of OP's -PerfOps run; subtracts the markers' "
        "GPU cost from OP's event time (needs --base-timeline)",
    )
    ap.add_argument("--json", help="write the factors here (for decode_model --calib)")
    add_model_args(ap)
    args = ap.parse_args(argv)
    spec, _dev = specs_from_args(args)
    fmap = op_family_map(args.map)
    op_tl = {}
    for item in args.op_timeline:
        op, sep, path = item.partition("=")
        if not sep or not path:
            ap.error(f"--op-timeline expects OP=JSON, got {item!r}")
        op_tl[op] = path
    if op_tl and not args.base_timeline:
        ap.error("--op-timeline needs --base-timeline")
    base_ms = dec_gpu_ms(args.base_timeline) if args.base_timeline else None

    ops: dict[str, dict] = {}
    for log in args.perf_log:
        tables = perf_op_tables(log)
        if not tables:
            print(f"WARNING: {log}: no [PERF] op tables", file=sys.stderr)
            continue
        for op in sorted({o for t in tables for o in t}):
            st = decode_op_stats(tables, op)
            if not st or op in ops:
                continue
            ops[op] = dict(st, log=log)
    # An op seen in only a handful of tables ran outside the decode loop (e.g.
    # the vision encoder's matmul), so its "per step" number is not one.
    most = max((v["steps"] for v in ops.values()), default=0)
    for op in [o for o, v in ops.items() if v["steps"] < most / 10]:
        print(f"skipping {op}: {ops[op]['steps']} tables vs {most}; not a decode op")
        del ops[op]

    for op, ev in ops.items():
        ev["raw_ms"] = ev["gpu_ms"]
        ev["marker_ms"] = None
        if op not in op_tl:
            continue
        delta = dec_gpu_ms(op_tl[op]) - base_ms
        if 0.0 <= delta < ev["raw_ms"]:
            ev["marker_ms"] = delta
            ev["gpu_ms"] = ev["raw_ms"] - delta
            print(
                f"{op}: markers add {delta:.3f} ms/step "
                f"({1000 * delta / ev['calls']:.1f} us/call); event "
                f"{ev['raw_ms']:.3f} -> {ev['gpu_ms']:.3f} ms"
            )
        else:
            print(
                f"WARNING: {op}: marker delta {delta:.3f} ms/step is outside "
                f"[0, {ev['raw_ms']:.3f}); not applied"
            )

    if not args.capture:
        print(f"{'op':<14}{'event ms/step':>14}{'calls':>7}{'steps':>7}  log")
        for op, ev in ops.items():
            print(
                f"{op:<14}{ev['gpu_ms']:>14.3f}{ev['calls']:>7}{ev['steps']:>7}  {ev['log']}"
            )
        return 0

    sqtt, unmapped, info = sqtt_per_step(args.capture, spec.layers, fmap)
    print(f"capture : {args.capture}")
    print(
        f"steps   : {info['steps']:g} ({info['method']}; decoder queue "
        f"{info['decoder_queue']}"
        + (
            f"; {info['rejected_windows']} window(s) rejected"
            if info["rejected_windows"]
            else ""
        )
        + ")"
    )
    print(f"kernels : {info['kernel_us_per_step'] / 1000:.3f} ms/step SQTT\n")

    head = f"{'op':<14}{'event ms':>9}{'calls':>7}{'SQTT ms':>9}{'disp':>7}{'d/call':>8}{'factor':>8}  check"
    print(head)
    print("-" * len(head))
    out: dict[str, dict] = {}
    bad = False
    for op, ev in ops.items():
        sq = sqtt.get(op)
        if sq is None:
            print(
                f"{op:<14}{ev['gpu_ms']:>9.3f}{ev['calls']:>7}{'-':>9}{'-':>7}{'-':>8}{'-':>8}  no mapped families in the capture"
            )
            continue
        per_call = sq["n"] / ev["calls"] if ev["calls"] else 0.0
        factor = (sq["us"] / 1000.0) / ev["gpu_ms"] if ev["gpu_ms"] else 0.0
        whole = per_call >= 0.95 and abs(per_call - round(per_call)) <= 0.1
        accepted = not whole and op in args.accept
        ok = whole or accepted
        bad |= not ok
        check = (
            "ok"
            if whole
            else "accepted by hand (--accept)"
            if accepted
            else "MAPPING? dispatches/call is not a whole number"
        )
        print(
            f"{op:<14}{ev['gpu_ms']:>9.3f}{ev['calls']:>7}{sq['us'] / 1000:>9.3f}"
            f"{sq['n']:>7.1f}{per_call:>8.2f}{factor:>8.3f}  {check}"
        )
        if not whole:
            for fam, n in sorted(sq["fams"].items(), key=lambda kv: -kv[1]):
                print(f"{'':14}  {fam:<40}{n:>7.1f} /step = {n / ev['calls']:.2f}/call")
        out[op] = {
            "factor": factor,
            "event_ms_per_step": ev["gpu_ms"],
            "event_ms_raw": ev["raw_ms"],
            "marker_ms_per_step": ev["marker_ms"],
            "event_calls_per_step": ev["calls"],
            "event_steps": ev["steps"],
            "sqtt_ms_per_step": sq["us"] / 1000.0,
            "sqtt_dispatches_per_step": sq["n"],
            "dispatches_per_call": per_call,
            "check_ok": ok,
            "check_accepted": accepted,
            "families": {k: list(v) for k, v in fmap[op].items()},
            "perf_log": ev["log"],
        }

    tot = info["kernel_us_per_step"]
    cal_us = sum(sqtt[o]["us"] for o in out)
    print(
        f"\ncalibrated ops cover {100 * cal_us / tot:.1f}% of SQTT kernel time;"
        " uncalibrated families stay SQTT-relative in decode_model."
    )
    if unmapped:
        print("families mapped to no op (share of SQTT kernel time):")
        for fam, us in sorted(unmapped.items(), key=lambda kv: -kv[1])[:12]:
            print(f"  {fam:<34}{us / 1000:>8.3f} ms {100 * us / tot:>6.1f}%")
    if bad:
        print(
            "\nWARNING: at least one op failed the dispatches/call check. Fix --map, or"
            " if the per-family counts above show a kernel only some calls launch,"
            " find it in the capture and pass --accept OP."
        )

    if args.json:
        with open(args.json, "w") as f:
            json.dump(
                {
                    "capture": args.capture,
                    "steps": info["steps"],
                    "sqtt_kernel_ms_per_step": tot / 1000.0,
                    "base_timeline": args.base_timeline,
                    "op_timelines": op_tl,
                    "ops": out,
                },
                f,
                indent=2,
            )
        print(f"\nwrote {args.json}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
