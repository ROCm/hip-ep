#!/usr/bin/env python3
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Per-decode-step host timeline from HIPDNN_EP_HOST_TIMELINE.

A run with HIPDNN_EP_HOST_TIMELINE=<prefix> leaves two CSVs per process, both on
the same steady_clock microsecond axis:

  <prefix>.ep.<pid>.csv    one row per EP Compute (MlirCustomOp):
                           op, enter, after_marshal, after_compute, exit
  <prefix>.sync.<pid>.csv  one row per hipdnn_ep_stream_sync: stream, enter, exit

Neither adds a sync, so unlike HIPDNN_EP_PERF this is the real timeline and
the throughput it was taken at is the throughput being explained. It answers
the questions an SQTT capture cannot: SQTT stretches idle gaps and host-side
waits heavily, so its step span and its gap numbers are not wall time.

Split of one decode period (decoder enter -> next decoder enter):

  dec_marshal   decoder input marshal (EP host)
  dec_launch    after marshal -> stream_sync entry: the host enqueueing kernels
  dec_sync      stream_sync entry -> exit: waiting for the GPU tail
  dec_post      stream_sync exit -> after compute (runtime epilogue)
  dec_tail      after compute -> Compute exit (D2H of host outputs, EP host)
  other_ep      every other EP Compute in the period (e.g. the VLM embedding)
  outside_ep    host time outside any EP Compute (ORT + GenAI)

The decoder is the op with the largest total time. A decode step is a pair of
consecutive decoder Computes of similar length; prefill chunks and the first
step of each generation fall out on that test, and the slowest 5% of periods
are dropped as warmup / sampling spikes.

GPU-bound vs host-bound: dec_sync > 0 means the GPU was still busy when the
host finished launching, so the step is GPU-bound and host savings inside
dec_launch buy nothing until they exceed the GPU tail. Host time outside the
decoder Compute (outside_ep + other_ep + marshal/tail) is serial with the GPU
and is always on the critical path.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import glob
import json
import os
import re
import statistics
import sys
from collections import defaultdict

COMPONENTS = (
    "dec_marshal",
    "dec_launch",
    "dec_sync",
    "dec_post",
    "dec_tail",
    "other_ep",
    "outside_ep",
)


def _read_ep(path: str) -> list[dict]:
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            rows.append(
                {
                    "op": r["op"],
                    "enter": float(r["enter_us"]),
                    "marshal": float(r["after_marshal_us"]),
                    "compute": float(r["after_compute_us"]),
                    "exit": float(r["exit_us"]),
                }
            )
    rows.sort(key=lambda r: r["enter"])
    return rows


def _read_sync(path: str) -> list[tuple[float, float]]:
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        out = [(float(r["enter_us"]), float(r["exit_us"])) for r in csv.DictReader(f)]
    out.sort()
    return out


def _pct(v: list[float], q: float) -> float:
    s = sorted(v)
    return s[min(len(s) - 1, int(q * len(s)))]


def steps_for_pid(ep_path: str, sync_path: str, drop_top: float) -> dict:
    ep = _read_ep(ep_path)
    sync = _read_sync(sync_path)
    sync_enter = [s[0] for s in sync]
    total = defaultdict(float)
    calls = defaultdict(int)
    for r in ep:
        total[r["op"]] += r["exit"] - r["enter"]
        calls[r["op"]] += 1
    if not total:
        return {"steps": [], "decoder": None, "ops": {}}
    dec = max(total, key=total.get)
    dec_idx = [i for i, r in enumerate(ep) if r["op"] == dec]

    steps = []
    for a, b in zip(dec_idx, dec_idx[1:]):
        p, d = ep[a], ep[b]
        pc, dc = p["compute"] - p["marshal"], d["compute"] - d["marshal"]
        # A prefill chunk next to a decode step is several times longer; so is
        # the first decode step, which pays for the prefill tail.
        if pc <= 0 or dc <= 0 or max(pc, dc) > 3 * min(pc, dc):
            continue
        rec = {
            "period": d["enter"] - p["enter"],
            "dec_marshal": p["marshal"] - p["enter"],
            "dec_tail": p["exit"] - p["compute"],
            "dec_compute": pc,
            "other_ep": 0.0,
            "outside_ep": 0.0,
        }
        prev_exit = p["exit"]
        for r in ep[a + 1 : b]:
            rec["outside_ep"] += max(0.0, r["enter"] - prev_exit)
            rec["other_ep"] += r["exit"] - r["enter"]
            prev_exit = r["exit"]
        rec["outside_ep"] += max(0.0, d["enter"] - prev_exit)
        # The decoder's stream_sync is the last one entered inside its compute.
        j = bisect.bisect_right(sync_enter, p["compute"]) - 1
        if j >= 0 and sync[j][0] >= p["marshal"]:
            s_in, s_out = sync[j]
            rec["dec_launch"] = s_in - p["marshal"]
            rec["dec_sync"] = s_out - s_in
            rec["dec_post"] = p["compute"] - s_out
        else:
            rec["dec_launch"] = pc
            rec["dec_sync"] = 0.0
            rec["dec_post"] = 0.0
            rec["unmatched"] = True
        steps.append(rec)

    if steps and drop_top > 0:
        cut = _pct([s["period"] for s in steps], 1.0 - drop_top)
        steps = [s for s in steps if s["period"] <= cut]
    ops = {
        op: {"calls": calls[op], "total_ms": total[op] / 1e3}
        for op in sorted(total, key=total.get, reverse=True)
    }
    return {
        "steps": steps,
        "decoder": dec,
        "ops": ops,
        "sync_rows": len(sync),
        "sync_matched": sum(1 for s in steps if not s.get("unmatched")),
    }


def resolve_files(prefix: str) -> list[tuple[str, str, str]]:
    """(pid, ep_csv, sync_csv) for every process the prefix captured."""
    out = []
    for ep in sorted(glob.glob(glob.escape(prefix) + ".ep.*.csv")):
        m = re.search(r"\.ep\.(\d+)\.csv$", ep)
        if not m:
            continue
        pid = m.group(1)
        out.append((pid, ep, f"{prefix}.sync.{pid}.csv"))
    return out


def summarize(steps: list[dict]) -> dict:
    keys = ("period", "dec_compute") + COMPONENTS
    return {
        k: {
            "median_us": statistics.median(s[k] for s in steps),
            "p10_us": _pct([s[k] for s in steps], 0.10),
            "p90_us": _pct([s[k] for s in steps], 0.90),
        }
        for k in keys
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("prefix", help="the HIPDNN_EP_HOST_TIMELINE value of the run")
    ap.add_argument("--pid", help="only this process (default: pool all)")
    ap.add_argument(
        "--drop-top",
        type=float,
        default=0.05,
        help="fraction of slowest periods dropped as spikes (default 0.05)",
    )
    ap.add_argument("--json", help="write the summary here (for decode_model.py)")
    args = ap.parse_args(argv)

    files = resolve_files(args.prefix)
    if args.pid:
        files = [f for f in files if f[0] == args.pid]
    if not files:
        print(f"no {args.prefix}.ep.<pid>.csv found", file=sys.stderr)
        return 2

    pooled: list[dict] = []
    per_pid = {}
    for pid, ep, sy in files:
        res = steps_for_pid(ep, sy, args.drop_top)
        per_pid[pid] = {k: v for k, v in res.items() if k != "steps"}
        per_pid[pid]["steps"] = len(res["steps"])
        pooled.extend(res["steps"])
        ops = ", ".join(
            f"{op} x{o['calls']} {o['total_ms']:.0f}ms" for op, o in res["ops"].items()
        )
        print(
            f"pid {pid}: decoder {res['decoder']}, steps {len(res['steps'])}, "
            f"sync rows {res.get('sync_rows', 0)} (matched "
            f"{res.get('sync_matched', 0)}); ops: {ops}"
        )
    if not pooled:
        print(
            "no decode steps found (need >= 2 similar decoder Computes)",
            file=sys.stderr,
        )
        return 2
    if any(s.get("unmatched") for s in pooled):
        n = sum(1 for s in pooled if s.get("unmatched"))
        print(
            f"WARNING: {n}/{len(pooled)} steps had no stream_sync row inside the "
            "decoder compute; their launch/sync split is unknown (counted as launch)"
        )

    summ = summarize(pooled)
    med = {k: v["median_us"] for k, v in summ.items()}
    print(f"\ndecode steps: {len(pooled)}   (medians; p10..p90 in brackets)")
    print(
        f"  {'period':12s} {med['period']:9.1f} us  -> {1e6 / med['period']:.2f} tok/s"
    )
    for k in COMPONENTS:
        v = summ[k]
        print(
            f"  {k:12s} {v['median_us']:9.1f} us  [{v['p10_us']:.1f}..{v['p90_us']:.1f}]"
        )
    host_serial = (
        med["dec_marshal"] + med["dec_tail"] + med["other_ep"] + med["outside_ep"]
    )
    print(
        f"\n  host serial with the GPU (marshal+tail+other_ep+outside_ep): "
        f"{host_serial:.0f} us = {100 * host_serial / med['period']:.1f}% of period"
    )
    print(
        f"  decoder compute {med['dec_compute']:.0f} us: launch {med['dec_launch']:.0f}"
        f" + sync wait {med['dec_sync']:.0f} + post {med['dec_post']:.0f}"
    )
    if med["dec_sync"] > 0:
        print(
            "  sync wait > 0: the GPU outlasts the launch phase, so the decoder is "
            "GPU-bound; launch-side savings do not shorten the step."
        )

    if args.json:
        with open(args.json, "w") as f:
            json.dump(
                {
                    "prefix": args.prefix,
                    "steps": len(pooled),
                    "per_pid": per_pid,
                    "summary": summ,
                },
                f,
                indent=2,
            )
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
