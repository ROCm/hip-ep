#!/usr/bin/env python3
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Diff two decode captures by kernel: dispatch count and busy time.

The per-item question this answers is narrow -- did the change remove the
dispatches it claimed to, and did the kernels it targeted get cheaper -- and it
is the only question a capture can answer honestly. Whether the step got faster
is for bench_tps.ps1; RGP pegs clocks, so a capture systematically flatters any
change that trades power for dispatches.

Both captures must be positioned identically (same -Op / -DecodeStep / -SeqLen),
or the windows hold different amounts of work and every row is noise. The
normaliser below is the count of the op the fence armed on, which is constant
per decode step, so a window holding 1.05 steps is reported as such rather than
silently inflating the totals.
"""

import argparse
import collections
import csv


def _rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _field(row, names):
    for n in names:
        if n in row and row[n] not in ("", None):
            return row[n]
    return None


def load(path, layers):
    rows = _rows(path)
    if not rows:
        raise SystemExit(f"{path}: no dispatches")
    count = collections.Counter()
    busy = collections.Counter()
    for r in rows:
        name = _field(r, ("kernel", "name", "kernel_name")) or "?"
        # Tensile kernels carry their whole tile configuration in the name, which
        # is 400 characters of schedule flags. Keep enough to tell two tilings
        # apart and no more.
        name = name.split("(")[0].strip()
        if len(name) > 52:
            name = name[:52] + "~"
        count[name] += 1
        d = _field(r, ("dur_us", "duration_us", "duration", "gpu_time_us", "time_us"))
        if d is not None:
            busy[name] += float(d)
    # topk_routing runs exactly once per layer per decode step, so it measures
    # how much of a step this window actually holds.
    router = sum(v for k, v in count.items() if k.startswith("topk_routing"))
    steps = (router / layers) if router else 1.0
    return count, busy, steps


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("before")
    ap.add_argument("after")
    ap.add_argument(
        "--layers",
        type=int,
        default=48,
        help="decoder layers, for the per-step normaliser (default: 48)",
    )
    ap.add_argument(
        "--min-us",
        type=float,
        default=10.0,
        help="hide rows whose count is unchanged and whose time moved less than this",
    )
    args = ap.parse_args()

    ca, ta, sa = load(args.before, args.layers)
    cb, tb, sb = load(args.after, args.layers)

    print(f"before : {args.before}")
    print(f"         {sum(ca.values())} dispatches over {sa:.2f} decode steps")
    print(f"after  : {args.after}")
    print(f"         {sum(cb.values())} dispatches over {sb:.2f} decode steps")
    if abs(sa - sb) > 0.15:
        print(
            "  WARNING: the windows hold different amounts of work; "
            "per-step figures below are normalised but treat them with care"
        )

    print(f"\n{'kernel':54} {'n/step':>14} {'us/step':>22}")
    print(f"{'':54} {'before':>6} {'after':>7} {'before':>9} {'after':>9} {'delta':>9}")
    keys = sorted(set(ca) | set(cb), key=lambda k: -(ta[k] / sa + tb[k] / sb))
    for k in keys:
        na, nb = ca[k] / sa, cb[k] / sb
        ua, ub = ta[k] / sa, tb[k] / sb
        if abs(na - nb) < 0.5 and abs(ua - ub) < args.min_us:
            continue
        print(f"{k:54} {na:6.0f} {nb:7.0f} {ua:9.1f} {ub:9.1f} {ub - ua:+9.1f}")

    dn = sum(cb.values()) / sb - sum(ca.values()) / sa
    du = sum(tb.values()) / sb - sum(ta.values()) / sa
    print(
        f"\n{'TOTAL':54} {sum(ca.values()) / sa:6.0f} {sum(cb.values()) / sb:7.0f} "
        f"{sum(ta.values()) / sa:9.1f} {sum(tb.values()) / sb:9.1f} {du:+9.1f}"
    )
    print(f"\ndispatches per step: {dn:+.0f}")
    print(
        f"busy us per step   : {du:+.1f}  ({100 * du / (sum(ta.values()) / sa):+.2f}%)"
    )
    print(
        "\nCapture time is not TPS: clocks are pegged here. Confirm with "
        "ab_interleaved.ps1 -Metric tps before believing it."
    )


if __name__ == "__main__":
    main()
