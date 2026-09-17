#!/usr/bin/env python3
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Paired summary of an interleaved A/B run.

Pairing is by round, so drift shared by the arms within a round cancels; the
reported interval is over the paired differences, not over the raw TTFTs, which
are far noisier. With --drop you can exclude rounds taken outside steady state
(see the thermal warning in ab_interleaved.ps1) -- do that from the absolute
level against a known baseline, never because a round disagrees with the others.

--metric selects which column carries the measurement. Both metrics are times,
so lower is better and a negative delta is a win. Decode is summarised as
ms/token and not tok/s on purpose: the paired difference of a rate is not the
rate of the paired difference, so a tok/s column would make the interval mean
something other than what it is read as.
"""

import argparse
import csv
import re
import statistics as st
from collections import defaultdict

# two-sided t critical values at 95% by degrees of freedom
_TCRIT = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
          8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160,
          14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093,
          20: 2.086}
DEFAULT_TAG = r"^ab\d*_(?P<arm>.+)_r(?P<round>\d+)$"

# column, unit, and decimals. TTFT is thousands of ms so it reads best as an
# integer; ms/token is tens, where rounding to an integer would quantise away
# most of the effects worth shipping.
METRICS = {
    "ttft": ("ttft_ms", "ms", 0),
    "tps": ("ms_per_token", "ms/tok", 3),
}


def tcrit(df: int) -> float:
    return _TCRIT.get(df, 1.96)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("csv", help="ttft_summary.csv or tps_summary.csv written by the bench scripts")
    ap.add_argument("--baseline", required=True, help="arm every other arm is compared against")
    ap.add_argument("--drop", type=int, nargs="*", default=[], help="round numbers to exclude")
    ap.add_argument("--metric", choices=sorted(METRICS), default="ttft",
                    help="which measurement column to summarise (default: ttft)")
    ap.add_argument("--tag-re", default=DEFAULT_TAG,
                    help="regex with 'arm' and 'round' groups, matched against the tag column")
    args = ap.parse_args()
    tag_re = re.compile(args.tag_re)
    column, unit, dp = METRICS[args.metric]

    rows = list(csv.DictReader(open(args.csv)))
    if rows and column not in rows[0]:
        raise SystemExit(
            f"{args.csv} has no '{column}' column (have: {', '.join(rows[0])}).\n"
            f"--metric {args.metric} expects the CSV written by "
            f"bench_{args.metric}.ps1.")

    runs: dict[int, dict[str, float]] = defaultdict(dict)
    for row in rows:
        m = tag_re.match(row["tag"])
        if m:
            runs[int(m.group("round"))][m.group("arm")] = float(row[column])

    kept = sorted(r for r in runs if r not in args.drop)
    if not kept:
        raise SystemExit("no rounds left after --drop")
    arms = [args.baseline] + sorted({a for r in kept for a in runs[r]} - {args.baseline})

    print(f"metric: {column} ({unit}); lower is better")
    print("round  " + "".join(f"{a:>12}" for a in arms))
    for r in sorted(runs):
        cells = "".join(f"{runs[r].get(a, float('nan')):12.{dp}f}" for a in arms)
        print(f"{r:5d}  {cells}" + ("   <- dropped" if r in args.drop else ""))

    print(f"\n{'arm':>10} {'n':>3} {'mean':>10} {'vs base':>11} {'sd of diff':>12} {'95% CI':>24} {'%':>8}")
    base = [runs[r][args.baseline] for r in kept if args.baseline in runs[r]]
    print(f"{args.baseline:>10} {len(base):3d} {st.mean(base):10.{dp}f} {'--':>11}")
    for arm in arms[1:]:
        pairs = [(runs[r][arm] - runs[r][args.baseline]) for r in kept
                 if arm in runs[r] and args.baseline in runs[r]]
        if len(pairs) < 2:
            print(f"{arm:>10} {len(pairs):3d}  (need >=2 paired rounds)")
            continue
        vals = [runs[r][arm] for r in kept if arm in runs[r]]
        m, sd = st.mean(pairs), st.stdev(pairs)
        half = tcrit(len(pairs) - 1) * sd / len(pairs) ** 0.5
        verdict = "" if (m - half) * (m + half) > 0 else "   (spans zero)"
        ci = f"[{m-half:+.{dp}f}, {m+half:+.{dp}f}]"
        print(f"{arm:>10} {len(pairs):3d} {st.mean(vals):10.{dp}f} {m:+11.{dp}f} {sd:12.{dp}f} "
              f"{ci:>24} {100*m/st.mean(base):+8.2f}{verdict}")


if __name__ == "__main__":
    main()
