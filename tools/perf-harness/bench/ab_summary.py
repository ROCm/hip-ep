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


def tcrit(df: int) -> float:
    return _TCRIT.get(df, 1.96)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("csv", help="ttft_summary.csv written by bench_ttft.ps1")
    ap.add_argument("--baseline", required=True, help="arm every other arm is compared against")
    ap.add_argument("--drop", type=int, nargs="*", default=[], help="round numbers to exclude")
    ap.add_argument("--tag-re", default=DEFAULT_TAG,
                    help="regex with 'arm' and 'round' groups, matched against the tag column")
    args = ap.parse_args()
    tag_re = re.compile(args.tag_re)

    runs: dict[int, dict[str, float]] = defaultdict(dict)
    for row in csv.DictReader(open(args.csv)):
        m = tag_re.match(row["tag"])
        if m:
            runs[int(m.group("round"))][m.group("arm")] = float(row["ttft_ms"])

    kept = sorted(r for r in runs if r not in args.drop)
    if not kept:
        raise SystemExit("no rounds left after --drop")
    arms = [args.baseline] + sorted({a for r in kept for a in runs[r]} - {args.baseline})

    print("round  " + "".join(f"{a:>10}" for a in arms))
    for r in sorted(runs):
        cells = "".join(f"{runs[r].get(a, float('nan')):10.0f}" for a in arms)
        print(f"{r:5d}  {cells}" + ("   <- dropped" if r in args.drop else ""))

    print(f"\n{'arm':>10} {'n':>3} {'mean':>8} {'vs base':>9} {'sd of diff':>11} {'95% CI':>18} {'%':>8}")
    base = [runs[r][args.baseline] for r in kept if args.baseline in runs[r]]
    print(f"{args.baseline:>10} {len(base):3d} {st.mean(base):8.0f} {'--':>9}")
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
        print(f"{arm:>10} {len(pairs):3d} {st.mean(vals):8.0f} {m:+9.0f} {sd:11.0f} "
              f"{f'[{m-half:+.0f}, {m+half:+.0f}]':>18} {100*m/st.mean(base):+8.2f}{verdict}")


if __name__ == "__main__":
    main()
