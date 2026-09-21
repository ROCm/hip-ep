#!/usr/bin/env python3
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Group one capture's dispatches by op label and kernel template, with the
achieved bandwidth and the bound classification the parser assigned.

capture_diff.py answers "did this change help"; this answers "which shape is
slow and why", which is what you need before deciding what to change. The
per-dispatch bandwidth column is the one that matters: two shapes moving the
same bytes through the same kernel should reach the same GB/s, and when they do
not, the gap is the headroom.
"""

import argparse
import collections
import csv


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("csv", help="<tag>_dispatches.csv from rgp_capture.ps1")
    ap.add_argument("--kernel", default="", help="substring filter on the kernel name")
    ap.add_argument("--top", type=int, default=30)
    args = ap.parse_args()

    with open(args.csv, newline="") as f:
        rows = list(csv.DictReader(f))

    agg = collections.defaultdict(
        lambda: {
            "n": 0,
            "us": 0.0,
            "gbps": 0.0,
            "occ": 0.0,
            "wg": set(),
            "bound": collections.Counter(),
        }
    )
    for r in rows:
        if args.kernel and args.kernel not in r["kernel"]:
            continue
        tmpl = r["kernel"].split("<", 1)[1].rstrip(">") if "<" in r["kernel"] else ""
        key = (r["kernel"].split("<", 1)[0], r["label"], r["tile"], tmpl)
        a = agg[key]
        a["n"] += 1
        a["us"] += float(r["dur_us"] or 0)
        a["gbps"] += float(r["mem_gbps"] or 0)
        a["occ"] += float(r["occ_pct"] or 0)
        a["wg"].add(r["workgroup"])
        a["bound"][r["bound_class"]] += 1

    hdr = (
        f"{'kernel':30} {'label':22} {'cfg':9} {'n':>4} "
        f"{'us tot':>9} {'us ea':>7} {'GB/s':>7} {'occ%':>6}  bound"
    )
    print(hdr)
    print("-" * len(hdr))
    for k, v in sorted(agg.items(), key=lambda x: -x[1]["us"])[: args.top]:
        n = v["n"]
        bound = v["bound"].most_common(1)[0][0] if v["bound"] else ""
        print(
            f"{k[0][:30]:30} {k[1][:22]:22} {k[3][:9]:9} {n:4d} "
            f"{v['us']:9.1f} {v['us'] / n:7.2f} {v['gbps'] / n:7.1f} "
            f"{v['occ'] / n:6.1f}  {bound[:24]}"
        )


if __name__ == "__main__":
    main()
