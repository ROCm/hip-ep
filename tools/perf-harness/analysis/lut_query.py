#!/usr/bin/env python3
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Query the matmul_nbits LUT for the config a given (n, k) shape resolves to.

The LUT is 2355 points over a 95-entry config table, so answering "what does
o-projection actually get dispatched with" by eye is not practical. Restricting
edits to shapes no other model reaches (the item-3 risk) needs the reverse
query too, hence --list-shapes.
"""

import argparse
import collections
import json


def cfg_str(configs, i):
    c = configs[i]
    if c["kind"] == "Gemv":
        return f'Gemv({c["threads"]},{c["tile_n"]})'
    return (f'{c["kind"]}(bm16={c["bm16"]},bn16={c["bn16"]},bk={c["bk"]},'
            f'wt={c["wt_m"]}x{c["wt_n"]},swz={c["swizzle"]},fused={c["fused"]})')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("lut")
    ap.add_argument("--phase", default="DecodeDp4a")
    ap.add_argument("--shape", action="append", default=[],
                    metavar="N,K", help="repeatable")
    ap.add_argument("--list-shapes", action="store_true",
                    help="every (n,k) present for the phase")
    ap.add_argument("--histogram", action="store_true")
    args = ap.parse_args()

    d = json.load(open(args.lut))
    C = d["configs"]
    pts = [p for p in d["points"] if p["phase"] == args.phase]
    print(f'{args.phase}: {len(pts)} points, {len(C)} configs in table')

    fb = [f for f in d["fallbacks"] if f["phase"] == args.phase]
    for f in fb:
        print(f'  fallback {f["bits"]}: config {f["config"]} '
              f'{cfg_str(C, f["config"])}')

    for s in args.shape:
        n, k = (int(x) for x in s.split(","))
        hits = [p for p in pts if p["n"] == n and p["k"] == k]
        print(f'--- n={n} k={k}: {len(hits)} entries')
        for p in hits:
            print(f'      {p["bits"]} {p["group_size"]} {p["zero_point"]} '
                  f'stride={p["row_stride"]} -> config {p["config"]} '
                  f'{cfg_str(C, p["config"])}')

    if args.list_shapes:
        shapes = collections.Counter((p["n"], p["k"]) for p in pts)
        print(f'--- {len(shapes)} distinct (n,k)')
        for (n, k), cnt in sorted(shapes.items()):
            cfgs = {cfg_str(C, p["config"])
                    for p in pts if p["n"] == n and p["k"] == k}
            print(f'      n={n:<6} k={k:<6} x{cnt:<3} {sorted(cfgs)}')

    if args.histogram:
        h = collections.Counter(cfg_str(C, p["config"]) for p in pts)
        print("--- config histogram")
        for kk, v in h.most_common():
            print(f'      {kk:<24} {v}')


if __name__ == "__main__":
    main()
