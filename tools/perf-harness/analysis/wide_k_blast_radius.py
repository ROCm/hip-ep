#!/usr/bin/env python3
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Replay the HIPDNN_EP_MATMUL_NBITS_WIDE_K promotion rule over every dp4a point
in the offline LUT and report which shapes it moves.

The rule lives in matmul_nbits_kernel.hip (wideKPromoteDp4aConfig); this is a
transcription of it, kept here because the interesting question -- "how many
shapes other than o-projection does this touch, and by how much" -- is a
property of the table, and answering it on hardware costs a build plus a run per
shape. Keep the two in sync; the assertions at the bottom are the tripwire.

The overhead model: matmul_nbits_gemv_dp4a_kernel issues 8 sudot4 per 32-element
chunk for a_sum_q plus 8 per column, so per column it costs 8 + 8/TILE_N units
against 8 of useful work. Predicted speedup from a promotion is the ratio of
those two costs -- an upper bound, since it assumes the kernel is purely
issue-bound.
"""

import argparse
import collections
import json

MIN_BLOCKS = 256
MIN_K = 2048


def cost_per_col(tile_n):
    """sudot4 issued per output column, per 32-element K chunk."""
    return 8.0 + 8.0 / tile_n


def promote(threads, tile_n, n, k):
    """Return (threads, tile_n) after promotion, or None if left alone."""
    if tile_n > 2 or k < MIN_K or k <= n:
        return None
    target = 0
    for tn in (8, 4):
        if n // tn >= MIN_BLOCKS and tn > tile_n:
            target = tn
            break
    if target == 0:
        return None
    chunks = k // 32
    t = 32
    while t < 256 and t * 2 <= chunks:
        t *= 2
    return (t, target)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("lut")
    ap.add_argument("--min-gain", type=float, default=1.0)
    args = ap.parse_args()

    d = json.load(open(args.lut))
    C = d["configs"]
    pts = [p for p in d["points"] if p["phase"] == "DecodeDp4a"]

    # The table can only express (threads, tile_n) pairs it already contains.
    have = {(c["threads"], c["tile_n"]) for c in C if c["kind"] == "Gemv"}

    moved, kept, unrepresentable = [], 0, []
    for p in pts:
        c = C[p["config"]]
        if c["kind"] != "Gemv":
            continue
        new = promote(c["threads"], c["tile_n"], p["n"], p["k"])
        if new is None:
            kept += 1
            continue
        if new not in have:
            unrepresentable.append((p["n"], p["k"], new))
            continue
        gain = cost_per_col(c["tile_n"]) / cost_per_col(new[1])
        moved.append((gain, p["n"], p["k"], p["group_size"], p["zero_point"],
                      (c["threads"], c["tile_n"]), new))

    print(f'dp4a points: {len(pts)}   promoted: {len(moved)}   unchanged: {kept}')
    if unrepresentable:
        print(f'  !! {len(unrepresentable)} promotions have no table entry: '
              f'{unrepresentable[:5]}')

    print(f'--- promoted shapes (predicted issue-rate ceiling >= {args.min_gain}x)')
    print(f'  {"n":>7} {"k":>7}  {"gs":<5} {"zp":<11} {"from":<10} {"to":<10} {"max x":>6}')
    for gain, n, k, gs, zp, old, new in sorted(moved, reverse=True):
        if gain < args.min_gain:
            continue
        print(f'  {n:7d} {k:7d}  {gs:<5} {zp:<11} '
              f'{f"({old[0]},{old[1]})":<10} {f"({new[0]},{new[1]})":<10} {gain:6.2f}')

    h = collections.Counter(g for g, *_ in moved)
    print("--- predicted ceiling histogram")
    for g, cnt in sorted(h.items(), reverse=True):
        print(f'      {g:.2f}x  {cnt} shapes')

    # Tripwires on the two shapes the item is actually about.
    assert promote(32, 1, 2048, 4096) == (128, 8), "o-projection must promote"
    assert promote(256, 4, 4096, 2048) is None, "q-projection must be left alone"
    assert promote(64, 2, 512, 2048) is None, "k/v must be left alone"
    print("--- Qwen3-30B attention shapes behave as intended")


if __name__ == "__main__":
    main()
