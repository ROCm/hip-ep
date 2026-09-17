#!/usr/bin/env python3
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Whole-prefill composition from two fence-positioned captures.

A capture holds one chunk, and a chunked prefill is many chunks, so a single
capture only generalises for work whose cost does not depend on how much context
already exists. That covers everything except full attention, which grows with
KV depth. Hence two captures: one shallow (early chunk) and one deep (late
chunk). The growth exponent is fitted from the pair rather than assumed to be
linear, and the sliding-window layers are measured -- not asserted -- to be
depth-independent, which is the check that the segmentation is right.

Expect the modelled total to land a few percent UNDER a real TTFT. RGP pegs
clocks to peak for the capture; production runs at whatever the power budget
allows. That gap is the reason a promising capture still has to be confirmed by
an A/B on TTFT before anyone believes it.
"""

import argparse
import math
from collections import defaultdict

from perfcommon import INT4_FAMILIES, Capture, add_common_args, specs_from_args

ATTN_FAMILY = "gqa_flash_prefill_v5"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap, many=True)
    ap.add_argument("--at-chunk", type=float, nargs="+", required=True,
                    help="chunk index each capture sits at, same order as the captures "
                         "(e.g. --at-chunk 2.5 31.5)")
    ap.add_argument("--measured-ttft-s", type=float,
                    help="clean TTFT to compare the model against")
    ap.add_argument("--attn-family", default=ATTN_FAMILY)
    args = ap.parse_args()
    spec, _ = specs_from_args(args)
    if len(args.captures) != 2 or len(args.at_chunk) != 2:
        raise SystemExit("need exactly two captures and two --at-chunk positions "
                         "(one shallow, one deep)")

    order = sorted(range(2), key=lambda i: args.at_chunk[i])
    names = ("shallow", "deep")
    caps, depth, per_chunk, attn = {}, {}, {}, {}

    for name, i in zip(names, order):
        cap = Capture(args.captures[i], spec)
        k = cap.layer_scale
        caps[name] = cap
        depth[name] = args.at_chunk[i]
        agg: dict[str, float] = defaultdict(float)
        for j, r in enumerate(cap.rows):
            fam, dur = r["family"], float(r["dur_us"])
            if j in cap.lm_head_idx:
                agg["lm_head"] += dur          # once per chunk, never scaled
            else:
                agg[f"{cap.regions[j]}:{fam}"] += dur * k
        per_chunk[name] = agg
        # The full-attention and sliding-window layers interleave; splitting the
        # per-layer durations at the median separates them without needing to
        # know the pattern.
        g = sorted((float(r["dur_us"]) for r in cap.rows
                    if r["family"] == args.attn_family), reverse=True)
        if len(g) < 2:
            raise SystemExit(f"{args.captures[i]}: too few '{args.attn_family}' dispatches")
        half = len(g) // 2
        attn[name] = (sum(g[:half]) / half, sum(g[half:]) / (len(g) - half),
                      k * len(g) / 2)

    print("=== attention: measured at two depths ===")
    for n in names:
        f, s, npl = attn[n]
        print(f"  {n:8} chunk~{depth[n]:4.1f}  full {f:9.1f} us/layer   "
              f"sliding {s:7.1f} us/layer   {npl:.1f} layers of each per chunk")
    f0, s0, npl = attn["shallow"]
    f1, s1, _ = attn["deep"]
    p = math.log(f1 / f0) / math.log(depth["deep"] / depth["shallow"])
    print(f"  full attention scales as KV^{p:.2f} "
          f"({f1/f0:.1f}x over {depth['deep']/depth['shallow']:.1f}x KV)")
    print(f"  sliding window is depth-independent: {s0:.1f} -> {s1:.1f} us "
          f"({100*(s1-s0)/s0:+.1f}%)  <- if this is large the segmentation is wrong")

    sl_ms = s0 * npl / 1000
    full_ms = [f0 * (c / depth["shallow"]) ** p * npl / 1000 for c in range(1, spec.chunks + 1)]
    attn_total = sum(full_ms) + sl_ms * spec.chunks

    base: dict[str, float] = defaultdict(float)
    for key in set(per_chunk["shallow"]) | set(per_chunk["deep"]):
        if args.attn_family in key:
            continue
        base[key] = (per_chunk["shallow"].get(key, 0.0)
                     + per_chunk["deep"].get(key, 0.0)) / 2 / 1000
    base_chunk = sum(base.values())
    total = base_chunk * spec.chunks + attn_total

    print(f"\n=== modelled prefill ({spec.chunks} chunks of {spec.chunk_tokens}) ===")
    print(f"  depth-independent  {base_chunk:7.1f} ms/chunk x {spec.chunks} = "
          f"{base_chunk*spec.chunks/1000:5.2f} s")
    print(f"  attention                                  = {attn_total/1000:5.2f} s")
    print(f"  modelled total                             = {total/1000:5.2f} s")
    if args.measured_ttft_s:
        d = 100 * (total / 1000 - args.measured_ttft_s) / args.measured_ttft_s
        print(f"  vs measured TTFT {args.measured_ttft_s:.2f} s -> {d:+.0f}% "
              f"(a few % under is expected: RGP pegs clocks)")

    share = dict(base)
    share[f"attention ({args.attn_family})"] = attn_total / spec.chunks
    print(f"\n=== whole-prefill ranking ===")
    print(f"{'component':46} {'s over prefill':>15} {'%':>7}")
    for key, v in sorted(share.items(), key=lambda kv: -kv[1])[:16]:
        print(f"{key:46} {v*spec.chunks/1000:15.2f} {100*v*spec.chunks/total:7.2f}")

    fam: dict[str, float] = defaultdict(float)
    for key, v in share.items():
        short = key.split(":")[-1]
        if key == "lm_head":
            fam["lm_head (logits for every row)"] += v
        elif key.startswith("qmoe:") and short in INT4_FAMILIES:
            fam["int4 matmul: qmoe expert GEMMs"] += v
        elif key.startswith("dense:") and short in INT4_FAMILIES:
            fam["int4 matmul: dense projections"] += v
        elif "attention" in key:
            fam["attention"] += v
        else:
            fam["everything else"] += v
    print(f"\n{'family':46} {'s over prefill':>15} {'%':>7}")
    for key, v in sorted(fam.items(), key=lambda kv: -kv[1]):
        print(f"{key:46} {v*spec.chunks/1000:15.2f} {100*v*spec.chunks/total:7.2f}")
    print("\nfeed headroom.py:  "
          f"--attention-s {attn_total/1000:.2f} --prefill-s {total/1000:.2f}")


if __name__ == "__main__":
    main()
