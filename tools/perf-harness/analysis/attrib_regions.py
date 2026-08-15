#!/usr/bin/env python3
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Split kernel time between the MoE expert loop and the dense projections.

The kernel name alone cannot attribute anything here: the same MatMulNBits*
kernels serve the experts and the QKV/o_proj/router projections, so a rollup by
kernel silently merges two very different workloads -- one running at M=1..2000
per expert, the other always at the full chunk. The dispatch stream can
attribute, because the MoE region is bracketed structurally: topk_routing opens
it, and the attention/layernorm kernels that never appear inside the expert loop
close it. No instrumentation involved.

The per-chunk numbers this prints are the `--dense-ms` and `--lm-head-ms` inputs
to headroom.py.
"""

import argparse
from collections import defaultdict

from perfcommon import INT4_FAMILIES, Capture, add_common_args, specs_from_args


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap, many=True)
    args = ap.parse_args()
    spec, _ = specs_from_args(args)

    for path in args.captures:
        cap = Capture(path, spec)
        k = cap.layer_scale
        by_region: dict[str, float] = defaultdict(float)
        scaled_region: dict[str, float] = defaultdict(float)
        int4_us: dict[tuple[str, str], float] = defaultdict(float)
        int4_n: dict[tuple[str, str], int] = defaultdict(int)
        dense_int4_chunk = 0.0

        for i, r in enumerate(cap.rows):
            fam, dur = r["family"], float(r["dur_us"])
            reg = "lm_head" if i in cap.lm_head_idx else cap.regions[i]
            by_region[reg] += dur
            scaled_region[reg] += cap.scaled_us(i)
            if fam in INT4_FAMILIES:
                int4_us[(reg, fam)] += dur
                int4_n[(reg, fam)] += 1
                if reg == "dense":
                    dense_int4_chunk += cap.scaled_us(i)

        total = cap.total_us
        chunk_us = cap.chunk_us

        print(f"\n######## {path}")
        print(f"window {total/1000:.1f} ms over {len(cap.rows)} dispatches, "
              f"{cap.layer_groups} layer-groups -> x{k:.2f}"
              + ("" if cap.layers_in_window else " (dated by attention markers)"))
        if spec.hybrid:
            # Say the census out loud: it is what makes the ms/chunk column
            # differ from a flat multiply, and a window holding none of one type
            # is the case where a reader should distrust that type's row.
            print(f"  hybrid stack: window holds {cap.linear_in_window} of "
                  f"{spec.linear_attn_layers} linear and {cap.full_in_window} of "
                  f"{spec.full_attn_layers} full-attention layers, scaled apart")
            print(f"  (only kernels owned by one layer type are scaled by type; the "
                  f"norms,\n   projections and MoE run in all {spec.layers} layers and "
                  f"take the flat x{k:.0f})")
        print(f"{'region':10} {'window ms':>10} {'%window':>8} {'ms/chunk':>10}")
        for reg in ("qmoe", "dense", "lm_head"):
            us = by_region.get(reg, 0.0)
            if not us:
                continue
            print(f"{reg:10} {us/1000:10.1f} {100*us/total:8.1f} "
                  f"{scaled_region[reg]/1000:10.1f}")
        print(f"{'chunk':10} {'':>10} {'':>8} {chunk_us/1000:10.1f}")

        print(f"\n--- int4 stack by region ---")
        print(f"{'region':8} {'kernel':32} {'count':>6} {'ms':>9} {'%window':>8}")
        for (reg, fam), us in sorted(int4_us.items(), key=lambda kv: -kv[1]):
            print(f"{reg:8} {fam:32} {int4_n[(reg, fam)]:6d} {us/1000:9.2f} {100*us/total:8.2f}")

        q = sum(v for (reg, _), v in int4_us.items() if reg == "qmoe")
        d = sum(v for (reg, _), v in int4_us.items() if reg == "dense")
        lm = sum(v for (reg, _), v in int4_us.items() if reg == "lm_head")
        tot4 = q + d + lm
        if tot4:
            print(f"\nint4 stack = {100*tot4/total:.1f}% of the window "
                  f"(qmoe {100*q/tot4:.1f}%, dense {100*d/tot4:.1f}%, lm_head {100*lm/tot4:.1f}% "
                  f"of the stack)")
        # headroom.py's dense floor is built from the QKV/o_proj/router weights,
        # so it must be fed the int4 projection time -- not the whole dense
        # region, which also carries attention, layernorm and rope.
        print(f"\nfeed headroom.py:  --dense-ms {dense_int4_chunk/1000:.1f} "
              f"--lm-head-ms {cap.lm_head_us/1000:.1f}")
        print(f"  (dense int4 projections only; the full dense region incl. attention "
              f"is {scaled_region.get('dense', 0)/1000:.1f} ms/chunk)")

        if cap.fused_moe:
            # No gather_tokens in the stream: this model's MoE is bucketed
            # grouped GEMMs, so there are no per-expert blocks to report. Roll
            # the region up by kernel instead of printing an empty table.
            print(f"\n--- MoE region: bucketed grouped GEMMs, no per-expert blocks ---")
            fused: dict[str, list] = defaultdict(lambda: [0, 0.0])
            for i, r in enumerate(cap.rows):
                if cap.regions[i] != "qmoe" or i in cap.lm_head_idx:
                    continue
                fused[r["family"]][0] += 1
                fused[r["family"]][1] += cap.scaled_us(i)
            for fam, (n_, us) in sorted(fused.items(), key=lambda kv: -kv[1][1]):
                print(f"  {fam:34} n={n_:4d}  {us/1000:8.2f} ms/chunk")
            continue

        print(f"\n--- expert blocks: {len(cap.blocks)} ---")
        paths: dict[str, list] = defaultdict(lambda: [0, 0.0, 0])
        for b in cap.blocks:
            key = "+".join(sorted(b.kernels)) or "(none)"
            paths[key][0] += 1
            paths[key][1] += b.dur_us
            paths[key][2] += b.m
        for key, (n, us, toks) in sorted(paths.items(), key=lambda kv: -kv[1][1]):
            print(f"  {key:45} n={n:4d}  {us/1000:8.2f} ms  mean {us/n:7.1f} us  "
                  f"mean M {toks/n:7.1f}")


if __name__ == "__main__":
    main()
