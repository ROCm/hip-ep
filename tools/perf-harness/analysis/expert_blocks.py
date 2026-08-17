#!/usr/bin/env python3
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Per-M-bucket cost of MoE expert blocks, against the bandwidth floor.

This is the report that tells you whether a dispatch-threshold change in
matmul_nbits_kernel.hip did anything. Buckets straddle the routing thresholds,
and the `path` column names the kernels each bucket actually landed on, so a
threshold move shows up as a bucket changing kernel.

Read it with two cautions:

  - Compare us/blk at matched mean M, not the per-chunk totals. Routing is
    input-dependent, so two captures of the same build see different numbers of
    blocks per bucket; only the per-block cost is comparable.
  - Treat the buckets whose kernel did not change as controls. If they drift as
    much as the bucket you changed, you measured noise.
"""

import argparse
from collections import defaultdict

from perfcommon import (
    M_BUCKETS,
    Capture,
    add_common_args,
    bucket_label,
    specs_from_args,
)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_common_args(ap, many=True)
    ap.add_argument(
        "--shapes",
        action="store_true",
        help="also list the routing distribution block by block",
    )
    args = ap.parse_args()
    spec, dev = specs_from_args(args)

    for path in args.captures:
        cap = Capture(path, spec)
        k = cap.layer_scale
        chunk_us = (cap.total_us - cap.lm_head_us) * k + cap.lm_head_us
        floor_us = spec.expert_weight_bytes / dev.bw_bytes_s * 1e6

        print(f"\n######## {path}")
        print(
            f"window {cap.total_us / 1000:.1f} ms, {len(cap.rows)} dispatches, "
            f"{cap.layers_in_window} layer-groups -> x{k:.2f} for {spec.layers} layers"
        )
        print(
            f"reconstructed chunk ({spec.chunk_tokens} tok): {chunk_us / 1000:.1f} ms"
            + (
                f"  (incl. lm_head {cap.lm_head_us / 1000:.1f} ms)"
                if cap.lm_head_us
                else ""
            )
        )
        print(
            f"one expert = {spec.expert_weight_bytes / 1e6:.1f} MB of weights+scales "
            f"-> {floor_us:.1f} us at {dev.bw_bytes_s / 1e9:.0f} GB/s\n"
        )

        print(
            f"{'M range':>10} {'blocks':>7} {'tokens':>7} {'ms/chunk':>9} {'%chunk':>7} "
            f"{'us/blk':>8} {'mean M':>7} {'x floor':>8} {'eff GB/s':>9}  path"
        )
        for lo, hi in M_BUCKETS:
            sel = [b for b in cap.blocks if lo <= b.m <= hi]
            if not sel:
                continue
            us = sum(b.dur_us for b in sel)
            mean = us / len(sel)
            toks = sum(b.m for b in sel)
            paths = defaultdict(int)
            for b in sel:
                paths[
                    "+".join(sorted(x.replace("MatMulNBits", "") for x in b.kernels))
                ] += 1
            pretty = " ".join(
                f"{n}x{c}" for n, c in sorted(paths.items(), key=lambda kv: -kv[1])
            )
            print(
                f"{bucket_label(lo, hi):>10} {len(sel):7d} {toks:7d} {us * k / 1000:9.2f} "
                f"{100 * us * k / chunk_us:7.2f} {mean:8.1f} {toks / len(sel):7.1f} "
                f"{mean / floor_us:8.2f} {spec.expert_weight_bytes / mean / 1e3:9.0f}  {pretty}"
            )

        small = [b for b in cap.blocks if b.m <= 63]
        if small and cap.blocks:
            tok_all = sum(b.m for b in cap.blocks)
            print(
                f"\nM<=63: {len(small)}/{len(cap.blocks)} blocks, "
                f"{100 * sum(b.m for b in small) / tok_all:.1f}% of routed tokens, "
                f"{100 * sum(b.dur_us for b in small) * k / chunk_us:.1f}% of a chunk"
            )

        # Same expert size, two different paths: does skipping the fused dequant
        # actually pay? Only comparable within one bucket.
        big = [b for b in cap.blocks if b.m >= 256]
        fused = [b for b in big if b.dequant_us == 0]
        split = [b for b in big if b.dequant_us > 0]
        if fused and split:
            print(
                "\n=== fused WMMA vs dequant+Fp16GEMM at the same expert size (M>=256) ==="
            )
            for label, sel in (
                ("fused WMMA only", fused),
                ("dequant + Fp16GEMM", split),
            ):
                us = sum(b.dur_us for b in sel)
                tok = sum(b.m for b in sel)
                print(
                    f"  {label:22} n={len(sel):3d}  meanM={tok / len(sel):6.1f}  "
                    f"{us / len(sel):8.1f} us/block  {us / tok:6.2f} us/token  "
                    f"dequant {sum(b.dequant_us for b in sel) / len(sel):6.1f} us/block"
                )

        if args.shapes:
            print("\n=== routing distribution ===")
            hist = defaultdict(int)
            for b in cap.blocks:
                hist[b.m] += 1
            for m in sorted(hist):
                print(f"  M={m:<6d} {hist[m]:4d} blocks")


if __name__ == "__main__":
    main()
