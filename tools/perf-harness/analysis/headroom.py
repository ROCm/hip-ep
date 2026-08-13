#!/usr/bin/env python3
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Rank optimisation candidates by under-utilisation, not by share of runtime.

A percentage of runtime says where time goes, not where it is wasted. An op can
be half the prefill and already sit at the hardware limit, and another can be
small and entirely unnecessary. So for each component this computes the floor
its own work implies -- max(bytes/BW, FLOP/peak) -- then reports utilisation
against that floor and the seconds between the two.

Two things the output is not:

  - The rows are floors in isolation, so their sum is a bound on a bound, not an
    achievable target. Use the table to order candidates by room, not to predict
    a result.
  - A floor says nothing about how hard the room is to take. The last column is
    a judgement, and it is usually what decides what to work on.
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


def block_floor(spec, dev, m: int) -> tuple[float, float, float]:
    """Seconds, bytes, FLOP for one expert block serving m tokens.

    Weights once, plus the activation traffic the block's own kernels move
    (gather, both GEMM's io, bias/swiglu, scatter). The dequantised-weight round
    trip that the non-fused path pays is deliberately NOT here: it is a
    consequence of the chosen path, not of the work, so it belongs on the
    headroom side of the ledger rather than in the floor.
    """
    h, inter = spec.hidden, spec.inter
    act = (
        spec.fp16(m * h) * 2
        + spec.fp16(m * h)
        + spec.fp16(m * 2 * inter) * 5
        + spec.fp16(m * h) * 6
    )
    byts = spec.expert_weight_bytes + act
    flop = spec.expert_flop_per_token * m
    return dev.floor_s(byts, flop), byts, flop


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_common_args(ap, many=True)
    ap.add_argument(
        "--dense-ms",
        type=float,
        required=True,
        help="measured dense projection ms per chunk (from attrib_regions.py)",
    )
    ap.add_argument(
        "--lm-head-ms",
        type=float,
        default=None,
        help="measured lm_head ms per chunk; omit on models where the "
        "prefill lm_head has already been pruned to the last row",
    )
    ap.add_argument(
        "--attention-s",
        type=float,
        required=True,
        help="measured attention seconds over the whole prefill "
        "(from prefill_model.py, which needs two capture depths)",
    )
    ap.add_argument(
        "--prefill-s",
        type=float,
        required=True,
        help="modelled or measured whole-prefill seconds, for the %% column",
    )
    args = ap.parse_args()
    spec, dev = specs_from_args(args)
    caps = [Capture(p, spec) for p in args.captures]
    n = len(caps)

    print(
        f"ceilings: {dev.peak_flops / 1e12:.1f} TFLOP/s fp16, {dev.bw_bytes_s / 1e9:.0f} GB/s"
    )
    print(
        f"one expert: {spec.expert_weight_bytes / 1e6:.1f} MB of weights "
        f"-> {spec.expert_weight_bytes / dev.bw_bytes_s * 1e6:.1f} us\n"
    )

    print("=== MoE expert blocks: measured vs the floor its own work implies ===")
    print(
        f"{'M':>10} {'blk/chunk':>9} {'meas ms':>8} {'floor ms':>9} {'binding':>10} "
        f"{'util':>6} {'headroom s':>11}"
    )
    m_all = f_all = small_m = small_f = 0.0
    for lo, hi in M_BUCKETS:
        meas = fl = nb = 0.0
        bind: dict[str, int] = defaultdict(int)
        for cap in caps:
            k = cap.layer_scale / n
            sel = [b for b in cap.blocks if lo <= b.m <= hi]
            meas += sum(b.dur_us for b in sel) * k
            nb += len(sel) * k
            for b in sel:
                f, byts, flop = block_floor(spec, dev, b.m)
                fl += f * 1e6 * k
                bind[
                    "compute"
                    if flop / dev.peak_flops > byts / dev.bw_bytes_s
                    else "bandwidth"
                ] += 1
        if not nb:
            continue
        m_all += meas
        f_all += fl
        if hi <= 63:
            small_m += meas
            small_f += fl
        b = max(bind, key=bind.get)
        print(
            f"{bucket_label(lo, hi):>10} {nb:9.0f} {meas / 1000:8.1f} {fl / 1000:9.1f} "
            f"{b:>10} {100 * fl / meas:5.0f}% {(meas - fl) * spec.chunks / 1e6:11.2f}"
        )
    print(
        f"{'all':>10} {'':>9} {m_all / 1000:8.1f} {f_all / 1000:9.1f} {'':>10} "
        f"{100 * f_all / m_all:5.0f}% {(m_all - f_all) * spec.chunks / 1e6:11.2f}"
    )

    # --- the rest of the chunk -------------------------------------------
    print("\n=== the rest, per chunk ===")
    c = spec.chunk_tokens
    dense_b = spec.layers * (
        spec.int4w(spec.hidden * spec.qkv_n)
        + spec.int4w(spec.o_proj_k * spec.hidden)
        + spec.int4w(spec.hidden * spec.router_n)
        + spec.fp16(c * spec.qkv_n)
        + spec.fp16(c * spec.hidden) * 4
    )
    dense_f = (
        spec.layers
        * 2
        * c
        * (
            spec.hidden * spec.qkv_n
            + spec.o_proj_k * spec.hidden
            + spec.hidden * spec.router_n
        )
    )
    lm_b = spec.int4w(spec.vocab * spec.hidden) + spec.fp16(c * spec.vocab)
    lm_f = 2 * c * spec.hidden * spec.vocab

    print(
        f"{'component':42} {'meas ms':>8} {'floor ms':>9} {'binding':>10} {'util':>6} "
        f"{'headroom s':>11}"
    )
    rest = [
        ("dense projections (QKV, o_proj, router)", args.dense_ms, dense_b, dense_f)
    ]
    if args.lm_head_ms is not None:
        rest.append(("lm_head (as executed, all rows)", args.lm_head_ms, lm_b, lm_f))
    for name, meas, byts, flop in rest:
        fb, fc = byts / dev.bw_bytes_s * 1e3, flop / dev.peak_flops * 1e3
        fl = max(fb, fc)
        print(
            f"{name:42} {meas:8.1f} {fl:9.1f} {('compute' if fc > fb else 'bandwidth'):>10} "
            f"{100 * fl / meas:5.0f}% {(meas - fl) * spec.chunks / 1e3:11.2f}"
        )

    # Attention floor: the QK^T and PV matmuls, plus the KV that has to be
    # streamed. The two layer groups cannot share one floor -- a global layer
    # attends to the whole context with kv2 x 512, a sliding layer to a fixed
    # window with kv8 x 256, so they differ in both how much they read and how
    # that grows with context. Summing one group's geometry over all 30 layers
    # is what made this floor wrong before.
    # Softmax/exp is real work this omits, so the utilisation is a LOWER bound
    # and is not comparable to the GEMM rows above.
    nfull = spec.full_attn_layers
    nslide = spec.layers - nfull

    # Global layers: context grows one chunk at a time, so both the FLOPs and
    # the KV read grow linearly with chunk index.
    ctx = [i * c for i in range(1, spec.chunks + 1)]
    full_flop = sum(nfull * 2 * 2 * c * t * spec.heads * spec.full_hd for t in ctx)
    full_kv_b = sum(nfull * t * spec.full_kv * spec.full_hd * 2 * 2 for t in ctx)

    # Sliding layers: each query sees at most `sliding_window` keys, so the cost
    # per chunk is constant once the window is full.
    win = [min(t, spec.sliding_window) for t in ctx]
    slide_flop = sum(nslide * 2 * 2 * c * w * spec.heads * spec.head_dim for w in win)
    slide_kv_b = sum(nslide * w * spec.kv_heads * spec.head_dim * 2 * 2 for w in win)

    att_bytes, att_flop = full_kv_b + slide_kv_b, full_flop + slide_flop
    att_floor_s = dev.floor_s(att_bytes, att_flop)
    att_bind = (
        "compute"
        if att_flop / dev.peak_flops > att_bytes / dev.bw_bytes_s
        else "bandwidth"
    )
    print(
        f"\n   attention floor: {nfull} global (kv{spec.full_kv} x {spec.full_hd}, "
        f"full context) + {nslide} sliding (kv{spec.kv_heads} x {spec.head_dim}, "
        f"window {spec.sliding_window})"
    )
    print(
        f"{'attention over the whole prefill':42} {args.attention_s * 1e3:8.1f} "
        f"{att_floor_s * 1e3:9.1f} {att_bind:>10} {100 * att_floor_s / args.attention_s:5.0f}% "
        f"{args.attention_s - att_floor_s:11.2f}"
    )
    print("   (matmul FLOPs only -- softmax/exp is real work this floor omits, so")
    print("    the utilisation is a lower bound, not comparable to the GEMM rows)")

    print("\n=== recoverable seconds, ranked ===")
    dense_fl_s = max(dense_b / dev.bw_bytes_s, dense_f / dev.peak_flops) * 1e3
    items = [
        (
            "MoE experts, large M (ordinary GEMM efficiency)",
            ((m_all - small_m) - (f_all - small_f)) * spec.chunks / 1e6,
            "hard",
        ),
        (
            "MoE experts, small M (structural: per-expert launch)",
            (small_m - small_f) * spec.chunks / 1e6,
            "new kernel",
        ),
        (
            "dense projections",
            (args.dense_ms - dense_fl_s) * spec.chunks / 1e3,
            "medium",
        ),
        (
            "attention (floor omits softmax; upper bound)",
            args.attention_s - att_floor_s,
            "already worked",
        ),
    ]
    if args.lm_head_ms is not None:
        items.append(
            (
                "lm_head: rows whose logits are never read",
                args.lm_head_ms * spec.chunks / 1e3 - lm_b / dev.bw_bytes_s,
                "delete it",
            )
        )
    print(f"{'':52} {'s':>6} {'% of prefill':>13}  cost to take")
    for name, s, cost in sorted(items, key=lambda x: -x[1]):
        print(f"{name:52} {s:6.2f} {100 * s / args.prefill_s:13.1f}  {cost}")


if __name__ == "__main__":
    main()
