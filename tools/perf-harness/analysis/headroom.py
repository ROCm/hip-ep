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

Two modes, because prefill and decode are different problems:

  default    Prefill. Compute can bind, expert blocks serve many tokens each, and
             the unit is a chunk scaled to the whole prompt.
  --decode   Decode. Every GEMM is a GEMV at M=1, so the FLOP term never binds
             and the whole step is a memory-traffic budget. The unit is one
             token, and the only term that moves with context is the KV read --
             which is why this mode takes several (kv_len, ms/token) pairs and
             shows how the ranking changes between them. A candidate that is
             worth taking at 128 tokens can be irrelevant at 16K.

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
    add_model_args,
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


def decode_main(argv) -> None:
    """Cross-context-length decode ranking.

    Each --at pairs a KV length with the measured ms/token there, and one
    source (a capture or a trace) supplies the component split. The weight terms
    are identical at every length; only the KV read grows, so the interesting
    output is which rows change position between the columns.
    """
    ap = argparse.ArgumentParser(
        prog="headroom.py --decode",
        description=decode_main.__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--decode", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument(
        "--at",
        action="append",
        required=True,
        metavar="KVLEN:MS[:SOURCE]",
        help="e.g. 144:14.59:dec128_dispatches.csv -- repeat per context length",
    )
    ap.add_argument(
        "--trace", action="store_true", help="sources are chrome traces, not CSVs"
    )
    ap.add_argument("--skip-runs", type=int, default=8)
    add_model_args(ap)
    args = ap.parse_args(argv)
    spec, dev = specs_from_args(args)

    import decode_model as dm

    cols = []
    for spec_str in args.at:
        parts = spec_str.split(":")
        if len(parts) < 2:
            raise SystemExit(f"--at wants KVLEN:MS[:SOURCE], got {spec_str!r}")
        kv, ms = int(parts[0]), float(parts[1])
        src = ":".join(parts[2:]) if len(parts) > 2 else None
        if src:
            if args.trace:
                times, _n, _p = dm.from_trace(src, spec, args.skip_runs)
            else:
                times, _n, _p = dm.from_dispatches(src, spec)
            total = sum(times.values()) / 1000.0
            # The measurement sets the magnitude; the source only sets the split
            # (a trace's own total is inflated by the profiler, and a capture
            # covers kernels only, not the gaps between them).
            times = {k: v / 1000.0 * ms / total for k, v in times.items()}
        else:
            times = None
        cols.append((kv, ms, times))

    print(f"roofline {dev.bw_bytes_s / 1e9:.0f} GB/s   model {args.preset}")
    print("weight traffic is context-independent; only kv_cache grows.\n")

    comps = dm.COMPONENT_ORDER
    header = f"{'component':<14}" + "".join(
        f"{'kv=' + str(kv):>22}" for kv, _ms, _t in cols
    )
    print(header)
    print(f"{'':14}" + "".join(f"{'meas  floor  recov':>22}" for _ in cols))
    print("-" * len(header))
    totals = []
    for comp in comps:
        row = f"{comp:<14}"
        for kv, _ms, times in cols:
            byts = spec.decode_bytes(kv)
            byts["norm_other"] = 0.0
            fl = byts.get(comp, 0.0) / dev.bw_bytes_s * 1000.0
            if times is None:
                row += f"{'-':>22}"
                continue
            meas = times.get(comp, 0.0)
            row += f"{meas:>8.2f}{fl:>7.2f}{max(0.0, meas - fl):>7.2f}"
        print(row)
    print("-" * len(header))
    row = f"{'TOTAL':<14}"
    for kv, ms, times in cols:
        fl = spec.decode_bytes_total(kv) / dev.bw_bytes_s * 1000.0
        totals.append((kv, ms, fl))
        row += f"{ms:>8.2f}{fl:>7.2f}{max(0.0, ms - fl):>7.2f}"
    print(row)

    print(f"\n{'kv_len':>8}{'tok/s now':>11}{'tok/s floor':>13}{'% of floor':>12}")
    for kv, ms, fl in totals:
        print(f"{kv:>8}{1000 / ms:>11.1f}{1000 / fl:>13.1f}{100 * fl / ms:>11.0f}%")

    print("\nranked by recoverable ms/token (at the longest context measured):")
    kv, ms, times = cols[-1]
    if times:
        byts = spec.decode_bytes(kv)
        byts["norm_other"] = 0.0
        rank = []
        for comp in comps:
            fl = byts.get(comp, 0.0) / dev.bw_bytes_s * 1000.0
            meas = times.get(comp, 0.0)
            rank.append((comp, meas, fl, max(0.0, meas - fl)))
        for comp, meas, fl, rec in sorted(rank, key=lambda r: -r[3]):
            if rec <= 0.01:
                continue
            pct = 100 * fl / meas if meas else 0
            print(
                f"  {comp:<14}{rec:>7.2f} ms   ({meas:.2f} -> {fl:.2f}, {pct:.0f}% of floor)"
            )


def main() -> None:
    import sys

    if "--decode" in sys.argv[1:]:
        return decode_main(sys.argv[1:])
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
    # A dense model has no expert loop, so the whole MoE half of this report is
    # not merely empty but undefined: spec.expert_weight_bytes is computed from
    # hidden/inter and would print a confident MB figure for a thing that does
    # not exist, and the `all` row divides by a measured time of zero.
    moe = any(cap.blocks for cap in caps)
    m_all = f_all = small_m = small_f = 0.0
    if moe:
        print(
            f"one expert: {spec.expert_weight_bytes / 1e6:.1f} MB of weights "
            f"-> {spec.expert_weight_bytes / dev.bw_bytes_s * 1e6:.1f} us\n"
        )
        print("=== MoE expert blocks: measured vs the floor its own work implies ===")
        print(
            f"{'M':>10} {'blk/chunk':>9} {'meas ms':>8} {'floor ms':>9} {'binding':>10} "
            f"{'util':>6} {'headroom s':>11} {'meas GB/s':>10} {'real?':>6}"
        )
    elif not spec.experts:
        print(f"dense model ({args.preset}, experts=0): MoE section skipped\n")
    else:
        # An MoE preset with no blocks is a different situation from a dense
        # model and must not be labelled as one: it means the window holds no
        # gather_tokens-delimited expert block, which on an MoE model usually
        # means the capture is a decode step or is positioned outside the
        # expert loop -- i.e. the capture is wrong for this report, not the model.
        print(
            f"{args.preset} is MoE but no expert block appears in this window "
            "-- wrong capture position, or a decode capture; MoE section skipped\n"
        )
    # Buckets whose floor is a bandwidth floor but whose measured bandwidth is at
    # or above the roofline. Their headroom is not recoverable time: the traffic
    # the floor charges DRAM for is being served from cache. Collected so the
    # ranking at the bottom can mark them rather than silently rank them first.
    # Keyed by the bucket's (lo, hi) bounds, not by its printed label: the small/
    # large split below is `hi <= 63`, the same test the totals use, and matching
    # on label text instead silently inverted which row got marked.
    bw_not_binding: dict[tuple[int, int], float] = {}

    for lo, hi in M_BUCKETS if moe else []:
        meas = fl = nb = meas_bytes = 0.0
        bind: dict[str, int] = defaultdict(int)
        for cap in caps:
            k = cap.layer_scale / n
            sel = [b for b in cap.blocks if lo <= b.m <= hi]
            meas += sum(b.dur_us for b in sel) * k
            meas_bytes += sum(b.mem_bytes for b in sel) * k
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
        # The floor says which resource *should* bind; SPM says what the hardware
        # did. Where the floor is bandwidth and the measured rate already exceeds
        # the roofline, the two disagree and the floor is the one that is wrong.
        meas_gbps = (meas_bytes / (meas * 1e-6) / 1e9) if meas else 0.0
        roof_gbps = dev.bw_bytes_s / 1e9
        if meas_gbps <= 0:
            real = "n/a"
        elif b == "bandwidth" and meas_gbps >= roof_gbps:
            real = "NO"
            bw_not_binding[(lo, hi)] = meas_gbps
        else:
            real = "yes"
        print(
            f"{bucket_label(lo, hi):>10} {nb:9.0f} {meas / 1000:8.1f} {fl / 1000:9.1f} "
            f"{b:>10} {100 * fl / meas:5.0f}% {(meas - fl) * spec.chunks / 1e6:11.2f} "
            f"{meas_gbps:10.0f} {real:>6}"
        )
    if m_all:
        print(
            f"{'all':>10} {'':>9} {m_all / 1000:8.1f} {f_all / 1000:9.1f} {'':>10} "
            f"{100 * f_all / m_all:5.0f}% {(m_all - f_all) * spec.chunks / 1e6:11.2f}"
        )

    # --- the rest of the chunk -------------------------------------------
    print("\n=== the rest, per chunk ===")
    c = spec.chunk_tokens
    # Any FFN that is NOT inside the expert loop has to be in this floor, because
    # its kernels are in the dense region and so are already inside --dense-ms.
    # On an MoE model that is the shared MLP running alongside the experts
    # (dense_inter). On a dense model it is the entire FFN -- gate, up and down
    # for every layer -- which is the bulk of the model. Leaving it out compares
    # a measured time that includes the FFN against a floor that does not, and
    # reports a utilisation several times lower than the truth.
    ffn_inter = spec.dense_inter or (spec.inter if not spec.experts else 0)
    dense_b = spec.layers * (
        spec.int4w(spec.hidden * spec.qkv_n)
        + spec.int4w(spec.o_proj_k * spec.hidden)
        + spec.int4w(spec.hidden * spec.router_n)
        + spec.fp16(c * spec.qkv_n)
        + spec.fp16(c * spec.hidden) * 4
        + spec.int4w(spec.hidden * 2 * ffn_inter)
        + spec.int4w(ffn_inter * spec.hidden)
        + spec.fp16(c * spec.hidden) * 2
        + spec.fp16(c * ffn_inter) * 6
    )
    dense_f = (
        spec.layers
        * 2
        * c
        * (
            spec.hidden * spec.qkv_n
            + spec.o_proj_k * spec.hidden
            + spec.hidden * spec.router_n
            + spec.hidden * 2 * ffn_inter
            + ffn_inter * spec.hidden
        )
    )
    lm_b = spec.int4w(spec.vocab * spec.hidden) + spec.fp16(c * spec.vocab)
    lm_f = 2 * c * spec.hidden * spec.vocab

    print(
        f"{'component':42} {'meas ms':>8} {'floor ms':>9} {'binding':>10} {'util':>6} "
        f"{'headroom s':>11}"
    )
    dense_label = "dense projections (QKV, o_proj, router" + (
        f", FFN x{ffn_inter})" if ffn_inter else ")"
    )
    rest = [(dense_label, args.dense_ms, dense_b, dense_f)]
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
    # A bucket whose bandwidth floor does not bind contributes volume, not time.
    # Ranking it as recoverable is how this report once put a 59% candidate at the
    # top that measured -4% when it was built, so say so in the row itself.
    small_unbound = [b for b in bw_not_binding if b[1] <= 63]
    large_unbound = [b for b in bw_not_binding if b[1] > 63]
    items = (
        [
            (
                "MoE experts, large M (ordinary GEMM efficiency)"
                + (" [BW FLOOR DOES NOT BIND]" if large_unbound else ""),
                ((m_all - small_m) - (f_all - small_f)) * spec.chunks / 1e6,
                "hard" if not large_unbound else "not recoverable: see above",
            ),
            (
                "MoE experts, small M (structural: per-expert launch)"
                + (" [BW FLOOR DOES NOT BIND]" if small_unbound else ""),
                (small_m - small_f) * spec.chunks / 1e6,
                "new kernel" if not small_unbound else "not recoverable: see above",
            ),
        ]
        if moe
        else []
    )
    items += [
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
    print(f"{'':74} {'s':>6} {'% of prefill':>13}  cost to take")
    for name, s, cost in sorted(items, key=lambda x: -x[1]):
        print(f"{name:74} {s:6.2f} {100 * s / args.prefill_s:13.1f}  {cost}")

    if bw_not_binding:
        print()
        print("!! Rows marked [BW FLOOR DOES NOT BIND] are not candidates.")
        for (lo, hi), gbps in sorted(bw_not_binding.items()):
            print(
                f"   M {bucket_label(lo, hi):>7}: measured {gbps:.0f} GB/s = "
                f"{100 * gbps / (dev.bw_bytes_s / 1e9):.0f}% of the "
                f"{dev.bw_bytes_s / 1e9:.0f} GB/s roofline"
            )
        print(
            "   Above the roofline the traffic is cache-served, so the floor is\n"
            "   charging DRAM time for bytes that never reach DRAM. Measured once:\n"
            "   a 59% 'candidate' whose kernels ran at 113-219% of roofline, and\n"
            "   which regressed TTFT by 4% once the traffic was actually removed.\n"
            "   Use the FLOP floor, or a measured control at a shape with nothing\n"
            "   to re-read, to size these instead."
        )


if __name__ == "__main__":
    main()
