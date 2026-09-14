#!/usr/bin/env python3
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Decode-step composition and per-component distance from the memory floor.

The prefill sibling (prefill_model.py) scales a captured chunk up to a whole
prompt. Decode needs no scaling -- one step *is* the unit -- so the question
changes from "how much of the prompt does this chunk represent" to "how much of
this step is recoverable". That makes the floor model the whole content here.

At M=1 every GEMM is a GEMV: one pass over the weights for a single row of
activations. Arithmetic intensity is ~2 FLOP/byte against a machine that wants
~232 to break even, so the compute term never binds and each component's floor
is just its bytes over the memory roofline. A component at 90% of that floor is
finished; one at 20% is where the time is.

Two inputs are supported, and which one is in use is always printed, because
they do not measure the same thing:

  dispatch CSV  from tools/rgp_parser. SQTT timing, minimally perturbing, but
                kernel-level -- components are recovered by kernel name.
  chrome trace  from HIPDNN_EP_TRACE_FILE. Exact op attribution (the EP names
                the ops itself) but the profiler's per-inference stream sync
                inflates the total. Structure yes, throughput no.
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import sys

from perfcommon import ModelSpec, add_model_args, specs_from_args

# EP op name -> component. The op names come from the build under test; run
# trace_ops.py on a trace to list them for a model this does not cover.
OP_COMPONENT = {
    "qmoe": "moe_experts",
    "matmul_nbits": "attn_proj",
    "gqa": "kv_cache",
    "layernorm": "norm_other",
    "skip_layernorm": "norm_other",
    "gather": "norm_other",
    "cast": "norm_other",
    "sub": "norm_other",
    "reduce_sum": "norm_other",
}

# `matmul` covers both the fp16 router and the fp16 lm_head, which differ by
# three orders of magnitude in bytes and want opposite fixes, so they are split
# on the N dimension rather than lumped together.
COMPONENT_ORDER = [
    "moe_experts",
    "attn_proj",
    "lm_head",
    "router",
    "kv_cache",
    "norm_other",
]


def classify_matmul(shape: str, spec: ModelSpec) -> str:
    """Split the fp16 `matmul` op into lm_head and router by its N dimension."""
    n = None
    for part in shape.split(","):
        part = part.strip()
        if part.startswith("n="):
            n = int(part[2:])
    if n is None:
        return "norm_other"
    return "lm_head" if n >= spec.vocab // 2 else "router"


def from_trace(
    path: str, spec: ModelSpec, skip: int, detail: dict | None = None
) -> tuple[dict[str, float], int, str]:
    """Mean per-component microseconds over the steady-state decode Runs."""
    sys.path.insert(0, __file__.rsplit("\\", 1)[0].rsplit("/", 1)[0])
    from trace_ops import load, split_runs

    runs = split_runs(load(path))
    # Prefill and decode run the same ops here, so the token dimension is what
    # separates them: gqa reports sq, and sq==1 is a decode step.
    decode = []
    for r in runs:
        gqa = next((e for e in r if e["name"] == "gqa"), None)
        sh = (gqa.get("args") or {}).get("shape", "") if gqa else ""
        if "sq=1," in sh:
            decode.append(r)
    if not decode:
        raise SystemExit(f"{path}: no decode Runs (no gqa with sq=1)")
    sel = decode[skip:] or decode[-1:]
    out: dict[str, float] = collections.defaultdict(float)
    for r in sel:
        for e in r:
            name = e["name"]
            shape = (e.get("args") or {}).get("shape", "")
            comp = (
                classify_matmul(shape, spec)
                if name == "matmul"
                else OP_COMPONENT.get(name, "norm_other")
            )
            dur = e.get("dur", 0.0)
            out[comp] += dur
            if detail is not None:
                d = detail.setdefault((comp, name, shape), [0, 0.0])
                d[0] += 1
                d[1] += dur
    for k in out:
        out[k] /= len(sel)
    if detail is not None:
        for k in detail:
            detail[k][0] /= len(sel)
            detail[k][1] /= len(sel)
    return dict(out), len(sel), "chrome trace (EP op names; timing inflated by the profiler)"


def from_dispatches(
    path: str, spec: ModelSpec, extra: dict | None = None
) -> tuple[dict[str, float], int, str]:
    """Per-component microseconds for ONE decode step, from an RGP dispatch CSV.

    A capture window rarely lands on exactly one step -- the fence arms at the
    start of a step and the buffer fills part-way into the next -- so totals are
    normalised by how many steps the window actually holds. topk_routing runs
    once per layer and nowhere else, so its count over `layers` measures that
    directly, in fractions.
    """
    rows = list(csv.DictReader(open(path)))
    routing = sum(1 for r in rows if r.get("family", "") == "topk_routing")
    if not routing:
        raise SystemExit(
            f"{path}: no topk_routing dispatches; cannot tell how many decode "
            "steps this window holds"
        )
    steps = routing / spec.layers
    # Classify per (family, threads) group rather than per row, so the rule can
    # use how often a group runs. Within one step that is the only thing
    # separating the lm_head from a projection: both are one matmul kernel over
    # int4 or fp16 weights, but the lm_head runs once and a projection runs once
    # per layer. See classify_kernel for why the thread count cannot do it.
    per_group: dict[tuple[str, int], int] = collections.Counter()
    for r in rows:
        per_group[(r.get("family", ""), int(r.get("threads") or 0))] += 1
    out: dict[str, float] = collections.defaultdict(float)
    for r in rows:
        fam = r.get("family", "")
        threads = int(r.get("threads") or 0)
        per_step = per_group[(fam, threads)] / steps
        out[classify_kernel(fam, threads, spec, per_step)] += float(r["dur_us"])
    out = {k: v / steps for k, v in out.items()}
    if extra is not None:
        extra.update(dispatch_stats(rows, steps))
    return (
        out,
        len(rows),
        f"RGP dispatch CSV (SQTT timing; window held {steps:.2f} decode steps)",
    )


# The fence drains the GPU and idles before arming, and RGP's own setup runs in
# that window, so the first gap of a capture is milliseconds of measurement
# rather than workload. Anything this large is not a launch gap.
_ARTIFACT_GAP_US = 500.0


def dispatch_stats(rows: list[dict], steps: float) -> dict:
    """Per-step busy/gap split and the SPM bound-class mix.

    The gap column is what makes a decode capture worth taking: ~400 expert
    blocks per token means the time between kernels is a real line item, not
    rounding.
    """
    busy = sum(float(r["dur_us"]) for r in rows)
    gaps = [float(r.get("gap_before_us") or 0) for r in rows]
    art = sum(g for g in gaps if g >= _ARTIFACT_GAP_US)
    gap = sum(g for g in gaps if g < _ARTIFACT_GAP_US)
    bound: dict[str, float] = collections.defaultdict(float)
    for r in rows:
        bound[r.get("bound_class") or "(unclassified)"] += float(r["dur_us"])
    return {
        "steps": steps,
        "busy_ms": busy / steps / 1000.0,
        "gap_ms": gap / steps / 1000.0,
        "artifact_ms": art / steps / 1000.0,
        "n_dispatch": len(rows) / steps,
        "bound": {k: v / busy for k, v in bound.items()},
    }


def _shape_fields(shape: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for part in shape.replace("x", ",").split(","):
        part = part.strip()
        if "=" in part:
            k, v = part.split("=", 1)
            try:
                out[k] = int(v)
            except ValueError:
                pass
    return out


def shape_bytes(op: str, shape: str, calls: float, spec: ModelSpec) -> float:
    """Weight bytes one op's dispatches must read, from the shape it reports.

    Only the weight-dominated ops get a floor. The norm/elementwise ops move
    activations whose size the shape string does not pin down, and they are
    0.5% of the step, so giving them a fabricated floor would add noise without
    adding information -- they return 0 and print as '-'.
    """
    f = _shape_fields(shape)
    if op == "matmul_nbits" and "n" in f and "k" in f:
        return calls * spec.int4w(f["n"] * f["k"])
    if op == "matmul" and "n" in f and "k" in f:
        return calls * spec.fp16(f["n"] * f["k"])
    if op == "qmoe":
        # "1x2048x768,e=128": hidden x inter, topk experts served per call.
        return calls * spec.topk * (
            spec.int4w(spec.hidden * 2 * spec.inter)
            + spec.int4w(spec.inter * spec.hidden)
        )
    if op == "gqa" and "skv" in f:
        # K and V for every past position, fp16.
        return calls * f["skv"] * 2 * spec.kv_heads * spec.head_dim * 2
    return 0.0


def classify_kernel(
    family: str, threads: int, spec: ModelSpec, per_step: float = 1e9
) -> str:
    """Map one dispatch to a component, given how often its group runs per step.

    Among the matmul families, `per_step` is the discriminator: the lm_head runs
    ONCE per decode step and every projection and router runs once per layer, so
    a group appearing far fewer than `layers` times is the lm_head and the rest
    are per-layer work. The remaining split is by weight dtype -- the quantised
    path (`matmul_nbits_*`) is the projections, the fp16 Tensile path (`gemm`)
    is whatever the export left unquantised.

    Thread count is deliberately NOT the discriminator, in either direction:

      - Tensile's count reflects the tile it chose, so the same router GEMM
        reports 256 threads at one context length and 2048 at another.
      - The hand-written GEMV launches many threads per output element, so a
        q-projection reports 262144 against a vocabulary of 151936.

    Both mistakes were made and both silently refiled work into the wrong
    component while leaving the totals intact.
    """
    f = family.lower()
    # Checked before the MoE families: top-k selection belongs with the router
    # that produced the logits, not with the experts it then dispatches to.
    if "topk" in f:
        return "router"
    if "moe" in f or "expert" in f or "swiglu" in f:
        return "moe_experts"
    if "gqa" in f or "flash" in f or "kv_cache" in f or "rope" in f:
        return "kv_cache"
    if "matmul" in f or "gemv" in f or "gemm" in f or "dequant" in f:
        if per_step < spec.layers / 2:
            return "lm_head"
        if "nbits" in f or "quant" in f:
            return "attn_proj"
        return "router"
    return "norm_other"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("source", help="*_dispatches.csv, or a chrome trace with --trace")
    ap.add_argument("--trace", action="store_true", help="source is a chrome trace")
    ap.add_argument(
        "--skip-runs",
        type=int,
        default=8,
        help="trace only: decode Runs to drop before averaging (autotune settling)",
    )
    ap.add_argument(
        "--kv-len",
        type=int,
        required=True,
        help="KV length the step ran at; the only seq-dependent term",
    )
    ap.add_argument(
        "--measured-ms",
        type=float,
        default=None,
        help="ms/token from bench_tps.ps1, to compare against the floor",
    )
    ap.add_argument(
        "--detail",
        action="store_true",
        help="also break each component down by op and shape (trace only)",
    )
    add_model_args(ap)
    args = ap.parse_args()
    spec, dev = specs_from_args(args)

    detail: dict = {} if args.detail else None
    stats: dict = {}
    if args.trace:
        times, n, provenance = from_trace(args.source, spec, args.skip_runs, detail)
        unit = f"mean of {n} steady-state decode Runs"
    else:
        times, n, provenance = from_dispatches(args.source, spec, stats)
        unit = f"{stats['n_dispatch']:.0f} dispatches per step ({n} captured)"

    byts = spec.decode_bytes(args.kv_len)
    byts["norm_other"] = 0.0  # activations only; not a weight-traffic component

    measured_total = sum(times.values()) / 1000.0
    floor_total = sum(byts.values()) / dev.bw_bytes_s * 1000.0

    # Both sources over-report, for different reasons: the trace carries the
    # profiler's per-inference stream sync, and a capture pays SQTT plus SPM
    # counter collection on every dispatch. Either way the kernel times sum to
    # more than the step really takes, and comparing that sum against the
    # measured ms/token would invent negative overhead. Both are trustworthy
    # for how the step divides up and neither for how long it is, so rescale
    # the shares onto the measurement and say so. Without --measured-ms there
    # is nothing to rescale onto and the absolute ms stay inflated.
    rescaled = False
    factor = 1.0
    if args.measured_ms and measured_total > 0:
        factor = args.measured_ms / measured_total
        raw_total = measured_total
        times = {k: v * factor for k, v in times.items()}
        measured_total = args.measured_ms
        rescaled = True

    print(f"source     : {args.source}")
    print(f"provenance : {provenance}")
    print(f"window     : {unit}")
    print(f"model      : {args.preset}  kv_len={args.kv_len}")
    print(f"roofline   : {dev.bw_bytes_s/1e9:.0f} GB/s")
    if rescaled:
        src = "trace" if args.trace else "capture"
        print(
            f"rescaled   : {src} shares x{factor:.3f} onto the measured "
            f"{args.measured_ms:.2f} ms/token ({src} total was {raw_total:.2f} ms)"
        )
    print()

    head = f"{'component':<14}{'MB':>9}{'floor_ms':>10}{'meas_ms':>9}{'%floor':>8}{'share':>8}{'recover':>9}"
    print(head)
    print("-" * len(head))
    rows = []
    for comp in COMPONENT_ORDER:
        ms = times.get(comp, 0.0) / 1000.0
        mb = byts.get(comp, 0.0) / 1e6
        fl = byts.get(comp, 0.0) / dev.bw_bytes_s * 1000.0
        pct = 100 * fl / ms if ms > 0 else 0.0
        share = 100 * ms / measured_total if measured_total else 0.0
        recover = max(0.0, ms - fl)
        rows.append((comp, mb, fl, ms, pct, share, recover))
        print(
            f"{comp:<14}{mb:>9.1f}{fl:>10.2f}{ms:>9.2f}"
            f"{pct:>7.0f}%{share:>7.1f}%{recover:>9.2f}"
        )
    print("-" * len(head))
    print(
        f"{'TOTAL':<14}{sum(byts.values())/1e6:>9.1f}{floor_total:>10.2f}"
        f"{measured_total:>9.2f}{100*floor_total/measured_total:>7.0f}%"
        f"{100:>7.1f}%{max(0.0, measured_total-floor_total):>9.2f}"
    )

    if args.measured_ms:
        print(
            f"floor at {dev.bw_bytes_s/1e9:.0f} GB/s          : {floor_total:.2f} ms"
            f"  -> {1000/floor_total:.0f} tok/s ceiling vs"
            f" {1000/args.measured_ms:.1f} tok/s measured"
            f"  ({100*floor_total/args.measured_ms:.0f}% of floor)"
        )

    if detail:
        # A component's average hides its spread. attn_proj is one number, but
        # underneath it the q/o projections and the 96 tiny k/v projections run
        # at completely different rates and want different fixes, so each
        # (op, shape) gets its own floor from the bytes its own shape implies.
        print("\nper-op detail (each row against its own floor):")
        head2 = f"  {'component':<12}{'op':<14}{'shape':<26}{'calls':>6}{'MB':>8}{'ms':>8}{'GB/s':>8}{'%floor':>8}"
        print(head2)
        print("  " + "-" * (len(head2) - 2))
        for (comp, op, shape), (calls, us) in sorted(
            detail.items(), key=lambda kv: -kv[1][1]
        ):
            ms = us / 1000.0 * (factor if rescaled else 1.0)
            mb = shape_bytes(op, shape, calls, spec) / 1e6
            if ms <= 0:
                continue
            gbs = mb / 1e3 / (ms / 1e3) if ms else 0
            fl = mb * 1e6 / dev.bw_bytes_s * 1000.0
            pct = 100 * fl / ms if ms else 0
            if mb == 0:
                print(f"  {comp:<12}{op:<14}{shape:<26}{calls:>6.0f}{'-':>8}{ms:>8.2f}{'-':>8}{'-':>8}")
            else:
                print(
                    f"  {comp:<12}{op:<14}{shape:<26}{calls:>6.0f}{mb:>8.1f}"
                    f"{ms:>8.2f}{gbs:>8.0f}{pct:>7.0f}%"
                )

    if stats:
        # SQTT plus SPM counter collection costs real time per dispatch, so a
        # capture's wall clock runs well above the step it is measuring. It is
        # authoritative for composition and for the bound classes -- which is
        # what it is here for -- and not for absolute ms.
        print(
            f"\ncapture wall clock: {stats['busy_ms']:.2f} ms kernels"
            f" + {stats['gap_ms']:.2f} ms between them"
            f" = {stats['busy_ms'] + stats['gap_ms']:.2f} ms/step"
        )
        if args.measured_ms:
            infl = (stats["busy_ms"] + stats["gap_ms"]) / args.measured_ms
            print(
                f"  vs {args.measured_ms:.2f} ms/token measured without the"
                f" profiler: x{infl:.2f} instrumentation cost"
            )
        print(
            f"  inter-kernel gap is {100*stats['gap_ms']/(stats['busy_ms']+stats['gap_ms']):.0f}%"
            f" of the captured step over ~{stats['n_dispatch']:.0f} dispatches"
            f" ({1000*stats['gap_ms']/stats['n_dispatch']:.1f} us each)"
        )
        if stats["artifact_ms"] > 0.01:
            print(
                f"  ({stats['artifact_ms']:.2f} ms of fence/capture-setup idle"
                " excluded as measurement, not workload)"
            )
        print("\n  SPM bound class, by share of kernel time:")
        for k, v in sorted(stats["bound"].items(), key=lambda kv: -kv[1]):
            print(f"    {k or '(unclassified)':<34}{100*v:>6.1f}%")

    print("\nranked by recoverable ms/token:")
    for comp, _mb, fl, ms, pct, _share, rec in sorted(rows, key=lambda r: -r[6]):
        if rec <= 0.01:
            continue
        print(f"  {comp:<14}{rec:>7.2f} ms   ({ms:.2f} -> {fl:.2f}, now at {pct:.0f}% of floor)")


if __name__ == "__main__":
    main()
