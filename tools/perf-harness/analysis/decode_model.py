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

Neither gives absolute time on its own. SQTT inflates kernels by a different
factor per family and stretches idle gaps far more (one gemma-4-26B capture
showed 2.8 ms of host gap per token against 0.73 ms real), so absolute ms,
recoverable ms and tok/s gains are printed only with both:

  --calib J          per-family SQTT factors from calibrate_sqtt.py; each
                     family is divided by max(1, its factor).
  --host-timeline P  the real per-token period and host split from
                     host_timeline.py (JSON, or the HIPDNN_EP_HOST_TIMELINE
                     prefix), in place of SQTT's gaps.

Without them the report is composition only, marked UNVALIDATED. Either way a
gain is an estimate until bench/ab_interleaved.ps1 has measured it.

A VLM decode step spans two queues (embedding session, then decoder). Steps
are cut where the stream switches queues, and the other queue's time is its
own line rather than decoder idle.
"""

from __future__ import annotations

import argparse
import bisect
import collections
import json
import os
import sys

from perfcommon import (
    LAYER_MARKERS,
    ModelSpec,
    add_model_args,
    decode_step_windows,
    load_dispatches,
    op_for_family,
    specs_from_args,
)

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
    "dense_mlp",
    "attn_proj",
    "lm_head",
    "router",
    "kv_cache",
    "norm_other",
]

UNVALIDATED = "UNVALIDATED (SQTT-relative)"
BUDGET_TOL_PCT = 5.0


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
    return (
        dict(out),
        len(sel),
        "chrome trace (EP op names; timing inflated by the profiler)",
    )


def load_calib(path: str) -> dict:
    """Per-op factors from calibrate_sqtt.py, and the family map they used."""
    with open(path) as f:
        d = json.load(f)
    ops = d.get("ops", {})
    raw = {op: v["factor"] for op, v in ops.items() if v.get("factor")}
    return {
        "path": path,
        "raw": raw,
        # SQTT and event time both over-state the uninstrumented kernel time
        # (see calibrate_sqtt.py), so a factor below 1 keeps SQTT.
        "factors": {op: max(1.0, f) for op, f in raw.items()},
        "fmap": {
            op: {
                "exact": tuple(v.get("families", {}).get("exact", ())),
                "prefix": tuple(v.get("families", {}).get("prefix", ())),
            }
            for op, v in ops.items()
        },
        "failed_check": [op for op, v in ops.items() if not v.get("check_ok", True)],
    }


def load_host_timeline(path: str) -> dict[str, float]:
    """Median per-step host timeline in ms: a host_timeline.py --json file, or
    the HIPDNN_EP_HOST_TIMELINE prefix of a run."""
    if path.lower().endswith(".json"):
        with open(path) as f:
            summ = json.load(f)["summary"]
    else:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import host_timeline as ht

        steps = []
        for _pid, ep, sy in ht.resolve_files(path):
            steps += ht.steps_for_pid(ep, sy, 0.05)["steps"]
        if not steps:
            raise SystemExit(f"{path}: no host-timeline decode steps")
        summ = ht.summarize(steps)
    return {k: v["median_us"] / 1000.0 for k, v in summ.items()}


def from_dispatches(
    path: str, spec: ModelSpec, extra: dict | None = None, calib: dict | None = None
) -> tuple[dict[str, float], int, str]:
    """Per-component microseconds for ONE decode step, from an RGP dispatch CSV.

    With two queues (VLM) the step is cut at queue switches and only complete
    steps count; the other queue's kernels go to `other_queue`. With one queue
    (text), a capture window rarely lands on exactly one step -- the fence arms
    at the start of a step and the buffer fills part-way into the next -- so
    totals are normalised by how many steps the window holds, counted by a
    once-per-layer marker family.

    With `calib`, each dispatch of a calibrated op is divided by that op's
    factor; the rest stay SQTT-relative and extra['calibrated_share'] says how
    much of the kernel time that is.
    """
    rows = load_dispatches(path)
    wins, main_q, rejected = decode_step_windows(rows, spec.layers)
    if wins:
        sel = [r for s, e in wins for r in rows[s:e]]
        steps = float(len(wins))
        how = f"{len(wins)} complete step(s) cut at queue switches"
    else:
        counts = collections.Counter(r.get("family", "") for r in rows)
        marker = next((f for f in LAYER_MARKERS if counts.get(f)), None)
        if marker is None:
            raise SystemExit(
                f"{path}: one queue and none of {', '.join(LAYER_MARKERS)}; "
                "cannot tell how many decode steps this window holds"
            )
        sel = rows
        steps = counts[marker] / spec.layers
        how = f"window held {steps:.2f} decode steps by {marker} count"
    dec = [r for r in sel if not wins or r.get("queue") == main_q]
    # Classify per (family, threads) group rather than per row, so the rule can
    # use how often a group runs. Within one step that is the only thing
    # separating the lm_head from a projection: both are one matmul kernel over
    # int4 or fp16 weights, but the lm_head runs once and a projection runs once
    # per layer. See classify_kernel for why the thread count cannot do it.
    per_group: dict[tuple[str, int], int] = collections.Counter()
    for r in dec:
        per_group[(r.get("family", ""), int(r.get("threads") or 0))] += 1
    by_position = positional_components(dec, spec)
    out: dict[str, float] = collections.defaultdict(float)
    raw_us = cal_raw_us = 0.0
    for i, r in enumerate(dec):
        fam = r.get("family", "")
        threads = int(r.get("threads") or 0)
        per_step = per_group[(fam, threads)] / steps
        dur = float(r["dur_us"])
        raw_us += dur
        comp = by_position.get(i) or classify_kernel(fam, threads, spec, per_step)
        if calib:
            op = op_for_family(fam, calib["fmap"])
            if op in calib["factors"]:
                cal_raw_us += dur
                dur /= calib["factors"][op]
            # The kernel name cannot say that a hipBLASLt gemm is attention;
            # the calibration's op map can (calibrate_sqtt.py --map gqa+=gemm).
            if op == "gqa":
                comp = "kv_cache"
        out[comp] += dur
    for r in sel:
        if wins and r.get("queue") != main_q:
            out["other_queue"] += float(r["dur_us"])
    out = {k: v / steps for k, v in out.items()}
    if extra is not None:
        extra.update(
            window_stats(rows, wins, main_q) if wins else dispatch_stats(rows, steps)
        )
        extra["rejected_windows"] = rejected
        if calib:
            extra["calibrated_share"] = cal_raw_us / raw_us if raw_us else 0.0
    return out, len(sel), f"RGP dispatch CSV (SQTT timing; {how})"


def window_stats(rows: list[dict], wins: list[tuple[int, int]], main_q: str) -> dict:
    """Per-step SQTT span, busy (union over queues), other-queue and idle time."""
    span = busy = other = 0.0
    n = 0
    bound: dict[str, float] = collections.defaultdict(float)
    for s, e in wins:
        w = rows[s:e]
        span += rows[e]["t0"] - rows[s]["t0"]
        cur0 = cur1 = None
        for r in w:
            if cur1 is None or r["t0"] > cur1:
                if cur1 is not None:
                    busy += cur1 - cur0
                cur0, cur1 = r["t0"], r["t1"]
            else:
                cur1 = max(cur1, r["t1"])
            if r.get("queue") != main_q:
                other += float(r["dur_us"])
            else:
                bound[r.get("bound_class") or "(unclassified)"] += float(r["dur_us"])
        busy += min(cur1, rows[e]["t0"]) - cur0
        n += len(w)
    k = len(wins)
    tot = sum(bound.values())
    return {
        "steps": float(k),
        "span_ms": span / k / 1000.0,
        "busy_ms": busy / k / 1000.0,
        "gap_ms": (span - busy) / k / 1000.0,
        "other_queue_ms": other / k / 1000.0,
        "artifact_ms": 0.0,
        "n_dispatch": n / k,
        "bound": {b: v / tot for b, v in bound.items()} if tot else {},
    }


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
    gaps = [float(r.get("gap_before_us") or 0) for r in rows if not r.get("artifact")]
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
        return (
            calls
            * spec.topk
            * (
                spec.int4w(spec.hidden * 2 * spec.inter)
                + spec.int4w(spec.inter * spec.hidden)
            )
        )
    if op == "gqa" and "skv" in f:
        # K and V for every past position, fp16.
        return calls * f["skv"] * 2 * spec.kv_heads * spec.head_dim * 2
    return 0.0


def positional_components(dec: list[dict], spec: ModelSpec) -> dict[int, str]:
    """Components of int4 GEMVs that only their position in the layer can name.

    SQTT carries no shapes, and a q-projection, an FFN GEMV and an int4 router
    all run the same matmul_nbits kernels once per layer, so classify_kernel
    files them all as attn_proj. Around them the layer is unambiguous: a gelu
    FFN runs gate GEMV, gelu, up GEMV, multiply, down GEMV, and an int4 router
    GEMV is followed directly by its softmax. Filing the FFN as attention put
    all of gemma-4-12b's FFN time against the q/k/v/o floor. Models without
    these kernels (swiglu MoE) are left to classify_kernel.
    """
    ffn = "dense_mlp" if spec.dense_inter or not spec.experts else None
    gemv = [
        i
        for i, r in enumerate(dec)
        if "matmul_nbits" in r.get("family", "") and "quant_act" not in r["family"]
    ]

    def with_quant(g: int) -> list[int]:
        # The activation-quantise kernel launched for a GEMV goes with it.
        prev = dec[g - 1].get("family", "") if g else ""
        return [g - 1, g] if "matmul_nbits_quant_act" in prev else [g]

    out: dict[int, str] = {}
    for i, r in enumerate(dec):
        fam = r.get("family", "")
        if ffn and "gelu" in fam:
            k = bisect.bisect_left(gemv, i)
            for g in gemv[max(0, k - 1) : k + 2]:
                for j in with_quant(g):
                    out[j] = ffn
        elif "softmax_row" in fam and i and i - 1 in gemv:
            for j in with_quant(i - 1):
                out[j] = "router"
    return out


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
        help="ms/token from bench_tps.ps1, to compare the whole step against "
        "the floor (never used to rescale components)",
    )
    ap.add_argument(
        "--calib",
        help="calibrate_sqtt.py --json output: per-family SQTT factors "
        "(dispatch CSV only)",
    )
    ap.add_argument(
        "--host-timeline",
        help="host_timeline.py --json output, or the HIPDNN_EP_HOST_TIMELINE "
        "prefix of a run at the same prompt length",
    )
    ap.add_argument(
        "--detail",
        action="store_true",
        help="also break each component down by op and shape (trace only)",
    )
    add_model_args(ap)
    args = ap.parse_args()
    spec, dev = specs_from_args(args)
    if args.calib and args.trace:
        raise SystemExit("--calib applies to a dispatch CSV, not a chrome trace")

    calib = load_calib(args.calib) if args.calib else None
    host = load_host_timeline(args.host_timeline) if args.host_timeline else None
    validated = bool(calib and host)
    missing = [n for n, v in (("--calib", calib), ("--host-timeline", host)) if not v]

    detail: dict = {} if args.detail else None
    stats: dict = {}
    if args.trace:
        times, n, provenance = from_trace(args.source, spec, args.skip_runs, detail)
        unit = f"mean of {n} steady-state decode Runs"
    else:
        times, n, provenance = from_dispatches(args.source, spec, stats, calib)
        unit = f"{stats['n_dispatch']:.0f} dispatches per step ({n} captured)"

    byts = spec.decode_bytes(args.kv_len)
    byts["norm_other"] = 0.0  # activations only; not a weight-traffic component

    # Decoder kernels only. The other queue (VLM embedding session) is serial
    # with the decoder and is reported on its own line.
    kernel_ms = sum(times.get(c, 0.0) for c in COMPONENT_ORDER) / 1000.0
    other_ms = times.get("other_queue", 0.0) / 1000.0
    floor_total = sum(byts.values()) / dev.bw_bytes_s * 1000.0
    # The decoder's GPU work fits inside its measured compute, so calibrated
    # kernels well past it mean the calibration did not remove the inflation
    # (gemma-4-12b: 19% over, with event and SQTT agreeing on matmul_nbits).
    # Absolute numbers built on that would carry the error into every gain.
    over_pct = None
    if validated:
        over_pct = 100 * (kernel_ms - host["dec_compute"]) / host["dec_compute"]
        validated = over_pct <= BUDGET_TOL_PCT

    print(f"source     : {args.source}")
    print(f"provenance : {provenance}")
    print(f"window     : {unit}")
    print(f"model      : {args.preset}  kv_len={args.kv_len}")
    print(f"roofline   : {dev.bw_bytes_s / 1e9:.0f} GB/s")
    if calib:
        share = stats.get("calibrated_share", 0.0)
        print(
            f"calibrated : {', '.join(f'{o} /{f:.3f}' for o, f in calib['factors'].items())}"
            f" ({100 * share:.0f}% of SQTT kernel time; the rest stays SQTT-relative)"
        )
        kept = [o for o, f in calib["raw"].items() if f < 1.0]
        if kept:
            print(
                f"             {', '.join(f'{o} {calib["raw"][o]:.3f}' for o in kept)}"
                " < 1: event pairs over-state more than SQTT there, so SQTT is kept"
            )
        if calib["failed_check"]:
            print(
                f"  WARNING: {', '.join(calib['failed_check'])} failed the "
                "dispatches/call check in calibrate_sqtt.py; factors suspect"
            )
    if host:
        print(
            f"host       : period {host['period']:.2f} ms"
            f" ({1000 / host['period']:.2f} tok/s), decoder compute"
            f" {host['dec_compute']:.2f} ms (launch {host['dec_launch']:.2f}"
            f" + sync wait {host['dec_sync']:.2f})"
        )
    if not validated:
        why = (
            f"missing {' and '.join(missing)}"
            if missing
            else f"calibrated kernels {kernel_ms:.2f} ms exceed the decoder compute"
            f" {host['dec_compute']:.2f} ms by {over_pct:.1f}% (tolerance"
            f" {BUDGET_TOL_PCT:g}%): some family is still inflated in both the capture"
            " and the event run, or the runs differ in build, env or prompt"
        )
        print(f"\n*** {UNVALIDATED}: {why}.")
        print(
            "*** Composition only. SQTT inflates kernels non-uniformly and idle"
            " gaps far more, so no absolute ms, recoverable ms or gains are shown."
        )
    print()

    rows = []
    if validated:
        period = host["period"]
        head = (
            f"{'component':<14}{'MB':>9}{'floor_ms':>10}{'ms':>8}{'%floor':>8}"
            f"{'%period':>9}{'recover':>9}{'tok/s +%':>10}"
        )
        print(head)
        print("-" * len(head))
        for comp in COMPONENT_ORDER:
            ms = times.get(comp, 0.0) / 1000.0
            mb = byts.get(comp, 0.0) / 1e6
            fl = byts.get(comp, 0.0) / dev.bw_bytes_s * 1000.0
            pct = 100 * fl / ms if ms > 0 and fl > 0 else 0.0
            # No floor, no recoverable claim: an activation-only component
            # (norm_other) would otherwise read as 100% recoverable.
            rec = max(0.0, ms - fl) if fl > 0 else 0.0
            # A GPU-bound step shrinks by the GPU time saved only until the
            # launch phase binds: at most the sync wait.
            eff = min(rec, host["dec_sync"])
            gain = 100 * (period / (period - eff) - 1) if eff > 0 else 0.0
            rows.append((comp, mb, fl, ms, pct, rec, eff, gain))
            print(
                f"{comp:<14}{mb:>9.1f}{fl:>10.2f}{ms:>8.2f}"
                + (f"{pct:>7.0f}%" if fl > 0 else f"{'-':>8}")
                + f"{100 * ms / period:>8.1f}%"
                + (f"{rec:>9.2f}{gain:>9.1f}%" if fl > 0 else f"{'-':>9}{'-':>10}")
            )
        print("-" * len(head))
        print(
            f"{'kernels':<14}{sum(byts.values()) / 1e6:>9.1f}{floor_total:>10.2f}"
            f"{kernel_ms:>8.2f}{100 * floor_total / kernel_ms:>7.0f}%"
            f"{100 * kernel_ms / period:>8.1f}%"
        )
        resid = host["dec_compute"] - kernel_ms
        print(
            f"\nstep budget (ms): period {period:.2f} = decoder compute"
            f" {host['dec_compute']:.2f} + host outside it"
            f" {period - host['dec_compute']:.2f}"
        )
        print(
            f"  decoder compute {host['dec_compute']:.2f} - calibrated kernels"
            f" {kernel_ms:.2f} = {resid:.2f} GPU idle inside the compute"
            " (plus calibration error)"
        )
        if resid < 0:
            print(
                f"  kernels exceed the decoder compute by {over_pct:.1f}% (within the"
                f" {BUDGET_TOL_PCT:g}% tolerance). Every calibrated number is an upper"
                " bound (event markers; SQTT stretch on uncalibrated families), so"
                " component ms read high by about this much."
            )
        if other_ms:
            print(
                f"  other queue (SQTT-relative, uncalibrated): {other_ms:.2f} ms;"
                f" host timeline says other EP Computes take {host['other_ep']:.2f} ms"
            )
    else:
        head = f"{'component':<14}{'MB':>9}{'floor_ms':>10}{'share':>8}"
        print(head)
        print("-" * len(head))
        for comp in COMPONENT_ORDER:
            ms = times.get(comp, 0.0) / 1000.0
            mb = byts.get(comp, 0.0) / 1e6
            fl = byts.get(comp, 0.0) / dev.bw_bytes_s * 1000.0
            share = 100 * ms / kernel_ms if kernel_ms else 0.0
            print(f"{comp:<14}{mb:>9.1f}{fl:>10.2f}{share:>7.1f}%")
        print("-" * len(head))
        print(
            f"{'TOTAL':<14}{sum(byts.values()) / 1e6:>9.1f}{floor_total:>10.2f}"
            f"{100:>7.1f}%   (share of decoder kernel time, {UNVALIDATED})"
        )
        if other_ms:
            print(
                f"other queue: {100 * other_ms / kernel_ms:.1f}% of decoder kernel time"
            )

    if args.measured_ms:
        print(
            f"floor at {dev.bw_bytes_s / 1e9:.0f} GB/s          : {floor_total:.2f} ms"
            f"  -> {1000 / floor_total:.0f} tok/s ceiling vs"
            f" {1000 / args.measured_ms:.1f} tok/s measured"
            f"  ({100 * floor_total / args.measured_ms:.0f}% of floor)"
        )

    if detail:
        # A component's average hides its spread. attn_proj is one number, but
        # underneath it the q/o projections and the 96 tiny k/v projections run
        # at completely different rates and want different fixes, so each
        # (op, shape) gets its own floor from the bytes its own shape implies.
        # Trace times carry the profiler's syncs, so only the split is shown.
        print(f"\nper-op detail ({UNVALIDATED}: trace timing is profiler-inflated):")
        tot_us = sum(us for _calls, us in detail.values()) or 1.0
        head2 = f"  {'component':<12}{'op':<14}{'shape':<26}{'calls':>6}{'MB':>8}{'share':>8}"
        print(head2)
        print("  " + "-" * (len(head2) - 2))
        for (comp, op, shape), (calls, us) in sorted(
            detail.items(), key=lambda kv: -kv[1][1]
        ):
            if us <= 0:
                continue
            mb = shape_bytes(op, shape, calls, spec) / 1e6
            mbs = f"{mb:>8.1f}" if mb else f"{'-':>8}"
            print(
                f"  {comp:<12}{op:<14}{shape:<26}{calls:>6.0f}{mbs}{100 * us / tot_us:>7.1f}%"
            )

    if stats:
        # SQTT plus SPM counter collection costs real time per dispatch and
        # stretches idle gaps far more than kernels, so none of this is wall
        # time. It is here for the bound classes and to show the inflation.
        span = stats.get("span_ms", stats["busy_ms"] + stats["gap_ms"])
        print(
            f"\nSQTT wall clock (inflated, not used for time): {span:.2f} ms/step ="
            f" {stats['busy_ms']:.2f} busy + {stats['gap_ms']:.2f} idle"
            + (
                f"; other queue {stats['other_queue_ms']:.2f} ms of the busy"
                if stats.get("other_queue_ms")
                else ""
            )
        )
        real = host["period"] if host else args.measured_ms
        if real:
            print(
                f"  vs {real:.2f} ms/token measured without the profiler: x{span / real:.2f}"
            )
        if host:
            print(
                f"  SQTT idle {stats['gap_ms']:.2f} ms vs host timeline: outside the"
                f" decoder {host['period'] - host['dec_compute']:.2f} ms"
            )
        if stats["artifact_ms"] > 0.01:
            print(
                f"  ({stats['artifact_ms']:.2f} ms of fence/capture-setup idle"
                " excluded as measurement, not workload)"
            )
        if stats["bound"]:
            print("\n  SPM bound class, by share of decoder kernel time:")
            for k, v in sorted(stats["bound"].items(), key=lambda kv: -kv[1]):
                print(f"    {k or '(unclassified)':<34}{100 * v:>6.1f}%")

    if validated:
        print("\nranked by recoverable ms/token (estimates -- confirm each with")
        print("bench/ab_interleaved.ps1 -Metric tps before quoting a gain):")
        for comp, _mb, fl, ms, pct, rec, eff, gain in sorted(rows, key=lambda r: -r[5]):
            if rec <= 0.01:
                continue
            cap = f", capped at sync wait {eff:.2f}" if eff < rec else ""
            print(
                f"  {comp:<14}{rec:>7.2f} ms   ({ms:.2f} -> {fl:.2f}, now at"
                f" {pct:.0f}% of floor{cap}) -> up to +{gain:.1f}% tok/s"
            )


if __name__ == "__main__":
    main()
