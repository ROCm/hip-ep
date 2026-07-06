#!/usr/bin/env python3
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""
bottleneck_report.py -- turn a bottleneck_probe.ps1 run directory into ONE
ranked, roofline-attributed bottleneck report (markdown + csv + json).

Consumes: <probe_dir>/manifest.json plus the raw logs it references.
Produces: <probe_dir>/bottleneck_report.{md,csv,json}

Phase 1 note: per-op byte counts (hence achieved GB/s and % of roofline) are
ESTIMATED from a shape-driven byte model here in the script. Phase 2 replaces
these with values measured inside the EP (bytes reported directly in [PERF]).
Sections that are estimates are labelled "(est)".

Pure standard library; runs anywhere with Python >= 3.10.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Byte model
# ---------------------------------------------------------------------------
# int4/int8 GEMV weight quantization: bits per weight keyed on (n, k) for the
# 35B model. Weights dominate; scales/act/out are secondary. Unknown shapes
# default to 4 bits (int4). Confirmed from the decode IR (section 3 of results).
BITS_BY_NK = {
    (4096, 2048): 8, (32, 2048): 8, (8192, 2048): 8, (2048, 4096): 8,
    (256, 2048): 4, (1, 2048): 4, (512, 2048): 4, (2048, 512): 4,
    (248320, 2048): 4,
}


def matmul_nbits_bytes(m, n, k, bits, block, dt_bytes=2):
    """Total bytes moved by an int-N GEMV: weights + scales + act + out."""
    weights = n * k * bits / 8.0
    scales = n * ((k + block - 1) // block) * dt_bytes
    act = m * k * dt_bytes
    out = m * n * dt_bytes
    return weights + scales + act + out


def bits_for(n, k):
    return BITS_BY_NK.get((n, k), 4)


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------
_MB_PROMPT = re.compile(r"Prompt processing.*?avg \(us\):\s*([\d.]+).*?avg \(tokens/s\):\s*([\d.]+)", re.S)
_MB_GEN = re.compile(r"Token generation.*?avg \(us\):\s*([\d.]+).*?avg \(tokens/s\):\s*([\d.]+)", re.S)
_MB_E2E = re.compile(r"E2E generation.*?avg \(ms\):\s*([\d.]+)", re.S)
_MB_WS = re.compile(r"Peak working set size \(bytes\):\s*(\d+)")


def parse_model_benchmark(text):
    out = {}
    if (mm := _MB_PROMPT.search(text)):
        out["ttft_s"] = float(mm.group(1)) / 1e6  # prompt-processing avg is microseconds
        out["prefill_tps"] = float(mm.group(2))
    if (mm := _MB_GEN.search(text)):
        out["decode_ms"] = float(mm.group(1)) / 1000.0
        out["decode_tps"] = float(mm.group(2))
    if (mm := _MB_E2E.search(text)):
        out["e2e_ms"] = float(mm.group(1))
    if (mm := _MB_WS.search(text)):
        out["peak_ws_gb"] = int(mm.group(1)) / (1024 ** 3)
    return out


# Row now optionally carries EP-measured MB / GB/s / %pk columns after gpu%.
_PERF_ROW = re.compile(
    r"^\[PERF\](\s+)(\S.*?)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)%"
    r"(?:\s+(\S+)\s+(\S+)\s+(\S+))?\s*$")
_PERF_TOTAL = re.compile(r"^\[PERF\]\s+TOTAL\s+([\d.]+)\s+([\d.]+)")
_PERF_WALL = re.compile(r"^\[PERF\]\s+#\d+\s+wall=([\d.]+).*?gpu=([\d.]+)(?:.*?launch_gap=([\d.]+))?")
_COLDSTART = re.compile(r"^\[COLDSTART\]\s+(\S+?)=([0-9.eE+-]+)")


def parse_perf_tables(text):
    """Return list of tables; each = {ops:{name:{calls,gpu,cpu,pct,shapes:{}}},
    total_gpu, total_cpu, wall}. Ops accumulate until a TOTAL line closes one."""
    tables = []
    cur = None
    last_op = None
    pending_wall = None
    pending_gap = None
    for line in text.splitlines():
        if (mw := _PERF_WALL.match(line)):
            pending_wall = float(mw.group(1))
            pending_gap = float(mw.group(3)) if mw.group(3) else None
            continue
        if (mt := _PERF_TOTAL.match(line)):
            if cur is not None:
                cur["total_gpu"] = float(mt.group(1))
                cur["total_cpu"] = float(mt.group(2))
                cur["wall"] = pending_wall
                cur["launch_gap"] = pending_gap
                tables.append(cur)
                cur = None
                last_op = None
                pending_wall = None
                pending_gap = None
            continue
        if (mr := _PERF_ROW.match(line)):
            indent, name, calls, gpu, cpu, pct, mb, gbps, pk = mr.groups()
            rec = {"calls": int(calls), "gpu": float(gpu), "cpu": float(cpu),
                   "pct": float(pct)}
            # EP-measured roofline columns (present only on byte-instrumented ops).
            if mb and mb != "-":
                rec["mb_meas"] = float(mb)
                rec["gbps_meas"] = float(gbps) if gbps not in (None, "-") else None
                rec["pk_meas"] = float(pk.rstrip("%")) if pk and pk != "-" else None
            name = name.strip()
            if name in ("", "calls"):
                continue
            if cur is None:
                cur = {"ops": {}}
            # op rows carry <=2 leading spaces; shape sub-rows are indented more.
            if len(indent) <= 2:
                rec["shapes"] = {}
                cur["ops"][name] = rec
                last_op = name
            elif last_op is not None:
                cur["ops"][last_op]["shapes"][name] = rec
    return tables


def parse_singleop_gpu(text):
    """Isolated per-shape kernel time: last matmul_nbits (or qmoe) shape row."""
    tables = parse_perf_tables(text)
    if not tables:
        return None
    ops = tables[-1]["ops"]
    for opname in ("matmul_nbits", "qmoe"):
        if opname in ops:
            op = ops[opname]
            if op["shapes"]:
                label, rec = list(op["shapes"].items())[-1]
                return {"op": opname, "shape": label, "gpu": rec["gpu"], "calls": rec["calls"]}
            return {"op": opname, "shape": "", "gpu": op["gpu"], "calls": op["calls"]}
    return None


_SHAPE_NK = re.compile(r"m=(\d+),n=(\d+),k=(\d+)")


def _pick_prefill_table(tables):
    """The prefill Compute is the one whose matmul_nbits shapes have m>1
    (prompt tokens); decode Computes are all m=1. Fall back to the first table."""
    for t in tables:
        mm = t["ops"].get("matmul_nbits")
        if mm:
            for label in mm.get("shapes", {}):
                sm = _SHAPE_NK.search(label)
                if sm and int(sm.group(1)) > 1:
                    return t
    return tables[0] if tables else None


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------
def load(probe_dir: Path):
    manifest = json.loads((probe_dir / "manifest.json").read_text())
    runs = manifest["runs"]

    def logtext(name):
        p = probe_dir / name
        return p.read_text(errors="replace") if p.exists() else ""

    def find(pred):
        return [r for r in runs if pred(r)]

    return manifest, runs, logtext, find


def build_report(probe_dir: Path):
    manifest, runs, logtext, find = load(probe_dir)
    peak = float(manifest.get("peak_gbps", 256.0))
    L = []  # markdown lines
    csv_rows = []
    j = {"model": manifest["model_name"], "peak_gbps": peak, "sections": {}}

    L.append(f"# Bottleneck report -- {manifest['model_name']}")
    L.append("")
    L.append(f"Probe dir: `{probe_dir.name}`  |  roofline peak: {peak:.0f} GB/s  |  "
             f"gen={manifest['gen']} reps={manifest['reps']} warmup={manifest['warmup']}")
    L.append("")
    L.append("> Phase 1 report: per-op GB/s and % peak are shape-model ESTIMATES (marked est). "
             "Phase 2 replaces them with EP-measured bytes.")
    L.append("")

    # --- A. Headline ---
    L.append("## A. Headline (model_benchmark)")
    L.append("")
    L.append("| prompt | TTFT (s) | prefill tok/s | decode tok/s | decode ms/tok | peak WS (GB) | kernel errors |")
    L.append("|---|---|---|---|---|---|---|")
    headline = {}
    for r in sorted(find(lambda r: r.get("type") == "headline"), key=lambda r: r["prompt"]):
        m = parse_model_benchmark(logtext(r["log"]))
        headline[r["prompt"]] = m
        L.append("| {p} | {ttft:.3f} | {pf:.0f} | {dc:.1f} | {dms:.2f} | {ws:.1f} | {ke} |".format(
            p=r["prompt"], ttft=m.get("ttft_s", float("nan")), pf=m.get("prefill_tps", float("nan")),
            dc=m.get("decode_tps", float("nan")), dms=m.get("decode_ms", float("nan")),
            ws=m.get("peak_ws_gb", float("nan")), ke=r.get("kernel_errors")))
        csv_rows.append(["headline", r["prompt"], "", m.get("prefill_tps"), m.get("decode_tps"),
                         m.get("ttft_s"), m.get("peak_ws_gb"), "", "", ""])
    j["sections"]["headline"] = {str(k): v for k, v in headline.items()}
    L.append("")

    # --- B/C. Decode per-op + per-shape roofline ---
    dp = find(lambda r: r.get("type") == "decode_profile")
    decode_rows = []
    if dp:
        tables = parse_perf_tables(logtext(dp[0]["log"]))
        if tables:
            t = tables[-1]  # steady-state decode Compute
            total_gpu = t.get("total_gpu") or sum(o["gpu"] for o in t["ops"].values())
            L.append(f"## B. Decode per-op roofline (steady token; total GPU {total_gpu:.1f} ms/Compute)")
            L.append("")
            L.append("| op | calls | gpu ms | % decode | bytes (MB, est) | GB/s (est) | % peak (est) |")
            L.append("|---|---|---|---|---|---|---|")
            ops_sorted = sorted(t["ops"].items(), key=lambda kv: kv[1]["gpu"], reverse=True)
            total_mb = 0.0
            for name, op in ops_sorted:
                mb, gbps, pctpeak = _op_bytes(name, op, block=32, peak=peak)
                total_mb += mb or 0.0
                mbs = f"{mb:.1f}" if mb else "-"
                gs = f"{gbps:.0f}" if gbps else "-"
                pp = f"{pctpeak:.0f}%" if pctpeak else "-"
                L.append(f"| {name} | {op['calls']} | {op['gpu']:.1f} | {op['pct']:.1f}% | {mbs} | {gs} | {pp} |")
                csv_rows.append(["decode_op", "", name, op["calls"], op["gpu"], op["pct"], mb, gbps, pctpeak, ""])
                decode_rows.append({"op": name, "calls": op["calls"], "gpu": op["gpu"],
                                    "pct": op["pct"], "mb": mb, "gbps": gbps, "pct_peak": pctpeak})
            L.append("")
            # Top-down aggregate memory roofline -- PROFILER-INDEPENDENT: byte
            # counts are correct even on degraded runs; multiply by the CLEAN
            # decode tok/s to get achieved aggregate bandwidth vs peak. This is
            # the roofline check that survives the profiling-path instability.
            dtps = (headline.get(128) or headline.get(512) or {}).get("decode_tps")
            if total_mb > 0 and dtps:
                ach = (total_mb / 1000.0) * dtps  # GB/s = (GB/token) * tok/s
                cov = 100.0 * sum(1 for d in decode_rows if d["mb"]) / max(1, len(decode_rows))
                L.append(f"- Top-down aggregate memory roofline (profiler-independent): "
                         f"~{total_mb/1000.0:.2f} GB/token (byte-model, ~{cov:.0f}% of ops covered) "
                         f"x {dtps:.1f} tok/s = ~{ach:.0f} GB/s achieved = ~{100.0*ach/peak:.0f}% of {peak:.0f} GB/s peak.")
                L.append(f"  Memory-roofline decode ceiling ~= {peak/(total_mb/1000.0):.0f} tok/s "
                         f"(peak/bytes-per-token); measured {dtps:.1f} tok/s.")
                L.append("")
                j["sections"]["aggregate_roofline"] = {
                    "gb_per_token": total_mb / 1000.0, "achieved_gbps": ach,
                    "pct_peak": 100.0 * ach / peak, "ceiling_tps": peak / (total_mb / 1000.0),
                    "ops_covered_pct": cov}
            # C. matmul_nbits per-shape
            if "matmul_nbits" in t["ops"] and t["ops"]["matmul_nbits"]["shapes"]:
                L.append("## C. matmul_nbits per-shape roofline (est)")
                L.append("")
                L.append("| shape | calls | gpu ms | % decode | bits | bytes (MB) | GB/s | % peak | class |")
                L.append("|---|---|---|---|---|---|---|---|---|")
                shapes = t["ops"]["matmul_nbits"]["shapes"]
                for label, rec in sorted(shapes.items(), key=lambda kv: kv[1]["gpu"], reverse=True):
                    sm = _SHAPE_NK.search(label)
                    row = f"| {label} | {rec['calls']} | {rec['gpu']:.1f} | {rec['pct']:.1f}% |"
                    mbv = gbps = pk = None
                    bits = ""
                    if rec.get("mb_meas"):
                        mbv = rec["mb_meas"]; gbps = rec.get("gbps_meas") or 0.0
                        pk = rec.get("pk_meas") or 0.0
                        if sm:
                            bits = bits_for(int(sm.group(2)), int(sm.group(3)))
                    elif sm:
                        mI, nI, kI = int(sm.group(1)), int(sm.group(2)), int(sm.group(3))
                        bits = bits_for(nI, kI)
                        tot = matmul_nbits_bytes(mI, nI, kI, bits, 32) * rec["calls"]
                        mbv = tot / 1e6
                        gbps = (tot / 1e9) / (rec["gpu"] / 1000.0) if rec["gpu"] > 0 else 0
                        pk = 100.0 * gbps / peak
                    if mbv is not None:
                        cls = "overhead-bound" if pk < 12 else ("headroom" if pk < 45 else "near-roofline")
                        row += f" {bits} | {mbv:.1f} | {gbps:.0f} | {pk:.0f}% | {cls} |"
                        csv_rows.append(["matmul_shape", "", label, rec["calls"], rec["gpu"], rec["pct"],
                                         mbv, gbps, pk, cls])
                    else:
                        row += " - | - | - | - | - |"
                    L.append(row)
                L.append("")
            # D. qmoe
            if "qmoe" in t["ops"]:
                q = t["ops"]["qmoe"]
                L.append("## D. qmoe")
                L.append("")
                L.append(f"- decode: {q['calls']} calls, {q['gpu']:.1f} ms/Compute, {q['pct']:.1f}% of decode GPU time.")
                L.append("- Sparsity realized (only k=8 active experts read); see prefill for the bigger qmoe cost.")
                L.append("")
            j["sections"]["decode"] = {"total_gpu_ms": total_gpu, "ops": decode_rows}
    else:
        L.append("## B. Decode per-op roofline")
        L.append("")
        L.append("_No decode profile log found._")
        L.append("")

    # --- E. Launch critical-path A/B ---
    ln = find(lambda r: r.get("type") == "launch_normal")
    lb = find(lambda r: r.get("type") == "launch_blocking")
    L.append("## E. Launch critical-path A/B (p128 decode)")
    L.append("")
    # EP-measured launch_gap (wall - gpu) distribution across decode Computes.
    # This is the definitive GPU-bound vs launch-bound signal (median gap
    # fraction): high => launch/dispatch on the critical path (fuse), low =>
    # GPU-compute-bound (tune kernels).
    if dp:
        _t = parse_perf_tables(logtext(dp[0]["log"]))
        gaps = [(x["wall"], x["launch_gap"]) for x in _t
                if x.get("launch_gap") is not None and x.get("wall")]
        # decode-token gaps: drop the first (prefill) sample if present.
        if len(gaps) > 1:
            gaps = gaps[1:]
        if gaps:
            fracs = sorted(100.0 * g / w for w, g in gaps if w > 0)
            med = fracs[len(fracs) // 2]
            L.append(f"- EP launch_gap (wall-gpu) over {len(gaps)} decode Computes: "
                     f"median {med:.0f}% of wall is host launch/dispatch not hidden by GPU "
                     f"(range {fracs[0]:.0f}-{fracs[-1]:.0f}%).")
            if med >= 25:
                L.append("  => launch/dispatch IS on the critical path -> fusion / launch-reduction helps.")
            else:
                L.append("  => GPU-compute-bound -> prioritize kernel efficiency over fusion.")
            j["sections"]["launch_gap_median_pct"] = med
            L.append("")
    if ln and lb:
        mn = parse_model_benchmark(logtext(ln[0]["log"]))
        mbk = parse_model_benchmark(logtext(lb[0]["log"]))
        n_ms = mn.get("decode_ms"); b_ms = mbk.get("decode_ms")
        L.append("| mode | decode tok/s | decode ms/tok |")
        L.append("|---|---|---|")
        L.append(f"| normal (async) | {mn.get('decode_tps', float('nan')):.1f} | {n_ms if n_ms else float('nan'):.2f} |")
        L.append(f"| HIP_LAUNCH_BLOCKING=1 | {mbk.get('decode_tps', float('nan')):.1f} | {b_ms if b_ms else float('nan'):.2f} |")
        L.append("")
        if n_ms and b_ms and b_ms > 0:
            ratio = b_ms / n_ms
            hidden = b_ms - n_ms
            L.append(f"- Serializing launches changes decode {n_ms:.2f} -> {b_ms:.2f} ms/tok "
                     f"({ratio:.2f}x). ~{hidden:.2f} ms/tok of launch overhead is currently HIDDEN by overlap.")
            if ratio < 1.15:
                L.append("- Interpretation: launches are largely ON the critical path -> kernel-fusion / "
                         "launch-reduction can help decode.")
            else:
                L.append("- Interpretation: launch overhead is mostly OVERLAPPED behind compute -> fusion "
                         "yields little decode TPS; focus on the big compute kernels.")
            j["sections"]["launch_ab"] = {"normal_ms": n_ms, "blocking_ms": b_ms, "ratio": ratio}
    else:
        L.append("_Launch A/B logs incomplete._")
    L.append("")

    # --- F. Prefill / TTFT ---
    L.append("## F. Prefill / TTFT")
    L.append("")
    if headline:
        L.append("- TTFT vs prompt (from section A): "
                 + ", ".join(f"p{p}={headline[p].get('ttft_s', float('nan')):.2f}s" for p in sorted(headline)))
        L.append("")
    # Prefill per-op attribution: the first Compute table (prompt tokens, m>1).
    if dp:
        ptables = parse_perf_tables(logtext(dp[0]["log"]))
        pref = _pick_prefill_table(ptables)
        if pref:
            ptot = pref.get("total_gpu") or sum(o["gpu"] for o in pref["ops"].values())
            L.append(f"- Prefill Compute per-op GPU breakdown (total {ptot:.1f} ms):")
            L.append("")
            L.append("| op | calls | gpu ms | % prefill |")
            L.append("|---|---|---|---|")
            for name, op in sorted(pref["ops"].items(), key=lambda kv: kv[1]["gpu"], reverse=True)[:8]:
                L.append(f"| {name} | {op['calls']} | {op['gpu']:.1f} | {op['pct']:.1f}% |")
                csv_rows.append(["prefill_op", "", name, op["calls"], op["gpu"], op["pct"], "", "", "", ""])
            j["sections"]["prefill"] = {"total_gpu_ms": ptot,
                                        "ops": [{"op": n, "gpu": o["gpu"], "pct": o["pct"]}
                                                for n, o in pref["ops"].items()]}
        else:
            L.append("- (No prefill Compute table found -- run needs a captured prompt-phase [PERF] table.)")
    L.append("")

    # --- G. Cold start ---
    cs = find(lambda r: r.get("type") == "coldstart")
    L.append("## G. Cold start")
    L.append("")
    if cs:
        L.append(f"- Minimal-run (l=8,g=1) process wall: {cs[0].get('wall_ms', 0)/1000.0:.1f} s "
                 "(model load + compile + weight upload + first-inference autotune, load-dominated).")
        # EP-measured [COLDSTART] phase timers (any log may carry them).
        # Cold-start lines are emitted once per process load. Parse a SINGLE
        # log (the coldstart run) to avoid multi-counting across battery runs.
        phases = {}
        counts = {}
        for line in logtext(cs[0]["log"]).splitlines():
            mm = _COLDSTART.match(line)
            if mm:
                k = mm.group(1)
                phases[k] = phases.get(k, 0.0) + float(mm.group(2))
                counts[k] = counts.get(k, 0) + 1
        if phases:
            L.append("- EP phase timers (measured, summed over cold-start events):")
            for k, v in sorted(phases.items(), key=lambda kv: kv[1], reverse=True):
                L.append(f"  - {k}: {v:.3f} s (n={counts[k]})")
            j["sections"]["coldstart_phases"] = phases
        else:
            L.append("- (No [COLDSTART] phase lines found -- rebuild with Phase 2 instrumentation.)")
    L.append("")

    # --- Per-shape isolated roofline (single_op) ---
    sops = find(lambda r: r.get("type") == "single_op")
    if sops:
        L.append("## (aux) Isolated per-shape kernel time (single_op seq1, gs128 graphs)")
        L.append("")
        L.append("| op graph | shape | calls | gpu ms |")
        L.append("|---|---|---|---|")
        for r in sops:
            si = parse_singleop_gpu(logtext(r["log"]))
            if si:
                L.append(f"| {r['op']} | {si['shape']} | {si['calls']} | {si['gpu']:.3f} |")
                csv_rows.append(["single_op", "", r["op"], si["calls"], si["gpu"], "", "", "", "", si["shape"]])
        L.append("")

    # --- H. Ranked bottleneck list ---
    L.append("## H. Ranked decode bottlenecks (by GPU-time share)")
    L.append("")
    if decode_rows:
        for i, d in enumerate(sorted(decode_rows, key=lambda x: x["pct"], reverse=True)[:6], 1):
            lever = _lever_for(d["op"], d.get("pct_peak"))
            pk = f"{d['pct_peak']:.0f}% peak" if d.get("pct_peak") else "n/a"
            L.append(f"{i}. **{d['op']}** -- {d['pct']:.1f}% of decode ({pk}). {lever}")
    L.append("")

    (probe_dir / "bottleneck_report.md").write_text("\n".join(L), encoding="utf-8")
    with (probe_dir / "bottleneck_report.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["section", "prompt", "name", "calls", "gpu_or_tps", "pct_or_ttft",
                    "mb_or_ws", "gbps", "pct_peak", "extra"])
        w.writerows(csv_rows)
    (probe_dir / "bottleneck_report.json").write_text(json.dumps(j, indent=2), encoding="utf-8")
    return "\n".join(L)


def _op_bytes(name, op, block, peak):
    """Return (MB, GB/s, %peak). Prefer EP-measured columns when present;
    otherwise fall back to the shape-driven estimate for matmul_nbits."""
    if op.get("mb_meas"):
        return op["mb_meas"], op.get("gbps_meas") or 0.0, op.get("pk_meas") or 0.0
    if name == "matmul_nbits" and op.get("shapes"):
        total = 0.0
        for label, rec in op["shapes"].items():
            sm = _SHAPE_NK.search(label)
            if sm:
                mI, nI, kI = int(sm.group(1)), int(sm.group(2)), int(sm.group(3))
                total += matmul_nbits_bytes(mI, nI, kI, bits_for(nI, kI), block) * rec["calls"]
        if total and op["gpu"] > 0:
            gbps = (total / 1e9) / (op["gpu"] / 1000.0)
            return total / 1e6, gbps, 100.0 * gbps / peak
    return 0.0, 0.0, 0.0


def _lever_for(op, pct_peak):
    if op == "matmul_nbits":
        return "Split into medium-N GEMV kernel efficiency (headroom) + small-N launch/batching (overhead-bound); see section C."
    if op == "qmoe":
        return "Grouped-expert GEMM (biggest prefill/TTFT lever); decode kernel BW tuning."
    if op in ("skip_layernorm", "layernorm", "activation", "power", "silu", "reduce_sum", "transpose"):
        return "Small-op tail -- fuse only if section E shows launches on the critical path."
    if op == "gqa":
        return "Full-attention decode kernel; check occupancy at m=1."
    return "Investigate per section E (launch-bound?) and roofline % above."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("probe_dir", help="perf-results/probe_<ts> directory")
    args = ap.parse_args()
    probe_dir = Path(args.probe_dir)
    if not (probe_dir / "manifest.json").exists():
        print(f"error: no manifest.json in {probe_dir}", file=sys.stderr)
        return 1
    build_report(probe_dir)
    print(f"wrote {probe_dir/'bottleneck_report.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
