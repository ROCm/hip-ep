#!/usr/bin/env python3
"""Compute TTFT / TPS (and profiling overhead) from benchmark_multimodal.py CSVs.

The Python OGA multimodal bench writes a one-row CSV of averaged metrics.
This reads one or more of them and prints the headline numbers -- plus, when
given exactly two, the profiled-vs-baseline overhead (e.g. HIPDNN_EP_PERF=1
vs =0).

Usage:
    python mm_csv_stats.py <csv> [<csv2> ...]
    python mm_csv_stats.py profiled=run1.csv baseline=run0.csv   # labeled

Prefill TPS is derived (prompt_tokens / TTFT_seconds); everything else is read
straight from the CSV columns. When two CSVs are given, the LAST is treated as
the baseline and the deltas are reported relative to it.
"""
import csv
import sys
from pathlib import Path


def load(path: Path) -> dict:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        sys.exit(f"ERROR: {path} has no data rows")
    r = rows[-1]

    def num(col: str) -> float:
        for k, v in r.items():
            if col.lower() in k.lower():
                return float(v)
        raise KeyError(f"no column matching {col!r} in {path}")

    prompt = int(num("Prompt Length"))
    ttft_ms = num("Prompt Latency")
    return {
        "prompt_tokens": prompt,
        "gen_tokens": int(num("Tokens Generated")),
        "ttft_ms": ttft_ms,
        "prefill_tps": prompt / (ttft_ms / 1000.0) if ttft_ms else 0.0,
        "decode_tps": num("Token Generation Throughput"),
        "decode_ms": num("Token Generation Latency"),
        "sampling_ms": num("Sampling Latency"),
        "wall_tps": num("Wall Clock Throughput"),
        "wall_s": num("Wall Clock Time"),
        "peak_gib": num("Memory Usage"),
    }


def fmt(label: str, m: dict) -> None:
    print(f"== {label} ==")
    print(f"  prompt / gen tokens : {m['prompt_tokens']} / {m['gen_tokens']}")
    print(f"  TTFT                : {m['ttft_ms']:.1f} ms  ({m['ttft_ms']/1000:.2f} s)")
    print(f"  Prefill TPS         : {m['prefill_tps']:.2f} tok/s  (derived prompt/TTFT)")
    print(f"  Decode  TPS         : {m['decode_tps']:.2f} tok/s  ({m['decode_ms']:.2f} ms/tok)")
    print(f"  Sampling            : {m['sampling_ms']:.3f} ms/tok")
    print(f"  Wall-clock TPS      : {m['wall_tps']:.2f} tok/s  (E2E {m['wall_s']:.2f} s)")
    print(f"  Peak working set    : {m['peak_gib']:.2f} GiB")


def pct(new: float, base: float) -> str:
    if base == 0:
        return "n/a"
    return f"{100.0 * (new - base) / base:+.1f}%"


def compare(prof_lbl, prof, base_lbl, base) -> None:
    print(f"== overhead: {prof_lbl} vs {base_lbl} (baseline) ==")
    rows = [
        ("Decode latency (ms/tok)", prof["decode_ms"], base["decode_ms"], "lower=better"),
        ("Decode TPS (tok/s)",      prof["decode_tps"], base["decode_tps"], "higher=better"),
        ("TTFT (ms)",               prof["ttft_ms"],    base["ttft_ms"],    "lower=better"),
        ("Prefill TPS (tok/s)",     prof["prefill_tps"],base["prefill_tps"],"higher=better"),
        ("E2E wall (s)",            prof["wall_s"],     base["wall_s"],     "lower=better"),
    ]
    w = max(len(r[0]) for r in rows)
    print(f"  {'metric':<{w}}  {prof_lbl:>12}  {base_lbl:>12}  {'delta':>10}  {'%':>8}")
    for name, p, b, _ in rows:
        print(f"  {name:<{w}}  {p:>12.2f}  {b:>12.2f}  {p - b:>+10.2f}  {pct(p, b):>8}")


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 1
    items = []  # (label, metrics)
    for a in argv:
        # "label=path" form; Windows drive letters use ':' not '=', so a bare
        # path never contains '=' -> splitting on the first '=' is safe.
        if "=" in a:
            label, _, p = a.partition("=")
        else:
            p = a
            label = Path(p).stem
        path = Path(p)
        if not path.is_file():
            sys.exit(f"ERROR: not a file: {path}")
        items.append((label, load(path)))

    for label, m in items:
        fmt(label, m)
        print()

    if len(items) == 2:
        (plbl, prof), (blbl, base) = items[0], items[1]
        compare(plbl, prof, blbl, base)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
