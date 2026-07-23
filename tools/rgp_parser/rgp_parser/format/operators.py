#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""operators.csv - per-kernel aggregate ('where does GPU time go').

Grouped by full kernel signature (template args kept distinct: ``foo<1024,1>`` and
``foo<256,64>`` are separate rows). ``wave_span=0`` launch-gap artifacts are counted
separately and flagged so real totals can exclude them.
"""

from __future__ import annotations

import csv
from statistics import median

from ..model import Trace

FIELDS = [
    "kernel",
    "library",
    "family",
    "dtype",
    "tile",
    "count",
    "artifact_count",
    "total_us",
    "pct_gpu",
    "mean_us",
    "median_us",
    "p90_us",
    "max_us",
    "avg_wavefronts",
    "avg_threads",
    "workgroup",
    "vgpr",
    "sgpr",
    "lds",
    "occ_pct",
    "occ_limiter",
    "bound_class",
    "avg_mem_gbps",
    "pc",
]

# _families.csv columns (rollup by op family / dtype / tile).
FAMILY_FIELDS = [
    "family",
    "dtype",
    "tile",
    "library",
    "label",
    "count",
    "total_us",
    "pct_gpu",
    "mean_us",
    "occ_pct",
    "occ_limiter",
    "bound_class",
    "avg_mem_gbps",
]


def _p90(xs):
    if not xs:
        return 0.0
    s = sorted(xs)
    return s[min(len(s) - 1, int(round(0.9 * (len(s) - 1))))]


def _dominant(strs):
    """Most common non-empty string in an iterable ("" if none)."""
    counts: dict[str, int] = {}
    for s in strs:
        if s:
            counts[s] = counts.get(s, 0) + 1
    return max(counts, key=counts.get) if counts else ""


def aggregate(trace: Trace):
    groups: dict[str, list] = {}
    for d in trace.dispatches:
        groups.setdefault(d.name, []).append(d)
    busy = trace.gpu_busy_us() or 1.0

    rows = []
    for name, ds in groups.items():
        real = [d for d in ds if not d.is_artifact and d.dur_us > 0]
        durs = [d.dur_us for d in real]
        total = sum(durs)
        mem = [d.mem_gbps for d in real if d.mem_gbps > 0]
        rows.append(
            {
                "kernel": name,
                "library": ds[0].library,
                "family": ds[0].family,
                "dtype": ds[0].dtype,
                "tile": ds[0].tile,
                "count": len(ds),
                "artifact_count": sum(1 for d in ds if d.is_artifact),
                "total_us": round(total, 3),
                "pct_gpu": round(100.0 * total / busy, 2),
                "mean_us": round(total / len(durs), 3) if durs else 0.0,
                "median_us": round(median(durs), 3) if durs else 0.0,
                "p90_us": round(_p90(durs), 3),
                "max_us": round(max(durs), 3) if durs else 0.0,
                "avg_wavefronts": round(sum(d.wavefronts for d in real) / len(real), 1)
                if real
                else 0,
                "avg_threads": round(sum(d.threads for d in real) / len(real), 1)
                if real
                else 0,
                "workgroup": ds[0].workgroup,
                "vgpr": ds[0].vgpr,
                "sgpr": ds[0].sgpr,
                "lds": ds[0].lds,
                "occ_pct": ds[0].occ_pct,
                "occ_limiter": ds[0].occ_limiter,
                "bound_class": _dominant(d.bound_class for d in real),
                "avg_mem_gbps": round(sum(mem) / len(mem), 2) if mem else 0.0,
                "pc": hex(ds[0].pc),
            }
        )
    rows.sort(key=lambda r: r["total_us"], reverse=True)
    return rows


def aggregate_families(trace: Trace):
    """Roll up dispatches by (family, dtype, tile) so GEMM tile variants and every
    instance of an op collapse into one row -- 'where does GPU time go by op'."""
    groups: dict[tuple, list] = {}
    for d in trace.dispatches:
        groups.setdefault((d.family, d.dtype, d.tile), []).append(d)
    busy = trace.gpu_busy_us() or 1.0

    rows = []
    for (family, dtype, tile), ds in groups.items():
        real = [d for d in ds if not d.is_artifact and d.dur_us > 0]
        durs = [d.dur_us for d in real]
        total = sum(durs)
        mem = [d.mem_gbps for d in real if d.mem_gbps > 0]
        occ = [d.occ_pct for d in real if d.occ_pct]
        rows.append(
            {
                "family": family or "(unknown)",
                "dtype": dtype,
                "tile": tile,
                "library": _dominant(d.library for d in ds),
                "label": _dominant(d.label for d in ds),
                "count": len(ds),
                "total_us": round(total, 3),
                "pct_gpu": round(100.0 * total / busy, 2),
                "mean_us": round(total / len(durs), 3) if durs else 0.0,
                "occ_pct": round(sum(occ) / len(occ), 1) if occ else 0.0,
                "occ_limiter": _dominant(d.occ_limiter for d in real),
                "bound_class": _dominant(d.bound_class for d in real),
                "avg_mem_gbps": round(sum(mem) / len(mem), 2) if mem else 0.0,
            }
        )
    rows.sort(key=lambda r: r["total_us"], reverse=True)
    return rows


def write(trace: Trace, path: str) -> None:
    rows = aggregate(trace)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)


def write_families(trace: Trace, path: str) -> None:
    rows = aggregate_families(trace)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FAMILY_FIELDS)
        w.writeheader()
        w.writerows(rows)
