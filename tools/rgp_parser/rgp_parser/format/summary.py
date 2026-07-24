#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""summary.{md,json} - compact, AI-primary overview of a capture.

High signal-per-token: capture metadata + top-N kernels rolled up. Small enough to
drop into a prompt, which is what 'analyze this trace' usually needs. ``write``
emits Markdown and a companion ``.json`` beside it.
"""

from __future__ import annotations

import json
import os

from ..model import Trace
from . import operators

TOP_N = 25


def _bottleneck_rollup(trace: Trace) -> list:
    """Time + count grouped by bound_class over real dispatches, descending."""
    busy = trace.gpu_busy_us() or 1.0
    agg: dict[str, list] = {}
    for d in trace.dispatches:
        if d.is_artifact or d.dur_us <= 0:
            continue
        cls = d.bound_class or "(unclassified)"
        cur = agg.setdefault(cls, [0.0, 0])
        cur[0] += d.dur_us
        cur[1] += 1
    rows = [
        {
            "class": cls,
            "total_us": round(t, 3),
            "count": n,
            "pct_gpu": round(100.0 * t / busy, 1),
        }
        for cls, (t, n) in agg.items()
    ]
    rows.sort(key=lambda r: r["total_us"], reverse=True)
    return rows


def build(trace: Trace, top_n: int = TOP_N) -> dict:
    ops = operators.aggregate(trace)
    fams = operators.aggregate_families(trace)
    busy = trace.gpu_busy_us()
    wall = trace.wall_span_us()
    n_art = sum(1 for d in trace.dispatches if d.is_artifact)
    gaps = trace.idle_gaps(top_n)
    total_idle = round(sum(d.idle_after_us for d in trace.dispatches), 3)
    overhead_us = round(
        sum(
            d.dur_us
            for d in trace.dispatches
            if d.bound_class == "overhead" and not d.is_artifact
        ),
        3,
    )
    return {
        "source": trace.meta.source,
        "gpu": trace.meta.extra.get("asic", ""),
        "shader_engines": trace.meta.shader_engines,
        "wave_size": trace.meta.wave_size,
        "queues": [f"me{m}/pipe{p}" for m, p in trace.meta.queues],
        "spm": trace.meta.extra.get("spm", ""),
        "dispatch_count": len(trace.dispatches),
        "artifact_count": n_art,
        "gpu_busy_us": round(busy, 3),
        "wall_span_us": round(wall, 3),
        # clamp to [0, 100]: busy can exceed wall when dispatches overlap across
        # queues (or on bad decode data), which would otherwise give a nonsense %.
        "gpu_idle_pct": round(min(100.0, max(0.0, 100.0 * (1 - busy / wall))), 1)
        if wall > 0
        else 0.0,
        "total_idle_us": total_idle,
        "overhead_us": overhead_us,
        "distinct_kernels": len(ops),
        "distinct_families": len(fams),
        "bottleneck_classes": _bottleneck_rollup(trace),
        "top_idle_gaps": gaps,
        "top_families": fams[:top_n],
        "top_kernels": ops[:top_n],
    }


def _md(s: dict) -> str:
    lines = [
        "# RGP capture summary",
        "",
        f"- **source**: `{s['source']}`",
        f"- **shader engines**: {s['shader_engines']}  |  **wave size**: {s['wave_size']}",
        f"- **queues**: {', '.join(s['queues']) or '-'}",
        f"- **dispatches**: {s['dispatch_count']} ({s['artifact_count']} artifacts)",
        f"- **GPU busy**: {s['gpu_busy_us']:.1f} us  |  **wall span**: {s['wall_span_us']:.1f} us"
        f"  |  **idle**: {s['gpu_idle_pct']:.1f}% ({s['total_idle_us']:.1f} us)",
        f"- **launch/overhead time**: {s['overhead_us']:.1f} us",
        f"- **distinct kernels**: {s['distinct_kernels']}  |  **distinct families**: {s['distinct_families']}",
        f"- **SPM**: {s['spm'] or 'n/a'}",
    ]

    # --- bottleneck classification -------------------------------------------
    lines += [
        "",
        "## Bottleneck classification (by GPU time)",
        "",
        "| class | total us | % gpu | count |",
        "|---|---:|---:|---:|",
    ]
    for b in s["bottleneck_classes"]:
        lines.append(
            f"| {b['class']} | {b['total_us']:.1f} | {b['pct_gpu']:.1f} | {b['count']} |"
        )

    # --- idle gaps ------------------------------------------------------------
    lines += [
        "",
        f"## Top {len(s['top_idle_gaps'])} idle gaps (GPU bubbles)",
        "",
        "| start us | dur us | preceding -> following | queue |",
        "|---:|---:|---|---|",
    ]
    for g in s["top_idle_gaps"]:
        lines.append(
            f"| {g['start_us']:.1f} | {g['dur_us']:.1f} | "
            f"`{g['prev_label']}` -> `{g['next_label']}` | {g['queue']} |"
        )

    # --- family rollup --------------------------------------------------------
    lines += [
        "",
        f"## Top {len(s['top_families'])} op families by GPU time",
        "",
        "| family | dtype | tile | count | total us | % gpu | mean us | occ % | limiter | bound | mem GB/s |",
        "|---|---|---|---:|---:|---:|---:|---:|---|---|---:|",
    ]
    for k in s["top_families"]:
        lines.append(
            f"| {k['family']} | {k['dtype'] or '-'} | {k['tile'] or '-'} | "
            f"{k['count']} | {k['total_us']:.1f} | {k['pct_gpu']:.1f} | "
            f"{k['mean_us']:.2f} | {k['occ_pct']:.0f} | {k['occ_limiter'] or '-'} | "
            f"{k['bound_class'] or '-'} | "
            f"{k['avg_mem_gbps'] if k['avg_mem_gbps'] else '-'} |"
        )

    # --- per-kernel detail ----------------------------------------------------
    lines += [
        "",
        f"## Top {len(s['top_kernels'])} kernels by GPU time",
        "",
        "| kernel | count | total us | % gpu | mean us | max us | avg wf | occ % | limiter | bound |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for k in s["top_kernels"]:
        lines.append(
            f"| `{k['kernel']}` | {k['count']} | {k['total_us']:.1f} | "
            f"{k['pct_gpu']:.1f} | {k['mean_us']:.2f} | {k['max_us']:.2f} | "
            f"{k['avg_wavefronts']} | {k['occ_pct']:.0f} | {k['occ_limiter'] or '-'} | "
            f"{k['bound_class'] or '-'} |"
        )
    return "\n".join(lines) + "\n"


def write(trace: Trace, path: str) -> None:
    s = build(trace)
    if path.endswith(".json"):
        json_path, md_path = path, os.path.splitext(path)[0] + ".md"
    else:
        md_path, json_path = path, os.path.splitext(path)[0] + ".json"
    with open(md_path, "w") as f:
        f.write(_md(s))
    with open(json_path, "w") as f:
        json.dump(s, f, indent=2)
