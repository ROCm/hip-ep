#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Output formats: one ``write_*`` per artifact, all pure functions of a Capture.

Given base ``out/x``: x_dispatches.csv (one row per dispatch), x_operators.csv
(per kernel signature), x_families.csv (per family/dtype/tile), x_summary.json
and .md (compact overview), x_trace.json (Chrome/Perfetto timeline).
"""

from __future__ import annotations

import csv
import json
import os
from statistics import median

from .model import Capture

DISPATCH_FIELDS = [
    "idx",
    "kernel",
    "family",
    "dtype",
    "tile",
    "label",
    "pc",
    "queue",
    "me",
    "pipe",
    "ts_us",
    "launch_ts_us",
    "dur_us",
    "gap_before_us",
    "idle_after_us",
    "is_artifact",
    "wavefronts",
    "exp_waves",
    "threads",
    "workgroup",
    "vgpr",
    "sgpr",
    "lds",
    "occ_pct",
    "occ_limiter",
    "bound_class",
    "bound_reason",
    "mem_gbps",
]
OPERATOR_FIELDS = [
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
TOP_N = 25


def write_dispatches(cap: Capture, path: str) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=DISPATCH_FIELDS)
        w.writeheader()
        for i, d in enumerate(sorted(cap.dispatches, key=lambda x: x.ts_us)):
            w.writerow(
                {
                    "idx": i,
                    "kernel": d.name,
                    "family": d.family,
                    "dtype": d.dtype,
                    "tile": d.tile,
                    "label": d.label,
                    "pc": hex(d.pc),
                    "queue": f"me{d.me}/pipe{d.pipe}",
                    "me": d.me,
                    "pipe": d.pipe,
                    "ts_us": round(d.ts_us, 4),
                    "launch_ts_us": round(d.launch_ts_us, 4),
                    "dur_us": round(d.dur_us, 4),
                    "gap_before_us": round(d.gap_before_us, 4),
                    "idle_after_us": round(d.idle_after_us, 4),
                    "is_artifact": int(d.is_artifact),
                    "wavefronts": d.wavefronts,
                    "exp_waves": d.exp_waves,
                    "threads": d.threads,
                    "workgroup": d.workgroup,
                    "vgpr": d.vgpr,
                    "sgpr": d.sgpr,
                    "lds": d.lds,
                    "occ_pct": d.occ_pct,
                    "occ_limiter": d.occ_limiter,
                    "bound_class": d.bound_class,
                    "bound_reason": d.bound_reason,
                    "mem_gbps": d.mem_gbps,
                }
            )


def _p90(xs):
    if not xs:
        return 0.0
    s = sorted(xs)
    return s[min(len(s) - 1, round(0.9 * (len(s) - 1)))]


def _dominant(strs):
    counts: dict[str, int] = {}
    for s in strs:
        if s:
            counts[s] = counts.get(s, 0) + 1
    return max(counts, key=counts.get) if counts else ""


def operators(cap: Capture) -> list[dict]:
    groups: dict[str, list] = {}
    for d in cap.dispatches:
        groups.setdefault(d.name, []).append(d)
    busy = cap.gpu_busy_us() or 1.0
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


def families(cap: Capture) -> list[dict]:
    groups: dict[tuple, list] = {}
    for d in cap.dispatches:
        groups.setdefault((d.family, d.dtype, d.tile), []).append(d)
    busy = cap.gpu_busy_us() or 1.0
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


def _write_rows(path: str, fields: list[str], rows: list[dict]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def write_operators(cap: Capture, path: str) -> None:
    _write_rows(path, OPERATOR_FIELDS, operators(cap))


def write_families(cap: Capture, path: str) -> None:
    _write_rows(path, FAMILY_FIELDS, families(cap))


def _bottlenecks(cap: Capture) -> list[dict]:
    busy = cap.gpu_busy_us() or 1.0
    agg: dict[str, list] = {}
    for d in cap.dispatches:
        if d.is_artifact or d.dur_us <= 0:
            continue
        cur = agg.setdefault(d.bound_class or "(unclassified)", [0.0, 0])
        cur[0] += d.dur_us
        cur[1] += 1
    rows = [
        {
            "class": k,
            "total_us": round(t, 3),
            "count": n,
            "pct_gpu": round(100.0 * t / busy, 1),
        }
        for k, (t, n) in agg.items()
    ]
    rows.sort(key=lambda r: r["total_us"], reverse=True)
    return rows


def summary(cap: Capture, top_n: int = TOP_N) -> dict:
    ops, fams = operators(cap), families(cap)
    busy, wall = cap.gpu_busy_us(), cap.wall_span_us()
    return {
        "source": cap.meta.source,
        "gpu": cap.meta.extra.get("asic", ""),
        "shader_engines": cap.meta.shader_engines,
        "wave_size": cap.meta.wave_size,
        "queues": [f"me{m}/pipe{p}" for m, p in cap.meta.queues],
        "spm": cap.meta.extra.get("spm", ""),
        "dispatch_count": len(cap.dispatches),
        "artifact_count": sum(1 for d in cap.dispatches if d.is_artifact),
        "gpu_busy_us": round(busy, 3),
        "wall_span_us": round(wall, 3),
        # Busy can exceed the span when queues overlap; clamp to a percentage.
        "gpu_idle_pct": round(min(100.0, max(0.0, 100.0 * (1 - busy / wall))), 1)
        if wall > 0
        else 0.0,
        "total_idle_us": round(sum(d.idle_after_us for d in cap.dispatches), 3),
        "overhead_us": round(
            sum(
                d.dur_us
                for d in cap.dispatches
                if d.bound_class == "overhead" and not d.is_artifact
            ),
            3,
        ),
        "distinct_kernels": len(ops),
        "distinct_families": len(fams),
        "bottleneck_classes": _bottlenecks(cap),
        "top_idle_gaps": cap.idle_gaps(top_n),
        "top_families": fams[:top_n],
        "top_kernels": ops[:top_n],
    }


def _markdown(s: dict) -> str:
    lines = [
        "# RGP capture summary",
        "",
        f"- **source**: `{s['source']}`",
        f"- **shader engines**: {s['shader_engines']}  |  **wave size**: {s['wave_size']}",
        f"- **queues**: {', '.join(s['queues']) or '-'}",
        f"- **dispatches**: {s['dispatch_count']} ({s['artifact_count']} artifacts)",
        (
            f"- **GPU busy**: {s['gpu_busy_us']:.1f} us  |  **wall span**: {s['wall_span_us']:.1f} us"
            f"  |  **idle**: {s['gpu_idle_pct']:.1f}% ({s['total_idle_us']:.1f} us)"
        ),
        f"- **launch/overhead time**: {s['overhead_us']:.1f} us",
        f"- **distinct kernels**: {s['distinct_kernels']}  |  **distinct families**: {s['distinct_families']}",
        f"- **SPM**: {s['spm'] or 'n/a'}",
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
            f"{k['bound_class'] or '-'} | {k['avg_mem_gbps'] if k['avg_mem_gbps'] else '-'} |"
        )
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


def write_summary(cap: Capture, path: str) -> None:
    s = summary(cap)
    base = os.path.splitext(path)[0]
    with open(base + ".md", "w") as f:
        f.write(_markdown(s))
    with open(base + ".json", "w") as f:
        json.dump(s, f, indent=2)


def _lane_pack(items):
    lanes: list[float] = []
    out = []
    for it in items:
        end = it["ts"] + max(it["dur"], 0.0)
        for i, last in enumerate(lanes):
            if it["ts"] >= last:
                lanes[i] = end
                out.append(i)
                break
        else:
            lanes.append(end)
            out.append(len(lanes) - 1)
    return out


TIMELINE_PROCESS = "SQTT (hipep_perf.rgp)"


def write_timeline(cap: Capture, path: str) -> None:
    queues = cap.meta.queues or sorted({(d.me, d.pipe) for d in cap.dispatches})
    primary = cap.meta.extra.get("primary_queue")
    out = []
    for qi, (me, pipe) in enumerate(queues):
        items = []
        for d in sorted(
            (d for d in cap.dispatches if d.me == me and d.pipe == pipe),
            key=lambda x: x.ts_us,
        ):
            args = {
                "queue": f"me{me}/pipe{pipe}",
                "pc": hex(d.pc),
                "workgroup_size": d.workgroup,
                "vgpr_alloc": d.vgpr,
                "sgpr_alloc": d.sgpr,
                "lds": d.lds,
                "is_artifact": d.is_artifact,
                "duration_us": round(max(d.dur_us, 0.0), 3),
                "total_threads": d.threads,
                "total_wavefronts": d.wavefronts,
                "occupancy_wavefronts": d.occupancy_wavefronts,
            }
            if d.wave_span_us >= 0:
                args["wave_span_us"] = d.wave_span_us
            if d.counts_source:
                args["counts_source"] = d.counts_source
            if d.duration_note:
                args["duration_note"] = d.duration_note
            items.append(
                {
                    "name": d.name,
                    "cat": "dispatch",
                    "ts": d.ts_us,
                    "dur": max(d.dur_us, 0.0),
                    "args": args,
                }
            )
        for b in cap.barriers:
            if b.me == me and b.pipe == pipe:
                items.append(
                    {
                        "name": b.name,
                        "cat": "barrier",
                        "ts": b.ts_us,
                        "dur": max(b.dur_us, 0.0),
                        "args": b.args,
                    }
                )
        items.sort(key=lambda x: x["ts"])
        base = 100 + qi * 10
        used = set()
        for it, lane in zip(items, _lane_pack(items)):
            used.add(lane)
            out.append(
                {
                    "name": it["name"],
                    "cat": it["cat"],
                    "ph": "X",
                    "pid": 0,
                    "tid": base + lane,
                    "ts": it["ts"],
                    "dur": it["dur"],
                    "args": it["args"],
                }
            )
        note = " [RGP Queue 1]" if primary == f"me{me}/pipe{pipe}" else ""
        for lane in sorted(used):
            out.append(
                {
                    "name": "thread_name",
                    "ph": "M",
                    "pid": 0,
                    "tid": base + lane,
                    "args": {
                        "name": f"Queue me{me}/pipe{pipe} (COMPUTE){note} - row {lane}"
                    },
                }
            )
    ex = cap.meta.extra
    out.append(
        {
            "name": "process_name",
            "ph": "M",
            "pid": 0,
            "tid": 0,
            "args": {
                "name": TIMELINE_PROCESS,
                "queue_mapping": ex.get("queue_mapping"),
                "primary_queue": ex.get("primary_queue"),
                "symbolization": ex.get("symbolization"),
                "spm": ex.get("spm"),
                "clock": ex.get("clock"),
            },
        }
    )
    out.append(
        {
            "name": "process_name",
            "ph": "M",
            "pid": 1,
            "tid": 0,
            "args": {"name": "GPU counters (occupancy / SPM)"},
        }
    )

    def counter(name, points, args_of):
        for ts, val in points:
            out.append(
                {
                    "name": name,
                    "cat": "counter",
                    "ph": "C",
                    "pid": 1,
                    "tid": 0,
                    "ts": float(ts),
                    "args": args_of(val),
                }
            )

    if cap.occupancy:
        counter(
            "Wavefront occupancy",
            [(s.ts_us, s.per_se) for s in cap.occupancy],
            lambda se: {f"se{k}": int(v) for k, v in se.items()},
        )
    for cv in cap.curves:
        counter(cv.display, cv.series, lambda v, key=cv.key: {key: round(float(v), 3)})
    with open(path, "w") as f:
        json.dump({"displayTimeUnit": "ns", "traceEvents": out}, f)


WRITERS = (
    (write_summary, "_summary.json"),
    (write_operators, "_operators.csv"),
    (write_families, "_families.csv"),
    (write_dispatches, "_dispatches.csv"),
    (write_timeline, "_trace.json"),
)


def write_all(cap: Capture, base: str) -> list[str]:
    root, ext = os.path.splitext(base)
    base = root if ext else base
    if os.path.dirname(base):
        os.makedirs(os.path.dirname(base), exist_ok=True)
    out = []
    for writer, suffix in WRITERS:
        writer(cap, base + suffix)
        out.append(base + suffix)
    return out
