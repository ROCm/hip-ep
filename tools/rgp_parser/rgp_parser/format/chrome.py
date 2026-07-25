#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""chrome_trace.json - Chrome/Perfetto timeline (human visual inspection).

Per-queue dispatch tracks (greedy lane-packed) + a barriers overlay + a dedicated
counters process carrying the wavefront-occupancy band and SPM memory/cache-bandwidth
curves. Open in https://ui.perfetto.dev or chrome://tracing.
"""

from __future__ import annotations

import json

from ..model import Trace

CTR_PID = 1  # dedicated process for counter ("C") tracks


def _lane_pack(items):
    lanes: list[float] = []
    out = []
    for it in items:
        placed = -1
        for L, last in enumerate(lanes):
            if it["ts"] >= last:
                placed = L
                lanes[L] = it["ts"] + max(it["dur"], 0.0)
                break
        if placed < 0:
            placed = len(lanes)
            lanes.append(it["ts"] + max(it["dur"], 0.0))
        out.append(placed)
    return out


def write(trace: Trace, path: str) -> None:
    queues = trace.meta.queues or sorted({(d.me, d.pipe) for d in trace.dispatches})
    qidx = {q: i for i, q in enumerate(queues)}
    primary = trace.meta.extra.get("primary_queue")
    events_out = []

    def meta(tid, name):
        events_out.append(
            {
                "name": "thread_name",
                "ph": "M",
                "pid": 0,
                "tid": tid,
                "args": {"name": name},
            }
        )

    for me, pipe in queues:
        qi = qidx[(me, pipe)]
        ds = sorted(
            [d for d in trace.dispatches if d.me == me and d.pipe == pipe],
            key=lambda x: x.ts_us,
        )
        items = []
        for d in ds:
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
        for b in trace.barriers:
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
        lanes = _lane_pack(items)
        base_tid = 100 + qi * 10
        used = set()
        for it, L in zip(items, lanes):
            used.add(L)
            events_out.append(
                {
                    "name": it["name"],
                    "cat": it["cat"],
                    "ph": "X",
                    "pid": 0,
                    "tid": base_tid + L,
                    "ts": it["ts"],
                    "dur": it["dur"],
                    "args": it["args"],
                }
            )
        rgp_note = " [RGP Queue 1]" if primary == f"me{me}/pipe{pipe}" else ""
        for L in sorted(used):
            meta(base_tid + L, f"Queue me{me}/pipe{pipe} (COMPUTE){rgp_note} - row {L}")

    # process metadata (queue mapping / symbolization / clock)
    ex = trace.meta.extra
    events_out.append(
        {
            "name": "process_name",
            "ph": "M",
            "pid": 0,
            "tid": 0,
            "args": {
                "name": "gfx1151 SQTT (rgp_parser, RGP-parity)",
                "queue_mapping": ex.get("queue_mapping"),
                "primary_queue": ex.get("primary_queue"),
                "symbolization": ex.get("symbolization"),
                "spm": ex.get("spm"),
                "clock": ex.get("clock"),
            },
        }
    )

    # ---- counter tracks: occupancy band (SQTT) + SPM curves -------------------
    ctr = [
        {
            "name": "process_name",
            "ph": "M",
            "pid": CTR_PID,
            "tid": 0,
            "args": {"name": "GPU counters (occupancy / SPM)"},
        }
    ]

    def emit_counter(name, points, arg_fn):
        for ts, val in points:
            ctr.append(
                {
                    "name": name,
                    "cat": "counter",
                    "ph": "C",
                    "pid": CTR_PID,
                    "tid": 0,
                    "ts": float(ts),
                    "args": arg_fn(val),
                }
            )

    ncurve = 0
    if trace.occupancy:
        pts = [(s.ts_us, s.per_se) for s in trace.occupancy]
        emit_counter(
            "Wavefront occupancy",
            pts,
            lambda se: {f"se{k}": int(v) for k, v in se.items()},
        )
        ncurve += 1
    for cv in trace.curves:
        emit_counter(
            cv.display, cv.series, lambda v, key=cv.key: {key: round(float(v), 3)}
        )
        ncurve += 1
    events_out.extend(ctr)

    with open(path, "w") as f:
        json.dump({"displayTimeUnit": "ns", "traceEvents": events_out}, f)
