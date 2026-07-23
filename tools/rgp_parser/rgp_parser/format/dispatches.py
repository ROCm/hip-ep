#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""dispatches.csv - one row per GPU dispatch (flat, for drill-down / custom analysis)."""

from __future__ import annotations

import csv

from ..model import Trace

FIELDS = [
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


def write(trace: Trace, path: str) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for i, d in enumerate(sorted(trace.dispatches, key=lambda x: x.ts_us)):
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
