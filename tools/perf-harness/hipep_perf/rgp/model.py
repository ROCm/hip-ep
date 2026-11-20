#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""In-memory capture model shared by every output format."""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field


@dataclass
class Dispatch:
    pc: int
    name: str
    me: int
    pipe: int
    ts_us: float  # first wave start
    launch_ts_us: float  # dispatch initiator
    dur_us: float  # min(wave span, next launch gap)
    wavefronts: int
    exp_waves: int
    threads: int
    workgroup: int
    vgpr: int
    sgpr: int
    lds: int
    flags: int = 0
    wave_span_us: float = -1.0
    occupancy_wavefronts: int = 0
    counts_source: str = ""
    duration_note: str = ""
    library: str = ""
    family: str = ""
    dtype: str = ""
    tile: str = ""
    label: str = ""
    gap_before_us: float = 0.0
    idle_after_us: float = 0.0
    occ_theoretical: int = 0
    occ_pct: float = 0.0
    occ_limiter: str = ""
    bound_class: str = ""
    bound_reason: str = ""
    mem_bytes: float = 0.0
    mem_gbps: float = 0.0

    @property
    def is_artifact(self) -> bool:
        """No occupancy waves were attributed, so the duration is a launch gap.

        Keyed off occupancy so an exact marker-grid wavefront count does not hide it.
        """
        occ = self.occupancy_wavefronts if self.counts_source else self.wavefronts
        return occ <= 0 or self.dur_us < 0


@dataclass
class Barrier:
    name: str
    me: int
    pipe: int
    ts_us: float
    dur_us: float
    args: dict = field(default_factory=dict)


@dataclass
class Event:
    me: int
    pipe: int
    ts_us: float
    type: int
    payload: int = 0


@dataclass
class OccupancySample:
    ts_us: float
    per_se: dict = field(default_factory=dict)


@dataclass
class Curve:
    display: str
    key: str
    series: list = field(default_factory=list)
    group: str = ""
    unit: str = ""


@dataclass
class Meta:
    source: str = ""
    rt_freq_hz: int = 0
    wave_size: int = 32
    shader_engines: int = 0
    queues: list = field(default_factory=list)
    extra: dict = field(default_factory=dict)


@dataclass
class Capture:
    meta: Meta = field(default_factory=Meta)
    dispatches: list[Dispatch] = field(default_factory=list)
    barriers: list[Barrier] = field(default_factory=list)
    events: list[Event] = field(default_factory=list)
    occupancy: list[OccupancySample] = field(default_factory=list)
    curves: list[Curve] = field(default_factory=list)

    def wall_span_us(self) -> float:
        ts = [d.ts_us for d in self.dispatches] + [
            d.ts_us + max(d.dur_us, 0.0) for d in self.dispatches
        ]
        return (max(ts) - min(ts)) if ts else 0.0

    def gpu_busy_us(self) -> float:
        return sum(
            d.dur_us for d in self.dispatches if not d.is_artifact and d.dur_us > 0
        )

    def idle_gaps(self, top_n: int = 15) -> list[dict]:
        by_q: dict = {}
        for d in self.dispatches:
            if not d.is_artifact:
                by_q.setdefault((d.me, d.pipe), []).append(d)
        gaps = []
        for (me, pipe), ds in by_q.items():
            ds.sort(key=lambda x: x.ts_us)
            for a, b in itertools.pairwise(ds):
                g = b.ts_us - (a.ts_us + max(a.dur_us, 0.0))
                if g > 0:
                    gaps.append(
                        {
                            "start_us": round(a.ts_us + max(a.dur_us, 0.0), 3),
                            "dur_us": round(g, 3),
                            "prev_kernel": a.name,
                            "next_kernel": b.name,
                            "prev_label": a.label or a.name,
                            "next_label": b.label or b.name,
                            "queue": f"me{me}/pipe{pipe}",
                        }
                    )
        gaps.sort(key=lambda x: x["dur_us"], reverse=True)
        return gaps[:top_n]
