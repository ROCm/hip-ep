#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Normalized in-memory model shared by every formatter.

The parse/decode stages populate a single :class:`Trace`; each formatter in
``rgp_parser.format`` is a pure function of that object. Keeping one canonical
representation means a new output format never has to re-parse the ``.rgp``.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Dispatch:
    """One GPU compute dispatch (kernel launch)."""

    pc: int  # entry-point GPU virtual address
    name: str  # symbolized kernel name (or hex PC if unresolved)
    me: int  # micro-engine id  (queue = (me, pipe))
    pipe: int  # pipe id
    ts_us: float  # first-wave start, microseconds from trace start
    launch_ts_us: float  # dispatch-initiator time (may precede ts_us)
    dur_us: float  # wave-span duration (last wave end - first start)
    wavefronts: int  # attributed occupancy waves
    exp_waves: int  # marker-derived expected wave count (cross-check)
    threads: int  # wavefronts * wave_size
    workgroup: int  # threads per workgroup (block size)
    vgpr: int
    sgpr: int
    lds: int
    flags: int = 0
    wave_span_us: float = -1.0  # raw occupancy wave-span (pre duration refinement)
    occupancy_wavefronts: int = 0  # occupancy-attributed waves (cross-check vs marker)
    counts_source: str = ""  # "Event-marker grid (exact)" | "occupancy (approx...)"
    duration_note: str = ""

    # --- decoded kernel identity (see rgp_parser.kernelinfo) -------------------
    library: str = ""  # tensile | hip | other | unknown
    family: str = ""  # normalized op family, e.g. gemm, rope, matmul_nbits_gemv_dp4a
    dtype: str = ""  # f16 | bf16 | f32 | i8 | ... ("" if undetermined)
    tile: str = ""  # macro tile "MxNxK" for GEMMs ("" otherwise)
    label: str = ""  # short human-readable label

    # --- idle-gap analysis (per queue, execution windows) ---------------------
    gap_before_us: float = 0.0  # GPU-idle before this dispatch on its queue
    idle_after_us: float = 0.0  # GPU-idle after this dispatch until the next one

    # --- occupancy + bottleneck classification (see rgp_parser.occupancy) ------
    occ_theoretical: int = 0  # theoretical resident waves/CU allowed by regs/LDS/WG
    occ_pct: float = 0.0  # theoretical occupancy: % of HW max waves/CU (0 if unknown)
    occ_limiter: str = ""  # VGPR | LDS | workgroup | "" (resource that binds)
    bound_class: str = ""  # overhead | memory-bound | compute-bound | latency/low-occupancy | compute-or-memory (undetermined)
    bound_reason: str = ""  # short auditable reason string

    # --- per-kernel SPM memory attribution (empty unless SPM present) ----------
    mem_bytes: float = 0.0  # attributed bytes moved over the dispatch window
    mem_gbps: float = 0.0  # attributed memory bandwidth (GB/s); 0.0 => unavailable

    @property
    def is_artifact(self) -> bool:
        """True when no real occupancy waves were attributed to this dispatch, so
        its duration is a launch-gap artifact rather than measured execution time.
        Excluded from trustworthy totals. Keyed off occupancy so an exact
        marker-grid wavefront
        override does not flip the classification."""
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
    per_se: dict = field(default_factory=dict)  # {se_index: peak alive waves}


@dataclass
class Curve:
    """A counter curve for the Chrome trace (SPM or occupancy)."""

    display: str  # track label, e.g. "Memory (bytes): L2 write [GB/s]"
    key: str  # per-point arg key (short counter name)
    series: list = field(default_factory=list)  # [(ts_us, value), ...]
    group: str = ""
    unit: str = ""


@dataclass
class CaptureMeta:
    source: str = ""  # .rgp path
    rt_freq_hz: int = 0  # realtime/GPU-timestamp clock
    wave_size: int = 32
    shader_engines: int = 0
    queues: list = field(default_factory=list)  # sorted [(me, pipe), ...]
    extra: dict = field(default_factory=dict)  # AsicInfo/ApiInfo/etc.


@dataclass
class Trace:
    meta: CaptureMeta = field(default_factory=CaptureMeta)
    dispatches: list[Dispatch] = field(default_factory=list)
    barriers: list[Barrier] = field(default_factory=list)
    events: list[Event] = field(default_factory=list)
    occupancy: list[OccupancySample] = field(default_factory=list)
    curves: list[Curve] = field(default_factory=list)  # SPM counter curves

    def wall_span_us(self) -> float:
        ts = [d.ts_us for d in self.dispatches] + [
            d.ts_us + max(d.dur_us, 0.0) for d in self.dispatches
        ]
        return (max(ts) - min(ts)) if ts else 0.0

    def gpu_busy_us(self) -> float:
        """Sum of real (non-artifact) dispatch durations."""
        return sum(
            d.dur_us for d in self.dispatches if not d.is_artifact and d.dur_us > 0
        )

    def idle_gaps(self, top_n: int = 15) -> list:
        """Top idle bubbles between consecutive real dispatches on the same queue.

        Uses ``idle_after_us`` computed in build.py (execution-window gap: the next
        dispatch's start minus this one's end). Returns descending by duration:
        ``[{start_us, dur_us, prev_kernel, next_kernel, queue, prev_label,
        next_label}]``.
        """
        real = [d for d in self.dispatches if not d.is_artifact]
        by_q: dict = {}
        for d in real:
            by_q.setdefault((d.me, d.pipe), []).append(d)
        gaps = []
        for (me, pipe), ds in by_q.items():
            ds.sort(key=lambda x: x.ts_us)
            for a, b in zip(ds, ds[1:]):
                g = b.ts_us - (a.ts_us + max(a.dur_us, 0.0))
                if g <= 0:
                    continue
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
