#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Assemble a ``model.Capture`` from an .rgp file.

container -> SQTT decode -> symbols -> markers (exact grids, barriers) ->
durations -> SPM -> kernel identity, occupancy, idle gaps, bound class.

A dispatch's duration is min(wave span, gap to the next launch on its queue):
the wave span alone can include waves of the next launch of the same kernel.
"""

from __future__ import annotations

import bisect
import itertools
import struct
from pathlib import Path

import numpy as np

from . import kernels, markers, occupancy, spm, sqtt, symbols
from .container import Container
from .model import Barrier, Capture, Curve, Dispatch, Event, Meta, OccupancySample


def _queues(c: Container):
    qinfo, qevents = [], []
    for ch in c.find("QueueInfo"):
        d = c.data(ch)
        if len(d) >= 32:
            hw = struct.unpack_from("<I", d, 24)[0]
            qinfo.append((hw & 0xFF, (hw >> 8) & 0xFF))
    for ch in c.find("QueueEvent"):
        d = c.data(ch)
        if len(d) >= 72:
            et, _cb, _frame, qi, _sub, _api, _cpu, g0, g1 = struct.unpack_from(
                "<IIQIIQQQQ", d, 16
            )
            if et == 0:
                qevents.append((qi, g0, g1))
    return qinfo, qevents


def build(
    path: str | Path,
    *,
    with_spm: bool = True,
    counters_xml: str | None = None,
    roofline_gbps: float = 256.0,
    bytes_per_req: int | None = None,
    legacy_spm: bool = False,
) -> Capture:
    """Decode one capture.

    legacy_spm reproduces tools/rgp_parser's SPM handling for parity runs: the
    first SPM sample pinned to the first dispatch and 64 bytes per EA request.
    Both are wrong against the calibration workload (see spm.py)."""
    if bytes_per_req is None:
        bytes_per_req = 64 if legacy_spm else 32
    path = str(path)
    c = Container.open(path)
    blobs = [c.data(ch) for ch in c.find("SqttData")]
    if not blobs:
        raise ValueError(f"{path}: no SqttData chunks (unusable capture)")
    rec = sqtt.decode(blobs)
    wave = rec["wave_size"]
    raw = rec["dispatches"]
    ev = rec["events"]

    pcw: dict[int, float] = {}
    for d in raw:
        pcw[d["pc"]] = pcw.get(d["pc"], 0.0) + max(0.0, d["dur"])
    names, sym_note = symbols.name_map(c, pcw)

    queues = sorted({(d["me"], d["pipe"]) for d in raw})
    disps: list[Dispatch] = []
    bars: list[Barrier] = []
    wave_sizes: set[int] = set()
    for me, pipe in queues:
        ds = sorted(
            (d for d in raw if d["me"] == me and d["pipe"] == pipe),
            key=lambda x: x["launch_ts"],
        )
        mbar, mgrid = markers.parse_queue(ev, me, pipe)
        grids = [g for _, g in mgrid] if mgrid else []
        exact = mgrid is not None and len(grids) == len(ds)
        launches = sorted(x["launch_ts"] for x in ds)
        for j, d in enumerate(ds):
            occ_wf = d["wavefronts"]
            span = max(0.0, d["dur"]) if d["dur"] >= 0 else 0.0
            lt = d["launch_ts"]
            li = bisect.bisect_right(launches, lt)
            gap = max(0.0, (launches[li] - lt) if li < len(launches) else span)
            dur = min(span, gap) if span > 0 else gap
            if exact:
                threads = grids[j] * d["wg"]
                wf = (
                    -(-threads // wave) if wave else 0
                )  # a partial workgroup still fills a wave
                wave_sizes.add(wave)
                src = "Event-marker grid (exact)"
            else:
                threads, wf = occ_wf * wave, occ_wf
                src = "occupancy (approx; marker grid unavailable)"
            disps.append(
                Dispatch(
                    pc=d["pc"],
                    name=names.get(d["pc"], hex(d["pc"])),
                    me=me,
                    pipe=pipe,
                    ts_us=float(d["ts"]),
                    launch_ts_us=float(lt),
                    dur_us=float(dur),
                    wavefronts=wf,
                    exp_waves=d["exp_waves"],
                    threads=threads,
                    workgroup=d["wg"],
                    vgpr=d["vgpr"],
                    sgpr=d["sgpr"],
                    lds=d["lds"],
                    flags=d["flags"],
                    wave_span_us=round(span, 3),
                    occupancy_wavefronts=occ_wf,
                    counts_source=src,
                    duration_note="min(occupancy wave-span, next-launch gap); approx",
                )
            )
        for b in mbar or []:
            bars.append(
                Barrier(
                    "CmdBarrier()", b["me"], b["pipe"], b["ts"], b["dur"], b["args"]
                )
            )

    events = [
        Event(e["me"], e["pipe"], float(e["ts"]), e["type"], e["payload"]) for e in ev
    ]
    occ = [
        OccupancySample(
            float(o["ts"]), {int(k[2:]): v for k, v in o.items() if k.startswith("se")}
        )
        for o in rec["occupancy"]
    ]

    curves: list[Curve] = []
    spm_note = ""
    mem_t = np.empty(0)
    mem_req = np.empty(0, dtype=np.int64)
    if with_spm:
        try:
            session = spm.parse(c)
        except Exception as e:  # noqa: BLE001 - SPM decode is best-effort
            session, spm_note = None, f"SPM decode skipped: {e}"
        if session:
            if legacy_spm:
                t0 = min((float(d["ts"]) for d in raw), default=0.0)
            else:
                t0 = spm.offset_us(session, rec["rt0"])
            mem_t, mem_req = spm.ea_requests(session, anchor_us=t0)
            for cv in spm.curves(
                session, counters_xml, anchor_us=t0, bytes_per_req=bytes_per_req
            ):
                curves.append(
                    Curve(
                        f"{cv['group']}: {cv['name']} [{cv['unit']}]",
                        cv["name"],
                        cv["series"],
                        cv["group"],
                        cv["unit"],
                    )
                )
            ms = spm.mem_summary(session, bytes_per_req, roofline_gbps)
            if ms:
                spm_note = (
                    f"unified-mem (EA ev55): ~{ms['avg_gbps']:.1f} GB/s avg over "
                    f"{ms['window_ms']:.1f} ms (~{ms['pct_roofline']:.0f}% of "
                    f"{roofline_gbps:.0f} GB/s roofline @{ms['bytes_per_req']}B/req)"
                )
        elif not spm_note:
            spm_note = "no SPM counters in capture"

    for d in disps:
        ki = kernels.parse(d.name)
        d.library, d.family, d.dtype, d.tile, d.label = (
            ki.library,
            ki.family,
            ki.dtype,
            ki.tile,
            ki.label,
        )
        d.occ_pct, d.occ_limiter, d.occ_theoretical = occupancy.theoretical(
            d.vgpr, d.sgpr, d.lds, d.workgroup, wave_size=wave
        )

    for q in queues:
        ds = sorted(
            (d for d in disps if (d.me, d.pipe) == q and not d.is_artifact),
            key=lambda x: x.ts_us,
        )
        for a, b in itertools.pairwise(ds):
            g = round(max(0.0, b.ts_us - (a.ts_us + max(a.dur_us, 0.0))), 3)
            a.idle_after_us = g
            b.gap_before_us = g

    # Per-dispatch bandwidth: EA requests arriving in the window times bytes per
    # request over the window, needing >= 3 samples. Summing counts (not averaging
    # per-sample rates) is what keeps it physical.
    if len(mem_t):
        csum = np.concatenate(([0], np.cumsum(mem_req)))
        for d in disps:
            if d.is_artifact or d.dur_us <= 0:
                continue
            lo = int(np.searchsorted(mem_t, d.ts_us, side="left"))
            hi = int(np.searchsorted(mem_t, d.ts_us + d.dur_us, side="right"))
            if hi - lo >= 3:
                span_us = max(float(mem_t[hi - 1] - mem_t[lo]), d.dur_us)
                moved = int(csum[hi] - csum[lo]) * bytes_per_req
                d.mem_bytes = round(moved, 1)
                d.mem_gbps = round(moved / (span_us * 1e3), 2) if span_us else 0.0

    for d in disps:
        if not d.is_artifact:
            d.bound_class, d.bound_reason = occupancy.classify(
                d,
                spm_present=bool(len(mem_t)),
                mem_reliable=d.mem_gbps > 0,
                roofline_gbps=roofline_gbps,
            )

    qinfo, qevents = _queues(c)
    per_q = {q: sum(1 for d in disps if (d.me, d.pipe) == q) for q in queues}
    primary = max(queues, key=lambda q: per_q[q]) if queues else None
    rt = rec["rt_freq"] or 100_000_000
    span_ms = (
        (max(g1 for _, _, g1 in qevents) - min(g0 for _, g0, _ in qevents)) / rt * 1e3
        if qevents
        else 0.0
    )
    n_compute = sum(1 for _qt, et in qinfo if et in (2, 3))
    meta = Meta(
        source=path,
        rt_freq_hz=rt,
        wave_size=wave,
        shader_engines=len(blobs),
        queues=queues,
        extra={
            "chunks": c.counts(),
            "symbolization": sym_note,
            "spm": spm_note,
            "queue_mapping": (
                f"{len(qevents)} cmdbuf submits; {n_compute} compute HW "
                f"queue(s); submit span {span_ms:.1f} ms"
            )
            if qevents
            else "no QueueEvent chunks",
            "primary_queue": f"me{primary[0]}/pipe{primary[1]}" if primary else None,
            "wave_sizes": sorted(wave_sizes) or [wave],
            "clock": f"GPU timestamp {rt / 1e6:.0f} MHz",
        },
    )
    return Capture(meta, disps, bars, events, occ, curves)


def verify(path: str | Path, need_spm: bool = False) -> tuple[bool, str]:
    """Is the capture decodable and does it hold what a decode needs?"""
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return False, "missing or empty file"
    try:
        c = Container.open(p)
    except ValueError as e:
        return False, str(e)
    counts = c.counts()
    if not counts.get("SqttData"):
        return False, f"no SqttData chunks ({counts})"
    if need_spm and not counts.get("SpmCounterData"):
        return False, f"no SpmCounterData chunks ({counts})"
    return True, ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
