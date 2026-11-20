#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Decode one SQTT stream per shader engine into dispatch / event / occupancy rows.

Each shader engine stamps its records with its own shader-clock counter, and the
counters drift apart by a different amount in every capture. Each engine also
records (shader clock, 100 MHz realtime) pairs, so engine N's wave times are
mapped onto engine 0's shader clock through realtime with engine N's own pairs,
and engine N's waves are matched against engine N's own launch records.
``HIPEP_RGP_PER_SE_CLOCK=0`` (or the legacy ``RGP_PARSER_PER_SE_CLOCK=0``) uses
engine 0's clock everywhere, for A/B comparisons.
"""

from __future__ import annotations

import bisect
import os

from . import markers, tokens

INF = float("inf")
RT_FREQ = 100_000_000  # RGP captures carry no RT frequency record; 100 MHz


def _per_se_clock() -> bool:
    for k in ("HIPEP_RGP_PER_SE_CLOCK", "RGP_PARSER_PER_SE_CLOCK"):
        if os.environ.get(k) == "0":
            return False
    return True


def _to_engine0(rts_se, rts_0):
    """f(shader clock on engine N) -> shader clock on engine 0, piecewise linear
    through realtime; None if either engine has fewer than two pairs."""
    a = sorted(rts_se)
    b = sorted(rts_0, key=lambda p: p[1])
    if len(a) < 2 or len(b) < 2:
        return None
    a_sc = [p[0] for p in a]
    b_rt = [p[1] for p in b]

    def interp(xs, pts, x, xi, yi):
        j = min(max(1, bisect.bisect_left(xs, x)), len(pts) - 1)
        p0, p1 = pts[j - 1], pts[j]
        dx = p1[xi] - p0[xi]
        return p0[yi] + (x - p0[xi]) * (p1[yi] - p0[yi]) / dx if dx else p0[yi]

    def conv(sc):
        return round(interp(b_rt, b, interp(a_sc, a, sc, 0, 1), 1, 0))

    return conv


def _own_launches(disps_se, disps_0):
    ds = sorted(disps_se, key=lambda d: d["t"])
    if len(ds) != len(disps_0):
        return None
    for a, b in zip(ds, disps_0):
        if (a["pc"], a["me"], a["pipe"]) != (b["pc"], b["me"], b["pipe"]):
            return None
    return [d["t"] for d in ds]


def _qkey(me: int, pipe: int) -> int:
    return ((me & 7) << 4) | (pipe & 0xF)


def decode(blobs: list[bytes], wave_size: int = 32) -> dict:
    disps, ev_raw, rts, occ0 = tokens.parse_engine(blobs[0], 0, True)
    disps.sort(key=lambda d: d["t"])
    per_se = _per_se_clock()
    occ_all, clock, own = [occ0], [None], [None]
    for i in range(1, len(blobs)):
        d_i, _, rts_i, occ_i = tokens.parse_engine(blobs[i], i, False, launches=per_se)
        occ_all.append(occ_i)
        clock.append(_to_engine0(rts_i, rts) if per_se else None)
        own.append(_own_launches(d_i, disps) if per_se else None)

    for d in disps:
        d.update(exp_waves=0, wf=0, min_start=INF, max_end=0)

    # Pair wave start/end per hardware slot; build the per-engine occupancy edges.
    pending: dict[int, tuple] = {}
    waves: dict[int, list] = {}
    edges: list[list[tuple[int, int]]] = [[] for _ in blobs]
    for se, occ in enumerate(occ_all):
        conv = clock[se]
        out = edges[se]
        for raw, start, cu, simd, wid, me, pipe, pc in occ:
            time = conv(raw) if conv is not None and raw else raw
            slot = (se << 24) | ((cu & 0x7F) << 16) | ((simd & 0xF) << 8) | (wid & 0xFF)
            out.append((time, 1 if start else -1))
            if start:
                pending[slot] = (time, pc, me, pipe, raw)
                continue
            p = pending.pop(slot, None)
            if p is None:
                continue
            st, ppc, pme, ppipe, st_raw = p
            waves.setdefault((ppc << 8) | _qkey(pme, ppipe), []).append(
                (st, time, se, st_raw)
            )

    # Expected waves per dispatch from the RGP marker grids, paired in launch order
    # per queue. This caps attribution so a sparse dispatch cannot absorb waves
    # from the idle gap before the next launch of the same kernel.
    grids = markers.grids_by_queue(ev_raw)
    by_queue: dict[int, list[int]] = {}
    for i, d in enumerate(disps):
        by_queue.setdefault(_qkey(d["me"], d["pipe"]), []).append(i)
    for qk, idxs in by_queue.items():
        g = grids.get(qk, [])
        for k, di in enumerate(idxs):
            d = disps[di]
            wg = d["wg"] or 1
            d["exp_waves"] = (g[k] if k < len(g) else 0) * (
                (wg + wave_size - 1) // wave_size
            )

    # Attribute waves to the latest launch of the same (pc, queue) before them.
    by_pc: dict[int, list[int]] = {}
    for i, d in enumerate(disps):
        by_pc.setdefault((d["pc"] << 8) | _qkey(d["me"], d["pipe"]), []).append(i)
    for key, idxs in by_pc.items():
        wv = waves.get(key)
        if not wv:
            continue
        times = [disps[i]["t"] for i in idxs]
        own_t = [None if sl is None else [sl[i] for i in idxs] for sl in own]
        wv.sort(key=lambda w: w[0])
        for ws, we, se, ws_raw in wv:
            if own_t[se] is not None:
                k = bisect.bisect_right(own_t[se], ws_raw) - 1
            else:
                k = bisect.bisect_right(times, ws) - 1
            d = disps[idxs[max(k, 0)]]
            if d["exp_waves"] > 0 and d["wf"] >= d["exp_waves"]:
                continue
            d["wf"] += 1
            d["min_start"] = min(d["min_start"], ws)
            d["max_end"] = max(d["max_end"], we)

    # Shader clock -> realtime microseconds from the first record.
    rts_sorted = sorted(rts, key=lambda r: r[0])
    rt_sh = [r[0] for r in rts_sorted]

    def sc_to_rt(sc):
        if not rts_sorted:
            return float(sc)
        if sc <= rts_sorted[0][0]:
            return float(rts_sorted[0][1])
        if sc >= rts_sorted[-1][0]:
            return float(rts_sorted[-1][1])
        j = bisect.bisect_left(rt_sh, sc)
        a, b = rts_sorted[j - 1], rts_sorted[j]
        return a[1] + (sc - a[0]) / max(1, b[0] - a[0]) * (b[1] - a[1])

    sc0 = min([d["t"] for d in disps] + [e[1] for e in ev_raw], default=INF)
    if sc0 == INF:
        sc0 = 0
    rt0 = sc_to_rt(sc0)
    us_per_tick = 1e6 / RT_FREQ

    def to_us(sc):
        return (sc_to_rt(sc) - rt0) * us_per_tick

    out_disp = []
    for d in disps:
        has = d["wf"] > 0 and d["min_start"] != INF
        out_disp.append(
            {
                "se": d["se"],
                "me": d["me"],
                "pipe": d["pipe"],
                "ts": round(to_us(d["min_start"] if has else d["t"]), 4),
                "launch_ts": round(to_us(d["t"]), 4),
                "dur": round(to_us(d["max_end"]) - to_us(d["min_start"]), 4)
                if has
                else -1.0,
                "wavefronts": d["wf"],
                "exp_waves": d["exp_waves"],
                "flags": d["flags"],
                "vgpr": d["vgpr"],
                "sgpr": d["sgpr"],
                "lds": d["lds"],
                "wg": d["wg"],
                "pc": d["pc"],
            }
        )
    out_ev = [
        {
            "me": me,
            "pipe": pipe,
            "ts": round(to_us(rawt), 4),
            "rawt": rawt,
            "type": ty,
            "flags": flags,
            "payload": payload,
            "seq": seq,
        }
        for seq, (ty, rawt, me, pipe, payload, flags) in enumerate(ev_raw)
    ]
    return {
        "rt_freq": RT_FREQ,
        "rt0": int(rt0),
        "wave_size": wave_size,
        "dispatches": out_disp,
        "events": out_ev,
        "occupancy": occupancy_band(edges, len(blobs), to_us),
    }


def occupancy_band(edges, n_se: int, to_us) -> list[dict]:
    """Peak alive waves per ~2 us bucket, per engine (4,000-120,000 buckets)."""
    ns = min(n_se, 8)
    times = [t for s in range(ns) for t, _ in edges[s]]
    if not times:
        return []
    tmin, tmax = min(times), max(times)
    if not tmax > tmin:
        return []
    nsamp = min(max(int((to_us(tmax) - to_us(tmin)) / 2.0), 4000), 120000)
    for s in range(ns):
        edges[s].sort()
    cur, idx = [0] * ns, [0] * ns
    span = tmax - tmin
    band = []
    for k in range(nsamp):
        t1 = tmin + int(span * (k + 1) / nsamp)
        row = {"ts": round(to_us(tmin + int(span * (k + 0.5) / nsamp)), 4)}
        for s in range(ns):
            e = edges[s]
            ci, i = cur[s], idx[s]
            peak = max(0, ci)
            while i < len(e) and e[i][0] < t1:
                ci += e[i][1]
                peak = max(peak, ci)
                i += 1
            cur[s], idx[s] = ci, i
            row[f"se{s}"] = peak
        band.append(row)
    return band
