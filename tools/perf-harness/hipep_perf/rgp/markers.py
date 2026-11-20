#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""RGP SQTT markers, reassembled from the USERDATA_2/3 register writes.

RGP (PAL's CmdInsertRgpTraceMarker) streams markers two dwords per packet into
the SQ thread-trace user-data registers; the token decoder surfaces each dword
as an EV_RGP_MARKER event. Read back in capture order they are Mesa's
rgp_sqtt_marker_* records (src/amd/common/ac_sqtt.h): per-dispatch EVENT
markers carry the grid dimensions, BARRIER_START/END bracket driver barriers
and carry the exact cache-invalidate and stall bits.
"""

from __future__ import annotations

from .tokens import EV_RGP_MARKER

# Dword count of each marker kind before its extension dwords.
_SIZE = {
    0x0: 3,
    0x1: 4,
    0x2: 3,
    0x3: 2,
    0x4: 2,
    0x5: 1,
    0x6: 1,
    0x7: 1,
    0x8: 3,
    0x9: 2,
    0xA: 3,
    0xC: 3,
}


def _qkey(me: int, pipe: int) -> int:
    return ((me & 7) << 4) | (pipe & 0xF)


def grids_by_queue(ev_raw: list[tuple]) -> dict[int, list[int]]:
    """Per queue, the grid workgroup count of each dispatch EVENT marker, in order.

    ``ev_raw`` rows are the token decoder's (type, t, me, pipe, payload, flags).
    """
    dwords: dict[int, list[int]] = {}
    for e in ev_raw:
        if e[0] == EV_RGP_MARKER:
            dwords.setdefault(_qkey(e[2], e[3]), []).append(e[4] & 0xFFFFFFFF)
    out = {}
    for qk, dw in dwords.items():
        grids, i, n = [], 0, len(dw)
        while i < n:
            d0 = dw[i]
            ident, ext = d0 & 0xF, (d0 >> 4) & 0x7
            if ident == 0x0:
                has = (d0 >> 31) & 1
                grids.append(
                    dw[i + 3] * dw[i + 4] * dw[i + 5] if (has and i + 5 < n) else 0
                )
                i += 3 + (3 if has else 0) + ext
            elif ident == 0x1:
                i += 4 + ext
            elif ident in (0x2, 0xC):
                i += 3 + ext
            elif ident in (0x3, 0x4, 0x9):
                i += 2 + ext
            else:
                i += 1 + ext
        out[qk] = grids
    return out


def _barrier_bits(d01: int, d02: int) -> dict:
    caches = [
        name
        for bit, name in (
            (3, "K"),
            (1, "L0"),
            (2, "I"),
            (26, "L1"),
            (5, "L2"),
            (4, "L2flush"),
        )
        if (d02 >> bit) & 1
    ]
    stalls = [
        name
        for bit, name in (
            (30, "CS"),
            (28, "VS"),
            (29, "PS"),
            (27, "waitEOP"),
            (31, "pfpSyncMe"),
        )
        if (d01 >> bit) & 1
    ]
    return {
        "caches_invalidated": ", ".join(caches) or "None",
        "frontend_sync": ", ".join(stalls) or "None",
        "cb_id": (d01 >> 7) & 0xFFFFF,
    }


def parse_queue(events: list[dict], me: int, pipe: int):
    """(barriers, [(ts, grid_workgroups)]) for one queue, or (None, None) if the
    capture carries no markers for it. ``events`` are decoded event dicts."""
    res = sorted(
        (
            e
            for e in events
            if e.get("type") == EV_RGP_MARKER
            and e.get("me") == me
            and e.get("pipe") == pipe
        ),
        key=lambda e: e.get("seq", 0),
    )
    if not res:
        return None, None
    dw = [(e["ts"], int(e["payload"]) & 0xFFFFFFFF) for e in res]
    barriers, grids = [], []
    i, pend = 0, None
    while i < len(dw):
        ts, d01 = dw[i]
        ident, ext = d01 & 0xF, (d01 >> 4) & 0x7
        if ident == 0x0:
            has = (d01 >> 31) & 1
            if has and i + 5 < len(dw):
                grids.append((ts, dw[i + 3][1] * dw[i + 4][1] * dw[i + 5][1]))
            i += 3 + (3 if has else 0) + ext
        elif ident == 0x3:
            pend = ts
            i += 2 + ext
        elif ident == 0x4:
            d02 = dw[i + 1][1] if i + 1 < len(dw) else 0
            start = pend if pend is not None else ts - 0.5
            span = max(ts - start, 0.0)
            barriers.append(
                {
                    "me": me,
                    "pipe": pipe,
                    "ts": start,
                    "dur": span,
                    "args": {
                        **_barrier_bits(d01, d02),
                        "barrier_type": "DRIVER",
                        "interval_us": round(span, 3),
                        "source": "RGP SQTT markers: cache/stall exact; "
                        "interval = BarrierEnd.ts - BarrierStart.ts",
                    },
                }
            )
            pend = None
            i += 2 + ext
        else:
            i += _SIZE.get(ident, 1) + ext
    return barriers, grids
