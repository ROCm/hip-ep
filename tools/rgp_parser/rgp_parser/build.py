#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Assemble a normalized :class:`~rgp_parser.model.Trace` from a ``.rgp`` capture.

Flow:  RDF -> (SQTT decode | records.json) -> symbolize -> markers/barriers ->
       exact counts + refined durations -> SPM/occupancy curves -> Trace

Full RGP-parity trace builder: deterministic CodeObject-hash symbolization, RGP
SQTT marker barriers, Event-marker grid thread/wavefront counts, min(wave-span,
next-launch gap) duration, wavefront-occupancy band, and SPM counter curves.
"""

from __future__ import annotations

import bisect
import struct

from . import kernelinfo, symbolize
from . import occupancy as occ_calc
from .model import Barrier, CaptureMeta, Curve, Dispatch, Event, OccupancySample, Trace
from .struct import coload, spm, sqtt
from .struct.codeobject import load_code_objects, load_code_objects_hashed
from .struct.rdf import RdfFile

# ---- RGP SQTT markers (reassembled from USERDATA_2 + USERDATA_3) --------------
# RGP streams markers 2 dwords/packet into USERDATA_2/3; the patched decoder surfaces
# each as EVENT type 15, payload = the 32-bit regdata, seq = capture order.
# Layouts from Mesa src/amd/common/ac_sqtt.h (rgp_sqtt_marker_*).
MARKER_EVENT_TYPE = 15
MARKER_BASE = {
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


def _decode_barrier_end(d01, d02):
    caches, stalls = [], []
    if (d02 >> 3) & 1:
        caches.append("K")  # inval_sqK (scalar)
    if (d02 >> 1) & 1:
        caches.append("L0")  # inval_tcp (L0 vector)
    if (d02 >> 2) & 1:
        caches.append("I")  # inval_sqI (instruction)
    if (d02 >> 26) & 1:
        caches.append("L1")  # inval_gl1
    if (d02 >> 5) & 1:
        caches.append("L2")  # inval_tcc
    if (d02 >> 4) & 1:
        caches.append("L2flush")  # flush_tcc
    if (d01 >> 30) & 1:
        stalls.append("CS")  # cs_partial_flush
    if (d01 >> 28) & 1:
        stalls.append("VS")
    if (d01 >> 29) & 1:
        stalls.append("PS")
    if (d01 >> 27) & 1:
        stalls.append("waitEOP")
    if (d01 >> 31) & 1:
        stalls.append("pfpSyncMe")
    return {
        "caches_invalidated": ", ".join(caches) or "None",
        "frontend_sync": ", ".join(stalls) or "None",
        "cb_id": (d01 >> 7) & 0xFFFFF,
    }


def _parse_markers(raw_events, me, pipe):
    """Walk the reassembled marker dword stream for one queue. Returns
    (barriers, events) where events = [(ts, grid_workgroups)] for dispatches, or
    (None, None) if no markers (unpatched decoder)."""
    res = sorted(
        [
            e
            for e in raw_events
            if e.get("type") == MARKER_EVENT_TYPE
            and e.get("me") == me
            and e.get("pipe") == pipe
        ],
        key=lambda e: e.get("seq", 0),
    )
    if not res:
        return None, None
    dw = [(e["ts"], int(e["payload"]) & 0xFFFFFFFF) for e in res]
    barriers, events = [], []
    i = 0
    pend_start = None
    while i < len(dw):
        ts, d01 = dw[i]
        ident = d01 & 0xF
        ext = (d01 >> 4) & 0x7
        if ident == 0x0:  # event (per draw/dispatch)
            has_dims = (d01 >> 31) & 1
            size = 3 + (3 if has_dims else 0) + ext
            if has_dims and i + 5 < len(dw):
                gx, gy, gz = dw[i + 3][1], dw[i + 4][1], dw[i + 5][1]
                events.append((ts, gx * gy * gz))
            i += size
        elif ident == 0x3:  # barrier start
            pend_start = ts
            i += 2 + ext
        elif ident == 0x4:  # barrier end
            d02 = dw[i + 1][1] if i + 1 < len(dw) else 0
            f = _decode_barrier_end(d01, d02)
            start = pend_start if pend_start is not None else ts - 0.5
            span = max(ts - start, 0.0)
            barriers.append(
                {
                    "me": me,
                    "pipe": pipe,
                    "ts": start,
                    "dur": span,
                    "args": {
                        **f,
                        "barrier_type": "DRIVER",
                        "interval_us": round(span, 3),
                        "source": "RGP SQTT markers: cache/stall exact; "
                        "interval = BarrierEnd.ts - BarrierStart.ts",
                    },
                }
            )
            pend_start = None
            i += 2 + ext
        else:
            i += MARKER_BASE.get(ident, 1) + ext
    return barriers, events


def _queue_context(rdf):
    """Queue mapping from QueueInfo/QueueEvent chunks."""
    qinfo, qevents = [], []
    for c in rdf.by_id("QueueInfo"):
        d = rdf.data(c)
        if len(d) >= 32:
            hw = struct.unpack_from("<I", d, 24)[0]
            qinfo.append((hw & 0xFF, (hw >> 8) & 0xFF))
    for c in rdf.by_id("QueueEvent"):
        d = rdf.data(c)
        if len(d) >= 72:
            et, cbid, frame, qii, sub, api, cpu, g0, g1 = struct.unpack_from(
                "<IIQIIQQQQ", d, 16
            )
            if et == 0:
                qevents.append((qii, g0, g1))
    return qinfo, qevents


def build_trace(
    rgp_path: str,
    records_path: str | None = None,
    *,
    spm_enabled: bool = True,
    occupancy_enabled: bool = True,
    counters_xml: str | None = None,
    mem_roofline_gbps: float = 256.0,
    mem_bytes_per_req: int = 64,
) -> Trace:
    """Decode + symbolize a capture into a Trace with full RGP-parity features.

    :param rgp_path: path to the ``.rgp`` capture.
    :param records_path: optional records.json from the C++ decoder. If omitted,
        the pure-Python token decoder is used (currently ``NotImplementedError``).
    :param spm_enabled: attach SPM counter curves (memory/cache bandwidth).
    :param occupancy_enabled: attach the wavefront-occupancy band.
    """
    rdf = RdfFile.from_path(rgp_path)

    if records_path:
        rec = sqtt.load_records(records_path)
    else:
        rec = sqtt.decode_tokens(sqtt.extract_blobs(rdf))

    WAVE = rec.wave_size
    raw_disp = rec.dispatches
    raw_events = rec.events

    # duration-weighted PCs for symbolization
    pcw: dict[int, float] = {}
    for d in raw_disp:
        pc = int(d["pc"], 16) if isinstance(d["pc"], str) else int(d["pc"])
        pcw[pc] = pcw.get(pc, 0.0) + max(0.0, d.get("dur", 0.0))

    # --- symbolize: prefer deterministic CodeObject-hash <-> COLoadEvent-base pairing
    sym_note = ""
    hash2base = coload.load_hash_bases(rdf)
    names: dict[int, str] = {}
    if hash2base and pcw:
        co_hash_funcs = load_code_objects_hashed(rdf)
        names, stats = symbolize.name_map_exact(co_hash_funcs, hash2base, pcw)
        sym_note = (
            f"exact CO-hash pairing: {stats[2]}/{stats[3]} code objects, "
            f"{stats[0]}/{stats[1]} PCs, {stats[4]:.0f}% of dispatch-duration"
        )
    if not names and pcw:  # fallback: coverage-based
        co_funcs = load_code_objects(rdf)
        bases = coload.load_bases(rdf)
        names = symbolize.name_map(co_funcs, bases, pcw)
        sym_note = "coverage-based CO<->base assignment (no COLoadEvent hashes)"

    # --- per-queue: markers -> barriers + exact grids; refined durations
    queues = sorted({(int(d.get("me", 0)), int(d.get("pipe", 0))) for d in raw_disp})
    dispatches: list[Dispatch] = []
    barriers: list[Barrier] = []
    wave_sizes: set[int] = set()
    for me, pipe in queues:
        ds = sorted(
            [
                d
                for d in raw_disp
                if int(d.get("me", 0)) == me and int(d.get("pipe", 0)) == pipe
            ],
            key=lambda x: x.get("launch_ts", x.get("ts", 0.0)),
        )
        mbar, mevents = _parse_markers(raw_events, me, pipe)
        grids = [g for _, g in mevents] if mevents else []
        counts_exact = (mevents is not None) and (len(grids) == len(ds))
        launches = sorted(x.get("launch_ts", x.get("ts", 0.0)) for x in ds)
        for j, d in enumerate(ds):
            pc = int(d["pc"], 16) if isinstance(d["pc"], str) else int(d["pc"])
            occ_wf = int(d.get("wavefronts", 0))
            span = (
                max(0.0, d["dur"])
                if d.get("dur") is not None and d["dur"] >= 0
                else 0.0
            )
            lt = d.get("launch_ts", d.get("ts", 0.0))
            li = bisect.bisect_right(launches, lt)
            gap = max(0.0, (launches[li] - lt) if li < len(launches) else span)
            dur = min(span, gap) if span > 0 else gap
            if counts_exact:
                threads = grids[j] * int(d.get("wg", 0))
                # ceil: a partial workgroup still occupies a full wavefront, which
                # is how RGP reports launched waves (e.g. 32 threads -> 1 wave).
                wf = -(-threads // WAVE) if WAVE else 0
                wave_sizes.add(WAVE)
                csrc = "Event-marker grid (exact)"
            else:
                threads = occ_wf * WAVE
                wf = occ_wf
                csrc = "occupancy (approx; marker grid unavailable)"
            dispatches.append(
                Dispatch(
                    pc=pc,
                    name=names.get(pc, hex(pc)),
                    me=me,
                    pipe=pipe,
                    ts_us=float(d.get("ts", 0.0)),
                    launch_ts_us=float(lt),
                    dur_us=float(dur),
                    wavefronts=wf,
                    exp_waves=int(d.get("exp_waves", 0)),
                    threads=threads,
                    workgroup=int(d.get("wg", 0)),
                    vgpr=int(d.get("vgpr", 0)),
                    sgpr=int(d.get("sgpr", 0)),
                    lds=int(d.get("lds", 0)),
                    flags=int(d.get("flags", 0)),
                    wave_span_us=round(span, 3),
                    occupancy_wavefronts=occ_wf,
                    counts_source=csrc,
                    duration_note="min(occupancy wave-span, next-launch gap); approx",
                )
            )
        if mbar:
            for b in mbar:
                barriers.append(
                    Barrier(
                        name="CmdBarrier()",
                        me=b["me"],
                        pipe=b["pipe"],
                        ts_us=b["ts"],
                        dur_us=b["dur"],
                        args=b["args"],
                    )
                )

    events = [
        Event(
            me=int(e.get("me", 0)),
            pipe=int(e.get("pipe", 0)),
            ts_us=float(e.get("ts", 0.0)),
            type=int(e.get("type", 0)),
            payload=int(e.get("payload", 0)),
        )
        for e in raw_events
    ]

    occupancy = []
    for o in rec.occupancy:
        per_se = {int(k[2:]): v for k, v in o.items() if k.startswith("se")}
        occupancy.append(OccupancySample(ts_us=float(o.get("ts", 0.0)), per_se=per_se))

    # --- SPM counter curves (memory/cache bandwidth) ---------------------------
    curves: list[Curve] = []
    spm_note = ""
    mem_xs: list[float] = []  # full-resolution EA sample times (us)
    mem_req: list[float] = []  # EA ev55 request counts per sample (for attribution)
    if spm_enabled:
        try:
            session = spm.parse_spm(rgp_path)
        except Exception as e:  # noqa: BLE001 - decode is best-effort
            session = None
            spm_note = f"SPM decode skipped: {e}"
        if session:
            trace_t0 = min((float(d.get("ts", 0.0)) for d in raw_disp), default=0.0)
            mem_xs, mem_req = spm.ea_request_series(session, anchor_us=trace_t0)
            for cv in spm.spm_curves(
                session,
                rocprof_xml=counters_xml,
                anchor_us=trace_t0,
                mem_roofline_gbps=mem_roofline_gbps,
                ea_bytes_per_req=mem_bytes_per_req,
            ):
                curves.append(
                    Curve(
                        display=f"{cv['group']}: {cv['name']} [{cv['unit']}]",
                        key=cv["name"],
                        series=cv["series"],
                        group=cv["group"],
                        unit=cv["unit"],
                    )
                )
            ms = spm.mem_traffic_summary(
                session,
                ea_bytes_per_req=mem_bytes_per_req,
                roofline_gbps=mem_roofline_gbps,
            )
            if ms:
                spm_note = (
                    f"unified-mem (EA ev55): ~{ms['avg_gbps']:.1f} GB/s avg over "
                    f"{ms['window_ms']:.1f} ms (~{ms['pct_roofline']:.0f}% of "
                    f"{mem_roofline_gbps:.0f} GB/s roofline @{ms['bytes_per_req']}B/req)"
                )
        elif not spm_note:
            spm_note = "no SPM counters in capture"

    # --- decode kernel identity + theoretical occupancy ------------------------
    for d in dispatches:
        ki = kernelinfo.parse_kernel_name(d.name)
        d.library, d.family, d.dtype, d.tile, d.label = (
            ki.library,
            ki.family,
            ki.dtype,
            ki.tile,
            ki.label,
        )
        occ_pct, limiter, waves_cu = occ_calc.theoretical_occupancy(
            d.vgpr, d.sgpr, d.lds, d.workgroup, wave_size=WAVE
        )
        d.occ_pct, d.occ_limiter, d.occ_theoretical = occ_pct, limiter, waves_cu

    # --- idle gaps per queue (execution-window based, real dispatches only) -----
    for q in queues:
        ds = sorted(
            [d for d in dispatches if (d.me, d.pipe) == q and not d.is_artifact],
            key=lambda x: x.ts_us,
        )
        for i, d in enumerate(ds):
            end = d.ts_us + max(d.dur_us, 0.0)
            if i + 1 < len(ds):
                gap = max(0.0, ds[i + 1].ts_us - end)
                d.idle_after_us = round(gap, 3)
                ds[i + 1].gap_before_us = round(gap, 3)

    # --- opportunistic per-kernel SPM memory attribution (no-op if SPM absent) --
    # Sustained BW = (EA requests arriving in the dispatch window * bytes/req) / window
    # time. Summing raw counts (not averaging per-sample GB/s) is the physically-sound
    # estimate: over the window, arrivals ~= completions, so their time-average is the
    # sustained bandwidth -- same argument that makes the whole-capture average valid.
    # Requires >=3 samples in-window to be trustworthy; short kernels stay 0 -> undet.
    spm_present = bool(mem_xs)
    if mem_xs:
        for d in dispatches:
            if d.is_artifact or d.dur_us <= 0:
                continue
            lo = bisect.bisect_left(mem_xs, d.ts_us)
            hi = bisect.bisect_right(mem_xs, d.ts_us + d.dur_us)
            n = hi - lo
            if n >= 3:
                # span the samples actually cover (avoids bias from partial windows)
                span_us = max(mem_xs[hi - 1] - mem_xs[lo], d.dur_us)
                req = sum(mem_req[lo:hi])
                bytes_moved = req * mem_bytes_per_req
                d.mem_bytes = round(bytes_moved, 1)
                d.mem_gbps = round(bytes_moved / (span_us * 1e3), 2) if span_us else 0.0

    # --- bottleneck classification (degrades gracefully without SPM) ------------
    for d in dispatches:
        if d.is_artifact:
            continue
        bc, br = occ_calc.classify_bound(
            d,
            spm_present=spm_present,
            mem_reliable=(d.mem_gbps > 0),
            roofline_gbps=mem_roofline_gbps,
        )
        d.bound_class, d.bound_reason = bc, br

    # --- queue mapping metadata ------------------------------------------------
    qinfo, qevents = _queue_context(rdf)
    disp_per_queue = {
        q: sum(1 for d in dispatches if (d.me, d.pipe) == q) for q in queues
    }
    primary_queue = max(queues, key=lambda q: disp_per_queue[q]) if queues else None
    rt_freq = rec.rt_freq or 100_000_000
    submit_span_ms = (
        (
            (max(g1 for _, _, g1 in qevents) - min(g0 for _, g0, _ in qevents))
            / rt_freq
            * 1e3
        )
        if qevents
        else 0.0
    )
    n_compute = sum(1 for _qt, et in qinfo if et in (2, 3))
    queue_note = (
        f"{len(qevents)} cmdbuf submits; {n_compute} compute HW queue(s); "
        f"submit span {submit_span_ms:.1f} ms"
        if qevents
        else "no QueueEvent chunks"
    )

    meta = CaptureMeta(
        source=rgp_path,
        rt_freq_hz=rt_freq,
        wave_size=WAVE,
        shader_engines=len(rdf.by_id("SqttData")) or rec.shader_engines,
        queues=queues,
        extra={
            "chunks": rdf.counts(),
            "symbolization": sym_note,
            "spm": spm_note,
            "queue_mapping": queue_note,
            "primary_queue": (
                f"me{primary_queue[0]}/pipe{primary_queue[1]}"
                if primary_queue
                else None
            ),
            "wave_sizes": sorted(wave_sizes) or [WAVE],
            "clock": f"GPU timestamp {rt_freq / 1e6:.0f} MHz",
        },
    )

    return Trace(
        meta=meta,
        dispatches=dispatches,
        barriers=barriers,
        events=events,
        occupancy=occupancy,
        curves=curves,
    )
