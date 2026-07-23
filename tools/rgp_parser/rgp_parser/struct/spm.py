#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Decode the SPM (streaming perf counter) chunks from an RGP .rgp capture.

Self-contained SPM counter decode (no external dependency).

The capture stores SPM in the RDF trace-source format:
  - SpmSession chunk: header {pciId, flags, samplingInterval, numTimestamps,
    numSpmCounters} in the RDF *header* region; data region = u64 timestamps[].
  - SpmCounterData chunks (one per counter): header {pciId, gpuBlock, blockInstance,
    eventIndex, dataSize} in the header region; data region = dataSize-byte samples[].

Counters are raw hardware (block, instance, eventIndex). RGP selected its own raw
hardware event selects, which only partially match rocprofiler's named XML - so we
name what matches and additionally compute memory-bandwidth curves from the GL2C EA
request counters (which DO match). See README "SPM counters" for the honest scope.
"""

from __future__ import annotations

import os
import re
import struct

# Pal::GpuBlock enum -> name. The SpmCounterData chunk stores the block as Pal::GpuBlock
# (pal/inc/core/palPerfExperiment.h), which the SpmGpuBlock enum mirrors 1:1
# (pal/shared/inc/sqtt_file_format.h). Verified against both.
GPU_BLOCK = {
    0: "CPF",
    1: "IA",
    2: "VGT",
    3: "PA",
    4: "SC",
    5: "SPI",
    6: "SQ",
    7: "SX",
    8: "TA",
    9: "TD",
    10: "TCP",
    11: "TCC",
    12: "TCA",
    13: "DB",
    14: "CB",
    15: "GDS",
    16: "SRBM",
    17: "GRBM",
    18: "GRBMSE",
    19: "RLC",
    20: "DMA",
    21: "MC",
    22: "CPG",
    23: "CPC",
    24: "WD",
    25: "TCS",
    26: "ATC",
    27: "ATCL2",
    28: "MCVML2",
    29: "EA",
    30: "RPB",
    31: "RMI",
    32: "UMCCH",
    33: "GE",
    34: "GL1A",
    35: "GL1C",
    36: "GL1CG",
    37: "GL2A",
    38: "GL2C",
    39: "CHA",
    40: "CHC",
    41: "CHCG",
    42: "GUS",
    43: "GCR",
    44: "PH",
    45: "UTCL1",
    46: "GEDIST",
    47: "GESE",
    48: "DFMALL",
    49: "SQWGP",
    50: "PC",
}

# RDF chunk versions this parser was verified against (pal gpuPerfExperimentTraceSource.h:
# SpmSessionChunkVersion / SpmCounterDataChunkVersion). A different version may reorder
# header fields, so we warn rather than silently misparse.
SPM_SESSION_VER = 2
SPM_COUNTER_VER = 2


def _walk_chunks(blob):
    # AMD_RDF: 32-byte file header, then 64-byte IndexEntry[] (libamdrdf spec).
    # IndexEntry: chunkIdentifier[16], compression+reserved (4B), version@20 (u32),
    # then chunkHeader/Data offset+size (5x i64) @24.
    _, ver, _, ioff, isz = struct.unpack_from("<8sIIqq", blob, 0)
    for i in range(isz // 64):
        b = ioff + i * 64
        cid = (
            struct.unpack_from("<16s", blob, b)[0]
            .split(b"\x00", 1)[0]
            .decode("utf-8", "replace")
        )
        cver = struct.unpack_from("<I", blob, b + 20)[0]
        ho, hs, do, ds, us = struct.unpack_from("<qqqqq", blob, b + 24)
        yield cid, cver, blob[ho : ho + hs], blob[do : do + ds]


def parse_spm(rgp_path):
    """Return {interval, timestamps:[u64], counters:[{block,name,instance,event,samples}]}
    or None if the capture has no SPM."""
    blob = open(rgp_path, "rb").read()
    session = None
    counters = []
    ts_freq_hz = None
    for cid, cver, hdr, data in _walk_chunks(blob):
        if cid == "AsicInfo" and len(data) >= 32:
            # TraceChunk::AsicInfo (pal asicInfoTraceSource.h) lives in the chunk DATA
            # region (its header is empty): u32 pciId + 4B pad, then u64 shaderCoreClock,
            # u64 memoryClock, u64 gpuTimestampFrequency @24. Verified: @24 = 100 MHz.
            f = struct.unpack_from("<Q", data, 24)[0]
            if 1e6 <= f <= 1e10:
                ts_freq_hz = f
        if cid == "SpmSession" and len(hdr) >= 20:
            if cver != SPM_SESSION_VER:
                print(
                    "WARNING: SpmSession chunk version %d != expected %d; header layout "
                    "may differ (verified against v%d)."
                    % (cver, SPM_SESSION_VER, SPM_SESSION_VER)
                )
            pci, flags, interval, nts, ncnt = struct.unpack_from("<IIIII", hdr, 0)
            ts = (
                list(struct.unpack_from("<%dQ" % nts, data, 0))
                if len(data) >= nts * 8
                else []
            )
            session = {"interval": interval, "timestamps": ts, "numCounters": ncnt}
        elif cid == "SpmCounterData" and len(hdr) >= 20:
            if cver != SPM_COUNTER_VER:
                print(
                    "WARNING: SpmCounterData chunk version %d != expected %d; header layout "
                    "may differ (verified against v%d)."
                    % (cver, SPM_COUNTER_VER, SPM_COUNTER_VER)
                )
            pci, block, inst, evt, dsz = struct.unpack_from("<IIIII", hdr, 0)
            n = len(data) // max(dsz, 1)
            fmt = {1: "B", 2: "H", 4: "I"}.get(dsz, "H")
            samples = struct.unpack_from("<%d%s" % (n, fmt), data, 0)
            counters.append(
                {
                    "block": block,
                    "blockname": GPU_BLOCK.get(block, str(block)),
                    "instance": inst,
                    "event": evt,
                    "samples": samples,
                }
            )
    if session is None:
        return None
    session["counters"] = counters
    session["ts_freq_hz"] = ts_freq_hz  # from AsicInfo; None if chunk absent
    return session


def mem_traffic_summary(spm, ea_bytes_per_req=64, roofline_gbps=256.0):
    """Honest whole-capture average of the EA ev55 unified-memory traffic.

    Returns None if EA ev55 absent. The AVERAGE is physically sane (unlike per-sample
    peaks, which exceed the roofline - see spm_curves). Reported as an order-of-
    magnitude estimate, not a calibrated bandwidth."""
    if not spm or not spm.get("timestamps"):
        return None
    ts = spm["timestamps"]
    freq = (
        spm.get("ts_freq_hz") or 1e8
    )  # AsicInfo gpuTimestampFrequency; 100 MHz fallback
    dur_s = (ts[-1] - ts[0]) / freq
    if dur_s <= 0:
        return None
    tot = 0
    for c in spm["counters"]:
        if c["blockname"] == "EA" and c["event"] == 55:
            tot += sum(c["samples"])
    if tot == 0:
        return None
    gbps = tot * ea_bytes_per_req / dur_s / 1e9
    return {
        "total_req": tot,
        "window_ms": dur_s * 1e3,
        "avg_gbps": gbps,
        "pct_roofline": 100.0 * gbps / roofline_gbps if roofline_gbps else None,
        "bytes_per_req": ea_bytes_per_req,
    }


def ea_request_series(spm, anchor_us=0.0):
    """Full-resolution EA ev55 unified-memory *request counts* per sample, for
    per-kernel attribution: returns (ts_us, req_counts) at the raw SPM cadence.

    Returns raw request COUNTS (not per-sample GB/s) on purpose: the correct
    per-kernel bandwidth is total requests-in-window * bytes / window-time (the same
    conservation argument that makes the whole-capture average trustworthy). The
    per-sample GB/s value is meaningless here -- EA samples are bursty arbiter
    *arrivals* that queue and drain, so single samples routinely exceed the roofline.
    Returns ([], []) if EA ev55 is absent.
    """
    if not spm or not spm.get("timestamps"):
        return [], []
    ts = spm["timestamps"]
    freq = spm.get("ts_freq_hz")
    ticks_per_us = freq / 1e6 if freq else 100.0
    ts_us = [anchor_us + (t - ts[0]) / ticks_per_us for t in ts]
    ea = None
    for c in spm["counters"]:
        if c["blockname"] == "EA" and c["event"] == 55:
            s = c["samples"]
            if ea is None:
                ea = list(s)
            else:
                for i in range(min(len(ea), len(s))):
                    ea[i] += s[i]
    if ea is None:
        return [], []
    return ts_us, list(ea)


def _load_xml_names(rocprof_xml):
    """(blockName, eventId) -> counterName from basic_counters.xml <gfx11>."""
    if not rocprof_xml or not os.path.exists(rocprof_xml):
        return {}
    txt = open(rocprof_xml, encoding="utf-8", errors="replace").read()
    m = re.search(r"<gfx11>(.*?)</gfx11>", txt, re.S)
    seg = m.group(1) if m else ""
    out = {}
    for mm in re.finditer(r'name="([^"]+)"\s+block=(\w+)\s+event=(\d+)', seg):
        out[(mm.group(2), int(mm.group(3)))] = mm.group(1)
    return out


def _median(xs):
    s = sorted(xs)
    m = len(s)
    return s[m // 2] if m else 0


def _downsample(ts_us, values, nsamp, reduce="max"):
    """Bucket a series into nsamp (ts_us, value) points.

    reduce="max"  -> bucket PEAK: matches RGP's curve display and preserves bursty
                     raw-counter activity (SPM counters are near-zero with spikes).
    reduce="median" -> robust sustained value: use for derived bandwidth, where rare
                     single-sample spikes are decode artifacts (a 90k-req sample would
                     imply TB/s) that must not dominate the GB/s reading.
    reduce="mean" -> bucket average.
    """
    if not ts_us:
        return []
    red = {"max": max, "median": _median, "mean": lambda xs: sum(xs) / len(xs)}[reduce]
    n = len(ts_us)
    if n <= nsamp:
        return list(zip(ts_us, values))
    out = []
    for k in range(nsamp):
        a = k * n // nsamp
        b = max(a + 1, (k + 1) * n // nsamp)
        out.append((ts_us[(a + b) // 2], red(values[a:b])))
    return out


def spm_curves(
    spm,
    rocprof_xml=None,
    nsamp=2000,
    anchor_us=0.0,
    mem_roofline_gbps=256.0,
    ea_bytes_per_req=64,
):
    """Turn parsed SPM into a list of curves for the Chrome trace.

    Each curve: {name, group, unit, series:[(ts_us, value)]}. Time base: SPM
    timestamps are in the GPU timestamp-clock domain (AsicInfo.gpuTimestampFrequency);
    ts_us = (ts - ts[0]) / (freq/1e6) + anchor.
    NOTE: SPM/SQTT co-start is assumed for the anchor (see README caveats).

    mem_roofline_gbps: LPDDR5X unified-memory peak for the % roofline curve (Strix
      Halo ~256 GB/s). ea_bytes_per_req: assumed bytes per EA memory request (see the
      EA-block note below - the exact unit is NOT verifiable from open counter defs).
    """
    if not spm or not spm["timestamps"]:
        return []
    names = _load_xml_names(rocprof_xml)
    ts = spm["timestamps"]
    # SPM timestamps are in the GPU timestamp-clock domain. Read the real frequency from
    # the capture's AsicInfo chunk (gpuTimestampFrequency, Hz) instead of assuming; fall
    # back to 100 MHz only if AsicInfo is absent. (Sanity: 4096-sclk interval @~2.7 GHz
    # = 1.52 us -> dtick ~152 ticks, i.e. ~100 ticks/us -> 100 MHz, which matches.)
    freq = spm.get("ts_freq_hz")
    if freq:
        ticks_per_us = freq / 1e6
    else:
        ticks_per_us = 100.0
        print(
            "SPM: no AsicInfo gpuTimestampFrequency; assuming 100 MHz timestamp clock"
        )
    ts_us = [anchor_us + (t - ts[0]) / ticks_per_us for t in ts]
    dt_us = (ts_us[-1] - ts_us[0]) / max(len(ts_us) - 1, 1)  # per-sample interval (us)

    # aggregate counters by (blockname, event): sum across instances per sample
    agg = {}
    for c in spm["counters"]:
        key = (c["blockname"], c["event"])
        s = agg.get(key)
        if s is None:
            agg[key] = list(c["samples"])
        else:
            for i in range(min(len(s), len(c["samples"]))):
                s[i] += c["samples"][i]

    curves = []
    # 1) named raw counters (whatever matches rocprofiler gfx11 XML)
    for (bn, evt), series in sorted(agg.items()):
        nm = names.get((bn, evt))
        if nm:
            curves.append(
                {
                    "name": nm,
                    "group": "SPM raw (%s)" % bn,
                    "unit": "count/sample",
                    "series": _downsample(ts_us, series, nsamp),
                }
            )

    def get(bn, evt):
        return agg.get((bn, evt))

    # 2) UNIFIED-MEMORY (LPDDR5X) bandwidth from the EA / GCEA block.
    #    On this APU (gfx1151) there is no discrete VRAM: all traffic that misses the
    #    GPU caches goes to shared LPDDR5X through the EA (Efficiency Arbiter / GCEA,
    #    SPM block 29). Event 55 carries that memory-request activity - it is by far
    #    the dominant memory counter in the capture (~184M req vs ~0.6 MB of GL2C EA
    #    read requests, which almost all hit in L2 here).
    #
    #    WHAT'S SOLID vs ASSUMED:
    #      * The whole-capture AVERAGE at 64 B/req ~= 65 GB/s (~26% of 256 GB/s) is a
    #        legitimate sustained-bandwidth estimate: over the capture, requests arriving
    #        at the arbiter must equal completions, so their time-average = sustained BW.
    #      * Instantaneous peaks (single samples ~55x the mean -> >roofline) are arbiter
    #        *arrival* bursts that queue and drain later, NOT sustained bandwidth. SPM
    #        timestamps are uniform (~1.52 us) so this is not a dt artifact. Hence the
    #        per-sample GB/s curve is median-downsampled and treated as relative intensity.
    #      * The 64 B/req unit is an ASSUMPTION (standard DRAM burst): rocprofiler has no
    #        gfx11 GCEA table and RGP's raw->name map is closed, so the exact byte size is
    #        not provable from open sources. Cross-check the ~65 GB/s average against RGP's
    #        own bandwidth readout to confirm; override via ea_bytes_per_req if calibrated.
    ea = get("EA", 55)
    if ea is not None:
        curves.append(
            {
                "name": "Unified-mem traffic (EA ev55, relative)",
                "group": "SPM raw (EA)",
                "unit": "req/sample",
                "series": _downsample(ts_us, ea, nsamp),
            }
        )
        bw = [
            (ea_bytes_per_req * ea[i]) / (dt_us * 1e3)
            if dt_us > 0 and i < len(ea)
            else 0.0
            for i in range(len(ts_us))
        ]
        # median downsample: rare burst samples (arbiter arrivals, not sustained BW)
        # must not dominate the reading.
        curves.append(
            {
                "name": "Unified LPDDR5X BW (est @%dB/req; avg trustworthy)"
                % ea_bytes_per_req,
                "group": "Memory (bytes)",
                "unit": "GB/s",
                "series": _downsample(ts_us, bw, nsamp, reduce="median"),
            }
        )

    # 3) L2->memory read/write from GL2C EA request counters (these DO match the XML).
    #    On an iGPU these are the *cache-miss* path to LPDDR5X; here they are small
    #    (the working set largely hits in L2), so they are a secondary detail, not the
    #    headline bandwidth. RDREQ_{32,64,96,128}B = reads of that size; WRREQ_64B writes.
    rd = {
        sz: get("GL2C", ev) for sz, ev in ((32, 99), (64, 100), (96, 101), (128, 102))
    }
    if any(v is not None for v in rd.values()):
        fetch = []
        for i in range(len(ts_us)):
            b = sum(sz * (rd[sz][i] if rd[sz] and i < len(rd[sz]) else 0) for sz in rd)
            fetch.append(b / (dt_us * 1e3) if dt_us > 0 else 0.0)  # bytes/us/1e3 = GB/s
        curves.append(
            {
                "name": "L2 miss read (GL2C->mem)",
                "group": "Memory (bytes)",
                "unit": "GB/s",
                "series": _downsample(ts_us, fetch, nsamp),
            }
        )
    wr64 = get("GL2C", 85)
    if wr64 is not None:
        wr = [
            (64 * wr64[i]) / (dt_us * 1e3) if dt_us > 0 and i < len(wr64) else 0.0
            for i in range(len(ts_us))
        ]
        curves.append(
            {
                "name": "L2 write (GL2C->mem)",
                "group": "Memory (bytes)",
                "unit": "GB/s",
                "series": _downsample(ts_us, wr, nsamp),
            }
        )
    return curves
