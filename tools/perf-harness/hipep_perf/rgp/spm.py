#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""SPM (streaming performance monitor) chunks of an RGP capture.

SpmSession: header {pciId, flags, samplingInterval, numTimestamps, numCounters};
data = u64 timestamps. SpmCounterData (one per counter): header {pciId,
gpuBlock, blockInstance, eventIndex, sampleSize}; data = samples. The block is
Pal::GpuBlock (pal/inc/core/palPerfExperiment.h). Timestamps run on the GPU
timestamp clock, whose frequency AsicInfo carries (data +24).

Memory bandwidth comes from the EA block's event 55, the unified-memory request
counter on gfx1151, at 32 bytes per request: calibration kernels that read,
write and copy known byte counts measure 2.0x those counts at 64 B/request
(``calib rgp --counters``). Only windowed totals are physical: single samples
are bursty arbiter arrivals and can exceed the roofline.

The SPM buffer is a ring: when a capture outlasts it, it keeps the most recent
samples, so its first sample is not the trace's start. Samples are placed on the
SQTT time axis by the shared 100 MHz counter (``offset_us``).
"""

from __future__ import annotations

import os
import re
import struct

import numpy as np

from .container import Container

GPU_BLOCK = {
    i: n
    for i, n in enumerate(
        (
            "CPF",
            "IA",
            "VGT",
            "PA",
            "SC",
            "SPI",
            "SQ",
            "SX",
            "TA",
            "TD",
            "TCP",
            "TCC",
            "TCA",
            "DB",
            "CB",
            "GDS",
            "SRBM",
            "GRBM",
            "GRBMSE",
            "RLC",
            "DMA",
            "MC",
            "CPG",
            "CPC",
            "WD",
            "TCS",
            "ATC",
            "ATCL2",
            "MCVML2",
            "EA",
            "RPB",
            "RMI",
            "UMCCH",
            "GE",
            "GL1A",
            "GL1C",
            "GL1CG",
            "GL2A",
            "GL2C",
            "CHA",
            "CHC",
            "CHCG",
            "GUS",
            "GCR",
            "PH",
            "UTCL1",
            "GEDIST",
            "GESE",
            "DFMALL",
            "SQWGP",
            "PC",
        )
    )
}

SESSION_VERSION = 2  # pal gpuPerfExperimentTraceSource.h
COUNTER_VERSION = 2


def parse(c: Container) -> dict | None:
    """{interval, timestamps (np.uint64), counters [...], ts_freq_hz} or None."""
    session, counters, freq = None, [], None
    for ch in c.chunks:
        if ch.identifier == "AsicInfo":
            data = c.data(ch)
            if len(data) >= 32:
                f = struct.unpack_from("<Q", data, 24)[0]
                if 1e6 <= f <= 1e10:
                    freq = f
        elif ch.identifier == "SpmSession":
            hdr = c.header(ch)
            if len(hdr) < 20:
                continue
            if ch.version != SESSION_VERSION:
                print(
                    f"WARNING: SpmSession chunk version {ch.version} != expected "
                    f"{SESSION_VERSION}; header layout may differ."
                )
            _, _, interval, nts, ncnt = struct.unpack_from("<IIIII", hdr, 0)
            data = c.data(ch)
            ts = (
                np.frombuffer(data, dtype="<u8", count=nts)
                if len(data) >= nts * 8
                else np.empty(0, dtype="<u8")
            )
            session = {"interval": interval, "timestamps": ts, "numCounters": ncnt}
        elif ch.identifier == "SpmCounterData":
            hdr = c.header(ch)
            if len(hdr) < 20:
                continue
            if ch.version != COUNTER_VERSION:
                print(
                    f"WARNING: SpmCounterData chunk version {ch.version} != expected "
                    f"{COUNTER_VERSION}; header layout may differ."
                )
            _, block, inst, evt, size = struct.unpack_from("<IIIII", hdr, 0)
            data = c.data(ch)
            dt = {1: "<u1", 2: "<u2", 4: "<u4"}.get(size, "<u2")
            n = len(data) // max(size, 1)
            counters.append(
                {
                    "block": block,
                    "blockname": GPU_BLOCK.get(block, str(block)),
                    "instance": inst,
                    "event": evt,
                    "samples": np.frombuffer(data, dtype=dt, count=n).astype(np.int64),
                }
            )
    if session is None:
        return None
    session["counters"] = counters
    session["ts_freq_hz"] = freq
    return session


def _ticks_per_us(s: dict) -> float:
    f = s.get("ts_freq_hz")
    return f / 1e6 if f else 100.0


def offset_us(s: dict, sqtt_rt0: int) -> float:
    """Microseconds from the SQTT time origin to the first SPM sample.

    SQTT's realtime base agrees with SPM's timestamps only in the low 32 bits,
    so the difference is taken modulo 2^32 ticks (42.9 s at 100 MHz) and read as
    signed: SPM may start just before the first dispatch."""
    d = (int(s["timestamps"][0]) - int(sqtt_rt0)) & 0xFFFFFFFF
    if d >= 1 << 31:
        d -= 1 << 32
    return d / _ticks_per_us(s)


def _times_us(s: dict, anchor_us: float) -> np.ndarray:
    ts = s["timestamps"].astype(np.int64)
    return anchor_us + (ts - ts[0]).astype(np.float64) / _ticks_per_us(s)


def _aggregate(s: dict) -> dict[tuple[str, int], np.ndarray]:
    agg: dict[tuple[str, int], np.ndarray] = {}
    for ctr in s["counters"]:
        key = (ctr["blockname"], ctr["event"])
        if key in agg:
            n = min(len(agg[key]), len(ctr["samples"]))
            agg[key] = agg[key].copy()
            agg[key][:n] += ctr["samples"][:n]
        else:
            agg[key] = ctr["samples"].copy()
    return agg


def ea_requests(
    s: dict | None, anchor_us: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    """(sample times in us, EA ev55 request counts per sample) at full cadence."""
    if not s or not len(s["timestamps"]):
        return np.empty(0), np.empty(0, dtype=np.int64)
    agg = _aggregate(s)
    ea = agg.get(("EA", 55))
    if ea is None:
        return np.empty(0), np.empty(0, dtype=np.int64)
    return _times_us(s, anchor_us), ea


def mem_summary(
    s: dict | None, bytes_per_req: int = 32, roofline_gbps: float = 256.0
) -> dict | None:
    if not s or not len(s["timestamps"]):
        return None
    ts = s["timestamps"]
    dur_s = (int(ts[-1]) - int(ts[0])) / (s.get("ts_freq_hz") or 1e8)
    if dur_s <= 0:
        return None
    tot = sum(
        int(ctr["samples"].sum())
        for ctr in s["counters"]
        if ctr["blockname"] == "EA" and ctr["event"] == 55
    )
    if tot == 0:
        return None
    gbps = tot * bytes_per_req / dur_s / 1e9
    return {
        "total_req": tot,
        "window_ms": dur_s * 1e3,
        "avg_gbps": gbps,
        "pct_roofline": 100.0 * gbps / roofline_gbps if roofline_gbps else None,
        "bytes_per_req": bytes_per_req,
    }


def _xml_names(path: str | None) -> dict[tuple[str, int], str]:
    if not path or not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8", errors="replace") as fh:
        txt = fh.read()
    m = re.search(r"<gfx11>(.*?)</gfx11>", txt, re.DOTALL)
    return {
        (mm.group(2), int(mm.group(3))): mm.group(1)
        for mm in re.finditer(
            r'name="([^"]+)"\s+block=(\w+)\s+event=(\d+)', m.group(1) if m else ""
        )
    }


def _downsample(ts_us: np.ndarray, values: np.ndarray, nsamp: int, reduce: str = "max"):
    """nsamp (ts, value) points: bucket peak ('max', as RGP shows counters) or the
    bucket's middle-ranked value ('median', for derived bandwidth where single
    burst samples must not dominate)."""
    n = len(ts_us)
    if n == 0:
        return []
    if n <= nsamp:
        return [(float(t), _py(v)) for t, v in zip(ts_us, values)]
    out = []
    for k in range(nsamp):
        a = k * n // nsamp
        b = max(a + 1, (k + 1) * n // nsamp)
        chunk = values[a:b]
        if reduce == "max":
            v = chunk.max()
        elif reduce == "median":
            v = np.sort(chunk)[len(chunk) // 2]
        else:
            v = sum(chunk.tolist()) / len(chunk)
        out.append((float(ts_us[(a + b) // 2]), _py(v)))
    return out


def _py(v):
    return int(v) if isinstance(v, (np.integer, int)) else float(v)


def curves(
    s: dict | None,
    rocprof_xml: str | None = None,
    nsamp: int = 2000,
    anchor_us: float = 0.0,
    bytes_per_req: int = 32,
) -> list[dict]:
    """Counter curves for the timeline: {name, group, unit, series}."""
    if not s or not len(s["timestamps"]):
        return []
    if not s.get("ts_freq_hz"):
        print(
            "SPM: no AsicInfo gpuTimestampFrequency; assuming 100 MHz timestamp clock"
        )
    names = _xml_names(rocprof_xml)
    ts_us = _times_us(s, anchor_us)
    dt_us = (ts_us[-1] - ts_us[0]) / max(len(ts_us) - 1, 1)
    agg = _aggregate(s)
    out = []
    for (bn, evt), series in sorted(agg.items()):
        nm = names.get((bn, evt))
        if nm:
            out.append(
                {
                    "name": nm,
                    "group": f"SPM raw ({bn})",
                    "unit": "count/sample",
                    "series": _downsample(ts_us, series, nsamp),
                }
            )

    def rate(counts: np.ndarray, width: int) -> np.ndarray:
        r = np.zeros(len(ts_us))
        n = min(len(counts), len(ts_us))
        if dt_us > 0:
            r[:n] = (width * counts[:n]) / (dt_us * 1e3)
        return r

    ea = agg.get(("EA", 55))
    if ea is not None:
        out.append(
            {
                "name": "Unified-mem traffic (EA ev55, relative)",
                "group": "SPM raw (EA)",
                "unit": "req/sample",
                "series": _downsample(ts_us, ea, nsamp),
            }
        )
        out.append(
            {
                "name": f"Unified LPDDR5X BW (est @{bytes_per_req}B/req; avg trustworthy)",
                "group": "Memory (bytes)",
                "unit": "GB/s",
                "series": _downsample(ts_us, rate(ea, bytes_per_req), nsamp, "median"),
            }
        )
    rd = {
        sz: agg.get(("GL2C", ev))
        for sz, ev in ((32, 99), (64, 100), (96, 101), (128, 102))
    }
    if any(v is not None for v in rd.values()):
        total = np.zeros(len(ts_us), dtype=np.int64)
        for sz, v in rd.items():
            if v is not None:
                n = min(len(v), len(ts_us))
                total[:n] += sz * v[:n]
        fetch = total / (dt_us * 1e3) if dt_us > 0 else np.zeros(len(ts_us))
        out.append(
            {
                "name": "L2 miss read (GL2C->mem)",
                "group": "Memory (bytes)",
                "unit": "GB/s",
                "series": _downsample(ts_us, fetch, nsamp),
            }
        )
    wr64 = agg.get(("GL2C", 85))
    if wr64 is not None:
        out.append(
            {
                "name": "L2 write (GL2C->mem)",
                "group": "Memory (bytes)",
                "unit": "GB/s",
                "series": _downsample(ts_us, rate(wr64, 64), nsamp),
            }
        )
    return out
