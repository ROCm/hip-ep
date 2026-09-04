#!/usr/bin/env python3
"""Diff the in-kernel autotuner's online winners against the shipped LUT.

    python lut_diff.py <console_log> [--lut lut/<arch>.json]

Reads the tuner-decision lines emitted under HIPDNN_MATMUL_AUTOTUNE_LOG=1 (run
with HIPDNN_MATMUL_AUTOTUNE_MODE=online) from a CI console log, resolves what
the offline table *would* have served for the same (phase, N, K, M, gs, zp)
using the exact runtime nearest-neighbour rule, and prints every shape where
the table's config differs from what the online sweep actually picked.

A DIFF means the shipped table is serving a suboptimal config for that shape:
online is what the current kernel measures as fastest.

Caveats (fields the winner log does not carry):
  * WMMA lines omit the row stride, so Padded vs Arrival is inferred from
    shouldPadRow(K) (mirrors lib/Runtime/real/matmul_nbits.cpp; assumes the
    no-bias prefill path, which is the padrow case). Both groups are checked
    and any disagreement is flagged.
  * GEMV (decode) lines omit zero_point; both zp groups are resolved and a
    disagreement is flagged.
WMMA (prefill, the TTFT driver) carries every key and is fully determined.
"""
from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEFAULT_LUT = ROOT / "lut" / "gfx1151.json"

# --- winner-line grammar (mirrors the printf formats in matmul_nbits_kernel.hip)
RE_WMMA = re.compile(
    r"WMMA autotune: M=(\d+) N=(\d+) K=(\d+) gs=(\d+) zp=(yes|no) -> "
    r"best config\[(\d+)\] (fused|dq\+gemm) "
    r"bm=(\d+) bn=(\d+) sw=(\d+) wt=(\d+)x(\d+) bk=(\d+) \(([\d.]+) ms/iter\)")
RE_GEMV = re.compile(
    r"(?<!-)GEMV autotune: M=(\d+) N=(\d+) K=(\d+) bs=(\d+) col_major=(\d+) -> "
    r"best config\[(\d+)\] threads=(\d+) tile_n=(\d+) \(([\d.]+) ms/iter\)")
RE_DP4A = re.compile(
    r"GEMV-dp4a autotune: N=(\d+) K=(\d+) bs=(\d+) zp=(yes|no) -> "
    r"best config\[(\d+)\] threads=(\d+) tile_n=(\d+) \(([\d.]+) ms/iter\)")


def should_pad_row(K: int) -> bool:
    """Mirror shouldPadRow() in lib/Runtime/real/matmul_nbits.cpp."""
    stride = K // 2
    if stride % 128 != 0:
        return False
    sl, b = stride // 128, 64
    while b:
        sl, b = b, sl % b
    return sl >= 8


class Lut:
    """The shipped table + the runtime's nearest-neighbour resolver."""

    def __init__(self, path: Path):
        import json
        d = json.loads(path.read_text())
        self.wm = d["weight_m"]
        self.wn = d["weight_n"]
        self.wk = d["weight_k"]
        self.configs = d["configs"]
        # group points by (phase, bits, group_size, zero_point, row_stride)
        self.groups: dict[tuple, list] = {}
        for p in d["points"]:
            key = (p["phase"], p["bits"], p["group_size"],
                   p["zero_point"], p["row_stride"])
            self.groups.setdefault(key, []).append(p)

    def resolve(self, phase, gs_str, zp_str, stride, m, n, k):
        """Nearest point by weighted log2 L2, returning its config dict."""
        key = (phase, "B4", gs_str, zp_str, stride)
        pts = self.groups.get(key)
        if not pts:
            return None, None, None
        qm, qn, qk = math.log2(max(m, 1)), math.log2(n), math.log2(k)

        def d2(p):
            dm = self.wm * (qm - math.log2(p["m"]))
            dn = self.wn * (qn - math.log2(p["n"]))
            dk = self.wk * (qk - math.log2(p["k"]))
            return dm * dm + dn * dn + dk * dk

        best = min(pts, key=d2)
        exact = best["m"] == m and best["n"] == n and best["k"] == k
        return self.configs[best["config"]], math.sqrt(d2(best)), exact

    @staticmethod
    def wmma_geo(cfg):
        return (cfg["fused"] != 0, cfg["bm16"] * 16, cfg["bn16"] * 16,
                cfg["swizzle"], cfg["wt_m"], cfg["wt_n"], cfg["bk"])

    @staticmethod
    def gemv_geo(cfg):
        return (cfg["threads"], cfg["tile_n"])


def parse(log_path: Path):
    """Yield dedup'd winner records from the console log."""
    seen = set()
    for line in log_path.read_text(errors="ignore").splitlines():
        m = RE_WMMA.search(line)
        if m:
            n, N, K = int(m[1]), int(m[2]), int(m[3])
            key = ("Prefill", n, N, K, int(m[4]), m[5])
            if key in seen:
                continue
            seen.add(key)
            yield {
                "phase": "Prefill", "m": n, "n": N, "k": K,
                "gs": int(m[4]), "zp": m[5] == "yes",
                "geo": (m[7] == "fused", int(m[8]), int(m[9]),
                        int(m[10]), int(m[11]), int(m[12]), int(m[13])),
                "ms": float(m[14]),
            }
            continue
        m = RE_DP4A.search(line)
        if m:
            N, K = int(m[1]), int(m[2])
            key = ("DecodeDp4a", 1, N, K, int(m[3]), m[4])
            if key in seen:
                continue
            seen.add(key)
            yield {
                "phase": "DecodeDp4a", "m": 1, "n": N, "k": K,
                "gs": int(m[3]), "zp": m[4] == "yes",
                "geo": (int(m[6]), int(m[7])), "ms": float(m[8]),
            }
            continue
        m = RE_GEMV.search(line)
        if m:
            M, N, K = int(m[1]), int(m[2]), int(m[3])
            key = ("Decode", M, N, K, int(m[4]))
            if key in seen:
                continue
            seen.add(key)
            yield {
                "phase": "Decode", "m": M, "n": N, "k": K,
                "gs": int(m[4]), "zp": None,   # not in the GEMV log
                "geo": (int(m[7]), int(m[8])), "ms": float(m[9]),
            }


def gs_class(gs: int) -> str:
    return {16: "G16", 32: "G32", 64: "G64", 128: "G128",
            256: "G256", 512: "G512"}.get(gs, "GAny")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("log", type=Path, help="CI console log with autotune winners")
    ap.add_argument("--lut", type=Path, default=DEFAULT_LUT)
    ap.add_argument("--all", action="store_true",
                    help="also print shapes where LUT == online")
    args = ap.parse_args()

    lut = Lut(args.lut)
    recs = list(parse(args.log))
    if not recs:
        print("no autotune winner lines found -- was HIPDNN_MATMUL_AUTOTUNE_LOG=1 "
              "set (correct name), stderr captured, and MODE=online?")
        return 1

    n_diff = n_match = n_nolut = 0
    rows = []
    for r in recs:
        gsc = gs_class(r["gs"])
        if r["phase"] == "Prefill":
            zp = "Asymmetric" if r["zp"] else "Symmetric"
            stride = "Padded" if should_pad_row(r["k"]) else "Arrival"
            cfg, dist, exact = lut.resolve("Prefill", gsc, zp, stride,
                                           r["m"], r["n"], r["k"])
            lut_geo = lut.wmma_geo(cfg) if cfg else None
            online_geo = r["geo"]
            fmt = lambda g: (f"{'fused' if g[0] else 'dq+gemm'} "
                             f"bm{g[1]} bn{g[2]} sw{g[3]} wt{g[4]}x{g[5]} bk{g[6]}"
                             if g else "(no LUT group)")
        else:
            zp = ("Asymmetric" if r["zp"] else "Symmetric") if r["zp"] is not None else None
            stride = "Any"
            if zp is None:
                # GEMV log lacks zp: resolve both, agree or flag
                ca, *_ = lut.resolve(r["phase"], gsc, "Asymmetric", stride,
                                     r["m"], r["n"], r["k"])
                cs, *_ = lut.resolve(r["phase"], gsc, "Symmetric", stride,
                                     r["m"], r["n"], r["k"])
                geos = {lut.gemv_geo(c) for c in (ca, cs) if c}
                lut_geo = next(iter(geos)) if len(geos) == 1 else None
                cfg = ca or cs
                dist = None
                exact = False
            else:
                cfg, dist, exact = lut.resolve(r["phase"], gsc, zp, stride,
                                               r["m"], r["n"], r["k"])
                lut_geo = lut.gemv_geo(cfg) if cfg else None
            online_geo = r["geo"]
            fmt = lambda g: (f"threads{g[0]} tile_n{g[1]}" if g else "(ambiguous/none)")

        if lut_geo is None:
            n_nolut += 1
            status = "NO-LUT"
        elif tuple(lut_geo) == tuple(online_geo):
            n_match += 1
            status = "match"
        else:
            n_diff += 1
            status = "DIFF"

        if status != "match" or args.all:
            zpc = "asym" if r["zp"] else ("sym" if r["zp"] is not None else "?")
            rows.append((status, r["phase"], r["n"], r["k"], r["m"], gsc, zpc,
                         stride, fmt(lut_geo), fmt(online_geo), r["ms"]))

    # sort: DIFFs first, then by shape
    rows.sort(key=lambda x: (x[0] != "DIFF", x[1], x[2], x[3], x[4]))
    hdr = ("status", "phase", "N", "K", "M", "gs", "zp", "stride",
           "LUT config", "online winner", "online ms")
    print(f"{hdr[0]:6} {hdr[1]:10} {hdr[2]:>6} {hdr[3]:>6} {hdr[4]:>6} "
          f"{hdr[5]:5} {hdr[6]:4} {hdr[7]:8} {hdr[8]:42} {hdr[9]:42} {hdr[10]:>9}")
    print("-" * 150)
    for row in rows:
        print(f"{row[0]:6} {row[1]:10} {row[2]:>6} {row[3]:>6} {row[4]:>6} "
              f"{row[5]:5} {row[6]:4} {row[7]:8} {row[8]:42} {row[9]:42} {row[10]:>9.4f}")

    total = n_diff + n_match + n_nolut
    print("-" * 150)
    print(f"{total} unique shapes: {n_diff} DIFF, {n_match} match, {n_nolut} no-LUT")
    if n_diff:
        print(f"=> the shipped table serves a suboptimal config on {n_diff} shape(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
