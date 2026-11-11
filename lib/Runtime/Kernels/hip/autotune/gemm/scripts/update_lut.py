#!/usr/bin/env python3
"""Gemm autotune LUT pipeline: measure -> build -> compile.

    python update_lut.py measure   # drive the GPU sweep, write data/*.log
    python update_lut.py build     # logs -> lut/<arch>.json
    python update_lut.py compile   # json -> lut/<arch>.fb (needs flatc)
    python update_lut.py all

Schema v2 (plan.md §2): points are keyed on (phase, act_dtype, wts_dtype,
trans_b) instead of v1's `type_bytes`, and cover four phases (Wmma / GemvNt /
GemvNn / TiledFma) across three real dtypes (f16, bf16, f32) instead of one
phase-pair keyed on element size. See plan.md §1 D2/D3/D4 for why the old key
could not express this and autotune/common/dtype.py for the shared vocabulary.

The winners come from the in-kernel autotuner, read off its debug log
(HIPDNN_EP_DEBUG=1). Nothing here re-implements "which config is fastest":
that judgement is made once, in the kernel, so the table can never drift from
what production actually runs (mirrors matmul_nbits/scripts/update_lut.py).
The one thing that *does* live here, not in the kernel, is the offline
tie-break (plan.md §4.4): a config within TIE_BAND of the fastest is broken
toward the larger warp tile, not toward whichever happened to run first.

Distance weights are fixed at 1.0 this round (plan.md §1 D8: fitting
per-group weights needs the schema to grow a [GemmGroupWeights] table first,
which is a future, schema_version-free addition). `fallbacks` is always
empty this round on purpose (plan.md §1 D5 note): an empty fallback table
plus exact-match-only groups is what gives a dtype with no measured points a
clean MISS back to the static default, instead of silently borrowing another
dtype's point.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                       # autotune/gemm
COMMON = ROOT.parent / "common"          # autotune/common (shared dtype vocab)
FBS = ROOT / "gemm_autotune.fbs"
LUT_DIR = ROOT / "lut"
DATA_DIR = ROOT / "scripts" / "data"
SHAPES = ROOT / "shapes" / "gemm_shapes.csv"

sys.path.insert(0, str(COMMON))
from dtype import NAME_TO_VALUE  # noqa: E402  (see autotune/common/dtype.py)

SCHEMA_VERSION = 3
KERNEL_ABI = "gemm-v2"

PHASES = ("Wmma", "GemvNt", "GemvNn", "TiledFma")
PHASE_TO_KIND = {"Wmma": "Wmma", "GemvNt": "Gemv", "GemvNn": "Gemv",
                 "TiledFma": "TiledFma"}


def dtype_enum_name(tok: str) -> str:
    """'f16' -> 'F16' etc (every NAME_TO_VALUE key upper-cases to its fbs
    enum member name), validated against the shared vocabulary so a typo in
    a log line fails loudly instead of writing an 'Any'-keyed point that
    silently never matches a real query (D7's spirit, ported from the old
    CSV-row validation to this log-line-based pipeline)."""
    if tok.lower() not in NAME_TO_VALUE:
        raise ValueError(f"unknown dtype token {tok!r}")
    return tok.upper()


# ---------------------------------------------------------------------------
# measure
# ---------------------------------------------------------------------------

def cmd_measure(args) -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"_{args.tag}" if args.tag else ""
    log = DATA_DIR / f"{args.arch}{tag}_sweep.log"

    env = dict(os.environ, HIPDNN_EP_DEBUG="1",
               HIPDNN_GEMM_AUTOTUNE_MODE="online")
    cmd = [args.sweep, "--shapes", str(args.shapes)]
    if args.m:
        cmd += ["--m", args.m]
    if args.dtypes:
        cmd += ["--dtypes", args.dtypes]
    if args.phases:
        cmd += ["--phases", args.phases]
    if args.limit:
        cmd += ["--limit", str(args.limit)]
    cmd += args.sweep_args
    print("[measure] " + " ".join(cmd))
    # Real file handle, not a shell redirect -- PowerShell wraps/truncates a
    # native command's stderr at console width (knowledge/hosts.md), which
    # would silently corrupt the autotune lines this pipeline parses.
    with open(log, "w", encoding="utf-8", errors="replace") as f:
        proc = subprocess.run(cmd, env=env, stderr=f, stdout=subprocess.DEVNULL)
    if proc.returncode != 0:
        print(f"[measure] sweep exited {proc.returncode}", file=sys.stderr)
        return proc.returncode
    print(f"[measure] -> {log} ({log.stat().st_size} bytes)")
    return 0


# ---------------------------------------------------------------------------
# parsing
# ---------------------------------------------------------------------------

SHAPE_RE = re.compile(
    r"^#SHAPE phase=(\w+) act=(\w+) wts=(\w+) out=(\w+) tb=(\d+) M=(\d+) "
    r"N=(\d+) K=(\d+)")

# Per-candidate lines. Each carries the FULL geometry, which is how a
# candidate's identity survives independent of its table index (plan.md §2.2:
# the .fb stores geometry, not index, precisely so a kWmma[] edit cannot
# silently point an old table at the wrong tile).
WMMA_CAND_RE = re.compile(
    r"^\[gemm\] wmma cfg\[(\d+)\] (\d+)x(\d+) wt=(\d+)x(\d+) sw=(\d+) "
    r"sk=(\d+) bk=(\d+) : ([\d.]+) ms")
# Retune (finalist re-measure) carries only cid + ms; geometry for that cid
# was already captured from the coarse pass above in the same shape block.
WMMA_RETUNE_RE = re.compile(
    r"^\[gemm\] wmma retune cfg\[(\d+)\] : ([\d.]+) ms \(peak\)")
WMMA_WIN_RE = re.compile(
    r"^\[gemm\] wmma autotune M=(\d+) N=(\d+) K=(\d+) tb=(\d+) -> "
    r"cfg\[(\d+)\] (\d+)x(\d+) wt=(\d+)x(\d+) sw=(\d+) sk=(\d+) bk=(\d+)")

GEMVNT_CAND_RE = re.compile(
    r"^\[gemm\] gemv-nt cfg\[(\d+)\] th=(\d+) tn=(\d+) : ([\d.]+) ms \(peak\)")
GEMVNT_WIN_RE = re.compile(
    r"^\[gemm\] gemv-nt autotune M=(\d+) N=(\d+) K=(\d+) -> cfg\[(\d+)\] "
    r"th=(\d+) tn=(\d+)")

GEMVNN_CAND_RE = re.compile(
    r"^\[gemm\] gemv-nn cfg\[(\d+)\] th=(\d+) : ([\d.]+) ms \(peak\)")
GEMVNN_WIN_RE = re.compile(
    r"^\[gemm\] gemv-nn autotune M=(\d+) N=(\d+) K=(\d+) -> cfg\[(\d+)\] "
    r"th=(\d+)")

TILEDFMA_CAND_RE = re.compile(
    r"^\[gemm\] tiledfma cfg\[(\d+)\] (\d+)x(\d+) bk=(\d+) tm=(\d+) tn=(\d+) "
    r"th=(\d+) : ([\d.]+) ms \(peak\)")
TILEDFMA_WIN_RE = re.compile(
    r"^\[gemm\] tiledfma autotune M=(\d+) N=(\d+) K=(\d+) ta=(\d+) tb=(\d+) "
    r"-> cfg\[(\d+)\] (\d+)x(\d+) bk=(\d+) tm=(\d+) tn=(\d+) th=(\d+)")


def _wmma_geom(bm, bn, wm, wn, sw, sk, bk):
    return ("Wmma", int(bm), int(bn), int(wm), int(wn), int(sw), int(sk),
            int(bk))


def _tiledfma_geom(bm, bn, bk, tm, tn, th):
    return ("TiledFma", int(bm), int(bn), int(bk), int(tm), int(tn), int(th))


def _gemv_geom(kind, th, tn=0):
    return (kind, int(th), int(tn))


def parse_log(path: Path):
    """Yield (shape_dict, winner_geom, {geom: ms}) per measured point.

    shape_dict has phase/act/wts/out/tb/m/n/k. `times` is every candidate
    geometry this shape actually got a timing for -- what makes the offline
    tie-break and margin computation possible instead of just trusting the
    kernel's single logged winner. The WMMA retune stage logs only cid + ms
    (no geometry), so `cid_geom` recovers each cid's geometry from the coarse
    pass earlier in the same shape block.
    """
    cur = None
    times: dict = {}
    cid_geom: dict = {}
    n_orphan = 0
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = SHAPE_RE.match(line)
            if m:
                if cur is not None:
                    n_orphan += 1
                cur = {"phase": m.group(1), "act": m.group(2),
                       "wts": m.group(3), "out": m.group(4),
                       "tb": int(m.group(5)), "m": int(m.group(6)),
                       "n": int(m.group(7)), "k": int(m.group(8))}
                times, cid_geom = {}, {}
                continue
            if cur is None:
                continue
            phase = cur["phase"]

            if phase == "Wmma":
                c = WMMA_CAND_RE.match(line)
                if c:
                    cid = int(c.group(1))
                    geom = _wmma_geom(*c.groups()[1:8])
                    cid_geom[cid] = geom
                    times[geom] = float(c.group(9))
                    continue
                r = WMMA_RETUNE_RE.match(line)
                if r:
                    cid = int(r.group(1))
                    geom = cid_geom.get(cid)
                    if geom is not None:
                        times[geom] = float(r.group(2))
                    continue
                w = WMMA_WIN_RE.match(line)
                if w:
                    geom = _wmma_geom(*w.groups()[4:11])
                    yield cur, geom, times
                    cur, times, cid_geom = None, {}, {}
                continue

            if phase == "GemvNt":
                c = GEMVNT_CAND_RE.match(line)
                if c:
                    times[_gemv_geom("Gemv", c.group(2), c.group(3))] = \
                        float(c.group(4))
                    continue
                w = GEMVNT_WIN_RE.match(line)
                if w:
                    geom = _gemv_geom("Gemv", w.group(5), w.group(6))
                    yield cur, geom, times
                    cur, times, cid_geom = None, {}, {}
                continue

            if phase == "GemvNn":
                c = GEMVNN_CAND_RE.match(line)
                if c:
                    times[_gemv_geom("Gemv", c.group(2))] = float(c.group(3))
                    continue
                w = GEMVNN_WIN_RE.match(line)
                if w:
                    geom = _gemv_geom("Gemv", w.group(5))
                    yield cur, geom, times
                    cur, times, cid_geom = None, {}, {}
                continue

            if phase == "TiledFma":
                c = TILEDFMA_CAND_RE.match(line)
                if c:
                    geom = _tiledfma_geom(*c.groups()[1:7])
                    times[geom] = float(c.group(8))
                    continue
                w = TILEDFMA_WIN_RE.match(line)
                if w:
                    geom = _tiledfma_geom(*w.groups()[5:11])
                    yield cur, geom, times
                    cur, times, cid_geom = None, {}, {}
                continue
    if n_orphan:
        print(f"[build] {n_orphan} shape markers had no autotune winner line "
              f"in {path.name}", file=sys.stderr)


def load_readings(args):
    data_dir = Path(args.data_dir) if args.data_dir else DATA_DIR
    logs = sorted(data_dir.glob(f"{args.arch}*_sweep.log"))
    if not logs:
        print(f"[build] no sweep logs in {data_dir}; run measure first",
              file=sys.stderr)
        return None
    readings = []
    for log in logs:
        got = list(parse_log(log))
        print(f"[build] {log.name}: {len(got)} readings")
        readings += got
    if not readings:
        print("[build] sweep logs have no usable readings", file=sys.stderr)
        return None
    print(f"[build] {len(readings)} readings total")
    return readings


# ---------------------------------------------------------------------------
# collate: readings -> one winner per point, offline tie-break, margin
# ---------------------------------------------------------------------------

def point_key(shape: dict):
    return ((shape["phase"], shape["act"], shape["wts"], shape["tb"]),
            shape["m"], shape["n"], shape["k"])


# Near-tie band (plan.md §4.4): configs within this factor of the fastest are
# measurement-equal, broken toward the larger warp tile instead of raw argmin
# noise deciding it.
TIE_BAND = 1.03


def _tiebreak_rank(geom):
    if geom[0] == "Wmma":
        _, bm, bn, wm, wn, _sw, _sk, _bk = geom
        return (wm * wn, bm * bn)
    if geom[0] == "TiledFma":
        _, bm, bn, _bk, tm, tn, _th = geom
        return (tm * tn, bm * bn)
    return (0, 0)  # Gemv: no tile concept, fastest wins


def select_winner(times: dict):
    fastest = min(times.values())
    band = fastest * TIE_BAND
    cands = [g for g, ms in times.items() if ms <= band]
    return max(cands, key=lambda g: (_tiebreak_rank(g), -times[g], g))


def margin_pct(times: dict, winner) -> float:
    """(runner_up - best) / best * 100, using the winner and the next-best
    DISTINCT geometry -- what plan.md §4.3 filters unstable shapes on."""
    if len(times) < 2:
        return 100.0  # only one candidate timed: nothing to be close to
    ordered = sorted(times.values())
    best = times[winner]
    runner_up = next((v for v in ordered if v > best), None)
    if runner_up is None:
        return 100.0
    return (runner_up - best) / best * 100.0


def collate(readings):
    """readings -> {point_key: (winner_geom, {geom: ms})}, plus unstable/list.

    Multiple readings of the same point (e.g. a base sweep plus a targeted
    `--tag remeasure` rerun, plan.md §4.3, or the same shape falling in two
    overlapping chunk logs) are reduced to the MIN per geometry before the
    winner is picked -- per `hip-kernel-perf-measurement` / pitfall 7a ("take
    the peak [least-throttled], never the average"), a slower repeat run is
    thermal/contention noise on top of the true achievable time, not signal,
    so averaging it in only pollutes the candidate's score. A non-physical
    reading (<= 0 ms -- a timer bug, not a real measurement) is dropped
    before the min so it can never win outright.
    """
    acc = collections.defaultdict(lambda: collections.defaultdict(list))
    n_nonpositive = 0
    for shape, _winner, times in readings:
        key = point_key(shape)
        for geom, ms in times.items():
            if ms <= 0.0:
                n_nonpositive += 1
                continue
            acc[key][geom].append(ms)
    if n_nonpositive:
        print(f"[build] dropped {n_nonpositive} non-physical (<=0 ms) "
              f"reading(s)", file=sys.stderr)

    out, margins, unstable = {}, {}, []
    for key, per_geom in acc.items():
        times = {g: min(v) for g, v in per_geom.items() if v}
        if not times:
            continue
        winner = select_winner(times)
        mp = margin_pct(times, winner)
        out[key] = (winner, times)
        margins[key] = mp
        if mp < 3.0:
            unstable.append((key, mp))
    return out, margins, unstable


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------

def config_entry(geom) -> dict:
    kind = geom[0]
    if kind == "Wmma":
        _, bm, bn, wm, wn, sw, sk, bk = geom
        return {"kind": "Wmma", "bm16": bm // 16, "bn16": bn // 16,
                "swizzle": sw, "wt_m": wm, "wt_n": wn, "bk": bk,
                "split_k": sk, "threads": 0, "tile_n": 0}
    if kind == "TiledFma":
        _, bm, bn, bk, tm, tn, th = geom
        return {"kind": "TiledFma", "bm16": bm // 16, "bn16": bn // 16,
                "swizzle": 0, "wt_m": tm, "wt_n": tn, "bk": bk,
                "split_k": 0, "threads": th, "tile_n": 0}
    # Gemv (GemvNt has tile_n, GemvNn does not -- tile_n stays 0 for it,
    # matching the fbs comment that the field is unused/0 for NN entries).
    _, th, tn = geom
    return {"kind": "Gemv", "bm16": 0, "bn16": 0, "swizzle": 0, "wt_m": 0,
            "wt_n": 0, "bk": 0, "split_k": 0, "threads": th, "tile_n": tn}


def cmd_build(args) -> int:
    readings = load_readings(args)
    if readings is None:
        return 1
    points, margins, unstable = collate(readings)
    stable_pct = 100.0 * (1 - len(unstable) / len(margins)) if margins else 0.0
    print(f"[build] {len(points)} distinct measured points, "
          f"{stable_pct:.1f}% stable (margin>=3%), "
          f"{len(unstable)} unstable (plan.md Stage 4.3)")

    if unstable:
        unstable_csv = DATA_DIR / "unstable_shapes.csv"
        with open(unstable_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["phase", "act", "wts", "tb", "m", "n", "k",
                       "margin_pct"])
            for (group, m, n, k), mp in unstable:
                phase, act, wts, tb = group
                w.writerow([phase, act, wts, tb, m, n, k, f"{mp:.2f}"])
        print(f"[build]   -> {unstable_csv} -- consider a targeted "
              f"`measure --tag remeasure` pass before trusting these points")

    pool, index = [], {}

    def config_index(geom) -> int:
        if geom not in index:
            index[geom] = len(pool)
            pool.append(config_entry(geom))
        return index[geom]

    point_rows = []
    for (group, m, n, k) in sorted(points):
        phase, act, wts, tb = group
        winner, _times = points[(group, m, n, k)]
        point_rows.append({
            "phase": phase,
            "act_dtype": dtype_enum_name(act),
            "wts_dtype": dtype_enum_name(wts),
            "out_dtype": dtype_enum_name(act),  # out == act this round, §2.2
            "trans_b": "NT" if tb else "NN",
            "config": config_index(winner),
            "m": m, "n": n, "k": k,
        })

    if len(pool) > 65535:
        print(f"[build] {len(pool)} distinct configs exceeds the uint16 "
              f"config index", file=sys.stderr)
        return 1

    doc = {
        "schema_version": SCHEMA_VERSION,
        "gpu_arch": args.arch,
        "rocm_version": args.rocm_version,
        "kernel_abi": KERNEL_ABI,
        "model_key": args.model_key,
        "weight_m": args.weight_m,
        "weight_n": args.weight_n,
        "weight_k": args.weight_k,
        "bf16_aliases_f16": args.bf16_aliases_f16,
    }
    LUT_DIR.mkdir(parents=True, exist_ok=True)
    out = Path(args.out) if args.out else LUT_DIR / f"{args.arch}.json"
    with open(out, "w", encoding="utf-8") as f:
        f.write("{\n")
        for k in doc:
            f.write(f' "{k}": {json.dumps(doc[k])},\n')
        f.write(' "configs": [\n')
        f.write(",\n".join("  " + json.dumps(c, sort_keys=True) for c in pool))
        # fallbacks intentionally always empty this round -- plan.md §1 D5.
        f.write("\n ],\n \"fallbacks\": [],\n \"points\": [\n")
        f.write(",\n".join("  " + json.dumps(r, sort_keys=True)
                           for r in point_rows))
        f.write("\n ]\n}\n")

    by_phase_dtype = collections.Counter(
        (r["phase"], r["act_dtype"]) for r in point_rows)
    print(f"[build] {len(point_rows)} points, {len(pool)} distinct configs "
          f"-> {out}")
    for (phase, dt), n_pts in sorted(by_phase_dtype.items()):
        print(f"[build]   {phase:<9s} {dt:<5s} {n_pts} points")
    return 0


def compact_points(points):
    """Conservatively prune same-config consecutive M runs.

    Lookup categories and N/K stay isolated.  Every maximal run of equal
    config in M order retains both endpoints, so boundaries and singleton
    runs remain represented; this is intentionally not the one-point variant.
    """
    groups = collections.defaultdict(list)
    for point in points:
        key = (point["phase"], point["act_dtype"], point["wts_dtype"],
               point.get("out_dtype"), point["trans_b"], point["n"], point["k"])
        groups[key].append(point)
    kept = []
    for key in sorted(groups):
        rows = sorted(groups[key], key=lambda p: (p["m"], p["config"]))
        i = 0
        while i < len(rows):
            j = i + 1
            while j < len(rows) and rows[j]["config"] == rows[i]["config"]:
                j += 1
            kept.append(rows[i])
            if j - i > 1:
                kept.append(rows[j - 1])
            i = j
    return sorted(kept, key=lambda p: (p["phase"], p["act_dtype"],
                                        p["wts_dtype"], p.get("out_dtype"),
                                        p["trans_b"], p["n"], p["k"], p["m"],
                                        p["config"]))


def cmd_compact(args) -> int:
    src = Path(args.input) if args.input else LUT_DIR / f"{args.arch}.json"
    dst = Path(args.out) if args.out else src
    with open(src, encoding="utf-8") as f:
        doc = json.load(f)
    if doc.get("schema_version") != 2:
        print(f"[compact] expected Q1 schema_version=2, got {doc.get('schema_version')}", file=sys.stderr)
        return 1
    if len(doc.get("configs", [])) > 255:
        print("[compact] config pool exceeds uint8 range", file=sys.stderr)
        return 1
    before = len(doc.get("points", []))
    doc["schema_version"] = SCHEMA_VERSION
    doc["points"] = compact_points(doc.get("points", []))
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=1, sort_keys=False)
        f.write("\n")
    print(f"[compact] {before} -> {len(doc['points'])} points, {len(doc['configs'])} configs -> {dst}")
    return 0


# ---------------------------------------------------------------------------
# compile
# ---------------------------------------------------------------------------

def cmd_compile(args) -> int:
    lut_json = LUT_DIR / f"{args.arch}.json"
    if not lut_json.exists():
        print(f"[compile] no {lut_json}; run build first", file=sys.stderr)
        return 1
    with tempfile.TemporaryDirectory() as tmp:
        cmd = [args.flatc, "--binary", "--strict-json", "-o", tmp, str(FBS),
               str(lut_json)]
        print("[compile] " + " ".join(cmd))
        proc = subprocess.run(cmd)
        if proc.returncode != 0:
            return proc.returncode
        produced = list(Path(tmp).glob("*.bin")) + list(Path(tmp).glob("*.fb"))
        if not produced:
            print("[compile] flatc produced nothing", file=sys.stderr)
            return 1
        dst = LUT_DIR / f"{args.arch}.fb"
        shutil.copy(produced[0], dst)
    # Only the .fb is committed; CMake embeds its bytes into the
    # custom_kernels DLL at configure time (file(READ ... HEX)) -- R7: a
    # `build`-only refresh of the .json leaves the DLL on the OLD .fb bytes,
    # a re-configure is required after this step, not just a rebuild.
    print(f"[compile] -> {dst} ({dst.stat().st_size} bytes)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("command",
                    choices=["measure", "build", "compact", "compile", "all"])
    ap.add_argument("--arch", default="gfx1151")
    ap.add_argument("--sweep", default="gemm_sweep.exe",
                    help="path to the built gemm_autotune_sweep binary")
    ap.add_argument("--shapes", default=str(SHAPES))
    ap.add_argument("--m", default=None,
                    help="comma-separated M list override for the sweep")
    ap.add_argument("--dtypes", default=None,
                    help="comma-separated dtype list override (f16,bf16,f32)")
    ap.add_argument("--phases", default=None,
                    help="comma-separated phase list override "
                         "(Wmma,GemvNt,GemvNn,TiledFma)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tag", default=None,
                    help="log suffix, e.g. a chunked sweep or a Stage 4.3 "
                         "margin<3% remeasure pass; build reads every "
                         "<arch>*_sweep.log in data/")
    ap.add_argument("--sweep-args", nargs=argparse.REMAINDER, default=[],
                    help="everything after this is passed to the sweep binary")
    ap.add_argument("--out", default=None, help="override the output json path")
    ap.add_argument("--input", default=None, help="input JSON for compact")
    ap.add_argument("--flatc", default="flatc")
    ap.add_argument("--rocm-version", type=int, default=70140)
    ap.add_argument("--model-key", default="gemm_shapes.csv (fp16/bf16/fp32)",
                    help="provenance label (D9: v1 defaulted to a "
                         "fp16-only label that was wrong post-migration)")
    ap.add_argument("--data-dir", default=None,
                    help="read sweep logs from here instead of scripts/data; "
                         "lets a table be rebuilt from an archived snapshot")
    ap.add_argument("--weight-m", type=float, default=1.0)
    ap.add_argument("--weight-n", type=float, default=1.0)
    ap.add_argument("--weight-k", type=float, default=1.0)
    ap.add_argument("--bf16-aliases-f16", action="store_true",
                    help="set only after Stage C's measured verdict (plan.md "
                         "§5 Stage C) -- do not guess this")
    args = ap.parse_args()

    if args.command in ("measure", "all"):
        rc = cmd_measure(args)
        if rc:
            return rc
    if args.command in ("build", "all"):
        rc = cmd_build(args)
        if rc:
            return rc
    if args.command == "compact":
        return cmd_compact(args)
    if args.command in ("compile", "all"):
        rc = cmd_compile(args)
        if rc:
            return rc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
