#!/usr/bin/env python3
"""MatMulNBits autotune LUT pipeline: measure -> fit -> build -> compile.

    python update_lut.py measure   # drive the GPU sweep, write data/*.log
    python update_lut.py fit       # choose the distance weights, report error
    python update_lut.py build     # logs -> lut/<arch>.json
    python update_lut.py compile   # json -> lut/<arch>.fb + lut/<arch>_lut_data.cpp (needs flatc)
    python update_lut.py all

The winners come from the in-kernel autotuner, read off its debug log. Nothing
here re-implements "which config is fastest": there is one implementation of
that and it is the one that ships, so the table cannot drift from the runtime's
own judgement.

The table is a set of measured points plus a per-phase fallback; a lookup takes
the nearest point in log space over (M, N, K) within an exactly-matched
categorical group. See matmul_nbits_autotune.fbs for why.

The sweep log carries the full per-config timing table at every measured point,
not just the winner, and that is what makes `fit` possible: the cost of handing
a shape its neighbour's config can be read off the shape's own timings rather
than guessed at.
"""
from __future__ import annotations

import argparse
import collections
import itertools
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
ROOT = HERE.parent                      # autotune/matmul_nbits
FBS = ROOT / "matmul_nbits_autotune.fbs"
LUT_DIR = ROOT / "lut"
DATA_DIR = ROOT / "scripts" / "data"
SHAPES = ROOT / "shapes" / "oga_models_bits4.csv"

SCHEMA_VERSION = 4
KERNEL_ABI = "matmul_nbits-v1"

GROUP_SIZES = {16: "G16", 32: "G32", 64: "G64", 128: "G128", 256: "G256",
               512: "G512"}
PATH_TO_PHASE = {"prefill": "Prefill", "prefill_padrow": "Prefill",
                 "decode": "Decode", "dp4a": "DecodeDp4a"}
PATH_TO_STRIDE = {"prefill": "Arrival", "prefill_padrow": "Padded",
                  "decode": "Any", "dp4a": "Any"}
PATH_TO_KIND = {"prefill": "Wmma", "prefill_padrow": "Wmma",
                "decode": "Gemv", "dp4a": "Gemv"}


# ---------------------------------------------------------------------------
# measure
# ---------------------------------------------------------------------------

def cmd_measure(args) -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"_{args.tag}" if args.tag else ""
    log = DATA_DIR / f"{args.arch}{tag}_sweep.log"

    # A cached shape is not re-tuned, so a stale cache silently truncates the
    # sweep. Clear it rather than trusting it.
    tmp = os.environ.get("TEMP") or os.environ.get("TMPDIR") or "/tmp"
    removed = 0
    for p in Path(tmp).glob("morphizen_*cache*"):
        try:
            p.unlink()
            removed += 1
        except OSError:
            pass
    print(f"[measure] cleared {removed} tune cache files")

    env = dict(os.environ, HIPDNN_EP_DEBUG="1")
    cmd = [args.sweep, "--shapes", str(args.shapes)]
    if args.m:
        cmd += ["--m", args.m]
    if args.limit:
        cmd += ["--limit", str(args.limit)]
    cmd += args.sweep_args
    print("[measure] " + " ".join(cmd))
    # Redirect into a real file handle. Do NOT let a shell do this on Windows:
    # PowerShell routes a native command's stderr through error-record
    # formatting and wraps it at the console width, which silently truncates
    # the autotune lines this pipeline parses.
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
    r"^#SHAPE path=(\w+) M=(\d+) N=(\d+) K=(\d+) gs=(\d+) zp=(\d+)")
WMMA_WIN_RE = re.compile(
    r"WMMA autotune:.*-> best config\[(\d+)\] (fused|dq\+gemm) "
    r"bm=(\d+) bn=(\d+) sw=(\d+) wt=(\d+)x(\d+) bk=(\d+)")
GEMV_WIN_RE = re.compile(
    r"GEMV(?:-dp4a)? autotune:.*-> best config\[(\d+)\] "
    r"threads=(\d+) tile_n=(\d+)")
# Per-config timing lines. Every config the tuner tried at this point, which is
# what lets `fit` price a wrong answer instead of only counting it.
#
# The label is "dq+gem" here and "dq+gemm" on the winner line: the timing log
# pads the field to six characters. Matching the winner's spelling silently
# dropped every separate-dequant timing, which left `fit` scoring against a
# best-of that only covered the fused half of the table.
WMMA_TIME_RE = re.compile(
    r"^\[custom_kernels\]\s+wmma config\[\s*\d+\] (fused|dq\+gemm?)\s+"
    r"bm=\s*(\d+) bn=\s*(\d+) sw=(\d+) wt=(\d+)x(\d+) bk=(\d+) : "
    r"([\d.]+) ms")
GEMV_TIME_RE = re.compile(
    r"^\[custom_kernels\]\s+(?:dp4a )?config\[\s*\d+\] "
    r"threads=\s*(\d+) tile_n=(\d+) : ([\d.]+) ms")


def answer_key(ans: dict):
    """Hashable identity of an answer: the dedup key and the comparison key."""
    if ans["kind"] == "Wmma":
        return ("Wmma", ans["bm"], ans["bn"], ans["swizzle"], ans["wt_m"],
                ans["wt_n"], ans["bk"], 1 if ans["fused"] else 0)
    return ("Gemv", ans["threads"], ans["tile_n"])


def config_entry(key) -> dict:
    """answer_key tuple -> a MatmulNbitsTuneConfig dict."""
    if key[0] == "Wmma":
        _, bm, bn, sw, wm, wn, bk, fused = key
        return {"kind": "Wmma", "bm16": bm // 16, "bn16": bn // 16,
                "swizzle": sw, "wt_m": wm, "wt_n": wn, "bk": bk,
                "fused": fused, "threads": 0, "tile_n": 0}
    _, threads, tile_n = key
    return {"kind": "Gemv", "bm16": 0, "bn16": 0, "swizzle": 0, "wt_m": 0,
            "wt_n": 0, "bk": 0, "fused": 0, "threads": threads,
            "tile_n": tile_n}


def parse_log(path: Path):
    """Yield (shape_tuple, winner_key, {answer_key: ms}) per measured point.

    shape_tuple is (path_kind, M, N, K, gs, zp). The timing dict is every config
    the tuner timed at that point.
    """
    cur = None
    times: dict = {}
    n_orphan = 0
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = SHAPE_RE.match(line)
            if m:
                if cur is not None:
                    n_orphan += 1          # marker with no winner line after
                cur = (m.group(1), int(m.group(2)), int(m.group(3)),
                       int(m.group(4)), int(m.group(5)), int(m.group(6)))
                times = {}
                continue
            if cur is None:
                continue
            want_wmma = PATH_TO_KIND[cur[0]] == "Wmma"

            if want_wmma:
                t = WMMA_TIME_RE.match(line)
                if t:
                    times[("Wmma", int(t.group(2)), int(t.group(3)),
                           int(t.group(4)), int(t.group(5)), int(t.group(6)),
                           int(t.group(7)), 1 if t.group(1) == "fused" else 0)
                          ] = float(t.group(8))
                    continue
                w = WMMA_WIN_RE.search(line)
                if w:
                    key = ("Wmma", int(w.group(3)), int(w.group(4)),
                           int(w.group(5)), int(w.group(6)), int(w.group(7)),
                           int(w.group(8)), 1 if w.group(2) == "fused" else 0)
                    yield cur, key, times
                    cur, times = None, {}
                continue

            t = GEMV_TIME_RE.match(line)
            if t:
                times[("Gemv", int(t.group(1)), int(t.group(2)))] = \
                    float(t.group(3))
                continue
            g = GEMV_WIN_RE.search(line)
            if g:
                yield cur, ("Gemv", int(g.group(2)), int(g.group(3))), times
                cur, times = None, {}
    if n_orphan:
        # Usually means the shape hit a cached entry (sweep run without a cache
        # clear) or fell through to a path with no tuner.
        print(f"[build] {n_orphan} shape markers had no autotune line",
              file=sys.stderr)


def load_readings(args):
    """Every *_sweep.log for this arch, merged.

    Not just the main one: shapes whose scope the main sweep caps (the vocab
    projections of the exports that declare logits as [batch, seq, vocab]) are
    measured by a separate run into their own log, and both are part of the
    same table.
    """
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


def point_key(shape):
    """(categorical group, M, N, K) -- the identity of a measured point."""
    pk, m, n, k, gs, zp = shape
    return ((PATH_TO_PHASE[pk], GROUP_SIZES.get(gs, "Any"),
             "Asymmetric" if zp else "Symmetric", PATH_TO_STRIDE[pk]),
            m, n, k)


# A reading this far below the median config time at the same point is not a
# config that is that good, it is a timing that did not happen. Every config at
# a point does identical work, so the median is a solid anchor: the real spread
# between the best and the median config in this data is under 4x, while the
# artifacts sit 100x or more below it.
IMPLAUSIBLE_RATIO = 20.0


def drop_implausible(times):
    """Strip readings too fast to be real. Returns (kept, n_dropped).

    The sweep occasionally records a near-zero time for one config -- 21 of the
    2018 measured points had one, always under a microsecond for five launches,
    which is below the cost of issuing them. Because every tuner keeps the
    minimum, such a reading wins its point every single time it occurs, so this
    cannot be left to average out: at 1% of points it is 1% of the table
    pointing at a config that was never actually the fastest.
    """
    if len(times) < 3:
        return times, 0
    ordered = sorted(times.values())
    median = ordered[len(ordered) // 2]
    floor = median / IMPLAUSIBLE_RATIO
    kept = {a: ms for a, ms in times.items() if ms >= floor}
    return (kept, len(times) - len(kept)) if kept else (times, 0)


# Near-tie band for the winner selection below. Configs within this factor of
# the fastest are treated as measurement-equal and separated structurally.
# Mirrors the value the WMMA kernel tuner used before that logic was moved here.
TIE_BAND = 1.03


def _tiebreak_rank(ans_key):
    """Structural preference among measurement-tied configs: larger warp tile
    (WT_M*WT_N) first, then larger tile area (BM*BN). A K-step issues WT_M*WT_N
    WMMA per (WT_M+WT_N) fragment loads, so the wider tile does the same
    arithmetic with fewer LDS reads -- a better tie-breaker than batch-timing
    noise, which cannot see cross-dispatch overlap. GEMV/dp4a configs have no
    warp tile, so they rank (0, 0) and the tie reduces to fastest-wins."""
    if ans_key[0] == "Wmma":
        _, bm, bn, _sw, wt_m, wt_n, _bk, _fused = ans_key
        return (wt_m * wt_n, bm * bn)
    return (0, 0)


def select_winner(times):
    """Fastest config, breaking near-ties (within TIE_BAND) toward the larger
    warp tile then tile area.

    This is the selection the WMMA kernel tuner used to do inline. It was moved
    here so the runtime tuner stays a plain single pass (no clock-settling,
    no multi-pass, no tie-break on the dispatch path); the table -- built from
    the full per-config timings the sweep logs -- carries the more careful
    choice instead. On the fp GEMV / dp4a paths there is no warp tile, so this
    reduces to argmin."""
    fastest = min(times.values())
    band = fastest * TIE_BAND
    cands = [a for a, ms in times.items() if ms <= band]
    # Largest warp tile, then area; ties broken toward the faster config, then
    # the answer key itself so the choice is reproducible from the readings.
    return max(cands, key=lambda a: (_tiebreak_rank(a), -times[a], a))


def collate(readings):
    """readings -> {point_key: (winner, {answer: ms})}.

    A point measured in several models is one point: its per-config timings are
    averaged across the models that measured it, then the winner is chosen by
    select_winner() from those timings -- fastest, with near-ties broken toward
    the larger warp tile. The winner is deliberately NOT taken from the tuner's
    logged choice: the runtime tuner is a plain single pass now, so its logged
    winner is a raw argmin (and occasionally a phantom); the careful selection
    lives here, offline, where it costs nothing at runtime.

    drop_implausible strips phantom readings from each measurement before they
    are averaged, so a near-zero (or negative) timing cannot win a point.
    """
    acc = collections.defaultdict(lambda: collections.defaultdict(list))
    winners_seen = collections.defaultdict(collections.Counter)
    poisoned = set()
    n_dropped = 0
    for shape, winner, times in readings:
        clean, dropped = drop_implausible(times)
        n_dropped += dropped
        key = point_key(shape)
        for ans, ms in clean.items():
            acc[key][ans].append(ms)
        if times and winner not in clean:
            poisoned.add(key)
        else:
            winners_seen[key][winner] += 1

    out = {}
    n_tiebroken = 0
    for key, per_ans in acc.items():
        times = {a: sum(v) / len(v) for a, v in per_ans.items()}
        if not times:
            continue
        winner = select_winner(times)
        # Count where the structural tie-break overrode the raw fastest.
        if winner != min(times.items(), key=lambda kv: (kv[1], kv[0]))[0]:
            n_tiebroken += 1
        out[key] = (winner, times)

    if n_dropped:
        print(f"[build] dropped {n_dropped} implausible readings "
              f"({len(poisoned)} points had their logged winner dropped)")
    print(f"[build] tie-break moved {n_tiebroken} points off the raw fastest "
          f"to a larger warp tile")

    # A point whose log had a winner but no timing table (an older log, or a
    # truncated one) still contributes its winner; it just cannot be scored or
    # tie-broken.
    for key, counter in winners_seen.items():
        if key not in out:
            out[key] = (counter.most_common(1)[0][0], {})
    return out


# ---------------------------------------------------------------------------
# the metric
# ---------------------------------------------------------------------------

def nearest(query, candidates, weights):
    """Candidates sorted by weighted log-space distance to `query`.

    query and candidates are (m, n, k). Mirrors resolve() in
    matmul_nbits_autotune.cpp; the two must agree or `fit` measures a metric
    the runtime does not use.
    """
    wm, wn, wk = weights
    qm, qn, qk = (math.log2(max(v, 1)) for v in query)
    scored = []
    for i, (cm, cn, ck) in enumerate(candidates):
        dm = wm * (qm - math.log2(max(cm, 1)))
        dn = wn * (qn - math.log2(max(cn, 1)))
        dk = wk * (qk - math.log2(max(ck, 1)))
        scored.append((dm * dm + dn * dn + dk * dk, i))
    scored.sort()
    return scored


def slowdowns(points, weights, holdout="shape"):
    """Leave-one-out cost of answering each point from its neighbours.

    holdout="point"  drop just this (M, N, K); the other M values of the same
                     layer stay in the table. Answers "how well does the table
                     interpolate an M it never saw".
    holdout="shape"  drop every M of this (N, K). Answers the question that
                     actually matters -- how well the table serves a model
                     nobody measured -- and is the number worth quoting.

    Returns the list of (cost, group, m, n, k), cost being the ratio of the
    chosen config's time at this point to the best time at this point. 1.0 is
    a perfect answer.
    """
    by_group = collections.defaultdict(list)
    for (group, m, n, k) in points:
        by_group[group].append((m, n, k))

    out = []
    for group, members in by_group.items():
        if len(members) < 2:
            continue
        for (m, n, k) in members:
            winner, times = points[(group, m, n, k)]
            if not times:
                continue
            best_ms = min(times.values())
            if best_ms <= 0:
                continue
            if holdout == "shape":
                pool = [c for c in members if (c[1], c[2]) != (n, k)]
            else:
                pool = [c for c in members if c != (m, n, k)]
            if not pool:
                continue
            # Walk outwards exactly as resolve() does: a config this point
            # never timed is one the validator would have refused here.
            for _d2, idx in nearest((m, n, k), pool, weights):
                cand = points[(group, ) + pool[idx]][0]
                if cand in times:
                    out.append((times[cand] / best_ms, group, m, n, k))
                    break
    return out


def summarize(costs):
    if not costs:
        return "no scorable points"
    vals = sorted(c[0] for c in costs)
    n = len(vals)
    geo = math.exp(sum(math.log(v) for v in vals) / n)
    within5 = sum(1 for v in vals if v <= 1.05) / n
    return (f"n={n} geomean={geo:.4f} median={vals[n // 2]:.4f} "
            f"p90={vals[int(n * 0.9)]:.4f} p99={vals[int(n * 0.99)]:.4f} "
            f"max={vals[-1]:.3f} within5%={within5 * 100:.1f}%")


# ---------------------------------------------------------------------------
# fit
# ---------------------------------------------------------------------------

def cmd_fit(args) -> int:
    readings = load_readings(args)
    if readings is None:
        return 1
    points = collate(readings)
    print(f"[fit] {len(points)} distinct measured points")

    # The metric is invariant to a common scale factor, so one weight is pinned
    # and the other two are searched around it.
    grid = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
    results = []
    for wm, wk in itertools.product(grid, repeat=2):
        weights = (wm, 1.0, wk)
        costs = slowdowns(points, weights, args.holdout)
        vals = [c[0] for c in costs]
        geo = math.exp(sum(math.log(v) for v in vals) / len(vals))
        results.append((geo, weights, costs))
    results.sort(key=lambda r: r[0])

    print(f"[fit] holdout={args.holdout}, best 5 of {len(results)} weightings:")
    for geo, weights, costs in results[:5]:
        print(f"      m={weights[0]:<5} n={weights[1]:<5} k={weights[2]:<5} "
              f"{summarize(costs)}")
    geo, weights, costs = results[0]
    print(f"[fit] -> weight_m={weights[0]} weight_n={weights[1]} "
          f"weight_k={weights[2]}")

    if args.verbose:
        worst = sorted(costs, reverse=True)[:20]
        print("[fit] worst points:")
        for cost, group, m, n, k in worst:
            print(f"      {cost:6.3f}x  {group[0]:<10s} gs={group[1]:<5s} "
                  f"{group[2]:<10s} {group[3]:<8s} M={m:<6d} N={n:<7d} K={k}")
    return 0


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------

def choose_fallbacks(points):
    """One config per phase, for a categorical group with no measured point.

    Picked as the config with the lowest geomean slowdown across every point of
    that phase that timed it, not the one that won most often. Winning often is
    a popularity contest among shapes that happened to be measured; what a last
    resort needs is to be the least bad when it is wrong, and that is a
    different config whenever the frequent winner is also a frequent disaster.
    Only configs timed at nearly every point are eligible, so a config that
    looks good on the handful of shapes it was legal for cannot win.
    """
    per_phase = collections.defaultdict(dict)
    for (group, m, n, k), (winner, times) in points.items():
        if times:
            per_phase[group[0]][(m, n, k)] = times

    out = {}
    for phase, pts in per_phase.items():
        seen = collections.Counter()
        for times in pts.values():
            for ans in times:
                seen[ans] += 1
        eligible = [a for a, c in seen.items() if c >= 0.9 * len(pts)]
        if not eligible:
            eligible = list(seen)
        scored = []
        for ans in eligible:
            logs = []
            for times in pts.values():
                if ans not in times:
                    continue
                best = min(times.values())
                if best > 0:
                    logs.append(math.log(times[ans] / best))
            if logs:
                scored.append((math.exp(sum(logs) / len(logs)), ans))
        if scored:
            scored.sort()
            out[phase] = scored[0][1]
            print(f"[build] fallback {phase:<11s} {scored[0][1]} "
                  f"geomean {scored[0][0]:.3f}x off best")
    return out


def cmd_build(args) -> int:
    readings = load_readings(args)
    if readings is None:
        return 1
    points = collate(readings)

    if args.fit_weights:
        grid = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
        best = None
        for wm, wk in itertools.product(grid, repeat=2):
            costs = slowdowns(points, (wm, 1.0, wk), "shape")
            vals = [c[0] for c in costs]
            geo = math.exp(sum(math.log(v) for v in vals) / len(vals))
            if best is None or geo < best[0]:
                best = (geo, (wm, 1.0, wk))
        weights = best[1]
        print(f"[build] fitted weights m={weights[0]} n={weights[1]} "
              f"k={weights[2]} (geomean {best[0]:.4f}x)")
    else:
        weights = (args.weight_m, args.weight_n, args.weight_k)

    fallbacks = choose_fallbacks(points)

    pool, index = [], {}

    def config_index(key) -> int:
        if key not in index:
            index[key] = len(pool)
            pool.append(config_entry(key))
        return index[key]

    # Sorted so the emitted table is a function of the readings alone: the
    # lookup breaks distance ties by position, so an unstable order here would
    # be an unstable answer there.
    point_rows = []
    for (group, m, n, k) in sorted(points):
        phase, gs, zp, stride = group
        winner, _times = points[(group, m, n, k)]
        point_rows.append({
            "phase": phase, "bits": "B4", "group_size": gs, "zero_point": zp,
            "row_stride": stride, "config": config_index(winner),
            "m": m, "n": n, "k": k,
        })

    fallback_rows = [
        {"phase": phase, "bits": "B4", "config": config_index(ans)}
        for phase, ans in sorted(fallbacks.items())
    ]

    if len(pool) > 256:
        print(f"[build] {len(pool)} distinct configs exceeds the ubyte config "
              f"index", file=sys.stderr)
        return 1

    doc = {
        "schema_version": SCHEMA_VERSION,
        "gpu_arch": args.arch,
        "rocm_version": args.rocm_version,
        "kernel_abi": KERNEL_ABI,
        "model_key": args.model_key,
        "weight_m": weights[0],
        "weight_n": weights[1],
        "weight_k": weights[2],
    }
    LUT_DIR.mkdir(parents=True, exist_ok=True)
    out = Path(args.out) if args.out else LUT_DIR / f"{args.arch}.json"
    # One entry per line: compact enough to keep the file reviewable and to make
    # a regeneration diff readable, without the 20x blowup of indent=1.
    with open(out, "w", encoding="utf-8") as f:
        f.write("{\n")
        for k in doc:
            f.write(f' "{k}": {json.dumps(doc[k])},\n')
        f.write(' "configs": [\n')
        f.write(",\n".join("  " + json.dumps(c, sort_keys=True) for c in pool))
        f.write("\n ],\n \"fallbacks\": [\n")
        f.write(",\n".join("  " + json.dumps(r, sort_keys=True)
                           for r in fallback_rows))
        f.write("\n ],\n \"points\": [\n")
        f.write(",\n".join("  " + json.dumps(r, sort_keys=True)
                           for r in point_rows))
        f.write("\n ]\n}\n")

    groups = len({(r["phase"], r["group_size"], r["zero_point"],
                   r["row_stride"]) for r in point_rows})
    print(f"[build] {len(point_rows)} points in {groups} groups, "
          f"{len(pool)} distinct configs -> {out}")

    costs = slowdowns(points, weights, "shape")
    print(f"[build] leave-one-shape-out: {summarize(costs)}")
    return 0


# ---------------------------------------------------------------------------
# compile
# ---------------------------------------------------------------------------

def embed_lut_data_cpp(fb: Path, out: Path) -> None:
    """Turn lut/<arch>.fb into the checked-in C++ source the build links."""
    data = fb.read_bytes()
    lines = [
        f"// Auto-generated from {fb.name} by update_lut.py compile; do not edit.",
        "#include <cstddef>",
        'extern "C" const unsigned char kMatmulNbitsLutData[] = {',
    ]
    col = 16
    for i in range(0, len(data), col):
        chunk = data[i : i + col]
        hex_chunk = ",".join(f"0x{b:02x}" for b in chunk)
        ascii_chunk = "".join(
            chr(b) if 32 <= b < 127 and b not in (ord("*"), ord("/"), ord("\\"))
            else "."
            for b in chunk
        )
        lines.append(
            f"/*{i:08x} */  {hex_chunk:<{col * 3}}, /* {ascii_chunk} */")
    lines.append("0x00};")
    lines.append(f'extern "C" const size_t kMatmulNbitsLutData_size = {len(data)};')
    out.write_text("\n".join(lines) + "\n")


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
    cpp = LUT_DIR / f"{args.arch}_lut_data.cpp"
    embed_lut_data_cpp(dst, cpp)
    print(f"[compile] -> {dst} ({dst.stat().st_size} bytes)")
    print(f"[compile] -> {cpp} ({cpp.stat().st_size} bytes)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("command",
                    choices=["measure", "fit", "build", "compile", "all"])
    ap.add_argument("--arch", default="gfx1151")
    ap.add_argument("--sweep", default="matmul_nbits_sweep.exe",
                    help="path to the built matmul_nbits_autotune_sweep binary")
    ap.add_argument("--shapes", default=str(SHAPES))
    ap.add_argument("--m", default=None,
                    help="comma-separated M list for the sweep")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tag", default=None,
                    help="log suffix, for a sweep that supplements the main "
                         "one; build reads every <arch>*_sweep.log")
    ap.add_argument("--sweep-args", nargs=argparse.REMAINDER, default=[],
                    help="everything after this is passed to the sweep binary")
    ap.add_argument("--out", default=None, help="override the output json path")
    ap.add_argument("--flatc", default="flatc")
    ap.add_argument("--rocm-version", type=int, default=70100)
    ap.add_argument("--model-key", default="oga_models bits=4")
    ap.add_argument("--data-dir", default=None,
                    help="read sweep logs from here instead of scripts/data; "
                         "lets a table be rebuilt from an archived snapshot")
    ap.add_argument("--holdout", choices=["shape", "point"], default="shape",
                    help="fit: shape drops every M of an (N, K), which is the "
                         "unmeasured-model case; point drops one (M, N, K)")
    ap.add_argument("--fit-weights", action="store_true",
                    help="build: fit the distance weights from the readings "
                         "instead of using --weight-*")
    ap.add_argument("--weight-m", type=float, default=1.0)
    ap.add_argument("--weight-n", type=float, default=1.0)
    ap.add_argument("--weight-k", type=float, default=1.0)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if args.command in ("measure", "all"):
        rc = cmd_measure(args)
        if rc:
            return rc
    if args.command == "fit":
        return cmd_fit(args)
    if args.command in ("build", "all"):
        rc = cmd_build(args)
        if rc:
            return rc
    if args.command in ("compile", "all"):
        rc = cmd_compile(args)
        if rc:
            return rc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
