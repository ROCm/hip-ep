#!/usr/bin/env python3
"""MatMulNBits autotune LUT pipeline: measure -> fit -> build -> compile.

    python update_lut.py measure   # drive the GPU sweep, write data/*.log
    python update_lut.py fit       # choose the distance weights, report error
    python update_lut.py build     # logs -> lut/<arch>.json
    python update_lut.py compile   # json -> lut/<arch>.fb (needs flatc)
    python update_lut.py all

Legacy sweep winners come from in-kernel autotuner logs. Kernel-change refreshes
use import-fast to consume one accepted fast ranking dataset directly. import-ut
also supports previously completed checked pinned-config data. Neither import
runs another GPU measurement pass.

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
import hashlib
import statistics
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

    # The tuners no longer persist winners to a %TEMP% cache file (that layer
    # was removed once the offline LUT became the durable source), so each sweep
    # process already starts with an empty in-process map -- nothing to clear.

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
        # Usually means the shape fell through to a path with no tuner.
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


def _answer_of(cfg: dict):
    """A table config entry -> answer_key, the inverse of config_entry()."""
    if cfg["kind"] == "Wmma":
        return ("Wmma", cfg["bm16"] * 16, cfg["bn16"] * 16, cfg["swizzle"],
                cfg["wt_m"], cfg["wt_n"], cfg["bk"], cfg["fused"])
    return ("Gemv", cfg["threads"], cfg["tile_n"])


def _fmt_answer(ans) -> str:
    if ans[0] == "Gemv":
        return f"{ans[1]}x{ans[2]}"
    return "wmma" + "/".join(str(v) for v in ans[1:])


def _geomean_slowdown(ans, pts):
    logs = [math.log(t[ans] / min(t.values())) for t in pts if ans in t]
    return math.exp(sum(logs) / len(logs)) if logs else float("inf")


def write_lut_json(out: Path, doc: dict, configs, fallbacks, points) -> None:
    # One entry per line: compact enough to keep the file reviewable and to make
    # a regeneration diff readable, without the 20x blowup of indent=1.
    with open(out, "w", encoding="utf-8") as f:
        f.write("{\n")
        for k in doc:
            f.write(f' "{k}": {json.dumps(doc[k])},\n')
        f.write(' "configs": [\n')
        f.write(",\n".join("  " + json.dumps(c, sort_keys=True) for c in configs))
        f.write("\n ],\n \"fallbacks\": [\n")
        f.write(",\n".join("  " + json.dumps(r, sort_keys=True)
                           for r in fallbacks))
        f.write("\n ],\n \"points\": [\n")
        f.write(",\n".join("  " + json.dumps(r, sort_keys=True)
                           for r in points))
        f.write("\n ]\n}\n")


def cmd_merge(args) -> int:
    """Re-measure one phase and change only the rows a new sweep clearly beats.

    Every other phase's rows, the header and the distance weights are kept as
    they are. For each measured point of --phase, the existing row is replaced
    only if the new winner is at least --min-gain faster than the row's current
    config *in the same sweep*; a near-tie stays put, so a regeneration diff
    shows real changes rather than timing noise. Points the table did not have
    are added.

    The phase fallback is kept unless --update-fallback is given. It only
    answers keys nothing was measured for (a group size absent from the
    table), so a geomean over the measured shapes is no evidence for it:
    moving the gfx1151 DecodeDp4a fallback from 32x1 to 64x1 on that basis
    made Nemotron-3.5-Lightning (all G64) 3% slower in-model. With the flag it
    follows the same --min-gain rule on its geomean slowdown.
    """
    readings = load_readings(args)
    if readings is None:
        return 1
    phase = args.phase
    measured = {k: v for k, v in collate(readings).items() if k[0][0] == phase}
    print(f"[merge] {len(measured)} measured {phase} points")
    src = LUT_DIR / f"{args.arch}.json"
    old = json.loads(src.read_text(encoding="utf-8"))
    pool = old["configs"]
    gain = 1.0 + args.min_gain

    def row_key(r):
        return ((r["phase"], r.get("group_size", "Any"),
                 r.get("zero_point", "Symmetric"), r.get("row_stride", "Any")),
                r["m"], r["n"], r["k"])

    changes, kept_ties, seen = [], 0, set()
    rows = []
    for r in old["points"]:
        key = row_key(r)
        if key not in measured:
            rows.append((r, None))
            continue
        seen.add(key)
        winner, times = measured[key]
        cur = _answer_of(pool[r["config"]])
        if winner == cur or not times:
            rows.append((r, None))
        elif cur not in times or times[cur] >= gain * times[winner]:
            rows.append((r, winner))
            changes.append((key, cur, winner, times.get(cur), times[winner]))
        else:
            rows.append((r, None))
            kept_ties += 1
    for key in sorted(set(measured) - seen):
        (ph, gs, zp, stride), m, n, k = key
        r = {"phase": ph, "bits": "B4", "group_size": gs, "zero_point": zp,
             "row_stride": stride, "m": m, "n": n, "k": k, "config": -1}
        rows.append((r, measured[key][0]))
        changes.append((key, None, measured[key][0], None,
                        measured[key][1].get(measured[key][0])))

    fb_rows = []
    fb_change = None
    new_fb = choose_fallbacks(measured).get(phase)
    for r in old["fallbacks"]:
        if r["phase"] == phase and new_fb is not None:
            pts = [t for _w, t in measured.values() if t]
            cur = _answer_of(pool[r["config"]])
            g_cur, g_new = (_geomean_slowdown(cur, pts),
                            _geomean_slowdown(new_fb, pts))
            if new_fb != cur and g_cur >= gain * g_new:
                fb_change = (cur, new_fb, g_cur, g_new)
                if args.update_fallback:
                    fb_rows.append((r, new_fb))
                    continue
        fb_rows.append((r, None))

    # Existing pool indices and row order are kept and new configs appended, so
    # the json diff is just the changed rows. A config no row uses any more
    # stays in the pool; the next full build drops it.
    new_pool = list(pool)
    index = {json.dumps(c, sort_keys=True): i for i, c in enumerate(pool)}

    def idx_of(entry: dict) -> int:
        key = json.dumps(entry, sort_keys=True)
        if key not in index:
            index[key] = len(new_pool)
            new_pool.append(entry)
        return index[key]

    def resolved(r, ans):
        out = dict(r)
        if ans is not None:
            out["config"] = idx_of(config_entry(ans))
        return out

    fallback_rows = [resolved(r, a) for r, a in fb_rows]
    point_rows = [resolved(r, a) for r, a in rows]
    doc = {k: v for k, v in old.items()
           if k not in ("configs", "fallbacks", "points")}
    out = Path(args.out) if args.out else LUT_DIR / f"{args.arch}.json"
    write_lut_json(out, doc, new_pool, fallback_rows, point_rows)
    print(f"[merge] {len(changes)} {phase} rows changed, {kept_ties} kept as "
          f"within {args.min_gain:.0%}, {len(point_rows)} points total -> {out}")

    if args.report:
        lines = [f"# {phase} LUT merge ({args.arch})", "",
                 f"Rows change only when the new winner is at least "
                 f"{args.min_gain:.0%} faster than the row's config in the same "
                 f"sweep. {len(measured)} points measured, {len(changes)} "
                 f"changed, {kept_ties} kept as near-ties.", ""]
        if fb_change:
            cur, new, g_cur, g_new = fb_change
            verb = "->" if args.update_fallback else "kept; candidate"
            lines += [f"Fallback: {_fmt_answer(cur)} {verb} {_fmt_answer(new)} "
                      f"(geomean slowdown over the measured shapes {g_cur:.3f}x "
                      f"-> {g_new:.3f}x)", ""]
        lines += ["| group | M | N | K | old | new | old ms | new ms | gain |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for (grp, m, n, k), cur, new, t_cur, t_new in sorted(
                changes, key=lambda c: -(c[3] or 0) / (c[4] or 1)):
            g = f"{t_cur / t_new:.2f}x" if t_cur and t_new else "new row"
            lines.append(
                f"| {grp[1]} {grp[2]} | {m} | {n} | {k} | "
                f"{_fmt_answer(cur) if cur else '-'} | {_fmt_answer(new)} | "
                f"{t_cur if t_cur is None else f'{t_cur:.5f}'} | "
                f"{t_new if t_new is None else f'{t_new:.5f}'} | {g} |")
        Path(args.report).write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"[merge] report -> {args.report}")
    return 0


def cmd_build(args) -> int:
    if args.merge_existing:
        return cmd_merge(args)
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
    write_lut_json(out, doc, pool, fallback_rows, point_rows)

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

def cmd_import_ut(args) -> int:
    """Incremental refresh from the accepted pinned standard-UT dataset.

    No GPU command is executed. The input binds the complete affected scope,
    all per-config process/block readings, the code decision, and original LUT.
    Existing pool indices, unrelated points, fallbacks and fitted weights stay
    unchanged; priority model points are added from the same measurements.
    """
    if not args.accepted_ut:
        raise ValueError('import-ut requires --accepted-ut')
    input_bytes = Path(args.accepted_ut).read_bytes()
    packet = json.loads(input_bytes)
    data = packet['dataset']
    digest = 'sha256:' + hashlib.sha256(json.dumps(data, sort_keys=True,
        separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()
    if (packet.get('status') != 'CANDIDATE_ACCEPTED' or packet.get('dataset_digest') != digest
            or data.get('source') != 'standard_pinned_ut' or data.get('pinned') is not True
            or data.get('symmetric') is not True or data.get('architecture') != args.arch
            or data.get('aggregation') != 'median_blocks_then_processes'
            or not data.get('noise_label') or not data.get('clock_policy')
            or not data.get('source_manifests') or not data.get('measurement_epoch')):
        raise ValueError('accepted architecture-specific checked pinned-UT evidence is required')
    source = LUT_DIR / f'{args.arch}.json'
    if packet.get('original_lut_sha256') != hashlib.sha256(source.read_bytes()).hexdigest():
        raise ValueError('accepted dataset does not match the original LUT')
    old = json.loads(source.read_text(encoding='utf-8'))
    if old.get('gpu_arch') != args.arch:
        raise ValueError('LUT architecture mismatch')
    scope = packet['scope']
    if scope.get('actual_launch_mapping_complete') is not True or not scope.get('family_mapping_digest'):
        raise ValueError('actual launch-family mapping is required')
    indices = scope['affected_point_indices']
    affected = set(indices)
    if len(indices) != len(affected) or any(type(i) is not int or i < 0 or i >= len(old['points']) for i in affected):
        raise ValueError('invalid affected LUT point index')
    required_ids = scope['required_ids']
    if (not isinstance(required_ids, list) or not required_ids
            or any(not isinstance(i, str) or not i for i in required_ids)
            or len(required_ids) != len(set(required_ids))):
        raise ValueError('required scope ids must be unique nonempty strings')
    rows = data['rows']
    if len(rows) != len({row['id'] for row in rows}) or {row['id'] for row in rows} != set(scope['required_ids']):
        raise ValueError('incomplete or duplicate affected/model/sentinel rows')
    winners = {}
    replacements = {}
    additions = []
    known_configs = {}
    known_keys = set()
    for row in rows:
        key = row['key']
        if (set(key) != {'phase', 'bits', 'group_size', 'zero_point', 'row_stride', 'm', 'n', 'k'}
                or key['phase'] != 'Prefill' or key['bits'] != 'B4'
                or key['group_size'] not in GROUP_SIZES.values()
                or key['zero_point'] not in {'Symmetric', 'Asymmetric'}
                or key['row_stride'] not in {'Arrival', 'Padded'}
                or any(type(key[d]) is not int or not 0 < key[d] <= limit
                       for d, limit in (('m', 65535), ('n', 4294967295), ('k', 65535)))):
            raise ValueError('invalid measured WMMA-prefill key')
        encoded_key = json.dumps(key, sort_keys=True)
        if encoded_key in known_keys:
            raise ValueError('duplicate measured key')
        known_keys.add(encoded_key)
        legal = row['legal_config_ids']
        configs = row['configs']
        if (not isinstance(legal, list) or not legal or len(legal) != len(set(legal))
                or any(type(i) is not int or i < 0 for i in legal)
                or len(configs) != len(legal) or {c['id'] for c in configs} != set(legal)):
            raise ValueError('every legal config must be measured')
        times = {'baseline': {}, 'candidate': {}}
        for config in configs:
            cfg = config['lut_config']
            fields = {'kind', 'bm16', 'bn16', 'swizzle', 'wt_m', 'wt_n', 'bk', 'fused', 'threads', 'tile_n'}
            if (set(cfg) != fields or cfg['kind'] != 'Wmma'
                    or any(type(cfg[f]) is not int for f in fields - {'kind'})
                    or any(not 0 < cfg[f] < 256 for f in ('bm16', 'bn16', 'swizzle', 'wt_m', 'wt_n'))
                    or cfg['bk'] not in {16, 32, 64} or cfg['fused'] not in {0, 1}
                    or cfg['threads'] != 0 or cfg['tile_n'] != 0
                    or (key['row_stride'] == 'Padded' and cfg['fused'] != 1)):
                raise ValueError('invalid pinned WMMA config geometry')
            if config['id'] in known_configs and known_configs[config['id']] != cfg:
                raise ValueError('pinned config identity changes between rows')
            known_configs[config['id']] = cfg
            if config.get('cpu_correct') is not True:
                raise ValueError('CPU correctness failed')
            for side in times:
                processes = config[side]
                if len(processes) < 5 or len(processes) % 2 != 1:
                    raise ValueError('an odd count of at least five processes per side is required')
                for blocks in processes:
                    if len(blocks) < 5 or any(isinstance(v, bool) or not isinstance(v, (int, float))
                            or not math.isfinite(v) or v <= 0 for v in blocks):
                        raise ValueError('five finite positive accepted blocks per process are required')
                times[side][config['id']] = statistics.median(statistics.median(b) for b in processes)
        resolution = row['resolution_fraction']
        if not isinstance(resolution, (int, float)) or isinstance(resolution, bool) or not math.isfinite(resolution) or not 0 <= resolution < 1:
            raise ValueError('invalid measured resolution')
        winner = min(times['candidate'], key=times['candidate'].get)
        if min(times['candidate'].values()) > min(times['baseline'].values()) * (1 + resolution):
            raise ValueError('candidate regression exceeds the accepted resolution')
        winners[row['id']] = winner
        config = next(c['lut_config'] for c in configs if c['id'] == winner)
        if config.get('kind') != 'Wmma':
            raise ValueError('this import path requires WMMA-prefill config evidence')
        if row.get('lut_point_index') is not None:
            index = row['lut_point_index']
            if index not in affected or index in replacements:
                raise ValueError('unexpected or duplicate affected LUT point')
            key = {k: v for k, v in old['points'][index].items() if k != 'config'}
            if row['key'] != key:
                raise ValueError('measured key differs from the affected LUT point')
            replacements[index] = config
        if row.get('priority') is True and row.get('add_exact_point') is True:
            additions.append((row['key'], config))
    if set(replacements) != affected or packet.get('accepted_winners') != winners:
        raise ValueError('scope/winners differ from the accepted dataset')
    pool = list(old['configs'])
    def index_of(config):
        answer = _answer_of(config)
        for index, existing in enumerate(pool):
            if _answer_of(existing) == answer:
                return index
        if len(pool) >= 256:
            raise ValueError('config pool exceeds the schema index capacity')
        pool.append(config)
        return len(pool) - 1
    points = [dict(row) for row in old['points']]
    for index, config in replacements.items():
        points[index]['config'] = index_of(config)
    key_to_index = {json.dumps({k: v for k, v in row.items() if k != 'config'}, sort_keys=True): i
                    for i, row in enumerate(points)}
    for key, config in additions:
        encoded = json.dumps(key, sort_keys=True)
        if encoded not in key_to_index:
            key_to_index[encoded] = len(points)
            points.append(dict(key, config=index_of(config)))
    doc = {k: v for k, v in old.items() if k not in ('configs', 'fallbacks', 'points')}
    out = Path(args.out) if args.out else source
    unchanged = all(points[i] == row for i, row in enumerate(old['points']) if i not in affected)
    if not unchanged or pool[:len(old['configs'])] != old['configs']:
        raise ValueError('incremental import changed unrelated points/configs')
    # Validate preservation before touching the destination. A failed write or
    # validation leaves the previous table intact.
    descriptor, name = tempfile.mkstemp(prefix=out.name + '.', suffix='.tmp', dir=out.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        write_lut_json(temporary, doc, pool, old['fallbacks'], points)
        expected = dict(doc, configs=pool, fallbacks=old['fallbacks'], points=points)
        if json.loads(temporary.read_text(encoding='utf-8')) != expected:
            raise ValueError('serialized LUT differs from the validated update')
        temporary.replace(out)
    finally:
        if temporary.exists():
            temporary.unlink()
    receipt = dict(status='PASS', dataset_digest=digest, architecture=args.arch,
                   input_sha256=hashlib.sha256(input_bytes).hexdigest(),
                   output_sha256=hashlib.sha256(out.read_bytes()).hexdigest(),
                   old_point_count=len(old['points']), new_point_count=len(points),
                   affected_point_count=len(affected), added_points=len(points)-len(old['points']),
                   winner_measurement_passes=1, unrelated_points_unchanged=unchanged,
                   unrelated_configs_unchanged=True, weights_unchanged=True,
                   abi_schema_decision='unchanged serialized configs and kernel semantics')
    if args.report:
        Path(args.report).write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
    print('[import-ut] ' + json.dumps(receipt, sort_keys=True))
    return 0


def validate_fast_winner_confirmation(row, source_manifest):
    """Validate a checked winner correction without replacing fast samples.

    The raw auditor establishes clocks, launch identity and source compatibility.
    This importer checks its bound attestation and recomputes acceptance from
    retained blocks. Prepared ranking and padded UT wrapper timings stay separate.
    """
    confirmation = row.get('formal_confirmation')
    if confirmation is None:
        return False
    if not isinstance(confirmation, dict):
        raise ValueError('checked winner confirmation must be an object')
    source_digest = hashlib.sha256(json.dumps(source_manifest, sort_keys=True,
                                              separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()
    selected = next((c for c in row['configs'] if c['id'] == row['winner']), None)
    operation = 'padded_wrapper' if row['key']['row_stride'] == 'Padded' else 'arrival_configured'
    if (selected is None or confirmation.get('source') != 'standard_pinned_ut'
            or confirmation.get('operation') != operation
            or confirmation.get('key') != row['key']
            or confirmation.get('config_id') != row['winner']
            or confirmation.get('lut_config') != selected['lut_config']
            or confirmation.get('final_source_manifest_digest') != source_digest
            or confirmation.get('cpu_correct') is not True
            or confirmation.get('clock_policy') != 'REALTIME_CPU_2pct'
            or confirmation.get('resolution_fraction') != .02
            or not re.fullmatch(r'[0-9a-f]{64}', str(confirmation.get('audited_row_digest', '')))
            or not re.fullmatch(r'[0-9a-f]{64}', str(confirmation.get('raw_audit_digest', '')))):
        raise ValueError('checked winner confirmation identity/acceptance binding is incomplete')
    values = {}
    sides = ['baseline', 'candidate', 'rejected_fast_candidate']
    if 'rejected_fast_baseline' in confirmation:
        if (confirmation.get('baseline_config_id') != row.get('baseline_source_config_id')
                or confirmation.get('rejected_fast_baseline_config_id') != row.get('baseline_source_config_id')):
            raise ValueError('displaced and corrected checks require the same original baseline config')
        sides.append('rejected_fast_baseline')
    for side in sides:
        processes = confirmation.get(side)
        if (not isinstance(processes, list) or len(processes) < 5 or len(processes) % 2 == 0
                or any(not isinstance(blocks, list) or len(blocks) < 5 for blocks in processes)
                or any(not isinstance(v, (int, float)) or isinstance(v, bool)
                       or not math.isfinite(v) or v <= 0 for blocks in processes for v in blocks)):
            raise ValueError('checked winner requires odd five-process, five-block evidence on each side')
        values[side] = statistics.median(statistics.median(blocks) for blocks in processes)
    if any(len(confirmation[side]) != len(confirmation['baseline']) for side in sides):
        raise ValueError('checked winner process counts must match')
    if values['candidate'] / values['baseline'] > 1.02:
        raise ValueError('checked winner has a confirmed regression')
    rejected_id = confirmation.get('rejected_fast_config_id')
    reason = confirmation.get('selection_reason', 'confirmed_regression')
    displaced_baseline = values.get('rejected_fast_baseline', values['baseline'])
    justified = values['rejected_fast_candidate'] / displaced_baseline > 1.02
    if reason == 'priority_checked_best':
        justified = (row.get('priority') is True
                     and values['rejected_fast_candidate'] / values['candidate'] > 1.02)
    elif reason != 'confirmed_regression':
        justified = False
    if (rejected_id == row['winner'] or rejected_id not in row['legal_config_ids']
            or not justified):
        raise ValueError('checked correction requires a confirmed regression or priority improvement')
    return True


def validate_fast_regression_selection(row, source_manifest, original_config,
                                      allow_pending=False):
    """Bind a bounded regression decision without altering ranking samples.

    Pending alternatives are staging tables only. Final checked alternatives use
    formal_confirmation; an incomplete bounded check may retain the original
    production config, as may a byte-identical original kernel.
    """
    selection = row.get('regression_selection')
    if selection is None:
        return False, False
    selected = next((c for c in row['configs'] if c['id'] == row['winner']), None)
    source_digest = hashlib.sha256(json.dumps(source_manifest, sort_keys=True,
                                              separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()
    if (not isinstance(selection, dict) or selected is None
            or selection.get('key') != row['key']
            or selection.get('config_id') != row['winner']
            or selection.get('lut_config') != selected['lut_config']
            or selection.get('final_source_manifest_digest') != source_digest
            or not re.fullmatch(r'[0-9a-f]{64}', str(selection.get('raw_audit_digest', '')))):
        raise ValueError('regression selection source/key/config binding is incomplete')
    if selection.get('kind') == 'pending_regression_fix':
        displaced = selection.get('displaced_config_id')
        if (not allow_pending or displaced == row['winner']
                or displaced not in row['legal_config_ids']
                or selection.get('baseline_config_id') != row.get('baseline_source_config_id')
                or selection.get('cpu_correct') is not True
                or selection.get('clock_policy') != 'REALTIME_CPU_2pct'):
            raise ValueError('pending regression alternative requires explicit staging mode')
        values = {}
        for side in ('baseline', 'candidate'):
            processes = selection.get(side)
            if (not isinstance(processes, list) or len(processes) < 5 or len(processes) % 2 == 0
                    or any(not isinstance(b, list) or len(b) < 5 for b in processes)
                    or any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0
                           for b in processes for v in b)):
                raise ValueError('pending fix requires a complete checked displaced-config loss')
            values[side] = statistics.median(statistics.median(b) for b in processes)
        if len(selection['baseline']) != len(selection['candidate']):
            raise ValueError('pending fix requires matched checked process counts')
        if values['candidate'] / values['baseline'] <= 1.02:
            raise ValueError('pending fix has no confirmed displaced-config regression')
        return True, True
    if (selection.get('kind') != 'retain_original_config'
            or row['winner'] != row.get('baseline_source_config_id')
            or (original_config is not None and _answer_of(selected['lut_config']) != _answer_of(original_config))):
        raise ValueError('bounded fallback must retain the original production config')
    reason = selection.get('reason')
    if reason == 'bounded_confirmation_incomplete':
        if (selection.get('requested_processes') != 5
                or type(selection.get('accepted_processes')) is not int
                or not 0 <= selection['accepted_processes'] < 5
                or selection.get('max_attempts_per_process') != 3):
            raise ValueError('incomplete fallback requires the bounded confirmation receipt')
    elif reason == 'unchanged_kernel_isa':
        payload = selection.get('baseline_payload_sha256')
        if (not re.fullmatch(r'[0-9a-f]{64}', str(payload))
                or payload != selection.get('candidate_payload_sha256')
                or selection.get('metadata_equal') is not True
                or selection.get('launch_geometry_equal') is not True):
            raise ValueError('unchanged-code fallback requires identical ISA and launch geometry')
    else:
        raise ValueError('unsupported bounded regression fallback')
    return True, False


def cmd_import_fast(args) -> int:
    """Build the incremental LUT directly from one accepted fast ranking file.

    This validates config ranking/provenance, not headline performance. The
    final production screen and formal suspect confirmations remain separate.
    """
    if not args.winner_data:
        raise ValueError('import-fast requires --winner-data')
    input_bytes = Path(args.winner_data).read_bytes()
    packet = json.loads(input_bytes)
    data = packet['dataset']
    digest = hashlib.sha256(json.dumps(data, sort_keys=True, separators=(',', ':'),
                                      ensure_ascii=False).encode()).hexdigest()
    if (packet.get('status') != 'FAST_RANKING_ACCEPTED' or packet.get('dataset_digest') != digest
            or data.get('purpose') != 'lut_ranking' or data.get('architecture') != args.arch
            or data.get('measurement_passes') != 1 or data.get('statistic') != 'median'
            or data.get('strategy') != 'settle_once_hold' or not data.get('source_manifest')
            or not data.get('build_identity') or not data.get('device_identity')):
        raise ValueError('single canonical architecture/source/build-bound ranking data required')
    source_manifest = data['source_manifest']
    required_sources = {'lib/Runtime/Kernels/hip/matmul_nbits_kernel.hip',
                        'lib/Runtime/Kernels/include/matmul_nbits_wmma_core.h'}
    if (not required_sources <= set(source_manifest)
            or packet.get('actual_launch_mapping_complete') is not True
            or not re.fullmatch(r'[0-9a-f]{64}', str(packet.get('family_mapping_digest', '')))
            or not isinstance(source_manifest, dict) or any(
            not re.fullmatch(r'[0-9a-f]{64}', str(value)) for value in source_manifest.values())
            or not re.fullmatch(r'[0-9a-f]{64}', str(data['build_identity'].get('candidate_library_sha256', '')))
            or data['device_identity'].get('gcn_arch') != args.arch):
        raise ValueError('verified source/build/device identities are required')
    repo = ROOT.parents[5]
    for name, expected in source_manifest.items():
        relative = Path(name)
        if (relative.is_absolute() or '..' in relative.parts
                or not str(relative).replace('\\', '/').startswith('lib/Runtime/Kernels/')
                or not (repo / relative).is_file()
                or hashlib.sha256((repo / relative).read_bytes()).hexdigest() != expected):
            raise ValueError('canonical dataset source differs from the current kernel tree')
    source = LUT_DIR / f'{args.arch}.json'
    if packet.get('original_lut_sha256') != hashlib.sha256(source.read_bytes()).hexdigest():
        raise ValueError('canonical dataset is already imported or does not match the original LUT')
    old = json.loads(source.read_text(encoding='utf-8'))
    if old.get('gpu_arch') != args.arch:
        raise ValueError('original LUT architecture mismatch')
    required = data['required_ids']
    rows = data['rows']
    if (not required or len(required) != len(set(required)) or len(rows) != len(required)
            or {r['id'] for r in rows} != set(required)):
        raise ValueError('incomplete/duplicate canonical shape coverage')
    affected = set(packet['affected_point_indices'])
    if len(affected) != len(packet['affected_point_indices']):
        raise ValueError('duplicate affected point indices')
    pool = list(old['configs'])
    points = [dict(point) for point in old['points']]
    replacements = set()
    winners = {}
    geometry = {}
    changed = 0
    pending_selections = 0
    def index_of(config):
        ans = _answer_of(config)
        for i, existing in enumerate(pool):
            if _answer_of(existing) == ans:
                return i
        if len(pool) >= 256:
            raise ValueError('config pool exceeds schema capacity')
        pool.append(config)
        return len(pool) - 1
    for row in rows:
        key = row['key']
        if (set(key) != {'phase', 'bits', 'group_size', 'zero_point', 'row_stride', 'm', 'n', 'k'}
                or key['phase'] != 'Prefill' or key['bits'] != 'B4'
                or key['group_size'] not in GROUP_SIZES.values()
                or key['zero_point'] not in {'Symmetric', 'Asymmetric'}
                or key['row_stride'] not in {'Arrival', 'Padded'}
                or any(type(key[d]) is not int or not 0 < key[d] <= limit
                       for d, limit in (('m', 65535), ('n', 4294967295), ('k', 65535)))):
            raise ValueError('invalid measured WMMA-prefill key')
        legal, configs = row['legal_config_ids'], row['configs']
        if not legal or len(legal) != len(set(legal)) or len(configs) != len(legal) or {c['id'] for c in configs} != set(legal):
            raise ValueError('every considered config must appear once')
        medians = {}
        for config in configs:
            cfg = config['lut_config']
            _answer_of(cfg)
            if (cfg['kind'] != 'Wmma' or cfg['bk'] not in {16, 32, 64}
                    or cfg['fused'] not in {0, 1} or cfg['threads'] or cfg['tile_n']
                    or any(type(cfg[f]) is not int or not 0 < cfg[f] < 256
                           for f in ('bm16', 'bn16', 'swizzle', 'wt_m', 'wt_n'))
                    or (key['row_stride'] == 'Padded' and cfg['fused'] != 1)):
                raise ValueError('invalid pinned WMMA geometry')
            if config['id'] in geometry and geometry[config['id']] != cfg:
                raise ValueError('config identity changed between measured rows')
            geometry[config['id']] = cfg
            samples = config['samples']
            if (config.get('cpu_correct') is not True or not isinstance(samples, list) or len(samples) < 3
                    or any(not isinstance(v, (int, float)) or isinstance(v, bool)
                           or not math.isfinite(v) or v <= 0 for v in samples)):
                raise ValueError('valid retained median samples and CPU correctness are required')
            medians[config['id']] = statistics.median(samples)
        winner = row['winner']
        checked = validate_fast_winner_confirmation(row, source_manifest)
        point_index = row.get('lut_point_index')
        original_config = (old['configs'][old['points'][point_index]['config']]
                           if type(point_index) is int and 0 <= point_index < len(old['points']) else None)
        regression_selected, pending = validate_fast_regression_selection(
            row, source_manifest, original_config, getattr(args, 'allow_pending_regression', False))
        pending_selections += int(pending)
        if winner not in medians or (medians[winner] > min(medians.values()) * 1.02 and not checked and not regression_selected):
            raise ValueError('winner differs from canonical median ranking/tie policy')
        winners[row['id']] = winner
        cfg = geometry[winner]
        index = row.get('lut_point_index')
        if index is not None:
            if index not in affected or index in replacements or not 0 <= index < len(points):
                raise ValueError('unexpected/duplicate affected LUT index')
            if key != {k: v for k, v in points[index].items() if k != 'config'}:
                raise ValueError('measured key differs from original LUT point')
            replacements.add(index)
            selected = index_of(cfg)
            changed += points[index]['config'] != selected
            points[index]['config'] = selected
        elif row.get('priority') is True and row.get('add_exact_point') is True:
            if any({k: v for k, v in p.items() if k != 'config'} == key for p in points):
                raise ValueError('priority exact point already exists')
            points.append(dict(key, config=index_of(cfg)))
    if replacements != affected or packet['accepted_winners'] != winners:
        raise ValueError('affected scope/winners differ from canonical dataset')
    if (any(points[i] != p for i, p in enumerate(old['points']) if i not in affected)
            or pool[:len(old['configs'])] != old['configs']):
        raise ValueError('unrelated points/configs changed')
    doc = {k: v for k, v in old.items() if k not in ('configs', 'fallbacks', 'points')}
    out = Path(args.out) if args.out else source
    descriptor, temporary_name = tempfile.mkstemp(prefix=out.name + '.', suffix='.tmp', dir=out.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        write_lut_json(temporary, doc, pool, old['fallbacks'], points)
        assert json.loads(temporary.read_text()) == dict(doc, configs=pool, fallbacks=old['fallbacks'], points=points)
        temporary.replace(out)
    finally:
        if temporary.exists(): temporary.unlink()
    receipt = dict(status='PENDING_PRODUCTION_CONFIRMATION' if pending_selections else 'PASS',
                   pending_regression_selections=pending_selections,
                   dataset_digest=digest, architecture=args.arch,
                   input_sha256=hashlib.sha256(input_bytes).hexdigest(),
                   output_sha256=hashlib.sha256(out.read_bytes()).hexdigest(),
                   old_point_count=len(old['points']), new_point_count=len(points),
                   affected_point_count=len(affected), changed_winners=changed,
                   added_points=len(points)-len(old['points']), winner_measurement_passes=1,
                   import_gpu_measurement_passes=0, unrelated_points_unchanged=True,
                   unrelated_configs_unchanged=True, weights_unchanged=True,
                   abi_schema_decision='unchanged serialized configs and numerical semantics')
    if args.report: Path(args.report).write_text(json.dumps(receipt, indent=2)+'\n')
    print('[import-fast] '+json.dumps(receipt, sort_keys=True))
    return 0


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
    # Only the .fb is checked in; the build turns it into the linkable
    # kMatmulNbitsLutBlobs[] symbol at compile time via cmake/embed_fb.cmake
    # (pure CMake, no Python), so nothing is generated here.
    print(f"[compile] -> {dst} ({dst.stat().st_size} bytes)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("command",
                    choices=["measure", "fit", "build", "compile", "all", "import-ut", "import-fast"])
    ap.add_argument("--winner-data", help="canonical fast winner dataset JSON; import-fast only")
    ap.add_argument("--allow-pending-regression", action="store_true",
                    help="build a staging LUT for already-confirmed loss alternatives; final checks still required")
    ap.add_argument("--accepted-ut", help="accepted pinned standard-UT dataset JSON; import-ut only")
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
    ap.add_argument("--merge-existing", action="store_true",
                    help="build: keep the current table and replace only the "
                         "--phase rows the readings beat by --min-gain")
    ap.add_argument("--phase", default="DecodeDp4a",
                    choices=["Prefill", "Decode", "DecodeDp4a"],
                    help="build --merge-existing: the phase to update")
    ap.add_argument("--min-gain", type=float, default=0.03,
                    help="build --merge-existing: required speedup of the new "
                         "winner over the row's current config")
    ap.add_argument("--report", default=None,
                    help="build --merge-existing: write a markdown diff here")
    ap.add_argument("--update-fallback", action="store_true",
                    help="build --merge-existing: also move the phase fallback; "
                         "it serves only unmeasured group sizes, so check those "
                         "in-model first")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if args.command == "import-fast":
        return cmd_import_fast(args)
    if args.command == "import-ut":
        return cmd_import_ut(args)
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
