#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Interleaved, order-reversed A/B (or A/B/C...) and its paired summary.

Why not run arm A a few times and then arm B: run-to-run spread on a thermally
managed APU is comparable to the effects worth shipping, so block-sequential
runs let drift land on one arm. Interleaving and pairing by round cancels it;
running one block reversed checks that position within a round was not what
got measured. Absolute levels also drift downward over a session as caches
warm, so a delta only means something within one invocation.

Arms come from a JSON manifest. An arm may swap DLLs into HIPEP_BIN, set
environment variables, or both:

    {"base": {"dll": "D:/b/custom_kernels_gfx1151.dll"},
     "cand": {"dll": ["D:/c/custom_kernels_gfx1151.dll", "D:/c/hipgpu.dll"]}}
    {"off": {"env": {"HIPDNN_EP_GQA_FUSE_APPEND": "0"}},
     "on":  {"env": {"HIPDNN_EP_GQA_FUSE_APPEND": "1"}}}

Env keys are unioned over all arms and every key is written for every arm, so
one arm's variables can never leak into the next. Each arm gets its own TEMP,
which keeps HIP's code-object cache separate across a DLL swap.
"""

from __future__ import annotations

import csv
import json
import re
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from . import bench
from .env import HarnessEnv
from .proc import Harness
from .stats import paired

DEFAULT_TAG = r"^ab\d*_(?P<arm>.+)_r(?P<round>\d+)$"
# tps_p50 and tps_p10 compare kernel speed when host stalls swamp the mean;
# p10 needs per-token times, which only the vlm driver records.
METRICS = {
    "ttft": ("ttft_ms", "ms", 0),
    "tps": ("ms_per_token", "ms/tok", 3),
    "tps_p50": ("p50_ms", "ms/tok", 3),
    "tps_p10": ("p10_ms", "ms/tok", 3),
}
DEFAULT_MAX_SLOW_PCT = 5.0


@dataclass
class Arm:
    name: str
    dlls: list[Path]
    env: dict[str, str]


def load_manifest(path: Path, order: list[str] | None = None) -> list[Arm]:
    raw = json.loads(Path(path).read_text())
    names = order or list(raw)
    arms = []
    for n in names:
        if n not in raw:
            raise SystemExit(f"arm {n!r} is not in {path} (have: {', '.join(raw)})")
        d = raw[n]
        dll = d.get("dll") or []
        dlls = [Path(p) for p in ([dll] if isinstance(dll, str) else dll)]
        env = {k: str(v) for k, v in (d.get("env") or {}).items()}
        if not dlls and not env:
            raise SystemExit(f"arm {n!r} defines neither 'dll' nor 'env'")
        for p in dlls:
            if not p.exists():
                raise SystemExit(f"arm {n!r}: DLL not found: {p}")
        arms.append(Arm(n, dlls, env))
    return arms


def run(
    env: HarnessEnv,
    arms: list[Arm],
    template: bench.BenchSpec,
    rounds: int = 3,
    start_round: int = 1,
    reverse: bool = False,
    prime: bool = True,
) -> Path:
    """Run the interleaved rounds; returns the summary CSV they appended to."""
    env.require("bin", "model")
    keys = sorted({k for a in arms for k in a.env})
    cache_root = env.out_root / "tunecache"
    out_dir = template.out_dir or env.out_root / template.metric

    def invoke(arm: Arm, tag: str, reps: int) -> None:
        for dll in arm.dlls:
            shutil.copy2(dll, env.bin / dll.name)
        temp = cache_root / arm.name
        temp.mkdir(parents=True, exist_ok=True)
        spec = bench.BenchSpec(**template.__dict__)
        spec.tag, spec.reps, spec.out_dir = tag, reps, out_dir
        spec.set_env = {
            **template.set_env,
            **{k: arm.env[k] for k in keys if k in arm.env},
        }
        spec.extra_env = {"TEMP": str(temp), "TMP": str(temp)}
        if bench.run(env, spec) is None:
            # A broken setup fails every run the same way; finishing the rounds
            # only produces a CSV with holes that looks like a completed A/B.
            raise SystemExit(f"run {tag} produced no result; A/B aborted")

    with Harness(env.out_root).lock():
        if prime:
            for a in arms:
                print(f"--- priming {a.name} (discarded)")
                invoke(a, f"prime_{a.name}", 1)
        order = list(reversed(arms)) if reverse else arms
        for r in range(start_round, start_round + rounds):
            for a in order:
                print(f"--- round {r} arm {a.name}")
                invoke(
                    a,
                    f"ab_{a.name}_r{r}",
                    template.reps or (4 if template.metric == "ttft" else 3),
                )
    return out_dir / f"{template.metric}_summary.csv"


def summarize(
    csv_path: Path,
    baseline: str,
    metric: str = "ttft",
    drop: list[int] | None = None,
    tag_re: str = DEFAULT_TAG,
    max_slow_pct: float = DEFAULT_MAX_SLOW_PCT,
) -> list[dict]:
    """Print the paired table; returns one dict per non-baseline arm."""
    column, unit, dp = METRICS[metric]
    with open(csv_path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if rows and column not in rows[0]:
        raise SystemExit(
            f"{csv_path} has no {column!r} column; --metric {metric} expects "
            f"the CSV written by the {metric.split('_')[0]} bench"
        )
    rx = re.compile(tag_re)
    runs: dict[int, dict[str, float]] = defaultdict(dict)
    slow: dict[int, float] = defaultdict(float)
    for row in rows:
        m = rx.match(row["tag"])
        if not m:
            continue
        if row[column] == "":
            raise SystemExit(
                f"{row['tag']} has no {column}; the model_benchmark driver records "
                "no per-token times, use --metric tps_p50"
            )
        rnd = int(m.group("round"))
        runs[rnd][m.group("arm")] = float(row[column])
        if row.get("slow_pct"):
            slow[rnd] = max(slow[rnd], float(row["slow_pct"]))
    drop = drop or []
    kept = sorted(r for r in runs if r not in drop)
    if not kept:
        raise SystemExit("no rounds left after --drop")
    arms = [baseline] + sorted({a for r in kept for a in runs[r]} - {baseline})
    print(f"metric: {column} ({unit}); lower is better")
    print("round  " + "".join(f"{a:>12}" for a in arms))
    stalled = sorted(r for r in runs if slow[r] > max_slow_pct)
    for r in sorted(runs):
        cells = "".join(f"{runs[r].get(a, float('nan')):12.{dp}f}" for a in arms)
        note = "   <- dropped" if r in drop else ""
        if r in stalled:
            note += f"   <- stalls: {slow[r]:.0f}% slow tokens"
        print(f"{r:5d}  {cells}{note}")
    if stalled:
        print(
            f"rounds {stalled} have more than {max_slow_pct:g}% stalled tokens in some "
            "arm: host noise, not kernel time. Compare --metric tps_p10/tps_p50, or "
            f"--drop {' '.join(map(str, stalled))} and re-run the rounds."
        )
    print(
        f"\n{'arm':>10} {'n':>3} {'mean':>10} {'vs base':>11} {'sd of diff':>12} "
        f"{'95% CI':>24} {'%':>8}"
    )
    base_all = [runs[r][baseline] for r in kept if baseline in runs[r]]
    print(
        f"{baseline:>10} {len(base_all):3d} {sum(base_all) / max(len(base_all), 1):10.{dp}f} {'--':>11}"
    )
    results = []
    for arm in arms[1:]:
        both = [r for r in kept if arm in runs[r] and baseline in runs[r]]
        if len(both) < 2:
            print(f"{arm:>10} {len(both):3d}  (need >=2 paired rounds)")
            continue
        p = paired([runs[r][baseline] for r in both], [runs[r][arm] for r in both])
        vals = [runs[r][arm] for r in both]
        ci = f"[{p.low:+.{dp}f}, {p.high:+.{dp}f}]"
        verdict = "   (spans zero)" if p.spans_zero else ""
        print(
            f"{arm:>10} {p.n:3d} {sum(vals) / len(vals):10.{dp}f} {p.mean_diff:+11.{dp}f} "
            f"{p.sd_diff:12.{dp}f} {ci:>24} {p.pct:+8.2f}{verdict}"
        )
        results.append(
            {
                "arm": arm,
                "n": p.n,
                "mean_diff": p.mean_diff,
                "low": p.low,
                "high": p.high,
                "pct": p.pct,
                "spans_zero": p.spans_zero,
            }
        )
    return results
