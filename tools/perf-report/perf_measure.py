#!/usr/bin/env python3
"""Reliable perf measurement harness for the HIPDNN/MorphiZen EP.

Motivation: decode TPS on this box (gfx1151, no clock pinning available) has
large run-to-run variance from GPU boost/thermal drift. Single runs and 2-sample
A/Bs mislead. This harness makes results trustworthy by:

  * NOISE-FLOOR mode: launch the model as N independent PROCESSES (each reloads +
    re-autotunes, so process-level variance = the real noise), report
    mean/median/stddev/CoV/95%CI, and the RESOLVABLE-WIN THRESHOLD (~2*stddev).
    Any optimization smaller than that threshold is NOT measurable here.

  * A/B mode: run K INTERLEAVED pairs (A,B,A,B,...). Paired per-pair differences
    cancel slow common-mode drift (the dominant noise). Reports the mean paired
    delta, its 95% CI, and a SIGNIFICANCE VERDICT (CI excludes 0 => real).

Each "sample" is a fresh process (captures the real variance); within a process
we use model_benchmark's own -r/-w to get a steady per-process number.

Config A/B are environment-variable dicts (e.g. B={"HIPDNN_EP_MIOPEN_ACT":"1"}),
so you can toggle a single code path on the SAME build for a clean paired A/B.

Usage:
  python perf_measure.py noise --reps 20
  python perf_measure.py ab --pairs 12 --b-env HIPDNN_EP_MIOPEN_ACT=1
  python perf_measure.py ab --pairs 12 --b-env HIPDNN_EP_MIOPEN_ACT=1,HIPDNN_EP_MIOPEN_NORM=1
"""
import argparse, os, re, statistics as st, subprocess, sys, time

DEF_MODEL = r"D:\Qwen3.5-35B-A3B-fp16-ve-fp16-int4-text-gs32-dml"
DEF_PKG = r"C:\Users\Administrator\workspace\gpu-test-package\bin"
DEF_IBIN = r"C:\Users\Administrator\workspace\install\bin"
DEF_ROCK = r"C:\Users\Administrator\workspace\therock-keep\bin"

TPS_RE = re.compile(r"Token generation:.*?avg \(tokens/s\):\s*([\d.]+)", re.S)
TTFT_RE = re.compile(r"Prompt processing.*?avg \(us\):\s*([\d.]+)", re.S)


def run_once(exe, model, prompt, gen, reps, warmup, extra_env):
    env = dict(os.environ)
    env["PATH"] = f"{DEF_PKG};{DEF_IBIN};{DEF_ROCK};" + env.get("PATH", "")
    # Clean perf/debug flags off unless the caller overrides.
    for k in ("HIPDNN_EP_PERF", "HIPDNN_EP_PERF_ISOLATE", "HIPDNN_EP_QMOE_COARSE",
              "HIPDNN_EP_QMOE_GROUPED", "HIPDNN_EP_QMOE_PARITY",
              "HIPDNN_EP_NORM_PARITY", "HIPDNN_EP_CAST_DIAG",
              "HIPDNN_EP_MIOPEN_ACT", "HIPDNN_EP_MIOPEN_NORM", "HIP_LAUNCH_BLOCKING"):
        env.pop(k, None)
    env.update(extra_env)
    cmd = [exe, "-i", model, "-l", str(prompt), "-g", str(gen),
           "-r", str(reps), "-w", str(warmup)]
    p = subprocess.run(cmd, env=env, capture_output=True, text=True)
    out = p.stdout + p.stderr
    m = TPS_RE.search(out)
    t = TTFT_RE.search(out)
    tps = float(m.group(1)) if m else None
    ttft = float(t.group(1)) / 1000.0 if t else None  # ms
    return tps, ttft, out


def stats(xs):
    n = len(xs)
    mean = st.mean(xs)
    sd = st.stdev(xs) if n > 1 else 0.0
    se = sd / (n ** 0.5) if n > 1 else 0.0
    return dict(n=n, mean=mean, median=st.median(xs), sd=sd, se=se,
                cov=100 * sd / mean if mean else 0, lo=min(xs), hi=max(xs),
                ci95=1.96 * se)


def parse_env(s):
    d = {}
    if not s:
        return d
    for kv in s.split(","):
        kv = kv.strip()
        if not kv:
            continue
        k, _, v = kv.partition("=")
        d[k.strip()] = v.strip() if v else "1"
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["noise", "ab"])
    ap.add_argument("--exe", default=os.path.join(DEF_PKG, "model_benchmark.exe"))
    ap.add_argument("--model", default=DEF_MODEL)
    ap.add_argument("--prompt", type=int, default=128)
    ap.add_argument("--gen", type=int, default=128)
    ap.add_argument("--reps", type=int, default=20, help="noise: #processes")
    ap.add_argument("--pairs", type=int, default=12, help="ab: #interleaved pairs")
    ap.add_argument("--inner-reps", type=int, default=3, help="model_benchmark -r")
    ap.add_argument("--warmup", type=int, default=2, help="model_benchmark -w")
    ap.add_argument("--drop", type=int, default=1, help="drop first N process samples (cold)")
    ap.add_argument("--a-env", default="")
    ap.add_argument("--b-env", default="")
    ap.add_argument("--metric", choices=["tps", "ttft"], default="tps")
    args = ap.parse_args()

    def sample(extra_env):
        tps, ttft, out = run_once(args.exe, args.model, args.prompt, args.gen,
                                  args.inner_reps, args.warmup, extra_env)
        val = tps if args.metric == "tps" else ttft
        if val is None:
            tail = "\n".join(out.strip().splitlines()[-3:])
            print(f"  [FAIL] no {args.metric} parsed. tail: {tail}", flush=True)
        return val

    unit = "tok/s" if args.metric == "tps" else "ms"
    if args.mode == "noise":
        aenv = parse_env(args.a_env)
        print(f"[noise] {args.reps} process reps, prompt={args.prompt} gen={args.gen} "
              f"inner -r{args.inner_reps} -w{args.warmup}, env={aenv or 'clean'}")
        xs = []
        for i in range(args.reps):
            v = sample(aenv)
            if v is not None:
                xs.append(v)
                print(f"  rep {i+1:2d}: {v:.2f} {unit}", flush=True)
        xs = xs[args.drop:]
        s = stats(xs)
        print(f"\n[noise floor] {args.metric} over n={s['n']} (dropped {args.drop} cold):")
        print(f"  mean={s['mean']:.2f}  median={s['median']:.2f}  stddev={s['sd']:.2f}  "
              f"CoV={s['cov']:.1f}%  min={s['lo']:.2f}  max={s['hi']:.2f}")
        print(f"  95% CI of mean = +/-{s['ci95']:.2f} {unit}")
        print(f"  RESOLVABLE-WIN THRESHOLD ~= 2*stddev = {2*s['sd']:.2f} {unit} "
              f"({200*s['sd']/s['mean']:.1f}% of mean) -- smaller deltas are noise.")
        return

    # A/B interleaved paired
    aenv, benv = parse_env(args.a_env), parse_env(args.b_env)
    print(f"[ab] {args.pairs} interleaved pairs, prompt={args.prompt} gen={args.gen}, "
          f"metric={args.metric}\n  A env={aenv or 'clean'}\n  B env={benv or 'clean'}")
    a_vals, b_vals, diffs = [], [], []
    for i in range(args.pairs):
        a = sample(aenv)
        b = sample(benv)
        if a is None or b is None:
            print(f"  pair {i+1:2d}: FAILED, skipping"); continue
        d = b - a  # positive = B faster (tps) / B slower (ttft)
        a_vals.append(a); b_vals.append(b); diffs.append(d)
        print(f"  pair {i+1:2d}: A={a:.2f}  B={b:.2f}  B-A={d:+.2f}", flush=True)
    if len(diffs) < 2:
        print("not enough pairs"); return
    sa, sb, sd = stats(a_vals), stats(b_vals), stats(diffs)
    print(f"\n[A/B result] {args.metric} ({unit})")
    print(f"  A: mean={sa['mean']:.2f} +/- {sa['ci95']:.2f} (CoV {sa['cov']:.1f}%)")
    print(f"  B: mean={sb['mean']:.2f} +/- {sb['ci95']:.2f} (CoV {sb['cov']:.1f}%)")
    pct = 100 * sd['mean'] / sa['mean'] if sa['mean'] else 0
    print(f"  paired delta (B-A): mean={sd['mean']:+.2f}  95% CI=+/-{sd['ci95']:.2f}  "
          f"({pct:+.1f}% vs A)")
    sig = abs(sd['mean']) > sd['ci95'] and sd['ci95'] > 0
    print(f"  VERDICT: {'SIGNIFICANT (CI excludes 0)' if sig else 'NOT significant (within noise)'}")


if __name__ == "__main__":
    main()
