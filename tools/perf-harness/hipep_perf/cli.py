#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Command line: ``python -m hipep_perf <command> ...`` (see README.md)."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

from . import ab, bench
from .env import HarnessEnv, parse_pairs

ANALYSES = (
    "attrib_regions",
    "calibrate_sqtt",
    "capture_diff",
    "decode_model",
    "expert_blocks",
    "headroom",
    "host_timeline",
    "inspect_capture",
    "lut_query",
    "prefill_model",
    "shape_breakdown",
    "trace_ops",
    "roctx_attrib",
)


def _bench_args(p: argparse.ArgumentParser, metric: str | None = None) -> None:
    p.add_argument("--tag", required=metric is not None)
    p.add_argument(
        "--driver", choices=("model_benchmark", "vlm"), default="model_benchmark"
    )
    p.add_argument("--seqlen", type=int)
    p.add_argument("--prompt-file", type=Path)
    p.add_argument("--gen", type=int)
    p.add_argument("--reps", type=int)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--max-tokens", type=int, default=4, help="vlm ttft only")
    p.add_argument("--max-length", type=int, help="vlm only: KV cache size")
    p.add_argument("--ep", default="follow_config", help="vlm only: execution provider")
    p.add_argument("--set-env", action="append", metavar="K=V", help="one run only")
    p.add_argument("--out-dir", type=Path)


def _spec(a: argparse.Namespace, metric: str) -> bench.BenchSpec:
    pf = a.prompt_file.resolve() if a.prompt_file else None
    if pf and not pf.exists():
        raise SystemExit(f"prompt file not found: {pf}")
    return bench.BenchSpec(
        tag=a.tag or "run",
        metric=metric,
        driver=a.driver,
        seqlen=a.seqlen,
        prompt_file=pf,
        gen=a.gen,
        reps=a.reps,
        warmup=a.warmup,
        max_tokens=a.max_tokens,
        max_length=a.max_length,
        execution_provider=a.ep,
        set_env=parse_pairs(a.set_env),
        out_dir=a.out_dir,
    )


def _fence(a: argparse.Namespace):
    from .traced import FenceSpec

    if not getattr(a, "fence_op", None):
        return None
    return FenceSpec(
        a.fence_op,
        a.fence_skip,
        a.fence_after_runs,
        a.fence_ms,
        a.fence_graph,
        a.sqtt_marker,
    )


def _fence_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--fence-op", help="fence on hip_ep.op.<name>")
    p.add_argument(
        "--fence-skip", type=int, default=0, help="instances to let pass first"
    )
    p.add_argument(
        "--fence-after-runs",
        type=int,
        default=0,
        help="hold off until this subgraph Run index (picks the phase)",
    )
    p.add_argument("--fence-ms", type=int, default=200)
    p.add_argument(
        "--fence-graph",
        default="",
        help="substring of the hip_ep.graph label that counts Runs (default: any 'sub=0')",
    )
    p.add_argument(
        "--sqtt-marker",
        action="store_true",
        help="launch hipep_trace_sqtt_marker at the fence (KTRACE_SQTT_KERNEL trigger)",
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="hipep_perf", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("bench", help="one clean TTFT or TPS measurement")
    b.add_argument("metric", choices=("ttft", "tps"))
    _bench_args(b, "x")
    b.add_argument(
        "--perf-ops", nargs="+", help="per-family event timing (throughput invalid)"
    )

    abp = sub.add_parser("ab", help="interleaved A/B")
    absub = abp.add_subparsers(dest="abcmd", required=True)
    r = absub.add_parser("run")
    r.add_argument("--manifest", type=Path, required=True)
    r.add_argument("--metric", choices=("ttft", "tps"), default="ttft")
    r.add_argument("--arms", nargs="+")
    r.add_argument("--rounds", type=int, default=3)
    r.add_argument("--start-round", type=int, default=1)
    r.add_argument("--reverse", action="store_true")
    r.add_argument("--skip-prime", action="store_true")
    _bench_args(r)
    s = absub.add_parser("summary")
    s.add_argument("csv", type=Path)
    s.add_argument("--baseline", required=True)
    s.add_argument("--metric", choices=("ttft", "tps"), default="ttft")
    s.add_argument("--drop", type=int, nargs="*", default=[])
    s.add_argument("--tag-re", default=ab.DEFAULT_TAG)
    s.add_argument("--json", type=Path)

    t = sub.add_parser("trace", help="run under the trace shim / post-process")
    tsub = t.add_subparsers(dest="tcmd", required=True)
    tr = tsub.add_parser("run")
    tr.add_argument("metric", choices=("ttft", "tps"))
    _bench_args(tr, "x")
    _fence_args(tr)
    tr.add_argument("--out", type=Path, required=True)
    tr.add_argument("--no-api", action="store_true")
    tc = tsub.add_parser("cmd", help="trace an arbitrary command")
    tc.add_argument("--out", type=Path, required=True)
    tc.add_argument("--no-api", action="store_true")
    _fence_args(tc)
    tc.add_argument("command", nargs=argparse.REMAINDER)
    tp = tsub.add_parser("report")
    tp.add_argument("out", type=Path)
    tp.add_argument("--api-slices", action="store_true")
    tf = tsub.add_parser(
        "finalize-ktrace",
        help="rocprofv3 export of a ktrace raw spool (what ktrace_postprocess finalize writes)",
    )
    tf.add_argument("spool", type=Path, help="ktrace's .<prefix>_<pid>_ops.csv")
    tf.add_argument("--out", type=Path, required=True)
    tf.add_argument("--prefix", default="trace")
    tk = tsub.add_parser("compare", help="compare a shim trace with a ktrace capture")
    tk.add_argument("out", type=Path, help="shim trace directory")
    tk.add_argument(
        "--ktrace", type=Path, required=True, help="ktrace output directory"
    )
    tk.add_argument("--prefix", default="trace", help="ktrace KTRACE_OUTPUT_FILE")
    tk.add_argument("--json", type=Path)

    c = sub.add_parser("capture", help="RGP capture positioned by the shim fence")
    c.add_argument("metric", choices=("ttft", "tps"))
    _bench_args(c, "x")
    _fence_args(c)
    c.add_argument("--counters", action="store_true", help="SPM counters")
    c.add_argument("--buf", default="minimum")
    c.add_argument("--op-count", type=int, default=3000)
    c.add_argument("--instructions", action="store_true")
    c.add_argument("--dwell", type=int, default=90)
    c.add_argument("--decode-step", type=int, help="convert to --fence-after-runs")
    c.add_argument("--prefill-chunk", type=int, default=512)

    g = sub.add_parser("rgp", help="decode .rgp captures")
    gsub = g.add_subparsers(dest="gcmd", required=True)
    gd = gsub.add_parser("decode")
    gd.add_argument("capture", type=Path)
    gd.add_argument("out_base", type=Path)
    gv = gsub.add_parser("verify")
    gv.add_argument("capture", type=Path)
    gv.add_argument("--need-spm", action="store_true")
    gp = gsub.add_parser("parity", help="compare against reference CSVs")
    gp.add_argument("corpus", type=Path, help="dir with rgp/ and ref/")
    gp.add_argument("--out", type=Path, required=True)

    k = sub.add_parser("calib", help="ground-truth calibration workload")
    ksub = k.add_subparsers(dest="kcmd", required=True)
    kb = ksub.add_parser("build")
    kb.add_argument("--arch", default="")
    ks = ksub.add_parser("score", help="score any rocprofv3-shaped kernel_trace.csv")
    ks.add_argument("--kernel-trace", type=Path, required=True)
    ks.add_argument(
        "--workload", type=Path, required=True, help="the workload's JSON stdout"
    )
    kg = ksub.add_parser("rgp", help="RGP-capture the workload and score the decoder")
    kg.add_argument("--out", type=Path, required=True)
    kg.add_argument("--iters", type=int, default=20)
    kg.add_argument("--counters", action="store_true", help="also check SPM bytes")
    kr = ksub.add_parser("check", help="trace the workload and score timings vs truth")
    kr.add_argument("--out", type=Path, required=True)
    kr.add_argument("--mb", type=int, default=256)
    kr.add_argument("--iters", type=int, default=20)

    an = sub.add_parser(
        "analysis", help="analysis scripts (pass --help after the name)"
    )
    an.add_argument("name", choices=ANALYSES)
    an.add_argument("rest", nargs=argparse.REMAINDER)

    lg = sub.add_parser("ledger", help="prediction ledger")
    lsub = lg.add_subparsers(dest="lcmd", required=True)
    lp = lsub.add_parser("predict")
    lp.add_argument("--id", required=True)
    lp.add_argument("--metric", choices=("ttft", "tps"), required=True)
    lp.add_argument(
        "--predicted-pct",
        type=float,
        required=True,
        help="predicted change in the metric (negative = faster)",
    )
    lp.add_argument("--source", default="", help="what produced the prediction")
    lp.add_argument("--note", default="")
    lm = lsub.add_parser("measure")
    lm.add_argument("--id", required=True)
    lm.add_argument(
        "--summary-json", type=Path, required=True, help="output of 'ab summary --json'"
    )
    lm.add_argument("--arm", required=True)
    lsub.add_parser("report")

    a = ap.parse_args(argv)
    env = HarnessEnv.resolve()

    if a.cmd == "bench":
        spec = _spec(a, a.metric)
        if a.perf_ops:
            logs = bench.run_perf_ops(env, spec, a.perf_ops)
            print(
                "calibrate with: python -m hipep_perf analysis calibrate_sqtt <dispatches.csv> "
                + " ".join(f"--perf-log {p}" for p in logs)
            )
            return 0
        return 0 if bench.run(env, spec) else 1

    if a.cmd == "ab":
        if a.abcmd == "run":
            arms = ab.load_manifest(a.manifest, a.arms)
            csv_path = ab.run(
                env,
                arms,
                _spec(a, a.metric),
                a.rounds,
                a.start_round,
                a.reverse,
                not a.skip_prime,
            )
            print(
                f"\nsummarise with: python -m hipep_perf ab summary {csv_path} "
                f"--metric {a.metric} --baseline {arms[0].name}"
            )
            return 0
        res = ab.summarize(a.csv, a.baseline, a.metric, a.drop, a.tag_re)
        if a.json:
            a.json.write_text(
                json.dumps(
                    {
                        "csv": str(a.csv),
                        "metric": a.metric,
                        "baseline": a.baseline,
                        "arms": res,
                    },
                    indent=2,
                )
            )
        return 0

    if a.cmd == "trace":
        from . import traced

        if a.tcmd == "report":
            print(traced.postprocess(a.out, api_slices=a.api_slices).read_text())
            return 0
        if a.tcmd == "finalize-ktrace":
            from . import ktrace_ref, rocprof

            for p in rocprof.write(ktrace_ref.load_spool(a.spool), a.out, a.prefix):
                print(f"wrote {p}")
            return 0
        if a.tcmd == "compare":
            from . import ktrace_ref, trace

            res = ktrace_ref.compare(
                trace.load(a.out), ktrace_ref.load(a.ktrace, a.prefix)
            )
            if a.json:
                a.json.write_text(json.dumps(res, indent=2, default=list))
            return 0
        if a.tcmd == "cmd":
            cmd = a.command[1:] if a.command[:1] == ["--"] else a.command
            if not cmd:
                raise SystemExit("trace cmd needs a command after --")
            fence = _fence(a)
            meta = traced.run_command(
                env,
                cmd,
                a.out,
                fence.env() if fence else None,
                api=not a.no_api,
                cwd=Path.cwd(),
            )
        else:
            meta = traced.run_bench(
                env, _spec(a, a.metric), a.out, _fence(a), api=not a.no_api
            )
        print(
            json.dumps(
                {
                    k: meta[k]
                    for k in (
                        "records",
                        "dropped",
                        "attached",
                        "tracer",
                        "api",
                        "api_detail",
                        "exit_code",
                    )
                    if k in meta
                }
            )
        )
        print(traced.postprocess(a.out).read_text())
        return 0 if meta.get("exit_code", 1) == 0 else 1

    if a.cmd == "capture":
        from . import capture

        return capture.main(env, a, _spec(a, a.metric))

    if a.cmd == "rgp":
        from .rgp import cli as rgp_cli

        return rgp_cli.dispatch(a)

    if a.cmd == "calib":
        from . import calib

        return calib.dispatch(env, a)

    if a.cmd == "analysis":
        mod = importlib.import_module(f"hipep_perf.analysis.{a.name}")
        sys.argv = [a.name, *a.rest]
        return int(mod.main() or 0)

    if a.cmd == "ledger":
        from . import ledger

        return ledger.dispatch(env, a)
    return 2
