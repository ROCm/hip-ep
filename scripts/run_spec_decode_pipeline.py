#!/usr/bin/env python3
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""EAGLE-3 / MTP speculative-decoding enablement pipeline for hip-ep.

Brings up a speculative-decoding stack on the hip-ep EP and measures it, in the
order you actually want when enabling a new target or a new draft head:

    preflight -> export -> sanity -> bench -> summarize

Each stage is a thin orchestration of the `tools/eagle3` harness, which is not
part of this repository -- it lives in the speculative-decoding workspace and is
located with `--harness` or `$SPEC_HARNESS`. Nothing is reimplemented here; the
value is the ordering, the preconditions and the presets.

Two stages exist because of how expensive the last one is. A full Spec-Bench
bake-off is hours: the 27B bake-off in Phase T ran 48 instances per process and
most of the wall clock is hip-ep compiling one autotune key per novel prompt
length. Discovering a mistyped head directory or a target graph without the
right aux taps at hour three is the failure this pipeline exists to prevent, so
`preflight` checks every file the run will touch and `sanity` does a six-prompt
run of every arm first.

Presets encode the Phase S and Phase T configurations, including the per-arm
policies. Those are not defaults anyone would guess: MTP wants `mp2` on the 9B
and `mp6` on the 27B, EAGLE-3 wants `cap2` except for the specdrift head, which
wants `mp3`. They came out of the break-even curves and are reproduced here so a
re-run means the same thing as the recorded results.

Usage:
    # what would run, without running it
    python scripts/run_spec_decode_pipeline.py --preset qwen36-27b --dry-run

    # bring up the 9B stack and check every arm loads
    python scripts/run_spec_decode_pipeline.py --preset qwen35-9b \
        --harness ../tools/eagle3 --models-root .. --stages preflight,sanity

    # the full Phase S bake-off
    python scripts/run_spec_decode_pipeline.py --preset qwen35-9b \
        --harness ../tools/eagle3 --models-root .. \
        --json results/phase_s.json
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

STAGES = ("preflight", "export", "sanity", "bench", "summarize")
DEFAULT_STAGES = ("preflight", "sanity", "bench", "summarize")

# Paths are relative to --models-root. The 9B keeps its target graph, MTP head,
# embedding table and tokenizer in one published directory; the 27B splits the
# tokenizer and embedding table out into the MTP source drop.
PRESETS: dict[str, dict] = {
    "qwen35-9b": {
        "describe": "Qwen3.5-9B, three EAGLE-3 heads vs MTP (Phase S section 4)",
        "target_dir": "models/amd_hipep_qwen35_9b",
        "target": "text_hipep_spec.onnx",
        "mtp": "mtp.onnx",
        "mtp_dir": None,
        "embed": "models/amd_hipep_qwen35_9b/embed_tokens_fp16.npy",
        "tokenizer_dir": "models/amd_hipep_qwen35_9b",
        "heads": {
            "blr2_plain": "models/qwen3.5-9b-eagle3-onnx-blr2_plain-int4-all",
            "incumbent": "models/qwen3.5-9b-eagle3-onnx-incumbent-int4-all",
            "blr2_sharegpt": "models/qwen3.5-9b-eagle3-onnx-blr2_sharegpt-int4-all",
        },
        "arms": "none,mtp:1:mp2,eagle3@blr2_plain:1:cap2,"
                "eagle3@incumbent:1:cap2,eagle3@blr2_sharegpt:1:cap2",
        "per_task": 20,
        "max_new": 128,
        "max_context": 1664,
    },
    "qwen36-27b": {
        "describe": "Qwen3.6-27B, incumbent + PRISM-full EAGLE-3 vs MTP (Phase T P1)",
        "target_dir": "models/qwen3.6-27b-int4-block-128",
        "target": "model_hipep_eagle3.onnx",
        "mtp": "mtp.onnx",
        "mtp_dir": None,
        "embed": "models/qwen3.6-27b-mtp-src/embed_tokens_fp16.npy",
        "tokenizer_dir": "models/qwen3.6-27b-mtp-src",
        "heads": {
            "incumbent": "models/qwen3.6-27b-eagle3-onnx-incumbent-int4-all",
            "prism_full": "models/qwen3.6-27b-eagle3-onnx-prism_full-int4-all",
        },
        "arms": "none,mtp:1:mp6,eagle3@incumbent:1:cap2,eagle3@prism_full:1:cap2",
        "per_task": 8,
        "max_new": 128,
        "max_context": 1792,
    },
    # A separate preset rather than a flag: the specdrift head taps L4/32/60 and
    # the other two tap L2/32/61, so it needs a different target graph and cannot
    # share a process with them. MTP is carried in both so the two processes can
    # be read against a common arm.
    "qwen36-27b-specdrift": {
        "describe": "Qwen3.6-27B, specdrift EAGLE-3 vs MTP (Phase T P2)",
        "target_dir": "models/qwen3.6-27b-int4-block-128",
        "target": "model_hipep_eagle3_sd.onnx",
        "mtp": "mtp.onnx",
        "mtp_dir": None,
        "embed": "models/qwen3.6-27b-mtp-src/embed_tokens_fp16.npy",
        "tokenizer_dir": "models/qwen3.6-27b-mtp-src",
        "heads": {
            "specdrift": "models/qwen3.6-27b-eagle3-onnx-specdrift-int4-all",
        },
        "arms": "none,mtp:1:mp6,eagle3@specdrift:1:mp3",
        "per_task": 8,
        "max_new": 128,
        "max_context": 1792,
    },
}


class Ctx:
    """Resolved paths and knobs shared by every stage."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.preset = PRESETS[args.preset]
        self.harness = Path(args.harness).resolve()
        self.models = Path(args.models_root).resolve()
        self.python = args.python or sys.executable
        self.dry_run = args.dry_run
        self.ep = args.ep
        self.iobinding = args.iobinding
        self.arms = args.arms or self.preset["arms"]
        self.per_task = args.per_task if args.per_task is not None \
            else self.preset["per_task"]
        self.max_new = args.max_new if args.max_new is not None \
            else self.preset["max_new"]
        self.questions = Path(args.questions).resolve() if args.questions else \
            self.models / "data" / "spec_bench" / "question.jsonl"
        self.json = Path(args.json).resolve() if args.json else None
        self.extra = args.extra or []

    def m(self, rel: str) -> Path:
        return self.models / rel

    @property
    def bench_script(self) -> Path:
        return self.harness / "qwen36_spec_bench.py"

    def bench_cmd(self, per_task: int, max_new: int,
                  json_path: Path | None) -> list[str]:
        """The one command both `sanity` and `bench` run, at different sizes."""
        p = self.preset
        head_dirs = ",".join(f"{tag}={self.m(d)}" for tag, d in p["heads"].items())
        cmd = [
            self.python, str(self.bench_script),
            "--target-dir", str(self.m(p["target_dir"])),
            "--target", p["target"],
            "--mtp", p["mtp"],
            "--embed", str(self.m(p["embed"])),
            "--tokenizer-dir", str(self.m(p["tokenizer_dir"])),
            "--questions", str(self.questions),
            "--arms", self.arms,
            "--per-task", str(per_task),
            "-n", str(max_new),
            "--max-context", str(p["max_context"]),
            "--ep", self.ep,
        ]
        if p["mtp_dir"]:
            cmd += ["--mtp-dir", str(self.m(p["mtp_dir"]))]
        if self.iobinding:
            cmd += ["--iobinding"]
        if json_path:
            cmd += ["--json", str(json_path)]
        return cmd + self.extra


def show(cmd: list[str]) -> str:
    return " ".join(shlex.quote(c) for c in cmd)


def run(ctx: Ctx, cmd: list[str], label: str) -> bool:
    print(f"  $ {show(cmd)}", flush=True)
    if ctx.dry_run:
        return True
    t0 = time.perf_counter()
    rc = subprocess.call(cmd)
    dt = (time.perf_counter() - t0) / 60
    if rc != 0:
        print(f"  {label} FAILED (exit {rc}) after {dt:.1f} min")
        return False
    print(f"  {label} ok in {dt:.1f} min")
    return True


# ---------------------------------------------------------------- stages


def stage_preflight(ctx: Ctx) -> bool:
    """Check every file the bench will open, before spending hours finding out.

    Reports all problems rather than stopping at the first, since the usual case
    is a model drop that is half in place.
    """
    p = ctx.preset
    checks: list[tuple[str, Path]] = [
        ("harness", ctx.bench_script),
        ("questions", ctx.questions),
        ("target graph", ctx.m(p["target_dir"]) / p["target"]),
        ("embedding table", ctx.m(p["embed"])),
        ("tokenizer dir", ctx.m(p["tokenizer_dir"])),
    ]
    if "mtp" in ctx.arms:
        mtp_dir = ctx.m(p["mtp_dir"]) if p["mtp_dir"] else ctx.m(p["target_dir"])
        checks.append(("mtp head", mtp_dir / p["mtp"]))
    for tag, d in p["heads"].items():
        if f"eagle3@{tag}" in ctx.arms:
            checks.append((f"eagle3 head {tag}", ctx.m(d)))

    ok = True
    for label, path in checks:
        good = path.exists()
        ok &= good
        print(f"  {'OK  ' if good else 'MISS'}  {label:<26} {path}")
    if not ok:
        print("  preflight found missing inputs; fix these before --stages bench")
    return ok


def stage_export(ctx: Ctx) -> bool:
    """Build the MTP head if it is not already next to the target.

    Only MTP is built here. An EAGLE-3 head is the output of a training and
    quantization pipeline with model-specific knobs (aux taps, rope mode, vocab
    pruning), so exporting one is not a step this orchestrator can honestly
    default; if a head directory is missing, preflight says so and the export
    command belongs in the head's own recipe.
    """
    p = ctx.preset
    mtp_dir = ctx.m(p["mtp_dir"]) if p["mtp_dir"] else ctx.m(p["target_dir"])
    out = mtp_dir / p["mtp"]
    if out.exists():
        print(f"  MTP head already present: {out}")
        return True
    script = ctx.harness / "qwen36_mtp_export.py"
    if not script.exists() and not ctx.dry_run:
        print(f"  missing exporter: {script}")
        return False
    return run(ctx, [
        ctx.python, str(script),
        "--target-dir", str(ctx.m(p["target_dir"])),
        "--target", p["target"],
        "--out", p["mtp"],
    ], "export")


def stage_sanity(ctx: Ctx) -> bool:
    """One instance per subtask, few tokens: does every arm load and decode?

    Six prompts is enough to fault a bad head directory, a target graph without
    the aux taps an EAGLE-3 arm needs, or an EP that silently fell back to CPU,
    and it costs minutes rather than hours. The numbers it prints are not worth
    reading -- too few tokens, and every prompt pays its own autotune.
    """
    return run(ctx, ctx.bench_cmd(per_task=1, max_new=16, json_path=None),
               "sanity")


def stage_bench(ctx: Ctx) -> bool:
    if ctx.json is None:
        print("  --json is required for the bench stage")
        return False
    ctx.json.parent.mkdir(parents=True, exist_ok=True)
    return run(ctx, ctx.bench_cmd(ctx.per_task, ctx.max_new, ctx.json), "bench")


def stage_summarize(ctx: Ctx) -> bool:
    """Re-print the tables from the recorded JSON.

    Separate from `bench` so a finished run can be re-read without the GPU, and
    so an interrupted one can still be summarized from whatever it wrote (the
    bench writes after every question).
    """
    if ctx.json is None:
        print("  --json is required for the summarize stage")
        return False
    if not ctx.json.exists() and not ctx.dry_run:
        print(f"  no results at {ctx.json}")
        return False
    return run(ctx, [ctx.python, str(ctx.bench_script),
                     "--summarize-only", "--json", str(ctx.json)], "summarize")


HANDLERS = {
    "preflight": stage_preflight,
    "export": stage_export,
    "sanity": stage_sanity,
    "bench": stage_bench,
    "summarize": stage_summarize,
}


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", choices=sorted(PRESETS), required=True)
    ap.add_argument("--harness", default=os.environ.get("SPEC_HARNESS", "tools/eagle3"),
                    help="directory holding qwen36_spec_bench.py; "
                         "defaults to $SPEC_HARNESS")
    ap.add_argument("--models-root", default=os.environ.get("SPEC_MODELS_ROOT", "."),
                    help="root the preset's models/... paths resolve against; "
                         "defaults to $SPEC_MODELS_ROOT")
    ap.add_argument("--stages", default=",".join(DEFAULT_STAGES),
                    help=f"comma-separated subset of {','.join(STAGES)}")
    ap.add_argument("--json", default=None, help="results file for bench/summarize")
    ap.add_argument("--arms", default=None, help="override the preset's arms")
    ap.add_argument("--per-task", type=int, default=None,
                    help="instances per subtask (six subtasks, 80 available each)")
    ap.add_argument("--max-new", type=int, default=None)
    ap.add_argument("--questions", default=None)
    ap.add_argument("--ep", default="hipgpu", choices=["cpu", "hipgpu", "amdgpu"])
    ap.add_argument("--no-iobinding", dest="iobinding", action="store_false")
    ap.add_argument("--python", default=None,
                    help="interpreter for the harness; defaults to this one")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the commands each stage would run and exit 0")
    ap.add_argument("--list-presets", action="store_true")
    ap.add_argument("extra", nargs="*",
                    help="extra flags passed through to qwen36_spec_bench.py")
    args = ap.parse_args()

    if args.list_presets:
        for name, p in sorted(PRESETS.items()):
            print(f"{name:<22} {p['describe']}")
            print(f"{'':<22} arms: {p['arms']}")
        return 0

    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    bad = [s for s in stages if s not in STAGES]
    if bad:
        print(f"unknown stage(s) {bad}; want {','.join(STAGES)}", file=sys.stderr)
        return 2

    ctx = Ctx(args)
    print(f"preset      {args.preset} -- {ctx.preset['describe']}")
    print(f"harness     {ctx.harness}")
    print(f"models      {ctx.models}")
    print(f"arms        {ctx.arms}")
    print(f"ep          {ctx.ep}{' +iobinding' if ctx.iobinding else ''}")
    print(f"stages      {','.join(stages)}"
          f"{'   (dry run)' if ctx.dry_run else ''}")

    results: list[tuple[str, bool]] = []
    for name in stages:
        print(f"\n--- {name} " + "-" * (58 - len(name)))
        ok = HANDLERS[name](ctx)
        results.append((name, ok))
        if not ok:
            print(f"\nstopping: {name} failed")
            break

    print("\n" + "=" * 62)
    for name, ok in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print("=" * 62)
    return 0 if all(ok for _, ok in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
