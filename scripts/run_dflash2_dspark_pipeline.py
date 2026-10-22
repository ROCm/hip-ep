#!/usr/bin/env python3
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""DSpark / DFlash2 block-drafter enablement pipeline for Qwen3.8-27B on hip-ep.

Also carries a `qwen36-27b` preset: Qwen3.6-27B has no official DSpark or DFlash2, so it
runs the z-lab DFlash and a community DSpark instead (see DRAFTERS_Q36).

Brings up the two block-diffusion drafters on the hip-ep EP and measures them
against the target alone and against Qwen's own MTP head, in the order you
actually want when enabling a new target or a new drafter:

    preflight -> export -> sanity -> bench -> summarize        (+ validate)

This is the sibling of `run_spec_decode_pipeline.py`, which covers EAGLE-3 and
MTP. The drafters differ in ways that change the stages: they are converted from
published checkpoints rather than trained here, they need a five-layer residual
tap on the target, and they share the target's lm_head through a hard link, so
`export` has three jobs where the EAGLE-3 pipeline has one.

Each stage is a thin orchestration of the `tools/eagle3` harness, which is not
part of this repository -- it lives in the speculative-decoding workspace and is
located with `--harness` or `$SPEC_HARNESS`, or found automatically when this
checkout sits inside the workspace as `$WS/hip-ep`. Nothing is reimplemented
here; the value is the ordering, the preconditions, the presets and a clean
environment.

Stages (always run in this order, whatever order `--stages` lists them in):
    preflight  every file the run will open, whether the target graph really
               carries the taps the arms need, and whether each drafter was
               exported for the layers that graph taps
    export     target tap graph, then each drafter's W4A16 ONNX graph; add
               `--fetch` to download the checkpoints at their pinned revisions
    sanity     one instance per subtask, 16 tokens, every arm
    bench      the Spec-Bench run, paired per prompt, JSON after every question
    summarize  re-print the tables from that JSON, no GPU
    validate   (not in the default set) record trajectories, then check each
               drafter's ONNX graph against the fp32 reference and replay its
               acceptance on hip-ep

The environment is cleaned for every child process: the hip-ep tuning levers are
removed (a leaked one silently changes what is measured), and
`HIPDNN_EP_STRICT=1` is set so a graph that cannot compile fully on the GPU
fails instead of falling back.

The MTP head is a prerequisite, not something this script builds: `mtp.onnx`
must already sit beside the target graph (see the EAGLE-3 pipeline).

`bench` refuses to start on an existing results file unless `--resume` (passed
through to the bench) or `--overwrite` says what to do with it, because the
bench otherwise starts a fresh document and the hours already recorded are lost.

Usage (from $WS/hip-ep; --harness, --models-root and --hipep-build are only
needed when the checkout is not inside the workspace):
    # what is there, in seconds
    python scripts/run_dflash2_dspark_pipeline.py --preset qwen38-27b --stages preflight

    # build everything from the pinned checkpoints
    python scripts/run_dflash2_dspark_pipeline.py --preset qwen38-27b \
        --fetch --stages preflight,export

    # check every arm loads
    python scripts/run_dflash2_dspark_pipeline.py --preset qwen38-27b \
        --stages preflight,sanity

    # the full 240-prompt Spec-Bench run, and continuing it after an interruption
    python scripts/run_dflash2_dspark_pipeline.py --preset qwen38-27b \
        --json results/qwen38_spec_bench.json
    python scripts/run_dflash2_dspark_pipeline.py --preset qwen38-27b \
        --json results/qwen38_spec_bench.json \
        --stages bench,summarize -- --resume
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

STAGES = ("preflight", "export", "sanity", "bench", "summarize", "validate")
DEFAULT_STAGES = ("preflight", "sanity", "bench", "summarize")

REPO = Path(__file__).resolve().parents[1]


def _workspace() -> Path | None:
    """The speculative-decoding workspace, when this checkout is `$WS/hip-ep`."""
    ws = REPO.parent
    return ws if (ws / "tools" / "eagle3").is_dir() else None


WORKSPACE = _workspace()


def _workspace_build() -> str | None:
    """`<workspace>/build/hip-ep`, when it is a hip-ep build tree."""
    build = WORKSPACE / "build" / "hip-ep" if WORKSPACE else None
    return str(build) if build and (build / "bin").is_dir() else None


# The EP plugin each --ep loads, as (Windows, POSIX) file names.
EP_LIBS = {
    "hipgpu": ("hipgpu.dll", "libhipgpu.so"),
    "amdgpu": ("amdgpu-ep.dll", "libamdgpu-ep.so"),
}


def _workspace_python() -> str | None:
    """The workspace virtual environment's interpreter, which has the harness's deps."""
    if WORKSPACE is None:
        return None
    exe = "Scripts/python.exe" if os.name == "nt" else "bin/python"
    py = WORKSPACE / ".venv" / exe
    return str(py) if py.exists() else None


# Modules the harness imports, per stage. A wrong interpreter otherwise passes
# preflight and sanity and only fails when export or validate reaches them.
STAGE_MODULES = {
    "export": ("onnx", "numpy", "safetensors"),
    "sanity": ("onnx", "onnxruntime", "numpy", "tokenizers"),
    "bench": ("onnx", "onnxruntime", "numpy", "tokenizers"),
    "validate": ("onnx", "onnxruntime", "numpy", "tokenizers", "safetensors"),
}

# hip-ep levers that change kernel selection or autotune behaviour. Every one is
# cleared for the children, because an inherited value turns a default-mode run
# into a differently tuned one without any visible difference in the output.
TUNING_VARS = (
    "HIPDNN_EP_DEBUG",
    "HIPDNN_EP_PERF",
    "HIPDNN_EP_MATMUL_DP4A",
    "HIPDNN_EP_MATMUL_DP4A_MAX_M",
    "HIPDNN_EP_MATMUL_SMALLM",
    "HIPDNN_EP_MATMUL_SMALLM_I8",
    "HIPDNN_EP_MATMUL_SMALLM_CFG",
    "HIPDNN_EP_MATMUL_SMALLM_I8_CFG",
    "HIPDNN_EP_MATMUL_TIER_TUNE",
    "HIPDNN_EP_MATMUL_WMMA_MIN_M",
    "HIPDNN_MATMUL_AUTOTUNE_MODE",
    "HIPDNN_GQA_AUTOTUNE_MODE",
)

# The drafters are published checkpoints, pinned to the revisions the recorded
# results were produced with. `dir` and `src` are relative to the preset's
# spec_dir; the harness looks the exported graph up as `<spec_dir>/<dir>`.
DRAFTERS: dict[str, dict] = {
    "dspark": {
        "repo": "RadixArk/Qwen3.8-27B-DSpark",
        "revision": "b9a5dbdf03bc999c6c73c426b19c2d9041cea393",
        "files": ("config.json", "model.safetensors"),
        "src": "src/dspark",
        "dir": "dspark-w4a16",
        "graph": "dspark.onnx",
        "host_files": (),
    },
    "dflash2": {
        "repo": "incoai/Qwen3.8-27B-DFlash2",
        "revision": "015e795645c74b1a0eeef3b570031fb62e769bc5",
        "files": ("config.json", "model.safetensors"),
        "src": "src/dflash2",
        "dir": "dflash2-w4a16",
        "graph": "dflash2.onnx",
        "host_files": ("predecessor_codebook.npy", "successor_codebook.npy"),
    },
}
# Qwen3.6-27B has no official DFlash2 or DSpark. What exists, and what the recorded
# results used (Phase X): the authors' plain DFlash, a community DSpark in the DeepSpec
# layout, and a community "DFlash2" that its card says was fine-tuned from the Qwen3.8
# drafter. `fetch_qwen36_block_drafters.py` records the revision it downloaded in a
# REVISION file; `revision` here is that value. Drafters trained for a 16/15-row block
# are served truncated (`--draft-block`) because verify is capped at eight rows.
DRAFTERS_Q36: dict[str, dict] = {
    "dflash": {
        "repo": "z-lab/Qwen3.6-27B-DFlash",
        "revision": "0919688658996800f86b895034249700e9481106",
        "files": ("config.json", "model.safetensors"),
        "src": "src/dflash",
        "dir": "dflash-w4a16",
        "graph": "dflash.onnx",
        "host_files": (),
        "export_args": ("--draft-block", "8"),
    },
    "dspark": {
        "repo": "satgeze/Qwen3.6-27B-DSpark",
        "revision": "47d8de04dd814f4a727bf3fc6a85613d2a3dcaa1",
        "files": ("config.json", "model.safetensors"),
        "src": "src/dspark",
        "dir": "dspark-w4a16",
        "graph": "dspark.onnx",
        "host_files": (),
        "export_args": ("--draft-block", "7"),
    },
    "dflash2": {
        "repo": "Fluxmire/Qwen3.6-27B-DFlash2",
        "revision": "2aaf558b35c26aa94bebfff87bfef7822e450e25",
        # the Hub file name; the exporter wants it called model.safetensors
        "files": ("config.json", "model-00001-of-00001.safetensors"),
        "src": "src/dflash2",
        "dir": "dflash2-w4a16",
        "graph": "dflash2.onnx",
        "host_files": ("predecessor_codebook.npy", "successor_codebook.npy"),
        "export_args": (),
    },
}
ARM_KINDS = ("mtp", "dspark", "dflash", "dflash2")

# Paths are relative to --models-root. The target and its tokenizer share one
# published directory; the embedding table comes from the MTP source drop.
_QWEN38 = {
    "target_dir": "models/amd_hipep_qwen38_27b",
    "base_graph": "text.onnx",
    "target": "text_hipep_all.onnx",
    "mtp": "mtp.onnx",
    "embed": "models/qwen3.8-27b-mtp-src/embed_tokens_fp16.npy",
    "tokenizer_dir": "models/amd_hipep_qwen38_27b",
    "spec_dir": "models/qwen3.8-27b-spec",
    "per_task": 40,
    "max_new": 128,
    "max_context": 1792,
    "drafters": DRAFTERS,
    # Unpruned lm_head scores every verify position; the hidden taps feed the MTP head
    # and the DFlash context tap feeds both block drafters, so one graph serves every
    # arm and the baseline runs on identical weights. The Qwen3.8 graph is already
    # repaired and GQA, so those two rewrites are off.
    "fixups_args": [
        "--no-repair-mask-ref",
        "--no-mha-to-gqa",
        "--unprune-lm-head",
        "--add-hidden-taps",
        "--add-dflash-context",
    ],
}
PRESETS: dict[str, dict] = {
    "qwen38-27b": {
        **_QWEN38,
        "describe": "Qwen3.8-27B, DSpark and DFlash2 at k=3 and 5 vs MTP",
        "arms": "none,mtp:2,dspark:3,dspark:5,dflash2:3,dflash2:5",
    },
    # Single-drafter presets rather than a flag: each loads only the drafter it
    # measures, which halves the weights resident next to the 13 GB target.
    "qwen38-27b-dflash2": {
        **_QWEN38,
        "describe": "Qwen3.8-27B, DFlash2 alone at k=3 and 5",
        "arms": "none,dflash2:3,dflash2:5",
    },
    "qwen38-27b-dspark": {
        **_QWEN38,
        "describe": "Qwen3.8-27B, DSpark alone at k=3 and 5",
        "arms": "none,dspark:3,dspark:5",
    },
}


# Qwen3.6-27B is a hybrid (48 Gated DeltaNet + 16 full-attention layers). Its AMD graph is
# the block-128 `model.onnx`; the tap graph also needs the mask-reference repair and the
# MHA->GQA rewrite, which qwen36_fixups.py does by default. The drafters tap the residual
# entering layers 2,17,32,47,62 (the checkpoints' "after layer" ids 1,16,31,46,61, plus
# one); the Qwen3.8 DFlash2 taps are the fixups default 6,20,34,48,62.
_QWEN36 = {
    "target_dir": "models/qwen3.6-27b-int4-block-128",
    "base_graph": "model.onnx",
    "target": "model_hipep_all.onnx",
    "mtp": "mtp.onnx",
    "embed": "models/qwen3.6-27b-mtp-src/embed_tokens_fp16.npy",
    "tokenizer_dir": "models/qwen3.6-27b-int4-block-128",
    "spec_dir": "models/qwen3.6-27b-spec",
    "per_task": 40,
    "max_new": 128,
    "max_context": 1792,
    "drafters": DRAFTERS_Q36,
    "fixups_args": [
        "--add-hidden-taps",
        "--unprune-lm-head",
        "--add-dflash-context",
        "--dflash-layers",
        "2,17,32,47,62",
    ],
}
PRESETS["qwen36-27b"] = {
    **_QWEN36,
    "describe": "Qwen3.6-27B, DFlash (z-lab) and DSpark (satgeze) at k=3 and 5 vs MTP",
    "arms": "none,mtp:1:mp6,dflash:3,dflash:5,dspark:3,dspark:5",
}
# The Fluxmire DFlash2 was trained on Qwen3.8 hidden states, so it needs its own tap graph
# (the Qwen3.8 layers) and is kept out of the main preset. Phase X records a slowdown.
PRESETS["qwen36-27b-dflash2"] = {
    **_QWEN36,
    "describe": "Qwen3.6-27B, community DFlash2 on Qwen3.8 taps; a no-go, kept for the record",
    "target": "model_hipep_all_d2.onnx",
    "fixups_args": ["--add-hidden-taps", "--unprune-lm-head", "--add-dflash-context"],
    "arms": "none,dflash2:3,dflash2:5",
}


def arm_kinds(arms: str) -> set[str]:
    """Drafter kinds named by an arm list, e.g. `none,mtp:2,dflash2:5`."""
    kinds = {a.split(":")[0].strip() for a in arms.split(",") if a.strip()}
    kinds.discard("none")
    bad = sorted(kinds - set(ARM_KINDS))
    if bad:
        raise SystemExit(
            f"unknown arm kind(s) {bad}; want none, " + ", ".join(ARM_KINDS)
        )
    return kinds


class Ctx:
    """Resolved paths and knobs shared by every stage."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.preset = PRESETS[args.preset]
        self.harness = Path(args.harness).resolve()
        self.models = Path(args.models_root).resolve()
        self.python = args.python or _workspace_python() or sys.executable
        self.stages: list[str] = []
        self.bench_ran = False
        self.dry_run = args.dry_run
        self.ep = args.ep
        self.iobinding = args.iobinding
        self.fetch = args.fetch
        self.arms = args.arms or self.preset["arms"]
        kinds = arm_kinds(self.arms)
        self.mtp = "mtp" in kinds
        self.drafters = [k for k in ARM_KINDS if k in kinds and k != "mtp"]
        missing = [k for k in self.drafters if k not in self.preset["drafters"]]
        if missing:
            raise SystemExit(
                f"preset {args.preset} has no {missing} drafter; "
                f"it offers {sorted(self.preset['drafters'])}"
            )
        p = self.preset
        self.per_task = args.per_task if args.per_task is not None else p["per_task"]
        self.max_new = args.max_new if args.max_new is not None else p["max_new"]
        self.max_context = args.max_context or p["max_context"]
        self.questions = (
            Path(args.questions).resolve()
            if args.questions
            else self.models / "data" / "spec_bench" / "question.jsonl"
        )
        self.report_prompt = (
            Path(args.report_prompt).resolve()
            if args.report_prompt
            else self.models / "design" / "data" / "dflash_report_prompt.txt"
        )
        self.json = Path(args.json).resolve() if args.json else None
        self.overwrite = args.overwrite
        self.extra = args.extra or []
        self.hipep_build = (
            Path(args.hipep_build).resolve() if args.hipep_build else None
        )
        self.env = self._child_env(args)

    def _child_env(self, args: argparse.Namespace) -> dict[str, str]:
        env = dict(os.environ)
        if not args.keep_env:
            for var in TUNING_VARS:
                env.pop(var, None)
        if not args.no_strict:
            env["HIPDNN_EP_STRICT"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        if self.hipep_build:
            env["HIPEP_EP_BIN"] = str(self.hipep_build / "bin")
            rocm = self.hipep_build / "_therock" / "bin"
            env["PATH"] = str(rocm) + os.pathsep + env.get("PATH", "")
        return env

    def m(self, rel: str) -> Path:
        return self.models / rel

    def spec(self, kind: str) -> dict:
        """The preset's catalogue entry for one drafter kind."""
        return self.preset["drafters"][kind]

    @property
    def target_dir(self) -> Path:
        return self.m(self.preset["target_dir"])

    @property
    def target_graph(self) -> Path:
        return self.target_dir / self.preset["target"]

    @property
    def spec_dir(self) -> Path:
        return self.m(self.preset["spec_dir"])

    def draft_dir(self, kind: str) -> Path:
        return self.spec_dir / self.spec(kind)["dir"]

    def draft_src(self, kind: str) -> Path:
        return self.spec_dir / self.spec(kind)["src"]

    def draft_outputs(self, kind: str) -> list[tuple[str, Path]]:
        """Every file one drafter's export leaves behind, and the runtime opens."""
        d = self.spec(kind)
        out = self.draft_dir(kind)
        files = [
            ("graph", out / d["graph"]),
            ("graph data", out / f"{d['graph']}.data"),
            ("config", out / "draft_config.json"),
            ("lm_head link", out / "target.onnx.data"),
        ]
        return files + [(name, out / name) for name in d["host_files"]]

    @property
    def target_data(self) -> Path:
        """The target's external weights, which every tap graph and drafter shares."""
        return self.target_dir / f"{self.preset['base_graph']}.data"

    def needed_taps(self) -> list[tuple[str, str]]:
        """Graph outputs the requested arms read, with a label for each."""
        needs = []
        if self.drafters:
            needs.append(("dflash_context", "the DFlash context tap"))
        if self.mtp:
            needs.append(("mtp_hidden_prenorm", "the MTP hidden tap"))
        return needs

    def tool(self, name: str) -> Path:
        return self.harness / name

    @property
    def bench_script(self) -> Path:
        return self.tool("qwen38_spec_bench.py")

    def common(self) -> list[str]:
        """Target, embedding, tokenizer and EP flags every runner takes."""
        p = self.preset
        cmd = [
            "--target-dir",
            str(self.target_dir),
            "--target",
            p["target"],
            "--embed",
            str(self.m(p["embed"])),
            "--tokenizer-dir",
            str(self.m(p["tokenizer_dir"])),
            "--ep",
            self.ep,
        ]
        return cmd + (["--iobinding"] if self.iobinding else [])

    def bench_cmd(
        self, per_task: int, max_new: int, json_path: Path | None
    ) -> list[str]:
        """The one command both `sanity` and `bench` run, at different sizes."""
        p = self.preset
        cmd = [
            self.python,
            str(self.bench_script),
            "--spec-dir",
            str(self.spec_dir),
            "--mtp",
            p["mtp"],
            "--questions",
            str(self.questions),
            "--arms",
            self.arms,
            "--per-task",
            str(per_task),
            "-n",
            str(max_new),
            "--max-context",
            str(self.max_context),
        ]
        cmd += self.common()
        if json_path:
            cmd += ["--json", str(json_path)]
        return cmd + self.extra


def show(cmd: list[str]) -> str:
    """The command as this platform's shell would take it, so it can be pasted."""
    return subprocess.list2cmdline(cmd) if os.name == "nt" else shlex.join(cmd)


def run(ctx: Ctx, cmd: list[str], label: str) -> bool:
    print(f"  $ {show(cmd)}", flush=True)
    if ctx.dry_run:
        return True
    t0 = time.perf_counter()
    rc = subprocess.call(cmd, env=ctx.env)
    dt = (time.perf_counter() - t0) / 60
    if rc != 0:
        print(f"  {label} FAILED (exit {rc}) after {dt:.1f} min")
        return False
    print(f"  {label} ok in {dt:.1f} min")
    return True


def graph_has(path: Path, name: str) -> bool:
    """Whether a structure-only .onnx names a tensor.

    The fallback when onnx is not importable. The tap graph is ~1.4 MB because
    the weights live in an external data file, so a byte search is cheap, but it
    only proves the name occurs somewhere, not that it is a graph output.
    """
    try:
        return name.encode() in path.read_bytes()
    except OSError:
        return False


def read_taps(path: Path) -> tuple[set[str], list[int] | None] | None:
    """A tap graph's outputs and the layers its `dflash_context` concatenates.

    None when onnx is unavailable or the graph does not parse; the caller then
    falls back to `graph_has` and cannot check layers.
    """
    try:
        import onnx
    except ImportError:
        return None
    try:
        model = onnx.load(str(path), load_external_data=False)
    except Exception:
        return None
    outputs = {o.name for o in model.graph.output}
    for node in model.graph.node:
        if "dflash_context" in node.output:
            found = [re.search(r"/layers\.(\d+)/", name) for name in node.input]
            return outputs, [int(m.group(1)) for m in found if m]
    return outputs, None


def draft_layers(config: Path) -> list[int] | None:
    """The target layers a drafter's context path expects, from `draft_config.json`.

    `context_layers` names them directly. Older exports only record the
    checkpoint's `target_layer_ids`, which count the layer whose output is
    tapped, so the residual entering the next layer is that id plus one.
    """
    try:
        cfg = json.loads(config.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if cfg.get("context_layers"):
        return [int(x) for x in cfg["context_layers"]]
    ids = cfg.get("target_layer_ids")
    return [int(x) + 1 for x in ids] if ids else None


def missing_taps(ctx: Ctx) -> list[str]:
    """Labels of the taps the arms need that the target graph does not export."""
    taps = read_taps(ctx.target_graph)
    if taps is None:
        return [w for n, w in ctx.needed_taps() if not graph_has(ctx.target_graph, n)]
    return [w for n, w in ctx.needed_taps() if n not in taps[0]]


# ---------------------------------------------------------------- stages


def stage_preflight(ctx: Ctx) -> bool:
    """Check every file the bench will open, before spending hours finding out.

    Reports all problems rather than stopping at the first, since the usual case
    is a model drop that is half in place. When `export` runs in the same
    invocation, what it builds is reported as TODO rather than missing, and its
    own inputs (the checkpoints) are checked instead.
    """
    p = ctx.preset
    exporting = "export" in ctx.stages
    # (label, path, export builds it)
    checks: list[tuple[str, Path, bool]] = [
        ("harness: spec bench", ctx.bench_script, False),
        ("harness: tap fixups", ctx.tool("qwen36_fixups.py"), False),
        ("harness: drafter export", ctx.tool("dflash_export.py"), False),
        ("questions", ctx.questions, False),
        ("base target graph", ctx.target_dir / p["base_graph"], False),
        ("target tap graph", ctx.target_graph, True),
        ("embedding table", ctx.m(p["embed"]), False),
        ("tokenizer dir", ctx.m(p["tokenizer_dir"]), False),
    ]
    if ctx.mtp:
        checks.append(("mtp head", ctx.target_dir / p["mtp"], False))
    for kind in ctx.drafters:
        outputs = ctx.draft_outputs(kind)
        checks += [(f"{kind} {label}", path, True) for label, path in outputs]
        if exporting and not all(path.exists() for _, path in outputs):
            # --fetch downloads it, so it only has to exist without --fetch
            src = ctx.draft_src(kind) / "model.safetensors"
            checks.append((f"{kind} checkpoint", src, ctx.fetch))
    gpu_stages = {"sanity", "bench", "validate"} & set(ctx.stages)
    if ctx.hipep_build and ctx.ep in EP_LIBS:
        lib = EP_LIBS[ctx.ep][0 if os.name == "nt" else 1]
        checks.append(("hip-ep EP library", ctx.hipep_build / "bin" / lib, False))
        checks.append(("hip-ep ROCm bin", ctx.hipep_build / "_therock" / "bin", False))

    ok = True
    todo = False
    for label, path, built in checks:
        if path.exists():
            status = "OK  "
        elif built and exporting:
            status, todo = "TODO", True
        else:
            status, ok = "MISS", False
        print(f"  {status}  {label:<32} {path}")
    if todo:
        print("  TODO items are built by the export stage of this run")
    modules = sorted({m for s in ctx.stages for m in STAGE_MODULES.get(s, ())})
    if modules:
        probe = subprocess.run(
            [ctx.python, "-c", f"import {', '.join(modules)}"],
            capture_output=True,
            text=True,
        )
        good = probe.returncode == 0
        ok &= good
        why = "" if good else ": " + (probe.stderr.strip().splitlines() or ["?"])[-1]
        print(
            f"  {'OK  ' if good else 'MISS'}  {'harness python':<32} {ctx.python}{why}"
        )
    if ctx.hipep_build and ctx.ep in EP_LIBS:
        lib = ctx.hipep_build / "bin" / EP_LIBS[ctx.ep][0 if os.name == "nt" else 1]
        if lib.exists():
            built = time.strftime("%Y-%m-%d %H:%M", time.localtime(lib.stat().st_mtime))
            print(f"  INFO  {'EP library built':<32} {built}")
    elif ctx.ep in EP_LIBS and gpu_stages:
        # The harness then searches install trees and stale copies of the EP can
        # win, without the ROCm runtime on PATH; the failure is a page of JIT
        # errors at session creation.
        ok = False
        print(
            f"  MISS  {'hip-ep build':<32} pass --hipep-build <build tree> or set "
            "$HIP_EP_BUILD (no <workspace>/build/hip-ep found)"
        )

    # A graph that exists but lacks a tap fails an hour in, on the first arm
    # that reads it, so look inside it as well.
    if ctx.target_graph.exists():
        missing = missing_taps(ctx)
        for _, what in ctx.needed_taps():
            good = what not in missing
            ok &= good
            print(
                f"  {'OK  ' if good else 'MISS'}  {what:<32} "
                f"{'in ' if good else 'not in '}{ctx.target_graph.name}"
            )

    # A drafter fed the wrong layers still loads, decodes and matches the
    # baseline token for token; only its acceptance drops. Nothing downstream
    # raises, so compare the layers the graph taps with the ones each drafter
    # was exported for.
    taps = read_taps(ctx.target_graph) if ctx.target_graph.exists() else None
    graph_layers = taps[1] if taps else None
    for kind in ctx.drafters:
        want = draft_layers(ctx.draft_dir(kind) / "draft_config.json")
        label = f"{kind} tap layers"
        if graph_layers is None or want is None:
            if ctx.target_graph.exists():
                print(
                    f"  SKIP  {label:<32} cannot read layers (needs onnx and both files)"
                )
            continue
        good = want == graph_layers
        ok &= good
        print(
            f"  {'OK  ' if good else 'MISS'}  {label:<32} drafter {want}, "
            f"graph {graph_layers}"
        )

    # The lm_head is shared with the target, never copied. The link must be the
    # same file as this target's weights: a copy costs 13 GB, and a link to a
    # different target's data is the wrong lm_head.
    for kind in ctx.drafters:
        link = ctx.draft_dir(kind) / "target.onnx.data"
        if not (link.exists() and ctx.target_data.exists()):
            continue
        if not os.path.samefile(link, ctx.target_data):
            ok = False
            print(
                f"  MISS  {kind + ' lm_head link':<32} not a hard link to {ctx.target_data}"
            )

    # The root may not exist yet on a first export, so ask the nearest parent.
    probe = ctx.models
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    free = shutil.disk_usage(probe).free / 1e9
    low = free < 10
    print(f"  {'WARN' if low else 'OK  '}  {'free disk':<32} {free:.0f} GB")
    if not ok:
        print(
            "  preflight found missing inputs. Missing graphs or drafters: add export "
            "to --stages (and --fetch for checkpoints). Anything else: see the doc's "
            "section 2."
        )
    return ok


def fetch_checkpoint(ctx: Ctx, kind: str) -> bool:
    """Download one drafter's checkpoint at its pinned revision."""
    d = ctx.spec(kind)
    src = ctx.draft_src(kind)
    print(f"  fetch {d['repo']} @ {d['revision'][:8]} -> {src}")
    if ctx.dry_run:
        for name in d["files"]:
            print(
                f"  $ hf_hub_download({d['repo']!r}, {name!r}, "
                f"revision={d['revision']!r}, local_dir={str(src)!r})"
            )
        return True
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("  --fetch needs huggingface_hub (pip install huggingface_hub)")
        return False
    for name in d["files"]:
        got = Path(
            hf_hub_download(d["repo"], name, revision=d["revision"], local_dir=str(src))
        )
        # the exporter reads `model.safetensors`, whatever the Hub file is called
        if name.endswith(".safetensors") and name != "model.safetensors":
            os.replace(got, src / "model.safetensors")
    return True


def stage_export(ctx: Ctx) -> bool:
    """Build what is missing: the target tap graph, then each drafter.

    Everything is skipped when it is already in place, so a re-run only fills
    gaps. Nothing here writes into the target's own files: the tap graph is a
    new structure-only file beside `text.onnx` that shares its external data,
    and each drafter reaches the target's lm_head through a hard link.
    """
    p = ctx.preset
    if ctx.target_graph.exists():
        # Skipping a graph built with other flags would leave preflight failing on
        # it forever, and rebuilding it here could clobber a hand-made one.
        missing = missing_taps(ctx)
        if missing:
            print(
                f"  {ctx.target_graph} exists but lacks {', '.join(missing)}; "
                "it was built with other flags. Delete it and re-run export."
            )
            return False
        print(f"  target tap graph already present: {ctx.target_graph}")
    else:
        # Unpruned lm_head scores every verify position; the hidden taps feed the
        # MTP head and the DFlash context tap feeds both block drafters, so one
        # graph serves every arm and the baseline runs on identical weights.
        fixups = [
            ctx.python,
            str(ctx.tool("qwen36_fixups.py")),
            "--in-dir",
            str(ctx.target_dir),
            "--in-model",
            p["base_graph"],
            "--out",
            p["target"],
            *p["fixups_args"],
        ]
        if not run(ctx, fixups, "tap graph"):
            return False

    for kind in ctx.drafters:
        d = ctx.spec(kind)
        out = ctx.draft_dir(kind)
        absent = [label for label, path in ctx.draft_outputs(kind) if not path.exists()]
        if not absent:
            print(f"  {kind} already exported: {out}")
            continue
        if len(absent) < len(ctx.draft_outputs(kind)):
            print(
                f"  {kind} export is partial (no {', '.join(absent)}); exporting again"
            )
        src = ctx.draft_src(kind)
        if not (src / "model.safetensors").exists():
            if not ctx.fetch:
                print(f"  {kind}: no checkpoint at {src}; pass --fetch to download it")
                return False
            if not fetch_checkpoint(ctx, kind):
                return False
        export = [
            ctx.python,
            str(ctx.tool("dflash_export.py")),
            "--src",
            str(src),
            "--out-dir",
            str(out),
            "--target-dir",
            str(ctx.target_dir),
            "--target",
            p["target"],
            *d.get("export_args", ()),
        ]
        if not run(ctx, export, f"{kind} export"):
            return False
    return True


def stage_sanity(ctx: Ctx) -> bool:
    """One instance per subtask, few tokens: does every arm load and decode?

    Six prompts is enough to fault a missing tap, a drafter graph that does not
    compile on the GPU, or an EP that silently fell back to CPU, and it costs
    minutes rather than hours. The numbers it prints are not worth reading --
    too few tokens, and every prompt pays its own autotune.
    """
    return run(ctx, ctx.bench_cmd(per_task=1, max_new=16, json_path=None), "sanity")


def stage_bench(ctx: Ctx) -> bool:
    if ctx.json is None:
        print("  --json is required for the bench stage")
        return False
    # Without --resume the bench starts a fresh document and rewrites the file
    # after the first prompt, so an existing run would be lost.
    if ctx.json.exists() and "--resume" not in ctx.extra and not ctx.overwrite:
        print(
            f"  {ctx.json} already exists; pass `-- --resume` to continue it, "
            "or --overwrite to replace it"
        )
        return ctx.dry_run
    ctx.json.parent.mkdir(parents=True, exist_ok=True)
    ctx.bench_ran = run(
        ctx, ctx.bench_cmd(ctx.per_task, ctx.max_new, ctx.json), "bench"
    )
    return ctx.bench_ran


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
    if ctx.bench_ran and not ctx.dry_run:
        print(f"  the bench above printed the tables; results in {ctx.json}")
        return True
    return run(
        ctx,
        [
            ctx.python,
            str(ctx.bench_script),
            "--summarize-only",
            "--json",
            str(ctx.json),
        ],
        "summarize",
    )


def stage_validate(ctx: Ctx) -> bool:
    """Record trajectories, then check each drafter against the reference.

    Spec-Bench says how fast a drafter is; this says whether the exported graph
    still computes the drafter it was exported from. Trajectories are plain
    greedy decodes of the tapped target that keep the context features, so
    every later step is a replay and needs no new decode. Existing trajectories
    are reused.

        parity   ONNX drafter vs the fp32 numpy reference, with the reference
                 given the same int4-rounded weights, so what is left is export
                 error rather than quantisation error
        replay   acceptance by depth of the ONNX drafter on hip-ep
    """
    ref = ctx.spec_dir / "ref"
    trajs = {"phasem": [], "report": ["--prompt-file", str(ctx.report_prompt)]}
    paths: dict[str, Path] = {}
    for tag, prompt in trajs.items():
        out = ref / f"traj_{tag}.npz"
        if tag == "report" and not ctx.report_prompt.exists() and not ctx.dry_run:
            print(
                f"  no report prompt at {ctx.report_prompt}; skipping that trajectory"
            )
            continue
        paths[tag] = out
        if out.exists():
            print(f"  trajectory present: {out}")
            continue
        cmd = [ctx.python, str(ctx.tool("dflash_capture.py")), *ctx.common()]
        cmd += ["--chat", "-n", "256", "--out", str(out), *prompt]
        if not ctx.dry_run:
            ref.mkdir(parents=True, exist_ok=True)
        if not run(ctx, cmd, f"capture {tag}"):
            return False

    parity_traj = paths.get("report") or paths.get("phasem")
    if parity_traj is None:
        print("  no trajectory to check against")
        return False
    for kind in ctx.drafters:
        parity = [
            ctx.python,
            str(ctx.tool("dflash_parity.py")),
            "--draft-dir",
            str(ctx.draft_dir(kind)),
            "--src",
            str(ctx.draft_src(kind)),
            "--traj",
            str(parity_traj),
            "--ref-quant",
            "--target-dir",
            str(ctx.target_dir),
            "--target",
            ctx.preset["target"],
            "--embed",
            str(ctx.m(ctx.preset["embed"])),
            "--ep",
            ctx.ep,
        ]
        parity += ["--iobinding"] if ctx.iobinding else []
        if not run(ctx, parity, f"{kind} parity"):
            return False
        replay = [
            ctx.python,
            str(ctx.tool("dflash_runner.py")),
            "--draft-dir",
            str(ctx.draft_dir(kind)),
            *ctx.common(),
            "--replay",
            *[str(p) for p in paths.values()],
        ]
        if not run(ctx, replay, f"{kind} replay"):
            return False
    return True


HANDLERS = {
    "preflight": stage_preflight,
    "export": stage_export,
    "sanity": stage_sanity,
    "bench": stage_bench,
    "summarize": stage_summarize,
    "validate": stage_validate,
}


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--preset", choices=sorted(PRESETS))
    ap.add_argument(
        "--harness",
        default=os.environ.get(
            "SPEC_HARNESS",
            str(WORKSPACE / "tools" / "eagle3") if WORKSPACE else "tools/eagle3",
        ),
        help="directory holding qwen38_spec_bench.py; defaults to $SPEC_HARNESS, "
        "else <workspace>/tools/eagle3 when this checkout is <workspace>/hip-ep",
    )
    ap.add_argument(
        "--models-root",
        default=os.environ.get(
            "SPEC_MODELS_ROOT", str(WORKSPACE) if WORKSPACE else "."
        ),
        help="root the preset's models/... paths resolve against; defaults to "
        "$SPEC_MODELS_ROOT, else the workspace this checkout sits in",
    )
    ap.add_argument(
        "--hipep-build",
        default=os.environ.get("HIP_EP_BUILD") or _workspace_build(),
        help="hip-ep build tree (bin/ and _therock/bin/); sets HIPEP_EP_BIN and "
        "PATH for the children; defaults to $HIP_EP_BUILD, else "
        "<workspace>/build/hip-ep",
    )
    ap.add_argument(
        "--stages",
        default=",".join(DEFAULT_STAGES),
        help=f"comma-separated subset of {','.join(STAGES)}",
    )
    ap.add_argument("--json", default=None, help="results file for bench/summarize")
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="let bench replace an existing --json instead of refusing to start",
    )
    ap.add_argument("--arms", default=None, help="override the preset's arms")
    ap.add_argument(
        "--per-task",
        type=int,
        default=None,
        help="instances per subtask (six subtasks, 80 available each)",
    )
    ap.add_argument("--max-new", type=int, default=None)
    ap.add_argument("--max-context", type=int, default=None)
    ap.add_argument("--questions", default=None)
    ap.add_argument(
        "--report-prompt",
        default=None,
        help="prompt file for the `report` trajectory of the validate stage",
    )
    ap.add_argument("--ep", default="hipgpu", choices=["cpu", "hipgpu", "amdgpu"])
    ap.add_argument("--no-iobinding", dest="iobinding", action="store_false")
    ap.add_argument(
        "--fetch",
        action="store_true",
        help="let `export` download a missing checkpoint at its pinned revision",
    )
    ap.add_argument(
        "--no-strict",
        action="store_true",
        help="do not set HIPDNN_EP_STRICT=1 for the children",
    )
    ap.add_argument(
        "--keep-env",
        action="store_true",
        help="do not clear the hip-ep tuning variables before running",
    )
    ap.add_argument(
        "--python",
        default=None,
        help="interpreter for the harness; defaults to <workspace>/.venv when it "
        "exists, else this one",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="print the commands each stage would run and exit 0",
    )
    ap.add_argument("--list-presets", action="store_true")
    ap.add_argument(
        "extra",
        nargs="*",
        help="flags passed through to qwen38_spec_bench.py, after a `--` "
        "(for example `-- --resume`)",
    )
    args = ap.parse_args()

    if args.list_presets:
        for name, p in sorted(PRESETS.items()):
            print(f"{name:<22} {p['describe']}")
            print(f"{'':<22} arms: {p['arms']}")
        return 0
    if not args.preset:
        ap.error("--preset is required (see --list-presets)")

    asked = [s.strip() for s in args.stages.split(",") if s.strip()]
    bad = [s for s in asked if s not in STAGES]
    if bad:
        print(f"unknown stage(s) {bad}; want {','.join(STAGES)}", file=sys.stderr)
        return 2
    # Each stage consumes what the earlier ones produce (bench before export
    # would measure a stale graph), so the order is fixed.
    stages = [s for s in STAGES if s in asked]
    # Checked now rather than when the stage is reached, which is after sanity.
    if not args.json and {"bench", "summarize"} & set(stages):
        ap.error(
            "--json is required by the bench and summarize stages "
            "(drop them with --stages, e.g. --stages preflight,sanity)"
        )

    ctx = Ctx(args)
    ctx.stages = stages
    print(f"preset      {args.preset} -- {ctx.preset['describe']}")
    print(f"harness     {ctx.harness}")
    print(f"python      {ctx.python}")
    print(f"hip-ep      {ctx.hipep_build or '(not set)'}")
    print(f"models      {ctx.models}")
    print(f"arms        {ctx.arms}")
    print(f"ep          {ctx.ep}{' +iobinding' if ctx.iobinding else ''}")
    print(
        f"env         strict={'off' if args.no_strict else 'on'}, "
        f"tuning levers {'kept' if args.keep_env else 'cleared'}"
    )
    print(f"stages      {','.join(stages)}{'   (dry run)' if ctx.dry_run else ''}")

    results: list[tuple[str, bool]] = []
    interrupted = False
    for name in stages:
        print(f"\n--- {name} " + "-" * (58 - len(name)))
        try:
            ok = HANDLERS[name](ctx)
            if ok and name == "export" and "preflight" in stages and not ctx.dry_run:
                # The tap-layer and hard-link checks need the built files.
                print("\n--- preflight, after export " + "-" * 34)
                ctx.stages = [s for s in stages if s != "export"]
                ok = stage_preflight(ctx)
        except KeyboardInterrupt:
            # The bench JSON is rewritten after every prompt, so what ran is kept
            # and `-- --resume` picks it up.
            ok, interrupted = False, True
        results.append((name, ok))
        if not ok:
            print(f"\nstopping: {name} {'interrupted' if interrupted else 'failed'}")
            break

    print("\n" + "=" * 62)
    for name, ok in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print("=" * 62)
    if interrupted:
        return 130
    return 0 if all(ok for _, ok in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
