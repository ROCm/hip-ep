#!/usr/bin/env python
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Standalone builder for Whisper variant ONNX models (fp32 + fp16).

Both precisions are produced by the SAME pinned OGA DirectML model builder, so
pinning the builder deps + the HF model revision makes both models reproducible.
This REPLACES the old external fp32 download (tonythethompson/whisper-large-v3-genai),
which was non-reproducible.

Isolation: the OGA model builder needs the STOCK onnxruntime-genai-directml, which
shadows the OGA FORK that build.py --build-oga installs (both import as
`onnxruntime_genai`). To keep the dev env's fork intact, this script builds the
models in a DEDICATED, PINNED venv (install/whisper-builder-venv/) and runs the
builder via subprocess in that venv. The dev env is never touched.

Run:
    conda activate hipdnn-ep
    python scripts/build_whisper_models.py            # build default variants (fp32 + fp16)
    python scripts/build_whisper_models.py --list     # show what would be built
    python scripts/build_whisper_models.py --variant tiny --variant base  # specific variants
    python build.py --build-whisper-models            # same, via the build wrapper

Outputs (raw OGA bundles; conftest then surgeries + fix_shapes them):
    models/whisper-{variant}-onnx/        fp32: encoder/decoder.onnx (+.data) + configs
    models/whisper-{variant}-onnx-fp16/   fp16: same layout, fp16 body + fp32 lm_head
"""

import argparse
import subprocess
import venv
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS = REPO_ROOT / "models"
BUILDER_VENV = REPO_ROOT / "install" / "whisper-builder-venv"

# ---------------------------------------------------------------------------
# Pinned per-variant sources (same idiom as build.py's version pins).
# DUPLICATED from conftest.WHISPER_VARIANTS so this builder stays standalone
# (no test-infra import). test_whisper_build_paths.py asserts the two match.
# ---------------------------------------------------------------------------
VARIANT_SOURCES = {
    # name            (hf_model_id,                    revision)
    "large-v3": ("openai/whisper-large-v3", "06f233fe06e710322aca913c1bc4249a0d71fce1"),
    "large-v3-turbo": ("openai/whisper-large-v3-turbo", "main"),
    "tiny": ("openai/whisper-tiny", "main"),
    "base": ("openai/whisper-base", "main"),
    "small": ("openai/whisper-small", "main"),
    "medium": ("openai/whisper-medium", "main"),
}

# Default build set when --variant is omitted (turbo + the small sizes; large-v3
# is built/downloaded via its own AMD-HF path in conftest, so it is not default).
DEFAULT_VARIANTS = ["large-v3-turbo", "tiny", "base", "small", "medium"]

# Exact builder deps. onnxruntime-genai-directml is the STOCK builder package;
# torch CPU is fine (the builder only reads HF weights). These live ONLY in the
# isolated venv, never the dev env.
BUILDER_REQUIREMENTS = [
    "onnxruntime-genai-directml==0.13.1",
    "torch==2.12.0",
    "transformers==4.49.0",
    "onnx",
    "onnx_ir==0.2.1",
]


def whisper_output_dirs(variant: str) -> "tuple[Path, Path]":
    """Return (fp32_dir, fp16_dir) under models/ for a variant.

    Naming convention MUST match conftest (models/whisper-{variant}-onnx[-fp16]).
    """
    return (
        MODELS / f"whisper-{variant}-onnx",
        MODELS / f"whisper-{variant}-onnx-fp16",
    )


def _venv_python(vdir: Path) -> Path:
    # Windows venv layout: Scripts/python.exe
    return vdir / "Scripts" / "python.exe"


def ensure_builder_venv() -> Path:
    """Create install/whisper-builder-venv/ with the pinned deps. Idempotent
    (a .ok sentinel records that the pinned requirements were installed)."""
    sentinel = BUILDER_VENV / ".deps.ok"
    py = _venv_python(BUILDER_VENV)
    if sentinel.exists() and py.exists():
        print(f"[whisper-build] builder venv ready: {BUILDER_VENV}")
        return py

    if not py.exists():
        print(f"[whisper-build] creating builder venv at {BUILDER_VENV} ...")
        venv.create(str(BUILDER_VENV), with_pip=True)

    print(f"[whisper-build] installing pinned builder deps: {BUILDER_REQUIREMENTS}")
    subprocess.run([str(py), "-m", "pip", "install", "--upgrade", "pip"], check=True)
    subprocess.run([str(py), "-m", "pip", "install", *BUILDER_REQUIREMENTS], check=True)
    sentinel.touch()
    print("[whisper-build] builder venv deps installed.")
    return py


# The build runs INSIDE the venv via `python -c <BUILDER_SNIPPET>` so the stock
# OGA import resolves there, not in the dev env. The qwen-class stub makes OGA's
# eager builders/__init__ import succeed under transformers 4.49.0 (the Whisper
# builder never uses those classes). This snippet is the SINGLE source of the OGA
# `create_model(... precision=..., execution_provider="dml" ...)` invocation for
# both precisions (the old test/python/whisper/build_fp16_model.py was retired).
_NUM_STUBS = 10
BUILDER_SNIPPET = r"""
import re, sys
import transformers
def _stub():
    for _ in range(%d):
        try:
            from onnxruntime_genai.models.builder import create_model  # noqa
            return create_model
        except ImportError as e:
            m = re.search(r"cannot import name '([^']+)'", str(e))
            # A non-"cannot import name" ImportError (e.g. OGA itself not
            # installed in the venv) is a real failure — surface it, don't stub.
            if not m: raise
            name = m.group(1)
            # Already stubbed this name yet still failing -> something else is
            # wrong; re-raise instead of looping forever on the same import.
            if hasattr(transformers, name): raise
            setattr(transformers, name, type(name, (), {}))
    raise RuntimeError(
        "could not satisfy OGA builder imports after %d transformers stubs")
create_model = _stub()
model_id, revision, precision, out_dir, cache_dir = sys.argv[1:6]

# --- Pin the HF revision so the weights are byte-stable across runs. ----------
# WHICH CALL PATH the OGA Whisper builder actually uses (verified against the
# stock onnxruntime-genai 0.13.1 source):
#   create_model -> transformers.AutoConfig.from_pretrained                (config)
#                -> base.Model.make_model -> AutoModelForSpeechSeq2Seq.from_pretrained  (WEIGHTS)
#                -> make_genai_config    -> AutoConfig / GenerationConfig.from_pretrained
#                -> save_processing      -> AutoTokenizer.from_pretrained   (tokenizer)
# ALL of these go through transformers -> `cached_file` -> hf_hub_download PER FILE.
# NONE of them call huggingface_hub.snapshot_download, and none thread a
# `revision=` kwarg, and there is no env var that hf_hub honors for the default
# revision (huggingface_hub hardcodes DEFAULT_REVISION="main"). So the ONLY
# reliable way to pin the revision is to default it into the from_pretrained
# calls the builder makes, plus hf_hub_download as a per-file catch-all.
import huggingface_hub
import transformers

def _wrap_revision(fn):
    def _wrapped(*a, **k):
        k.setdefault("revision", revision)
        return fn(*a, **k)
    return _wrapped

# from_pretrained classmethods are the primary hooks (config/weights/tokenizer).
for _cls_name in (
    "AutoConfig",
    "AutoModelForSpeechSeq2Seq",  # Whisper weights
    "AutoTokenizer",              # save_processing
    "AutoProcessor",              # belt-and-suspenders (not used by Whisper today)
    "GenerationConfig",           # make_genai_config
):
    _cls = getattr(transformers, _cls_name, None)
    if _cls is not None and hasattr(_cls, "from_pretrained"):
        _cls.from_pretrained = _wrap_revision(_cls.from_pretrained)

# Per-file catch-all: every transformers `cached_file` lands here eventually.
huggingface_hub.hf_hub_download = _wrap_revision(huggingface_hub.hf_hub_download)
# Harmless belt-and-suspenders (the builder does not actually call this).
huggingface_hub.snapshot_download = _wrap_revision(huggingface_hub.snapshot_download)

create_model(
    model_name=model_id, input_path="", output_dir=out_dir,
    precision=precision, execution_provider="dml", cache_dir=cache_dir,
)
print("[builder-subproc] done:", precision, "->", out_dir)
""" % (_NUM_STUBS, _NUM_STUBS)


def build_one(
    py: Path, model_id: str, revision: str, precision: str, out_dir: Path
) -> None:
    """Build one (variant, precision) via the OGA builder in the isolated venv.
    Idempotent (skips if the raw encoder+decoder already exist)."""
    if (out_dir / "encoder.onnx").exists() and (out_dir / "decoder.onnx").exists():
        print(f"[whisper-build] {out_dir.name}: already built")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = BUILDER_VENV / "_hf_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"[whisper-build] building {precision} {model_id} -> {out_dir}")
    subprocess.run(
        [
            str(py),
            "-c",
            BUILDER_SNIPPET,
            model_id,
            revision,
            precision,
            str(out_dir),
            str(cache_dir),
        ],
        check=True,
    )


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Build reproducible Whisper fp32 + fp16 ONNX via the pinned "
        "OGA DirectML model builder (isolated venv)."
    )
    ap.add_argument(
        "--variant",
        action="append",
        default=None,
        help="variant to build (repeatable, or comma-separated). "
        f"Known: {sorted(VARIANT_SOURCES)}. Default: {DEFAULT_VARIANTS}.",
    )
    ap.add_argument(
        "--precision",
        choices=["fp32", "fp16", "both"],
        default="both",
        help="which precision(s) to build (default: both)",
    )
    ap.add_argument(
        "--list",
        action="store_true",
        help="print the resolved variant -> output-dir mapping and exit",
    )
    args = ap.parse_args()

    # Resolve --variant (repeatable AND comma-list) -> flat list; default set.
    if args.variant:
        variants = []
        for v in args.variant:
            variants.extend(p.strip() for p in v.split(",") if p.strip())
    else:
        variants = list(DEFAULT_VARIANTS)
    for v in variants:
        if v not in VARIANT_SOURCES:
            ap.error(f"unknown variant {v!r}; known: {sorted(VARIANT_SOURCES)}")

    if args.list:
        for v in variants:
            fp32, fp16 = whisper_output_dirs(v)
            print(f"{v}: fp32={fp32}  fp16={fp16}  src={VARIANT_SOURCES[v]}")
        return 0

    py = ensure_builder_venv()
    for v in variants:
        model_id, revision = VARIANT_SOURCES[v]
        fp32_dir, fp16_dir = whisper_output_dirs(v)
        for prec, out_dir in (("fp32", fp32_dir), ("fp16", fp16_dir)):
            if args.precision in (prec, "both"):
                build_one(py, model_id, revision, prec, out_dir)

    print("[whisper-build] DONE.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
