#!/usr/bin/env python
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Prepare an already-built Whisper-large-v3 ONNX for the MorphiZen EP.

One-shot wrapper around ``conftest.setup_whisper_model_dir`` so users don't have
to paste a multi-line ``python -c`` (which is awkward to quote/continue in
PowerShell). Runs identically in PowerShell and Git Bash:

    python scripts/setup_whisper_model.py          # fp16 (default)
    python scripts/setup_whisper_model.py --fp32   # fp32 model dir

This script does NOT download or build the raw model — it only PREPARES one: it
applies the ONNX surgery (past_sequence_length input + position-/token-embed
fixes) and fix_shapes to produce the static-shape variants the EP compiles. The
DEFAULT is the fp16 model; ``--fp32`` selects the fp32 model directory instead.
The raw model must already exist; build it first with::

    python build.py --build-whisper-models

If the raw model is absent, this script prints a build hint and exits non-zero.
Idempotent: re-running is a no-op once the prepared files exist. Works from any
current directory.
"""

import argparse
import pathlib
import sys

# Resolve the repo root from this script's location (scripts/ is at repo root),
# so the command works regardless of the current working directory.
REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "test" / "python"))

from conftest import (  # noqa: E402
    setup_whisper_fp16_model_dir,
    setup_whisper_model_dir,
)

MODEL_DIR = REPO_ROOT / "models" / "whisper-large-v3-onnx"
MODEL_DIR_FP16 = REPO_ROOT / "models" / "whisper-large-v3-onnx-fp16"


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Prepare an already-built Whisper-large-v3 ONNX for the EP."
    )
    ap.add_argument(
        "--fp32",
        action="store_true",
        help="select the fp32 model directory instead of the default fp16 one",
    )
    args = ap.parse_args()

    use_fp16 = not args.fp32
    model_dir = MODEL_DIR_FP16 if use_fp16 else MODEL_DIR
    prec = "fp16" if use_fp16 else "fp32"
    print(f"[whisper-setup] target: {model_dir}")
    try:
        if use_fp16:
            setup_whisper_fp16_model_dir(model_dir)
        else:
            setup_whisper_model_dir(model_dir)
    except FileNotFoundError as e:
        print(f"[whisper-setup] ERROR: {e}")
        print("[whisper-setup] Build the raw models first:")
        print("    python build.py --build-whisper-models")
        return 1
    expected = [
        "encoder_fixed.onnx",
        "decoder_surgery.onnx",
        "decoder_fixed_prefill.onnx",
        "decoder_fixed_decode.onnx",
    ]
    missing = [f for f in expected if not (model_dir / f).exists()]
    if missing:
        print(f"[whisper-setup] ERROR: missing expected files: {missing}")
        return 1
    print(f"[whisper-setup] DONE — {prec} model ready at:")
    print(f"    {model_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
