#!/usr/bin/env python
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Download + prepare the Whisper-large-v3 ONNX for the MorphiZen EP.

One-shot wrapper around ``conftest.setup_whisper_model_dir`` so users don't have
to paste a multi-line ``python -c`` (which is awkward to quote/continue in
PowerShell). Runs identically in PowerShell and Git Bash:

    python scripts/setup_whisper_model.py

It downloads the ~6 GB fp32 ONNX from HuggingFace, applies the required ONNX
surgery (past_sequence_length input + position-/token-embed fixes), and runs
fix_shapes to produce the static-shape variants the EP compiles. Idempotent:
re-running is a no-op once the files exist. Works from any current directory.
"""

import pathlib
import sys

# Resolve the repo root from this script's location (scripts/ is at repo root),
# so the command works regardless of the current working directory.
REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "test" / "python"))

from conftest import setup_whisper_model_dir  # noqa: E402

MODEL_DIR = REPO_ROOT / "models" / "whisper-large-v3-onnx"


def main() -> int:
    print(f"[whisper-setup] target: {MODEL_DIR}")
    print("[whisper-setup] downloading + preparing fp32 Whisper-large-v3 ONNX")
    print("[whisper-setup] (first run downloads ~6 GB; later runs are a no-op)")
    setup_whisper_model_dir(MODEL_DIR)
    expected = [
        "encoder_fixed.onnx",
        "decoder_surgery.onnx",
        "decoder_fixed_prefill.onnx",
        "decoder_fixed_decode.onnx",
    ]
    missing = [f for f in expected if not (MODEL_DIR / f).exists()]
    if missing:
        print(f"[whisper-setup] ERROR: missing expected files: {missing}")
        return 1
    print("[whisper-setup] DONE — model ready at:")
    print(f"    {MODEL_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
