#!/usr/bin/env python
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Prepare an already-built Whisper variant ONNX for the MorphiZen EP.

Ensures the raw OGA bundle is present (large-v3 auto-downloads from AMD HF; other
variants are local-build primary — `python scripts/build_whisper_models.py
--variant <name>`), then applies the decoder surgery + fix_shapes to emit the
static-shape variants the EP compiles. Default variant is large-v3, default
precision fp16.

    python scripts/setup_whisper_model.py                       # large-v3 fp16
    python scripts/setup_whisper_model.py --variant tiny        # tiny fp16
    python scripts/setup_whisper_model.py --variant small --fp32
"""

import argparse
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "test" / "python"))

from conftest import setup_whisper_variant  # noqa: E402


def _resolve_args(argv):
    ap = argparse.ArgumentParser(
        description="Prepare a Whisper variant ONNX for the EP."
    )
    ap.add_argument("--variant", default="large-v3", help="variant name")
    ap.add_argument("--fp32", action="store_true", help="select fp32 (default: fp16)")
    ns = ap.parse_args(argv)
    return ns.variant, ("fp32" if ns.fp32 else "fp16")


def main() -> int:
    name, precision = _resolve_args(sys.argv[1:])
    print(f"[whisper-setup] target: {name} ({precision})")
    try:
        model_dir, _var = setup_whisper_variant(name, precision)
    except (FileNotFoundError, KeyError) as e:
        print(f"[whisper-setup] ERROR: {e}")
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
    print(f"[whisper-setup] DONE — {name} {precision} ready at:\n    {model_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
