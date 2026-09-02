#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

from __future__ import annotations

import argparse
from pathlib import Path

from .bundle import DEFAULT_OUTPUT_DIR_NAME, detect_bundle
from .pipeline import INTERMEDIATES_DIR_NAME, _default_output_dir, convert_bundle

__all__ = ["main"]


def main(argv: list[str] | None = None) -> int:
    """Convert a split ONNX pipeline bundle into a single merged model."""
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing split embedding / decoder / lm_head ONNX files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=f"Output directory (default: <input-dir>/{DEFAULT_OUTPUT_DIR_NAME})",
    )
    parser.add_argument(
        "--keep-intermediates",
        action="store_true",
        help=(
            f"Keep step-by-step ONNX artifacts under <output-dir>/{INTERMEDIATES_DIR_NAME}/ "
            "(decoder weights in model.data; default: temp dir, deleted when done)"
        ),
    )
    args = parser.parse_args(argv)

    bundle = detect_bundle(args.input_dir)
    out = args.output_dir or _default_output_dir(bundle)
    convert_bundle(bundle, out, keep_intermediates=args.keep_intermediates)
    return 0
