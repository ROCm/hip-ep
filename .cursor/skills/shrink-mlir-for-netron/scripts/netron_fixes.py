#!/usr/bin/env python3
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Streaming Netron-oriented fixes for shrunk hip EP ONNX MLIR (.view.mlir).

- Quote unquoted ``onnx.Cast`` ``to`` attributes (``to = f32`` -> ``to = "f32"``).
- Normalize unsigned MLIR types in type strings (``ui8`` -> ``i8``, etc.).
- Optionally drop ``onnx.Constant`` / ``onnx.NoValue`` lines (SSA may be invalid; view-only).

Output is not valid MLIR for compilation.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

_CAST_TO_RE = re.compile(r'to = (?!")([A-Za-z_][A-Za-z0-9_]*)')
# MLIR unsigned elems appear as ``...xui8`` or ``<ui8`` (not always word-bound before ``ui``).
_UI_TYPE_RE = re.compile(r"(?<=[x<,])ui(\d+)\b")
_UI_TYPE_X_RE = re.compile(r"xui(\d+)\b")
_CONST_LINE_RE = re.compile(r'^\s*%\d+\s*=\s*"(onnx\.Constant|onnx\.NoValue)"')


def default_output_path(input_path: Path) -> Path:
    name = input_path.name
    if name.endswith(".view.mlir"):
        base = name[: -len(".view.mlir")]
        return input_path.with_name(f"{base}.netron.mlir")
    if name.endswith(".frontend.mlir"):
        stem = input_path.stem
        return input_path.with_name(f"{stem}.netron.mlir")
    return input_path.with_name(f"{input_path.stem}.netron.mlir")


def fix_line(line: str, *, strip_constants: bool) -> tuple[str | None, dict[str, int]]:
    stats = {"cast_fixes": 0, "ui_fixes": 0, "lines_dropped": 0}

    if strip_constants and _CONST_LINE_RE.match(line):
        stats["lines_dropped"] = 1
        return None, stats

    if '"onnx.Cast"' in line:
        new_line, n = _CAST_TO_RE.subn(r'to = "\1"', line)
        if n:
            stats["cast_fixes"] = n
            line = new_line

    line, n_xui = _UI_TYPE_X_RE.subn(r"xi\1", line)
    line, n_ui = _UI_TYPE_RE.subn(r"i\1", line)
    n_ui_total = n_xui + n_ui
    if n_ui_total:
        stats["ui_fixes"] = n_ui_total

    return line, stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input", type=Path, help="input .mlir (usually *.view.mlir)")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="output path (default: <stem>.netron.mlir beside input)",
    )
    parser.add_argument(
        "--keep-constants",
        action="store_true",
        help="keep onnx.Constant / onnx.NoValue lines (default: strip for cleaner graph)",
    )
    args = parser.parse_args(argv)

    if not args.input.is_file():
        parser.error(f"input file not found: {args.input}")

    out_path = args.output or default_output_path(args.input)
    if out_path.resolve() == args.input.resolve():
        parser.error("refusing to overwrite the input file; choose a different -o")

    strip_constants = not args.keep_constants
    in_bytes = args.input.stat().st_size
    totals = {
        "lines_in": 0,
        "lines_out": 0,
        "cast_fixes": 0,
        "ui_fixes": 0,
        "lines_dropped": 0,
    }
    out_bytes = 0

    with (
        args.input.open("r", encoding="utf-8", errors="replace") as fin,
        out_path.open("w", encoding="utf-8", newline="\n") as fout,
    ):
        for line in fin:
            totals["lines_in"] += 1
            fixed, st = fix_line(line, strip_constants=strip_constants)
            totals["cast_fixes"] += st["cast_fixes"]
            totals["ui_fixes"] += st["ui_fixes"]
            totals["lines_dropped"] += st["lines_dropped"]
            if fixed is None:
                continue
            fout.write(fixed)
            totals["lines_out"] += 1
            out_bytes += len(fixed.encode("utf-8"))

    print(f"input : {args.input}  ({in_bytes:,} bytes)")
    print(f"output: {out_path}  ({out_bytes:,} bytes)")
    print(
        f"cast to= quoted: {totals['cast_fixes']:,}; "
        f"ui* -> i*: {totals['ui_fixes']:,} substitution(s)"
    )
    if strip_constants:
        print(
            f"dropped {totals['lines_dropped']:,} Constant/NoValue line(s) "
            f"({totals['lines_in']:,} -> {totals['lines_out']:,} lines)"
        )
    else:
        print("constants: kept (--keep-constants)")
    print("NOTE: output is for Netron/viewing only — not valid MLIR for compilation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
