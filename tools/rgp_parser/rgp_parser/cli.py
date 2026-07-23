#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""main.py CLI: decode a .rgp and emit all analyzed files at once.

    python main.py <capture>.rgp <output-base>

Given output base ``sample`` (a name or path, extension ignored), emits:
    sample_summary.json, sample_summary.md, sample_operators.csv,
    sample_dispatches.csv, sample_trace.json
"""

from __future__ import annotations

import os
import sys

from .build import build_trace
from .format import WRITERS


def _strip_ext(base: str) -> str:
    # allow the user to pass "sample", "sample.json", "out/sample" etc.
    root, ext = os.path.splitext(base)
    return root if ext else base


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    # minimal positional parsing: <rgp> <output-base> [--records PATH]
    records = None
    if "--records" in argv:
        i = argv.index("--records")
        try:
            records = argv[i + 1]
            del argv[i : i + 2]
        except IndexError:
            print("error: --records needs a path", file=sys.stderr)
            return 2
    if len(argv) != 2 or argv[0] in ("-h", "--help"):
        print(
            "usage: main.py <capture>.rgp <output-base> [--records records.json]\n"
            "  emits <output-base>_{summary.json,summary.md,operators.csv,"
            "dispatches.csv,trace.json}",
            file=sys.stderr,
        )
        return 0 if argv and argv[0] in ("-h", "--help") else 2

    rgp, out_base = argv
    if not os.path.isfile(rgp):
        print(f"error: no such file: {rgp}", file=sys.stderr)
        return 2

    base = _strip_ext(out_base)
    outdir = os.path.dirname(base)
    if outdir:
        os.makedirs(outdir, exist_ok=True)

    try:
        trace = build_trace(rgp, records_path=records)
    except NotImplementedError as e:
        print(f"error: {e}", file=sys.stderr)
        return 3

    for _fmt, (writer, suffix) in WRITERS.items():
        out = base + suffix
        writer(trace, out)
        print(f"wrote {out}")
    return 0
