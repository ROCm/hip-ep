#!/usr/bin/env python3
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Entry point: decode a .rgp capture and emit all analyzed files at once.

    python main.py <capture>.rgp <output-base>

e.g. `python main.py sample.rgp sample` -> sample_summary.json/.md,
sample_operators.csv, sample_dispatches.csv, sample_trace.json.

The logic lives in the rgp_parser package (kept in a package so its `struct`
subpackage never shadows the stdlib `struct` module).
"""

from rgp_parser.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
