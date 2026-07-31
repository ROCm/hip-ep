#!/usr/bin/env python3
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Entry point for split-pipeline -> merged ONNX conversion.

See README.md for usage.
Implementation lives in the ``merged_convert`` package.
"""

from merged_convert.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
