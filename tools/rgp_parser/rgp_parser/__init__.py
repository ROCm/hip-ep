#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""rgp_parser: decode a Radeon GPU Profiler (RGP) .rgp capture into analysis-friendly
formats (per-kernel summary, operators CSV, per-dispatch CSV, Chrome trace).

The package is intentionally nested under this directory so the `struct` subpackage is
imported as ``rgp_parser.struct`` and never shadows Python's stdlib ``struct`` module.
"""

__version__ = "0.1.0"
