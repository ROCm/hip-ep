#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""Numeric verification framework for an ORT execution provider.

Each test builds (or loads) a small ONNX model, runs it through a pluggable
backend, and compares the output against an ORT CPU reference value. The
reference value is either computed on demand or served from a per-test cache
so large CPU MatMuls are not re-run on every test invocation.

The framework is EP-agnostic; the canonical wiring for this repo's
build of the MorphiZen EP lives in ``test/numeric/README.md`` under
"Example: MorphiZen EP".
"""
