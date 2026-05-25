#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""Numeric verification for ONNX Shape (static-input compile-time fold).

Shape has two lowering paths in `lib/Conversion/OnnxToHip/ShapeConversion.cpp`:

  * `ShapeToConstant` (benefit=2): when the input has a fully static
    shape, rewrites to a single `arith.constant` carrying the i64 dim
    vector. No HIP op, no runtime function, no per-inference work.

  * `ShapeToHip` (benefit=1): when at least one input dim is dynamic
    (typically a Category-C producer such as NonZero), emits `hip.shape`
    with one DimSpec per output element. The dynamic-input path is
    exercised end-to-end in `test_nonzero_composition.py::Shape(NonZero(X))`.

This file covers the static-input fold:
  * Multiple input ranks (1D / 2D / 3D / 4D).
  * Multiple element types (Shape is type-agnostic on the input; output
    is always int64 of length `rank`).
  * The opset-15+ `start` / `end` attributes (positive and negative
    values, including out-of-range clamping per the ONNX spec).
"""

from __future__ import annotations

import numpy as np
import pytest
from onnx import TensorProto, helper

from framework.onnx_utils import make_model_from_nodes, np_to_onnx_type


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_shape_model(
    dtype: np.dtype,
    input_shape: list[int],
    start: int | None = None,
    end: int | None = None,
):
    tp = np_to_onnx_type(dtype)
    X = helper.make_tensor_value_info("X", tp, input_shape)
    # Compute expected output length (mirrors ShapeConversion.cpp clamping).
    rank = len(input_shape)
    s = 0 if start is None else (start if start >= 0 else start + rank)
    e = rank if end is None else (end if end >= 0 else end + rank)
    s = max(0, min(rank, s))
    e = max(0, min(rank, e))
    if e < s:
        e = s
    out_len = e - s
    Y = helper.make_tensor_value_info("Y", TensorProto.INT64, [out_len])
    attrs = {}
    if start is not None:
        attrs["start"] = start
    if end is not None:
        attrs["end"] = end
    node = helper.make_node("Shape", ["X"], ["Y"], **attrs)
    return make_model_from_nodes([node], [X], [Y])


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "shape",
    [
        [5],
        [3, 4],
        [2, 3, 4],
        [1, 8, 1, 16],
    ],
)
@pytest.mark.parametrize(
    "dtype",
    [np.float32, np.float16, np.int64, np.int32],
)
class TestShape:
    """Shape op folds to constant for static-shape input."""

    def test_shape_no_slice(self, model_runner, shape, dtype):
        model = _make_shape_model(dtype, shape)
        rng = np.random.default_rng(0)
        x = rng.standard_normal(size=shape).astype(dtype)
        actual, expected = model_runner.run_sample(model, [x])
        assert len(actual) == 1 and len(expected) == 1
        assert actual[0].dtype == np.int64
        assert tuple(actual[0].tolist()) == tuple(shape)
        np.testing.assert_array_equal(actual[0], expected[0])


@pytest.mark.parametrize(
    "shape,start,end,expected",
    [
        # Positive start / end (slice middle).
        ([2, 3, 4, 5], 1, 3, [3, 4]),
        # Negative end (Python-style trailing slice).
        ([2, 3, 4, 5], 0, -1, [2, 3, 4]),
        # Negative start.
        ([2, 3, 4, 5], -2, 4, [4, 5]),
        # start > end after normalisation -> empty.
        ([2, 3, 4], 2, 1, []),
        # Out-of-range clamping (start < 0 collapses below 0, end > rank
        # clamps to rank).
        ([3, 4], -10, 10, [3, 4]),
    ],
)
class TestShapeStartEnd:
    """`start` / `end` attributes select a contiguous slice of the shape."""

    def test_shape_slice(self, model_runner, shape, start, end, expected):
        model = _make_shape_model(np.float32, shape, start=start, end=end)
        rng = np.random.default_rng(0)
        x = rng.standard_normal(size=shape).astype(np.float32)
        actual, ref = model_runner.run_sample(model, [x])
        assert len(actual) == 1 and len(ref) == 1
        assert actual[0].dtype == np.int64
        # CPU reference is the source of truth; check our output matches it
        # and matches the hand-computed expected for the chosen slice.
        np.testing.assert_array_equal(actual[0], ref[0])
        assert actual[0].tolist() == expected
