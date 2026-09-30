#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""Numeric coverage for ONNX ArgMax against the ORT CPU reference.

Indices are bit-exact (atol=0). Ties are included so select_last_index is
checked, not only the unique-maximum case. ui32 and ui64 are intentionally
absent: the kernel has no unsigned 32/64 compare.
"""

from __future__ import annotations

import numpy as np
import pytest
from onnx import helper

from framework.comparator import compare_outputs
from framework.onnx_utils import make_model_from_nodes, np_to_onnx_type


def _out_shape(shape: list[int], axis: int, keepdims: int) -> list[int]:
    axis = axis if axis >= 0 else axis + len(shape)
    out = list(shape)
    if keepdims:
        out[axis] = 1
    else:
        del out[axis]
    return out


def _make_argmax_model(
    shape: list[int],
    axis: int,
    keepdims: int,
    select_last_index: int,
    dtype: np.dtype,
):
    tp = np_to_onnx_type(dtype)
    x = helper.make_tensor_value_info("x", tp, list(shape))
    y = helper.make_tensor_value_info(
        "y", np_to_onnx_type(np.int64), _out_shape(shape, axis, keepdims)
    )
    node = helper.make_node(
        "ArgMax",
        ["x"],
        ["y"],
        axis=axis,
        keepdims=keepdims,
        select_last_index=select_last_index,
    )
    return make_model_from_nodes([node], [x], [y])


class TestArgMax:
    @pytest.mark.parametrize(
        "dtype", [np.float32, np.float16, np.int32, np.int64, np.uint8]
    )
    @pytest.mark.parametrize("keepdims", [0, 1])
    @pytest.mark.parametrize("axis", [0, -1])
    def test_unique_max(self, model_runner, dtype, keepdims, axis):
        shape = [2, 5]
        model = _make_argmax_model(shape, axis, keepdims, 0, dtype)
        rng = np.random.default_rng(7)
        # Distinct values along each reduced slice.
        cols = np.stack([rng.permutation(shape[1]) for _ in range(shape[0])])
        x = cols.astype(dtype)
        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=0)

    @pytest.mark.parametrize("select_last_index", [0, 1])
    def test_ties(self, model_runner, select_last_index):
        # Row 0: max 3 at indices 1 and 2. First -> 1, last -> 2.
        # Row 1: max 4 only at index 0.
        x = np.array([[1, 3, 3, 2], [4, 0, 1, 2]], dtype=np.int32)
        model = _make_argmax_model(
            list(x.shape),
            axis=1,
            keepdims=0,
            select_last_index=select_last_index,
            dtype=np.int32,
        )
        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=0)

    def test_negative_axis_keepdims(self, model_runner):
        x = np.array([[0, 2, 1], [5, 5, 1]], dtype=np.float32)
        model = _make_argmax_model(
            list(x.shape), axis=-1, keepdims=1, select_last_index=1, dtype=np.float32
        )
        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=0)
