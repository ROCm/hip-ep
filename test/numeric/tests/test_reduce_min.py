#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""Tests for the ONNX ReduceMin op.

The runtime (lib/Runtime/real/reduce_min.cpp) supports f16/i32/i64 and assumes
the reduced axis occupies the trailing portion of the buffer (reduce_size =
data_elems / output_elems), so these tests reduce the LAST axis -- the natural
contiguous case. Integer dtypes (i32/i64, both supported by the EP kernel and
by ORT CPU ReduceMin) give a bit-exact reference (atol=0). keepdims is covered
in both settings.
"""

from __future__ import annotations

import numpy as np
import pytest
from onnx import helper

from framework.comparator import compare_outputs
from framework.onnx_utils import make_model_from_nodes, np_to_onnx_type


def _make_reduce_min_model(
    shape: list[int], axes: list[int], keepdims: int, dtype: np.dtype
):
    out_shape = list(shape)
    for a in axes:
        out_shape[a] = 1
    if not keepdims:
        out_shape = [d for i, d in enumerate(out_shape) if i not in set(axes)]
    tp = np_to_onnx_type(dtype)
    X = helper.make_tensor_value_info("data", tp, list(shape))
    Y = helper.make_tensor_value_info("reduced", tp, out_shape)
    node = helper.make_node(
        "ReduceMin", ["data"], ["reduced"], axes=axes, keepdims=keepdims
    )
    return make_model_from_nodes([node], [X], [Y])


class TestReduceMin:
    @pytest.mark.parametrize("dtype", [np.int32, np.int64])
    @pytest.mark.parametrize("keepdims", [0, 1])
    def test_reduce_min_last_axis_2d(self, model_runner, dtype, keepdims):
        shape = [4, 6]
        axes = [1]  # trailing axis
        model = _make_reduce_min_model(shape, axes, keepdims, dtype)
        rng = np.random.default_rng(1101)
        data = rng.integers(-100, 100, shape, dtype=dtype)
        actual, expected = model_runner.run_sample(model, [data])
        compare_outputs(actual, expected, atol=0)

    @pytest.mark.parametrize("dtype", [np.int32, np.int64])
    def test_reduce_min_last_axis_3d(self, model_runner, dtype):
        shape = [2, 3, 5]
        axes = [2]  # trailing axis
        model = _make_reduce_min_model(shape, axes, keepdims=0, dtype=dtype)
        rng = np.random.default_rng(1102)
        data = rng.integers(-100, 100, shape, dtype=dtype)
        actual, expected = model_runner.run_sample(model, [data])
        compare_outputs(actual, expected, atol=0)
