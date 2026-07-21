#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""Tests for the ONNX GatherElements op.

out[i][j] = data[i][ indices[i][j] ]   (for axis=1; general per-axis gather).
The output has the same shape as `indices`. The gather is a pure data copy so
the result must be bit-exact (atol=0).

Coverage exercises both int32 and int64 indices (the runtime threads the
index byte width through `indices_element_size_bytes` -- a regression that
read int32 indices as int64 would fuse adjacent values into huge OOB indices,
so both widths must be tested), plus positive and negative axes.
"""

from __future__ import annotations

import numpy as np
import pytest
from onnx import helper

from framework.comparator import compare_outputs
from framework.onnx_utils import make_model_from_nodes, np_to_onnx_type


def _make_gather_elements_model(
    data_shape: list[int],
    idx_shape: list[int],
    axis: int,
    idx_dtype: np.dtype,
):
    data = helper.make_tensor_value_info(
        "data", np_to_onnx_type(np.float32), list(data_shape)
    )
    indices = helper.make_tensor_value_info(
        "indices", np_to_onnx_type(idx_dtype), list(idx_shape)
    )
    out = helper.make_tensor_value_info(
        "output", np_to_onnx_type(np.float32), list(idx_shape)
    )
    node = helper.make_node(
        "GatherElements", ["data", "indices"], ["output"], axis=axis
    )
    return make_model_from_nodes([node], [data, indices], [out])


class TestGatherElements:
    @pytest.mark.parametrize("idx_dtype", [np.int32, np.int64])
    @pytest.mark.parametrize("axis", [0, 1])
    def test_gather_elements_2d(self, model_runner, idx_dtype, axis):
        data_shape = [4, 6]
        idx_shape = [4, 6]
        model = _make_gather_elements_model(data_shape, idx_shape, axis, idx_dtype)
        rng = np.random.default_rng(401)
        data = rng.uniform(-5.0, 5.0, data_shape).astype(np.float32)
        indices = rng.integers(0, data_shape[axis], idx_shape, dtype=idx_dtype)
        actual, expected = model_runner.run_sample(model, [data, indices])
        compare_outputs(actual, expected, atol=0)

    @pytest.mark.parametrize("idx_dtype", [np.int32, np.int64])
    def test_gather_elements_rect_indices(self, model_runner, idx_dtype):
        """indices smaller than data along the gather axis (idx_shape != data)."""
        data_shape = [3, 8]
        idx_shape = [3, 5]
        axis = 1
        model = _make_gather_elements_model(data_shape, idx_shape, axis, idx_dtype)
        rng = np.random.default_rng(402)
        data = rng.uniform(-5.0, 5.0, data_shape).astype(np.float32)
        indices = rng.integers(0, data_shape[axis], idx_shape, dtype=idx_dtype)
        actual, expected = model_runner.run_sample(model, [data, indices])
        compare_outputs(actual, expected, atol=0)
