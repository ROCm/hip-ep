#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""Tests for the ONNX ScatterElements op (reduction='none').

output = copy(data); then output[i][ indices[i][j] ] = updates[i][j]  (axis=1).
The output has the same shape as `data`. With reduction='none', duplicate
indices along the scatter axis make the result write-order dependent (last
write wins) and thus non-deterministic across implementations -- so every
test here uses per-row DISTINCT indices (each target position written exactly
once), which makes the output bit-exact (atol=0) regardless of write order.

Both int32 and int64 indices are covered (the runtime threads the index byte
width through `indices_element_size_bytes`).
"""

from __future__ import annotations

import numpy as np
import pytest
from onnx import helper

from framework.comparator import compare_outputs
from framework.onnx_utils import make_model_from_nodes, np_to_onnx_type


def _make_scatter_elements_model(
    data_shape: list[int],
    upd_shape: list[int],
    axis: int,
    idx_dtype: np.dtype,
):
    data = helper.make_tensor_value_info(
        "data", np_to_onnx_type(np.float32), list(data_shape)
    )
    indices = helper.make_tensor_value_info(
        "indices", np_to_onnx_type(idx_dtype), list(upd_shape)
    )
    updates = helper.make_tensor_value_info(
        "updates", np_to_onnx_type(np.float32), list(upd_shape)
    )
    out = helper.make_tensor_value_info(
        "output", np_to_onnx_type(np.float32), list(data_shape)
    )
    node = helper.make_node(
        "ScatterElements",
        ["data", "indices", "updates"],
        ["output"],
        axis=axis,
    )
    return make_model_from_nodes([node], [data, indices, updates], [out])


def _distinct_row_indices(rng, rows: int, cols: int, axis_dim: int, idx_dtype):
    """Per-row indices with no duplicates (permutation prefix), so scatter is
    deterministic under reduction='none'."""
    idx = np.empty((rows, cols), dtype=idx_dtype)
    for r in range(rows):
        idx[r] = rng.permutation(axis_dim)[:cols].astype(idx_dtype)
    return idx


class TestScatterElements:
    @pytest.mark.parametrize("idx_dtype", [np.int32, np.int64])
    def test_scatter_elements_axis1(self, model_runner, idx_dtype):
        data_shape = [4, 6]
        upd_shape = [4, 4]
        axis = 1
        model = _make_scatter_elements_model(data_shape, upd_shape, axis, idx_dtype)
        rng = np.random.default_rng(501)
        data = rng.uniform(-5.0, 5.0, data_shape).astype(np.float32)
        updates = rng.uniform(-5.0, 5.0, upd_shape).astype(np.float32)
        indices = _distinct_row_indices(
            rng, upd_shape[0], upd_shape[1], data_shape[axis], idx_dtype
        )
        actual, expected = model_runner.run_sample(model, [data, indices, updates])
        compare_outputs(actual, expected, atol=0)

    @pytest.mark.parametrize("idx_dtype", [np.int32, np.int64])
    def test_scatter_elements_full_overwrite(self, model_runner, idx_dtype):
        """updates covers every column (full permutation) -> output == permuted
        updates, no original data survives along the axis."""
        data_shape = [3, 5]
        upd_shape = [3, 5]
        axis = 1
        model = _make_scatter_elements_model(data_shape, upd_shape, axis, idx_dtype)
        rng = np.random.default_rng(502)
        data = rng.uniform(-5.0, 5.0, data_shape).astype(np.float32)
        updates = rng.uniform(-5.0, 5.0, upd_shape).astype(np.float32)
        indices = _distinct_row_indices(
            rng, upd_shape[0], upd_shape[1], data_shape[axis], idx_dtype
        )
        actual, expected = model_runner.run_sample(model, [data, indices, updates])
        compare_outputs(actual, expected, atol=0)
