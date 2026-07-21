#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""Tests for the ONNX OneHot op.

output rank = indices rank + 1; the new dimension (size `depth`) is inserted
at `axis`. Each output slice is `on_value` at the index position and
`off_value` elsewhere. `depth` and `values` are constant initializers so the
output extent resolves statically; `indices` is a runtime input. The op writes
exact literal values, so the comparison is bit-exact (atol=0).
"""

from __future__ import annotations

import numpy as np
import pytest
from onnx import helper, numpy_helper

from framework.comparator import compare_outputs
from framework.onnx_utils import make_model_from_nodes, np_to_onnx_type


def _make_one_hot_model(
    idx_shape: list[int],
    depth: int,
    axis: int,
    off_on: tuple[float, float],
    out_shape: list[int],
):
    indices = helper.make_tensor_value_info(
        "indices", np_to_onnx_type(np.int64), list(idx_shape)
    )
    out = helper.make_tensor_value_info(
        "output", np_to_onnx_type(np.float32), list(out_shape)
    )
    depth_init = numpy_helper.from_array(np.array(depth, dtype=np.int64), name="depth")
    values_init = numpy_helper.from_array(
        np.array(off_on, dtype=np.float32), name="values"
    )
    node = helper.make_node(
        "OneHot", ["indices", "depth", "values"], ["output"], axis=axis
    )
    return make_model_from_nodes(
        [node], [indices], [out], initializers=[depth_init, values_init]
    )


class TestOneHot:
    def test_one_hot_axis1(self, model_runner):
        idx_shape = [2, 2]
        depth = 10
        axis = 1
        out_shape = [2, depth, 2]
        model = _make_one_hot_model(idx_shape, depth, axis, (0.0, 1.0), out_shape)
        rng = np.random.default_rng(801)
        indices = rng.integers(0, depth, idx_shape, dtype=np.int64)
        actual, expected = model_runner.run_sample(model, [indices])
        compare_outputs(actual, expected, atol=0)

    def test_one_hot_axis_last(self, model_runner):
        """axis at the end (rank r indices -> [.., depth] output)."""
        idx_shape = [3, 4]
        depth = 6
        axis = 2  # == indices_rank, appends the one-hot dim last
        out_shape = [3, 4, depth]
        model = _make_one_hot_model(idx_shape, depth, axis, (-1.0, 3.0), out_shape)
        rng = np.random.default_rng(802)
        indices = rng.integers(0, depth, idx_shape, dtype=np.int64)
        actual, expected = model_runner.run_sample(model, [indices])
        compare_outputs(actual, expected, atol=0)
