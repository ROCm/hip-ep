#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""Tests for the ONNX TopK op (values + indices outputs).

K is normally supplied as a constant initializer, so the output extent (k) is
static and the compiler passes it by value to wrap_top_k_static. One case
instead feeds K as a graph input with a symbolic output extent, so the runtime
reads it from the K buffer through wrap_top_k; both entry points run the same
kernel and must agree with ORT. `largest` and `sorted` are both covered.

Inputs use DISTINCT per-row values (no ties), so the selected set and its
sorted order -- and therefore the returned indices -- are unambiguous and can
be compared bit-exact against the ORT CPU reference (atol=0).
"""

from __future__ import annotations

import numpy as np
import pytest
from onnx import helper, numpy_helper

from framework.comparator import compare_outputs
from framework.onnx_utils import make_model_from_nodes, np_to_onnx_type


def _make_topk_model(x_shape: list[int], k: int, axis: int, largest: int, sorted_: int):
    out_shape = list(x_shape)
    out_shape[axis] = k
    X = helper.make_tensor_value_info("x", np_to_onnx_type(np.float32), list(x_shape))
    values = helper.make_tensor_value_info(
        "values", np_to_onnx_type(np.float32), out_shape
    )
    indices = helper.make_tensor_value_info(
        "indices", np_to_onnx_type(np.int64), out_shape
    )
    k_init = numpy_helper.from_array(np.array([k], dtype=np.int64), name="k")
    node = helper.make_node(
        "TopK",
        ["x", "k"],
        ["values", "indices"],
        axis=axis,
        largest=largest,
        sorted=sorted_,
    )
    return make_model_from_nodes([node], [X], [values, indices], initializers=[k_init])


def _distinct_rows(rng, rows: int, cols: int) -> np.ndarray:
    """Each row is a shuffled arange -> globally distinct within a row, so
    TopK has no ties to break."""
    out = np.empty((rows, cols), dtype=np.float32)
    for r in range(rows):
        out[r] = rng.permutation(cols).astype(np.float32) + rng.uniform(
            -0.25, 0.25, cols
        ).astype(np.float32)
    return out


class TestTopK:
    @pytest.mark.parametrize("largest", [1, 0])
    @pytest.mark.parametrize("sorted_", [1])
    def test_topk_axis1(self, model_runner, largest, sorted_):
        x_shape = [3, 8]
        k = 3
        model = _make_topk_model(x_shape, k, axis=1, largest=largest, sorted_=sorted_)
        rng = np.random.default_rng(601)
        x = _distinct_rows(rng, x_shape[0], x_shape[1])
        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=0)

    def test_topk_full_k(self, model_runner):
        """k == axis length -> a full sort of each row."""
        x_shape = [4, 5]
        k = 5
        model = _make_topk_model(x_shape, k, axis=1, largest=1, sorted_=1)
        rng = np.random.default_rng(602)
        x = _distinct_rows(rng, x_shape[0], x_shape[1])
        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=0)

    def test_topk_runtime_k(self, model_runner):
        """K as a graph input with a symbolic output extent: the runtime reads
        K from its buffer (wrap_top_k) instead of receiving it by value."""
        x_shape = [3, 8]
        k = 3
        X = helper.make_tensor_value_info("x", np_to_onnx_type(np.float32), x_shape)
        K = helper.make_tensor_value_info("k", np_to_onnx_type(np.int64), [1])
        values = helper.make_tensor_value_info(
            "values", np_to_onnx_type(np.float32), [x_shape[0], "k_out"]
        )
        indices = helper.make_tensor_value_info(
            "indices", np_to_onnx_type(np.int64), [x_shape[0], "k_out"]
        )
        node = helper.make_node(
            "TopK", ["x", "k"], ["values", "indices"], axis=1, largest=1, sorted=1
        )
        model = make_model_from_nodes([node], [X, K], [values, indices])
        rng = np.random.default_rng(603)
        x = _distinct_rows(rng, x_shape[0], x_shape[1])
        actual, expected = model_runner.run_sample(
            model, [x, np.array([k], dtype=np.int64)]
        )
        compare_outputs(actual, expected, atol=0)
