#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""Tests for the logical/comparison ops Or and Greater.

Or      : bool inputs, bool output (element-wise logical OR).
Greater : numeric inputs, bool output. The converter decomposes Greater(a, b)
          into Less(b, a) (operand swap), so this also guards that the swap is
          wired correctly. Both float and integer inputs are covered.

The runtime kernels see equal-shaped operands (broadcasting is materialised by
an upstream Expand), so tests use same-shape A/B. Bool results are bit-exact
(atol=0). Inputs are seeded to yield a mix of True/False.
"""

from __future__ import annotations

import numpy as np
import pytest
from onnx import helper

from framework.comparator import compare_outputs
from framework.onnx_utils import make_model_from_nodes, np_to_onnx_type

SEQ_LENS = [1, 128]


def _make_binary_model(
    op_type: str, in_dtype: np.dtype, out_dtype: np.dtype, shape: list[int]
):
    A = helper.make_tensor_value_info("A", np_to_onnx_type(in_dtype), list(shape))
    B = helper.make_tensor_value_info("B", np_to_onnx_type(in_dtype), list(shape))
    Y = helper.make_tensor_value_info("Y", np_to_onnx_type(out_dtype), list(shape))
    node = helper.make_node(op_type, ["A", "B"], ["Y"])
    return make_model_from_nodes([node], [A, B], [Y])


class TestOr:
    @pytest.mark.parametrize("shape", [[4, 8], [3, 5, 7]])
    def test_or(self, model_runner, shape):
        model = _make_binary_model("Or", np.bool_, np.bool_, shape)
        rng = np.random.default_rng(1001)
        a = rng.integers(0, 2, shape).astype(np.bool_)
        b = rng.integers(0, 2, shape).astype(np.bool_)
        actual, expected = model_runner.run_sample(model, [a, b])
        compare_outputs(actual, expected, atol=0)

    @pytest.mark.parametrize("seq_len", SEQ_LENS)
    def test_or_attention_mask_shape(self, model_runner, seq_len):
        shape = [1, seq_len]
        model = _make_binary_model("Or", np.bool_, np.bool_, shape)
        rng = np.random.default_rng(1002)
        a = rng.integers(0, 2, shape).astype(np.bool_)
        b = rng.integers(0, 2, shape).astype(np.bool_)
        actual, expected = model_runner.run_sample(model, [a, b])
        compare_outputs(actual, expected, atol=0)


class TestGreater:
    @pytest.mark.parametrize("dtype", [np.float16, np.float32, np.int32, np.int64])
    def test_greater(self, model_runner, dtype):
        shape = [4, 8]
        model = _make_binary_model("Greater", dtype, np.bool_, shape)
        rng = np.random.default_rng(1003)
        if np.issubdtype(dtype, np.integer):
            a = rng.integers(-10, 10, shape, dtype=dtype)
            b = rng.integers(-10, 10, shape, dtype=dtype)
        else:
            a = rng.uniform(-3.0, 3.0, shape).astype(dtype)
            b = rng.uniform(-3.0, 3.0, shape).astype(dtype)
        actual, expected = model_runner.run_sample(model, [a, b])
        compare_outputs(actual, expected, atol=0)

    @pytest.mark.parametrize("seq_len", SEQ_LENS)
    def test_greater_threshold_shape(self, model_runner, seq_len):
        """i64 [1, S] `>` pattern (position-id thresholding)."""
        shape = [1, seq_len]
        model = _make_binary_model("Greater", np.int64, np.bool_, shape)
        rng = np.random.default_rng(1004)
        a = rng.integers(0, seq_len, shape, dtype=np.int64)
        b = np.full(shape, max(1, seq_len // 2), dtype=np.int64)
        actual, expected = model_runner.run_sample(model, [a, b])
        compare_outputs(actual, expected, atol=0)
