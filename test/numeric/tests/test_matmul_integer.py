#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""MatMulInteger.

A reduction of at most 258 accumulates in f32, which is exact for an 8-bit
dot product. A longer K accumulates in i32.
"""

from __future__ import annotations

import numpy as np
from onnx import TensorProto, helper, numpy_helper

from framework.comparator import compare_outputs
from framework.onnx_utils import make_model_from_nodes


def _matmul_integer(a, b, a_zp=None, b_zp=None):
    a_dtype = TensorProto.INT8 if a.dtype == np.int8 else TensorProto.UINT8
    batch = list(np.broadcast_shapes(a.shape[:-2], b.shape[:-2]))
    out_shape = batch + [a.shape[-2], b.shape[-1]]
    a_info = helper.make_tensor_value_info("A", a_dtype, list(a.shape))
    y_info = helper.make_tensor_value_info("Y", TensorProto.INT32, out_shape)
    inputs = ["A", "B"]
    inits = [numpy_helper.from_array(b, "B")]
    if a_zp is not None:
        inputs.append("a_zp")
        inits.append(numpy_helper.from_array(np.asarray(a_zp, dtype=a.dtype), "a_zp"))
    if b_zp is not None:
        if a_zp is None:
            raise ValueError("b_zero_point requires a_zero_point to be present")
        inputs.append("b_zp")
        inits.append(numpy_helper.from_array(np.asarray(b_zp, dtype=b.dtype), "b_zp"))
    node = helper.make_node("MatMulInteger", inputs, ["Y"])
    return make_model_from_nodes(
        [node], [a_info], [y_info], initializers=inits, opset=13
    )


class TestMatMulInteger:
    def test_per_column_zero_point(self, model_runner):
        a = np.array([[1, 2, 3, 4]], dtype=np.uint8)
        b = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.uint8)
        model = _matmul_integer(a, b, 1, np.array([1, 2], dtype=np.uint8))
        actual, expected = model_runner.run_sample(model, [a])
        compare_outputs(actual, expected, atol=0, rtol=0)

    def test_per_row_zero_point(self, model_runner):
        a = np.array([[4, 5], [6, 7]], dtype=np.uint8)
        b = np.array([[3], [4]], dtype=np.uint8)
        model = _matmul_integer(a, b, np.array([1, 2], dtype=np.uint8), np.uint8(1))
        actual, expected = model_runner.run_sample(model, [a])
        compare_outputs(actual, expected, atol=0, rtol=0)

    def test_no_zero_point_int8(self, model_runner):
        a = np.array([[2, -1]], dtype=np.int8)
        b = np.array([[3], [4]], dtype=np.int8)
        model = _matmul_integer(a, b)
        actual, expected = model_runner.run_sample(model, [a])
        compare_outputs(actual, expected, atol=0, rtol=0)

    def test_batched(self, model_runner):
        a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.uint8)
        b = np.array([[1, 0], [0, 1]], dtype=np.uint8)
        model = _matmul_integer(a, b, np.uint8(0), np.uint8(0))
        actual, expected = model_runner.run_sample(model, [a])
        compare_outputs(actual, expected, atol=0, rtol=0)

    def test_broadcast_batch(self, model_runner):
        a = np.arange(6, dtype=np.uint8).reshape(1, 2, 3)
        b = np.arange(60, dtype=np.uint8).reshape(4, 3, 5)
        model = _matmul_integer(a, b, np.uint8(0), np.uint8(0))
        actual, expected = model_runner.run_sample(model, [a])
        compare_outputs(actual, expected, atol=0, rtol=0)

    def test_wide_k(self, model_runner):
        a = np.full((1, 259), 255, dtype=np.uint8)
        b = np.full((259, 1), 255, dtype=np.uint8)
        model = _matmul_integer(a, b)
        actual, expected = model_runner.run_sample(model, [a])
        compare_outputs(actual, expected, atol=0, rtol=0)
        assert int(np.asarray(actual[0]).reshape(-1)[0]) == 16841475

    def test_wide_k_batched(self, model_runner):
        # K=259 takes the i32 kernel. Each batch has its own B, and each
        # row and column has a different factor, so a wrong batch stride or
        # a swapped index changes the result.
        batch, rows, k, cols = 2, 3, 259, 4
        a = np.empty((batch, rows, k), dtype=np.uint8)
        b = np.empty((batch, k, cols), dtype=np.uint8)
        for bi in range(batch):
            for row in range(rows):
                a[bi, row, :] = row + 1
            for col in range(cols):
                b[bi, :, col] = (bi + 1) * (col + 1)
        model = _matmul_integer(a, b)
        actual, expected = model_runner.run_sample(model, [a])
        compare_outputs(actual, expected, atol=0, rtol=0)
        assert int(np.asarray(actual[0])[1, 2, 3]) == 3 * 2 * 4 * k

    def test_wide_k_shared_b(self, model_runner):
        # Same i32 kernel, with one rank-2 B reused by every batch of A.
        batch, rows, k, cols = 2, 3, 259, 4
        a = np.empty((batch, rows, k), dtype=np.uint8)
        for row in range(rows):
            a[:, row, :] = row + 1
        b = np.empty((k, cols), dtype=np.uint8)
        for col in range(cols):
            b[:, col] = col + 1
        model = _matmul_integer(a, b)
        actual, expected = model_runner.run_sample(model, [a])
        compare_outputs(actual, expected, atol=0, rtol=0)
        got = np.asarray(actual[0])
        assert int(got[0, 2, 3]) == 3 * 4 * k
        assert int(got[1, 2, 3]) == int(got[0, 2, 3])
