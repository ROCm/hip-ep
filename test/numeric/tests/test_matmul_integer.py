#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""MatMulInteger.

Values stay inside the f32 mantissa, so the f32 hip.matmul accumulator matches
the int32 dot product exactly. A long K is not bit-identical.
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
