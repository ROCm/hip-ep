#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""Tests for ONNX QLinearMatMul.

The MobileNetV3 int8 graph uses this shape: uint8 [1, K] times int8 [K, N],
with scalar scales and zero points. The exact cases use a scale ratio of 1.
Separate cases hit halfway values, so ties-to-even differs from truncation,
and products outside the 8-bit range, so a missing clamp fails.
"""

from __future__ import annotations

import numpy as np
from onnx import TensorProto, helper, numpy_helper

from framework.comparator import compare_outputs
from framework.onnx_utils import make_model_from_nodes


def _make_qlinear_matmul(m: int, k: int, n: int, b_zero_point: int):
    a_info = helper.make_tensor_value_info("a", TensorProto.UINT8, [m, k])
    y_info = helper.make_tensor_value_info("y", TensorProto.UINT8, [m, n])
    node = helper.make_node(
        "QLinearMatMul",
        [
            "a",
            "a_scale",
            "a_zero_point",
            "b",
            "b_scale",
            "b_zero_point",
            "y_scale",
            "y_zero_point",
        ],
        ["y"],
    )
    rng = np.random.default_rng(7)
    b = rng.integers(-1, 2, size=(k, n), dtype=np.int8)
    inits = [
        numpy_helper.from_array(np.array(0.5, dtype=np.float32), "a_scale"),
        numpy_helper.from_array(np.array(1, dtype=np.uint8), "a_zero_point"),
        numpy_helper.from_array(b, "b"),
        numpy_helper.from_array(np.array(0.5, dtype=np.float32), "b_scale"),
        numpy_helper.from_array(np.array(b_zero_point, dtype=np.int8), "b_zero_point"),
        numpy_helper.from_array(np.array(0.25, dtype=np.float32), "y_scale"),
        numpy_helper.from_array(np.array(2, dtype=np.uint8), "y_zero_point"),
    ]
    return make_model_from_nodes(
        [node], [a_info], [y_info], initializers=inits, opset=12
    )


def _make_qlinear_matmul_case(a_shape, b, y_dtype, y_np, scales):
    """Build one QLinearMatMul.

    `scales` is a_scale, a_zp, b_scale, b_zp, y_scale, y_zp.
    """
    a_scale, a_zp, b_scale, b_zp, y_scale, y_zp = scales
    a_info = helper.make_tensor_value_info("a", TensorProto.UINT8, list(a_shape))
    y_info = helper.make_tensor_value_info("y", y_dtype, [a_shape[0], b.shape[1]])
    node = helper.make_node(
        "QLinearMatMul",
        [
            "a",
            "a_scale",
            "a_zero_point",
            "b",
            "b_scale",
            "b_zero_point",
            "y_scale",
            "y_zero_point",
        ],
        ["y"],
    )
    inits = [
        numpy_helper.from_array(np.array(a_scale, dtype=np.float32), "a_scale"),
        numpy_helper.from_array(np.array(a_zp, dtype=np.uint8), "a_zero_point"),
        numpy_helper.from_array(b, "b"),
        numpy_helper.from_array(np.array(b_scale, dtype=np.float32), "b_scale"),
        numpy_helper.from_array(np.array(b_zp, dtype=np.int8), "b_zero_point"),
        numpy_helper.from_array(np.array(y_scale, dtype=np.float32), "y_scale"),
        numpy_helper.from_array(np.array(y_zp, dtype=y_np), "y_zero_point"),
    ]
    return make_model_from_nodes(
        [node], [a_info], [y_info], initializers=inits, opset=12
    )


class TestQLinearMatMul:
    def test_row_vector(self, model_runner):
        model = _make_qlinear_matmul(1, 4, 3, 0)
        a = np.random.default_rng(11).integers(0, 5, size=(1, 4), dtype=np.uint8)
        actual, expected = model_runner.run_sample(model, [a])
        compare_outputs(actual, expected, atol=0, rtol=0)

    def test_matrix_nonzero_b_zp(self, model_runner):
        model = _make_qlinear_matmul(2, 4, 3, 1)
        a = np.random.default_rng(12).integers(0, 5, size=(2, 4), dtype=np.uint8)
        actual, expected = model_runner.run_sample(model, [a])
        compare_outputs(actual, expected, atol=0, rtol=0)

    def test_rounding_ties_to_even(self, model_runner):
        # K = 1, so each output is one product. y_scale = 2 gives
        # 0.5, 1.5, -0.5, -1.5. Ties-to-even is 0, 2, 0, -2; truncation is
        # 0, 1, 0, -1. The output is int8 so the negative results are kept.
        b = np.array([[1, 3, -1, -3]], dtype=np.int8)
        model = _make_qlinear_matmul_case(
            (1, 1),
            b,
            TensorProto.INT8,
            np.int8,
            (1.0, 0, 1.0, 0, 2.0, 0),
        )
        a = np.array([[1]], dtype=np.uint8)
        actual, expected = model_runner.run_sample(model, [a])
        compare_outputs(actual, expected, atol=0, rtol=0)

    def test_saturation(self, model_runner):
        # 200 * 2 = 400 and 200 * -2 = -400. uint8 clamps those to 255 and 0.
        b = np.array([[2, -2]], dtype=np.int8)
        model = _make_qlinear_matmul_case(
            (1, 1),
            b,
            TensorProto.UINT8,
            np.uint8,
            (1.0, 0, 1.0, 0, 1.0, 0),
        )
        a = np.array([[200]], dtype=np.uint8)
        actual, expected = model_runner.run_sample(model, [a])
        compare_outputs(actual, expected, atol=0, rtol=0)
