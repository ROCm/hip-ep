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
import pytest
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


def _make_qlinear_matmul_case(a_shape, b, y_dtype, y_np, scales, a_np=np.uint8):
    """Build one QLinearMatMul.

    `scales` is a_scale, a_zp, b_scale, b_zp, y_scale, y_zp.
    """
    a_scale, a_zp, b_scale, b_zp, y_scale, y_zp = scales
    a_type = TensorProto.UINT8 if a_np == np.uint8 else TensorProto.INT8
    a_info = helper.make_tensor_value_info("a", a_type, list(a_shape))
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
        numpy_helper.from_array(np.array(a_zp, dtype=a_np), "a_zero_point"),
        numpy_helper.from_array(b, "b"),
        numpy_helper.from_array(np.array(b_scale, dtype=np.float32), "b_scale"),
        numpy_helper.from_array(np.array(b_zp, dtype=np.int8), "b_zero_point"),
        numpy_helper.from_array(np.array(y_scale, dtype=np.float32), "y_scale"),
        numpy_helper.from_array(np.array(y_zp, dtype=y_np), "y_zero_point"),
    ]
    return make_model_from_nodes(
        [node], [a_info], [y_info], initializers=inits, opset=12
    )


def _make_qlinear_matmul_arrays(
    a_shape, b, a_scale, a_zp, b_scale, b_zp, y_scale, y_zp
):
    """Build one QLinearMatMul from numpy parameters.

    B, its scale, and its zero point may be per-tensor or one value per column.
    """
    y_dtype = (
        TensorProto.UINT8 if np.dtype(y_zp.dtype) == np.uint8 else TensorProto.INT8
    )
    a_type = (
        TensorProto.UINT8 if np.asarray(a_zp).dtype == np.uint8 else TensorProto.INT8
    )
    a_info = helper.make_tensor_value_info("a", a_type, list(a_shape))
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
        numpy_helper.from_array(np.asarray(a_scale, dtype=np.float32), "a_scale"),
        numpy_helper.from_array(np.asarray(a_zp), "a_zero_point"),
        numpy_helper.from_array(np.asarray(b), "b"),
        numpy_helper.from_array(np.asarray(b_scale, dtype=np.float32), "b_scale"),
        numpy_helper.from_array(np.asarray(b_zp), "b_zero_point"),
        numpy_helper.from_array(np.asarray(y_scale, dtype=np.float32), "y_scale"),
        numpy_helper.from_array(np.asarray(y_zp), "y_zero_point"),
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
        # 0, 1, 0, -1. The output is int8 so the negative results are kept;
        # A is int8 too, as the CPU reference needs the two types to match.
        b = np.array([[1, 3, -1, -3]], dtype=np.int8)
        model = _make_qlinear_matmul_case(
            (1, 1),
            b,
            TensorProto.INT8,
            np.int8,
            (1.0, 0, 1.0, 0, 2.0, 0),
            a_np=np.int8,
        )
        a = np.array([[1]], dtype=np.int8)
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

    def test_per_column_b(self, model_runner):
        # B's scale and zero point have one value per output column. A constant
        # fill would hide a kernel that still reads element 0.
        b = np.array([[2, 1, 0], [3, 0, 4], [1, 2, 1], [0, 3, 2]], dtype=np.uint8)
        b_scale = np.array([0.5, 1.0, 2.0], dtype=np.float32)
        b_zp = np.array([1, 0, 2], dtype=np.uint8)
        a = np.array([[1, 2, 3, 4], [0, 5, 1, 2]], dtype=np.uint8)
        model = _make_qlinear_matmul_arrays(
            a.shape,
            b,
            np.float32(1.0),
            np.uint8(1),
            b_scale,
            b_zp,
            np.float32(1.0),
            np.uint8(3),
        )
        actual, expected = model_runner.run_sample(model, [a])
        compare_outputs(actual, expected, atol=0, rtol=0)
        got = np.asarray(actual[0])

        def cell(row, col):
            acc = 0
            for ki in range(b.shape[0]):
                acc += (int(a[row, ki]) - 1) * (int(b[ki, col]) - int(b_zp[col]))
            q = np.rint(acc * float(b_scale[col])) + 3
            return int(np.clip(q, 0, 255))

        for row, col in ((0, 0), (0, 2), (1, 1)):
            assert int(got[row, col]) == cell(row, col)

    def test_per_column_b_scalar_zp(self, model_runner):
        # Scale is per-column and the zero point stays a scalar, so the two
        # counts are independent.
        b = np.array([[1, 4], [2, 0], [3, 1]], dtype=np.int8)
        b_scale = np.array([2.0, 0.5], dtype=np.float32)
        a = np.array([[2, 1, 0]], dtype=np.uint8)
        model = _make_qlinear_matmul_arrays(
            a.shape,
            b,
            np.float32(1.0),
            np.uint8(0),
            b_scale,
            np.int8(1),
            np.float32(1.0),
            np.uint8(0),
        )
        actual, expected = model_runner.run_sample(model, [a])
        compare_outputs(actual, expected, atol=0, rtol=0)
        got = np.asarray(actual[0])
        # Column 0: dot 1, scale 2. Column 1: dot 5, scale 0.5, rint(2.5) == 2.
        assert int(got[0, 0]) == 2
        assert int(got[0, 1]) == 2


def _make_requant_matmul(
    rng,
    m: int,
    k: int,
    n: int,
    *,
    a_dtype=np.uint8,
    a_zp: int,
    y_zp: int,
    a_scale: float = 0.0173,
    y_scale: float = 0.0912,
):
    """A QLinearMatMul with non-dyadic, per-column B scales.

    The output type matches A, as the CPU reference requires. B stays within
    +-63: the reference's AVX2 u8 x s8 kernel sums pairs of products in int16,
    which saturates for larger values.
    """
    info = np.iinfo(a_dtype)
    b = rng.integers(-63, 64, size=(k, n), dtype=np.int8)
    model = _make_qlinear_matmul_arrays(
        (m, k),
        b,
        np.float32(a_scale),
        np.array(a_zp, dtype=a_dtype),
        rng.uniform(0.002, 0.02, size=(n,)).astype(np.float32),
        np.zeros((n,), dtype=np.int8),
        np.float32(y_scale),
        np.array(y_zp, dtype=a_dtype),
    )
    a = rng.integers(info.min, info.max + 1, size=(m, k), dtype=a_dtype)
    return model, a


class TestQLinearMatMulRequantize:
    """Full-range activations and zero points with an f32 requantize.

    These graphs meet the rocMLIR route's conditions: constant parameters and a
    zero B zero point. With HIPDNN_EP_ROCMLIR=1 they run as one rocMLIR
    kernel, where a u8 A is rebiased to a - 128 and a per-column constant
    corrects the zero point. B above the ORT bridge clone threshold (127 bytes
    by default) is not an inline constant; raise
    XLNX_ABI_2_0_CLONE_EXTERNAL_DATA_THRESHOLD so that every case here takes
    the rocMLIR route.
    """

    @pytest.mark.parametrize("a_zp", [0, 115, 255])
    def test_u8_zero_point(self, model_runner, a_zp):
        rng = np.random.default_rng(600 + a_zp)
        model, a = _make_requant_matmul(rng, 8, 64, 24, a_zp=a_zp, y_zp=128)
        actual, expected = model_runner.run_sample(model, [a])
        compare_outputs(actual, expected, atol=0, rtol=0)

    def test_row_vector(self, model_runner):
        rng = np.random.default_rng(700)
        model, a = _make_requant_matmul(rng, 1, 96, 40, a_zp=42, y_zp=100)
        actual, expected = model_runner.run_sample(model, [a])
        compare_outputs(actual, expected, atol=0, rtol=0)

    def test_i8(self, model_runner):
        rng = np.random.default_rng(800)
        model, a = _make_requant_matmul(
            rng, 8, 64, 24, a_dtype=np.int8, a_zp=-20, y_zp=5
        )
        actual, expected = model_runner.run_sample(model, [a])
        compare_outputs(actual, expected, atol=0, rtol=0)

    @pytest.mark.parametrize("a_dtype", [np.uint8, np.int8], ids=["u8", "i8"])
    def test_saturation(self, model_runner, a_dtype):
        # A small output scale pushes most results past both ends of the range.
        rng = np.random.default_rng(900)
        model, a = _make_requant_matmul(
            rng,
            8,
            64,
            24,
            a_dtype=a_dtype,
            a_zp=128 if a_dtype == np.uint8 else 0,
            y_zp=0,
            y_scale=0.004,
        )
        actual, expected = model_runner.run_sample(model, [a])
        compare_outputs(actual, expected, atol=0, rtol=0)
        got = np.asarray(actual[0])
        info = np.iinfo(a_dtype)
        low = info.min if a_dtype == np.int8 else 0
        assert (got == low).any() and (got == info.max).any()
