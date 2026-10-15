#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""Numeric checks for com.microsoft QLinearAdd / QLinearMul / QLinearConcat /
QLinearGlobalAveragePool.

LIT only checks the lowered IR. These compare hip-ep against the ORT CPU
reference, including rounding, broadcast, mixed int8/uint8 concat, and the
channels_last pool path.
"""

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from framework.comparator import compare_outputs
from framework.onnx_utils import make_model_from_nodes

_MS = [helper.make_opsetid("com.microsoft", 1)]


def _init(name, value, dtype):
    return numpy_helper.from_array(np.array(value, dtype=dtype), name=name)


def _model(nodes, inputs, outputs, initializers):
    return make_model_from_nodes(
        nodes,
        inputs,
        outputs,
        initializers=initializers,
        extra_opsets=_MS,
    )


def _qlinear_math_model(
    op_type, a_dtype, b_dtype, out_dtype, a_shape, b_shape, out_shape
):
    """QLinearAdd / QLinearMul: A, scale, zp, B, scale, zp, C_scale, C_zp."""
    A = helper.make_tensor_value_info("A", a_dtype, a_shape)
    B = helper.make_tensor_value_info("B", b_dtype, b_shape)
    C = helper.make_tensor_value_info("C", out_dtype, out_shape)
    inits = [
        _init("A_scale", 0.5, np.float32),
        _init("A_zp", 1, np.int8 if a_dtype == TensorProto.INT8 else np.uint8),
        _init("B_scale", 0.25, np.float32),
        _init("B_zp", 2, np.int8 if b_dtype == TensorProto.INT8 else np.uint8),
        _init("C_scale", 0.5, np.float32),
        _init("C_zp", 3, np.int8 if out_dtype == TensorProto.INT8 else np.uint8),
    ]
    node = helper.make_node(
        op_type,
        ["A", "A_scale", "A_zp", "B", "B_scale", "B_zp", "C_scale", "C_zp"],
        ["C"],
        domain="com.microsoft",
    )
    return _model([node], [A, B], [C], inits)


class TestQLinearAdd:
    def test_rounding(self, model_runner):
        """Values that land on a .5 boundary exercise quantize rounding."""
        model = _qlinear_math_model(
            "QLinearAdd",
            TensorProto.INT8,
            TensorProto.INT8,
            TensorProto.INT8,
            [8],
            [8],
            [8],
        )
        # (q - zp) * scale sits on halves after the add and re-quant.
        a = np.array([0, 1, 2, 3, -4, -1, 7, -8], dtype=np.int8)
        b = np.array([2, 2, 4, 6, 0, 2, 10, -2], dtype=np.int8)
        actual, expected = model_runner.run_sample(model, [a, b])
        compare_outputs(actual, expected, atol=1)


class TestQLinearMul:
    def test_broadcast(self, model_runner):
        model = _qlinear_math_model(
            "QLinearMul",
            TensorProto.UINT8,
            TensorProto.UINT8,
            TensorProto.UINT8,
            [1, 4],
            [4],
            [1, 4],
        )
        a = np.array([[1, 8, 16, 40]], dtype=np.uint8)
        b = np.array([2, 4, 8, 12], dtype=np.uint8)
        actual, expected = model_runner.run_sample(model, [a, b])
        compare_outputs(actual, expected, atol=1)


def _dequant(q, scale, zp):
    return (q.astype(np.int32) - np.int32(zp)).astype(np.float32) * np.float32(scale)


def _quant(values, scale, zp, dtype):
    info = np.iinfo(dtype)
    rounded = np.rint(values / np.float32(scale)).astype(np.int32) + np.int32(zp)
    return np.clip(rounded, info.min, info.max).astype(dtype)


class TestQLinearConcat:
    def test_same_type(self, model_runner):
        A = helper.make_tensor_value_info("A", TensorProto.UINT8, [1, 2])
        B = helper.make_tensor_value_info("B", TensorProto.UINT8, [1, 3])
        Y = helper.make_tensor_value_info("Y", TensorProto.UINT8, [1, 5])
        inits = [
            _init("Y_scale", 0.5, np.float32),
            _init("Y_zp", 1, np.uint8),
            _init("A_scale", 0.25, np.float32),
            _init("A_zp", 2, np.uint8),
            _init("B_scale", 0.5, np.float32),
            _init("B_zp", 4, np.uint8),
        ]
        node = helper.make_node(
            "QLinearConcat",
            ["Y_scale", "Y_zp", "A", "A_scale", "A_zp", "B", "B_scale", "B_zp"],
            ["Y"],
            domain="com.microsoft",
            axis=1,
        )
        model = _model([node], [A, B], [Y], inits)
        a = np.array([[4, 10]], dtype=np.uint8)
        b = np.array([[0, 8, 20]], dtype=np.uint8)
        actual, expected = model_runner.run_sample(model, [a, b])
        compare_outputs(actual, expected, atol=1)

    def test_mixed_signedness(self, model_runner, tmp_path):
        """int8 and uint8 inputs. ORT's CPU kernel rejects mismatched zero-point
        types, so the reference is the dequant-concat-quant formula."""
        A = helper.make_tensor_value_info("A", TensorProto.INT8, [1, 2])
        B = helper.make_tensor_value_info("B", TensorProto.UINT8, [1, 3])
        Y = helper.make_tensor_value_info("Y", TensorProto.UINT8, [1, 5])
        inits = [
            _init("Y_scale", 0.5, np.float32),
            _init("Y_zp", 1, np.uint8),
            _init("A_scale", 0.25, np.float32),
            _init("A_zp", -2, np.int8),
            _init("B_scale", 0.5, np.float32),
            _init("B_zp", 4, np.uint8),
        ]
        node = helper.make_node(
            "QLinearConcat",
            ["Y_scale", "Y_zp", "A", "A_scale", "A_zp", "B", "B_scale", "B_zp"],
            ["Y"],
            domain="com.microsoft",
            axis=1,
        )
        model = _model([node], [A, B], [Y], inits)
        a = np.array([[-4, 6]], dtype=np.int8)
        b = np.array([[0, 8, 20]], dtype=np.uint8)
        model_path = tmp_path / "mixed.onnx"
        onnx.save(model, model_path)
        actual = model_runner.backend.run(str(model_path), [a, b])
        parts = np.concatenate([_dequant(a, 0.25, -2), _dequant(b, 0.5, 4)], axis=1)
        expected = [_quant(parts, 0.5, 1, np.uint8)]
        compare_outputs(actual, expected, atol=1)


class TestQLinearGlobalAveragePool:
    @pytest.mark.parametrize(
        "channels_last,in_shape,out_shape",
        [
            (0, [1, 4, 2, 2], [1, 4, 1, 1]),
            (1, [1, 2, 2, 4], [1, 1, 1, 4]),
        ],
    )
    def test_pool(self, model_runner, channels_last, in_shape, out_shape):
        X = helper.make_tensor_value_info("X", TensorProto.UINT8, in_shape)
        Y = helper.make_tensor_value_info("Y", TensorProto.UINT8, out_shape)
        inits = [
            _init("X_scale", 0.5, np.float32),
            _init("X_zp", 2, np.uint8),
            _init("Y_scale", 0.25, np.float32),
            _init("Y_zp", 1, np.uint8),
        ]
        node = helper.make_node(
            "QLinearGlobalAveragePool",
            ["X", "X_scale", "X_zp", "Y_scale", "Y_zp"],
            ["Y"],
            domain="com.microsoft",
            channels_last=channels_last,
        )
        model = _model([node], [X], [Y], inits)
        rng = np.random.default_rng(7)
        x = rng.integers(0, 32, in_shape, dtype=np.uint8)
        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=1)
