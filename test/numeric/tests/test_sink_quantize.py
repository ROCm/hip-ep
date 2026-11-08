#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""A QuantizeLinear after ops that only move elements (Concat, Transpose,
nearest Resize) of DequantizeLinear outputs.

With HIPDNN_EP_ROCMLIR=1 the quantize moves back through those ops, so they
run on the 8-bit values. A dequantize with the quantize's scale and zero point
folds away; any other input is requantized alone. Each output equals ORT CPU
except where a requantized value rounds at a half.
"""

from __future__ import annotations

import numpy as np
from onnx import TensorProto, helper, numpy_helper

from framework.comparator import compare_outputs
from framework.onnx_utils import make_model_from_nodes

_RNG = np.random.default_rng(11)
_MS = [helper.make_opsetid("com.microsoft", 1)]
_SHAPE = [1, 16, 10, 10]


def _scalar(name, value, dtype):
    return numpy_helper.from_array(np.array(value, dtype=dtype), name)


def _u8(shape):
    return _RNG.integers(0, 256, size=shape, dtype=np.uint8)


def _u8_info(name, shape):
    return helper.make_tensor_value_info(name, TensorProto.UINT8, shape)


def _qdq_concat(a_params, b_params, y_params):
    """DQ(a), DQ(b) -> Concat(axis 1) -> Q, each with (scale, zero point)."""
    nodes = [
        helper.make_node("DequantizeLinear", ["a", "as", "az"], ["af"]),
        helper.make_node("DequantizeLinear", ["b", "bs", "bz"], ["bf"]),
        helper.make_node("Concat", ["af", "bf"], ["c"], axis=1),
        helper.make_node("QuantizeLinear", ["c", "ys", "yz"], ["y"]),
    ]
    inits = []
    for prefix, (scale, zp) in zip("aby", (a_params, b_params, y_params)):
        inits += [
            _scalar(prefix + "s", scale, np.float32),
            _scalar(prefix + "z", zp, np.uint8),
        ]
    out = [_SHAPE[0], 2 * _SHAPE[1], *_SHAPE[2:]]
    return make_model_from_nodes(
        nodes,
        [_u8_info("a", _SHAPE), _u8_info("b", _SHAPE)],
        [_u8_info("y", out)],
        initializers=inits,
        opset=13,
    )


def _qdq_unary(op_nodes, in_shape, out_shape, inits=()):
    """DQ(a) -> op_nodes (reading "af", writing "f") -> Q, one scale."""
    nodes = [
        helper.make_node("DequantizeLinear", ["a", "s", "z"], ["af"]),
        *op_nodes,
        helper.make_node("QuantizeLinear", ["f", "s", "z"], ["y"]),
    ]
    return make_model_from_nodes(
        nodes,
        [_u8_info("a", in_shape)],
        [_u8_info("y", out_shape)],
        initializers=[
            _scalar("s", 0.0078125, np.float32),
            _scalar("z", 128, np.uint8),
            *inits,
        ],
        opset=13,
    )


def _nearest_resize(src, dst):
    return helper.make_node("Resize", [src, "", "scales"], [dst], mode="nearest")


_SCALES = numpy_helper.from_array(np.array([1, 1, 2, 2], dtype=np.float32), "scales")
_UPSAMPLED = [*_SHAPE[:2], 2 * _SHAPE[2], 2 * _SHAPE[3]]


class TestSinkQuantize:
    def test_concat_matched(self, model_runner):
        params = (0.0078125, 128)
        model = _qdq_concat(params, params, params)
        actual, expected = model_runner.run_sample(model, [_u8(_SHAPE), _u8(_SHAPE)])
        compare_outputs(actual, expected, atol=0, rtol=0)

    def test_concat_mismatched(self, model_runner):
        model = _qdq_concat((0.0078125, 128), (0.0123, 120), (0.0078125, 128))
        actual, expected = model_runner.run_sample(model, [_u8(_SHAPE), _u8(_SHAPE)])
        compare_outputs(actual, expected, atol=1, rtol=0)

    def test_qlinear_concat(self, model_runner):
        """The scales of a QLinearConcat in a quantized SqueezeNet: one input
        has the output scale and the other does not."""
        y_scale, b_scale = 5.773518085479736, 5.398648262023926
        node = helper.make_node(
            "QLinearConcat",
            ["ys", "yz", "a", "as", "az", "b", "bs", "bz"],
            ["y"],
            axis=1,
            domain="com.microsoft",
        )
        inits = []
        for prefix, scale in (("y", y_scale), ("a", y_scale), ("b", b_scale)):
            inits += [
                _scalar(prefix + "s", scale, np.float32),
                _scalar(prefix + "z", 0, np.uint8),
            ]
        out = [_SHAPE[0], 2 * _SHAPE[1], *_SHAPE[2:]]
        model = make_model_from_nodes(
            [node],
            [_u8_info("a", _SHAPE), _u8_info("b", _SHAPE)],
            [_u8_info("y", out)],
            initializers=inits,
            opset=13,
            extra_opsets=_MS,
        )
        actual, expected = model_runner.run_sample(model, [_u8(_SHAPE), _u8(_SHAPE)])
        compare_outputs(actual, expected, atol=1, rtol=0)

    def test_transpose(self, model_runner):
        nhwc = [_SHAPE[0], *_SHAPE[2:], _SHAPE[1]]
        model = _qdq_unary(
            [helper.make_node("Transpose", ["af"], ["f"], perm=[0, 3, 1, 2])],
            nhwc,
            _SHAPE,
        )
        actual, expected = model_runner.run_sample(model, [_u8(nhwc)])
        compare_outputs(actual, expected, atol=0, rtol=0)

    def test_resize_nearest(self, model_runner):
        model = _qdq_unary(
            [_nearest_resize("af", "f")], _SHAPE, _UPSAMPLED, inits=[_SCALES]
        )
        actual, expected = model_runner.run_sample(model, [_u8(_SHAPE)])
        compare_outputs(actual, expected, atol=0, rtol=0)

    def test_u8_resize_nearest(self, model_runner):
        """A nearest Resize of u8 values runs the u8 runtime kernel."""
        model = make_model_from_nodes(
            [_nearest_resize("a", "y")],
            [_u8_info("a", _SHAPE)],
            [_u8_info("y", _UPSAMPLED)],
            initializers=[_SCALES],
            opset=13,
        )
        actual, expected = model_runner.run_sample(model, [_u8(_SHAPE)])
        compare_outputs(actual, expected, atol=0, rtol=0)
