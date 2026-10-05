#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""Tests for ONNX QLinearConv.

The MobileNet int8 graph is entirely this shape: uint8 activations, int8
weights, scalar scales and zero points, an int32 bias, and either a grouped
3x3 or a 1x1 convolution. Scales here are dyadic so the integer dot product
requantizes exactly.
"""

from __future__ import annotations

import numpy as np
from onnx import TensorProto, helper, numpy_helper

from framework.comparator import compare_outputs
from framework.onnx_utils import make_model_from_nodes


def _out_dim(in_dim: int, pad: int, kernel: int, stride: int) -> int:
    return (in_dim + pad + pad - (kernel - 1) - 1) // stride + 1


def _make_qlinear_conv(
    x_shape: list[int],
    w_shape: list[int],
    *,
    group: int,
    stride: int,
    pad: int,
    bias: bool,
    per_channel: bool,
):
    cout = w_shape[0]
    out_shape = [
        x_shape[0],
        cout,
        _out_dim(x_shape[2], pad, w_shape[2], stride),
        _out_dim(x_shape[3], pad, w_shape[3], stride),
    ]
    x_info = helper.make_tensor_value_info("x", TensorProto.UINT8, x_shape)
    y_info = helper.make_tensor_value_info("y", TensorProto.UINT8, out_shape)
    inputs = [
        "x",
        "x_scale",
        "x_zero_point",
        "w",
        "w_scale",
        "w_zero_point",
        "y_scale",
        "y_zero_point",
    ]
    if bias:
        inputs.append("B")
    node = helper.make_node(
        "QLinearConv",
        inputs,
        ["y"],
        kernel_shape=[w_shape[2], w_shape[3]],
        strides=[stride, stride],
        pads=[pad, pad, pad, pad],
        dilations=[1, 1],
        group=group,
        auto_pad="NOTSET",
    )
    rng = np.random.default_rng(7)
    w = rng.integers(-1, 2, size=w_shape, dtype=np.int8)
    if per_channel:
        w_scale = np.array([0.5, 1.0, 0.5, 1.0][:cout], dtype=np.float32)
        w_zp = np.zeros((cout,), dtype=np.int8)
    else:
        w_scale = np.array(0.5, dtype=np.float32)
        w_zp = np.array(0, dtype=np.int8)
    inits = [
        numpy_helper.from_array(np.array(0.5, dtype=np.float32), "x_scale"),
        numpy_helper.from_array(np.array(1, dtype=np.uint8), "x_zero_point"),
        numpy_helper.from_array(w, "w"),
        numpy_helper.from_array(w_scale, "w_scale"),
        numpy_helper.from_array(w_zp, "w_zero_point"),
        numpy_helper.from_array(np.array(0.25, dtype=np.float32), "y_scale"),
        numpy_helper.from_array(np.array(2, dtype=np.uint8), "y_zero_point"),
    ]
    if bias:
        b = rng.integers(-1, 2, size=(cout,), dtype=np.int32)
        inits.append(numpy_helper.from_array(b, "B"))
    return make_model_from_nodes(
        [node], [x_info], [y_info], initializers=inits, opset=12
    )


class TestQLinearConv:
    def test_stride2_with_bias(self, model_runner):
        model = _make_qlinear_conv(
            [1, 3, 8, 8],
            [4, 3, 3, 3],
            group=1,
            stride=2,
            pad=1,
            bias=True,
            per_channel=False,
        )
        x = np.random.default_rng(11).integers(0, 5, size=(1, 3, 8, 8), dtype=np.uint8)
        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=0, rtol=0)

    def test_depthwise(self, model_runner):
        model = _make_qlinear_conv(
            [1, 4, 6, 6],
            [4, 1, 3, 3],
            group=4,
            stride=1,
            pad=1,
            bias=True,
            per_channel=False,
        )
        x = np.random.default_rng(12).integers(0, 5, size=(1, 4, 6, 6), dtype=np.uint8)
        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=0, rtol=0)

    def test_pointwise_no_bias(self, model_runner):
        model = _make_qlinear_conv(
            [1, 4, 5, 5],
            [6, 4, 1, 1],
            group=1,
            stride=1,
            pad=0,
            bias=False,
            per_channel=False,
        )
        x = np.random.default_rng(13).integers(0, 5, size=(1, 4, 5, 5), dtype=np.uint8)
        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=0, rtol=0)

    def test_per_channel_weight_scale(self, model_runner):
        model = _make_qlinear_conv(
            [1, 2, 3, 3],
            [4, 2, 1, 1],
            group=1,
            stride=1,
            pad=0,
            bias=True,
            per_channel=True,
        )
        x = np.random.default_rng(14).integers(0, 5, size=(1, 2, 3, 3), dtype=np.uint8)
        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=0, rtol=0)
