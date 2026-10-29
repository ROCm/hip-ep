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
import pytest
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

    def test_same_upper_odd_pad(self, model_runner):
        # 4x4 input, 3x3 kernel, stride 2 produces 2x2. The pad budget is 1,
        # so SAME_UPPER pads only the bottom and right. A symmetric pad changes
        # the corner sums below.
        x = np.arange(16, dtype=np.uint8).reshape(1, 1, 4, 4)
        w = np.ones((1, 1, 3, 3), dtype=np.int8)
        x_info = helper.make_tensor_value_info("x", TensorProto.UINT8, [1, 1, 4, 4])
        y_info = helper.make_tensor_value_info("y", TensorProto.UINT8, [1, 1, 2, 2])
        node = helper.make_node(
            "QLinearConv",
            [
                "x",
                "x_scale",
                "x_zero_point",
                "w",
                "w_scale",
                "w_zero_point",
                "y_scale",
                "y_zero_point",
            ],
            ["y"],
            kernel_shape=[3, 3],
            strides=[2, 2],
            dilations=[1, 1],
            group=1,
            auto_pad="SAME_UPPER",
        )
        inits = [
            numpy_helper.from_array(np.array(1.0, dtype=np.float32), "x_scale"),
            numpy_helper.from_array(np.array(0, dtype=np.uint8), "x_zero_point"),
            numpy_helper.from_array(w, "w"),
            numpy_helper.from_array(np.array(1.0, dtype=np.float32), "w_scale"),
            numpy_helper.from_array(np.array(0, dtype=np.int8), "w_zero_point"),
            numpy_helper.from_array(np.array(1.0, dtype=np.float32), "y_scale"),
            numpy_helper.from_array(np.array(0, dtype=np.uint8), "y_zero_point"),
        ]
        model = make_model_from_nodes(
            [node], [x_info], [y_info], initializers=inits, opset=12
        )
        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=0, rtol=0)
        got = np.asarray(actual[0])
        assert int(got[0, 0, 0, 0]) == 45
        assert int(got[0, 0, 0, 1]) == 39
        assert int(got[0, 0, 1, 0]) == 66
        assert int(got[0, 0, 1, 1]) == 50


_NP_TO_ONNX = {np.uint8: TensorProto.UINT8, np.int8: TensorProto.INT8}


def _random_q(rng, dtype, shape):
    info = np.iinfo(dtype)
    return rng.integers(info.min, info.max + 1, size=shape, dtype=dtype)


def _make_requant_conv(
    rng,
    x_shape: list[int],
    w_shape: list[int],
    *,
    x_dtype=np.uint8,
    y_dtype=np.uint8,
    x_zp: int,
    y_zp: int,
    x_scale: float = 0.0173,
    y_scale: float = 0.0912,
    group: int = 1,
    stride: int = 1,
    pad: int = 1,
):
    """A QLinearConv with non-dyadic, per-channel scales and a bias.

    Weights stay within +-63: the CPU reference's AVX2 u8 x s8 kernel sums
    pairs of products in int16, which saturates for larger weights.
    """
    cout = w_shape[0]
    out_shape = [
        x_shape[0],
        cout,
        _out_dim(x_shape[2], pad, w_shape[2], stride),
        _out_dim(x_shape[3], pad, w_shape[3], stride),
    ]
    x_info = helper.make_tensor_value_info("x", _NP_TO_ONNX[x_dtype], x_shape)
    y_info = helper.make_tensor_value_info("y", _NP_TO_ONNX[y_dtype], out_shape)
    node = helper.make_node(
        "QLinearConv",
        [
            "x",
            "x_scale",
            "x_zero_point",
            "w",
            "w_scale",
            "w_zero_point",
            "y_scale",
            "y_zero_point",
            "B",
        ],
        ["y"],
        kernel_shape=[w_shape[2], w_shape[3]],
        strides=[stride, stride],
        pads=[pad, pad, pad, pad],
        dilations=[1, 1],
        group=group,
    )
    w = rng.integers(-63, 64, size=w_shape, dtype=np.int8)
    w_scale = rng.uniform(0.002, 0.02, size=(cout,)).astype(np.float32)
    b = rng.integers(-2000, 2001, size=(cout,), dtype=np.int32)
    inits = [
        numpy_helper.from_array(np.array(x_scale, dtype=np.float32), "x_scale"),
        numpy_helper.from_array(np.array(x_zp, dtype=x_dtype), "x_zero_point"),
        numpy_helper.from_array(w, "w"),
        numpy_helper.from_array(w_scale, "w_scale"),
        numpy_helper.from_array(np.zeros((cout,), dtype=np.int8), "w_zero_point"),
        numpy_helper.from_array(np.array(y_scale, dtype=np.float32), "y_scale"),
        numpy_helper.from_array(np.array(y_zp, dtype=y_dtype), "y_zero_point"),
        numpy_helper.from_array(b, "B"),
    ]
    model = make_model_from_nodes(
        [node], [x_info], [y_info], initializers=inits, opset=12
    )
    return model, _random_q(rng, x_dtype, x_shape)


class TestQLinearConvRequantize:
    """Full-range activations and zero points with an f32 requantize.

    These graphs meet the rocMLIR route's conditions: constant parameters, a
    zero weight zero point, and per-tensor activation parameters. With
    HIPDNN_EP_ROCMLIR=1 they run as one rocMLIR kernel, where a u8 input is
    rebiased to x - 128 and a constant term corrects the zero point. A padded
    border reads fewer taps, so its correction differs from the interior.
    Weights above the ORT bridge clone threshold (127 bytes by default) are
    not inline constants; raise XLNX_ABI_2_0_CLONE_EXTERNAL_DATA_THRESHOLD so
    that every case here takes the rocMLIR route.
    """

    @pytest.mark.parametrize("x_zp", [0, 115, 255])
    def test_u8_zero_point_padded(self, model_runner, x_zp):
        rng = np.random.default_rng(100 + x_zp)
        model, x = _make_requant_conv(
            rng, [1, 3, 7, 7], [4, 3, 3, 3], x_zp=x_zp, y_zp=128
        )
        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=0, rtol=0)

    @pytest.mark.parametrize(
        "pad,stride", [(0, 1), (2, 1), (1, 2), (2, 2)], ids=lambda v: str(v)
    )
    def test_pad_stride(self, model_runner, pad, stride):
        rng = np.random.default_rng(200 + 10 * pad + stride)
        model, x = _make_requant_conv(
            rng,
            [1, 4, 9, 9],
            [6, 4, 3, 3],
            x_zp=77,
            y_zp=100,
            pad=pad,
            stride=stride,
        )
        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=0, rtol=0)

    @pytest.mark.parametrize("group", [2, 8], ids=["grouped", "depthwise"])
    def test_groups(self, model_runner, group):
        rng = np.random.default_rng(300 + group)
        model, x = _make_requant_conv(
            rng, [1, 8, 6, 6], [8, 8 // group, 3, 3], x_zp=200, y_zp=60, group=group
        )
        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=0, rtol=0)

    def test_i8_input_and_output(self, model_runner):
        rng = np.random.default_rng(400)
        model, x = _make_requant_conv(
            rng,
            [1, 3, 7, 7],
            [4, 3, 3, 3],
            x_dtype=np.int8,
            y_dtype=np.int8,
            x_zp=-20,
            y_zp=5,
        )
        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=0, rtol=0)

    @pytest.mark.parametrize("y_dtype", [np.uint8, np.int8], ids=["u8", "i8"])
    def test_saturation(self, model_runner, y_dtype):
        # A small output scale pushes most results past both ends of the range.
        # The CPU reference needs the input and output types to match.
        rng = np.random.default_rng(500)
        model, x = _make_requant_conv(
            rng,
            [1, 3, 7, 7],
            [4, 3, 3, 3],
            x_dtype=y_dtype,
            y_dtype=y_dtype,
            x_zp=128 if y_dtype == np.uint8 else 0,
            y_zp=0,
            y_scale=0.004,
        )
        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=0, rtol=0)
        got = np.asarray(actual[0])
        info = np.iinfo(y_dtype)
        if y_dtype == np.int8:
            assert (got == info.min).any() and (got == info.max).any()
        else:
            assert (got == info.max).any() and (got == 0).any()
