#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""DynamicQuantizeLinear and ConvInteger.

A reduction of at most 258 accumulates in f32, which is exact for an 8-bit
dot product. A longer reduction accumulates in i32.
"""

from __future__ import annotations

import numpy as np
from onnx import TensorProto, helper, numpy_helper

from framework.comparator import compare_outputs
from framework.onnx_utils import make_model_from_nodes


def _conv_integer(x, w, x_zp, w_zp, *, stride=1, pad=0):
    x_shape = list(x.shape)
    w_shape = list(w.shape)
    out_h = (x_shape[2] + pad + pad - w_shape[2]) // stride + 1
    out_w = (x_shape[3] + pad + pad - w_shape[3]) // stride + 1
    out_shape = [x_shape[0], w_shape[0], out_h, out_w]
    x_dtype = TensorProto.INT8 if x.dtype == np.int8 else TensorProto.UINT8
    x_info = helper.make_tensor_value_info("x", x_dtype, x_shape)
    y_info = helper.make_tensor_value_info("y", TensorProto.INT32, out_shape)
    node = helper.make_node(
        "ConvInteger",
        ["x", "w", "x_zp", "w_zp"],
        ["y"],
        kernel_shape=[w_shape[2], w_shape[3]],
        strides=[stride, stride],
        pads=[pad, pad, pad, pad],
        dilations=[1, 1],
        group=1,
        auto_pad="NOTSET",
    )
    inits = [
        numpy_helper.from_array(w, "w"),
        numpy_helper.from_array(np.asarray(x_zp, dtype=x.dtype), "x_zp"),
        numpy_helper.from_array(np.asarray(w_zp, dtype=w.dtype), "w_zp"),
    ]
    return make_model_from_nodes(
        [node], [x_info], [y_info], initializers=inits, opset=13
    )


class TestDynamicQuantizeLinear:
    def test_range_through_zero(self, model_runner):
        x = np.array([[[[-1.0, 0.0], [1.0, 2.0]]]], dtype=np.float32)
        x_info = helper.make_tensor_value_info("x", TensorProto.FLOAT, list(x.shape))
        outputs = [
            helper.make_tensor_value_info("y", TensorProto.UINT8, list(x.shape)),
            helper.make_tensor_value_info("y_scale", TensorProto.FLOAT, []),
            helper.make_tensor_value_info("y_zero_point", TensorProto.UINT8, []),
        ]
        node = helper.make_node(
            "DynamicQuantizeLinear", ["x"], ["y", "y_scale", "y_zero_point"]
        )
        model = make_model_from_nodes([node], [x_info], outputs, opset=13)
        actual, expected = model_runner.run_sample(model, [x])
        actual[1:] = [
            np.asarray(a).reshape(np.asarray(e).shape)
            for a, e in zip(actual[1:], expected[1:])
        ]
        compare_outputs(actual, expected, atol=1e-6, rtol=0)

    def test_all_zero_input(self, model_runner):
        x = np.zeros((1, 1, 2, 2), dtype=np.float32)
        x_info = helper.make_tensor_value_info("x", TensorProto.FLOAT, list(x.shape))
        outputs = [
            helper.make_tensor_value_info("y", TensorProto.UINT8, list(x.shape)),
            helper.make_tensor_value_info("y_scale", TensorProto.FLOAT, []),
            helper.make_tensor_value_info("y_zero_point", TensorProto.UINT8, []),
        ]
        node = helper.make_node(
            "DynamicQuantizeLinear", ["x"], ["y", "y_scale", "y_zero_point"]
        )
        model = make_model_from_nodes([node], [x_info], outputs, opset=13)
        actual, expected = model_runner.run_sample(model, [x])
        # A scalar graph output comes back as a length-1 vector from the CPU
        # reference and as a rank-0 tensor from the EP. The value is the same.
        actual = [
            np.asarray(a).reshape(np.asarray(e).shape) for a, e in zip(actual, expected)
        ]
        compare_outputs(actual, expected, atol=0, rtol=0)
        assert float(np.asarray(actual[1]).reshape(-1)[0]) == 1.0
        assert int(np.asarray(actual[2]).reshape(-1)[0]) == 0
        assert np.all(np.asarray(actual[0]) == 0)


class TestConvInteger:
    def test_scalar_zero_points(self, model_runner):
        x = np.full((1, 1, 2, 2), 10, dtype=np.uint8)
        w = np.full((1, 1, 1, 1), 3, dtype=np.uint8)
        model = _conv_integer(x, w, 1, 2)
        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=0, rtol=0)

    def test_per_channel_weight_zero_point(self, model_runner):
        x = np.full((1, 1, 2, 2), 10, dtype=np.uint8)
        w = np.full((2, 1, 1, 1), 5, dtype=np.uint8)
        model = _conv_integer(x, w, 0, np.array([1, 2], dtype=np.uint8))
        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=0, rtol=0)

    def test_stride2(self, model_runner):
        x = np.full((1, 1, 4, 4), 2, dtype=np.uint8)
        w = np.full((1, 1, 3, 3), 1, dtype=np.uint8)
        model = _conv_integer(x, w, 0, 0, stride=2)
        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=0, rtol=0)

    def test_int8(self, model_runner):
        x = np.full((1, 1, 2, 2), 2, dtype=np.int8)
        w = np.full((1, 1, 1, 1), 3, dtype=np.int8)
        model = _conv_integer(x, w, np.int8(0), np.int8(0))
        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=0, rtol=0)

    def test_wide_channels(self, model_runner):
        x = np.full((1, 259, 1, 1), 255, dtype=np.uint8)
        w = np.full((1, 259, 1, 1), 255, dtype=np.uint8)
        model = _conv_integer(x, w, np.uint8(0), np.uint8(0))
        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=0, rtol=0)
        assert int(np.asarray(actual[0]).reshape(-1)[0]) == 16841475
