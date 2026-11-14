#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""DequantizeLinear -> Conv -> LeakyRelu -> QuantizeLinear, as QDQ models
export a quantized conv.

With HIPDNN_EP_ROCMLIR=1 the conv runs as an integer conv on the quantized
operands, scaled by x_scale * w_scale in f32, and the bias, activation, and
quantize run in the same kernel. ORT CPU sums the dequantized f32 values, so
the two results may differ by one unit where the value rounds at a half.
"""

from __future__ import annotations

import numpy as np
import pytest
from onnx import TensorProto, helper, numpy_helper

from framework.comparator import compare_outputs
from framework.onnx_utils import make_model_from_nodes

_N, _C, _K, _H, _W = 1, 16, 32, 20, 20
_RNG = np.random.default_rng(7)


def _scalar(name, value, dtype):
    return numpy_helper.from_array(np.array(value, dtype=dtype), name)


def _qdq_conv(x_dtype, x_zp, w_scale, pads):
    x_scale, b_scale, y_scale, y_zp = 0.0123, 0.00071, 0.0157, 128
    w = _RNG.integers(-127, 128, size=(_K, _C, 3, 3), dtype=np.int8)
    b = _RNG.integers(-127, 128, size=(_K,), dtype=np.int8)
    w_axis = {"axis": 0} if np.ndim(w_scale) else {}
    nodes = [
        helper.make_node("DequantizeLinear", ["x", "xs", "xz"], ["xf"]),
        helper.make_node("DequantizeLinear", ["w", "ws", "wz"], ["wf"], **w_axis),
        helper.make_node("DequantizeLinear", ["b", "bs", "bz"], ["bf"]),
        helper.make_node(
            "Conv",
            ["xf", "wf", "bf"],
            ["c"],
            kernel_shape=[3, 3],
            strides=[1, 1],
            pads=pads,
            dilations=[1, 1],
            group=1,
        ),
        helper.make_node("LeakyRelu", ["c"], ["l"], alpha=0.19921875),
        helper.make_node("QuantizeLinear", ["l", "ys", "yz"], ["y"]),
    ]
    inits = [
        _scalar("xs", x_scale, np.float32),
        _scalar("xz", x_zp, x_dtype),
        numpy_helper.from_array(w, "w"),
        numpy_helper.from_array(np.asarray(w_scale, dtype=np.float32), "ws"),
        numpy_helper.from_array(np.zeros(np.shape(w_scale), dtype=np.int8), "wz"),
        numpy_helper.from_array(b, "b"),
        _scalar("bs", b_scale, np.float32),
        _scalar("bz", 0, np.int8),
        _scalar("ys", y_scale, np.float32),
        _scalar("yz", y_zp, np.uint8),
    ]
    out_hw = _H - 2 + pads[0] + pads[2]
    elem = TensorProto.UINT8 if x_dtype == np.uint8 else TensorProto.INT8
    return make_model_from_nodes(
        nodes,
        [helper.make_tensor_value_info("x", elem, [_N, _C, _H, _W])],
        [
            helper.make_tensor_value_info(
                "y", TensorProto.UINT8, [_N, _K, out_hw, out_hw]
            )
        ],
        initializers=inits,
        opset=13,
    )


def _input(dtype):
    info = np.iinfo(dtype)
    return _RNG.integers(info.min, info.max + 1, size=(_N, _C, _H, _W), dtype=dtype)


class TestQdqConv:
    @pytest.mark.parametrize("pads", [[1, 1, 1, 1], [0, 0, 0, 0]], ids=["pad", "valid"])
    def test_u8_per_tensor(self, model_runner, pads):
        model = _qdq_conv(np.uint8, 128, 0.0041, pads)
        actual, expected = model_runner.run_sample(model, [_input(np.uint8)])
        compare_outputs(actual, expected, atol=1, rtol=0)

    def test_u8_per_channel(self, model_runner):
        scales = np.linspace(0.002, 0.006, _K, dtype=np.float32)
        model = _qdq_conv(np.uint8, 128, scales, [1, 1, 1, 1])
        actual, expected = model_runner.run_sample(model, [_input(np.uint8)])
        compare_outputs(actual, expected, atol=1, rtol=0)

    @pytest.mark.parametrize("x_zp", [-5, 0], ids=["zp", "nozp"])
    def test_i8_zero_point(self, model_runner, x_zp):
        """A nonzero input zero point with padding needs a per-element correction."""
        model = _qdq_conv(np.int8, x_zp, 0.0041, [1, 1, 1, 1])
        actual, expected = model_runner.run_sample(model, [_input(np.int8)])
        compare_outputs(actual, expected, atol=1, rtol=0)

    def test_u8_zero_point_shift(self, model_runner):
        """A u8 zero point other than 128 needs a correction after the rebias."""
        model = _qdq_conv(np.uint8, 120, 0.0041, [1, 1, 1, 1])
        actual, expected = model_runner.run_sample(model, [_input(np.uint8)])
        compare_outputs(actual, expected, atol=1, rtol=0)
