#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""u8 quantized chains between integer convs.

With HIPDNN_EP_ROCMLIR=1 a u8 tree whose readers include an integer conv
becomes i8 with every zero point lowered by 128, so the conv reads signed
bytes. The rewrite is exact; the outputs may differ from ORT CPU by one unit
where the conv's f32 requantize rounds at a half.
"""

from __future__ import annotations

import numpy as np
import pytest
from onnx import TensorProto, helper, numpy_helper

from framework.comparator import compare_outputs
from framework.onnx_utils import make_model_from_nodes

_RNG = np.random.default_rng(23)
_C, _H, _W = 8, 12, 12


def _scalar(name, value, dtype):
    return numpy_helper.from_array(np.array(value, dtype=dtype), name)


def _weights(name, k, c):
    return numpy_helper.from_array(
        _RNG.integers(-127, 128, size=(k, c, 3, 3), dtype=np.int8), name
    )


def _f32_info(name, shape):
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, shape)


def _u8_info(name, shape):
    return helper.make_tensor_value_info(name, TensorProto.UINT8, shape)


def _conv(prefix, src, dst, k):
    """DQ(src) and DQ(w) -> Conv (pad 1) -> LeakyRelu -> Q, one scale and zero
    point for all the activations."""
    return [
        helper.make_node("DequantizeLinear", [src, "s", "z"], [prefix + "xf"]),
        helper.make_node(
            "DequantizeLinear", [prefix + "w", "ws", "wz"], [prefix + "wf"]
        ),
        helper.make_node(
            "Conv",
            [prefix + "xf", prefix + "wf"],
            [prefix + "c"],
            kernel_shape=[3, 3],
            pads=[1, 1, 1, 1],
        ),
        helper.make_node("LeakyRelu", [prefix + "c"], [prefix + "l"], alpha=0.2),
        helper.make_node("QuantizeLinear", [prefix + "l", "s", "z"], [dst]),
    ], [_weights(prefix + "w", k, _C if prefix != "c3" else 2 * _C)]


_QDQ_INITS = [
    _scalar("s", 0.05, np.float32),
    _scalar("z", 120, np.uint8),
    _scalar("ws", 0.002, np.float32),
    _scalar("wz", 0, np.int8),
]


def _qdq_chain():
    """Q(x) -> conv1 -> {conv2, Transpose -> Transpose} -> Concat -> conv3.

    Every u8 value between the input quantize and conv3 reaches a conv, so the
    whole tree becomes i8. The output quantize of conv3 is a graph output and
    stays u8.
    """
    nodes = [helper.make_node("QuantizeLinear", ["x", "s", "z"], ["q0"])]
    inits = list(_QDQ_INITS)
    for prefix, src, dst, k in (
        ("c1", "q0", "y1", _C),
        ("c2", "y1", "y2", _C),
    ):
        conv_nodes, conv_inits = _conv(prefix, src, dst, k)
        nodes += conv_nodes
        inits += conv_inits
    nodes += [
        helper.make_node("DequantizeLinear", ["y1", "s", "z"], ["y1f"]),
        helper.make_node("Transpose", ["y1f"], ["t0"], perm=[0, 2, 3, 1]),
        helper.make_node("Transpose", ["t0"], ["t1"], perm=[0, 3, 1, 2]),
        helper.make_node("DequantizeLinear", ["y2", "s", "z"], ["y2f"]),
        helper.make_node("Concat", ["t1", "y2f"], ["cat"], axis=1),
        helper.make_node("QuantizeLinear", ["cat", "s", "z"], ["qcat"]),
    ]
    conv_nodes, conv_inits = _conv("c3", "qcat", "y", _C)
    nodes += conv_nodes
    inits += conv_inits
    return make_model_from_nodes(
        nodes,
        [_f32_info("x", [1, _C, _H, _W])],
        [_u8_info("y", [1, _C, _H, _W])],
        initializers=inits,
        opset=13,
    )


def _qlinear_chain(x_zp, mid_zp):
    """Q(x) -> QLinearConv -> QLinearConv, both with i8 weights."""
    nodes = [
        helper.make_node("QuantizeLinear", ["x", "xs", "xz"], ["q"]),
        helper.make_node(
            "QLinearConv",
            ["q", "xs", "xz", "w1", "ws", "wz", "ms", "mz", "b1"],
            ["m"],
            kernel_shape=[3, 3],
            pads=[1, 1, 1, 1],
        ),
        helper.make_node(
            "QLinearConv",
            ["m", "ms", "mz", "w2", "ws", "wz", "ys", "yz"],
            ["y"],
            kernel_shape=[3, 3],
            pads=[1, 1, 1, 1],
        ),
    ]
    inits = [
        _scalar("xs", 0.05, np.float32),
        _scalar("xz", x_zp, np.uint8),
        _weights("w1", _C, _C),
        _weights("w2", _C, _C),
        _scalar("ws", 0.002, np.float32),
        _scalar("wz", 0, np.int8),
        numpy_helper.from_array(
            _RNG.integers(-500, 500, size=(_C,), dtype=np.int32), "b1"
        ),
        _scalar("ms", 0.04, np.float32),
        _scalar("mz", mid_zp, np.uint8),
        _scalar("ys", 0.03, np.float32),
        _scalar("yz", 128, np.uint8),
    ]
    return make_model_from_nodes(
        nodes,
        [_f32_info("x", [1, _C, _H, _W])],
        [_u8_info("y", [1, _C, _H, _W])],
        initializers=inits,
        opset=13,
    )


def _input():
    return _RNG.uniform(-6.0, 6.0, size=(1, _C, _H, _W)).astype(np.float32)


class TestRebiasUnsigned:
    def test_qdq_chain(self, model_runner):
        actual, expected = model_runner.run_sample(_qdq_chain(), [_input()])
        compare_outputs(actual, expected, atol=1, rtol=0)

    @pytest.mark.parametrize(
        "x_zp, mid_zp", [(128, 128), (120, 3), (0, 255)], ids=["mid", "shift", "edge"]
    )
    def test_qlinear_chain(self, model_runner, x_zp, mid_zp):
        actual, expected = model_runner.run_sample(
            _qlinear_chain(x_zp, mid_zp), [_input()]
        )
        compare_outputs(actual, expected, atol=1, rtol=0)
