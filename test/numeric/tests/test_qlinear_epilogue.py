#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""Quantized ops that read a QLinearConv, with the scales of MobileNetV3 int8.

With HIPDNN_EP_ROCMLIR=1 the conv and the ops after it run as one rocMLIR
kernel. There, QLinearAdd and QLinearMul compute ORT CPU's own formulas, so
at these scales the scalar add and the mul match it exactly over every input.
A tensor add can differ by one unit where its sum rounds at a half and the
compiler does not fuse its outer multiply-add, as ORT does. Ops that
dequantize to f32 and quantize again (Div, HardSigmoid) may also differ by
one unit. The HIP kernels of the default path keep their tolerance of one
unit.

The conv is an identity: a 1x1 filter of one with the input's scale and zero
point on both sides, so its output equals its input. A 256x256 input, with
the row index as one operand and the column index as the other, covers every
pair of 8-bit values.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
from onnx import TensorProto, helper, numpy_helper

from framework.comparator import compare_outputs
from framework.onnx_utils import make_model_from_nodes

_MS = [helper.make_opsetid("com.microsoft", 1)]
_EXACT = 0 if os.environ.get("HIPDNN_EP_ROCMLIR") == "1" else 1
_SHAPE = [1, 1, 256, 256]

_ROWS = np.arange(256, dtype=np.uint8).reshape(1, 1, 256, 1).repeat(256, axis=3)
_COLS = np.arange(256, dtype=np.uint8).reshape(1, 1, 1, 256).repeat(256, axis=2)


def _scalar(name, value, dtype):
    return numpy_helper.from_array(np.array(value, dtype=dtype), name)


def _identity_conv(x, scale, zero_point, out):
    nodes = [
        helper.make_node(
            "QLinearConv",
            [x, "cs", "cz", "cw", "cws", "cwz", "cs", "cz"],
            [out],
            kernel_shape=[1, 1],
            strides=[1, 1],
            pads=[0, 0, 0, 0],
            dilations=[1, 1],
            group=1,
        )
    ]
    inits = [
        _scalar("cs", scale, np.float32),
        _scalar("cz", zero_point, np.uint8),
        numpy_helper.from_array(np.ones((1, 1, 1, 1), dtype=np.int8), "cw"),
        _scalar("cws", 1.0, np.float32),
        _scalar("cwz", 0, np.int8),
    ]
    return nodes, inits


def _binary(op, a, b, params, out, prefix):
    sa, za, sb, zb, sc, zc = params
    names = [f"{prefix}{n}" for n in ("sa", "za", "sb", "zb", "sc", "zc")]
    node = helper.make_node(
        op,
        [a, names[0], names[1], b, names[2], names[3], names[4], names[5]],
        [out],
        domain="com.microsoft",
    )
    inits = [
        _scalar(names[0], sa, np.float32),
        _scalar(names[1], za, np.uint8),
        _scalar(names[2], sb, np.float32),
        _scalar(names[3], zb, np.uint8),
        _scalar(names[4], sc, np.float32),
        _scalar(names[5], zc, np.uint8),
    ]
    return node, inits


def _u8(name, shape=None):
    return helper.make_tensor_value_info(name, TensorProto.UINT8, shape or _SHAPE)


# (sA, zA, sB, zB, sC, zC) of MobileNetV3 int8 sites.
_ADD_TENSOR = [
    (2.2196123600006104, 108, 3.7732431888580322, 161, 3.166368007659912, 143),
    (11.868009567260742, 98, 2.3189637660980225, 116, 10.961501121520996, 87),
    (59.150054931640625, 123, 10.961501121520996, 87, 57.80239486694336, 133),
]
_ADD_SCALAR = [
    (44.117183685302734, 134, 0.0117647061124444, 0, 0.0235294122248888, 0),
    (3.899467706680298, 134, 0.0117647061124444, 0, 0.0235294122248888, 0),
    (7.179993629455566, 120, 0.0117647061124444, 0, 0.0235294122248888, 0),
]
_MUL = [
    (44.117183685302734, 134, 0.003921568859368563, 0, 21.004295349121094, 0),
    (0.0028467855881899595, 0, 1.9674113988876343, 0, 0.8652156591415405, 0),
    (3.899467706680298, 134, 0.003921568859368563, 0, 1.853278636932373, 0),
]


class TestQLinearBinaryAfterConv:
    @pytest.mark.parametrize("params", _ADD_TENSOR, ids=["s0", "s1", "s2"])
    def test_add_tensor(self, model_runner, params):
        conv, inits = _identity_conv("x", params[0], params[1], "y")
        add, add_inits = _binary("QLinearAdd", "y", "r", params, "z", "a_")
        model = make_model_from_nodes(
            conv + [add],
            [_u8("x"), _u8("r")],
            [_u8("z")],
            initializers=inits + add_inits,
            opset=12,
            extra_opsets=_MS,
        )
        actual, expected = model_runner.run_sample(model, [_ROWS, _COLS])
        compare_outputs(actual, expected, atol=1, rtol=0)

    @pytest.mark.parametrize("params", _ADD_SCALAR, ids=["s0", "s1", "s2"])
    def test_add_constant_scalar(self, model_runner, params):
        conv, inits = _identity_conv("x", params[0], params[1], "y")
        add, add_inits = _binary("QLinearAdd", "y", "three", params, "z", "a_")
        inits = inits + add_inits + [_scalar("three", 255, np.uint8)]
        model = make_model_from_nodes(
            conv + [add],
            [_u8("x")],
            [_u8("z")],
            initializers=inits,
            opset=12,
            extra_opsets=_MS,
        )
        actual, expected = model_runner.run_sample(model, [_ROWS])
        compare_outputs(actual, expected, atol=_EXACT, rtol=0)

    @pytest.mark.parametrize("params", _MUL, ids=["s0", "s1", "s2"])
    def test_mul_tensor(self, model_runner, params):
        conv, inits = _identity_conv("x", params[0], params[1], "y")
        mul, mul_inits = _binary("QLinearMul", "y", "r", params, "z", "m_")
        model = make_model_from_nodes(
            conv + [mul],
            [_u8("x"), _u8("r")],
            [_u8("z")],
            initializers=inits + mul_inits,
            opset=12,
            extra_opsets=_MS,
        )
        actual, expected = model_runner.run_sample(model, [_ROWS, _COLS])
        compare_outputs(actual, expected, atol=_EXACT, rtol=0)


def _requantize(op, x, in_scale, in_zp, out_scale, out_zp, prefix, **attrs):
    """DequantizeLinear -> op -> QuantizeLinear, with a constant 6 for Div."""
    nodes = [
        helper.make_node(
            "DequantizeLinear", [x, f"{prefix}si", f"{prefix}zi"], [f"{prefix}f"]
        )
    ]
    inits = [
        _scalar(f"{prefix}si", in_scale, np.float32),
        _scalar(f"{prefix}zi", in_zp, np.uint8),
        _scalar(f"{prefix}so", out_scale, np.float32),
        _scalar(f"{prefix}zo", out_zp, np.uint8),
    ]
    inputs = [f"{prefix}f"]
    if op == "Div":
        inputs.append(f"{prefix}six")
        inits.append(_scalar(f"{prefix}six", 6.0, np.float32))
    nodes.append(helper.make_node(op, inputs, [f"{prefix}g"], **attrs))
    nodes.append(
        helper.make_node(
            "QuantizeLinear",
            [f"{prefix}g", f"{prefix}so", f"{prefix}zo"],
            [f"{prefix}q"],
        )
    )
    return nodes, inits


class TestRequantizedEpilogue:
    def test_div(self, model_runner):
        conv, inits = _identity_conv("x", 0.0235294122248888, 0, "y")
        div, div_inits = _requantize(
            "Div", "y", 0.0235294122248888, 0, 0.003921568859368563, 0, "d_"
        )
        model = make_model_from_nodes(
            conv + div,
            [_u8("x")],
            [_u8("d_q")],
            initializers=inits + div_inits,
            opset=12,
        )
        actual, expected = model_runner.run_sample(model, [_ROWS])
        compare_outputs(actual, expected, atol=1, rtol=0)

    def test_hard_sigmoid(self, model_runner):
        conv, inits = _identity_conv("x", 0.009775694459676743, 116, "y")
        hs, hs_inits = _requantize(
            "HardSigmoid",
            "y",
            0.009775694459676743,
            116,
            0.0028467855881899595,
            0,
            "h_",
            alpha=0.1666666716337204,
        )
        model = make_model_from_nodes(
            conv + hs,
            [_u8("x")],
            [_u8("h_q")],
            initializers=inits + hs_inits,
            opset=12,
        )
        actual, expected = model_runner.run_sample(model, [_ROWS])
        compare_outputs(actual, expected, atol=1, rtol=0)

    def test_hard_swish_block(self, model_runner):
        """y * q(q(y + 3) / 6), the first block of MobileNetV3 int8."""
        sy, zy = 44.117183685302734, 134
        conv, inits = _identity_conv("x", sy, zy, "y")
        add, add_inits = _binary(
            "QLinearAdd",
            "y",
            "three",
            (sy, zy, 0.0117647061124444, 0, 0.0235294122248888, 0),
            "t",
            "a_",
        )
        div, div_inits = _requantize(
            "Div", "t", 0.0235294122248888, 0, 0.003921568859368563, 0, "d_"
        )
        mul, mul_inits = _binary(
            "QLinearMul",
            "y",
            "d_q",
            (sy, zy, 0.003921568859368563, 0, 21.004295349121094, 0),
            "z",
            "m_",
        )
        model = make_model_from_nodes(
            conv + [add] + div + [mul],
            [_u8("x")],
            [_u8("z")],
            initializers=inits
            + add_inits
            + div_inits
            + mul_inits
            + [_scalar("three", 255, np.uint8)],
            opset=12,
            extra_opsets=_MS,
        )
        actual, expected = model_runner.run_sample(model, [_ROWS])
        compare_outputs(actual, expected, atol=1, rtol=0)
