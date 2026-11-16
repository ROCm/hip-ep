#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""Unsigned ops that read the u8 result of a QLinearConv.

rocMLIR takes only signless integers. With HIPDNN_EP_ROCMLIR=1 a u8 Max, Min
or Clip joins the conv kernel through a form widened to i32 that keeps the
unsigned order. An Equal or Less returns BOOL, which is u8 in hip; it runs
after the kernel as a HIP op. The kernel must compile, so run with
HIPDNN_EP_STRICT=1, and every case must keep values of 128 and above in
order.

The conv is an identity: a 1x1 filter of one with the input's scale and zero
point on both sides, so its output equals its input, which holds every u8
value.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
from onnx import TensorProto, helper, numpy_helper

from framework.comparator import compare_outputs
from framework.onnx_utils import make_model_from_nodes

# The HIP elementwise kernels of the default path have no u8 Max or Min.
_ROCMLIR_ONLY = pytest.mark.skipif(
    os.environ.get("HIPDNN_EP_ROCMLIR") != "1",
    reason="the HIP elementwise kernels have no u8 Max or Min",
)
_SHAPE = [1, 1, 16, 16]
_X = np.arange(256, dtype=np.uint8).reshape(_SHAPE)


def _scalar(name, value, dtype):
    return numpy_helper.from_array(np.array(value, dtype=dtype), name)


def _identity_conv(x, out):
    node = helper.make_node(
        "QLinearConv",
        [x, "cs", "cz", "cw", "cws", "cwz", "cs", "cz"],
        [out],
        kernel_shape=[1, 1],
        strides=[1, 1],
        pads=[0, 0, 0, 0],
        dilations=[1, 1],
        group=1,
    )
    inits = [
        _scalar("cs", 0.5, np.float32),
        _scalar("cz", 128, np.uint8),
        numpy_helper.from_array(np.ones((1, 1, 1, 1), dtype=np.int8), "cw"),
        _scalar("cws", 1.0, np.float32),
        _scalar("cwz", 0, np.int8),
    ]
    return [node], inits


def _tensor(name, elem_type):
    return helper.make_tensor_value_info(name, elem_type, _SHAPE)


def _run(model_runner, nodes, inits, out_type):
    conv, conv_inits = _identity_conv("x", "y")
    model = make_model_from_nodes(
        conv + nodes,
        [_tensor("x", TensorProto.UINT8)],
        [_tensor("z", out_type)],
        initializers=conv_inits + inits,
        opset=13,
    )
    actual, expected = model_runner.run_sample(model, [_X])
    compare_outputs(actual, expected, atol=0, rtol=0)


class TestUnsignedAfterQLinearConv:
    @_ROCMLIR_ONLY
    @pytest.mark.parametrize("op", ["Max", "Min"])
    def test_max_min(self, model_runner, op):
        node = helper.make_node(op, ["y", "c"], ["z"])
        _run(model_runner, [node], [_scalar("c", 200, np.uint8)], TensorProto.UINT8)

    @_ROCMLIR_ONLY
    def test_clip(self, model_runner):
        node = helper.make_node("Clip", ["y", "lo", "hi"], ["z"])
        inits = [_scalar("lo", 100, np.uint8), _scalar("hi", 200, np.uint8)]
        _run(model_runner, [node], inits, TensorProto.UINT8)

    @_ROCMLIR_ONLY
    def test_max_then_min(self, model_runner):
        nodes = [
            helper.make_node("Max", ["y", "lo"], ["m"]),
            helper.make_node("Min", ["m", "hi"], ["z"]),
        ]
        inits = [_scalar("lo", 100, np.uint8), _scalar("hi", 200, np.uint8)]
        _run(model_runner, nodes, inits, TensorProto.UINT8)

    @pytest.mark.xfail(strict=True, reason="the HIP Less kernel has no u8 form")
    def test_less(self, model_runner):
        node = helper.make_node("Less", ["y", "c"], ["z"])
        _run(model_runner, [node], [_scalar("c", 200, np.uint8)], TensorProto.BOOL)

    def test_equal(self, model_runner):
        node = helper.make_node("Equal", ["y", "c"], ["z"])
        _run(model_runner, [node], [_scalar("c", 200, np.uint8)], TensorProto.BOOL)
