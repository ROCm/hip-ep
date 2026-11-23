#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""A 1x1 Conv whose weights are a per-channel int4 DequantizeLinear, as int4
QDQ LLMs export their projections.

With HIPDNN_EP_ROCMLIR=1 and autotuning, the rocMLIR conv kernel reads the
packed int4 weights itself: two signed values per byte, low nibble first.
The weights here use every value from -8 to 7 in both nibble positions, so a
swapped nibble order or an unsigned read fails the comparison.
"""

from __future__ import annotations

import numpy as np
import pytest
from onnx import TensorProto, helper, numpy_helper

from framework.comparator import compare_outputs
from framework.onnx_utils import make_model_from_nodes

_RNG = np.random.default_rng(11)


def _int4_weights(cout, cin, with_zp):
    weights = _RNG.integers(-8, 8, size=(cout, cin, 1, 1), dtype=np.int8)
    scales = _RNG.uniform(0.002, 0.02, size=(cout,)).astype(np.float32)
    inits = [
        helper.make_tensor(
            "w", TensorProto.INT4, [cout, cin, 1, 1], weights.ravel().tolist()
        ),
        numpy_helper.from_array(scales, "ws"),
    ]
    inputs = ["w", "ws"]
    if with_zp:
        zero_points = _RNG.integers(-8, 8, size=(cout,), dtype=np.int8)
        inits.append(
            helper.make_tensor("wz", TensorProto.INT4, [cout], zero_points.tolist())
        )
        inputs.append("wz")
    node = helper.make_node("DequantizeLinear", inputs, ["wf"], axis=0)
    return node, inits


def _conv(x, w, y):
    return helper.make_node(
        "Conv",
        [x, w],
        [y],
        kernel_shape=[1, 1],
        strides=[1, 1],
        pads=[0, 0, 0, 0],
        dilations=[1, 1],
        group=1,
    )


def _float_model(cout, cin, positions, with_zp):
    dq, inits = _int4_weights(cout, cin, with_zp)
    return make_model_from_nodes(
        [dq, _conv("x", "wf", "y")],
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, cin, 1, positions])],
        [
            helper.make_tensor_value_info(
                "y", TensorProto.FLOAT, [1, cout, 1, positions]
            )
        ],
        initializers=inits,
        opset=21,
    )


def _uint16_model(cout, cin, positions):
    """uint16 activation in and out, as in a QDQ model with 16-bit activations."""
    dq, inits = _int4_weights(cout, cin, with_zp=True)
    inits += [
        numpy_helper.from_array(np.array(1.0e-4, np.float32), "xs"),
        numpy_helper.from_array(np.array(32768, np.uint16), "xz"),
        numpy_helper.from_array(np.array(2.5e-3, np.float32), "ys"),
        numpy_helper.from_array(np.array(32768, np.uint16), "yz"),
    ]
    nodes = [
        helper.make_node("DequantizeLinear", ["x", "xs", "xz"], ["xf"]),
        dq,
        _conv("xf", "wf", "c"),
        helper.make_node("QuantizeLinear", ["c", "ys", "yz"], ["y"]),
    ]
    return make_model_from_nodes(
        nodes,
        [
            helper.make_tensor_value_info(
                "x", TensorProto.UINT16, [1, cin, 1, positions]
            )
        ],
        [
            helper.make_tensor_value_info(
                "y", TensorProto.UINT16, [1, cout, 1, positions]
            )
        ],
        initializers=inits,
        opset=21,
    )


class TestInt4WeightConv:
    @pytest.mark.parametrize("with_zp", [True, False], ids=["zp", "nozp"])
    @pytest.mark.parametrize(
        "cout,cin,positions",
        [(32, 16, 8), (1024, 2048, 16)],
        ids=["small", "llm"],
    )
    def test_float_activation(self, model_runner, cout, cin, positions, with_zp):
        model = _float_model(cout, cin, positions, with_zp)
        x = _RNG.uniform(-1.0, 1.0, size=(1, cin, 1, positions)).astype(np.float32)
        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=1e-3, rtol=1e-3)

    def test_uint4_weights(self, model_runner):
        """UINT4 weights with a zero point: the kernel zero-extends each nibble."""
        cout, cin, positions = 64, 32, 8
        weights = _RNG.integers(0, 16, size=(cout, cin, 1, 1), dtype=np.uint8)
        scales = _RNG.uniform(0.002, 0.02, size=(cout,)).astype(np.float32)
        zero_points = _RNG.integers(0, 16, size=(cout,), dtype=np.uint8)
        inits = [
            helper.make_tensor(
                "w", TensorProto.UINT4, [cout, cin, 1, 1], weights.ravel().tolist()
            ),
            numpy_helper.from_array(scales, "ws"),
            helper.make_tensor("wz", TensorProto.UINT4, [cout], zero_points.tolist()),
        ]
        model = make_model_from_nodes(
            [
                helper.make_node("DequantizeLinear", ["w", "ws", "wz"], ["wf"], axis=0),
                _conv("x", "wf", "y"),
            ],
            [
                helper.make_tensor_value_info(
                    "x", TensorProto.FLOAT, [1, cin, 1, positions]
                )
            ],
            [
                helper.make_tensor_value_info(
                    "y", TensorProto.FLOAT, [1, cout, 1, positions]
                )
            ],
            initializers=inits,
            opset=21,
        )
        x = _RNG.uniform(-1.0, 1.0, size=(1, cin, 1, positions)).astype(np.float32)
        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=1e-3, rtol=1e-3)

    def test_uint16_activation(self, model_runner):
        model = _uint16_model(512, 2048, 16)
        x = _RNG.integers(30000, 35536, size=(1, 2048, 1, 16), dtype=np.uint16)
        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=1, rtol=0)
