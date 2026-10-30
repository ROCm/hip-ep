#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""f16 graphs that widen to f32 only around a nearest Resize.

The hip.cast canonicalizer folds Cast(f16->f32) -> Resize(nearest) ->
Cast(f32->f16) to an f16 Resize. A nearest resize only copies elements, so
the result is exact.
"""

from __future__ import annotations

import numpy as np
from onnx import TensorProto, helper, numpy_helper

from framework.comparator import compare_outputs
from framework.onnx_utils import make_model_from_nodes

_RNG = np.random.default_rng(31)
_C, _H, _W = 8, 12, 14


def _f16_info(name, shape):
    return helper.make_tensor_value_info(name, TensorProto.FLOAT16, shape)


def _resize_round_trip(src, dst):
    return [
        helper.make_node("Cast", [src], [dst + "_w"], to=TensorProto.FLOAT),
        helper.make_node(
            "Resize",
            [dst + "_w", "", "scales"],
            [dst + "_r"],
            mode="nearest",
            coordinate_transformation_mode="asymmetric",
            nearest_mode="floor",
        ),
        helper.make_node("Cast", [dst + "_r"], [dst], to=TensorProto.FLOAT16),
    ]


_SCALES = numpy_helper.from_array(
    np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32), "scales"
)


def _resize_add():
    """Cast -> Resize(nearest, 2x) -> Cast -> Add, all f16 outside the casts."""
    nodes = _resize_round_trip("x", "u") + [
        helper.make_node("Add", ["u", "bias"], ["y"]),
    ]
    bias = numpy_helper.from_array(
        _RNG.uniform(-1.0, 1.0, size=(1, _C, 1, 1)).astype(np.float16), "bias"
    )
    return make_model_from_nodes(
        nodes,
        [_f16_info("x", [1, _C, _H, _W])],
        [_f16_info("y", [1, _C, 2 * _H, 2 * _W])],
        initializers=[_SCALES, bias],
        opset=13,
    )


def _conv_resize_conv():
    """f16 Conv -> the round trip -> f16 Conv. On the rocMLIR path the fold
    runs before the convs' kernels could absorb either cast."""
    nodes = [
        helper.make_node(
            "Conv", ["x", "w1"], ["c1"], kernel_shape=[3, 3], pads=[1, 1, 1, 1]
        ),
        *_resize_round_trip("c1", "u"),
        helper.make_node(
            "Conv", ["u", "w2"], ["y"], kernel_shape=[3, 3], pads=[1, 1, 1, 1]
        ),
    ]
    weights = [
        numpy_helper.from_array(
            _RNG.uniform(-0.2, 0.2, size=(_C, _C, 3, 3)).astype(np.float16), name
        )
        for name in ("w1", "w2")
    ]
    return make_model_from_nodes(
        nodes,
        [_f16_info("x", [1, _C, _H, _W])],
        [_f16_info("y", [1, _C, 2 * _H, 2 * _W])],
        initializers=[_SCALES, *weights],
        opset=13,
    )


def _input():
    return _RNG.uniform(-2.0, 2.0, size=(1, _C, _H, _W)).astype(np.float16)


class TestCastRoundTrip:
    def test_resize_add(self, model_runner):
        actual, expected = model_runner.run_sample(_resize_add(), [_input()])
        compare_outputs(actual, expected, atol=0, rtol=0)

    def test_conv_resize_conv(self, model_runner):
        actual, expected = model_runner.run_sample(_conv_resize_conv(), [_input()])
        # The resize is exact; the f16 convs may round differently from the
        # CPU reference.
        compare_outputs(actual, expected, atol=2e-2, rtol=2e-2)
