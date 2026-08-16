#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""GPU-vs-CPU correctness for 2D NCHW ConvTranspose.

The dynamic-shape case is the regression guard for destination construction:
output channels come from ``weights[1] * group`` and spatial dimensions use
the ONNX transposed-convolution formula rather than positional input extents.
"""

import numpy as np
import pytest
from onnx import TensorProto, helper, numpy_helper

from framework.comparator import compare_outputs
from framework.onnx_utils import make_model_from_nodes


def _make_convtranspose_model(
    in_c: int,
    out_c: int,
    group: int,
    stride: int,
    pad: int,
    output_padding: int,
    *,
    dynamic: bool,
    seed: int = 17,
):
    assert in_c % group == 0 and out_c % group == 0
    k = 3
    input_shape = [None, in_c, None, None] if dynamic else [1, in_c, 4, 5]
    output_shape = (
        [None, out_c, None, None]
        if dynamic
        else [
            1,
            out_c,
            stride * (4 - 1) + output_padding + k - 2 * pad,
            stride * (5 - 1) + output_padding + k - 2 * pad,
        ]
    )

    inp = helper.make_tensor_value_info("input", TensorProto.FLOAT16, input_shape)
    out = helper.make_tensor_value_info("output", TensorProto.FLOAT16, output_shape)

    rng = np.random.default_rng(seed)
    # ONNX ConvTranspose layout: [C_in, C_out / group, kH, kW].
    weights = rng.uniform(-0.1, 0.1, [in_c, out_c // group, k, k]).astype(np.float16)
    bias = rng.uniform(-0.1, 0.1, [out_c]).astype(np.float16)

    node = helper.make_node(
        "ConvTranspose",
        ["input", "weights", "bias"],
        ["output"],
        group=group,
        kernel_shape=[k, k],
        strides=[stride, stride],
        pads=[pad, pad, pad, pad],
        dilations=[1, 1],
        output_padding=[output_padding, output_padding],
    )
    return make_model_from_nodes(
        [node],
        [inp],
        [out],
        initializers=[
            numpy_helper.from_array(weights, name="weights"),
            numpy_helper.from_array(bias, name="bias"),
        ],
    )


class TestConvTranspose:
    @pytest.mark.parametrize(
        "in_c,out_c,group,stride,pad,output_padding,dynamic",
        [
            (2, 4, 1, 1, 0, 0, False),
            (2, 4, 1, 2, 1, 1, True),
            (4, 6, 2, 2, 1, 0, True),
        ],
    )
    def test_convtranspose(
        self,
        model_runner,
        in_c,
        out_c,
        group,
        stride,
        pad,
        output_padding,
        dynamic,
    ):
        model = _make_convtranspose_model(
            in_c,
            out_c,
            group,
            stride,
            pad,
            output_padding,
            dynamic=dynamic,
        )
        rng = np.random.default_rng(123)
        x = rng.uniform(-1.0, 1.0, [1, in_c, 4, 5]).astype(np.float16)
        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=1e-2, rtol=1e-2, cos_threshold=0.999)
