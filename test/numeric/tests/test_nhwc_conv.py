#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""Numeric checks for com.microsoft.NhwcConv.

Activations are NHWC and weights are [M, kH, kW, C/group]. The lowering
transposes both into hip.conv and transposes the result back.
"""

import numpy as np
import pytest
from onnx import TensorProto, helper, numpy_helper

from framework.comparator import compare_outputs
from framework.onnx_utils import make_model_from_nodes

_MS = [helper.make_opsetid("com.microsoft", 1)]


def _make_nhwc_conv(
    n,
    h,
    w,
    c,
    m,
    kh,
    kw,
    pads,
    group=1,
    bias=True,
    seed=7,
    auto_pad="NOTSET",
    dynamic=False,
):
    assert c % group == 0 and m % group == 0
    if auto_pad == "VALID":
        ho = h - kh + 1
        wo = w - kw + 1
    else:
        ho = (h + pads[0] + pads[2] - kh) // 1 + 1
        wo = (w + pads[1] + pads[3] - kw) // 1 + 1
    in_shape = [None, None, None, c] if dynamic else [n, h, w, c]
    out_shape = [None, None, None, m] if dynamic else [n, ho, wo, m]
    inp = helper.make_tensor_value_info("X", TensorProto.FLOAT16, in_shape)
    out = helper.make_tensor_value_info("Y", TensorProto.FLOAT16, out_shape)
    rng = np.random.default_rng(seed)
    weight = rng.uniform(-0.1, 0.1, [m, kh, kw, c // group]).astype(np.float16)
    inputs = ["X", "W"]
    inits = [numpy_helper.from_array(weight, name="W")]
    if bias:
        inputs.append("B")
        inits.append(
            numpy_helper.from_array(
                rng.uniform(-0.1, 0.1, [m]).astype(np.float16), name="B"
            )
        )
    node = helper.make_node(
        "NhwcConv",
        inputs,
        ["Y"],
        domain="com.microsoft",
        auto_pad=auto_pad,
        group=group,
        kernel_shape=[kh, kw],
        strides=[1, 1],
        dilations=[1, 1],
        **({} if auto_pad == "VALID" else {"pads": pads}),
    )
    return make_model_from_nodes(
        [node], [inp], [out], initializers=inits, extra_opsets=_MS
    )


class TestNhwcConv:
    @pytest.mark.parametrize(
        "c,m,kh,pads,group,bias",
        [
            (4, 4, 1, [0, 0, 0, 0], 1, True),
            (4, 8, 3, [1, 1, 1, 1], 1, True),
            (4, 4, 3, [1, 1, 1, 1], 2, True),
            (4, 4, 1, [0, 0, 0, 0], 1, False),
        ],
    )
    def test_nhwc_conv(self, model_runner, c, m, kh, pads, group, bias):
        model = _make_nhwc_conv(1, 8, 8, c, m, kh, kh, pads, group, bias)
        rng = np.random.default_rng(11)
        x = rng.uniform(-1.0, 1.0, [1, 8, 8, c]).astype(np.float16)
        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=1e-2, rtol=1e-2, cos_threshold=0.999)

    def test_nhwc_conv_valid(self, model_runner):
        """auto_pad=VALID drops the spatial border: 8x8, 3x3 -> 6x6."""
        model = _make_nhwc_conv(1, 8, 8, 4, 4, 3, 3, [0, 0, 0, 0], auto_pad="VALID")
        rng = np.random.default_rng(11)
        x = rng.uniform(-1.0, 1.0, [1, 8, 8, 4]).astype(np.float16)
        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=1e-2, rtol=1e-2, cos_threshold=0.999)
        assert actual[0].shape == (1, 6, 6, 4)

    def test_nhwc_conv_dynamic(self, model_runner):
        """Dynamic N/H/W with pad 1 keeps the concrete spatial size."""
        model = _make_nhwc_conv(2, 5, 6, 4, 8, 3, 3, [1, 1, 1, 1], dynamic=True)
        rng = np.random.default_rng(11)
        x = rng.uniform(-1.0, 1.0, [2, 5, 6, 4]).astype(np.float16)
        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=1e-2, rtol=1e-2, cos_threshold=0.999)
        assert actual[0].shape == (2, 5, 6, 8)
