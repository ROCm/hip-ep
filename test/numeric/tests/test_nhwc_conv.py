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


def _make_nhwc_conv(n, h, w, c, m, kh, kw, pads, group=1, bias=True, seed=7):
    assert c % group == 0 and m % group == 0
    ho = (h + pads[0] + pads[2] - kh) // 1 + 1
    wo = (w + pads[1] + pads[3] - kw) // 1 + 1
    inp = helper.make_tensor_value_info("X", TensorProto.FLOAT16, [n, h, w, c])
    out = helper.make_tensor_value_info("Y", TensorProto.FLOAT16, [n, ho, wo, m])
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
        auto_pad="NOTSET",
        group=group,
        kernel_shape=[kh, kw],
        pads=pads,
        strides=[1, 1],
        dilations=[1, 1],
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
