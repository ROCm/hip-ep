#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""Tests for ONNX LRN.

GoogLeNet opset 12 uses size 5, alpha 0.0001, beta 0.75, and bias 1 on
1x64x56x56 and 1x192x56x56. These cases keep those attributes and shrink
only the spatial size.
"""

from __future__ import annotations

import numpy as np
from onnx import helper

from framework.comparator import compare_outputs
from framework.onnx_utils import make_model_from_nodes, np_to_onnx_type


def _make_lrn(shape, size=5, alpha=0.0001, beta=0.75, bias=1.0):
    dtype = np.float32
    tp = np_to_onnx_type(dtype)
    x_info = helper.make_tensor_value_info("X", tp, list(shape))
    y_info = helper.make_tensor_value_info("Y", tp, list(shape))
    node = helper.make_node(
        "LRN",
        ["X"],
        ["Y"],
        size=size,
        alpha=alpha,
        beta=beta,
        bias=bias,
    )
    return make_model_from_nodes([node], [x_info], [y_info], opset=12)


class TestLRN:
    def test_googlenet_64_channels(self, model_runner):
        shape = (1, 64, 8, 8)
        model = _make_lrn(shape)
        x = np.random.default_rng(7).uniform(-1, 1, shape).astype(np.float32)
        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=1e-5, rtol=1e-5)

    def test_googlenet_192_channels(self, model_runner):
        shape = (1, 192, 8, 8)
        model = _make_lrn(shape)
        x = np.random.default_rng(8).uniform(-1, 1, shape).astype(np.float32)
        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=1e-5, rtol=1e-5)
