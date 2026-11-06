#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""Tests for ONNX PRelu, lowered through hip.prelu.

Y = X if X >= 0 else slope * X. slope is unidirectional-broadcastable onto X.
The per-channel 1xCx1x1 slope is the form used by ArcFace ResNet-100.
"""

from __future__ import annotations

import numpy as np
import pytest
from onnx import helper

from framework.comparator import compare_outputs
from framework.onnx_utils import make_model_from_nodes, np_to_onnx_type


def _make_prelu_model(x_shape: list[int], slope: np.ndarray):
    y_info = helper.make_tensor_value_info("Y", np_to_onnx_type(slope.dtype), x_shape)
    x_info = helper.make_tensor_value_info("X", np_to_onnx_type(slope.dtype), x_shape)
    slope_init = helper.make_tensor(
        "slope",
        np_to_onnx_type(slope.dtype),
        list(slope.shape),
        slope.reshape(-1).tolist(),
    )
    node = helper.make_node("PRelu", ["X", "slope"], ["Y"])
    return make_model_from_nodes([node], [x_info], [y_info], initializers=[slope_init])


class TestPRelu:
    @pytest.mark.parametrize(
        "x_shape,slope_shape",
        [
            ([1, 4, 3, 3], [1, 4, 1, 1]),
            ([1, 64, 8, 8], [1, 64, 1, 1]),
            ([2, 3], []),
            ([2, 4], [4]),
        ],
    )
    def test_f32(self, model_runner, x_shape, slope_shape):
        rng = np.random.default_rng(7)
        slope = rng.uniform(-0.5, 0.5, slope_shape).astype(np.float32)
        model = _make_prelu_model(x_shape, slope)
        x = rng.uniform(-2.0, 2.0, x_shape).astype(np.float32)
        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=0)

    def test_f16_per_channel(self, model_runner):
        rng = np.random.default_rng(8)
        x_shape = [1, 4, 3, 3]
        slope = rng.uniform(-0.5, 0.5, [1, 4, 1, 1]).astype(np.float16)
        model = _make_prelu_model(x_shape, slope)
        x = rng.uniform(-2.0, 2.0, x_shape).astype(np.float16)
        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=1e-3, rtol=1e-3)
