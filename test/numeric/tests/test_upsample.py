#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""Tests for ONNX Upsample (schema 9), lowered through hip.resize.

Upsample is Resize with asymmetric coordinates. Nearest sampling uses floor.
output_dim = floor(input_dim * scale). The two Deeplab shapes are the
linear nodes in ul_deeplabv3_fp16.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from onnx import TensorProto, helper

from framework.comparator import compare_outputs
from framework.onnx_utils import make_model_from_nodes, np_to_onnx_type


def _make_upsample_model(
    shape: list[int], scales: list[float], dtype: np.dtype, mode: str
):
    out_shape = [int(math.floor(dim * scale)) for dim, scale in zip(shape, scales)]
    x_info = helper.make_tensor_value_info("X", np_to_onnx_type(dtype), list(shape))
    y_info = helper.make_tensor_value_info("Y", np_to_onnx_type(dtype), list(out_shape))
    scales_init = helper.make_tensor("scales", TensorProto.FLOAT, [len(scales)], scales)
    node = helper.make_node("Upsample", ["X", "scales"], ["Y"], mode=mode)
    # Upsample was removed after opset 9.
    return make_model_from_nodes(
        [node], [x_info], [y_info], initializers=[scales_init], opset=9
    )


class TestUpsample:
    @pytest.mark.parametrize(
        "shape,scales,dtype",
        [
            ([1, 256, 1, 1], [1.0, 1.0, 65.0, 65.0], np.float32),
            ([1, 21, 65, 65], [1.0, 1.0, 7.890625, 7.890625], np.float32),
            ([1, 4, 8, 8], [1.0, 1.0, 2.0, 2.0], np.float16),
        ],
    )
    def test_linear(self, model_runner, shape, scales, dtype):
        model = _make_upsample_model(shape, scales, dtype, "linear")
        rng = np.random.default_rng(201)
        x = rng.uniform(-1.0, 1.0, shape).astype(dtype)
        actual, expected = model_runner.run_sample(model, [x])
        atol = 2e-2 if dtype == np.float16 else 1e-4
        compare_outputs(actual, expected, atol=atol, rtol=atol)

    def test_nearest(self, model_runner):
        shape = [1, 2, 3, 4]
        scales = [1.0, 1.0, 2.0, 3.0]
        model = _make_upsample_model(shape, scales, np.float32, "nearest")
        rng = np.random.default_rng(202)
        x = rng.uniform(-2.0, 2.0, shape).astype(np.float32)
        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=0)
