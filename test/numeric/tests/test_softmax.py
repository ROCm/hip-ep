#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""Tests for last-axis ONNX Softmax, lowered through hip.miopen.softmax.

The row kernel is chosen by width, so the widths below cover every launch
path and its boundaries: exact one-wavefront vector rows (128, 256, 512,
1024), masked wavefront rows (1 to 512), a 256-thread block per row with
16-byte loads (widths divisible by the vector width) or scalar loads (odd
widths) up to 8192, a 1024-thread block up to 32768, and the three-pass
kernel past that.
"""

from __future__ import annotations

import numpy as np
import pytest
from onnx import helper

from framework.comparator import compare_outputs
from framework.onnx_utils import make_model_from_nodes, np_to_onnx_type

_WIDTHS = [
    1,
    20,
    32,
    33,
    77,
    128,
    197,
    256,
    257,
    512,
    513,
    577,
    1000,
    1024,
    1025,
    4096,
    8192,
    8193,
    14400,
    16384,
    16385,
    32768,
    40000,
]


def _make_softmax_model(shape: list[int], dtype: type):
    tp = np_to_onnx_type(dtype)
    x_info = helper.make_tensor_value_info("X", tp, shape)
    y_info = helper.make_tensor_value_info("Y", tp, shape)
    node = helper.make_node("Softmax", ["X"], ["Y"], axis=-1)
    return make_model_from_nodes([node], [x_info], [y_info])


def _run(model_runner, shape: list[int], dtype: type, seed: int):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-8.0, 8.0, shape).astype(dtype)
    actual, expected = model_runner.run_sample(
        _make_softmax_model(shape, dtype), [x], reference="cpu"
    )
    return actual, expected


class TestSoftmax:
    @pytest.mark.parametrize("cols", _WIDTHS)
    def test_f32(self, model_runner, cols):
        actual, expected = _run(model_runner, [2, 3, cols], np.float32, cols)
        compare_outputs(actual, expected, atol=1e-6, rtol=1e-4)

    @pytest.mark.parametrize("cols", _WIDTHS)
    def test_f16(self, model_runner, cols):
        actual, expected = _run(model_runner, [2, 3, cols], np.float16, cols)
        compare_outputs(actual, expected, atol=1e-4, rtol=1e-2)

    @pytest.mark.parametrize(
        "shape,dtype",
        [
            ([4096, 49], np.float32),
            ([2048, 197], np.float16),
            ([256, 4096], np.float32),
            ([64, 14400], np.float16),
        ],
    )
    def test_many_rows(self, model_runner, shape, dtype):
        actual, expected = _run(model_runner, shape, dtype, shape[-1])
        if dtype == np.float16:
            compare_outputs(actual, expected, atol=1e-4, rtol=1e-2)
        else:
            compare_outputs(actual, expected, atol=1e-6, rtol=1e-4)
