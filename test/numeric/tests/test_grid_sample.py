#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""Numeric tests for ONNX GridSample (4-D NCHW)."""

import numpy as np
import pytest
from onnx import TensorProto, helper

from framework.comparator import compare_outputs
from framework.onnx_utils import make_model_from_nodes


def _make_grid_sample_model(
    input_shape,
    grid_shape,
    dtype=np.float32,
    mode="bilinear",
    padding_mode="zeros",
    align_corners=0,
):
    tp = {np.float16: TensorProto.FLOAT16, np.float32: TensorProto.FLOAT}[dtype]
    n, c, _, _ = input_shape
    _, oh, ow, _ = grid_shape
    X = helper.make_tensor_value_info("X", tp, input_shape)
    Grid = helper.make_tensor_value_info("grid", tp, grid_shape)
    Y = helper.make_tensor_value_info("Y", tp, [n, c, oh, ow])
    node = helper.make_node(
        "GridSample",
        ["X", "grid"],
        ["Y"],
        mode=mode,
        padding_mode=padding_mode,
        align_corners=align_corners,
    )
    return make_model_from_nodes([node], [X, Grid], [Y], opset=17)


def _identity_grid(n, oh, ow, dtype):
    """Normalized grid covering the interior of [-1, 1]."""
    ys = np.linspace(-0.9, 0.9, oh, dtype=np.float32)
    xs = np.linspace(-0.9, 0.9, ow, dtype=np.float32)
    grid_y, grid_x = np.meshgrid(ys, xs, indexing="ij")
    grid = np.stack([grid_x, grid_y], axis=-1)
    return np.broadcast_to(grid, (n, oh, ow, 2)).astype(dtype, copy=True)


class TestGridSample:
    @pytest.mark.parametrize(
        "mode,padding_mode,align_corners",
        [
            ("bilinear", "zeros", 0),
            ("bilinear", "border", 0),
            ("nearest", "zeros", 0),
            ("bilinear", "zeros", 1),
        ],
    )
    def test_grid_sample_f32(
        self, model_runner, mode, padding_mode, align_corners
    ):
        input_shape = [1, 3, 8, 8]
        grid_shape = [1, 4, 4, 2]
        model = _make_grid_sample_model(
            input_shape,
            grid_shape,
            np.float32,
            mode=mode,
            padding_mode=padding_mode,
            align_corners=align_corners,
        )
        rng = np.random.default_rng(11)
        x = rng.uniform(-2, 2, input_shape).astype(np.float32)
        grid = _identity_grid(1, 4, 4, np.float32)
        actual, expected = model_runner.run_sample(model, [x, grid])
        compare_outputs(actual, expected, atol=1e-5, rtol=1e-5)

    def test_grid_sample_f16(self, model_runner):
        input_shape = [1, 2, 8, 8]
        grid_shape = [1, 3, 3, 2]
        model = _make_grid_sample_model(
            input_shape, grid_shape, np.float16, mode="bilinear"
        )
        rng = np.random.default_rng(12)
        x = rng.uniform(-2, 2, input_shape).astype(np.float16)
        grid = _identity_grid(1, 3, 3, np.float16)
        actual, expected = model_runner.run_sample(model, [x, grid])
        compare_outputs(actual, expected, atol=2e-3, rtol=1e-3)
