#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""Nearest integer upsample feeding a convolution, with rocMLIR forced on.

--hip-fuse-rocmlir absorbs the resize into the rocMLIR convolution as an
expand_shape -> hip.expand -> collapse_shape view, so no resize kernel runs.
The reference is ONNX Runtime's CPU EP running the unfused Resize then Conv.
The 3D case builds a rank-8 view and the 2D case a rank-6 view.
"""

from __future__ import annotations

import numpy as np
import pytest
from onnx import TensorProto, helper

from framework.comparator import compare_outputs
from framework.onnx_utils import make_model_from_nodes


def _make_resize_conv_model(
    in_shape: list[int],
    spatial_scale: int,
    out_channels: int,
    nearest_mode: str,
    use_sizes: bool,
    seed: int,
):
    rng = np.random.default_rng(seed)
    spatial = len(in_shape) - 2
    up_shape = in_shape[:2] + [d * spatial_scale for d in in_shape[2:]]
    kernel = [3] * spatial
    w = rng.uniform(-0.5, 0.5, [out_channels, in_shape[1], *kernel])
    b = rng.uniform(-0.5, 0.5, [out_channels])
    w_init = helper.make_tensor(
        "W", TensorProto.FLOAT16, w.shape, w.astype(np.float16).ravel()
    )
    b_init = helper.make_tensor(
        "B", TensorProto.FLOAT16, b.shape, b.astype(np.float16).ravel()
    )

    if use_sizes:
        resize_init = helper.make_tensor(
            "sizes", TensorProto.INT64, [len(up_shape)], up_shape
        )
        resize_inputs = ["X", "", "", "sizes"]
    else:
        scales = [1.0, 1.0] + [float(spatial_scale)] * spatial
        resize_init = helper.make_tensor(
            "scales", TensorProto.FLOAT, [len(scales)], scales
        )
        resize_inputs = ["X", "", "scales"]

    resize = helper.make_node(
        "Resize",
        resize_inputs,
        ["U"],
        mode="nearest",
        coordinate_transformation_mode="asymmetric",
        nearest_mode=nearest_mode,
    )
    conv = helper.make_node(
        "Conv",
        ["U", "W", "B"],
        ["Y"],
        kernel_shape=kernel,
        pads=[1] * (2 * spatial),
        strides=[1] * spatial,
    )
    out_shape = [in_shape[0], out_channels, *up_shape[2:]]
    x_info = helper.make_tensor_value_info("X", TensorProto.FLOAT16, in_shape)
    y_info = helper.make_tensor_value_info("Y", TensorProto.FLOAT16, out_shape)
    return make_model_from_nodes(
        [resize, conv],
        [x_info],
        [y_info],
        initializers=[resize_init, w_init, b_init],
        opset=13,
    )


@pytest.mark.parametrize(
    "in_shape,spatial_scale,nearest_mode,use_sizes",
    [
        # 2D, floor, explicit scales: rank-6 view.
        ([1, 8, 16, 16], 2, "floor", False),
        # 2D, round_prefer_floor at scale 2 agrees with floor on every sample.
        ([1, 8, 16, 16], 2, "round_prefer_floor", True),
        # 2D, scale 4 with floor.
        ([1, 4, 8, 8], 4, "floor", True),
        # 3D: rank-8 view.
        ([1, 4, 4, 8, 8], 2, "floor", False),
    ],
    ids=["2d_x2_floor", "2d_x2_round_prefer_floor", "2d_x4_floor", "3d_x2_floor"],
)
def test_nearest_upsample_conv_rocmlir(
    model_runner, monkeypatch, in_shape, spatial_scale, nearest_mode, use_sizes
):
    monkeypatch.setenv("HIPDNN_EP_ROCMLIR", "1")
    model = _make_resize_conv_model(
        in_shape,
        spatial_scale,
        out_channels=8,
        nearest_mode=nearest_mode,
        use_sizes=use_sizes,
        seed=11,
    )
    x = np.random.default_rng(5).uniform(-1, 1, in_shape).astype(np.float16)
    actual, expected = model_runner.run_sample(model, [x])
    assert np.isfinite(actual[0]).all()
    compare_outputs(actual, expected, atol=2e-2, rtol=2e-2)
