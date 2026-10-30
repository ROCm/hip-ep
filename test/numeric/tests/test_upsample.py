#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""Tests for ONNX Upsample and Resize, lowered through hip.resize.

Upsample is Resize with asymmetric coordinates. Nearest sampling uses floor.
output_dim = floor(input_dim * scale). Schema 9 passes scales as an input.
Opset 7 and 8 store the same list as an attribute. The two Deeplab shapes
are the linear nodes in ul_deeplabv3_fp16.

Resize pytorch_half_pixel maps every output of a length-1 axis to input 0.
Other lengths use half_pixel. When floor(input * scale) rounds, the
coordinate transform must use the supplied scale, not output / input.

The runtime kernel writes the widest store of 16, 8, 4 or 2 bytes that tiles
the output row, and one element otherwise.
TestResizeRowKernel covers each width for f32, f16 and 8-bit types, ranks 1 to
3, downsampling, and a scale of 3, where a reciprocal multiply would floor to
the wrong source index. The ONNX importer reads bf16 tensors as f32, so a bf16
Resize cannot reach the bf16 kernel from a model.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from onnx import TensorProto, helper

from framework.comparator import compare_outputs
from framework.onnx_utils import make_model_from_nodes, np_to_onnx_type


def _make_upsample_attr_model(
    shape: list[int], scales: list[float], dtype: np.dtype, mode: str
):
    out_shape = [int(math.floor(dim * scale)) for dim, scale in zip(shape, scales)]
    x_info = helper.make_tensor_value_info("X", np_to_onnx_type(dtype), list(shape))
    y_info = helper.make_tensor_value_info("Y", np_to_onnx_type(dtype), list(out_shape))
    node = helper.make_node("Upsample", ["X"], ["Y"], mode=mode, scales=scales)
    # Opset 9 moved scales from an attribute to an input.
    return make_model_from_nodes([node], [x_info], [y_info], opset=8)


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


def _make_resize_sizes_model(
    shape: list[int], sizes: list[int], dtype: np.dtype, mode: str, coord: str
):
    x_info = helper.make_tensor_value_info("X", np_to_onnx_type(dtype), list(shape))
    y_info = helper.make_tensor_value_info("Y", np_to_onnx_type(dtype), list(sizes))
    sizes_init = helper.make_tensor("sizes", TensorProto.INT64, [len(sizes)], sizes)
    node = helper.make_node(
        "Resize",
        ["X", "", "", "sizes"],
        ["Y"],
        mode=mode,
        coordinate_transformation_mode=coord,
    )
    return make_model_from_nodes(
        [node], [x_info], [y_info], initializers=[sizes_init], opset=13
    )


def _make_resize_scales_model(
    shape: list[int],
    scales: list[float],
    dtype: np.dtype,
    mode: str,
    coord: str,
    nearest_mode: str = "round_prefer_floor",
    out_shape: list[int] | None = None,
    cast_through: list[int] = (),
):
    if out_shape is None:
        out_shape = [int(math.floor(dim * s)) for dim, s in zip(shape, scales)]
    x_info = helper.make_tensor_value_info("X", np_to_onnx_type(dtype), list(shape))
    y_info = helper.make_tensor_value_info("Y", np_to_onnx_type(dtype), out_shape)
    scales_init = helper.make_tensor("scales", TensorProto.FLOAT, [len(scales)], scales)
    # Each Cast rounds the scales to its element type before Resize reads them.
    cast_types = list(cast_through)
    if cast_types:
        cast_types.append(TensorProto.FLOAT)
    nodes = []
    scales_name = "scales"
    for i, to in enumerate(cast_types):
        nodes.append(helper.make_node("Cast", [scales_name], [f"scales_{i}"], to=to))
        scales_name = f"scales_{i}"
    nodes.append(
        helper.make_node(
            "Resize",
            ["X", "", scales_name],
            ["Y"],
            mode=mode,
            coordinate_transformation_mode=coord,
            nearest_mode=nearest_mode,
        )
    )
    return make_model_from_nodes(
        nodes, [x_info], [y_info], initializers=[scales_init], opset=13
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

    def test_nearest_opset8_scales_attribute(self, model_runner):
        shape = [1, 128, 56, 56]
        scales = [1.0, 1.0, 2.0, 2.0]
        model = _make_upsample_attr_model(shape, scales, np.float32, "nearest")
        rng = np.random.default_rng(203)
        x = rng.uniform(-2.0, 2.0, shape).astype(np.float32)
        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=0)


class TestResize:
    @pytest.mark.parametrize(
        "out_len,expected",
        [
            (1, [0.0]),
            (2, [0.5, 2.5]),
        ],
    )
    def test_pytorch_half_pixel_linear(self, model_runner, out_len, expected):
        model = _make_resize_sizes_model(
            [1, 1, 4], [1, 1, out_len], np.float32, "linear", "pytorch_half_pixel"
        )
        x = np.arange(4, dtype=np.float32).reshape(1, 1, 4)
        actual, reference = model_runner.run_sample(model, [x])
        compare_outputs(actual, reference, atol=1e-6)
        np.testing.assert_allclose(
            actual[0].reshape(-1), np.array(expected, np.float32), atol=1e-6
        )

    def test_scales_kept_when_floor_rounds(self, model_runner):
        model = _make_resize_scales_model(
            [1, 1, 3], [1.0, 1.0, 1.5], np.float32, "linear", "half_pixel"
        )
        x = np.arange(3, dtype=np.float32).reshape(1, 1, 3)
        actual, reference = model_runner.run_sample(model, [x])
        compare_outputs(actual, reference, atol=1e-5)
        np.testing.assert_allclose(
            actual[0].reshape(-1),
            np.array([0.0, 0.5, 1.1666667, 1.8333334], np.float32),
            atol=1e-5,
        )

    @pytest.mark.parametrize(
        "shape,scales,mode,coord",
        [
            ([1, 3, 7, 9], [1.0, 1.0, 1.5, 2.3], "linear", "asymmetric"),
            ([1, 3, 7, 9], [1.0, 1.0, 1.5, 2.3], "linear", "pytorch_half_pixel"),
            ([1, 3, 7, 9], [1.0, 1.0, 1.5, 2.3], "nearest", "half_pixel"),
            ([1, 3, 8, 8], [1.0, 1.0, 2.0, 2.0], "linear", "half_pixel"),
        ],
    )
    def test_scales(self, model_runner, shape, scales, mode, coord):
        model = _make_resize_scales_model(shape, scales, np.float32, mode, coord)
        rng = np.random.default_rng(301)
        x = rng.uniform(-1.0, 1.0, shape).astype(np.float32)
        actual, reference = model_runner.run_sample(model, [x])
        compare_outputs(actual, reference, atol=1e-4, rtol=1e-4)

    def test_scales_unchanged_extent_is_resampled(self, model_runner):
        # floor(3 * 1.1) = 3 keeps the H extent, but H is still resampled.
        model = _make_resize_scales_model(
            [1, 3, 3, 1],
            [1.0, 1.1, 1.5, 1.0],
            np.float32,
            "nearest",
            "pytorch_half_pixel",
            nearest_mode="floor",
        )
        x = np.repeat(np.array([0.0, 10.0, 20.0], np.float32), 3).reshape(1, 3, 3, 1)
        actual, reference = model_runner.run_sample(model, [x])
        compare_outputs(actual, reference, atol=0)
        np.testing.assert_array_equal(actual[0][0, :, 0, 0], [0.0, 0.0, 10.0])

    def test_scales_rounded_by_cast(self, model_runner):
        # 2.0001 through f16 is 2, so [0, 1, 2, 3] doubles exactly.
        model = _make_resize_scales_model(
            [1, 1, 4],
            [1.0, 1.0, 2.0001],
            np.float32,
            "nearest",
            "asymmetric",
            nearest_mode="floor",
            out_shape=[1, 1, 8],
            cast_through=[TensorProto.FLOAT16],
        )
        x = np.arange(4, dtype=np.float32).reshape(1, 1, 4)
        actual, reference = model_runner.run_sample(model, [x])
        compare_outputs(actual, reference, atol=0)
        np.testing.assert_array_equal(actual[0].reshape(-1), [0, 0, 1, 1, 2, 2, 3, 3])


class TestResizeRowKernel:
    @pytest.mark.parametrize(
        "shape,scales,coord,nearest_mode",
        [
            # 14 outputs per row: 2-element stores, not the f32 vector of 4.
            ([1, 3, 5, 7], [1.0, 1.0, 2.0, 2.0], "asymmetric", "floor"),
            ([1, 2, 4, 5], [1.0, 1.0, 3.0, 3.0], "asymmetric", "floor"),
            ([1, 2, 4, 6], [1.0, 1.0, 3.0, 3.0], "half_pixel", "round_prefer_floor"),
            ([1, 2, 16, 32], [1.0, 1.0, 0.5, 0.5], "asymmetric", "floor"),
            ([1, 2, 16, 32], [1.0, 1.0, 0.5, 0.5], "half_pixel", "round_prefer_ceil"),
            ([1, 3, 12, 16], [1.0, 1.0, 2.0, 2.0], "align_corners", "ceil"),
            ([1, 3, 12, 16], [1.0, 1.0, 2.0, 2.0], "pytorch_half_pixel", "floor"),
            ([1, 2, 3, 5, 8], [1.0, 1.0, 2.0, 2.0, 2.0], "asymmetric", "floor"),
            (
                [1, 2, 3, 5, 7],
                [1.0, 1.0, 1.5, 2.0, 1.5],
                "half_pixel",
                "round_prefer_floor",
            ),
            ([1, 4, 50], [1.0, 1.0, 2.5], "half_pixel", "round_prefer_floor"),
            ([1, 4, 48], [1.0, 1.0, 2.0], "asymmetric", "floor"),
        ],
    )
    def test_nearest_f32(self, model_runner, shape, scales, coord, nearest_mode):
        model = _make_resize_scales_model(
            shape, scales, np.float32, "nearest", coord, nearest_mode=nearest_mode
        )
        rng = np.random.default_rng(401)
        x = rng.uniform(-2.0, 2.0, shape).astype(np.float32)
        actual, reference = model_runner.run_sample(model, [x])
        compare_outputs(actual, reference, atol=0)

    @pytest.mark.parametrize(
        "shape,scales,coord",
        [
            ([1, 3, 5, 7], [1.0, 1.0, 2.0, 2.0], "half_pixel"),
            ([1, 2, 4, 5], [1.0, 1.0, 3.0, 3.0], "asymmetric"),
            ([1, 2, 16, 32], [1.0, 1.0, 0.5, 0.5], "half_pixel"),
            ([1, 3, 12, 16], [1.0, 1.0, 2.0, 2.0], "asymmetric"),
            ([1, 3, 12, 16], [1.0, 1.0, 2.0, 2.0], "half_pixel"),
            ([1, 3, 12, 16], [1.0, 1.0, 2.0, 2.0], "align_corners"),
            ([1, 3, 12, 16], [1.0, 1.0, 2.0, 2.0], "pytorch_half_pixel"),
            ([1, 2, 3, 5, 8], [1.0, 1.0, 2.0, 2.0, 2.0], "half_pixel"),
            ([1, 2, 3, 5, 7], [1.0, 1.0, 1.5, 2.0, 1.5], "asymmetric"),
            ([1, 4, 50], [1.0, 1.0, 2.5], "half_pixel"),
            ([1, 4, 48], [1.0, 1.0, 2.0], "asymmetric"),
        ],
    )
    def test_linear_f32(self, model_runner, shape, scales, coord):
        model = _make_resize_scales_model(shape, scales, np.float32, "linear", coord)
        rng = np.random.default_rng(402)
        x = rng.uniform(-2.0, 2.0, shape).astype(np.float32)
        actual, reference = model_runner.run_sample(model, [x])
        compare_outputs(actual, reference, atol=1e-5, rtol=1e-5)

    # 14, 20 and 24 outputs per row use 2-, 4- and 8-element f16 stores.
    @pytest.mark.parametrize("width", [7, 10, 12])
    @pytest.mark.parametrize("mode", ["nearest", "linear"])
    def test_f16(self, model_runner, width, mode):
        shape = [1, 4, 6, width]
        scales = [1.0, 1.0, 2.0, 2.0]
        model = _make_resize_scales_model(shape, scales, np.float16, mode, "half_pixel")
        rng = np.random.default_rng(403)
        x = rng.uniform(-2.0, 2.0, shape).astype(np.float16)
        actual, reference = model_runner.run_sample(model, [x])
        atol = 0 if mode == "nearest" else 1e-2
        compare_outputs(actual, reference, atol=atol, rtol=atol)

    # 20, 32, 40 and 50 outputs per row use 4-, 16-, 8- and 2-byte stores.
    @pytest.mark.parametrize("width", [10, 16, 20, 25])
    @pytest.mark.parametrize("dtype", [np.int8, np.uint8])
    def test_nearest_8bit(self, model_runner, width, dtype):
        shape = [1, 4, 6, width]
        scales = [1.0, 1.0, 2.0, 2.0]
        model = _make_resize_scales_model(
            shape, scales, dtype, "nearest", "asymmetric", nearest_mode="floor"
        )
        rng = np.random.default_rng(405)
        info = np.iinfo(dtype)
        x = rng.integers(info.min, info.max + 1, shape).astype(dtype)
        actual, reference = model_runner.run_sample(model, [x])
        compare_outputs(actual, reference, atol=0)
