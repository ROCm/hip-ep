#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""Numeric tests for ONNX RoiAlign (4-D NCHW)."""

import ml_dtypes
import numpy as np
import pytest
from onnx import TensorProto, helper

from framework.comparator import compare_outputs
from framework.onnx_utils import make_model_from_nodes, np_to_onnx_type


def _roi_align_node(inputs, output, out_h, out_w, mode, ctm, sampling_ratio, scale):
    # ctm=None omits the attribute, as an opset-10 model does.
    attrs = {} if ctm is None else {"coordinate_transformation_mode": ctm}
    return helper.make_node(
        "RoiAlign",
        inputs,
        [output],
        mode=mode,
        output_height=out_h,
        output_width=out_w,
        sampling_ratio=sampling_ratio,
        spatial_scale=scale,
        **attrs,
    )


def _make_roi_align_model(
    input_shape,
    num_rois,
    out_h,
    out_w,
    dtype=np.float32,
    mode="avg",
    coordinate_transformation_mode="half_pixel",
    sampling_ratio=2,
    spatial_scale=1.0,
    opset=16,
):
    tp = np_to_onnx_type(dtype)
    c = input_shape[1]
    X = helper.make_tensor_value_info("X", tp, input_shape)
    rois = helper.make_tensor_value_info("rois", tp, [num_rois, 4])
    idx = helper.make_tensor_value_info("batch_indices", TensorProto.INT64, [num_rois])
    Y = helper.make_tensor_value_info("Y", tp, [num_rois, c, out_h, out_w])
    node = _roi_align_node(
        ["X", "rois", "batch_indices"],
        "Y",
        out_h,
        out_w,
        mode,
        coordinate_transformation_mode,
        sampling_ratio,
        spatial_scale,
    )
    return make_model_from_nodes([node], [X, rois, idx], [Y], opset=opset)


def _random_rois(rng, num_rois, image_h, image_w, batch, dtype):
    x1 = rng.uniform(-2, image_w * 0.6, num_rois)
    y1 = rng.uniform(-2, image_h * 0.6, num_rois)
    x2 = x1 + rng.uniform(0.5, image_w * 0.6, num_rois)
    y2 = y1 + rng.uniform(0.5, image_h * 0.6, num_rois)
    rois = np.stack([x1, y1, x2, y2], axis=1).astype(dtype)
    idx = rng.integers(0, batch, num_rois).astype(np.int64)
    return rois, idx


class TestRoiAlign:
    @pytest.mark.parametrize(
        "mode,ctm,sampling_ratio",
        [
            ("avg", "half_pixel", 2),
            ("avg", "output_half_pixel", 2),
            ("avg", "half_pixel", 0),
            ("max", "half_pixel", 2),
            ("max", "output_half_pixel", 0),
        ],
    )
    def test_roi_align_f32(self, model_runner, mode, ctm, sampling_ratio):
        input_shape = [2, 4, 16, 16]
        spatial_scale = 0.5
        model = _make_roi_align_model(
            input_shape,
            "num_rois",
            3,
            3,
            np.float32,
            mode=mode,
            coordinate_transformation_mode=ctm,
            sampling_ratio=sampling_ratio,
            spatial_scale=spatial_scale,
        )
        rng = np.random.default_rng(21)
        x = rng.uniform(-2, 2, input_shape).astype(np.float32)
        rois, idx = _random_rois(rng, 6, 32, 32, input_shape[0], np.float32)
        actual, expected = model_runner.run_sample(model, [x, rois, idx])
        compare_outputs(actual, expected, atol=1e-5, rtol=1e-5)

    def test_roi_align_f16(self, model_runner):
        input_shape = [1, 3, 12, 12]
        model = _make_roi_align_model(
            input_shape, "num_rois", 2, 2, np.float16, sampling_ratio=2
        )
        rng = np.random.default_rng(22)
        x = rng.uniform(-2, 2, input_shape).astype(np.float16)
        rois, idx = _random_rois(rng, 4, 12, 12, 1, np.float16)
        actual, expected = model_runner.run_sample(
            model,
            [x, rois, idx],
            reference_model=_make_cast_reference_model(
                input_shape, "num_rois", 2, 2, np.float16
            ),
        )
        compare_outputs(actual, expected, atol=2e-3, rtol=1e-3)

    def test_roi_align_bf16(self, model_runner):
        input_shape = [1, 3, 12, 12]
        bf16 = ml_dtypes.bfloat16
        model = _make_roi_align_model(
            input_shape, "num_rois", 2, 2, bf16, sampling_ratio=2
        )
        rng = np.random.default_rng(23)
        x = rng.uniform(-2, 2, input_shape).astype(bf16)
        rois, idx = _random_rois(rng, 4, 12, 12, 1, bf16)
        actual, expected = model_runner.run_sample(
            model,
            [x, rois, idx],
            reference_model=_make_cast_reference_model(
                input_shape, "num_rois", 2, 2, bf16
            ),
        )
        compare_outputs(actual, expected, atol=3e-2, rtol=1e-2)

    @pytest.mark.parametrize("mode", ["avg", "max"])
    def test_roi_align_f64(self, model_runner, mode):
        input_shape = [2, 4, 16, 16]
        model = _make_roi_align_model(
            input_shape,
            "num_rois",
            3,
            3,
            np.float64,
            mode=mode,
            sampling_ratio=0,
            spatial_scale=0.5,
        )
        rng = np.random.default_rng(24)
        x = rng.uniform(-2, 2, input_shape).astype(np.float64)
        rois, idx = _random_rois(rng, 5, 32, 32, input_shape[0], np.float64)
        actual, expected = model_runner.run_sample(model, [x, rois, idx])
        compare_outputs(actual, expected, atol=1e-10, rtol=1e-10)

    @pytest.mark.parametrize("mode", ["avg", "max"])
    def test_roi_align_opset10_output_half_pixel(self, model_runner, mode):
        """RoiAlign-10 has no coordinate_transformation_mode and samples like
        output_half_pixel; a half_pixel default would shift every sample."""
        input_shape = [1, 4, 16, 16]
        model = _make_roi_align_model(
            input_shape,
            "num_rois",
            3,
            3,
            mode=mode,
            coordinate_transformation_mode=None,
            spatial_scale=0.5,
            opset=10,
        )
        rng = np.random.default_rng(25)
        x = rng.uniform(-2, 2, input_shape).astype(np.float32)
        rois, idx = _random_rois(rng, 4, 32, 32, 1, np.float32)
        actual, expected = model_runner.run_sample(model, [x, rois, idx])
        compare_outputs(actual, expected, atol=1e-5, rtol=1e-5)

    def test_roi_align_zero_rois(self, model_runner):
        input_shape = [1, 4, 8, 8]
        model = _make_roi_align_model(input_shape, "num_rois", 2, 2)
        x = np.random.default_rng(26).uniform(-1, 1, input_shape).astype(np.float32)
        rois = np.zeros((0, 4), np.float32)
        idx = np.zeros((0,), np.int64)
        actual, expected = model_runner.run_sample(model, [x, rois, idx])
        compare_outputs(actual, expected, atol=0)
        assert actual[0].shape == (0, 4, 2, 2)


def _make_cast_reference_model(input_shape, num_rois, out_h, out_w, dtype):
    """ORT CPU has no f16/bf16 RoiAlign; run it in f32 behind Casts."""
    c = input_shape[1]
    tp = np_to_onnx_type(dtype)
    X = helper.make_tensor_value_info("X", tp, input_shape)
    rois = helper.make_tensor_value_info("rois", tp, [num_rois, 4])
    idx = helper.make_tensor_value_info("batch_indices", TensorProto.INT64, [num_rois])
    Y = helper.make_tensor_value_info("Y", tp, [num_rois, c, out_h, out_w])
    nodes = [
        helper.make_node("Cast", ["X"], ["X32"], to=TensorProto.FLOAT),
        helper.make_node("Cast", ["rois"], ["rois32"], to=TensorProto.FLOAT),
        _roi_align_node(
            ["X32", "rois32", "batch_indices"],
            "Y32",
            out_h,
            out_w,
            "avg",
            "half_pixel",
            2,
            1.0,
        ),
        helper.make_node("Cast", ["Y32"], ["Y"], to=tp),
    ]
    return make_model_from_nodes(nodes, [X, rois, idx], [Y], opset=16)
