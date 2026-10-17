#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""Numeric coverage for ONNX RandomNormalLike.

ONNX does not define the generator, so hip-ep values are not compared to
the ORT CPU reference except for the shape, the dtype, and the scale=0
case (every element is the mean). A fixed seed must repeat, and the input
values must not affect the output.
"""

from __future__ import annotations

import ml_dtypes
import numpy as np
import pytest
from onnx import helper

from framework.comparator import compare_outputs
from framework.onnx_utils import make_model_from_nodes, np_to_onnx_type


def _make_model(
    shape,
    in_dtype,
    out_dtype,
    *,
    mean=0.0,
    scale=1.0,
    seed=1.0,
    symbolic=False,
):
    in_type = np_to_onnx_type(in_dtype)
    out_type = np_to_onnx_type(out_dtype)
    dims = ["d" + str(i) for i in range(len(shape))] if symbolic else list(shape)
    x = helper.make_tensor_value_info("x", in_type, dims)
    y = helper.make_tensor_value_info("y", out_type, dims)
    attrs = {"mean": float(mean), "scale": float(scale)}
    if seed is not None:
        attrs["seed"] = float(seed)
    if in_dtype != out_dtype:
        attrs["dtype"] = int(out_type)
    node = helper.make_node("RandomNormalLike", ["x"], ["y"], **attrs)
    return make_model_from_nodes([node], [x], [y], opset=21)


def _as_f64(values):
    if values.dtype == ml_dtypes.bfloat16 or values.dtype == np.dtype("bfloat16"):
        return values.astype(np.float32).astype(np.float64)
    return values.astype(np.float64)


def _assert_normal(values, mean, scale, mean_tol, std_tol):
    flat = _as_f64(values).reshape(-1)
    assert flat.shape[0] > 0
    assert np.isfinite(flat).all()
    assert abs(flat.mean() - mean) < mean_tol
    if scale == 0.0:
        assert np.all(flat == mean)
    else:
        assert abs(flat.std() - abs(scale)) < std_tol


class TestRandomNormalLike:
    @pytest.mark.parametrize(
        "dtype,mean_tol,std_tol",
        [
            (np.float32, 0.15, 0.2),
            (np.float64, 0.15, 0.2),
            (np.float16, 0.25, 0.3),
            (ml_dtypes.bfloat16, 0.25, 0.35),
        ],
    )
    def test_distribution(self, model_runner, dtype, mean_tol, std_tol):
        shape = [64, 128]
        model = _make_model(shape, dtype, dtype, mean=0.5, scale=2.0, seed=7.0)
        x = np.zeros(shape, dtype=dtype)
        actual, expected = model_runner.run_sample(model, [x])
        assert actual[0].shape == tuple(shape)
        assert actual[0].dtype == expected[0].dtype
        _assert_normal(actual[0], 0.5, 2.0, mean_tol, std_tol)

    def test_seed_repeats_and_ignores_input_values(self, model_runner):
        shape = [32, 32]
        model = _make_model(shape, np.float32, np.float32, seed=11.0)
        zeros = np.zeros(shape, dtype=np.float32)
        other = np.full(shape, 9.0, dtype=np.float32)
        first, _ = model_runner.run_sample(model, [zeros], name="rnl_seed_a")
        second, _ = model_runner.run_sample(model, [zeros], name="rnl_seed_b")
        third, _ = model_runner.run_sample(model, [other], name="rnl_seed_c")
        compare_outputs(first, second, atol=0)
        compare_outputs(first, third, atol=0)

    def test_different_seeds_differ(self, model_runner):
        shape = [64, 16]
        x = np.zeros(shape, dtype=np.float32)
        a, _ = model_runner.run_sample(
            _make_model(shape, np.float32, np.float32, seed=1.0),
            [x],
            name="rnl_seed_1",
        )
        b, _ = model_runner.run_sample(
            _make_model(shape, np.float32, np.float32, seed=2.0),
            [x],
            name="rnl_seed_2",
        )
        assert not np.array_equal(a[0], b[0])

    def test_scale_zero_is_mean(self, model_runner):
        shape = [8, 8]
        model = _make_model(
            shape, np.float32, np.float32, mean=3.0, scale=0.0, seed=1.0
        )
        x = np.zeros(shape, dtype=np.float32)
        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=0)
        assert np.all(actual[0] == np.float32(3.0))

    def test_dynamic_shape_and_int_input(self, model_runner):
        shape = [16, 16, 16]
        model = _make_model(
            shape,
            np.int32,
            np.float32,
            mean=0.0,
            scale=1.0,
            seed=5.0,
            symbolic=True,
        )
        x = np.arange(np.prod(shape), dtype=np.int32).reshape(shape)
        actual, expected = model_runner.run_sample(model, [x])
        assert actual[0].shape == tuple(shape)
        assert actual[0].dtype == np.float32
        assert expected[0].shape == tuple(shape)
        _assert_normal(actual[0], 0.0, 1.0, 0.15, 0.2)
