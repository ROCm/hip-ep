#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""Tests for the ONNX AveragePool / MaxPool operators.

The cases cover the three pool kernels by shape:

* 2x2 / 3x3 windows: the small-window kernel. 2x2 stride-2 pools that tile
  the input exactly take its vectorized path (fp16 needs W % 8 == 0, fp32
  W % 4 == 0); unaligned widths, odd heights, ceil mode and padding take the
  generic path. The tail cases have an output width that is not a multiple
  of the outputs one thread writes, across several planes.
* Windows of 64 or more taps: the wide kernel, 32 lanes per output. The
  extra cases exercise the k2 carry (3-D 5x5x5), a 1x64 window whose step0
  is 0, dilation, count_include_pad, LpPool, and fp16.
* Everything else, and MaxPool with argmax indices: the one-thread-per-output
  kernel.
* An average pool whose one window covers the whole input does not reach a pool
  kernel: the compiler turns it into a mean over the spatial dims.
"""

import numpy as np
import pytest
from onnx import TensorProto, helper

from framework.comparator import compare_outputs
from framework.onnx_utils import make_model_from_nodes, np_to_onnx_type


def _pool_output_shape(input_shape, attrs):
    """ONNX pooling output extent per spatial axis (floor or ceil mode)."""
    kernel = attrs["kernel_shape"]
    rank = len(kernel)
    strides = attrs.get("strides", [1] * rank)
    pads = attrs.get("pads", [0] * (2 * rank))
    dilations = attrs.get("dilations", [1] * rank)
    out = list(input_shape[:2])
    for i in range(rank):
        span = (kernel[i] - 1) * dilations[i] + 1
        num = input_shape[2 + i] + pads[i] + pads[i + rank] - span
        o = (
            -(-num // strides[i]) if attrs.get("ceil_mode", 0) else num // strides[i]
        ) + 1
        # A ceil-mode window must start inside the input or the left pad.
        if (
            attrs.get("ceil_mode", 0)
            and (o - 1) * strides[i] >= input_shape[2 + i] + pads[i]
        ):
            o -= 1
        out.append(o)
    return out


def _make_pool_model(op, input_shape, dtype, attrs, with_indices=False):
    tp = np_to_onnx_type(dtype)
    X = helper.make_tensor_value_info("X", tp, list(input_shape))
    out_shape = _pool_output_shape(input_shape, attrs)
    outputs = [helper.make_tensor_value_info("Y", tp, out_shape)]
    names = ["Y"]
    if with_indices:
        outputs.append(helper.make_tensor_value_info("I", TensorProto.INT64, out_shape))
        names.append("I")
    node = helper.make_node(op, ["X"], names, **attrs)
    return make_model_from_nodes([node], [X], outputs, opset=19)


SMALL_CASES = [
    ("AveragePool", [1, 48, 96, 112], dict(kernel_shape=[2, 2], strides=[2, 2])),
    ("MaxPool", [2, 4, 32, 64], dict(kernel_shape=[2, 2], strides=[2, 2])),
    ("AveragePool", [1, 3, 20, 84], dict(kernel_shape=[2, 2], strides=[2, 2])),
    (
        "AveragePool",
        [1, 5, 17, 23],
        dict(kernel_shape=[2, 2], strides=[2, 2], ceil_mode=1),
    ),
    (
        "AveragePool",
        [1, 4, 16, 16],
        dict(
            kernel_shape=[2, 2], strides=[2, 2], pads=[1, 1, 1, 1], count_include_pad=1
        ),
    ),
    ("MaxPool", [1, 3, 9, 11], dict(kernel_shape=[2, 2], strides=[1, 1])),
    (
        "AveragePool",
        [1, 8, 33, 35],
        dict(kernel_shape=[3, 3], strides=[1, 1], pads=[1, 1, 1, 1]),
    ),
    (
        "MaxPool",
        [2, 5, 40, 41],
        dict(kernel_shape=[3, 3], strides=[2, 2], pads=[1, 1, 1, 1]),
    ),
    ("MaxPool", [1, 7, 27, 29], dict(kernel_shape=[3, 3], strides=[2, 2], ceil_mode=1)),
    # OW = 9, and the 2x2 kernel writes V = 2 outputs per thread.
    ("AveragePool", [2, 3, 8, 10], dict(kernel_shape=[2, 2], strides=[1, 1])),
    # OW = 7, and the 3x3 stride-2 kernel also writes V = 2.
    ("MaxPool", [2, 4, 15, 16], dict(kernel_shape=[3, 3], strides=[2, 2])),
]
SMALL_IDS = [
    "avg2x2_vec",
    "max2x2_vec",
    "avg2x2_unaligned",
    "avg2x2_ceil",
    "avg2x2_pad_include",
    "max2x2_s1",
    "avg3x3_s1_pad",
    "max3x3_s2_pad",
    "max3x3_ceil",
    "avg2x2_ow_tail",
    "max3x3_ow_tail",
]

WIDE_CASES = [
    ("AveragePool", [1, 3, 768, 768], dict(kernel_shape=[32, 32], strides=[32, 32])),
    (
        "AveragePool",
        [2, 4, 33, 35],
        dict(kernel_shape=[9, 9], strides=[1, 1], pads=[4, 4, 4, 4]),
    ),
    (
        "MaxPool",
        [2, 5, 40, 40],
        dict(kernel_shape=[8, 8], strides=[4, 4], pads=[2, 2, 2, 2]),
    ),
    # k = (5, 5, 5) decomposes 32 as step (1, 1, 2), so the k2 carry runs.
    (
        "AveragePool",
        [1, 2, 8, 8, 8],
        dict(kernel_shape=[5, 5, 5], strides=[2, 2, 2]),
    ),
    # k1 * k2 = 64, so step0 is 0 and the position advances along the width.
    ("MaxPool", [1, 2, 4, 80], dict(kernel_shape=[1, 64], strides=[1, 1])),
    (
        "AveragePool",
        [1, 2, 20, 20],
        dict(kernel_shape=[8, 8], strides=[4, 4], dilations=[2, 2]),
    ),
    (
        "AveragePool",
        [1, 2, 20, 20],
        dict(
            kernel_shape=[8, 8],
            strides=[8, 8],
            pads=[2, 2, 2, 2],
            count_include_pad=1,
        ),
    ),
    ("LpPool", [1, 2, 16, 16], dict(kernel_shape=[8, 8], strides=[8, 8], p=2)),
]
WIDE_IDS = [
    "avg32x32",
    "avg9x9_pad",
    "max8x8_pad",
    "avg5x5x5",
    "max1x64",
    "avg8x8_dil2",
    "avg8x8_pad_include",
    "lp8x8",
]


# Average pools whose one window covers the whole input (the compiler turns
# them into a mean over the spatial dims). A kernel larger than the input is
# only legal with ceil_mode; either count_include_pad mode then divides by the
# number of input elements.
WHOLE_INPUT_CASES = [
    ([2, 32, 7, 7], dict(kernel_shape=[7, 7], strides=[7, 7])),
    ([1, 5, 9, 4], dict(kernel_shape=[9, 4], strides=[1, 1])),
    ([2, 8, 7, 7], dict(kernel_shape=[9, 9], strides=[9, 9], ceil_mode=1)),
    (
        [1, 8, 7, 7],
        dict(kernel_shape=[9, 9], strides=[9, 9], ceil_mode=1, count_include_pad=1),
    ),
    ([1, 4, 5, 3], dict(kernel_shape=[32, 32], strides=[32, 32], ceil_mode=1)),
    (
        [1, 4, 5, 3],
        dict(kernel_shape=[32, 32], strides=[32, 32], ceil_mode=1, count_include_pad=1),
    ),
    ([2, 6, 10], dict(kernel_shape=[10], strides=[10])),
    ([2, 6, 10], dict(kernel_shape=[12], strides=[12], ceil_mode=1)),
    (
        [2, 6, 10],
        dict(kernel_shape=[12], strides=[12], ceil_mode=1, count_include_pad=1),
    ),
    ([1, 3, 4, 5, 6], dict(kernel_shape=[4, 5, 6], strides=[1, 1, 1])),
]
WHOLE_INPUT_IDS = [
    "global7x7",
    "global9x4",
    "larger_ceil",
    "larger_ceil_include_pad",
    "much_larger_ceil",
    "much_larger_ceil_include_pad",
    "1d",
    "1d_larger_ceil",
    "1d_larger_ceil_include_pad",
    "3d",
]


class TestPool:
    @pytest.mark.parametrize("dtype", [np.float32, np.float16], ids=["fp32", "fp16"])
    @pytest.mark.parametrize("op, shape, attrs", SMALL_CASES, ids=SMALL_IDS)
    def test_pool_small_window(self, model_runner, op, shape, attrs, dtype):
        rng = np.random.default_rng(61)
        x = rng.uniform(-3, 3, shape).astype(dtype)
        model = _make_pool_model(op, shape, dtype, attrs)
        actual, expected = model_runner.run_sample(model, [x])
        tol = 2e-3 if dtype == np.float16 else 1e-5
        compare_outputs(actual, expected, atol=tol, rtol=tol)

    @pytest.mark.parametrize("op, shape, attrs", WIDE_CASES, ids=WIDE_IDS)
    def test_pool_wide_window(self, model_runner, op, shape, attrs):
        rng = np.random.default_rng(62)
        x = rng.uniform(-3, 3, shape).astype(np.float32)
        model = _make_pool_model(op, shape, np.float32, attrs)
        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=1e-5, rtol=1e-4)

    def test_pool_wide_window_fp16(self, model_runner):
        """fp16 wide window. The kernel accumulates in fp32, so the tolerance
        covers a reference that sums in fp16."""
        shape = [1, 2, 16, 16]
        attrs = dict(kernel_shape=[8, 8], strides=[8, 8])
        rng = np.random.default_rng(65)
        x = rng.uniform(-3, 3, shape).astype(np.float16)
        model = _make_pool_model("AveragePool", shape, np.float16, attrs)
        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=2e-2, rtol=2e-2)

    @pytest.mark.parametrize("dtype", [np.float32, np.float16], ids=["fp32", "fp16"])
    @pytest.mark.parametrize("shape, attrs", WHOLE_INPUT_CASES, ids=WHOLE_INPUT_IDS)
    def test_avgpool_whole_input(self, model_runner, shape, attrs, dtype):
        rng = np.random.default_rng(66)
        x = rng.uniform(-3, 3, shape).astype(dtype)
        model = _make_pool_model("AveragePool", shape, dtype, attrs)
        actual, expected = model_runner.run_sample(model, [x])
        tol = 2e-3 if dtype == np.float16 else 1e-5
        compare_outputs(actual, expected, atol=tol, rtol=tol)

    @pytest.mark.parametrize("dtype", [np.float32, np.float16], ids=["fp32", "fp16"])
    def test_global_average_pool(self, model_runner, dtype):
        shape = [2, 32, 7, 7]
        tp = np_to_onnx_type(dtype)
        X = helper.make_tensor_value_info("X", tp, shape)
        Y = helper.make_tensor_value_info("Y", tp, [2, 32, 1, 1])
        node = helper.make_node("GlobalAveragePool", ["X"], ["Y"])
        model = make_model_from_nodes([node], [X], [Y], opset=19)
        rng = np.random.default_rng(67)
        x = rng.uniform(-3, 3, shape).astype(dtype)
        actual, expected = model_runner.run_sample(model, [x])
        tol = 2e-3 if dtype == np.float16 else 1e-5
        compare_outputs(actual, expected, atol=tol, rtol=tol)

    @pytest.mark.parametrize("batch", [1, 3])
    @pytest.mark.parametrize("dtype", [np.float32, np.float16], ids=["fp32", "fp16"])
    def test_avgpool_whole_input_dynamic_batch(self, model_runner, batch, dtype):
        """The batch stays dynamic through the rewrite; spatial extents are static."""
        tp = np_to_onnx_type(dtype)
        X = helper.make_tensor_value_info("X", tp, ["batch", 16, 7, 7])
        Y = helper.make_tensor_value_info("Y", tp, ["batch", 16, 1, 1])
        node = helper.make_node(
            "AveragePool", ["X"], ["Y"], kernel_shape=[7, 7], strides=[7, 7]
        )
        model = make_model_from_nodes([node], [X], [Y], opset=19)
        rng = np.random.default_rng(68)
        x = rng.uniform(-3, 3, [batch, 16, 7, 7]).astype(dtype)
        actual, expected = model_runner.run_sample(model, [x])
        assert actual[0].shape == (batch, 16, 1, 1)
        tol = 2e-3 if dtype == np.float16 else 1e-5
        compare_outputs(actual, expected, atol=tol, rtol=tol)

    def test_maxpool_with_indices(self, model_runner):
        """Argmax indices keep the one-thread-per-output kernel."""
        shape = [1, 4, 16, 16]
        rng = np.random.default_rng(63)
        x = rng.uniform(-3, 3, shape).astype(np.float32)
        model = _make_pool_model(
            "MaxPool",
            shape,
            np.float32,
            dict(kernel_shape=[2, 2], strides=[2, 2]),
            with_indices=True,
        )
        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=0, rtol=0)

    @pytest.mark.parametrize("dtype", [np.float32, np.float16], ids=["fp32", "fp16"])
    def test_maxpool_2x2_vec_nonfinite_windows(self, model_runner, dtype):
        """An all-NaN or all -inf window gives -FLT_MAX (cast to the data
        type) on the vectorized 2x2 path, as on the other pool kernels."""
        shape = [1, 2, 8, 16]
        rng = np.random.default_rng(64)
        x = rng.uniform(-3, 3, shape).astype(dtype)
        x[0, 0, 0:2, 0:2] = np.nan
        x[0, 0, 0:2, 2:4] = -np.inf
        model = _make_pool_model(
            "MaxPool", shape, dtype, dict(kernel_shape=[2, 2], strides=[2, 2])
        )
        actual, expected = model_runner.run_sample(model, [x])
        expected = [expected[0].copy()]
        floor = np.array(-np.finfo(np.float32).max, dtype=np.float32).astype(dtype)
        expected[0][0, 0, 0, 0:2] = floor
        np.testing.assert_array_equal(actual[0], expected[0])
