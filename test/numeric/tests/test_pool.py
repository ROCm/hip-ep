#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""Tests for the ONNX AveragePool / MaxPool operators.

The cases cover the three pool kernels by shape:

* 2x2 / 3x3 windows: the small-window kernel. 2x2 stride-2 pools that tile
  the input exactly take its vectorized path (fp16 needs W % 8 == 0, fp32
  W % 4 == 0); unaligned widths, odd heights, ceil mode and padding take the
  generic path.
* Windows of 64 or more taps: the wide kernel, 32 lanes per output.
* Everything else, and MaxPool with argmax indices: the one-thread-per-output
  kernel.
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
]
WIDE_IDS = ["avg32x32", "avg9x9_pad", "max8x8_pad"]


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
        compare_outputs(actual, expected, atol=0)
