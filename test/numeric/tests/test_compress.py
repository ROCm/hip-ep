#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""Tests for the ONNX Compress op (axis mode).

output = input selected along `axis` at positions where condition[k] is True.
The static-shape EP sizes the output from the model's declared output extent,
so each test supplies a `condition` whose number of True entries EXACTLY
matches the declared output size along the axis -- otherwise the EP's static
output and ORT CPU's dynamically-sized output would differ in shape. The
selection is a pure copy, so results are bit-exact (atol=0).

`condition` is a runtime input (mirrors the e2e LIT form) so the compress
kernel is genuinely exercised rather than constant-folded away.
"""

from __future__ import annotations

import numpy as np
from onnx import helper

from framework.comparator import compare_outputs
from framework.onnx_utils import make_model_from_nodes, np_to_onnx_type


def _make_compress_model(
    in_shape: list[int], cond_len: int, axis: int, out_shape: list[int]
):
    inp = helper.make_tensor_value_info(
        "input", np_to_onnx_type(np.float32), list(in_shape)
    )
    cond = helper.make_tensor_value_info(
        "condition", np_to_onnx_type(np.bool_), [cond_len]
    )
    out = helper.make_tensor_value_info(
        "output", np_to_onnx_type(np.float32), list(out_shape)
    )
    node = helper.make_node("Compress", ["input", "condition"], ["output"], axis=axis)
    return make_model_from_nodes([node], [inp, cond], [out])


class TestCompress:
    def test_compress_axis0(self, model_runner):
        in_shape = [3, 2]
        axis = 0
        condition = np.array([True, False, True], dtype=np.bool_)
        out_shape = [int(condition.sum()), 2]
        model = _make_compress_model(in_shape, condition.shape[0], axis, out_shape)
        rng = np.random.default_rng(701)
        x = rng.uniform(-5.0, 5.0, in_shape).astype(np.float32)
        actual, expected = model_runner.run_sample(model, [x, condition])
        compare_outputs(actual, expected, atol=0)

    def test_compress_axis1(self, model_runner):
        in_shape = [2, 4]
        axis = 1
        condition = np.array([False, True, True, False], dtype=np.bool_)
        out_shape = [2, int(condition.sum())]
        model = _make_compress_model(in_shape, condition.shape[0], axis, out_shape)
        rng = np.random.default_rng(702)
        x = rng.uniform(-5.0, 5.0, in_shape).astype(np.float32)
        actual, expected = model_runner.run_sample(model, [x, condition])
        compare_outputs(actual, expected, atol=0)
