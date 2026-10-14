#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""Numeric checks for binary Einsum lowered to batched MatMul."""

import numpy as np
import pytest
from onnx import TensorProto, helper

from framework.comparator import compare_outputs
from framework.onnx_utils import make_model_from_nodes


def _make_einsum_model(equation, a_shape, b_shape, y_shape, dtype):
    tp = {np.float16: TensorProto.FLOAT16, np.float32: TensorProto.FLOAT}[dtype]
    a = helper.make_tensor_value_info("A", tp, a_shape)
    b = helper.make_tensor_value_info("B", tp, b_shape)
    y = helper.make_tensor_value_info("Y", tp, y_shape)
    node = helper.make_node("Einsum", ["A", "B"], ["Y"], equation=equation)
    return make_model_from_nodes([node], [a, b], [y])


CASES = [
    ("bhwc,hkc->bhwk", [2, 3, 4, 8], [3, 5, 8], [2, 3, 4, 5]),
    ("bhwc,wkc->bhwk", [2, 3, 4, 8], [4, 5, 8], [2, 3, 4, 5]),
    ("ij,jk->ik", [4, 8], [8, 6], [4, 6]),
]


class TestEinsum:
    @pytest.mark.parametrize("equation,a_shape,b_shape,y_shape", CASES)
    def test_einsum_f32(self, model_runner, equation, a_shape, b_shape, y_shape):
        model = _make_einsum_model(equation, a_shape, b_shape, y_shape, np.float32)
        rng = np.random.default_rng(11)
        a = rng.uniform(-1, 1, a_shape).astype(np.float32)
        b = rng.uniform(-1, 1, b_shape).astype(np.float32)
        actual, expected = model_runner.run_sample(model, [a, b])
        compare_outputs(actual, expected, atol=1e-4, rtol=1e-4)

    @pytest.mark.parametrize("equation,a_shape,b_shape,y_shape", CASES)
    def test_einsum_f16(self, model_runner, equation, a_shape, b_shape, y_shape):
        model = _make_einsum_model(equation, a_shape, b_shape, y_shape, np.float16)
        rng = np.random.default_rng(12)
        a = rng.uniform(-1, 1, a_shape).astype(np.float16)
        b = rng.uniform(-1, 1, b_shape).astype(np.float16)
        actual, expected = model_runner.run_sample(model, [a, b])
        compare_outputs(actual, expected, atol=2e-2, rtol=1e-2)
