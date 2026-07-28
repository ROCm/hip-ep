#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""Numeric coverage for transpose-aware dynamic Gemm result dimensions."""

import numpy as np
import pytest
from onnx import TensorProto, helper

from framework.comparator import compare_outputs
from framework.onnx_utils import make_model_from_nodes

M = 3
K = 4
N = 5


def _make_gemm_model(trans_a: int, trans_b: int):
    a_shape = [K, "M"] if trans_a else ["M", K]
    b_shape = ["N", K] if trans_b else [K, "N"]
    a = helper.make_tensor_value_info("A", TensorProto.FLOAT16, a_shape)
    b = helper.make_tensor_value_info("B", TensorProto.FLOAT16, b_shape)
    c = helper.make_tensor_value_info("C", TensorProto.FLOAT16, ["N"])
    output = helper.make_tensor_value_info("Y", TensorProto.FLOAT16, ["M", "N"])
    node = helper.make_node(
        "Gemm",
        ["A", "B", "C"],
        ["Y"],
        transA=trans_a,
        transB=trans_b,
    )
    return make_model_from_nodes([node], [a, b, c], [output])


class TestGemm:
    @pytest.mark.parametrize(
        "trans_a,trans_b",
        [(0, 0), (1, 0), (0, 1), (1, 1)],
    )
    def test_dynamic_m_n_all_transposes(self, model_runner, trans_a, trans_b):
        model = _make_gemm_model(trans_a, trans_b)
        rng = np.random.default_rng(44)
        a_shape = (K, M) if trans_a else (M, K)
        b_shape = (N, K) if trans_b else (K, N)
        inputs = [
            rng.uniform(-0.5, 0.5, a_shape).astype(np.float16),
            rng.uniform(-0.5, 0.5, b_shape).astype(np.float16),
            rng.uniform(-0.5, 0.5, (N,)).astype(np.float16),
        ]

        actual, expected = model_runner.run_sample(model, inputs, reference="cpu")
        compare_outputs(actual, expected, atol=1e-3)
