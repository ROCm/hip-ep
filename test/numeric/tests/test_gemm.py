#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""Numeric verification for the ONNX Gemm op with a broadcast C.

Y = alpha * A' * B' + beta * C, where C is unidirectionally broadcast to
[M, N]. A C that is not [M, N] is first copied into the output on the device,
and hipBLASLt then applies beta to it in place. A, B and C are runtime inputs
so nothing constant-folds.
"""

import numpy as np
import pytest
from onnx import TensorProto, helper

from framework.comparator import compare_outputs
from framework.onnx_utils import make_model_from_nodes

M, K, N = 6, 32, 10


def _make_gemm_model(c_shape: list[int], alpha: float, beta: float, trans_b: int):
    b_shape = [N, K] if trans_b else [K, N]
    A = helper.make_tensor_value_info("A", TensorProto.FLOAT, [M, K])
    B = helper.make_tensor_value_info("B", TensorProto.FLOAT, b_shape)
    C = helper.make_tensor_value_info("C", TensorProto.FLOAT, c_shape)
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [M, N])
    node = helper.make_node(
        "Gemm", ["A", "B", "C"], ["Y"], alpha=alpha, beta=beta, transB=trans_b
    )
    return make_model_from_nodes([node], [A, B, C], [Y])


class TestGemm:
    @pytest.mark.parametrize(
        "c_shape",
        [[N], [1, N], [M, 1], [1, 1], [M, N]],
        ids=["row", "row2d", "col", "scalar", "full"],
    )
    @pytest.mark.parametrize("beta", [1.0, 0.5, -2.0, 0.0])
    def test_gemm_broadcast_c(self, model_runner, c_shape, beta):
        model = _make_gemm_model(c_shape, alpha=0.75, beta=beta, trans_b=1)
        rng = np.random.default_rng(901)
        a = rng.uniform(-1, 1, [M, K]).astype(np.float32)
        b = rng.uniform(-1, 1, [N, K]).astype(np.float32)
        c = rng.uniform(-4, 4, c_shape).astype(np.float32)
        actual, expected = model_runner.run_sample(model, [a, b, c], reference="cpu")
        compare_outputs(actual, expected, atol=1e-3)
