#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""MatMulBnb4.

NF4 weights, high nibble first, block absmax, then A @ B_dequant^T.
Requires an ONNX Runtime build that registers com.microsoft.MatMulBnb4.
"""

from __future__ import annotations

import numpy as np
from onnx import TensorProto, helper, numpy_helper

from framework.comparator import compare_outputs
from framework.onnx_utils import make_model_from_nodes


def _matmul_bnb4(a, n, k, block_size, quant_type, rng):
    numel = n * k
    codes = rng.integers(0, 16, size=numel, dtype=np.uint8)
    packed = ((codes[0::2] << 4) | codes[1::2]).astype(np.uint8)
    absmax = rng.uniform(0.25, 1.5, size=numel // block_size).astype(np.float32)
    a_info = helper.make_tensor_value_info("A", TensorProto.FLOAT, list(a.shape))
    y_info = helper.make_tensor_value_info(
        "Y", TensorProto.FLOAT, list(a.shape[:-1]) + [n]
    )
    node = helper.make_node(
        "MatMulBnb4",
        ["A", "B", "absmax"],
        ["Y"],
        domain="com.microsoft",
        K=k,
        N=n,
        block_size=block_size,
        quant_type=quant_type,
        transB=1,
    )
    return make_model_from_nodes(
        [node],
        [a_info],
        [y_info],
        initializers=[
            numpy_helper.from_array(packed, "B"),
            numpy_helper.from_array(absmax, "absmax"),
        ],
        opset=13,
        extra_opsets=[helper.make_opsetid("com.microsoft", 1)],
    )


class TestMatMulBnb4:
    def test_nf4(self, model_runner):
        rng = np.random.default_rng(0)
        a = rng.standard_normal((1, 2, 16)).astype(np.float32)
        model = _matmul_bnb4(a, n=4, k=16, block_size=16, quant_type=1, rng=rng)
        actual, expected = model_runner.run_sample(model, [a])
        compare_outputs(actual, expected, atol=1e-4, rtol=1e-4)

    def test_fp4(self, model_runner):
        rng = np.random.default_rng(1)
        a = rng.standard_normal((2, 16)).astype(np.float32)
        model = _matmul_bnb4(a, n=2, k=16, block_size=16, quant_type=0, rng=rng)
        actual, expected = model_runner.run_sample(model, [a])
        compare_outputs(actual, expected, atol=1e-4, rtol=1e-4)
