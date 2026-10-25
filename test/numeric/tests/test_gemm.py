#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""Check Gemm with dynamic result dimensions and all transpose pairs."""

import numpy as np
import pytest
from onnx import TensorProto, helper

from framework.comparator import compare_outputs
from framework.onnx_utils import make_model_from_nodes


@pytest.mark.parametrize("trans_a,trans_b", [(0, 0), (1, 0), (0, 1), (1, 1)])
def test_gemm_dynamic_result(model_runner, trans_a, trans_b):
    a_shape = [4, 3] if trans_a else [3, 4]
    b_shape = [5, 4] if trans_b else [4, 5]
    a_type = [4, None] if trans_a else [None, 4]
    b_type = [None, 4] if trans_b else [4, None]
    inputs = [
        helper.make_tensor_value_info("A", TensorProto.FLOAT, a_type),
        helper.make_tensor_value_info("B", TensorProto.FLOAT, b_type),
    ]
    output = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [None, None])
    node = helper.make_node(
        "Gemm", ["A", "B"], ["Y"], transA=trans_a, transB=trans_b, beta=0.0
    )
    model = make_model_from_nodes([node], inputs, [output])
    rng = np.random.default_rng(42)
    a = rng.uniform(-1, 1, a_shape).astype(np.float32)
    b = rng.uniform(-1, 1, b_shape).astype(np.float32)
    actual, expected = model_runner.run_sample(model, [a, b], reference="cpu")
    compare_outputs(actual, expected, atol=1e-5)
