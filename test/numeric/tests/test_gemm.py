#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""ONNX Gemm numeric coverage for the custom dense runtime entry."""

import numpy as np
import pytest
from onnx import TensorProto, helper, numpy_helper

from framework.comparator import compare_outputs
from framework.onnx_utils import make_model_from_nodes


@pytest.mark.parametrize("dtype", [np.float16, np.float32])
@pytest.mark.parametrize("trans_a,trans_b", [(0, 0), (0, 1), (1, 0), (1, 1)])
@pytest.mark.parametrize("bias", ["none", "scalar", "row", "column", "matrix"])
def test_gemm_transpose_bias(model_runner, dtype, trans_a, trans_b, bias):
    """Alpha/beta and every legal C broadcast survive both transpose flags."""
    m, n, k = 19, 21, 37
    a_shape = [k, m] if trans_a else [m, k]
    b_shape = [n, k] if trans_b else [k, n]
    elem_type = TensorProto.FLOAT16 if dtype == np.float16 else TensorProto.FLOAT
    rng = np.random.default_rng(519)
    a = rng.uniform(-0.25, 0.25, a_shape).astype(dtype)
    b = rng.uniform(-0.25, 0.25, b_shape).astype(dtype)
    inputs = ["A", "B"]
    initializers = [numpy_helper.from_array(b, name="B")]
    if bias != "none":
        c_shape = {"scalar": [], "row": [n], "column": [m, 1], "matrix": [m, n]}[bias]
        c = np.asarray(rng.uniform(-0.25, 0.25, c_shape), dtype=dtype)
        initializers.append(numpy_helper.from_array(c, name="C"))
        inputs.append("C")
    node = helper.make_node(
        "Gemm", inputs, ["Y"], transA=trans_a, transB=trans_b, alpha=0.75, beta=0.25
    )
    model = make_model_from_nodes(
        [node],
        [helper.make_tensor_value_info("A", elem_type, a_shape)],
        [helper.make_tensor_value_info("Y", elem_type, [m, n])],
        initializers=initializers,
    )
    actual, expected = model_runner.run_sample(model, [a], reference="cpu")
    compare_outputs(actual, expected, atol=1e-3 if dtype == np.float16 else 1e-5)
