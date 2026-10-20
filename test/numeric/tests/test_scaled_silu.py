#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""com.microsoft ScaledSiLU: y = z * sigmoid(z), z = alpha * x (Qwen3.8)."""

import numpy as np
import pytest
from onnx import TensorProto, helper

from framework.comparator import compare_outputs
from framework.onnx_utils import make_model_from_nodes

_TENSOR_PROTO = {
    np.float16: TensorProto.FLOAT16,
    np.float32: TensorProto.FLOAT,
}


def _make_model(shape, dtype, alpha):
    tp = _TENSOR_PROTO[dtype]
    X = helper.make_tensor_value_info("X", tp, shape)
    Y = helper.make_tensor_value_info("Y", tp, shape)
    node = helper.make_node(
        "ScaledSiLU", ["X"], ["Y"], domain="com.microsoft", alpha=alpha
    )
    return make_model_from_nodes(
        [node], [X], [Y], extra_opsets=[helper.make_opsetid("com.microsoft", 1)]
    )


@pytest.mark.parametrize(
    "model_shape, run_shape, dtype, alpha",
    [
        (["batch", "seq", 320], [1, 1, 320], np.float16, 0.25),
        (["batch", "seq", 320], [2, 37, 320], np.float16, 0.25),
        ([3, 5, 64], [3, 5, 64], np.float32, 0.5),
        ([4, 64], [4, 64], np.float16, 1.0),
    ],
)
def test_scaled_silu(model_runner, model_shape, run_shape, dtype, alpha):
    model = _make_model(model_shape, dtype, alpha)
    rng = np.random.default_rng(31)
    x = rng.uniform(-8, 8, run_shape).astype(dtype)
    actual, expected = model_runner.run_sample(model, [x])
    if dtype == np.float16:
        compare_outputs(actual, expected, atol=1e-3, rtol=1e-3)
    else:
        compare_outputs(actual, expected, atol=1e-6, rtol=1e-5)
