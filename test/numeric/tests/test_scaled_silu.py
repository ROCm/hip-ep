#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""com.microsoft ScaledSiLU: y = z * sigmoid(z), z = alpha * x (Qwen3.8)."""

import numpy as np
import pytest
from onnx import TensorProto, helper, numpy_helper

from framework.comparator import compare_outputs
from framework.onnx_utils import make_model_from_nodes

_TENSOR_PROTO = {
    np.float16: TensorProto.FLOAT16,
    np.float32: TensorProto.FLOAT,
}


def _make_model(shape, dtype, alpha, scale):
    tp = _TENSOR_PROTO[dtype]
    X = helper.make_tensor_value_info("X", tp, shape)
    Y = helper.make_tensor_value_info("Y", tp, shape)
    inputs = ["X"]
    inits = []
    if scale is not None:
        inputs.append("scale")
        inits.append(numpy_helper.from_array(np.array(scale, dtype=dtype), "scale"))
    node = helper.make_node(
        "ScaledSiLU", inputs, ["Y"], domain="com.microsoft", alpha=alpha
    )
    return make_model_from_nodes(
        [node],
        [X],
        [Y],
        initializers=inits,
        extra_opsets=[helper.make_opsetid("com.microsoft", 1)],
    )


@pytest.mark.parametrize(
    "model_shape, run_shape, dtype, alpha, scale",
    [
        (["batch", "seq", 320], [1, 1, 320], np.float16, 0.25, None),
        (["batch", "seq", 320], [2, 37, 320], np.float16, 0.25, None),
        ([3, 5, 64], [3, 5, 64], np.float32, 0.5, None),
        ([4, 64], [4, 64], np.float16, 1.0, None),
        # The scale input overrides alpha.
        ([4, 64], [4, 64], np.float16, 1.0, 0.125),
    ],
)
def test_scaled_silu(model_runner, model_shape, run_shape, dtype, alpha, scale):
    model = _make_model(model_shape, dtype, alpha, scale)
    rng = np.random.default_rng(31)
    x = rng.uniform(-8, 8, run_shape).astype(dtype)
    actual, expected = model_runner.run_sample(model, [x])
    if dtype == np.float16:
        compare_outputs(actual, expected, atol=1e-3, rtol=1e-3)
    else:
        compare_outputs(actual, expected, atol=1e-6, rtol=1e-5)
