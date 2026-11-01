#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""com.microsoft GatedRMSNorm: RMSNorm over scale-length groups times act(gate)."""

import numpy as np
import pytest
from onnx import TensorProto, helper, numpy_helper

from framework.comparator import compare_outputs
from framework.onnx_utils import make_model_from_nodes

_TENSOR_PROTO = {
    np.float16: TensorProto.FLOAT16,
    np.float32: TensorProto.FLOAT,
}


def _make_model(model_shape, group, dtype, activation):
    tp = _TENSOR_PROTO[dtype]
    X = helper.make_tensor_value_info("X", tp, model_shape)
    Z = helper.make_tensor_value_info("gate", tp, model_shape)
    Y = helper.make_tensor_value_info("Y", tp, model_shape)
    rng = np.random.default_rng(67)
    scale = numpy_helper.from_array(
        rng.uniform(0.5, 1.5, [group]).astype(dtype), name="scale"
    )
    node = helper.make_node(
        "GatedRMSNorm",
        ["X", "scale", "gate"],
        ["Y"],
        domain="com.microsoft",
        epsilon=1e-6,
        activation=activation,
    )
    return make_model_from_nodes(
        [node],
        [X, Z],
        [Y],
        initializers=[scale],
        extra_opsets=[helper.make_opsetid("com.microsoft", 1)],
    )


@pytest.mark.parametrize(
    "model_shape, run_shape, group, dtype, activation",
    [
        # Qwen3.8 GatedDeltaNet output norm: 48 heads x 128.
        (["batch", "seq", 6144], [1, 1, 6144], 128, np.float16, "sigmoid"),
        (["batch", "seq", 6144], [2, 17, 6144], 128, np.float16, "sigmoid"),
        ([3, 5, 512], [3, 5, 512], 128, np.float16, "silu"),
        ([3, 5, 512], [3, 5, 512], 512, np.float32, "swish"),
        ([4, 300], [4, 300], 100, np.float32, "sigmoid"),
    ],
)
def test_gated_rms_norm(model_runner, model_shape, run_shape, group, dtype, activation):
    model = _make_model(model_shape, group, dtype, activation)
    rng = np.random.default_rng(71)
    x = rng.uniform(-2, 2, run_shape).astype(dtype)
    z = rng.uniform(-4, 4, run_shape).astype(dtype)
    actual, expected = model_runner.run_sample(model, [x, z])
    if dtype == np.float16:
        compare_outputs(actual, expected, atol=2e-3, rtol=1e-3)
    else:
        compare_outputs(actual, expected, atol=1e-5, rtol=1e-4)
