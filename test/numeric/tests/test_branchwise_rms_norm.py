#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""com.microsoft BranchwiseRMSNorm: per-branch RMSNorm of C branches of H."""

import numpy as np
import pytest
from onnx import TensorProto, helper, numpy_helper

from framework.comparator import compare_outputs
from framework.onnx_utils import make_model_from_nodes

_TENSOR_PROTO = {
    np.float16: TensorProto.FLOAT16,
    np.float32: TensorProto.FLOAT,
}


def _make_model(model_shape, scale_shape, dtype, num_branches):
    tp = _TENSOR_PROTO[dtype]
    X = helper.make_tensor_value_info("X", tp, model_shape)
    Y = helper.make_tensor_value_info("Y", tp, model_shape)
    rng = np.random.default_rng(53)
    scale = numpy_helper.from_array(
        rng.uniform(0.5, 1.5, scale_shape).astype(dtype), name="scale"
    )
    node = helper.make_node(
        "BranchwiseRMSNorm",
        ["X", "scale"],
        ["Y"],
        domain="com.microsoft",
        epsilon=1e-6,
        num_branches=num_branches,
    )
    return make_model_from_nodes(
        [node],
        [X],
        [Y],
        initializers=[scale],
        extra_opsets=[helper.make_opsetid("com.microsoft", 1)],
    )


@pytest.mark.parametrize(
    "model_shape, run_shape, scale_shape, dtype, num_branches",
    [
        # Qwen3.8 hyper-connection norm: 4 branches of 2560, per-branch gain.
        (["batch", "seq", 10240], [1, 1, 10240], [10240], np.float16, 4),
        (["batch", "seq", 10240], [2, 13, 10240], [10240], np.float16, 4),
        ([2, 3, 4 * 96], [2, 3, 4 * 96], [4, 96], np.float32, 4),
        # One gain shared by every branch.
        ([2, 5, 4 * 64], [2, 5, 4 * 64], [64], np.float16, 4),
        # Grouped layout (..., C, H).
        ([2, 3, 4, 96], [2, 3, 4, 96], [4, 96], np.float32, 0),
    ],
)
def test_branchwise_rms_norm(
    model_runner, model_shape, run_shape, scale_shape, dtype, num_branches
):
    model = _make_model(model_shape, scale_shape, dtype, num_branches)
    rng = np.random.default_rng(59)
    x = rng.uniform(-2, 2, run_shape).astype(dtype)
    actual, expected = model_runner.run_sample(model, [x])
    if dtype == np.float16:
        compare_outputs(actual, expected, atol=2e-3, rtol=1e-3)
    else:
        compare_outputs(actual, expected, atol=1e-5, rtol=1e-4)
