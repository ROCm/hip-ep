#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""com.microsoft GatedAdd: out = X + round_T(Y * gate), gate is (..., 1)."""

import numpy as np
import pytest
from onnx import TensorProto, helper

from framework.comparator import compare_outputs
from framework.onnx_utils import make_model_from_nodes

_TENSOR_PROTO = {
    np.float16: TensorProto.FLOAT16,
    np.float32: TensorProto.FLOAT,
}


def _make_model(model_shape, dtype):
    tp = _TENSOR_PROTO[dtype]
    gate_shape = list(model_shape[:-1]) + [1]
    X = helper.make_tensor_value_info("X", tp, model_shape)
    Yin = helper.make_tensor_value_info("Y", tp, model_shape)
    G = helper.make_tensor_value_info("gate", tp, gate_shape)
    out = helper.make_tensor_value_info("out", tp, model_shape)
    node = helper.make_node(
        "GatedAdd", ["X", "Y", "gate"], ["out"], domain="com.microsoft"
    )
    return make_model_from_nodes(
        [node],
        [X, Yin, G],
        [out],
        extra_opsets=[helper.make_opsetid("com.microsoft", 1)],
    )


@pytest.mark.parametrize(
    "model_shape, run_shape, dtype",
    [
        (["batch", "seq", 2560], [1, 1, 2560], np.float16),
        (["batch", "seq", 2560], [2, 19, 2560], np.float16),
        ([3, 7, 96], [3, 7, 96], np.float32),
    ],
)
def test_gated_add(model_runner, model_shape, run_shape, dtype):
    model = _make_model(model_shape, dtype)
    rng = np.random.default_rng(41)
    x = rng.uniform(-4, 4, run_shape).astype(dtype)
    y = rng.uniform(-4, 4, run_shape).astype(dtype)
    gate = rng.uniform(0, 1, list(run_shape[:-1]) + [1]).astype(dtype)
    actual, expected = model_runner.run_sample(model, [x, y, gate])
    # Mul then Add, each rounded to T, is the contract itself.
    compare_outputs(actual, expected, atol=0.0, rtol=0.0)
