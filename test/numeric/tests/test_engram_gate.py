#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""com.microsoft EngramGate: similarity gate on the engram value (Qwen3.8)."""

import numpy as np
import pytest
from onnx import TensorProto, helper, numpy_helper

from framework.comparator import compare_outputs
from framework.onnx_utils import make_model_from_nodes

_TENSOR_PROTO = {
    np.float16: TensorProto.FLOAT16,
    np.float32: TensorProto.FLOAT,
}


def _make_model(hc, hidden, dtype, normed, zero_query_scale=False):
    tp = _TENSOR_PROTO[dtype]
    kq_shape = ["batch", "seq", hc, hidden]
    K = helper.make_tensor_value_info("key", tp, kq_shape)
    Q = helper.make_tensor_value_info("query", tp, kq_shape)
    V = helper.make_tensor_value_info("value", tp, ["batch", "seq", hidden])
    rng = np.random.default_rng(83)

    def scale(name):
        data = rng.uniform(0.5, 1.5, [hc, hidden]).astype(dtype)
        if zero_query_scale and name == "query_scale":
            data[:] = 0
        return numpy_helper.from_array(data, name=name)

    inits = [scale("key_scale"), scale("query_scale")]
    inputs = ["key", "query", "value", "key_scale", "query_scale"]
    outputs = [helper.make_tensor_value_info("out", tp, kq_shape)]
    node_outputs = ["out"]
    if normed:
        inits.append(scale("conv_scale"))
        inputs.append("conv_scale")
        outputs.append(helper.make_tensor_value_info("normed", tp, kq_shape))
        node_outputs.append("normed")
    node = helper.make_node(
        "EngramGate", inputs, node_outputs, domain="com.microsoft", epsilon=1e-6
    )
    return make_model_from_nodes(
        [node],
        [K, Q, V],
        outputs,
        initializers=inits,
        extra_opsets=[helper.make_opsetid("com.microsoft", 1)],
    )


@pytest.mark.parametrize(
    "batch, seq, hc, hidden, dtype, normed, zero_query_scale",
    [
        # Qwen3.8 layer-1 engram injection: 4 hyper-connection branches of 2560.
        (1, 1, 4, 2560, np.float16, True, False),
        (2, 9, 4, 2560, np.float16, True, False),
        (1, 3, 2, 96, np.float32, False, False),
        # A zero dot product gives a gate of exactly 0.5.
        (1, 2, 2, 64, np.float32, True, True),
    ],
)
def test_engram_gate(
    model_runner, batch, seq, hc, hidden, dtype, normed, zero_query_scale
):
    model = _make_model(hc, hidden, dtype, normed, zero_query_scale)
    rng = np.random.default_rng(89)
    key = rng.uniform(-2, 2, [batch, seq, hc, hidden]).astype(dtype)
    query = rng.uniform(-2, 2, [batch, seq, hc, hidden]).astype(dtype)
    value = rng.uniform(-2, 2, [batch, seq, hidden]).astype(dtype)
    actual, expected = model_runner.run_sample(model, [key, query, value])
    if dtype == np.float16:
        compare_outputs(actual, expected, atol=2e-3, rtol=1e-3)
    else:
        compare_outputs(actual, expected, atol=1e-5, rtol=1e-4)
