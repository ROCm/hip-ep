#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""Tests for com.microsoft GatedDeltaNet (packed QKV, qwen gate, sigmoid beta,
qk_l2_norm=1).

ORT CPU has no GatedDeltaNet kernel, so the reference is an equivalent graph:
Split + L2 norm + qwen gate / sigmoid beta from standard ops feeding CPU
LinearAttention (f32, K-major state) with the V-major state transposed.
"""

import numpy as np
import pytest
from onnx import TensorProto as TP
from onnx import helper, numpy_helper

from framework.comparator import compare_outputs
from framework.onnx_utils import make_model_from_nodes

_OPSETS = dict(opset=22, extra_opsets=[helper.make_opsetid("com.microsoft", 1)])


def _io(Hq, Hv, dk, dv, B, L, t_onnx):
    P = 2 * Hq * dk + Hv * dv
    inputs = [
        helper.make_tensor_value_info("query", t_onnx, [B, L, P]),
        helper.make_tensor_value_info("decay", TP.FLOAT, [B, L, Hv]),
        helper.make_tensor_value_info("beta", TP.FLOAT, [B, L, Hv]),
        helper.make_tensor_value_info("initial_state", TP.FLOAT, [B, Hv, dv, dk]),
    ]
    outputs = [
        helper.make_tensor_value_info("output", t_onnx, [B, L, Hv, dv]),
        helper.make_tensor_value_info("final_state", TP.FLOAT, [B, Hv, dv, dk]),
    ]
    return inputs, outputs


def _finish(model):
    model.ir_version = 10
    return model


def _make_gdn_model(Hq, Hv, dk, dv, B, L, t_onnx, a_log, dt_bias):
    node = helper.make_node(
        "GatedDeltaNet",
        ["query", "", "", "", "decay", "beta", "initial_state", "a_log", "dt_bias"],
        ["output", "final_state"],
        domain="com.microsoft",
        update_rule="gated_delta",
        gate_activation="qwen",
        beta_activation="sigmoid",
        qk_l2_norm=1,
        scale=0.0,
        chunk_size=64,
    )
    inputs, outputs = _io(Hq, Hv, dk, dv, B, L, t_onnx)
    inits = [
        numpy_helper.from_array(a_log, "a_log"),
        numpy_helper.from_array(dt_bias, "dt_bias"),
    ]
    return _finish(make_model_from_nodes([node], inputs, outputs, inits, **_OPSETS))


def _make_reference_model(Hq, Hv, dk, dv, B, L, t_onnx, a_log, dt_bias):
    nodes = [
        helper.make_node("Cast", ["query"], ["qkv32"], to=TP.FLOAT),
        helper.make_node("Split", ["qkv32", "split"], ["q", "k", "v"], axis=-1),
        helper.make_node("Reshape", ["q", "qshape"], ["q4"]),
        helper.make_node("Reshape", ["k", "qshape"], ["k4"]),
    ]
    for x in ("q4", "k4"):
        nodes += [
            helper.make_node(
                "ReduceSumSquare", [x, "axis_last"], [x + "_ss"], keepdims=1
            ),
            helper.make_node("Add", [x + "_ss", "eps"], [x + "_sse"]),
            helper.make_node("Sqrt", [x + "_sse"], [x + "_n"]),
            helper.make_node("Div", [x, x + "_n"], [x + "_u"]),
            helper.make_node("Reshape", [x + "_u", "flat3"], [x + "_l2"]),
        ]
    nodes += [
        helper.make_node("Add", ["decay", "dt_bias"], ["d_b"]),
        helper.make_node("Softplus", ["d_b"], ["sp"]),
        helper.make_node("Exp", ["a_log"], ["ea"]),
        helper.make_node("Neg", ["ea"], ["nea"]),
        helper.make_node("Mul", ["nea", "sp"], ["g"]),
        helper.make_node("Sigmoid", ["beta"], ["bs"]),
        helper.make_node(
            "Transpose", ["initial_state"], ["la_state"], perm=[0, 1, 3, 2]
        ),
        helper.make_node(
            "LinearAttention",
            ["q4_l2", "k4_l2", "v", "la_state", "g", "bs"],
            ["o3", "la_present"],
            domain="com.microsoft",
            q_num_heads=Hq,
            kv_num_heads=Hv,
            update_rule="gated_delta",
        ),
        helper.make_node("Reshape", ["o3", "oshape"], ["o4"]),
        helper.make_node("Cast", ["o4"], ["output"], to=t_onnx),
        helper.make_node(
            "Transpose", ["la_present"], ["final_state"], perm=[0, 1, 3, 2]
        ),
    ]
    inits = [
        numpy_helper.from_array(a_log, "a_log"),
        numpy_helper.from_array(dt_bias, "dt_bias"),
        helper.make_tensor("split", TP.INT64, [3], [Hq * dk, Hq * dk, Hv * dv]),
        helper.make_tensor("qshape", TP.INT64, [4], [0, 0, Hq, dk]),
        helper.make_tensor("flat3", TP.INT64, [3], [0, 0, Hq * dk]),
        helper.make_tensor("oshape", TP.INT64, [4], [0, 0, Hv, dv]),
        helper.make_tensor("axis_last", TP.INT64, [1], [-1]),
        helper.make_tensor("eps", TP.FLOAT, [], [1e-12]),
    ]
    inputs, outputs = _io(Hq, Hv, dk, dv, B, L, t_onnx)
    return _finish(make_model_from_nodes(nodes, inputs, outputs, inits, **_OPSETS))


_CASES = [
    # (Hq, Hv, dk, dv, B, L, dtype)
    *[(16, 48, 128, 128, 1, L, np.float16) for L in (1, 7, 64)],
    (2, 6, 32, 16, 2, 5, np.float16),
    (2, 6, 32, 16, 2, 5, np.float32),
    (4, 4, 64, 64, 1, 3, np.float32),
    (2, 4, 256, 128, 1, 9, np.float16),
]


@pytest.mark.parametrize(
    "Hq,Hv,dk,dv,B,L,dtype",
    _CASES,
    ids=[
        f"hq{c[0]}_hv{c[1]}_dk{c[2]}_dv{c[3]}_b{c[4]}_l{c[5]}_{c[6].__name__}"
        for c in _CASES
    ],
)
def test_gated_delta_net(model_runner, Hq, Hv, dk, dv, B, L, dtype):
    t_onnx = TP.FLOAT16 if dtype == np.float16 else TP.FLOAT
    rng = np.random.default_rng(1234 + L)

    def u(*shape):
        return rng.uniform(-1, 1, shape).astype(np.float32)

    query = u(B, L, 2 * Hq * dk + Hv * dv).astype(dtype)
    decay = (-0.05 * (u(B, L, Hv) + 1.2)).astype(np.float32)
    beta = u(B, L, Hv)
    initial_state = (0.05 * u(B, Hv, dv, dk)).astype(np.float32)
    a_log = (0.1 * u(Hv)).astype(np.float32)
    dt_bias = (0.1 * u(Hv)).astype(np.float32)

    args = (Hq, Hv, dk, dv, B, L, t_onnx, a_log, dt_bias)
    actual, expected = model_runner.run_sample(
        _make_gdn_model(*args),
        [query, decay, beta, initial_state],
        reference_model=_make_reference_model(*args),
    )
    if dtype == np.float16:
        out_tol, state_tol = (3e-3, 1e-2), (1e-4, 1e-3)
    else:
        out_tol = state_tol = (1e-4, 1e-4)
    compare_outputs(actual[:1], expected[:1], atol=out_tol[0], rtol=out_tol[1])
    compare_outputs(actual[1:], expected[1:], atol=state_tol[0], rtol=state_tol[1])
