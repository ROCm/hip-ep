#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""Tests for com.microsoft GatedDeltaNet (packed QKV, qwen gate, sigmoid beta,
qk_l2_norm=1), with and without cu_seqlens.

ORT CPU has no GatedDeltaNet kernel, so the reference is an equivalent graph:
Split + L2 norm + qwen gate / sigmoid beta from standard ops feeding CPU
LinearAttention (f32, K-major state) with the V-major state transposed. For the
ragged form every request is sliced out and run through that dense graph on its
own state slice; the request boundaries are baked into the reference graph, so
its cu_seqlens input goes unused.
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


def _io_ragged(Hq, Hv, dk, dv, seq_lens, t_onnx):
    P = 2 * Hq * dk + Hv * dv
    T = sum(seq_lens)
    B = len(seq_lens)
    inputs = [
        helper.make_tensor_value_info("query", t_onnx, [T, P]),
        helper.make_tensor_value_info("cu_seqlens", TP.INT32, [B + 1]),
        helper.make_tensor_value_info("decay", TP.FLOAT, [T, Hv]),
        helper.make_tensor_value_info("beta", TP.FLOAT, [T, Hv]),
        helper.make_tensor_value_info("initial_state", TP.FLOAT, [B, Hv, dv, dk]),
    ]
    outputs = [
        helper.make_tensor_value_info("output", t_onnx, [T, Hv, dv]),
        helper.make_tensor_value_info("final_state", TP.FLOAT, [B, Hv, dv, dk]),
    ]
    return inputs, outputs


def _finish(model):
    model.ir_version = 10
    return model


def _gdn_node(cu_name):
    return helper.make_node(
        "GatedDeltaNet",
        [
            "query",
            "",
            "",
            cu_name,
            "decay",
            "beta",
            "initial_state",
            "a_log",
            "dt_bias",
        ],
        ["output", "final_state"],
        domain="com.microsoft",
        update_rule="gated_delta",
        gate_activation="qwen",
        beta_activation="sigmoid",
        qk_l2_norm=1,
        scale=0.0,
        chunk_size=64,
    )


def _make_gdn_model(Hq, Hv, dk, dv, B, L, t_onnx, a_log, dt_bias):
    inputs, outputs = _io(Hq, Hv, dk, dv, B, L, t_onnx)
    inits = [
        numpy_helper.from_array(a_log, "a_log"),
        numpy_helper.from_array(dt_bias, "dt_bias"),
    ]
    return _finish(
        make_model_from_nodes([_gdn_node("")], inputs, outputs, inits, **_OPSETS)
    )


def _make_ragged_gdn_model(Hq, Hv, dk, dv, seq_lens, t_onnx, a_log, dt_bias):
    inputs, outputs = _io_ragged(Hq, Hv, dk, dv, seq_lens, t_onnx)
    inits = [
        numpy_helper.from_array(a_log, "a_log"),
        numpy_helper.from_array(dt_bias, "dt_bias"),
    ]
    return _finish(
        make_model_from_nodes(
            [_gdn_node("cu_seqlens")], inputs, outputs, inits, **_OPSETS
        )
    )


def _dense_nodes(p, q_in, decay_in, beta_in, state_in, o_out, s_out, Hq, Hv, t_onnx):
    """Equivalent graph for dense [B, L] inputs. Intermediates carry the prefix
    p; the constant initializers are shared between instances."""

    def n(name):
        return p + name

    nodes = [
        helper.make_node("Cast", [q_in], [n("qkv32")], to=TP.FLOAT),
        helper.make_node(
            "Split", [n("qkv32"), "split"], [n("q"), n("k"), n("v")], axis=-1
        ),
        helper.make_node("Reshape", [n("q"), "qshape"], [n("q4")]),
        helper.make_node("Reshape", [n("k"), "qshape"], [n("k4")]),
    ]
    for x in ("q4", "k4"):
        nodes += [
            helper.make_node(
                "ReduceSumSquare", [n(x), "axis_last"], [n(x + "_ss")], keepdims=1
            ),
            helper.make_node("Add", [n(x + "_ss"), "eps"], [n(x + "_sse")]),
            helper.make_node("Sqrt", [n(x + "_sse")], [n(x + "_n")]),
            helper.make_node("Div", [n(x), n(x + "_n")], [n(x + "_u")]),
            helper.make_node("Reshape", [n(x + "_u"), "flat3"], [n(x + "_l2")]),
        ]
    nodes += [
        helper.make_node("Add", [decay_in, "dt_bias"], [n("d_b")]),
        helper.make_node("Softplus", [n("d_b")], [n("sp")]),
        helper.make_node("Exp", ["a_log"], [n("ea")]),
        helper.make_node("Neg", [n("ea")], [n("nea")]),
        helper.make_node("Mul", [n("nea"), n("sp")], [n("g")]),
        helper.make_node("Sigmoid", [beta_in], [n("bs")]),
        helper.make_node("Transpose", [state_in], [n("la_state")], perm=[0, 1, 3, 2]),
        helper.make_node(
            "LinearAttention",
            [n("q4_l2"), n("k4_l2"), n("v"), n("la_state"), n("g"), n("bs")],
            [n("o3"), n("la_present")],
            domain="com.microsoft",
            q_num_heads=Hq,
            kv_num_heads=Hv,
            update_rule="gated_delta",
        ),
        helper.make_node("Reshape", [n("o3"), "oshape"], [n("o4")]),
        helper.make_node("Cast", [n("o4")], [o_out], to=t_onnx),
        helper.make_node("Transpose", [n("la_present")], [s_out], perm=[0, 1, 3, 2]),
    ]
    return nodes


def _shared_inits(Hq, Hv, dk, dv, a_log, dt_bias):
    return [
        numpy_helper.from_array(a_log, "a_log"),
        numpy_helper.from_array(dt_bias, "dt_bias"),
        helper.make_tensor("split", TP.INT64, [3], [Hq * dk, Hq * dk, Hv * dv]),
        helper.make_tensor("qshape", TP.INT64, [4], [0, 0, Hq, dk]),
        helper.make_tensor("flat3", TP.INT64, [3], [0, 0, Hq * dk]),
        helper.make_tensor("oshape", TP.INT64, [4], [0, 0, Hv, dv]),
        helper.make_tensor("axis_last", TP.INT64, [1], [-1]),
        helper.make_tensor("eps", TP.FLOAT, [], [1e-12]),
    ]


def _make_reference_model(Hq, Hv, dk, dv, B, L, t_onnx, a_log, dt_bias):
    nodes = _dense_nodes(
        "",
        "query",
        "decay",
        "beta",
        "initial_state",
        "output",
        "final_state",
        Hq,
        Hv,
        t_onnx,
    )
    inits = _shared_inits(Hq, Hv, dk, dv, a_log, dt_bias)
    inputs, outputs = _io(Hq, Hv, dk, dv, B, L, t_onnx)
    return _finish(make_model_from_nodes(nodes, inputs, outputs, inits, **_OPSETS))


def _make_ragged_reference_model(Hq, Hv, dk, dv, seq_lens, t_onnx, a_log, dt_bias):
    inits = _shared_inits(Hq, Hv, dk, dv, a_log, dt_bias)
    inits.append(numpy_helper.from_array(np.array([0], dtype=np.int64), "axes0"))

    def const(name, values):
        inits.append(numpy_helper.from_array(np.array(values, dtype=np.int64), name))
        return name

    def slice_rows(src, dst, name, start, end):
        return helper.make_node(
            "Slice",
            [
                src,
                const(f"{name}_s", [start]),
                const(f"{name}_e", [end]),
                const(f"{name}_ax", [0]),
            ],
            [dst],
        )

    nodes, outs, finals = [], [], []
    start = 0
    for b, n in enumerate(seq_lens):
        r = f"r{b}_"
        nodes.append(slice_rows("initial_state", r + "state", r + "st", b, b + 1))
        if n == 0:
            # A request without tokens passes its state through.
            finals.append(r + "state")
            continue
        for src, name in (("query", "q"), ("decay", "d"), ("beta", "be")):
            nodes += [
                slice_rows(src, r + name + "_rows", r + name, start, start + n),
                helper.make_node(
                    "Unsqueeze",
                    [r + name + "_rows", "axes0"],
                    [r + name + "_in"],
                ),
            ]
        nodes += _dense_nodes(
            r,
            r + "q_in",
            r + "d_in",
            r + "be_in",
            r + "state",
            r + "out",
            r + "final",
            Hq,
            Hv,
            t_onnx,
        )
        outs.append(r + "out")
        finals.append(r + "final")
        start += n

    total = sum(seq_lens)
    nodes += [
        helper.make_node("Concat", outs, ["out_cat"], axis=1),
        helper.make_node(
            "Reshape", ["out_cat", const("out_shape", [total, Hv, dv])], ["output"]
        ),
        helper.make_node("Concat", finals, ["final_state"], axis=0),
    ]
    inputs, outputs = _io_ragged(Hq, Hv, dk, dv, seq_lens, t_onnx)
    return _finish(make_model_from_nodes(nodes, inputs, outputs, inits, **_OPSETS))


_CASES = [
    # (Hq, Hv, dk, dv, B, L, dtype, inputs)
    *[(16, 48, 128, 128, 1, L, np.float16, "rand") for L in (1, 7, 64, 130, 512)],
    (16, 48, 128, 128, 2, 40, np.float16, "rand"),
    (16, 48, 128, 128, 1, 512, np.float16, "weak_decay"),
    (16, 48, 128, 128, 1, 64, np.float16, "alternating_keys"),
    (4, 4, 128, 64, 1, 100, np.float16, "rand"),
    (2, 6, 32, 16, 2, 5, np.float16, "rand"),
    (2, 6, 32, 16, 2, 5, np.float32, "rand"),
    (4, 4, 64, 64, 1, 3, np.float32, "rand"),
    (2, 4, 256, 128, 1, 9, np.float16, "rand"),
]


def _tolerances(dtype):
    if dtype == np.float16:
        return (3e-3, 1e-2), (1e-4, 1e-3)
    return (1e-4, 1e-4), (1e-4, 1e-4)


@pytest.mark.parametrize(
    "Hq,Hv,dk,dv,B,L,dtype,inputs",
    _CASES,
    ids=[
        f"hq{c[0]}_hv{c[1]}_dk{c[2]}_dv{c[3]}_b{c[4]}_l{c[5]}_{c[6].__name__}_{c[7]}"
        for c in _CASES
    ],
)
def test_gated_delta_net(model_runner, Hq, Hv, dk, dv, B, L, dtype, inputs):
    t_onnx = TP.FLOAT16 if dtype == np.float16 else TP.FLOAT
    rng = np.random.default_rng(1234 + L)

    def u(*shape):
        return rng.uniform(-1, 1, shape).astype(np.float32)

    query = u(B, L, 2 * Hq * dk + Hv * dv).astype(dtype)
    decay = (-0.05 * (u(B, L, Hv) + 1.2)).astype(np.float32)
    beta = u(B, L, Hv)
    if inputs == "weak_decay":
        # Per-token log decay of about -0.007 to -0.05: the state keeps
        # contributions from the whole sequence.
        decay = (u(B, L, Hv) - 4.0).astype(np.float32)
    elif inputs == "alternating_keys":
        # Each token's key is +-k0 with beta ~ 1 and almost no decay.
        k = query[:, :, Hq * dk : 2 * Hq * dk]
        sign = np.where(np.arange(L) % 2 == 0, 1, -1).astype(dtype)
        k[:] = k[:, :1, :] * sign[None, :, None]
        decay = np.full((B, L, Hv), -20.0, np.float32)
        beta = np.full((B, L, Hv), 10.0, np.float32)
    initial_state = (0.05 * u(B, Hv, dv, dk)).astype(np.float32)
    a_log = (0.1 * u(Hv)).astype(np.float32)
    dt_bias = (0.1 * u(Hv)).astype(np.float32)

    args = (Hq, Hv, dk, dv, B, L, t_onnx, a_log, dt_bias)
    actual, expected = model_runner.run_sample(
        _make_gdn_model(*args),
        [query, decay, beta, initial_state],
        reference_model=_make_reference_model(*args),
    )
    out_tol, state_tol = _tolerances(dtype)
    compare_outputs(actual[:1], expected[:1], atol=out_tol[0], rtol=out_tol[1])
    compare_outputs(actual[1:], expected[1:], atol=state_tol[0], rtol=state_tol[1])


_RAGGED_CASES = [
    # (Hq, Hv, dk, dv, seq_lens, dtype)
    (16, 48, 128, 128, (5, 1, 0, 9), np.float16),
    (16, 48, 128, 128, (130, 7), np.float16),
    (2, 6, 32, 16, (3, 4), np.float32),
]


@pytest.mark.parametrize(
    "Hq,Hv,dk,dv,seq_lens,dtype",
    _RAGGED_CASES,
    ids=[
        f"hq{c[0]}_hv{c[1]}_dk{c[2]}_dv{c[3]}_seq{'_'.join(map(str, c[4]))}_{c[5].__name__}"
        for c in _RAGGED_CASES
    ],
)
def test_gated_delta_net_cu_seqlens(model_runner, Hq, Hv, dk, dv, seq_lens, dtype):
    t_onnx = TP.FLOAT16 if dtype == np.float16 else TP.FLOAT
    total, B = sum(seq_lens), len(seq_lens)
    rng = np.random.default_rng(4321 + total)

    def u(*shape):
        return rng.uniform(-1, 1, shape).astype(np.float32)

    query = u(total, 2 * Hq * dk + Hv * dv).astype(dtype)
    cu = np.concatenate([[0], np.cumsum(seq_lens)]).astype(np.int32)
    decay = (-0.05 * (u(total, Hv) + 1.2)).astype(np.float32)
    beta = u(total, Hv)
    initial_state = (0.05 * u(B, Hv, dv, dk)).astype(np.float32)
    a_log = (0.1 * u(Hv)).astype(np.float32)
    dt_bias = (0.1 * u(Hv)).astype(np.float32)

    args = (Hq, Hv, dk, dv, seq_lens, t_onnx, a_log, dt_bias)
    actual, expected = model_runner.run_sample(
        _make_ragged_gdn_model(*args),
        [query, cu, decay, beta, initial_state],
        reference_model=_make_ragged_reference_model(*args),
    )
    out_tol, state_tol = _tolerances(dtype)
    compare_outputs(actual[:1], expected[:1], atol=out_tol[0], rtol=out_tol[1])
    compare_outputs(actual[1:], expected[1:], atol=state_tol[0], rtol=state_tol[1])
