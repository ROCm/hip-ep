#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""Tests for com.microsoft VarlenCausalConvWithState.

The op convolves a packed, token-major batch of variable-length sequences,
each carrying its own committed state of (kernel_size - 1) * dilation samples.

ORT CPU has no VarlenCausalConvWithState kernel, so the reference is an
equivalent graph: every sequence is sliced out, prefixed with its state and run
through a depthwise Conv, and the last state_len samples of that virtual
sequence become its final state. The sequence boundaries are baked into the
reference graph, so its cumulative_sequence_length input goes unused.
"""

import numpy as np
import pytest
from onnx import TensorProto as TP
from onnx import helper, numpy_helper

from framework.comparator import compare_outputs
from framework.onnx_utils import make_model_from_nodes

_OPSETS = dict(opset=22, extra_opsets=[helper.make_opsetid("com.microsoft", 1)])
_TP = {np.float16: TP.FLOAT16, np.float32: TP.FLOAT}
_TOL = {np.float16: (2e-3, 1e-2), np.float32: (1e-4, 1e-4)}

CHANNELS = 64
KERNEL = 4


def _io(seq_lens, state_len, tp):
    total = sum(seq_lens)
    batch = len(seq_lens)
    inputs = [
        helper.make_tensor_value_info("input", tp, [total, CHANNELS]),
        helper.make_tensor_value_info(
            "cumulative_sequence_length", TP.INT32, [batch + 1]
        ),
        helper.make_tensor_value_info(
            "initial_state", tp, [batch, CHANNELS, state_len]
        ),
    ]
    outputs = [
        helper.make_tensor_value_info("output", tp, [total, CHANNELS]),
        helper.make_tensor_value_info("final_state", tp, [batch, CHANNELS, state_len]),
    ]
    return inputs, outputs


def _make_model(seq_lens, dilation, activation, weight, bias, tp):
    state_len = (KERNEL - 1) * dilation
    node = helper.make_node(
        "VarlenCausalConvWithState",
        [
            "input",
            "weight",
            "cumulative_sequence_length",
            "bias" if bias is not None else "",
            "initial_state",
        ],
        ["output", "final_state"],
        domain="com.microsoft",
        activation=activation,
        dilation=dilation,
    )
    inits = [numpy_helper.from_array(weight, "weight")]
    if bias is not None:
        inits.append(numpy_helper.from_array(bias, "bias"))
    inputs, outputs = _io(seq_lens, state_len, tp)
    return make_model_from_nodes([node], inputs, outputs, inits, **_OPSETS)


def _make_reference_model(seq_lens, dilation, activation, weight, bias, tp):
    state_len = (KERNEL - 1) * dilation
    inits = [numpy_helper.from_array(weight.astype(np.float32), "weight32")]
    conv_inputs = ["seq", "weight32"]
    if bias is not None:
        inits.append(numpy_helper.from_array(bias.astype(np.float32), "bias32"))
        conv_inputs.append("bias32")

    def const(name, values):
        inits.append(numpy_helper.from_array(np.array(values, dtype=np.int64), name))
        return name

    nodes = [
        helper.make_node("Cast", ["input"], ["x32"], to=TP.FLOAT),
        helper.make_node("Cast", ["initial_state"], ["state32"], to=TP.FLOAT),
    ]
    outs, finals = [], []
    start = 0
    for b, n in enumerate(seq_lens):
        s = f"s{b}"
        nodes += [
            helper.make_node(
                "Slice",
                [
                    "x32",
                    const(f"{s}_xs", [start]),
                    const(f"{s}_xe", [start + n]),
                    const(f"{s}_ax", [0]),
                ],
                [f"{s}_x"],
            ),
            helper.make_node("Transpose", [f"{s}_x"], [f"{s}_xt"], perm=[1, 0]),
            helper.make_node(
                "Reshape",
                [f"{s}_xt", const(f"{s}_xshape", [1, CHANNELS, n])],
                [f"{s}_x3"],
            ),
            helper.make_node(
                "Slice",
                [
                    "state32",
                    const(f"{s}_ss", [b]),
                    const(f"{s}_se", [b + 1]),
                    const(f"{s}_sax", [0]),
                ],
                [f"{s}_state"],
            ),
            helper.make_node("Concat", [f"{s}_state", f"{s}_x3"], [f"{s}_seq"], axis=2),
        ]
        conv_in = [f"{s}_seq"] + conv_inputs[1:]
        nodes.append(
            helper.make_node(
                "Conv",
                conv_in,
                [f"{s}_conv"],
                group=CHANNELS,
                kernel_shape=[KERNEL],
                dilations=[dilation],
            )
        )
        act = f"{s}_conv"
        if activation != "none":
            nodes += [
                helper.make_node("Sigmoid", [f"{s}_conv"], [f"{s}_sig"]),
                helper.make_node("Mul", [f"{s}_conv", f"{s}_sig"], [f"{s}_act"]),
            ]
            act = f"{s}_act"
        nodes += [
            helper.make_node(
                "Reshape", [act, const(f"{s}_yshape", [CHANNELS, n])], [f"{s}_y"]
            ),
            helper.make_node("Transpose", [f"{s}_y"], [f"{s}_yt"], perm=[1, 0]),
            helper.make_node(
                "Slice",
                [
                    f"{s}_seq",
                    const(f"{s}_fs", [n]),
                    const(f"{s}_fe", [n + state_len]),
                    const(f"{s}_fax", [2]),
                ],
                [f"{s}_final"],
            ),
        ]
        outs.append(f"{s}_yt")
        finals.append(f"{s}_final")
        start += n

    nodes += [
        helper.make_node("Concat", outs, ["out32"], axis=0),
        helper.make_node("Concat", finals, ["final32"], axis=0),
        helper.make_node("Cast", ["out32"], ["output"], to=tp),
        helper.make_node("Cast", ["final32"], ["final_state"], to=tp),
    ]
    inputs, outputs = _io(seq_lens, state_len, tp)
    return make_model_from_nodes(nodes, inputs, outputs, inits, **_OPSETS)


class TestVarlenCausalConvWithState:
    @pytest.mark.parametrize(
        "seq_lens,dilation,with_bias,activation,dtype",
        [
            ([5, 3], 1, True, "silu", np.float16),
            # Decode: one token per request.
            ([1, 1, 1, 1], 1, False, "silu", np.float16),
            # Sequences shorter than the state, so most taps come from it.
            ([2, 9], 3, False, "silu", np.float16),
            ([7], 3, True, "none", np.float32),
            ([1, 12, 4], 2, True, "silu", np.float32),
        ],
    )
    def test_varlen_causal_conv(
        self, model_runner, seq_lens, dilation, with_bias, activation, dtype
    ):
        rng = np.random.default_rng(0xC0DE + sum(seq_lens) + dilation)
        state_len = (KERNEL - 1) * dilation
        weight = (rng.standard_normal((CHANNELS, 1, KERNEL)) * 0.1).astype(dtype)
        bias = (
            (rng.standard_normal((CHANNELS,)) * 0.05).astype(dtype)
            if with_bias
            else None
        )
        tp = _TP[dtype]

        x = (rng.standard_normal((sum(seq_lens), CHANNELS)) * 0.5).astype(dtype)
        cumulative = np.concatenate([[0], np.cumsum(seq_lens)]).astype(np.int32)
        state = (
            rng.standard_normal((len(seq_lens), CHANNELS, state_len)) * 0.5
        ).astype(dtype)

        actual, expected = model_runner.run_sample(
            _make_model(seq_lens, dilation, activation, weight, bias, tp),
            [x, cumulative, state],
            reference_model=_make_reference_model(
                seq_lens, dilation, activation, weight, bias, tp
            ),
        )
        atol, rtol = _TOL[dtype]
        compare_outputs(actual, expected, atol=atol, rtol=rtol)
