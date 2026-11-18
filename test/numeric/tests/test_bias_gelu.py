#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""Tests for the fused com.microsoft.BiasGelu and FastGelu ops.

BiasGelu : output = Gelu_erf(data + broadcast(bias))
FastGelu : output = Gelu_tanh(input + broadcast(bias))   (bias optional)

It also covers GELU written out as primitive ops (Mul/Add/Div/Pow/Tanh/Erf),
which the compiler fuses back into one hip.gelu.

`bias` is a 1-D vector broadcast over the last dim of the activation. The
runtime kernels (lib/Runtime/real/{bias_gelu,fast_gelu}.cpp) support
f16/f32/bf16/f64, but ORT's CPU contrib kernels for both ops are
`TypeConstraint("T", float)` only (see contrib_ops/cpu/bert/{bias_gelu,
fast_gelu}.cc), so the CPU-referenced numeric tests here are fp32. The fp16
path is exercised end-to-end by the model perf/accuracy suites.

These tests turn a silent CPU fallback (op fails to compile -> EP declines ->
session-create aborts because the numeric backend sets
`session.disable_cpu_ep_fallback=1`) into a hard failure.
"""

from __future__ import annotations

import numpy as np
import pytest
from onnx import helper, numpy_helper

from framework.comparator import compare_outputs
from framework.onnx_utils import make_model_from_nodes, np_to_onnx_type

MS = helper.make_opsetid("com.microsoft", 1)
SEQ_LENS = [1, 128]
HIDDEN = 768


def _make_gelu_model(op_type: str, shape: list[int], with_bias: bool):
    tp = np_to_onnx_type(np.float32)
    X = helper.make_tensor_value_info("X", tp, list(shape))
    Y = helper.make_tensor_value_info("Y", tp, list(shape))
    inputs = [X]
    node_inputs = ["X"]
    if with_bias:
        bias = helper.make_tensor_value_info("bias", tp, [shape[-1]])
        inputs.append(bias)
        node_inputs.append("bias")
    node = helper.make_node(op_type, node_inputs, ["Y"], domain="com.microsoft")
    return make_model_from_nodes([node], inputs, [Y], extra_opsets=[MS])


class TestBiasGelu:
    @pytest.mark.parametrize("shape", [[2, 16], [4, 8, 32]])
    def test_bias_gelu_smoke(self, model_runner, shape):
        model = _make_gelu_model("BiasGelu", shape, with_bias=True)
        rng = np.random.default_rng(301)
        x = rng.uniform(-3.0, 3.0, shape).astype(np.float32)
        bias = rng.uniform(-1.0, 1.0, [shape[-1]]).astype(np.float32)
        actual, expected = model_runner.run_sample(model, [x, bias])
        compare_outputs(actual, expected, atol=1e-4, rtol=1e-3)

    @pytest.mark.parametrize("seq_len", SEQ_LENS)
    def test_bias_gelu_bert_shape(self, model_runner, seq_len):
        shape = [1, seq_len, HIDDEN]
        model = _make_gelu_model("BiasGelu", shape, with_bias=True)
        rng = np.random.default_rng(302)
        x = rng.uniform(-3.0, 3.0, shape).astype(np.float32)
        bias = rng.uniform(-1.0, 1.0, [HIDDEN]).astype(np.float32)
        actual, expected = model_runner.run_sample(model, [x, bias])
        compare_outputs(actual, expected, atol=1e-4, rtol=1e-3)


class TestMicrosoftGelu:
    """com.microsoft.Gelu is the erf formula, lowered to hip.gelu."""

    @pytest.mark.parametrize("shape", [[2, 16], [4, 8, 32]])
    def test_microsoft_gelu_smoke(self, model_runner, shape):
        model = _make_gelu_model("Gelu", shape, with_bias=False)
        rng = np.random.default_rng(306)
        x = rng.uniform(-3.0, 3.0, shape).astype(np.float32)
        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=1e-4, rtol=1e-3)


class TestFastGelu:
    @pytest.mark.parametrize("shape", [[2, 16], [4, 8, 32]])
    def test_fast_gelu_smoke(self, model_runner, shape):
        model = _make_gelu_model("FastGelu", shape, with_bias=True)
        rng = np.random.default_rng(303)
        x = rng.uniform(-3.0, 3.0, shape).astype(np.float32)
        bias = rng.uniform(-1.0, 1.0, [shape[-1]]).astype(np.float32)
        actual, expected = model_runner.run_sample(model, [x, bias])
        compare_outputs(actual, expected, atol=1e-4, rtol=1e-3)

    def test_fast_gelu_no_bias(self, model_runner):
        """FastGelu bias is optional; bias_len==0 in the runtime path."""
        shape = [4, 8, 32]
        model = _make_gelu_model("FastGelu", shape, with_bias=False)
        rng = np.random.default_rng(304)
        x = rng.uniform(-3.0, 3.0, shape).astype(np.float32)
        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=1e-4, rtol=1e-3)

    @pytest.mark.parametrize("seq_len", SEQ_LENS)
    def test_fast_gelu_bert_shape(self, model_runner, seq_len):
        shape = [1, seq_len, HIDDEN]
        model = _make_gelu_model("FastGelu", shape, with_bias=True)
        rng = np.random.default_rng(305)
        x = rng.uniform(-3.0, 3.0, shape).astype(np.float32)
        bias = rng.uniform(-1.0, 1.0, [HIDDEN]).astype(np.float32)
        actual, expected = model_runner.run_sample(model, [x, bias])
        compare_outputs(actual, expected, atol=1e-4, rtol=1e-3)


def _const(name: str, value: float):
    return numpy_helper.from_array(np.array(value, dtype=np.float32), name)


def _make_primitive_gelu_model(form: str, shape: list[int]):
    """GELU spelled as primitive ops; `form` is 'tanh' or 'erf'."""
    tp = np_to_onnx_type(np.float32)
    X = helper.make_tensor_value_info("X", tp, list(shape))
    Y = helper.make_tensor_value_info("Y", tp, list(shape))
    if form == "tanh":
        consts = {
            "half": 0.5,
            "one": 1.0,
            "cube": 3.0,
            "k": 0.044715,
            "c": 0.7978845608028654,
        }
        nodes = [
            helper.make_node("Mul", ["X", "half"], ["h"]),
            helper.make_node("Pow", ["X", "cube"], ["x3"]),
            helper.make_node("Mul", ["x3", "k"], ["kx3"]),
            helper.make_node("Add", ["X", "kx3"], ["s"]),
            helper.make_node("Mul", ["s", "c"], ["t"]),
            helper.make_node("Tanh", ["t"], ["th"]),
            helper.make_node("Add", ["th", "one"], ["p"]),
            helper.make_node("Mul", ["h", "p"], ["Y"]),
        ]
    else:
        consts = {"half": 0.5, "one": 1.0, "sqrt2": 1.4142135623730951}
        nodes = [
            helper.make_node("Mul", ["X", "half"], ["h"]),
            helper.make_node("Div", ["X", "sqrt2"], ["d"]),
            helper.make_node("Erf", ["d"], ["e"]),
            helper.make_node("Add", ["e", "one"], ["p"]),
            helper.make_node("Mul", ["h", "p"], ["Y"]),
        ]
    inits = [_const(n, v) for n, v in consts.items()]
    return make_model_from_nodes(nodes, [X], [Y], initializers=inits)


class TestPrimitiveGeluChain:
    @pytest.mark.parametrize("form", ["tanh", "erf"])
    @pytest.mark.parametrize("shape", [[2, 16], [1, 128, HIDDEN]])
    def test_primitive_gelu(self, model_runner, form, shape):
        model = _make_primitive_gelu_model(form, shape)
        rng = np.random.default_rng(307)
        x = rng.uniform(-4.0, 4.0, shape).astype(np.float32)
        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=1e-4, rtol=1e-3)
