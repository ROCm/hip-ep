#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""Tests for activation / unary operations: Sigmoid, Tanh, Sqrt, Reciprocal,
Softplus."""

import numpy as np
import pytest
from onnx import helper

from framework.comparator import compare_outputs
from framework.onnx_utils import make_model_from_nodes, np_to_onnx_type

SEQ_LENS = [1, 128]
INTERMEDIATE = 14336

# Shapes used by the chunk_opt prefill/decode models.  Sqrt/Reciprocal
# are paired with the per-head L2-norm chain (16 heads) and Softplus is
# used in the 32-channel gating chain.
CHUNK_OPT_SEQ_LENS = [1, 128]
CHUNK_OPT_HEADS = 16
CHUNK_OPT_GATE_CH = 32


def _make_unary_model(op_type: str, dtype, shape: list[int], **attrs):
    """Build a single-input unary ONNX model."""
    tp = np_to_onnx_type(dtype)
    X = helper.make_tensor_value_info("X", tp, shape)
    Y = helper.make_tensor_value_info("Y", tp, shape)
    node = helper.make_node(op_type, ["X"], ["Y"], **attrs)
    return make_model_from_nodes([node], [X], [Y])


class TestSigmoid:
    @pytest.mark.parametrize(
        "dtype,shape",
        [
            (np.float16, [1, 10]),
            (np.float16, [4, 256]),
        ],
    )
    def test_sigmoid(self, model_runner, dtype, shape):
        model = _make_unary_model("Sigmoid", dtype, shape)

        rng = np.random.default_rng(42)
        x = rng.uniform(-5, 5, shape).astype(dtype)

        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=1e-3)

    @pytest.mark.parametrize("seq_len", SEQ_LENS)
    def test_sigmoid_llama_shape(self, model_runner, seq_len):
        """Sigmoid with shapes matching Llama MLP gate activation."""
        shape = [1, seq_len, INTERMEDIATE]
        model = _make_unary_model("Sigmoid", np.float16, shape)

        rng = np.random.default_rng(42)
        x = rng.uniform(-5, 5, shape).astype(np.float16)

        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=1e-3)


class TestTanh:
    @pytest.mark.parametrize(
        "dtype,shape",
        [
            (np.float16, [1, 10]),
            (np.float16, [4, 256]),
        ],
    )
    def test_tanh(self, model_runner, dtype, shape):
        model = _make_unary_model("Tanh", dtype, shape)

        rng = np.random.default_rng(42)
        x = rng.uniform(-5, 5, shape).astype(dtype)

        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=1e-3)


class TestSqrt:
    """Sqrt as used in the per-head L2-norm chain of chunk_opt models:
    Sqrt(SumSq + eps) -> per-head scale.  Operates on strictly positive
    fp16 inputs, so we sample from a positive distribution to mirror the
    runtime input domain."""

    @pytest.mark.parametrize(
        "dtype,shape",
        [
            (np.float16, [1, 10]),
            (np.float32, [4, 256]),
        ],
    )
    def test_sqrt(self, model_runner, dtype, shape):
        model = _make_unary_model("Sqrt", dtype, shape)

        rng = np.random.default_rng(7)
        # Strictly positive to match Sqrt's mathematical domain.
        x = rng.uniform(1e-3, 10.0, shape).astype(dtype)

        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=1e-3)

    @pytest.mark.parametrize("seq_len", CHUNK_OPT_SEQ_LENS)
    def test_sqrt_chunk_opt_l2norm_shape(self, model_runner, seq_len):
        """fp16 Sqrt over the per-head L2-norm tensor: [1, S, 16, 1].

        Driven by chunk_opt prefill/decode (60 occurrences each).
        """
        shape = [1, seq_len, CHUNK_OPT_HEADS, 1]
        model = _make_unary_model("Sqrt", np.float16, shape)

        rng = np.random.default_rng(11)
        # Magnitudes resembling SumSq + eps over a hidden dim of ~128.
        x = rng.uniform(0.1, 50.0, shape).astype(np.float16)

        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=1e-2)


class TestReciprocal:
    """Reciprocal as used downstream of Sqrt in the L2-norm chain:
    1 / Sqrt(SumSq + eps).  Inputs are strictly positive fp16."""

    @pytest.mark.parametrize(
        "dtype,shape",
        [
            (np.float16, [1, 10]),
            (np.float32, [4, 256]),
        ],
    )
    def test_reciprocal(self, model_runner, dtype, shape):
        model = _make_unary_model("Reciprocal", dtype, shape)

        rng = np.random.default_rng(13)
        # Stay clear of zero to avoid Inf/NaN that would defeat allclose.
        x = rng.uniform(0.1, 10.0, shape).astype(dtype)
        sign = rng.choice([-1.0, 1.0], shape).astype(dtype)
        x = x * sign

        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=1e-3, rtol=1e-3)

    @pytest.mark.parametrize("seq_len", CHUNK_OPT_SEQ_LENS)
    def test_reciprocal_chunk_opt_l2norm_shape(self, model_runner, seq_len):
        """fp16 Reciprocal over the per-head L2-norm scale: [1, S, 16, 1].

        Driven by chunk_opt prefill/decode (60 occurrences each).
        """
        shape = [1, seq_len, CHUNK_OPT_HEADS, 1]
        model = _make_unary_model("Reciprocal", np.float16, shape)

        rng = np.random.default_rng(17)
        # Positive only: the actual runtime feeds Sqrt(...) here.
        x = rng.uniform(0.5, 20.0, shape).astype(np.float16)

        actual, expected = model_runner.run_sample(model, [x])
        # fp16 Reciprocal has visible relative error for small inputs;
        # relax atol slightly while keeping rtol tight.
        compare_outputs(actual, expected, atol=5e-3, rtol=1e-2)


class TestSoftplus:
    """Softplus(x) = ln(1 + exp(x)).  Used in the 32-channel gating
    chain of chunk_opt models (30 occurrences per graph)."""

    @pytest.mark.parametrize(
        "dtype,shape",
        [
            (np.float32, [1, 10]),
            (np.float32, [4, 256]),
            (np.float16, [1, 10]),
            (np.float16, [4, 256]),
        ],
    )
    def test_softplus(self, model_runner, dtype, shape):
        model = _make_unary_model("Softplus", dtype, shape)

        rng = np.random.default_rng(19)
        x = rng.uniform(-5, 5, shape).astype(dtype)

        actual, expected = model_runner.run_sample(model, [x])
        atol = 1e-3 if dtype == np.float16 else 1e-4
        compare_outputs(actual, expected, atol=atol)

    @pytest.mark.parametrize("seq_len", CHUNK_OPT_SEQ_LENS)
    def test_softplus_chunk_opt_gate_shape(self, model_runner, seq_len):
        """fp32 Softplus over the gating tensor: [1, S, 32].

        Driven by chunk_opt prefill/decode (30 occurrences each).
        """
        shape = [1, seq_len, CHUNK_OPT_GATE_CH]
        model = _make_unary_model("Softplus", np.float32, shape)

        rng = np.random.default_rng(23)
        x = rng.uniform(-5, 5, shape).astype(np.float32)

        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=1e-4)
