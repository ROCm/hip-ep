#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""Tests for unary elementwise ops: Neg, Sign, Cos, Sin, Not.

These five ops are added by the qwen-vision-kernels PR. They share the same
host-side shape: one input, one output, identical shape, no broadcasting.
The runtime dispatches each to its own HIP elementwise kernel.

Coverage per op (matching the runtime dtype tables in
lib/Runtime/real/<op>.cpp):

    Neg   : f16, f32, i32, i64
    Sign  : f16, f32, i32, i64
    Cos   : f16, f32           (trig kernels are float-only)
    Sin   : f16, f32           (trig kernels are float-only)
    Not   : bool only          (treated as 1-byte stream)
    IsInf : f16, f32, f64       (output: bool)

All tests cover a small shape (smoke) plus a llama-3.1-8B-style
[1, S, 4096] shape for S in {1, 128}, which is the dominant tensor
geometry these ops appear at in the Qwen / Llama families.
"""

from __future__ import annotations

import numpy as np
import pytest
from onnx import TensorProto, helper

from framework.comparator import compare_outputs
from framework.onnx_utils import make_model_from_nodes, np_to_onnx_type

SEQ_LENS = [1, 128]
HIDDEN = 4096


def _make_unary_model(op_type: str, dtype: np.dtype, shape: list[int]):
    tp = np_to_onnx_type(dtype)
    X = helper.make_tensor_value_info("X", tp, list(shape))
    Y = helper.make_tensor_value_info("Y", tp, list(shape))
    node = helper.make_node(op_type, ["X"], ["Y"])
    return make_model_from_nodes([node], [X], [Y])


def _rng_input(rng, dtype: np.dtype, shape: list[int]):
    if np.issubdtype(dtype, np.integer):
        return rng.integers(-10, 11, shape, dtype=dtype)
    return rng.uniform(-3.0, 3.0, shape).astype(dtype)


# ---------------------------------------------------------------------------
# Neg : y = -x
# ---------------------------------------------------------------------------
class TestNeg:
    @pytest.mark.parametrize(
        "dtype,shape",
        [
            (np.float16, [4, 8]),
            (np.float32, [4, 8]),
            (np.int32, [4, 8]),
            (np.int64, [4, 8]),
        ],
    )
    def test_neg(self, model_runner, dtype, shape):
        model = _make_unary_model("Neg", dtype, shape)
        rng = np.random.default_rng(101)
        x = _rng_input(rng, dtype, shape)
        actual, expected = model_runner.run_sample(model, [x])
        # Neg is bit-exact for every supported dtype: the generic kernel
        # emits a single neg instruction (i32/i64/f32) and the fp16
        # specialisation negates in fp32 (every fp16 value round-trips
        # exactly through fp32). No rounding budget needed.
        compare_outputs(actual, expected, atol=0)

    @pytest.mark.parametrize("seq_len", SEQ_LENS)
    def test_neg_llama_shape(self, model_runner, seq_len):
        shape = [1, seq_len, HIDDEN]
        model = _make_unary_model("Neg", np.float16, shape)
        rng = np.random.default_rng(102)
        x = rng.uniform(-3.0, 3.0, shape).astype(np.float16)
        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=0)


# ---------------------------------------------------------------------------
# Sign : y = sign(x)  (-1, 0, +1)
# ---------------------------------------------------------------------------
class TestSign:
    @pytest.mark.parametrize(
        "dtype,shape",
        [
            (np.float16, [4, 8]),
            (np.float32, [4, 8]),
            (np.int32, [4, 8]),
            (np.int64, [4, 8]),
        ],
    )
    def test_sign(self, model_runner, dtype, shape):
        model = _make_unary_model("Sign", dtype, shape)
        rng = np.random.default_rng(103)
        x = _rng_input(rng, dtype, shape)
        # Ensure both negative and exactly-zero cases are exercised.
        if shape[0] >= 2 and shape[1] >= 2:
            x[0, 0] = 0
            x[1, 1] = 0
        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=0)

    @pytest.mark.parametrize("seq_len", SEQ_LENS)
    def test_sign_llama_shape(self, model_runner, seq_len):
        shape = [1, seq_len, HIDDEN]
        model = _make_unary_model("Sign", np.float16, shape)
        rng = np.random.default_rng(104)
        x = rng.uniform(-1.0, 1.0, shape).astype(np.float16)
        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=0)


# ---------------------------------------------------------------------------
# Cos : y = cos(x)
# ---------------------------------------------------------------------------
class TestCos:
    @pytest.mark.parametrize(
        "dtype,shape",
        [
            (np.float16, [4, 8]),
            (np.float32, [4, 8]),
        ],
    )
    def test_cos(self, model_runner, dtype, shape):
        model = _make_unary_model("Cos", dtype, shape)
        rng = np.random.default_rng(105)
        # Match the typical rope-cache input range; large magnitudes blow up
        # fp16 cos precision because the argument reduction is fp16 internally.
        x = rng.uniform(-np.pi, np.pi, shape).astype(dtype)
        actual, expected = model_runner.run_sample(model, [x])
        # fp16 trig has up to ~2e-3 ULP near +-1; fp32 ~ 1e-6
        atol = 3e-3 if dtype == np.float16 else 1e-5
        compare_outputs(actual, expected, atol=atol)

    @pytest.mark.parametrize("seq_len", SEQ_LENS)
    def test_cos_llama_shape(self, model_runner, seq_len):
        shape = [1, seq_len, HIDDEN]
        model = _make_unary_model("Cos", np.float16, shape)
        rng = np.random.default_rng(106)
        x = rng.uniform(-np.pi, np.pi, shape).astype(np.float16)
        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=3e-3)


# ---------------------------------------------------------------------------
# Sin : y = sin(x)
# ---------------------------------------------------------------------------
class TestSin:
    @pytest.mark.parametrize(
        "dtype,shape",
        [
            (np.float16, [4, 8]),
            (np.float32, [4, 8]),
        ],
    )
    def test_sin(self, model_runner, dtype, shape):
        model = _make_unary_model("Sin", dtype, shape)
        rng = np.random.default_rng(107)
        x = rng.uniform(-np.pi, np.pi, shape).astype(dtype)
        actual, expected = model_runner.run_sample(model, [x])
        atol = 3e-3 if dtype == np.float16 else 1e-5
        compare_outputs(actual, expected, atol=atol)

    @pytest.mark.parametrize("seq_len", SEQ_LENS)
    def test_sin_llama_shape(self, model_runner, seq_len):
        shape = [1, seq_len, HIDDEN]
        model = _make_unary_model("Sin", np.float16, shape)
        rng = np.random.default_rng(108)
        x = rng.uniform(-np.pi, np.pi, shape).astype(np.float16)
        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=3e-3)


# ---------------------------------------------------------------------------
# Not : y = !x  (bool-only per ONNX spec)
# ---------------------------------------------------------------------------
class TestNot:
    @pytest.mark.parametrize("shape", [[4, 8], [1, 64], [2, 3, 5]])
    def test_not(self, model_runner, shape):
        model = _make_unary_model("Not", np.bool_, shape)
        rng = np.random.default_rng(109)
        x = rng.integers(0, 2, shape, dtype=np.bool_)
        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=0)

    @pytest.mark.parametrize("seq_len", SEQ_LENS)
    def test_not_attention_mask_shape(self, model_runner, seq_len):
        """Bool [1, S] -- attention-mask negation pattern."""
        shape = [1, seq_len]
        model = _make_unary_model("Not", np.bool_, shape)
        rng = np.random.default_rng(110)
        x = rng.integers(0, 2, shape, dtype=np.bool_)
        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=0)


def _make_isinf_model(
    dtype: np.dtype,
    shape: list[int],
    detect_negative: int = 1,
    detect_positive: int = 1,
):
    tp = np_to_onnx_type(dtype)
    X = helper.make_tensor_value_info("X", tp, list(shape))
    Y = helper.make_tensor_value_info("Y", TensorProto.BOOL, list(shape))
    node = helper.make_node(
        "IsInf",
        ["X"],
        ["Y"],
        detect_negative=detect_negative,
        detect_positive=detect_positive,
    )
    return make_model_from_nodes([node], [X], [Y])


# ---------------------------------------------------------------------------
# IsInf : y = isinf(x), bool output
# ---------------------------------------------------------------------------
class TestIsInf:
    @pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
    def test_isinf(self, model_runner, dtype):
        shape = [4, 8]
        model = _make_isinf_model(dtype, shape)
        x = np.array(
            [[0.0, 1.0, np.inf, -np.inf], [np.nan, -1.0, 2.0, -2.0]],
            dtype=dtype,
        )
        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=0)

    @pytest.mark.parametrize("dtype", [np.float16, np.float32])
    def test_isinf_detect_positive_only(self, model_runner, dtype):
        shape = [4]
        model = _make_isinf_model(dtype, shape, detect_negative=0, detect_positive=1)
        x = np.array([0.0, np.inf, -np.inf, 1.0], dtype=dtype)
        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=0)

    @pytest.mark.parametrize("seq_len", SEQ_LENS)
    def test_isinf_llama_shape(self, model_runner, seq_len):
        shape = [1, seq_len, HIDDEN]
        model = _make_isinf_model(np.float16, shape)
        rng = np.random.default_rng(111)
        x = rng.uniform(-3.0, 3.0, shape).astype(np.float16)
        x[0, 0, 0] = np.inf
        x[0, 0, 1] = -np.inf
        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=0)
