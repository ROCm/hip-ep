#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""Tests for binary elementwise ops: Div, Equal, Less, Mod, And.

These five ops are added by the qwen-vision-kernels PR. The runtime expects
A and B to have the **same shape** -- broadcasting is performed by the
upstream OnnxToHip pre-pass that materialises an Expand before the binary
op, so the kernels themselves see equal-sized inputs.

Per the runtime dtype tables in lib/Runtime/real/<op>.cpp:

    Div   : f16, f32, i32, i64
    Equal : f16, f32, i32, i64  (output: bool)
    Less  : f16, f32, i32, i64  (output: bool)
    Mod   : f16/f32 require fmod=1; i32/i64 require fmod=0
    And   : bool only (output: bool); mirrors ORT v1.22.2
            SPECIALIZED_BINARY_ELEMENTWISE_IMPL(And, bool)

We exercise both small smoke shapes and a llama-style [1, S] integer shape
to mirror attention-mask arithmetic.
"""

from __future__ import annotations

import numpy as np
import pytest
from onnx import helper

from framework.comparator import compare_outputs
from framework.onnx_utils import make_model_from_nodes, np_to_onnx_type

SEQ_LENS = [1, 128]


def _make_binary_model(
    op_type: str,
    in_dtype: np.dtype,
    out_dtype: np.dtype,
    shape: list[int],
    extra_attrs: dict | None = None,
):
    """A,B same shape; output `shape` is `shape` too (no broadcast)."""
    in_tp = np_to_onnx_type(in_dtype)
    out_tp = np_to_onnx_type(out_dtype)
    A = helper.make_tensor_value_info("A", in_tp, list(shape))
    B = helper.make_tensor_value_info("B", in_tp, list(shape))
    Y = helper.make_tensor_value_info("Y", out_tp, list(shape))
    node = helper.make_node(op_type, ["A", "B"], ["Y"], **(extra_attrs or {}))
    return make_model_from_nodes([node], [A, B], [Y])


def _safe_divisor(rng, dtype: np.dtype, shape: list[int]) -> np.ndarray:
    """Return B with no zero elements (Div / Mod would be UB on zeros)."""
    if np.issubdtype(dtype, np.integer):
        b = rng.integers(1, 10, shape, dtype=dtype)
        sign = rng.integers(0, 2, shape, dtype=dtype) * 2 - 1
        return (b * sign).astype(dtype)
    b = rng.uniform(0.5, 3.0, shape).astype(dtype)
    sign = rng.choice([-1.0, 1.0], size=shape).astype(dtype)
    return (b * sign).astype(dtype)


# ---------------------------------------------------------------------------
# Div : y = a / b
# ---------------------------------------------------------------------------
class TestDiv:
    @pytest.mark.parametrize(
        "dtype,shape",
        [
            (np.float16, [4, 8]),
            (np.float32, [4, 8]),
            (np.int32, [4, 8]),
            (np.int64, [4, 8]),
        ],
    )
    def test_div(self, model_runner, dtype, shape):
        model = _make_binary_model("Div", dtype, dtype, shape)
        rng = np.random.default_rng(201)
        if np.issubdtype(dtype, np.integer):
            a = rng.integers(-50, 50, shape, dtype=dtype)
        else:
            a = rng.uniform(-3.0, 3.0, shape).astype(dtype)
        b = _safe_divisor(rng, dtype, shape)
        actual, expected = model_runner.run_sample(model, [a, b])
        # Integer Div: exact; fp16: ~1 ULP; fp32: tighter.
        if np.issubdtype(dtype, np.integer):
            atol = 0
        elif dtype == np.float16:
            atol = 1e-3
        else:
            atol = 1e-5
        compare_outputs(actual, expected, atol=atol, rtol=1e-3)

    def test_div_5d_broadcast(self, model_runner):
        """Rank-5 Div with a broadcast channel axis."""
        lhs_shape = [2, 3, 4, 2, 5]
        rhs_shape = [2, 1, 4, 2, 5]
        in_tp = np_to_onnx_type(np.float32)
        A = helper.make_tensor_value_info("A", in_tp, lhs_shape)
        B = helper.make_tensor_value_info("B", in_tp, rhs_shape)
        Y = helper.make_tensor_value_info("Y", in_tp, lhs_shape)
        node = helper.make_node("Div", ["A", "B"], ["Y"])
        model = make_model_from_nodes([node], [A, B], [Y])

        rng = np.random.default_rng(202)
        a = rng.uniform(-3.0, 3.0, lhs_shape).astype(np.float32)
        b = _safe_divisor(rng, np.float32, rhs_shape)
        actual, expected = model_runner.run_sample(model, [a, b])
        compare_outputs(actual, expected, atol=1e-5, rtol=1e-3)


# ---------------------------------------------------------------------------
# Equal : y = (a == b), bool output
# ---------------------------------------------------------------------------
class TestEqual:
    @pytest.mark.parametrize(
        "dtype,shape",
        [
            (np.float16, [4, 8]),
            (np.float32, [4, 8]),
            (np.int32, [4, 8]),
            (np.int64, [4, 8]),
        ],
    )
    def test_equal(self, model_runner, dtype, shape):
        model = _make_binary_model("Equal", dtype, np.bool_, shape)
        rng = np.random.default_rng(202)
        if np.issubdtype(dtype, np.integer):
            a = rng.integers(0, 5, shape, dtype=dtype)
            b = rng.integers(0, 5, shape, dtype=dtype)
        else:
            # Use a small discrete set so we actually exercise equality hits.
            a = rng.choice([-1.0, 0.0, 1.0, 2.0], size=shape).astype(dtype)
            b = rng.choice([-1.0, 0.0, 1.0, 2.0], size=shape).astype(dtype)
        actual, expected = model_runner.run_sample(model, [a, b])
        compare_outputs(actual, expected, atol=0)

    @pytest.mark.parametrize("seq_len", SEQ_LENS)
    def test_equal_attention_mask_shape(self, model_runner, seq_len):
        """i64 [1, S] equality, mirroring `attention_mask == 0` patterns."""
        shape = [1, seq_len]
        model = _make_binary_model("Equal", np.int64, np.bool_, shape)
        rng = np.random.default_rng(203)
        a = rng.integers(0, 2, shape, dtype=np.int64)
        b = np.zeros(shape, dtype=np.int64)
        actual, expected = model_runner.run_sample(model, [a, b])
        compare_outputs(actual, expected, atol=0)


# ---------------------------------------------------------------------------
# Less : y = (a < b), bool output
# ---------------------------------------------------------------------------
class TestLess:
    @pytest.mark.parametrize(
        "dtype,shape",
        [
            (np.float16, [4, 8]),
            (np.float32, [4, 8]),
            (np.int32, [4, 8]),
            (np.int64, [4, 8]),
        ],
    )
    def test_less(self, model_runner, dtype, shape):
        model = _make_binary_model("Less", dtype, np.bool_, shape)
        rng = np.random.default_rng(204)
        if np.issubdtype(dtype, np.integer):
            a = rng.integers(-10, 10, shape, dtype=dtype)
            b = rng.integers(-10, 10, shape, dtype=dtype)
        else:
            a = rng.uniform(-3.0, 3.0, shape).astype(dtype)
            b = rng.uniform(-3.0, 3.0, shape).astype(dtype)
        actual, expected = model_runner.run_sample(model, [a, b])
        compare_outputs(actual, expected, atol=0)

    @pytest.mark.parametrize("seq_len", SEQ_LENS)
    def test_less_position_threshold_shape(self, model_runner, seq_len):
        """i64 [1, S] `< threshold` pattern (position-id thresholding)."""
        shape = [1, seq_len]
        model = _make_binary_model("Less", np.int64, np.bool_, shape)
        rng = np.random.default_rng(205)
        a = rng.integers(0, seq_len, shape, dtype=np.int64)
        b = np.full(shape, max(1, seq_len // 2), dtype=np.int64)
        actual, expected = model_runner.run_sample(model, [a, b])
        compare_outputs(actual, expected, atol=0)

    def test_less_5d_broadcast(self, model_runner):
        """Rank-5 Less with a broadcast channel axis."""
        lhs_shape = [2, 3, 4, 2, 5]
        rhs_shape = [2, 1, 4, 2, 5]
        in_tp = np_to_onnx_type(np.float32)
        out_tp = np_to_onnx_type(np.bool_)
        A = helper.make_tensor_value_info("A", in_tp, lhs_shape)
        B = helper.make_tensor_value_info("B", in_tp, rhs_shape)
        Y = helper.make_tensor_value_info("Y", out_tp, lhs_shape)
        node = helper.make_node("Less", ["A", "B"], ["Y"])
        model = make_model_from_nodes([node], [A, B], [Y])

        rng = np.random.default_rng(206)
        a = rng.uniform(-3.0, 3.0, lhs_shape).astype(np.float32)
        b = rng.uniform(-3.0, 3.0, rhs_shape).astype(np.float32)
        actual, expected = model_runner.run_sample(model, [a, b])
        compare_outputs(actual, expected, atol=0)


# ---------------------------------------------------------------------------
# Mod : y = a % b   (fmod=0 -> Python-style int; fmod=1 -> C fmod, float)
# ---------------------------------------------------------------------------
class TestMod:
    @pytest.mark.parametrize(
        "dtype,shape",
        [
            (np.int32, [4, 8]),
            (np.int64, [4, 8]),
        ],
    )
    def test_mod_int(self, model_runner, dtype, shape):
        """ONNX Mod with fmod=0 (default) -- integer Python-style modulo."""
        model = _make_binary_model("Mod", dtype, dtype, shape)
        rng = np.random.default_rng(206)
        a = rng.integers(-50, 50, shape, dtype=dtype)
        b = _safe_divisor(rng, dtype, shape)
        actual, expected = model_runner.run_sample(model, [a, b])
        compare_outputs(actual, expected, atol=0)

    @pytest.mark.parametrize(
        "dtype,shape",
        [
            (np.float16, [4, 8]),
            (np.float32, [4, 8]),
        ],
    )
    def test_mod_fmod(self, model_runner, dtype, shape):
        """ONNX Mod with fmod=1 -- C `fmod` semantics on float inputs."""
        model = _make_binary_model("Mod", dtype, dtype, shape, extra_attrs={"fmod": 1})
        rng = np.random.default_rng(207)
        a = rng.uniform(-3.0, 3.0, shape).astype(dtype)
        b = _safe_divisor(rng, dtype, shape)
        actual, expected = model_runner.run_sample(model, [a, b])
        atol = 1e-3 if dtype == np.float16 else 1e-5
        compare_outputs(actual, expected, atol=atol, rtol=1e-3)


# ---------------------------------------------------------------------------
# And : y = a & b, bool inputs / bool output
# ---------------------------------------------------------------------------
class TestAnd:
    @pytest.mark.parametrize(
        "shape",
        [
            [4, 8],
            [3, 5, 7],
        ],
    )
    def test_and(self, model_runner, shape):
        """ONNX And on bool tensors (the only dtype ORT v1.22.2 specializes)."""
        model = _make_binary_model("And", np.bool_, np.bool_, shape)
        rng = np.random.default_rng(208)
        a = rng.integers(0, 2, shape).astype(np.bool_)
        b = rng.integers(0, 2, shape).astype(np.bool_)
        actual, expected = model_runner.run_sample(model, [a, b])
        compare_outputs(actual, expected, atol=0)

    @pytest.mark.parametrize("seq_len", SEQ_LENS)
    def test_and_attention_mask_shape(self, model_runner, seq_len):
        """bool [1, S] AND, mirroring attention-mask combination patterns."""
        shape = [1, seq_len]
        model = _make_binary_model("And", np.bool_, np.bool_, shape)
        rng = np.random.default_rng(209)
        a = rng.integers(0, 2, shape).astype(np.bool_)
        b = rng.integers(0, 2, shape).astype(np.bool_)
        actual, expected = model_runner.run_sample(model, [a, b])
        compare_outputs(actual, expected, atol=0)
