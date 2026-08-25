#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""Tests for the unary math ops Abs, Ceil, Round, Atan, Floor, Log, Erf.

These route through the shared unary elementwise HIP kernel family
(lib/Runtime/real/{abs,ceil,round,atan,floor,log,erf}.cpp). Per those dtype
tables:

    Abs   : f16, f32, i32, i64   (bit-exact; abs is value-preserving)
    Ceil  : f16, f32             (bit-exact; result is integral)
    Round : f16, f32             (bit-exact; ONNX ties-to-even)
    Atan  : f16, f32             (fp16 ~1e-3, fp32 ~1e-5)
    Floor : f16, f32             (bit-exact; result is integral)
    Log   : f16, f32             (positive inputs only; fp16 ~1e-3, fp32 ~1e-6)

Each op is checked at a small smoke shape plus a llama-style [1, S, 4096]
shape for S in {1, 128}.
"""

from __future__ import annotations

import numpy as np
import pytest
from onnx import helper

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


# ---------------------------------------------------------------------------
# Abs : y = |x|
# ---------------------------------------------------------------------------
class TestAbs:
    @pytest.mark.parametrize(
        "dtype",
        [np.float16, np.float32, np.int32, np.int64],
    )
    def test_abs(self, model_runner, dtype):
        shape = [4, 8]
        model = _make_unary_model("Abs", dtype, shape)
        rng = np.random.default_rng(901)
        if np.issubdtype(dtype, np.integer):
            x = rng.integers(-50, 50, shape, dtype=dtype)
        else:
            x = rng.uniform(-5.0, 5.0, shape).astype(dtype)
        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=0)

    @pytest.mark.parametrize("seq_len", SEQ_LENS)
    def test_abs_llama_shape(self, model_runner, seq_len):
        shape = [1, seq_len, HIDDEN]
        model = _make_unary_model("Abs", np.float16, shape)
        rng = np.random.default_rng(902)
        x = rng.uniform(-5.0, 5.0, shape).astype(np.float16)
        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=0)


# ---------------------------------------------------------------------------
# Ceil : y = ceil(x)
# ---------------------------------------------------------------------------
class TestCeil:
    @pytest.mark.parametrize("dtype", [np.float16, np.float32])
    def test_ceil(self, model_runner, dtype):
        shape = [4, 8]
        model = _make_unary_model("Ceil", dtype, shape)
        rng = np.random.default_rng(903)
        x = rng.uniform(-5.0, 5.0, shape).astype(dtype)
        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=0)

    @pytest.mark.parametrize("seq_len", SEQ_LENS)
    def test_ceil_llama_shape(self, model_runner, seq_len):
        shape = [1, seq_len, HIDDEN]
        model = _make_unary_model("Ceil", np.float16, shape)
        rng = np.random.default_rng(904)
        x = rng.uniform(-5.0, 5.0, shape).astype(np.float16)
        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=0)


# ---------------------------------------------------------------------------
# Round : y = round_ties_to_even(x)  (ONNX Round)
# ---------------------------------------------------------------------------
class TestRound:
    @pytest.mark.parametrize("dtype", [np.float16, np.float32])
    def test_round(self, model_runner, dtype):
        shape = [4, 8]
        model = _make_unary_model("Round", dtype, shape)
        rng = np.random.default_rng(915)
        x = rng.uniform(-5.0, 5.0, shape).astype(dtype)
        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=0)

    def test_round_ties_to_even(self, model_runner):
        model = _make_unary_model("Round", np.float32, [7])
        x = np.array([0.9, 2.5, 2.3, 1.5, -4.5, -1.5, 0.0], dtype=np.float32)
        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=0)

    @pytest.mark.parametrize("seq_len", SEQ_LENS)
    def test_round_llama_shape(self, model_runner, seq_len):
        shape = [1, seq_len, HIDDEN]
        model = _make_unary_model("Round", np.float16, shape)
        rng = np.random.default_rng(916)
        x = rng.uniform(-5.0, 5.0, shape).astype(np.float16)
        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=0)


# ---------------------------------------------------------------------------
# Atan : y = arctan(x)
# ---------------------------------------------------------------------------
class TestAtan:
    @pytest.mark.parametrize("dtype", [np.float16, np.float32])
    def test_atan(self, model_runner, dtype):
        shape = [4, 8]
        model = _make_unary_model("Atan", dtype, shape)
        rng = np.random.default_rng(917)
        x = rng.uniform(-5.0, 5.0, shape).astype(dtype)
        actual, expected = model_runner.run_sample(model, [x])
        atol = 1e-3 if dtype == np.float16 else 1e-5
        compare_outputs(actual, expected, atol=atol, rtol=1e-3)

    @pytest.mark.parametrize("seq_len", SEQ_LENS)
    def test_atan_llama_shape(self, model_runner, seq_len):
        shape = [1, seq_len, HIDDEN]
        model = _make_unary_model("Atan", np.float16, shape)
        rng = np.random.default_rng(918)
        x = rng.uniform(-5.0, 5.0, shape).astype(np.float16)
        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=1e-3, rtol=1e-3)


# ---------------------------------------------------------------------------
# Floor : y = floor(x)
# ---------------------------------------------------------------------------
class TestFloor:
    @pytest.mark.parametrize("dtype", [np.float16, np.float32])
    def test_floor(self, model_runner, dtype):
        shape = [4, 8]
        model = _make_unary_model("Floor", dtype, shape)
        rng = np.random.default_rng(913)
        x = rng.uniform(-5.0, 5.0, shape).astype(dtype)
        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=0)

    @pytest.mark.parametrize("seq_len", SEQ_LENS)
    def test_floor_llama_shape(self, model_runner, seq_len):
        shape = [1, seq_len, HIDDEN]
        model = _make_unary_model("Floor", np.float16, shape)
        rng = np.random.default_rng(914)
        x = rng.uniform(-5.0, 5.0, shape).astype(np.float16)
        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=0)


# ---------------------------------------------------------------------------
# Log : y = ln(x)  (positive domain)
# ---------------------------------------------------------------------------
class TestLog:
    @pytest.mark.parametrize("dtype", [np.float16, np.float32])
    def test_log(self, model_runner, dtype):
        shape = [4, 8]
        model = _make_unary_model("Log", dtype, shape)
        rng = np.random.default_rng(905)
        x = rng.uniform(0.1, 10.0, shape).astype(dtype)
        actual, expected = model_runner.run_sample(model, [x])
        atol = 1e-3 if dtype == np.float16 else 1e-5
        compare_outputs(actual, expected, atol=atol, rtol=1e-3)

    @pytest.mark.parametrize("seq_len", SEQ_LENS)
    def test_log_llama_shape(self, model_runner, seq_len):
        shape = [1, seq_len, HIDDEN]
        model = _make_unary_model("Log", np.float16, shape)
        rng = np.random.default_rng(906)
        x = rng.uniform(0.1, 10.0, shape).astype(np.float16)
        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=1e-3, rtol=1e-3)


# ---------------------------------------------------------------------------
# Erf : y = erf(x)
# ---------------------------------------------------------------------------
class TestErf:
    @pytest.mark.parametrize("dtype", [np.float16, np.float32])
    def test_erf(self, model_runner, dtype):
        shape = [4, 8]
        model = _make_unary_model("Erf", dtype, shape)
        rng = np.random.default_rng(907)
        x = rng.uniform(-3.0, 3.0, shape).astype(dtype)
        actual, expected = model_runner.run_sample(model, [x])
        atol = 2e-3 if dtype == np.float16 else 1e-5
        compare_outputs(actual, expected, atol=atol, rtol=1e-3)

    def test_erf_bev_shape(self, model_runner):
        shape = [1, 128, 200, 200]
        model = _make_unary_model("Erf", np.float32, shape)
        rng = np.random.default_rng(908)
        x = rng.uniform(-2.0, 2.0, shape).astype(np.float32)
        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=1e-5, rtol=1e-5)
