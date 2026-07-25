#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""Tests for reduce operations: ReduceSum, ReduceMax, ReduceProd.

The Llama-3.1-8B-fixed model uses ReduceSum on the attention_mask
([1, 128] i64, axis=1, keepdims=1) in the attention mask subgraph.

ReduceMax / ReduceProd are added by the qwen-vision-kernels PR. Per
lib/Runtime/real/reduce_{max,prod}.cpp the runtime supports only
{f16, i32, i64} for both -- f32 is NOT in the dispatch table, so we
exercise the supported set only.
"""

import numpy as np
import pytest
from onnx import helper, numpy_helper

from framework.comparator import compare_outputs
from framework.onnx_utils import make_model_from_nodes, np_to_onnx_type

# Sequence lengths chosen to cover:
#   * S=1   : decode (chunk_opt/decode_p512m16384.onnx, 60 fp16 ReduceSum)
#   * S=128 : moderate prefill / smoke
SEQ_LENS = [1, 128]


def _reduced_shape(input_shape: list[int], axes: list[int], keepdims: int):
    out = []
    for i, dim in enumerate(input_shape):
        if i in axes or (i - len(input_shape)) in axes:
            if keepdims:
                out.append(1)
        else:
            out.append(dim)
    if not out:
        out = [1] if keepdims else []
    return out


def _make_reduce_model(
    op_type: str,
    dtype,
    input_shape: list[int],
    axes: list[int],
    keepdims: int = 1,
):
    """Build a Reduce* ONNX model with axes as initializer (opset 18 style).

    ReduceSum, ReduceMax, ReduceProd all share the same ONNX-18 signature
    (axes as a tensor input plus a `keepdims` attribute) -- but only
    ReduceSum got that form at opset 13. ReduceMax and ReduceProd require
    opset 18 to accept `axes` as an input rather than an attribute, so we
    bump the opset import here.
    """
    tp = np_to_onnx_type(dtype)
    X = helper.make_tensor_value_info("X", tp, input_shape)
    axes_init = numpy_helper.from_array(np.array(axes, dtype=np.int64), name="axes")
    output_shape = _reduced_shape(input_shape, axes, keepdims)
    Y = helper.make_tensor_value_info("Y", tp, output_shape if output_shape else None)
    node = helper.make_node(op_type, ["X", "axes"], ["Y"], keepdims=keepdims)
    return make_model_from_nodes([node], [X], [Y], initializers=[axes_init], opset=18)


def _make_reduce_sum_model(
    dtype, input_shape: list[int], axes: list[int], keepdims: int = 1
):
    return _make_reduce_model("ReduceSum", dtype, input_shape, axes, keepdims)


class TestReduceSum:
    @pytest.mark.parametrize(
        "dtype,shape,axes,keepdims",
        [
            (np.int64, [1, 128], [1], 1),
            (np.int64, [2, 64], [1], 1),
        ],
    )
    def test_reduce_sum(self, model_runner, dtype, shape, axes, keepdims):
        model = _make_reduce_sum_model(dtype, shape, axes, keepdims)

        rng = np.random.default_rng(55)
        if np.issubdtype(dtype, np.integer):
            x = rng.integers(-10, 10, shape, dtype=dtype)
        else:
            x = rng.uniform(-5, 5, shape).astype(dtype)

        actual, expected = model_runner.run_sample(model, [x])
        atol = 0 if np.issubdtype(dtype, np.integer) else 1e-4
        compare_outputs(actual, expected, atol=atol)

    @pytest.mark.parametrize("seq_len", SEQ_LENS)
    def test_reduce_sum_attention_mask_llama_shape(self, model_runner, seq_len):
        """ReduceSum on attention_mask [1, S] i64, axis=1 (Llama model pattern)."""
        model = _make_reduce_sum_model(np.int64, [1, seq_len], [1], keepdims=1)

        rng = np.random.default_rng(55)
        x = rng.integers(0, 2, [1, seq_len], dtype=np.int64)

        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=0)

    # ------------------------------------------------------------------
    # Qwen3.5-9B (text.onnx) -- 48 ReduceSum ops on fp16 in linear-attn
    # q_l2norm / k_l2norm SumSq blocks.  Each one is:
    #
    #     in   : fp16 [B, S, 16, 128]    (per-head squared values)
    #     axes : i64 [-1]
    #     out  : fp16 [B, S, 16, 1]      (keepdims=1)
    #
    # H=16 (Q heads) and D=128 (head_dim).  These tests pin both the
    # decode (S=1) and prefill-ish (S=128) cases so we exercise the
    # narrow-reduce (D=128) fp16 accumulation path used by L2 norm.
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("seq_len", SEQ_LENS)
    def test_reduce_sum_qwen9b_l2norm_sumsq_shape(self, model_runner, seq_len):
        """fp16 ReduceSum over last dim: [1, S, 16, 128] -> [1, S, 16, 1]."""
        shape = [1, seq_len, 16, 128]
        model = _make_reduce_sum_model(np.float16, shape, [-1], keepdims=1)

        rng = np.random.default_rng(56)
        # Match the SumSq input distribution: pre-squared (non-negative),
        # bounded so the K=128 fp16 accumulation doesn't overflow.
        x = rng.uniform(0.0, 0.25, shape).astype(np.float16)

        actual, expected = model_runner.run_sample(model, [x])
        # fp16 accumulation over 128 partials of values up to 0.25
        # yields outputs around 16 with absolute fp16 ULP near 1e-2.
        compare_outputs(actual, expected, atol=2e-2, rtol=1e-2)


# ---------------------------------------------------------------------------
# ReduceMean -- first-class hip.reduce_mean op. The mean division happens
# in-kernel (reduce_size = num_input / num_output), so the op needs no static
# reduce dim. Runtime dtype: f16 only (ONNX ReduceMean is float-domain; the
# true-fp16 EP path only ever feeds half tensors).
# ---------------------------------------------------------------------------


class TestReduceMean:
    @pytest.mark.parametrize(
        "shape,axes,keepdims",
        [
            ([4, 8], [1], 1),
            ([4, 8], [1], 0),
            ([1, 256, 1152], [-1], 1),
        ],
    )
    def test_reduce_mean(self, model_runner, shape, axes, keepdims):
        model = _make_reduce_model("ReduceMean", np.float16, shape, axes, keepdims)
        rng = np.random.default_rng(401)
        x = rng.uniform(-3.0, 3.0, shape).astype(np.float16)
        actual, expected = model_runner.run_sample(model, [x])
        # fp16 accumulate-in-float then divide; allow a small ULP tolerance.
        compare_outputs(actual, expected, atol=2e-2, rtol=1e-2)

    @pytest.mark.parametrize("seq_len", SEQ_LENS)
    def test_reduce_mean_last_dim(self, model_runner, seq_len):
        """fp16 ReduceMean over last dim: [1, S, 16, 128] -> [1, S, 16, 1]."""
        shape = [1, seq_len, 16, 128]
        model = _make_reduce_model("ReduceMean", np.float16, shape, [-1], 1)
        rng = np.random.default_rng(402)
        x = rng.uniform(-2.0, 2.0, shape).astype(np.float16)
        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=2e-2, rtol=1e-2)

    def test_reduce_mean_strided_channel_axis(self, model_runner):
        """Non-trailing (strided) channel-axis mean over NCHW: axes=[1].
        Exercises the inner_size > 1 path (reduced elements are H*W apart)."""
        shape = [2, 8, 3, 5]
        model = _make_reduce_model("ReduceMean", np.float16, shape, [1], 1)
        rng = np.random.default_rng(403)
        x = rng.uniform(-2.0, 2.0, shape).astype(np.float16)
        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=2e-2, rtol=1e-2)


# ---------------------------------------------------------------------------
# ReduceMax (added by qwen-vision-kernels PR)
# Runtime dtypes: f16, i32, i64  (NO f32 -- not in the dispatch table)
# ---------------------------------------------------------------------------


class TestReduceMax:
    @pytest.mark.parametrize(
        "dtype,shape,axes,keepdims",
        [
            (np.float16, [4, 8], [1], 1),
            (np.float16, [4, 8], [1], 0),
            (np.int32, [1, 64], [1], 1),
            (np.int64, [1, 64], [1], 1),
        ],
    )
    def test_reduce_max(self, model_runner, dtype, shape, axes, keepdims):
        model = _make_reduce_model("ReduceMax", dtype, shape, axes, keepdims)
        rng = np.random.default_rng(301)
        if np.issubdtype(dtype, np.integer):
            x = rng.integers(-100, 100, shape, dtype=dtype)
        else:
            x = rng.uniform(-3.0, 3.0, shape).astype(dtype)
        actual, expected = model_runner.run_sample(model, [x])
        # Max is a select, not an arithmetic accumulator -- exact match
        # expected even in fp16.
        compare_outputs(actual, expected, atol=0)

    @pytest.mark.parametrize("seq_len", SEQ_LENS)
    def test_reduce_max_last_dim_llama_shape(self, model_runner, seq_len):
        """fp16 ReduceMax over last dim, mirroring a per-head max pattern."""
        shape = [1, seq_len, 16, 128]
        model = _make_reduce_model("ReduceMax", np.float16, shape, [-1], 1)
        rng = np.random.default_rng(302)
        x = rng.uniform(-2.0, 2.0, shape).astype(np.float16)
        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=0)

    def test_reduce_max_qwen35b_vision_full_reduce(self, model_runner):
        """ReduceMax exactly as it appears in Qwen3.5-35B vision.onnx:
            in: i64 [1, 2]
            no `axes` input, no `axes` attribute (full reduction)
            attrs: keepdims=0, noop_with_empty_axes=0
            out: i64 [] (scalar)

        This exercises the conversion-time `noop_with_empty_axes==0 + no
        axes` branch in ReduceMaxConversion.cpp (reduce over all axes),
        which is structurally different from the axes-as-input path
        covered above.
        """
        shape = [1, 2]
        tp = np_to_onnx_type(np.int64)
        X = helper.make_tensor_value_info("X", tp, shape)
        Y = helper.make_tensor_value_info("Y", tp, [])
        # No `axes` input, no `axes` attribute -- relies on
        # noop_with_empty_axes=0 to mean "reduce over every axis".
        node = helper.make_node(
            "ReduceMax",
            ["X"],
            ["Y"],
            keepdims=0,
            noop_with_empty_axes=0,
        )
        model = make_model_from_nodes([node], [X], [Y], opset=18)
        rng = np.random.default_rng(305)
        x = rng.integers(-100, 100, shape, dtype=np.int64)
        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=0)


# ---------------------------------------------------------------------------
# ReduceProd (added by qwen-vision-kernels PR)
# Runtime dtypes: f16, i32, i64
# ---------------------------------------------------------------------------


class TestReduceProd:
    @pytest.mark.parametrize(
        "dtype,shape,axes,keepdims",
        [
            (np.float16, [4, 8], [1], 1),
            (np.int32, [3, 4], [1], 1),
            (np.int64, [3, 4], [1], 0),
        ],
    )
    def test_reduce_prod(self, model_runner, dtype, shape, axes, keepdims):
        model = _make_reduce_model("ReduceProd", dtype, shape, axes, keepdims)
        rng = np.random.default_rng(303)
        if np.issubdtype(dtype, np.integer):
            # Tight bounds: prod of 8 ints in [-3, 3] stays in int32 safely
            # (max |prod| = 3^8 = 6561 << 2^31).
            x = rng.integers(-3, 4, shape, dtype=dtype)
        else:
            # fp16 cumulative product blows up FAST. With axis-size 8 we keep
            # |values| <= 1.2 so |prod| <= 1.2^8 ~= 4.3 -- well within fp16
            # dynamic range.
            x = rng.uniform(-1.2, 1.2, shape).astype(dtype)
        actual, expected = model_runner.run_sample(model, [x])
        if np.issubdtype(dtype, np.integer):
            atol = 0
        else:
            # fp16 ReduceProd kernel accumulates in fp32 and rounds once on
            # the output write -- end-to-end error is ~1 ULP of the fp16
            # output. ULP(4.3) in fp16 ~= 4e-3, so 5e-3 covers it with
            # slack for the small (~1e-6 rel) drift between our tree
            # reduction and ORT CPU's sequential one.
            atol = 5e-3
        compare_outputs(actual, expected, atol=atol, rtol=1e-3)

    def test_reduce_prod_small_axis_llama_shape(self, model_runner):
        """fp16 ReduceProd over a small (last) dim -- guards the f16 path
        without overflowing the dynamic range."""
        shape = [1, 1, 16, 4]
        model = _make_reduce_model("ReduceProd", np.float16, shape, [-1], 1)
        rng = np.random.default_rng(304)
        # |values| <= 1.1 -> |prod| <= 1.1^4 = 1.46 (safe in fp16).
        x = rng.uniform(-1.1, 1.1, shape).astype(np.float16)
        actual, expected = model_runner.run_sample(model, [x])
        # ULP(1.46) in fp16 ~= 1.4e-3; 2e-3 covers the single output-write
        # rounding (fp32-internal kernel).
        compare_outputs(actual, expected, atol=2e-3, rtol=1e-3)
