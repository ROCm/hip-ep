#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""Tests for elementwise operations: Sub, Mul, Add.

The Llama-3.1-8B-fixed model uses:
  Sub: [1, 1] - [1] (broadcast) in the attention mask subgraph
  Mul: [1, S, 14336] * [1, S, 14336] for the SiLU gate (gate * sigmoid)
"""

import numpy as np
import pytest
from onnx import helper

from framework.comparator import compare_outputs
from framework.onnx_utils import make_model_from_nodes, np_to_onnx_type

# Sequence lengths chosen to cover:
#   * S=1   : decode (e.g. chunk_opt/decode_p512m16384.onnx)
#   * S=128 : moderate prefill / smoke
SEQ_LENS = [1, 128]
INTERMEDIATE = 14336


def _make_binary_model(op_type: str, dtype, shape: list[int]):
    """Build a 2-input elementwise ONNX model (same shape for both inputs)."""
    tp = np_to_onnx_type(dtype)
    X = helper.make_tensor_value_info("X", tp, shape)
    Y = helper.make_tensor_value_info("Y", tp, shape)
    Z = helper.make_tensor_value_info("Z", tp, shape)
    node = helper.make_node(op_type, ["X", "Y"], ["Z"])
    return make_model_from_nodes([node], [X, Y], [Z])


def _make_broadcast_binary_model(
    op_type: str,
    dtype,
    lhs_shape: list[int],
    rhs_shape: list[int],
    out_shape: list[int],
):
    """Build a 2-input elementwise ONNX model with broadcasting."""
    tp = np_to_onnx_type(dtype)
    X = helper.make_tensor_value_info("X", tp, lhs_shape)
    Y = helper.make_tensor_value_info("Y", tp, rhs_shape)
    Z = helper.make_tensor_value_info("Z", tp, out_shape)
    node = helper.make_node(op_type, ["X", "Y"], ["Z"])
    return make_model_from_nodes([node], [X, Y], [Z])


class TestElementwiseSub:
    @pytest.mark.parametrize(
        "dtype,shape",
        [
            (np.int64, [1, 1]),
            (np.int64, [2, 128]),
        ],
    )
    def test_sub(self, model_runner, dtype, shape):
        model = _make_binary_model("Sub", dtype, shape)

        rng = np.random.default_rng(42)
        if np.issubdtype(dtype, np.integer):
            lhs = rng.integers(-100, 100, shape, dtype=dtype)
            rhs = rng.integers(-100, 100, shape, dtype=dtype)
        else:
            lhs = rng.uniform(-10, 10, shape).astype(dtype)
            rhs = rng.uniform(-10, 10, shape).astype(dtype)

        actual, expected = model_runner.run_sample(model, [lhs, rhs])
        atol = 0 if np.issubdtype(dtype, np.integer) else 1e-5
        compare_outputs(actual, expected, atol=atol)

    @pytest.mark.parametrize("seq_len", SEQ_LENS)
    def test_sub_llama_shape(self, model_runner, seq_len):
        """Sub i64 with shapes matching Llama attention mask subgraph."""
        shape = [1, seq_len]
        model = _make_binary_model("Sub", np.int64, shape)

        rng = np.random.default_rng(42)
        lhs = rng.integers(0, seq_len + 1, shape, dtype=np.int64)
        rhs = rng.integers(0, seq_len + 1, shape, dtype=np.int64)

        actual, expected = model_runner.run_sample(model, [lhs, rhs])
        compare_outputs(actual, expected, atol=0)

    def test_sub_broadcast_llama_shape(self, model_runner):
        """Sub i64 broadcast [1, 1] - [1] matching Llama attn mask preprocessing.

        The model computes ReduceSum(attention_mask) -> [1, 1], then subtracts
        a scalar constant [1] (1D tensor with value 1) to get seqlens_k.
        """
        model = _make_broadcast_binary_model(
            "Sub",
            np.int64,
            [1, 1],
            [1],
            [1, 1],
        )

        rng = np.random.default_rng(42)
        lhs = rng.integers(1, 257, [1, 1], dtype=np.int64)
        rhs = np.array([1], dtype=np.int64)

        actual, expected = model_runner.run_sample(model, [lhs, rhs])
        compare_outputs(actual, expected, atol=0)


class TestElementwiseMul:
    @pytest.mark.parametrize(
        "dtype,shape",
        [
            (np.float16, [2, 32]),
            (np.float16, [4, 256]),
        ],
    )
    def test_mul(self, model_runner, dtype, shape):
        model = _make_binary_model("Mul", dtype, shape)

        rng = np.random.default_rng(77)
        lhs = rng.uniform(-2, 2, shape).astype(dtype)
        rhs = rng.uniform(-2, 2, shape).astype(dtype)

        actual, expected = model_runner.run_sample(model, [lhs, rhs])
        compare_outputs(actual, expected, atol=1e-4)

    @pytest.mark.parametrize("seq_len", SEQ_LENS)
    def test_mul_llama_shape(self, model_runner, seq_len):
        """Mul with shapes matching Llama MLP SiLU gate (gate * sigmoid)."""
        shape = [1, seq_len, INTERMEDIATE]
        model = _make_binary_model("Mul", np.float16, shape)

        rng = np.random.default_rng(77)
        lhs = rng.uniform(-2, 2, shape).astype(np.float16)
        rhs = rng.uniform(-2, 2, shape).astype(np.float16)

        actual, expected = model_runner.run_sample(model, [lhs, rhs])
        compare_outputs(actual, expected, atol=1e-4)

    def test_mul_5d_broadcast(self, model_runner):
        """Rank-5 broadcast matching SimpleBEV feature-volume masking."""
        lhs_shape = [2, 3, 4, 2, 5]
        rhs_shape = [2, 1, 4, 2, 5]
        model = _make_broadcast_binary_model(
            "Mul", np.float32, lhs_shape, rhs_shape, lhs_shape
        )

        rng = np.random.default_rng(78)
        lhs = rng.uniform(-2, 2, lhs_shape).astype(np.float32)
        rhs = rng.uniform(-2, 2, rhs_shape).astype(np.float32)

        actual, expected = model_runner.run_sample(model, [lhs, rhs])
        compare_outputs(actual, expected, atol=1e-6)

    def test_mul_6d_same_shape(self, model_runner):
        """Rank-6 same-shape multiplication packed through the 4-D path."""
        shape = [1, 2, 3, 2, 3, 4]
        model = _make_binary_model("Mul", np.float16, shape)

        rng = np.random.default_rng(79)
        lhs = rng.uniform(-2, 2, shape).astype(np.float16)
        rhs = rng.uniform(-2, 2, shape).astype(np.float16)

        actual, expected = model_runner.run_sample(model, [lhs, rhs])
        compare_outputs(actual, expected, atol=1e-3)


class TestElementwiseAdd:
    """Add op coverage matching the per-projection bias adds in
    prefill_p3000m4096.onnx.  Each Add takes the projection output
    [1, S, N] and broadcasts a learned bias vector [N]."""

    @pytest.mark.parametrize(
        "dtype,shape",
        [
            (np.float16, [2, 32]),
            (np.float16, [4, 256]),
        ],
    )
    def test_add(self, model_runner, dtype, shape):
        """Add with small same-shape operands for fast sanity checking."""
        model = _make_binary_model("Add", dtype, shape)

        rng = np.random.default_rng(123)
        lhs = rng.uniform(-2, 2, shape).astype(dtype)
        rhs = rng.uniform(-2, 2, shape).astype(dtype)

        actual, expected = model_runner.run_sample(model, [lhs, rhs])
        compare_outputs(actual, expected, atol=1e-4)

    # ------------------------------------------------------------------
    # Qwen3.5-9B (text.onnx) -- bias adds in the rotary-emb / linear-attn
    # subgraphs.  Two distinct broadcast signatures appear:
    #
    #   x48  fp16 Add  [B, S, 16, 1] + [1]    -> [B, S, 16, 1]   (4-D + scalar)
    #   x24  fp32 Add  [B, S, 32]    + [32]   -> [B, S, 32]      (3-D + last-dim)
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("seq_len", SEQ_LENS)
    def test_add_qwen9b_l2norm_eps_shape(self, model_runner, seq_len):
        """fp16 Add: [1, S, 16, 1] + [1] -> [1, S, 16, 1]   (L2-norm epsilon)."""
        lhs_shape = [1, seq_len, 16, 1]
        rhs_shape = [1]
        model = _make_broadcast_binary_model(
            "Add",
            np.float16,
            lhs_shape,
            rhs_shape,
            lhs_shape,
        )

        rng = np.random.default_rng(124)
        lhs = rng.uniform(0.0, 16.0, lhs_shape).astype(np.float16)
        rhs = np.array([1e-5], dtype=np.float16)

        actual, expected = model_runner.run_sample(model, [lhs, rhs])
        compare_outputs(actual, expected, atol=1e-3)

    @pytest.mark.parametrize("seq_len", SEQ_LENS)
    def test_add_qwen9b_rope_bias_shape(self, model_runner, seq_len):
        """fp32 Add: [1, S, 32] + [32] -> [1, S, 32]   (RoPE per-dim bias)."""
        lhs_shape = [1, seq_len, 32]
        rhs_shape = [32]
        model = _make_broadcast_binary_model(
            "Add",
            np.float32,
            lhs_shape,
            rhs_shape,
            lhs_shape,
        )

        rng = np.random.default_rng(125)
        lhs = rng.uniform(-1, 1, lhs_shape).astype(np.float32)
        rhs = rng.uniform(-1, 1, rhs_shape).astype(np.float32)

        actual, expected = model_runner.run_sample(model, [lhs, rhs])
        compare_outputs(actual, expected, atol=1e-6)

    # ------------------------------------------------------------------
    # Qwen3.5-9B Mul broadcast cases (covered by the new 3-way
    # broadcast logic in elementwise.cpp):
    #
    #   x48  fp16 Mul  [B, S, 16, 128] * [B, S, 16, 1]   -- per-head L2 scale
    #   x24  fp16 Mul  [B, S, 2048]    * [1]             -- scalar broadcast
    #   x24  fp32 Mul  [32]            * [B, S, 32]      -- 3-D + last-dim
    # ------------------------------------------------------------------


class TestElementwiseMulBroadcast:
    """Mul cases that go through the new three-way broadcast path."""

    @pytest.mark.parametrize("seq_len", SEQ_LENS)
    def test_mul_qwen9b_l2norm_scale_4d(self, model_runner, seq_len):
        """fp16 Mul: [1, S, 16, 128] * [1, S, 16, 1] -> [1, S, 16, 128]."""
        full_shape = [1, seq_len, 16, 128]
        scale_shape = [1, seq_len, 16, 1]
        model = _make_broadcast_binary_model(
            "Mul",
            np.float16,
            full_shape,
            scale_shape,
            full_shape,
        )

        rng = np.random.default_rng(126)
        lhs = rng.uniform(-1, 1, full_shape).astype(np.float16)
        rhs = rng.uniform(0.5, 1.5, scale_shape).astype(np.float16)

        actual, expected = model_runner.run_sample(model, [lhs, rhs])
        compare_outputs(actual, expected, atol=1e-3)

    @pytest.mark.parametrize("seq_len", SEQ_LENS)
    def test_mul_qwen9b_scalar_bcast_3d(self, model_runner, seq_len):
        """fp16 Mul: [1, S, 2048] * [1] -> [1, S, 2048]   (scalar broadcast)."""
        lhs_shape = [1, seq_len, 2048]
        rhs_shape = [1]
        model = _make_broadcast_binary_model(
            "Mul",
            np.float16,
            lhs_shape,
            rhs_shape,
            lhs_shape,
        )

        rng = np.random.default_rng(127)
        lhs = rng.uniform(-1, 1, lhs_shape).astype(np.float16)
        rhs = np.array([0.5], dtype=np.float16)

        actual, expected = model_runner.run_sample(model, [lhs, rhs])
        compare_outputs(actual, expected, atol=1e-4)

    @pytest.mark.parametrize("seq_len", SEQ_LENS)
    def test_mul_chunk_opt_lastdim_bcast_3d(self, model_runner, seq_len):
        """fp16 Mul: [1, S, 2048] * [1, S, 1] -> [1, S, 2048]   (last-dim singleton).

        Appears x40 in chunk_opt/{prefill,decode}_p512m16384.onnx as the
        per-token L2-norm rescale on the 2048-wide hidden state.  Differs
        from the ``scalar_bcast`` case above in that the singleton lives
        on the *last* axis only -- a different broadcast iterator path
        in the runtime.
        """
        lhs_shape = [1, seq_len, 2048]
        rhs_shape = [1, seq_len, 1]
        model = _make_broadcast_binary_model(
            "Mul",
            np.float16,
            lhs_shape,
            rhs_shape,
            lhs_shape,
        )

        rng = np.random.default_rng(129)
        lhs = rng.uniform(-1, 1, lhs_shape).astype(np.float16)
        rhs = rng.uniform(0.5, 1.5, rhs_shape).astype(np.float16)

        actual, expected = model_runner.run_sample(model, [lhs, rhs])
        compare_outputs(actual, expected, atol=1e-4)

    @pytest.mark.parametrize("seq_len", SEQ_LENS)
    def test_mul_qwen9b_rope_perdim_3d(self, model_runner, seq_len):
        """fp32 Mul: [32] * [1, S, 32] -> [1, S, 32]   (broadcast on lhs)."""
        lhs_shape = [32]
        rhs_shape = [1, seq_len, 32]
        out_shape = rhs_shape
        model = _make_broadcast_binary_model(
            "Mul",
            np.float32,
            lhs_shape,
            rhs_shape,
            out_shape,
        )

        rng = np.random.default_rng(128)
        lhs = rng.uniform(-1, 1, lhs_shape).astype(np.float32)
        rhs = rng.uniform(-1, 1, rhs_shape).astype(np.float32)

        actual, expected = model_runner.run_sample(model, [lhs, rhs])
        compare_outputs(actual, expected, atol=1e-6)
