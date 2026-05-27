#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""Tests for com.microsoft.MultiHeadAttention custom op.

These tests target the standard "Q_K_V_BSNH" path the runtime currently
implements:

    query : [B, S_q,  N * H]   fp16
    key   : [B, S_kv, N * H]   fp16
    value : [B, S_kv, N * H]   fp16
    output: [B, S_q,  N * H]   fp16

  attributes used:
    num_heads
    scale  (default 1/sqrt(head_size) when 0.0)
    unidirectional  (0 = no mask, 1 = causal mask)

  not yet exercised here (validated by runtime to fail loudly):
    bias / key_padding_mask / attention_bias
    past_key / past_value / past_sequence_length / cache_indirection
    present_key / present_value / qk outputs
    packed QKV layouts (rank-5 query, rank-5 key)

Shapes are kept small for fast iteration plus one Llama-style shape sanity
case (batch=1, S=128, num_heads=32, head_dim=128 -> hidden=4096) to match
the LIT e2e test in onnx-hipdnn-ep.
"""

import numpy as np
import pytest
from onnx import TensorProto, helper

from framework.comparator import compare_outputs
from framework.onnx_utils import make_model_from_nodes

SEQ_LENS = [1, 128]


def _make_mha_model(
    batch: int,
    seq_q: int,
    seq_kv: int,
    num_heads: int,
    head_dim: int,
    scale: float = 0.0,
    unidirectional: int = 0,
):
    """Build a self-/cross-attention MHA ONNX model (Q_K_V_BSNH fp16).

    Cross-attention is exercised when ``seq_q != seq_kv``. ``scale=0.0`` keeps
    the runtime default (``1/sqrt(head_dim)``) -- matches GQA's convention
    and the LIT e2e test.
    """
    hidden = num_heads * head_dim

    query = helper.make_tensor_value_info(
        "query", TensorProto.FLOAT16, [batch, seq_q, hidden]
    )
    key = helper.make_tensor_value_info(
        "key", TensorProto.FLOAT16, [batch, seq_kv, hidden]
    )
    value = helper.make_tensor_value_info(
        "value", TensorProto.FLOAT16, [batch, seq_kv, hidden]
    )
    output = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT16, [batch, seq_q, hidden]
    )

    # Effective scale forwarded to the runtime: 0.0 means "auto-compute
    # 1/sqrt(head_dim) at runtime" (matches the GQA convention and the
    # default the lowering propagates when the attribute is absent).
    eff_scale = scale if scale != 0.0 else float(1.0 / np.sqrt(head_dim))

    node = helper.make_node(
        "MultiHeadAttention",
        ["query", "key", "value"],
        ["output"],
        domain="com.microsoft",
        num_heads=num_heads,
        scale=eff_scale,
        mask_filter_value=-10000.0,
        unidirectional=unidirectional,
    )
    ms_opset = helper.make_opsetid("com.microsoft", 1)
    return make_model_from_nodes(
        [node],
        [query, key, value],
        [output],
        extra_opsets=[ms_opset],
    )


def _qkv_inputs(batch, seq_q, seq_kv, num_heads, head_dim, seed=42, scale=1.0):
    """Sample Q/K/V tensors with a small, reproducible range.

    A modest range (`scale` ~ 1.0) keeps softmax inside fp16's safe band and
    matches what natural attention scores look like during inference.
    """
    rng = np.random.default_rng(seed)
    hidden = num_heads * head_dim
    q = rng.uniform(-scale, scale, [batch, seq_q, hidden]).astype(np.float16)
    k = rng.uniform(-scale, scale, [batch, seq_kv, hidden]).astype(np.float16)
    v = rng.uniform(-scale, scale, [batch, seq_kv, hidden]).astype(np.float16)
    return q, k, v


class TestMultiHeadAttention:
    """End-to-end correctness for the Q_K_V_BSNH MHA path."""

    @pytest.mark.parametrize("seq_q", SEQ_LENS)
    def test_self_attention_small(self, model_runner, seq_q):
        """Self-attention (S_q == S_kv) with a small head config.

        Shape: B=1, N=4, H=16 -> hidden=64. Catches kernel + GEMM stride
        bugs early at a size that exercises both transpose (S>1) and decode
        (S==1) code paths.
        """
        batch, num_heads, head_dim = 1, 4, 16
        model = _make_mha_model(batch, seq_q, seq_q, num_heads, head_dim)

        q, k, v = _qkv_inputs(batch, seq_q, seq_q, num_heads, head_dim)

        actual, expected = model_runner.run_sample(model, [q, k, v])
        compare_outputs(actual, expected, atol=1e-2, rtol=1e-2, cos_threshold=0.999)

    @pytest.mark.parametrize("seq_q", SEQ_LENS)
    def test_self_attention_causal_small(self, model_runner, seq_q):
        """Self-attention with unidirectional=1 (causal mask).

        Same small shape as ``test_self_attention_small`` but
        ``unidirectional=1`` so the runtime applies the lower-triangular
        causal mask before softmax. At S==1 the mask is a no-op (a single
        query attends to a single key), so this test also covers the
        S_q==1 + unidirectional=1 corner.
        """
        batch, num_heads, head_dim = 1, 4, 16
        model = _make_mha_model(
            batch,
            seq_q,
            seq_q,
            num_heads,
            head_dim,
            unidirectional=1,
        )

        q, k, v = _qkv_inputs(batch, seq_q, seq_q, num_heads, head_dim)

        actual, expected = model_runner.run_sample(model, [q, k, v])
        compare_outputs(actual, expected, atol=1e-2, rtol=1e-2, cos_threshold=0.999)

    def test_cross_attention_small(self, model_runner):
        """Cross-attention: S_q != S_kv.

        The runtime takes the BSHD->BNSD transpose path on both the query
        (S_q=4 != 1) and the K/V (S_kv=6 != 1), so this exercises the
        general transpose / strided-batched GEMM combination.
        """
        batch, num_heads, head_dim = 1, 4, 16
        seq_q, seq_kv = 4, 6
        model = _make_mha_model(batch, seq_q, seq_kv, num_heads, head_dim)

        q, k, v = _qkv_inputs(batch, seq_q, seq_kv, num_heads, head_dim)

        actual, expected = model_runner.run_sample(model, [q, k, v])
        compare_outputs(actual, expected, atol=1e-2, rtol=1e-2, cos_threshold=0.999)

    def test_batched_self_attention(self, model_runner):
        """Batch > 1, exercising the strided-batched GEMM batch dimension."""
        batch, seq, num_heads, head_dim = 2, 8, 4, 32
        model = _make_mha_model(batch, seq, seq, num_heads, head_dim)

        q, k, v = _qkv_inputs(batch, seq, seq, num_heads, head_dim)

        actual, expected = model_runner.run_sample(model, [q, k, v])
        compare_outputs(actual, expected, atol=1e-2, rtol=1e-2, cos_threshold=0.999)

    def test_self_attention_llama_shape(self, model_runner):
        """Self-attention at the Llama-style shape used in the LIT e2e test.

        B=1, S=128, N=32, H=128 -> hidden=4096. Exercises the realistic
        prefill workload: 4 KB-tile Score GEMM and Value GEMM, B*N=32
        batch count for the strided-batched matmul.
        """
        batch, seq, num_heads, head_dim = 1, 128, 32, 128
        model = _make_mha_model(batch, seq, seq, num_heads, head_dim)

        q, k, v = _qkv_inputs(batch, seq, seq, num_heads, head_dim)

        actual, expected = model_runner.run_sample(model, [q, k, v])
        # Slightly more forgiving thresholds at hidden=4096 because the
        # softmax + Value-GEMM accumulation stretches fp16 precision.
        compare_outputs(actual, expected, atol=2e-2, rtol=2e-2, cos_threshold=0.998)
