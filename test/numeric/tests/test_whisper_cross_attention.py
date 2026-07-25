#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""Numeric tests for Whisper-large-v3 decoder CROSS-attention.

Whisper's decoder cross-attn uses the 8-input ``com.microsoft.MultiHeadAttention``
form: Q from the decoder (3-D ``[B, Sq, H*d]``) and K/V from the encoder output
already split into BNSH (``[B, H, Skv, d]``), with the 5 trailing optional slots
(bias / mask / attention_bias / past_key / past_value) empty and
``unidirectional=0`` (cross-attn is bidirectional).  MorphiZen lowers this to
``hip.gqa(no_causal=true)`` with HPG=1 and a compile-time constant
``seqlens_k = [Skv]`` (see MultiHeadAttentionConversion.cpp, branch 1).

Operand layout is taken from the Task-8 LIT fixture
``test/lit/Conversion/onnx-to-hip/test_whisper_cross_mha.mlir``:
K/V are rank-4 BNSH — the converter rejects rank-3 cross-attn keys.

Reference: ORT CPU EP runs the SAME 8-input MHA model with bidirectional
attention, so the standard run_sample CPU comparison applies directly (no
weight-convention divergence, unlike the encoder Attention op).
"""

import numpy as np
import pytest
from onnx import TensorProto, helper

from framework.comparator import compare_outputs
from framework.onnx_utils import make_model_from_nodes

# Whisper-large-v3 decoder geometry.
NUM_HEADS = 20
HEAD_DIM = 64
HIDDEN = NUM_HEADS * HEAD_DIM  # 1280

# num_heads for every supported variant (head_dim == 64 for all): tiny 6, base 8,
# small 12, medium 16, large-v3 & turbo 20. Cross-attn hidden = num_heads*64.
VARIANT_HEADS = [6, 8, 12, 16, 20]


def _make_cross_mha_model(
    batch, seq_q, seq_kv, num_heads, head_dim, np_dtype=np.float16
):
    """Build an 8-input com.microsoft.MultiHeadAttention cross-attn model.

    Q is 3-D ``[B, Sq, H*d]``; K/V are rank-4 BNSH ``[B, H, Skv, d]`` (the
    Whisper cross-attn layout the converter expects).  Output is ``[B, Sq, H*d]``.

    ``np_dtype`` selects fp16 (default) or fp32; fp32 drives the fp32 decomposed
    GQA path (element_size_bytes=4).
    """
    hidden = num_heads * head_dim
    onnx_dtype = TensorProto.FLOAT if np_dtype == np.float32 else TensorProto.FLOAT16

    query = helper.make_tensor_value_info("query", onnx_dtype, [batch, seq_q, hidden])
    key = helper.make_tensor_value_info(
        "key", onnx_dtype, [batch, num_heads, seq_kv, head_dim]
    )
    value = helper.make_tensor_value_info(
        "value", onnx_dtype, [batch, num_heads, seq_kv, head_dim]
    )
    output = helper.make_tensor_value_info("output", onnx_dtype, [batch, seq_q, hidden])

    scale = float(1.0 / np.sqrt(head_dim))
    # 8-input form: Q, K, V, then 5 empty optional slots (bias, key_padding_mask,
    # attention_bias, past_key, past_value).  This is the cross-attn dispatch
    # signal for MultiHeadAttentionConversion branch 1.
    node = helper.make_node(
        "MultiHeadAttention",
        ["query", "key", "value", "", "", "", "", ""],
        ["output"],
        domain="com.microsoft",
        num_heads=num_heads,
        scale=scale,
        unidirectional=0,
        mask_filter_value=-10000.0,
    )
    ms_opset = helper.make_opsetid("com.microsoft", 1)
    return make_model_from_nodes(
        [node],
        [query, key, value],
        [output],
        extra_opsets=[ms_opset],
    )


def _cross_inputs(
    batch, seq_q, seq_kv, num_heads, head_dim, seed=7, np_dtype=np.float16
):
    rng = np.random.default_rng(seed)
    hidden = num_heads * head_dim
    # Modest range keeps softmax inside fp16's safe band.
    q = rng.uniform(-1.0, 1.0, [batch, seq_q, hidden]).astype(np_dtype)
    k = rng.uniform(-1.0, 1.0, [batch, num_heads, seq_kv, head_dim]).astype(np_dtype)
    v = rng.uniform(-1.0, 1.0, [batch, num_heads, seq_kv, head_dim]).astype(np_dtype)
    return q, k, v


# NOTE (was a bug, fixed in the no_causal seqlens_k exemption): same no_causal
# GQA seqlens_k off-by-one as the encoder.  MultiHeadAttentionConversion.cpp
# branch 1 emits a compile-time `seqlens_k = [Skv]` and `total_seq = Skv`, but
# the runtime `gqa_forward_hipblaslt` used to apply `total_seq = seqlens_k[0] + 1`
# -> over-counted by 1 (Skv+1 > present_seq=Skv) -> rc=-1 / zeroed output.
# Cross-attn additionally has Skv != seq_q (K/V are the full rank-4 BNSH encoder
# output, seq_q is the decoder query count), so the past_len = total_seq - sq
# derivation was nonsense too.  The runtime now exempts no_causal: total_seq =
# skv, past_len = 0, KV populated by a direct BNSH copy of all Skv keys (see
# lib/Runtime/real/gqa.cpp update_kv_cache no_causal branch).  Cross-attn is
# bidirectional with the full encoder output valid, so no_causal is correct.
# This test is the executable repro that locks the fix in.


class TestWhisperCrossAttention:
    """8-input MultiHeadAttention cross-attn -> hip.gqa(no_causal=true)."""

    @pytest.mark.parametrize("num_heads", VARIANT_HEADS)
    @pytest.mark.parametrize("seq_q", [1, 4])
    @pytest.mark.parametrize("seq_kv", [64, 256])
    def test_cross_attention(self, model_runner, seq_q, seq_kv, num_heads):
        """Whisper decoder cross-attn across every variant's num_heads.

        Real Whisper Skv=1500; we test 64 / 256 for speed (kernel is
        shape-agnostic).  Bidirectional, compared directly against ORT CPU.
        """
        batch = 1
        model = _make_cross_mha_model(batch, seq_q, seq_kv, num_heads, HEAD_DIM)
        q, k, v = _cross_inputs(batch, seq_q, seq_kv, num_heads, HEAD_DIM)

        actual, expected = model_runner.run_sample(model, [q, k, v])
        compare_outputs(actual, expected, atol=2e-2, rtol=2e-2, cos_threshold=0.999)

    @pytest.mark.parametrize("num_heads", VARIANT_HEADS)
    @pytest.mark.parametrize("seq_q", [1, 4])
    @pytest.mark.parametrize("seq_kv", [64, 256])
    def test_cross_attention_fp32(self, model_runner, seq_q, seq_kv, num_heads):
        """fp32 variant: drives the fp32 decomposed GQA path (elem_size=4).

        seq_q=1 / seq_kv>1 exercises the no_causal D2D-copy KV population +
        no-expand decode GEMM in fp32 (the decoder cross-attn decode shape).
        Compared directly against ORT CPU fp32 MHA; tighter thresholds than the
        fp16 test because both sides are fp32.
        """
        batch = 1
        model = _make_cross_mha_model(
            batch, seq_q, seq_kv, num_heads, HEAD_DIM, np_dtype=np.float32
        )
        q, k, v = _cross_inputs(
            batch, seq_q, seq_kv, num_heads, HEAD_DIM, np_dtype=np.float32
        )

        actual, expected = model_runner.run_sample(model, [q, k, v])
        compare_outputs(actual, expected, atol=2e-3, rtol=2e-3, cos_threshold=0.9999)
