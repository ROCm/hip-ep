#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""Numeric tests for Whisper-large-v3 ENCODER self-attention.

Whisper's encoder uses the legacy fused-QKV ``com.microsoft.Attention`` op
(unidirectional=0 -> bidirectional). MorphiZen lowers it to
``hip.gqa(no_causal=true)`` with HPG=1 (num_heads == kv_num_heads) via a single
fused QKV projection + 3 slices (see AttentionConversion.cpp).

Reference strategy: a numpy fp32 bidirectional-MHA reference, NOT ORT CPU on the
same model.  fp32 reference avoids the fp16 CPU softmax accumulation artefact
seen elsewhere in this project (CPU fp16 attention can drift to cosine ~0.6).

Weight layout: ``[hidden, 3*hidden]`` = ``[H, 3H]`` — the ORT
com.microsoft.Attention spec convention (``[input_hidden,
q_hidden+k_hidden+v_hidden]``) and what Whisper-large-v3 actually exports
(``qkv_proj.weight`` is ``[1280, 3840]``).  MorphiZen's AttentionConversion
consumes this layout directly in the fused QKV matmul (no transpose); the numpy
reference projects with ``x @ qkv_w`` accordingly.

Whisper-large-v3 encoder shapes: d_model=1280, num_heads=20, head_dim=64,
S=1500.  We test small S (16, 64) for speed plus one S=256 case; the kernel is
shape-agnostic so S=1500 is not needed to prove correctness.
"""

import numpy as np
import pytest
from onnx import TensorProto, helper, numpy_helper

from framework.comparator import compare_outputs
from framework.onnx_utils import make_model_from_nodes

# Whisper-large-v3 encoder geometry.
HIDDEN = 1280
NUM_HEADS = 20
HEAD_DIM = HIDDEN // NUM_HEADS  # 64

# (hidden, num_heads) for every supported Whisper variant; head_dim == 64 for all
# (tiny 384/6, base 512/8, small 768/12, medium 1024/16, large-v3 & turbo
# 1280/20 — turbo shares large-v3's geometry, so 5 unique shapes). The attention
# kernel is shape-agnostic, so covering each variant's d_model/num_heads proves
# the per-variant GEMM/softmax shapes dispatch correctly.
VARIANT_SHAPES = [(384, 6), (512, 8), (768, 12), (1024, 16), (1280, 20)]


def _make_encoder_attention_model(
    batch, seq_len, hidden, num_heads, seed=0xA77, np_dtype=np.float16
):
    """Build a single-op com.microsoft.Attention model (fused QKV, bidirectional).

    Weight layout is the real Whisper / ORT-spec convention that MorphiZen's
    AttentionConversion expects: ``qkv_w`` is ``[hidden, 3*hidden]`` =
    ``[H, 3H]`` (``[input_hidden, q+k+v hidden]``) and ``qkv_b`` is
    ``[3*hidden]``.  Returns ``(model, qkv_w, qkv_b)`` so the test can feed the
    same weights to the numpy reference.

    ``np_dtype`` selects fp16 (default) or fp32; fp32 exercises the fp32
    decomposed GQA path (element_size_bytes=4) that the Whisper greedy-correct
    pipeline relies on.
    """
    qkv_hidden = 3 * hidden
    onnx_dtype = TensorProto.FLOAT if np_dtype == np.float32 else TensorProto.FLOAT16

    hidden_vi = helper.make_tensor_value_info(
        "hidden", onnx_dtype, [batch, seq_len, hidden]
    )
    out_vi = helper.make_tensor_value_info(
        "output", onnx_dtype, [batch, seq_len, hidden]
    )

    rng = np.random.default_rng(seed)
    # Modest range keeps softmax inside fp16's safe band.
    # [H, 3H] layout — the real Whisper qkv_proj.weight shape (= [1280, 3840]).
    qkv_w = (rng.standard_normal((hidden, qkv_hidden)) * 0.05).astype(np_dtype)
    qkv_b = (rng.standard_normal((qkv_hidden,)) * 0.05).astype(np_dtype)

    initializers = [
        numpy_helper.from_array(qkv_w, name="qkv_w"),
        numpy_helper.from_array(qkv_b, name="qkv_b"),
    ]

    scale = float(1.0 / np.sqrt(hidden // num_heads))
    # 7-input form (input, weights, bias, then 4 empty optional slots) matching
    # the Whisper encoder node and the LIT test test_whisper_encoder_attention.
    node = helper.make_node(
        "Attention",
        ["hidden", "qkv_w", "qkv_b", "", "", "", ""],
        ["output"],
        domain="com.microsoft",
        num_heads=num_heads,
        qkv_hidden_sizes=[hidden, hidden, hidden],
        unidirectional=0,
        do_rotary=0,
        past_present_share_buffer=0,
        rotary_embedding_dim=0,
        scale=scale,
        mask_filter_value=-10000.0,
    )
    ms_opset = helper.make_opsetid("com.microsoft", 1)
    model = make_model_from_nodes(
        [node],
        [hidden_vi],
        [out_vi],
        initializers=initializers,
        extra_opsets=[ms_opset],
    )
    return model, qkv_w, qkv_b


def _bidirectional_mha_ref(x, qkv_w, qkv_b, num_heads):
    """fp32 numpy bidirectional fused-QKV MHA reference.

    ``qkv_w`` is ``[H, 3H]`` (the real Whisper / ORT-spec layout); the
    projection is ``x @ qkv_w + qkv_b``.  Slices the projected ``[B, S, 3H]``
    activation into Q/K/V along the last axis exactly like AttentionConversion's
    three extract_slice ops, then runs standard softmax(QK^T / sqrt(d)) V with
    NO causal mask.
    """
    x = x.astype(np.float64)
    w = qkv_w.astype(np.float64)
    b = qkv_b.astype(np.float64)
    batch, seq_len, hidden = x.shape
    head_dim = hidden // num_heads

    qkv = x @ w + b  # [B, S, H] @ [H, 3H] -> [B, S, 3H]
    q = qkv[..., :hidden]
    k = qkv[..., hidden : 2 * hidden]
    v = qkv[..., 2 * hidden :]

    def to_heads(t):
        return t.reshape(batch, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)

    q, k, v = to_heads(q), to_heads(k), to_heads(v)
    scale = 1.0 / np.sqrt(head_dim)
    scores = (q @ k.transpose(0, 1, 3, 2)) * scale  # [B, nh, S, S]
    scores = scores - scores.max(axis=-1, keepdims=True)
    probs = np.exp(scores)
    probs = probs / probs.sum(axis=-1, keepdims=True)
    out = probs @ v  # [B, nh, S, d]
    return out.transpose(0, 2, 1, 3).reshape(batch, seq_len, hidden)


# NOTE (was a bug, fixed in the no_causal seqlens_k exemption): the no_causal
# GQA path used to have a seqlens_k off-by-one.  AttentionConversion.cpp emits a
# compile-time `seqlens_k = [S]` AND `total_seq = S`, but the runtime
# `gqa_forward_hipblaslt` applied the ORT decode convention
# `total_seq = seqlens_k[0] + 1` unconditionally.  For the encoder (S valid
# tokens, no padding) that over-counted by 1: `total_seq = S+1 > present_seq = S`
# -> the validation returned rc=-1, GQA left the output zero-initialised, and the
# result was garbage (cosine ~0).  The no_causal path had NO end-to-end runtime
# coverage before this test (LIT only checks IR), so the bug shipped unnoticed in
# Tasks 7/8.  The runtime now exempts no_causal from the +1 (total_seq = skv,
# past_len = 0) -- see lib/Runtime/real/gqa.cpp.  This test is the executable
# repro that locks the fix in.


class TestWhisperEncoderAttention:
    """com.microsoft.Attention (fused QKV, bidirectional) -> hip.gqa(no_causal)."""

    @pytest.mark.parametrize("hidden,num_heads", VARIANT_SHAPES)
    @pytest.mark.parametrize("seq_len", [16, 64, 256])
    def test_encoder_self_attention(self, model_runner, seq_len, hidden, num_heads):
        """Whisper encoder self-attn across every variant's d_model/num_heads.

        Bidirectional (unidirectional=0).  Compared against an fp32 numpy
        reference (see module docstring for why ORT CPU is unsuitable here).
        """
        batch = 1
        model, qkv_w, qkv_b = _make_encoder_attention_model(
            batch, seq_len, hidden, num_heads
        )

        rng = np.random.default_rng(seq_len * 1000 + hidden)
        x = (rng.standard_normal((batch, seq_len, hidden)) * 0.5).astype(np.float16)

        # Drive MorphiZen directly; CPU-on-same-model would mis-interpret the
        # weight layout (see module docstring).  session.disable_cpu_ep_fallback
        # is set by the backend, so this run also proves GPU dispatch.
        actual = model_runner.backend.run(_persist(model_runner, model), [x])
        expected = [_bidirectional_mha_ref(x, qkv_w, qkv_b, num_heads)]

        compare_outputs(actual, expected, atol=2e-2, rtol=2e-2, cos_threshold=0.999)

    @pytest.mark.parametrize("hidden,num_heads", VARIANT_SHAPES)
    @pytest.mark.parametrize("seq_len", [16, 64, 256])
    def test_encoder_self_attention_fp32(
        self, model_runner, seq_len, hidden, num_heads
    ):
        """fp32 variant: drives the fp32 decomposed GQA path (elem_size=4).

        Same geometry as the fp16 test but fp32 inputs/weights. The fp32 path
        is what makes Whisper greedy transcription correct on GPU (fp16 is
        argmax-lossy on the wide lm_head). Tighter thresholds than fp16 because
        fp32 GEMM + fp32 softmax should track the fp64 numpy reference closely.
        """
        batch = 1
        model, qkv_w, qkv_b = _make_encoder_attention_model(
            batch, seq_len, hidden, num_heads, np_dtype=np.float32
        )

        rng = np.random.default_rng(seq_len * 1000 + hidden)
        x = (rng.standard_normal((batch, seq_len, hidden)) * 0.5).astype(np.float32)

        actual = model_runner.backend.run(_persist(model_runner, model), [x])
        expected = [_bidirectional_mha_ref(x, qkv_w, qkv_b, num_heads)]

        compare_outputs(actual, expected, atol=2e-3, rtol=2e-3, cos_threshold=0.9999)


def _persist(model_runner, model):
    """Write *model* to a fresh per-test work subdir and return its path.

    Mirrors what ModelRunner.run_sample does internally; needed because we
    bypass run_sample to supply our own numpy reference instead of the CPU EP.
    """
    sub = model_runner._next_subdir()
    path = sub / "model.onnx"
    path.write_bytes(model.SerializeToString())
    return str(path)
