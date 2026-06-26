#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""Numeric tests for Whisper-large-v3 decoder SELF-attention (with past KV).

After the Task-9 ONNX surgery, Whisper's decoder self-attn is the 9-input
``com.microsoft.MultiHeadAttention`` form: Q/K/V 3-D ``[B, Sq, H*d]``,
past_key/past_value BNSH ``[B, H, 448, d]`` (pre-allocated shared buffer),
and an injected ``past_sequence_length`` (1-D i32) operand, with
``unidirectional=1`` (causal).  MorphiZen
lowers this to ``hip.gqa(no_causal=false)`` with HPG=1, ``seqlens_k =
operand[8]`` (the runtime past_sequence_length) and ``total_seq_len`` baked to
the static cache buffer length (448) — see MultiHeadAttentionConversion.cpp
branch 2 and the Task-8 LIT fixture test_whisper_self_mha_with_past.mlir.

Reference strategy (FALLBACK OPTION 2 from the task brief): ORT CPU does NOT
accept past_key/value of a DIFFERENT shape than present (the shared-buffer
448-slot form with a partial valid prefix has no single-op ORT CPU equivalent).
So we build a SEPARATE reference model: the standard 8-input MHA-with-past (no
share-buffer) whose ``past_key/value`` carry exactly the ``past_seq_val`` valid
prefix.  The
two are mathematically equal for the decode output:
  - MorphiZen 9-input: attends Q over KV slots [0, past_seq_val] of the 448
    buffer (valid prefix + the freshly-appended token).
  - CPU 8-input: attends Q over its [past_seq_val] past tokens + the new token.
Both reduce over the identical KV content, so ``output`` matches.  (We compare
``output`` only; the present-KV buffers have different shapes — 448-slot shared
vs past_seq_val+1 concat — and the shared buffer's tail is intentionally
garbage.)  Equivalence validated offline against a numpy causal-decode reference.
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
CACHE = 448  # Whisper decoder max KV buffer (BNSH dim 2) — same for every variant

# num_heads for every supported variant (head_dim 64, cache 448 for all): tiny 6,
# base 8, small 12, medium 16, large-v3 & turbo 20.
VARIANT_HEADS = [6, 8, 12, 16, 20]


def _make_self_mha_share_buffer_model(batch, seq_q, cache, num_heads, head_dim):
    """9-input MHA, past_present_share_buffer=1 (post-Task-9-surgery form).

    past_key/value are BNSH ``[B, H, cache, d]`` (shared buffer); present_key/
    value mirror that shape; the 9th input is ``past_sequence_length`` (1-D i32).
    """
    hidden = num_heads * head_dim

    query = helper.make_tensor_value_info(
        "query", TensorProto.FLOAT16, [batch, seq_q, hidden]
    )
    key = helper.make_tensor_value_info(
        "key", TensorProto.FLOAT16, [batch, seq_q, hidden]
    )
    value = helper.make_tensor_value_info(
        "value", TensorProto.FLOAT16, [batch, seq_q, hidden]
    )
    past_key = helper.make_tensor_value_info(
        "past_key", TensorProto.FLOAT16, [batch, num_heads, cache, head_dim]
    )
    past_value = helper.make_tensor_value_info(
        "past_value", TensorProto.FLOAT16, [batch, num_heads, cache, head_dim]
    )
    past_seq = helper.make_tensor_value_info(
        "past_sequence_length", TensorProto.INT32, [1]
    )

    output = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT16, [batch, seq_q, hidden]
    )
    present_key = helper.make_tensor_value_info(
        "present_key", TensorProto.FLOAT16, [batch, num_heads, cache, head_dim]
    )
    present_value = helper.make_tensor_value_info(
        "present_value", TensorProto.FLOAT16, [batch, num_heads, cache, head_dim]
    )

    scale = float(1.0 / np.sqrt(head_dim))
    # NO past_present_share_buffer attribute: ORT's MHA schema rejects it (it
    # lives on GroupQueryAttention/Attention). The 9-input form with the slot-8
    # past_sequence_length operand is the post-surgery signal MorphiZen keys on
    # (MultiHeadAttentionConversion.cpp branch 2), and it IS ORT-loadable.
    node = helper.make_node(
        "MultiHeadAttention",
        [
            "query",
            "key",
            "value",
            "",
            "",
            "",
            "past_key",
            "past_value",
            "past_sequence_length",
        ],
        ["output", "present_key", "present_value"],
        domain="com.microsoft",
        num_heads=num_heads,
        scale=scale,
        unidirectional=1,
        mask_filter_value=-10000.0,
    )
    ms_opset = helper.make_opsetid("com.microsoft", 1)
    return make_model_from_nodes(
        [node],
        [query, key, value, past_key, past_value, past_seq],
        [output, present_key, present_value],
        extra_opsets=[ms_opset],
    )


def _make_self_mha_plain_past_model(batch, seq_q, past, num_heads, head_dim):
    """Standard 8-input MHA-with-past (no share-buffer) — the CPU reference.

    past_key/value are BNSH ``[B, H, past, d]`` carrying exactly the valid
    prefix; present_key/value are ``[B, H, past+seq_q, d]`` (concat).  ORT CPU
    accepts this form; we compare only its ``output``.
    """
    hidden = num_heads * head_dim

    query = helper.make_tensor_value_info(
        "query", TensorProto.FLOAT16, [batch, seq_q, hidden]
    )
    key = helper.make_tensor_value_info(
        "key", TensorProto.FLOAT16, [batch, seq_q, hidden]
    )
    value = helper.make_tensor_value_info(
        "value", TensorProto.FLOAT16, [batch, seq_q, hidden]
    )
    past_key = helper.make_tensor_value_info(
        "past_key", TensorProto.FLOAT16, [batch, num_heads, past, head_dim]
    )
    past_value = helper.make_tensor_value_info(
        "past_value", TensorProto.FLOAT16, [batch, num_heads, past, head_dim]
    )

    output = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT16, [batch, seq_q, hidden]
    )
    present_key = helper.make_tensor_value_info(
        "present_key",
        TensorProto.FLOAT16,
        [batch, num_heads, past + seq_q, head_dim],
    )
    present_value = helper.make_tensor_value_info(
        "present_value",
        TensorProto.FLOAT16,
        [batch, num_heads, past + seq_q, head_dim],
    )

    scale = float(1.0 / np.sqrt(head_dim))
    node = helper.make_node(
        "MultiHeadAttention",
        ["query", "key", "value", "", "", "", "past_key", "past_value"],
        ["output", "present_key", "present_value"],
        domain="com.microsoft",
        num_heads=num_heads,
        scale=scale,
        unidirectional=1,
        mask_filter_value=-10000.0,
    )
    ms_opset = helper.make_opsetid("com.microsoft", 1)
    return make_model_from_nodes(
        [node],
        [query, key, value, past_key, past_value],
        [output, present_key, present_value],
        extra_opsets=[ms_opset],
    )


# HISTORICAL NOTE (Task 11 → resolved in Task 12): an earlier surgery emitted a
# `past_present_share_buffer=1` ATTRIBUTE on the MHA node, which ORT's registered
# `com.microsoft.MultiHeadAttention` schema does NOT define — ORT rejected the
# model at graph-load with "Unrecognized attribute: past_present_share_buffer"
# before any EP saw it, so this standalone numeric test had to be skipped.
# Task 12 dropped that attribute: the 9-input form with the slot-8
# `past_sequence_length` OPERAND is now the sole post-surgery signal (the operand
# is schema-valid, so ORT loads it), and MultiHeadAttentionConversion.cpp branch
# 2 keys on operand presence. The standalone model is therefore loadable now and
# this test runs (CPU reference = standard 8-input MHA-with-past, fallback
# option 2; equivalence validated offline against a numpy causal-decode
# reference, max diff ~3e-4).


class TestWhisperSelfAttention:
    """9-input share-buffer MHA -> hip.gqa(no_causal=false), HPG=1, decode."""

    @pytest.mark.parametrize("num_heads", VARIANT_HEADS)
    @pytest.mark.parametrize("past_seq_val", [1, 64, 256, 447])
    def test_self_attention_decode(self, model_runner, past_seq_val, num_heads):
        """Whisper decoder self-attn decode (Sq=1) across variant num_heads.

        MorphiZen runs the 9-input share-buffer form (448-slot buffer, valid
        prefix = past_seq_val).  Reference = ORT CPU on the standard 8-input
        MHA-with-past whose past tensor is exactly the valid prefix.
        """
        batch, seq_q = 1, 1
        hidden = num_heads * HEAD_DIM

        rng = np.random.default_rng(past_seq_val * 100 + num_heads)
        q = rng.uniform(-1.0, 1.0, [batch, seq_q, hidden]).astype(np.float16)
        k = rng.uniform(-1.0, 1.0, [batch, seq_q, hidden]).astype(np.float16)
        v = rng.uniform(-1.0, 1.0, [batch, seq_q, hidden]).astype(np.float16)

        # Valid KV prefix shared by both models.
        pk_valid = rng.uniform(
            -1.0, 1.0, [batch, num_heads, past_seq_val, HEAD_DIM]
        ).astype(np.float16)
        pv_valid = rng.uniform(
            -1.0, 1.0, [batch, num_heads, past_seq_val, HEAD_DIM]
        ).astype(np.float16)

        # ---- MorphiZen: 9-input share-buffer form.  past_* is the 448-slot
        # buffer with the valid prefix in [0, past_seq_val) and garbage tail
        # (zeros) afterward; the kernel only reads [0, past_seq_val].
        pk_buf = np.zeros([batch, num_heads, CACHE, HEAD_DIM], dtype=np.float16)
        pv_buf = np.zeros([batch, num_heads, CACHE, HEAD_DIM], dtype=np.float16)
        pk_buf[:, :, :past_seq_val, :] = pk_valid
        pv_buf[:, :, :past_seq_val, :] = pv_valid
        # past_sequence_length = number of valid past tokens (GQA seqlens_k
        # convention: count of populated KV slots before the current token).
        past_seq = np.array([past_seq_val], dtype=np.int32)

        share_model = _make_self_mha_share_buffer_model(
            batch, seq_q, CACHE, num_heads, HEAD_DIM
        )
        sub = model_runner._next_subdir("share_buffer")
        share_path = str(sub / "model.onnx")
        (sub / "model.onnx").write_bytes(share_model.SerializeToString())
        ep_outputs = model_runner.backend.run(
            share_path, [q, k, v, pk_buf, pv_buf, past_seq]
        )
        ep_output = ep_outputs[0]  # [B, 1, hidden]

        # ---- CPU reference: standard 8-input MHA-with-past (no share-buffer),
        # past tensor == the valid prefix only.  ORT CPU rejects the 9-input
        # share-buffer form, so this equal-but-acceptable form is the reference.
        ref_model = _make_self_mha_plain_past_model(
            batch, seq_q, past_seq_val, num_heads, HEAD_DIM
        )
        _, ref_expected = model_runner.run_sample(
            ref_model,
            [q, k, v, pk_valid, pv_valid],
            name="plain_past_ref",
        )
        cpu_output = ref_expected[0]  # [B, 1, hidden]

        compare_outputs(
            [ep_output], [cpu_output], atol=2e-2, rtol=2e-2, cos_threshold=0.999
        )
