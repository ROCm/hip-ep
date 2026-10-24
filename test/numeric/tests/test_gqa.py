#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""Tests for GroupQueryAttention custom op (com.microsoft).

GQA is a complex attention mechanism with grouped key-value heads.
Past/present KV cache uses BNSH layout [batch, kv_num_heads, seq, head_dim].

Two model families are tested:

Llama-3.1-8B (separate Q/K/V, no in-op rotary):
  num_heads = 32, kv_num_heads = 8, head_dim = 128
  hidden = 4096, kv_hidden = 1024
  scale = 1/sqrt(128), do_rotary = 0

GPT-OSS-20B (packed QKV, in-op rotary, attention sinks):
  num_heads = 64, kv_num_heads = 8, head_dim = 64
  q_hidden = 4096, kv_hidden = 512, qkv_hidden = 5120
  scale = 1/sqrt(64) = 0.125, do_rotary = 1
  cos/sin cache: [131072, 32], sinks: [64]
  Variants: local_window_size = 128  and  local_window_size = -1 (full)
"""

import numpy as np
import pytest
from onnx import TensorProto, helper, numpy_helper

from framework.comparator import compare_outputs
from framework.onnx_utils import make_model_from_nodes

SEQ_LENS = [1, 128]

# --- Llama-3.1-8B constants ---
LLAMA_HIDDEN = 4096
LLAMA_KV_HIDDEN = 1024
LLAMA_NUM_HEADS = 32
LLAMA_KV_NUM_HEADS = 8
LLAMA_HEAD_DIM = 128
LLAMA_CACHE_SIZE = 256

# --- GPT-OSS-20B constants ---
GPT_OSS_NUM_HEADS = 64
GPT_OSS_KV_NUM_HEADS = 8
GPT_OSS_HEAD_DIM = 64
GPT_OSS_Q_HIDDEN = GPT_OSS_NUM_HEADS * GPT_OSS_HEAD_DIM  # 4096
GPT_OSS_KV_HIDDEN = GPT_OSS_KV_NUM_HEADS * GPT_OSS_HEAD_DIM  # 512
GPT_OSS_QKV_HIDDEN = GPT_OSS_Q_HIDDEN + 2 * GPT_OSS_KV_HIDDEN  # 5120
GPT_OSS_MAX_POS = 131072
GPT_OSS_HALF_HEAD_DIM = GPT_OSS_HEAD_DIM // 2  # 32


def _make_gqa_model(
    batch, seq_len, past_seq, hidden, kv_hidden, num_heads, kv_num_heads
):
    """Build a GroupQueryAttention ONNX model."""
    head_dim = hidden // num_heads

    query = helper.make_tensor_value_info(
        "query", TensorProto.FLOAT16, [batch, seq_len, hidden]
    )
    key = helper.make_tensor_value_info(
        "key", TensorProto.FLOAT16, [batch, seq_len, kv_hidden]
    )
    value = helper.make_tensor_value_info(
        "value", TensorProto.FLOAT16, [batch, seq_len, kv_hidden]
    )
    past_key = helper.make_tensor_value_info(
        "past_key",
        TensorProto.FLOAT16,
        [batch, kv_num_heads, past_seq, head_dim],
    )
    past_value = helper.make_tensor_value_info(
        "past_value",
        TensorProto.FLOAT16,
        [batch, kv_num_heads, past_seq, head_dim],
    )
    seqlens_k = helper.make_tensor_value_info("seqlens_k", TensorProto.INT32, [batch])
    total_seq = helper.make_tensor_value_info("total_seq", TensorProto.INT32, [])

    total_kv_seq = past_seq + seq_len
    output = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT16, [batch, seq_len, hidden]
    )
    present_key = helper.make_tensor_value_info(
        "present_key",
        TensorProto.FLOAT16,
        [batch, kv_num_heads, total_kv_seq, head_dim],
    )
    present_value = helper.make_tensor_value_info(
        "present_value",
        TensorProto.FLOAT16,
        [batch, kv_num_heads, total_kv_seq, head_dim],
    )

    scale = 1.0 / np.sqrt(head_dim).item()
    node = helper.make_node(
        "GroupQueryAttention",
        [
            "query",
            "key",
            "value",
            "past_key",
            "past_value",
            "seqlens_k",
            "total_seq",
            "",
            "",
        ],
        ["output", "present_key", "present_value"],
        domain="com.microsoft",
        num_heads=num_heads,
        kv_num_heads=kv_num_heads,
        scale=scale,
        do_rotary=0,
        rotary_interleaved=0,
        softcap=0.0,
    )

    ms_opset = helper.make_opsetid("com.microsoft", 1)
    return make_model_from_nodes(
        [node],
        [query, key, value, past_key, past_value, seqlens_k, total_seq],
        [output, present_key, present_value],
        extra_opsets=[ms_opset],
    )


def _make_gqa_fixed_cache_model(
    batch, seq_len, cache_size, hidden, kv_hidden, num_heads, kv_num_heads
):
    """Build a GQA ONNX model with pre-allocated fixed-size KV cache.

    In this pattern (used by the Llama-3.1-8B-fixed model), past and present
    key/value tensors share the same buffer size.  The total_seq scalar
    equals cache_size and seqlens_k indicates how many past tokens are valid.
    """
    head_dim = hidden // num_heads

    query = helper.make_tensor_value_info(
        "query", TensorProto.FLOAT16, [batch, seq_len, hidden]
    )
    key = helper.make_tensor_value_info(
        "key", TensorProto.FLOAT16, [batch, seq_len, kv_hidden]
    )
    value = helper.make_tensor_value_info(
        "value", TensorProto.FLOAT16, [batch, seq_len, kv_hidden]
    )
    past_key = helper.make_tensor_value_info(
        "past_key",
        TensorProto.FLOAT16,
        [batch, kv_num_heads, cache_size, head_dim],
    )
    past_value = helper.make_tensor_value_info(
        "past_value",
        TensorProto.FLOAT16,
        [batch, kv_num_heads, cache_size, head_dim],
    )
    seqlens_k = helper.make_tensor_value_info("seqlens_k", TensorProto.INT32, [batch])
    total_seq = helper.make_tensor_value_info("total_seq", TensorProto.INT32, [])

    output = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT16, [batch, seq_len, hidden]
    )
    present_key = helper.make_tensor_value_info(
        "present_key",
        TensorProto.FLOAT16,
        [batch, kv_num_heads, cache_size, head_dim],
    )
    present_value = helper.make_tensor_value_info(
        "present_value",
        TensorProto.FLOAT16,
        [batch, kv_num_heads, cache_size, head_dim],
    )

    scale = 1.0 / np.sqrt(head_dim).item()
    node = helper.make_node(
        "GroupQueryAttention",
        [
            "query",
            "key",
            "value",
            "past_key",
            "past_value",
            "seqlens_k",
            "total_seq",
            "",
            "",
        ],
        ["output", "present_key", "present_value"],
        domain="com.microsoft",
        num_heads=num_heads,
        kv_num_heads=kv_num_heads,
        scale=scale,
        do_rotary=0,
        rotary_interleaved=0,
        softcap=0.0,
    )

    ms_opset = helper.make_opsetid("com.microsoft", 1)
    return make_model_from_nodes(
        [node],
        [query, key, value, past_key, past_value, seqlens_k, total_seq],
        [output, present_key, present_value],
        extra_opsets=[ms_opset],
    )


def _make_gqa_packed_rotary_model(
    batch,
    seq_len,
    past_seq,
    num_heads,
    kv_num_heads,
    head_dim,
    max_pos,
    local_window_size=-1,
    seed=42,
):
    """Build a GQA model with packed QKV, in-op rotary, and attention sinks.

    Matches the GPT-OSS-20B pattern:
      - Single packed QKV input (Q, K, V are empty strings)
      - do_rotary=1 with cos/sin cache as initializers
      - Attention sinks vector
      - Optional local window attention
    """
    q_hidden = num_heads * head_dim
    kv_hidden = kv_num_heads * head_dim
    qkv_hidden = q_hidden + 2 * kv_hidden
    half_head_dim = head_dim // 2
    total_kv_seq = past_seq + seq_len

    packed_qkv = helper.make_tensor_value_info(
        "packed_qkv",
        TensorProto.FLOAT16,
        [batch, seq_len, qkv_hidden],
    )
    past_key = helper.make_tensor_value_info(
        "past_key",
        TensorProto.FLOAT16,
        [batch, kv_num_heads, past_seq, head_dim],
    )
    past_value = helper.make_tensor_value_info(
        "past_value",
        TensorProto.FLOAT16,
        [batch, kv_num_heads, past_seq, head_dim],
    )
    seqlens_k = helper.make_tensor_value_info(
        "seqlens_k",
        TensorProto.INT32,
        [batch, 1],
    )
    total_seq_vi = helper.make_tensor_value_info(
        "total_seq",
        TensorProto.INT32,
        [],
    )

    output = helper.make_tensor_value_info(
        "output",
        TensorProto.FLOAT16,
        [batch, seq_len, q_hidden],
    )
    present_key = helper.make_tensor_value_info(
        "present_key",
        TensorProto.FLOAT16,
        [batch, kv_num_heads, total_kv_seq, head_dim],
    )
    present_value = helper.make_tensor_value_info(
        "present_value",
        TensorProto.FLOAT16,
        [batch, kv_num_heads, total_kv_seq, head_dim],
    )

    rng = np.random.default_rng(seed)
    cos_data = rng.uniform(-1.5, 1.5, [max_pos, half_head_dim]).astype(np.float16)
    sin_data = rng.uniform(-1.5, 1.5, [max_pos, half_head_dim]).astype(np.float16)
    sinks_data = rng.uniform(-3.0, 5.0, [head_dim]).astype(np.float16)

    initializers = [
        numpy_helper.from_array(cos_data, name="cos_cache"),
        numpy_helper.from_array(sin_data, name="sin_cache"),
        numpy_helper.from_array(sinks_data, name="sinks"),
    ]

    scale = 1.0 / np.sqrt(head_dim).item()
    node = helper.make_node(
        "GroupQueryAttention",
        [
            "packed_qkv",
            "",
            "",
            "past_key",
            "past_value",
            "seqlens_k",
            "total_seq",
            "cos_cache",
            "sin_cache",
            "",
            "",
            "sinks",
        ],
        ["output", "present_key", "present_value"],
        domain="com.microsoft",
        num_heads=num_heads,
        kv_num_heads=kv_num_heads,
        scale=scale,
        local_window_size=local_window_size,
        do_rotary=1,
        rotary_interleaved=0,
        softcap=0.0,
    )

    ms_opset = helper.make_opsetid("com.microsoft", 1)
    return make_model_from_nodes(
        [node],
        [packed_qkv, past_key, past_value, seqlens_k, total_seq_vi],
        [output, present_key, present_value],
        initializers=initializers,
        opset=21,
        extra_opsets=[ms_opset],
    )


class TestGroupQueryAttention:
    @pytest.mark.parametrize(
        "seq_len,hidden,kv_hidden,num_heads,kv_num_heads",
        [
            (4, 256, 128, 4, 2),
            (8, 256, 128, 4, 2),
        ],
    )
    def test_gqa(
        self, model_runner, seq_len, hidden, kv_hidden, num_heads, kv_num_heads
    ):
        """GQA with small shapes for fast sanity checking."""
        head_dim = hidden // num_heads
        past_seq = seq_len
        total_kv_seq = past_seq + seq_len

        model = _make_gqa_model(
            1,
            seq_len,
            past_seq,
            hidden,
            kv_hidden,
            num_heads,
            kv_num_heads,
        )

        rng = np.random.default_rng(42)
        q = rng.uniform(-1, 1, [1, seq_len, hidden]).astype(np.float16)
        k = rng.uniform(-1, 1, [1, seq_len, kv_hidden]).astype(np.float16)
        v = rng.uniform(-1, 1, [1, seq_len, kv_hidden]).astype(np.float16)
        pk = rng.uniform(
            -1,
            1,
            [1, kv_num_heads, past_seq, head_dim],
        ).astype(np.float16)
        pv = rng.uniform(
            -1,
            1,
            [1, kv_num_heads, past_seq, head_dim],
        ).astype(np.float16)
        seqlens = np.array([total_kv_seq - 1], dtype=np.int32)
        total = np.array(total_kv_seq, dtype=np.int32)

        actual, expected = model_runner.run_sample(
            model,
            [q, k, v, pk, pv, seqlens, total],
        )
        compare_outputs(actual, expected, atol=1e-3)

    @pytest.mark.parametrize("seq_len", SEQ_LENS)
    def test_gqa_llama_shape(self, model_runner, seq_len):
        """GQA with Llama-3.1-8B shapes (past_seq == seq_len)."""
        past_seq = seq_len
        total_kv_seq = past_seq + seq_len

        model = _make_gqa_model(
            1,
            seq_len,
            past_seq,
            LLAMA_HIDDEN,
            LLAMA_KV_HIDDEN,
            LLAMA_NUM_HEADS,
            LLAMA_KV_NUM_HEADS,
        )

        rng = np.random.default_rng(42)
        q = rng.uniform(-1, 1, [1, seq_len, LLAMA_HIDDEN]).astype(np.float16)
        k = rng.uniform(-1, 1, [1, seq_len, LLAMA_KV_HIDDEN]).astype(np.float16)
        v = rng.uniform(-1, 1, [1, seq_len, LLAMA_KV_HIDDEN]).astype(np.float16)
        pk = rng.uniform(
            -1,
            1,
            [1, LLAMA_KV_NUM_HEADS, past_seq, LLAMA_HEAD_DIM],
        ).astype(np.float16)
        pv = rng.uniform(
            -1,
            1,
            [1, LLAMA_KV_NUM_HEADS, past_seq, LLAMA_HEAD_DIM],
        ).astype(np.float16)
        seqlens = np.array([total_kv_seq - 1], dtype=np.int32)
        total = np.array(total_kv_seq, dtype=np.int32)

        actual, expected = model_runner.run_sample(
            model,
            [q, k, v, pk, pv, seqlens, total],
        )
        compare_outputs(actual, expected, atol=1e-3)

    @pytest.mark.parametrize("seq_len", SEQ_LENS)
    def test_gqa_local_gpt_oss_shape(self, model_runner, seq_len):
        """GQA packed-QKV + rotary + sinks, local_window=128 (GPT-OSS-20B).

        Input: packed QKV from qkv_proj [1, S, 5120], range [-2, 2].
        64 Q heads, 8 KV heads, head_dim=64, rotary applied inside GQA.
        """
        past_seq = seq_len
        total_kv_seq = past_seq + seq_len

        model = _make_gqa_packed_rotary_model(
            1,
            seq_len,
            past_seq,
            GPT_OSS_NUM_HEADS,
            GPT_OSS_KV_NUM_HEADS,
            GPT_OSS_HEAD_DIM,
            GPT_OSS_MAX_POS,
            local_window_size=128,
        )

        rng = np.random.default_rng(42)
        qkv = rng.uniform(
            -2,
            2,
            [1, seq_len, GPT_OSS_QKV_HIDDEN],
        ).astype(np.float16)
        pk = rng.uniform(
            -1,
            1,
            [1, GPT_OSS_KV_NUM_HEADS, past_seq, GPT_OSS_HEAD_DIM],
        ).astype(np.float16)
        pv = rng.uniform(
            -1,
            1,
            [1, GPT_OSS_KV_NUM_HEADS, past_seq, GPT_OSS_HEAD_DIM],
        ).astype(np.float16)
        seqlens = np.array([[total_kv_seq - 1]], dtype=np.int32)
        total = np.array(total_kv_seq, dtype=np.int32)

        actual, expected = model_runner.run_sample(
            model,
            [qkv, pk, pv, seqlens, total],
        )
        compare_outputs(actual, expected, atol=1e-2, rtol=1e-2, cos_threshold=0.999)

    @pytest.mark.parametrize("seq_len", SEQ_LENS)
    def test_gqa_full_gpt_oss_shape(self, model_runner, seq_len):
        """GQA packed-QKV + rotary + sinks, full attention (GPT-OSS-20B).

        Same as local variant but with local_window_size=-1 (full attention).
        """
        past_seq = seq_len
        total_kv_seq = past_seq + seq_len

        model = _make_gqa_packed_rotary_model(
            1,
            seq_len,
            past_seq,
            GPT_OSS_NUM_HEADS,
            GPT_OSS_KV_NUM_HEADS,
            GPT_OSS_HEAD_DIM,
            GPT_OSS_MAX_POS,
            local_window_size=-1,
        )

        rng = np.random.default_rng(42)
        qkv = rng.uniform(
            -2,
            2,
            [1, seq_len, GPT_OSS_QKV_HIDDEN],
        ).astype(np.float16)
        pk = rng.uniform(
            -1,
            1,
            [1, GPT_OSS_KV_NUM_HEADS, past_seq, GPT_OSS_HEAD_DIM],
        ).astype(np.float16)
        pv = rng.uniform(
            -1,
            1,
            [1, GPT_OSS_KV_NUM_HEADS, past_seq, GPT_OSS_HEAD_DIM],
        ).astype(np.float16)
        seqlens = np.array([[total_kv_seq - 1]], dtype=np.int32)
        total = np.array(total_kv_seq, dtype=np.int32)

        actual, expected = model_runner.run_sample(
            model,
            [qkv, pk, pv, seqlens, total],
        )
        compare_outputs(actual, expected, atol=1e-2, rtol=1e-2, cos_threshold=0.999)

    @pytest.mark.parametrize("past_valid", [0, 64, 128])
    def test_gqa_prefill_fixed_cache_llama_shape(self, model_runner, past_valid):
        """GQA prefill with pre-allocated KV cache (Llama-3.1-8B-fixed pattern).

        128-token prefill into a 256-slot pre-allocated buffer.  past_valid=0
        is the first inference (empty cache); past_valid>0 simulates a
        multi-turn chat where earlier turns already filled part of the cache.
        seqlens_k = past_valid + 128 - 1.
        """
        seq_len = 128

        model = _make_gqa_fixed_cache_model(
            1,
            seq_len,
            LLAMA_CACHE_SIZE,
            LLAMA_HIDDEN,
            LLAMA_KV_HIDDEN,
            LLAMA_NUM_HEADS,
            LLAMA_KV_NUM_HEADS,
        )

        rng = np.random.default_rng(42)
        q = rng.uniform(-1, 1, [1, seq_len, LLAMA_HIDDEN]).astype(np.float16)
        k = rng.uniform(-1, 1, [1, seq_len, LLAMA_KV_HIDDEN]).astype(np.float16)
        v = rng.uniform(-1, 1, [1, seq_len, LLAMA_KV_HIDDEN]).astype(np.float16)
        pk = rng.uniform(
            -1,
            1,
            [1, LLAMA_KV_NUM_HEADS, LLAMA_CACHE_SIZE, LLAMA_HEAD_DIM],
        ).astype(np.float16)
        if past_valid == 0:
            pk[:] = 0
        pv = rng.uniform(
            -1,
            1,
            [1, LLAMA_KV_NUM_HEADS, LLAMA_CACHE_SIZE, LLAMA_HEAD_DIM],
        ).astype(np.float16)
        if past_valid == 0:
            pv[:] = 0
        seqlens = np.array([past_valid + seq_len - 1], dtype=np.int32)
        total = np.array(LLAMA_CACHE_SIZE, dtype=np.int32)

        actual, expected = model_runner.run_sample(
            model,
            [q, k, v, pk, pv, seqlens, total],
        )
        compare_outputs(actual, expected, atol=1e-3)

    @pytest.mark.parametrize("past_valid", [1, 64, 128, 200, 255])
    def test_gqa_decode_fixed_cache_llama_shape(self, model_runner, past_valid):
        """GQA decode with pre-allocated KV cache (Llama-3.1-8B-fixed pattern).

        Single-token decode with various past lengths in the 256-slot KV
        buffer.  seqlens_k = past_valid + 1 - 1 = past_valid.
        """
        seq_len = 1

        model = _make_gqa_fixed_cache_model(
            1,
            seq_len,
            LLAMA_CACHE_SIZE,
            LLAMA_HIDDEN,
            LLAMA_KV_HIDDEN,
            LLAMA_NUM_HEADS,
            LLAMA_KV_NUM_HEADS,
        )

        rng = np.random.default_rng(42)
        q = rng.uniform(-1, 1, [1, seq_len, LLAMA_HIDDEN]).astype(np.float16)
        k = rng.uniform(-1, 1, [1, seq_len, LLAMA_KV_HIDDEN]).astype(np.float16)
        v = rng.uniform(-1, 1, [1, seq_len, LLAMA_KV_HIDDEN]).astype(np.float16)
        pk = rng.uniform(
            -1,
            1,
            [1, LLAMA_KV_NUM_HEADS, LLAMA_CACHE_SIZE, LLAMA_HEAD_DIM],
        ).astype(np.float16)
        pv = rng.uniform(
            -1,
            1,
            [1, LLAMA_KV_NUM_HEADS, LLAMA_CACHE_SIZE, LLAMA_HEAD_DIM],
        ).astype(np.float16)
        seqlens = np.array([past_valid + seq_len - 1], dtype=np.int32)
        total = np.array(LLAMA_CACHE_SIZE, dtype=np.int32)

        actual, expected = model_runner.run_sample(
            model,
            [q, k, v, pk, pv, seqlens, total],
        )
        compare_outputs(actual, expected, atol=1e-3)

    def test_gqa_decode_deep_cache_llama_shape(self, model_runner):
        """GQA single-token decode with a large KV cache (Llama-3.1-8B pattern).

        Simulates a decode step late in generation: seq_len=1 with 255 past
        tokens already in the cache, matching the model's 256-slot total.
        """
        seq_len = 1
        past_seq = 255
        total_kv_seq = past_seq + seq_len

        model = _make_gqa_model(
            1,
            seq_len,
            past_seq,
            LLAMA_HIDDEN,
            LLAMA_KV_HIDDEN,
            LLAMA_NUM_HEADS,
            LLAMA_KV_NUM_HEADS,
        )

        rng = np.random.default_rng(42)
        q = rng.uniform(-1, 1, [1, seq_len, LLAMA_HIDDEN]).astype(np.float16)
        k = rng.uniform(-1, 1, [1, seq_len, LLAMA_KV_HIDDEN]).astype(np.float16)
        v = rng.uniform(-1, 1, [1, seq_len, LLAMA_KV_HIDDEN]).astype(np.float16)
        pk = rng.uniform(
            -1,
            1,
            [1, LLAMA_KV_NUM_HEADS, past_seq, LLAMA_HEAD_DIM],
        ).astype(np.float16)
        pv = rng.uniform(
            -1,
            1,
            [1, LLAMA_KV_NUM_HEADS, past_seq, LLAMA_HEAD_DIM],
        ).astype(np.float16)
        seqlens = np.array([total_kv_seq - 1], dtype=np.int32)
        total = np.array(total_kv_seq, dtype=np.int32)

        actual, expected = model_runner.run_sample(
            model,
            [q, k, v, pk, pv, seqlens, total],
        )
        compare_outputs(actual, expected, atol=1e-3)

    @pytest.mark.parametrize(
        "seq_len,real_count,past_valid",
        [
            # Matches genai chunk-opt: window=128, prompt=22 tokens, no past KV.
            (128, 22, 0),
            # Smallest non-trivial real prefix.
            (128, 1, 0),
            # Larger real prefix that still leaves the chunk partially padded.
            (128, 64, 0),
            # Boundary case: real prefix exactly half the window.
            (128, 32, 0),
        ],
    )
    def test_gqa_padded_chunk_prefill_llama_shape(
        self, model_runner, seq_len, real_count, past_valid
    ):
        """GQA chunk-opt padded prefill: seq_len_q > seqlens_k+1.

        Regression for the bug fixed in lib/Runtime/real/gqa.cpp where
        ``past_len = total_seq - sq`` went negative for chunk-opt padded
        prefill (e.g. genai chunked prefill with window=512 and a 22-token
        prompt: total_seq=22, sq=512 -> past_len=-490).  The host-side
        validation rejected this as invalid, returning rc=-1 from every
        GroupQueryAttention layer, which cascaded to all-zero logits and
        the LM emitting token id 0 (``!``) for every step.

        The fix clamps past_len to 0 in that case and lets the kernel
        attend Q[0..sq-1] over K[0..total_seq-1].  Q positions beyond
        real_count are downstream garbage that the model's own
        attention_mask handling masks out.

        Scope: this regression covers the *fresh-prefill* chunk-opt
        scenario (past_valid == 0) which is what the genai sliding-window
        config drives for any prompt shorter than one chunk window.  The
        analogous mid-conversation case (past_valid > 0 with real_count
        < seq_len) needs a richer fix because past_len no longer equals
        zero -- tracked separately.

        Verification strategy: the ORT CPU EP doesn't accept the padded
        configuration directly (it asserts seqlens_k+1 == past+seq), so
        we compare the EP's output for the first ``real_count`` Q
        positions against the CPU reference output of an *unpadded*
        equivalent model (seq_len=real_count, same K/V/past in those
        positions).  Both kernels see exactly the same Q-vs-K reduction
        for positions ``[0, real_count)`` because each Q position only
        attends to K[0..min(i, total_seq-1)], which is independent of the
        padded tail.
        """
        assert real_count <= seq_len
        total_kv_seq = past_valid + real_count

        rng = np.random.default_rng(42)
        # Real Q/K/V values for the [0, real_count) prefix.
        q_real = rng.uniform(-1, 1, [1, real_count, LLAMA_HIDDEN]).astype(np.float16)
        k_real = rng.uniform(-1, 1, [1, real_count, LLAMA_KV_HIDDEN]).astype(np.float16)
        v_real = rng.uniform(-1, 1, [1, real_count, LLAMA_KV_HIDDEN]).astype(np.float16)
        # Padded tail uses zeros to mimic genai's padded embeddings.
        pad_q = np.zeros([1, seq_len - real_count, LLAMA_HIDDEN], dtype=np.float16)
        pad_k = np.zeros([1, seq_len - real_count, LLAMA_KV_HIDDEN], dtype=np.float16)
        pad_v = np.zeros([1, seq_len - real_count, LLAMA_KV_HIDDEN], dtype=np.float16)
        q_full = np.concatenate([q_real, pad_q], axis=1)
        k_full = np.concatenate([k_real, pad_k], axis=1)
        v_full = np.concatenate([v_real, pad_v], axis=1)

        pk = rng.uniform(
            -1,
            1,
            [1, LLAMA_KV_NUM_HEADS, LLAMA_CACHE_SIZE, LLAMA_HEAD_DIM],
        ).astype(np.float16)
        pv = rng.uniform(
            -1,
            1,
            [1, LLAMA_KV_NUM_HEADS, LLAMA_CACHE_SIZE, LLAMA_HEAD_DIM],
        ).astype(np.float16)
        if past_valid == 0:
            pk[:] = 0
            pv[:] = 0

        # ---- Padded (chunk-opt) model: run on EP only.  ORT CPU EP does
        # not accept seq_len_q > seqlens_k+1, so we drive the backend
        # directly without the reference comparison.
        wide_model = _make_gqa_fixed_cache_model(
            1,
            seq_len,
            LLAMA_CACHE_SIZE,
            LLAMA_HIDDEN,
            LLAMA_KV_HIDDEN,
            LLAMA_NUM_HEADS,
            LLAMA_KV_NUM_HEADS,
        )
        sub = model_runner._next_subdir("wide_padded")
        wide_path = str(sub / "model.onnx")
        (sub / "model.onnx").write_bytes(wide_model.SerializeToString())
        seqlens_full = np.array([total_kv_seq - 1], dtype=np.int32)
        total = np.array(LLAMA_CACHE_SIZE, dtype=np.int32)
        ep_outputs_full = model_runner.backend.run(
            wide_path,
            [q_full, k_full, v_full, pk, pv, seqlens_full, total],
        )
        ep_output_full = ep_outputs_full[0]

        # ---- Narrow (unpadded) reference model: same K/V prefix, run via
        # the standard run_sample helper (CPU is the reference). ORT CPU's
        # outputs for [0, real_count) are mathematically identical to what
        # the padded EP run should produce for the same prefix.
        narrow_model = _make_gqa_fixed_cache_model(
            1,
            real_count,
            LLAMA_CACHE_SIZE,
            LLAMA_HIDDEN,
            LLAMA_KV_HIDDEN,
            LLAMA_NUM_HEADS,
            LLAMA_KV_NUM_HEADS,
        )
        seqlens_narrow = np.array([total_kv_seq - 1], dtype=np.int32)
        _, narrow_expected = model_runner.run_sample(
            narrow_model,
            [q_real, k_real, v_real, pk.copy(), pv.copy(), seqlens_narrow, total],
            name="narrow_ref",
        )
        narrow_cpu_output = narrow_expected[0]

        # ---- Compare the [0, real_count) Q slice
        ep_prefix = ep_output_full[:, :real_count, :]
        compare_outputs([ep_prefix], [narrow_cpu_output], atol=2e-3)

        # ---- Sanity: the prefix must not be all-zero (the original bug
        # symptom was rc=-1 -> output left at zero-init, cascading to
        # all-zero logits and "!" tokens).
        assert np.any(ep_prefix != 0), (
            "GQA prefix output is all-zero; the chunk-opt clamp regressed."
        )

    def test_gqa_decode_early_cache_llama_shape(self, model_runner):
        """GQA single-token decode right after prefill (Llama-3.1-8B pattern).

        Simulates the first decode step: seq_len=1 with 128 past tokens
        from a 128-token prefill.
        """
        seq_len = 1
        past_seq = 128
        total_kv_seq = past_seq + seq_len

        model = _make_gqa_model(
            1,
            seq_len,
            past_seq,
            LLAMA_HIDDEN,
            LLAMA_KV_HIDDEN,
            LLAMA_NUM_HEADS,
            LLAMA_KV_NUM_HEADS,
        )

        rng = np.random.default_rng(42)
        q = rng.uniform(-1, 1, [1, seq_len, LLAMA_HIDDEN]).astype(np.float16)
        k = rng.uniform(-1, 1, [1, seq_len, LLAMA_KV_HIDDEN]).astype(np.float16)
        v = rng.uniform(-1, 1, [1, seq_len, LLAMA_KV_HIDDEN]).astype(np.float16)
        pk = rng.uniform(
            -1,
            1,
            [1, LLAMA_KV_NUM_HEADS, past_seq, LLAMA_HEAD_DIM],
        ).astype(np.float16)
        pv = rng.uniform(
            -1,
            1,
            [1, LLAMA_KV_NUM_HEADS, past_seq, LLAMA_HEAD_DIM],
        ).astype(np.float16)
        seqlens = np.array([total_kv_seq - 1], dtype=np.int32)
        total = np.array(total_kv_seq, dtype=np.int32)

        actual, expected = model_runner.run_sample(
            model,
            [q, k, v, pk, pv, seqlens, total],
        )
        compare_outputs(actual, expected, atol=1e-3)


def _ring_fp32_reference(query, key_cache, value_cache, past):
    qlen = query.shape[1]
    total = past + qlen
    q = query.reshape(1, qlen, 24, 256).transpose(0, 2, 1, 3).astype(np.float32)
    k = np.repeat(key_cache[:, :, :total].astype(np.float32), 6, axis=1)
    v = np.repeat(value_cache[:, :, :total].astype(np.float32), 6, axis=1)
    scores = (q @ k.transpose(0, 1, 3, 2)) * np.float32(0.0625)
    mask = np.arange(total)[None, :] > past + np.arange(qlen)[:, None]
    scores = np.where(mask, -np.inf, scores)
    weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
    weights /= weights.sum(axis=-1, keepdims=True)
    return (weights @ v).transpose(0, 2, 1, 3).reshape(1, qlen, 6144)


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize(
    "query_length,past_length", [(1, 0), (1, 257), (17, 0), (33, 129)]
)
def test_gqa_ring_geometry(model_runner, tmp_path, dynamic, query_length, past_length):
    """Verify the common ONNX contract for default and Ring-enabled builds.

    The backend disables CPU fallback. Compare against FP32 attention computed
    from the same FP16-rounded inputs; check the entire valid present cache too.
    Unused capacity in separately allocated outputs has no defined contents.
    """
    import onnx

    qlen, past = query_length, past_length
    total, capacity = past + qlen, past + qlen + 19
    model = _make_gqa_fixed_cache_model(
        1, "query_length" if dynamic else qlen, capacity, 6144, 1024, 24, 4
    )
    rng = np.random.default_rng(984)
    query = rng.standard_normal((1, qlen, 6144)).astype(np.float16)
    key = rng.standard_normal((1, qlen, 1024)).astype(np.float16)
    value = rng.standard_normal((1, qlen, 1024)).astype(np.float16)
    old_k = rng.standard_normal((1, 4, capacity, 256)).astype(np.float16)
    old_v = rng.standard_normal((1, 4, capacity, 256)).astype(np.float16)
    seqlens = np.array([total - 1], dtype=np.int32)
    length = np.array(capacity, dtype=np.int32)
    path = tmp_path / "ring_geometry.onnx"
    onnx.save(model, path)
    actual = model_runner.backend.run(
        str(path), [query, key, value, old_k, old_v, seqlens, length]
    )
    expected_k, expected_v = old_k.copy(), old_v.copy()
    expected_k[:, :, past:total] = key.reshape(1, qlen, 4, 256).transpose(0, 2, 1, 3)
    expected_v[:, :, past:total] = value.reshape(1, qlen, 4, 256).transpose(0, 2, 1, 3)
    expected = _ring_fp32_reference(query, expected_k, expected_v, past)
    np.testing.assert_allclose(actual[0], expected, atol=0.005, rtol=0.005)
    np.testing.assert_array_equal(actual[1][:, :, :total], expected_k[:, :, :total])
    np.testing.assert_array_equal(actual[2][:, :, :total], expected_v[:, :, :total])


def test_gqa_ring_shared_cache_transitions(model_runner, tmp_path):
    """Bind actual GPU OrtValues as both past and present across prefill/decode."""
    import onnx
    import onnxruntime as ort

    backend = model_runner.backend
    if not hasattr(backend, "_ep_device"):
        pytest.skip("requires the ORT EP backend and its GPU allocator")
    capacity = 80
    model = _make_gqa_fixed_cache_model(1, "query_length", capacity, 6144, 1024, 24, 4)
    path = tmp_path / "ring_shared.onnx"
    onnx.save(model, path)
    options = ort.SessionOptions()
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    options.add_provider_for_devices(
        [backend._ep_device], dict(backend._provider_options)
    )
    session = ort.InferenceSession(str(path), sess_options=options)
    expected_k = np.full((1, 4, capacity, 256), 7.0, dtype=np.float16)
    expected_v = expected_k.copy()
    caches = [
        ort.OrtValue.ortvalue_from_shape_and_type(
            list(expected_k.shape), np.float16, device_type="gpu", vendor_id=0x1002
        )
        for _ in range(2)
    ]
    for cache, values in zip(caches, [expected_k, expected_v]):
        cache.update_inplace(values)
    rng = np.random.default_rng(987)
    past = 0
    for qlen in (17, 1, 33, 1):
        total = past + qlen
        query = rng.standard_normal((1, qlen, 6144)).astype(np.float16)
        key = rng.standard_normal((1, qlen, 1024)).astype(np.float16)
        value = rng.standard_normal((1, qlen, 1024)).astype(np.float16)
        inputs = {
            "query": query,
            "key": key,
            "value": value,
            "seqlens_k": np.array([total - 1], dtype=np.int32),
            "total_seq": np.array(capacity, dtype=np.int32),
        }
        io = session.io_binding()
        for name, array in inputs.items():
            io.bind_cpu_input(name, array)
        io.bind_output("output", "cpu")
        for kind, cache in zip(["key", "value"], caches):
            io.bind_ortvalue_input("past_" + kind, cache)
            io.bind_ortvalue_output("present_" + kind, cache)
        session.run_with_iobinding(io)
        io.synchronize_outputs()
        outputs = io.copy_outputs_to_cpu()
        for actual, cache in zip(io.get_outputs()[1:], caches):
            assert actual.data_ptr() == cache.data_ptr()
        expected_k[:, :, past:total] = key.reshape(1, qlen, 4, 256).transpose(
            0, 2, 1, 3
        )
        expected_v[:, :, past:total] = value.reshape(1, qlen, 4, 256).transpose(
            0, 2, 1, 3
        )
        expected = _ring_fp32_reference(query, expected_k, expected_v, past)
        np.testing.assert_allclose(outputs[0], expected, atol=0.005, rtol=0.005)
        np.testing.assert_array_equal(outputs[1], expected_k)
        np.testing.assert_array_equal(outputs[2], expected_v)
        past = total


@pytest.mark.parametrize(
    "query_length,past_length",
    [
        (8192, 0),
        (32768, 0),
        (4096, 28800),
        (1, 32895),
        (65536, 0),
        (4096, 61568),
        (1, 65663),
        (1, 131071),
        (1, 262143),
        (17, 262255),
    ],
)
def test_gqa_ring_long_context(model_runner, tmp_path, query_length, past_length):
    """Check long-query admission and the final cache boundary through ONNX.

    Sample query rows but scan every causally valid key in an FP32 reference;
    verify full output finiteness and every valid present-cache element.
    """
    import onnx

    qlen, past = query_length, past_length
    total = past + qlen
    capacity = max(32915, total + 19)
    model = _make_gqa_fixed_cache_model(1, "query_length", capacity, 6144, 1024, 24, 4)
    rng = np.random.default_rng(985)
    query = rng.standard_normal((1, qlen, 6144)).astype(np.float16)
    key = rng.standard_normal((1, qlen, 1024)).astype(np.float16)
    value = rng.standard_normal((1, qlen, 1024)).astype(np.float16)
    old_k = rng.standard_normal((1, 4, capacity, 256)).astype(np.float16)
    old_v = rng.standard_normal((1, 4, capacity, 256)).astype(np.float16)
    path = tmp_path / "ring_long_context.onnx"
    onnx.save(model, path)
    actual = model_runner.backend.run(
        str(path),
        [
            query,
            key,
            value,
            old_k,
            old_v,
            np.array([total - 1], dtype=np.int32),
            np.array(capacity, dtype=np.int32),
        ],
    )
    old_k[:, :, past:total] = key.reshape(1, qlen, 4, 256).transpose(0, 2, 1, 3)
    old_v[:, :, past:total] = value.reshape(1, qlen, 4, 256).transpose(0, 2, 1, 3)
    assert np.isfinite(actual[0]).all()
    np.testing.assert_array_equal(actual[1][:, :, :total], old_k[:, :, :total])
    np.testing.assert_array_equal(actual[2][:, :, :total], old_v[:, :, :total])
    indices = np.array(
        sorted(
            {
                0,
                qlen // 4,
                qlen // 2,
                3 * qlen // 4,
                qlen - 1,
                min(15, qlen - 1),
                min(16, qlen - 1),
            }
        )
    )
    q = query.reshape(qlen, 24, 256)[indices].astype(np.float32)
    expected = np.empty_like(q)
    mask = np.arange(total)[None, :] > past + indices[:, None]
    for group in range(4):
        heads = slice(group * 6, (group + 1) * 6)
        scores = (
            q[:, heads].transpose(1, 0, 2)
            @ old_k[0, group, :total].astype(np.float32).T
        ) * np.float32(0.0625)
        scores = np.where(mask[None, :, :], -np.inf, scores)
        weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
        weights /= weights.sum(axis=-1, keepdims=True)
        expected[:, heads] = (
            weights @ old_v[0, group, :total].astype(np.float32)
        ).transpose(1, 0, 2)
    np.testing.assert_allclose(
        actual[0].reshape(qlen, 24, 256)[indices], expected, atol=0.005, rtol=0.005
    )


@pytest.mark.parametrize(
    "total",
    [
        15,
        16,
        17,
        31,
        32,
        33,
        127,
        128,
        129,
        1023,
        1024,
        1025,
        8191,
        8192,
        8193,
        65663,
        65664,
        65665,
        131072,
        262143,
        262144,
        262272,
    ],
)
def test_gqa_ring_decode_regimes(model_runner, tmp_path, total):
    """Exercise partial tiles and both sides of partition dispatch boundaries."""
    test_gqa_ring_geometry(model_runner, tmp_path, True, 1, total - 1)


@pytest.mark.parametrize("total", [2, 17, 65, 8193, 65665, 262144])
def test_gqa_ring_decode_regimes_probability_precision(model_runner, tmp_path, total):
    """A tiny probability times a large FP16 value must survive decode.

    exp(-18) rounds to zero in FP16, but its FP32 product with 60000 is
    representable in the output. This catches FP16 probability narrowing.
    """
    import onnx

    capacity = total + 17
    model = _make_gqa_fixed_cache_model(1, 1, capacity, 6144, 1024, 24, 4)
    query = np.zeros((1, 1, 6144), dtype=np.float16)
    query.reshape(24, 256)[:, 0] = 16
    key = np.zeros((1, 1, 1024), dtype=np.float16)
    key.reshape(4, 256)[:, 0] = -18
    value = np.full((1, 1, 1024), 60000, dtype=np.float16)
    old_k = np.zeros((1, 4, capacity, 256), dtype=np.float16)
    old_k[:, :, 1 : total - 1, 0] = -18
    old_v = np.zeros_like(old_k)
    path = tmp_path / "ring_decode_probability_precision.onnx"
    onnx.save(model, path)
    actual = model_runner.backend.run(
        str(path),
        [
            query,
            key,
            value,
            old_k,
            old_v,
            np.array([total - 1], dtype=np.int32),
            np.array(capacity, dtype=np.int32),
        ],
    )
    expected = 60000 * np.exp(-18.0) / (1 + (total - 1) * np.exp(-18.0))
    np.testing.assert_allclose(actual[0], expected, atol=0.000002, rtol=0.001)


@pytest.mark.parametrize("past,delta", [(0, 18), (0, 24), (65536, 32), (262255, 32)])
def test_gqa_ring_prefill_probability_precision(model_runner, tmp_path, past, delta):
    """Tiny prefill probabilities must contribute when multiplied by large V."""
    import ctypes

    import onnx

    # This regression tests Ring's precision contract. The unchanged default
    # prefill narrows probabilities to FP16 and does not implement this fix.
    ep_path = getattr(model_runner.backend, "_ep_dll", None)
    if ep_path is None:
        pytest.skip("requires the Ring-enabled gfx1151 EP")
    libraries = [
        ep_path.parent / "custom_kernels_gfx1151.dll",
        ep_path.parent / "libcustom_kernels_gfx1151.so",
    ]
    library = next((path for path in libraries if path.exists()), None)
    if library is None:
        pytest.skip("requires the gfx1151 native kernel library")
    native = ctypes.CDLL(str(library))
    supported = getattr(native, "hip_gqa_ring_device_supported", None)
    if supported is None:
        pytest.skip("Ring attention is disabled in this build")
    supported.argtypes = []
    supported.restype = ctypes.c_int
    if not supported():
        pytest.skip("requires a Ring-supported device")

    qlen = 17
    total = past + qlen
    capacity = total + 19
    model = _make_gqa_fixed_cache_model(1, "query_length", capacity, 6144, 1024, 24, 4)
    query = np.zeros((1, qlen, 6144), dtype=np.float16)
    query.reshape(qlen, 24, 256)[:, :, 0] = 16
    key = np.zeros((1, qlen, 1024), dtype=np.float16)
    key.reshape(qlen, 4, 256)[:, :, 0] = -delta
    value = np.full((1, qlen, 1024), 60000, dtype=np.float16)
    old_k = np.zeros((1, 4, capacity, 256), dtype=np.float16)
    old_k[:, :, :past, 0] = -delta
    old_v = np.full_like(old_k, 60000)
    if past:
        old_k[:, :, 0] = 0
        old_v[:, :, 0] = 0
    else:
        key[:, 0] = 0
        value[:, 0] = 0
    path = tmp_path / "ring_prefill_probability_precision.onnx"
    onnx.save(model, path)
    actual = model_runner.backend.run(
        str(path),
        [
            query,
            key,
            value,
            old_k,
            old_v,
            np.array([total - 1], dtype=np.int32),
            np.array(capacity, dtype=np.int32),
        ],
    )
    mass = (past + np.arange(qlen)) * np.exp(-float(delta))
    expected = np.broadcast_to((60000 * mass / (1 + mass))[None, :, None], query.shape)
    np.testing.assert_allclose(actual[0], expected, atol=0.000002, rtol=0.001)
