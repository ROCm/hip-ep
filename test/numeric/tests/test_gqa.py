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
    batch,
    seq_len,
    cache_size,
    hidden,
    kv_hidden,
    num_heads,
    kv_num_heads,
    elem_type=TensorProto.FLOAT16,
):
    """Build a GQA ONNX model with pre-allocated fixed-size KV cache.

    In this pattern (used by the Llama-3.1-8B-fixed model), past and present
    key/value tensors share the same buffer size.  The total_seq scalar
    equals cache_size and seqlens_k indicates how many past tokens are valid.
    elem_type sets the element type of the activations and the cache.
    """
    head_dim = hidden // num_heads

    query = helper.make_tensor_value_info("query", elem_type, [batch, seq_len, hidden])
    key = helper.make_tensor_value_info("key", elem_type, [batch, seq_len, kv_hidden])
    value = helper.make_tensor_value_info(
        "value", elem_type, [batch, seq_len, kv_hidden]
    )
    past_key = helper.make_tensor_value_info(
        "past_key",
        elem_type,
        [batch, kv_num_heads, cache_size, head_dim],
    )
    past_value = helper.make_tensor_value_info(
        "past_value",
        elem_type,
        [batch, kv_num_heads, cache_size, head_dim],
    )
    seqlens_k = helper.make_tensor_value_info("seqlens_k", TensorProto.INT32, [batch])
    total_seq = helper.make_tensor_value_info("total_seq", TensorProto.INT32, [])

    output = helper.make_tensor_value_info(
        "output", elem_type, [batch, seq_len, hidden]
    )
    present_key = helper.make_tensor_value_info(
        "present_key",
        elem_type,
        [batch, kv_num_heads, cache_size, head_dim],
    )
    present_value = helper.make_tensor_value_info(
        "present_value",
        elem_type,
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


def _make_gqa_fixed_cache_rotary_model(seq_len, elem_type=TensorProto.FLOAT16):
    """The Llama fixed-cache GQA model with RoPE inside GQA (cos/sin caches)."""
    f16 = elem_type
    vi = helper.make_tensor_value_info
    cache = [1, LLAMA_KV_NUM_HEADS, LLAMA_CACHE_SIZE, LLAMA_HEAD_DIM]
    rot = [LLAMA_CACHE_SIZE, LLAMA_HEAD_DIM // 2]
    inputs = [
        vi("query", f16, [1, seq_len, LLAMA_HIDDEN]),
        vi("key", f16, [1, seq_len, LLAMA_KV_HIDDEN]),
        vi("value", f16, [1, seq_len, LLAMA_KV_HIDDEN]),
        vi("past_key", f16, cache),
        vi("past_value", f16, cache),
        vi("seqlens_k", TensorProto.INT32, [1]),
        vi("total_seq", TensorProto.INT32, []),
        vi("cos_cache", f16, rot),
        vi("sin_cache", f16, rot),
    ]
    outputs = [
        vi("output", f16, [1, seq_len, LLAMA_HIDDEN]),
        vi("present_key", f16, cache),
        vi("present_value", f16, cache),
    ]
    node = helper.make_node(
        "GroupQueryAttention",
        [i.name for i in inputs],
        [o.name for o in outputs],
        domain="com.microsoft",
        num_heads=LLAMA_NUM_HEADS,
        kv_num_heads=LLAMA_KV_NUM_HEADS,
        scale=1.0 / np.sqrt(LLAMA_HEAD_DIM).item(),
        do_rotary=1,
        rotary_interleaved=0,
    )
    return make_model_from_nodes(
        [node], inputs, outputs, extra_opsets=[helper.make_opsetid("com.microsoft", 1)]
    )


def _rope_tables(np_dtype=np.float16):
    pos = np.arange(LLAMA_CACHE_SIZE, dtype=np.float32)[:, None]
    inv = 1.0 / (10000.0 ** (np.arange(0, LLAMA_HEAD_DIM, 2) / LLAMA_HEAD_DIM))
    ang = pos * inv[None, :].astype(np.float32)
    return np.cos(ang).astype(np_dtype), np.sin(ang).astype(np_dtype)


def _ep_session(model_runner, model, name):
    """An EP session for IO binding: persists *model* and opens it like the
    backend does (CPU fallback disabled, same provider options)."""
    import onnxruntime as ort

    backend = model_runner.backend
    sub = model_runner._next_subdir(name)
    (sub / "model.onnx").write_bytes(model.SerializeToString())
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    opts.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    opts.add_provider_for_devices([backend._ep_device], dict(backend._provider_options))
    return ort.InferenceSession(str(sub / "model.onnx"), sess_options=opts)


def _gpu(arr):
    import onnxruntime as ort

    return ort.OrtValue.ortvalue_from_numpy(arr, device_type="gpu", vendor_id=0x1002)


def _cpu_gqa(model, inputs):
    """ORT CPU reference for *model* (independent of the EP)."""
    import onnxruntime as ort

    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    return sess.run(None, dict(zip([i.name for i in sess.get_inputs()], inputs)))


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
            # A full window: no padding, the ordinary prefill path.
            (128, 128, 0),
        ],
    )
    # float16 runs the fused prefill; float32 activations with an unquantized
    # cache are not fused_supported and run the decomposed hipBLASLt pipeline.
    @pytest.mark.parametrize(
        "elem_type,np_dtype",
        [(TensorProto.FLOAT16, np.float16), (TensorProto.FLOAT, np.float32)],
        ids=["fp16_fused", "fp32_hipblaslt"],
    )
    def test_gqa_padded_chunk_prefill_llama_shape(
        self, model_runner, seq_len, real_count, past_valid, elem_type, np_dtype
    ):
        """GQA chunk-opt padded prefill: seq_len_q > seqlens_k+1.

        Regression for the bug fixed in lib/Runtime/real/gqa.cpp where
        ``past_len = total_seq - sq`` went negative for chunk-opt padded
        prefill (e.g. genai chunked prefill with window=512 and a 22-token
        prompt: total_seq=22, sq=512 -> past_len=-490).  The host-side
        validation rejected this as invalid, returning rc=-1 from every
        GroupQueryAttention layer, which cascaded to all-zero logits and
        the LM emitting token id 0 (``!``) for every step.

        The window is marked as a first prompt the way ORT marks one:
        total_sequence_length equals the window (seq_len). The runtime then
        clamps past_len to 0 and lets Q[0..sq-1] attend over
        K[0..total_seq-1]; Q positions beyond real_count are padding the
        model discards. A padded window with any other total (a step after a
        past, a total of the cache size) is refused instead - see
        test_gqa_padded_window_is_rejected.

        Verification: the EP's output for the first ``real_count`` Q
        positions, and the cache slots they fill, are compared with an
        *unpadded* equivalent model (seq_len=real_count) on ORT CPU. That
        reference is independent of the padding convention; each Q position
        only attends to K[0..min(i, total_seq-1)], so the padded tail does
        not enter it.
        """
        assert real_count <= seq_len
        total_kv_seq = past_valid + real_count

        rng = np.random.default_rng(42)
        # Real Q/K/V values for the [0, real_count) prefix.
        q_real = rng.uniform(-1, 1, [1, real_count, LLAMA_HIDDEN]).astype(np_dtype)
        k_real = rng.uniform(-1, 1, [1, real_count, LLAMA_KV_HIDDEN]).astype(np_dtype)
        v_real = rng.uniform(-1, 1, [1, real_count, LLAMA_KV_HIDDEN]).astype(np_dtype)
        # Padded tail uses zeros to mimic genai's padded embeddings.
        pad_q = np.zeros([1, seq_len - real_count, LLAMA_HIDDEN], dtype=np_dtype)
        pad_k = np.zeros([1, seq_len - real_count, LLAMA_KV_HIDDEN], dtype=np_dtype)
        pad_v = np.zeros([1, seq_len - real_count, LLAMA_KV_HIDDEN], dtype=np_dtype)
        q_full = np.concatenate([q_real, pad_q], axis=1)
        k_full = np.concatenate([k_real, pad_k], axis=1)
        v_full = np.concatenate([v_real, pad_v], axis=1)

        pk = rng.uniform(
            -1,
            1,
            [1, LLAMA_KV_NUM_HEADS, LLAMA_CACHE_SIZE, LLAMA_HEAD_DIM],
        ).astype(np_dtype)
        pv = rng.uniform(
            -1,
            1,
            [1, LLAMA_KV_NUM_HEADS, LLAMA_CACHE_SIZE, LLAMA_HEAD_DIM],
        ).astype(np_dtype)
        if past_valid == 0:
            pk[:] = 0
            pv[:] = 0

        # ---- Padded model on the EP; the unpadded model below is the
        # reference.
        wide_model = _make_gqa_fixed_cache_model(
            1,
            seq_len,
            LLAMA_CACHE_SIZE,
            LLAMA_HIDDEN,
            LLAMA_KV_HIDDEN,
            LLAMA_NUM_HEADS,
            LLAMA_KV_NUM_HEADS,
            elem_type,
        )
        sub = model_runner._next_subdir("wide_padded")
        wide_path = str(sub / "model.onnx")
        (sub / "model.onnx").write_bytes(wide_model.SerializeToString())
        seqlens_full = np.array([total_kv_seq - 1], dtype=np.int32)
        # A padded first prompt is marked as ORT marks one: its total is the
        # window (seq_len), not the cache size.
        total_full = np.array(seq_len, dtype=np.int32)
        ep_outputs_full = model_runner.backend.run(
            wide_path,
            [q_full, k_full, v_full, pk, pv, seqlens_full, total_full],
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
            elem_type,
        )
        seqlens_narrow = np.array([total_kv_seq - 1], dtype=np.int32)
        total = np.array(LLAMA_CACHE_SIZE, dtype=np.int32)
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

        # ---- The KV-cache append and RoPE kernels derive the past from
        # seqlens_k on the device as well. Unclamped, the negative past made
        # the append kernels skip the batch, leaving the cache unwritten, so
        # the prompt's keys and values must land in slots [0, real_count).
        for ep_cache, cpu_cache, name in (
            (ep_outputs_full[1], narrow_expected[1], "present_key"),
            (ep_outputs_full[2], narrow_expected[2], "present_value"),
        ):
            ep_slots = ep_cache[:, :, :real_count, :]
            cpu_slots = cpu_cache[:, :, :real_count, :]
            compare_outputs([ep_slots], [cpu_slots], atol=2e-3)
            assert np.any(ep_slots != 0), (
                f"{name} slots [0, {real_count}) are all-zero; the padded "
                "prompt was not appended to the KV cache."
            )

    @pytest.mark.parametrize(
        "elem_type,np_dtype",
        [(TensorProto.FLOAT16, np.float16), (TensorProto.FLOAT, np.float32)],
        ids=["fp16_fused", "fp32_hipblaslt"],
    )
    @pytest.mark.parametrize("real_count", [1, 22, 128])
    def test_gqa_padded_first_prompt_rotary(
        self, model_runner, real_count, elem_type, np_dtype
    ):
        """A padded first prompt with RoPE inside GQA, marked as ORT marks one.

        The padded window (total_sequence_length = the window) must give the
        token rows, and the cache slots they fill, of the same tokens run
        unpadded on ORT CPU - which checks the rotary positions too, and,
        since the padding rows hold nonzero values, that they are not read.
        ORT CPU also accepts the padded graph itself; the EP must match it as
        well.
        """
        seq_len = 128
        rng = np.random.default_rng(5)
        pad = seq_len - real_count

        def rows(width):
            return rng.uniform(-1, 1, [1, real_count, width]).astype(np_dtype)

        def padded(x):
            # Finite, nonzero padding: the token rows and their cache slots
            # must not depend on what the padding rows hold.
            fill = rng.uniform(-4, 4, [1, pad, x.shape[2]]).astype(np_dtype)
            return np.concatenate([x, fill], 1)

        q, k, v = rows(LLAMA_HIDDEN), rows(LLAMA_KV_HIDDEN), rows(LLAMA_KV_HIDDEN)
        cos, sin = _rope_tables(np_dtype)
        cache = np.zeros(
            [1, LLAMA_KV_NUM_HEADS, LLAMA_CACHE_SIZE, LLAMA_HEAD_DIM], np_dtype
        )
        seqlens = np.array([real_count - 1], np.int32)
        model = _make_gqa_fixed_cache_rotary_model(seq_len, elem_type)
        inputs = [padded(q), padded(k), padded(v), cache, cache.copy(), seqlens]
        inputs += [np.array(seq_len, np.int32), cos, sin]
        actual, same_graph = model_runner.run_sample(model, inputs, reference="cpu")
        unpadded = _cpu_gqa(
            _make_gqa_fixed_cache_rotary_model(real_count, elem_type),
            [q, k, v, cache, cache.copy(), seqlens, np.array(real_count, np.int32)]
            + [cos, sin],
        )
        for expected in (unpadded, same_graph):
            compare_outputs(
                [actual[0][:, :real_count]]
                + [actual[i][:, :, :real_count] for i in (1, 2)],
                [expected[0][:, :real_count]]
                + [expected[i][:, :, :real_count] for i in (1, 2)],
                atol=2e-3,
            )

    @pytest.mark.parametrize(
        "elem_type,np_dtype",
        [(TensorProto.FLOAT16, np.float16), (TensorProto.FLOAT, np.float32)],
        ids=["fp16_fused", "fp32_hipblaslt"],
    )
    def test_gqa_padded_prefill_then_decode_in_place(
        self, model_runner, elem_type, np_dtype
    ):
        """A padded first prompt, then a decode step, on caches shared in place.

        past_* and present_* are bound to the same GPU buffers, as
        onnxruntime-genai binds a shared cache. The step after the padded
        prompt must see exactly its tokens: output and cache slots match the
        same sequence run unpadded on ORT CPU.
        """
        seq_len, real_count = 128, 22
        rng = np.random.default_rng(9)

        def rows(n, width):
            return rng.uniform(-1, 1, [1, n, width]).astype(np_dtype)

        q, k, v = (
            rows(real_count, LLAMA_HIDDEN),
            rows(real_count, LLAMA_KV_HIDDEN),
            rows(real_count, LLAMA_KV_HIDDEN),
        )
        dq, dk, dv = (
            rows(1, LLAMA_HIDDEN),
            rows(1, LLAMA_KV_HIDDEN),
            rows(1, LLAMA_KV_HIDDEN),
        )
        shape = [1, LLAMA_KV_NUM_HEADS, LLAMA_CACHE_SIZE, LLAMA_HEAD_DIM]
        cache = np.zeros(shape, np_dtype)

        def padded(x):
            pad = np.zeros([1, seq_len - real_count, x.shape[2]], np_dtype)
            return np.concatenate([x, pad], 1)

        def model(n):
            return _make_gqa_fixed_cache_model(
                1,
                n,
                LLAMA_CACHE_SIZE,
                LLAMA_HIDDEN,
                LLAMA_KV_HIDDEN,
                LLAMA_NUM_HEADS,
                LLAMA_KV_NUM_HEADS,
                elem_type=elem_type,
            )

        # ORT CPU, unpadded: the prompt, then the decode step on its caches.
        ref1 = _cpu_gqa(
            model(real_count),
            [
                q,
                k,
                v,
                cache,
                cache.copy(),
                np.array([real_count - 1], np.int32),
                np.array(real_count, np.int32),
            ],
        )
        ref2 = _cpu_gqa(
            model(1),
            [
                dq,
                dk,
                dv,
                ref1[1],
                ref1[2],
                np.array([real_count], np.int32),
                np.array(real_count + 1, np.int32),
            ],
        )

        cache_k, cache_v = _gpu(cache.copy()), _gpu(cache.copy())
        steps = [
            (
                _ep_session(model_runner, model(seq_len), "padded_prefill"),
                [padded(q), padded(k), padded(v)],
                real_count - 1,
                seq_len,
            ),
            (
                _ep_session(model_runner, model(1), "decode"),
                [dq, dk, dv],
                real_count,
                real_count + 1,
            ),
        ]
        outputs = []
        for sess, (sq, sk, sv), seqlens, total in steps:
            io = sess.io_binding()
            for name, arr in zip(["query", "key", "value"], [sq, sk, sv]):
                io.bind_cpu_input(name, arr)
            io.bind_cpu_input("seqlens_k", np.array([seqlens], np.int32))
            io.bind_cpu_input("total_seq", np.array(total, np.int32))
            io.bind_ortvalue_input("past_key", cache_k)
            io.bind_ortvalue_input("past_value", cache_v)
            io.bind_output("output")
            io.bind_ortvalue_output("present_key", cache_k)
            io.bind_ortvalue_output("present_value", cache_v)
            sess.run_with_iobinding(io)
            outputs.append(io.copy_outputs_to_cpu()[0])

        n = real_count + 1
        compare_outputs(
            [
                outputs[0][:, :real_count],
                outputs[1],
                cache_k.numpy()[:, :, :n],
                cache_v.numpy()[:, :, :n],
            ],
            [ref1[0], ref2[0], ref2[1][:, :, :n], ref2[2][:, :, :n]],
            atol=2e-3,
        )

    @pytest.mark.parametrize(
        "elem_type,np_dtype",
        [(TensorProto.FLOAT16, np.float16), (TensorProto.FLOAT, np.float32)],
        ids=["fp16_fused", "fp32_hipblaslt"],
    )
    @pytest.mark.parametrize(
        "past,total",
        [
            # A padded step after a past: total is the past plus its tokens.
            # Read as a first prompt it would overwrite the cached tokens.
            (40, 50),
            # A padded window whose total is its token count, not the window.
            (0, 10),
            # A padded window whose total is the KV cache size: what an
            # onnxruntime-genai sliding-window model passes, whether it pads
            # after its tokens or before them (its default), which GQA's
            # inputs cannot tell apart.
            (0, LLAMA_CACHE_SIZE),
        ],
        ids=["after_past", "total_is_token_count", "total_is_cache_size"],
    )
    def test_gqa_padded_window_is_rejected(
        self, model_runner, capfd, past, total, elem_type, np_dtype
    ):
        """A padded window not marked as a first prompt is refused, cache intact.

        The runtime's status is dropped by the generated call today, so the
        refusal is checked by its own diagnostic, captured from this run only,
        and by the caches: past_* and present_* are bound to the same GPU
        buffers (in place), filled with distinct values, and must come back
        byte for byte unchanged.
        """
        seq_len, real_count = 128, 10
        rng = np.random.default_rng(11)

        def rows(width):
            real = rng.uniform(-1, 1, [1, real_count, width]).astype(np_dtype)
            pad = np.zeros([1, seq_len - real_count, width], np_dtype)
            return np.concatenate([real, pad], 1)

        shape = [1, LLAMA_KV_NUM_HEADS, LLAMA_CACHE_SIZE, LLAMA_HEAD_DIM]
        count = int(np.prod(shape))
        before_k = ((np.arange(count) % 2039) / 2039.0 - 0.5).astype(np_dtype)
        before_v = ((np.arange(count) % 1987) / 1987.0 + 0.25).astype(np_dtype)
        before_k, before_v = before_k.reshape(shape), before_v.reshape(shape)
        cache_k, cache_v = _gpu(before_k.copy()), _gpu(before_v.copy())

        sess = _ep_session(
            model_runner,
            _make_gqa_fixed_cache_model(
                1,
                seq_len,
                LLAMA_CACHE_SIZE,
                LLAMA_HIDDEN,
                LLAMA_KV_HIDDEN,
                LLAMA_NUM_HEADS,
                LLAMA_KV_NUM_HEADS,
                elem_type=elem_type,
            ),
            "padded_rejected",
        )
        io = sess.io_binding()
        for name, width in (
            ("query", LLAMA_HIDDEN),
            ("key", LLAMA_KV_HIDDEN),
            ("value", LLAMA_KV_HIDDEN),
        ):
            io.bind_cpu_input(name, rows(width))
        io.bind_cpu_input("seqlens_k", np.array([past + real_count - 1], np.int32))
        io.bind_cpu_input("total_seq", np.array(total, np.int32))
        io.bind_ortvalue_input("past_key", cache_k)
        io.bind_ortvalue_input("past_value", cache_v)
        io.bind_output("output")
        io.bind_ortvalue_output("present_key", cache_k)
        io.bind_ortvalue_output("present_value", cache_v)

        capfd.readouterr()  # only this inference's output below
        sess.run_with_iobinding(io)
        err = capfd.readouterr().err

        expected = (
            f"invalid seqlens_k[0]+1={past + real_count} (sq={seq_len}, "
            f"past_len={past + real_count - seq_len},"
        )
        assert expected in err, f"no refusal diagnostic in:\n{err}"
        path_failed = (
            "gqa_forward_fused failed (rc=-1)"
            if np_dtype == np.float16
            else "legacy decomposed pipeline failed (rc=-1)"
        )
        assert path_failed in err, err
        assert cache_k.numpy().tobytes() == before_k.tobytes(), "present_key changed"
        assert cache_v.numpy().tobytes() == before_v.tobytes(), "present_value changed"

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
