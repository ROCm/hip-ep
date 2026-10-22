#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""Tests for com.microsoft DynamicSparseAttention (attention_mode=selected_only,
selected_kv_source=main, separate Q/K/V with a past cache, fp16).

ORT CPU has no DynamicSparseAttention kernel, so the reference is a NumPy port
of the ORT CUDA semantics (fp32 logits and softmax, invalid candidates skipped,
empty candidate set -> 0, K/V written at seqlens_k + 1 - S + s when inside the
cache).
"""

import numpy as np
import pytest
from onnx import TensorProto as TP
from onnx import helper

from framework.comparator import compare_outputs
from framework.onnx_utils import make_model_from_nodes

_OPSETS = dict(opset=22, extra_opsets=[helper.make_opsetid("com.microsoft", 1)])
_FLT_MAX = np.float32(np.finfo(np.float32).max)


def _make_model(N, KV, D, K, scale, symbolic_head_dim=False):
    attrs = dict(
        num_heads=N,
        kv_num_heads=KV,
        is_causal=1,
        attention_mode="selected_only",
        selected_kv_source="main",
        do_rotary=0,
        rotary_interleaved=0,
        qk_norm_epsilon=1e-6,
    )
    if scale is not None:
        attrs["scale"] = scale
    node = helper.make_node(
        "DynamicSparseAttention",
        [
            "query",
            "key",
            "value",
            "past_key",
            "past_value",
            "",
            "",
            "selected_indices",
            "selected_counts",
            "seqlens_k",
            "total_sequence_length",
            "",
            "",
            "",
            "",
            "",
            "",
        ],
        ["output", "present_key", "present_value"],
        domain="com.microsoft",
        **attrs,
    )
    cache = ["B", KV, "cap", "kv_cache_dim" if symbolic_head_dim else D]
    inputs = [
        helper.make_tensor_value_info("query", TP.FLOAT16, ["B", "S", N * D]),
        helper.make_tensor_value_info("key", TP.FLOAT16, ["B", "S", KV * D]),
        helper.make_tensor_value_info("value", TP.FLOAT16, ["B", "S", KV * D]),
        helper.make_tensor_value_info("past_key", TP.FLOAT16, cache),
        helper.make_tensor_value_info("past_value", TP.FLOAT16, cache),
        helper.make_tensor_value_info("selected_indices", TP.INT32, ["R", K]),
        helper.make_tensor_value_info("selected_counts", TP.INT32, ["R"]),
        helper.make_tensor_value_info("seqlens_k", TP.INT32, ["B"]),
        helper.make_tensor_value_info("total_sequence_length", TP.INT32, []),
    ]
    outputs = [
        helper.make_tensor_value_info("output", TP.FLOAT16, ["B", "S", N * D]),
        helper.make_tensor_value_info("present_key", TP.FLOAT16, cache),
        helper.make_tensor_value_info("present_value", TP.FLOAT16, cache),
    ]
    model = make_model_from_nodes([node], inputs, outputs, **_OPSETS)
    model.ir_version = 10
    return model


def _dsa_reference(q, k, v, past_k, past_v, sel, counts, seqlens, N, KV, scale):
    """Returns (output, present_key, present_value) per ORT v1.31 CUDA."""
    B, S, _ = q.shape
    cap, D = past_k.shape[2], past_k.shape[3]
    present_k = past_k.copy()
    present_v = past_v.copy()
    k_in = k.reshape(B, S, KV, D)
    v_in = v.reshape(B, S, KV, D)
    for b in range(B):
        for s in range(S):
            dest = int(seqlens[b]) + 1 - S + s
            if 0 <= dest < cap:
                present_k[b, :, dest] = k_in[b, s]
                present_v[b, :, dest] = v_in[b, s]

    pk = present_k.astype(np.float32)
    pv = present_v.astype(np.float32)
    q32 = q.astype(np.float32).reshape(B, S, N, D)
    sc = np.float32(scale)
    max_sel = sel.shape[1]
    group = N // KV
    out = np.zeros((B, S, N, D), np.float32)
    for b in range(B):
        L = int(seqlens[b]) + 1
        if not S <= L <= cap:
            continue
        for s in range(S):
            row = b * S + s
            qpos = L - S + s
            count = max(0, min(int(counts[row]), max_sel))
            cand = sel[row, :count].astype(np.int64)
            cand = cand[(cand >= 0) & (cand < L) & (cand <= qpos) & (cand < cap)]
            if cand.size == 0:
                continue
            for g in range(KV):
                heads = q32[b, s, g * group : (g + 1) * group]
                logits = (heads @ pk[b, g, cand].T).astype(np.float32) * sc
                m = np.maximum(logits.max(axis=1, keepdims=True), -_FLT_MAX)
                w = np.exp(logits - m, dtype=np.float32)
                den = w.sum(axis=1, keepdims=True, dtype=np.float32)
                num = (w @ pv[b, g, cand]).astype(np.float32)
                out[b, s, g * group : (g + 1) * group] = np.where(
                    den == 0, 0.0, num / np.where(den == 0, 1, den)
                )
    return out.reshape(B, S, N * D).astype(np.float16), present_k, present_v


def _dense_causal_reference(q, present_k, present_v, seqlens, N, KV, scale):
    B, S, _ = q.shape
    D = present_k.shape[3]
    group = N // KV
    q32 = q.astype(np.float64).reshape(B, S, N, D)
    out = np.zeros((B, S, N, D))
    for b in range(B):
        L = int(seqlens[b]) + 1
        for s in range(S):
            qpos = L - S + s
            for h in range(N):
                kk = present_k[b, h // group, : qpos + 1].astype(np.float64)
                vv = present_v[b, h // group, : qpos + 1].astype(np.float64)
                x = kk @ q32[b, s, h] * scale
                p = np.exp(x - x.max())
                out[b, s, h] = p @ vv / p.sum()
    return out.reshape(B, S, N * D)


def _block_indices(rng, qpos, count):
    """SparseAttentionIndexer-style list: ascending blocks of 4 below the
    query position, then a tail of the most recent positions."""
    n_tail = min(3, count, qpos + 1)
    tail = np.arange(qpos + 1 - n_tail, qpos + 1)
    nblocks = -(-(count - n_tail) // 4)
    avail = max(1, (qpos + 1) // 4)
    starts = rng.choice(avail, size=nblocks, replace=nblocks > avail)
    blocks = np.sort(starts)[:, None] * 4 + np.arange(4)[None, :]
    idx = np.concatenate([blocks.reshape(-1)[: count - n_tail], tail])
    return idx.astype(np.int32)


def _run(
    model_runner,
    q,
    k,
    v,
    past_k,
    past_v,
    sel,
    counts,
    seqlens,
    N,
    KV,
    scale,
    symbolic_head_dim=False,
):
    D = past_k.shape[3]
    model = _make_model(N, KV, D, sel.shape[1], scale, symbolic_head_dim)
    total = np.array(
        min(max(int(seqlens.max()) + 1, q.shape[1]), past_k.shape[2]), np.int32
    )
    actual = model_runner.backend.run(
        _persist(model_runner, model),
        [q, k, v, past_k, past_v, sel, counts, seqlens, total],
    )
    eff_scale = scale if scale is not None else 1.0 / np.sqrt(D)
    expected = _dsa_reference(
        q, k, v, past_k, past_v, sel, counts, seqlens, N, KV, eff_scale
    )
    compare_outputs(actual[:1], expected[:1], atol=5e-3, rtol=0.0)
    np.testing.assert_array_equal(actual[1], expected[1])
    np.testing.assert_array_equal(actual[2], expected[2])
    return actual, expected


def _inputs(rng, B, S, N, KV, D, cap):
    def u(*shape):
        return rng.uniform(-1, 1, shape).astype(np.float16)

    return (
        u(B, S, N * D),
        u(B, S, KV * D),
        u(B, S, KV * D),
        u(B, KV, cap, D),
        u(B, KV, cap, D),
    )


_DECODE_CASES = [
    # (L, count, kind)
    (1, 1, "block"),
    (37, 37, "block"),
    (37, 0, "block"),
    (3000, 1, "block"),
    (3000, 2051, "block"),
    (3000, 64, "skipped"),
]


@pytest.mark.parametrize(
    "L,count,kind",
    _DECODE_CASES,
    ids=[f"L{c[0]}_n{c[1]}_{c[2]}" for c in _DECODE_CASES],
)
def test_decode_qwen_geometry(model_runner, L, count, kind):
    N, KV, D, cap, K = 24, 2, 256, 4096, 2051
    rng = np.random.default_rng(L * 7919 + count)
    q, k, v, past_k, past_v = _inputs(rng, 1, 1, N, KV, D, cap)
    # Rows past the live length are never read; NaN there catches a kernel
    # that touches a skipped candidate.
    past_k[:, :, L:] = np.nan
    past_v[:, :, L:] = np.nan
    qpos = L - 1
    sel = np.full((1, K), -1, np.int32)
    if kind == "block":
        sel[0, :count] = _block_indices(rng, qpos, count)
    else:
        idx = _block_indices(rng, qpos, count)
        idx[::5] = idx[1::5][: idx[::5].size]  # duplicates
        idx[3::16] = -7
        idx[7::16] = L + 5  # beyond the live length
        idx[11::16] = cap + 3  # beyond the cache
        sel[0, :count] = idx
    counts = np.array([count], np.int32)
    seqlens = np.array([L - 1], np.int32)
    actual, _ = _run(
        model_runner, q, k, v, past_k, past_v, sel, counts, seqlens, N, KV, 0.0625
    )
    if count == 0:
        assert not np.any(actual[0])


@pytest.mark.parametrize("S,past", [(64, 0), (64, 100), (128, 0)])
def test_prefill_causal_subsets(model_runner, S, past):
    N, KV, D, cap, K = 24, 2, 256, 256, 48
    rng = np.random.default_rng(past + 11)
    q, k, v, past_k, past_v = _inputs(rng, 1, S, N, KV, D, cap)
    L = past + S
    sel = np.full((S, K), -1, np.int32)
    counts = np.zeros(S, np.int32)
    for s in range(S):
        qpos = L - S + s
        n = min(qpos + 1, K)
        sel[s, :n] = np.sort(rng.choice(qpos + 1, size=n, replace=False))
        counts[s] = n
    seqlens = np.array([L - 1], np.int32)
    _run(model_runner, q, k, v, past_k, past_v, sel, counts, seqlens, N, KV, 0.0625)


def test_batch_with_invalid_rows(model_runner):
    """Batch 0 has fewer cached tokens than new ones (L < S), batch 2 more than
    the cache holds (L > Cap); both produce zeros but still write the tokens
    that land inside the cache."""
    N, KV, D, S, cap, K = 24, 2, 256, 4, 128, 64
    rng = np.random.default_rng(5)
    B = 3
    q, k, v, past_k, past_v = _inputs(rng, B, S, N, KV, D, cap)
    seqlens = np.array([1, 99, cap + 1], np.int32)
    sel = np.full((B * S, K), -1, np.int32)
    counts = np.zeros(B * S, np.int32)
    for b in range(B):
        L = int(seqlens[b]) + 1
        for s in range(S):
            qpos = min(L - S + s, cap - 1)
            n = min(max(qpos + 1, 1), K)
            sel[b * S + s, :n] = _block_indices(rng, max(qpos, 0), n)
            counts[b * S + s] = n + 3  # counts past the valid prefix read -1
    actual, _ = _run(
        model_runner, q, k, v, past_k, past_v, sel, counts, seqlens, N, KV, 0.0625
    )
    out = actual[0].reshape(B, S, -1)
    assert not np.any(out[0]) and not np.any(out[2]) and np.any(out[1])


def test_all_positions_selected_is_dense_causal(model_runner):
    N, KV, D, S, P, cap = 24, 2, 256, 16, 40, 64
    K = P + S
    rng = np.random.default_rng(3)
    q, k, v, past_k, past_v = _inputs(rng, 1, S, N, KV, D, cap)
    sel = np.full((S, K), -1, np.int32)
    counts = np.zeros(S, np.int32)
    for s in range(S):
        sel[s, : P + s + 1] = np.arange(P + s + 1)
        counts[s] = P + s + 1
    seqlens = np.array([P + S - 1], np.int32)
    actual, expected = _run(
        model_runner, q, k, v, past_k, past_v, sel, counts, seqlens, N, KV, 0.0625
    )
    dense = _dense_causal_reference(q, expected[1], expected[2], seqlens, N, KV, 0.0625)
    compare_outputs(actual[:1], [dense], atol=5e-3, rtol=0.0)


def test_head_dim_128_default_scale(model_runner):
    N, KV, D, S, P, cap, K = 16, 2, 128, 3, 70, 96, 40
    rng = np.random.default_rng(17)
    q, k, v, past_k, past_v = _inputs(rng, 1, S, N, KV, D, cap)
    L = P + S
    sel = np.full((S, K), -1, np.int32)
    counts = np.full(S, K, np.int32)
    for s in range(S):
        sel[s] = _block_indices(rng, L - S + s, K)
    seqlens = np.array([L - 1], np.int32)
    _run(model_runner, q, k, v, past_k, past_v, sel, counts, seqlens, N, KV, None)


def test_symbolic_cache_head_dim(model_runner):
    N, KV, D, S, P, cap, K = 24, 2, 256, 2, 61, 80, 36
    rng = np.random.default_rng(23)
    q, k, v, past_k, past_v = _inputs(rng, 1, S, N, KV, D, cap)
    L = P + S
    sel = np.full((S, K), -1, np.int32)
    counts = np.full(S, K, np.int32)
    for s in range(S):
        sel[s] = _block_indices(rng, L - S + s, K)
    seqlens = np.array([L - 1], np.int32)
    _run(
        model_runner,
        q,
        k,
        v,
        past_k,
        past_v,
        sel,
        counts,
        seqlens,
        N,
        KV,
        0.0625,
        symbolic_head_dim=True,
    )


@pytest.mark.parametrize("N,KV,D", [(2, 2, 256), (4, 4, 128), (32, 2, 128)])
def test_heads_per_group(model_runner, N, KV, D):
    S, P, cap, K = 3, 90, 128, 40
    rng = np.random.default_rng(N * 31 + KV)
    q, k, v, past_k, past_v = _inputs(rng, 1, S, N, KV, D, cap)
    L = P + S
    sel = np.full((S, K), -1, np.int32)
    counts = np.full(S, K, np.int32)
    for s in range(S):
        sel[s] = _block_indices(rng, L - S + s, K)
    seqlens = np.array([L - 1], np.int32)
    _run(model_runner, q, k, v, past_k, past_v, sel, counts, seqlens, N, KV, None)


def test_batch2_decode_split(model_runner):
    """Two decode rows with long candidate lists take the split-K path."""
    N, KV, D, cap, K = 24, 2, 256, 4096, 2051
    B = 2
    rng = np.random.default_rng(29)
    q, k, v, past_k, past_v = _inputs(rng, B, 1, N, KV, D, cap)
    seqlens = np.array([2999, 3700], np.int32)
    counts = np.array([K, 1500], np.int32)
    sel = np.full((B, K), -1, np.int32)
    for b in range(B):
        sel[b, : counts[b]] = _block_indices(rng, int(seqlens[b]), int(counts[b]))
    _run(model_runner, q, k, v, past_k, past_v, sel, counts, seqlens, N, KV, 0.0625)


def _persist(model_runner, model):
    sub = model_runner._next_subdir()
    path = sub / "model.onnx"
    path.write_bytes(model.SerializeToString())
    return str(path)
