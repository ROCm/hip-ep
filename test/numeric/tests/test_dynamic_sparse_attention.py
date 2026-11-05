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

The fused cases surround the op with the graph the compiler folds into it: a
Split of one packed qkv input, per-head SimplifiedLayerNormalization and
MRotaryEmbedding of q and k, and Mul(output, Sigmoid(gate)).
"""

import numpy as np
import pytest
from onnx import TensorProto as TP
from onnx import helper, numpy_helper

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


def test_shared_past_present_decode(model_runner):
    """OGA-style decode: past_* and present_* are bound to the same device
    buffers, so the cache is updated in place across two steps."""
    import onnxruntime as ort

    backend = model_runner.backend
    if not hasattr(backend, "create_session"):
        pytest.skip("backend does not expose an ORT session for IO binding")
    N, KV, D, cap, K = 24, 2, 256, 512, 96
    rng = np.random.default_rng(31)
    sess = backend.create_session(
        _persist(model_runner, _make_model(N, KV, D, K, 0.0625))
    )
    _, _, _, past_k, past_v = _inputs(rng, 1, 1, N, KV, D, cap)
    L = 300
    past_k[:, :, L:] = np.nan
    past_v[:, :, L:] = np.nan

    def gpu(arr):
        return ort.OrtValue.ortvalue_from_numpy(
            arr, device_type="gpu", vendor_id=0x1002
        )

    cache_k, cache_v = gpu(past_k), gpu(past_v)
    for step in range(2):
        q, k, v, _, _ = _inputs(rng, 1, 1, N, KV, D, 1)
        seqlens = np.array([L - 1 + step], np.int32)
        sel = _block_indices(rng, L - 1 + step, K)[None, :]
        counts = np.array([K], np.int32)
        total = np.array(L + step, np.int32)
        io = sess.io_binding()
        for name, arr in zip(
            ["query", "key", "value", "selected_indices", "selected_counts"]
            + ["seqlens_k", "total_sequence_length"],
            [q, k, v, sel, counts, seqlens, total],
        ):
            io.bind_cpu_input(name, arr)
        io.bind_ortvalue_input("past_key", cache_k)
        io.bind_ortvalue_input("past_value", cache_v)
        io.bind_output("output")
        io.bind_ortvalue_output("present_key", cache_k)
        io.bind_ortvalue_output("present_value", cache_v)
        sess.run_with_iobinding(io)

        out, past_k, past_v = _dsa_reference(
            q, k, v, past_k, past_v, sel, counts, seqlens, N, KV, 0.0625
        )
        compare_outputs([io.copy_outputs_to_cpu()[0]], [out], atol=5e-3, rtol=0.0)
        np.testing.assert_array_equal(cache_k.numpy(), past_k)
        np.testing.assert_array_equal(cache_v.numpy(), past_v)


def _selection(rng, B, S, P, width):
    """Flattened [B * S, width] candidate lists, -1 padded, and their counts."""
    flat = np.full((B * S, width), -1, np.int32)
    counts = np.zeros(B * S, np.int32)
    for row in range(B * S):
        qpos = P + row % S
        n = min(qpos + 1, width)
        flat[row, :n] = _block_indices(rng, qpos, n)
        counts[row] = n
    return flat, counts


def _rope_base_tables(rows, half, theta=1e7):
    """Standard RoPE base tables [rows, half]: cos/sin of pos * theta^(-f/half),
    the layout of the exported cos_cache/sin_cache initializers."""
    inv_freq = theta ** (-np.arange(half, dtype=np.float64) / half)
    ang = np.arange(rows, dtype=np.float64)[:, None] * inv_freq[None, :]
    return np.cos(ang).astype(np.float16), np.sin(ang).astype(np.float16)


def _step_positions(B, P, S):
    """[3, B, S] T/H/W positions of the S tokens after P cached ones. The rows
    differ, so a frequency that reads the wrong section shows in the output."""
    pos = np.arange(P, P + S)
    pos = np.stack([pos, pos // 2, (3 * pos) // 4])
    return np.broadcast_to(pos[:, None, :], (3, B, S)).copy()


def _mrope_sections(half, section, layout):
    """MRoPE section (0=T, 1=H, 2=W) of each rotary frequency, as mrope_stream
    in rope_kernel.hip. Layout 0 uses contiguous sections; layout 1 puts H and
    W on every third frequency."""
    sec_t, sec_h, sec_w = section
    f = np.arange(half)
    if layout == 0:
        return np.where(f < sec_t, 0, np.where(f < sec_t + sec_h, 1, 2))
    h = (f % 3 == 1) & (f < 3 * sec_h)
    w = (f % 3 == 2) & (f < 3 * sec_w)
    return np.where(h, 1, np.where(w, 2, 0))


def _norm_rope(x, heads, weight, position_ids, cos_base, sin_base, c):
    """The exported per-head SimplifiedLayerNormalization (fp32 stash), then
    MRotaryEmbedding (interleaved=0, fp32 rotation rounded once)."""
    B, S, W = x.shape
    x32 = x.reshape(B, S, heads, -1).astype(np.float32)
    ms = np.mean(x32 * x32, axis=-1, keepdims=True, dtype=np.float32)
    inv = np.float32(1.0) / np.sqrt(ms + np.float32(c["eps"]), dtype=np.float32)
    normed = (x32 * inv * weight.astype(np.float32)).astype(np.float16)
    half = c["rotary_dim"] // 2
    sections = _mrope_sections(half, c["section"], c["layout"])
    rows = np.moveaxis(position_ids[sections], 0, -1)
    cs = cos_base[rows, np.arange(half)].astype(np.float32)[:, :, None, :]
    sn = sin_base[rows, np.arange(half)].astype(np.float32)[:, :, None, :]
    x0 = normed[..., :half].astype(np.float32)
    x1 = normed[..., half : 2 * half].astype(np.float32)
    out = normed.copy()
    out[..., :half] = (x0 * cs - x1 * sn).astype(np.float16)
    out[..., half : 2 * half] = (x1 * cs + x0 * sn).astype(np.float16)
    return out.reshape(B, S, W)


def _output_gate(attn, gate):
    """Mul(attn, Sigmoid(gate)) in fp16: the sigmoid rounds to fp16 before the
    product, which rounds again."""
    sig = (1.0 / (1.0 + np.exp(-gate.astype(np.float32)))).astype(np.float16)
    return (attn.astype(np.float32) * sig.astype(np.float32)).astype(np.float16)


def _constant(nodes, name, values):
    nodes.append(
        helper.make_node(
            "Constant",
            [],
            [name],
            value=numpy_helper.from_array(np.array(values, np.int64)),
        )
    )
    return name


def _make_fused_model(c, gate_width):
    """DSA with the optional packed-qkv Split, q/k norm + MRoPE prologue and
    output gate of case ``c``; ``gate_width`` 0 means no gate."""
    N, KV, D = c["N"], c["KV"], c["D"]
    nodes, inits = [], []
    f16 = TP.FLOAT16
    inputs = []
    if c["packed"]:
        src = dict(q="q0", k="k0", v="v0")
        widths = _constant(nodes, "widths", [N * D, KV * D, KV * D])
        nodes.append(
            helper.make_node("Split", ["qkv", widths], list(src.values()), axis=-1)
        )
        inputs.append(
            helper.make_tensor_value_info("qkv", f16, ["B", "S", (N + 2 * KV) * D])
        )
    else:
        src = dict(q="query", k="key", v="value")
        inputs += [
            helper.make_tensor_value_info(src[x], f16, ["B", "S", h * D])
            for x, h in (("q", N), ("k", KV), ("v", KV))
        ]
    q, k = src["q"], src["k"]
    if c["prologue"]:
        per_head = _constant(nodes, "per_head", [0, -1, D])
        rd = c["rotary_dim"]
        cos, sin = _rope_base_tables(c["table_rows"], rd // 2)
        inits += [
            numpy_helper.from_array(cos, "cos_cache"),
            numpy_helper.from_array(sin, "sin_cache"),
        ]
        rotated = []
        for x, heads in (("q", N), ("k", KV)):
            inits.append(numpy_helper.from_array(c[f"{x}_weight"], f"{x}_weight"))
            merged = _constant(nodes, f"{x}_merged", [0, -1, heads * D])
            nodes += [
                helper.make_node("Reshape", [src[x], per_head], [f"{x}_heads"]),
                helper.make_node(
                    "SimplifiedLayerNormalization",
                    [f"{x}_heads", f"{x}_weight"],
                    [f"{x}_normed"],
                    epsilon=c["eps"],
                    axis=-1,
                    stash_type=1,
                ),
                helper.make_node("Reshape", [f"{x}_normed", merged], [f"{x}_flat"]),
                helper.make_node(
                    "MRotaryEmbedding",
                    [f"{x}_flat", "position_ids", "cos_cache", "sin_cache"],
                    [f"{x}_rot"],
                    domain="com.microsoft",
                    interleaved=0,
                    rotary_embedding_dim=rd,
                    num_heads=heads,
                    mrope_section=c["section"],
                    mrope_layout=c["layout"],
                    is_packed_batching=0,
                ),
            ]
            rotated.append(f"{x}_rot")
        q, k = rotated
        inputs.append(
            helper.make_tensor_value_info("position_ids", TP.INT64, [3, "B", "S"])
        )
    attn = "attn" if gate_width else "output"
    nodes.append(
        helper.make_node(
            "DynamicSparseAttention",
            [q, k, src["v"], "past_key", "past_value", "", ""]
            + ["selected_indices", "selected_counts", "seqlens_k"]
            + ["total_sequence_length"]
            + [""] * 6,
            [attn, "present_key", "present_value"],
            domain="com.microsoft",
            num_heads=N,
            kv_num_heads=KV,
            scale=c["scale"],
            is_causal=1,
            attention_mode="selected_only",
            selected_kv_source="main",
            do_rotary=0,
            rotary_interleaved=0,
            qk_norm_epsilon=1e-6,
        )
    )
    if gate_width:
        nodes += [
            helper.make_node("Sigmoid", ["gate"], ["gate_sig"]),
            helper.make_node("Mul", [attn, "gate_sig"], ["output"]),
        ]
        inputs.append(
            helper.make_tensor_value_info("gate", f16, ["B", "S", gate_width])
        )
    cache = ["B", KV, "cap", D]
    inputs += [
        helper.make_tensor_value_info("past_key", f16, cache),
        helper.make_tensor_value_info("past_value", f16, cache),
        helper.make_tensor_value_info("selected_indices", TP.INT32, ["R", c["K"]]),
        helper.make_tensor_value_info("selected_counts", TP.INT32, ["R"]),
        helper.make_tensor_value_info("seqlens_k", TP.INT32, ["B"]),
        helper.make_tensor_value_info("total_sequence_length", TP.INT32, []),
    ]
    outputs = [
        helper.make_tensor_value_info("output", f16, ["B", "S", N * D]),
        helper.make_tensor_value_info("present_key", f16, cache),
        helper.make_tensor_value_info("present_value", f16, cache),
    ]
    model = make_model_from_nodes(nodes, inputs, outputs, inits, **_OPSETS)
    model.ir_version = 10
    return model


_FUSED_CASES = {
    # Folded pieces, geometry (heads, KV heads, head size), batch, new and
    # cached tokens, candidate width.
    "packed": dict(packed=True, prologue=False, gate=False, B=1, S=3, P=70),
    "gate": dict(packed=False, prologue=False, gate=True, B=2, S=4, P=90),
    "broadcast_gate": dict(packed=True, prologue=False, gate="broadcast", B=1, S=3),
    "prologue": dict(packed=False, prologue=True, gate=False, B=2, S=5, P=90),
    "packed_prologue": dict(packed=True, prologue=True, gate=False, B=1, S=64, P=0),
    "all_d128_layout0": dict(
        packed=True, prologue=True, gate=True, B=1, S=7, P=60, N=4, KV=1, D=128
    ),
    "all_decode_split": dict(
        packed=True, prologue=True, gate=True, B=2, S=1, P=3000, K=2051
    ),
}
for _c in _FUSED_CASES.values():
    _c.setdefault("P", 50)
    _c.setdefault("N", 24)
    _c.setdefault("KV", 2)
    _c.setdefault("D", 256)
    _c.setdefault("K", 48)


@pytest.mark.parametrize("case", list(_FUSED_CASES), ids=list(_FUSED_CASES))
def test_fused_prologue_and_gate(model_runner, case):
    c = dict(_FUSED_CASES[case])
    N, KV, D, B, S, P, K = (c[x] for x in ("N", "KV", "D", "B", "S", "P", "K"))
    rng = np.random.default_rng(sum(map(ord, case)))
    half = 16 if D == 128 else 32
    c.update(
        scale=0.0625,
        eps=1e-6,
        rotary_dim=2 * half,
        section=[6, 5, 5] if D == 128 else [11, 11, 10],
        layout=0 if "layout0" in case else 1,
        table_rows=P + S + 16,
        q_weight=(1 + 0.1 * rng.standard_normal(D)).astype(np.float16),
        k_weight=(1 + 0.1 * rng.standard_normal(D)).astype(np.float16),
    )
    gate_width = {True: N * D, "broadcast": 1, False: 0}[c["gate"]]
    cap = P + S + 8
    q, k, v, past_k, past_v = _inputs(rng, B, S, N, KV, D, cap)
    past_k[:, :, P:] = np.nan
    past_v[:, :, P:] = np.nan
    sel, counts = _selection(rng, B, S, P, K)
    seqlens = np.full(B, P + S - 1, np.int32)
    total = np.array(P + S, np.int32)
    feeds = {}
    if c["packed"]:
        feeds["qkv"] = np.concatenate([q, k, v], axis=-1)
    else:
        feeds.update(query=q, key=k, value=v)
    if c["prologue"]:
        pos = _step_positions(B, P, S)
        feeds["position_ids"] = pos
        cos, sin = _rope_base_tables(c["table_rows"], half)
        q = _norm_rope(q, N, c["q_weight"], pos, cos, sin, c)
        k = _norm_rope(k, KV, c["k_weight"], pos, cos, sin, c)
    gate = None
    if gate_width:
        gate = rng.uniform(-4, 4, (B, S, gate_width)).astype(np.float16)
        feeds["gate"] = gate
    feeds.update(
        past_key=past_k,
        past_value=past_v,
        selected_indices=sel,
        selected_counts=counts,
        seqlens_k=seqlens,
        total_sequence_length=total,
    )
    model = _make_fused_model(c, gate_width)
    names = [i.name for i in model.graph.input]
    actual = model_runner.backend.run(
        _persist(model_runner, model), [feeds[x] for x in names]
    )
    expected, pk, pv = _dsa_reference(
        q, k, v, past_k, past_v, sel, counts, seqlens, N, KV, c["scale"]
    )
    if gate is not None:
        expected = _output_gate(expected, gate)
    compare_outputs(actual[:1], [expected], atol=5e-3, rtol=0.0)
    if c["prologue"]:
        np.testing.assert_allclose(actual[1], pk, rtol=4e-3, atol=4e-3)
    else:
        np.testing.assert_array_equal(actual[1], pk)
    np.testing.assert_array_equal(actual[2], pv)


def _persist(model_runner, model):
    sub = model_runner._next_subdir()
    path = sub / "model.onnx"
    path.write_bytes(model.SerializeToString())
    return str(path)
