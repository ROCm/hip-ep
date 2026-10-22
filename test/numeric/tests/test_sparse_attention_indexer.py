#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""Tests for com.microsoft SparseAttentionIndexer (policy_mode=qsa, packed QK,
no past_sequence_length).

ORT CPU has no SparseAttentionIndexer kernel, so the reference is a NumPy port
of the ORT CUDA semantics. Scores are fp32 sums whose order differs between
NumPy and the GPU, so blocks whose reference scores differ by at most
_TIE_RTOL may come out in either order. Exactly equal scores rank by block
index ascending, and everything else must match exactly.
"""

import numpy as np
import pytest
from onnx import TensorProto as TP
from onnx import helper

from framework.onnx_utils import make_model_from_nodes

_OPSETS = dict(opset=22, extra_opsets=[helper.make_opsetid("com.microsoft", 1)])
_D = 128
_TIE_RTOL = 1e-5


def _make_model(N, R, rank3, has_mask, dtype, ratio, budget, epsilon, scale):
    elem = TP.FLOAT16 if dtype == np.float16 else TP.FLOAT
    attrs = dict(policy_mode="qsa", compress_ratio=ratio, token_budget=budget)
    if epsilon is not None:
        attrs["epsilon"] = epsilon
    if scale is not None:
        attrs["scale"] = scale
    node = helper.make_node(
        "SparseAttentionIndexer",
        [
            "query",
            "",
            "query_norm_weight",
            "key_norm_weight",
            "cos_cache",
            "sin_cache",
            "mask" if has_mask else "",
            "past_key",
        ],
        ["selected_indices", "present_key"],
        domain="com.microsoft",
        **attrs,
    )
    cache = ["B", "X", R] if rank3 else ["X", R]
    inputs = [
        helper.make_tensor_value_info("query", elem, ["B", "S", (N + 1) * _D]),
        helper.make_tensor_value_info("query_norm_weight", elem, [_D]),
        helper.make_tensor_value_info("key_norm_weight", elem, [_D]),
        helper.make_tensor_value_info("cos_cache", elem, cache),
        helper.make_tensor_value_info("sin_cache", elem, cache),
    ]
    if has_mask:
        inputs.append(helper.make_tensor_value_info("mask", TP.INT64, ["B", "T"]))
    inputs.append(helper.make_tensor_value_info("past_key", elem, ["B", "P", _D]))
    outputs = [
        helper.make_tensor_value_info(
            "selected_indices", TP.INT32, ["B", "S", budget + ratio - 1]
        ),
        helper.make_tensor_value_info("present_key", elem, ["B", "T", _D]),
    ]
    model = make_model_from_nodes([node], inputs, outputs, **_OPSETS)
    model.ir_version = 10
    return model


def _rms_norm(x, w, eps):
    ss = np.sum(x * x, axis=-1, keepdims=True, dtype=np.float32)
    inv = np.float32(1.0) / np.sqrt(
        ss / np.float32(x.shape[-1]) + np.float32(eps), dtype=np.float32
    )
    return (x * inv * w).astype(np.float32)


def _leading_rope(x, cos_row, sin_row):
    r = cos_row.shape[-1]
    h = r // 2
    out = x.copy()
    xr = x[..., :r]
    paired = np.concatenate([-xr[..., h:], xr[..., :h]], axis=-1)
    out[..., :r] = xr * cos_row + paired * sin_row
    return out.astype(np.float32)


def _sai_reference(query, qw, kw, cos, sin, mask, past, ratio, budget, eps, scale):
    """Returns (selected_indices, present_key, rows) where rows[b][s] is
    (visible positions, block scores) for the tie-aware comparison."""
    B, S, W = query.shape
    N = W // _D - 1
    P = past.shape[1]
    total = P + S
    cap = budget + ratio - 1
    if cos.ndim == 2:
        cos = np.broadcast_to(cos, (B,) + cos.shape)
        sin = np.broadcast_to(sin, (B,) + sin.shape)
    maxpos = cos.shape[1]
    cos32 = cos.astype(np.float32)
    sin32 = sin.astype(np.float32)
    if mask is None:
        mask = np.ones((B, total), np.int64)

    present = np.concatenate([past, query[:, :, N * _D :]], axis=1)
    pk32 = present.astype(np.float32)
    q32 = query[:, :, : N * _D].astype(np.float32).reshape(B, S, N, _D)
    qw32 = qw.astype(np.float32)
    kw32 = kw.astype(np.float32)
    sc = np.float32(scale)

    def clamp(p):
        return min(max(int(p), 0), maxpos - 1)

    selected = np.full((B, S, cap), -1, np.int32)
    rows = []
    for b in range(B):
        visible_all = np.nonzero(mask[b] != 0)[0]
        nb_all = len(visible_all) // ratio
        groups_all = visible_all[: nb_all * ratio].reshape(nb_all, ratio)
        krot = np.zeros((nb_all, _D), np.float32)
        if nb_all:
            pooled = pk32[b, groups_all[:, 0]].copy()
            for t in range(1, ratio):
                pooled = pooled + pk32[b, groups_all[:, t]]
            pooled = (pooled / np.float32(ratio)).astype(np.float32)
            pooled = _rms_norm(pooled, kw32, eps)
            kpos = [clamp(p) for p in groups_all[:, 0]]
            krot = _leading_rope(pooled, cos32[b, kpos], sin32[b, kpos])
        row_info = []
        for s in range(S):
            qp = clamp(P + s)
            qrot = _leading_rope(
                _rms_norm(q32[b, s], qw32, eps), cos32[b, qp], sin32[b, qp]
            )
            visible = visible_all[visible_all <= P + s]
            nb = len(visible) // ratio
            emitted = min(budget // ratio, nb)
            score = np.zeros(nb, np.float32)
            if nb:
                dots = (qrot @ krot[:nb].T).astype(np.float32)
                for h in range(N):
                    score = score + np.fmax(dots[h], np.float32(0.0))
                score = (score * sc).astype(np.float32) + np.float32(0.0)
                order = np.lexsort((np.arange(nb), -score))
                for rank in range(emitted):
                    selected[b, s, rank * ratio : (rank + 1) * ratio] = groups_all[
                        order[rank]
                    ]
            tail = visible[nb * ratio :]
            selected[b, s, emitted * ratio : emitted * ratio + len(tail)] = tail
            row_info.append((visible, score))
        rows.append(row_info)
    return selected, present, rows


def _check_selected(actual, expected, rows, ratio, budget):
    """Exact match, except that blocks whose reference scores differ but lie
    within _TIE_RTOL of each other may be ranked in either order."""
    B, S, _ = expected.shape
    swapped = 0
    for b in range(B):
        for s in range(S):
            if np.array_equal(actual[b, s], expected[b, s]):
                continue
            swapped += 1
            visible, score = rows[b][s]
            nb = len(score)
            k = min(budget // ratio, nb)
            np.testing.assert_array_equal(
                actual[b, s, k * ratio :], expected[b, s, k * ratio :]
            )
            groups = actual[b, s, : k * ratio].reshape(k, ratio)
            pos = np.searchsorted(visible, groups[:, 0])
            assert np.all(pos % ratio == 0) and np.all(pos // ratio < nb), (b, s)
            blocks = pos // ratio
            np.testing.assert_array_equal(
                groups, visible[: nb * ratio].reshape(nb, ratio)[blocks]
            )
            assert len(set(blocks.tolist())) == k, (b, s)
            tol = _TIE_RTOL * max(1.0, float(np.abs(score).max()))
            picked = score[blocks]
            assert np.all(np.diff(picked) <= tol), (b, s, "rank order")
            for i in range(k - 1):
                later = blocks[i + 1 :][picked[i + 1 :] == picked[i]]
                assert np.all(later > blocks[i]), (b, s, "exact tie order")
            rest = np.setdiff1d(np.arange(nb), blocks)
            if rest.size:
                assert score[rest].max() <= picked.min() + tol, (b, s, "selected set")
                for r in rest:
                    tied = blocks[picked == score[r]]
                    assert np.all(tied < r), (b, s, "exact tie set")
    return swapped


def _rope_tables(rng, rows, R, dtype, batch=None):
    inv_freq = 1.0 / (10000.0 ** (np.arange(R // 2) / (R // 2)))
    shape = (rows,) if batch is None else (batch, rows)
    pos = np.arange(rows, dtype=np.float64)
    if batch is not None:
        pos = pos[None, :] + rng.integers(0, 64, size=(batch, 1))
    ang = pos[..., None] * inv_freq
    ang = np.concatenate([ang, ang], axis=-1).reshape(shape + (R,))
    return np.cos(ang).astype(dtype), np.sin(ang).astype(dtype)


def _run_case(
    model_runner,
    *,
    B,
    S,
    P,
    seed,
    mask=None,
    N=4,
    R=64,
    X=None,
    rank3=True,
    dtype=np.float16,
    ratio=4,
    budget=2048,
    epsilon=1e-6,
    scale=0.0883883461,
):
    rng = np.random.default_rng(seed)
    total = P + S
    X = total if X is None else X
    query = rng.uniform(-1, 1, (B, S, (N + 1) * _D)).astype(dtype)
    qw = rng.uniform(0.5, 1.5, _D).astype(dtype)
    kw = rng.uniform(0.5, 1.5, _D).astype(dtype)
    cos, sin = _rope_tables(rng, X, R, dtype, B if rank3 else None)
    past = rng.uniform(-1, 1, (B, P, _D)).astype(dtype)
    return _run_feeds(
        model_runner,
        query,
        qw,
        kw,
        cos,
        sin,
        mask,
        past,
        ratio=ratio,
        budget=budget,
        epsilon=epsilon,
        scale=scale,
    )


def _run_feeds(
    model_runner,
    query,
    qw,
    kw,
    cos,
    sin,
    mask,
    past,
    *,
    ratio=4,
    budget=2048,
    epsilon=1e-6,
    scale=0.0883883461,
    exact=False,
):
    N = query.shape[2] // _D - 1
    model = _make_model(
        N,
        cos.shape[-1],
        cos.ndim == 3,
        mask is not None,
        query.dtype.type,
        ratio,
        budget,
        epsilon,
        scale,
    )
    feeds = [query, qw, kw, cos, sin] + ([mask] if mask is not None else []) + [past]
    actual = model_runner.backend.run(_persist(model_runner, model), feeds)
    eff_eps = 1e-6 if epsilon is None else epsilon
    eff_scale = 1.0 / np.sqrt(_D) if scale is None else scale
    expected, present, rows = _sai_reference(
        query, qw, kw, cos, sin, mask, past, ratio, budget, eff_eps, eff_scale
    )
    np.testing.assert_array_equal(actual[1], present)
    assert actual[0].dtype == np.int32 and actual[0].shape == expected.shape
    if exact:
        np.testing.assert_array_equal(actual[0], expected)
        return 0
    return _check_selected(actual[0], expected, rows, ratio, budget)


@pytest.mark.parametrize("S", [1, 7, 64, 600])
def test_prefill(model_runner, S):
    _run_case(model_runner, B=1, S=S, P=0, seed=S)


@pytest.mark.parametrize("P", [1000, 4000])
def test_decode(model_runner, P):
    _run_case(model_runner, B=1, S=1, P=P, seed=P)


def test_decode_past_crossing_topk(model_runner):
    """S=2 rows on either side of 512 blocks, and 64 prefill rows on top of a
    2000-token cache."""
    _run_case(model_runner, B=1, S=2, P=2046, seed=1)
    _run_case(model_runner, B=1, S=64, P=2000, seed=2)


def test_masked_batch(model_runner):
    """Batch 0 is left-padded, batch 1 has holes in the middle."""
    B, S, P = 2, 24, 300
    total = P + S
    rng = np.random.default_rng(7)
    mask = np.ones((B, total), np.int64)
    mask[0, :37] = 0
    holes = rng.random(total) < 0.3
    holes[-S:] = False
    mask[1, holes] = 0
    mask[1, 50] = 1 << 40
    _run_case(model_runner, B=B, S=S, P=P, seed=3, mask=mask)


def test_fewer_visible_than_ratio(model_runner):
    mask = np.array([[0, 0, 1, 0, 0, 1, 1]], np.int64)
    _run_case(model_runner, B=1, S=3, P=4, seed=4, mask=mask)
    _run_case(model_runner, B=1, S=2, P=1, seed=5)


def test_rank2_cache_clamped_rows(model_runner):
    """Rotary rows X < P + S: positions past the table use its last row."""
    _run_case(model_runner, B=2, S=5, P=700, seed=6, rank3=False, X=300)


def test_rank3_cache_clamped_rows(model_runner):
    _run_case(model_runner, B=1, S=1, P=3000, seed=8, X=1024)


def test_f32_small_config(model_runner):
    _run_case(
        model_runner,
        B=2,
        S=9,
        P=150,
        seed=9,
        N=2,
        R=32,
        rank3=False,
        dtype=np.float32,
        budget=64,
        epsilon=None,
        scale=None,
    )


@pytest.mark.parametrize("S", [1, 8])
def test_exact_ties_keep_block_order(model_runner, S):
    """With an identity rotary, identical keys give every block the same
    positive score and keys opposite to the queries give every block 0."""
    B, P, N, R = 2, 200, 4, 64
    dt = np.float16
    rng = np.random.default_rng(20 + S)
    cos = np.ones((B, P + S, R), dt)
    sin = np.zeros((B, P + S, R), dt)
    qw = rng.uniform(0.5, 1.5, _D).astype(dt)
    kw = rng.uniform(0.5, 1.5, _D).astype(dt)

    key_row = rng.uniform(-1, 1, _D).astype(dt)
    query = rng.uniform(-1, 1, (B, S, (N + 1) * _D)).astype(dt)
    query[:, :, N * _D :] = key_row
    past = np.broadcast_to(key_row, (B, P, _D)).copy()
    _run_feeds(model_runner, query, qw, kw, cos, sin, None, past, budget=64, exact=True)

    query = rng.uniform(0.1, 1, (B, S, (N + 1) * _D)).astype(dt)
    query[:, :, N * _D :] = -rng.uniform(0.1, 1, (B, S, _D)).astype(dt)
    past = -rng.uniform(0.1, 1, (B, P, _D)).astype(dt)
    _run_feeds(model_runner, query, qw, kw, cos, sin, None, past, budget=64, exact=True)


def _persist(model_runner, model):
    sub = model_runner._next_subdir()
    path = sub / "model.onnx"
    path.write_bytes(model.SerializeToString())
    return str(path)
