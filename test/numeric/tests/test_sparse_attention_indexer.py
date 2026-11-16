#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""Tests for com.microsoft SparseAttentionIndexer (policy_mode=qsa, packed QK).

ORT CPU has no SparseAttentionIndexer kernel, so the reference is a NumPy port
of the ORT CUDA semantics. Scores are fp32 sums whose order differs between
NumPy and the GPU, so blocks whose reference scores differ by at most
_TIE_RTOL may come out in either order; that includes non-zero scores that
are equal only by fp32 coincidence. Zero scores (every head clamped) tie
exactly on both sides and rank by block index ascending, and everything else
must match exactly.
"""

import numpy as np
import pytest
from onnx import TensorProto as TP
from onnx import helper, numpy_helper

from framework.onnx_utils import make_model_from_nodes

_OPSETS = dict(opset=22, extra_opsets=[helper.make_opsetid("com.microsoft", 1)])
_D = 128
_TIE_RTOL = 1e-5


def _make_model(
    N, R, rank3, has_mask, dtype, ratio, budget, epsilon, scale, shared=False
):
    elem = TP.FLOAT16 if dtype == np.float16 else TP.FLOAT
    attrs = dict(policy_mode="qsa", compress_ratio=ratio, token_budget=budget)
    if epsilon is not None:
        attrs["epsilon"] = epsilon
    if scale is not None:
        attrs["scale"] = scale
    node_inputs = [
        "query",
        "",
        "query_norm_weight",
        "key_norm_weight",
        "cos_cache",
        "sin_cache",
        "mask" if has_mask else "",
        "past_key",
    ]
    if shared:
        node_inputs += ["", "", "", "", "past_sequence_length"]
    node = helper.make_node(
        "SparseAttentionIndexer",
        node_inputs,
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
    if shared:
        inputs.append(
            helper.make_tensor_value_info("past_sequence_length", TP.INT32, [1])
        )
    outputs = [
        helper.make_tensor_value_info(
            "selected_indices", TP.INT32, ["B", "S", budget + ratio - 1]
        ),
        helper.make_tensor_value_info(
            "present_key", elem, ["B", "P" if shared else "T", _D]
        ),
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


def _block_scores(qrot, krot, sc):
    """qrot [..., N, D] x krot [nb, D] -> [..., nb] fp32 head-summed ReLU."""
    dots = np.matmul(qrot, krot.T).astype(np.float32)
    score = np.zeros(dots.shape[:-2] + dots.shape[-1:], np.float32)
    for h in range(dots.shape[-2]):
        score = score + np.fmax(dots[..., h, :], np.float32(0.0))
    return (score * sc).astype(np.float32) + np.float32(0.0)


_REF_ROWS = 64


def _sai_reference(query, qw, kw, cos, sin, mask, past, ratio, budget, eps, scale):
    """Returns (selected_indices, present_key, row_info) where row_info(b, s)
    is (visible positions, block scores) for the tie-aware comparison."""
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

    selected = np.full((B, S, cap), -1, np.int32)
    per_batch = []
    rows_idx = np.arange(S)
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
            kpos = np.clip(groups_all[:, 0], 0, maxpos - 1)
            krot = _leading_rope(pooled, cos32[b, kpos], sin32[b, kpos])
        qp = np.clip(P + rows_idx, 0, maxpos - 1)
        qrot = _leading_rope(
            _rms_norm(q32[b], qw32, eps), cos32[b, qp][:, None], sin32[b, qp][:, None]
        )
        nvis = np.searchsorted(visible_all, P + rows_idx, side="right")
        nb = nvis // ratio
        emitted = np.minimum(budget // ratio, nb)
        for s0 in range(0, S, _REF_ROWS):
            s1 = min(S, s0 + _REF_ROWS)
            kmax = int(emitted[s0:s1].max())
            if kmax == 0:
                continue
            width = int(nb[s1 - 1])
            score = _block_scores(qrot[s0:s1], krot[:width], sc)
            score[np.arange(width)[None, :] >= nb[s0:s1, None]] = -np.inf
            order = np.argsort(-score, axis=1, kind="stable")[:, :kmax]
            picks = groups_all[order].reshape(s1 - s0, kmax * ratio)
            keep = np.arange(kmax * ratio)[None, :] < emitted[s0:s1, None] * ratio
            selected[b, s0:s1, : kmax * ratio] = np.where(keep, picks, -1)
        for t in range(ratio - 1):
            idx = nb * ratio + t
            ok = np.nonzero(idx < nvis)[0]
            selected[b, ok, emitted[ok] * ratio + t] = visible_all[idx[ok]]
        per_batch.append((visible_all, nvis, qrot, krot))

    def row_info(b, s):
        visible_all, nvis, qrot, krot = per_batch[b]
        visible = visible_all[: nvis[s]]
        return visible, _block_scores(qrot[s], krot[: len(visible) // ratio], sc)

    return selected, present, row_info


def _check_selected(actual, expected, row_info, ratio, budget):
    """Exact match, except that blocks whose reference scores differ but lie
    within _TIE_RTOL of each other may be ranked in either order."""
    B, S, _ = expected.shape
    swapped = 0
    for b in range(B):
        for s in range(S):
            if np.array_equal(actual[b, s], expected[b, s]):
                continue
            swapped += 1
            visible, score = row_info(b, s)
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
            zero = blocks[picked == 0]
            assert np.all(np.diff(zero) > 0), (b, s, "zero tie order")
            rest = np.setdiff1d(np.arange(nb), blocks)
            if rest.size:
                assert score[rest].max() <= picked.min() + tol, (b, s, "selected set")
                if zero.size:
                    assert np.all(rest[score[rest] == 0] > zero.max()), (
                        b,
                        s,
                        "zero tie set",
                    )
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
    expected, present, row_info = _sai_reference(
        query, qw, kw, cos, sin, mask, past, ratio, budget, eff_eps, eff_scale
    )
    np.testing.assert_array_equal(actual[1], present)
    assert actual[0].dtype == np.int32 and actual[0].shape == expected.shape
    if exact:
        np.testing.assert_array_equal(actual[0], expected)
        return 0
    return _check_selected(actual[0], expected, row_info, ratio, budget)


@pytest.mark.parametrize("S", [1, 7, 64, 600])
def test_prefill(model_runner, S):
    _run_case(model_runner, B=1, S=S, P=0, seed=S)


@pytest.mark.parametrize("P", [1000, 4000])
def test_decode(model_runner, P):
    _run_case(model_runner, B=1, S=1, P=P, seed=P)


def test_decode_long_context(model_runner):
    """Few rows of at least 8192 blocks select on one 1024-thread block."""
    _run_case(model_runner, B=2, S=3, P=33000, seed=12)


def test_prefill_multi_chunk(model_runner):
    """Scores past a 128 MiB budget split the query rows into chunks of
    budget / (B * ((P + S) / ratio + N * D) * 4 bytes) rows."""
    B, S, P, N, ratio = 1, 2100, 60000, 8, 4
    chunk_rows = (128 << 20) // (B * ((P + S) // ratio + N * _D) * 4)
    assert 0 < chunk_rows < S
    _run_case(model_runner, B=B, S=S, P=P, seed=10, N=N, ratio=ratio)


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


@pytest.mark.parametrize("R", [8, 48, 128])
def test_rotary_widths(model_runner, R):
    """Covers each RoPE partner path in both kernel layouts.

    - Halves 4 and 64 swap partners between lanes; half 24 goes through LDS.
    - S = 5 runs the separate pooled and score kernels; S = 1 runs the fused
      pooled-score kernel.
    """
    _run_case(model_runner, B=1, S=5, P=900, seed=30 + R, R=R)
    _run_case(model_runner, B=1, S=1, P=2100, seed=31 + R, R=R)


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


def _make_gathered_rotary_model(N, R, rows, budget, masks, M, picks):
    """Returns an indexer model that builds its MRoPE rotary tables:

    cache = Concat(w, w)
    w     = Where(m2, g2, Where(m1, g1, g0))
    g_k   = Gather(base, Gather(rotary_position_ids, picks[k]))

    rotary_position_ids is [M, B, X].
    """
    elem = TP.FLOAT16
    nodes = [
        helper.make_node(
            "Constant", [], [f"k{k}"], value=numpy_helper.from_array(np.int64(pick))
        )
        for k, pick in enumerate(picks)
    ]
    nodes += [
        helper.make_node("Gather", ["rotary_position_ids", f"k{k}"], [f"pos{k}"])
        for k in range(3)
    ]
    for kind in ("cos", "sin"):
        nodes += [
            helper.make_node("Gather", [f"{kind}_base", f"pos{k}"], [f"{kind}{k}"])
            for k in range(3)
        ]
        nodes.append(
            helper.make_node("Where", ["m1", f"{kind}1", f"{kind}0"], [f"{kind}_w1"])
        )
        nodes.append(
            helper.make_node("Where", ["m2", f"{kind}2", f"{kind}_w1"], [f"{kind}_w"])
        )
        nodes.append(
            helper.make_node(
                "Concat", [f"{kind}_w", f"{kind}_w"], [f"{kind}_cache"], axis=-1
            )
        )
    nodes.append(
        helper.make_node(
            "SparseAttentionIndexer",
            ["query", "", "query_norm_weight", "key_norm_weight"]
            + ["cos_cache", "sin_cache", "", "past_key"],
            ["selected_indices", "present_key"],
            domain="com.microsoft",
            policy_mode="qsa",
            compress_ratio=4,
            token_budget=budget,
        )
    )
    inputs = [
        helper.make_tensor_value_info("query", elem, ["B", "S", (N + 1) * _D]),
        helper.make_tensor_value_info("query_norm_weight", elem, [_D]),
        helper.make_tensor_value_info("key_norm_weight", elem, [_D]),
        helper.make_tensor_value_info("cos_base", elem, [rows, R // 2]),
        helper.make_tensor_value_info("sin_base", elem, [rows, R // 2]),
        helper.make_tensor_value_info("rotary_position_ids", TP.INT64, [M, "B", "X"]),
        helper.make_tensor_value_info("past_key", elem, ["B", "P", _D]),
    ]
    outputs = [
        helper.make_tensor_value_info(
            "selected_indices", TP.INT32, ["B", "S", budget + 3]
        ),
        helper.make_tensor_value_info("present_key", elem, ["B", "T", _D]),
    ]
    inits = [numpy_helper.from_array(m, f"m{i}") for i, m in enumerate(masks, 1)]
    model = make_model_from_nodes(nodes, inputs, outputs, inits, **_OPSETS)
    model.ir_version = 10
    return model


@pytest.mark.parametrize("M, picks", [(3, (0, 1, 2)), (6, (0, 4, 5))])
@pytest.mark.parametrize("S", [1, 64])
def test_rotary_fold_matches_tables(model_runner, S, M, picks):
    """The folded graph must match the indexer fed materialized tables bit for bit.

    - The ids cover three distinct sections, per-batch offsets, negative
      positions and a history X longer than the context.
    - M = 3 reads the packed section map; M = 6 reads the per-channel one.
    """
    B, P, N, R, rows, budget = 2, 1000, 4, 64, 2048, 2048
    X = P + S + 100
    dt = np.float16
    rng = np.random.default_rng(50 + S + M)
    freq = np.arange(R // 2)
    masks = (freq % 3 == 1, freq % 3 == 2)
    ids = rng.integers(-rows, rows, (M, B, X))
    ids[picks[0]] = np.arange(X) + rng.integers(0, 64, (B, 1))
    ang = np.arange(rows)[:, None] / (10000.0 ** (freq / (R // 2)))
    cos_base, sin_base = np.cos(ang).astype(dt), np.sin(ang).astype(dt)

    def table(base):
        g0, g1, g2 = (base[ids[pick]] for pick in picks)
        w = np.where(masks[1], g2, np.where(masks[0], g1, g0))
        return np.concatenate([w, w], axis=-1)

    cos, sin = table(cos_base), table(sin_base)
    query = rng.uniform(-1, 1, (B, S, (N + 1) * _D)).astype(dt)
    qw = rng.uniform(0.5, 1.5, _D).astype(dt)
    kw = rng.uniform(0.5, 1.5, _D).astype(dt)
    past = rng.uniform(-1, 1, (B, P, _D)).astype(dt)

    folded = model_runner.backend.run(
        _persist(
            model_runner,
            _make_gathered_rotary_model(N, R, rows, budget, masks, M, picks),
        ),
        [query, qw, kw, cos_base, sin_base, ids, past],
    )
    tabled = _make_model(N, R, True, False, dt, 4, budget, None, None)
    tabled = model_runner.backend.run(
        _persist(model_runner, tabled), [query, qw, kw, cos, sin, past]
    )
    for got, want in zip(folded, tabled):
        np.testing.assert_array_equal(got, want)

    expected, present, row_info = _sai_reference(
        query, qw, kw, cos, sin, None, past, 4, budget, 1e-6, 1 / np.sqrt(_D)
    )
    np.testing.assert_array_equal(folded[1], present)
    _check_selected(folded[0], expected, row_info, 4, budget)


@pytest.mark.parametrize("S", [1, 64])
def test_counts_fold(model_runner, S):
    """The folded counts equal the number of valid entries in each selected row.

    The history is shorter than the budget, so each row has its own count.
    """
    B, P, N, R, budget = 2, 100, 4, 64, 2048
    dt = np.float16
    rng = np.random.default_rng(70 + S)
    query = rng.uniform(-1, 1, (B, S, (N + 1) * _D)).astype(dt)
    qw = rng.uniform(0.5, 1.5, _D).astype(dt)
    kw = rng.uniform(0.5, 1.5, _D).astype(dt)
    cos, sin = _rope_tables(rng, P + S, R, dt, B)
    past = rng.uniform(-1, 1, (B, P, _D)).astype(dt)

    model = _make_model(N, R, True, False, dt, 4, budget, None, None)
    constants = {"zero": np.int32(0), "one": np.int32(1), "axes": np.array([-1])}
    model.graph.node.extend(
        [
            helper.make_node(
                "Constant", [], [name], value=numpy_helper.from_array(value)
            )
            for name, value in constants.items()
        ]
        + [
            helper.make_node("GreaterOrEqual", ["selected_indices", "zero"], ["ge"]),
            helper.make_node("Where", ["ge", "one", "zero"], ["valid"]),
            helper.make_node(
                "ReduceSum", ["valid", "axes"], ["selected_counts"], keepdims=0
            ),
        ]
    )
    model.graph.output.append(
        helper.make_tensor_value_info("selected_counts", TP.INT32, ["B", "S"])
    )
    sel, present, counts = model_runner.backend.run(
        _persist(model_runner, model), [query, qw, kw, cos, sin, past]
    )

    expected, want_present, row_info = _sai_reference(
        query, qw, kw, cos, sin, None, past, 4, budget, 1e-6, 1 / np.sqrt(_D)
    )
    np.testing.assert_array_equal(present, want_present)
    _check_selected(sel, expected, row_info, 4, budget)
    np.testing.assert_array_equal(counts, (expected >= 0).sum(-1, dtype=np.int32))


@pytest.mark.parametrize("bind_shared", [True, False])
@pytest.mark.parametrize("has_mask", [False, True])
def test_past_sequence_length(model_runner, bind_shared, has_mask):
    """past_key / present_key are [B, C, D] caches holding past_sequence_length
    valid rows. With bind_shared both are bound to one device buffer and the
    rows past the valid ones must stay untouched."""
    import onnxruntime as ort

    backend = model_runner.backend
    if not hasattr(backend, "create_session"):
        pytest.skip("backend does not expose an ORT session for IO binding")
    B, N, R, C, L = 2, 4, 64, 640, 300
    dt = np.float16
    rng = np.random.default_rng(40 + 2 * bind_shared + has_mask)
    model = _make_model(N, R, True, has_mask, dt, 4, 64, 1e-6, None, shared=True)
    sess = backend.create_session(_persist(model_runner, model))
    qw = rng.uniform(0.5, 1.5, _D).astype(dt)
    kw = rng.uniform(0.5, 1.5, _D).astype(dt)
    cos, sin = _rope_tables(rng, C, R, dt, B)
    cache = np.full((B, C, _D), np.nan, dt)
    cache[:, :L] = rng.uniform(-1, 1, (B, L, _D)).astype(dt)
    cache_dev = ort.OrtValue.ortvalue_from_numpy(
        cache, device_type="gpu", vendor_id=0x1002
    )

    for S in [1, 5, 1]:
        query = rng.uniform(-1, 1, (B, S, (N + 1) * _D)).astype(dt)
        mask = None
        if has_mask:
            mask = np.ones((B, L + S), np.int64)
            mask[1, :23] = 0
        io = sess.io_binding()
        names = ["query", "query_norm_weight", "key_norm_weight", "cos_cache"]
        names += ["sin_cache", "past_sequence_length"]
        feeds = [query, qw, kw, cos, sin, np.array([L], np.int32)]
        if has_mask:
            names.append("mask")
            feeds.append(mask)
        for name, arr in zip(names, feeds):
            io.bind_cpu_input(name, arr)
        io.bind_output("selected_indices")
        if bind_shared:
            io.bind_ortvalue_input("past_key", cache_dev)
            io.bind_ortvalue_output("present_key", cache_dev)
        else:
            io.bind_cpu_input("past_key", cache)
            io.bind_output("present_key")
        sess.run_with_iobinding(io)
        outputs = io.copy_outputs_to_cpu()

        expected, present, row_info = _sai_reference(
            query, qw, kw, cos, sin, mask, cache[:, :L], 4, 64, 1e-6, 1 / np.sqrt(_D)
        )
        _check_selected(outputs[0], expected, row_info, 4, 64)
        cache[:, : L + S] = present
        actual = cache_dev.numpy() if bind_shared else outputs[1]
        np.testing.assert_array_equal(actual[:, : L + S], present)
        if bind_shared:
            np.testing.assert_array_equal(actual, cache)
        L += S


def _persist(model_runner, model):
    sub = model_runner._next_subdir()
    path = sub / "model.onnx"
    path.write_bytes(model.SerializeToString())
    return str(path)
