#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""Numeric tests for a windowed ``onnx.Attention`` layer on the decomposed path.

These cover the geometry that had no numeric coverage at all: a sliding-window
attention layer running through ``gqa_forward_hipblaslt``. The two windowed
cases in ``test_gqa.py`` both use ``head_dim == 64``, which
``flash_decode_geometry_ok`` accepts, so they exercise the fused kernels and
never reach the decomposed pipeline.

The model here is Gemma-4 26B-A4B's local attention layer as exported:

  ``onnx.Attention`` (standard domain, opset 24), 3D Q/K/V + additive mask +
  past KV, ``q_num_heads=16``, ``kv_num_heads=8``, ``head_dim=256``,
  ``is_causal=0``.

Three properties of that export matter and are reproduced exactly:

  * An additive mask operand is present, which makes ``fused_supported`` false
    (the lean fused path would silently drop the bias), so every one of these
    runs is on the decomposed path by construction.
  * ``is_causal=0`` lowers to ``no_causal=true``, so the built-in causal/window
    mask kernel does not run and the mask operand carries all masking.
  * The window is not an attribute anywhere. It is baked into the mask, and the
    mask is BUILT IN THE GRAPH from the same node shape the real export uses --
    ``Where(And(qpos >= kpos, qpos - kpos < W), 0, -65504)``. That is what makes
    these tests exercise the shipping mechanism: the converter's
    AttentionWindowFold has to recognize that subgraph and recover ``W``, or the
    runtime never narrows and the narrowed path goes untested. Handing the mask
    in as a graph input would test the arithmetic while skipping the part that
    decides whether it runs at all.

Reference is fp64 numpy rather than ORT CPU on the same model: the comparison
needs to be against the mask semantics we intend, and fp16 CPU attention
accumulates enough softmax error on a 2K key range to make a mismatch
ambiguous.
"""

from __future__ import annotations

import numpy as np
import pytest
from onnx import TensorProto, helper, numpy_helper

from framework.comparator import compare_outputs

# Gemma-4 26B-A4B local-layer geometry.
NUM_HEADS = 16
KV_NUM_HEADS = 8
HEAD_DIM = 256
Q_HIDDEN = NUM_HEADS * HEAD_DIM  # 4096
KV_HIDDEN = KV_NUM_HEADS * HEAD_DIM  # 2048

# The value the real export's mask Where selects for a disallowed position.
MASK_FILL = -65504.0


def _mask_subgraph(prefix, qidx, kidx, window, out):
    """Nodes + initializers building an additive mask the converter can read.

    Reproduces the real export's node shape, which is what
    AttentionWindowFold matches on:

        qpos   = Unsqueeze(qidx, [2])          # varies along the query axis
        kpos   = Unsqueeze(kidx, [1])          # varies along the key axis
        keep   = And(qpos >= kpos, qpos - kpos < W)
        bias   = Where(keep, 0.0, -65504.0)
        <out>  = Unsqueeze(Cast(bias, fp16), [1])

    The compare and the subtraction share the same two values in the same
    order; that identity is what lets the converter conclude the bound is on
    the query-key distance. ``window <= 0`` emits the bare causal compare
    instead, which is the global-layer shape and must NOT yield a window.
    """
    p = f"{prefix}_"
    inits = [
        numpy_helper.from_array(np.array([1], dtype=np.int64), p + "ax1"),
        numpy_helper.from_array(np.array([2], dtype=np.int64), p + "ax2"),
        numpy_helper.from_array(np.array(0.0, dtype=np.float32), p + "keepv"),
        numpy_helper.from_array(np.array(MASK_FILL, dtype=np.float32), p + "dropv"),
    ]
    nodes = [
        helper.make_node("Unsqueeze", [qidx, p + "ax2"], [p + "qpos"]),
        helper.make_node("Unsqueeze", [kidx, p + "ax1"], [p + "kpos"]),
        helper.make_node("GreaterOrEqual", [p + "qpos", p + "kpos"], [p + "causal"]),
    ]
    if window > 0:
        inits.append(numpy_helper.from_array(np.array(window, dtype=np.int64), p + "w"))
        nodes += [
            helper.make_node("Sub", [p + "qpos", p + "kpos"], [p + "dist"]),
            helper.make_node("Less", [p + "dist", p + "w"], [p + "inwin"]),
            helper.make_node("And", [p + "causal", p + "inwin"], [p + "keep"]),
        ]
        keep = p + "keep"
    else:
        keep = p + "causal"
    nodes += [
        helper.make_node("Where", [keep, p + "keepv", p + "dropv"], [p + "bias32"]),
        helper.make_node(
            "Cast", [p + "bias32"], [p + "bias16"], to=TensorProto.FLOAT16
        ),
        helper.make_node("Unsqueeze", [p + "bias16", p + "ax1"], [out]),
    ]
    return nodes, inits


def _attention_attrs():
    return dict(
        q_num_heads=NUM_HEADS,
        kv_num_heads=KV_NUM_HEADS,
        scale=float(1.0 / np.sqrt(HEAD_DIM)),
        softcap=0.0,
        is_causal=0,
    )


def _make_attention_model(batch, seq_len, window):
    """Build a single-op ``onnx.Attention`` model in the Gemma-4 export shape.

    Q/K/V are rank-3 ``[B, S, heads*head_dim]``, past/present KV are BNSH, and
    the additive mask is computed in-graph from the absolute query/key position
    vectors so the window is recoverable. ``is_causal=0`` so the mask is the
    only source of masking, matching the export.

    The KV sequence dims are symbolic, matching the export's
    ``past_sequence_len`` / ``past_sequence_len + sequence_len``. This is not
    cosmetic: with every dim static, the converter's synthesised ``seqlens_k``
    folds to a compile-time constant that lands in the host constants blob, and
    the decomposed path's D2H readback of it fails with "invalid argument"
    (rc=-1, zero-filled output). Symbolic KV dims keep ``seqlens_k`` a runtime
    device value, which is what the real model produces.
    """
    past_dim = "past_seq"
    total_dim = "past_seq + seq"

    def vi(name, dtype, shape):
        return helper.make_tensor_value_info(name, dtype, shape)

    inputs = [
        vi("query", TensorProto.FLOAT16, [batch, seq_len, Q_HIDDEN]),
        vi("key", TensorProto.FLOAT16, [batch, seq_len, KV_HIDDEN]),
        vi("value", TensorProto.FLOAT16, [batch, seq_len, KV_HIDDEN]),
        vi("qidx", TensorProto.INT64, [batch, seq_len]),
        vi("kidx", TensorProto.INT64, [batch, total_dim]),
        vi(
            "past_key",
            TensorProto.FLOAT16,
            [batch, KV_NUM_HEADS, past_dim, HEAD_DIM],
        ),
        vi(
            "past_value",
            TensorProto.FLOAT16,
            [batch, KV_NUM_HEADS, past_dim, HEAD_DIM],
        ),
    ]
    outputs = [
        vi("output", TensorProto.FLOAT16, [batch, seq_len, Q_HIDDEN]),
        vi(
            "present_key",
            TensorProto.FLOAT16,
            [batch, KV_NUM_HEADS, total_dim, HEAD_DIM],
        ),
        vi(
            "present_value",
            TensorProto.FLOAT16,
            [batch, KV_NUM_HEADS, total_dim, HEAD_DIM],
        ),
    ]

    mask_nodes, mask_inits = _mask_subgraph("m", "qidx", "kidx", window, "attn_mask")
    node = helper.make_node(
        "Attention",
        ["query", "key", "value", "attn_mask", "past_key", "past_value"],
        ["output", "present_key", "present_value"],
        **_attention_attrs(),
    )

    graph = helper.make_graph(
        mask_nodes + [node],
        "attention_sliding_window",
        inputs,
        outputs,
        initializer=mask_inits,
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 24)])


def _make_chained_model(batch, second_seq, window):
    """Two ``Attention`` nodes where the first's ``present`` is the second's past.

    This is the shape of the one hazard the decode copy narrowing introduces. A
    narrowed decode leaves ``present`` valid only from ``kv_lo`` up, on the
    promise that whoever reads it next is another narrowed decode. A call with
    ``sq > 1`` and ``past_len > 0`` breaks that promise: prefill is deliberately
    not narrowed, so it reads the whole range, including the part the decode
    never wrote.

    Chaining the two nodes in one graph makes ORT hand node A's ``present``
    buffer to node B as its ``past``, which is the same aliasing a multi-turn
    generate loop produces across session runs -- and unlike that loop, it is
    reproducible in one call.
    """
    dim_a_past = "past_seq"
    dim_a_total = "past_seq_plus_1"
    dim_b_total = "past_seq_plus_1_plus_s2"

    def vi(name, shape, dtype=TensorProto.FLOAT16):
        return helper.make_tensor_value_info(name, dtype, shape)

    i64 = TensorProto.INT64
    inputs = [
        vi("query_a", [batch, 1, Q_HIDDEN]),
        vi("key_a", [batch, 1, KV_HIDDEN]),
        vi("value_a", [batch, 1, KV_HIDDEN]),
        vi("qidx_a", [batch, 1], i64),
        vi("kidx_a", [batch, dim_a_total], i64),
        vi("past_key", [batch, KV_NUM_HEADS, dim_a_past, HEAD_DIM]),
        vi("past_value", [batch, KV_NUM_HEADS, dim_a_past, HEAD_DIM]),
        vi("query_b", [batch, second_seq, Q_HIDDEN]),
        vi("key_b", [batch, second_seq, KV_HIDDEN]),
        vi("value_b", [batch, second_seq, KV_HIDDEN]),
        vi("qidx_b", [batch, second_seq], i64),
        vi("kidx_b", [batch, dim_b_total], i64),
    ]
    # Every ``present`` is a graph output, node A's included even though node B
    # also consumes it. The EP's meta-def requires a declared value info for
    # each subgraph output (morphizen-ep.cpp asserts the counts match), and
    # exposing node A's present is useful in its own right: it is the tensor
    # whose lower region the narrowing deliberately leaves unwritten.
    outputs = [
        vi("output_b", [batch, second_seq, Q_HIDDEN]),
        vi("output_a", [batch, 1, Q_HIDDEN]),
        vi("present_a_key", [batch, KV_NUM_HEADS, dim_a_total, HEAD_DIM]),
        vi("present_a_value", [batch, KV_NUM_HEADS, dim_a_total, HEAD_DIM]),
        vi("present_b_key", [batch, KV_NUM_HEADS, dim_b_total, HEAD_DIM]),
        vi("present_b_value", [batch, KV_NUM_HEADS, dim_b_total, HEAD_DIM]),
    ]

    attrs = _attention_attrs()
    nodes_a, inits_a = _mask_subgraph("ma", "qidx_a", "kidx_a", window, "mask_a")
    nodes_b, inits_b = _mask_subgraph("mb", "qidx_b", "kidx_b", window, "mask_b")
    node_a = helper.make_node(
        "Attention",
        ["query_a", "key_a", "value_a", "mask_a", "past_key", "past_value"],
        ["output_a", "present_a_key", "present_a_value"],
        name="decode",
        **attrs,
    )
    node_b = helper.make_node(
        "Attention",
        ["query_b", "key_b", "value_b", "mask_b", "present_a_key", "present_a_value"],
        ["output_b", "present_b_key", "present_b_value"],
        name="prefill_after_decode",
        **attrs,
    )

    graph = helper.make_graph(
        nodes_a + nodes_b + [node_a, node_b],
        "decode_then_prefill",
        inputs,
        outputs,
        initializer=inits_a + inits_b,
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 24)])


def _positions(batch, seq_len, past_seq):
    """Absolute query and key positions, the mask subgraph's two inputs."""
    total_kv = past_seq + seq_len
    qidx = np.broadcast_to(
        past_seq + np.arange(seq_len, dtype=np.int64), (batch, seq_len)
    ).copy()
    kidx = np.broadcast_to(
        np.arange(total_kv, dtype=np.int64), (batch, total_kv)
    ).copy()
    return qidx, kidx


def _make_window_mask(batch, seq_len, past_seq, window):
    """The reference for what the in-graph mask subgraph computes.

    Query row ``i`` has absolute position ``past_seq + i`` and may attend to
    key ``j`` when ``j <= abs_q`` (causal) and ``j > abs_q - window`` (window,
    i.e. ``abs_q - j < window``). ``window <= 0`` means causal only.
    """
    total_kv = past_seq + seq_len
    abs_q = (past_seq + np.arange(seq_len))[:, None]
    keys = np.arange(total_kv)[None, :]
    allowed = keys <= abs_q
    if window > 0:
        allowed &= abs_q - keys < window
    mask = np.where(allowed, 0.0, MASK_FILL).astype(np.float32)
    return np.broadcast_to(mask, (batch, 1, seq_len, total_kv)).copy()


def _gqa_reference(q, k, v, past_k, past_v, mask):
    """fp64 numpy GQA reference: (output, present_key, present_value).

    Q/K/V arrive rank-3 ``[B, S, heads*head_dim]``; past/present are BNSH. KV
    heads are repeated to the query head count, the additive mask is applied to
    the scaled scores, and softmax runs over the whole key range -- so a key
    whose mask entry is MASK_FILL contributes ~0 without being dropped, which
    is exactly what the narrowed path has to reproduce.
    """
    batch, seq_len, _ = q.shape

    present_k = np.concatenate(
        [past_k.astype(np.float64), _to_bnsh(k, batch, seq_len)], axis=2
    )
    present_v = np.concatenate(
        [past_v.astype(np.float64), _to_bnsh(v, batch, seq_len)], axis=2
    )
    return _attend(q, present_k, present_v, mask), present_k, present_v


def _attend(q, present_k, present_v, mask):
    """fp64 attention of rank-3 ``q`` over BNSH ``present_k``/``present_v``."""
    batch, seq_len, _ = q.shape
    hpg = NUM_HEADS // KV_NUM_HEADS

    qh = q.astype(np.float64).reshape(batch, seq_len, NUM_HEADS, HEAD_DIM)
    qh = qh.transpose(0, 2, 1, 3)  # [B, H, S, d]
    kh = np.repeat(present_k, hpg, axis=1)  # [B, H, T, d]
    vh = np.repeat(present_v, hpg, axis=1)

    scores = (qh @ kh.transpose(0, 1, 3, 2)) * (1.0 / np.sqrt(HEAD_DIM))
    scores = scores + mask.astype(np.float64)  # broadcasts over the head axis
    scores = scores - scores.max(axis=-1, keepdims=True)
    probs = np.exp(scores)
    probs /= probs.sum(axis=-1, keepdims=True)
    out = probs @ vh
    return out.transpose(0, 2, 1, 3).reshape(batch, seq_len, Q_HIDDEN)


def _to_bnsh(t, batch, seq_len):
    t = t.astype(np.float64).reshape(batch, seq_len, KV_NUM_HEADS, HEAD_DIM)
    return t.transpose(0, 2, 1, 3)


def _assert_live_present(got, ref, total_kv, window):
    """Assert ``present`` matches the reference over the range decode will read.

    The lower bound is the same ``kv_lo`` the EP narrows to, so this checks the
    exact region the copy is still responsible for. Below it the copy is allowed
    to write nothing at all, so that region is not compared -- and it must not
    be, since asserting anything about it (even that it is zero) would be
    asserting on memory nobody promised anything about.

    Bit-exactness is the right bar here rather than a tolerance: the concat is a
    copy, so any difference at all in the live region means the wrong bytes
    moved.
    """
    kv_lo = max(0, total_kv - window) if window > 0 else 0
    got = np.asarray(got)[:, :, kv_lo:, :].astype(np.float32)
    ref = np.asarray(ref)[:, :, kv_lo:, :].astype(np.float32)
    assert got.shape == ref.shape, f"{got.shape} vs {ref.shape}"
    mismatched = int(np.count_nonzero(got != ref))
    assert mismatched == 0, (
        f"present[{kv_lo}:] differs from the reference in {mismatched} of "
        f"{got.size} elements: the narrowed copy did not write everything the "
        f"narrowed read consumes"
    )


def _persist(model_runner, model):
    """Write *model* to a fresh per-test work subdir and return its path."""
    sub = model_runner._next_subdir()
    path = sub / "model.onnx"
    path.write_bytes(model.SerializeToString())
    return str(path)


def _run_case(model_runner, seq_len, past_seq, window, seed):
    batch = 1
    model = _make_attention_model(batch, seq_len, window)

    rng = np.random.default_rng(seed)

    def rand(shape, spread=0.5):
        return (rng.standard_normal(shape) * spread).astype(np.float16)

    q = rand([batch, seq_len, Q_HIDDEN])
    k = rand([batch, seq_len, KV_HIDDEN])
    v = rand([batch, seq_len, KV_HIDDEN])
    past_k = rand([batch, KV_NUM_HEADS, past_seq, HEAD_DIM])
    past_v = rand([batch, KV_NUM_HEADS, past_seq, HEAD_DIM])
    qidx, kidx = _positions(batch, seq_len, past_seq)
    mask = _make_window_mask(batch, seq_len, past_seq, window)

    actual = model_runner.backend.run(
        _persist(model_runner, model),
        [q, k, v, qidx, kidx, past_k, past_v],
    )
    expected = _gqa_reference(q, k, v, past_k, past_v, mask)
    compare_outputs(
        actual[:1], list(expected[:1]), atol=2e-2, rtol=2e-2, cos_threshold=0.999
    )

    # present KV is checked over the range this call's read consumes, and only
    # that range: the copy is narrowed to the same bound as the read, so
    # everything below it is legitimately unwritten. The bound is the case's
    # own window, because that is now what the EP narrows by -- it recovered it
    # from this model's mask subgraph. A window at or above the context does not
    # narrow, and neither does a prefill.
    total_kv = past_seq + seq_len
    narrowed = seq_len == 1 and 0 < window < total_kv
    _assert_live_present(actual[1], expected[1], total_kv, window if narrowed else 0)
    _assert_live_present(actual[2], expected[2], total_kv, window if narrowed else 0)


class TestAttentionSlidingWindow:
    """Windowed ``onnx.Attention`` through ``gqa_forward_hipblaslt``."""

    @pytest.mark.parametrize(
        "past_seq,window",
        [
            # Window saturated well inside the context: the case the narrowing
            # exists for, and the one where a wrong kv_lo shows up as a large
            # error rather than a rounding difference.
            (2048, 1024),
            # Window one key short of the context: kv_lo == 1, which catches an
            # off-by-one that a power-of-two offset would hide.
            (1024, 1024),
            # Window wider than the context: narrowing must not engage, and the
            # result must match the same reference.
            (512, 4096),
        ],
    )
    def test_decode_windowed(self, model_runner, past_seq, window):
        """Single-token decode (sq == 1) with the window carried by the mask."""
        _run_case(model_runner, seq_len=1, past_seq=past_seq, window=window, seed=7)

    def test_decode_full_attention(self, model_runner):
        """Decode with a causal-only mask: the no-window control.

        This is the global-layer shape. The mask subgraph has no distance bound,
        so the converter must recover no window and the runtime must read the
        whole key range. If it narrowed anyway the output would be wrong here,
        which makes this the guard on over-eager recognition.
        """
        _run_case(model_runner, seq_len=1, past_seq=1024, window=0, seed=11)

    def test_decode_then_prefill_reads_full_range(self, model_runner):
        """A prefill reading the cache a narrowed decode just wrote.

        The decode narrowing skips writing ``present`` below ``kv_lo`` because
        the next decode will not read below it. This case is the exception: node
        B has ``sq > 1``, so it is not narrowed and reads the whole range,
        including what node A skipped. It is the one sequence that can turn the
        copy narrowing into wrong numbers rather than fewer bytes.

        Node B's mask is windowed, as a real export's is -- the window belongs to
        the layer, so prefill carries it too. Each of node B's query rows sits at
        absolute position >= ``past_seq + 1``, so its window admits nothing below
        ``kv_lo``: the skipped region is masked for every row, and any finite
        content there is suppressed by the ``-65504`` fill.

        What is asserted, and what deliberately is not:

        * Node A's output and the live ``[kv_lo, total)`` part of its ``present``
          are asserted exactly. That is the narrowing's own contract.
        * Node B's output is asserted only to be free of NaN, which is the
          failure this case exists to detect: a stale non-finite value in the
          skipped region survives any finite mask and would take the whole
          softmax row with it.
        * Node B's *values* are not asserted, because they are wrong for an
          unrelated reason. In a chained graph the EP misderives node B's
          geometry -- it reports ``total_seq = past_buf_seq`` and hence
          ``past_len = total_seq - sq``, writing the new tokens over valid past
          keys. The same geometry as a single node scores 1.000000, and the
          error is identical with the narrowing off, so asserting on it here
          would only encode a pre-existing bug.
        """
        batch, past_seq, second_seq, window = 1, 2048, 8, 1024
        rng = np.random.default_rng(23)

        def rand(shape):
            return (rng.standard_normal(shape) * 0.5).astype(np.float16)

        q_a = rand([batch, 1, Q_HIDDEN])
        k_a = rand([batch, 1, KV_HIDDEN])
        v_a = rand([batch, 1, KV_HIDDEN])
        past_k = rand([batch, KV_NUM_HEADS, past_seq, HEAD_DIM])
        past_v = rand([batch, KV_NUM_HEADS, past_seq, HEAD_DIM])
        qidx_a, kidx_a = _positions(batch, 1, past_seq)
        mask_a = _make_window_mask(batch, 1, past_seq, window)

        q_b = rand([batch, second_seq, Q_HIDDEN])
        k_b = rand([batch, second_seq, KV_HIDDEN])
        v_b = rand([batch, second_seq, KV_HIDDEN])
        qidx_b, kidx_b = _positions(batch, second_seq, past_seq + 1)

        actual = model_runner.backend.run(
            _persist(model_runner, _make_chained_model(batch, second_seq, window)),
            [
                q_a,
                k_a,
                v_a,
                qidx_a,
                kidx_a,
                past_k,
                past_v,
                q_b,
                k_b,
                v_b,
                qidx_b,
                kidx_b,
            ],
        )
        out_b, out_a, present_a_k, present_a_v = (
            actual[0],
            actual[1],
            actual[2],
            actual[3],
        )

        ref_pa_k = np.concatenate(
            [past_k.astype(np.float64), _to_bnsh(k_a, batch, 1)], axis=2
        )
        ref_pa_v = np.concatenate(
            [past_v.astype(np.float64), _to_bnsh(v_a, batch, 1)], axis=2
        )

        assert not np.isnan(np.asarray(out_b, dtype=np.float32)).any(), (
            "prefill-after-decode output has NaN: it read cache positions the "
            "narrowed decode left unwritten, and a finite mask cannot suppress "
            "a non-finite value"
        )
        _assert_live_present(present_a_k, ref_pa_k, past_seq + 1, window)
        _assert_live_present(present_a_v, ref_pa_v, past_seq + 1, window)
        compare_outputs(
            [out_a],
            [_attend(q_a, ref_pa_k, ref_pa_v, mask_a)],
            atol=2e-2,
            rtol=2e-2,
            cos_threshold=0.999,
        )

    @pytest.mark.parametrize("seq_len,past_seq,window", [(128, 0, 32), (64, 64, 32)])
    def test_prefill_windowed(self, model_runner, seq_len, past_seq, window):
        """Windowed prefill (sq > 1) must be unaffected by decode narrowing.

        The decode narrowing is gated on ``sq == 1``; a banded prefill is a
        separate problem. This is the guard that the gate holds: a recovered
        window is present on these runs too, and they still have to match the
        full reference.
        """
        _run_case(
            model_runner, seq_len=seq_len, past_seq=past_seq, window=window, seed=13
        )
