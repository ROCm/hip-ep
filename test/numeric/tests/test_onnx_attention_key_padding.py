#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""ONNX Attention (opset 23) with is_causal=1 and an additive mask built from
the padding mask AND the causal compare, optionally AND a sliding window.

The compiler reduces that mask to its per-key padding leg (hip.gqa key_bias)
and leaves causality and the window to the kernels, which index absolutely.
That is exact for every query that is a real token, the only rows a generator
reads; a padded query's own row is not (a fully masked row is
implementation-defined, and a right-padded query's clamped position puts its
window elsewhere), and no real token ever attends to it. The comparison is
therefore restricted to real-token rows of Y. present_key / present_value are
compared from the oldest key the window keeps: a sliding layer does not copy
the past below it into present, since no later step reads it.

Head geometry follows the targets of the fused key-bias kernels: head_dim
128, 6 query heads per KV head on full-attention layers and 9 on sliding ones.
"""

import numpy as np
import pytest
from onnx import TensorProto, helper, numpy_helper

from framework.comparator import compare_outputs
from framework.onnx_utils import make_model_from_nodes

HEAD_DIM = 128
KV_HEADS = 2
DROP = -65504.0


def _make_model(q_heads, window):
    """Attention whose mask is Unsqueeze(Cast_f16(Where(keep, 0, -65504))) with
    keep = And(pad, causal[, window]) over CumSum positions of the mask; the
    query positions are the last S of them."""
    hd = HEAD_DIM
    inits = [
        numpy_helper.from_array(np.array(1, np.int64), "ax1"),
        numpy_helper.from_array(np.array([1], np.int64), "c1"),
        numpy_helper.from_array(np.array([2], np.int64), "c2"),
        numpy_helper.from_array(np.array([np.iinfo(np.int64).max], np.int64), "q_end"),
        numpy_helper.from_array(np.array(0.0, np.float32), "keep_v"),
        numpy_helper.from_array(np.array(DROP, np.float32), "drop_v"),
    ]
    nodes = [
        helper.make_node("Shape", ["q"], ["q_len"], start=1, end=2),
        helper.make_node("Neg", ["q_len"], ["q_beg"]),
        helper.make_node("CumSum", ["attention_mask", "ax1"], ["kidx"]),
        helper.make_node("Slice", ["kidx", "q_beg", "q_end", "c1"], ["qidx"]),
        helper.make_node("Unsqueeze", ["qidx", "c2"], ["qpos"]),
        helper.make_node("Unsqueeze", ["kidx", "c1"], ["kpos"]),
        helper.make_node("GreaterOrEqual", ["qpos", "kpos"], ["causal"]),
        helper.make_node("Unsqueeze", ["attention_mask", "c1"], ["pad_i"]),
        helper.make_node("Cast", ["pad_i"], ["pad"], to=TensorProto.BOOL),
    ]
    leg = "causal"
    if window:
        inits.append(numpy_helper.from_array(np.array(window, np.int64), "win"))
        nodes += [
            helper.make_node("Sub", ["qpos", "kpos"], ["dist"]),
            helper.make_node("Less", ["dist", "win"], ["in_win"]),
            helper.make_node("And", ["causal", "in_win"], ["win_leg"]),
        ]
        leg = "win_leg"
    nodes += [
        helper.make_node("And", ["pad", leg], ["keep"]),
        helper.make_node("Where", ["keep", "keep_v", "drop_v"], ["bias32"]),
        helper.make_node("Cast", ["bias32"], ["bias16"], to=TensorProto.FLOAT16),
        helper.make_node("Unsqueeze", ["bias16", "c1"], ["mask"]),
        helper.make_node(
            "Attention",
            ["q", "k", "v", "mask", "past_key", "past_value"],
            ["y", "present_key", "present_value"],
            is_causal=1,
            q_num_heads=q_heads,
            kv_num_heads=KV_HEADS,
        ),
    ]
    # Symbolic batch and lengths, as in the exports.
    f16 = TensorProto.FLOAT16
    inputs = [
        helper.make_tensor_value_info("q", f16, ["B", "S", q_heads * hd]),
        helper.make_tensor_value_info("k", f16, ["B", "S", KV_HEADS * hd]),
        helper.make_tensor_value_info("v", f16, ["B", "S", KV_HEADS * hd]),
        helper.make_tensor_value_info("past_key", f16, ["B", KV_HEADS, "P", hd]),
        helper.make_tensor_value_info("past_value", f16, ["B", KV_HEADS, "P", hd]),
        helper.make_tensor_value_info("attention_mask", TensorProto.INT64, ["B", "T"]),
    ]
    outputs = [
        helper.make_tensor_value_info("y", f16, ["B", "S", q_heads * hd]),
        helper.make_tensor_value_info("present_key", f16, ["B", KV_HEADS, "T", hd]),
        helper.make_tensor_value_info("present_value", f16, ["B", KV_HEADS, "T", hd]),
    ]
    return make_model_from_nodes(nodes, inputs, outputs, initializers=inits, opset=23)


def _run(model_runner, batch, seq, past, q_heads, window, pad, left):
    """Batch b > 0 pads b * pad keys, on the left (generation) or the right."""
    total = seq + past
    rng = np.random.default_rng(seq * 131 + past * 7 + q_heads)
    hd = HEAD_DIM
    q = rng.standard_normal([batch, seq, q_heads * hd]).astype(np.float16)
    k = rng.standard_normal([batch, seq, KV_HEADS * hd]).astype(np.float16)
    v = rng.standard_normal([batch, seq, KV_HEADS * hd]).astype(np.float16)
    pk = rng.standard_normal([batch, KV_HEADS, past, hd]).astype(np.float16)
    pv = rng.standard_normal([batch, KV_HEADS, past, hd]).astype(np.float16)
    mask = np.ones([batch, total], np.int64)
    for b in range(1, batch):
        n = b * pad
        if left:
            mask[b, :n] = 0
        else:
            mask[b, total - n :] = 0

    model = _make_model(q_heads, window)
    actual, expected = model_runner.run_sample(model, [q, k, v, pk, pv, mask])
    real = mask[:, past:].astype(bool)
    kept = max(0, past - window + 1) if window else 0
    compare_outputs(
        [actual[0][real]] + [p[:, :, kept:] for p in actual[1:]],
        [expected[0][real]] + [p[:, :, kept:] for p in expected[1:]],
        atol=1e-2,
        rtol=1e-2,
        cos_threshold=0.999,
    )


class TestOnnxAttentionKeyPadding:
    @pytest.mark.parametrize(
        "q_heads,window",
        [pytest.param(12, 0, id="full_hpg6"), pytest.param(18, 64, id="sliding_hpg9")],
    )
    @pytest.mark.parametrize("left", [False, True], ids=["right_pad", "left_pad"])
    def test_prefill(self, model_runner, q_heads, window, left):
        """Pure prefill over an empty past: the fused prefill kernel."""
        _run(
            model_runner,
            batch=2,
            seq=256,
            past=0,
            q_heads=q_heads,
            window=window,
            pad=48,
            left=left,
        )

    @pytest.mark.parametrize(
        "q_heads,window",
        [pytest.param(12, 0, id="full_hpg6"), pytest.param(18, 64, id="sliding_hpg9")],
    )
    def test_chunked_prefill(self, model_runner, q_heads, window):
        """A later prefill chunk whose left padding sits in the past."""
        _run(
            model_runner,
            batch=2,
            seq=64,
            past=192,
            q_heads=q_heads,
            window=window,
            pad=40,
            left=True,
        )

    @pytest.mark.parametrize(
        "q_heads,window",
        [pytest.param(12, 0, id="full_hpg6"), pytest.param(18, 64, id="sliding_hpg9")],
    )
    @pytest.mark.parametrize("batch", [1, 2])
    def test_decode(self, model_runner, q_heads, window, batch):
        """One new token over a left-padded cache: the flash-decode kernels."""
        _run(
            model_runner,
            batch=batch,
            seq=1,
            past=255,
            q_heads=q_heads,
            window=window,
            pad=70,
            left=True,
        )
