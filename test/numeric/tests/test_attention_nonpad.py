#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""Tests for causal onnx.Attention bounded by nonpad_kv_seqlen.

K/V are a fixed buffer of `rows` rows whose first nonpad = ReduceSum(mask)
rows are valid; the query is the last S of them. hip-ep lowers this to
hip.kv_attention, which reads the buffer in place. Rows past the valid prefix
hold large noise, so reading any of them shows up in the output. The cases
cover a first prefill, a chunked prefill over a past, and decode steps, for
BNSH (rank-4) and BSD (rank-3) Q/K/V with 8 query heads over 2 KV heads.
"""

import numpy as np
import pytest
from onnx import TensorProto, helper, numpy_helper

from framework.comparator import compare_outputs
from framework.onnx_utils import make_model_from_nodes

H = 8
G = 2


def _make_model(d, bnsh):
    """(q, k, v, mask) -> o with nonpad = ReduceSum(mask)."""
    inits = [numpy_helper.from_array(np.array([1], np.int64), "axis1")]
    nodes = [
        helper.make_node(
            "ReduceSum", ["attention_mask", "axis1"], ["nonpad"], keepdims=0
        ),
        helper.make_node(
            "Attention",
            ["q", "k", "v", "", "", "", "nonpad"],
            ["o"],
            q_num_heads=H,
            kv_num_heads=G,
            is_causal=1,
        ),
    ]
    f16 = TensorProto.FLOAT16
    q_shape = [1, H, "S", d] if bnsh else [1, "S", H * d]
    kv_shape = [1, G, "P", d] if bnsh else [1, "P", G * d]
    inputs = [
        helper.make_tensor_value_info("q", f16, q_shape),
        helper.make_tensor_value_info("k", f16, kv_shape),
        helper.make_tensor_value_info("v", f16, kv_shape),
        helper.make_tensor_value_info("attention_mask", TensorProto.INT64, [1, "T"]),
    ]
    outputs = [helper.make_tensor_value_info("o", f16, q_shape)]
    model = make_model_from_nodes(
        nodes, inputs, outputs, initializers=inits, opset=24
    )
    # Opset 24 needs IR 11; newer onnx packages stamp a version ORT may not
    # load yet.
    model.ir_version = 11
    return model


class TestAttentionNonpad:
    @pytest.mark.parametrize("bnsh", [True, False], ids=["bnsh", "bsd"])
    @pytest.mark.parametrize("d", [64, 128])
    @pytest.mark.parametrize(
        "past,seq,rows",
        [(0, 128, 128), (0, 128, 256), (100, 37, 200), (511, 1, 700),
         (2047, 1, 2200)],
        ids=["prefill_full", "prefill", "chunk", "decode", "decode_long"],
    )
    def test_nonpad_attention(self, model_runner, bnsh, d, past, seq, rows):
        rng = np.random.default_rng(4121)
        model = _make_model(d, bnsh)
        q = rng.standard_normal((1, H, seq, d)) * 0.5
        if not bnsh:
            q = q.transpose(0, 2, 1, 3).reshape(1, seq, H * d)
        q = q.astype(np.float16)
        kv = []
        for _ in range(2):
            x = rng.standard_normal((1, G, rows, d))
            x[:, :, past + seq:] *= 8
            if not bnsh:
                x = x.transpose(0, 2, 1, 3).reshape(1, rows, G * d)
            kv.append(x.astype(np.float16))
        mask = np.ones((1, past + seq), np.int64)
        actual, expected = model_runner.run_sample(model, [q, *kv, mask])
        compare_outputs(actual, expected, atol=1e-2, rtol=1e-2, cos_threshold=0.999)
