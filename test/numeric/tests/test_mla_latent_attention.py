#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""Tests for attention over a latent (MLA) KV cache.

The graph keeps one row [c (256), k_r (64)] per token in a fixed
[B, 1, rows, 320] buffer written with TensorScatter at nonpad - S, expands it
into per-head keys and values (c @ W_k / c @ W_v up-projections, k_r
broadcast over heads) and runs a causal onnx.Attention bounded by
nonpad_kv_seqlen. hip-ep rewrites that into hip.mla_attention, which expands
for prefill and moves the query into the latent space for decode, so the
cases cover a first prefill, a chunked prefill over a past, and decode
steps at short and long cache lengths.

Geometry: H = 32, kv_lora_rank = 256, qk_rope = 64, qk_nope = 64, v = 128;
up-projections as int4 MatMulNBits (block 128) or fp16 MatMul.
"""

import numpy as np
import pytest
from onnx import TensorProto, helper, numpy_helper

from framework.comparator import compare_outputs
from framework.onnx_utils import make_model_from_nodes

H = 32
LAT = 256
ROPE = 64
NOPE = 64
DV = 128
BLOCK = 128
SCALE = 0.19496949


def _quantize_int4(w):
    """RTN int4 with the implicit zero point 8: packed, scales, dequantized."""
    n, k = w.shape
    blocks = w.reshape(n, k // BLOCK, BLOCK)
    scales = np.abs(blocks).max(-1, keepdims=True) / 7.0
    scales = scales.astype(np.float16).astype(np.float32)
    q = np.clip(np.rint(blocks / scales) + 8, 0, 15).astype(np.uint8)
    packed = q[..., 0::2] | (q[..., 1::2] << 4)
    deq = ((q.astype(np.float32) - 8) * scales).reshape(n, k)
    return packed, scales.reshape(-1).astype(np.float16), deq


def _make_model(rng, dense):
    """One latent-cache attention layer: (q, c_norm, k_pe, past_k, mask) ->
    (o, present_k). The new rows go into the fixed past_k buffer at
    nonpad - S, nonpad = ReduceSum(mask), and the causal Attention reads the
    first nonpad rows."""
    inits = [
        numpy_helper.from_array(np.array([1], np.int64), "axis1"),
        numpy_helper.from_array(np.array([LAT, ROPE], np.int64), "split"),
        numpy_helper.from_array(np.array([0, 0, H, NOPE], np.int64), "shape_kn"),
        numpy_helper.from_array(np.array([0, 0, 1, ROPE], np.int64), "shape_kr"),
        numpy_helper.from_array(np.array([1, 1, H, 1], np.int64), "heads"),
        numpy_helper.from_array(np.array([0, 0, -1], np.int64), "flat"),
    ]
    projections = []
    extra_opsets = None
    for name, out, n in (("kb", "k_nope", H * NOPE), ("vb", "v", H * DV)):
        w = (rng.standard_normal((n, LAT)) * 0.06).astype(np.float32)
        packed, scales, deq = _quantize_int4(w)
        if dense:
            inits.append(
                numpy_helper.from_array(deq.T.astype(np.float16).copy(), name)
            )
            projections.append(helper.make_node("MatMul", ["c", name], [out]))
        else:
            inits.append(numpy_helper.from_array(packed, name))
            inits.append(numpy_helper.from_array(scales, name + "_s"))
            projections.append(
                helper.make_node(
                    "MatMulNBits",
                    ["c", name, name + "_s"],
                    [out],
                    domain="com.microsoft",
                    K=LAT,
                    N=n,
                    bits=4,
                    block_size=BLOCK,
                )
            )
            extra_opsets = [helper.make_opsetid("com.microsoft", 1)]

    nodes = [
        helper.make_node(
            "ReduceSum", ["attention_mask", "axis1"], ["nonpad"], keepdims=0
        ),
        helper.make_node("Shape", ["q"], ["S"], start=1, end=2),
        helper.make_node("Sub", ["nonpad", "S"], ["write_idx"]),
        helper.make_node("Concat", ["c_norm", "k_pe"], ["latent"], axis=-1),
        helper.make_node("Unsqueeze", ["latent", "axis1"], ["latent4"]),
        helper.make_node(
            "TensorScatter",
            ["past_k", "latent4", "write_idx"],
            ["present_k"],
            axis=2,
        ),
        helper.make_node("Squeeze", ["present_k", "axis1"], ["lat"]),
        helper.make_node("Split", ["lat", "split"], ["c", "k_r"], axis=-1),
        *projections,
        helper.make_node("Reshape", ["k_nope", "shape_kn"], ["k_nope4"]),
        helper.make_node("Reshape", ["k_r", "shape_kr"], ["k_r4"]),
        helper.make_node("Expand", ["k_r4", "heads"], ["k_rh"]),
        helper.make_node("Concat", ["k_nope4", "k_rh"], ["k4"], axis=-1),
        helper.make_node("Reshape", ["k4", "flat"], ["k"]),
        helper.make_node(
            "Attention",
            ["q", "k", "v", "", "", "", "nonpad"],
            ["o"],
            q_num_heads=H,
            kv_num_heads=H,
            scale=SCALE,
            is_causal=1,
        ),
    ]

    f16 = TensorProto.FLOAT16
    inputs = [
        helper.make_tensor_value_info("q", f16, [1, "S", H * (NOPE + ROPE)]),
        helper.make_tensor_value_info("c_norm", f16, [1, "S", LAT]),
        helper.make_tensor_value_info("k_pe", f16, [1, "S", ROPE]),
        helper.make_tensor_value_info("past_k", f16, [1, 1, "P", LAT + ROPE]),
        helper.make_tensor_value_info("attention_mask", TensorProto.INT64, [1, "T"]),
    ]
    outputs = [
        helper.make_tensor_value_info("o", f16, [1, "S", H * DV]),
        helper.make_tensor_value_info("present_k", f16, [1, 1, "P", LAT + ROPE]),
    ]
    model = make_model_from_nodes(
        nodes,
        inputs,
        outputs,
        initializers=inits,
        opset=24,
        extra_opsets=extra_opsets,
    )
    # Opset 24 needs IR 11; newer onnx packages stamp a version ORT may not
    # load yet.
    model.ir_version = 11
    return model


class TestMlaLatentAttention:
    @pytest.mark.parametrize("dense", [False, True], ids=["int4", "fp16"])
    @pytest.mark.parametrize(
        "past,seq,rows",
        [(0, 256, 256), (0, 256, 512), (384, 128, 640), (100, 37, 200),
         (511, 1, 700), (2047, 1, 2200)],
        ids=["prefill_full", "prefill", "chunk", "chunk_odd", "decode",
             "decode_long"],
    )
    def test_latent_attention(self, model_runner, dense, past, seq, rows):
        """Rows past the valid prefix hold noise the attention must not read."""
        rng = np.random.default_rng(7302)
        model = _make_model(rng, dense)
        q = (rng.standard_normal((1, seq, H * (NOPE + ROPE))) * 0.5).astype(np.float16)
        c_norm = rng.standard_normal((1, seq, LAT)).astype(np.float16)
        k_pe = rng.standard_normal((1, seq, ROPE)).astype(np.float16)
        past_k = rng.standard_normal((1, 1, rows, LAT + ROPE))
        past_k[:, :, past:] *= 8
        past_k = past_k.astype(np.float16)
        mask = np.ones((1, past + seq), np.int64)
        actual, expected = model_runner.run_sample(
            model, [q, c_norm, k_pe, past_k, mask]
        )
        compare_outputs(actual, expected, atol=1e-2, rtol=1e-2, cos_threshold=0.999)
