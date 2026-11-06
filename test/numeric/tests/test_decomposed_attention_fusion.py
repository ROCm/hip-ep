#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""Primitive scaled dot-product attention, as exporters without a fused op
write it, against the ORT CPU reference.

  q, k, v : [B, S, N * H] fp16
  per head: Reshape [B, S, N, H] -> Transpose, optional scalar Mul on Q and K
  scores  : MatMul(Q, K^T), optional scalar Div, Softmax(axis=-1)
  output  : MatMul(P, V) -> Transpose [B, S, N, H]

The pre-lowering DecomposedAttentionFusion rewrites the chain into
com.microsoft.MultiHeadAttention, which runs on the flash-attention prefill
kernel; test_decomposed_attention_fusion.mlir checks the rewrite itself. Here
the fused GPU result is compared with the CPU running the primitives.
"""

import numpy as np
import pytest
from onnx import TensorProto, helper, numpy_helper

from framework.comparator import compare_outputs
from framework.onnx_utils import make_model_from_nodes


def _make_sdpa_model(batch, seq_q, seq_kv, num_heads, head_dim, scale_form):
    """Build the primitive attention chain.

    scale_form: "split" multiplies Q and K by sqrt(1/sqrt(H)) each (the
    diffusers UNet export), "div" divides the scores by sqrt(H).
    """
    hidden = num_heads * head_dim
    fp16 = TensorProto.FLOAT16
    inputs = [
        helper.make_tensor_value_info("q", fp16, [batch, seq_q, hidden]),
        helper.make_tensor_value_info("k", fp16, [batch, seq_kv, hidden]),
        helper.make_tensor_value_info("v", fp16, [batch, seq_kv, hidden]),
    ]
    output = helper.make_tensor_value_info(
        "out", fp16, [batch, seq_q, num_heads, head_dim]
    )
    split = np.array([batch, -1, num_heads, head_dim], dtype=np.int64)
    inits = [numpy_helper.from_array(split, "split")]

    nodes = []
    for name, perm in (("q", [0, 2, 1, 3]), ("k", [0, 2, 3, 1]), ("v", [0, 2, 1, 3])):
        nodes.append(helper.make_node("Reshape", [name, "split"], [f"{name}4"]))
        nodes.append(
            helper.make_node("Transpose", [f"{name}4"], [f"{name}t"], perm=perm)
        )

    q_in, k_in = "qt", "kt"
    if scale_form == "split":
        half = np.array(head_dim**-0.25, dtype=np.float16)
        inits.append(numpy_helper.from_array(half, "half_scale"))
        nodes.append(helper.make_node("Mul", ["qt", "half_scale"], ["qs"]))
        nodes.append(helper.make_node("Mul", ["kt", "half_scale"], ["ks"]))
        q_in, k_in = "qs", "ks"
    nodes.append(helper.make_node("MatMul", [q_in, k_in], ["scores"]))
    scores = "scores"
    if scale_form == "div":
        root = np.array(head_dim**0.5, dtype=np.float16)
        inits.append(numpy_helper.from_array(root, "sqrt_h"))
        nodes.append(helper.make_node("Div", ["scores", "sqrt_h"], ["scaled"]))
        scores = "scaled"
    nodes.append(helper.make_node("Softmax", [scores], ["probs"], axis=-1))
    nodes.append(helper.make_node("MatMul", ["probs", "vt"], ["attn"]))
    nodes.append(helper.make_node("Transpose", ["attn"], ["out"], perm=[0, 2, 1, 3]))
    return make_model_from_nodes(nodes, inputs, [output], initializers=inits)


def _qkv(batch, seq_q, seq_kv, hidden, seed=7):
    rng = np.random.default_rng(seed)
    q = rng.uniform(-1.0, 1.0, [batch, seq_q, hidden]).astype(np.float16)
    k = rng.uniform(-1.0, 1.0, [batch, seq_kv, hidden]).astype(np.float16)
    v = rng.uniform(-1.0, 1.0, [batch, seq_kv, hidden]).astype(np.float16)
    return q, k, v


class TestDecomposedAttentionFusion:
    @pytest.mark.parametrize("head_dim", [40, 160])
    @pytest.mark.parametrize("seq_kv", [256, 77])
    def test_split_scale(self, model_runner, head_dim, seq_kv):
        """The SD1.5 UNet form: scale split over Q and K, self- and cross-attention."""
        batch, seq_q, num_heads = 2, 256, 8
        model = _make_sdpa_model(batch, seq_q, seq_kv, num_heads, head_dim, "split")
        q, k, v = _qkv(batch, seq_q, seq_kv, num_heads * head_dim)
        actual, expected = model_runner.run_sample(model, [q, k, v])
        compare_outputs(actual, expected, atol=1e-2, rtol=1e-2, cos_threshold=0.999)

    def test_score_div(self, model_runner):
        """Scale applied to the scores with a Div."""
        batch, seq, num_heads, head_dim = 1, 128, 4, 64
        model = _make_sdpa_model(batch, seq, seq, num_heads, head_dim, "div")
        q, k, v = _qkv(batch, seq, seq, num_heads * head_dim)
        actual, expected = model_runner.run_sample(model, [q, k, v])
        compare_outputs(actual, expected, atol=1e-2, rtol=1e-2, cos_threshold=0.999)
