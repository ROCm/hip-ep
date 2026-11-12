#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""Primitive scaled-dot-product attention, as diffusers exports it, against CPU.

The graph is Transpose / Mul / MatMul / Softmax / MatMul over the projections,
which the frontend rewrites into com.microsoft.MultiHeadAttention and so runs
on its flash kernel. The cases are the SD1.5 UNet's self- and cross-attention
at its head sizes 40, 80 and 160.
"""

import numpy as np
import pytest
from onnx import TensorProto, helper, numpy_helper

from framework.comparator import compare_outputs
from framework.onnx_utils import make_model_from_nodes

F16 = TensorProto.FLOAT16


def _const(name, array):
    return numpy_helper.from_array(np.asarray(array), name=name)


def _make_attention_model(batch, sq, skv, heads, head_dim):
    """SD-style attention: d^-1/4 on Q and on K^T, BSNH in and out."""
    hidden = heads * head_dim
    q = helper.make_tensor_value_info("q", F16, [batch, sq, hidden])
    k = helper.make_tensor_value_info("k", F16, [batch, skv, hidden])
    v = helper.make_tensor_value_info("v", F16, [batch, skv, hidden])
    out = helper.make_tensor_value_info("out", F16, [batch, sq, hidden])
    inits = [
        _const("split", np.array([batch, -1, heads, head_dim], np.int64)),
        _const("merge", np.array([batch, sq, hidden], np.int64)),
        _const("s", np.array([head_dim**-0.25], np.float16)),
    ]
    nodes = [
        helper.make_node("Reshape", ["q", "split"], ["qr"]),
        helper.make_node("Transpose", ["qr"], ["qt"], perm=[0, 2, 1, 3]),
        helper.make_node("Mul", ["qt", "s"], ["qs"]),
        helper.make_node("Reshape", ["k", "split"], ["kr"]),
        helper.make_node("Transpose", ["kr"], ["kt"], perm=[0, 2, 3, 1]),
        helper.make_node("Mul", ["kt", "s"], ["ks"]),
        helper.make_node("Reshape", ["v", "split"], ["vr"]),
        helper.make_node("Transpose", ["vr"], ["vt"], perm=[0, 2, 1, 3]),
        helper.make_node("MatMul", ["qs", "ks"], ["scores"]),
        helper.make_node("Softmax", ["scores"], ["probs"], axis=-1),
        helper.make_node("MatMul", ["probs", "vt"], ["o"]),
        helper.make_node("Transpose", ["o"], ["ot"], perm=[0, 2, 1, 3]),
        helper.make_node("Reshape", ["ot", "merge"], ["out"]),
    ]
    return make_model_from_nodes(nodes, [q, k, v], [out], initializers=inits)


class TestDecomposedAttention:
    @pytest.mark.parametrize(
        "sq,skv,head_dim",
        [
            (256, 256, 40),  # 64x64 level, self
            (256, 77, 40),  # 64x64 level, cross
            (128, 128, 80),  # 32x32 level
            (64, 64, 160),  # 16x16 / 8x8 levels
            (64, 77, 160),
        ],
    )
    def test_sd_attention(self, model_runner, sq, skv, head_dim):
        batch, heads = 2, 8
        model = _make_attention_model(batch, sq, skv, heads, head_dim)
        rng = np.random.default_rng(11)
        hidden = heads * head_dim
        q = rng.standard_normal((batch, sq, hidden)).astype(np.float16)
        k = rng.standard_normal((batch, skv, hidden)).astype(np.float16)
        v = rng.standard_normal((batch, skv, hidden)).astype(np.float16)
        actual, expected = model_runner.run_sample(model, [q, k, v])
        # The reference rounds Q*s, K*s and the scores to fp16 before the
        # softmax; the flash kernel accumulates the scores in fp32.
        compare_outputs(actual, expected, atol=1e-2, rtol=1e-2)
