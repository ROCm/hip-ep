#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""Check ONNX Attention over a preallocated cache whose live length is in the mask.

The past cache and the additive mask both span the full cache capacity. Only
the first ``past_len`` slots hold valid keys; the rest hold stale data that
the mask rules out with ``fill``. The CPU reference runs the same inputs
through a graph that slices the past cache and the mask to the live length.
"""

import numpy as np
import pytest
from onnx import TensorProto, helper

from framework.onnx_utils import make_model_from_nodes

CAPACITY = 16
HEAD_DIM = 64
Q_HEADS = 2
KV_HEADS = 1


def _slice(name, out, end, axis):
    starts, ends, axes = (f"{out}_{part}" for part in ("starts", "ends", "axes"))
    initializers = [
        helper.make_tensor(starts, TensorProto.INT64, [1], [0]),
        helper.make_tensor(ends, TensorProto.INT64, [1], [end]),
        helper.make_tensor(axes, TensorProto.INT64, [1], [axis]),
    ]
    return helper.make_node("Slice", [name, starts, ends, axes], [out]), initializers


def make_shared_cache_case(past_len, seq_len, fill):
    tp = TensorProto.FLOAT16
    tensor = helper.make_tensor_value_info
    inputs = [
        tensor("q", tp, [1, seq_len, Q_HEADS * HEAD_DIM]),
        tensor("k", tp, [1, seq_len, KV_HEADS * HEAD_DIM]),
        tensor("v", tp, [1, seq_len, KV_HEADS * HEAD_DIM]),
        tensor("mask", tp, [1, 1, seq_len, CAPACITY]),
        tensor("past_k", tp, [1, KV_HEADS, CAPACITY, HEAD_DIM]),
        tensor("past_v", tp, [1, KV_HEADS, CAPACITY, HEAD_DIM]),
    ]
    outputs = [tensor("out", tp, [1, seq_len, Q_HEADS * HEAD_DIM])]
    attrs = dict(q_num_heads=Q_HEADS, kv_num_heads=KV_HEADS, is_causal=0)

    model = make_model_from_nodes(
        [
            helper.make_node(
                "Attention",
                ["q", "k", "v", "mask", "past_k", "past_v"],
                ["out"],
                **attrs,
            )
        ],
        inputs,
        outputs,
        opset=23,
    )

    nodes, initializers = [], []
    for name, end, axis in (
        ("past_k", past_len, 2),
        ("past_v", past_len, 2),
        ("mask", past_len + seq_len, 3),
    ):
        node, inits = _slice(name, f"{name}_live", end, axis)
        nodes.append(node)
        initializers.extend(inits)
    # The CPU kernel requires present outputs whenever a past cache is given.
    nodes.append(
        helper.make_node(
            "Attention",
            ["q", "k", "v", "mask_live", "past_k_live", "past_v_live"],
            ["out", "present_k_live", "present_v_live"],
            **attrs,
        )
    )
    reference = make_model_from_nodes(
        nodes, inputs, outputs, initializers=initializers, opset=23
    )

    rng = np.random.default_rng(7)

    def uniform(*shape):
        return rng.uniform(-1, 1, shape).astype(np.float16)

    q = uniform(1, seq_len, Q_HEADS * HEAD_DIM)
    k = uniform(1, seq_len, KV_HEADS * HEAD_DIM)
    v = uniform(1, seq_len, KV_HEADS * HEAD_DIM)
    past_k = uniform(1, KV_HEADS, CAPACITY, HEAD_DIM)
    past_v = uniform(1, KV_HEADS, CAPACITY, HEAD_DIM)
    rows = past_len + np.arange(seq_len)[:, None]
    cols = np.arange(CAPACITY)[None, :]
    mask = np.where(cols <= rows, 0.0, fill).astype(np.float16)[None, None]
    return model, reference, [q, k, v, mask, past_k, past_v]


@pytest.mark.parametrize("fill", [-10000.0, -65504.0, -np.inf])
@pytest.mark.parametrize("past_len,seq_len", [(0, 4), (5, 1), (5, 4)])
def test_onnx_attention_shared_cache_mask_fill(model_runner, fill, past_len, seq_len):
    model, reference, inputs = make_shared_cache_case(past_len, seq_len, fill)
    actual, expected = model_runner.run_sample(
        model,
        inputs,
        name=f"shared_cache_{int(max(fill, -1e9))}_{past_len}_{seq_len}",
        reference="cpu",
        reference_model=reference,
    )
    np.testing.assert_allclose(actual[0], expected[0], rtol=2e-3, atol=1e-3)
