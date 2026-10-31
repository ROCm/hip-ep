#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""Check GQA lengths computed from two different past-cache shapes."""

import numpy as np
from onnx import TensorProto, helper

from framework.onnx_utils import make_model_from_nodes


def make_host_length_case(seq_len=1, dtype=np.float16):
    """Return a graph, inputs, and a NumPy reference for two GQA calls."""
    tp = TensorProto.FLOAT16 if dtype == np.float16 else TensorProto.FLOAT
    tensor = helper.make_tensor_value_info
    inputs = [tensor(name, tp, [1, None, 64]) for name in ("q", "k", "v")]
    nodes = []
    rng = np.random.default_rng(42)
    data = [rng.uniform(-0.5, 0.5, (1, seq_len, 64)).astype(dtype) for _ in inputs]
    q, k, v = data
    outputs, expected = [], []
    for layer, past_len in enumerate((2, 5)):
        suffix = str(layer)
        pk, pv = "pk" + suffix, "pv" + suffix
        inputs.extend(
            [tensor(pk, tp, [1, 1, None, 64]), tensor(pv, tp, [1, 1, None, 64])]
        )
        past_k, past_v = [
            rng.uniform(-0.5, 0.5, (1, 1, past_len, 64)).astype(dtype) for _ in range(2)
        ]
        data.extend([past_k, past_v])
        out, present_k, present_v = [
            name + suffix for name in ("out", "present_k", "present_v")
        ]
        nodes.extend(
            [
                helper.make_node(
                    "Attention",
                    ["q", "k", "v", "", pk, pv],
                    [out, present_k, present_v],
                    q_num_heads=1,
                    kv_num_heads=1,
                    scale=0.125,
                    is_causal=1,
                ),
            ]
        )
        outputs.extend(
            [
                tensor(out, tp, [1, None, 64]),
                tensor(present_k, tp, [1, 1, None, 64]),
                tensor(present_v, tp, [1, 1, None, 64]),
            ]
        )
        full_k = np.concatenate([past_k[0, 0], k[0]], axis=0)
        full_v = np.concatenate([past_v[0, 0], v[0]], axis=0)
        scores = q[0].astype(np.float32) @ full_k.astype(np.float32).T * 0.125
        mask = (
            np.arange(past_len + seq_len)[None, :]
            > past_len + np.arange(seq_len)[:, None]
        )
        scores[mask] = -np.inf
        probs = np.exp(scores - scores.max(axis=-1, keepdims=True))
        probs /= probs.sum(axis=-1, keepdims=True)
        result = (probs @ full_v.astype(np.float32)).astype(dtype)[None, :, :]
        expected.extend([result, full_k[None, None], full_v[None, None]])
    model = make_model_from_nodes(
        nodes,
        inputs,
        outputs,
        opset=23,
    )
    return model, data, expected


def test_gqa_host_lengths(model_runner):
    for dtype in (np.float16, np.float32):
        for seq_len in (1, 4):
            model, inputs, expected = make_host_length_case(seq_len, dtype)
            actual, reference = model_runner.run_sample(
                model, inputs, name=f"host_lengths_{np.dtype(dtype).name}_{seq_len}"
            )
            for result, cpu, numpy in zip(actual, reference, expected):
                np.testing.assert_allclose(result, cpu, rtol=2e-3, atol=1e-3)
                np.testing.assert_allclose(result, numpy, rtol=2e-3, atol=1e-3)
