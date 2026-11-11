#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""Decomposed attention GEMMs with grouped cache and query chunk strides."""

import numpy as np
import pytest
from onnx import TensorProto, helper, numpy_helper

from framework.comparator import compare_outputs
from framework.onnx_utils import make_model_from_nodes


@pytest.mark.parametrize("batch,seq,past", [(2, 1, 7), (2, 19, 13), (1, 257, 300)])
def test_gqa_decomposed_grouped_cache(model_runner, batch, seq, past):
    """D32 exercises the decomposed path; the long query also permits chunking.

    Run with HIPDNN_EP_GQA_NO_EXPAND=0 and with NO_EXPAND_PREFILL=1 to
    cover both cache layouts. A 1 MiB score budget chunks the long-query case.
    """
    heads, groups, dim = 4, 2, 32
    total = seq + past
    shapes = [
        [batch, seq, heads * dim],
        [batch, seq, groups * dim],
        [batch, seq, groups * dim],
        [batch, groups, past, dim],
        [batch, groups, past, dim],
        [batch],
        [],
    ]
    names = ["query", "key", "value", "past_key", "past_value", "seqlens", "total"]
    inputs = [
        helper.make_tensor_value_info(
            name, TensorProto.INT32 if i >= 5 else TensorProto.FLOAT16, shape
        )
        for i, (name, shape) in enumerate(zip(names, shapes))
    ]
    outputs = [
        helper.make_tensor_value_info(
            "Y", TensorProto.FLOAT16, [batch, seq, heads * dim]
        ),
        helper.make_tensor_value_info(
            "PK", TensorProto.FLOAT16, [batch, groups, total, dim]
        ),
        helper.make_tensor_value_info(
            "PV", TensorProto.FLOAT16, [batch, groups, total, dim]
        ),
    ]
    node = helper.make_node(
        "GroupQueryAttention",
        names,
        ["Y", "PK", "PV"],
        domain="com.microsoft",
        num_heads=heads,
        kv_num_heads=groups,
        scale=float(1 / np.sqrt(dim)),
        do_rotary=0,
    )
    model = make_model_from_nodes(
        [node], inputs, outputs, extra_opsets=[helper.make_opsetid("com.microsoft", 1)]
    )
    rng = np.random.default_rng(521)
    values = [
        rng.uniform(-0.25, 0.25, shape).astype(np.float16) for shape in shapes[:5]
    ]
    values += [
        np.full([batch], total - 1, dtype=np.int32),
        np.asarray(total, dtype=np.int32),
    ]
    # ORT CPU GQA rejects batch>1 with both prefill and a past context. Build
    # an independent mathematical reference for every case, including that
    # supported HIP layout, and feed it through the existing reference API.
    q = values[0].reshape(batch, seq, heads, dim).transpose(0, 2, 1, 3)
    k = values[1].reshape(batch, seq, groups, dim).transpose(0, 2, 1, 3)
    v = values[2].reshape(batch, seq, groups, dim).transpose(0, 2, 1, 3)
    present_k = np.concatenate([values[3], k], axis=2)
    present_v = np.concatenate([values[4], v], axis=2)
    expanded_k = np.repeat(present_k.astype(np.float64), heads // groups, axis=1)
    expanded_v = np.repeat(present_v.astype(np.float64), heads // groups, axis=1)
    scores = np.matmul(q.astype(np.float64), expanded_k.swapaxes(-1, -2)) / np.sqrt(dim)
    future = np.arange(total)[None, :] > (past + np.arange(seq))[:, None]
    scores = np.where(future, -np.inf, scores)
    probabilities = np.exp(scores - scores.max(axis=-1, keepdims=True))
    probabilities /= probabilities.sum(axis=-1, keepdims=True)
    result = np.matmul(probabilities, expanded_v).transpose(0, 2, 1, 3)
    result = result.reshape(batch, seq, heads * dim).astype(np.float16)
    reference_values = [result, present_k, present_v]
    initializers = [
        numpy_helper.from_array(value, name=f"ref_{i}")
        for i, value in enumerate(reference_values)
    ]
    reference_nodes = [
        helper.make_node("Identity", [f"ref_{i}"], [output.name])
        for i, output in enumerate(outputs)
    ]
    reference = make_model_from_nodes(
        reference_nodes, inputs, outputs, initializers=initializers
    )
    actual, expected = model_runner.run_sample(
        model, values, reference="cpu", reference_model=reference
    )
    compare_outputs(actual, expected, atol=2e-3, rtol=1e-2)
