#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""ONNX Attention (opset 23) without a mask or past KV: bidirectional encoder
attention, as in windowed vision blocks.

When the KV sequence length is static, seqlens_k is known at compile time.
With a batch that is not a static 1 it is broadcast on the device, so the
constant must live in device-readable storage. A folded host constant made
the broadcast fault on the GPU. The cases cover a dynamic batch and a static
batch above 1, next to the dynamic-length and static-batch-1 forms that never
broadcast a constant.
"""

import numpy as np
import pytest
from onnx import TensorProto, helper

from framework.comparator import compare_outputs
from framework.onnx_utils import make_model_from_nodes

HEADS = 16


def _make_model(batch, seq, head_dim):
    hidden = HEADS * head_dim
    shape = [batch, seq, hidden]
    names = ("q", "k", "v")
    inputs = [
        helper.make_tensor_value_info(n, TensorProto.FLOAT16, shape) for n in names
    ]
    output = helper.make_tensor_value_info("y", TensorProto.FLOAT16, shape)
    node = helper.make_node(
        "Attention",
        list(names),
        ["y"],
        q_num_heads=HEADS,
        kv_num_heads=HEADS,
        is_causal=0,
        scale=float(1.0 / np.sqrt(head_dim)),
    )
    return make_model_from_nodes([node], inputs, [output], opset=23)


@pytest.mark.parametrize(
    "batch,seq,run_batch,run_seq,head_dim",
    [
        ("B", 4, 64, 4, 72),
        ("B", 16, 1, 16, 64),
        ("B", 16, 3, 16, 64),
        (4, 16, 4, 16, 64),
        ("B", "S", 4, 16, 64),
        (1, 16, 1, 16, 64),
    ],
    ids=[
        "dyn_batch_static_seq4",
        "dyn_batch_static_seq16_run1",
        "dyn_batch_static_seq16_run3",
        "static_batch4_static_seq",
        "dynamic_seq",
        "static_batch1",
    ],
)
def test_onnx_attention_encoder(model_runner, batch, seq, run_batch, run_seq, head_dim):
    model = _make_model(batch, seq, head_dim)
    rng = np.random.default_rng(23)
    shape = [run_batch, run_seq, HEADS * head_dim]
    inputs = [rng.uniform(-2, 2, shape).astype(np.float16) for _ in range(3)]
    actual, expected = model_runner.run_sample(model, inputs)
    compare_outputs(actual, expected, atol=1e-2, rtol=1e-2, cos_threshold=0.9999)
