#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""Tests for com.microsoft MRotaryEmbedding (rank-3 input, interleaved=0).

The Qwen3.8 cases use the model geometry: head 256, rotary 64,
mrope_section [11, 11, 10], mrope_layout 1, num_heads 24 (query) and 2 (key).
"""

import numpy as np
import pytest
from onnx import TensorProto as TP
from onnx import helper, numpy_helper

from framework.comparator import compare_outputs
from framework.onnx_utils import make_model_from_nodes

_OPSETS = dict(opset=22, extra_opsets=[helper.make_opsetid("com.microsoft", 1)])
_NP = {TP.FLOAT16: np.float16, TP.FLOAT: np.float32}
_ATOL = {TP.FLOAT16: 2e-3, TP.FLOAT: 1e-5}


def _run(
    model_runner,
    *,
    batch,
    seq,
    num_heads,
    head,
    rot_attr,
    section,
    layout,
    t_onnx,
    max_pos=4096,
    scale=None,
    symbolic=False,
):
    rot = rot_attr or head
    hidden = num_heads * head
    dtype = _NP[t_onnx]
    rng = np.random.default_rng(1234 + seq)
    cos = rng.uniform(-1, 1, [max_pos, rot // 2]).astype(dtype)
    sin = rng.uniform(-1, 1, [max_pos, rot // 2]).astype(dtype)

    attrs = dict(
        num_heads=num_heads,
        rotary_embedding_dim=rot_attr,
        mrope_section=list(section),
        mrope_layout=layout,
        interleaved=0,
        is_packed_batching=0,
    )
    if scale is not None:
        attrs["scale"] = scale
    node = helper.make_node(
        "MRotaryEmbedding",
        ["input", "position_ids", "cos_cache", "sin_cache"],
        ["output"],
        domain="com.microsoft",
        **attrs,
    )
    b_dim, s_dim = ("B", "S") if symbolic else (batch, seq)
    model = make_model_from_nodes(
        [node],
        [
            helper.make_tensor_value_info("input", t_onnx, [b_dim, s_dim, hidden]),
            helper.make_tensor_value_info("position_ids", TP.INT64, [3, b_dim, s_dim]),
        ],
        [helper.make_tensor_value_info("output", t_onnx, [b_dim, s_dim, hidden])],
        initializers=[
            numpy_helper.from_array(cos, "cos_cache"),
            numpy_helper.from_array(sin, "sin_cache"),
        ],
        **_OPSETS,
    )

    x = rng.uniform(-1, 1, [batch, seq, hidden]).astype(dtype)
    pos = rng.integers(0, max_pos, [3, batch, seq], dtype=np.int64)
    actual, expected = model_runner.run_sample(model, [x, pos])
    compare_outputs(actual, expected, atol=_ATOL[t_onnx])


class TestMRotaryEmbedding:
    @pytest.mark.parametrize("seq", [1, 7, 64])
    @pytest.mark.parametrize("num_heads", [24, 2])
    def test_qwen38(self, model_runner, num_heads, seq):
        _run(
            model_runner,
            batch=1,
            seq=seq,
            num_heads=num_heads,
            head=256,
            rot_attr=64,
            section=(11, 11, 10),
            layout=1,
            t_onnx=TP.FLOAT16,
        )

    @pytest.mark.parametrize("batch,seq", [(1, 1), (2, 9)])
    def test_qwen38_symbolic_dims(self, model_runner, batch, seq):
        _run(
            model_runner,
            batch=batch,
            seq=seq,
            num_heads=24,
            head=256,
            rot_attr=64,
            section=(11, 11, 10),
            layout=1,
            t_onnx=TP.FLOAT16,
            symbolic=True,
        )

    def test_qwen38_full_cache(self, model_runner):
        _run(
            model_runner,
            batch=1,
            seq=7,
            num_heads=24,
            head=256,
            rot_attr=64,
            section=(11, 11, 10),
            layout=1,
            t_onnx=TP.FLOAT16,
            max_pos=262144,
        )

    @pytest.mark.parametrize("layout", [0, 1])
    @pytest.mark.parametrize("t_onnx", [TP.FLOAT16, TP.FLOAT])
    def test_partial_rotary_batch2(self, model_runner, t_onnx, layout):
        _run(
            model_runner,
            batch=2,
            seq=7,
            num_heads=4,
            head=128,
            rot_attr=64,
            section=(16, 8, 8),
            layout=layout,
            t_onnx=t_onnx,
        )

    @pytest.mark.parametrize("layout", [0, 1])
    @pytest.mark.parametrize("t_onnx", [TP.FLOAT16, TP.FLOAT])
    def test_full_rotary(self, model_runner, t_onnx, layout):
        _run(
            model_runner,
            batch=2,
            seq=7,
            num_heads=2,
            head=32,
            rot_attr=0,
            section=(4, 6, 6),
            layout=layout,
            t_onnx=t_onnx,
        )

    @pytest.mark.parametrize("t_onnx", [TP.FLOAT16, TP.FLOAT])
    def test_scale(self, model_runner, t_onnx):
        _run(
            model_runner,
            batch=2,
            seq=7,
            num_heads=2,
            head=256,
            rot_attr=64,
            section=(11, 11, 10),
            layout=1,
            t_onnx=t_onnx,
            scale=0.7,
        )
