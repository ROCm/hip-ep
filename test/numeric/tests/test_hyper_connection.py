#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""com.microsoft HyperConnectionPreMix and HyperConnectionPostMix."""

import numpy as np
import pytest
from onnx import TensorProto, helper

from framework.comparator import compare_outputs
from framework.onnx_utils import make_model_from_nodes

_TENSOR_PROTO = {
    np.float16: TensorProto.FLOAT16,
    np.float32: TensorProto.FLOAT,
}


def _model(nodes, inputs, outputs):
    return make_model_from_nodes(
        nodes,
        inputs,
        outputs,
        extra_opsets=[helper.make_opsetid("com.microsoft", 1)],
    )


def _compare(actual, expected, dtype):
    if dtype == np.float16:
        compare_outputs(actual, expected, atol=2e-3, rtol=1e-3)
    else:
        compare_outputs(actual, expected, atol=1e-5, rtol=1e-4)


def _vi(name, dtype, shape):
    return helper.make_tensor_value_info(name, _TENSOR_PROTO[dtype], shape)


@pytest.mark.parametrize(
    "streams, gate, branches, scale, dtype",
    [
        # Flattened streams, 4 branches, hidden scaled down from 2560 to 8.
        (["batch", "seq", 32], ["batch", "seq", 32], 4, 0.25, np.float16),
        (["batch", "seq", 10240], ["batch", "seq", 10240], 4, 0.25, np.float16),
        # Per-branch gate on flattened streams.
        ([2, 3, 20], [2, 3, 4], 4, 1.0, np.float32),
        # Grouped streams, per-feature gate.
        ([2, 3, 4, 6], [2, 3, 4, 6], 0, 0.5, np.float16),
        # (C, 1) gate.
        ([1, 2, 4, 8], [1, 2, 4, 1], 0, 1.0, np.float32),
    ],
)
def test_hyper_connection_pre_mix(model_runner, streams, gate, branches, scale, dtype):
    hidden = streams[-1] // branches if branches else streams[-1]
    out_shape = list(streams[: -1 if branches else -2]) + [hidden]
    attrs = {"reduction_scale": scale}
    if branches:
        attrs["num_branches"] = branches
    node = helper.make_node(
        "HyperConnectionPreMix",
        ["streams", "pre_mix"],
        ["out"],
        domain="com.microsoft",
        **attrs,
    )
    model = _model(
        [node],
        [_vi("streams", dtype, streams), _vi("pre_mix", dtype, gate)],
        [_vi("out", dtype, out_shape)],
    )
    rng = np.random.default_rng(38)
    run_streams = [2 if d == "batch" else 3 if d == "seq" else d for d in streams]
    run_gate = [2 if d == "batch" else 3 if d == "seq" else d for d in gate]
    actual, expected = model_runner.run_sample(
        model,
        [
            rng.uniform(-1, 1, run_streams).astype(dtype),
            rng.uniform(0, 1, run_gate).astype(dtype),
        ],
    )
    _compare(actual, expected, dtype)


@pytest.mark.parametrize(
    "streams, gate, mix, branches, dtype",
    [
        (["batch", "seq", 32], ["batch", "seq", 4], None, 4, np.float16),
        (["batch", "seq", 10240], ["batch", "seq", 4], None, 4, np.float16),
        ([2, 3, 4, 8], [2, 3, 4, 8], [2, 3, 4, 4], 0, np.float32),
        ([1, 2, 4, 8], [1, 2, 4, 1], None, 0, np.float16),
    ],
)
def test_hyper_connection_post_mix(model_runner, streams, gate, mix, branches, dtype):
    hidden = streams[-1] // branches if branches else streams[-1]
    block = list(streams[: -1 if branches else -2]) + [hidden]
    inputs = ["streams", "block", "post_mix"] + (["stream_mix"] if mix else [])
    attrs = {"num_branches": branches} if branches else {}
    node = helper.make_node(
        "HyperConnectionPostMix",
        inputs,
        ["out"],
        domain="com.microsoft",
        **attrs,
    )
    vis = [
        _vi("streams", dtype, streams),
        _vi("block", dtype, block),
        _vi("post_mix", dtype, gate),
    ]
    if mix:
        vis.append(_vi("stream_mix", dtype, mix))
    model = _model([node], vis, [_vi("out", dtype, streams)])
    rng = np.random.default_rng(39)

    def run_shape(shape):
        return [2 if d == "batch" else 3 if d == "seq" else d for d in shape]

    feeds = [
        rng.uniform(-1, 1, run_shape(streams)).astype(dtype),
        rng.uniform(-1, 1, run_shape(block)).astype(dtype),
        rng.uniform(0, 1, run_shape(gate)).astype(dtype),
    ]
    if mix:
        feeds.append(rng.uniform(-0.5, 0.5, run_shape(mix)).astype(dtype))
    actual, expected = model_runner.run_sample(model, feeds)
    _compare(actual, expected, dtype)
