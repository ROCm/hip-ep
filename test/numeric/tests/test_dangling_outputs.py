#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""Graphs with dead-end nodes whose outputs are neither consumed nor graph
outputs. The EP must still claim them (the suite disables CPU fallback) and
must not expose their values as fused-node outputs."""

from __future__ import annotations

import numpy as np
from onnx import TensorProto, helper, numpy_helper

from framework.comparator import compare_outputs
from framework.onnx_utils import make_model_from_nodes

_F32 = TensorProto.FLOAT


def _matmul_add_model(dead_nodes, extra_initializers=()):
    x = helper.make_tensor_value_info("x", _F32, [4, 8])
    y = helper.make_tensor_value_info("y", _F32, [4, 8])
    rng = np.random.default_rng(1301)
    w = numpy_helper.from_array(rng.standard_normal((8, 8)).astype(np.float32), "w")
    b = numpy_helper.from_array(rng.standard_normal((8,)).astype(np.float32), "b")
    nodes = [
        helper.make_node("MatMul", ["x", "w"], ["h"]),
        helper.make_node("Add", ["h", "b"], ["y"]),
        *dead_nodes,
    ]
    return make_model_from_nodes(
        nodes, [x], [y], initializers=[w, b, *extra_initializers]
    )


class TestDanglingOutputs:
    def test_dead_node_on_graph_input(self, model_runner):
        x = helper.make_tensor_value_info("x", _F32, [4, 8])
        y = helper.make_tensor_value_info("y", _F32, [4, 8])
        nodes = [
            helper.make_node("Relu", ["x"], ["y"]),
            helper.make_node("Neg", ["x"], ["dead"]),
        ]
        model = make_model_from_nodes(nodes, [x], [y])
        data = np.random.default_rng(1300).standard_normal((4, 8)).astype(np.float32)
        actual, expected = model_runner.run_sample(model, [data])
        compare_outputs(actual, expected)

    def test_dead_topk_chain(self, model_runner):
        k = numpy_helper.from_array(np.array([2], dtype=np.int64), "k")
        model = _matmul_add_model(
            [
                helper.make_node("TopK", ["h", "k"], ["values", "indices"], axis=1),
                helper.make_node("Cast", ["values"], ["dead"], to=TensorProto.FLOAT16),
            ],
            extra_initializers=[k],
        )
        data = np.random.default_rng(1302).standard_normal((4, 8)).astype(np.float32)
        actual, expected = model_runner.run_sample(model, [data])
        compare_outputs(actual, expected, rtol=1e-3, atol=1e-3)

    def test_dead_multi_level_chain(self, model_runner):
        model = _matmul_add_model(
            [
                helper.make_node("Abs", ["h"], ["a"]),
                helper.make_node("Sigmoid", ["a"], ["s"]),
                helper.make_node("Neg", ["s"], ["dead"]),
            ]
        )
        data = np.random.default_rng(1303).standard_normal((4, 8)).astype(np.float32)
        actual, expected = model_runner.run_sample(model, [data])
        compare_outputs(actual, expected, rtol=1e-3, atol=1e-3)
