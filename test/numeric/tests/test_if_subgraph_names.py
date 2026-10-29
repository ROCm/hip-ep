#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""If bodies whose nodes produce values named like the If's own outputs, as
in merged decoders (`present.*` from both branches). The importer must
resolve those names inside the body instead of capturing the outer,
not-yet-produced If output."""

from __future__ import annotations

import numpy as np
from onnx import TensorProto, helper

from framework.comparator import compare_outputs
from framework.onnx_utils import make_model_from_nodes

_F32 = TensorProto.FLOAT
_SHAPE = [4, 8]


def _branch(name, op):
    return helper.make_graph(
        [
            helper.make_node(op, ["x"], ["h"]),
            helper.make_node("Add", ["h", "x"], ["present"]),
        ],
        name,
        [],
        [helper.make_tensor_value_info("present", _F32, _SHAPE)],
    )


def _model():
    x = helper.make_tensor_value_info("x", _F32, _SHAPE)
    cond = helper.make_tensor_value_info("cond", TensorProto.BOOL, [])
    present = helper.make_tensor_value_info("present", _F32, _SHAPE)
    node = helper.make_node(
        "If",
        ["cond"],
        ["present"],
        then_branch=_branch("then_body", "Relu"),
        else_branch=_branch("else_body", "Neg"),
    )
    return make_model_from_nodes([node], [x, cond], [present])


class TestIfSubgraphNames:
    def test_then_branch(self, model_runner):
        data = np.random.default_rng(1400).standard_normal(_SHAPE).astype(np.float32)
        actual, expected = model_runner.run_sample(_model(), [data, np.array(True)])
        compare_outputs(actual, expected)

    def test_else_branch(self, model_runner):
        data = np.random.default_rng(1401).standard_normal(_SHAPE).astype(np.float32)
        actual, expected = model_runner.run_sample(_model(), [data, np.array(False)])
        compare_outputs(actual, expected)
