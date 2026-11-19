#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""Numeric coverage for the diffusers-style GEGLU frontend fusion."""

import numpy as np
import pytest
from onnx import TensorProto, helper, numpy_helper

from framework.comparator import compare_outputs
from framework.onnx_utils import make_model_from_nodes

F16 = TensorProto.FLOAT16


def _const(name, array):
    return numpy_helper.from_array(np.asarray(array), name=name)


def _make_geglu_model(rows, hidden):
    projection = helper.make_tensor_value_info("projection", F16, [2, rows, 2 * hidden])
    output = helper.make_tensor_value_info("output", F16, [2, rows, hidden])
    initializers = [
        _const("zero", np.array([0], np.int64)),
        _const("hidden", np.array([hidden], np.int64)),
        _const("two_hidden", np.array([2 * hidden], np.int64)),
        _const("axis", np.array([-1], np.int64)),
        _const("sqrt2", np.array(1.4140625, np.float16)),
        _const("one", np.array(1.0, np.float16)),
        _const("half", np.array(0.5, np.float16)),
    ]
    nodes = [
        helper.make_node("Slice", ["projection", "zero", "hidden", "axis"], ["value"]),
        helper.make_node(
            "Slice", ["projection", "hidden", "two_hidden", "axis"], ["gate"]
        ),
        helper.make_node("Div", ["gate", "sqrt2"], ["scaled_gate"]),
        helper.make_node("Erf", ["scaled_gate"], ["erf_gate"]),
        helper.make_node("Add", ["erf_gate", "one"], ["one_plus_erf"]),
        helper.make_node("Mul", ["gate", "one_plus_erf"], ["gate_times_erf"]),
        helper.make_node("Mul", ["gate_times_erf", "half"], ["gelu_gate"]),
        helper.make_node("Mul", ["value", "gelu_gate"], ["output"]),
    ]
    return make_model_from_nodes(
        nodes, [projection], [output], initializers=initializers
    )


class TestGeglu:
    @pytest.mark.parametrize(
        "rows,hidden",
        [
            (256, 320),
            (64, 640),
            (16, 1280),
        ],
    )
    def test_sd15_shapes(self, model_runner, rows, hidden):
        model = _make_geglu_model(rows, hidden)
        rng = np.random.default_rng(12)
        projection = (rng.standard_normal((2, rows, 2 * hidden)) * 2).astype(np.float16)
        actual, expected = model_runner.run_sample(model, [projection])
        compare_outputs(actual, expected, atol=2e-2, rtol=1e-2)
