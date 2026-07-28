#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""Numeric coverage for variadic Max/Min and multidirectional broadcasting."""

import numpy as np
import pytest
from onnx import TensorProto, helper

from framework.comparator import compare_outputs
from framework.onnx_utils import make_model_from_nodes


def _make_binary_model(op_type: str):
    lhs = helper.make_tensor_value_info("lhs", TensorProto.FLOAT, ["M", 1])
    rhs = helper.make_tensor_value_info("rhs", TensorProto.FLOAT, [1, "N"])
    output = helper.make_tensor_value_info("output", TensorProto.FLOAT, ["M", "N"])
    node = helper.make_node(op_type, ["lhs", "rhs"], ["output"])
    return make_model_from_nodes([node], [lhs, rhs], [output])


def _make_variadic_model(op_type: str):
    a = helper.make_tensor_value_info("a", TensorProto.FLOAT, [4])
    b = helper.make_tensor_value_info("b", TensorProto.FLOAT, [3, 4])
    c = helper.make_tensor_value_info("c", TensorProto.FLOAT, [2, 3, 4])
    output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [2, 3, 4])
    node = helper.make_node(op_type, ["a", "b", "c"], ["output"])
    return make_model_from_nodes([node], [a, b, c], [output])


class TestMaxMinBroadcast:
    @pytest.mark.parametrize("op_type", ["Max", "Min"])
    def test_asymmetric_dynamic_broadcast(self, model_runner, op_type):
        model = _make_binary_model(op_type)
        rng = np.random.default_rng(42)
        lhs = rng.uniform(-1, 1, (3, 1)).astype(np.float32)
        rhs = rng.uniform(-1, 1, (1, 5)).astype(np.float32)

        actual, expected = model_runner.run_sample(model, [lhs, rhs], reference="cpu")
        compare_outputs(actual, expected, atol=1e-6)

    @pytest.mark.parametrize("op_type", ["Max", "Min"])
    def test_variadic_rank_growth(self, model_runner, op_type):
        model = _make_variadic_model(op_type)
        rng = np.random.default_rng(43)
        inputs = [
            rng.uniform(-1, 1, (4,)).astype(np.float32),
            rng.uniform(-1, 1, (3, 4)).astype(np.float32),
            rng.uniform(-1, 1, (2, 3, 4)).astype(np.float32),
        ]

        actual, expected = model_runner.run_sample(model, inputs, reference="cpu")
        compare_outputs(actual, expected, atol=1e-6)
