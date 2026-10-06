#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""Numeric checks for onnx.Trilu against the ORT CPU reference."""

import numpy as np
import pytest
from onnx import TensorProto, helper, numpy_helper

from framework.comparator import compare_outputs
from framework.onnx_utils import make_model_from_nodes


def _model(shape, upper, k=None):
    x = helper.make_tensor_value_info("X", TensorProto.FLOAT, shape)
    y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, shape)
    inputs = ["X"]
    inits = []
    if k is not None:
        inputs.append("K")
        inits.append(numpy_helper.from_array(np.array(k, dtype=np.int64), "K"))
    node = helper.make_node("Trilu", inputs, ["Y"], upper=upper)
    return make_model_from_nodes([node], [x], [y], initializers=inits)


class TestTrilu:
    def test_lower_triangle(self, model_runner):
        model = _model([2, 4, 4], upper=0, k=0)
        data = np.arange(32, dtype=np.float32).reshape(2, 4, 4)
        actual, expected = model_runner.run_sample(model, [data])
        compare_outputs(actual, expected)

    def test_upper_diagonal(self, model_runner):
        model = _model([1, 5, 3], upper=1, k=1)
        data = np.arange(15, dtype=np.float32).reshape(1, 5, 3)
        actual, expected = model_runner.run_sample(model, [data])
        compare_outputs(actual, expected)

    def test_default_upper(self, model_runner):
        model = _model([3, 3], upper=1)
        data = np.arange(9, dtype=np.float32).reshape(3, 3)
        actual, expected = model_runner.run_sample(model, [data])
        compare_outputs(actual, expected)

    @pytest.mark.parametrize("k", [-2, -1, 0, 2])
    def test_negative_and_positive_k(self, model_runner, k):
        model = _model([1, 4, 6], upper=0, k=k)
        data = np.arange(24, dtype=np.float32).reshape(1, 4, 6)
        actual, expected = model_runner.run_sample(model, [data])
        compare_outputs(actual, expected)

    def test_constant_of_shape_rank0_fill(self, model_runner):
        """Trilu of a dynamic ConstantOfShape broadcasts one host scalar.

        The shape is a graph input so ConstantOfShape cannot fold into a
        full splat. The kernel must take the one-element input path.
        """
        shape = helper.make_tensor_value_info("shape", TensorProto.INT64, [3])
        y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, ["B", "M", "N"])
        ones = helper.make_node(
            "ConstantOfShape",
            ["shape"],
            ["ones"],
            value=numpy_helper.from_array(np.array([1.0], dtype=np.float32)),
        )
        k = numpy_helper.from_array(np.array(0, dtype=np.int64), "K")
        trilu = helper.make_node("Trilu", ["ones", "K"], ["Y"], upper=0)
        model = make_model_from_nodes([ones, trilu], [shape], [y], initializers=[k])
        actual, expected = model_runner.run_sample(
            model, [np.array([1, 4, 4], dtype=np.int64)]
        )
        compare_outputs(actual, expected)
        assert actual[0].shape == (1, 4, 4)
        assert np.all(actual[0] == np.tril(np.ones((1, 4, 4), dtype=np.float32)))
