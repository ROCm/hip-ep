#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""Numeric coverage for a transient MatMul -> Softmax -> MatMul chain."""

import numpy as np
import pytest
from onnx import TensorProto, helper, numpy_helper

from framework.comparator import compare_outputs
from framework.onnx_utils import make_model_from_nodes


def _make_model(dtype):
    onnx_dtype = TensorProto.FLOAT16 if dtype == np.float16 else TensorProto.FLOAT
    x_info = helper.make_tensor_value_info("X", onnx_dtype, [2, 4, 8])
    output_info = helper.make_tensor_value_info("output", onnx_dtype, [2, 4, 5])

    rng = np.random.default_rng(725)
    key = rng.uniform(-0.5, 0.5, [8, 6]).astype(dtype)
    value = rng.uniform(-0.5, 0.5, [6, 5]).astype(dtype)
    initializers = [
        numpy_helper.from_array(key, name="key"),
        numpy_helper.from_array(value, name="value"),
    ]

    nodes = [
        helper.make_node("MatMul", ["X", "key"], ["scores"]),
        helper.make_node("Softmax", ["scores"], ["probabilities"], axis=-1),
        helper.make_node("MatMul", ["probabilities", "value"], ["output"]),
    ]
    return make_model_from_nodes(
        nodes, [x_info], [output_info], initializers=initializers
    )


@pytest.mark.parametrize("dtype", [np.float16, np.float32])
def test_transient_softmax_inplace(model_runner, dtype):
    model = _make_model(dtype)
    rng = np.random.default_rng(726)
    x = rng.uniform(-1.0, 1.0, [2, 4, 8]).astype(dtype)

    actual, expected = model_runner.run_sample(model, [x], reference="cpu")
    tolerance = 2e-3 if dtype == np.float16 else 1e-5
    compare_outputs(actual, expected, atol=tolerance, rtol=tolerance)
