#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""Check a copy-filled Pad shape that also supplies a scalar readback."""

import numpy as np
import pytest
from onnx import TensorProto, helper

from framework.onnx_utils import make_model_from_nodes


@pytest.mark.parametrize("amount", [0, 1, 4])
def test_copy_readback_pad_shape(model_runner, amount):
    tensor = helper.make_tensor_value_info
    inputs = [
        tensor("x", TensorProto.FLOAT, [2, 3]),
        tensor("amount", TensorProto.INT64, [1]),
    ]
    outputs = [tensor("y", TensorProto.FLOAT, [2, None])]
    initializers = [helper.make_tensor("zero", TensorProto.INT64, [1], [0])]
    nodes = [
        helper.make_node(
            "Concat", ["zero", "amount", "zero", "zero"], ["pads"], axis=0
        ),
        helper.make_node("Pad", ["x", "pads"], ["y"], mode="constant"),
    ]
    model = make_model_from_nodes(nodes, inputs, outputs, initializers, opset=13)
    data = np.arange(6, dtype=np.float32).reshape(2, 3)
    actual, reference = model_runner.run_sample(
        model, [data, np.array([amount], dtype=np.int64)]
    )
    np.testing.assert_array_equal(actual[0], reference[0])
    np.testing.assert_array_equal(actual[0], np.pad(data, ((0, 0), (amount, 0))))
