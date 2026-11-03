#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""Check host-built Slice bounds after data-dependent NonZero output."""

import numpy as np
import pytest
from onnx import TensorProto, helper

from framework.onnx_utils import make_model_from_nodes


def make_nonzero_gather_slice_model(always_empty=False):
    shape = [1, 1, 60]
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, shape)
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [None])
    zero = helper.make_tensor("zero", TensorProto.FLOAT, [], [0.0])
    starts = helper.make_tensor("starts", TensorProto.INT64, [1], [0])
    axes = helper.make_tensor("axes", TensorProto.INT64, [1], [0])
    nodes = [
        helper.make_node("Less", ["x", "x" if always_empty else "zero"], ["mask"]),
        helper.make_node("NonZero", ["mask"], ["coords"]),
        helper.make_node("Transpose", ["coords"], ["indices"], perm=[1, 0]),
        helper.make_node("GatherND", ["x", "indices"], ["gathered"]),
        helper.make_node("Shape", ["gathered"], ["ends"]),
        helper.make_node("Slice", ["gathered", "starts", "ends", "axes"], ["y"]),
    ]
    initializers = [starts, axes] if always_empty else [zero, starts, axes]
    return make_model_from_nodes(nodes, [x], [y], initializers, opset=13)


@pytest.mark.parametrize("selected", [0, 1, 30, 60])
def test_nonzero_gather_slice(model_runner, selected):
    model = make_nonzero_gather_slice_model()
    data = np.arange(1, 61, dtype=np.float32)
    data[:selected] *= -1
    actual, expected = model_runner.run_sample(model, [data.reshape(1, 1, 60)])
    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_array_equal(actual[0], data[:selected])


def test_nonzero_always_empty_gather_slice(model_runner):
    model = make_nonzero_gather_slice_model(always_empty=True)
    data = np.arange(60, dtype=np.float32).reshape(1, 1, 60)
    actual, expected = model_runner.run_sample(model, [data])
    np.testing.assert_array_equal(actual[0], expected[0])
    assert actual[0].shape == (0,)
