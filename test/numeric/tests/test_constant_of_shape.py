#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""Numeric checks for onnx.ConstantOfShape against the ORT CPU reference."""

import numpy as np
import pytest
from onnx import TensorProto, helper, numpy_helper

from framework.comparator import compare_outputs
from framework.onnx_utils import make_model_from_nodes

_FLT_MIN = float(np.finfo(np.float32).min)


def _fill_node(value, dtype):
    return helper.make_node(
        "ConstantOfShape",
        ["shape"],
        ["fill"],
        value=numpy_helper.from_array(np.array([value], dtype=dtype)),
    )


class TestConstantOfShape:
    @pytest.mark.parametrize("value", [2.5, _FLT_MIN])
    def test_dynamic_shape_add(self, model_runner, value):
        """A runtime-shaped fill consumed by a non-Where op.

        The shape is a graph input, so the result has dynamic dims and the
        full-size buffer must be filled on the device.
        """
        shape = helper.make_tensor_value_info("shape", TensorProto.INT64, [2])
        x = helper.make_tensor_value_info("X", TensorProto.FLOAT, ["M", "N"])
        y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, ["M", "N"])
        add = helper.make_node("Add", ["fill", "X"], ["Y"])
        model = make_model_from_nodes(
            [_fill_node(value, np.float32), add], [shape, x], [y]
        )
        data = np.arange(12, dtype=np.float32).reshape(3, 4)
        actual, expected = model_runner.run_sample(
            model, [np.array([3, 4], dtype=np.int64), data]
        )
        compare_outputs(actual, expected)

    def test_causal_mask_with_shape_consumer(self, model_runner):
        """Whisper decoder causal mask: Where(mask, 0, fill) where the fill is
        also read by Shape, so the rank-0 Where shortcut does not apply."""
        x = helper.make_tensor_value_info("X", TensorProto.FLOAT, ["S", "S"])
        y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, ["S", "S"])
        zero = numpy_helper.from_array(np.array(0.0, dtype=np.float32), "zero")
        limit = numpy_helper.from_array(np.array(7.0, dtype=np.float32), "limit")
        nodes = [
            helper.make_node("Shape", ["X"], ["shape"]),
            _fill_node(_FLT_MIN, np.float32),
            helper.make_node("Shape", ["fill"], ["fill_shape"]),
            helper.make_node("Less", ["X", "limit"], ["mask"]),
            helper.make_node("Where", ["mask", "zero", "fill"], ["masked"]),
            helper.make_node("Reshape", ["masked", "fill_shape"], ["Y"]),
        ]
        model = make_model_from_nodes(nodes, [x], [y], initializers=[zero, limit])
        data = np.arange(16, dtype=np.float32).reshape(4, 4)
        actual, expected = model_runner.run_sample(model, [data])
        compare_outputs(actual, expected)
