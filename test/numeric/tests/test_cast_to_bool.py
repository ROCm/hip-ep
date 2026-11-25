#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""Cast to bool is x != 0, read back through Where.

hip-decompose-bool-cast rewrites a non-bool Cast(to=BOOL) as Not(Equal(x, 0)).
A by-value byte cast would store int64 5 as 5 and 256 as 0, so the inputs mix
zero, one, values above one, values whose low byte is zero, negatives, -0.0 and
NaN. The mask selects between two float tensors, so any value other than 0/1
in it shows up in the output.
"""

from __future__ import annotations

import numpy as np
import pytest
from onnx import TensorProto, helper

from framework.comparator import compare_outputs
from framework.onnx_utils import make_model_from_nodes

_N = 16

_CASES = {
    "int64": (
        TensorProto.INT64,
        np.array(
            [0, 1, 5, 256, -3, 0, 2, -1, 1 << 40, 0, 7, 0, 255, 512, 0, 9],
            dtype=np.int64,
        ),
    ),
    "int32": (
        TensorProto.INT32,
        np.array(
            [0, 1, 5, 256, -3, 0, 2, -1, 1 << 24, 0, 7, 0, 255, 512, 0, 9],
            dtype=np.int32,
        ),
    ),
    "uint8": (
        TensorProto.UINT8,
        np.array([0, 1, 5, 255, 2, 0, 128, 0, 3, 0, 1, 0, 64, 0, 0, 9], dtype=np.uint8),
    ),
    "float32": (
        TensorProto.FLOAT,
        np.array(
            [
                0.0,
                -0.0,
                2.5,
                np.nan,
                1e-30,
                -1.0,
                0.0,
                0.5,
                np.inf,
                0.0,
                -np.inf,
                3.0,
                0.0,
                1.0,
                -0.0,
                7.0,
            ],
            dtype=np.float32,
        ),
    ),
    "float16": (
        TensorProto.FLOAT16,
        np.array(
            [
                0.0,
                -0.0,
                2.5,
                np.nan,
                1e-7,
                -1.0,
                0.0,
                0.5,
                np.inf,
                0.0,
                -np.inf,
                3.0,
                0.0,
                1.0,
                -0.0,
                7.0,
            ],
            dtype=np.float16,
        ),
    ),
}


def _cast_where(elem_type):
    nodes = [
        helper.make_node("Cast", ["x"], ["mask"], to=TensorProto.BOOL),
        helper.make_node("Where", ["mask", "a", "b"], ["y"]),
    ]
    inputs = [
        helper.make_tensor_value_info("x", elem_type, [_N]),
        helper.make_tensor_value_info("a", TensorProto.FLOAT, [_N]),
        helper.make_tensor_value_info("b", TensorProto.FLOAT, [_N]),
    ]
    outputs = [helper.make_tensor_value_info("y", TensorProto.FLOAT, [_N])]
    return make_model_from_nodes(nodes, inputs, outputs, opset=13)


class TestCastToBool:
    @pytest.mark.parametrize("name", list(_CASES))
    def test_cast_where(self, model_runner, name):
        elem_type, x = _CASES[name]
        a = np.arange(1, _N + 1, dtype=np.float32)
        b = -a
        actual, expected = model_runner.run_sample(
            _cast_where(elem_type), [x, a, b], name=f"cast_to_bool_{name}"
        )
        compare_outputs(actual, expected, atol=0, rtol=0)
